from typing import Optional
from einops import rearrange

import torch
import torch.nn as nn
import torch.nn.functional as F

from peft import LoraConfig, get_peft_model
from GET_Transformer.models.GET import GET
from transformers import (AutoTokenizer,
                          BlipForConditionalGeneration,
                          BlipForImageTextRetrieval)


class EventVLM(nn.Module):
    def __init__(self,
                 event_encoder: GET,
                 device: torch.device = None):

        super().__init__()

        # Store the components
        self.event_encoder = event_encoder
        self.aligner = nn.Sequential(nn.Linear(self.event_encoder.num_features[-1], 512),
                                     nn.GELU(),
                                     nn.LayerNorm(512),
                                     nn.Linear(512, 768))
        self.text_decoder = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").text_decoder
        self.text_encoder = BlipForImageTextRetrieval.from_pretrained("Salesforce/blip-itm-base-coco").text_encoder

        # Configure LoRA
        lora_config = LoraConfig(r=8,
                                 lora_alpha=16,
                                 target_modules=["query", "value"],
                                 lora_dropout=0.1,
                                 bias="none",
                                 task_type="CAUSAL_LM")
        self.text_decoder = get_peft_model(self.text_decoder, lora_config)
        self.text_decoder.print_trainable_parameters()

        # Set up tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("Salesforce/blip-image-captioning-base")

        # Add special tokens
        special_tokens_dict = {}
        if self.tokenizer.pad_token is None:
            special_tokens_dict['pad_token'] = '[PAD]'
        self.tokenizer.add_special_tokens(special_tokens_dict)
        self.text_decoder.resize_token_embeddings(len(self.tokenizer))

        # Freeze vision encoder
        for param in self.event_encoder.parameters():
            param.requires_grad = False
            
        # Freeze the text encoder
        for param in self.text_encoder.parameters():
            param.requires_grad = False

        # Move the components to the device if provided
        if device is not None:
            self.to(device=device)

    @property
    def event_vision_device(self):
        return next(self.event_encoder.parameters()).device

    @property
    def aligner_device(self):
        return next(self.aligner.parameters()).device

    @property
    def language_device(self):
        return next(self.language_model.parameters()).device

    @property
    def device(self):
        return next(self.parameters()).device

    def to(self, device: torch.device):
        self.event_encoder = self.event_encoder.to(device)
        self.aligner = self.aligner.to(device)
        self.text_decoder = self.text_decoder.to(device)
        self.text_encoder = self.text_encoder.to(device)
        return self

    def forward(self,
                event_representations: torch.FloatTensor = None,
                captions: Optional[torch.LongTensor] = None,
                **kwargs):

        # Make sure that event and text encoders are in evaluation mode
        self.event_encoder.eval()
        self.text_encoder.eval()

        # Encode event features [B, N, C, H, W] -> [B, num_visual_tokens, 384]
        event_features = self.event_encoder(rearrange(event_representations, "B N C H W -> (B N) C H W"))

        # Project to decoder's dimension [B, 16, 768]
        visual_context = self.aligner(event_features)

        # Get visual embedding by averaging visual tokens
        visual_embeds = visual_context.mean(dim=1)  # [B, 768]

        # Tokenize captions
        inputs = self.tokenizer(captions, return_tensors="pt", padding=True, truncation=True).to(self.device)
        input_ids = inputs.input_ids  # [B, seq_len]
        attention_mask = inputs.attention_mask
        
        # Contrastive loss
        text_embeds = self.text_encoder(input_ids=input_ids,
                                        attention_mask=attention_mask).last_hidden_state[:, 0, :]  # [B, 768]

        # Normalize features
        visual_embeds = visual_embeds / visual_embeds.norm(dim=1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(dim=1, keepdim=True)

        # Compute similarity matrix
        logits_per_image = visual_embeds @ text_embeds.t()  # [B, B]
        logits_per_text = logits_per_image.t()  # [B, B]

        # Contrastive loss (similar to BLIP)
        batch_size = visual_embeds.shape[0]
        targets = torch.arange(batch_size, device=visual_embeds.device)

        contrastive_loss = (
                                   F.cross_entropy(logits_per_image, targets) +
                                   F.cross_entropy(logits_per_text, targets)
                           ) / 2

        # Forward pass with visual context
        logits = self.text_decoder(input_ids=input_ids,
                                   attention_mask=attention_mask,
                                   encoder_hidden_states=visual_context,
                                   encoder_attention_mask=None).logits

        # Caption loss
        # Shift targets left and ignore last logit
        shifted_logits = logits[:, :-1, :].contiguous()  # [B, seq_len-1, vocab_size]
        shifted_input_ids = input_ids[:, 1:].contiguous()  # [B, seq_len-1]
        caption_loss = F.cross_entropy(shifted_logits.view(-1, shifted_logits.size(-1)),
                                       shifted_input_ids.view(-1),
                                       ignore_index=self.tokenizer.pad_token_id)

        # Combine losses
        total_loss = caption_loss + contrastive_loss
        
        return total_loss

    def generate(self,
                 event_representations: torch.FloatTensor,
                 max_length: int = 30):

        self.eval()
        with torch.no_grad():

            # Encode the visual input
            event_features = self.event_encoder(rearrange(event_representations, "B N C H W -> (B N) C H W"))
            visual_context = self.aligner(event_features)

            # Provide [CLS] token for the start of the sequence (no text prompt)
            input_ids = torch.tensor([[self.tokenizer.cls_token_id]], device=self.device)

            # Generate the text conditioned on the visual context (greedy decoding)
            generated_ids = self.text_decoder.generate(input_ids=input_ids,
                                                       encoder_hidden_states=visual_context,
                                                       encoder_attention_mask=None,
                                                       max_length=max_length,
                                                       num_beams=1,
                                                       do_sample=False)
            captions = self.tokenizer.batch_decode(generated_ids,
                                                   skip_special_tokens=True)

            return captions


def build_model(config: dict,
                device: torch.device = None) -> EventVLM:
    """
    Builds and configures the complete EventVLM model with all components.

    Args:
        config (dict): Dictionary containing the configuration of the experiment
        device (torch.device): Device where the model is located.

    Returns:
        EventVLM: Fully configured EventVLM model instance

    Raises:
        ValueError: If the type of the models was not recognised.
        RuntimeError: If required components are missing from the loaded model.
    """

    def _get_event_encoder(model_config: dict) -> GET:
        """
        Builds the event vision encoder from config.

        Args:
            model_config (dict): Dictionary containing the configuration of the event vision model

        Returns:
            GET: Loaded instance of the event vision encoder
        """

        if model_config['type'] == 'GET':
            model_parameters = model_config['parameters']

            # Initialize GET
            model = GET(patch_size=model_parameters['patch_size'],
                        num_classes=-1,  # No classification head
                        embed_dim=model_parameters['embed_dim'],
                        depths=model_parameters['depths'],
                        num_heads=model_parameters['num_heads'],
                        window_size=model_parameters['window_size'],
                        mlp_ratio=model_parameters['mlp_ratio'],
                        drop_rate=model_parameters['drop_rate'],
                        attn_drop_rate=model_parameters['attn_drop_rate'],
                        drop_path_rate=model_parameters['drop_path_rate'],
                        use_checkpoint=model_parameters['use_checkpoint'],
                        embed_split=model_parameters['embed_split'],
                        group_num=model_parameters['group_num'])

            return model
        else:
            raise ValueError(f"Unrecognized event vision model type: {model_config['type']}")

    # Build event vision encoder
    event_encoder = _get_event_encoder(model_config=config['event_vision_model'])

    # Assemble the final model
    return EventVLM(event_encoder=event_encoder,
                    device=device)
