from einops import rearrange
from typing import Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM
from deepseek_vl.models import MultiModalityCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast

from GET_Transformer.models.GET import GET


class EventVLM(nn.Module):
    def __init__(self,
                 event_vision_model: GET,
                 aligner: nn.Module,
                 language_model: nn.Module,
                 device: torch.device = None):

        super().__init__()

        # Store the components
        self.event_vision_model = event_vision_model
        self.aligner = aligner
        self.language_model = language_model.eval()

        # Freeze the language model
        for param in self.language_model.parameters():
            param.requires_grad = False

        # Move the components to the device if provided
        if device is not None:
            self.to(device=device)

    @property
    def event_vision_device(self):
        return next(self.event_vision_model.parameters()).device

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
        self.event_vision_model = self.event_vision_model.to(device)
        self.aligner = self.aligner.to(device)
        self.language_model = self.language_model.to(device)
        return self

    def get_input_embeddings(self,
                             input_ids: torch.LongTensor,
                             event_representations: torch.FloatTensor,
                             event_seq_mask: torch.BoolTensor,
                             event_emb_mask: torch.BoolTensor,
                             **kwargs) -> torch.FloatTensor:
        """
        Combine text embeddings with aligned event embeddings by replacing event token placeholders.

        Args:
            input_ids (torch.LongTensor): Padded tokenized text with event token placeholders
                            Shape: [batch_size, max_seq_len]
            event_representations (torch.FloatTensor): Processed event data
                            Shape: [batch_size, max_event_streams_per_sample, token_dim, grid_height, grid_width]
            event_seq_mask (torch.BoolTensor): Mask indicating event token placeholder positions in the prompt/conversation
                            Shape: [batch_size, max_seq_len]
            event_emb_mask (torch.BoolTensor): Mask for valid event tokens in event_representations tensor
                            Shape: [batch_size, max_event_streams_per_sample, num_event_tokens]

        Returns:
            inputs_embeds (torch.FloatTensor): Combined embeddings with event representations inserted at placeholder positions
                            Shape: [batch_size, max_seq_len, hidden_size]

        Raises:
            ValueError: If torch.sum(event_seq_mask) != torch.sum(event_emb_mask), i.e. the number of event tokens
                        mismatches in two masks
        """

        if torch.sum(event_seq_mask) != torch.sum(event_emb_mask):
            raise ValueError("event_seq_mask and event_emb_mask should mask out the same number of elements.")

        batch_size, max_event_streams_per_sample = event_representations.shape[0:2]

        # Flatten batch and event stream dimensions for parallel processing
        # [batch_size x max_event_streams_per_sample, token_dim, grid_height, grid_width]
        event_tokens = rearrange(event_representations, "B N C H W -> (B N) C H W")

        # Process event tokens through event vision model and aligner
        # [batch_size x max_event_streams_per_sample, num_event_tokens, hidden_size]
        event_embeddings = self.aligner(self.event_vision_model(event_tokens))

        # Reshape back to separate batch and event stream dimensions
        # [batch_size, max_event_streams_per_sample x num_event_tokens, hidden_size]
        event_embeddings = rearrange(event_embeddings, "(B N) T D -> B (N T) D",
                                     B=batch_size, N=max_event_streams_per_sample)

        # Flatten the mask for valid event tokens to match the reshaped embeddings
        # [batch_size, max_event_streams_per_sample x num_event_tokens]
        event_emb_mask = rearrange(event_emb_mask, "B N T -> B (N T)")

        # Get the input text embeddings
        # [batch_size, max_seq_len, hidden_size]
        inputs_embeds = self.language_model.get_input_embeddings()(input_ids)

        # Replace event token placeholders with corresponding event embeddings
        # [batch_size, max_seq_len, hidden_size]
        inputs_embeds[event_seq_mask] = event_embeddings[event_emb_mask]

        return inputs_embeds

    def forward(self,
                inputs_embeds: torch.FloatTensor,
                attention_mask: torch.LongTensor,
                input_ids: torch.LongTensor = None,
                event_representations: torch.FloatTensor = None,
                event_seq_mask: torch.BoolTensor = None,
                event_emb_mask: torch.BoolTensor = None,
                labels: Optional[torch.LongTensor] = None,
                **kwargs) -> CausalLMOutputWithPast:

        """
        Forward pass for the EventVLM model.

        Args:
            inputs_embeds (torch.FloatTensor): Embedded representation of input tokens
                Shape: [batch_size, max_seq_len, hidden_size]
            attention_mask (torch.Tensor): Attention mask for text and event tokens
                Shape: [batch_size, max_seq_len]
            input_ids (torch.Tensor): Padded tokenized text with event token placeholders
                Shape: [batch_size, max_seq_len]
            event_representations (torch.Tensor): Processed event data
                Shape: [batch_size, max_event_streams_per_sample, token_dim, grid_height, grid_width]
            event_seq_mask (torch.BoolTensor): Mask indicating event token positions
                Shape: [batch_size, max_seq_len]
            event_emb_mask (torch.BoolTensor): Mask for valid event tokens
                Shape: [batch_size, max_event_streams_per_sample, num_event_tokens]
            labels (torch.LongTensor, optional): Labels for computing the masked language modeling loss
                Shape: [batch_size, max_seq_len]

        Returns:
            model_output (transformers.modeling_outputs.CausalLMOutputWithPast):
                Language model output with logits, loss (if labels provided), etc.

        Raises:
            ValueError: If neither or both of input_ids and inputs_embeds are specified.
            ValueError: If input_ids is specified but required embedding generation inputs
                       (event_representations, event_seq_mask, or event_emb_mask) are missing.
        """

        # Validate that exactly one of input_ids or inputs_embeds is provided
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds.")

        # Generate inputs_embeds if not provided
        if inputs_embeds is None:
            if None in (event_representations, event_seq_mask, event_emb_mask):
                raise ValueError("When using input_ids, you must provide all of event_representations,"
                                 " event_seq_mask, and event_emb_mask.")
            inputs_embeds = self.get_input_embeddings(input_ids=input_ids,
                                                      event_representations=event_representations,
                                                      event_seq_mask=event_seq_mask,
                                                      event_emb_mask=event_emb_mask)

        # Ensure all tensors are on the same device
        inputs_embeds = inputs_embeds.to(self.language_device)
        attention_mask = attention_mask.to(self.language_device)

        if labels is not None:
            labels = labels.to(self.language_device)

        # Pass through language model
        return self.language_model(inputs_embeds=inputs_embeds,
                                   attention_mask=attention_mask,
                                   labels=labels,
                                   **kwargs)

    def generate(self,
                 inputs_embeds: torch.FloatTensor,
                 attention_mask: torch.LongTensor,
                 pad_token_id: int,
                 bos_token_id: int,
                 eos_token_id: int,
                 max_new_tokens: int = 512,
                 do_sample: bool = False,
                 use_cache: bool = True) -> torch.Tensor:

        # Ensure all tensors are on the same device
        inputs_embeds = inputs_embeds.to(self.language_device)
        attention_mask = attention_mask.to(self.language_device)

        return self.language_model.generate(inputs_embeds=inputs_embeds,
                                            attention_mask=attention_mask,
                                            pad_token_id=pad_token_id,
                                            bos_token_id=bos_token_id,
                                            eos_token_id=eos_token_id,
                                            max_new_tokens=max_new_tokens,
                                            do_sample=do_sample,
                                            use_cache=use_cache)


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

    def _get_vision_language_model(model_config: dict) -> MultiModalityCausalLM:
        """
        Loads the pretrained vision-language backbone model.

        Args:
            model_config (dict): Dictionary containing the configuration of the vision language model
                                 (e.g. model path, data type, etc.)

        Returns:
            MultiModalityCausalLM: Loaded instance of the vision-language backbone model

        Raises:
            ValueError: If the type of the model was not recognised.
        """

        if model_config['type'] == 'DeepSeek-VL':
            return AutoModelForCausalLM.from_pretrained(model_config['model_path'],
                                                        trust_remote_code=True,
                                                        torch_dtype=getattr(torch, model_config['dtype']))
        else:
            raise ValueError(f"Unrecognized vision language model type: {model_config['type']}")

    def _get_event_vision_model(model_config: dict) -> GET:
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
                        num_classes=model_parameters['num_classes'],
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

    def _get_components(model: MultiModalityCausalLM) -> Tuple[nn.Module, nn.Module]:
        """
        Extracts aligner and language model components from VL model.

        Args:
            model (MultiModalityCausalLM): Loaded vision-language backbone model

        Returns:
            tuple: Aligner and language model components of the vision-language backbone model

        Raises:
            RuntimeError: If required components are missing.
        """

        if not hasattr(model, 'aligner'):
            raise RuntimeError("Loaded model missing required 'aligner' component.")
        if not hasattr(model, 'language_model'):
            raise RuntimeError("Loaded model missing required 'language_model' component.")

        return model.aligner, model.language_model

    # Load backbone model
    vl_model = _get_vision_language_model(model_config=config['vision_language_model'])

    # Build event vision encoder
    event_vision_model = _get_event_vision_model(model_config=config['event_vision_model'])

    # Extract components
    aligner, language_model = _get_components(model=vl_model)

    # Assemble the final model
    return EventVLM(event_vision_model=event_vision_model,
                    aligner=aligner,
                    language_model=language_model,
                    device=device)
