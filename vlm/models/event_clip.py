import numpy as np
from einops import rearrange

import clip
import torch
import torch.nn as nn
from torch.nn import functional as F

from GET_Transformer.models.GET import GET


class EventCLIP(nn.Module):
    def __init__(self,
                 event_encoder: GET,
                 projection_dim: int = 128,
                 temperature: float = 0.07,
                 n_prompts: int = 2,
                 use_gate: bool = True,
                 device: torch.device = None):

        super().__init__()

        # Load clip model
        self.clip, _ = clip.load("ViT-B/32", device=device)

        # Store the components
        self.event_encoder = event_encoder
        self.text_transformer = self.clip.transformer.float()
        self.token_embedding = self.clip.token_embedding
        self.positional_embedding = self.clip.positional_embedding
        self.ln_final = nn.LayerNorm(self.text_transformer.width)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / temperature))

        # Check the validity of the input
        if use_gate and n_prompts == 0:
            raise ValueError("Using gating is possible only if the number of prompts greater than zero.")
        if n_prompts < 0:
            raise ValueError("The number of prompts must be non-negative.")

        # Set learnable prompts
        self.use_gate = use_gate
        self.n_prompts = n_prompts
        if self.n_prompts > 0:
            self.prompt_weights = nn.Parameter(torch.randn(n_prompts, self.token_embedding.weight.shape[-1]))
        else:
            self.prompt_weights = None
        if self.use_gate:
            self.gate_weights = nn.Parameter(torch.zeros(1, self.token_embedding.weight.shape[-1]))
            nn.init.constant_(self.gate_weights, -1)

        # Projection heads to align event/text embeddings
        self.event_proj = nn.Linear(self.event_encoder.num_features[-1], projection_dim)
        self.text_proj = nn.Linear(self.text_transformer.width, projection_dim)

        # Freeze CLIP and put it in eval mode
        self.clip.eval()
        for param in self.clip.parameters():
            param.requires_grad = False

        # Move the components to the device if provided
        if device is not None:
            self.to(device=device)

    @property
    def dtype(self):
        return list(self.event_encoder.parameters())[0].dtype

    @property
    def device(self):
        return next(self.parameters()).device

    def to(self, device: torch.device):
        self.event_encoder = self.event_encoder.to(device)
        self.text_transformer = self.text_transformer.to(device)
        self.token_embedding = self.token_embedding.to(device)
        self.positional_embedding = self.positional_embedding.to(device)
        self.ln_final = self.ln_final.to(device)
        self.logit_scale = nn.Parameter(self.logit_scale.to(device))
        self.event_proj = self.event_proj.to(device)
        self.text_proj = self.text_proj.to(device)

        #TODO
        if self.prompt_weights is not None:
            self.prompt_weights = nn.Parameter(self.prompt_weights.to(device))
        if self.use_gate:
            self.gate_weights = nn.Parameter(self.gate_weights.to(device))

        return self

    def encode_events(self,
                      event_representations: torch.Tensor) -> torch.Tensor:
        """
        Encodes event data into the shared embedding space.

        Args:
            event_representations (torch.Tensor): Input event data with shape
                [batch_size, max_event_streams_per_sample, token_dim, grid_height, grid_width]

        Returns:
            torch.Tensor: Projected embeddings in contrastive space with shape
                [batch_size * max_event_streams_per_sample, projection_dim]
        """

        # Flatten batch and event stream dimensions for parallel processing
        # [batch_size x max_event_streams_per_sample, token_dim, grid_height, grid_width]
        event_tokens = rearrange(event_representations, "B N C H W -> (B N) C H W")

        # Process event tokens through event vision model
        # [batch_size x max_event_streams_per_sample, hidden_size]
        event_embeddings = self.event_encoder(event_tokens)

        # Pool the event tokens
        # [batch_size x max_event_streams_per_sample, hidden_size, 1]
        pooling = nn.AdaptiveAvgPool1d(output_size=1)
        event_embeddings = pooling(event_embeddings.transpose(dim0=1, dim1=2))

        # Project to shared space
        # [batch_size x max_event_streams_per_sample, projection_dim]
        projection = self.event_proj(event_embeddings.flatten(start_dim=1))

        return projection

    def encode_text(self,
                    text: list[str]) -> torch.Tensor:
        """
        Encodes text into the shared embedding space.

        Args:
            text (list[str]): List of text strings to encode.

        Returns:
            torch.Tensor: Projected text embeddings with shape
                [batch_size, projection_dim].
        """

        # Tokenize and move to device
        text_tokens = clip.tokenize(text).to(self.device)

        # Get token embeddings and add positional encoding
        token_embeddings = self.token_embedding(text_tokens)
        token_embeddings = token_embeddings + self.positional_embedding
        # TODO
        modified = token_embeddings.clone()
        if self.use_gate:
            gate = torch.sigmoid(self.gate_weights)
            modified[:, 1:self.n_prompts + 1] = (1 - gate) * token_embeddings[:, 1:self.n_prompts + 1] + gate * self.prompt_weights.unsqueeze(0)
        elif self.n_prompts > 0:
            modified[:, 1:self.n_prompts + 1] += self.prompt_weights.unsqueeze(0)

        # Process through transformer (LND format)
        token_embeddings = modified.permute(1, 0, 2)
        #token_embeddings = token_embeddings.permute(1, 0, 2)  # NLD → LND
        text_embeddings = self.text_transformer(token_embeddings)
        text_embeddings = text_embeddings.permute(1, 0, 2)  # LND → NLD

        # Final layer norm and extract features at EOT token position
        text_embeddings = self.ln_final(text_embeddings)
        text_embeddings = text_embeddings[torch.arange(text_embeddings.shape[0]),
                                          text_tokens.argmax(dim=-1)]

        # Project to shared embedding space
        projection = self.text_proj(text_embeddings)

        return projection

    def extract_features(self,
                         event_representations: torch.Tensor) -> torch.Tensor:
        """
        Extract features from event representations (before projection head).
        Used for linear evaluation by freezing the backbone and training a classifier on top.

        Args:
            event_representations (torch.Tensor): Input event data tensor with shape
                [batch_size, max_event_streams_per_sample, token_dim, grid_height, grid_width]

        Returns:
            torch.Tensor: Feature embeddings with shape
                [batch_size * max_event_streams_per_sample, hidden_size]

        Note:
            - Automatically sets model to eval mode during feature extraction
            - Disables gradient computation via @torch.no_grad()
            - Preserves original training mode when exiting
        """

        # Store original mode
        original_mode = self.training

        try:
            # Set to eval mode
            self.eval()

            # Flatten batch and event stream dimensions for parallel processing
            # [batch_size x max_event_streams_per_sample, token_dim, grid_height, grid_width]
            event_tokens = rearrange(event_representations, "B N C H W -> (B N) C H W")

            # Process event tokens through event vision model
            # [batch_size x max_event_streams_per_sample, num_event_tokens, hidden_size]
            event_embeddings = self.event_encoder(event_tokens)

            # Pool the event tokens
            # [batch_size x max_event_streams_per_sample, hidden_size, 1]
            pooling = nn.AdaptiveAvgPool1d(output_size=1)
            event_embeddings = pooling(event_embeddings.transpose(dim0=1, dim1=2))

            return event_embeddings.flatten(start_dim=1)
        finally:
            # Restore original mode
            self.train(original_mode)

    def forward(self,
                event_representations: torch.Tensor,
                text: list[str],
                **kwargs) -> torch.Tensor:
        """
        Computes normalized similarity scores between event and text embeddings.

            Args:
                event_representations (torch.Tensor): Input event data with shape
                    [batch_size, max_event_streams_per_sample, token_dim, grid_height, grid_width]
                text (list[str]): List of text strings (length batch_size).

            Returns:
                torch.Tensor: Similarity matrix with shape [batch_size, batch_size],
                    where matrix[i,j] = cos_sim(event_i, text_j) / temperature.
        """

        event_features = self.encode_events(event_representations)
        text_features = self.encode_text(text)

        # Normalize features
        event_features = F.normalize(event_features, p=2, dim=-1)
        text_features = F.normalize(text_features, p=2, dim=-1)

        # Similarity matrix
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * event_features @ text_features.T

        return logits


def build_model(config: dict,
                device: torch.device = None) -> EventCLIP:
    """
    Builds and configures the complete EventCLIP model.

    Args:
        config (dict): Dictionary containing the configuration of the experiment
        device (torch.device): Device where the model is located.

    Returns:
        EventCLIP: Fully configured EventCLIP model instance

    Raises:
        ValueError: If the type of the models was not recognised.
    """

    def _get_event_encoder(model_config: dict) -> GET:
        """
        Builds the event encoder from config.

        Args:
            model_config (dict): Dictionary containing the configuration of the event vision backbone

        Returns:
            GET: Loaded instance of the event encoder
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

    # Load event encoder
    event_encoder = _get_event_encoder(model_config=config['event_vision_model'])

    # Assemble the final model
    return EventCLIP(event_encoder=event_encoder,
                     projection_dim=config['event_vision_model']['projection_dim'],
                     temperature=config['train']['objective']['temperature'],
                     device=device)
