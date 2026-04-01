from typing import Union
from einops import rearrange

import torch
import torch.nn as nn

from GET_Transformer.models.GET import GET


class EventCLR(nn.Module):
    def __init__(self,
                 backbone: GET,
                 projection_dim: int = 128,
                 device: torch.device = None):
        """
        Event Contrastive Learning Representation (EventCLR) model.
        This model learns representations from event-based data using a contrastive learning framework (SimCLR-style).

        Args:
            backbone (GET): Backbone model for feature extraction. Must have num_features attribute
            projection_dim (int, optional): Dimension of the projected embedding space. Defaults to 128.
            device (torch.device, optional): Target device (CPU/GPU). Defaults to None.

        Attributes:
            backbone (GET): Feature extraction backbone.
            pooling (nn.AdaptiveAvgPool1d): Pooling layer to aggregate event token embeddings.
            head (nn.Sequential): Projection head (MLP) for contrastive learning.
        """

        super().__init__()

        # Store the components
        self.backbone = backbone
        self.pooling = nn.AdaptiveAvgPool1d(output_size=1)
        self.head = nn.Linear(backbone.num_features[-1], projection_dim)

        # Move the components to the device if provided
        if device is not None:
            self.to(device=device)

    @property
    def device(self):
        return next(self.parameters()).device

    def to(self, device: torch.device):
        self.backbone = self.backbone.to(device)
        self.pooling = self.pooling.to(device)
        self.head = self.head.to(device)
        return self

    def _forward(self,
                 event_representations: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for processing event representations through the backbone and projection head.

        Args:
            event_representations (torch.Tensor): Input event data tensor with shape
                [batch_size, max_event_streams_per_sample, token_dim, grid_height, grid_width]

        Returns:
            torch.Tensor: Projected embeddings in contrastive space with shape
                [batch_size * max_event_streams_per_sample, projection_dim]
        """

        # Flatten batch and event stream dimensions for parallel processing
        # [batch_size x max_event_streams_per_sample, token_dim, grid_height, grid_width]
        event_tokens = rearrange(event_representations, "B N C H W -> (B N) C H W")

        # Process event tokens through event vision model
        # [batch_size x max_event_streams_per_sample, num_event_tokens, hidden_size]
        event_embeddings = self.backbone(event_tokens)

        # Pool the event tokens
        # [batch_size x max_event_streams_per_sample, hidden_size, 1]
        event_embeddings = self.pooling(event_embeddings.transpose(dim0=1, dim1=2))

        # Project to contrastive space
        # [batch_size x max_event_streams_per_sample, projection_dim]
        projection = self.head(event_embeddings.flatten(start_dim=1))

        return projection

    def forward(self,
                anchor_representations: torch.FloatTensor,
                positive_representations: torch.FloatTensor,
                **kwargs) -> Union[tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """
        Forward pass for the EventCLR model.

        Args:
            anchor_representations (torch.Tensor): Processed anchor event data
                Shape: [batch_size, max_event_streams_per_sample, token_dim, grid_height, grid_width]
            positive_representations (torch.Tensor): Processed positive event data
                Shape: [batch_size, max_event_streams_per_sample, token_dim, grid_height, grid_width]
        Returns:
            tuple[torch.Tensor, torch.Tensor] or torch.Tensor: Projected embeddings of anchor-positive pairs for contrastive loss.
                Shape of each tensor: [batch_size * max_event_streams_per_sample, projection_dim].
        """

        anchor_proj = self._forward(anchor_representations)
        positive_proj = self._forward(positive_representations)

        return anchor_proj, positive_proj

    @torch.no_grad()
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
            event_embeddings = self.backbone(event_tokens)

            # Pool the event tokens
            # [batch_size x max_event_streams_per_sample, hidden_size, 1]
            event_embeddings = self.pooling(event_embeddings.transpose(dim0=1, dim1=2))

            return event_embeddings.flatten(start_dim=1)
        finally:
            # Restore original mode
            self.train(original_mode)


def build_model(config: dict,
                device: torch.device = None) -> EventCLR:
    """
    Builds and configures the complete EventCLR model.

    Args:
        config (dict): Dictionary containing the configuration of the experiment
        device (torch.device): Device where the model is located.

    Returns:
        EventCLR: Fully configured EventCLR model instance

    Raises:
        ValueError: If the type of the models was not recognised.
    """

    def _get_backbone(model_config: dict) -> GET:
        """
        Builds the event vision backbone from config.

        Args:
            model_config (dict): Dictionary containing the configuration of the event vision backbone

        Returns:
            GET: Loaded instance of the event vision backbone
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

    # Load backbone model
    backbone = _get_backbone(model_config=config['event_vision_model'])

    # Assemble the final model
    return EventCLR(backbone=backbone,
                    projection_dim=config['event_vision_model']['projection_dim'],
                    device=device)
