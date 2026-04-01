import torch
import torch.nn as nn

"""
Code reference: https://github.com/Peterande/GET-Group-Event-Transformer/blob/master/event_based/event_token.py
"""


class EventTokenizer(nn.Module):

    @torch.no_grad()
    def __init__(self,
                 ref_resolution: int = 128,
                 embed_split: int = 12,
                 patch_size: int = 4,
                 device: torch.device = None):

        super().__init__()
        self.ref_resolution = ref_resolution
        self.time_div = embed_split // 2
        self.patch_size = patch_size
        self.num_patches = ref_resolution // patch_size

        if ref_resolution <= 0 or embed_split <= 0 or patch_size <= 0:
            raise ValueError("Input characteristics cannot be zero or negative.")

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

    @torch.no_grad()
    def forward(self,
                events: torch.Tensor,
                eps: float = 1e-4) -> torch.Tensor:
        """
        Converts a single event stream into a token representation according to the GET paper.
        Automatically resizes the input to the reference resolution while preserving spatiotemporal
        relationships.

        Args:
            events (torch.Tensor): Raw event stream containing (t, x, y, p).
                                  Shape: [num_events, 4]
                                  Note: Use torch.inf for padding if needed.
            eps (float): Epsilon for numerical stability.

        Returns:
            torch.Tensor: Fixed-size token representation with shape [1, C, H, W] where:
                         - C = 2 * embed_split * (patch_size ** 2)  (feature channels)
                         - H = W = ref_resolution // patch_size   (grid dimensions)
        """

        events = events.to(self.device)

        # Remove padding
        events = events[events != torch.inf].reshape(-1, 4)

        # Initialize output tensor
        token_num = self.num_patches ** 2
        patch_area = int(self.patch_size ** 2)
        event_tokens = torch.zeros([self.time_div, 2, 2, patch_area, token_num],
                                   device=self.device)

        # Handle empty input
        if len(events) == 0:
            return event_tokens.reshape(1, -1, self.num_patches, self.num_patches)

        # Rescale to the reference resolution
        max_x = int(events[:, 1].max())
        max_y = int(events[:, 2].max())

        # Handle case where all events are at (0,0)
        if max_x == 0 or max_y == 0:
            scale_x = scale_y = 1.0
        else:
            scale_x = (self.ref_resolution - 1) / max(1, max_x)
            scale_y = (self.ref_resolution - 1) / max(1, max_y)

        events_scaled = events.clone()
        events_scaled[:, 1] = (events[:, 1] * scale_x).clamp(0, self.ref_resolution - 1)  # Scale x coordinate
        events_scaled[:, 2] = (events[:, 2] * scale_y).clamp(0, self.ref_resolution - 1)  # Scale y coordinate

        # Histogram weights
        weight_p = events_scaled[:, 3] != 2
        weight_t = torch.div(events_scaled[:, 0] - events_scaled[0, 0],
                             events_scaled[-1, 0] - events_scaled[0, 0] + eps)

        # Compute global position
        coord_to_patch = (self.num_patches - 1) / (self.ref_resolution - 1)
        grid_x = (events_scaled[:, 1] * coord_to_patch).long().clamp(0, self.num_patches - 1)
        grid_y = (events_scaled[:, 2] * coord_to_patch).long().clamp(0, self.num_patches - 1)
        grid_pos = grid_x + grid_y * self.num_patches

        # Compute local position
        local_x = (events_scaled[:, 1] % self.patch_size).long().clamp(0, self.patch_size - 1)
        local_y = (events_scaled[:, 2] % self.patch_size).long().clamp(0, self.patch_size - 1)
        patch_pos = local_x + local_y * self.patch_size

        # Time normalization
        time_double = events_scaled[:, 0].double()
        time_pos = torch.floor(self.time_div * torch.div(time_double - time_double[0],
                                                         time_double[-1] - time_double[0] + 1))

        # Polarity (convert to 0/1 if needed)
        polarity = events_scaled[:, 3].long().clamp(0, 1)

        # Verify all indices are within bounds
        assert (grid_pos < token_num).all(), "Grid position out of bounds"
        assert (patch_pos < patch_area).all(), "Patch position out of bounds"
        assert (time_pos < self.time_div).all(), "Time position out of bounds"

        # Mapping from 4-D to 1-D.
        bins = torch.as_tensor((self.time_div, 2, patch_area, token_num)).to(self.device)
        repr_4d = torch.stack((time_pos, polarity, patch_pos, grid_pos), dim=1).permute(1, 0).int()
        repr_1d, index = index_mapping(repr_4d, bins)

        # Get 1-D histogram which encodes the event tokens
        event_tokens[:, :, 0, :, :], event_tokens[:, :, 1, :, :] = get_repr(repr_1d, index,
                                                                            bins=bins,
                                                                            weights=[weight_p, weight_t])

        return event_tokens.reshape(1, -1, self.num_patches, self.num_patches)


def build_event_tokenizer(config: dict,
                          device: torch.device = None) -> EventTokenizer:
    """
    Constructs an EventTokenizer instance for converting an single event stream to a token representation.

    This factory function creates a tokenizer that transforms an asynchronous event stream into
    a fixed-size grid-based token representation suitable for transformer-based architectures,
    following the GET (Group Event Transformer) approach.

    Args:
        config (dict): Dictionary containing the configuration of the event tokenizer
        device (torch.device, optional): Target device for computation. If None, will
                                         automatically use CUDA if available. Defaults to None.

    Returns:
        EventTokenizer: Configured tokenizer instance capable of processing a single event stream.

    Note:
        The tokenizer performs the following key transformations:
           1. Scales input coordinates to the reference resolution
           2. Splits events into temporal groups
           3. Organizes events into spatial patches
           4. Creates separate channels for positive/negative polarities
           5. Outputs a fixed-size tensor regardless of input dimensions
    """

    data_config = config['data']
    model_parameters = config['event_vision_model']['parameters']

    return EventTokenizer(ref_resolution=data_config['ref_resolution'],
                          embed_split=model_parameters['embed_split'],
                          patch_size=model_parameters['patch_size'],
                          device=device)


def index_mapping(sample,
                  bins=None):
    """
    Multi-index mapping method from N-D to 1-D.
    """
    device = sample.device
    bins = torch.as_tensor(bins).to(device)
    y = torch.max(sample, torch.zeros([], device=device, dtype=torch.int32))
    y = torch.min(y, bins.reshape(-1, 1))
    index = torch.ones_like(bins)
    index[1:] = torch.cumprod(torch.flip(bins[1:], [0]), -1).int()
    index = torch.flip(index, [0])
    l = torch.sum((index.reshape(-1, 1)) * y, 0)
    return l, index


def get_repr(l,
             index,
             bins=None,
             weights=None):
    """
    Function to return histograms.
    """
    hist = torch.bincount(l, minlength=index[0] * bins[0], weights=weights[0])
    hist = hist.reshape(tuple(bins))
    if len(weights) > 1:
        hist2 = torch.bincount(l, minlength=index[0] * bins[0], weights=weights[1])
        hist2 = hist2.reshape(tuple(bins))
    else:
        return hist
    if len(weights) > 2:
        hist3 = torch.bincount(l, minlength=index[0] * bins[0], weights=weights[2])
        hist3 = hist3.reshape(tuple(bins))
    else:
        return hist, hist2
    if len(weights) > 3:
        hist4 = torch.bincount(l, minlength=index[0] * bins[0], weights=weights[3])
        hist4 = hist4.reshape(tuple(bins))
    else:
        return hist, hist2, hist3
    return hist, hist2, hist3, hist4


class E2IMG(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.args = args

    def forward(self, x):
        y = torch.stack([self.Ree(x[i]) for i in range(len(x))], dim=0)
        return y

    def Ree(self, x):
        """
        Convert events into images.
        """
        sz = self.args[0]
        y = 255 * torch.ones([3, int(sz[1]), int(sz[0])], dtype=x.dtype, device=x.device)
        if len(x):
            y[0, torch.floor(x[:, 2]).long(), torch.floor(x[:, 1]).long()] = 255 - 255 * (x[:, 3] == 1).to(
                dtype=y.dtype)
            y[1, torch.floor(x[:, 2]).long(), torch.floor(x[:, 1]).long()] = 255 - 255 * (x[:, 3] == 0).to(
                dtype=y.dtype)
            y[2] = y[0] + y[1]
        return y.permute(1, 2, 0)
