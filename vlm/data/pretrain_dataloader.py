from typing import List, Tuple
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset, DataLoader, Sampler

from data.processor import DictOutput
from data.event_tokenizer import EventTokenizer


@dataclass
class AugmentedBatch(DictOutput):
    """
    Container for a batch of anchor and positive pairs of augmented event sequences with
    device transfer capabilities.
    Handles the storage and device management of multiple event sequence tensors.

    Attributes:
        anchors (List[torch.Tensor]): List of anchor event camera streams in format (t, x, y, p)
                        Shape of an element in the list: [1, num_events, 4]
        positives (List[torch.Tensor]): List of positive event camera streams in format (t, x, y, p)
                        Shape of an element in the list: [1, num_events, 4]
    """

    anchors: List[torch.Tensor]  # Shape of each element: [1, num_events, 4]
    positives: List[torch.Tensor]  # Shape of each element: [1, num_events, 4]

    def __len__(self) -> int:
        return len(self.anchors)

    def to(self,
           device: torch.device = None,
           dtype: torch.dtype = torch.float32):
        """
        Transfers all tensors to the specified device with optional type conversion.

        Args:
            device (torch.device or str): Target device (e.g., 'cuda', 'cpu')
            dtype (torch.dtype, optional): Data type for the event stream data.
                                           Defaults to torch.float32.
        """

        # Set the device in case None is provided
        if device is None:
            device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        move_list = lambda list_, device_, dtype_: [x.to(device=device_, dtype=dtype_)
                                                    for x in list_] if list_ is not None else None

        self.anchors = move_list(self.anchors, device, dtype)
        self.positives = move_list(self.positives, device, dtype)

        return self


@dataclass
class TokenizedBatch(DictOutput):
    """
    Container for a batch of anchor and positive event pairs with their tokenized representations.
    Handles the storage and device management of event sequences along with their tokenized representations.

    Attributes:
        anchors (List[torch.Tensor]): List of anchor event camera streams in format (t, x, y, p)
                        Shape of an element in the list: [1, num_events, 4]
        positives (List[torch.Tensor]): List of positive event camera streams in format (t, x, y, p)
                        Shape of an element in the list: [1, num_events, 4]
        anchor_representations (torch.Tensor): Tensor containing the representations
                        of the tokenized anchor events. Shape: [batch_size, 1, token_dim, grid_height, grid_width]
        positive_representations (torch.Tensor): Tensor containing the representations
                        of the tokenized positive events. Shape: [batch_size, 1, token_dim, grid_height, grid_width]
    """

    anchors: List[torch.Tensor]  # Anchor sequences
    positives: List[torch.Tensor]  # Positive sequences
    anchor_representations: torch.Tensor  # Shape: [batch_size, 1, token_dim, grid_height, grid_width]
    positive_representations: torch.Tensor  # Shape: [batch_size, 1, token_dim, grid_height, grid_width]

    def __len__(self) -> int:
        return len(self.anchors)

    def to(self,
           device: torch.device = None,
           dtype: torch.dtype = torch.float32):
        """
        Transfers all tensors to the specified device with optional type conversion.

        Args:
            device (torch.device or str): Target device (e.g., 'cuda', 'cpu')
            dtype (torch.dtype, optional): Data type for the event stream data.
                                           Defaults to torch.float32.
        """

        # Set the device in case None is provided
        if device is None:
            device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        move_tensor = lambda tensor_, device_: tensor_.to(device=device_) if tensor_ is not None else None
        move_list = lambda list_, device_, dtype_: [x.to(device=device_, dtype=dtype_)
                                                    for x in list_] if list_ is not None else None

        self.anchors = move_list(self.anchors, device, dtype)
        self.positives = move_list(self.positives, device, dtype)

        self.anchor_representations = move_tensor(self.anchor_representations, device)
        self.positive_representations = move_tensor(self.positive_representations, device)

        return self


class AugmentedCollator:
    """
    Collates multiple augmented views for contrastive learning.
    """

    def __call__(self,
                 batch: List[Tuple[torch.Tensor, torch.Tensor]]):

        if not batch:
            raise ValueError("Cannot collate an empty batch.")

        # Extract data from the batch
        anchors, positives = zip(*batch)

        return AugmentedBatch(anchors=[events.unsqueeze(0) for events in anchors],
                              positives=[events.unsqueeze(0) for events in positives])


def build_dataloader(event_dataset: Dataset,
                     event_tokenizer: EventTokenizer,
                     batch_size: int = 64,
                     num_workers: int = 8,
                     local_rank: int = 0,
                     shuffle: bool = True,
                     sampler: Sampler = None,
                     drop_last: bool = True):
    """
    Builds a tokenizing dataloader for contrastive learning with event data.

    Creates a DataLoader that:
    1. Collates the augmented data
    2. Automatically tokenizes event data using the specified tokenizer
    3. Manages device placement (CPU/GPU)
    4. Supports distributed training environments

    Args:
        event_dataset (Dataset): Dataset containing event sequences for contrastive learning
        event_tokenizer (EventTokenizer): Tokenizer for converting raw events to model inputs
        batch_size (int, optional): Number of samples per batch. Defaults to 64.
        num_workers (int, optional): Number of subprocesses for data loading. Defaults to 8.
        local_rank (int, optional): The rank (index) of the current process within its local machine.
                                    Defaults to 0.
        shuffle (bool, optional): Whether to shuffle the data. Defaults to True.
        sampler (Sampler, optional): Distributed sampler instance. Defaults to None.
        drop_last (bool, optional): Whether to drop incomplete batches. Defaults to True.

    Returns:
        TokenizedDataLoader: A DataLoader-like iterator that yields TokenizedBatch objects containing:
            - anchors (list): List of raw anchor event sequences
            - positives (list): List of raw positive event sequences
            - anchor_representations (torch.Tensor): Tokenized anchor representations.
                                        Shape: [batch_size, 1, token_dim, grid_height, grid_width]
            - positive_representations (torch.Tensor): Tokenized positive representations
                                        Shape: [batch_size, 1, token_dim, grid_height, grid_width]

    Note:
        - In distributed mode, shuffling should be handled by the sampler
        - Tokenization happens in the main thread to avoid GPU contention
        - Output batches are automatically moved to the appropriate device
    """

    # Define data collator
    collate_fn = AugmentedCollator()

    # Create a data loader that handles data augmentation
    augmented_dataloader = DataLoader(event_dataset,
                                      batch_size=batch_size,
                                      collate_fn=collate_fn,
                                      num_workers=num_workers,
                                      shuffle=shuffle if sampler is None else False,
                                      sampler=sampler,
                                      drop_last=drop_last,
                                      persistent_workers=True,
                                      pin_memory=False)

    # Wrap the data loader to add data tokenization
    class TokenizedDataLoader:
        """
        Wrapper that adds tokenization to augmented batches in main thread.
        """

        def __init__(self,
                     augmented_dataloader: DataLoader,
                     event_tokenizer: EventTokenizer,
                     local_rank: int = 0):

            self.augmented_dataloader = augmented_dataloader
            self.sampler = augmented_dataloader.sampler
            self.event_tokenizer = event_tokenizer
            self.local_rank = local_rank

        def __iter__(self):
            for batch in self.augmented_dataloader:
                # Move the batch of augmented data to GPU
                device = f'cuda:{self.local_rank}'
                batch = batch.to(device)

                # Tokenize anchors and positives
                anchor_representations = torch.stack([self.event_tokenizer(sample)
                                                      for sample in batch.anchors], dim=0).to(device)

                positive_representations = torch.stack([self.event_tokenizer(sample)
                                                        for sample in batch.positives], dim=0).to(device)

                yield TokenizedBatch(anchors=batch.anchors,
                                     positives=batch.positives,
                                     anchor_representations=anchor_representations,
                                     positive_representations=positive_representations)

        def __len__(self):
            return len(self.augmented_dataloader)

    return TokenizedDataLoader(augmented_dataloader=augmented_dataloader,
                               event_tokenizer=event_tokenizer,
                               local_rank=local_rank)

# TODO:
# different convertion to video (histograms instead???)