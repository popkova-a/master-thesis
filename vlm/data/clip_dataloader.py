from typing import List, Tuple
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset, DataLoader, Sampler

from data.processor import DictOutput
from data.event_tokenizer import EventTokenizer


@dataclass
class RawBatch(DictOutput):
    """
    A batch container for classification with event data and corresponding class labels.
    Handles the storage and device management of multiple event sequence tensors.

    Attributes:
        event_data_list (List[torch.Tensor]): List of raw event camera stream in format (t, x, y, p)
                         Shape of an element in the list: [1, num_events, 4]
        labels (torch.LongTensor): Tensor of ground truth class indices for the classification task.
    """

    event_data_list: List[torch.Tensor]  # Shape of each element: [1, num_events, 4]
    labels: torch.LongTensor

    def __len__(self) -> int:
        return len(self.event_data_list)

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

        self.event_data_list = move_list(self.event_data_list, device, dtype)
        self.labels = move_tensor(self.labels, device)

        return self


class RawBatchCollator:
    """
    Collates events and corresponding labels for classification task.
    """

    def __call__(self,
                 batch: List[Tuple[torch.Tensor, int]]):

        if not batch:
            raise ValueError("Cannot collate an empty batch.")

        # Extract data from the batch
        event_data_list, labels = zip(*batch)

        return RawBatch(event_data_list=[events.unsqueeze(0) for events in event_data_list],
                        labels=torch.tensor(labels, dtype=torch.long))


@dataclass
class TokenizedBatch(DictOutput):
    """
    Container for a batch of event streams with their tokenized representations.
    Handles the storage and device management of event sequences along with their tokenized representations.

    Attributes:
       event_data_list (List[torch.Tensor]): List of raw event camera stream in format (t, x, y, p)
                         Shape of an element in the list: [1, num_events, 4]
       event_representations (torch.Tensor): Tensor containing the representations
                        of the tokenized events. Shape: [batch_size, 1, token_dim, grid_height, grid_width]
       labels (torch.LongTensor): Tensor of ground truth class indices for the classification task.
    """

    event_data_list: List[torch.Tensor]
    event_representations: torch.Tensor  # Shape: [batch_size, 1, token_dim, grid_height, grid_width]
    labels: torch.LongTensor

    def __len__(self) -> int:
        return len(self.event_data_list)

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

        self.event_data_list = move_list(self.event_data_list, device, dtype)
        self.event_representations = move_tensor(self.event_representations, device)
        self.labels = move_tensor(self.labels, device)

        return self


def build_dataloader(event_dataset: Dataset,
                     event_tokenizer: EventTokenizer,
                     batch_size: int = 64,
                     num_workers: int = 8,
                     local_rank: int = 0,
                     shuffle: bool = True,
                     sampler: Sampler = None,
                     drop_last: bool = True):
    """
    Builds a tokenizing dataloader for CLIP-style contrastive learning with event data.

    Creates a DataLoader that:
    1. Automatically tokenizes event data using the specified tokenizer
    2. Manages device placement (CPU/GPU)
    3. Supports distributed training environments

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
        TokenizedDataLoader: A DataLoader-like iterator that yields TokenizedCLIPBatch objects containing:
            - event_data_list (List[torch.Tensor]): List of raw event camera stream in format (t, x, y, p)
                             Shape of an element in the list: [1, num_events, 4]
            - event_representations (torch.Tensor): Tensor containing the representations
                             of the tokenized events. Shape: [batch_size, 1, token_dim, grid_height, grid_width]
            - labels (torch.LongTensor): Tensor of ground truth class indices for the classification task.

    Note:
        - In distributed mode, shuffling should be handled by the sampler
        - Tokenization happens in the main thread to avoid GPU contention
        - Output batches are automatically moved to the appropriate device
    """

    # Define data collator
    collate_fn = RawBatchCollator()

    # Create a data loader that handles raw data collection
    raw_dataloader = DataLoader(event_dataset,
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
                     raw_dataloader: DataLoader,
                     event_tokenizer: EventTokenizer,
                     local_rank: int = 0):

            self.raw_dataloader = raw_dataloader
            self.sampler = raw_dataloader.sampler
            self.event_tokenizer = event_tokenizer
            self.classes = raw_dataloader.dataset.classes
            self.local_rank = local_rank

        def __iter__(self):
            for batch in self.raw_dataloader:
                # Move the batch of augmented data to GPU
                device = f'cuda:{self.local_rank}'
                batch = batch.to(device)

                # Tokenize events
                event_representations = torch.stack([self.event_tokenizer(sample)
                                                     for sample in batch.event_data_list], dim=0).to(device)

                yield TokenizedBatch(event_data_list=batch.event_data_list,
                                     event_representations=event_representations,
                                     labels=batch.labels)

        def __len__(self):
            return len(self.raw_dataloader)

    return TokenizedDataLoader(raw_dataloader=raw_dataloader,
                               event_tokenizer=event_tokenizer,
                               local_rank=local_rank)
