import random
from typing import Union, List
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset, DataLoader, Sampler

from data.processor import DictOutput, EventVLProcessor


def get_classification_prompt(event_tag: str = "<image_placeholder>") -> str:
    """
    Generates a natural language prompt for visual classification task.

    Args:
        event_tag: String placeholder representing where the event should be inserted.
                  Defaults to "<image_placeholder>".

    Returns:
        str: A randomly selected classification prompt.
    """

    prompts = [f"What do you see? {event_tag}",
               f"Describe the object. {event_tag}",
               f"Which object is shown? {event_tag}",
               f"Identify this item: {event_tag}",
               f"Identify the subject of {event_tag}.",
               f"What is depicted in this {event_tag}?",
               f"{event_tag} Describe the main subject of this event stream.",
               f"{event_tag} What category does this event data belong to?",
               f"{event_tag} Recognize the object from these events."]

    return random.choice(prompts)


def get_label_description(idx: Union[int, torch.Tensor]) -> str:
    """
    Generate a classification label description for a given class IDs.

    Args:
        idx (Union[int, torch.Tensor]): A single class ID.

    Returns:
        str: A randomly selected label description of the class.
    """

    prefixes = ['This is ', 'The answer is ', 'I see ', 'It appears to be ',
                'That looks like ', '', 'My guess is ', 'It resembles ',
                'After analyzing, I say it is ']
    class_descs = [["an airplane.",
                    "a powered aircraft with fixed wings and engines.",
                    "a flying vehicle with wings, which is called an airplane.",
                    "an airplane, a large, mechanical vehicle designed for air travel with a fuselage, wings, and tail."
                    ],
                    ["an automobile.",
                     "a road vehicle with four wheels powered by an engine.",
                     "a motor vehicle for land transport, which is called an automobile.",
                     "a wheeled vehicle typically used for personal transport, containing an enclosed cabin and seating."
                    ],
                    ["a bird.",
                     "a bird, warm-blooded animal with feathers and a beak.",
                     "a feathered animal with wings, which is called a bird.",
                     "a vertebrate with wings, feathers, and typically the ability to fly, belonging to the class Aves."
                    ],
                    ["a cat.",
                     "a cat, a four-legged feline pet with retractable claws.",
                     "a small domestic mammal with fur, which is called a cat.",
                     "a domesticated animal of the Felidae family, known for its agility, soft coat, and independent behavior."
                    ],
                    ["a deer.",
                     "a hoofed mammal with antlers which is called a deer.",
                     "a deer, a grazing animal found in forests or fields.",
                     "a ruminant mammal with long legs and, in many species, branched antlers found mainly on males."
                    ],
                    ["a dog.",
                     "a domesticated four-legged mammal, which is called a dog.",
                     "a dog, a furry animal often kept as a pet or working companion.",
                     "a canid species selectively bred for traits like loyalty, obedience, and companionship."
                    ],
                    ["a frog.",
                     "a small amphibian with long legs, which is called a frog.",
                     "a frog, a cold-blooded animal with moist skin and webbed feet.",
                     "an amphibian belonging to the order Anura, known for its smooth skin, bulging eyes, and jumping ability."
                    ],
                    ["a horse.",
                     "a large, four-legged hoofed animal, which is called a horse.",
                     "a horse, an animal with a mane, used historically for riding and labor.",
                     "a domesticated mammal of the Equidae family, recognized for its muscular build, long legs, and mane."
                    ],
                    ["a ship.",
                     "a large watercraft for ocean travel, which is called a ship.",
                     "a ship, a vessel designed to carry goods or passengers over water.",
                     "a large, engineered structure that floats on water and is used for marine transportation."
                    ],
                    ["a truck.",
                     "a truck, a motor vehicle designed to transport cargo.",
                     "a heavy vehicle used on roads, which is called a truck.",
                     "a road vehicle with a strong frame and large cargo area, typically used in logistics and transport."
                    ]]

    if isinstance(idx, int) or (isinstance(idx, torch.Tensor) and idx.dim() == 0):
        return random.choice(prefixes) + random.choice(class_descs[idx])

    raise TypeError("The input should be int or 0-dim tensor.")


@dataclass
class MultimodalSample(DictOutput):
    """
    A single multimodal sample containing event data paired with text annotations.

    Attributes:
        event_data (torch.Tensor): Raw event camera stream in format (t, x, y, p)
            Shape: [num_events, 4] where each event contains:
                - t: Timestamp
                - x: X coordinate (pixel position)
                - y: Y coordinate (pixel position)
                - p: Polarity (-1 or 1 for negative/positive change)
        prompt (str): Natural language instruction/prompt containing an event placeholder
        caption (str): Natural language description of the event sequence
    """

    event_data: torch.Tensor
    prompt: str
    caption: str


class MultimodalDataset(Dataset):
    """
    Event-language dataset containing raw event data, captioning prompts and captions describing
    the corresponding event sequence.
    """

    def __init__(self,
                 event_text_dataset: Dataset):
        self.event_text_dataset = event_text_dataset

    def __len__(self):
        return len(self.event_text_dataset)

    def __getitem__(self,
                    idx: int) -> MultimodalSample:
        """
        Retrieves a single multimodal sample.

        Args:
            idx: Index of the sample to retrieve

        Returns:
            MultimodalSample: Container holding unprocessed data sample with:
                - event_data (torch.Tensor): Raw event camera stream in format (t, x, y, p)
                                    Shape: [num_events, 4]
                - prompt (str): Natural language instruction/prompt containing an event placeholder
                - caption (str): Natural language description of the event sequence

         Note:
            The prompt generation uses random selection from predefined templates to ensure diversity
            in the training data.
        """

        # Get event data and caption from the underlying dataset
        event_data, caption = self.event_text_dataset[idx]

        # Generate a random classification prompt with an event placeholder
        prompt = get_classification_prompt()

        return MultimodalSample(event_data=event_data,
                                prompt=prompt,
                                caption=caption)


@dataclass
class RawBatch(DictOutput):
    """
    Container for holding raw, unprocessed batch data before applying transformations.

    This dataclass serves as an intermediate storage structure that collects raw data samples
    from multiple workers before processing in the main process. It maintains the original
    structure of the data without any model-specific transformations.

    Attributes:
        event_data_list (List[torch.Tensor]): List of raw event camera stream in format (t, x, y, p)
                         Shape of an element in the list: [1, num_events, 4]
        prompt_list (List[str]): List of natural language instructions/prompts containing an event placeholders
        caption_list (List[str]): List of natural language descriptions of the event sequences
    """

    event_data_list: List[torch.Tensor]
    prompt_list: List[str]
    caption_list: List[str]

    def __len__(self) -> int:
        return len(self.event_data_list)

    def to(self,
           device: torch.device = None,
           dtype: torch.dtype = torch.float32):
        """
        Transfers all tensors to the specified device with optional type conversion.

        Args:
            device (torch.device or str): Target device (e.g., 'cuda', 'cpu')
            dtype (torch.dtype, optional): Data type for raw event data.
                                           Defaults to torch.float32.
        """

        # Set the device in case None is provided
        if device is None:
            device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        move_list = lambda list_, device_, dtype_: [x.to(device=device_, dtype=dtype_)
                                                    for x in list_] if list_ is not None else None

        self.event_data_list = move_list(self.event_data_list, device, dtype)

        return self


class RawBatchCollator:
    """
    Collates raw data samples without applying any processor transformations.
    """

    def __call__(self,
                 batch: List):
        """
        Collates a list of samples into a raw batch container.

        Args:
            batch (List[MultimodalSample]): List of unprocessed data samples.

        Returns:
            RawBatch: Container holding batched but unprocessed data with:
                - event_data_list (List[torch.Tensor]): List of raw event camera stream in format (t, x, y, p)
                             Shape of an element in the list: [1, num_events, 4]
                - prompt_list (List[str]): List of natural language instructions/prompts containing an event placeholders
                - caption_list (List[str]): List of natural language descriptions of the event sequences
        """

        if not batch:
            raise ValueError("Cannot collate an empty batch.")

        # Gather all fields from batch samples
        event_data_list = [sample.event_data.unsqueeze(0) for sample in batch]
        prompt_list = [sample.prompt for sample in batch]
        caption_list = [sample.caption for sample in batch]

        return RawBatch(event_data_list=event_data_list,
                        prompt_list=prompt_list,
                        caption_list=caption_list)


@dataclass
class ProcessedBatch(DictOutput):
    """
    A fully processed and batched training sample ready for model input.

    Attributes:
        sft_format (List[str]): List of formatted prompt strings
        input_ids (torch.Tensor): Padded tokenized text with event token placeholders
            Shape: [batch_size, max_seq_len]
        event_representations (torch.Tensor): Processed event data
            Shape: [batch_size, 1, token_dim, grid_height, grid_width]
        attention_mask (torch.Tensor): Attention mask for text and event tokens
            Shape: [batch_size, max_seq_len]
        event_seq_mask (torch.BoolTensor): Mask indicating event token positions
            Shape: [batch_size, max_seq_len]
        event_emb_mask (torch.BoolTensor): Mask for valid event tokens
            Shape: [batch_size, 1, num_event_tokens]
        target_ids (torch.Tensor): Tokenized captions
            Shape: [batch_size, max_seq_len]
        captions (List[str]): List of natural language descriptions of the event sequences
    """

    sft_format: List[str]
    input_ids: torch.LongTensor
    event_representations: torch.Tensor
    attention_mask: torch.Tensor
    event_seq_mask: torch.BoolTensor
    event_emb_mask: torch.BoolTensor
    target_ids: torch.Tensor
    captions: List[str]

    def __len__(self) -> int:
        return len(self.input_ids)

    def to(self, device=None, dtype=torch.float32):
        """
        Transfers all tensors to the specified device with optional type conversion.

        Args:
            device (torch.device or str): Target device (e.g., 'cuda', 'cpu')
            dtype (torch.dtype, optional): Data type for event representations.
                                           Defaults to torch.float32.
        """

        # Set the device in case None is provided
        if device is None:
            device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        self.input_ids = self.input_ids.to(device)
        self.event_representations = self.event_representations.to(device=device, dtype=dtype)
        self.attention_mask = self.attention_mask.to(device)
        self.event_seq_mask = self.event_seq_mask.to(device)
        self.event_emb_mask = self.event_emb_mask.to(device)
        self.target_ids = self.target_ids.to(device)

        return self


class ProcessedBatchCollator:
    """
    Applies full processing pipeline to raw batches in the main process.

    This collator takes raw batches collected by workers and applies:
    1. Event data and prompt processing through EventVLProcessor
    2. Tokenization for natural language descriptions of the event sequences
    3. Proper batching and padding of all components

    Attributes:
        processor (EventVLProcessor): Processor instance for handling both event data and text for multimodal models
        pad_id (int): Padding token ID from the processor's vocabulary
    """

    def __init__(self,
                 processor: EventVLProcessor):
        self.processor = processor
        self.pad_id = processor.pad_id

    def __call__(self,
                 raw_batch: RawBatch):
        """
        Transforms raw samples into fully processed, model-ready batch.

        Args:
            raw_batch (RawBatch): Unprocessed batch data collected by workers.

        Returns:
            ProcessedBatch: Fully processed batch containing:
                - sft_format (List[str]): List of formatted prompt strings
                - input_ids (torch.Tensor): Padded tokenized text with event token placeholders
                            Shape: [batch_size, max_seq_len]
                - event_representations (torch.Tensor): Processed event data
                            Shape: [batch_size, 1, token_dim, grid_height, grid_width]
                - attention_mask (torch.Tensor): Attention mask for text and event tokens
                            Shape: [batch_size, max_seq_len]
                - event_seq_mask (torch.BoolTensor): Mask indicating event token positions
                            Shape: [batch_size, max_seq_len]
                - event_emb_mask (torch.BoolTensor): Mask for valid event tokens
                            Shape: [batch_size, 1, num_event_tokens]
                - target_ids (torch.Tensor): Tokenized captions
                            Shape: [batch_size, max_seq_len]
                - captions (List[str]): List of natural language descriptions of the event sequences
        """

        # Compute the batch size
        batch_size = len(raw_batch)

        # Conversations remain empty, only prompts are used
        conversations_list = [None] * batch_size

        # Process the data
        batched_prepares = self.processor.process_batch(prompt_list=raw_batch.prompt_list,
                                                        conversations_list=conversations_list,
                                                        event_data_list=raw_batch.event_data_list)

        # Get the target ids
        target_ids_list = self.processor.language_tokenizer(raw_batch.caption_list)['input_ids']

        # Batchify the target ids
        max_seq_len = batched_prepares.input_ids.shape[1]
        batched_target_ids = torch.full((batch_size, max_seq_len),
                                        self.pad_id,
                                        dtype=torch.long)
        for i, target_ids in enumerate(target_ids_list):
            seq_len = min(len(target_ids), max_seq_len)   # Truncate the target in case it is too long
            target_ids = torch.tensor(target_ids,
                                      dtype=torch.long)
            batched_target_ids[i, -seq_len:] = target_ids[-seq_len:]

        return ProcessedBatch(input_ids=batched_prepares.input_ids,
                              attention_mask=batched_prepares.attention_mask,
                              event_representations=batched_prepares.event_representations,
                              event_seq_mask=batched_prepares.event_seq_mask,
                              event_emb_mask=batched_prepares.event_emb_mask,
                              sft_format=batched_prepares.sft_format,
                              target_ids=batched_target_ids,
                              captions=raw_batch.caption_list)


def build_dataloader(event_text_dataset: Dataset,
                     processor: EventVLProcessor,
                     batch_size: int = 64,
                     num_workers: int = 8,
                     local_rank: int = 0,
                     shuffle: bool = True,
                     sampler: Sampler = None,
                     drop_last: bool = True):
    """
    Creates a DataLoader pipeline for multimodal event-text data processing.

    Args:
        event_text_dataset (Dataset): Dataset containing raw event data (t, x, y, p) and corresponding captions
        processor (EventVLProcessor): Processor for handling both event data and text for multimodal models
        batch_size (int, optional): Batch size for data loading. Defaults to 64.
        num_workers (int, optional): Number of workers for data loading. Defaults to 8.
        local_rank (int, optional): The rank (index) of the current process within its local machine.
                                    Defaults to 0.
        shuffle (bool, optional): Whether to shuffle the data before batching. Defaults to True.
        sampler (Sampler, optional): Distributed sampler if using DDP. Defaults to None.
        drop_last (bool, optional): Whether to drop incomplete batches. Defaults to True.

    Returns:
        ProcessedDataLoader: A DataLoader-like iterator that yields processed batches
    """

    # Wrap raw event data with text generation capabilities
    multimodal_dataset = MultimodalDataset(event_text_dataset=event_text_dataset)

    # Create a data loader that handles raw data collection
    raw_dataloader = DataLoader(multimodal_dataset,
                                batch_size=batch_size,
                                collate_fn=RawBatchCollator(),
                                num_workers=num_workers,
                                shuffle=shuffle if sampler is None else False,
                                sampler=sampler,
                                drop_last=drop_last,
                                persistent_workers=True,
                                pin_memory=False)

    # Wrap the data loader to add data processing
    class ProcessedDataLoader:
        """
        Wrapper that adds processing to raw batches in main thread.
        """

        def __init__(self,
                     raw_dataloader: DataLoader,
                     processor: EventVLProcessor,
                     local_rank: int = 0):

            self.raw_dataloader = raw_dataloader
            self.sampler = raw_dataloader.sampler
            self.processor = processor
            self.local_rank = local_rank

            # Collator handles all CUDA operations safely in main process
            self.processed_collator = ProcessedBatchCollator(processor=processor)

        def __iter__(self):
            for raw_batch in self.raw_dataloader:

                # Move the batch of unprocessed data to GPU
                device = f'cuda:{self.local_rank}'
                raw_batch = raw_batch.to(device)

                # Return a batch of processed data
                yield self.processed_collator(raw_batch=raw_batch)

        def __len__(self):
            return len(self.raw_dataloader)

    return ProcessedDataLoader(raw_dataloader=raw_dataloader,
                               processor=processor,
                               local_rank=local_rank)


# only one event stream per sample here