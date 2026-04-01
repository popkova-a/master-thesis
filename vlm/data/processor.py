from dataclasses import dataclass
from typing import List, Dict, Tuple

import torch
from deepseek_vl.models import VLChatProcessor
from transformers import LlamaTokenizerFast, AutoTokenizer

from data.event_tokenizer import EventTokenizer


# Register custom tokenizer
AutoTokenizer.register("EventTokenizer", EventTokenizer)


class DictOutput(object):
    """
    Base class that enables dictionary-like access to object attributes.
    """

    def keys(self):
        return self.__dict__.keys()

    def __getitem__(self, item):
        return self.__dict__[item]

    def __setitem__(self, key, value):
        self.__dict__[key] = value


@dataclass
class EventVLProcessorOutput(DictOutput):
    """
    Container for processed event and text data for a single sample.

    Attributes:
        sft_format (str): Formatted prompt string
        input_ids (torch.Tensor): Tokenized text with event token placeholders
                                  Shape: [text_length + num_event_tokens]
        event_representations (torch.Tensor): Processed event data
                                  Shape: [event_streams_per_sample, token_dim, grid_height, grid_width]
    """

    sft_format: str
    input_ids: torch.Tensor
    event_representations: torch.Tensor

    def __len__(self) -> int:
        return len(self.input_ids)


@dataclass
class BatchedEventVLProcessorOutput(DictOutput):
    """
    Container for batched processed event and text data.

    Attributes:
        sft_format (List[str]): List of formatted prompt strings
        input_ids (torch.Tensor): Padded tokenized text with event token placeholders
                                  Shape: [batch_size, max_seq_len]
        event_representations (torch.Tensor): Processed event data
                                  Shape: [batch_size, max_event_streams_per_sample, token_dim, grid_height, grid_width]
        attention_mask (torch.Tensor): Attention mask for text and event tokens
                                  Shape: [batch_size, max_seq_len]
        event_seq_mask (torch.BoolTensor): Mask indicating event token positions
                                   Shape: [batch_size, max_seq_len]
        event_emb_mask (torch.BoolTensor): Mask for valid event tokens
                                   Shape: [batch_size, max_event_streams_per_sample, num_event_tokens]
    """

    sft_format: List[str]
    input_ids: torch.Tensor
    event_representations: torch.Tensor
    attention_mask: torch.Tensor
    event_seq_mask: torch.BoolTensor
    event_emb_mask: torch.BoolTensor

    def __len__(self) -> int:
        return len(self.input_ids)

    def to(self, device, dtype=torch.float32):
        self.input_ids = self.input_ids.to(device)
        self.attention_mask = self.attention_mask.to(device)
        self.event_seq_mask = self.event_seq_mask.to(device)
        self.event_emb_mask = self.event_emb_mask.to(device)
        self.event_representations = self.event_representations.to(device=device, dtype=dtype)
        return self


class EventVLProcessor(VLChatProcessor):
    """
    Processor for handling both event data and text for multimodal models.

    Inherits from VLChatProcessor and specializes in processing event camera data
    along with text, using image tokens as placeholders for event representations.
    """

    event_tokenizer_class = "EventTokenizer"
    language_tokenizer_class = ("LlamaTokenizer", "LlamaTokenizerFast")
    attributes = ["event_tokenizer", "language_tokenizer"]

    system_prompt = (
        "You are a helpful language and event vision assistant. "
        "You are able to understand the event visual content (received using an event camera,"
        "also known as a neuromorphic camera, such as the Dynamic Vision Sensor (DVS)),"
        "that the user provides, and assist the user with a variety of tasks using natural language."
    )

    def __init__(self,
                 event_tokenizer: EventTokenizer,
                 language_tokenizer: LlamaTokenizerFast,
                 num_event_tokens: int,
                 image_tag: str = "<image_placeholder>",
                 add_special_token: bool = False,
                 sft_format: str = "deepseek",
                 mask_prompt: bool = True,
                 ignore_id: int = -100,
                 **kwargs,
                 ):
        """
        Initialize the EventVLProcessor.

        Args:
            event_tokenizer (EventTokenizer): Tokenizer for processing event data
            language_tokenizer (LlamaTokenizerFast): Tokenizer for processing text
            num_event_tokens (int): Number of tokens allocated for each event stream
            image_tag (str): Special token used as placeholder for event representations
            add_special_token (bool): Whether to add special tokens
            sft_format (str): Format for supervised fine-tuning
            mask_prompt (bool): Whether to mask prompt tokens during training
            ignore_id (int): Token ID to ignore during loss calculation
            **kwargs: Additional arguments passed to parent class
        """

        self.event_tokenizer = event_tokenizer
        self.language_tokenizer = language_tokenizer
        self.image_tag = image_tag
        self.num_event_tokens = num_event_tokens
        self.add_special_token = add_special_token
        self.sft_format = sft_format
        self.mask_prompt = mask_prompt
        self.ignore_id = ignore_id

        self.event_representation_shape = self.get_event_representation_shape()

        super().__init__(event_tokenizer,
                         language_tokenizer,
                         image_tag,
                         num_event_tokens,
                         add_special_token,
                         sft_format,
                         mask_prompt,
                         ignore_id,
                         **kwargs)

    @property
    def event_tag(self) -> str:
        """
        Get the special token string used as placeholder for event representations.

        Returns:
            str: The event tag/placeholder string (for consistency with DeepSeek-VL pretraining
                 we take "<image_placeholder>")
        """
        return self.image_tag

    @property
    def event_id(self) -> int:
        """
        Get the token ID used to represent events in the tokenized text.

        Returns:
            int: The numerical token ID corresponding to the event placeholder (for consistency with DeepSeek-VL
                 pretraining we take the ID corresponding to "<image_placeholder>")
        """

        return self.image_id

    def add_event_token(self,
                        event_indices: List[int],
                        input_ids: torch.LongTensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Insert event placeholder tokens into tokenized text at specified positions.
        Wraps add_image_token() with event-focused naming.

        Args:
            event_indices (torch.Tensor): Positions to insert event tokens
                                          Shape: [event_streams_per_sample, 1]
            input_ids: Tokenized text without event token placeholders
                      Shape: [text_length]

        Returns:
             input_ids: Tokenized text with event token placeholders
                        Shape: [text_length + num_event_tokens]
             num_event_tokens: Number of tokens allocated for each event stream
                        Shape: [event_streams_per_sample]
        """

        return self.add_image_token(image_indices=event_indices,
                                    input_ids=input_ids)

    def get_event_representation_shape(self) -> Tuple[int, int, int]:
        """
        Compute the shape of event representation based on the event tokenizer configuration.

        Returns:
            Tuple[int, int, int]:
                - token_dim: Dimensionality of each token (i.e., feature channels per spatial location)
                - grid_height: Number of token rows (vertical patches)
                - grid_width: Number of token columns (horizontal patches)
        """

        height = width = self.event_tokenizer.ref_resolution
        grid_height = grid_width = height // self.event_tokenizer.patch_size
        token_dim = self.event_tokenizer.time_div * 4 * self.event_tokenizer.patch_size ** 2

        return token_dim, grid_height, grid_width

    def process_one(self,
                    prompt: str = None,
                    conversations: List[Dict[str, str]] = None,
                    event_data: torch.Tensor = None,
                    **kwargs) -> EventVLProcessorOutput:
        """
        Process a single sample containing text and optional event data.

        Args:
            prompt (str): Prompt string
            conversations (List[Dict]): List of conversation turns as dictionaries
            event_data (torch.Tensor): Raw event camera data
                                       Shape: [event_streams_per_sample, num_events, 4] in format (t, x, y, p)
            **kwargs: Additional arguments

        Returns:
            EventVLProcessorOutput: Processed sample containing:
                - sft_format (str): Formatted prompt string
                - input_ids (torch.LongTensor): Tokenized text with event token placeholders
                                                Shape: [text_length + num_event_tokens]
                - event_representations (torch.FloatTensor): Processed event data
                                                Shape: [event_streams_per_sample, token_dim, grid_height, grid_width]

        Raises:
            ValueError: If number of event token placeholders in prompt doesn't match the provided event data
        """

        assert (
                prompt is None or conversations is None
        ), "prompt and conversations cannot be used at the same time."

        # Initialize empty event data if none provided
        if event_data is None:
            event_data = torch.zeros((0, 0, 4))

        # Format the prompt or conversation
        if prompt is None:
            # Apply sft format
            sft_format = self.apply_sft_template_for_multi_turn_prompts(conversations=conversations,
                                                                        sft_format=self.sft_format,
                                                                        system_prompt=self.system_prompt)
        else:
            sft_format = self.system_prompt + 'User:' + prompt + '. Assistant:'

        # Tokenize the text
        input_ids = self.language_tokenizer.encode(sft_format)
        input_ids = torch.LongTensor(input_ids)

        # Find positions of event tokens in the text
        event_token_mask = input_ids == self.event_id
        event_indices = event_token_mask.nonzero()

        # Validate event token count matches provided event data
        if len(event_indices) != len(event_data):
            raise ValueError(
                f"Found {len(event_indices)} image token(s) in the prompt but {len(event_data)} "
                "event streams provided. Please, pass the correct number of event streams."
            )

        # Add event tokens to the input_ids
        input_ids, num_event_tokens = self.add_event_token(event_indices=event_indices,
                                                           input_ids=input_ids)

        # Process event data
        if len(event_data) == 0:
            event_representations = torch.zeros((0, *self.event_representation_shape))
        else:
            event_representations = self.event_tokenizer(event_data)

        return EventVLProcessorOutput(sft_format=sft_format,
                                      input_ids=input_ids,
                                      event_representations=event_representations)

    def process_batch(self,
                      prompt_list: List[str] = None,
                      conversations_list: List[List[Dict[str, str]]] = None,
                      event_data_list: List[torch.Tensor] = None,
                      **kwargs) -> BatchedEventVLProcessorOutput:
        """
        Process a batch of samples containing text and event data.

        Args:
            prompt_list (List[str]): List of pre-formatted prompt strings
            conversations_list (List[List[Dict]]): List of conversation turns as dictionaries
            event_data_list (List[torch.Tensor]): List of raw event camera data
                       Shape of an element in the list: [event_streams_per_sample, num_events, 4] in format (t, x, y, p)
            num_workers (int): Number of worker processes to use
            **kwargs: Additional arguments

        Returns:
            BatchedEventVLProcessorOutput: Batched data ready for model input containing:
                        sft_format (List[str]): List of formatted prompt strings
                        input_ids (torch.Tensor): Padded tokenized text with event token placeholders
                            Shape: [batch_size, max_seq_len]
                        event_representations (torch.Tensor): Processed event data
                            Shape: [batch_size, max_event_streams_per_sample, token_dim, grid_height, grid_width]
                        attention_mask (torch.Tensor): Attention mask for text and event tokens
                            Shape: [batch_size, max_seq_len]
                        event_seq_mask (torch.BoolTensor): Mask indicating event token positions
                            Shape: [batch_size, max_seq_len]
                        event_emb_mask (torch.BoolTensor): Mask for valid event tokens
                            Shape: [batch_size, max_event_streams_per_sample, num_event_tokens]

        Raises:
            ValueError: If the input lists vary in lengths.

        Note:
            batch_size here is determined by the length of the input list.
        """

        # Validate prompt or conversations count matches the event data list
        if not(len(prompt_list) == len(conversations_list) == len(event_data_list)):
            raise ValueError(
                f"Found {len(prompt_list)} prompts, {len(conversations_list)} conversations and {len(event_data_list)}"
                f" event inputs provided. Their number should coincide."
            )

        # Process the inputs
        prepare_list = [self.process_one(prompt=prompt,
                                         conversations=conversations,
                                         event_data=event_data)
                        for prompt, conversations, event_data in zip(prompt_list,
                                                                     conversations_list,
                                                                     event_data_list)]

        # Batchify the results
        return self.batchify(prepare_list=prepare_list)

    def batchify(self,
                 prepare_list: List[EventVLProcessorOutput]) -> BatchedEventVLProcessorOutput:
        """
        Convert a list of individual samples into a batched format.

        Args:
            prepare_list (List[EventVLProcessorOutput]): List of processed samples

        Returns:
            BatchedEventVLProcessorOutput: Batched data ready for model input
        """

        batch_size = len(prepare_list)
        sft_format = []
        n_event_streams = []
        seq_lens = []
        for prepare in prepare_list:
            n_event_streams.append(len(prepare.event_representations))
            seq_lens.append(len(prepare))

        input_token_max_len = max(seq_lens)
        max_n_event_streams = max(1, max(n_event_streams))

        # Initialize batched tensors
        batched_input_ids = torch.full((batch_size, input_token_max_len), self.pad_id).long()
        batched_attention_mask = torch.zeros((batch_size, input_token_max_len)).long()
        batched_event_representations = torch.zeros((batch_size, max_n_event_streams,
                                                     *self.event_representation_shape)).float()
        batched_event_seq_mask = torch.zeros((batch_size, input_token_max_len)).bool()
        batched_event_emb_mask = torch.zeros((batch_size, max_n_event_streams, self.num_event_tokens)).bool()

        for i, prepare in enumerate(prepare_list):
            input_ids = prepare.input_ids

            # Left-padding
            batched_attention_mask[i, -seq_lens[i]:] = 1
            batched_input_ids[i, -seq_lens[i]:] = torch.LongTensor(input_ids)
            batched_event_seq_mask[i, -seq_lens[i]:] = input_ids == self.event_id

            if n_event_streams[i] > 0:
                batched_event_representations[i, :n_event_streams[i]] = prepare.event_representations
                for j in range(len(prepare.event_representations)):
                    batched_event_emb_mask[i, j, :self.num_event_tokens] = True

            sft_format.append(prepare.sft_format)

        batched_prepares = BatchedEventVLProcessorOutput(input_ids=batched_input_ids,
                                                         attention_mask=batched_attention_mask,
                                                         event_representations=batched_event_representations,
                                                         event_seq_mask=batched_event_seq_mask,
                                                         event_emb_mask=batched_event_emb_mask,
                                                         sft_format=sft_format)

        return batched_prepares

    def __call__(
            self,
            prompt: str = None,
            conversations: List[Dict[str, str]] = None,
            event_data: torch.Tensor = None,
            force_batchify: bool = True,
            **kwargs):
        """
        Args:
            prompt (str): Pre-formatted prompt string
            conversations (List[Dict]): List of conversation turns as dictionaries
            event_data (torch.Tensor): Raw event camera data
                                       Shape: [event_streams_per_sample, num_events, 4] in format (t, x, y, p)
            force_batchify (bool): flag to force inputs batchifying
            **kwargs:

        Returns:
           EventVLProcessorOutput: (only for a single conversation)
                - sft_format (str): Formatted prompt string
                - input_ids (torch.LongTensor): Tokenized text with event token placeholders
                                                Shape: [text_length + num_event_tokens]
                - event_representations (torch.FloatTensor): Processed event data
                                                Shape: [event_streams_per_sample, token_dim, grid_height, grid_width]
        or
           BatchedEventVLProcessorOutput: (only for a single conversation)
                - sft_format (List[str]): List of formatted prompt strings
                - input_ids (torch.Tensor): Padded tokenized text with event token placeholders
                                          Shape: [1, text_length + num_event_tokens]
                - event_representations (torch.Tensor): Processed event data
                                          Shape: [1, event_streams_per_sample, token_dim, grid_height, grid_width]
                - attention_mask (torch.Tensor): Attention mask for text and event tokens
                                          Shape: [1, text_length + num_event_tokens]
                - event_seq_mask (torch.BoolTensor): Mask indicating event token positions
                                           Shape: [1, text_length + num_event_tokens]
                - event_emb_mask (torch.BoolTensor): Mask for valid event tokens
                                           Shape: [1, event_streams_per_sample, num_event_tokens]
        """

        prepare = self.process_one(prompt=prompt,
                                   conversations=conversations,
                                   event_data=event_data)

        if force_batchify:
            prepare = self.batchify([prepare])

        return prepare


def build_processor(config: dict) -> EventVLProcessor:
    """
    Builds and configures an EventVLProcessor based on the provided configuration.

    Args:
        config (dict): Dictionary containing the configuration of the experiment

    Returns:
        EventVLProcessor: Fully configured EventVLProcessor instance
    """

    def _get_event_tokenizer(config: dict) -> EventTokenizer:
        """
        Initializes and returns an EventTokenizer from config.

        Args:
            config (dict): Dictionary containing the configuration of the experiment

        Returns:
            EventTokenizer: Configured event tokenizer from GET paper (https://arxiv.org/pdf/2310.02642)

        Raises:
            KeyError: If required configuration keys are missing.
        """

        data_config = config['data']
        model_parameters = config['event_vision_model']['parameters']

        return EventTokenizer(ref_resolution=data_config['ref_resolution'],
                              embed_split=model_parameters['embed_split'],
                              patch_size=model_parameters['patch_size'])

    def _get_language_tokenizer(config: dict) -> LlamaTokenizerFast:
        """
        Loads and returns a LlamaTokenizerFast from a pretrained VLChatProcessor.

        Args:
            config (dict): Dictionary containing the configuration of the experiment

        Returns:
            LlamaTokenizerFast: Tokenizer from the pretrained vision-language processor
        """

        model_path = config['vision_language_model']['model_path']
        vl_processor = VLChatProcessor.from_pretrained(model_path)
        return vl_processor.tokenizer

    # Load an event tokenizer
    event_tokenizer = _get_event_tokenizer(config)

    # Load a language tokenizer
    language_tokenizer = _get_language_tokenizer(config)

    # Assemble the processor
    return EventVLProcessor(event_tokenizer=event_tokenizer,
                            language_tokenizer=language_tokenizer,
                            num_event_tokens=config['event_vision_model']['num_event_tokens'])
