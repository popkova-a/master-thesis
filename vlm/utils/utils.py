# Disable warnings
import warnings
warnings.filterwarnings('ignore')

import os
import sys
import random
import numpy as np
from typing import Union

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

# Constants
REPOSITORY_PATHS = {'main_repo': '/data/storage/anastasia/repos',
                    'deepseek': '/data/storage/anastasia/repos/DeepSeek-VL',
                    'GET': '/data/storage/anastasia/repos/GET_Transformer'}
HF_CACHE_DIR = "/data/storage/anastasia/anaconda3/.conda_cache/huggingface"

# Configure system paths
sys.path.extend(REPOSITORY_PATHS.values())
os.environ["HF_HOME"] = HF_CACHE_DIR


def set_seed(seed: int,
             rank: int) -> None:
    """
    Configure all random seeds for reproducibility.

    Args:
        seed (int): Base random seed.
        rank (int): Process rank to create unique seeds per process.
    """

    seed += rank  # Ensure different seeds across processes
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True


def setup_distributed_environment(rank: int,
                                  local_rank: int,
                                  world_size: int,
                                  seed: int = 123) -> torch.device:
    """
    Initialize the distributed training environment for multi-GPU training and
    set a seed for reproducibility.

    Args:
        rank (int): Unique identifier for each process (0 <= rank < world_size)
        local_rank (int): The rank (index) of the current process within its local machine.
        world_size (int): Total number of processes participating in the job
        seed (int, optional): Base random seed. Defaults to 123.
                             Note: Actual seed will be seed + rank for per-process uniqueness.

    Raises:
        Exception: If process group initialization fails
    """

    if rank == 0:
        os.environ['MASTER_ADDR'] = os.getenv('MASTER_ADDR', 'localhost')
        os.environ['MASTER_PORT'] = os.getenv('MASTER_PORT', '12355')

    # Set the current device for the rank
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)

    try:
        if not torch.distributed.is_initialized():
            # Initialize the process group
            torch.distributed.init_process_group(backend='nccl',
                                                 init_method='env://',
                                                 rank=rank,
                                                 world_size=world_size,
                                                 device_id=device)
            # Print the result
            print(f"Initialized distributed environment on rank {rank}")

        # Sync all processes
        torch.distributed.barrier()
    except Exception as e:
        print(f"Failed to initialize distributed environment: {e}")
        raise

    # Set a random seed on different ranks
    set_seed(seed=seed,
             rank=rank)

    return device


def cleanup_distributed_environment() -> None:
    """
    Clean up the distributed training environment.

    Safely destroys the process group if it was initialized.
    This should be called at the end of training to prevent resource leaks.
    """

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def get_parameter_groups(model: nn.Module,
                         lr: Union[float, list],
                         weight_decay: float) -> list:
    """
    This function splits parameters into backbone and head components, then further divides
    them into weight-decayed and non-decayed groups (for biases/normalization layers).
    Each group maintains proper learning rate inheritance from its parent component.

    Args:
        model (nn.Module): Model whose parameters will be grouped.
        lr: Learning rate specification. Can be:
            - float: Used for both backbone and head
            - list[float]: [lr] or [backbone_lr, head_lr]
        weight_decay (float): Weight decay (L2 penalty) value to apply to non-bias parameters.

    Returns:
        list: List of parameter groups formatted for PyTorch optimizers.
    """

    # Verify the learning rate settings
    if isinstance(lr, float):
        backbone_lr = head_lr = lr
    elif isinstance(lr, list) and len(lr) == 1:
        backbone_lr = head_lr = lr[0]
    else:
        backbone_lr, head_lr = lr[:2]  # Unpack exactly 2 values

    # Define component prefixes
    backbone_prefixes = ('event_vision_model.', 'backbone.')
    head_prefixes = ('aligner.', 'head.')

    # Initialize parameter groups
    groups = {'backbone_decay': {'params': [], 'lr': backbone_lr, 'weight_decay': weight_decay, 'name': 'backbone_decay'},
              'head_decay': {'params': [], 'lr': head_lr, 'weight_decay': weight_decay, 'name': 'head_decay'},
              'backbone_no_decay': {'params': [], 'lr': backbone_lr, 'weight_decay': 0.0, 'name': 'backbone_no_decay'},
              'head_no_decay': {'params': [], 'lr': head_lr, 'weight_decay': 0.0, 'name': 'head_no_decay'}}

    # Categorize parameters
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # Skip frozen params

        if name.startswith(backbone_prefixes):
            group_prefix = 'backbone'
        elif name.startswith(head_prefixes):
            group_prefix = 'head'
        else:
            continue

        # Determine decay status
        decay_status = 'no_decay' if any(word in name for word in ['bias', 'norm', 'ln_']) else 'decay'

        # Add to appropriate group
        groups[f'{group_prefix}_{decay_status}']['params'].append(param)

    # Return only non-empty groups
    return [g for g in groups.values() if g['params']]


def load_pretrained_checkpoint(checkpoint_path: str,
                               model: nn.Module,
                               input_backbone_name: str = 'backbone',
                               output_backbone_name: str = 'backbone',
                               device: torch.device = None,
                               verbose: bool = True) -> nn.Module:
    """
    Load pretrained checkpoint into a model with optional backbone name adaptation.

    Args:
        checkpoint_path (str): Path to the checkpoint file
        model (nn.Module): Model to load the weights into
        input_backbone_name (str): Name of the input backbone in the checkpoint
        output_backbone_name (str): Name of the output backbone in the model
        device (torch.device): Device to load the model on
        verbose (bool): Whether to print loading messages

    Returns:
        nn.Module: The model with loaded weights
    """

    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Retrieve model state dict
    if 'model_state_dict' not in checkpoint:
        raise KeyError(f"The checkpoint must contain a 'model_state_dict' key.")
    model_state_dict = checkpoint['model_state_dict']

    # Adapt multi-gpu checkpoint for a single-gpu setting
    world_size = torch.distributed.get_world_size()
    if world_size == 1:
        model_state_dict = {k.replace('module.', ''): v for k, v in model_state_dict.items()}

    # Change the backbone name
    if input_backbone_name != output_backbone_name:
        model_state_dict = {k.replace(f'{input_backbone_name}.', f'{output_backbone_name}.'): v
                            for k, v in model_state_dict.items()}

    # Move model to device before loading state dict
    if device is not None:
        model = model.to(device)

    # Load the state dictionary into the provided model
    msg = model.load_state_dict(model_state_dict, strict=False)

    if verbose:
        print(f"Loading messages: {msg}")
        if msg.missing_keys:
            print(f"Missing keys: {msg.missing_keys}")
        if msg.unexpected_keys:
            print(f"Unexpected keys: {msg.unexpected_keys}")

    return model
