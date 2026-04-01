import os
import torch
import torch.nn as nn


class Checkpointer:
    """
    Saves and loads .pth.tar checkpoint.

    Args:
        checkpoint_dir (str): Path to the folder with checkpoint
        model_config (dict): Model name and parameters
    """

    def __init__(self,
                 checkpoint_dir: str,
                 model_config: dict):

        self.checkpoint_dir = checkpoint_dir
        self.model_config = model_config

    def save_checkpoint(self,
                        epoch: int,
                        model: nn.Module,
                        optimizer: torch.optim.Optimizer,
                        scheduler: torch.optim.lr_scheduler._LRScheduler = None,
                        checkpoint_name: str = 'model'):
        """
        Saves state dicts as a checkpoint in a .pth.tar file.

        Args:
            epoch (int): Epoch number
            model (torch.nn.Module): Torch model
            optimizer (torch.optim.Optimizer): Torch optimizer
            scheduler (torch.optim.lr_scheduler._LRScheduler): Torch scheduler
            checkpoint_name (str): Name of the checkpoint to be saved
        """

        # Exclude language model weights from the state dictionary
        model_state_dict = {k: v for k,v in model.state_dict().items()
                            if not('clip' in k or
                                   'language_model' in k or
                                   'text_transformer' in k)}

        checkpoint = {'epoch': epoch,
                      'model_config': self.model_config,
                      'model_state_dict': model_state_dict,
                      'optimizer_state_dict': optimizer.state_dict()}

        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()

        torch.save(checkpoint, os.path.join(self.checkpoint_dir, f'{checkpoint_name}.pth.tar'))

    def load_checkpoint(self,
                        device: torch.device,
                        checkpoint_name: str = 'model'):
        """
        Loads a checkpoint with state dicts to torch device from a .pth.tar file.

        Args:
            device (torch.device): Torch device
            checkpoint_name (str): Name of the checkpoint to be loaded

        Returns:
            epoch (int): epoch number
            model_state_dict (dict): Torch model state dict
            optimizer_state_dict (dict): Torch optimizer state dict
            scheduler_state_dict (dict): Torch scheduler state dict if available

        Raises:
            FileNotFoundError: If the checkpoint file does not exist
            ValueError: Error loading the checkpoint file,
                        if one of the required components do not exist or
                        the model configurations mismatch
        """

        checkpoint_path = os.path.join(self.checkpoint_dir, f'{checkpoint_name}.pth.tar')

        # Check if file exists first
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found at {checkpoint_path}")

        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
        except Exception as e:
            raise ValueError(f"Error loading checkpoint: {str(e)}")

        # Required components
        required_components = {'epoch': checkpoint.get('epoch'),
                               'model_config': checkpoint.get('model_config'),
                               'model_state_dict': checkpoint.get('model_state_dict'),
                               'optimizer_state_dict': checkpoint.get('optimizer_state_dict')}

        # Check for missing required components
        for k, v in required_components.items():
            if v is None:
                raise ValueError(f"Required component {k} missing in the checkpoint.")

        # Verify model configuration
        if required_components['model_config'] != self.model_config:
            raise ValueError("The model configurations mismatch."
                             f"Expected configuration: {self.model_config}"
                             f"Loaded configuration: {required_components['model_config']}")

        # Scheduler is optional
        scheduler_state_dict = checkpoint.get('scheduler_state_dict')

        return (required_components['epoch'],
                required_components['model_state_dict'],
                required_components['optimizer_state_dict'],
                scheduler_state_dict)


def build_checkpointer(checkpoint_dir: str,
                       model_config: dict) -> Checkpointer:
    """
    Creates a Checkpointer instance for managing model checkpoints.

    Args:
        checkpoint_dir: Path to the directory where checkpoints will be saved/loaded.
        model_config: Dictionary containing the model configuration parameters.

    Returns:
        Checkpointer: An initialized checkpointer instance configured to save
        and load checkpoints in the specified directory with the given model config.
    """

    return Checkpointer(checkpoint_dir=checkpoint_dir,
                        model_config=model_config)
