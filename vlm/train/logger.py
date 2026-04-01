import os
import wandb
from typing import List, Tuple

import torch


class WandBLogger:
    """
    Logs metrics and media with the interface provided by wandb.ai.

    Args:
        log_dir (str): Path to the directory where logs should be stored.
        project (str): Name of the Weights & Biases project (optional).
        config (dict): Configuration dictionary to be logged (optional).
        resume (bool): Whether to resume logging to an existing run (default: False).
    """

    def __init__(self,
                 log_dir: str,
                 project: str = None,
                 config: dict = None,
                 resume: bool = False):

        self.log_dir = log_dir

        if resume:
            run_id = self._get_latest_run_id()
        else:
            run_id = wandb.util.generate_id()

        wandb.init(project=project if project is not None else os.path.split(os.getcwd())[1],
                   config=config,
                   name=os.path.split(log_dir)[1],
                   dir=log_dir,
                   resume='allow',
                   id=run_id)

        self.scalars_dict = {}
        self.table = None
        self.videos = []

    def _get_latest_run_id(self):
        """
        Retrieves the ID of the most recent wandb run by examining the 'latest-run' directory.

        Returns:
            str: The run ID of the latest wandb run.
        """

        latest_run_file_name = [file_name for file_name in os.listdir(os.path.join(self.log_dir, 'wandb', 'latest-run'))
                                if file_name.endswith(".wandb")][0]
        return os.path.splitext(latest_run_file_name)[0].split('-')[-1]

    def log(self,
            tag: str,
            step: int):
        """
        Logs the current scalars and/or table to wandb under a specific tag and step.

        Args:
            tag (str): Tag used to namespace the logged data.
            step (int): Training step or epoch number associated with this log.
        """

        log_dict = {}
        log_dict.update(self._flatten_dict(dict_={tag: self.scalars_dict}, separator='/'))
        if self.table is not None:
            log_dict.update(self._flatten_dict(dict_={tag: self.table}, separator='.'))

        # Add videos to log_dict
        for i, video in enumerate(self.videos):
            video_key = f"{tag}/video_{i + 1}"
            log_dict.update(self._flatten_dict(dict_={video_key: video}, separator='/'))

        wandb.log(log_dict, step=step)

        self.scalars_dict = {}
        self.table = None
        self.videos = []

    def add_scalars_dict(self, **scalars_dict):
        """
        Adds scalar values to be logged in the next log() call.

        Args:
            **scalars_dict: Arbitrary keyword arguments representing scalar metrics.
        """

        self.scalars_dict.update(scalars_dict)

    def add_table(self,
                  ground_truth_captions: List[str],
                  generated_captions: List[str]):
        """
        Creates a wandb.Table for logging structured data such as model predictions.

        Args:
            ground_truth_captions (list): List of ground truth or reference descriptions.
            generated_captions (list): List of model-generated descriptions or outputs.
        """

        columns = ["ground truth captions", "generated captions"]
        data = [[ground_truth_caption, generated_caption]
                for ground_truth_caption, generated_caption in zip(ground_truth_captions,
                                                                   generated_captions)]

        self.table = wandb.Table(columns=columns, data=data)

    def add_video(self,
                  events: torch.Tensor,
                  events_per_frame: int = 1000,
                  resolution: Tuple[int, int] = (128, 128),
                  caption: str = None,
                  device: torch.device = None) -> None:
        """
        Convert event tensor to a video of colored frames for visualization.

        Args:
            events (torch.Tensor): Event stream of shape [N, 4] (t, x, y, p) with polarities 0 or 1.
            events_per_frame (int): Number of events per video frame.
            resolution (tuple): Output frame size (height, width).
            device (torch.device): Target device (CPU/GPU).
        """

        # Move the event stream to device
        device = device if device is not None else torch.device('cpu')
        events = events.to(device)

        height, width = resolution

        # Create video frames
        frames = []
        for start in range(0, len(events), events_per_frame):
            end = min(start + events_per_frame - 1, len(events))

            # Split the vents
            chunk = events[start:end]
            chunk_x = torch.clamp(chunk[:, 1].long(), min=0, max=width - 1)
            chunk_y = torch.clamp(chunk[:, 2].long(), min=0, max=height - 1)
            chunk_p = chunk[:, 3].long()

            # Create a frame with white background
            frame = torch.ones(3, height, width, device=device, dtype=torch.uint8) * 255

            # Plot positive events (red)
            pos_mask = chunk_p == 1
            if pos_mask.any():
                frame[0, chunk_y[pos_mask], chunk_x[pos_mask]] = 255
                frame[1, chunk_y[pos_mask], chunk_x[pos_mask]] = 0
                frame[2, chunk_y[pos_mask], chunk_x[pos_mask]] = 0

            # Plot negative events (blue)
            neg_mask = chunk_p == 0
            if neg_mask.any():
                frame[0, chunk_y[neg_mask], chunk_x[neg_mask]] = 0
                frame[1, chunk_y[neg_mask], chunk_x[neg_mask]] = 0
                frame[2, chunk_y[neg_mask], chunk_x[neg_mask]] = 255

            frames.append(frame)

        self.videos.append(wandb.Video(torch.stack(frames, dim=0).cpu(),
                                       fps=10,
                                       caption=caption))

    @staticmethod
    def _flatten_dict(dict_: dict,
                      separator: str = '/'):
        """
        Flattens a nested dictionary for logging to wandb with namespaced keys.

        Args:
            dict_ (dict): Dictionary to flatten.
            separator (str): String used to separate keys in the flattened dictionary.

        Returns:
            dict: Flattened dictionary with compound keys.
        """

        prefix = ''
        stack = [(dict_, prefix)]
        flat_dict = {}

        while stack:
            current_dict, current_prefix = stack.pop()
            for key, value in current_dict.items():
                new_key = current_prefix + separator + key if current_prefix else key
                if isinstance(value, dict):
                    stack.append((value, new_key))
                else:
                    flat_dict[new_key] = value

        return flat_dict

    def __del__(self):
        """
        Finalizes the wandb run when the logger object is deleted.
        """

        wandb.finish()


def build_logger(log_dir: str,
                 project: str = None,
                 config: dict = None,
                 resume: bool = False) -> WandBLogger:
    """
    Create a WandBLogger instance initialized with the given logger parameters.

    Args:
        log_dir (str): Path to the directory where logs should be stored.
        project (str): Name of the Weights & Biases project (optional).
        config (dict): Configuration dictionary to be logged (optional).
        resume (bool): Whether to resume logging to an existing run (default: False).

    Returns:
        WandBLogger: A Weights & Biases logger instance configured with the specified parameters.
    """

    return WandBLogger(log_dir=log_dir,
                       project=project,
                       config=config,
                       resume=resume)
