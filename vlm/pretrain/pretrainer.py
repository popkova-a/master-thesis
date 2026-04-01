import random
from tqdm import tqdm
from typing import Union

# Set the bar colors for distributed setting
GREEN = '\033[92m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
MAGENTA = '\033[95m'
RESET = '\033[0m'
colors = [GREEN, YELLOW, BLUE, MAGENTA]

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.utils.data.distributed as dist

from train.logger import WandBLogger
from train.checkpointer import Checkpointer
from pretrain.evaluator import LinearProbeEvaluator
from data.pretrain_dataloader import TokenizedBatch
from train.amp_scaler import GradScalerWithNormTracking


class Pretrainer:
    def __init__(self,
                 model: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 objective: nn.Module,
                 scheduler: torch.optim.lr_scheduler._LRScheduler = None,
                 logger: WandBLogger = None,
                 evaluator: LinearProbeEvaluator = None,
                 checkpointer: Checkpointer = None,
                 amp_scaler: GradScalerWithNormTracking = None,
                 max_grad_norm: float = 1.0,
                 rank: int = 0,
                 local_rank: int = 0,
                 last_epoch: int = 0):

        self.model = model
        self.optimizer = optimizer
        self.objective = objective
        self.scheduler = scheduler
        self.logger = logger
        self.evaluator = evaluator
        self.checkpointer = checkpointer
        self.max_grad_norm = max_grad_norm
        self.rank = rank
        self.local_rank = local_rank
        self.last_epoch = last_epoch

        # Define if amp should be enabled
        if amp_scaler is not None:
            self.amp_scaler = amp_scaler
            self.amp_enabled = True
        else:
            self.amp_enabled = False

        # Move the model to the device if not DDP
        if not isinstance(self.model, nn.parallel.DistributedDataParallel):
            self.model.to(f'cuda:{self.local_rank}')

    def _optimize(self,
                  loss: torch.Tensor) -> Union[torch.Tensor, None]:
        """
        Performs a single optimization step.

        Args:
            loss (torch.Tensor): Computed loss value to backpropogate

        Returns:
            torch.Tensor or None: Gradient norm if computed, else None
        """

        world_size = (torch.distributed.get_world_size()
                      if torch.distributed.is_initialized()
                      else 1)
        model_parameters = (self.model.module.parameters()
                            if world_size > 1 else self.model.parameters())

        # Clear the gradients of all optimized tensors
        self.optimizer.zero_grad()

        # Define a gradient norm variable
        grad_norm = None

        if self.amp_enabled:
            # Handle mixed-precision training: scaled backprop + weight update
            grad_norm = self.amp_scaler(parameters=model_parameters,
                                        loss=loss,
                                        optimizer=self.optimizer,
                                        max_grad_norm=self.max_grad_norm)
        else:
            # Back propagate the loss
            loss.backward()

            # Clip the gradient norm
            torch.nn.utils.clip_grad_norm_(model_parameters,
                                           max_norm=self.max_grad_norm)

            # Update model parameters
            self.optimizer.step()

        if self.scheduler is not None:
            self.scheduler.step()

        return grad_norm

    def _get_lr(self,
                name: str) -> float:
        """
        Gets the current learning rate for a specific parameter group by name.

        Searches through all parameter groups in the optimizer to find the group
        with the matching name and returns its current learning rate.

        Args:
            name: The name of the parameter group whose learning rate should be returned.
                This should match the 'name' key in one of the optimizer's param_groups.

        Returns:
            float: The learning rate of the specified parameter group.
        """

        for param_group in self.optimizer.param_groups:
            if name in param_group['name']:
                return param_group['lr']

    @torch.no_grad()
    def _log_epoch_statistics(self,
                              batch: TokenizedBatch,
                              anchor_proj: torch.Tensor,
                              positive_proj: torch.Tensor,
                              loss: float,
                              grad_norm: float,
                              epoch: int,
                              phase: str) -> None:
        """
        Logs epoch results to WandB.

        Args:
            batch (TokenizedBatch): Last batch of the epoch to log
            anchor_proj (torch.Tensor): Anchor embeddings of the batch
            positive_proj (torch.Tensor): Positive embeddings of the batch
            loss (float): Loss for the epoch
            grad_norm (float): Gradient norm for the epoch (only used when training and AMP is enabled)
            epoch (int): Current epoch number
            phase (str): Either 'train' or 'val'
        """

        if self.logger is None or self.rank != 0:
            return

        # Log loss, ranking metrics and the learning rate
        if phase == 'train':
            if self.amp_enabled:
                self.logger.add_scalars_dict(epoch_loss=loss,
                                             epoch_grad_norm=grad_norm,
                                             lr_backbone=self._get_lr(name='backbone'),
                                             lr_head=self._get_lr(name='head'),
                                             **self.objective.compute_metrics(anchor_proj, positive_proj))
            else:
                self.logger.add_scalars_dict(epoch_loss=loss,
                                             lr_backbone=self._get_lr(name='backbone'),
                                             lr_head=self._get_lr(name='head'),
                                             **self.objective.compute_metrics(anchor_proj, positive_proj))
        else:
            self.logger.add_scalars_dict(epoch_loss=loss,
                                         **self.objective.compute_metrics(anchor_proj, positive_proj))

        # Visualize the augmentation
        for i in range(4):
            idx = random.randint(0, len(batch) - 1)
            self.logger.add_video(events=batch.anchors[idx].squeeze(0),
                                  device=torch.device(f'cuda:{self.local_rank}'))
            self.logger.add_video(events=batch.positives[idx].squeeze(0),
                                  device=torch.device(f'cuda:{self.local_rank}'))

        # Commit logs
        self.logger.log(tag=phase, step=epoch)

    def _barrier(self):
        """
        Applies barrier function in case of distributed training.
        """

        if torch.distributed.is_initialized():
            torch.distributed.barrier()

    def run_epoch(self,
                  dataloader: DataLoader,
                  epoch: int,
                  phase: str) -> None:
        """
        Runs a single training or validation epoch.

        Args:
            dataloader (DataLoader): DataLoader for the current phase
            epoch (int): Current epoch number
            phase (str): Either 'train' or 'val'
        """

        # Set training/evaluation mode
        is_train = (phase == 'train')
        self.model.train() if is_train else self.model.eval()

        # Set the sampler epoch for proper shuffling in DDP
        if isinstance(dataloader.sampler, dist.DistributedSampler):
            dataloader.sampler.set_epoch(epoch)

        # Initialize metrics and variables
        epoch_loss = torch.tensor(0.0,
                                  device=torch.device(f'cuda:{self.local_rank}'))
        epoch_grad_norm = (torch.tensor(0.0,
                                        device=torch.device(f'cuda:{self.local_rank}'))
                           if is_train else None)
        num_batches = len(dataloader)
        world_size = (torch.distributed.get_world_size()
                      if torch.distributed.is_initialized()
                      else 1)

        # Initialize progress bar
        pbar = tqdm(iterable=dataloader,
                    desc=f'{colors[self.rank % len(colors)]}Rank {self.rank}{RESET} | Epoch {epoch:02} | {phase}',
                    ncols=150,
                    leave=False,
                    unit='batch')

        with torch.set_grad_enabled(is_train):  # Control gradient computation
            for batch_idx, batch in enumerate(pbar):
                batch = batch.to(f'cuda:{self.local_rank}')

                # Synchronize if distributed training
                self._barrier()

                # Forward pass
                with torch.cuda.amp.autocast(enabled=self.amp_enabled):
                    anchor_proj, positive_proj = self.model(**batch)
                    loss = self.objective(anchor_proj, positive_proj)
                epoch_loss += loss.item()

                # Backward pass if training
                if is_train:
                    grad_norm = self._optimize(loss)

                    if grad_norm is not None and not torch.isnan(grad_norm):
                        epoch_grad_norm += grad_norm

                    # Update progress bar
                    pbar.set_postfix_str(f"loss: {loss.item():.5f} | grad_norm: {grad_norm:.5f}")
                    pbar.refresh()
                else:
                    # Update progress bar
                    pbar.set_postfix_str(f"loss: {loss.item():.5f}")
                    pbar.refresh()

                # Synchronize if distributed training
                self._barrier()

                # Wait for all kernels in all streams of the device to complete
                torch.cuda.synchronize()

                # Clean cache
                torch.cuda.empty_cache()

        # Calculate average loss for the epoch across ranks
        epoch_loss /= num_batches
        if torch.distributed.is_initialized():
            torch.distributed.reduce(epoch_loss, dst=0)
        epoch_avg_loss = epoch_loss.item() / world_size

        # Calculate average gradient norm for the training epoch across ranks
        if is_train:
            if epoch_grad_norm is not None:
                epoch_grad_norm /= num_batches
                if torch.distributed.is_initialized():
                    torch.distributed.reduce(epoch_grad_norm, dst=0)
                epoch_avg_grad_norm = epoch_grad_norm.item() / world_size
            else:
                epoch_avg_grad_norm = 0.0
        else:
            epoch_avg_grad_norm = 0.0

        # Synchronize if distributed training
        self._barrier()

        # Log epoch statistics and outputs
        self._log_epoch_statistics(batch=batch,
                                   anchor_proj=anchor_proj,
                                   positive_proj=positive_proj,
                                   loss=epoch_avg_loss,
                                   grad_norm=epoch_avg_grad_norm,
                                   epoch=epoch,
                                   phase=phase)

        # Synchronize if distributed training
        self._barrier()

    def pretrain(self,
                 train_dataloader: DataLoader,
                 val_dataloader: DataLoader,
                 num_epochs: int,
                 experiment_name: str = 'experiment') -> None:
        """
        Runs the complete training process.

        Args:
            train_dataloader (DataLoader): DataLoader for training data of self-supervised objective
            val_dataloader (DataLoader): DataLoader for validation data of self-supervised objective
            num_epochs (int): Number of epochs to train
            experiment_name (str): Name of the experiment to checkpoint
        """

        for epoch in range(self.last_epoch + 1, self.last_epoch + 1 + num_epochs):
            # Training epoch
            self.run_epoch(dataloader=train_dataloader,
                           epoch=epoch,
                           phase='train')

            # Synchronize if distributed training
            self._barrier()

            # Validation epoch
            self.run_epoch(dataloader=val_dataloader,
                           epoch=epoch,
                           phase='val')

            # Synchronize if distributed training
            self._barrier()

            # Linear probe
            if self.evaluator is not None:

                train_acc, val_acc = self.evaluator.evaluate()

                if self.rank == 0:
                    self.logger.add_scalars_dict(probe_train_acc=train_acc,
                                                 probe_val_acc=val_acc)
                    self.logger.log(tag='val', step=epoch)

                # Synchronize if distributed training
                self._barrier()

            # Checkpoint
            if self.checkpointer is not None:
                self.checkpointer.save_checkpoint(epoch=epoch,
                                                  model=self.model,
                                                  optimizer=self.optimizer,
                                                  scheduler=self.scheduler,
                                                  checkpoint_name=experiment_name)
