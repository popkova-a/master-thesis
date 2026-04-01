import math
import torch
from typing import Union, List

from torch.optim.lr_scheduler import _LRScheduler
from timm.scheduler.cosine_lr import CosineLRScheduler

"""
Code reference: https://github.com/katsura-jp/pytorch-cosine-annealing-with-warmup/tree/master
"""


class CosineAnnealingWarmupRestarts(_LRScheduler):
    """
    A learning rate scheduler combining warmup and cosine annealing with restarts.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        first_cycle_steps (int): First cycle step size.
        cycle_mult (float): Cycle steps magnification. Default: 1.
        max_lr (float): First cycle's max learning rate. Default: 0.1.
        min_lr (float): Min learning rate. Default: 0.001.
        warmup_steps (int): Linear warmup step size. Default: 0.
        gamma (float): Decrease rate of max learning rate by cycle. Default: 1.
        last_epoch (int): The index of last epoch. Default: -1.
    """

    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 first_cycle_steps: int,
                 cycle_mult: float = 1.,
                 max_lr: Union[float, list] = 0.1,
                 min_lr: Union[float, list] = 0.001,
                 warmup_steps: int = 0,
                 gamma: float = 1.,
                 last_epoch: int = -1):

        assert warmup_steps < first_cycle_steps

        # Handling different types of input
        group_num = len(optimizer.param_groups)
        self.base_max_lrs = self._validity_check(lr=max_lr,
                                                 group_num=group_num) # first max learning rate
        self.max_lrs = self._validity_check(lr=max_lr,
                                            group_num=group_num)  # max learning rate in the current cycle
        self.min_lrs = self._validity_check(lr=min_lr,
                                            group_num=group_num)  # min learning rate

        self.first_cycle_steps = first_cycle_steps  # first cycle step size
        self.cycle_mult = cycle_mult  # cycle steps magnification
        self.warmup_steps = warmup_steps  # warmup step size
        self.gamma = gamma  # decrease rate of max learning rate by cycle

        self.cur_cycle_steps = first_cycle_steps  # first cycle step size
        self.cycle = 0  # cycle count
        self.step_in_cycle = last_epoch  # step size of the current cycle

        super(CosineAnnealingWarmupRestarts, self).__init__(optimizer, last_epoch)

        # set learning rate min_lr
        self.init_lr()

    def _validity_check(self,
                        lr: Union[float, List[float]],
                        group_num: int) -> List:
        """
        Validate and format learning rate inputs.

        Args:
            lr (float or list[float]): Learning rate(s) to validate
            group_num (int): Number of parameter groups in the optimizer

        Returns:
            list: List of learning rates (one per parameter group).

        Raises:
            ValueError: If lr is neither a float nor a list with length matching group_num.
        """

        if isinstance(lr, float):
            return [lr] * group_num
        elif isinstance(lr, list) and group_num != len(lr) * 2:  # Bias / non-bias within each group for weight decay compensation
            raise ValueError("The number of learning rates should correspond to the number of parameter groups.")
        elif not isinstance(lr, list):
            raise ValueError("Learning rate should be a float or list of floats.")
        else:
            return lr * 2  # Bias / non-bias within each group for weight decay compensation

    def init_lr(self) -> None:
        """
        Initialize learning rates to min_lr for all parameter groups.

        This is called during scheduler initialization to set the base learning rates.
        """

        self.base_lrs = self.min_lrs.copy()
        for i, param_group in enumerate(self.optimizer.param_groups):
            param_group['lr'] = self.base_lrs[i]

    def get_lr(self) -> List:
        """
        Compute the current learning rate for each parameter group.

        Returns:
            list[float]: Learning rates for each parameter group. Calculated as:
                - base_lrs if no steps taken (step_in_cycle == -1).
                - Linear warmup if in warmup phase (step_in_cycle < warmup_steps).
                - Cosine annealing otherwise.
        """

        if self.step_in_cycle == -1:
            return self.base_lrs
        elif self.step_in_cycle < self.warmup_steps:
            return [(max_lr - base_lr) * self.step_in_cycle / self.warmup_steps + base_lr
                    for max_lr, base_lr in zip(self.max_lrs, self.base_lrs)]
        else:
            return [base_lr + (max_lr - base_lr) *
                    (1 + math.cos(math.pi * (self.step_in_cycle - self.warmup_steps) /
                                 (self.cur_cycle_steps - self.warmup_steps))) / 2
                    for max_lr, base_lr in zip(self.max_lrs, self.base_lrs)]

    def step(self,
             epoch: int = None) -> None:
        """
        Update the learning rate and cycle state.

        Args:
            epoch (int): Current epoch. If None, increments internal step count.s
        """

        if epoch is None:
            epoch = self.last_epoch + 1
            self.step_in_cycle = self.step_in_cycle + 1
            if self.step_in_cycle >= self.cur_cycle_steps:
                self.cycle += 1
                self.step_in_cycle = self.step_in_cycle - self.cur_cycle_steps
                self.cur_cycle_steps = int(
                    (self.cur_cycle_steps - self.warmup_steps) * self.cycle_mult) + self.warmup_steps
        else:
            if epoch >= self.first_cycle_steps:
                if self.cycle_mult == 1.:
                    self.step_in_cycle = epoch % self.first_cycle_steps
                    self.cycle = epoch // self.first_cycle_steps
                else:
                    n = int(math.log((epoch / self.first_cycle_steps * (self.cycle_mult - 1) + 1), self.cycle_mult))
                    self.cycle = n
                    self.step_in_cycle = epoch - int(
                        self.first_cycle_steps * (self.cycle_mult ** n - 1) / (self.cycle_mult - 1))
                    self.cur_cycle_steps = self.first_cycle_steps * self.cycle_mult ** (n)
            else:
                self.cur_cycle_steps = self.first_cycle_steps
                self.step_in_cycle = epoch

        self.max_lrs = [base_max_lr * (self.gamma ** self.cycle) for base_max_lr in self.base_max_lrs]
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr


def build_scheduler(optimizer: torch.optim.Optimizer,
                    config: dict,
                    iter_per_epoch: int,
                    last_epoch: int = -1) -> Union[CosineAnnealingWarmupRestarts, CosineLRScheduler]:
    """
    Builds a learning rate scheduler.

    Args:
        optimizer (torch.optim.Optimizer): Wrapped optimizer (e.g., Adam, SGD)
        config (dict): Configuration dictionary containing scheduler parameters under config['train']['optimizer']
        iter_per_epoch (int): Number of iterations in each epoch, i.e. number of batches per epoch.
        last_epoch (int: The index of the last epoch (for resuming training). Default: -1.

    Returns:
         A configured scheduler instance of the specified type:
            - CosineAnnealingWarmupRestarts if name is 'cosine_restarts'
            - CosineLRScheduler if name is 'cosine'

    Raises:
        ValueError: If an unsupported scheduler name is provided.
    """

    lr_config = config['train']['scheduler']
    if lr_config['name'] == 'cosine_restarts':
        return CosineAnnealingWarmupRestarts(optimizer=optimizer,
                                             first_cycle_steps=lr_config['first_cycle_steps'],
                                             cycle_mult=lr_config['cycle_mult'],
                                             max_lr=lr_config['max_lr'],
                                             min_lr=lr_config['min_lr'],
                                             warmup_steps=lr_config['warmup_steps'],
                                             gamma=lr_config['gamma'],
                                             last_epoch=last_epoch)
    elif lr_config['name'] == 'cosine':
        num_steps = config['train']['num_epochs'] * iter_per_epoch
        warmup_steps = lr_config['warmup_epochs'] * iter_per_epoch
        return CosineLRScheduler(optimizer=optimizer,
                                 t_initial=(num_steps - warmup_steps) if lr_config['warmup_prefix'] else num_steps,
                                 lr_min=lr_config['min_lr'],
                                 cycle_mul=lr_config['cycle_mult'],
                                 cycle_limit=lr_config['cycle_limit'],
                                 warmup_t=warmup_steps,
                                 warmup_lr_init=lr_config['warmup_lr_init'],
                                 warmup_prefix=lr_config['warmup_prefix'],
                                 t_in_epochs=lr_config['t_in_epochs'])
    else:
        raise ValueError(f"Unsupported scheduler name: {lr_config['name']}. "
                         f"Available options: 'cosine', 'cosine_restarts'")
