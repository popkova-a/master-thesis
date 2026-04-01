import math

import torch
import torch.nn as nn
import torch.distributed as dist


class SyncFunction(torch.autograd.Function):
    """
    Synchronizes tensors across all processes in distributed training.
    Forward pass gathers tensors from all processes, backward pass scatters gradients.
    """

    @staticmethod
    def forward(ctx,
                tensor: torch.Tensor) -> torch.Tensor:
        """
        Gathers tensors from all processes and returns concatenated result.

        Args:
            ctx: Context object to save information for backward pass
            tensor (torch.Tensor): Input tensor to synchronize

        Returns:
            torch.Tensor: Concatenated tensors from all processes
        """

        if not dist.is_initialized():
            return tensor

        ctx.batch_size = tensor.size(0)
        world_size = dist.get_world_size()

        # Validate input tensor
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()

        # Pre-allocate gather list with proper device placement
        gathered_tensors = [torch.empty_like(tensor) for _ in range(world_size)]

        # Perform all-gather
        dist.all_gather(gathered_tensors, tensor)

        # Concatenate results
        return torch.cat(gathered_tensors, dim=0)

    @staticmethod
    def backward(ctx,
                 grad_output: torch.Tensor) -> torch.Tensor:
        """
        Scatters gradients back to original processes.

        Args:
            ctx: Context object to save information for backward pass
            grad_output (torch.Tensor): Gradient of the concatenated output

        Returns:
            torch.Tensor: Gradient for the original input tensor
        """

        if not dist.is_initialized():
            return grad_output

        # Validate gradient tensor
        if not grad_output.is_contiguous():
            grad_output = grad_output.contiguous()

        rank = dist.get_rank()
        batch_size = ctx.batch_size

        # Reduce gradients across all processes (sum)
        grad_input = grad_output.clone()
        dist.all_reduce(grad_input, op=dist.ReduceOp.SUM, async_op=False)

        # Calculate slice indices for this process
        start_idx = rank * batch_size
        end_idx = start_idx + batch_size

        # Handle incomplete batches (if any)
        end_idx = min(end_idx, grad_input.size(0))

        # Get gradient slice for this process
        return grad_input[start_idx:end_idx]


class NTXentLoss(nn.Module):
    def __init__(self,
                 temperature: float = 1.0,
                 eps: float = 1e-6):
        """
        Normalized Temperature-scaled Cross Entropy Loss (NT-Xent) for contrastive learning.

        Computes the InfoNCE loss using positive and negative pairs from multiple views.
        Based on the SimCLR paper: https://arxiv.org/abs/2002.05709

         Attributes:
            temperature (float): Temperature parameter for scaling logits. Defaults to 1.0.
            eps (float): Small value for numerical stability. Defaults to 1e-6.
        """

        super().__init__()
        self.name = 'ntxent'
        self.temperature = temperature
        self.eps = eps

    def compute_metrics(self,
                        anchor_proj: torch.Tensor,
                        positive_proj: torch.Tensor) -> dict:
        """
        Computes ranking metrics for contrastive learning evaluation.

        Args:
            anchor_proj (Tensor): Anchor embeddings [batch_size, feature_dim]
            positive_proj (Tensor): Positive embeddings [batch_size, feature_dim]

        Returns:
            dict: Dictionary containing
                - pos_sim (torch.Tensor): Average cosine similarity between positive pairs
                - neg_sim (torch.Tensor): Average cosine similarity between negative pairs
                - top1_acc (torch.Tensor): Accuracy of positive pair being ranked first
                - top5_acc (torch.Tensor): Accuracy of positive pair being in top 5
                - mean_rank (torch.Tensor): Mean rank position of positive pairs (lower is better)
        """

        # Normalize features
        anchor_proj = nn.functional.normalize(anchor_proj, dim=-1, eps=self.eps)
        positive_proj = nn.functional.normalize(positive_proj, dim=-1, eps=self.eps)

        # Compute positive similarities (diagonal elements)
        pos_sim = (anchor_proj * positive_proj).sum(dim=-1)

        # Compute similarity matrix
        sim_matrix = torch.mm(anchor_proj, positive_proj.t().contiguous())

        # Compute negative similarities (off-diagonal elements)
        mask = ~torch.eye(anchor_proj.size(0),
                          dtype=torch.bool,
                          device=anchor_proj.device)
        neg_sim = sim_matrix[mask]

        # Find the ranks of the positive samples
        # Count how many negatives are more similar
        ranks = (sim_matrix > pos_sim.unsqueeze(1)).sum(dim=-1)

        # Calculate metrics
        top1_acc = (ranks == 0).float().mean()
        top5_acc = (ranks < 5).float().mean()
        mean_rank = ranks.float().mean() + 1

        return {'pos_sim': pos_sim.mean(),
                'neg_sim': neg_sim.mean(),
                'top1_acc': top1_acc,
                'top5_acc': top5_acc,
                'mean_rank': mean_rank}

    def forward(self,
                anchor_proj: torch.Tensor,
                positive_proj: torch.Tensor) -> torch.Tensor:
        """
        Computes the NT-Xent loss using anchor-positive pairs.

        Args:
            anchor_proj (Tensor): Anchor embeddings [batch_size, feature_dim]
            positive_proj (Tensor): Positive embeddings [batch_size, feature_dim]

        Returns:
            Tensor: Scalar loss value
        """
        # Normalize the input features
        anchor_proj = nn.functional.normalize(anchor_proj, dim=1, eps=self.eps)
        positive_proj = nn.functional.normalize(positive_proj, dim=1, eps=self.eps)

        # Gather representations in case of distributed training
        if dist.is_available() and dist.is_initialized():
            anchor_proj_dist = SyncFunction.apply(anchor_proj)
            positive_proj_dist = SyncFunction.apply(positive_proj)
        else:
            anchor_proj_dist = anchor_proj
            positive_proj_dist = positive_proj

        # Concatenate all features for negative examples
        # features: [2 * batch_size, feature_dim]
        # features_dist: [2 * batch_size * world_size, feature_dim]
        features = torch.cat([anchor_proj, positive_proj], dim=0)
        features_dist = torch.cat([anchor_proj_dist, positive_proj_dist], dim=0)

        # Compute similarity matrices
        # cov: [2 * batch_size, 2 * batch_size * world_size]
        cov = torch.mm(features, features_dist.t().contiguous())
        sim = torch.exp(cov / self.temperature)

        # Compute negative similarities
        neg = sim.sum(dim=-1)

        # Subtract self-similarity
        row_sub = torch.Tensor(neg.shape).fill_(math.e ** (1 / self.temperature)).to(neg.device)
        neg = torch.clamp(neg - row_sub, min=self.eps)

        # Compute positive similarities
        pos = torch.exp(torch.sum(anchor_proj * positive_proj, dim=-1) / self.temperature)
        pos = torch.cat([pos, pos], dim=0)  # [2 * batch_size]

        # Compute final loss
        loss = -torch.log(pos / (neg + self.eps)).mean()

        return loss


class ClassificationLoss(nn.Module):
    def __init__(self,
                 label_smoothing: float = 0.1,
                 ignore_index: int = -100):
        """
        Classification cross-entropy loss with label smoothing.

        Attributes:
            label_smoothing: Smoothing factor for label smoothing. Defaults to 0.1.
            ignore_index: Target value that is ignored and does not contribute to gradient
        """

        super().__init__()
        self.name = 'classification'
        self._label_smoothing = label_smoothing
        self._ignore_index = ignore_index

        # Define criterion
        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing,
                                             ignore_index=ignore_index)

    def set_ignore_index(self,
                         ignore_index: int) -> None:
        """
        Updates the ignore index for the classification loss and recreates the criterion.

        Args:
            ignore_index: New target value that is ignored and does not contribute to gradient
                         Typical values:
                         - -100 (default for many frameworks)
                         - tokenizer.pad_token_id (when using transformers)

        Note:
            Reinitializes the CrossEntropyLoss criterion with both:
            - The new ignore_index
            - The previously configured label_smoothing value
        """

        self._ignore_index = ignore_index
        self.criterion = nn.CrossEntropyLoss(label_smoothing=self._label_smoothing,
                                             ignore_index=ignore_index)

    def compute_metrics(self,
                        input: torch.Tensor,
                        target: torch.Tensor) -> dict:
        """
        Computes ranking metrics for contrastive learning evaluation.

        Args:
            input: Model predictions (logits) of shape [batch_size, num_classes]
            target: Ground truth class indices of shape [batch_size]

        Returns:
            dict: Dictionary containing
                - pos_sim (torch.Tensor): Average cosine similarity between positive pairs
                - neg_sim (torch.Tensor): Average cosine similarity between negative pairs
                - top1_acc (torch.Tensor): Accuracy of positive pair being ranked first
                - top5_acc (torch.Tensor): Accuracy of positive pair being in top 5
                - mean_rank (torch.Tensor): Mean rank position of positive pairs (lower is better)
        """

        # Compute positive similarities
        pos_sim = input[torch.arange(len(input)), target]

        # Compute negative similarities
        mask = torch.ones_like(input, dtype=torch.bool)
        mask[torch.arange(len(input)), target] = False
        neg_sim = input[mask].view(input.size(0), -1)

        # Find the ranks of the positive samples
        # Count how many negatives are more similar
        ranks = (input > pos_sim.unsqueeze(1)).sum(dim=-1)

        # Calculate metrics
        top1_acc = (ranks == 0).float().mean()
        top5_acc = (ranks < 5).float().mean()
        mean_rank = ranks.float().mean() + 1

        return {'pos_sim': pos_sim.mean(),
                'neg_sim': neg_sim.mean(),
                'top1_acc': top1_acc,
                'top5_acc': top5_acc,
                'mean_rank': mean_rank}

    def forward(self,
                input: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        """
        Computes the classification cross-entropy loss with label smoothing.

        Args:
            input: Model predictions of shape [batch_size, num_classes]
            target: Ground truth class indices of shape [batch_size]

        Returns:
            torch.Tensor: Scalar loss value
        """

        return self.criterion(input=input,
                              target=target)


def build_objective(config: dict) -> nn.Module:
    """
    Builds and configures a corresponding loss module.

    Args:
        config (dict): Dictionary containing the configuration of the experiment

    Returns:
        nn.Module: Initialized loss module

    Note:
        The configuration must include the following keys:
            For 'ntxent':
                - batch_size: int
                - n_views: int
                - temperature: float (default=1.0)
            For 'classification':
                - label_smoothing: float (default=0.1)
                - ignore_index: int (default=-100)

    Raises:
        ValueError: If loss_name is not recognized or the required arguments are missing.
    """

    # Lowercase the objective name
    objective_name = config['train']['objective']['name'].lower()

    # Build the corresponding loss module
    if objective_name == 'ntxent':
        return NTXentLoss(temperature=config['train']['objective'].get('temperature', 1.0),
                          eps=config['train']['objective'].get('eps', 1e-6))
    elif objective_name == 'classification':
        return ClassificationLoss(label_smoothing=config['train']['objective'].get('label_smoothing', 0.1),
                                  ignore_index=config['train']['objective'].get('ignore_index', -100))
    else:
        raise ValueError(f"Unknown loss name: {objective_name}. "
                         f"Available options: 'ntxent', 'classification'")
