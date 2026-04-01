import numpy as np
from tqdm import tqdm
from typing import List, Tuple, Optional
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

# Set the bar colors for distributed setting
GREEN = '\033[92m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
MAGENTA = '\033[95m'
RESET = '\033[0m'
colors = [GREEN, YELLOW, BLUE, MAGENTA]

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, Sampler

from data.event_tokenizer import EventTokenizer


class LinearProbeEvaluator:
    def __init__(self,
                 model: nn.Module,
                 event_tokenizer: EventTokenizer,
                 train_dataset: Dataset,
                 val_dataset: Dataset,
                 train_sampler: Sampler = None,
                 val_sampler: Sampler = None,
                 c_values: List[float] = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
                 rank: int = 0,
                 local_rank: int = 0):
        """
        Evaluates model representations using linear probing.

        This class handles feature extraction from a frozen model and trains a linear classifier
        on top to evaluate representation quality. Supports distributed training environments.

        Args:
            model (nn.Module): The model whose representations will be evaluated (will be frozen)
            event_tokenizer (EventTokenizer): Tokenizer for converting events to model inputs
            train_dataset (Dataset): Training dataset for linear probe
            val_dataset (Dataset): Validation dataset for linear probe
            train_sampler (Sampler, optional): Distributed sampler for training data
            val_sampler (Sampler, optional): Distributed sampler for validation data
            c_values (List[float], optional): List of inverse of regularization strength for linear probe
            rank (int, optional): Process rank in distributed training. Defaults to 0.
            local_rank (int): The rank (index) of the current process within its local machine. Defaults to 0.
        """

        self.model = model
        self.event_tokenizer = event_tokenizer
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.train_sampler = train_sampler
        self.val_sampler = val_sampler
        self.c_values = c_values
        self.rank = rank
        self.local_rank = local_rank
        self.device = f'cuda:{local_rank}'

    @staticmethod
    def _collate_fn(batch: List[Tuple[torch.Tensor, int]]) -> Tuple[List[torch.Tensor], List[int]]:
        """
        Collates a batch of event data samples into separated events and labels.

        This function takes a list of (events, label) tuples and separates them into:
        1. A list of event tensors
        2. A list of corresponding integer labels

        Args:
            batch: A list of samples where each sample is a tuple containing:
                - events (torch.Tensor): Event data tensor of shape [num_events, 4] containing (t, x, y, p) entries
                - label (int): Integer class label for the sample

        Returns:
            A tuple containing:
                - events (List[torch.Tensor]): List of event tensors from the batch
                - labels (List[int]): List of corresponding integer labels
        """

        events = [sample[0] for sample in batch]
        labels = [sample[1] for sample in batch]

        return events, labels

    def _build_dataloader(self,
                          event_dataset: Dataset,
                          batch_size: int = 64,
                          num_workers: int = 8,
                          shuffle: bool = False,
                          sampler: Sampler = None,
                          drop_last: bool = False):
        """
        Builds a tokenizing dataloader for event data.

        Args:
            event_dataset (Dataset): Dataset containing event data of shape [num_events, 4] containing (t, x, y, p) entries.
            batch_size (int, optional): Batch size. Defaults to 64.
            num_workers (int, optional): Number of workers for data loading. Defaults to 8.
            shuffle (bool, optional): Whether to shuffle data. Defaults to False.
            sampler (Sampler, optional): Distributed sampler. Defaults to None.
            drop_last (bool, optional): Whether to drop incomplete batches. Defaults to False.

            Returns:
                TokenizedDataLoader: A DataLoader-like iterator that yields tokenized batches
        """

        # Create a data loader
        dataloader = DataLoader(event_dataset,
                                batch_size=batch_size,
                                collate_fn=self._collate_fn,
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
                         dataloader: DataLoader,
                         event_tokenizer: EventTokenizer,
                         local_rank: int = 0):

                self.dataloader = dataloader
                self.sampler = dataloader.sampler
                self.event_tokenizer = event_tokenizer
                self.local_rank = local_rank

            def __iter__(self):
                for events, labels in self.dataloader:
                    device = f'cuda:{self.local_rank}'

                    # Tokenize the events
                    event_representations = torch.stack([self.event_tokenizer(sample)
                                                         for sample in events], dim=0).to(device)

                    # Convert the label list to tensor
                    labels = torch.tensor(labels).to(device)

                    yield event_representations, labels

            def __len__(self):
                return len(self.dataloader)

        return TokenizedDataLoader(dataloader=dataloader,
                                   event_tokenizer=self.event_tokenizer,
                                   local_rank=self.local_rank)

    @torch.no_grad()
    def extract_features(self,
                         dataloader: DataLoader) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Extracts features from the frozen model.

        Args:
            dataloader (Dataloader): Dataloader providing batches of tokenized events each of shape
                                                [batch_size, 1, token_dim, grid_height, grid_width]

        Returns:
            Tuple containing:
                - features (torch.Tensor or None): Extracted features (on rank 0) or None
                - targets (torch.Tensor or None): Corresponding labels (on rank 0) or None
        """

        features, targets = [], []

        # Initialize progress bar
        pbar = tqdm(iterable=dataloader,
                    desc=f'Extracting features | {colors[self.rank % len(colors)]}Rank {self.rank}{RESET}',
                    ncols=100,
                    leave=False,
                    unit='batch')

        for batch in pbar:
            # Assuming batch is (events, label)
            events, labels = batch[0].to(self.device), batch[1].to(self.device)

            # Get features from frozen model
            if dist.get_world_size() > 1:
                feat = self.model.module.extract_features(events)
            else:
                feat = self.model.extract_features(events)
            features.append(feat)
            targets.append(labels)

        features = torch.cat(features, dim=0)
        targets = torch.cat(targets, dim=0)

        # Gather features across all GPUs if distributed
        if dist.is_initialized():
            if self.rank == 0:
                all_features = [torch.zeros_like(features) for _ in range(dist.get_world_size())]
                all_targets = [torch.zeros_like(targets) for _ in range(dist.get_world_size())]
            else:
                all_features = None
                all_targets = None

            dist.gather(features, all_features, dst=0)
            dist.gather(targets, all_targets, dst=0)

            if self.rank == 0:
                features = torch.cat(all_features, dim=0).cpu()
                targets = torch.cat(all_targets, dim=0).cpu()
            else:
                return None, None

        return features, targets

    def evaluate(self) -> Tuple[float, float, float]:
        """
        Runs full linear evaluation pipeline.

        Returns:
            Tuple containing:
                - train_acc (float): Accuracy on linear probe training set
                - val_acc (float): Accuracy on linear probe validation set
         """

        # Build corresponding dataloaders
        train_dataloader = self._build_dataloader(event_dataset=self.train_dataset,
                                                  sampler=self.train_sampler)
        val_dataloader = self._build_dataloader(event_dataset=self.val_dataset,
                                                sampler=self.val_sampler)

        # Extract features
        train_features, train_targets = self.extract_features(train_dataloader)
        val_features, val_targets = self.extract_features(val_dataloader)

        if self.rank != 0:
            return 0.0, 0.0, 0.0  # Only rank 0 computes the actual metrics

        # Standardize features
        mean = train_features.mean(dim=0)
        std = train_features.std(dim=0)
        train_features = (train_features - mean) / (std + 1e-6)
        val_features = (val_features - mean) / (std + 1e-6)

        # Reserve variables for the best result
        best_C = 1.0
        best_val_acc = 0.0
        best_train_acc = 0.0

        # Sweep over C values
        for C in self.c_values:
            # Train logistic regression
            classifier = LogisticRegression(random_state=0,
                                            max_iter=3000,
                                            solver='lbfgs',
                                            C=C,
                                            n_jobs=-1)
            classifier.fit(train_features.numpy(), train_targets.numpy())

            # Calculate accuracies
            train_acc = classifier.score(train_features.numpy(), train_targets.numpy())
            val_acc = classifier.score(val_features.numpy(), val_targets.numpy())

            # Update best parameters if current is better
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_train_acc = train_acc
                best_C = C

        return best_train_acc, best_val_acc, best_C


class RetrievalEvaluator(LinearProbeEvaluator):
    def __init__(self,
                 model: torch.nn.Module,
                 event_tokenizer: EventTokenizer,
                 query_dataset: Dataset,
                 gallery_dataset: Dataset = None,
                 query_sampler: Sampler = None,
                 gallery_sampler: Sampler = None,
                 top_k: List[int] = [1, 5, 10],
                 rank: int = 0,
                 local_rank: int = 0):
        """
        Evaluates model representations using mean Average Precision (mAP) for retrieval tasks.
        Inherits from LinearProbeEvaluator to reuse feature extraction infrastructure while
        implementing retrieval-specific evaluation metrics.

        Args:
            model (nn.Module): The model whose representations will be evaluated (will be frozen)
            event_tokenizer (EventTokenizer): Tokenizer for converting events to model inputs
            query_dataset (Dataset): Dataset containing query samples
            gallery_dataset (Dataset): Dataset containing gallery samples to retrieve from
            query_sampler (Sampler, optional): Distributed sampler for query data
            gallery_sampler (Sampler, optional): Distributed sampler for gallery data
            top_k (List[int], optional): List of k values for mAP@k evaluation
            rank (int, optional): Process rank in distributed training. Defaults to 0.
            local_rank (int): The rank (index) of the current process within its local machine. Defaults to 0.
        """

        # If no gallery dataset is provided, copy the query dataset
        self.same_dataset = False
        if gallery_dataset is None:
            gallery_dataset = query_dataset
            gallery_sampler = query_sampler
            self.same_dataset = True

        super().__init__(model=model,
                         event_tokenizer=event_tokenizer,
                         train_dataset=query_dataset,
                         val_dataset=gallery_dataset,
                         train_sampler=query_sampler,
                         val_sampler=gallery_sampler,
                         rank=rank,
                         local_rank=local_rank)

        self.query_dataset = query_dataset
        self.gallery_dataset = gallery_dataset
        self.query_sampler = query_sampler
        self.gallery_sampler = gallery_sampler
        self.top_k = top_k
        self.device = f'cuda:{local_rank}'

    @staticmethod
    def compute_similarity_matrix(query_features: torch.Tensor,
                                  gallery_features: torch.Tensor) -> np.ndarray:
        """
        Computes cosine similarity matrix between query and gallery features.

        Args:
            query_features (torch.Tensor): Feature vectors for query samples.
                                          Shape: [num_queries, feature_dim]
            gallery_features (torch.Tensor): Feature vectors for gallery samples.
                                            Shape: [num_gallery, feature_dim]

        Returns:
            np.ndarray: Cosine similarity matrix of shape [num_queries, num_gallery]
                        where element [i,j] contains the cosine similarity between
                        query i and gallery item j.
        """

        # Normalize features
        query_features = query_features / query_features.norm(dim=1, keepdim=True)
        gallery_features = gallery_features / gallery_features.norm(dim=1, keepdim=True)

        # Compute cosine similarity
        sim_matrix = torch.mm(query_features, gallery_features.T)

        return sim_matrix.cpu().numpy()

    @staticmethod
    def compute_ap(scores: np.ndarray,
                   labels: np.ndarray) -> float:
        """
        Computes Average Precision (AP) for a single query.

        Args:
            scores (np.ndarray): Similarity scores for gallery items relative to the query.
                                Shape: [num_gallery]
            labels (np.ndarray): Binary relevance labels (1=relevant, 0=irrelevant) for
                                gallery items relative to the query. Shape: [num_gallery]

        Returns:
            float: Average Precision score between 0 and 1, where 1 indicates perfect
                   retrieval with all relevant items ranked first.
        """

        return average_precision_score(labels, scores)

    def evaluate(self) -> dict:
        """
        Executes the full mAP evaluation pipeline.

        Steps:
        1. Extracts features for all query and gallery samples
        2. Computes cosine similarity matrix between queries and gallery
        3. Calculates mAP@k for each specified k value
        4. Returns results dictionary with mAP scores

        Returns:
            dict: Dictionary containing mAP scores for each k value in top_k.
                  Format: {'mAP@1': float, 'mAP@5': float, ...}

        Note:
            In distributed settings, only rank 0 returns actual metrics.
            Other ranks return zero-filled dictionaries.
        """

        # Build corresponding dataloaders and extract features
        query_dataloader = self._build_dataloader(event_dataset=self.query_dataset,
                                                  sampler=self.query_sampler)
        query_features, query_targets = self.extract_features(query_dataloader)

        if self.same_dataset:
            gallery_features, gallery_targets = query_features, query_targets
        else:
            gallery_dataloader = self._build_dataloader(event_dataset=self.gallery_dataset,
                                                        sampler=self.gallery_sampler)
            gallery_features, gallery_targets = self.extract_features(gallery_dataloader)

        if self.rank != 0:
            return {f'mAP@{k}': 0.0 for k in self.top_k}  # Only rank 0 computes the actual metrics

        # Compute similarity matrix
        sim_matrix = self.compute_similarity_matrix(query_features=query_features.cuda(),
                                                    gallery_features=gallery_features.cuda())

        # Convert targets to numpy
        query_targets = query_targets.numpy()
        gallery_targets = gallery_targets.numpy()

        # Compute mAP@k
        results = {}
        for k in self.top_k:
            aps = []
            for i in range(len(query_targets)):

                # Get similarity scores and relevance for query i
                scores = sim_matrix[i]
                y_true = (gallery_targets == query_targets[i]).astype(int)

                # Only exclude self-retrieval if query == gallery
                if self.same_dataset:
                    scores[i] = -np.inf  # Mask out the query itself

                # For mAP@k, consider only top-k results
                if k < len(gallery_targets):
                    top_k_indices = np.argpartition(-scores, k)[:k]
                    scores = scores[top_k_indices]
                    y_true = y_true[top_k_indices]

                ap = self.compute_ap(scores, y_true)
                aps.append(ap)

            results[f'mAP@{k}'] = np.mean(aps)

        return results


def build_evaluator(model: nn.Module,
                    event_tokenizer: EventTokenizer,
                    train_dataset: Dataset,
                    val_dataset: Dataset,
                    train_sampler: Sampler = None,
                    val_sampler: Sampler = None,
                    rank: int = 0,
                    local_rank: int = 0) -> LinearProbeEvaluator:
    """
    Factory function to create a LinearProbeEvaluator instance.

    Args:
        model (nn.Module): Model to evaluate
        event_tokenizer (EventTokenizer): Tokenizer for event data
        train_dataset (Dataset): Linear probe training dataset
        val_dataset (Dataset): Linear probe validation dataset
        train_sampler (Sampler, optional): Training data sampler
        val_sampler (Sampler, optional): Validation data sampler
        rank (int, optional): Process rank. Defaults to 0.
        local_rank (int, optional): The rank (index) of the current process within its local machine. Defaults to 0.

    Returns:
        LinearProbeEvaluator: Configured evaluator instance
    """

    return LinearProbeEvaluator(model=model,
                                event_tokenizer=event_tokenizer,
                                train_dataset=train_dataset,
                                val_dataset=val_dataset,
                                train_sampler=train_sampler,
                                val_sampler=val_sampler,
                                rank=rank,
                                local_rank=local_rank)
