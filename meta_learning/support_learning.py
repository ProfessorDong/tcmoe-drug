"""
Support-based Learning for Few-Shot Molecular Property Prediction
Similar to Matching Networks approach
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random


class SupportLearner(nn.Module):
    """
    Support-based learner using attention over support set.
    Implements the matching networks approach for molecular property prediction.
    """

    def __init__(self, encoder, embedding_dim=128, use_lstm_embedding=True,
                 lstm_depth=3):
        """
        Args:
            encoder: Feature encoder
            embedding_dim: Embedding dimension
            use_lstm_embedding: Use LSTM-based embedding refinement
            lstm_depth: Number of LSTM processing steps
        """
        super(SupportLearner, self).__init__()

        self.encoder = encoder
        self.embedding_dim = embedding_dim
        self.use_lstm_embedding = use_lstm_embedding
        self.lstm_depth = lstm_depth

        encoder_dim = encoder.output_dim

        # Projection to embedding space
        self.projection = nn.Sequential(
            nn.Linear(encoder_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.ReLU()
        )

        # LSTM for embedding refinement (Full Context Embeddings)
        if use_lstm_embedding:
            self.lstm = nn.LSTMCell(embedding_dim * 2, embedding_dim)
            self.attention_layer = nn.Linear(embedding_dim, 1)

    def encode(self, x, smiles=None, lengths=None):
        """Encode inputs."""
        if smiles is not None:
            emb = self.encoder(x, smiles, lengths)
        else:
            emb = self.encoder(x)

        if isinstance(emb, dict):
            emb = emb['fused']

        return self.projection(emb)

    def forward(self, query_x, support_x, support_y,
                query_smiles=None, query_lengths=None,
                support_smiles=None, support_lengths=None):
        """
        Predict using support set.

        Args:
            query_x: Query inputs
            support_x: Support inputs
            support_y: Support labels
            *_smiles, *_lengths: Optional SMILES data

        Returns:
            predictions: Predicted probabilities for queries
        """
        # Encode
        query_emb = self.encode(query_x, query_smiles, query_lengths)
        support_emb = self.encode(support_x, support_smiles, support_lengths)

        # Refine embeddings with LSTM
        if self.use_lstm_embedding:
            query_emb, support_emb = self._lstm_embedding(
                query_emb, support_emb
            )

        # Compute attention (cosine similarity)
        query_norm = F.normalize(query_emb, dim=-1)
        support_norm = F.normalize(support_emb, dim=-1)

        attention = torch.mm(query_norm, support_norm.t())  # [n_query, n_support]
        attention = F.softmax(attention, dim=-1)

        # Weighted sum of support labels
        support_y = support_y.float()
        predictions = torch.mv(attention, support_y)

        return predictions

    def _lstm_embedding(self, query_emb, support_emb):
        """
        Refine embeddings using bidirectional LSTM attention.
        Implements Full Context Embeddings from Matching Networks.
        """
        n_query = query_emb.size(0)
        n_support = support_emb.size(0)

        # Initialize
        h_query = query_emb.clone()
        h_support = support_emb.clone()

        # LSTM states
        lstm_h = torch.zeros(n_query, self.embedding_dim, device=query_emb.device)
        lstm_c = torch.zeros(n_query, self.embedding_dim, device=query_emb.device)

        for _ in range(self.lstm_depth):
            # Attention from query to support
            attn = torch.mm(h_query, h_support.t())
            attn = F.softmax(attn, dim=-1)
            read = torch.mm(attn, h_support)

            # LSTM step
            lstm_input = torch.cat([h_query, read], dim=-1)
            lstm_h, lstm_c = self.lstm(lstm_input, (lstm_h, lstm_c))

            # Update query embedding
            h_query = h_query + lstm_h

        return h_query, h_support


class EpisodeSampler:
    """
    Samples episodes for few-shot learning.
    """

    def __init__(self, dataset, n_way=2, n_support=5, n_query=15,
                 n_episodes=100):
        """
        Args:
            dataset: Dataset with molecules and labels
            n_way: Number of classes per episode
            n_support: Number of support samples per class
            n_query: Number of query samples per class
            n_episodes: Number of episodes to generate
        """
        self.dataset = dataset
        self.n_way = n_way
        self.n_support = n_support
        self.n_query = n_query
        self.n_episodes = n_episodes

        # Build class indices
        self._build_class_indices()

    def _build_class_indices(self):
        """Build mapping from class to sample indices."""
        self.class_indices = {}

        if hasattr(self.dataset, 'y'):
            labels = self.dataset.y
        elif hasattr(self.dataset, 'labels'):
            labels = self.dataset.labels
        else:
            raise ValueError("Dataset must have 'y' or 'labels' attribute")

        for idx, label in enumerate(labels):
            if isinstance(label, np.ndarray):
                label = label.item() if label.size == 1 else tuple(label)
            if label not in self.class_indices:
                self.class_indices[label] = []
            self.class_indices[label].append(idx)

        # Filter classes with enough samples
        min_samples = self.n_support + self.n_query
        self.valid_classes = [
            c for c, idxs in self.class_indices.items()
            if len(idxs) >= min_samples
        ]

    def __iter__(self):
        """Generate episodes."""
        for _ in range(self.n_episodes):
            yield self._sample_episode()

    def _sample_episode(self):
        """Sample single episode."""
        # Sample classes
        classes = random.sample(self.valid_classes, self.n_way)

        support_indices = []
        query_indices = []
        support_labels = []
        query_labels = []

        for i, c in enumerate(classes):
            indices = random.sample(
                self.class_indices[c],
                self.n_support + self.n_query
            )

            support_indices.extend(indices[:self.n_support])
            query_indices.extend(indices[self.n_support:])
            support_labels.extend([i] * self.n_support)
            query_labels.extend([i] * self.n_query)

        # Get data
        support_x = self._get_batch(support_indices)
        query_x = self._get_batch(query_indices)

        episode = {
            'support': (support_x, torch.tensor(support_labels)),
            'query': (query_x, torch.tensor(query_labels)),
            'classes': classes
        }

        return episode

    def _get_batch(self, indices):
        """Get batch of samples."""
        if hasattr(self.dataset, 'X'):
            return torch.tensor(self.dataset.X[indices])
        elif hasattr(self.dataset, '__getitem__'):
            batch = [self.dataset[i] for i in indices]
            return torch.stack(batch)
        else:
            raise ValueError("Dataset format not supported")


class TaskSupportGenerator:
    """
    Generates support sets for different property prediction tasks.
    Supports multi-task learning scenarios.
    """

    def __init__(self, dataset, tasks, n_pos=5, n_neg=5, n_trials=100,
                 train_dataset=None):
        """
        Args:
            dataset: Test dataset
            tasks: List of task names
            n_pos: Number of positive samples
            n_neg: Number of negative samples
            n_trials: Number of trials per task
            train_dataset: Optional separate training dataset for support
        """
        self.dataset = dataset
        self.tasks = tasks
        self.n_pos = n_pos
        self.n_neg = n_neg
        self.n_trials = n_trials
        self.train_dataset = train_dataset if train_dataset else dataset

        # Build task indices
        self._build_task_indices()

    def _build_task_indices(self):
        """Build positive/negative indices for each task."""
        self.task_pos_indices = {}
        self.task_neg_indices = {}

        y = self.train_dataset.y if hasattr(self.train_dataset, 'y') else None

        for task_idx, task in enumerate(self.tasks):
            if y is not None:
                if y.ndim > 1:
                    task_y = y[:, task_idx]
                else:
                    task_y = y

                self.task_pos_indices[task] = np.where(task_y == 1)[0].tolist()
                self.task_neg_indices[task] = np.where(task_y == 0)[0].tolist()
            else:
                self.task_pos_indices[task] = []
                self.task_neg_indices[task] = []

    def __iter__(self):
        """Generate (task, support) pairs."""
        for task in self.tasks:
            for _ in range(self.n_trials):
                support = self._sample_support(task)
                yield task, support

    def _sample_support(self, task):
        """Sample support set for task."""
        pos_indices = self.task_pos_indices.get(task, [])
        neg_indices = self.task_neg_indices.get(task, [])

        # Sample positive examples
        n_pos = min(self.n_pos, len(pos_indices))
        sampled_pos = random.sample(pos_indices, n_pos) if n_pos > 0 else []

        # Sample negative examples
        n_neg = min(self.n_neg, len(neg_indices))
        sampled_neg = random.sample(neg_indices, n_neg) if n_neg > 0 else []

        # Combine
        indices = sampled_pos + sampled_neg
        labels = [1] * len(sampled_pos) + [0] * len(sampled_neg)

        # Shuffle
        combined = list(zip(indices, labels))
        random.shuffle(combined)
        indices, labels = zip(*combined) if combined else ([], [])

        # Get data
        support_x = self._get_batch(list(indices))
        support_y = torch.tensor(list(labels))

        return {'x': support_x, 'y': support_y, 'indices': indices}

    def _get_batch(self, indices):
        """Get batch of samples from dataset."""
        if len(indices) == 0:
            return None

        if hasattr(self.train_dataset, 'X'):
            return torch.tensor(self.train_dataset.X[indices])
        return None
