"""
Prototypical Networks for Few-Shot Molecular Property Prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class PrototypicalNetwork(nn.Module):
    """
    Prototypical Networks for few-shot learning.

    Computes class prototypes from support set and classifies queries
    based on distance to prototypes.
    """

    def __init__(self, encoder, distance='euclidean', temperature=1.0):
        """
        Args:
            encoder: Feature encoder network
            distance: Distance metric ('euclidean' or 'cosine')
            temperature: Temperature for softmax
        """
        super(PrototypicalNetwork, self).__init__()

        self.encoder = encoder
        self.distance = distance
        self.temperature = temperature

    def compute_prototypes(self, support_x, support_y, support_smiles=None,
                           support_lengths=None):
        """
        Compute class prototypes from support set.

        Args:
            support_x: Support set features/graphs
            support_y: Support set labels
            support_smiles: SMILES data (for dual encoder)
            support_lengths: SMILES lengths

        Returns:
            prototypes: Class prototypes [n_classes, embed_dim]
            classes: Unique class labels
        """
        # Encode support set
        if support_smiles is not None:
            support_emb = self.encoder(support_x, support_smiles, support_lengths)
        else:
            support_emb = self.encoder(support_x)

        if isinstance(support_emb, dict):
            support_emb = support_emb['fused']

        # Compute prototypes
        classes = torch.unique(support_y)
        prototypes = []

        for c in classes:
            mask = support_y == c
            class_emb = support_emb[mask]
            prototype = class_emb.mean(dim=0)
            prototypes.append(prototype)

        prototypes = torch.stack(prototypes)
        return prototypes, classes

    def forward(self, query_x, support_x, support_y,
                query_smiles=None, query_lengths=None,
                support_smiles=None, support_lengths=None):
        """
        Classify query samples using prototypes.

        Args:
            query_x: Query features/graphs
            support_x: Support features/graphs
            support_y: Support labels
            *_smiles, *_lengths: Optional SMILES data

        Returns:
            log_probs: Log probabilities [n_query, n_classes]
        """
        # Compute prototypes
        prototypes, classes = self.compute_prototypes(
            support_x, support_y, support_smiles, support_lengths
        )

        # Encode queries
        if query_smiles is not None:
            query_emb = self.encoder(query_x, query_smiles, query_lengths)
        else:
            query_emb = self.encoder(query_x)

        if isinstance(query_emb, dict):
            query_emb = query_emb['fused']

        # Compute distances
        if self.distance == 'euclidean':
            # Squared Euclidean distance
            dists = torch.cdist(query_emb, prototypes, p=2) ** 2
            logits = -dists / self.temperature
        elif self.distance == 'cosine':
            # Cosine similarity
            query_norm = F.normalize(query_emb, dim=-1)
            proto_norm = F.normalize(prototypes, dim=-1)
            logits = torch.mm(query_norm, proto_norm.t()) / self.temperature
        else:
            raise ValueError(f"Unknown distance: {self.distance}")

        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs, classes


class ProtoTrainer:
    """Trainer for Prototypical Networks."""

    def __init__(self, proto_net, lr=0.001, weight_decay=1e-5):
        self.proto_net = proto_net
        self.optimizer = torch.optim.Adam(
            proto_net.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=20, gamma=0.5
        )

    def train_epoch(self, episode_generator, n_episodes=100):
        """Train one epoch on episodes."""
        self.proto_net.train()
        total_loss = 0
        total_acc = 0

        for i, episode in enumerate(episode_generator):
            if i >= n_episodes:
                break

            loss, acc = self._train_episode(episode)
            total_loss += loss
            total_acc += acc

        self.scheduler.step()

        return total_loss / n_episodes, total_acc / n_episodes

    def _train_episode(self, episode):
        """Train on single episode."""
        support_x, support_y = episode['support']
        query_x, query_y = episode['query']

        # Optional SMILES data
        support_smiles = episode.get('support_smiles')
        support_lengths = episode.get('support_lengths')
        query_smiles = episode.get('query_smiles')
        query_lengths = episode.get('query_lengths')

        self.optimizer.zero_grad()

        # Forward
        log_probs, classes = self.proto_net(
            query_x, support_x, support_y,
            query_smiles, query_lengths,
            support_smiles, support_lengths
        )

        # Map query labels to class indices
        class_to_idx = {c.item(): i for i, c in enumerate(classes)}
        query_y_idx = torch.tensor([class_to_idx[y.item()] for y in query_y],
                                    device=query_y.device)

        # Loss
        loss = F.nll_loss(log_probs, query_y_idx)
        loss.backward()
        self.optimizer.step()

        # Accuracy
        pred = log_probs.argmax(dim=-1)
        acc = (pred == query_y_idx).float().mean().item()

        return loss.item(), acc

    def evaluate(self, episode_generator, n_episodes=100):
        """Evaluate on episodes."""
        self.proto_net.eval()
        total_acc = 0

        with torch.no_grad():
            for i, episode in enumerate(episode_generator):
                if i >= n_episodes:
                    break

                support_x, support_y = episode['support']
                query_x, query_y = episode['query']

                log_probs, classes = self.proto_net(
                    query_x, support_x, support_y
                )

                class_to_idx = {c.item(): i for i, c in enumerate(classes)}
                query_y_idx = torch.tensor([class_to_idx[y.item()] for y in query_y],
                                            device=query_y.device)

                pred = log_probs.argmax(dim=-1)
                acc = (pred == query_y_idx).float().mean().item()
                total_acc += acc

        return total_acc / n_episodes
