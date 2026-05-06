"""
Training Utilities
Trainer class, early stopping, and metric tracking
"""

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import time
import os


class EarlyStopping:
    """Early stopping to prevent overfitting."""

    def __init__(self, patience=10, min_delta=0.0, mode='min'):
        """
        Args:
            patience: Number of epochs to wait
            min_delta: Minimum improvement
            mode: 'min' or 'max' for loss or accuracy
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score):
        """
        Check if should stop.

        Args:
            score: Current score

        Returns:
            True if improved, False otherwise
        """
        if self.best_score is None:
            self.best_score = score
            return True

        if self.mode == 'min':
            improved = score < self.best_score - self.min_delta
        else:
            improved = score > self.best_score + self.min_delta

        if improved:
            self.best_score = score
            self.counter = 0
            return True
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return False


class MetricTracker:
    """Track training metrics."""

    def __init__(self, metrics=None):
        """
        Args:
            metrics: List of metric names to track
        """
        self.metrics = metrics or ['loss']
        self.history = {m: [] for m in self.metrics}
        self.epoch_values = {m: [] for m in self.metrics}

    def update(self, metric_name, value):
        """Update metric value."""
        if metric_name in self.epoch_values:
            self.epoch_values[metric_name].append(value)

    def end_epoch(self):
        """End epoch and compute averages."""
        for metric in self.metrics:
            if self.epoch_values[metric]:
                avg = np.mean(self.epoch_values[metric])
                self.history[metric].append(avg)
                self.epoch_values[metric] = []

    def get_history(self, metric_name):
        """Get metric history."""
        return self.history.get(metric_name, [])

    def get_last(self, metric_name):
        """Get last metric value."""
        history = self.history.get(metric_name, [])
        return history[-1] if history else None


class Trainer:
    """
    Generic trainer for PyTorch models.
    """

    def __init__(self, model, optimizer, criterion, device='cuda',
                 scheduler=None, grad_clip=None):
        """
        Args:
            model: PyTorch model
            optimizer: Optimizer
            criterion: Loss function
            device: Device to use
            scheduler: Learning rate scheduler
            grad_clip: Gradient clipping value
        """
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.scheduler = scheduler
        self.grad_clip = grad_clip

        self.model.to(device)

    def train_epoch(self, dataloader, epoch=0):
        """
        Train for one epoch.

        Args:
            dataloader: Training data loader
            epoch: Current epoch number

        Returns:
            Average loss
        """
        self.model.train()
        total_loss = 0
        n_batches = 0

        pbar = tqdm(dataloader, desc=f'Epoch {epoch}')

        for batch in pbar:
            loss = self._train_step(batch)
            total_loss += loss
            n_batches += 1

            pbar.set_postfix({'loss': f'{loss:.4f}'})

        return total_loss / n_batches

    def _train_step(self, batch):
        """Single training step."""
        self.optimizer.zero_grad()

        # Handle different batch formats
        if isinstance(batch, dict):
            inputs = {k: v.to(self.device) for k, v in batch.items()
                      if k != 'labels' and isinstance(v, torch.Tensor)}
            labels = batch.get('labels', batch.get('y'))
            if labels is not None:
                labels = labels.to(self.device)
        elif isinstance(batch, (tuple, list)):
            inputs, labels = batch[0], batch[1]
            if isinstance(inputs, torch.Tensor):
                inputs = inputs.to(self.device)
            labels = labels.to(self.device)
        else:
            inputs = batch.to(self.device)
            labels = None

        # Forward
        outputs = self.model(inputs) if not isinstance(inputs, dict) else self.model(**inputs)

        # Compute loss
        if isinstance(outputs, dict):
            outputs = list(outputs.values())[0]
        loss = self.criterion(outputs.squeeze(), labels.float() if labels is not None else outputs)

        # Backward
        loss.backward()

        if self.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

        self.optimizer.step()

        return loss.item()

    @torch.no_grad()
    def evaluate(self, dataloader):
        """
        Evaluate model.

        Args:
            dataloader: Evaluation data loader

        Returns:
            Dict with evaluation metrics
        """
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        n_batches = 0

        for batch in dataloader:
            # Handle batch format
            if isinstance(batch, dict):
                inputs = {k: v.to(self.device) for k, v in batch.items()
                          if k != 'labels' and isinstance(v, torch.Tensor)}
                labels = batch.get('labels', batch.get('y'))
                if labels is not None:
                    labels = labels.to(self.device)
            elif isinstance(batch, (tuple, list)):
                inputs, labels = batch[0], batch[1]
                if isinstance(inputs, torch.Tensor):
                    inputs = inputs.to(self.device)
                labels = labels.to(self.device)
            else:
                inputs = batch.to(self.device)
                labels = None

            # Forward
            outputs = self.model(inputs) if not isinstance(inputs, dict) else self.model(**inputs)

            if isinstance(outputs, dict):
                outputs = list(outputs.values())[0]

            # Loss
            if labels is not None:
                loss = self.criterion(outputs.squeeze(), labels.float())
                total_loss += loss.item()

            all_preds.append(outputs.cpu())
            if labels is not None:
                all_labels.append(labels.cpu())
            n_batches += 1

        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels) if all_labels else None

        return {
            'loss': total_loss / n_batches,
            'predictions': all_preds,
            'labels': all_labels
        }

    def fit(self, train_loader, val_loader=None, n_epochs=100,
            early_stopping=None, save_path=None):
        """
        Full training loop.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            n_epochs: Number of epochs
            early_stopping: EarlyStopping instance
            save_path: Path to save best model

        Returns:
            Training history
        """
        history = {'train_loss': [], 'val_loss': []}
        best_val_loss = float('inf')

        for epoch in range(n_epochs):
            # Train
            train_loss = self.train_epoch(train_loader, epoch)
            history['train_loss'].append(train_loss)

            # Validate
            if val_loader is not None:
                val_results = self.evaluate(val_loader)
                val_loss = val_results['loss']
                history['val_loss'].append(val_loss)

                print(f'Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}')

                # Early stopping
                if early_stopping is not None:
                    improved = early_stopping(val_loss)
                    if improved and save_path is not None:
                        self.save_model(save_path)
                    if early_stopping.early_stop:
                        print(f'Early stopping at epoch {epoch}')
                        break

                # Save best
                if val_loss < best_val_loss and save_path is not None:
                    best_val_loss = val_loss
                    self.save_model(save_path)
            else:
                print(f'Epoch {epoch}: train_loss={train_loss:.4f}')

            # Scheduler step
            if self.scheduler is not None:
                self.scheduler.step()

        return history

    def save_model(self, path):
        """Save model checkpoint."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)

    def load_model(self, path):
        """Load model checkpoint."""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
