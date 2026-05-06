"""
Model-Agnostic Meta-Learning (MAML) for Drug Discovery
Enables rapid adaptation to new molecular property prediction tasks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad
import copy
from collections import OrderedDict
import numpy as np
from tqdm import tqdm


class MAML(nn.Module):
    """
    Model-Agnostic Meta-Learning (MAML) wrapper.

    Enables rapid adaptation of any model to new tasks with few examples.
    Particularly useful for predicting new molecular properties with limited data.
    """

    def __init__(self, model, inner_lr=0.01, first_order=False):
        """
        Args:
            model: Base model to be meta-learned
            inner_lr: Learning rate for inner (task-specific) updates
            first_order: Use first-order approximation (faster but less accurate)
        """
        super(MAML, self).__init__()

        self.model = model
        self.inner_lr = inner_lr
        self.first_order = first_order

        # Store parameter names for easy access
        self.param_names = [name for name, _ in model.named_parameters()]

    def forward(self, *args, **kwargs):
        """Forward pass through the base model."""
        return self.model(*args, **kwargs)

    def adapt(self, support_x, support_y, num_steps=1, task_name=None):
        """
        Adapt model to a new task using support set.

        Args:
            support_x: Support set inputs
            support_y: Support set labels
            num_steps: Number of gradient steps
            task_name: Task name for multi-task models

        Returns:
            adapted_params: Task-adapted parameters
        """
        # Get current parameters
        params = OrderedDict(self.model.named_parameters())

        # Perform inner loop updates
        for step in range(num_steps):
            # Forward pass with current params
            if task_name is not None:
                pred = self._forward_with_params(support_x, params)[task_name]
            else:
                pred = self._forward_with_params(support_x, params)

            # Compute loss
            if pred.dim() > 1 and pred.size(-1) > 1:
                loss = F.cross_entropy(pred, support_y)
            else:
                loss = F.mse_loss(pred.squeeze(), support_y.float())

            # Compute gradients
            grads = grad(
                loss, params.values(),
                create_graph=not self.first_order,
                allow_unused=True
            )

            # Update parameters
            params = OrderedDict(
                (name, param - self.inner_lr * g if g is not None else param)
                for (name, param), g in zip(params.items(), grads)
            )

        return params

    def _forward_with_params(self, x, params):
        """Forward pass with given parameters."""
        # Temporarily replace model parameters
        original_params = {}
        for name, p in self.model.named_parameters():
            original_params[name] = p

        for name, param in params.items():
            self._set_param(self.model, name, param)

        # Forward
        output = self.model(x)

        # Restore original parameters
        for name, p in original_params.items():
            self._set_param(self.model, name, p)

        return output

    def _set_param(self, model, name, value):
        """Set a parameter by name, wrapping as nn.Parameter if needed."""
        parts = name.split('.')
        module = model
        for part in parts[:-1]:
            if part.isdigit():
                module = module[int(part)]
            else:
                module = getattr(module, part)
        # Wrap plain tensors as nn.Parameter to satisfy nn.Module
        if not isinstance(value, nn.Parameter):
            value = nn.Parameter(value)
        setattr(module, parts[-1], value)

    def meta_forward(self, support_x, support_y, query_x, num_steps=1, task_name=None):
        """
        Full meta-learning forward pass.

        Args:
            support_x: Support set inputs
            support_y: Support set labels
            query_x: Query set inputs
            num_steps: Inner loop steps
            task_name: Task name

        Returns:
            Query set predictions with adapted parameters
        """
        # Adapt to task
        adapted_params = self.adapt(support_x, support_y, num_steps, task_name)

        # Predict on query with adapted params
        if task_name is not None:
            return self._forward_with_params(query_x, adapted_params)[task_name]
        return self._forward_with_params(query_x, adapted_params)


class MAMLTrainer:
    """
    Training procedure for MAML in drug discovery.
    """

    def __init__(self, maml_model, meta_lr=0.001, inner_lr=0.01,
                 inner_steps=5, first_order=False):
        """
        Args:
            maml_model: MAML wrapped model
            meta_lr: Outer loop learning rate
            inner_lr: Inner loop learning rate
            inner_steps: Number of inner loop steps
            first_order: Use first-order approximation
        """
        self.maml = maml_model
        self.meta_lr = meta_lr
        self.inner_lr = inner_lr
        self.inner_steps = inner_steps
        self.first_order = first_order

        # Meta-optimizer
        self.meta_optimizer = torch.optim.Adam(
            self.maml.model.parameters(), lr=meta_lr
        )

    def train_epoch(self, task_generator, n_tasks_per_batch=4):
        """
        Train one epoch on sampled tasks.

        Args:
            task_generator: Generator yielding (support, query) pairs
            n_tasks_per_batch: Number of tasks per meta-batch

        Returns:
            Average meta-loss
        """
        self.maml.train()
        total_loss = 0
        n_batches = 0

        task_batch = []

        for task_data in task_generator:
            task_batch.append(task_data)

            if len(task_batch) == n_tasks_per_batch:
                loss = self._meta_step(task_batch)
                total_loss += loss
                n_batches += 1
                task_batch = []

        # Handle remaining tasks
        if task_batch:
            loss = self._meta_step(task_batch)
            total_loss += loss
            n_batches += 1

        return total_loss / max(n_batches, 1)

    def _meta_step(self, task_batch):
        """
        Perform meta-update on a batch of tasks.
        """
        self.meta_optimizer.zero_grad()
        meta_loss = 0

        for task_data in task_batch:
            support_x, support_y = task_data['support']
            query_x, query_y = task_data['query']
            task_name = task_data.get('task_name', None)

            # Forward with adaptation
            query_pred = self.maml.meta_forward(
                support_x, support_y, query_x,
                num_steps=self.inner_steps,
                task_name=task_name
            )

            # Compute query loss
            if query_pred.dim() > 1 and query_pred.size(-1) > 1:
                task_loss = F.cross_entropy(query_pred, query_y)
            else:
                task_loss = F.mse_loss(query_pred.squeeze(), query_y.float())

            meta_loss += task_loss

        # Average and backprop
        meta_loss = meta_loss / len(task_batch)
        meta_loss.backward()
        self.meta_optimizer.step()

        return meta_loss.item()

    def evaluate(self, task_generator, n_tasks=100):
        """
        Evaluate meta-learning performance.

        Args:
            task_generator: Task generator
            n_tasks: Number of tasks to evaluate

        Returns:
            Dict with evaluation metrics
        """
        self.maml.eval()

        pre_adapt_losses = []
        post_adapt_losses = []
        pre_adapt_accs = []
        post_adapt_accs = []

        with torch.no_grad():
            for i, task_data in enumerate(task_generator):
                if i >= n_tasks:
                    break

                support_x, support_y = task_data['support']
                query_x, query_y = task_data['query']
                task_name = task_data.get('task_name', None)

                # Pre-adaptation prediction
                pre_pred = self.maml(query_x)
                if task_name:
                    pre_pred = pre_pred[task_name]

                # Post-adaptation prediction
                post_pred = self.maml.meta_forward(
                    support_x, support_y, query_x,
                    num_steps=self.inner_steps,
                    task_name=task_name
                )

                # Compute metrics
                is_classification = post_pred.dim() > 1 and post_pred.size(-1) > 1

                if is_classification:
                    pre_loss = F.cross_entropy(pre_pred, query_y)
                    post_loss = F.cross_entropy(post_pred, query_y)

                    pre_acc = (pre_pred.argmax(-1) == query_y).float().mean()
                    post_acc = (post_pred.argmax(-1) == query_y).float().mean()

                    pre_adapt_accs.append(pre_acc.item())
                    post_adapt_accs.append(post_acc.item())
                else:
                    pre_loss = F.mse_loss(pre_pred.squeeze(), query_y.float())
                    post_loss = F.mse_loss(post_pred.squeeze(), query_y.float())

                pre_adapt_losses.append(pre_loss.item())
                post_adapt_losses.append(post_loss.item())

        results = {
            'pre_adapt_loss': np.mean(pre_adapt_losses),
            'post_adapt_loss': np.mean(post_adapt_losses),
            'improvement': np.mean(pre_adapt_losses) - np.mean(post_adapt_losses)
        }

        if pre_adapt_accs:
            results['pre_adapt_acc'] = np.mean(pre_adapt_accs)
            results['post_adapt_acc'] = np.mean(post_adapt_accs)
            results['acc_improvement'] = np.mean(post_adapt_accs) - np.mean(pre_adapt_accs)

        return results

    def fine_tune(self, support_x, support_y, num_steps=10, task_name=None):
        """
        Fine-tune model on a specific task.

        Args:
            support_x: Support set inputs
            support_y: Support set labels
            num_steps: Number of fine-tuning steps
            task_name: Task name

        Returns:
            Fine-tuned model
        """
        # Create a copy of the model
        model_copy = copy.deepcopy(self.maml.model)
        optimizer = torch.optim.SGD(model_copy.parameters(), lr=self.inner_lr)

        model_copy.train()
        for _ in range(num_steps):
            optimizer.zero_grad()

            pred = model_copy(support_x)
            if task_name:
                pred = pred[task_name]

            if pred.dim() > 1 and pred.size(-1) > 1:
                loss = F.cross_entropy(pred, support_y)
            else:
                loss = F.mse_loss(pred.squeeze(), support_y.float())

            loss.backward()
            optimizer.step()

        return model_copy


class TaskAugmentedMAML(MAML):
    """
    MAML with task embeddings for better task conditioning.
    Useful when tasks have learnable structure.
    """

    def __init__(self, model, task_embedding_dim=32, num_tasks=None,
                 inner_lr=0.01, first_order=False):
        super(TaskAugmentedMAML, self).__init__(model, inner_lr, first_order)

        self.task_embedding_dim = task_embedding_dim

        if num_tasks is not None:
            self.task_embeddings = nn.Embedding(num_tasks, task_embedding_dim)
        else:
            self.task_embeddings = None

        # Task encoder for learning task embeddings from support set
        self.task_encoder = nn.Sequential(
            nn.Linear(model.output_dim, 128),
            nn.ReLU(),
            nn.Linear(128, task_embedding_dim)
        )

    def encode_task(self, support_x, support_y):
        """Encode task from support set."""
        with torch.no_grad():
            support_repr = self.model(support_x)
            if isinstance(support_repr, dict):
                support_repr = list(support_repr.values())[0]

        # Weighted by labels for asymmetric encoding
        weights = support_y.float().unsqueeze(-1)
        weighted_repr = (support_repr * weights).mean(0)

        task_emb = self.task_encoder(weighted_repr)
        return task_emb

    def adapt_with_task(self, support_x, support_y, num_steps=1, task_id=None):
        """Adapt with task embedding."""
        if task_id is not None and self.task_embeddings is not None:
            task_emb = self.task_embeddings(torch.tensor([task_id]))
        else:
            task_emb = self.encode_task(support_x, support_y)

        # Use task embedding to modulate adaptation
        # This is a simplified version - full implementation would use
        # task embedding to generate adaptation parameters

        return self.adapt(support_x, support_y, num_steps)
