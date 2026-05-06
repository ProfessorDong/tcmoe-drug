"""
Evaluation Utilities
Metrics computation and evaluation
"""

import torch
import numpy as np
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, recall_score,
    f1_score, mean_squared_error, mean_absolute_error, r2_score
)


def compute_metrics(y_true, y_pred, task_type='classification'):
    """
    Compute evaluation metrics.

    Args:
        y_true: True labels
        y_pred: Predictions
        task_type: 'classification' or 'regression'

    Returns:
        Dict of metrics
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.numpy()

    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()

    if task_type == 'classification':
        return _classification_metrics(y_true, y_pred)
    else:
        return _regression_metrics(y_true, y_pred)


def _classification_metrics(y_true, y_pred):
    """Compute classification metrics."""
    metrics = {}

    # Probabilities to binary predictions
    if y_pred.max() <= 1 and y_pred.min() >= 0:
        y_pred_binary = (y_pred > 0.5).astype(int)
    else:
        y_pred_binary = y_pred.astype(int)

    try:
        metrics['accuracy'] = accuracy_score(y_true, y_pred_binary)
    except:
        metrics['accuracy'] = None

    try:
        metrics['precision'] = precision_score(y_true, y_pred_binary, zero_division=0)
    except:
        metrics['precision'] = None

    try:
        metrics['recall'] = recall_score(y_true, y_pred_binary, zero_division=0)
    except:
        metrics['recall'] = None

    try:
        metrics['f1'] = f1_score(y_true, y_pred_binary, zero_division=0)
    except:
        metrics['f1'] = None

    try:
        if len(np.unique(y_true)) > 1:
            metrics['roc_auc'] = roc_auc_score(y_true, y_pred)
        else:
            metrics['roc_auc'] = None
    except:
        metrics['roc_auc'] = None

    return metrics


def _regression_metrics(y_true, y_pred):
    """Compute regression metrics."""
    metrics = {}

    try:
        metrics['mse'] = mean_squared_error(y_true, y_pred)
        metrics['rmse'] = np.sqrt(metrics['mse'])
    except:
        metrics['mse'] = None
        metrics['rmse'] = None

    try:
        metrics['mae'] = mean_absolute_error(y_true, y_pred)
    except:
        metrics['mae'] = None

    try:
        metrics['r2'] = r2_score(y_true, y_pred)
    except:
        metrics['r2'] = None

    return metrics


class Evaluator:
    """Evaluator for model performance."""

    def __init__(self, model, task_type='classification', device='cuda'):
        """
        Args:
            model: Model to evaluate
            task_type: 'classification' or 'regression'
            device: Device
        """
        self.model = model
        self.task_type = task_type
        self.device = device

    def evaluate(self, dataloader, return_predictions=False):
        """
        Evaluate model on dataloader.

        Args:
            dataloader: Data loader
            return_predictions: Return predictions

        Returns:
            Dict of metrics
        """
        self.model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in dataloader:
                # Handle batch format
                if isinstance(batch, dict):
                    inputs = {k: v.to(self.device) for k, v in batch.items()
                              if k not in ['labels', 'y', 'smiles'] and isinstance(v, torch.Tensor)}
                    labels = batch.get('labels', batch.get('y'))
                    outputs = self.model(**inputs) if inputs else self.model(batch)
                elif isinstance(batch, (tuple, list)):
                    inputs, labels = batch[0], batch[1]
                    if hasattr(inputs, 'to'):
                        inputs = inputs.to(self.device)
                    outputs = self.model(inputs)
                else:
                    outputs = self.model(batch.to(self.device))
                    labels = None

                if isinstance(outputs, dict):
                    outputs = list(outputs.values())[0]

                all_preds.append(outputs.cpu())
                if labels is not None:
                    all_labels.append(labels.cpu() if hasattr(labels, 'cpu') else torch.tensor(labels))

        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels) if all_labels else None

        metrics = compute_metrics(all_labels, all_preds, self.task_type)

        if return_predictions:
            return metrics, all_preds, all_labels

        return metrics

    def evaluate_per_task(self, dataloader, task_names):
        """
        Evaluate multi-task model per task.

        Args:
            dataloader: Data loader
            task_names: List of task names

        Returns:
            Dict of metrics per task
        """
        self.model.eval()
        task_preds = {task: [] for task in task_names}
        task_labels = {task: [] for task in task_names}

        with torch.no_grad():
            for batch in dataloader:
                outputs = self.model(batch)

                for task in task_names:
                    if task in outputs:
                        task_preds[task].append(outputs[task].cpu())
                        # Get corresponding labels
                        if 'labels' in batch:
                            task_labels[task].append(batch['labels'][task].cpu())

        results = {}
        for task in task_names:
            if task_preds[task] and task_labels[task]:
                preds = torch.cat(task_preds[task])
                labels = torch.cat(task_labels[task])
                results[task] = compute_metrics(labels, preds, self.task_type)

        return results


def generation_metrics(generated_smiles, reference_smiles=None):
    """
    Compute generation metrics.

    Args:
        generated_smiles: List of generated SMILES
        reference_smiles: Optional reference set

    Returns:
        Dict of metrics
    """
    from .chemistry import validate_smiles, get_unique_smiles, compute_properties

    metrics = {}

    # Validity
    valid = [s for s in generated_smiles if validate_smiles(s)]
    metrics['validity'] = len(valid) / len(generated_smiles) if generated_smiles else 0

    # Uniqueness
    unique = get_unique_smiles(generated_smiles)
    metrics['uniqueness'] = len(unique) / len(valid) if valid else 0

    # Novelty (if reference provided)
    if reference_smiles is not None:
        ref_set = set(get_unique_smiles(reference_smiles))
        novel = [s for s in unique if s not in ref_set]
        metrics['novelty'] = len(novel) / len(unique) if unique else 0

    # Property distributions
    if valid:
        mw = []
        logp = []
        for smiles in valid[:1000]:  # Limit for speed
            props = compute_properties(smiles, ['mw', 'logp'])
            if props:
                mw.append(props['mw'])
                logp.append(props['logp'])

        if mw:
            metrics['mean_mw'] = np.mean(mw)
            metrics['std_mw'] = np.std(mw)
        if logp:
            metrics['mean_logp'] = np.mean(logp)
            metrics['std_logp'] = np.std(logp)

    return metrics
