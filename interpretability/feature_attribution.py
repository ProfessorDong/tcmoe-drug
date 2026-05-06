"""
Feature Attribution Methods for Molecular Property Prediction
Implements Integrated Gradients and SHAP-like explanations
"""

import torch
import torch.nn as nn
import numpy as np


class IntegratedGradients:
    """
    Integrated Gradients for feature attribution.
    Computes attribution by integrating gradients along path from baseline.
    """

    def __init__(self, model, baseline_type='zeros'):
        """
        Args:
            model: Neural network model
            baseline_type: Type of baseline ('zeros', 'random', 'mean')
        """
        self.model = model
        self.baseline_type = baseline_type

    def attribute(self, x, target=None, task_name=None, n_steps=50):
        """
        Compute integrated gradients attribution.

        Args:
            x: Input tensor [batch, features] or graph data
            target: Target class index (for classification)
            task_name: Task name for multi-task models
            n_steps: Number of integration steps

        Returns:
            attributions: Feature attributions [batch, features]
        """
        self.model.eval()

        # Create baseline
        baseline = self._create_baseline(x)

        # Generate interpolated inputs
        alphas = torch.linspace(0, 1, n_steps).view(-1, 1)
        if x.dim() > 1:
            alphas = alphas.unsqueeze(-1).expand(-1, x.size(0), x.size(1))

        # Compute gradients at each step
        gradients = []
        for alpha in torch.linspace(0, 1, n_steps):
            x_interp = baseline + alpha * (x - baseline)
            x_interp = x_interp.clone().requires_grad_(True)

            # Forward pass
            output = self.model(x_interp)
            if isinstance(output, dict):
                output = output.get(task_name, list(output.values())[0])

            if target is not None:
                output = output[:, target]
            else:
                output = output.sum()

            # Backward pass
            grad = torch.autograd.grad(output, x_interp)[0]
            gradients.append(grad)

        # Average gradients
        avg_gradients = torch.stack(gradients).mean(dim=0)

        # Compute attributions
        attributions = (x - baseline) * avg_gradients

        return attributions.detach()

    def _create_baseline(self, x):
        """Create baseline tensor."""
        if self.baseline_type == 'zeros':
            return torch.zeros_like(x)
        elif self.baseline_type == 'random':
            return torch.randn_like(x) * 0.1
        elif self.baseline_type == 'mean':
            return x.mean(dim=0, keepdim=True).expand_as(x)
        else:
            return torch.zeros_like(x)

    def visualize_attributions(self, smiles, attributions, feature_names=None):
        """
        Visualize feature attributions.

        Args:
            smiles: SMILES string
            attributions: Attribution values
            feature_names: Names of features

        Returns:
            Dict with visualization data
        """
        attributions = attributions.cpu().numpy().squeeze()

        # Sort by absolute value
        sorted_idx = np.argsort(np.abs(attributions))[::-1]

        result = {
            'smiles': smiles,
            'top_features': [],
            'attributions': attributions
        }

        for i, idx in enumerate(sorted_idx[:10]):
            name = feature_names[idx] if feature_names else f"Feature_{idx}"
            result['top_features'].append({
                'name': name,
                'attribution': float(attributions[idx]),
                'rank': i + 1
            })

        return result


class SHAP_Explainer:
    """
    SHAP-like explainer using kernel approximation.
    Provides model-agnostic feature attributions.
    """

    def __init__(self, model, background_data, n_samples=100):
        """
        Args:
            model: Prediction model
            background_data: Background dataset for reference
            n_samples: Number of samples for approximation
        """
        self.model = model
        self.background = background_data
        self.n_samples = n_samples

    def explain(self, x, task_name=None):
        """
        Compute SHAP values.

        Args:
            x: Input tensor [1, features]
            task_name: Task name for multi-task models

        Returns:
            shap_values: Feature attributions
        """
        n_features = x.size(1)

        # Baseline prediction (expected value)
        baseline_pred = self._predict(self.background, task_name).mean()

        # Instance prediction
        instance_pred = self._predict(x, task_name)

        # Approximate SHAP values using sampling
        shap_values = np.zeros(n_features)

        for _ in range(self.n_samples):
            # Random permutation
            perm = np.random.permutation(n_features)

            # Iterate through features
            x_with = self._sample_background()
            x_without = x_with.clone()

            for i, feat_idx in enumerate(perm):
                # Include feature
                x_with[:, feat_idx] = x[:, feat_idx]

                # Marginal contribution
                pred_with = self._predict(x_with, task_name)
                pred_without = self._predict(x_without, task_name)

                shap_values[feat_idx] += (pred_with - pred_without).item()

                x_without = x_with.clone()

        shap_values /= self.n_samples

        return shap_values

    def _predict(self, x, task_name):
        """Make prediction."""
        self.model.eval()
        with torch.no_grad():
            output = self.model(x)
            if isinstance(output, dict):
                output = output.get(task_name, list(output.values())[0])
        return output

    def _sample_background(self):
        """Sample from background data."""
        idx = np.random.randint(len(self.background))
        if isinstance(self.background, torch.Tensor):
            return self.background[idx:idx+1].clone()
        return torch.tensor(self.background[idx:idx+1]).float()


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping for graph neural networks.
    Highlights important substructures for predictions.
    """

    def __init__(self, model, target_layer_name):
        """
        Args:
            model: GNN model
            target_layer_name: Name of layer to compute CAM from
        """
        self.model = model
        self.target_layer_name = target_layer_name

        self.activations = None
        self.gradients = None

        # Register hooks
        self._register_hooks()

    def _register_hooks(self):
        """Register forward and backward hooks."""
        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        # Find target layer
        for name, module in self.model.named_modules():
            if name == self.target_layer_name:
                module.register_forward_hook(forward_hook)
                module.register_full_backward_hook(backward_hook)
                break

    def compute_cam(self, x, target=None, task_name=None):
        """
        Compute GradCAM.

        Args:
            x: Input data
            target: Target class
            task_name: Task name

        Returns:
            cam: Class activation map
        """
        self.model.eval()

        # Forward pass
        output = self.model(x)
        if isinstance(output, dict):
            output = output.get(task_name, list(output.values())[0])

        if target is not None:
            score = output[:, target]
        else:
            score = output.sum()

        # Backward pass
        self.model.zero_grad()
        score.backward(retain_graph=True)

        # Compute CAM
        weights = self.gradients.mean(dim=-1, keepdim=True)
        cam = (weights * self.activations).sum(dim=-1)
        cam = torch.relu(cam)

        # Normalize
        cam = cam / (cam.max() + 1e-8)

        return cam.detach()

    def visualize_cam(self, smiles, cam, save_path=None):
        """
        Visualize GradCAM on molecule.

        Args:
            smiles: SMILES string
            cam: CAM values [n_atoms]
            save_path: Path to save image

        Returns:
            PIL Image
        """
        from rdkit import Chem
        from rdkit.Chem.Draw import rdMolDraw2D
        import io
        from PIL import Image

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")

        cam = cam.cpu().numpy()

        # Create drawer
        drawer = rdMolDraw2D.MolDraw2DCairo(400, 400)

        # Color atoms by CAM
        atom_colors = {}
        for i, c in enumerate(cam[:mol.GetNumAtoms()]):
            # Red intensity based on CAM
            atom_colors[i] = (1.0, 1.0 - c, 1.0 - c)

        drawer.DrawMolecule(
            mol,
            highlightAtoms=list(range(min(len(cam), mol.GetNumAtoms()))),
            highlightAtomColors=atom_colors
        )
        drawer.FinishDrawing()

        png_data = drawer.GetDrawingText()
        img = Image.open(io.BytesIO(png_data))

        if save_path:
            img.save(save_path)

        return img
