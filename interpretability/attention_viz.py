"""
Attention Visualization for Molecular Models
Provides interpretable explanations via attention weights
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D
import io
from PIL import Image


class AttentionVisualizer:
    """
    Visualizes attention weights from dual encoder models.
    Supports both SMILES sequence and graph attention.
    """

    def __init__(self, model, device='cuda'):
        """
        Args:
            model: DualEncoder or similar model with attention
            device: Device for computation
        """
        self.model = model
        self.device = device

    def get_smiles_attention(self, smiles_tensor, lengths=None):
        """
        Get attention weights for SMILES encoding.

        Args:
            smiles_tensor: Tokenized SMILES [1, seq_len]
            lengths: Sequence length

        Returns:
            attention_weights: [seq_len] attention weights
        """
        self.model.eval()
        with torch.no_grad():
            _, attention = self.model.smiles_encoder(
                smiles_tensor, lengths, return_attention=True
            )
        return attention.squeeze().cpu().numpy()

    def get_graph_attention(self, graph_data):
        """
        Get attention weights for graph encoding.

        Args:
            graph_data: Graph data object

        Returns:
            node_attention: Node-level attention weights
            edge_attention: Edge-level attention weights (if available)
        """
        self.model.eval()
        with torch.no_grad():
            _, attention_list = self.model.graph_encoder(
                graph_data, return_attention=True
            )

        # Extract node attention (from pooling layer)
        node_attention = attention_list[-1].cpu().numpy()

        # Extract edge attention (from GAT layers)
        edge_attention = [att.cpu().numpy() for att in attention_list[:-1]]

        return node_attention, edge_attention

    def visualize_smiles_attention(self, smiles, tokenizer, save_path=None):
        """
        Visualize SMILES attention as heatmap.

        Args:
            smiles: SMILES string
            tokenizer: Tokenizer with encode method
            save_path: Optional path to save figure

        Returns:
            matplotlib figure
        """
        # Tokenize
        tokens = list(smiles)
        tensor = tokenizer.encode(smiles).unsqueeze(0).to(self.device)
        length = torch.tensor([len(smiles)])

        # Get attention
        attention = self.get_smiles_attention(tensor, length)

        # Create figure
        fig, ax = plt.subplots(figsize=(len(tokens) * 0.5 + 2, 2))

        # Plot attention as colored bars
        colors = plt.cm.Reds(attention / attention.max())
        ax.bar(range(len(tokens)), attention, color=colors)

        # Add token labels
        ax.set_xticks(range(len(tokens)))
        ax.set_xticklabels(tokens, fontsize=12)
        ax.set_ylabel('Attention Weight')
        ax.set_title('SMILES Attention Visualization')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        return fig

    def visualize_molecule_attention(self, smiles, attention_weights, save_path=None):
        """
        Visualize attention on molecular structure.

        Args:
            smiles: SMILES string
            attention_weights: Atom-level attention weights
            save_path: Optional path to save figure

        Returns:
            PIL Image of molecule with attention highlighting
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")

        # Normalize attention
        attention = np.array(attention_weights)
        attention = (attention - attention.min()) / (attention.max() - attention.min() + 1e-8)

        # Create drawer
        drawer = rdMolDraw2D.MolDraw2DCairo(400, 400)

        # Set atom highlights based on attention
        atom_colors = {}
        atom_radii = {}

        for i, att in enumerate(attention):
            # Red color with intensity based on attention
            atom_colors[i] = (1.0, 1.0 - att, 1.0 - att)
            atom_radii[i] = 0.3 + att * 0.3

        drawer.DrawMolecule(
            mol,
            highlightAtoms=list(range(len(attention))),
            highlightAtomColors=atom_colors,
            highlightAtomRadii=atom_radii
        )
        drawer.FinishDrawing()

        # Convert to PIL Image
        png_data = drawer.GetDrawingText()
        img = Image.open(io.BytesIO(png_data))

        if save_path:
            img.save(save_path)

        return img


class GraphAttentionExplainer:
    """
    Explains predictions using graph attention weights.
    Identifies important atoms and bonds for predictions.
    """

    def __init__(self, model):
        """
        Args:
            model: Graph-based model with attention
        """
        self.model = model

    def explain(self, graph_data, task_name=None, top_k=5):
        """
        Generate explanation for prediction.

        Args:
            graph_data: Input graph data
            task_name: Task name for multi-task models
            top_k: Number of top atoms to highlight

        Returns:
            explanation: Dict with important atoms, bonds, and attention
        """
        self.model.eval()

        # Get prediction with attention
        output = self.model(graph_data, return_attention=True)

        if isinstance(output, dict):
            prediction = output.get('fused', output.get(task_name))
            attention_info = output.get('attention', {})
        else:
            prediction, attention_info = output

        # Extract node importance
        if 'graph' in attention_info:
            node_attention = attention_info['graph'][-1]  # Pooling attention
            if isinstance(node_attention, torch.Tensor):
                node_attention = node_attention.cpu().numpy()
        else:
            node_attention = None

        # Find top-k important atoms
        important_atoms = []
        if node_attention is not None:
            top_indices = np.argsort(node_attention)[-top_k:][::-1]
            important_atoms = [
                {'index': int(idx), 'attention': float(node_attention[idx])}
                for idx in top_indices
            ]

        # Extract edge attention if available
        edge_attention = None
        if 'graph' in attention_info and len(attention_info['graph']) > 1:
            edge_attention = attention_info['graph'][0]
            if isinstance(edge_attention, torch.Tensor):
                edge_attention = edge_attention.cpu().numpy()

        return {
            'prediction': prediction.item() if prediction.numel() == 1 else prediction.tolist(),
            'important_atoms': important_atoms,
            'node_attention': node_attention,
            'edge_attention': edge_attention
        }

    def visualize_explanation(self, smiles, explanation, save_path=None):
        """
        Visualize explanation on molecular structure.

        Args:
            smiles: SMILES string
            explanation: Output from explain()
            save_path: Path to save image

        Returns:
            PIL Image
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")

        node_attention = explanation.get('node_attention')
        if node_attention is None:
            # Fallback to simple drawing
            img = Draw.MolToImage(mol)
            if save_path:
                img.save(save_path)
            return img

        # Normalize attention
        att = np.array(node_attention[:mol.GetNumAtoms()])
        att = (att - att.min()) / (att.max() - att.min() + 1e-8)

        # Create drawer
        drawer = rdMolDraw2D.MolDraw2DCairo(400, 400)

        # Color atoms by attention
        atom_colors = {}
        for i, a in enumerate(att):
            # Blue (low) to Red (high) colormap
            atom_colors[i] = (a, 0.2, 1.0 - a)

        drawer.DrawMolecule(
            mol,
            highlightAtoms=list(range(len(att))),
            highlightAtomColors=atom_colors
        )
        drawer.FinishDrawing()

        png_data = drawer.GetDrawingText()
        img = Image.open(io.BytesIO(png_data))

        if save_path:
            img.save(save_path)

        return img

    def compare_explanations(self, smiles_list, explanations, save_path=None):
        """
        Compare explanations across multiple molecules.

        Args:
            smiles_list: List of SMILES
            explanations: List of explanation dicts
            save_path: Path to save figure

        Returns:
            matplotlib figure
        """
        n_mols = len(smiles_list)
        fig, axes = plt.subplots(1, n_mols, figsize=(4 * n_mols, 4))

        if n_mols == 1:
            axes = [axes]

        for ax, smiles, exp in zip(axes, smiles_list, explanations):
            img = self.visualize_explanation(smiles, exp)
            ax.imshow(img)
            ax.axis('off')
            pred = exp.get('prediction', 'N/A')
            ax.set_title(f'Pred: {pred:.3f}' if isinstance(pred, float) else f'Pred: {pred}')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        return fig
