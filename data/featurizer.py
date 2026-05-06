"""
Molecular Featurization
SMILES tokenization and Graph feature extraction
"""

import torch
import numpy as np
from rdkit import Chem


class SMILESTokenizer:
    """
    SMILES string tokenizer.
    Converts SMILES to integer sequences.
    """

    def __init__(self, tokens=None, add_special_tokens=True):
        """
        Args:
            tokens: List of valid tokens (None = auto-detect)
            add_special_tokens: Add <PAD>, <START>, <END>, <UNK>
        """
        if tokens is None:
            self.tokens = [
                '<', '>', '#', '%', ')', '(', '+', '-', '/', '.', '1', '0', '3', '2', '5', '4', '7',
                '6', '9', '8', '=', 'A', '@', 'C', 'B', 'F', 'I', 'H', 'O', 'N', 'P', 'S', '[', ']',
                '\\', 'c', 'e', 'i', 'l', 'o', 'n', 'p', 's', 'r', '\n'
            ]
        else:
            self.tokens = list(tokens)

        if add_special_tokens:
            self.special_tokens = ['<PAD>', '<START>', '<END>', '<UNK>']
            self.all_tokens = self.special_tokens + self.tokens
        else:
            self.special_tokens = []
            self.all_tokens = self.tokens

        self.token_to_idx = {t: i for i, t in enumerate(self.all_tokens)}
        self.idx_to_token = {i: t for t, i in self.token_to_idx.items()}

        self.pad_idx = self.token_to_idx.get('<PAD>', 0)
        self.start_idx = self.token_to_idx.get('<START>', 1) if '<START>' in self.token_to_idx else None
        self.end_idx = self.token_to_idx.get('<END>', 2) if '<END>' in self.token_to_idx else None
        self.unk_idx = self.token_to_idx.get('<UNK>', 3) if '<UNK>' in self.token_to_idx else 0

    @property
    def vocab_size(self):
        return len(self.all_tokens)

    def encode(self, smiles, add_start_end=False):
        """
        Encode SMILES to token indices.

        Args:
            smiles: SMILES string
            add_start_end: Add start/end tokens

        Returns:
            List of token indices
        """
        indices = []

        if add_start_end and self.start_idx is not None:
            indices.append(self.start_idx)

        for char in smiles:
            idx = self.token_to_idx.get(char, self.unk_idx)
            indices.append(idx)

        if add_start_end and self.end_idx is not None:
            indices.append(self.end_idx)

        return indices

    def decode(self, indices, remove_special=True):
        """
        Decode token indices to SMILES.

        Args:
            indices: List of token indices
            remove_special: Remove special tokens

        Returns:
            SMILES string
        """
        tokens = []
        for idx in indices:
            token = self.idx_to_token.get(idx, '')
            if remove_special and token in self.special_tokens:
                continue
            tokens.append(token)

        return ''.join(tokens)

    def batch_encode(self, smiles_list, max_length=100, padding=True):
        """
        Encode batch of SMILES.

        Args:
            smiles_list: List of SMILES strings
            max_length: Maximum sequence length
            padding: Pad to max_length

        Returns:
            Tensor of shape [batch, max_length]
        """
        encoded = []
        lengths = []

        for smiles in smiles_list:
            tokens = self.encode(smiles)[:max_length]
            lengths.append(len(tokens))

            if padding:
                tokens = tokens + [self.pad_idx] * (max_length - len(tokens))

            encoded.append(tokens)

        return torch.tensor(encoded), torch.tensor(lengths)


class GraphFeaturizer:
    """
    Graph featurizer for molecular graphs.
    Extracts node and edge features for GNNs.
    """

    def __init__(self, node_feature_dim=75, edge_feature_dim=6):
        """
        Args:
            node_feature_dim: Dimension of node features
            edge_feature_dim: Dimension of edge features
        """
        self.node_feature_dim = node_feature_dim
        self.edge_feature_dim = edge_feature_dim

        # Atom types
        self.atom_types = [
            'C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca',
            'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag',
            'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni',
            'Cd', 'In', 'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 'Pb', 'Unknown'
        ]

    def featurize(self, mol):
        """
        Featurize molecule as graph.

        Args:
            mol: RDKit molecule

        Returns:
            Dict with x, edge_index, edge_attr
        """
        if mol is None:
            return None

        # Node features
        x = []
        for atom in mol.GetAtoms():
            x.append(self._atom_features(atom))

        x = torch.tensor(x, dtype=torch.float)

        # Edge features
        edge_index = []
        edge_attr = []

        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            edge_index.extend([[i, j], [j, i]])
            bf = self._bond_features(bond)
            edge_attr.extend([bf, bf])

        if edge_index:
            edge_index = torch.tensor(edge_index, dtype=torch.long).t()
            edge_attr = torch.tensor(edge_attr, dtype=torch.float)
        else:
            edge_index = torch.zeros(2, 0, dtype=torch.long)
            edge_attr = torch.zeros(0, self.edge_feature_dim)

        return {
            'x': x,
            'edge_index': edge_index,
            'edge_attr': edge_attr
        }

    def _atom_features(self, atom):
        """Extract atom features."""
        features = []

        # Atom type one-hot
        atom_type = atom.GetSymbol()
        atom_idx = self.atom_types.index(atom_type) if atom_type in self.atom_types else -1
        features.extend([1 if i == atom_idx else 0 for i in range(len(self.atom_types))])

        # Degree one-hot
        features.extend([1 if atom.GetDegree() == i else 0 for i in range(11)])

        # Num hydrogens one-hot
        features.extend([1 if atom.GetTotalNumHs() == i else 0 for i in range(5)])

        # Implicit valence one-hot
        features.extend([1 if atom.GetImplicitValence() == i else 0 for i in range(7)])

        # Other features
        features.append(atom.GetFormalCharge())
        features.append(atom.GetNumRadicalElectrons())

        # Hybridization one-hot
        hyb = atom.GetHybridization()
        hyb_types = [
            Chem.rdchem.HybridizationType.SP,
            Chem.rdchem.HybridizationType.SP2,
            Chem.rdchem.HybridizationType.SP3,
            Chem.rdchem.HybridizationType.SP3D,
            Chem.rdchem.HybridizationType.SP3D2
        ]
        features.extend([1 if hyb == h else 0 for h in hyb_types])

        # Aromatic
        features.append(1 if atom.GetIsAromatic() else 0)

        return features

    def _bond_features(self, bond):
        """Extract bond features."""
        bt = bond.GetBondType()
        return [
            int(bt == Chem.rdchem.BondType.SINGLE),
            int(bt == Chem.rdchem.BondType.DOUBLE),
            int(bt == Chem.rdchem.BondType.TRIPLE),
            int(bt == Chem.rdchem.BondType.AROMATIC),
            int(bond.GetIsConjugated()),
            int(bond.IsInRing())
        ]


class MoleculeFeaturizer:
    """
    Combined featurizer for molecules.
    Returns both graph and SMILES representations.
    """

    def __init__(self, tokenizer=None, graph_featurizer=None):
        """
        Args:
            tokenizer: SMILES tokenizer
            graph_featurizer: Graph featurizer
        """
        self.tokenizer = tokenizer or SMILESTokenizer()
        self.graph_featurizer = graph_featurizer or GraphFeaturizer()

    def featurize(self, smiles):
        """
        Featurize SMILES string.

        Args:
            smiles: SMILES string

        Returns:
            Dict with graph and SMILES features
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        return {
            'graph': self.graph_featurizer.featurize(mol),
            'smiles_tokens': self.tokenizer.encode(smiles),
            'smiles': smiles
        }

    def batch_featurize(self, smiles_list):
        """
        Featurize batch of SMILES.

        Args:
            smiles_list: List of SMILES strings

        Returns:
            Batch of features
        """
        features = []
        for smiles in smiles_list:
            feat = self.featurize(smiles)
            if feat is not None:
                features.append(feat)
        return features
