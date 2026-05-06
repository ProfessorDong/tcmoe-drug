"""
Dataset Classes for Drug Discovery
Supports SMILES, Graph, and Dual representations
"""

import torch
from torch.utils.data import Dataset
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem


class GeneratorTokenizer:
    """Tokenizer compatible with DualDataset that uses the same char→index
    mapping as GeneratorData/MoleculeGenerator.

    Usage:
        from data.loader import GeneratorData
        gen_data = GeneratorData(smiles_list=smiles, tokens=None, max_len=120)
        tokenizer = GeneratorTokenizer(gen_data.all_characters)
        dataset = DualDataset(smiles, tokenizer=tokenizer)
    """

    def __init__(self, all_characters):
        """
        Args:
            all_characters: List of characters from GeneratorData.all_characters
        """
        self.all_characters = list(all_characters)
        self.char2idx = {c: i for i, c in enumerate(self.all_characters)}
        self.vocab_size = len(self.all_characters)

    def encode(self, smiles):
        """Convert SMILES string to list of token indices."""
        return [self.char2idx.get(c, 0) for c in smiles]

    def decode(self, indices):
        """Convert list of token indices back to SMILES string."""
        return ''.join(self.all_characters[i] for i in indices
                       if 0 <= i < len(self.all_characters))


class MoleculeDataset(Dataset):
    """Base dataset class for molecules."""

    def __init__(self, smiles_list, labels=None, transform=None):
        """
        Args:
            smiles_list: List of SMILES strings
            labels: Optional labels
            transform: Optional transform function
        """
        self.smiles_list = smiles_list
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.smiles_list)

    def __getitem__(self, idx):
        smiles = self.smiles_list[idx]

        if self.transform:
            data = self.transform(smiles)
        else:
            data = smiles

        if self.labels is not None:
            label = self.labels[idx]
            return data, label

        return data


class SMILESDataset(Dataset):
    """Dataset for SMILES-based models."""

    def __init__(self, smiles_list, labels=None, tokenizer=None, max_length=100):
        """
        Args:
            smiles_list: List of SMILES strings
            labels: Optional labels
            tokenizer: SMILES tokenizer
            max_length: Maximum sequence length
        """
        self.smiles_list = smiles_list
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.smiles_list)

    def __getitem__(self, idx):
        smiles = self.smiles_list[idx]

        # Tokenize
        if self.tokenizer:
            tokens = self.tokenizer.encode(smiles)
        else:
            tokens = list(smiles)

        # Pad/truncate
        length = len(tokens)
        if length > self.max_length:
            tokens = tokens[:self.max_length]
            length = self.max_length

        # Convert to tensor
        tensor = torch.zeros(self.max_length, dtype=torch.long)
        tensor[:len(tokens)] = torch.tensor(tokens)

        data = {
            'tokens': tensor,
            'length': length,
            'smiles': smiles
        }

        if self.labels is not None:
            data['label'] = torch.tensor(self.labels[idx])

        return data


class GraphDataset(Dataset):
    """Dataset for graph-based models."""

    def __init__(self, smiles_list, labels=None, featurizer=None):
        """
        Args:
            smiles_list: List of SMILES strings
            labels: Optional labels
            featurizer: Graph featurizer
        """
        self.smiles_list = smiles_list
        self.labels = labels
        self.featurizer = featurizer

        # Pre-compute graphs
        self.graphs = []
        self.valid_indices = []

        for idx, smiles in enumerate(smiles_list):
            graph = self._smiles_to_graph(smiles)
            if graph is not None:
                self.graphs.append(graph)
                self.valid_indices.append(idx)

    def _smiles_to_graph(self, smiles):
        """Convert SMILES to graph representation."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        if self.featurizer:
            return self.featurizer.featurize(mol)

        # Default featurization
        n_atoms = mol.GetNumAtoms()

        # Node features
        node_features = []
        for atom in mol.GetAtoms():
            features = self._get_atom_features(atom)
            node_features.append(features)

        x = torch.tensor(node_features, dtype=torch.float)

        # Edge indices
        edge_index = []
        edge_attr = []

        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_index.extend([[i, j], [j, i]])

            bond_features = self._get_bond_features(bond)
            edge_attr.extend([bond_features, bond_features])

        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)

        return {'x': x, 'edge_index': edge_index, 'edge_attr': edge_attr}

    def _get_atom_features(self, atom):
        """Get atom features."""
        return [
            atom.GetAtomicNum(),
            atom.GetDegree(),
            atom.GetFormalCharge(),
            atom.GetNumRadicalElectrons(),
            int(atom.GetIsAromatic()),
            atom.GetTotalNumHs(),
            int(atom.GetHybridization()),
        ]

    def _get_bond_features(self, bond):
        """Get bond features."""
        return [
            bond.GetBondTypeAsDouble(),
            bond.GetIsConjugated(),
            bond.IsInRing(),
        ]

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        graph = self.graphs[idx]

        if self.labels is not None:
            orig_idx = self.valid_indices[idx]
            label = torch.tensor(self.labels[orig_idx])
            return graph, label

        return graph


class DualDataset(Dataset):
    """
    Dataset for dual representation (Graph + SMILES).
    Returns both graph and SMILES features.
    """

    def __init__(self, smiles_list, labels=None, tokenizer=None,
                 graph_featurizer=None, max_length=100):
        """
        Args:
            smiles_list: List of SMILES strings
            labels: Optional labels
            tokenizer: SMILES tokenizer
            graph_featurizer: Graph featurizer
            max_length: Maximum SMILES length
        """
        self.smiles_list = smiles_list
        self.labels = labels
        self.tokenizer = tokenizer
        self.graph_featurizer = graph_featurizer
        self.max_length = max_length

        # Pre-process and validate
        self.valid_data = []

        for idx, smiles in enumerate(smiles_list):
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue

            # Graph features
            graph = self._mol_to_graph(mol)

            # SMILES tokens
            tokens = self._tokenize_smiles(smiles)

            self.valid_data.append({
                'smiles': smiles,
                'graph': graph,
                'tokens': tokens,
                'length': min(len(smiles), max_length),
                'label': labels[idx] if labels is not None else None
            })

    def _mol_to_graph(self, mol):
        """Convert molecule to graph."""
        if self.graph_featurizer:
            return self.graph_featurizer.featurize(mol)

        n_atoms = mol.GetNumAtoms()

        # Node features (75-dim like DeepChem)
        node_features = []
        for atom in mol.GetAtoms():
            features = self._get_atom_features_75(atom)
            node_features.append(features)

        x = torch.tensor(node_features, dtype=torch.float)

        # Edges
        edge_index = []
        edge_attr = []

        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            edge_index.extend([[i, j], [j, i]])
            bf = self._get_bond_features(bond)
            edge_attr.extend([bf, bf])

        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float) if edge_attr else torch.zeros(0, 6)

        return {'x': x, 'edge_index': edge_index, 'edge_attr': edge_attr}

    def _get_atom_features_75(self, atom):
        """Get 75-dimensional atom features."""
        # Atom type one-hot (44 dims)
        atom_types = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg',
                      'Na', 'Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl',
                      'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H',
                      'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr',
                      'Pt', 'Hg', 'Pb', 'Unknown']

        atom_type = atom.GetSymbol()
        atom_type_idx = atom_types.index(atom_type) if atom_type in atom_types else -1

        one_hot = [1 if i == atom_type_idx else 0 for i in range(len(atom_types))]

        # Degree (11 dims)
        degree = [1 if atom.GetDegree() == i else 0 for i in range(11)]

        # Hydrogens (5 dims)
        n_hs = [1 if atom.GetTotalNumHs() == i else 0 for i in range(5)]

        # Valence (7 dims)
        valence = [1 if atom.GetImplicitValence() == i else 0 for i in range(7)]

        # Other features
        other = [
            atom.GetFormalCharge(),
            atom.GetNumRadicalElectrons(),
            1 if atom.GetIsAromatic() else 0
        ]

        # Hybridization (5 dims)
        hyb = atom.GetHybridization()
        hyb_types = [Chem.rdchem.HybridizationType.SP,
                     Chem.rdchem.HybridizationType.SP2,
                     Chem.rdchem.HybridizationType.SP3,
                     Chem.rdchem.HybridizationType.SP3D,
                     Chem.rdchem.HybridizationType.SP3D2]
        hyb_feat = [1 if hyb == h else 0 for h in hyb_types]

        return one_hot + degree + n_hs + valence + other + hyb_feat

    def _get_bond_features(self, bond):
        """Get bond features."""
        bt = bond.GetBondType()
        return [
            bt == Chem.rdchem.BondType.SINGLE,
            bt == Chem.rdchem.BondType.DOUBLE,
            bt == Chem.rdchem.BondType.TRIPLE,
            bt == Chem.rdchem.BondType.AROMATIC,
            bond.GetIsConjugated(),
            bond.IsInRing()
        ]

    def _tokenize_smiles(self, smiles):
        """Tokenize SMILES string."""
        if self.tokenizer:
            tokens = self.tokenizer.encode(smiles)
        else:
            tokens = [ord(c) for c in smiles]

        # Pad/truncate
        tensor = torch.zeros(self.max_length, dtype=torch.long)
        length = min(len(tokens), self.max_length)
        tensor[:length] = torch.tensor(tokens[:length])

        return tensor

    def __len__(self):
        return len(self.valid_data)

    def __getitem__(self, idx):
        data = self.valid_data[idx]

        result = {
            'graph': data['graph'],
            'tokens': data['tokens'],
            'length': data['length'],
            'smiles': data['smiles']
        }

        if data['label'] is not None:
            result['label'] = torch.tensor(data['label'])

        return result


def collate_graphs(batch):
    """Collate function for graph batches."""
    # Separate graphs and labels
    if isinstance(batch[0], tuple):
        graphs, labels = zip(*batch)
        labels = torch.stack(labels)
    else:
        graphs = batch
        labels = None

    # Batch graphs
    x_list = []
    edge_index_list = []
    edge_attr_list = []
    batch_list = []

    cumsum = 0
    for i, g in enumerate(graphs):
        x_list.append(g['x'])
        edge_index_list.append(g['edge_index'] + cumsum)
        if 'edge_attr' in g:
            edge_attr_list.append(g['edge_attr'])
        batch_list.append(torch.full((g['x'].size(0),), i, dtype=torch.long))
        cumsum += g['x'].size(0)

    batched = {
        'x': torch.cat(x_list, dim=0),
        'edge_index': torch.cat(edge_index_list, dim=1),
        'batch': torch.cat(batch_list)
    }

    if edge_attr_list:
        batched['edge_attr'] = torch.cat(edge_attr_list, dim=0)

    if labels is not None:
        batched['y'] = labels

    return batched


def collate_dual(batch):
    """Collate function for dual representation batches."""
    graphs = [b['graph'] for b in batch]
    tokens = torch.stack([b['tokens'] for b in batch])
    lengths = torch.tensor([b['length'] for b in batch])

    # Batch graphs
    graph_batch = collate_graphs(graphs)

    result = {
        'graph': graph_batch,
        'tokens': tokens,
        'lengths': lengths
    }

    if 'label' in batch[0]:
        result['labels'] = torch.stack([b['label'] for b in batch])

    return result
