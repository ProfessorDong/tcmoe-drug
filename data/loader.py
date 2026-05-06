"""
Data Loading Utilities
Generator data format compatible with original ReLeaSE
"""

import torch
import numpy as np
import random
from torch.utils.data import DataLoader


class GeneratorData:
    """
    Data container for SMILES generation.
    Compatible with ReLeaSE format.
    """

    def __init__(self, training_data_path=None, smiles_list=None, tokens=None,
                 start_token='<', end_token='>', max_len=120, delimiter='\t',
                 cols_to_read=None, keep_header=False, use_cuda=None):
        """
        Args:
            training_data_path: Path to training data file
            smiles_list: Alternative: provide SMILES directly
            tokens: List of valid tokens
            start_token: Start of sequence token
            end_token: End of sequence token
            max_len: Maximum SMILES length
            delimiter: File delimiter
            cols_to_read: Columns to read from file
            keep_header: Skip header line
            use_cuda: Use GPU
        """
        self.start_token = start_token
        self.end_token = end_token
        self.max_len = max_len

        # Load data
        if smiles_list is not None:
            data = smiles_list
        elif training_data_path is not None:
            data = self._read_file(training_data_path, delimiter, cols_to_read, keep_header)
        else:
            data = []

        # Filter and format
        self.file = []
        for smiles in data:
            if len(smiles) <= max_len:
                self.file.append(start_token + smiles + end_token)

        self.file_len = len(self.file)

        # Setup vocabulary
        if tokens is None:
            tokens = self._extract_tokens()

        self.all_characters = tokens
        self.char2idx = {c: i for i, c in enumerate(tokens)}
        self.n_characters = len(tokens)

        # CUDA
        self.use_cuda = use_cuda
        if self.use_cuda is None:
            self.use_cuda = torch.cuda.is_available()

    def _read_file(self, path, delimiter, cols_to_read, keep_header):
        """Read SMILES from file."""
        data = []
        with open(path, 'r') as f:
            if keep_header:
                next(f)  # Skip header

            for line in f:
                parts = line.strip().split(delimiter)
                if cols_to_read:
                    smiles = parts[cols_to_read[0]] if cols_to_read[0] < len(parts) else ''
                else:
                    smiles = parts[0]
                if smiles:
                    data.append(smiles)

        return data

    def _extract_tokens(self):
        """Extract tokens from data."""
        chars = set()
        for smiles in self.file:
            chars.update(smiles)
        return sorted(list(chars))

    def random_chunk(self):
        """Get random SMILES from dataset."""
        idx = random.randint(0, self.file_len - 1)
        return self.file[idx]

    def char_tensor(self, string):
        """Convert string to tensor of indices."""
        tensor = torch.zeros(len(string)).long()
        for i, c in enumerate(string):
            tensor[i] = self.char2idx.get(c, 0)

        if self.use_cuda:
            return tensor.cuda()
        return tensor

    def random_training_set(self, smiles_augmentation=None):
        """Get random training pair (input, target)."""
        chunk = self.random_chunk()

        if smiles_augmentation is not None:
            chunk = self.start_token + smiles_augmentation.randomize_smiles(chunk[1:-1]) + self.end_token

        inp = self.char_tensor(chunk[:-1])
        target = self.char_tensor(chunk[1:])

        return inp, target

    def load_dictionary(self, tokens, char2idx):
        """Load external vocabulary."""
        self.all_characters = tokens
        self.char2idx = char2idx
        self.n_characters = len(tokens)


def create_data_loaders(train_dataset, val_dataset=None, test_dataset=None,
                        batch_size=32, num_workers=4, collate_fn=None):
    """
    Create data loaders.

    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        test_dataset: Test dataset
        batch_size: Batch size
        num_workers: Number of workers
        collate_fn: Custom collate function

    Returns:
        Dict of data loaders
    """
    loaders = {}

    loaders['train'] = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn
    )

    if val_dataset is not None:
        loaders['val'] = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_fn
        )

    if test_dataset is not None:
        loaders['test'] = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_fn
        )

    return loaders


def load_smiles_from_file(filepath, delimiter='\t', smiles_col=0, header=True):
    """
    Load SMILES from file.

    Args:
        filepath: Path to file
        delimiter: Column delimiter
        smiles_col: SMILES column index
        header: File has header

    Returns:
        List of SMILES strings
    """
    smiles_list = []

    with open(filepath, 'r') as f:
        if header:
            next(f)

        for line in f:
            parts = line.strip().split(delimiter)
            if len(parts) > smiles_col:
                smiles_list.append(parts[smiles_col])

    return smiles_list


def load_property_file(filepath, delimiter='\t', smiles_col=0, property_cols=None,
                       header=True):
    """
    Load SMILES with properties.

    Args:
        filepath: Path to file
        delimiter: Column delimiter
        smiles_col: SMILES column index
        property_cols: Property column indices
        header: File has header

    Returns:
        smiles_list, properties array
    """
    smiles_list = []
    properties = []

    with open(filepath, 'r') as f:
        if header:
            next(f)

        for line in f:
            parts = line.strip().split(delimiter)
            if len(parts) > smiles_col:
                smiles_list.append(parts[smiles_col])

                if property_cols:
                    props = [float(parts[col]) if col < len(parts) else np.nan
                             for col in property_cols]
                    properties.append(props)

    if properties:
        properties = np.array(properties)

    return smiles_list, properties if properties else None
