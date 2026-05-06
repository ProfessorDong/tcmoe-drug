"""
Experience Replay for Molecular Generation
Stores and samples past experiences for training stability
"""

import torch
import numpy as np
import random
from collections import deque


class ExperienceReplay:
    """
    Standard experience replay buffer.
    Stores (smiles, reward) pairs for replay during training.
    """

    def __init__(self, capacity=10000, min_samples=100):
        """
        Args:
            capacity: Maximum buffer capacity
            min_samples: Minimum samples before replay starts
        """
        self.capacity = capacity
        self.min_samples = min_samples
        self.buffer = deque(maxlen=capacity)

    def add(self, smiles, reward, log_probs=None, features=None):
        """
        Add experience to buffer.

        Args:
            smiles: SMILES string
            reward: Reward value
            log_probs: Optional log probabilities
            features: Optional molecular features
        """
        experience = {
            'smiles': smiles,
            'reward': reward,
            'log_probs': log_probs,
            'features': features
        }
        self.buffer.append(experience)

    def sample(self, batch_size):
        """
        Sample batch of experiences.

        Args:
            batch_size: Number of samples

        Returns:
            List of experience dicts
        """
        if len(self.buffer) < self.min_samples:
            return []

        batch_size = min(batch_size, len(self.buffer))
        return random.sample(list(self.buffer), batch_size)

    def get_top_k(self, k):
        """Get top k experiences by reward."""
        sorted_buffer = sorted(self.buffer, key=lambda x: x['reward'], reverse=True)
        return sorted_buffer[:k]

    def clear(self):
        """Clear the buffer."""
        self.buffer.clear()

    def __len__(self):
        return len(self.buffer)


class PrioritizedReplay:
    """
    Prioritized Experience Replay.
    Samples experiences with probability proportional to their priority.
    """

    def __init__(self, capacity=10000, alpha=0.6, beta=0.4, beta_increment=0.001):
        """
        Args:
            capacity: Maximum buffer capacity
            alpha: Priority exponent (0 = uniform, 1 = full priority)
            beta: Importance sampling exponent
            beta_increment: Beta annealing rate
        """
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment

        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.max_priority = 1.0

    def add(self, smiles, reward, log_probs=None, features=None):
        """Add experience with max priority."""
        experience = {
            'smiles': smiles,
            'reward': reward,
            'log_probs': log_probs,
            'features': features
        }

        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience

        self.priorities[self.position] = self.max_priority
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        """
        Sample batch with prioritized sampling.

        Returns:
            experiences: List of experience dicts
            indices: Buffer indices for priority updates
            weights: Importance sampling weights
        """
        if len(self.buffer) == 0:
            return [], [], []

        buffer_len = len(self.buffer)
        batch_size = min(batch_size, buffer_len)

        # Compute sampling probabilities
        priorities = self.priorities[:buffer_len] ** self.alpha
        probs = priorities / priorities.sum()

        # Sample indices
        indices = np.random.choice(buffer_len, batch_size, p=probs, replace=False)

        # Compute importance sampling weights
        self.beta = min(1.0, self.beta + self.beta_increment)
        weights = (buffer_len * probs[indices]) ** (-self.beta)
        weights /= weights.max()

        experiences = [self.buffer[i] for i in indices]

        return experiences, indices.tolist(), weights.tolist()

    def update_priorities(self, indices, priorities):
        """Update priorities for sampled experiences."""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)

    def __len__(self):
        return len(self.buffer)


class MoleculeMemory:
    """
    Memory of generated molecules with diversity tracking.
    Keeps track of unique valid molecules and their properties.
    """

    def __init__(self, max_molecules=10000, similarity_threshold=0.9):
        """
        Args:
            max_molecules: Maximum molecules to store
            similarity_threshold: Threshold for considering molecules similar
        """
        self.max_molecules = max_molecules
        self.similarity_threshold = similarity_threshold

        self.molecules = {}  # SMILES -> properties dict
        self.fingerprints = {}  # SMILES -> fingerprint
        self.rewards = []  # History of rewards

    def add(self, smiles, reward, properties=None):
        """
        Add molecule to memory if unique.

        Args:
            smiles: SMILES string
            reward: Reward value
            properties: Optional property dict

        Returns:
            True if added (novel), False if duplicate
        """
        from rdkit import Chem
        from rdkit.Chem import AllChem, DataStructs

        # Canonicalize
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False

        canonical = Chem.MolToSmiles(mol)

        # Check if already exists
        if canonical in self.molecules:
            return False

        # Check similarity
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)

        for existing_smiles, existing_fp in self.fingerprints.items():
            similarity = DataStructs.TanimotoSimilarity(fp, existing_fp)
            if similarity > self.similarity_threshold:
                return False

        # Add molecule
        self.molecules[canonical] = {
            'reward': reward,
            'properties': properties or {},
            'count': 1
        }
        self.fingerprints[canonical] = fp
        self.rewards.append(reward)

        # Enforce capacity
        if len(self.molecules) > self.max_molecules:
            self._remove_worst()

        return True

    def _remove_worst(self):
        """Remove molecule with lowest reward."""
        worst_smiles = min(self.molecules, key=lambda s: self.molecules[s]['reward'])
        del self.molecules[worst_smiles]
        del self.fingerprints[worst_smiles]

    def get_statistics(self):
        """Get memory statistics."""
        if not self.molecules:
            return {}

        rewards = [m['reward'] for m in self.molecules.values()]
        return {
            'n_molecules': len(self.molecules),
            'mean_reward': np.mean(rewards),
            'max_reward': np.max(rewards),
            'min_reward': np.min(rewards),
            'std_reward': np.std(rewards)
        }

    def get_top_k(self, k):
        """Get top k molecules by reward."""
        sorted_mols = sorted(
            self.molecules.items(),
            key=lambda x: x[1]['reward'],
            reverse=True
        )
        return [(smiles, data) for smiles, data in sorted_mols[:k]]

    def __len__(self):
        return len(self.molecules)

    def __contains__(self, smiles):
        from rdkit import Chem
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False
        canonical = Chem.MolToSmiles(mol)
        return canonical in self.molecules
