"""
Reward Functions for Molecular Generation
Includes property-based and one-shot learning rewards
"""

import torch
import torch.nn.functional as F
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, QED


class RewardFunction:
    """Base class for reward functions."""

    def __call__(self, smiles, predictor=None, **kwargs):
        """
        Compute reward for a SMILES string.

        Args:
            smiles: SMILES string
            predictor: Optional predictor model
            **kwargs: Additional arguments

        Returns:
            reward: Float reward value
        """
        raise NotImplementedError


class LogPReward(RewardFunction):
    """Reward based on LogP (lipophilicity)."""

    def __init__(self, target_range=(1.0, 4.0), use_rdkit=True,
                 invalid_reward=0.0):
        """
        Args:
            target_range: Desired LogP range (min, max)
            use_rdkit: Use RDKit for LogP calculation
            invalid_reward: Reward for invalid molecules
        """
        self.target_range = target_range
        self.use_rdkit = use_rdkit
        self.invalid_reward = invalid_reward

    def __call__(self, smiles, predictor=None, **kwargs):
        """Compute LogP-based reward."""
        # Validate molecule
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return self.invalid_reward

        # Compute LogP
        if self.use_rdkit:
            logp = Crippen.MolLogP(mol)
        elif predictor is not None:
            _, pred, _ = predictor.predict([smiles])
            if len(pred) == 0:
                return self.invalid_reward
            logp = pred[0]
        else:
            return self.invalid_reward

        # Compute reward based on target range
        min_logp, max_logp = self.target_range

        if min_logp <= logp <= max_logp:
            # Maximum reward in target range
            return 11.0
        else:
            # Smaller reward outside range
            return 1.0


class QEDReward(RewardFunction):
    """Reward based on QED (Quantitative Estimate of Drug-likeness)."""

    def __init__(self, threshold=0.5, invalid_reward=0.0):
        """
        Args:
            threshold: QED threshold for bonus reward
            invalid_reward: Reward for invalid molecules
        """
        self.threshold = threshold
        self.invalid_reward = invalid_reward

    def __call__(self, smiles, predictor=None, **kwargs):
        """Compute QED-based reward."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return self.invalid_reward

        try:
            qed_value = QED.qed(mol)
            return qed_value * 10.0  # Scale to reasonable range
        except:
            return self.invalid_reward


class OneShotReward(RewardFunction):
    """
    Reward using one-shot learning predictor.
    Uses support set to predict property of generated molecules.
    """

    def __init__(self, one_shot_predictor, tasks, target_classes,
                 class_weights=None, invalid_reward=0.0):
        """
        Args:
            one_shot_predictor: One-shot learning model
            tasks: List of task names
            target_classes: Target class indices to optimize for
            class_weights: Weights for each class
            invalid_reward: Reward for invalid molecules
        """
        self.predictor = one_shot_predictor
        self.tasks = tasks
        self.target_classes = target_classes
        self.class_weights = class_weights or [1.0] * len(target_classes)
        self.invalid_reward = invalid_reward

    def __call__(self, smiles, support_data=None, **kwargs):
        """
        Compute one-shot based reward.

        Args:
            smiles: SMILES string
            support_data: Support set data for one-shot prediction
        """
        # Validate molecule
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return self.invalid_reward

        if support_data is None:
            return self.invalid_reward

        # Featurize molecule
        try:
            query_features = self._featurize(smiles)
        except:
            return self.invalid_reward

        # Get predictions from one-shot model
        predictions = self.predictor.predict(
            query_features,
            support_data['x'],
            support_data['y']
        )

        # Compute weighted reward
        reward = 0.0
        for cls, weight in zip(self.target_classes, self.class_weights):
            reward += weight * predictions[cls].item()

        return reward

    def _featurize(self, smiles):
        """Featurize SMILES for prediction."""
        # This should match the featurization used for training
        # Placeholder - implement based on your featurization
        raise NotImplementedError


class MultiObjectiveReward(RewardFunction):
    """
    Multi-objective reward combining multiple properties.
    Uses scalarization or Pareto-based methods.
    """

    def __init__(self, objectives, weights=None, aggregation='weighted_sum',
                 invalid_reward=0.0):
        """
        Args:
            objectives: Dict of objective name -> reward function
            weights: Dict of objective name -> weight
            aggregation: 'weighted_sum', 'min', 'product'
            invalid_reward: Reward for invalid molecules
        """
        self.objectives = objectives
        self.weights = weights or {k: 1.0 for k in objectives}
        self.aggregation = aggregation
        self.invalid_reward = invalid_reward

        # Normalize weights
        total = sum(self.weights.values())
        self.weights = {k: v / total for k, v in self.weights.items()}

    def __call__(self, smiles, predictor=None, **kwargs):
        """Compute multi-objective reward."""
        # Validate molecule
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return self.invalid_reward

        # Compute individual rewards
        rewards = {}
        for name, obj_fn in self.objectives.items():
            rewards[name] = obj_fn(smiles, predictor, **kwargs)

        # Aggregate
        if self.aggregation == 'weighted_sum':
            return sum(self.weights[k] * rewards[k] for k in rewards)
        elif self.aggregation == 'min':
            return min(rewards.values())
        elif self.aggregation == 'product':
            prod = 1.0
            for r in rewards.values():
                prod *= max(r, 0.01)  # Avoid zero
            return prod ** (1.0 / len(rewards))
        else:
            return sum(rewards.values()) / len(rewards)


class SAReward(RewardFunction):
    """Reward based on Synthetic Accessibility."""

    def __init__(self, sa_threshold=6.0, invalid_reward=0.0):
        """
        Args:
            sa_threshold: SA score threshold (lower is better)
            invalid_reward: Reward for invalid molecules
        """
        self.sa_threshold = sa_threshold
        self.invalid_reward = invalid_reward

    def __call__(self, smiles, predictor=None, **kwargs):
        """Compute SA-based reward."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return self.invalid_reward

        try:
            # Import SA calculation (assumes SAcal.py is available)
            from SAcal import calculateScore
            sa_score = calculateScore(mol)

            # Lower SA is better, so invert
            if sa_score < self.sa_threshold:
                return (self.sa_threshold - sa_score) / self.sa_threshold * 10.0
            else:
                return 1.0
        except:
            return self.invalid_reward


class TargetSimilarityReward(RewardFunction):
    """Reward based on similarity to target molecule."""

    def __init__(self, target_smiles, fingerprint='morgan', threshold=0.7,
                 invalid_reward=0.0):
        """
        Args:
            target_smiles: Target molecule SMILES
            fingerprint: Fingerprint type ('morgan', 'rdkit', 'maccs')
            threshold: Similarity threshold for bonus
            invalid_reward: Reward for invalid molecules
        """
        from rdkit.Chem import AllChem, MACCSkeys
        from rdkit import DataStructs

        self.target_smiles = target_smiles
        self.fingerprint = fingerprint
        self.threshold = threshold
        self.invalid_reward = invalid_reward

        # Compute target fingerprint
        target_mol = Chem.MolFromSmiles(target_smiles)
        self.target_fp = self._get_fingerprint(target_mol)

    def _get_fingerprint(self, mol):
        """Get fingerprint for molecule."""
        from rdkit.Chem import AllChem, MACCSkeys

        if self.fingerprint == 'morgan':
            return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
        elif self.fingerprint == 'rdkit':
            return Chem.RDKFingerprint(mol)
        elif self.fingerprint == 'maccs':
            return MACCSkeys.GenMACCSKeys(mol)
        else:
            return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)

    def __call__(self, smiles, predictor=None, **kwargs):
        """Compute similarity-based reward."""
        from rdkit import DataStructs

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return self.invalid_reward

        try:
            fp = self._get_fingerprint(mol)
            similarity = DataStructs.TanimotoSimilarity(self.target_fp, fp)

            # Scale reward
            if similarity >= self.threshold:
                return similarity * 10.0
            else:
                return similarity * 5.0
        except:
            return self.invalid_reward


class ConstrainedReward(RewardFunction):
    """Reward with hard constraints."""

    def __init__(self, base_reward, constraints, invalid_reward=0.0):
        """
        Args:
            base_reward: Base reward function
            constraints: List of constraint functions (return True if satisfied)
            invalid_reward: Reward for invalid/constrained molecules
        """
        self.base_reward = base_reward
        self.constraints = constraints
        self.invalid_reward = invalid_reward

    def __call__(self, smiles, predictor=None, **kwargs):
        """Compute constrained reward."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return self.invalid_reward

        # Check constraints
        for constraint in self.constraints:
            if not constraint(mol):
                return self.invalid_reward

        # Return base reward
        return self.base_reward(smiles, predictor, **kwargs)


# Common constraints
def mw_constraint(min_mw=200, max_mw=500):
    """Molecular weight constraint."""
    def check(mol):
        mw = Descriptors.MolWt(mol)
        return min_mw <= mw <= max_mw
    return check


def hbd_constraint(max_hbd=5):
    """Hydrogen bond donor constraint."""
    def check(mol):
        return Descriptors.NumHDonors(mol) <= max_hbd
    return check


def hba_constraint(max_hba=10):
    """Hydrogen bond acceptor constraint."""
    def check(mol):
        return Descriptors.NumHAcceptors(mol) <= max_hba
    return check


def rotatable_bonds_constraint(max_rotatable=10):
    """Rotatable bonds constraint."""
    def check(mol):
        return Descriptors.NumRotatableBonds(mol) <= max_rotatable
    return check


def lipinski_constraint():
    """Lipinski's Rule of Five."""
    def check(mol):
        mw = Descriptors.MolWt(mol)
        logp = Crippen.MolLogP(mol)
        hbd = Descriptors.NumHDonors(mol)
        hba = Descriptors.NumHAcceptors(mol)

        violations = 0
        if mw > 500:
            violations += 1
        if logp > 5:
            violations += 1
        if hbd > 5:
            violations += 1
        if hba > 10:
            violations += 1

        return violations <= 1  # Allow one violation
    return check
