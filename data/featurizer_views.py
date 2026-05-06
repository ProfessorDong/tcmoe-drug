"""
Multi-view molecular feature extraction for the MoE predictor.

Provides four representations on a common interface:
    1. Morgan (ECFP4) fingerprints, 1024-bit, radius 2
    2. Dual-encoder embedding (loaded from a trained DualEncoder)
    3. MACCS keys (167-bit)
    4. Standardized RDKit physicochemical descriptors

Each view returns a fixed-dimensional float32 numpy array per SMILES.
"""

from __future__ import annotations
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys, Descriptors, Crippen, QED
from rdkit import DataStructs


MORGAN_BITS = 1024
MACCS_BITS = 167


# ---------------------------------------------------------------------------
# Morgan / MACCS
# ---------------------------------------------------------------------------

def morgan_fp(smiles: str, n_bits: int = MORGAN_BITS, radius: int = 2) -> np.ndarray:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(n_bits, dtype=np.float32)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    arr = np.zeros(n_bits, dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def maccs_keys(smiles: str) -> np.ndarray:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(MACCS_BITS, dtype=np.float32)
    fp = MACCSkeys.GenMACCSKeys(mol)
    arr = np.zeros(MACCS_BITS, dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


# ---------------------------------------------------------------------------
# RDKit physicochemical descriptors (selected, robust subset)
# ---------------------------------------------------------------------------

# 24 descriptors covering size, polarity, lipophilicity, ring/topology,
# charge, and drug-likeness. Intentionally robust (no electronic-state
# descriptors that are slow or fragile).
DESCRIPTOR_FNS = [
    ('MolWt', Descriptors.MolWt),
    ('HeavyAtomCount', Descriptors.HeavyAtomCount),
    ('NumHeteroatoms', Descriptors.NumHeteroatoms),
    ('NumHDonors', Descriptors.NumHDonors),
    ('NumHAcceptors', Descriptors.NumHAcceptors),
    ('NumRotatableBonds', Descriptors.NumRotatableBonds),
    ('NumAromaticRings', Descriptors.NumAromaticRings),
    ('NumAliphaticRings', Descriptors.NumAliphaticRings),
    ('NumSaturatedRings', Descriptors.NumSaturatedRings),
    ('RingCount', Descriptors.RingCount),
    ('FractionCSP3', Descriptors.FractionCSP3),
    ('TPSA', Descriptors.TPSA),
    ('LabuteASA', Descriptors.LabuteASA),
    ('MolLogP', Crippen.MolLogP),
    ('MolMR', Crippen.MolMR),
    ('BalabanJ', Descriptors.BalabanJ),
    ('BertzCT', Descriptors.BertzCT),
    ('Chi0v', Descriptors.Chi0v),
    ('Chi1v', Descriptors.Chi1v),
    ('Kappa1', Descriptors.Kappa1),
    ('Kappa2', Descriptors.Kappa2),
    ('NumValenceElectrons', Descriptors.NumValenceElectrons),
    ('QED', QED.qed),
    ('FormalCharge', lambda m: Chem.GetFormalCharge(m)),
]
DESCRIPTOR_DIM = len(DESCRIPTOR_FNS)


def rdkit_descriptors(smiles: str) -> np.ndarray:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(DESCRIPTOR_DIM, dtype=np.float32)
    out = np.empty(DESCRIPTOR_DIM, dtype=np.float32)
    for i, (_, fn) in enumerate(DESCRIPTOR_FNS):
        try:
            v = fn(mol)
            out[i] = float(v) if np.isfinite(float(v)) else 0.0
        except Exception:
            out[i] = 0.0
    return out


class DescriptorScaler:
    """Robust standardizer for RDKit descriptors.

    Uses median / IQR to be insensitive to descriptor outliers (e.g.,
    extreme MolWt molecules). Stores per-descriptor center and scale
    so test data uses training-time statistics.
    """

    def __init__(self) -> None:
        self.center: np.ndarray | None = None
        self.scale: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> 'DescriptorScaler':
        med = np.median(X, axis=0)
        q1 = np.percentile(X, 25, axis=0)
        q3 = np.percentile(X, 75, axis=0)
        iqr = q3 - q1
        iqr[iqr < 1e-6] = 1.0
        self.center = med.astype(np.float32)
        self.scale = iqr.astype(np.float32)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        assert self.center is not None and self.scale is not None
        Z = (X - self.center) / self.scale
        # Clip extreme outliers to keep gradients well-behaved
        return np.clip(Z, -5.0, 5.0).astype(np.float32)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.transform(X)


# ---------------------------------------------------------------------------
# Batch helpers
# ---------------------------------------------------------------------------

def batch_morgan(smiles_list, n_bits: int = MORGAN_BITS) -> np.ndarray:
    return np.stack([morgan_fp(s, n_bits=n_bits) for s in smiles_list])


def batch_maccs(smiles_list) -> np.ndarray:
    return np.stack([maccs_keys(s) for s in smiles_list])


def batch_descriptors(smiles_list) -> np.ndarray:
    return np.stack([rdkit_descriptors(s) for s in smiles_list])
