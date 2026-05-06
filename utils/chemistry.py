"""
Chemistry Utilities
SMILES validation, canonicalization, and property computation
"""

import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, QED, AllChem


def canonical_smiles(smiles_list, sanitize=True, throw_warning=False):
    """
    Convert SMILES to canonical form.

    Args:
        smiles_list: List of SMILES strings
        sanitize: Sanitize molecules
        throw_warning: Print warnings for invalid SMILES

    Returns:
        List of canonical SMILES (None for invalid)
    """
    canonical = []

    for smiles in smiles_list:
        try:
            mol = Chem.MolFromSmiles(smiles, sanitize=sanitize)
            if mol is not None:
                canonical.append(Chem.MolToSmiles(mol))
            else:
                canonical.append(None)
                if throw_warning:
                    print(f"Warning: Invalid SMILES: {smiles}")
        except:
            canonical.append(None)
            if throw_warning:
                print(f"Warning: Error processing SMILES: {smiles}")

    return canonical


def validate_smiles(smiles):
    """
    Validate SMILES string.

    Args:
        smiles: SMILES string

    Returns:
        True if valid, False otherwise
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        return mol is not None
    except:
        return False


def compute_properties(smiles, properties=None):
    """
    Compute molecular properties.

    Args:
        smiles: SMILES string
        properties: List of properties to compute (None = all)

    Returns:
        Dict of property values
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    available_properties = {
        'mw': lambda m: Descriptors.MolWt(m),
        'logp': lambda m: Crippen.MolLogP(m),
        'hbd': lambda m: Descriptors.NumHDonors(m),
        'hba': lambda m: Descriptors.NumHAcceptors(m),
        'tpsa': lambda m: Descriptors.TPSA(m),
        'rotatable_bonds': lambda m: Descriptors.NumRotatableBonds(m),
        'qed': lambda m: QED.qed(m),
        'n_rings': lambda m: Descriptors.RingCount(m),
        'n_aromatic_rings': lambda m: Descriptors.NumAromaticRings(m),
        'n_heavy_atoms': lambda m: mol.GetNumHeavyAtoms(),
        'fraction_csp3': lambda m: Descriptors.FractionCSP3(m),
    }

    if properties is None:
        properties = list(available_properties.keys())

    result = {}
    for prop in properties:
        if prop in available_properties:
            try:
                result[prop] = available_properties[prop](mol)
            except:
                result[prop] = None

    return result


def compute_fingerprint(smiles, fp_type='morgan', radius=2, nbits=1024):
    """
    Compute molecular fingerprint.

    Args:
        smiles: SMILES string
        fp_type: Fingerprint type ('morgan', 'rdkit', 'maccs')
        radius: Morgan fingerprint radius
        nbits: Number of bits

    Returns:
        Numpy array fingerprint
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    if fp_type == 'morgan':
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)
    elif fp_type == 'rdkit':
        fp = Chem.RDKFingerprint(mol, fpSize=nbits)
    elif fp_type == 'maccs':
        from rdkit.Chem import MACCSkeys
        fp = MACCSkeys.GenMACCSKeys(mol)
    else:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)

    from rdkit import DataStructs
    arr = np.zeros((len(fp),), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def compute_similarity(smiles1, smiles2, fp_type='morgan'):
    """
    Compute Tanimoto similarity between two molecules.

    Args:
        smiles1: First SMILES
        smiles2: Second SMILES
        fp_type: Fingerprint type

    Returns:
        Similarity score (0-1)
    """
    from rdkit import DataStructs

    mol1 = Chem.MolFromSmiles(smiles1)
    mol2 = Chem.MolFromSmiles(smiles2)

    if mol1 is None or mol2 is None:
        return None

    if fp_type == 'morgan':
        fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 2, nBits=1024)
        fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 2, nBits=1024)
    else:
        fp1 = Chem.RDKFingerprint(mol1)
        fp2 = Chem.RDKFingerprint(mol2)

    return DataStructs.TanimotoSimilarity(fp1, fp2)


def filter_valid_smiles(smiles_list):
    """
    Filter list to keep only valid SMILES.

    Args:
        smiles_list: List of SMILES

    Returns:
        List of valid SMILES
    """
    return [s for s in smiles_list if validate_smiles(s)]


def get_unique_smiles(smiles_list):
    """
    Get unique canonical SMILES.

    Args:
        smiles_list: List of SMILES

    Returns:
        List of unique canonical SMILES
    """
    canonical = canonical_smiles(smiles_list)
    unique = []
    seen = set()

    for smi in canonical:
        if smi is not None and smi not in seen:
            unique.append(smi)
            seen.add(smi)

    return unique


def lipinski_filter(smiles):
    """
    Check if molecule passes Lipinski's Rule of Five.

    Args:
        smiles: SMILES string

    Returns:
        True if passes, False otherwise
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False

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

    return violations <= 1


# ---- Scaffold-based splitting ----

def scaffold_split(smiles_list, frac_train=0.8, frac_val=0.1, seed=42):
    """Split molecules by Murcko scaffold to avoid data leakage.

    Returns:
        train_idx, val_idx, test_idx: Lists of indices.
    """
    from rdkit.Chem.Scaffolds import MurckoScaffold
    import random

    scaffolds = {}
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        try:
            scaffold = MurckoScaffold.MurckoScaffoldSmiles(
                mol=mol, includeChirality=False
            )
        except Exception:
            scaffold = ''
        scaffolds.setdefault(scaffold, []).append(i)

    # Sort scaffold groups by size (largest first) — standard scaffold split
    # assigns largest groups to training to maximize training diversity
    groups = sorted(scaffolds.values(), key=len, reverse=True)

    n = len(smiles_list)
    train_idx, val_idx, test_idx = [], [], []
    for group in groups:
        if len(train_idx) / n < frac_train:
            train_idx.extend(group)
        elif len(val_idx) / n < frac_val:
            val_idx.extend(group)
        else:
            test_idx.extend(group)

    return train_idx, val_idx, test_idx


# ---- PubChem novelty checking ----

def check_pubchem_novelty(smiles_list, batch_size=50):
    """Check which SMILES correspond to known PubChem compounds.

    Returns:
        known: List of (smiles, cid) for compounds found in PubChem.
        novel: List of smiles NOT found in PubChem.
    """
    import urllib.request
    import json
    import time

    known, novel = [], []
    for i in range(0, len(smiles_list), batch_size):
        batch = smiles_list[i:i + batch_size]
        for smi in batch:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                novel.append(smi)
                continue
            can = Chem.MolToSmiles(mol)
            try:
                url = (
                    f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/"
                    f"compound/smiles/{urllib.parse.quote(can, safe='')}"
                    f"/cids/JSON"
                )
                req = urllib.request.Request(url)
                with urllib.request.urlopen(req, timeout=10) as resp:
                    data = json.loads(resp.read())
                cids = data.get('IdentifierList', {}).get('CID', [])
                if cids:
                    known.append((smi, cids[0]))
                else:
                    novel.append(smi)
            except Exception:
                novel.append(smi)  # treat failures as novel (conservative)
            time.sleep(0.25)  # rate limit

    return known, novel
