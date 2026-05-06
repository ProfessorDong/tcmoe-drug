#!/usr/bin/env python
"""
REINVENT 4 Baseline Runner for Theranostics Drug Discovery Framework
=====================================================================

Wraps REINVENT 4 (AstraZeneca's molecular generative model) as a baseline
for comparison against our Theranostics framework.

REINVENT 4 uses:
  - RNN-based (Reinvent) or Transformer-based (Mol2Mol) generative models
  - Reinforcement Learning (staged/curriculum learning) for property optimization
  - Transfer Learning to focus the prior on target-relevant chemical space
  - A plugin-based scoring system with built-in QED, SA, physchem descriptors

Workflow:
  1. (Optional) Transfer learning on known actives from the dataset
  2. Staged RL with multi-objective scoring (QED, MW, LogP, similarity to actives)
  3. Sample molecules from the optimized agent
  4. Evaluate generated molecules on standard metrics

Usage:
  python reinvent4_baseline.py \\
      --dataset scd1 \\
      --data_path /path/to/scd1_binding.csv \\
      --output_dir /path/to/results

  python reinvent4_baseline.py \\
      --dataset drd2 \\
      --data_path /path/to/drd2_bioactivity.csv \\
      --output_dir ./results/drd2 \\
      --num_samples 2000 \\
      --rl_steps 300 \\
      --device cuda:0

  # Evaluation only (on previously generated molecules)
  python reinvent4_baseline.py \\
      --evaluate_only \\
      --generated_smiles /path/to/generated.csv \\
      --data_path /path/to/dataset.csv \\
      --output_dir ./results

Requirements:
  - REINVENT 4 installed (pip install -e path/to/REINVENT4-main)
  - Prior model file (reinvent.prior) in REINVENT4-main/priors/
  - rdkit, torch, pandas, numpy
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import shutil
import tempfile
import logging
from pathlib import Path
from typing import List, Optional, Dict, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Resolve REINVENT4 path before any reinvent imports
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
BASELINES_DIR = SCRIPT_DIR
PROJECT_DIR = SCRIPT_DIR.parent
REINVENT4_DIR = PROJECT_DIR.parent / "REINVENT4-main"

# Ensure REINVENT4 is on sys.path
if str(REINVENT4_DIR) not in sys.path:
    sys.path.insert(0, str(REINVENT4_DIR))

# ---------------------------------------------------------------------------
# RDKit imports (always available)
# ---------------------------------------------------------------------------
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors, Crippen, QED as QED_module
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit import DataStructs

RDLogger.DisableLog("rdApp.*")

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("reinvent4_baseline")


# ============================================================================
# Metrics computation (RDKit-only, no REINVENT dependency)
# ============================================================================

def compute_sa_score(mol):
    """Compute synthetic accessibility score (Ertl, 2009).

    Uses the SA score implementation bundled with REINVENT 4 if available,
    otherwise falls back to a simplified version.
    """
    try:
        from reinvent_plugins.components.SAScore.comp_sascore import SAScore as _SA
        # Use REINVENT's bundled SA score
        scorer = _SA()
        result = scorer([mol])
        return result[0][0]
    except Exception:
        pass

    # Fallback: use rdkit's built-in SA contrib if available
    try:
        from rdkit.Chem import RDConfig
        sa_model_path = os.path.join(RDConfig.RDContribDir, "SA_Score", "sascorer.py")
        if os.path.exists(sa_model_path):
            import importlib.util
            spec = importlib.util.spec_from_file_location("sascorer", sa_model_path)
            sascorer = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(sascorer)
            return sascorer.calculateScore(mol)
    except Exception:
        pass

    # Last fallback: simplified heuristic
    ring_info = mol.GetRingInfo()
    num_rings = ring_info.NumRings()
    num_stereo = Chem.CalcNumAtomStereoCenters(mol)
    num_rot = Descriptors.NumRotatableBonds(mol)
    mw = Descriptors.MolWt(mol)

    # Heuristic: higher score = harder to synthesize (1-10 scale)
    score = 1.0 + 0.5 * num_rings + 0.8 * num_stereo + 0.1 * num_rot
    score += max(0, (mw - 400) / 200)
    return min(score, 10.0)


def canonical_smiles(smi: str) -> Optional[str]:
    """Canonicalize a SMILES string. Returns None if invalid."""
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol)
    except Exception:
        return None


def compute_molecule_properties(smiles_list: List[str]) -> pd.DataFrame:
    """Compute molecular properties for a list of SMILES.

    Returns a DataFrame with columns:
        SMILES, Valid, QED, SA_Score, LogP, MW, TPSA, HBA, HBD, NumRotBonds
    """
    records = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            records.append({
                "SMILES": smi, "Valid": False,
                "QED": np.nan, "SA_Score": np.nan, "LogP": np.nan,
                "MW": np.nan, "TPSA": np.nan, "HBA": np.nan, "HBD": np.nan,
                "NumRotBonds": np.nan,
            })
            continue

        try:
            qed = QED_module.qed(mol)
        except Exception:
            qed = np.nan
        try:
            sa = compute_sa_score(mol)
        except Exception:
            sa = np.nan

        records.append({
            "SMILES": smi,
            "Valid": True,
            "QED": round(qed, 4),
            "SA_Score": round(sa, 4),
            "LogP": round(Crippen.MolLogP(mol), 4),
            "MW": round(Descriptors.MolWt(mol), 2),
            "TPSA": round(Descriptors.TPSA(mol), 2),
            "HBA": Descriptors.NumHAcceptors(mol),
            "HBD": Descriptors.NumHDonors(mol),
            "NumRotBonds": Descriptors.NumRotatableBonds(mol),
        })

    return pd.DataFrame(records)


def compute_fingerprint(smi: str, radius: int = 3, n_bits: int = 2048):
    """Compute Morgan fingerprint for a SMILES."""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)


def compute_internal_diversity(smiles_list: List[str], sample_size: int = 1000) -> float:
    """Compute internal diversity as 1 - average pairwise Tanimoto similarity."""
    fps = []
    for smi in smiles_list:
        fp = compute_fingerprint(smi)
        if fp is not None:
            fps.append(fp)

    if len(fps) < 2:
        return 0.0

    # Sample if too many
    if len(fps) > sample_size:
        indices = np.random.choice(len(fps), sample_size, replace=False)
        fps = [fps[i] for i in indices]

    similarities = []
    for i in range(len(fps)):
        for j in range(i + 1, len(fps)):
            sim = DataStructs.TanimotoSimilarity(fps[i], fps[j])
            similarities.append(sim)

    return 1.0 - np.mean(similarities)


def compute_novelty(generated: List[str], reference: List[str]) -> float:
    """Fraction of generated molecules not in the reference set."""
    ref_set = set()
    for smi in reference:
        can = canonical_smiles(smi)
        if can:
            ref_set.add(can)

    novel = 0
    total = 0
    for smi in generated:
        can = canonical_smiles(smi)
        if can:
            total += 1
            if can not in ref_set:
                novel += 1

    return novel / total if total > 0 else 0.0


def compute_generation_metrics(
    generated_smiles: List[str],
    reference_smiles: List[str],
) -> Dict[str, float]:
    """Compute standard generative model metrics.

    Returns dict with: validity, uniqueness, novelty, diversity
    """
    # Validity
    valid_smiles = []
    for smi in generated_smiles:
        can = canonical_smiles(smi)
        if can:
            valid_smiles.append(can)
    validity = len(valid_smiles) / len(generated_smiles) if generated_smiles else 0.0

    # Uniqueness
    unique_smiles = list(set(valid_smiles))
    uniqueness = len(unique_smiles) / len(valid_smiles) if valid_smiles else 0.0

    # Novelty
    novelty = compute_novelty(unique_smiles, reference_smiles)

    # Internal diversity
    diversity = compute_internal_diversity(unique_smiles)

    return {
        "Total Generated": len(generated_smiles),
        "Valid": len(valid_smiles),
        "Unique": len(unique_smiles),
        "Validity (%)": round(validity * 100, 2),
        "Uniqueness (%)": round(uniqueness * 100, 2),
        "Novelty (%)": round(novelty * 100, 2),
        "Internal Diversity": round(diversity, 4),
    }


def compute_property_statistics(props_df: pd.DataFrame) -> Dict[str, str]:
    """Compute summary statistics for molecular properties."""
    valid_df = props_df[props_df["Valid"] == True]
    if len(valid_df) == 0:
        return {"Error": "No valid molecules"}

    stats = {}
    for col in ["QED", "SA_Score", "LogP", "MW", "TPSA"]:
        if col in valid_df.columns:
            vals = valid_df[col].dropna()
            if len(vals) > 0:
                stats[f"{col} mean"] = f"{vals.mean():.3f}"
                stats[f"{col} std"] = f"{vals.std():.3f}"
                stats[f"{col} median"] = f"{vals.median():.3f}"
                stats[f"{col} [min, max]"] = f"[{vals.min():.3f}, {vals.max():.3f}]"
    return stats


# ============================================================================
# REINVENT 4 Interface
# ============================================================================

def find_prior_model() -> Optional[str]:
    """Locate the REINVENT prior model file."""
    # Check standard locations
    candidates = [
        REINVENT4_DIR / "priors" / "reinvent.prior",
        Path(os.environ.get("REINVENT_PRIOR_BASE", "")) / "reinvent.prior",
    ]

    # Check via reinvent package if importable
    try:
        import reinvent
        pkg_dir = Path(reinvent.__file__).parent
        candidates.append(pkg_dir.parent / "priors" / "reinvent.prior")
    except ImportError:
        pass

    for path in candidates:
        if path.exists():
            return str(path)

    return None


def prepare_smiles_file(data_path: str, output_path: str, max_smiles: int = 500) -> str:
    """Extract SMILES from a dataset CSV and write to a .smi file for REINVENT.

    Takes the top compounds by pIC50 (if available) as reference actives.
    """
    df = pd.read_csv(data_path)

    # Identify SMILES column
    smiles_col = None
    for col in ["SMILES", "smiles", "Smiles", "canonical_smiles"]:
        if col in df.columns:
            smiles_col = col
            break
    if smiles_col is None:
        raise ValueError(f"No SMILES column found in {data_path}")

    # Sort by pIC50 if available (take most active compounds)
    for col in ["pIC50", "pIC50_value", "pic50", "activity"]:
        if col in df.columns:
            df = df.sort_values(col, ascending=False)
            break

    # Take top compounds
    smiles_list = df[smiles_col].head(max_smiles).tolist()

    # Validate and canonicalize
    valid_smiles = []
    for smi in smiles_list:
        can = canonical_smiles(str(smi))
        if can:
            valid_smiles.append(can)

    with open(output_path, "w") as f:
        for smi in valid_smiles:
            f.write(f"{smi}\n")

    logger.info(f"Wrote {len(valid_smiles)} SMILES to {output_path}")
    return output_path


def get_reference_actives(data_path: str, top_n: int = 5) -> List[str]:
    """Get top active SMILES from dataset for use in similarity scoring."""
    df = pd.read_csv(data_path)

    smiles_col = None
    for col in ["SMILES", "smiles", "Smiles", "canonical_smiles"]:
        if col in df.columns:
            smiles_col = col
            break
    if smiles_col is None:
        return []

    activity_col = None
    for col in ["pIC50", "pIC50_value", "pic50", "activity"]:
        if col in df.columns:
            activity_col = col
            break

    if activity_col:
        df = df.sort_values(activity_col, ascending=False)

    smiles = []
    for smi in df[smiles_col].head(top_n * 2).tolist():
        can = canonical_smiles(str(smi))
        if can and len(can) < 200:  # Skip very long SMILES
            smiles.append(can)
        if len(smiles) >= top_n:
            break

    return smiles


def create_rl_config(
    dataset: str,
    prior_file: str,
    agent_file: str,
    output_dir: str,
    reference_smiles: List[str],
    device: str = "cuda:0",
    batch_size: int = 128,
    rl_steps: int = 300,
) -> str:
    """Create a REINVENT 4 staged learning TOML config programmatically.

    This generates the config dynamically rather than using a template,
    so it can incorporate reference SMILES from the actual dataset.
    """

    # Build reference SMILES TOML list
    ref_smiles_toml = ",\n    ".join(f'"{smi}"' for smi in reference_smiles[:5])

    config = f'''# Auto-generated REINVENT 4 config for {dataset}
# Generated by reinvent4_baseline.py

run_type = "staged_learning"
device = "{device}"
tb_logdir = "tb_{dataset}"
json_out_config = "_{dataset}_config.json"

[parameters]

prior_file = "{prior_file}"
agent_file = "{agent_file}"
summary_csv_prefix = "{dataset}_rl"

batch_size = {batch_size}
use_checkpoint = false
purge_memories = false
unique_sequences = true
randomize_smiles = true

[learning_strategy]
type = "dap"
sigma = 128
rate = 0.0001

[diversity_filter]
type = "IdenticalMurckoScaffold"
bucket_size = 25
minscore = 0.4
minsimilarity = 0.4

# Stage 1: Drug-likeness + structural alerts
[[stage]]
max_score = 0.7
min_steps = 30
max_steps = {rl_steps // 2}
chkpt_file = "{dataset}_stage1.chkpt"
termination = "simple"

[stage.scoring]
type = "geometric_mean"

[[stage.scoring.component]]
[stage.scoring.component.custom_alerts]

[[stage.scoring.component.custom_alerts.endpoint]]
name = "Structural Alerts"
params.smarts = [
    "[*;r8]", "[*;r9]", "[*;r10]", "[*;r11]", "[*;r12]",
    "[*;r13]", "[*;r14]", "[*;r15]", "[*;r16]", "[*;r17]",
    "[#8][#8]", "[#6;+]", "[#16][#16]",
    "[#7;!n][S;!$(S(=O)=O)]", "[#7;!n][#7;!n]",
    "C#C", "C(=[O,S])[O,S]",
    "[#7;!n][C;!$(C(=[O,N])[N,O])][#16;!s]",
    "[#7;!n][C;!$(C(=[O,N])[N,O])][#7;!n]",
    "[#7;!n][C;!$(C(=[O,N])[N,O])][#8;!o]",
    "[#8;!o][C;!$(C(=[O,N])[N,O])][#16;!s]",
    "[#8;!o][C;!$(C(=[O,N])[N,O])][#8;!o]",
    "[#16;!s][C;!$(C(=[O,N])[N,O])][#16;!s]"
]

[[stage.scoring.component]]
[stage.scoring.component.QED]

[[stage.scoring.component.QED.endpoint]]
name = "QED"
weight = 0.5

[[stage.scoring.component]]
[stage.scoring.component.MolecularWeight]

[[stage.scoring.component.MolecularWeight.endpoint]]
name = "Molecular Weight"
weight = 0.3
transform.type = "double_sigmoid"
transform.high = 550.0
transform.low = 200.0
transform.coef_div = 500.0
transform.coef_si = 20.0
transform.coef_se = 20.0

[[stage.scoring.component]]
[stage.scoring.component.SlogP]

[[stage.scoring.component.SlogP.endpoint]]
name = "SLogP"
weight = 0.2
transform.type = "double_sigmoid"
transform.high = 5.0
transform.low = 1.0
transform.coef_div = 500.0
transform.coef_si = 20.0
transform.coef_se = 20.0

# Stage 2: Similarity to known actives + drug-likeness
[[stage]]
max_score = 0.85
min_steps = 30
max_steps = {rl_steps}
chkpt_file = "{dataset}_stage2.chkpt"
termination = "simple"

[stage.scoring]
type = "geometric_mean"

[[stage.scoring.component]]
[stage.scoring.component.custom_alerts]

[[stage.scoring.component.custom_alerts.endpoint]]
name = "Structural Alerts"
params.smarts = [
    "[*;r8]", "[*;r9]", "[*;r10]", "[*;r11]", "[*;r12]",
    "[*;r13]", "[*;r14]", "[*;r15]", "[*;r16]", "[*;r17]",
    "[#8][#8]", "[#6;+]", "[#16][#16]",
    "[#7;!n][S;!$(S(=O)=O)]", "[#7;!n][#7;!n]",
    "C#C", "C(=[O,S])[O,S]",
    "[#7;!n][C;!$(C(=[O,N])[N,O])][#16;!s]",
    "[#7;!n][C;!$(C(=[O,N])[N,O])][#7;!n]",
    "[#7;!n][C;!$(C(=[O,N])[N,O])][#8;!o]",
    "[#8;!o][C;!$(C(=[O,N])[N,O])][#16;!s]",
    "[#8;!o][C;!$(C(=[O,N])[N,O])][#8;!o]",
    "[#16;!s][C;!$(C(=[O,N])[N,O])][#16;!s]"
]

[[stage.scoring.component]]
[stage.scoring.component.QED]

[[stage.scoring.component.QED.endpoint]]
name = "QED"
weight = 0.3

[[stage.scoring.component]]
[stage.scoring.component.TanimotoSimilarity]

[[stage.scoring.component.TanimotoSimilarity.endpoint]]
name = "{dataset.upper()} Active Similarity"
weight = 0.5
params.smiles = [
    {ref_smiles_toml}
]
params.radius = 3
params.use_counts = true
params.use_features = true

[[stage.scoring.component]]
[stage.scoring.component.MolecularWeight]

[[stage.scoring.component.MolecularWeight.endpoint]]
name = "Molecular Weight"
weight = 0.2
transform.type = "double_sigmoid"
transform.high = 550.0
transform.low = 200.0
transform.coef_div = 500.0
transform.coef_si = 20.0
transform.coef_se = 20.0
'''
    config_path = os.path.join(output_dir, f"reinvent4_{dataset}_rl.toml")
    with open(config_path, "w") as f:
        f.write(config)

    logger.info(f"Wrote RL config to {config_path}")
    return config_path


def create_sampling_config(
    model_file: str,
    output_file: str,
    output_dir: str,
    num_smiles: int = 1000,
    device: str = "cuda:0",
) -> str:
    """Create a REINVENT 4 sampling TOML config."""

    config = f'''# Auto-generated REINVENT 4 sampling config
run_type = "sampling"
device = "{device}"
json_out_config = "_sampling.json"

[parameters]
model_file = "{model_file}"
output_file = "{output_file}"
num_smiles = {num_smiles}
unique_molecules = true
randomize_smiles = true
'''
    config_path = os.path.join(output_dir, "reinvent4_sampling.toml")
    with open(config_path, "w") as f:
        f.write(config)

    logger.info(f"Wrote sampling config to {config_path}")
    return config_path


def create_tl_config(
    prior_file: str,
    smiles_file: str,
    output_model_file: str,
    output_dir: str,
    num_epochs: int = 50,
    device: str = "cuda:0",
) -> str:
    """Create a REINVENT 4 transfer learning TOML config."""

    config = f'''# Auto-generated REINVENT 4 transfer learning config
run_type = "transfer_learning"
device = "{device}"
tb_logdir = "tb_TL"
json_out_config = "_tl_config.json"

[parameters]
num_epochs = {num_epochs}
save_every_n_epochs = {max(1, num_epochs // 5)}
batch_size = 128
num_refs = 0
sample_batch_size = 128

input_model_file = "{prior_file}"
smiles_file = "{smiles_file}"
output_model_file = "{output_model_file}"
validation_smiles_file = "{smiles_file}"
'''
    config_path = os.path.join(output_dir, "reinvent4_tl.toml")
    with open(config_path, "w") as f:
        f.write(config)

    logger.info(f"Wrote TL config to {config_path}")
    return config_path


def run_reinvent4(config_path: str, log_file: str, device: str = None) -> bool:
    """Run REINVENT 4 via the command-line interface.

    Uses subprocess to call 'reinvent' CLI to avoid internal state issues.
    Falls back to Python API if CLI is not available.
    """
    import subprocess

    cmd = ["reinvent", "-l", log_file]
    if device:
        cmd.extend(["-d", device])
    cmd.append(config_path)

    logger.info(f"Running: {' '.join(cmd)}")
    logger.info(f"Log file: {log_file}")

    try:
        env = os.environ.copy()
        env["MKL_SERVICE_FORCE_INTEL"] = "1"
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=7200,  # 2 hour timeout
            env=env,
        )
        if result.returncode != 0:
            logger.error(f"REINVENT 4 failed with return code {result.returncode}")
            logger.error(f"stderr: {result.stderr[-2000:]}")
            return False
        logger.info("REINVENT 4 run completed successfully")
        return True
    except FileNotFoundError:
        logger.warning("'reinvent' CLI not found on PATH, trying Python API...")
        return run_reinvent4_python_api(config_path, log_file, device)
    except subprocess.TimeoutExpired:
        logger.error("REINVENT 4 run timed out (2 hours)")
        return False


def run_reinvent4_python_api(config_path: str, log_file: str, device: str = None) -> bool:
    """Run REINVENT 4 via its Python API (fallback)."""
    try:
        from reinvent.Reinvent import main
        from types import SimpleNamespace

        args = SimpleNamespace(
            config_filename=Path(config_path).resolve(),
            config_format="toml",
            device=device,
            log_filename=log_file,
            log_level="info",
            seed=None,
            dotenv_filename=None,
            enable_rdkit_log_levels=None,
        )

        # Change to the config directory so relative paths work
        original_cwd = os.getcwd()
        os.chdir(Path(config_path).parent)
        try:
            main(args)
        finally:
            os.chdir(original_cwd)

        return True
    except Exception as e:
        logger.error(f"REINVENT 4 Python API failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def collect_generated_smiles(output_dir: str, prefix: str) -> List[str]:
    """Collect generated SMILES from REINVENT 4 CSV output files."""
    smiles = []

    # Look for RL output CSVs (staged_learning generates prefix_N.csv)
    for csv_file in sorted(Path(output_dir).glob(f"{prefix}_*.csv")):
        try:
            df = pd.read_csv(csv_file)
            if "SMILES" in df.columns:
                valid = df[df.get("SMILES_state", 1) != 0]["SMILES"].tolist()
                smiles.extend(valid)
        except Exception as e:
            logger.warning(f"Could not read {csv_file}: {e}")

    # Look for sampling output
    for csv_file in sorted(Path(output_dir).glob("sampling*.csv")):
        try:
            df = pd.read_csv(csv_file)
            if "SMILES" in df.columns:
                smiles.extend(df["SMILES"].tolist())
        except Exception as e:
            logger.warning(f"Could not read {csv_file}: {e}")

    # Deduplicate while preserving order
    seen = set()
    unique = []
    for smi in smiles:
        can = canonical_smiles(str(smi))
        if can and can not in seen:
            seen.add(can)
            unique.append(can)

    return unique


def print_summary_table(metrics: Dict, properties: Dict, dataset: str):
    """Print a formatted summary table of results."""
    width = 60
    print("\n" + "=" * width)
    print(f"  REINVENT 4 Baseline Results - {dataset.upper()}")
    print("=" * width)

    print("\n--- Generation Metrics ---")
    for key, val in metrics.items():
        print(f"  {key:<30s} {val}")

    print("\n--- Property Statistics ---")
    for key, val in properties.items():
        print(f"  {key:<30s} {val}")

    print("=" * width + "\n")


# ============================================================================
# Main Pipeline
# ============================================================================

def run_baseline(args):
    """Execute the full REINVENT 4 baseline pipeline."""

    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    dataset = args.dataset
    device = args.device

    logger.info(f"REINVENT 4 Baseline for dataset: {dataset}")
    logger.info(f"Output directory: {output_dir}")

    # Load reference data
    if args.data_path:
        data_path = os.path.abspath(args.data_path)
        logger.info(f"Dataset path: {data_path}")
        ref_df = pd.read_csv(data_path)

        smiles_col = None
        for col in ["SMILES", "smiles", "Smiles"]:
            if col in ref_df.columns:
                smiles_col = col
                break
        reference_smiles = ref_df[smiles_col].tolist() if smiles_col else []
    else:
        data_path = None
        reference_smiles = []

    # ------------------------------------------------------------------
    # Evaluate-only mode
    # ------------------------------------------------------------------
    if args.evaluate_only:
        if not args.generated_smiles:
            logger.error("--evaluate_only requires --generated_smiles")
            sys.exit(1)

        gen_df = pd.read_csv(args.generated_smiles)
        if "SMILES" in gen_df.columns:
            generated = gen_df["SMILES"].tolist()
        else:
            generated = gen_df.iloc[:, 0].tolist()

        logger.info(f"Evaluating {len(generated)} generated molecules")

        metrics = compute_generation_metrics(generated, reference_smiles)
        props_df = compute_molecule_properties(generated)
        prop_stats = compute_property_statistics(props_df)

        props_df.to_csv(os.path.join(output_dir, "reinvent4_properties.csv"), index=False)

        results = {"metrics": metrics, "properties": prop_stats, "dataset": dataset}
        with open(os.path.join(output_dir, "reinvent4_results.json"), "w") as f:
            json.dump(results, f, indent=2)

        print_summary_table(metrics, prop_stats, dataset)
        return

    # ------------------------------------------------------------------
    # Check REINVENT 4 availability
    # ------------------------------------------------------------------
    reinvent_available = False
    try:
        import reinvent
        logger.info(f"REINVENT 4 version: {reinvent.__version__}")
        reinvent_available = True
    except ImportError as e:
        logger.warning(f"Cannot import reinvent: {e}")
        logger.warning("REINVENT 4 is not installed. Install with:")
        logger.warning(f"  cd {REINVENT4_DIR} && pip install -e . --no-deps")

    prior_file = find_prior_model()
    if prior_file:
        logger.info(f"Prior model found: {prior_file}")
    else:
        logger.warning("Prior model not found!")
        logger.warning("Download from: https://zenodo.org/records/15641296")
        logger.warning(f"Place in: {REINVENT4_DIR / 'priors' / 'reinvent.prior'}")

    can_run = reinvent_available and prior_file is not None

    if not can_run:
        logger.warning("=" * 50)
        logger.warning("REINVENT 4 cannot run (missing install or prior model).")
        logger.warning("Generating configs and evaluation pipeline only.")
        logger.warning("=" * 50)

    # ------------------------------------------------------------------
    # Step 1: Prepare input data
    # ------------------------------------------------------------------
    if data_path:
        smiles_file = os.path.join(output_dir, f"{dataset}_actives.smi")
        prepare_smiles_file(data_path, smiles_file, max_smiles=args.max_reference)
        ref_actives = get_reference_actives(data_path, top_n=5)
    else:
        smiles_file = None
        ref_actives = []

    # ------------------------------------------------------------------
    # Step 2: Create configs
    # ------------------------------------------------------------------
    if prior_file:
        # Transfer learning config
        tl_model_file = os.path.join(output_dir, f"{dataset}_tl.model")
        tl_config = create_tl_config(
            prior_file=prior_file,
            smiles_file=smiles_file or "",
            output_model_file=tl_model_file,
            output_dir=output_dir,
            num_epochs=args.tl_epochs,
            device=device,
        )

        # RL config (uses prior as starting point; after TL, use tl_model)
        agent_file = prior_file  # Start from prior; change to tl_model_file after TL
        rl_config = create_rl_config(
            dataset=dataset,
            prior_file=prior_file,
            agent_file=agent_file,
            output_dir=output_dir,
            reference_smiles=ref_actives,
            device=device,
            batch_size=args.batch_size,
            rl_steps=args.rl_steps,
        )

        # Sampling config
        sampling_output = os.path.join(output_dir, "sampling_results.csv")
        sampling_config = create_sampling_config(
            model_file=prior_file,  # Will be updated to checkpoint after RL
            output_file=sampling_output,
            output_dir=output_dir,
            num_smiles=args.num_samples,
            device=device,
        )
    else:
        tl_config = rl_config = sampling_config = None

    # ------------------------------------------------------------------
    # Step 3: Run REINVENT 4 (if available)
    # ------------------------------------------------------------------
    generated_smiles = []

    if can_run and not args.config_only:
        start_time = time.time()

        # Change working directory for REINVENT (it uses relative paths)
        original_cwd = os.getcwd()
        os.chdir(output_dir)

        try:
            # Optional: Transfer Learning
            if args.run_tl and smiles_file:
                logger.info("=" * 40)
                logger.info("Running Transfer Learning...")
                logger.info("=" * 40)
                tl_log = os.path.join(output_dir, "reinvent4_tl.log")
                tl_success = run_reinvent4(tl_config, tl_log, device)

                if tl_success and os.path.exists(tl_model_file):
                    # Update RL config to use TL model as agent
                    logger.info("TL complete. Updating RL config to use TL model.")
                    rl_config = create_rl_config(
                        dataset=dataset,
                        prior_file=prior_file,
                        agent_file=tl_model_file,
                        output_dir=output_dir,
                        reference_smiles=ref_actives,
                        device=device,
                        batch_size=args.batch_size,
                        rl_steps=args.rl_steps,
                    )

            # Reinforcement Learning (staged learning)
            logger.info("=" * 40)
            logger.info("Running Reinforcement Learning...")
            logger.info("=" * 40)
            rl_log = os.path.join(output_dir, "reinvent4_rl.log")
            rl_success = run_reinvent4(rl_config, rl_log, device)

            if rl_success:
                # Find the final checkpoint
                chkpts = sorted(Path(output_dir).glob(f"{dataset}_stage*.chkpt"))
                if chkpts:
                    final_model = str(chkpts[-1])
                    logger.info(f"Using final checkpoint: {final_model}")

                    # Update sampling config to use trained model
                    sampling_config = create_sampling_config(
                        model_file=final_model,
                        output_file=sampling_output,
                        output_dir=output_dir,
                        num_smiles=args.num_samples,
                        device=device,
                    )

                    # Sample from trained model
                    logger.info("=" * 40)
                    logger.info("Sampling from trained model...")
                    logger.info("=" * 40)
                    sampling_log = os.path.join(output_dir, "reinvent4_sampling.log")
                    run_reinvent4(sampling_config, sampling_log, device)

            # Collect all generated SMILES
            generated_smiles = collect_generated_smiles(output_dir, f"{dataset}_rl")

        finally:
            os.chdir(original_cwd)

        elapsed = time.time() - start_time
        logger.info(f"Total REINVENT 4 runtime: {elapsed:.1f}s ({elapsed/60:.1f}min)")

    # ------------------------------------------------------------------
    # Step 4: Evaluate results
    # ------------------------------------------------------------------
    if generated_smiles:
        logger.info(f"Collected {len(generated_smiles)} unique generated molecules")

        metrics = compute_generation_metrics(generated_smiles, reference_smiles)
        props_df = compute_molecule_properties(generated_smiles)
        prop_stats = compute_property_statistics(props_df)

        # Save results
        props_df.to_csv(os.path.join(output_dir, "reinvent4_properties.csv"), index=False)

        gen_df = pd.DataFrame({"SMILES": generated_smiles})
        gen_df.to_csv(os.path.join(output_dir, "reinvent4_generated.csv"), index=False)

        results = {
            "metrics": metrics,
            "properties": prop_stats,
            "dataset": dataset,
            "model": "REINVENT 4",
            "version": None,
            "config": {
                "rl_steps": args.rl_steps,
                "batch_size": args.batch_size,
                "num_samples": args.num_samples,
                "device": device,
            },
        }
        try:
            import reinvent
            results["version"] = reinvent.__version__
        except ImportError:
            pass

        with open(os.path.join(output_dir, "reinvent4_results.json"), "w") as f:
            json.dump(results, f, indent=2)

        print_summary_table(metrics, prop_stats, dataset)
    else:
        logger.info("No generated molecules to evaluate.")
        if not can_run:
            logger.info("Configs have been created. To run the full pipeline:")
            logger.info("  1. Install REINVENT 4: bash setup_reinvent4.sh")
            logger.info("  2. Download prior model")
            logger.info("  3. Re-run this script")

    # Print config file locations
    logger.info("Generated configuration files:")
    for f in sorted(Path(output_dir).glob("*.toml")):
        logger.info(f"  {f}")


# ============================================================================
# CLI
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="REINVENT 4 Baseline Runner for Theranostics Drug Discovery",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="scd1",
        choices=["scd1", "nk1r", "drd2", "fads2", "custom"],
        help="Target dataset name",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="Path to dataset CSV file (SMILES, pIC50 columns)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results/reinvent4",
        help="Output directory for results",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="PyTorch device (cuda:0, cpu)",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1000,
        help="Number of molecules to generate in final sampling",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Batch size for RL",
    )
    parser.add_argument(
        "--rl_steps",
        type=int,
        default=300,
        help="Maximum RL steps per stage",
    )
    parser.add_argument(
        "--tl_epochs",
        type=int,
        default=50,
        help="Number of transfer learning epochs",
    )
    parser.add_argument(
        "--max_reference",
        type=int,
        default=500,
        help="Max reference SMILES for transfer learning",
    )
    parser.add_argument(
        "--run_tl",
        action="store_true",
        default=False,
        help="Run transfer learning before RL",
    )
    parser.add_argument(
        "--config_only",
        action="store_true",
        default=False,
        help="Only generate config files, do not run REINVENT 4",
    )
    parser.add_argument(
        "--evaluate_only",
        action="store_true",
        default=False,
        help="Only evaluate previously generated molecules",
    )
    parser.add_argument(
        "--generated_smiles",
        type=str,
        default=None,
        help="Path to CSV with generated SMILES (for --evaluate_only)",
    )

    return parser.parse_args()


def resolve_default_data_path(dataset: str) -> Optional[str]:
    """Resolve the default data path for a known dataset."""
    data_dir = PROJECT_DIR / "data"
    old_data_dir = PROJECT_DIR.parent / "theranostics_old" / "data"

    dataset_map = {
        "scd1": [
            old_data_dir / "scd1_binding.csv",
            data_dir / "fatty_acid_desaturase_bioactivity.csv",
        ],
        "nk1r": [
            data_dir / "nk1r_bioactivity.csv",
        ],
        "drd2": [
            data_dir / "drd2_bioactivity.csv",
        ],
        "fads2": [
            data_dir / "fads2_bioactivity.csv",
            data_dir / "fatty_acid_desaturase_bioactivity.csv",
        ],
    }

    if dataset in dataset_map:
        for path in dataset_map[dataset]:
            if path.exists():
                return str(path)

    return None


if __name__ == "__main__":
    args = parse_args()

    # Resolve default data path if not provided
    if args.data_path is None and args.dataset != "custom" and not args.evaluate_only:
        default_path = resolve_default_data_path(args.dataset)
        if default_path:
            args.data_path = default_path
            logger.info(f"Using default data path: {args.data_path}")
        else:
            logger.warning(f"No default data path found for dataset '{args.dataset}'")
            logger.warning("Please provide --data_path explicitly")

    run_baseline(args)
