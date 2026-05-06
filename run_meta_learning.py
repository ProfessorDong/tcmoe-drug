#!/usr/bin/env python
"""
Meta-Learning Experiment
=========================
Train meta-model on source targets (DRD2 + NK1R), adapt to held-out target
(SCD-1) with k-shot (k=5,10,20,50).  Compare against training from scratch
and simple transfer learning.

Usage:
    python run_meta_learning.py --device cuda --epochs 30
    python run_meta_learning.py --held_out scd1 --quick
"""

import os
import sys
import copy
import json
import argparse
import logging
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from rdkit import DataStructs

RDLogger.DisableLog('rdApp.*')

# Framework imports
from models.property_predictor import PropertyPredictor
from meta_learning.maml import MAML, MAMLTrainer
from utils.training import EarlyStopping

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TARGET_PATHS = {
    'scd1': os.path.normpath(
        os.path.join(BASE_DIR, 'data', 'scd1_binding.csv')
    ),
    'nk1r': os.path.join(BASE_DIR, 'data', 'nk1r_combined.csv'),
    'drd2': os.path.join(BASE_DIR, 'data', 'drd2_bioactivity.csv'),
    'fads2': os.path.join(BASE_DIR, 'data', 'fads2_bioactivity.csv'),
    'fads2_expanded': os.path.join(BASE_DIR, 'data', 'fatty_acid_desaturase_bioactivity.csv'),
}


def setup_logging(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(output_dir, 'meta_learning.log')),
            logging.StreamHandler(sys.stdout),
        ],
    )
    return logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def _fp_from_smiles(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
    arr = np.zeros(1024, dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def load_target(target):
    path = TARGET_PATHS[target]
    df = pd.read_csv(path)
    smiles = df['SMILES'].tolist()
    activity = df.iloc[:, 1].values.astype(np.float32)
    X, y, smi_list = [], [], []
    for s, a in zip(smiles, activity):
        if np.isfinite(a):
            fp = _fp_from_smiles(s)
            if fp is not None:
                X.append(fp)
                y.append(a)
                smi_list.append(s)
    return np.array(X), np.array(y, dtype=np.float32), smi_list


def scaffold_support_query_split(smiles_list, k, seed=42):
    """Split held-out data into support (k) and query using Murcko scaffolds.
    Ensures no scaffold overlap between support and query sets."""
    from rdkit.Chem.Scaffolds import MurckoScaffold
    import random as rng_mod

    scaffolds = {}
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        try:
            scaf = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
        except Exception:
            scaf = ''
        scaffolds.setdefault(scaf, []).append(i)

    groups = sorted(scaffolds.values(), key=len, reverse=True)
    rng = rng_mod.Random(seed)
    # Shuffle groups for randomness across trials (different seed per trial)
    rng.shuffle(groups)

    support_idx, query_idx = [], []
    for group in groups:
        if len(support_idx) < k:
            support_idx.extend(group)
        else:
            query_idx.extend(group)

    # Trim support to exactly k if we overshot due to group sizes
    if len(support_idx) > k:
        overflow = support_idx[k:]
        support_idx = support_idx[:k]
        query_idx = overflow + query_idx

    # Cap query at 100 for efficiency
    if len(query_idx) > 100:
        rng.shuffle(query_idx)
        query_idx = query_idx[:100]

    return np.array(support_idx), np.array(query_idx)


# ---------------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------------

def _make_predictor(device):
    task_configs = {'pIC50': {'type': 'regression'}}
    model = PropertyPredictor(
        input_dim=1024,
        task_configs=task_configs,
        hidden_dims=[512, 256],
        dropout=0.3,
    ).to(device)
    return model


# ---------------------------------------------------------------------------
# Approach 1: Meta-learning (MAML) on source, adapt to held-out
# ---------------------------------------------------------------------------

def run_maml(source_data, held_out_data, k_shots, n_meta_epochs, device, logger,
             inner_lr=0.01, meta_lr=1e-3, inner_steps=5, quick=False,
             held_out_smiles=None):
    """Train MAML on source tasks, then adapt to held-out with different k."""
    logger.info('-' * 50)
    logger.info('Approach: MAML')
    logger.info('-' * 50)

    base_model = _make_predictor(device)
    maml = MAML(base_model, inner_lr=inner_lr, first_order=True)
    maml_trainer = MAMLTrainer(
        maml, meta_lr=meta_lr, inner_lr=inner_lr, inner_steps=inner_steps,
    )

    # Meta-training
    for epoch in range(n_meta_epochs):
        # Build tasks from source targets
        tasks = []
        for name, (X, y) in source_data.items():
            n = len(X)
            k = min(50, n // 3)
            perm = np.random.permutation(n)
            sx = torch.tensor(X[perm[:k]], dtype=torch.float32, device=device)
            sy = torch.tensor(y[perm[:k]], dtype=torch.float32, device=device)
            qx = torch.tensor(X[perm[k:k*2]], dtype=torch.float32, device=device)
            qy = torch.tensor(y[perm[k:k*2]], dtype=torch.float32, device=device)
            tasks.append({
                'support': (sx, sy), 'query': (qx, qy), 'task_name': 'pIC50',
            })

        if not tasks:
            break
        loss = maml_trainer._meta_step(tasks)
        if (epoch + 1) % max(n_meta_epochs // 10, 1) == 0:
            logger.info(f'MAML meta-epoch {epoch+1}/{n_meta_epochs}  loss={loss:.4f}')

    # Adapt to held-out with different k (scaffold-based support/query split)
    X_ho, y_ho = held_out_data
    results = {}
    for k in k_shots:
        rmses = []
        n_trials = 1 if quick else 5
        for trial in range(n_trials):
            if held_out_smiles is not None:
                support_idx, query_idx = scaffold_support_query_split(
                    held_out_smiles, k, seed=trial * 1000 + k)
            else:
                perm = np.random.permutation(len(X_ho))
                support_idx = perm[:k]
                query_idx = perm[k:k + min(100, len(X_ho) - k)]

            sx = torch.tensor(X_ho[support_idx], dtype=torch.float32, device=device)
            sy = torch.tensor(y_ho[support_idx], dtype=torch.float32, device=device)
            qx = torch.tensor(X_ho[query_idx], dtype=torch.float32, device=device)
            qy = torch.tensor(y_ho[query_idx], dtype=torch.float32, device=device)

            preds = maml.meta_forward(sx, sy, qx, num_steps=inner_steps, task_name='pIC50')
            rmse = float(torch.sqrt(F.mse_loss(preds.squeeze().detach(), qy)).item())
            rmses.append(rmse)

        mean_rmse = float(np.mean(rmses))
        std_rmse = float(np.std(rmses))
        results[k] = {'mean_rmse': mean_rmse, 'std_rmse': std_rmse}
        logger.info(f'  k={k}: RMSE = {mean_rmse:.4f} +/- {std_rmse:.4f}')

    return results


# ---------------------------------------------------------------------------
# Approach 2: Transfer learning (pretrain on source, fine-tune on held-out)
# ---------------------------------------------------------------------------

def run_transfer(source_data, held_out_data, k_shots, n_pretrain_epochs,
                 device, logger, lr=1e-3, quick=False, held_out_smiles=None):
    logger.info('-' * 50)
    logger.info('Approach: Transfer Learning')
    logger.info('-' * 50)

    # Pretrain on combined source data
    X_src = np.concatenate([d[0] for d in source_data.values()], axis=0)
    y_src = np.concatenate([d[1] for d in source_data.values()], axis=0)

    model = _make_predictor(device)
    optimizer = Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    ds = TensorDataset(
        torch.tensor(X_src, dtype=torch.float32),
        torch.tensor(y_src, dtype=torch.float32),
    )
    loader = DataLoader(ds, batch_size=64, shuffle=True)

    model.train()
    for epoch in range(n_pretrain_epochs):
        total_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb, task_name='pIC50')['pIC50'].squeeze()
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch + 1) % max(n_pretrain_epochs // 5, 1) == 0:
            logger.info(f'Transfer pretrain epoch {epoch+1}  loss={total_loss/len(loader):.4f}')

    # Fine-tune on held-out k-shot (scaffold-based support/query split)
    X_ho, y_ho = held_out_data
    results = {}
    for k in k_shots:
        rmses = []
        n_trials = 1 if quick else 5
        for trial in range(n_trials):
            if held_out_smiles is not None:
                support_idx, query_idx = scaffold_support_query_split(
                    held_out_smiles, k, seed=trial * 1000 + k)
            else:
                perm = np.random.permutation(len(X_ho))
                support_idx = perm[:k]
                query_idx = perm[k:k + min(100, len(X_ho) - k)]

            ft_model = copy.deepcopy(model)
            ft_opt = Adam(ft_model.parameters(), lr=1e-3)
            ft_model.train()

            sx = torch.tensor(X_ho[support_idx], dtype=torch.float32, device=device)
            sy = torch.tensor(y_ho[support_idx], dtype=torch.float32, device=device)

            for _ in range(50):
                ft_opt.zero_grad()
                pred = ft_model(sx, task_name='pIC50')['pIC50'].squeeze()
                loss = criterion(pred, sy)
                loss.backward()
                ft_opt.step()

            ft_model.eval()
            qx = torch.tensor(X_ho[query_idx], dtype=torch.float32, device=device)
            qy = torch.tensor(y_ho[query_idx], dtype=torch.float32, device=device)
            with torch.no_grad():
                preds = ft_model(qx, task_name='pIC50')['pIC50'].squeeze()
                rmse = float(torch.sqrt(F.mse_loss(preds, qy)).item())
                rmses.append(rmse)

        mean_rmse = float(np.mean(rmses))
        std_rmse = float(np.std(rmses))
        results[k] = {'mean_rmse': mean_rmse, 'std_rmse': std_rmse}
        logger.info(f'  k={k}: RMSE = {mean_rmse:.4f} +/- {std_rmse:.4f}')

    return results


# ---------------------------------------------------------------------------
# Approach 3: Train from scratch on held-out (k-shot)
# ---------------------------------------------------------------------------

def run_scratch(held_out_data, k_shots, device, logger, lr=1e-3, quick=False,
                held_out_smiles=None):
    logger.info('-' * 50)
    logger.info('Approach: Train from Scratch')
    logger.info('-' * 50)

    criterion = nn.MSELoss()
    X_ho, y_ho = held_out_data
    results = {}

    for k in k_shots:
        rmses = []
        n_trials = 1 if quick else 5
        for trial in range(n_trials):
            if held_out_smiles is not None:
                support_idx, query_idx = scaffold_support_query_split(
                    held_out_smiles, k, seed=trial * 1000 + k)
            else:
                perm = np.random.permutation(len(X_ho))
                support_idx = perm[:k]
                query_idx = perm[k:k + min(100, len(X_ho) - k)]

            model = _make_predictor(device)
            optimizer = Adam(model.parameters(), lr=lr)
            model.train()

            sx = torch.tensor(X_ho[support_idx], dtype=torch.float32, device=device)
            sy = torch.tensor(y_ho[support_idx], dtype=torch.float32, device=device)

            for _ in range(100):
                optimizer.zero_grad()
                pred = model(sx, task_name='pIC50')['pIC50'].squeeze()
                loss = criterion(pred, sy)
                loss.backward()
                optimizer.step()

            model.eval()
            qx = torch.tensor(X_ho[query_idx], dtype=torch.float32, device=device)
            qy = torch.tensor(y_ho[query_idx], dtype=torch.float32, device=device)
            with torch.no_grad():
                preds = model(qx, task_name='pIC50')['pIC50'].squeeze()
                rmse = float(torch.sqrt(F.mse_loss(preds, qy)).item())
                rmses.append(rmse)

        mean_rmse = float(np.mean(rmses))
        std_rmse = float(np.std(rmses))
        results[k] = {'mean_rmse': mean_rmse, 'std_rmse': std_rmse}
        logger.info(f'  k={k}: RMSE = {mean_rmse:.4f} +/- {std_rmse:.4f}')

    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_comparison(maml_results, transfer_results, scratch_results, k_shots, output_dir):
    fig, ax = plt.subplots(figsize=(10, 6))

    for label, results, color, marker in [
        ('MAML', maml_results, 'royalblue', 'o'),
        ('Transfer', transfer_results, 'darkorange', 's'),
        ('Scratch', scratch_results, 'seagreen', '^'),
    ]:
        means = [results[k]['mean_rmse'] for k in k_shots]
        stds = [results[k]['std_rmse'] for k in k_shots]
        ax.errorbar(k_shots, means, yerr=stds, label=label, marker=marker,
                    color=color, capsize=4, linewidth=2, markersize=8)

    ax.set_xlabel('k (number of support samples)', fontsize=13)
    ax.set_ylabel('RMSE (pIC50)', fontsize=13)
    ax.set_title('Few-Shot Adaptation: MAML vs Transfer vs Scratch', fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(output_dir, 'meta_learning_comparison.png')
    plt.savefig(path, dpi=150)
    plt.close()
    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description='Meta-Learning Experiment')
    parser.add_argument('--held_out', type=str, default='scd1',
                        choices=['scd1', 'nk1r', 'drd2', 'fads2', 'fads2_expanded'],
                        help='Held-out target for adaptation')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Meta-training epochs')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--quick', action='store_true',
                        help='Quick smoke test')
    return parser.parse_args()


def main():
    args = parse_args()

    if args.device == 'cuda' and not torch.cuda.is_available():
        args.device = 'cpu'

    if args.output_dir is None:
        args.output_dir = os.path.join(BASE_DIR, 'outputs', f'meta_learning_{args.held_out}')
    os.makedirs(args.output_dir, exist_ok=True)

    logger = setup_logging(args.output_dir)
    logger.info(f'Arguments: {vars(args)}')
    logger.info(f'Device: {args.device}')

    # Load all targets (X=fingerprints, y=activity, smi=SMILES for scaffold split)
    all_data = {}
    all_smiles = {}
    for t, path in TARGET_PATHS.items():
        if os.path.exists(path):
            X, y, smi = load_target(t)
            all_data[t] = (X, y)
            all_smiles[t] = smi
            logger.info(f'Loaded {t}: {X.shape[0]} compounds')
        else:
            logger.warning(f'Data file not found for {t}: {path}')

    if args.held_out not in all_data:
        logger.error(f'Held-out target {args.held_out} has no data.')
        sys.exit(1)

    held_out_data = all_data[args.held_out]
    held_out_smi = all_smiles.get(args.held_out)
    # Source targets: exclude the held-out target and its variants (e.g., fads2/fads2_expanded)
    fads2_keys = {'fads2', 'fads2_expanded'}
    if args.held_out in fads2_keys:
        exclude_keys = fads2_keys  # exclude both FADS2 variants from sources
    else:
        exclude_keys = {args.held_out}
    source_data = {k: v for k, v in all_data.items() if k not in exclude_keys}

    if not source_data:
        logger.error('No source targets available for meta-learning.')
        sys.exit(1)

    logger.info(f'Source targets: {list(source_data.keys())}')
    logger.info(f'Held-out target: {args.held_out}')

    n_held_out = len(held_out_data[0])
    # Use k values that leave at least 3 samples for query set
    all_k_shots = [5, 10, 20, 50]
    k_shots = [k for k in all_k_shots if k < n_held_out - 2]
    if not k_shots:
        logger.error(f'Held-out target has too few samples ({n_held_out}) for any k-shot setting.')
        sys.exit(1)
    logger.info(f'k-shot values: {k_shots} (held-out has {n_held_out} samples)')

    if args.quick:
        k_shots = [k for k in [5, 20] if k < n_held_out - 2]
        if not k_shots:
            k_shots = [k_shots[0]] if k_shots else [5]
        args.epochs = min(args.epochs, 5)

    t0 = time.time()

    # Run three approaches
    maml_results = run_maml(
        source_data, held_out_data, k_shots, args.epochs,
        args.device, logger, quick=args.quick,
        held_out_smiles=held_out_smi,
    )
    transfer_results = run_transfer(
        source_data, held_out_data, k_shots,
        n_pretrain_epochs=args.epochs, device=args.device, logger=logger,
        quick=args.quick, held_out_smiles=held_out_smi,
    )
    scratch_results = run_scratch(
        held_out_data, k_shots, args.device, logger, quick=args.quick,
        held_out_smiles=held_out_smi,
    )

    elapsed = time.time() - t0
    logger.info(f'Experiment completed in {elapsed:.1f}s')

    # Summary table
    logger.info('')
    logger.info('=' * 70)
    logger.info(f'{"k":>5s}  {"MAML RMSE":>15s}  {"Transfer RMSE":>15s}  {"Scratch RMSE":>15s}')
    logger.info('-' * 70)
    for k in k_shots:
        m = maml_results[k]
        t_ = transfer_results[k]
        s = scratch_results[k]
        logger.info(
            f'{k:>5d}  {m["mean_rmse"]:>6.4f}+/-{m["std_rmse"]:.4f}  '
            f'{t_["mean_rmse"]:>6.4f}+/-{t_["std_rmse"]:.4f}  '
            f'{s["mean_rmse"]:>6.4f}+/-{s["std_rmse"]:.4f}'
        )
    logger.info('=' * 70)

    # Save results
    all_results = {
        'held_out': args.held_out,
        'source_targets': list(source_data.keys()),
        'k_shots': k_shots,
        'maml': {str(k): v for k, v in maml_results.items()},
        'transfer': {str(k): v for k, v in transfer_results.items()},
        'scratch': {str(k): v for k, v in scratch_results.items()},
    }
    results_path = os.path.join(args.output_dir, 'meta_learning_results.json')
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    logger.info(f'Results saved to {results_path}')

    # Plot
    fig_path = plot_comparison(maml_results, transfer_results, scratch_results,
                               k_shots, args.output_dir)
    logger.info(f'Comparison figure: {fig_path}')


if __name__ == '__main__':
    main()
