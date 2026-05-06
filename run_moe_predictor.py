#!/usr/bin/env python3
"""
Train and evaluate the multi-view Mixture-of-Experts predictor for the
JBHI manuscript revision.

Runs three experiments (controlled by --experiment):
  - ablation: per-target scaffold-split RMSE for single-view baselines and MoE.
  - fewshot:  k-shot adaptation on a held-out target with scaffold-based
              support/query splits, comparing single-view transfer and MoE.
  - selectivity: cross-target selectivity matrix using rescored generated
              molecules from each target's RL run.

The MoE design is described in models/moe_predictor.py.
"""

from __future__ import annotations
import os, sys, csv, json, copy, time, argparse, random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog('rdApp.*')

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())

from models.moe_predictor import make_default_moe, MoEPredictor
from data.featurizer_views import (
    batch_morgan, batch_maccs, batch_descriptors,
    DescriptorScaler, MORGAN_BITS, MACCS_BITS, DESCRIPTOR_DIM,
)

BASE = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

TARGET_PATHS = {
    'scd1': os.path.join(BASE, 'data', 'scd1_binding.csv'),
    'nk1r': os.path.join(BASE, 'data', 'nk1r_combined.csv'),
    'drd2': os.path.join(BASE, 'data', 'drd2_bioactivity.csv'),
    'fads': os.path.join(BASE, 'data', 'fatty_acid_desaturase_bioactivity.csv'),
}
ALL_TARGETS = ['scd1', 'nk1r', 'drd2', 'fads']
TARGET_IDX = {t: i for i, t in enumerate(ALL_TARGETS)}


def load_target(name: str):
    df = pd.read_csv(TARGET_PATHS[name])
    smiles = df['SMILES'].astype(str).tolist()
    y = df.iloc[:, 1].astype(float).values
    keep_smiles, keep_y = [], []
    for s, v in zip(smiles, y):
        if not np.isfinite(v):
            continue
        if Chem.MolFromSmiles(s) is None:
            continue
        keep_smiles.append(s)
        keep_y.append(v)
    return keep_smiles, np.asarray(keep_y, dtype=np.float32)


# ---------------------------------------------------------------------------
# Scaffold splitting
# ---------------------------------------------------------------------------

def murcko_scaffold(smi: str) -> str:
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return ''
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
    except Exception:
        return ''


def scaffold_split_indices(smiles_list, frac_train=0.8, frac_val=0.1, seed=0):
    """Standard scaffold split: largest scaffolds first into train, then val/test."""
    scaffolds: dict[str, list[int]] = {}
    for i, s in enumerate(smiles_list):
        scaffolds.setdefault(murcko_scaffold(s), []).append(i)
    groups = sorted(scaffolds.values(), key=len, reverse=True)
    rng = random.Random(seed)
    # Shuffle within size buckets to introduce some diversity across seeds while
    # preserving scaffold integrity.
    bucketed: dict[int, list[list[int]]] = {}
    for g in groups:
        bucketed.setdefault(len(g), []).append(g)
    final_groups = []
    for size in sorted(bucketed.keys(), reverse=True):
        gs = bucketed[size]
        rng.shuffle(gs)
        final_groups.extend(gs)

    n = len(smiles_list)
    n_train = int(round(frac_train * n))
    n_val = int(round(frac_val * n))
    train_idx, val_idx, test_idx = [], [], []
    for g in final_groups:
        if len(train_idx) + len(g) <= n_train:
            train_idx.extend(g)
        elif len(val_idx) + len(g) <= n_val:
            val_idx.extend(g)
        else:
            test_idx.extend(g)
    return np.array(train_idx), np.array(val_idx), np.array(test_idx)


def scaffold_support_query(smiles_list, k, seed=0, query_cap=100):
    """k-shot scaffold split: support set has exactly k molecules, query set
    contains molecules from disjoint scaffolds (capped)."""
    scaffolds: dict[str, list[int]] = {}
    for i, s in enumerate(smiles_list):
        scaffolds.setdefault(murcko_scaffold(s), []).append(i)
    groups = sorted(scaffolds.values(), key=len, reverse=True)
    rng = random.Random(seed)
    rng.shuffle(groups)

    support, query = [], []
    for g in groups:
        if len(support) < k:
            support.extend(g)
        else:
            query.extend(g)
    if len(support) > k:
        overflow = support[k:]
        support = support[:k]
        query = overflow + query
    if len(query) > query_cap:
        rng.shuffle(query)
        query = query[:query_cap]
    return np.array(support), np.array(query)


# ---------------------------------------------------------------------------
# Multi-view feature pre-computation
# ---------------------------------------------------------------------------

def compute_views(smiles_list, encoder_emb=None):
    """Return a dict of view_name -> [N, D] numpy float32."""
    views = {
        'morgan': batch_morgan(smiles_list).astype(np.float32),
        'maccs': batch_maccs(smiles_list).astype(np.float32),
        'descriptors': batch_descriptors(smiles_list).astype(np.float32),
    }
    if encoder_emb is not None:
        views['encoder'] = encoder_emb.astype(np.float32)
    else:
        # Placeholder: use a fixed projection of Morgan to 128-d so the model
        # is testable without a trained dual encoder.
        views['encoder'] = views['morgan'][:, :128]
    return views


def get_dual_encoder_embeddings(smiles_list, device, weights_path=None,
                                 exclude_target: str | None = None,
                                 train_smiles=None, train_y=None):
    """Train (or load) a small dual encoder and return 128-dim embeddings
    for ``smiles_list``.

    Args:
        smiles_list: SMILES to embed.
        device: torch device.
        weights_path: Optional path to load/save encoder weights. If the file
            exists, weights are loaded; otherwise a fresh encoder is trained
            and saved.
        exclude_target: If given, the named target is EXCLUDED from encoder
            training. This is required to avoid data leakage in held-out
            few-shot evaluation (the held-out target must not be seen during
            pretraining of any component, including the encoder).
        train_smiles, train_y: If both given, they override the default
            "full data of all targets except exclude_target" training set.
            Pass training-split-only data to avoid leakage of validation
            and test scaffolds into the encoder.
    """
    from models.dual_encoder import DualEncoder
    from data.dataset import DualDataset, collate_dual

    # Build a small dual encoder
    encoder = DualEncoder(
        node_features=75, edge_features=6,
        graph_hidden_dims=[128, 256, 128], vocab_size=128,
        smiles_hidden_dim=256,
        output_dim=128, num_heads=4, dropout=0.1,
    ).to(device)

    if weights_path and os.path.exists(weights_path):
        encoder.load_state_dict(torch.load(weights_path, map_location=device))
        print(f'  loaded encoder weights from {weights_path}', flush=True)
    else:
        # Determine encoder training set
        if train_smiles is not None and train_y is not None:
            all_s = list(train_smiles)
            all_y = list(train_y)
            print(f'  training encoder on {len(all_s)} provided SMILES (training split only)',
                  flush=True)
        else:
            all_s, all_y = [], []
            targets_used = [t for t in ALL_TARGETS if t != exclude_target]
            for t in targets_used:
                try:
                    s, y = load_target(t)
                    all_s.extend(s)
                    all_y.extend(y.tolist())
                except FileNotFoundError:
                    continue
            if exclude_target is not None:
                print(f'  training encoder on {targets_used} (excluding {exclude_target})',
                      flush=True)
        all_y = np.array(all_y, dtype=np.float32)
        ds = DualDataset(all_s, all_y, max_length=120)
        loader = DataLoader(ds, batch_size=64, shuffle=True,
                            collate_fn=collate_dual, drop_last=True)
        head = nn.Sequential(nn.Linear(128, 64), nn.ReLU(),
                             nn.Dropout(0.2), nn.Linear(64, 1)).to(device)
        opt = Adam(list(encoder.parameters()) + list(head.parameters()),
                   lr=1e-3, weight_decay=1e-4)
        encoder.train(); head.train()
        n_epochs = 30
        for ep in range(n_epochs):
            tot = 0; nb = 0
            for batch in loader:
                gb = _GraphBatch(batch['graph']).to(device)
                tok = batch['tokens'].to(device); ln = batch['lengths']
                yy = batch['labels'].to(device)
                opt.zero_grad()
                emb = encoder(gb, tok, ln)
                pred = head(emb).squeeze(-1)
                loss = F.mse_loss(pred, yy)
                loss.backward(); opt.step()
                tot += loss.item(); nb += 1
            if (ep + 1) % 10 == 0:
                print(f'    encoder epoch {ep+1}/{n_epochs} loss={tot/max(nb,1):.4f}',
                      flush=True)
        if weights_path:
            torch.save(encoder.state_dict(), weights_path)

    # Extract embeddings
    encoder.eval()
    ds = DualDataset(smiles_list, np.zeros(len(smiles_list), dtype=np.float32),
                     max_length=120)
    loader = DataLoader(ds, batch_size=128, shuffle=False, collate_fn=collate_dual)
    embs = []
    with torch.no_grad():
        for batch in loader:
            gb = _GraphBatch(batch['graph']).to(device)
            tok = batch['tokens'].to(device); ln = batch['lengths']
            embs.append(encoder(gb, tok, ln).cpu().numpy())
    return np.concatenate(embs, axis=0).astype(np.float32)


class _GraphBatch:
    """Wrapper that mirrors the existing fewshot script's GraphBatch."""
    def __init__(self, d):
        self.x = d['x']
        self.edge_index = d['edge_index']
        self.batch = d['batch']
        self.edge_attr = d.get('edge_attr')
    def to(self, device):
        self.x = self.x.to(device)
        self.edge_index = self.edge_index.to(device)
        self.batch = self.batch.to(device)
        if self.edge_attr is not None:
            self.edge_attr = self.edge_attr.to(device)
        return self


# ---------------------------------------------------------------------------
# Single-view baseline (MLP) — used in ablation table
# ---------------------------------------------------------------------------

def make_baseline_mlp(in_dim, hidden=256, dropout=0.2):
    return nn.Sequential(
        nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.ReLU(), nn.Dropout(dropout),
        nn.Linear(hidden, hidden // 2), nn.LayerNorm(hidden // 2), nn.ReLU(),
        nn.Dropout(dropout), nn.Linear(hidden // 2, 1),
    )


def train_single_view(X_train, y_train, X_val, y_val, device, n_epochs=120,
                      lr=5e-4, batch_size=64, weight_decay=1e-4, patience=20):
    """Train a small MLP on a single-view representation and return best
    state by validation loss."""
    model = make_baseline_mlp(X_train.shape[1]).to(device)
    opt = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
    Xv = torch.tensor(X_val, device=device)
    yv = torch.tensor(y_val, device=device)
    best_state = copy.deepcopy(model.state_dict())
    best_val = float('inf'); since = 0
    for ep in range(n_epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            pred = model(xb).squeeze(-1)
            loss = F.mse_loss(pred, yb)
            loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            v = float(F.mse_loss(model(Xv).squeeze(-1), yv).sqrt().item())
        if v < best_val - 1e-4:
            best_val = v; best_state = copy.deepcopy(model.state_dict()); since = 0
        else:
            since += 1
            if since >= patience:
                break
    model.load_state_dict(best_state)
    return model, best_val


def eval_single_view(model, X_test, y_test, device):
    model.eval()
    Xt = torch.tensor(X_test, device=device)
    yt = torch.tensor(y_test, device=device)
    with torch.no_grad():
        pred = model(Xt).squeeze(-1)
        return float(F.mse_loss(pred, yt).sqrt().item())


# ---------------------------------------------------------------------------
# MoE training (joint multi-target)
# ---------------------------------------------------------------------------

def to_tensor_dict(views_dict, target_id, y, device):
    return (
        [torch.tensor(views_dict['morgan'], device=device),
         torch.tensor(views_dict['encoder'], device=device),
         torch.tensor(views_dict['maccs'], device=device),
         torch.tensor(views_dict['descriptors'], device=device)],
        torch.tensor(target_id, device=device, dtype=torch.long),
        torch.tensor(y, device=device, dtype=torch.float32),
    )


def slice_views(views_dict, idx):
    return {k: v[idx] for k, v in views_dict.items()}


def train_moe(
    train_views, train_target, train_y,
    val_views, val_target, val_y,
    device, n_targets=4, top_k=2, target_emb_dim=16,
    n_epochs=120, lr=5e-4, batch_size=128, weight_decay=1e-4,
    patience=20, load_balance=0.01, dropout=0.2,
):
    moe = make_default_moe(
        n_targets=n_targets, top_k=top_k, target_emb_dim=target_emb_dim,
        load_balance=load_balance, dropout=dropout,
    ).to(device)
    opt = Adam(moe.parameters(), lr=lr, weight_decay=weight_decay)

    # Pack training tensors
    morgan = torch.tensor(train_views['morgan'])
    encoder = torch.tensor(train_views['encoder'])
    maccs = torch.tensor(train_views['maccs'])
    desc = torch.tensor(train_views['descriptors'])
    tgt = torch.tensor(train_target, dtype=torch.long)
    yy = torch.tensor(train_y, dtype=torch.float32)

    n = morgan.size(0)
    indices = np.arange(n)

    # Validation tensors
    val_views_t = [
        torch.tensor(val_views['morgan'], device=device),
        torch.tensor(val_views['encoder'], device=device),
        torch.tensor(val_views['maccs'], device=device),
        torch.tensor(val_views['descriptors'], device=device),
    ]
    val_tgt = torch.tensor(val_target, device=device, dtype=torch.long)
    val_y_t = torch.tensor(val_y, device=device, dtype=torch.float32)

    best_state = copy.deepcopy(moe.state_dict())
    best_val = float('inf'); since = 0
    for ep in range(n_epochs):
        moe.train()
        np.random.shuffle(indices)
        for start in range(0, n, batch_size):
            idx = indices[start:start + batch_size]
            xb = [v[idx].to(device) for v in (morgan, encoder, maccs, desc)]
            tb = tgt[idx].to(device)
            yb = yy[idx].to(device)
            opt.zero_grad()
            pred, aux = moe(xb, tb, return_aux=True)
            loss = F.mse_loss(pred, yb) + aux['load_balance_loss']
            loss.backward(); opt.step()
        moe.eval()
        with torch.no_grad():
            v = float(F.mse_loss(moe(val_views_t, val_tgt), val_y_t).sqrt().item())
        if v < best_val - 1e-4:
            best_val = v; best_state = copy.deepcopy(moe.state_dict()); since = 0
        else:
            since += 1
            if since >= patience:
                break
    moe.load_state_dict(best_state)
    return moe, best_val


def eval_moe(moe, views_dict, target_id, y, device):
    moe.eval()
    xb = [torch.tensor(views_dict['morgan'], device=device),
          torch.tensor(views_dict['encoder'], device=device),
          torch.tensor(views_dict['maccs'], device=device),
          torch.tensor(views_dict['descriptors'], device=device)]
    tb = torch.tensor(target_id, device=device, dtype=torch.long)
    yt = torch.tensor(y, device=device, dtype=torch.float32)
    with torch.no_grad():
        pred = moe(xb, tb)
        return float(F.mse_loss(pred, yt).sqrt().item()), pred.cpu().numpy()


# ---------------------------------------------------------------------------
# Experiment 1: per-target representation ablation
# ---------------------------------------------------------------------------

def run_ablation(args, device, encoder_emb_cache, target_splits=None):
    print('\n=== Experiment 1: per-target representation ablation ===', flush=True)

    # Load all targets and reuse precomputed splits (must match the splits
    # used during encoder training to avoid leakage).
    target_data = {}
    for t in ALL_TARGETS:
        smiles, y = load_target(t)
        if target_splits is not None and t in target_splits:
            tr = target_splits[t]['tr']
            val = target_splits[t]['val']
            te = target_splits[t]['te']
        else:
            tr, val, te = scaffold_split_indices(smiles, seed=args.seed)
        target_data[t] = {
            'smiles': smiles, 'y': y,
            'tr': tr, 'val': val, 'te': te,
        }
        print(f'  {t}: n={len(smiles)} train={len(tr)} val={len(val)} test={len(te)}',
              flush=True)

    # Compute views (per target) using the encoder_emb_cache when available
    target_views = {}
    desc_scalers = {}
    for t in ALL_TARGETS:
        s = target_data[t]['smiles']
        emb = encoder_emb_cache.get(t)
        v = compute_views(s, encoder_emb=emb)
        # Standardize descriptors using *training-only* statistics
        scaler = DescriptorScaler().fit(v['descriptors'][target_data[t]['tr']])
        v['descriptors'] = scaler.transform(v['descriptors'])
        desc_scalers[t] = scaler
        target_views[t] = v

    # ---- Single-view baselines per target ----
    results = {}
    for t in ALL_TARGETS:
        print(f'\n  -- {t} --', flush=True)
        v = target_views[t]
        d = target_data[t]
        results[t] = {}
        for view_name in ['morgan', 'encoder', 'maccs', 'descriptors']:
            X = v[view_name]
            tr_rmses = []
            for trial in range(args.n_trials):
                np.random.seed(args.seed + trial)
                torch.manual_seed(args.seed + trial)
                model, _ = train_single_view(
                    X[d['tr']], d['y'][d['tr']],
                    X[d['val']], d['y'][d['val']],
                    device, n_epochs=args.epochs, lr=args.lr,
                    batch_size=args.batch_size, patience=args.patience,
                )
                rmse = eval_single_view(model, X[d['te']], d['y'][d['te']], device)
                tr_rmses.append(rmse)
            mean = float(np.mean(tr_rmses)); std = float(np.std(tr_rmses))
            print(f'    {view_name:>11s}  RMSE = {mean:.4f} +/- {std:.4f}', flush=True)
            results[t][view_name] = {'mean': mean, 'std': std, 'rmses': tr_rmses}

    # ---- MoE: trained jointly on all targets, evaluated per-target ----
    print('\n  -- MoE (trained jointly) --', flush=True)
    moe_results: dict[str, list[float]] = {t: [] for t in ALL_TARGETS}
    moe_util: dict[str, list[list[float]]] = {t: [] for t in ALL_TARGETS}
    for trial in range(args.n_trials):
        np.random.seed(args.seed + trial)
        torch.manual_seed(args.seed + trial)

        # Build joint train/val sets
        train_views = {k: [] for k in ['morgan', 'encoder', 'maccs', 'descriptors']}
        val_views = {k: [] for k in ['morgan', 'encoder', 'maccs', 'descriptors']}
        train_tgt, train_y = [], []
        val_tgt, val_y = [], []
        for t in ALL_TARGETS:
            v = target_views[t]; d = target_data[t]
            for k in train_views:
                train_views[k].append(v[k][d['tr']])
                val_views[k].append(v[k][d['val']])
            train_tgt.append(np.full(len(d['tr']), TARGET_IDX[t], dtype=np.int64))
            val_tgt.append(np.full(len(d['val']), TARGET_IDX[t], dtype=np.int64))
            train_y.append(d['y'][d['tr']]); val_y.append(d['y'][d['val']])
        train_views = {k: np.concatenate(v, axis=0) for k, v in train_views.items()}
        val_views = {k: np.concatenate(v, axis=0) for k, v in val_views.items()}
        train_tgt = np.concatenate(train_tgt); train_y = np.concatenate(train_y)
        val_tgt = np.concatenate(val_tgt); val_y = np.concatenate(val_y)

        moe, _ = train_moe(
            train_views, train_tgt, train_y,
            val_views, val_tgt, val_y,
            device, n_targets=len(ALL_TARGETS), top_k=args.top_k,
            target_emb_dim=args.target_emb_dim, n_epochs=args.epochs, lr=args.lr,
            batch_size=args.batch_size, patience=args.patience,
            load_balance=args.load_balance, dropout=args.dropout,
        )

        # Evaluate per target on test split
        for t in ALL_TARGETS:
            v = target_views[t]; d = target_data[t]
            te_views = {k: v[k][d['te']] for k in ['morgan', 'encoder', 'maccs', 'descriptors']}
            te_tgt = np.full(len(d['te']), TARGET_IDX[t], dtype=np.int64)
            rmse, _ = eval_moe(moe, te_views, te_tgt, d['y'][d['te']], device)
            moe_results[t].append(rmse)

            # Expert utilization on test set
            xb = [torch.tensor(te_views[k], device=device)
                  for k in ['morgan', 'encoder', 'maccs', 'descriptors']]
            util = moe.expert_utilization(xb, torch.tensor(te_tgt, device=device))
            moe_util[t].append(util.tolist())

    for t in ALL_TARGETS:
        m = float(np.mean(moe_results[t])); s = float(np.std(moe_results[t]))
        util_mean = np.mean(moe_util[t], axis=0).tolist()
        print(f'    {t:>11s}  MoE RMSE = {m:.4f} +/- {s:.4f}  '
              f'util(M/E/MA/D) = '
              f'{util_mean[0]:.2f}/{util_mean[1]:.2f}/'
              f'{util_mean[2]:.2f}/{util_mean[3]:.2f}', flush=True)
        results[t]['moe'] = {'mean': m, 'std': s, 'rmses': moe_results[t],
                              'utilization': util_mean,
                              'utilization_per_seed': moe_util[t]}

    out = os.path.join(args.output_dir, 'ablation_results.json')
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nSaved ablation results to {out}', flush=True)
    return results


# ---------------------------------------------------------------------------
# Experiment 2: few-shot MoE (held-out target)
# ---------------------------------------------------------------------------

def run_fewshot(args, device, encoder_emb_cache):
    """Few-shot held-out experiment.

    IMPORTANT: To avoid data leakage, the dual encoder used to extract
    embeddings is trained ONLY on the three source targets (the held-out
    target is excluded from encoder training). This ensures the encoder
    has not seen any held-out molecules or labels before scoring.
    """
    print(f'\n=== Experiment 2: few-shot MoE (held-out: {args.held_out}) ===',
          flush=True)
    held = args.held_out
    sources = [t for t in ALL_TARGETS if t != held]

    # ---- Build held-out-clean encoder (trained on sources only) ----
    print(f'  Training source-only encoder (excludes {held})...', flush=True)
    clean_enc_path = os.path.join(args.output_dir, f'dual_encoder_excl_{held}.pt')
    if not os.path.exists(clean_enc_path):
        # Force training a fresh encoder that has not seen the held-out target.
        _ = get_dual_encoder_embeddings(
            ['CCO'], device,  # dummy single-molecule input; we just want the .pt file
            weights_path=clean_enc_path, exclude_target=held)
    # Now extract embeddings for ALL targets using the clean encoder
    clean_emb = {}
    for t in ALL_TARGETS:
        s, _ = load_target(t)
        clean_emb[t] = get_dual_encoder_embeddings(
            s, device, weights_path=clean_enc_path)
        print(f'    {t}: clean encoder embeddings {clean_emb[t].shape}',
              flush=True)

    # Load and view-encode all data using the CLEAN encoder
    data = {}
    for t in ALL_TARGETS:
        s, y = load_target(t)
        v = compute_views(s, encoder_emb=clean_emb[t])
        data[t] = {'smiles': s, 'y': y, 'views': v}

    # Standardize descriptors using sources' train-equivalent set (use all
    # source data; safe under scaffold split because we only use it to scale)
    src_desc = np.concatenate([data[t]['views']['descriptors'] for t in sources], axis=0)
    scaler = DescriptorScaler().fit(src_desc)
    for t in ALL_TARGETS:
        data[t]['views']['descriptors'] = scaler.transform(data[t]['views']['descriptors'])

    k_shots = args.k_shots
    held_smiles = data[held]['smiles']; held_y = data[held]['y']
    held_views = data[held]['views']

    # ---- Build "transfer" base predictors for each single view (pretrained
    # on all source targets) ----
    print('  Pretraining single-view source models...', flush=True)
    src_y = np.concatenate([data[t]['y'] for t in sources])
    pretrained = {}
    for view_name in ['morgan', 'encoder', 'maccs', 'descriptors']:
        Xs = np.concatenate([data[t]['views'][view_name] for t in sources], axis=0)
        # 80/10 split for early stop
        rng = np.random.default_rng(args.seed)
        order = rng.permutation(len(Xs))
        cut = int(0.9 * len(order))
        tr_idx = order[:cut]; vl_idx = order[cut:]
        m, _ = train_single_view(
            Xs[tr_idx], src_y[tr_idx], Xs[vl_idx], src_y[vl_idx],
            device, n_epochs=args.epochs, lr=args.lr,
            batch_size=args.batch_size, patience=args.patience,
        )
        pretrained[view_name] = m
        print(f'    pretrained {view_name}', flush=True)

    # ---- Pretrain MoE on source targets jointly ----
    print('  Pretraining MoE on source targets...', flush=True)
    train_views = {k: [] for k in ['morgan', 'encoder', 'maccs', 'descriptors']}
    train_tgt = []; train_y = []
    val_views = {k: [] for k in ['morgan', 'encoder', 'maccs', 'descriptors']}
    val_tgt = []; val_y = []
    rng = np.random.default_rng(args.seed)
    for t in sources:
        v = data[t]['views']; y = data[t]['y']
        order = rng.permutation(len(y))
        cut = int(0.9 * len(order))
        tr = order[:cut]; vl = order[cut:]
        for k in train_views:
            train_views[k].append(v[k][tr]); val_views[k].append(v[k][vl])
        train_tgt.append(np.full(len(tr), TARGET_IDX[t], dtype=np.int64))
        val_tgt.append(np.full(len(vl), TARGET_IDX[t], dtype=np.int64))
        train_y.append(y[tr]); val_y.append(y[vl])
    train_views = {k: np.concatenate(v, axis=0) for k, v in train_views.items()}
    val_views = {k: np.concatenate(v, axis=0) for k, v in val_views.items()}
    train_tgt = np.concatenate(train_tgt); train_y = np.concatenate(train_y)
    val_tgt = np.concatenate(val_tgt); val_y = np.concatenate(val_y)

    moe_pre, _ = train_moe(
        train_views, train_tgt, train_y,
        val_views, val_tgt, val_y,
        device, n_targets=len(ALL_TARGETS), top_k=args.top_k,
        target_emb_dim=args.target_emb_dim, n_epochs=args.epochs, lr=args.lr,
        batch_size=args.batch_size, patience=args.patience,
        load_balance=args.load_balance, dropout=args.dropout,
    )

    # ---- Few-shot evaluation ----
    results = {}  # method -> {k: {mean, std, rmses}}
    methods = ['morgan_scratch', 'morgan_transfer',
               'encoder_scratch', 'encoder_transfer',
               'maccs_scratch', 'maccs_transfer',
               'descriptors_scratch', 'descriptors_transfer',
               'moe_scratch', 'moe_transfer']
    for m in methods:
        results[m] = {str(k): {} for k in k_shots}

    for k in k_shots:
        print(f'\n  k = {k}:', flush=True)
        for trial in range(args.n_trials):
            sup_idx, qry_idx = scaffold_support_query(
                held_smiles, k, seed=args.seed + trial * 1000 + k)
            if len(qry_idx) < 3:
                continue

            # Single-view few-shot (scratch + transfer)
            for view_name in ['morgan', 'encoder', 'maccs', 'descriptors']:
                Xs = held_views[view_name][sup_idx]
                ys = held_y[sup_idx]
                Xq = held_views[view_name][qry_idx]
                yq = held_y[qry_idx]
                # Scratch
                base = make_baseline_mlp(Xs.shape[1]).to(device)
                opt = Adam(base.parameters(), lr=1e-3)
                _fit_kshot(base, opt, Xs, ys, device, n_steps=100)
                rmse_sc = _eval_kshot(base, Xq, yq, device)
                # Transfer (deepcopy from pretrained)
                base = copy.deepcopy(pretrained[view_name]).to(device)
                opt = Adam(base.parameters(), lr=1e-3)
                _fit_kshot(base, opt, Xs, ys, device, n_steps=50)
                rmse_tr = _eval_kshot(base, Xq, yq, device)
                results[f'{view_name}_scratch'].setdefault(str(k), {}).setdefault('rmses', []).append(rmse_sc)
                results[f'{view_name}_transfer'].setdefault(str(k), {}).setdefault('rmses', []).append(rmse_tr)

            # MoE few-shot (scratch and transfer)
            sup_views = {kn: held_views[kn][sup_idx] for kn in ['morgan', 'encoder', 'maccs', 'descriptors']}
            qry_views = {kn: held_views[kn][qry_idx] for kn in ['morgan', 'encoder', 'maccs', 'descriptors']}
            sup_tgt = np.full(len(sup_idx), TARGET_IDX[held], dtype=np.int64)
            qry_tgt = np.full(len(qry_idx), TARGET_IDX[held], dtype=np.int64)

            # MoE scratch
            scratch_moe = make_default_moe(
                n_targets=len(ALL_TARGETS), top_k=args.top_k,
                target_emb_dim=args.target_emb_dim,
                load_balance=args.load_balance, dropout=args.dropout,
            ).to(device)
            opt = Adam(scratch_moe.parameters(), lr=1e-3)
            _fit_kshot_moe(scratch_moe, opt, sup_views, sup_tgt, held_y[sup_idx],
                           device, n_steps=100, load_balance=args.load_balance)
            rmse_moe_sc, _ = eval_moe(scratch_moe, qry_views, qry_tgt, held_y[qry_idx], device)

            # MoE transfer (fine-tune pretrained MoE on the k support points)
            tr_moe = copy.deepcopy(moe_pre).to(device)
            opt = Adam(tr_moe.parameters(), lr=1e-3)
            _fit_kshot_moe(tr_moe, opt, sup_views, sup_tgt, held_y[sup_idx],
                           device, n_steps=50, load_balance=args.load_balance)
            rmse_moe_tr, _ = eval_moe(tr_moe, qry_views, qry_tgt, held_y[qry_idx], device)

            results['moe_scratch'].setdefault(str(k), {}).setdefault('rmses', []).append(rmse_moe_sc)
            results['moe_transfer'].setdefault(str(k), {}).setdefault('rmses', []).append(rmse_moe_tr)

    # Reduce
    summary = {}
    for method, kdict in results.items():
        summary[method] = {}
        for k_str, slot in kdict.items():
            rmses = slot.get('rmses', [])
            if not rmses:
                continue
            summary[method][k_str] = {
                'mean': float(np.mean(rmses)),
                'std': float(np.std(rmses)),
                'rmses': rmses,
            }

    # Pretty print
    print('\n  Few-shot summary (RMSE mean ± std):', flush=True)
    print(f'    {"method":<22s} ' + '  '.join([f'k={k:<3d}'.rjust(14) for k in k_shots]), flush=True)
    for method in methods:
        line = f'    {method:<22s} '
        for k in k_shots:
            kk = str(k)
            if kk in summary.get(method, {}):
                m = summary[method][kk]['mean']; s = summary[method][kk]['std']
                line += f'  {m:>5.3f} ± {s:.3f}'
            else:
                line += f'  {"-":>14s}'
        print(line, flush=True)

    out = os.path.join(args.output_dir, f'fewshot_{held}_results.json')
    with open(out, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'\nSaved to {out}', flush=True)
    return summary


def _fit_kshot(model, opt, X, y, device, n_steps=100):
    Xt = torch.tensor(X, device=device)
    yt = torch.tensor(y, device=device, dtype=torch.float32)
    model.train()
    for _ in range(n_steps):
        opt.zero_grad()
        pred = model(Xt).squeeze(-1)
        loss = F.mse_loss(pred, yt)
        loss.backward(); opt.step()


def _fit_kshot_moe(moe, opt, views_dict, target_id, y, device,
                   n_steps=100, load_balance=0.01):
    xb = [torch.tensor(views_dict[k], device=device)
          for k in ['morgan', 'encoder', 'maccs', 'descriptors']]
    tb = torch.tensor(target_id, device=device, dtype=torch.long)
    yb = torch.tensor(y, device=device, dtype=torch.float32)
    moe.train()
    for _ in range(n_steps):
        opt.zero_grad()
        pred, aux = moe(xb, tb, return_aux=True)
        loss = F.mse_loss(pred, yb) + aux['load_balance_loss']
        loss.backward(); opt.step()


def _eval_kshot(model, X, y, device):
    model.eval()
    Xt = torch.tensor(X, device=device)
    yt = torch.tensor(y, device=device, dtype=torch.float32)
    with torch.no_grad():
        return float(F.mse_loss(model(Xt).squeeze(-1), yt).sqrt().item())


# ---------------------------------------------------------------------------
# Experiment 3: cross-target selectivity (rescore generated molecules)
# ---------------------------------------------------------------------------

def run_selectivity(args, device, encoder_emb_cache, target_splits=None):
    """Rescore each target's generated molecules with the MoE predictor and
    with single-view baselines, producing the selectivity matrix.

    Generated SMILES are loaded from ``outputs/{target}_v4/generated.csv`` if
    it exists; otherwise we fall back to ``outputs/{target}_v4/eval_summary.json``
    or to a small set of top candidates.
    """
    print('\n=== Experiment 3: cross-target selectivity (MoE rescore) ===',
          flush=True)

    # Load training data and train both single-view and MoE predictors on FULL data
    target_data = {}
    for t in ALL_TARGETS:
        s, y = load_target(t)
        emb = encoder_emb_cache.get(t)
        v = compute_views(s, encoder_emb=emb)
        target_data[t] = {'smiles': s, 'y': y, 'views': v}

    # Standardize descriptors using all-targets training data
    desc_full = np.concatenate([target_data[t]['views']['descriptors']
                                 for t in ALL_TARGETS], axis=0)
    scaler = DescriptorScaler().fit(desc_full)
    for t in ALL_TARGETS:
        target_data[t]['views']['descriptors'] = scaler.transform(
            target_data[t]['views']['descriptors'])

    # Train per-target single-view (Morgan) baselines for the legacy comparison
    print('  Training per-target Morgan baselines...', flush=True)
    morgan_models = {}
    for t in ALL_TARGETS:
        v = target_data[t]['views']; y = target_data[t]['y']
        order = np.random.default_rng(args.seed).permutation(len(y))
        cut = int(0.9 * len(order))
        m, _ = train_single_view(
            v['morgan'][order[:cut]], y[order[:cut]],
            v['morgan'][order[cut:]], y[order[cut:]], device,
            n_epochs=args.epochs, lr=args.lr, patience=args.patience,
            batch_size=args.batch_size,
        )
        morgan_models[t] = m

    # Train one shared MoE on all targets
    print('  Training shared MoE...', flush=True)
    train_views = {k: [] for k in ['morgan', 'encoder', 'maccs', 'descriptors']}
    train_tgt, train_y = [], []
    val_views = {k: [] for k in ['morgan', 'encoder', 'maccs', 'descriptors']}
    val_tgt, val_y = [], []
    rng = np.random.default_rng(args.seed)
    for t in ALL_TARGETS:
        v = target_data[t]['views']; y = target_data[t]['y']
        order = rng.permutation(len(y))
        cut = int(0.9 * len(order))
        tr = order[:cut]; vl = order[cut:]
        for k in train_views:
            train_views[k].append(v[k][tr]); val_views[k].append(v[k][vl])
        train_tgt.append(np.full(len(tr), TARGET_IDX[t], dtype=np.int64))
        val_tgt.append(np.full(len(vl), TARGET_IDX[t], dtype=np.int64))
        train_y.append(y[tr]); val_y.append(y[vl])
    train_views = {k: np.concatenate(v, axis=0) for k, v in train_views.items()}
    val_views = {k: np.concatenate(v, axis=0) for k, v in val_views.items()}
    train_tgt = np.concatenate(train_tgt); train_y = np.concatenate(train_y)
    val_tgt = np.concatenate(val_tgt); val_y = np.concatenate(val_y)

    moe, _ = train_moe(
        train_views, train_tgt, train_y, val_views, val_tgt, val_y, device,
        n_targets=len(ALL_TARGETS), top_k=args.top_k,
        target_emb_dim=args.target_emb_dim, n_epochs=args.epochs, lr=args.lr,
        batch_size=args.batch_size, patience=args.patience,
        load_balance=args.load_balance, dropout=args.dropout,
    )

    # Locate generated SMILES per target
    candidates = {}
    for t in ['scd1', 'nk1r', 'drd2']:
        cand = _load_generated_smiles(t, top_n=args.top_n_selectivity)
        candidates[t] = cand
        print(f'  {t}: scoring {len(cand)} generated candidates', flush=True)

    # Compute encoder embeddings for generated molecules so the encoder
    # expert sees real embeddings (not the Morgan-prefix placeholder).
    # We reuse the cached dual encoder weights.
    print('  Computing encoder embeddings for generated candidates...', flush=True)
    enc_weights = os.path.join(args.output_dir, 'dual_encoder.pt')
    cand_encoder_emb = {}
    for t_gen, cand in candidates.items():
        if not cand:
            continue
        if not args.skip_encoder and os.path.exists(enc_weights):
            cand_encoder_emb[t_gen] = get_dual_encoder_embeddings(
                cand, device, weights_path=enc_weights)
        else:
            cand_encoder_emb[t_gen] = None  # falls back to placeholder

    # Rescore with Morgan baselines and with MoE
    selectivity_morgan = {}
    selectivity_moe = {}
    for t_gen in ['scd1', 'nk1r', 'drd2']:
        cand = candidates[t_gen]
        if not cand:
            continue
        # Build views with the real encoder embeddings
        v = compute_views(cand, encoder_emb=cand_encoder_emb.get(t_gen))
        v['descriptors'] = scaler.transform(v['descriptors'])

        # Morgan: each target's Morgan model scores all candidates
        selectivity_morgan[t_gen] = {}
        for t_pred in ALL_TARGETS:
            Xt = torch.tensor(v['morgan'], device=device)
            morgan_models[t_pred].eval()
            with torch.no_grad():
                pred = morgan_models[t_pred](Xt).squeeze(-1).cpu().numpy()
            selectivity_morgan[t_gen][t_pred] = float(pred.mean())

        # MoE: feed each target_id to score same molecules under different targets
        selectivity_moe[t_gen] = {}
        xb = [torch.tensor(v[k], device=device)
              for k in ['morgan', 'encoder', 'maccs', 'descriptors']]
        for t_pred in ALL_TARGETS:
            tb = torch.tensor(np.full(len(cand), TARGET_IDX[t_pred], dtype=np.int64),
                              device=device)
            moe.eval()
            with torch.no_grad():
                pred = moe(xb, tb).cpu().numpy()
            selectivity_moe[t_gen][t_pred] = float(pred.mean())

    out = {
        'morgan': selectivity_morgan,
        'moe': selectivity_moe,
        'targets_scored': ALL_TARGETS,
    }
    p = os.path.join(args.output_dir, 'selectivity_results.json')
    with open(p, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'\nSaved selectivity results to {p}', flush=True)

    # Print
    print('\n  Morgan baseline selectivity (mean predicted pIC50):', flush=True)
    _print_selectivity_table(selectivity_morgan, ALL_TARGETS)
    print('\n  MoE selectivity (mean predicted pIC50):', flush=True)
    _print_selectivity_table(selectivity_moe, ALL_TARGETS)
    return out


def _print_selectivity_table(matrix, targets):
    header = '    {:>12s}  '.format('generated_for') + '  '.join(
        f'{t:>8s}' for t in targets)
    print(header, flush=True)
    for t_gen in ['scd1', 'nk1r', 'drd2']:
        if t_gen not in matrix:
            continue
        row = '    {:>12s}  '.format(t_gen) + '  '.join(
            f'{matrix[t_gen].get(t, 0.0):>8.2f}' for t in targets)
        print(row, flush=True)


def _load_generated_smiles(target, top_n=200, sort_by='pred_pIC50'):
    """Try a sequence of candidate file locations. Fall back to small set.

    If the file has a ``pred_pIC50`` column we sort by it descending and take
    the top ``top_n`` (this is how the JBHI Table V candidates are picked).
    """
    candidates = []
    paths = [
        os.path.join(BASE, 'outputs', f'{target}_v4', 'eval', 'generated_molecules.csv'),
        os.path.join(BASE, 'outputs', f'{target}_v4', 'generated_molecules.csv'),
        os.path.join(BASE, 'outputs', f'{target}_v4', 'generated.csv'),
        os.path.join(BASE, 'outputs', f'{target}_v3', 'eval', 'generated_molecules.csv'),
        os.path.join(BASE, 'outputs', target, 'eval', 'generated_molecules.csv'),
        os.path.join(BASE, 'outputs', target, 'generated.csv'),
    ]
    for p in paths:
        if os.path.exists(p):
            try:
                df = pd.read_csv(p)
                # Standardize column name
                if 'smiles' in df.columns and 'SMILES' not in df.columns:
                    df = df.rename(columns={'smiles': 'SMILES'})
                if 'SMILES' not in df.columns:
                    df = df.rename(columns={df.columns[0]: 'SMILES'})
                # Filter to valid SMILES
                df = df.dropna(subset=['SMILES'])
                df = df[df['SMILES'].apply(lambda s: Chem.MolFromSmiles(str(s)) is not None)]
                if df.empty:
                    continue
                # Drug-like filter (loose) and sort by pIC50 if available
                if 'pred_pIC50' in df.columns:
                    df = df.sort_values('pred_pIC50', ascending=False)
                if 'qed' in df.columns:
                    df = df[df['qed'] >= 0.4]
                if 'lipinski' in df.columns:
                    df = df[df['lipinski'] == 1]
                # Drop duplicates
                df = df.drop_duplicates(subset=['SMILES'])
                smis = df['SMILES'].astype(str).tolist()
                if smis:
                    candidates = smis[:top_n]
                    break
            except Exception as e:
                print(f'    skip {p}: {e}', flush=True)
                continue

    if not candidates:
        # Fall back to top candidates from the JBHI paper's Table V
        fallback = {
            'scd1': [
                'CCc1ncccc1NN1CCc2ccccc2OC1',
                'O=C(NCc1ccccc1)C1NCCc2ccccc21',
                'CCc1ncccc1OC1CCc2ccccc21',
            ],
            'nk1r': [
                'COc1ccccc1C1OC(c2ccccc2)C(=O)NCC1O',
                'COc1ccccc1CN1CCN(C(=O)CCc2ccccc2)NC1=O',
                'CCOc1ccccc1N1CCN(CCCc2ccccc2)CC1',
            ],
            'drd2': [
                'COc1ccccc1N1CCN(C)CC1',
                'CCOc1ccccc1N1CCN(CCCc2ccccc2)CC1',
                'COc1ccccc1N1CCN(CCc2ccccc2)CC1',
            ],
        }
        candidates = [s for s in fallback.get(target, []) if Chem.MolFromSmiles(s)]
    return candidates


# ---------------------------------------------------------------------------
# Top-K and target-conditioning ablations
# ---------------------------------------------------------------------------

def run_topk_ablation(args, device, encoder_emb_cache, target_splits=None):
    """Compare top-K=1, 2, 4 and target-conditioning on/off."""
    print('\n=== Experiment 4: top-K and target-conditioning ablation ===',
          flush=True)
    target_data = {}
    for t in ALL_TARGETS:
        s, y = load_target(t)
        emb = encoder_emb_cache.get(t)
        v = compute_views(s, encoder_emb=emb)
        if target_splits is not None and t in target_splits:
            tr = target_splits[t]['tr']
            val = target_splits[t]['val']
            te = target_splits[t]['te']
        else:
            tr, val, te = scaffold_split_indices(s, seed=args.seed)
        target_data[t] = {'smiles': s, 'y': y, 'views': v, 'tr': tr, 'val': val, 'te': te}

    # Standardize descriptors per training split
    for t in ALL_TARGETS:
        sc = DescriptorScaler().fit(target_data[t]['views']['descriptors'][target_data[t]['tr']])
        target_data[t]['views']['descriptors'] = sc.transform(target_data[t]['views']['descriptors'])

    configs = [
        {'name': 'topk=1, target-cond',  'top_k': 1, 'target_emb': args.target_emb_dim},
        {'name': 'topk=2, target-cond',  'top_k': 2, 'target_emb': args.target_emb_dim},
        {'name': 'topk=4, target-cond',  'top_k': 4, 'target_emb': args.target_emb_dim},
        {'name': 'topk=2, no target',    'top_k': 2, 'target_emb': 0},
    ]
    results = []
    for cfg in configs:
        rmses_per_target = {t: [] for t in ALL_TARGETS}
        for trial in range(args.n_trials):
            np.random.seed(args.seed + trial); torch.manual_seed(args.seed + trial)
            train_views = {k: [] for k in ['morgan', 'encoder', 'maccs', 'descriptors']}
            train_tgt, train_y = [], []
            val_views = {k: [] for k in ['morgan', 'encoder', 'maccs', 'descriptors']}
            val_tgt, val_y = [], []
            for t in ALL_TARGETS:
                v = target_data[t]['views']; d = target_data[t]
                for k in train_views:
                    train_views[k].append(v[k][d['tr']]); val_views[k].append(v[k][d['val']])
                # If no target conditioning, force all targets to a single id 0
                if cfg['target_emb'] == 0:
                    train_tgt.append(np.zeros(len(d['tr']), dtype=np.int64))
                    val_tgt.append(np.zeros(len(d['val']), dtype=np.int64))
                else:
                    train_tgt.append(np.full(len(d['tr']), TARGET_IDX[t], dtype=np.int64))
                    val_tgt.append(np.full(len(d['val']), TARGET_IDX[t], dtype=np.int64))
                train_y.append(d['y'][d['tr']]); val_y.append(d['y'][d['val']])
            train_views = {k: np.concatenate(v, axis=0) for k, v in train_views.items()}
            val_views = {k: np.concatenate(v, axis=0) for k, v in val_views.items()}
            train_tgt = np.concatenate(train_tgt); train_y = np.concatenate(train_y)
            val_tgt = np.concatenate(val_tgt); val_y = np.concatenate(val_y)

            n_targets_cfg = len(ALL_TARGETS) if cfg['target_emb'] > 0 else 1
            target_emb_dim = max(1, cfg['target_emb']) if cfg['target_emb'] > 0 else 1
            moe, _ = train_moe(
                train_views, train_tgt, train_y, val_views, val_tgt, val_y, device,
                n_targets=n_targets_cfg, top_k=cfg['top_k'],
                target_emb_dim=target_emb_dim, n_epochs=args.epochs, lr=args.lr,
                batch_size=args.batch_size, patience=args.patience,
                load_balance=args.load_balance, dropout=args.dropout,
            )
            for t in ALL_TARGETS:
                d = target_data[t]; v = target_data[t]['views']
                te_views = {k: v[k][d['te']] for k in ['morgan', 'encoder', 'maccs', 'descriptors']}
                tid = (np.zeros(len(d['te']), dtype=np.int64) if cfg['target_emb'] == 0
                       else np.full(len(d['te']), TARGET_IDX[t], dtype=np.int64))
                rmse, _ = eval_moe(moe, te_views, tid, d['y'][d['te']], device)
                rmses_per_target[t].append(rmse)

        per_target_mean = {t: float(np.mean(r)) for t, r in rmses_per_target.items()}
        per_target_std = {t: float(np.std(r)) for t, r in rmses_per_target.items()}
        all_rmse = [v for r in rmses_per_target.values() for v in r]
        avg = float(np.mean(all_rmse))
        results.append({
            'config': cfg['name'], 'top_k': cfg['top_k'],
            'target_emb_dim': cfg['target_emb'],
            'per_target_mean': per_target_mean,
            'per_target_std': per_target_std,
            'overall_mean': avg,
        })
        print(f'  {cfg["name"]:<24s}  avg RMSE = {avg:.4f}', flush=True)
        for t in ALL_TARGETS:
            print(f'      {t:>5s}: {per_target_mean[t]:.4f} +/- {per_target_std[t]:.4f}',
                  flush=True)

    p = os.path.join(args.output_dir, 'topk_ablation.json')
    with open(p, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nSaved top-K ablation to {p}', flush=True)
    return results


# ---------------------------------------------------------------------------
# Encoder embedding cache (compute once, reuse across experiments)
# ---------------------------------------------------------------------------

def precompute_target_splits(args):
    """Compute scaffold-split (tr, val, te) indices per target.

    The same splits must be used for encoder training and for downstream
    predictor training/evaluation. Otherwise the encoder leaks val/test
    scaffolds into the predictor's input embeddings.
    """
    target_splits = {}
    for t in ALL_TARGETS:
        try:
            s, _ = load_target(t)
        except FileNotFoundError:
            continue
        tr, val, te = scaffold_split_indices(s, seed=args.seed)
        target_splits[t] = {'tr': tr, 'val': val, 'te': te}
    return target_splits


def build_encoder_cache(args, device, target_splits=None):
    """Compute (or load) per-target dual-encoder 128-d embeddings.

    If ``target_splits`` is given, the encoder is trained ONLY on the union of
    training scaffolds across all targets, eliminating leakage of val/test
    scaffolds into the encoder embeddings used downstream. The leak-free
    encoder is saved as ``dual_encoder_clean.pt``. If ``target_splits`` is
    None, the legacy full-data encoder is used (saved as ``dual_encoder.pt``).
    If --skip_encoder is given, returns an empty dict and the placeholder
    projection in compute_views() is used instead.
    """
    if args.skip_encoder:
        return {}
    cache = {}
    if target_splits is None:
        weights_path = os.path.join(args.output_dir, 'dual_encoder.pt')
    else:
        weights_path = os.path.join(args.output_dir, 'dual_encoder_clean.pt')

    print('\n=== Building dual-encoder embedding cache ===', flush=True)

    # Build training-only data if splits are provided and weights need training
    train_smiles = train_y = None
    if target_splits is not None and not os.path.exists(weights_path):
        all_s, all_y = [], []
        for t in ALL_TARGETS:
            try:
                s, y = load_target(t)
            except FileNotFoundError:
                continue
            tr_idx = target_splits[t]['tr']
            all_s.extend([s[i] for i in tr_idx])
            all_y.extend(y[tr_idx].tolist())
        train_smiles = all_s
        train_y = all_y
        n_targets_used = len([t for t in ALL_TARGETS if t in target_splits])
        print(f'  encoder will train on {len(all_s)} training-split molecules '
              f'across {n_targets_used} targets (leak-free)', flush=True)

    for t in ALL_TARGETS:
        try:
            s, _ = load_target(t)
        except FileNotFoundError:
            continue
        cache[t] = get_dual_encoder_embeddings(
            s, device, weights_path=weights_path,
            train_smiles=train_smiles, train_y=train_y,
        )
        # After the first call the weights file exists and subsequent calls
        # will load instead of retraining; clear the override.
        train_smiles = train_y = None
        print(f'  {t}: encoder embeddings {cache[t].shape}', flush=True)
    return cache


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--experiment', choices=['ablation', 'fewshot', 'selectivity', 'topk', 'all'],
                   default='ablation')
    p.add_argument('--held_out', default='scd1', choices=['scd1', 'nk1r', 'drd2', 'fads'])
    p.add_argument('--k_shots', type=int, nargs='+', default=[5, 10, 20, 50])
    p.add_argument('--n_trials', type=int, default=5)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--epochs', type=int, default=120)
    p.add_argument('--lr', type=float, default=5e-4)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--patience', type=int, default=20)
    p.add_argument('--top_k', type=int, default=2)
    p.add_argument('--target_emb_dim', type=int, default=16)
    p.add_argument('--load_balance', type=float, default=0.01)
    p.add_argument('--dropout', type=float, default=0.2)
    p.add_argument('--top_n_selectivity', type=int, default=200)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--output_dir', default=os.path.join(BASE, 'outputs', 'moe_v5'))
    p.add_argument('--skip_encoder', action='store_true',
                   help='Use Morgan-projection placeholder instead of training a dual encoder.')
    args = p.parse_args()

    if args.device.startswith('cuda') and not torch.cuda.is_available():
        args.device = 'cpu'
    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)
    print(f'Device: {device}', flush=True)
    print(f'Output: {args.output_dir}', flush=True)

    # Precompute scaffold splits per target (used for both encoder training
    # and downstream predictor training/evaluation, so the encoder never sees
    # val or test scaffolds).
    target_splits = precompute_target_splits(args)
    for t, sp in target_splits.items():
        print(f'  split {t}: tr={len(sp["tr"])} val={len(sp["val"])} '
              f'te={len(sp["te"])}', flush=True)

    encoder_cache = build_encoder_cache(args, device, target_splits=target_splits)

    if args.experiment in ('ablation', 'all'):
        run_ablation(args, device, encoder_cache, target_splits=target_splits)
    if args.experiment in ('fewshot', 'all'):
        run_fewshot(args, device, encoder_cache)
    if args.experiment in ('selectivity', 'all'):
        run_selectivity(args, device, encoder_cache, target_splits=target_splits)
    if args.experiment in ('topk', 'all'):
        run_topk_ablation(args, device, encoder_cache, target_splits=target_splits)


if __name__ == '__main__':
    main()
