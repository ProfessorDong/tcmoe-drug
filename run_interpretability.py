#!/usr/bin/env python
"""
Interpretability Analysis & M1 Comparison for SCD-1 Cancer Theranostics
=========================================================================
1. Feature Attribution via Integrated Gradients
2. Rule Extraction via Decision Tree
3. M1 Comparison
4. Atom Sensitivity Analysis (attention-like for RNN generator)
"""

import os
import sys
import json
import warnings
import traceback
import numpy as np
import pandas as pd
import torch
from collections import defaultdict

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# ---- Setup paths ----
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)
sys.path.insert(0, BASE_DIR)

OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs', 'interpretability')
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Output directory: {OUTPUT_DIR}")

# ---- RDKit imports ----
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Crippen, QED, DataStructs

# ---- Model imports ----
from models.property_predictor import PropertyPredictor


# ============================================================
# Helper: load predictor
# ============================================================
def load_predictor():
    ckpt_path = os.path.join(BASE_DIR, 'outputs', 'scd1', 'predictor_scd1.pt')
    ckpt = torch.load(ckpt_path, map_location='cpu')
    predictor = PropertyPredictor(
        input_dim=1024,
        task_configs=ckpt['task_configs'],
        hidden_dims=[512, 256],
        dropout=0.2
    )
    predictor.load_state_dict(ckpt['model_state_dict'])
    predictor.eval()
    return predictor


# ============================================================
# Helper: Morgan fingerprint with bitInfo
# ============================================================
def morgan_fp_with_info(smiles, radius=2, nbits=1024):
    """Return (numpy array, bitInfo dict) or (None, None)."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None
    bit_info = {}
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits, bitInfo=bit_info)
    arr = np.zeros(nbits, dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr, bit_info


# ============================================================
# Helper: compute molecular descriptors
# ============================================================
def compute_descriptors(smiles):
    """Return dict of descriptors or None."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        return {
            'MolWt': Descriptors.MolWt(mol),
            'LogP': Crippen.MolLogP(mol),
            'NumHDonors': Descriptors.NumHDonors(mol),
            'NumHAcceptors': Descriptors.NumHAcceptors(mol),
            'TPSA': Descriptors.TPSA(mol),
            'NumRotatableBonds': Descriptors.NumRotatableBonds(mol),
            'RingCount': Descriptors.RingCount(mol),
            'NumAromaticRings': Descriptors.NumAromaticRings(mol),
            'FractionCSP3': Descriptors.FractionCSP3(mol),
            'NumHeavyAtoms': mol.GetNumHeavyAtoms(),
            'QED': QED.qed(mol),
        }
    except Exception:
        return None


# ============================================================
# Load generated molecules
# ============================================================
print("\n" + "=" * 60)
print("Loading generated molecules...")
csv_path = os.path.join(BASE_DIR, 'outputs', 'scd1', 'eval', 'generated_molecules.csv')
df = pd.read_csv(csv_path)
print(f"  Total generated: {len(df)}")

# Sort by predicted pIC50 descending
df_sorted = df.sort_values('pred_pIC50', ascending=False).reset_index(drop=True)

# Top 100 by pIC50
top100 = df_sorted.head(100).copy()
print(f"  Top 100 pIC50 range: {top100['pred_pIC50'].min():.3f} - {top100['pred_pIC50'].max():.3f}")

# Load predictor
predictor = load_predictor()
print("  Predictor loaded.")


# ============================================================
# TASK 1: Feature Attribution via Integrated Gradients
# ============================================================
print("\n" + "=" * 60)
print("TASK 1: Feature Attribution via Integrated Gradients")
print("=" * 60)

try:
    n_steps = 50
    all_attributions = np.zeros(1024, dtype=np.float64)
    # Collect bit_info across molecules for substructure mapping
    bit_info_all = defaultdict(list)  # bit_idx -> list of (smiles, atom_ids, radius)
    n_valid = 0

    for idx, row in top100.iterrows():
        smi = row['smiles']
        fp_arr, bit_info = morgan_fp_with_info(smi)
        if fp_arr is None:
            continue

        # Store bitInfo
        for bit_idx, info_list in bit_info.items():
            for (center_atom, rad) in info_list:
                bit_info_all[bit_idx].append((smi, center_atom, rad))

        # Integrated Gradients: baseline = zeros, input = fp
        baseline = torch.zeros(1, 1024, dtype=torch.float32)
        inp = torch.tensor(fp_arr, dtype=torch.float32).unsqueeze(0)

        # Accumulate gradients along the path
        attributions = torch.zeros(1024, dtype=torch.float64)
        for step in range(n_steps + 1):
            alpha = step / n_steps
            interp = baseline + alpha * (inp - baseline)
            interp.requires_grad_(True)

            pred = predictor(interp, task_name='pIC50')['pIC50']
            pred.backward()

            if interp.grad is not None:
                attributions += interp.grad.squeeze().double()
                interp.grad = None

        # Scale: (input - baseline) * avg_gradient
        attributions = (inp.squeeze().double() - baseline.squeeze().double()) * attributions / (n_steps + 1)
        all_attributions += attributions.detach().numpy()
        n_valid += 1

        if (n_valid) % 20 == 0:
            print(f"  Processed {n_valid} molecules...")

    print(f"  Total valid molecules: {n_valid}")

    # Average attributions
    avg_attributions = all_attributions / max(n_valid, 1)

    # Top 20 bits by absolute attribution
    abs_attr = np.abs(avg_attributions)
    top_bit_indices = np.argsort(abs_attr)[::-1][:20]

    # Map bits to substructure SMARTS
    bit_substructures = {}
    for bit_idx in top_bit_indices:
        bit_idx_int = int(bit_idx)
        smarts_set = set()
        if bit_idx_int in bit_info_all:
            for (smi, center_atom, rad) in bit_info_all[bit_idx_int][:10]:  # limit samples
                mol = Chem.MolFromSmiles(smi)
                if mol is None:
                    continue
                if rad == 0:
                    # Single atom - use atomic symbol
                    atom = mol.GetAtomWithIdx(center_atom)
                    smarts_set.add(f"[{atom.GetSymbol()}]")
                else:
                    # Get substructure environment
                    env = Chem.FindAtomEnvironmentOfRadiusN(mol, rad, center_atom)
                    if env:
                        atoms_in_env = set()
                        for bond_idx in env:
                            bond = mol.GetBondWithIdx(bond_idx)
                            atoms_in_env.add(bond.GetBeginAtomIdx())
                            atoms_in_env.add(bond.GetEndAtomIdx())
                        if atoms_in_env:
                            try:
                                submol = Chem.PathToSubmol(mol, env)
                                sma = Chem.MolToSmarts(submol)
                                if sma:
                                    smarts_set.add(sma)
                            except Exception:
                                pass
        bit_substructures[str(bit_idx_int)] = list(smarts_set)[:5]  # limit to 5 examples

    result_task1 = {
        "top_bits": [int(b) for b in top_bit_indices],
        "bit_importance": [float(avg_attributions[b]) for b in top_bit_indices],
        "bit_substructures": bit_substructures,
        "mean_attribution_magnitude": float(np.mean(np.abs(avg_attributions)))
    }

    out_path = os.path.join(OUTPUT_DIR, 'feature_attributions.json')
    with open(out_path, 'w') as f:
        json.dump(result_task1, f, indent=2)
    print(f"  Saved: {out_path}")
    print(f"  Mean attribution magnitude: {result_task1['mean_attribution_magnitude']:.6f}")
    print(f"  Top 5 bits: {result_task1['top_bits'][:5]}")
    print(f"  Top 5 importances: {[f'{v:.4f}' for v in result_task1['bit_importance'][:5]]}")

except Exception as e:
    print(f"  ERROR in Task 1: {e}")
    traceback.print_exc()


# ============================================================
# TASK 2: Rule Extraction via Decision Tree
# ============================================================
print("\n" + "=" * 60)
print("TASK 2: Rule Extraction via Decision Tree")
print("=" * 60)

try:
    from sklearn.tree import DecisionTreeClassifier, export_text
    from sklearn.metrics import accuracy_score

    # Get top 100 hits (pIC50 >= 7.0) and 100 random non-hits (pIC50 < 7.0)
    hits = df_sorted[df_sorted['pred_pIC50'] >= 7.0].head(100)
    non_hits = df_sorted[df_sorted['pred_pIC50'] < 7.0].sample(n=min(100, len(df_sorted[df_sorted['pred_pIC50'] < 7.0])), random_state=42)

    combined = pd.concat([hits, non_hits], ignore_index=True)
    print(f"  Hits: {len(hits)}, Non-hits: {len(non_hits)}")

    descriptor_names = ['MolWt', 'LogP', 'NumHDonors', 'NumHAcceptors', 'TPSA',
                        'NumRotatableBonds', 'RingCount', 'NumAromaticRings',
                        'FractionCSP3', 'NumHeavyAtoms', 'QED']

    X_list = []
    y_list = []
    valid_smiles = []

    for _, row in combined.iterrows():
        desc = compute_descriptors(row['smiles'])
        if desc is None:
            continue
        feat = [desc[k] for k in descriptor_names]
        if any(v is None for v in feat):
            continue
        X_list.append(feat)
        y_list.append(1 if row['pred_pIC50'] >= 7.0 else 0)
        valid_smiles.append(row['smiles'])

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)
    print(f"  Valid samples: {len(X)} (positive: {np.sum(y)}, negative: {np.sum(y == 0)})")

    # Train decision tree
    dt = DecisionTreeClassifier(max_depth=4, random_state=42)
    dt.fit(X, y)

    y_pred = dt.predict(X)
    accuracy = float(accuracy_score(y, y_pred))
    print(f"  Decision tree accuracy: {accuracy:.4f}")

    # Feature importances
    feat_importances = {name: float(imp) for name, imp in zip(descriptor_names, dt.feature_importances_)}
    print(f"  Feature importances: {feat_importances}")

    # Extract rules as text
    tree_text = export_text(dt, feature_names=descriptor_names)
    print(f"  Decision tree rules:\n{tree_text}")

    # Extract rules from tree structure
    def extract_rules(tree, feature_names):
        """Extract human-readable rules from decision tree."""
        tree_ = tree.tree_
        feature_name = [
            feature_names[i] if i != -2 else "undefined!"
            for i in tree_.feature
        ]
        rules = []

        def recurse(node, conditions):
            if tree_.feature[node] == -2:
                # Leaf node
                # tree_.value may be normalized; use n_node_samples for actual counts
                support = int(tree_.n_node_samples[node])
                class_dist = tree_.value[node][0]
                pred_class = int(np.argmax(class_dist))
                confidence = float(class_dist[pred_class] / class_dist.sum()) if class_dist.sum() > 0 else 0.0
                prediction = "active" if pred_class == 1 else "inactive"
                if support >= 3:  # Only keep rules with reasonable support
                    rules.append({
                        "condition": " AND ".join(conditions) if conditions else "default",
                        "prediction": prediction,
                        "support": support,
                        "confidence": round(confidence, 4)
                    })
                return

            name = feature_name[node]
            threshold = round(float(tree_.threshold[node]), 4)

            # Left: feature <= threshold
            left_cond = conditions + [f"{name} <= {threshold}"]
            recurse(tree_.children_left[node], left_cond)

            # Right: feature > threshold
            right_cond = conditions + [f"{name} > {threshold}"]
            recurse(tree_.children_right[node], right_cond)

        recurse(0, [])
        return rules

    rules = extract_rules(dt, descriptor_names)

    result_task2 = {
        "rules": rules,
        "tree_depth": int(dt.get_depth()),
        "accuracy": accuracy,
        "feature_importances": feat_importances,
        "tree_text": tree_text
    }

    out_path = os.path.join(OUTPUT_DIR, 'extracted_rules.json')
    with open(out_path, 'w') as f:
        json.dump(result_task2, f, indent=2)
    print(f"  Saved: {out_path}")

except Exception as e:
    print(f"  ERROR in Task 2: {e}")
    traceback.print_exc()


# ============================================================
# TASK 3: M1 Comparison
# ============================================================
print("\n" + "=" * 60)
print("TASK 3: M1 Comparison")
print("=" * 60)

try:
    # M1 known properties from conference paper
    m1_properties = {
        "pIC50": 7.1,
        "logP": 3.8
    }

    # Try to find M1 SMILES from the training data
    m1_smiles = None
    # Check multiple possible locations for SCD-1 training data
    scd1_data_candidates = [
        os.path.join(BASE_DIR, 'data', 'scd1_binding.csv'),
        os.path.join(BASE_DIR, 'data', 'scd1_binding.csv'),
    ]
    train_data_path = None
    for p in scd1_data_candidates:
        if os.path.exists(p):
            train_data_path = p
            break
    if train_data_path is None:
        train_data_path = scd1_data_candidates[0]  # fallback for error message
    if os.path.exists(train_data_path):
        train_df = pd.read_csv(train_data_path)
        print(f"  Training data: {len(train_df)} compounds from {train_data_path}")
        # Check column names
        print(f"  Training columns: {list(train_df.columns)}")
        smiles_col = 'SMILES' if 'SMILES' in train_df.columns else 'smiles'

        # Look for M1: compound with pIC50 closest to 7.1 AND logP close to 3.8
        if 'pIC50' in train_df.columns and smiles_col in train_df.columns:
            # Compute logP for all training compounds to find M1
            train_logps = []
            for smi in train_df[smiles_col]:
                d = compute_descriptors(smi)
                train_logps.append(d['LogP'] if d else None)
            train_df['computed_logP'] = train_logps

            # Filter to those with valid logP and close to M1 properties
            valid_mask = train_df['computed_logP'].notna()
            train_valid = train_df[valid_mask].copy()
            train_valid['m1_dist'] = np.sqrt(
                ((train_valid['pIC50'] - 7.1) / 2.0) ** 2 +
                ((train_valid['computed_logP'] - 3.8) / 2.0) ** 2
            )
            m1_candidates = train_valid.nsmallest(5, 'm1_dist')
            print(f"  M1 closest candidates from training data:")
            for _, r in m1_candidates.iterrows():
                smi = r[smiles_col]
                logp = r['computed_logP']
                print(f"    {smi[:50]}... pIC50={r['pIC50']:.2f}, logP={logp:.2f}")
                if m1_smiles is None and abs(r['pIC50'] - 7.1) < 0.5 and abs(logp - 3.8) < 1.5:
                    m1_smiles = smi
                    m1_properties['logP'] = round(float(logp), 2)
                    m1_properties['pIC50'] = round(float(r['pIC50']), 2)
                    print(f"    -> Selected as M1 reference")
    else:
        print(f"  Training data not found at {train_data_path}")

    # Top 10 candidates
    top10 = df_sorted.head(10).copy()
    candidates = []

    for _, row in top10.iterrows():
        smi = row['smiles']
        desc = compute_descriptors(smi)
        if desc is None:
            continue

        cand = {
            "smiles": smi,
            "pIC50": round(float(row['pred_pIC50']), 4),
            "qed": round(desc['QED'], 4),
            "logp": round(desc['LogP'], 4),
            "mw": round(desc['MolWt'], 2),
            "tpsa": round(desc['TPSA'], 2),
            "num_aromatic_rings": int(desc['NumAromaticRings']),
            "num_h_donors": int(desc['NumHDonors']),
            "num_h_acceptors": int(desc['NumHAcceptors']),
            "ring_count": int(desc['RingCount']),
            "fraction_csp3": round(desc['FractionCSP3'], 4),
            "num_heavy_atoms": int(desc['NumHeavyAtoms']),
        }

        # Compute Tanimoto similarity to M1 (if available) or to training set mean
        if m1_smiles is not None:
            from utils.chemistry import compute_similarity
            sim = compute_similarity(smi, m1_smiles)
            cand["tanimoto_to_m1"] = round(float(sim), 4) if sim is not None else None
        else:
            cand["tanimoto_to_m1"] = None

        candidates.append(cand)

    # Count how many match or exceed M1
    n_better_pIC50 = sum(1 for c in candidates if c['pIC50'] >= m1_properties['pIC50'])
    n_better_logP = sum(1 for c in candidates if c['logp'] is not None and 1.0 <= c['logp'] <= 5.0)

    summary = (
        f"{n_better_pIC50} of {len(candidates)} candidates match or exceed M1 pIC50 ({m1_properties['pIC50']}). "
        f"{n_better_logP} of {len(candidates)} have logP in drug-like range (1-5)."
    )

    # Also compute Tanimoto to training set (always useful context)
    if os.path.exists(train_data_path):
        if 'train_df' not in dir() or train_df is None:
            train_df = pd.read_csv(train_data_path)
        smiles_col = 'SMILES' if 'SMILES' in train_df.columns else 'smiles'
        if smiles_col in train_df.columns:
            print("  Computing Tanimoto to nearest training molecule...")
            train_smiles = train_df[smiles_col].dropna().tolist()[:200]  # limit for speed
            for cand in candidates:
                max_sim = 0.0
                for ts in train_smiles:
                    from utils.chemistry import compute_similarity
                    sim = compute_similarity(cand['smiles'], ts)
                    if sim is not None and sim > max_sim:
                        max_sim = sim
                cand["max_tanimoto_to_train"] = round(float(max_sim), 4)

    result_task3 = {
        "m1_properties": m1_properties,
        "m1_smiles": m1_smiles,
        "top_candidates": candidates,
        "comparison_summary": summary
    }

    out_path = os.path.join(OUTPUT_DIR, 'm1_comparison.json')
    with open(out_path, 'w') as f:
        json.dump(result_task3, f, indent=2)
    print(f"  Saved: {out_path}")
    print(f"  Summary: {summary}")

    # Print candidate table
    print(f"\n  {'Rank':>4} {'pIC50':>7} {'QED':>6} {'logP':>6} {'MW':>8} {'TPSA':>7} {'AromR':>5} {'SMILES'}")
    print(f"  {'-'*4} {'-'*7} {'-'*6} {'-'*6} {'-'*8} {'-'*7} {'-'*5} {'-'*30}")
    for i, c in enumerate(candidates):
        print(f"  {i+1:4d} {c['pIC50']:7.3f} {c['qed']:6.3f} {c['logp']:6.2f} "
              f"{c['mw']:8.2f} {c['tpsa']:7.2f} {c['num_aromatic_rings']:5d} {c['smiles'][:40]}")

except Exception as e:
    print(f"  ERROR in Task 3: {e}")
    traceback.print_exc()


# ============================================================
# TASK 4: Atom Sensitivity Analysis
# ============================================================
print("\n" + "=" * 60)
print("TASK 4: Atom Sensitivity Analysis")
print("=" * 60)

try:
    top5 = df_sorted.head(5).copy()
    sensitivity_results = []

    for rank, (_, row) in enumerate(top5.iterrows()):
        smi = row['smiles']
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue

        n_atoms = mol.GetNumAtoms()
        print(f"\n  Molecule {rank+1}: {smi} (atoms: {n_atoms})")

        # Baseline prediction (full fingerprint)
        fp_full, bit_info_full = morgan_fp_with_info(smi)
        if fp_full is None:
            continue

        with torch.no_grad():
            x_full = torch.tensor(fp_full, dtype=torch.float32).unsqueeze(0)
            baseline_pred = predictor(x_full, task_name='pIC50')['pIC50'].item()

        print(f"    Baseline pIC50: {baseline_pred:.4f}")

        # For each atom, compute its contribution to the fingerprint
        # by masking the bits that this atom is responsible for
        atom_importances = []

        for atom_idx in range(n_atoms):
            # Find which bits this atom contributes to
            bits_for_atom = set()
            for bit_idx, info_list in bit_info_full.items():
                for (center, rad) in info_list:
                    if center == atom_idx:
                        bits_for_atom.add(bit_idx)
                    elif rad > 0:
                        # Check if atom is in the environment
                        env = Chem.FindAtomEnvironmentOfRadiusN(mol, rad, center)
                        if env:
                            atoms_in_env = set()
                            for bond_idx_e in env:
                                bond = mol.GetBondWithIdx(bond_idx_e)
                                atoms_in_env.add(bond.GetBeginAtomIdx())
                                atoms_in_env.add(bond.GetEndAtomIdx())
                            if atom_idx in atoms_in_env:
                                bits_for_atom.add(bit_idx)

            # Create masked fingerprint (zero out bits this atom contributes to)
            fp_masked = fp_full.copy()
            for b in bits_for_atom:
                fp_masked[b] = 0.0

            with torch.no_grad():
                x_masked = torch.tensor(fp_masked, dtype=torch.float32).unsqueeze(0)
                masked_pred = predictor(x_masked, task_name='pIC50')['pIC50'].item()

            importance = baseline_pred - masked_pred  # positive = atom contributes positively
            atom_symbol = mol.GetAtomWithIdx(atom_idx).GetSymbol()
            atom_importances.append({
                "atom_idx": atom_idx,
                "atom_symbol": atom_symbol,
                "importance": round(float(importance), 6),
                "n_bits_affected": len(bits_for_atom),
                "masked_pIC50": round(float(masked_pred), 4)
            })

        # Sort by absolute importance
        atom_importances.sort(key=lambda x: abs(x['importance']), reverse=True)

        # Print top important atoms
        print(f"    Top 5 important atoms:")
        for ai in atom_importances[:5]:
            print(f"      Atom {ai['atom_idx']} ({ai['atom_symbol']}): "
                  f"importance={ai['importance']:.4f}, bits={ai['n_bits_affected']}")

        sensitivity_results.append({
            "smiles": smi,
            "n_atoms": n_atoms,
            "baseline_pIC50": round(float(baseline_pred), 4),
            "atom_importances": atom_importances
        })

    result_task4 = {
        "method": "fingerprint_masking_sensitivity",
        "description": (
            "For each atom in the top 5 SCD-1 molecules, we mask its contribution "
            "to the Morgan fingerprint and measure the drop in predicted pIC50. "
            "A larger positive importance means the atom contributes more to predicted activity."
        ),
        "molecules": sensitivity_results
    }

    out_path = os.path.join(OUTPUT_DIR, 'atom_sensitivity.json')
    with open(out_path, 'w') as f:
        json.dump(result_task4, f, indent=2)
    print(f"\n  Saved: {out_path}")

except Exception as e:
    print(f"  ERROR in Task 4: {e}")
    traceback.print_exc()


# ============================================================
# FINAL SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("INTERPRETABILITY ANALYSIS COMPLETE")
print("=" * 60)
print(f"\nOutput files:")
for fname in sorted(os.listdir(OUTPUT_DIR)):
    fpath = os.path.join(OUTPUT_DIR, fname)
    size = os.path.getsize(fpath)
    print(f"  {fname}: {size:,} bytes")
print("\nDone!")
