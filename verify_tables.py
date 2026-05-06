#!/usr/bin/env python3
"""Comprehensive verification: every number in the paper vs actual data."""
import json, csv, os, sys, numpy as np

os.chdir(os.path.dirname(os.path.abspath(__file__)))

print("=" * 70)
print("COMPREHENSIVE TABLE & FIGURE VERIFICATION")
print("=" * 70, flush=True)

# === TABLE II: Generation Quality ===
print("\n--- TABLE II: Generation Quality ---", flush=True)
for target, path in [('SCD-1','outputs/scd1_v2_full/eval_summary.json'),
                     ('NK1R','outputs/nk1r_v2/eval_summary.json'),
                     ('DRD2','outputs/drd2_v2/eval_summary.json')]:
    d = json.load(open(path))
    print(f"{target}: validity={d['validity']*100:.1f}% unique={d['uniqueness']*100:.1f}% "
          f"novelty={d['novelty']*100:.1f}% diversity={d['diversity']:.3f}", flush=True)

# === TABLE III: Property Profiles ===
print("\n--- TABLE III: Property Profiles ---", flush=True)
for target, path in [('SCD-1','outputs/scd1_v2_full/eval_summary.json'),
                     ('NK1R','outputs/nk1r_v2/eval_summary.json'),
                     ('DRD2','outputs/drd2_v2/eval_summary.json')]:
    d = json.load(open(path))
    print(f"{target}: pIC50={d['pIC50_mean']:.2f}+-{d['pIC50_std']:.2f} "
          f"hit={d['pIC50_hit_rate']*100:.1f}% QED={d['qed_mean']:.3f}+-{d['qed_std']:.3f} "
          f"logP={d['logp_mean']:.2f}+-{d['logp_std']:.2f} MW={d['mw_mean']:.0f}+-{d['mw_std']:.0f} "
          f"Lipinski={d['lipinski_pct']:.2f}%", flush=True)

# === TABLE IV: Top Candidates ===
print("\n--- TABLE IV: Top Candidates ---", flush=True)
for target, path in [('SCD-1','outputs/scd1_v2_full/generated_molecules.csv'),
                     ('NK1R','outputs/nk1r_v2/generated_molecules.csv'),
                     ('DRD2','outputs/drd2_v2/generated_molecules.csv')]:
    rows = list(csv.DictReader(open(path)))
    dl = [r for r in rows if float(r['qed'])>0.5 and float(r['mw'])>150 and 0.5<float(r['logp'])<5.5]
    dl.sort(key=lambda r: float(r['pred_pIC50']), reverse=True)
    print(f"  {target}:", flush=True)
    for r in dl[:3]:
        print(f"    {r['smiles'][:42]:<44} pIC50={float(r['pred_pIC50']):.2f} QED={float(r['qed']):.3f} "
              f"logP={float(r['logp']):.2f} MW={float(r['mw']):.0f}", flush=True)

# === TABLE V: Meta SCD-1 ===
print("\n--- TABLE V: Meta SCD-1 held out ---", flush=True)
d = json.load(open('outputs/meta_learning_v2/meta_learning_results.json'))
print(f"Source: {d['source_targets']}", flush=True)
for method in ['scratch','transfer','maml']:
    vals = [f"k={k}: {d[method][str(k)]['mean_rmse']:.3f}+-{d[method][str(k)]['std_rmse']:.2f}" for k in [5,10,20,50]]
    print(f"  {method}: {', '.join(vals)}", flush=True)

# === TABLE VI: Meta FADS ===
print("\n--- TABLE VI: Meta FADS held out ---", flush=True)
d = json.load(open('outputs/meta_learning_fads2_v2/meta_learning_results.json'))
print(f"Source: {d['source_targets']}", flush=True)
for method in ['scratch','transfer','maml']:
    vals = [f"k={k}: {d[method][str(k)]['mean_rmse']:.3f}+-{d[method][str(k)]['std_rmse']:.2f}" for k in [5,10,20,50]]
    print(f"  {method}: {', '.join(vals)}", flush=True)

# === TABLE VII: Ablation ===
print("\n--- TABLE VII(a): Representation Ablation ---", flush=True)
for target, path in [('SCD-1','outputs/ablation_v2/ablation_results.json'),
                     ('NK1R','outputs/ablation_v2/nk1r/ablation_results.json'),
                     ('DRD2','outputs/ablation_v2/drd2/ablation_results.json')]:
    d = json.load(open(path))
    ra = d['representation_ablation']
    print(f"  {target}: SMILES={ra['smiles_only']['val_rmse']:.3f} "
          f"Graph={ra['graph_only']['val_rmse']:.3f} Dual={ra['dual']['val_rmse']:.3f}", flush=True)

print("\n--- TABLE VII(b): Training Regime (SCD-1) ---", flush=True)
d = json.load(open('outputs/ablation_v2/ablation_results.json'))
for regime in ['supervised_only','standard_rl','meta_rl']:
    r = d['regime_ablation'][regime]
    print(f"  {regime}: validity={r['validity']*100:.1f}% unique={r['uniqueness']*100:.1f}% qed={r['qed_mean']:.3f}", flush=True)
# Full framework = main eval
df = json.load(open('outputs/scd1_v2_full/eval_summary.json'))
print(f"  full_framework: validity={df['validity']*100:.1f}% unique={df['uniqueness']*100:.1f}% qed={df['qed_mean']:.3f}", flush=True)

# === TABLE VIII: Selectivity ===
print("\n--- TABLE VIII: Cross-target Selectivity ---", flush=True)
d = json.load(open('outputs/cross_target_selectivity.json'))
for gt in ['scd1','nk1r','drd2']:
    print(f"  {gt.upper()}: vs_SCD1={d[gt]['scd1']:.2f} vs_NK1R={d[gt]['nk1r']:.2f} vs_DRD2={d[gt]['drd2']:.2f}", flush=True)

# === TABLE I: Datasets ===
print("\n--- TABLE I: Dataset Statistics ---", flush=True)
for target, path in [('SCD-1', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'scd1_binding.csv')),
                     ('NK1R', 'data/nk1r_bioactivity.csv'),
                     ('DRD2', 'data/drd2_bioactivity.csv'),
                     ('FADS', 'data/fatty_acid_desaturase_bioactivity.csv')]:
    rows = list(csv.DictReader(open(path)))
    pics = [float(r['pIC50']) for r in rows if 'pIC50' in r]
    print(f"  {target}: {len(rows)} compounds, range={min(pics):.2f}--{max(pics):.2f}, "
          f"mean={np.mean(pics):.2f}, std={np.std(pics):.2f}", flush=True)

print("\n=== FIGURE SOURCE VERIFICATION ===", flush=True)
print("Fig 1: TikZ (data-independent) - OK", flush=True)
print("Fig 2: TikZ (data-independent) - OK", flush=True)

import datetime
for name, path in [
    ('Fig 3 (property_distributions.eps)', '../../JournalPapers/figures/property_distributions.eps'),
    ('Fig 4 (pareto_fronts.eps)', '../../JournalPapers/figures/pareto_fronts.eps'),
    ('Fig 5 (top_candidates_all.png)', '../../JournalPapers/figures/top_candidates_all.png'),
    ('Fig 6 (meta_learning_curves.eps)', '../../JournalPapers/figures/meta_learning_curves.eps'),
    ('Fig 7: TikZ (data-independent)', None),
]:
    if path and os.path.exists(path):
        mtime = os.path.getmtime(path)
        dt = datetime.datetime.fromtimestamp(mtime)
        v2_start = datetime.datetime(2026, 3, 13, 21, 0)  # v2 experiments started
        status = "OK (v2)" if dt > v2_start else "WARNING: pre-v2!"
        print(f"  {name}: modified {dt.strftime('%Y-%m-%d %H:%M')} - {status}", flush=True)
    elif path is None:
        print(f"  {name} - OK", flush=True)
    else:
        print(f"  {name}: FILE NOT FOUND!", flush=True)

print("\nDone!", flush=True)
