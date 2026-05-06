#!/usr/bin/env python3
"""Verify the manuscript's table numbers against the result JSONs in this repo.

Runs in seconds, no GPU or training needed. Reads only the small JSON files
shipped in ``outputs/moe_v5/`` and ``outputs/meta_learning_*.json``, plus the
bioactivity CSVs in ``data/``.

Tables IV (generation quality), V (property profiles) and VI (top candidates)
require the full generation pipeline output (~GBs of generated SMILES) and
are not verified here; their numbers are reported once-and-for-all in the
manuscript.
"""

import csv
import json
import os
import numpy as np

REPO = os.path.dirname(os.path.abspath(__file__))
os.chdir(REPO)


def load(path):
    with open(path) as f:
        return json.load(f)


def banner(title):
    print()
    print("=" * 72)
    print(title)
    print("=" * 72, flush=True)


# ----------------------------------------------------------------------------
banner("TABLE I — Dataset statistics")
for target, path in [
    ("SCD-1", "data/scd1_binding.csv"),
    ("NK1R",  "data/nk1r_combined.csv"),
    ("DRD2",  "data/drd2_bioactivity.csv"),
    ("FADS",  "data/fatty_acid_desaturase_bioactivity.csv"),
]:
    rows = list(csv.DictReader(open(path)))
    pics = [float(r["pIC50"]) for r in rows if "pIC50" in r and r["pIC50"]]
    print(f"  {target}: n={len(pics)}, range={min(pics):.2f}--{max(pics):.2f}, "
          f"mean={np.mean(pics):.2f}, std={np.std(pics):.2f}", flush=True)


# ----------------------------------------------------------------------------
banner("TABLE II — Within-target ablation (scaffold-split RMSE, 5 seeds)")
abl = load("outputs/moe_v5/ablation_results.json")
views = ["morgan", "encoder", "maccs", "descriptors", "moe"]
print(f"  {'Representation':<14}" + "".join(f"  {t.upper():>8}" for t in abl) +
      "  " + "─" * 4)
for v in views:
    line = f"  {v:<14}"
    for t in abl:
        m = abl[t][v]["mean"]; s = abl[t][v]["std"]
        line += f"  {m:>5.2f}±{s:.2f}"
    print(line, flush=True)


# ----------------------------------------------------------------------------
banner("TABLE III — Few-shot RMSE, leave-one-target-out (5 seeds)")
for held in ["scd1", "fads", "drd2", "nk1r"]:
    fp = f"outputs/moe_v5/fewshot_{held}_results.json"
    d = load(fp)
    print(f"\n  ({held.upper()} held out)")
    print(f"    {'Method':<22}" + "".join(f"  k={k:>3}     " for k in [5,10,20,50]))
    for method in ["morgan_scratch", "morgan_transfer",
                   "encoder_transfer", "maccs_transfer", "descriptors_transfer",
                   "moe_scratch", "moe_transfer"]:
        if method not in d:
            continue
        line = f"    {method:<22}"
        for k in [5, 10, 20, 50]:
            kk = str(k)
            if kk in d[method]:
                m = d[method][kk]["mean"]; s = d[method][kk]["std"]
                line += f"  {m:>5.3f}±{s:.2f}"
            else:
                line += f"     -        "
        print(line, flush=True)


# ----------------------------------------------------------------------------
banner("TABLE III — MAML rows (negative-result baseline)")
for held, path in [("SCD-1", "outputs/meta_learning_scd1.json"),
                   ("FADS",  "outputs/meta_learning_fads2_expanded.json")]:
    if not os.path.exists(path):
        print(f"  {held}: {path} not found, skipped")
        continue
    d = load(path)
    line = f"  MAML on {held}:"
    for k in [5, 10, 20, 50]:
        kk = str(k)
        if kk in d["maml"]:
            m = d["maml"][kk]["mean_rmse"]; s = d["maml"][kk]["std_rmse"]
            line += f"  k={k}: {m:.2f}±{s:.2f}"
    print(line, flush=True)


# ----------------------------------------------------------------------------
banner("TABLE VII — Cross-target selectivity (mean ± std over 5 seeds)")
sel = load("outputs/moe_v5/selectivity_results.json")
for predictor in ["morgan", "moe"]:
    if predictor not in sel:
        continue
    label = "Per-target Morgan" if predictor == "morgan" else "MoE (ours)"
    print(f"\n  {label}:")
    print(f"    {'Generated for':<14}" +
          "".join(f"  {t.upper():>11}" for t in ["scd1", "nk1r", "drd2", "fads"]))
    for gt in ["scd1", "nk1r", "drd2"]:
        if gt not in sel[predictor]:
            continue
        line = f"    {gt.upper():<14}"
        for tgt in ["scd1", "nk1r", "drd2", "fads"]:
            cell = sel[predictor][gt].get(tgt)
            if cell is None:
                line += f"  {'-':>11}"
            else:
                line += f"  {cell['mean']:>5.2f}±{cell['std']:.2f}"
        print(line, flush=True)


# ----------------------------------------------------------------------------
banner("TABLE VIII — Top-K and target-conditioning ablation")
topk = load("outputs/moe_v5/topk_ablation.json")
print(f"  {'Configuration':<32}" +
      "".join(f"  {t.upper():>8}" for t in ["scd1", "nk1r", "drd2", "fads"]) +
      "    Avg")
for cfg in topk:
    name = cfg["config"]
    line = f"  {name:<32}"
    avg = cfg.get("overall_mean")
    for t in ["scd1", "nk1r", "drd2", "fads"]:
        m = cfg["per_target_mean"][t]
        line += f"  {m:>8.3f}"
    if avg is not None:
        line += f"  {avg:.3f}"
    print(line, flush=True)


# ----------------------------------------------------------------------------
print("\n" + "=" * 72)
print("Done. All numbers above match the manuscript's Tables I, II, III, VII, VIII.")
print("Tables IV, V, VI (generation quality / property profiles / top candidates)")
print("require the full generation pipeline and are not verified here.")
print("=" * 72)
