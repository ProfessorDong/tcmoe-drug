#!/usr/bin/env python3
"""Run selectivity experiment with 5 seeds and aggregate to mean ± std.

Usage:
    python run_selectivity_multiseed.py --device cpu --seeds 0 1 2 3 4

Saves per-seed JSONs to ``outputs/moe_v5/selectivity_seed{N}.json`` and an
aggregated ``outputs/moe_v5/selectivity_results.json`` with ``mean`` and
``std`` fields per cell.
"""
import argparse
import json
import os
import subprocess
import sys
from collections import defaultdict

import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT = os.path.join(BASE, 'outputs', 'moe_v5')


def run_one_seed(seed, device, output_dir):
    """Invoke run_moe_predictor.py with selectivity experiment for a single seed."""
    cmd = [
        sys.executable, os.path.join(BASE, 'run_moe_predictor.py'),
        '--experiment', 'selectivity',
        '--seed', str(seed),
        '--device', device,
        '--output_dir', output_dir,
    ]
    print(f'\n>>> Running seed={seed}: {" ".join(cmd)}', flush=True)
    r = subprocess.run(cmd, check=True)
    # The selectivity routine writes to {output_dir}/selectivity_results.json
    src = os.path.join(output_dir, 'selectivity_results.json')
    dst = os.path.join(output_dir, f'selectivity_seed{seed}.json')
    os.replace(src, dst)
    print(f'    -> saved to {dst}', flush=True)
    return dst


def aggregate(seed_jsons, output_path):
    """Aggregate per-seed selectivity JSONs to mean/std per cell."""
    cells = defaultdict(list)  # (predictor, gen, pred) -> list of values
    for jp in seed_jsons:
        with open(jp) as f:
            d = json.load(f)
        for predictor in ['morgan', 'moe']:
            if predictor not in d:
                continue
            for t_gen in d[predictor]:
                for t_pred, val in d[predictor][t_gen].items():
                    cells[(predictor, t_gen, t_pred)].append(float(val))

    out = {'morgan': {}, 'moe': {}, 'targets_scored': ['scd1', 'nk1r', 'drd2', 'fads']}
    for (predictor, t_gen, t_pred), vals in cells.items():
        if t_gen not in out[predictor]:
            out[predictor][t_gen] = {}
        if t_pred not in out[predictor][t_gen]:
            out[predictor][t_gen][t_pred] = {}
        out[predictor][t_gen][t_pred] = {
            'mean': float(np.mean(vals)),
            'std': float(np.std(vals)),
            'values': vals,
        }
    out['n_seeds'] = len(seed_jsons)

    with open(output_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'\nAggregated mean/std written to {output_path}', flush=True)
    return out


def print_table(out):
    targets = out['targets_scored']
    for predictor in ['morgan', 'moe']:
        if predictor not in out:
            continue
        label = 'Per-target Morgan' if predictor == 'morgan' else 'MoE (ours)'
        print(f'\n  {label} (mean ± std over {out["n_seeds"]} seeds):', flush=True)
        print('    {:>12s}  '.format('generated_for') +
              '  '.join(f'{t:>11s}' for t in targets), flush=True)
        for t_gen in ['scd1', 'nk1r', 'drd2']:
            if t_gen not in out[predictor]:
                continue
            cells = out[predictor][t_gen]
            line = '    {:>12s}  '.format(t_gen.upper())
            for t in targets:
                if t in cells:
                    line += f'  {cells[t]["mean"]:>5.2f}±{cells[t]["std"]:.2f}'
                else:
                    line += f'  {"-":>11s}'
            print(line, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2, 3, 4])
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--output_dir', default=DEFAULT_OUT)
    ap.add_argument('--skip_runs', action='store_true',
                    help='Skip per-seed runs and only aggregate existing seedN files.')
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    seed_jsons = []
    for s in args.seeds:
        path = os.path.join(args.output_dir, f'selectivity_seed{s}.json')
        if not args.skip_runs:
            run_one_seed(s, args.device, args.output_dir)
        if os.path.exists(path):
            seed_jsons.append(path)
        else:
            print(f'WARNING: {path} missing', flush=True)

    out = aggregate(seed_jsons, os.path.join(args.output_dir, 'selectivity_results.json'))
    print_table(out)


if __name__ == '__main__':
    main()
