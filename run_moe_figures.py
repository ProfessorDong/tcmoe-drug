#!/usr/bin/env python3
"""Generate MoE-related figures for the JBHI manuscript.

Produces:
  figures/moe_utilization.eps    -- per-target expert utilization bar chart
  figures/moe_fewshot.eps        -- few-shot RMSE vs k for MoE and baselines
  figures/moe_topk_ablation.eps  -- top-K and target-conditioning ablation
"""

from __future__ import annotations
import os, json, argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.normpath(os.path.join(BASE, '..', '..', 'JournalPapers', 'figures'))
os.makedirs(FIG_DIR, exist_ok=True)


VIEW_LABELS = ['Morgan', 'Dual encoder', 'MACCS', 'Descriptors']
VIEW_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
TARGET_LABELS = {'scd1': 'SCD-1', 'nk1r': 'NK1R', 'drd2': 'DRD2', 'fads': 'FADS'}


def fig_utilization(ablation_path, out_path):
    with open(ablation_path) as f:
        data = json.load(f)
    targets = ['scd1', 'nk1r', 'drd2', 'fads']
    util = np.array([data[t]['moe']['utilization'] for t in targets])  # [n_targets, 4]

    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    width = 0.18
    x = np.arange(len(targets))
    y_max = max(util.max() * 1.22, 0.55)
    for i, (lbl, c) in enumerate(zip(VIEW_LABELS, VIEW_COLORS)):
        positions = x + (i - 1.5) * width
        ax.bar(positions, util[:, i], width, label=lbl, color=c,
               edgecolor='black', linewidth=0.4)
        # Annotate descriptors view (column index 3) since values are ~0 and bars are not visible
        if lbl == 'Descriptors':
            for xp, val in zip(positions, util[:, i]):
                tag = '<0.001' if val < 1e-3 else f'{val:.3f}'
                ax.text(xp, 0.012, tag, ha='center', va='bottom', fontsize=7,
                        color=c, rotation=90)
    ax.set_xticks(x)
    ax.set_xticklabels([TARGET_LABELS[t] for t in targets], fontsize=10)
    ax.set_ylabel('Mean expert utilization', fontsize=11)
    ax.set_xlabel('Target', fontsize=11)
    ax.set_ylim(0, y_max)
    ax.grid(axis='y', alpha=0.3)
    ax.legend(loc='upper center', ncol=4, fontsize=9, frameon=False,
              bbox_to_anchor=(0.5, 1.18))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_path, format='eps', bbox_inches='tight')
    plt.savefig(out_path.replace('.eps', '.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print(f'Saved {out_path}')


def fig_fewshot(fewshot_path, out_path, held_out='SCD-1'):
    with open(fewshot_path) as f:
        data = json.load(f)
    k_shots = sorted({int(k) for m in data.values() for k in m})

    plt.figure(figsize=(6.5, 4.5))
    methods = [
        ('morgan_transfer',     'Morgan transfer',     '-', 'o', '#1f77b4'),
        ('encoder_transfer',    'Dual encoder transfer','-', 's', '#ff7f0e'),
        ('maccs_transfer',      'MACCS transfer',      '-', '^', '#2ca02c'),
        ('descriptors_transfer','Descriptors transfer','-', 'v', '#d62728'),
        ('moe_transfer',        'MoE transfer (ours)', '-', 'D', '#9467bd'),
        ('morgan_scratch',      'Morgan scratch',      '--', 'o', '#1f77b4'),
        ('moe_scratch',         'MoE scratch',         '--', 'D', '#9467bd'),
    ]
    for method, label, ls, marker, color in methods:
        if method not in data:
            continue
        means, stds = [], []
        for k in k_shots:
            kk = str(k)
            if kk in data[method]:
                means.append(data[method][kk]['mean'])
                stds.append(data[method][kk]['std'])
            else:
                means.append(np.nan); stds.append(0)
        means = np.asarray(means); stds = np.asarray(stds)
        plt.errorbar(k_shots, means, yerr=stds, label=label, color=color,
                     marker=marker, ls=ls, linewidth=1.6 if 'moe' in method else 1.2,
                     markersize=6 if 'moe' in method else 5, capsize=3)

    plt.xlabel('Number of support examples $k$', fontsize=15)
    plt.ylabel('RMSE on $\\mathrm{pIC}_{50}$', fontsize=15)
    plt.title(f'Few-shot adaptation, {held_out} held out (scaffold split)',
              fontsize=15)
    plt.xticks(k_shots, fontsize=13)
    plt.yticks(fontsize=13)
    plt.grid(alpha=0.3)
    leg = plt.legend(fontsize=12, ncol=2, loc='upper right',
                     framealpha=0.6, fancybox=True)
    leg.get_frame().set_facecolor('white')
    leg.get_frame().set_alpha(0.6)
    plt.tight_layout()
    plt.savefig(out_path, format='eps', bbox_inches='tight')
    plt.savefig(out_path.replace('.eps', '.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print(f'Saved {out_path}')


def fig_topk(topk_path, out_path):
    with open(topk_path) as f:
        data = json.load(f)
    targets = ['scd1', 'nk1r', 'drd2', 'fads']
    n_cfg = len(data); n_targ = len(targets)
    fig, ax = plt.subplots(figsize=(7.5, 3.5))
    width = 0.2
    x = np.arange(n_targ)
    for i, cfg in enumerate(data):
        means = [cfg['per_target_mean'][t] for t in targets]
        stds = [cfg['per_target_std'][t] for t in targets]
        ax.bar(x + (i - (n_cfg - 1) / 2) * width, means, width,
               yerr=stds, capsize=2, label=cfg['config'], edgecolor='black',
               linewidth=0.3)
    ax.set_xticks(x)
    ax.set_xticklabels([TARGET_LABELS[t] for t in targets], fontsize=10)
    ax.set_ylabel('RMSE on $\\mathrm{pIC}_{50}$', fontsize=11)
    ax.set_xlabel('Target', fontsize=11)
    ax.legend(fontsize=8, ncol=2, loc='upper center', frameon=False,
              bbox_to_anchor=(0.5, 1.20))
    ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_path, format='eps', bbox_inches='tight')
    plt.savefig(out_path.replace('.eps', '.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print(f'Saved {out_path}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--moe_dir', default=os.path.join(BASE, 'outputs', 'moe_v5'))
    args = p.parse_args()

    abl = os.path.join(args.moe_dir, 'ablation_results.json')
    if os.path.exists(abl):
        fig_utilization(abl, os.path.join(FIG_DIR, 'moe_utilization.eps'))

    for held in ['scd1', 'fads']:
        fp = os.path.join(args.moe_dir, f'fewshot_{held}_results.json')
        if os.path.exists(fp):
            fig_fewshot(fp, os.path.join(FIG_DIR, f'moe_fewshot_{held}.eps'),
                        held_out=TARGET_LABELS[held])

    topk = os.path.join(args.moe_dir, 'topk_ablation.json')
    if os.path.exists(topk):
        fig_topk(topk, os.path.join(FIG_DIR, 'moe_topk_ablation.eps'))


if __name__ == '__main__':
    main()
