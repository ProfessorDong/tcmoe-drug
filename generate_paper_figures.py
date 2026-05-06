#!/usr/bin/env python3
"""
Generate all publication-quality figures for the Cancer Theranostics paper.
Saves figures as both PNG (300 dpi) and EPS to the paper figures directory.
"""

import os
import sys
import json
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

# Publication-quality style
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'font.family': 'sans-serif',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')
FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

TARGETS = ['scd1', 'nk1r', 'drd2']
TARGET_LABELS = {'scd1': 'SCD-1', 'nk1r': 'NK1R', 'drd2': 'DRD2'}

# Training data paths
TRAIN_DATA_PATHS = {
    'scd1': os.path.join(BASE_DIR, 'data', 'scd1_binding.csv'),
    'nk1r': os.path.join(BASE_DIR, 'data/nk1r_combined.csv'),
    'drd2': os.path.join(BASE_DIR, 'data/drd2_bioactivity.csv'),
}

# pIC50 thresholds per target
PIC50_THRESHOLDS = {'scd1': 7.0, 'nk1r': 7.0, 'drd2': 6.5}


def save_fig(fig, name):
    """Save figure in both PNG and EPS formats."""
    png_path = os.path.join(FIG_DIR, f'{name}.png')
    eps_path = os.path.join(FIG_DIR, f'{name}.eps')
    fig.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(eps_path, format='eps', bbox_inches='tight', facecolor='white')
    print(f"  Saved: {png_path}")
    print(f"  Saved: {eps_path}")
    plt.close(fig)


def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)


def load_generated_molecules(target):
    path = os.path.join(OUTPUT_DIR, target, 'eval', 'generated_molecules.csv')
    return pd.read_csv(path)


def load_training_data(target):
    path = TRAIN_DATA_PATHS[target]
    return pd.read_csv(path)


# ============================================================================
# FIGURE 1: Property Distributions (3x3 grid)
# ============================================================================
def fig1_property_distributions():
    print("\n[Figure 1] Property distributions...")
    fig, axes = plt.subplots(3, 3, figsize=(14, 12))

    properties = [
        ('pred_pIC50', 'pIC50', 'Predicted pIC50'),
        ('qed', None, 'QED'),
        ('logp', None, 'logP'),
    ]

    for col, target in enumerate(TARGETS):
        gen_df = load_generated_molecules(target)
        train_df = load_training_data(target)

        # Compute training QED and logP using RDKit
        from rdkit import Chem
        from rdkit.Chem import Descriptors, QED as QED_module

        train_qed = []
        train_logp = []
        train_pic50 = train_df['pIC50'].values

        for smi in train_df['SMILES']:
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                train_qed.append(QED_module.qed(mol))
                train_logp.append(Descriptors.MolLogP(mol))
            else:
                train_qed.append(np.nan)
                train_logp.append(np.nan)

        train_qed = np.array(train_qed)
        train_logp = np.array(train_logp)

        for row, (gen_col, train_col, prop_label) in enumerate(properties):
            ax = axes[row, col]

            if gen_col == 'pred_pIC50':
                gen_vals = gen_df[gen_col].dropna().values
                tr_vals = train_pic50[~np.isnan(train_pic50)]
            elif gen_col == 'qed':
                gen_vals = gen_df['qed'].dropna().values
                tr_vals = train_qed[~np.isnan(train_qed)]
            else:  # logp
                gen_vals = gen_df['logp'].dropna().values
                tr_vals = train_logp[~np.isnan(train_logp)]

            # Histogram overlay
            bins = np.linspace(
                min(gen_vals.min(), tr_vals.min()),
                max(gen_vals.max(), tr_vals.max()),
                40
            )
            ax.hist(tr_vals, bins=bins, alpha=0.55, color='coral',
                    label='Training', density=True, edgecolor='white', linewidth=0.5)
            ax.hist(gen_vals, bins=bins, alpha=0.55, color='steelblue',
                    label='Generated', density=True, edgecolor='white', linewidth=0.5)

            # Vertical lines for means
            ax.axvline(np.mean(tr_vals), color='coral', linestyle='--',
                       linewidth=1.5, alpha=0.8)
            ax.axvline(np.mean(gen_vals), color='steelblue', linestyle='--',
                       linewidth=1.5, alpha=0.8)

            # Threshold lines where applicable
            if gen_col == 'pred_pIC50':
                threshold = PIC50_THRESHOLDS[target]
                ax.axvline(threshold, color='red', linestyle=':', linewidth=1.5,
                           label=f'Threshold ({threshold})')
            elif gen_col == 'qed':
                ax.axvline(0.5, color='gray', linestyle=':', linewidth=1.2,
                           label='QED=0.5')
            elif gen_col == 'logp':
                ax.axvline(5.0, color='gray', linestyle=':', linewidth=1.2,
                           label='Lipinski (5)')

            ax.set_ylabel('Density' if col == 0 else '', fontsize=14)
            if row == 0:
                ax.set_title(f'{TARGET_LABELS[target]}', fontsize=17, fontweight='bold')
            if row == 2:
                ax.set_xlabel(prop_label, fontsize=15)
            else:
                ax.set_xlabel('')

            ax.tick_params(axis='both', labelsize=12)

            # Property label on left
            if col == 0:
                ax.set_ylabel(f'{prop_label}\nDensity', fontsize=14)

            if col == 2:
                leg = ax.legend(loc='upper right', fontsize=12, framealpha=0.6,
                                fancybox=True)
                leg.get_frame().set_facecolor('white')
                leg.get_frame().set_alpha(0.6)

    fig.tight_layout(h_pad=2.0, w_pad=1.5)
    save_fig(fig, 'property_distributions')


# ============================================================================
# FIGURE 2: Pareto Fronts (1x3)
# ============================================================================
def fig2_pareto_fronts():
    print("\n[Figure 2] Pareto fronts...")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    for i, target in enumerate(TARGETS):
        ax = axes[i]
        df = load_generated_molecules(target)
        threshold = PIC50_THRESHOLDS[target]

        pic50 = df['pred_pIC50'].values
        qed = df['qed'].values

        # Classify as hits
        is_hit = pic50 >= threshold

        # Plot non-hits first, then hits
        ax.scatter(qed[~is_hit], pic50[~is_hit], s=8, alpha=0.25,
                   c='silver', label='Non-hit', rasterized=True)
        ax.scatter(qed[is_hit], pic50[is_hit], s=12, alpha=0.5,
                   c='steelblue', label='Hit', rasterized=True)

        # Draw approximate Pareto front
        # Find non-dominated points (maximize both pIC50 and QED)
        points = np.column_stack([qed, pic50])
        pareto_mask = np.ones(len(points), dtype=bool)
        for j in range(len(points)):
            if pareto_mask[j]:
                # A point is dominated if another point is >= in all objectives and > in at least one
                for k in range(len(points)):
                    if j != k and pareto_mask[k]:
                        if points[k, 0] >= points[j, 0] and points[k, 1] >= points[j, 1]:
                            if points[k, 0] > points[j, 0] or points[k, 1] > points[j, 1]:
                                pareto_mask[j] = False
                                break

        pareto_points = points[pareto_mask]
        # Sort by QED for drawing line
        sort_idx = np.argsort(pareto_points[:, 0])
        pareto_sorted = pareto_points[sort_idx]

        ax.plot(pareto_sorted[:, 0], pareto_sorted[:, 1], 'r-', linewidth=1.5,
                alpha=0.7, label='Pareto front', zorder=5)

        # Threshold lines
        ax.axhline(threshold, color='red', linestyle=':', linewidth=1, alpha=0.6)
        ax.axvline(0.5, color='green', linestyle=':', linewidth=1, alpha=0.6)

        ax.set_xlabel('QED')
        ax.set_ylabel('Predicted pIC50' if i == 0 else '')
        ax.set_title(TARGET_LABELS[target], fontweight='bold')

        if i == 2:
            ax.legend(loc='lower right', fontsize=9, framealpha=0.9, markerscale=1.5)

    fig.tight_layout(w_pad=2.0)
    save_fig(fig, 'pareto_fronts')


# ============================================================================
# FIGURE 3: Meta-Learning Curves
# ============================================================================
def fig3_meta_learning_curves():
    print("\n[Figure 3] Meta-learning curves...")
    meta_data = load_json(os.path.join(OUTPUT_DIR, 'meta_learning', 'meta_learning_results.json'))

    fig, ax = plt.subplots(figsize=(8, 5))

    k_shots = meta_data['k_shots']
    methods = {
        'maml': {'color': 'royalblue', 'marker': 'o', 'label': 'MAML'},
        'transfer': {'color': 'darkorange', 'marker': 's', 'label': 'Transfer Learning'},
        'scratch': {'color': 'seagreen', 'marker': '^', 'label': 'From Scratch'},
    }

    for method_key, style in methods.items():
        means = [meta_data[method_key][str(k)]['mean_rmse'] for k in k_shots]
        stds = [meta_data[method_key][str(k)]['std_rmse'] for k in k_shots]

        ax.errorbar(k_shots, means, yerr=stds,
                    color=style['color'], marker=style['marker'],
                    markersize=8, linewidth=2, capsize=5, capthick=1.5,
                    label=style['label'], markerfacecolor='white',
                    markeredgewidth=2)

    ax.set_xlabel('Number of Support Samples (k-shot)')
    ax.set_ylabel('Validation RMSE')
    ax.set_title('Few-Shot Property Prediction (SCD-1 Held Out)', fontweight='bold')
    ax.set_xticks(k_shots)
    ax.legend(loc='upper right', framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')

    fig.tight_layout()
    save_fig(fig, 'meta_learning_curves')


# ============================================================================
# FIGURE 4: Ablation Comparison
# ============================================================================
def fig4_ablation_comparison():
    print("\n[Figure 4] Ablation comparison...")

    # Load ablation data for all targets
    ablation_data = {}
    for target in TARGETS:
        ablation_data[target] = load_json(
            os.path.join(OUTPUT_DIR, 'ablation', target, 'ablation_results.json')
        )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # --- Left: Representation Ablation (Bar + dots) ---
    rep_types = ['dual', 'graph_only', 'smiles_only']
    rep_labels = ['Dual\n(Graph+SMILES)', 'Graph Only', 'SMILES Only']
    colors_rep = ['#4C72B0', '#55A868', '#C44E52']

    # Collect per-target values
    per_target_vals = {rep: [] for rep in rep_types}
    for target in TARGETS:
        for rep in rep_types:
            per_target_vals[rep].append(
                ablation_data[target]['representation_ablation'][rep]['val_rmse']
            )

    means = [np.mean(per_target_vals[rep]) for rep in rep_types]
    stds = [np.std(per_target_vals[rep]) for rep in rep_types]

    x_pos = np.arange(len(rep_types))
    bars = ax1.bar(x_pos, means, yerr=stds, width=0.5,
                   color=colors_rep, alpha=0.75, capsize=5, edgecolor='black', linewidth=0.8)

    # Overlay per-target dots
    target_markers = {'scd1': 'o', 'nk1r': 's', 'drd2': '^'}
    for j, rep in enumerate(rep_types):
        for k, target in enumerate(TARGETS):
            val = per_target_vals[rep][k]
            ax1.scatter(x_pos[j] + (k - 1) * 0.12, val, s=50,
                        marker=target_markers[target], color='black',
                        zorder=5, edgecolors='white', linewidths=0.5)

    # Add legend for target dots
    for target in TARGETS:
        ax1.scatter([], [], marker=target_markers[target], color='black',
                    s=50, label=TARGET_LABELS[target])
    ax1.legend(loc='upper left', fontsize=9, framealpha=0.9)

    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(rep_labels, fontsize=10)
    ax1.set_ylabel('Validation RMSE')
    ax1.set_title('(a) Representation Ablation', fontweight='bold')
    ax1.grid(True, alpha=0.2, axis='y', linestyle='--')

    # --- Right: Training Regime Ablation (Grouped bars for SCD-1) ---
    scd1_ablation = ablation_data['scd1']
    regime_names = ['supervised_only', 'standard_rl', 'meta_rl']
    regime_labels = ['Supervised\nOnly', 'Standard\nRL', 'Meta-RL']
    # Add full framework from eval summary
    eval_summary = load_json(os.path.join(OUTPUT_DIR, 'scd1', 'eval', 'eval_summary.json'))

    validity_vals = [scd1_ablation['regime_ablation'][r]['validity'] for r in regime_names]
    validity_vals.append(eval_summary['validity'])

    qed_vals = [scd1_ablation['regime_ablation'][r]['qed_mean'] for r in regime_names]
    qed_vals.append(eval_summary['qed_mean'])

    regime_labels_full = regime_labels + ['Full\nFramework']

    x_pos2 = np.arange(len(regime_labels_full))
    width = 0.35

    bars1 = ax2.bar(x_pos2 - width / 2, validity_vals, width,
                    label='Validity', color='#4C72B0', alpha=0.8, edgecolor='black', linewidth=0.8)
    bars2 = ax2.bar(x_pos2 + width / 2, qed_vals, width,
                    label='Mean QED', color='#DD8452', alpha=0.8, edgecolor='black', linewidth=0.8)

    # Add value labels on bars
    for bar_group in [bars1, bars2]:
        for bar in bar_group:
            height = bar.get_height()
            ax2.annotate(f'{height:.2f}',
                         xy=(bar.get_x() + bar.get_width() / 2, height),
                         xytext=(0, 3), textcoords="offset points",
                         ha='center', va='bottom', fontsize=8)

    ax2.set_xticks(x_pos2)
    ax2.set_xticklabels(regime_labels_full, fontsize=10)
    ax2.set_ylabel('Value')
    ax2.set_title('(b) Training Regime Ablation (SCD-1)', fontweight='bold')
    ax2.legend(loc='upper left', fontsize=9, framealpha=0.9)
    ax2.grid(True, alpha=0.2, axis='y', linestyle='--')
    ax2.set_ylim(0, 1.05)

    fig.tight_layout(w_pad=3.0)
    save_fig(fig, 'ablation_comparison')


# ============================================================================
# FIGURE 5: Training Curves (1x3)
# ============================================================================
def fig5_training_curves():
    print("\n[Figure 5] Training curves...")

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 4))

    target_colors = {'scd1': '#4C72B0', 'nk1r': '#DD8452', 'drd2': '#55A868'}

    # --- Left: Pretrain loss ---
    pretrain = load_json(os.path.join(OUTPUT_DIR, 'scd1', 'pretrain_history.json'))
    epochs_pt = range(1, len(pretrain['pretrain_loss']) + 1)
    ax1.plot(epochs_pt, pretrain['pretrain_loss'], color='#4C72B0', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss (NLL)')
    ax1.set_title('(a) Pretraining Loss', fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')

    # --- Middle: Predictor val RMSE ---
    for target in TARGETS:
        hist = load_json(os.path.join(OUTPUT_DIR, target,
                                       f'predict_{target}_history.json'))
        if 'val_rmse' in hist:
            epochs = range(1, len(hist['val_rmse']) + 1)
            ax2.plot(epochs, hist['val_rmse'], color=target_colors[target],
                     linewidth=2, label=TARGET_LABELS[target])

    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Validation RMSE')
    ax2.set_title('(b) Predictor Training', fontweight='bold')
    ax2.legend(framealpha=0.9)
    ax2.grid(True, alpha=0.3, linestyle='--')

    # --- Right: RL reward ---
    for target in TARGETS:
        hist = load_json(os.path.join(OUTPUT_DIR, target,
                                       f'rl_{target}_history.json'))
        if 'rl_reward' in hist:
            rewards = hist['rl_reward']
            steps = range(1, len(rewards) + 1)
            # Smooth with rolling average
            window = max(1, len(rewards) // 20)
            smoothed = pd.Series(rewards).rolling(window=window, min_periods=1).mean().values
            ax3.plot(steps, smoothed, color=target_colors[target],
                     linewidth=2, label=TARGET_LABELS[target])
            ax3.plot(steps, rewards, color=target_colors[target],
                     linewidth=0.5, alpha=0.25)

    ax3.set_xlabel('RL Step')
    ax3.set_ylabel('Mean Reward')
    ax3.set_title('(c) RL Training Reward', fontweight='bold')
    ax3.legend(framealpha=0.9)
    ax3.grid(True, alpha=0.3, linestyle='--')

    fig.tight_layout(w_pad=2.5)
    save_fig(fig, 'training_curves')


# ============================================================================
# FIGURE 6: Feature Importance (Gradient-based)
# ============================================================================
def fig6_feature_importance():
    print("\n[Figure 6] Feature importance...")
    import torch
    from rdkit import Chem
    from rdkit.Chem import AllChem

    sys.path.insert(0, BASE_DIR)
    from models.property_predictor import PropertyPredictor

    # Load predictor
    ckpt = torch.load(os.path.join(OUTPUT_DIR, 'scd1', 'predictor_scd1.pt'),
                      map_location='cpu')
    predictor = PropertyPredictor(
        input_dim=1024,
        task_configs=ckpt['task_configs'],
        hidden_dims=[512, 256],
        dropout=0.2
    )
    predictor.load_state_dict(ckpt['model_state_dict'])
    predictor.eval()

    # Load generated molecules - top 100 by pIC50
    df = load_generated_molecules('scd1')
    df_sorted = df.nlargest(100, 'pred_pIC50')

    # Compute Morgan fingerprints and gradients
    all_grads = []
    bit_info_collection = {}  # bit -> list of (smiles, substructure info)

    for _, row in df_sorted.iterrows():
        mol = Chem.MolFromSmiles(row['smiles'])
        if mol is None:
            continue

        bit_info = {}
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024,
                                                     bitInfo=bit_info)
        fp_array = np.zeros(1024)
        for bit in fp.GetOnBits():
            fp_array[bit] = 1.0

        # Store bit info for interpretation
        for bit, info_list in bit_info.items():
            if bit not in bit_info_collection:
                bit_info_collection[bit] = []
            bit_info_collection[bit].append((row['smiles'], info_list))

        # Compute gradient
        fp_tensor = torch.tensor(fp_array, dtype=torch.float32).unsqueeze(0)
        fp_tensor.requires_grad_(True)

        pred = predictor(fp_tensor, task_name='pIC50')['pIC50']
        pred.backward()

        grad = fp_tensor.grad.detach().numpy().flatten()
        all_grads.append(np.abs(grad))

    # Average absolute gradients
    mean_importance = np.mean(all_grads, axis=0)

    # Top 20 bits
    top_indices = np.argsort(mean_importance)[::-1][:20]
    top_importances = mean_importance[top_indices]

    # Try to get substructure descriptions for each top bit
    bit_labels = []
    for bit_idx in top_indices:
        label = f'Bit {bit_idx}'
        if bit_idx in bit_info_collection:
            # Get the most common radius for this bit
            radii = []
            for smi, info_list in bit_info_collection[bit_idx]:
                for atom_idx, radius in info_list:
                    radii.append(radius)
            if radii:
                most_common_radius = max(set(radii), key=radii.count)
                label += f' (r={most_common_radius})'
        bit_labels.append(label)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))

    y_pos = np.arange(len(top_indices))[::-1]
    colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(top_indices)))

    bars = ax.barh(y_pos, top_importances, color=colors, edgecolor='black',
                   linewidth=0.5, height=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(bit_labels)
    ax.set_xlabel('Mean |Gradient| (Feature Importance)')
    ax.set_title('Top 20 Morgan Fingerprint Bits for SCD-1 pIC50 Prediction',
                 fontweight='bold')
    ax.grid(True, alpha=0.2, axis='x', linestyle='--')

    # Annotate values
    for bar, val in zip(bars, top_importances):
        ax.text(bar.get_width() + max(top_importances) * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f'{val:.4f}', va='center', fontsize=8)

    fig.tight_layout()
    save_fig(fig, 'feature_importance')


# ============================================================================
# FIGURE 7: Molecular Visualization
# ============================================================================
def fig7_mol_visualization():
    print("\n[Figure 7] Molecule visualization...")
    from rdkit import Chem
    from rdkit.Chem import Draw, AllChem
    from PIL import Image
    import io

    # Top 5 SCD-1 candidates by pIC50
    df = load_generated_molecules('scd1')
    top5 = df.nlargest(5, 'pred_pIC50')

    mols = []
    legends = []
    valid_rows = []

    for _, row in top5.iterrows():
        mol = Chem.MolFromSmiles(row['smiles'])
        if mol is not None:
            AllChem.Compute2DCoords(mol)
            mols.append(mol)
            leg = (f"pIC50={row['pred_pIC50']:.2f}\n"
                   f"QED={row['qed']:.3f}\n"
                   f"logP={row['logp']:.2f}")
            legends.append(leg)
            valid_rows.append(row)

    if len(mols) == 0:
        print("  WARNING: No valid molecules to visualize. Skipping.")
        return

    # Use RDKit's MolsToGridImage
    n_mols = len(mols)
    img = Draw.MolsToGridImage(
        mols,
        molsPerRow=min(n_mols, 5),
        subImgSize=(400, 350),
        legends=legends,
        legendFontSize=20,
    )

    # Convert PIL Image to matplotlib figure
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.imshow(img)
    ax.axis('off')
    ax.set_title('Top 5 SCD-1 Candidates by Predicted pIC50', fontweight='bold',
                 fontsize=14, pad=10)

    fig.tight_layout()
    save_fig(fig, 'mol_visualization')


# ============================================================================
# Main
# ============================================================================
def main():
    print("=" * 70)
    print("  Generating publication figures for Cancer Theranostics paper")
    print("=" * 70)
    print(f"  Output directory: {FIG_DIR}")

    fig1_property_distributions()
    fig2_pareto_fronts()
    fig3_meta_learning_curves()
    fig4_ablation_comparison()
    fig5_training_curves()
    fig6_feature_importance()
    fig7_mol_visualization()

    print("\n" + "=" * 70)
    print("  All figures generated successfully!")
    print("=" * 70)


if __name__ == '__main__':
    main()
