#!/usr/bin/env python
"""Evaluate REINVENT4 baseline results for all 3 targets."""

import pandas as pd
import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, Descriptors, Crippen, QED as QED_module
import json, os, time

RDLogger.DisableLog('rdApp.*')

BASE = os.path.dirname(os.path.abspath(__file__))

TARGETS = {
    'scd1': {
        'output_dir': os.path.join(BASE, 'outputs/reinvent4_scd1'),
        'data_path': os.path.join(os.path.dirname(BASE), 'theranostics_old/data/scd1_binding.csv'),
    },
    'nk1r': {
        'output_dir': os.path.join(BASE, 'outputs/reinvent4_nk1r'),
        'data_path': os.path.join(BASE, 'data/nk1r_bioactivity.csv'),
    },
    'drd2': {
        'output_dir': os.path.join(BASE, 'outputs/reinvent4_drd2'),
        'data_path': os.path.join(BASE, 'data/drd2_bioactivity.csv'),
    },
}


def evaluate_target(target, output_dir, data_path):
    t0 = time.time()

    # Read sampling results
    sampling_file = os.path.join(output_dir, 'sampling_results.csv')
    df = pd.read_csv(sampling_file)
    all_smiles = df[df['SMILES_state'] == 1]['SMILES'].tolist()
    total_sampled = len(df)

    # Canonicalize and get valid
    valid_smiles = []
    for smi in all_smiles:
        try:
            mol = Chem.MolFromSmiles(str(smi))
            if mol:
                valid_smiles.append(Chem.MolToSmiles(mol))
        except Exception:
            pass

    # Uniqueness
    unique_smiles = list(set(valid_smiles))

    # Reference SMILES for novelty
    ref_df = pd.read_csv(data_path)
    smiles_col = [c for c in ref_df.columns if c.lower() == 'smiles'][0]
    ref_set = set()
    for smi in ref_df[smiles_col]:
        mol = Chem.MolFromSmiles(str(smi))
        if mol:
            ref_set.add(Chem.MolToSmiles(mol))

    novel = [s for s in unique_smiles if s not in ref_set]

    # Diversity (sample 1000 for speed)
    sample_for_div = unique_smiles[:1000]
    fps = []
    for smi in sample_for_div:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024))

    n = min(500, len(fps))
    sims = []
    for i in range(n):
        for j in range(i + 1, n):
            sims.append(DataStructs.TanimotoSimilarity(fps[i], fps[j]))
    diversity = 1.0 - np.mean(sims) if sims else 0.0

    # Properties for unique molecules
    qeds, logps, mws, lipinski_pass = [], [], [], 0
    for smi in unique_smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            qeds.append(QED_module.qed(mol))
            logps.append(Crippen.MolLogP(mol))
            mws.append(Descriptors.MolWt(mol))
            if (Descriptors.MolWt(mol) <= 500 and
                Crippen.MolLogP(mol) <= 5 and
                Descriptors.NumHDonors(mol) <= 5 and
                Descriptors.NumHAcceptors(mol) <= 10):
                lipinski_pass += 1

    results = {
        'target': target,
        'total_sampled': total_sampled,
        'validity': round(len(valid_smiles) / total_sampled * 100, 1),
        'uniqueness': round(len(unique_smiles) / len(valid_smiles) * 100, 1) if valid_smiles else 0,
        'novelty': round(len(novel) / len(unique_smiles) * 100, 1) if unique_smiles else 0,
        'diversity': round(diversity, 4),
        'qed_mean': round(np.mean(qeds), 3),
        'qed_std': round(np.std(qeds), 3),
        'logp_mean': round(np.mean(logps), 3),
        'logp_std': round(np.std(logps), 3),
        'mw_mean': round(np.mean(mws), 1),
        'mw_std': round(np.std(mws), 1),
        'lipinski_pct': round(lipinski_pass / len(unique_smiles) * 100, 1) if unique_smiles else 0,
        'num_valid': len(valid_smiles),
        'num_unique': len(unique_smiles),
        'num_novel': len(novel),
    }

    with open(os.path.join(output_dir, 'reinvent4_eval.json'), 'w') as f:
        json.dump(results, f, indent=2)

    elapsed = time.time() - t0
    print(f'\n=== {target.upper()} ===')
    print(f'  Validity:   {results["validity"]}% ({results["num_valid"]}/{total_sampled})')
    print(f'  Uniqueness: {results["uniqueness"]}% ({results["num_unique"]}/{results["num_valid"]})')
    print(f'  Novelty:    {results["novelty"]}% ({results["num_novel"]}/{results["num_unique"]})')
    print(f'  Diversity:  {results["diversity"]}')
    print(f'  QED:        {results["qed_mean"]} ± {results["qed_std"]}')
    print(f'  LogP:       {results["logp_mean"]} ± {results["logp_std"]}')
    print(f'  MW:         {results["mw_mean"]} ± {results["mw_std"]}')
    print(f'  Lipinski:   {results["lipinski_pct"]}%')
    print(f'  Eval time:  {elapsed:.1f}s')
    return results


if __name__ == '__main__':
    all_results = {}
    for target, cfg in TARGETS.items():
        all_results[target] = evaluate_target(target, cfg['output_dir'], cfg['data_path'])

    # Summary comparison table
    print('\n' + '=' * 80)
    print('REINVENT 4 Baseline Results Summary')
    print('=' * 80)
    print(f'{"Metric":<20} {"SCD-1":>12} {"NK1R":>12} {"DRD2":>12}')
    print('-' * 56)
    for key in ['validity', 'uniqueness', 'novelty', 'diversity',
                'qed_mean', 'logp_mean', 'mw_mean', 'lipinski_pct']:
        vals = [str(all_results[t][key]) for t in ['scd1', 'nk1r', 'drd2']]
        print(f'{key:<20} {vals[0]:>12} {vals[1]:>12} {vals[2]:>12}')

    # Save combined results
    with open(os.path.join(BASE, 'outputs/reinvent4_combined_results.json'), 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f'\nCombined results saved to outputs/reinvent4_combined_results.json')
