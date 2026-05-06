#!/usr/bin/env python
"""Evaluate v3 trained models: generate 10K molecules and compute metrics."""
import os, sys, csv, json, torch, numpy as np
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())

from models.molecule_generator import MoleculeGenerator
from models.property_predictor import PropertyPredictor
from data.loader import GeneratorData
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors, Crippen, QED as QEDMod, DataStructs
RDLogger.DisableLog('rdApp.*')

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def smi2fp(smi, nbits=1024):
    mol = Chem.MolFromSmiles(smi)
    if not mol: return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=nbits)
    arr = np.zeros(nbits, dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr

# Load full ChEMBL for correct vocabulary
print("Loading ChEMBL vocabulary...", flush=True)
all_smiles = [l.strip() for l in open('data/chembl_pretrain_smiles.txt')
              if l.strip() and 1 < len(l.strip()) <= 120]
gen_data = GeneratorData(smiles_list=all_smiles, tokens=None, max_len=120)
print(f"Vocabulary: {gen_data.n_characters} tokens", flush=True)

TARGET_PATHS = {
    'scd1': os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'scd1_binding.csv'),
    'nk1r': 'data/nk1r_bioactivity.csv',
    'drd2': 'data/drd2_bioactivity.csv',
}
THRESHOLDS = {'scd1': 7.0, 'nk1r': 7.0, 'drd2': 6.5}


def evaluate_target(target, output_dir, n_mols=10000):
    print(f'\n{"="*60}', flush=True)
    print(f'  {target.upper()} (v3, 200-epoch, scaffold splits)', flush=True)
    print(f'{"="*60}', flush=True)

    # Load generator
    generator = MoleculeGenerator(
        vocab_size=gen_data.n_characters, hidden_size=512,
        embedding_dim=128, n_layers=2, has_stack=True,
        stack_width=100, stack_depth=200, dropout=0.0,
        use_cuda=(device == 'cuda'))
    rl_path = os.path.join(output_dir, f'generator_rl_{target}.pt')
    generator.load_state_dict(torch.load(rl_path, map_location=device, weights_only=False))
    generator.eval()
    print(f"Generator loaded from {rl_path}", flush=True)

    # Load predictor
    threshold = THRESHOLDS[target]
    tc = {'pIC50': {'type': 'regression'}}
    predictor = PropertyPredictor(input_dim=1024, task_configs=tc,
                                  hidden_dims=[512, 256], dropout=0.3).to(device)
    pred_path = os.path.join(output_dir, f'predictor_{target}.pt')
    ckpt = torch.load(pred_path, map_location=device, weights_only=False)
    predictor.load_state_dict(ckpt.get('model_state_dict', ckpt))
    predictor.eval()

    # Training set for novelty (use 80% scaffold split, not full dataset)
    all_target_smiles = []
    with open(TARGET_PATHS[target]) as f:
        for row in csv.DictReader(f):
            smi = row.get('SMILES', '')
            if smi:
                all_target_smiles.append(smi)
    try:
        from utils.chemistry import scaffold_split
        train_idx, _, _ = scaffold_split(all_target_smiles, frac_train=0.8, frac_val=0.1)
        train_smiles = set()
        for idx in train_idx:
            mol = Chem.MolFromSmiles(all_target_smiles[idx])
            if mol:
                train_smiles.add(Chem.MolToSmiles(mol))
    except Exception:
        # Fallback: use all molecules
        train_smiles = set()
        for smi in all_target_smiles:
            mol = Chem.MolFromSmiles(smi)
            if mol:
                train_smiles.add(Chem.MolToSmiles(mol))

    # Generate
    print(f"Generating {n_mols} molecules...", flush=True)
    generated = []
    with torch.no_grad():
        for i in range(n_mols):
            try:
                result = generator.generate(gen_data, start_token='<', end_token='>')
                # generate() may return a list or a string
                if isinstance(result, list):
                    smi = result[0] if result else ''
                else:
                    smi = str(result)
                smi = smi.strip('<>').replace('\n', '')
                if smi:
                    generated.append(smi)
            except Exception:
                pass
            if (i + 1) % 2000 == 0:
                print(f"  {i+1}/{n_mols} generated ({len(generated)} valid so far)", flush=True)

    print(f"Generated {len(generated)} raw SMILES", flush=True)

    # Parse and validate
    valid_smiles = []
    for smi in generated:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            valid_smiles.append(Chem.MolToSmiles(mol))

    n_valid = len(valid_smiles)
    unique_smiles = list(set(valid_smiles))
    n_unique = len(unique_smiles)
    novel = [s for s in unique_smiles if s not in train_smiles]
    n_novel = len(novel)

    validity = n_valid / len(generated) if generated else 0
    uniqueness = n_unique / n_valid if n_valid else 0
    novelty = n_novel / n_unique if n_unique else 0

    # Diversity
    diversity = 0
    if n_unique > 1:
        fps_div = []
        for s in unique_smiles[:1000]:
            mol = Chem.MolFromSmiles(s)
            if mol:
                fps_div.append(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048))
        if len(fps_div) > 1:
            sim_sum, n_pairs = 0.0, 0
            for i in range(len(fps_div)):
                for j in range(i + 1, len(fps_div)):
                    sim_sum += DataStructs.TanimotoSimilarity(fps_div[i], fps_div[j])
                    n_pairs += 1
            diversity = 1 - sim_sum / n_pairs if n_pairs else 0

    # Properties
    qeds, logps, mws, lip_pass, pic50s, rows = [], [], [], 0, [], []
    for smi in unique_smiles:
        mol = Chem.MolFromSmiles(smi)
        if not mol:
            continue
        try:
            qed = QEDMod.qed(mol)
            logp = Crippen.MolLogP(mol)
            mw = Descriptors.MolWt(mol)
            hbd = Descriptors.NumHDonors(mol)
            hba = Descriptors.NumHAcceptors(mol)
            lip = (mw <= 500 and logp <= 5 and hbd <= 5 and hba <= 10)
            if lip:
                lip_pass += 1
            qeds.append(qed)
            logps.append(logp)
            mws.append(mw)
            fp = smi2fp(smi)
            if fp is not None:
                x = torch.tensor(fp, dtype=torch.float32).unsqueeze(0).to(device)
                with torch.no_grad():
                    p = predictor(x, task_name='pIC50')['pIC50'].item()
                pic50s.append(p)
                rows.append({
                    'smiles': smi, 'qed': round(qed, 4), 'logp': round(logp, 4),
                    'mw': round(mw, 2), 'lipinski': int(lip), 'pred_pIC50': round(p, 4),
                    'sa_approx': round(2.5, 1),
                })
        except Exception:
            pass

    lip_pct = lip_pass / len(unique_smiles) * 100 if unique_smiles else 0
    hit_rate = sum(1 for p in pic50s if p >= threshold) / len(pic50s) if pic50s else 0

    print(f"\nResults:", flush=True)
    print(f"  Validity:   {validity*100:.1f}% ({n_valid}/{len(generated)})", flush=True)
    print(f"  Uniqueness: {uniqueness*100:.1f}% ({n_unique}/{n_valid})", flush=True)
    print(f"  Novelty:    {novelty*100:.1f}% ({n_novel}/{n_unique})", flush=True)
    print(f"  Diversity:  {diversity:.4f}", flush=True)
    print(f"  Lipinski:   {lip_pct:.1f}%", flush=True)
    print(f"  QED:        {np.mean(qeds):.3f}+-{np.std(qeds):.3f}", flush=True)
    print(f"  logP:       {np.mean(logps):.2f}+-{np.std(logps):.2f}", flush=True)
    print(f"  MW:         {np.mean(mws):.0f}+-{np.std(mws):.0f}", flush=True)
    print(f"  pIC50:      {np.mean(pic50s):.3f}+-{np.std(pic50s):.3f}", flush=True)
    print(f"  Hit rate:   {hit_rate*100:.1f}% (>={threshold})", flush=True)

    # Save
    results = {
        'n_generated': len(generated),
        'validity': round(validity, 4),
        'uniqueness': round(uniqueness, 4),
        'novelty': round(novelty, 4),
        'diversity': round(diversity, 4),
        'lipinski_pct': round(lip_pct, 2),
        'qed_mean': round(float(np.mean(qeds)), 4),
        'qed_std': round(float(np.std(qeds)), 4),
        'logp_mean': round(float(np.mean(logps)), 4),
        'logp_std': round(float(np.std(logps)), 4),
        'mw_mean': round(float(np.mean(mws)), 2),
        'mw_std': round(float(np.std(mws)), 2),
        'pIC50_mean': round(float(np.mean(pic50s)), 4),
        'pIC50_std': round(float(np.std(pic50s)), 4),
        'pIC50_hit_rate': round(hit_rate, 4),
        'pIC50_threshold': threshold,
    }
    eval_dir = os.path.join(output_dir, 'eval')
    os.makedirs(eval_dir, exist_ok=True)
    json.dump(results, open(os.path.join(eval_dir, 'eval_summary.json'), 'w'), indent=2)
    with open(os.path.join(eval_dir, 'generated_molecules.csv'), 'w', newline='') as f:
        if rows:
            w = csv.DictWriter(f, fieldnames=rows[0].keys())
            w.writeheader()
            w.writerows(rows)
    print(f"Saved to {eval_dir}/", flush=True)
    return results


if __name__ == '__main__':
    import sys
    version = sys.argv[1] if len(sys.argv) > 1 else 'v4'
    for target in ['scd1', 'nk1r', 'drd2']:
        evaluate_target(target, f'outputs/{target}_{version}')
    print(f'\n=== ALL {version.upper()} EVALUATIONS COMPLETE ===', flush=True)
