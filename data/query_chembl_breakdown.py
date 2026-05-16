#!/usr/bin/env python3
"""
Query ChEMBL for IC50/Ki/EC50/Kd records per target and report endpoint-type
breakdown for the SMILES present in each curated training dataset.

For each target:
  1. Find target_chembl_id via UniProt accession.
  2. Pull all IC50, Ki, EC50, Kd records.
  3. Canonicalize ChEMBL SMILES.
  4. Match against training-CSV canonical SMILES.
  5. Report counts per standard_type for the matched (training) records.

The pIC50 values themselves are NOT changed; only the disclosure column for
Table I is computed here.
"""
from __future__ import annotations
import os
import sys
import json
import time
import numpy as np
import pandas as pd
from chembl_webresource_client.new_client import new_client
from rdkit import Chem, RDLogger
from rdkit.Chem import MolFromSmiles, MolToSmiles

RDLogger.DisableLog('rdApp.*')

BASE = os.path.dirname(os.path.abspath(__file__))

# UniProt accessions per target. SCD-1, NK1R (TACR1), DRD2 are human single chains.
# FADS is reported as FADS1 + FADS2 union per the manuscript caption.
TARGETS = {
    'scd1':  {'uniprot': ['O00767'],            'csv': os.path.join(BASE, 'scd1_binding.csv'),                     'expected': 762},
    'nk1r':  {'uniprot': ['P25103'],            'csv': os.path.join(BASE, 'nk1r_combined.csv'),                    'expected': 3056},
    'drd2':  {'uniprot': ['P14416'],            'csv': os.path.join(BASE, 'drd2_bioactivity.csv'),                 'expected': 9966},
    'fads':  {'uniprot': ['O60427', 'O95864'],  'csv': os.path.join(BASE, 'fatty_acid_desaturase_bioactivity.csv'), 'expected': 1187},
}


def canonicalize(smiles_iter):
    out = []
    for s in smiles_iter:
        if not isinstance(s, str) or not s:
            out.append(None)
            continue
        mol = MolFromSmiles(s)
        if mol is None:
            out.append(None)
        else:
            try:
                out.append(MolToSmiles(mol))
            except Exception:
                out.append(None)
    return out


def find_target_chembl_ids(uniprot_ids):
    """Return all ChEMBL target IDs for the given UniProt accessions."""
    target_api = new_client.target
    ids = []
    for u in uniprot_ids:
        results = list(target_api.filter(target_components__accession=u))
        for r in results:
            ids.append((u, r['target_chembl_id'], r.get('pref_name'), r.get('organism')))
    return ids


def pull_activities(target_chembl_id):
    """Pull all IC50/Ki/EC50/Kd activity records for a target.

    Returns a DataFrame with columns: canonical_smiles, standard_type,
    standard_value, standard_units, molecule_chembl_id.
    """
    activity = new_client.activity
    rows = []
    for std_type in ['IC50', 'Ki', 'EC50', 'Kd']:
        print(f"    Querying {std_type}...", flush=True)
        t0 = time.time()
        records = activity.filter(
            target_chembl_id=target_chembl_id,
            standard_type=std_type,
            standard_units='nM',
        ).only([
            'canonical_smiles', 'standard_type', 'standard_value',
            'standard_units', 'molecule_chembl_id',
        ])
        count = 0
        for r in records:
            rows.append(r)
            count += 1
        dt = time.time() - t0
        print(f"      {count} records ({dt:.1f}s)", flush=True)
    if not rows:
        return pd.DataFrame(columns=['canonical_smiles', 'standard_type',
                                      'standard_value', 'standard_units',
                                      'molecule_chembl_id'])
    df = pd.DataFrame(rows)
    return df


def process_chembl_records(df):
    """Filter + canonicalize. Returns DataFrame keyed by canonical SMILES with
    standard_type set to the median-pIC50 record's type (deterministic tie-break
    by alphabetical order)."""
    if len(df) == 0:
        return df
    df = df.copy()
    df['standard_value'] = pd.to_numeric(df['standard_value'], errors='coerce')
    mask = (
        df['standard_value'].notna() &
        (df['standard_value'] > 0) &
        df['canonical_smiles'].notna() &
        (df['canonical_smiles'] != '')
    )
    df = df[mask].copy()
    df['pIC50'] = 9.0 - np.log10(df['standard_value'].astype(float).values)
    df = df[(df['pIC50'] >= -2) & (df['pIC50'] <= 14)].copy()
    df['canon'] = canonicalize(df['canonical_smiles'])
    df = df[df['canon'].notna()].copy()
    return df


def breakdown_for_target(name, info):
    print(f"\n{'='*70}\nTarget: {name}\n{'='*70}", flush=True)
    print(f"Training CSV: {info['csv']}", flush=True)
    train_df = pd.read_csv(info['csv'])
    print(f"  Training rows: {len(train_df)} (expected {info['expected']})", flush=True)
    train_df['canon'] = canonicalize(train_df['SMILES'])
    train_df = train_df[train_df['canon'].notna()].copy()
    train_canon = set(train_df['canon'])
    print(f"  Unique canonical SMILES in training: {len(train_canon)}", flush=True)

    # Find ChEMBL target IDs
    print(f"  UniProt accessions: {info['uniprot']}", flush=True)
    target_ids = find_target_chembl_ids(info['uniprot'])
    print(f"  ChEMBL target IDs found: {len(target_ids)}", flush=True)
    for u, tid, pname, org in target_ids:
        print(f"    {u} -> {tid} ({pname}, {org})", flush=True)

    # Pull activities
    all_records = []
    for u, tid, pname, org in target_ids:
        if org and 'sapiens' not in org.lower() and 'human' not in (org or '').lower():
            # Skip non-human orthologs unless the target itself is a complex
            continue
        print(f"  Pulling activities for {tid} ({org}):", flush=True)
        rec = pull_activities(tid)
        all_records.append(rec)
    if not all_records:
        print(f"  No records pulled.")
        return None
    chembl_df = pd.concat(all_records, ignore_index=True)
    chembl_df = process_chembl_records(chembl_df)
    print(f"  ChEMBL processed records: {len(chembl_df)}", flush=True)

    # For each canonical SMILES in training set, find all standard_types
    # present in ChEMBL. If a SMILES has multiple ChEMBL records with different
    # types (e.g. IC50 and Ki), report the per-type counts at the record level.
    chembl_df_matched = chembl_df[chembl_df['canon'].isin(train_canon)].copy()
    print(f"  ChEMBL records matching a training SMILES: {len(chembl_df_matched)}", flush=True)

    matched_smiles = set(chembl_df_matched['canon'])
    print(f"  Unique training SMILES with ChEMBL match: {len(matched_smiles)}", flush=True)
    print(f"  Unique training SMILES with NO ChEMBL match: {len(train_canon - matched_smiles)}", flush=True)

    # Per-SMILES dominant endpoint type (most-frequent among that SMILES's records)
    per_smiles_type = (
        chembl_df_matched.groupby('canon')['standard_type']
        .agg(lambda s: s.value_counts().index[0])
    )
    smiles_type_breakdown = per_smiles_type.value_counts().to_dict()

    # Per-record breakdown (counts of records, not unique SMILES)
    per_record_type = chembl_df_matched['standard_type'].value_counts().to_dict()

    return {
        'target': name,
        'training_compounds': len(train_canon),
        'matched_smiles': len(matched_smiles),
        'unmatched_smiles': len(train_canon - matched_smiles),
        'breakdown_by_smiles_dominant_type': smiles_type_breakdown,
        'breakdown_by_record': per_record_type,
    }


def main():
    out = {}
    for name, info in TARGETS.items():
        try:
            res = breakdown_for_target(name, info)
            if res is not None:
                out[name] = res
        except Exception as e:
            print(f"  ERROR on {name}: {e}", flush=True)
            import traceback
            traceback.print_exc()

    out_path = os.path.join(BASE, 'endpoint_type_breakdown.json')
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved breakdown -> {out_path}")

    # Print human-readable summary
    print(f"\n{'='*70}\nENDPOINT-TYPE BREAKDOWN SUMMARY\n{'='*70}")
    for name, r in out.items():
        print(f"\n{name.upper()}:  training={r['training_compounds']}, "
              f"matched={r['matched_smiles']} ({100*r['matched_smiles']/r['training_compounds']:.1f}%), "
              f"unmatched={r['unmatched_smiles']}")
        print(f"  By unique SMILES (dominant type): {r['breakdown_by_smiles_dominant_type']}")
        print(f"  By raw record: {r['breakdown_by_record']}")


if __name__ == '__main__':
    main()
