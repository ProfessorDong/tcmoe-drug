#!/usr/bin/env python3
"""
Query ChEMBL for FADS2 and FADS1 bioactivity data.

FADS2 (Delta-6 desaturase): UniProt O95864
FADS1 (Delta-5 desaturase): UniProt O60427
"""

import os
import sys
import numpy as np
import pandas as pd
from chembl_webresource_client.new_client import new_client
from rdkit import Chem
from rdkit.Chem import MolToSmiles, MolFromSmiles

DATA_DIR = "os.path.join(os.path.dirname(os.path.abspath(__file__)))"

def search_target(uniprot_id, name):
    """Search for a ChEMBL target by UniProt accession."""
    target = new_client.target
    results = target.filter(target_components__accession=uniprot_id)
    results_list = list(results)
    print(f"\n{'='*60}")
    print(f"Target search for {name} (UniProt: {uniprot_id})")
    print(f"{'='*60}")
    if not results_list:
        print(f"  WARNING: No target found for UniProt {uniprot_id}")
        return None
    for r in results_list:
        print(f"  ChEMBL ID: {r['target_chembl_id']}")
        print(f"  Name: {r['pref_name']}")
        print(f"  Type: {r['target_type']}")
        print(f"  Organism: {r['organism']}")
    return results_list


def get_bioactivity(target_chembl_id, target_name):
    """Download all bioactivity records for a target."""
    activity = new_client.activity
    print(f"\nQuerying bioactivity for {target_name} ({target_chembl_id})...")

    # Query for IC50, Ki, EC50, Kd
    records = []
    for std_type in ['IC50', 'Ki', 'EC50', 'Kd']:
        print(f"  Querying {std_type}...")
        acts = activity.filter(
            target_chembl_id=target_chembl_id,
            standard_type=std_type
        )
        acts_list = list(acts)
        print(f"    Found {len(acts_list)} raw {std_type} records")
        records.extend(acts_list)

    print(f"  Total raw records: {len(records)}")
    return records


def process_records(records, source_label):
    """Filter, convert to pIC50, deduplicate, validate SMILES."""
    print(f"\nProcessing records from {source_label}...")

    # Convert to DataFrame
    if not records:
        print("  No records to process")
        return pd.DataFrame(columns=['SMILES', 'pIC50', 'standard_type', 'chembl_id', 'source'])

    df = pd.DataFrame(records)
    print(f"  Total raw records: {len(df)}")

    # Filter: must have standard_value, standard_units == 'nM', canonical_smiles
    mask = (
        df['standard_value'].notna() &
        (df['standard_units'] == 'nM') &
        df['canonical_smiles'].notna() &
        (df['canonical_smiles'] != '')
    )
    df = df[mask].copy()
    print(f"  After filtering (nM, non-null value/SMILES): {len(df)}")

    if len(df) == 0:
        return pd.DataFrame(columns=['SMILES', 'pIC50', 'standard_type', 'chembl_id', 'source'])

    # Convert standard_value to float
    df['standard_value'] = pd.to_numeric(df['standard_value'], errors='coerce')
    df = df[df['standard_value'].notna() & (df['standard_value'] > 0)].copy()
    print(f"  After removing non-positive/NaN values: {len(df)}")

    # Convert to pIC50: -log10(value_in_M) = 9 - log10(value_in_nM)
    df['pIC50'] = 9 - np.log10(df['standard_value'].values)

    # Filter out unreasonable pIC50 values (< -2 or > 14)
    df = df[(df['pIC50'] >= -2) & (df['pIC50'] <= 14)].copy()
    print(f"  After filtering unreasonable pIC50: {len(df)}")

    # Validate and canonicalize SMILES with RDKit
    valid_smiles = []
    valid_indices = []
    for idx, smi in zip(df.index, df['canonical_smiles']):
        try:
            mol = MolFromSmiles(smi)
            if mol is not None:
                can_smi = MolToSmiles(mol)
                valid_smiles.append(can_smi)
                valid_indices.append(idx)
        except Exception:
            continue

    df = df.loc[valid_indices].copy()
    df['canonical_smiles'] = valid_smiles
    print(f"  After RDKit validation: {len(df)}")

    # Deduplicate by canonical SMILES (keep median pIC50)
    dedup = df.groupby('canonical_smiles').agg(
        pIC50=('pIC50', 'median'),
        standard_type=('standard_type', 'first'),
        molecule_chembl_id=('molecule_chembl_id', 'first'),
        n_measurements=('pIC50', 'count')
    ).reset_index()

    print(f"  After deduplication (median pIC50): {len(dedup)}")

    result = pd.DataFrame({
        'SMILES': dedup['canonical_smiles'],
        'pIC50': dedup['pIC50'].round(4),
        'standard_type': dedup['standard_type'],
        'chembl_id': dedup['molecule_chembl_id'],
        'n_measurements': dedup['n_measurements'],
        'source': source_label
    })

    return result


def main():
    # ==========================================
    # Step 1: Search for targets
    # ==========================================
    fads2_targets = search_target('O95864', 'FADS2')
    fads1_targets = search_target('O60427', 'FADS1')

    # ==========================================
    # Step 2: Get bioactivity data
    # ==========================================
    all_fads2_records = []
    all_fads1_records = []

    if fads2_targets:
        for t in fads2_targets:
            tid = t['target_chembl_id']
            records = get_bioactivity(tid, f"FADS2 ({tid})")
            all_fads2_records.extend(records)

    if fads1_targets:
        for t in fads1_targets:
            tid = t['target_chembl_id']
            records = get_bioactivity(tid, f"FADS1 ({tid})")
            all_fads1_records.extend(records)

    # ==========================================
    # Step 3: Process records
    # ==========================================
    fads2_df = process_records(all_fads2_records, 'ChEMBL_FADS2')
    fads1_df = process_records(all_fads1_records, 'ChEMBL_FADS1')

    # ==========================================
    # Step 4: Save individual files
    # ==========================================
    fads2_path = os.path.join(DATA_DIR, 'fads2_chembl_bioactivity.csv')
    fads1_path = os.path.join(DATA_DIR, 'fads1_chembl_bioactivity.csv')
    combined_path = os.path.join(DATA_DIR, 'fads_combined_bioactivity.csv')

    fads2_df.to_csv(fads2_path, index=False)
    print(f"\nSaved FADS2 ChEMBL data: {len(fads2_df)} compounds -> {fads2_path}")

    fads1_df.to_csv(fads1_path, index=False)
    print(f"Saved FADS1 ChEMBL data: {len(fads1_df)} compounds -> {fads1_path}")

    # Combined
    combined_df = pd.concat([fads2_df, fads1_df], ignore_index=True)
    # Deduplicate combined by SMILES (keep median pIC50 again)
    if len(combined_df) > 0:
        combined_dedup = combined_df.groupby('SMILES').agg(
            pIC50=('pIC50', 'median'),
            standard_type=('standard_type', 'first'),
            chembl_id=('chembl_id', 'first'),
            n_measurements=('n_measurements', 'sum'),
            source=('source', lambda x: '+'.join(sorted(set(x))))
        ).reset_index()
        combined_dedup['pIC50'] = combined_dedup['pIC50'].round(4)
    else:
        combined_dedup = combined_df

    combined_dedup.to_csv(combined_path, index=False)
    print(f"Saved combined FADS1+FADS2 data: {len(combined_dedup)} compounds -> {combined_path}")

    # ==========================================
    # Step 5: Merge with existing datasets
    # ==========================================
    print(f"\n{'='*60}")
    print("Merging with existing datasets")
    print(f"{'='*60}")

    # Read existing FADS2 dataset (23 compounds)
    existing_fads2_path = os.path.join(DATA_DIR, 'fads2_bioactivity.csv')
    existing_fads2 = pd.read_csv(existing_fads2_path)
    print(f"\nExisting fads2_bioactivity.csv: {len(existing_fads2)} compounds")

    # Canonicalize existing SMILES
    canon_existing = []
    valid_idx = []
    for i, smi in enumerate(existing_fads2['SMILES']):
        mol = MolFromSmiles(smi)
        if mol:
            canon_existing.append(MolToSmiles(mol))
            valid_idx.append(i)
    existing_fads2 = existing_fads2.iloc[valid_idx].copy()
    existing_fads2['SMILES'] = canon_existing
    existing_fads2['source'] = 'original_fads2'

    # Read existing fatty acid desaturase dataset (1187 compounds)
    existing_fad_path = os.path.join(DATA_DIR, 'fatty_acid_desaturase_bioactivity.csv')
    existing_fad = pd.read_csv(existing_fad_path)
    print(f"Existing fatty_acid_desaturase_bioactivity.csv: {len(existing_fad)} compounds")

    # Canonicalize
    canon_fad = []
    valid_idx2 = []
    for i, smi in enumerate(existing_fad['SMILES']):
        mol = MolFromSmiles(smi)
        if mol:
            canon_fad.append(MolToSmiles(mol))
            valid_idx2.append(i)
    existing_fad = existing_fad.iloc[valid_idx2].copy()
    existing_fad['SMILES'] = canon_fad
    existing_fad['source'] = 'original_fatty_acid_desaturase'

    # Merge all FADS2-relevant data
    # Combine: original_fads2 + original_fatty_acid_desaturase + ChEMBL_FADS2 + ChEMBL_FADS1
    all_dfs = []

    # Add existing datasets (only SMILES and pIC50 columns needed)
    for label, edf in [('original_fads2', existing_fads2), ('original_fatty_acid_desaturase', existing_fad)]:
        tmp = pd.DataFrame({
            'SMILES': edf['SMILES'],
            'pIC50': edf['pIC50'],
            'source': label
        })
        all_dfs.append(tmp)

    # Add ChEMBL data
    for label, cdf in [('ChEMBL_FADS2', fads2_df), ('ChEMBL_FADS1', fads1_df)]:
        if len(cdf) > 0:
            tmp = pd.DataFrame({
                'SMILES': cdf['SMILES'],
                'pIC50': cdf['pIC50'],
                'source': label
            })
            all_dfs.append(tmp)

    merged = pd.concat(all_dfs, ignore_index=True)
    print(f"\nBefore deduplication: {len(merged)} total rows")

    # Count overlaps
    smiles_sets = {}
    for label, edf in [('original_fads2', existing_fads2),
                         ('original_fatty_acid_desaturase', existing_fad)]:
        smiles_sets[label] = set(edf['SMILES'])
    if len(fads2_df) > 0:
        smiles_sets['ChEMBL_FADS2'] = set(fads2_df['SMILES'])
    if len(fads1_df) > 0:
        smiles_sets['ChEMBL_FADS1'] = set(fads1_df['SMILES'])

    print("\nOverlap analysis:")
    labels = list(smiles_sets.keys())
    for i in range(len(labels)):
        for j in range(i+1, len(labels)):
            overlap = smiles_sets[labels[i]] & smiles_sets[labels[j]]
            print(f"  {labels[i]} & {labels[j]}: {len(overlap)} shared SMILES")

    # Deduplicate by SMILES (keep median pIC50)
    expanded = merged.groupby('SMILES').agg(
        pIC50=('pIC50', 'median'),
        source=('source', lambda x: '+'.join(sorted(set(x))))
    ).reset_index()
    expanded['pIC50'] = expanded['pIC50'].round(4)

    expanded_path = os.path.join(DATA_DIR, 'fads2_expanded_bioactivity.csv')
    expanded.to_csv(expanded_path, index=False)
    print(f"\nSaved expanded dataset: {len(expanded)} compounds -> {expanded_path}")

    # ==========================================
    # Step 6: Print summary
    # ==========================================
    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print(f"{'='*60}")

    print(f"\nSource breakdown:")
    print(f"  original_fads2:                  {len(existing_fads2):>6d} compounds")
    print(f"  original_fatty_acid_desaturase:  {len(existing_fad):>6d} compounds")
    print(f"  ChEMBL_FADS2 (new):              {len(fads2_df):>6d} compounds")
    print(f"  ChEMBL_FADS1 (new):              {len(fads1_df):>6d} compounds")

    print(f"\nOutput files:")
    print(f"  fads2_chembl_bioactivity.csv:     {len(fads2_df):>6d} compounds")
    print(f"  fads1_chembl_bioactivity.csv:     {len(fads1_df):>6d} compounds")
    print(f"  fads_combined_bioactivity.csv:    {len(combined_dedup):>6d} compounds")
    print(f"  fads2_expanded_bioactivity.csv:   {len(expanded):>6d} compounds (all sources merged)")

    print(f"\nExpanded dataset statistics:")
    print(f"  Total compounds: {len(expanded)}")
    print(f"  pIC50 range:     [{expanded['pIC50'].min():.4f}, {expanded['pIC50'].max():.4f}]")
    print(f"  pIC50 mean:      {expanded['pIC50'].mean():.4f}")
    print(f"  pIC50 std:       {expanded['pIC50'].std():.4f}")
    print(f"  pIC50 median:    {expanded['pIC50'].median():.4f}")

    # Per-source stats for new ChEMBL data
    for label, cdf in [('ChEMBL FADS2', fads2_df), ('ChEMBL FADS1', fads1_df)]:
        if len(cdf) > 0:
            print(f"\n  {label} stats:")
            print(f"    Compounds: {len(cdf)}")
            print(f"    pIC50 range: [{cdf['pIC50'].min():.4f}, {cdf['pIC50'].max():.4f}]")
            print(f"    pIC50 mean:  {cdf['pIC50'].mean():.4f}")
            print(f"    pIC50 std:   {cdf['pIC50'].std():.4f}")
            print(f"    Assay types: {dict(cdf['standard_type'].value_counts())}")


if __name__ == '__main__':
    main()
