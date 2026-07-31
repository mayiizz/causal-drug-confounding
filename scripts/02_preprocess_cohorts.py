#!/usr/bin/env python3
"""
Phase 2: Preprocessing & Cohort Definition.
Binarizes pathway scores and selects valid drug-class cohorts.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
from src.preprocessing.binarize_treatment import binarize_all_pathways
from src.preprocessing.select_drug_classes import select_drug_classes, find_cross_dataset_cohorts

PROCESSED = project_root / 'data' / 'processed'
PROCESSED.mkdir(parents=True, exist_ok=True)


def main():
    print("=" * 60)
    print("PHASE 2: Preprocessing & Cohort Definition")
    print("=" * 60)
    
    # ------------------------------------------------------------------
    # 1. Load unified data
    # ------------------------------------------------------------------
    print("\n[1/4] Loading unified datasets...")
    gdsc2 = pd.read_parquet(PROCESSED / 'gdsc2_unified.parquet')
    ccle = pd.read_parquet(PROCESSED / 'ccle_unified.parquet')
    print(f"  GDSC2: {gdsc2.shape[0]:,} rows")
    print(f"  CCLE:  {ccle.shape[0]:,} rows")
    
    # Identify pathway columns
    pw_cols = [c for c in gdsc2.columns if c.startswith('HALLMARK_')]
    print(f"  Pathways: {pw_cols}")
    
    # ------------------------------------------------------------------
    # 2. Binarize GDSC2
    # ------------------------------------------------------------------
    print("\n[2/4] Binarizing GDSC2...")
    gdsc_results = binarize_all_pathways(gdsc2, pathway_cols=pw_cols)
    gdsc_selected = select_drug_classes(gdsc_results)
    
    # Save per-pathway tables
    for pw, info in gdsc_selected.items():
        pw_short = pw.replace('HALLMARK_', '')
        info['data'].to_parquet(PROCESSED / f'gdsc2_cohort_{pw_short}.parquet', index=False)
        info['thresholds'].to_parquet(PROCESSED / f'gdsc2_thresholds_{pw_short}.parquet', index=False)
    
    # ------------------------------------------------------------------
    # 3. Binarize CCLE
    # ------------------------------------------------------------------
    print("\n[3/4] Binarizing CCLE...")
    ccle_results = binarize_all_pathways(ccle, pathway_cols=pw_cols)
    ccle_selected = select_drug_classes(ccle_results)
    
    for pw, info in ccle_selected.items():
        pw_short = pw.replace('HALLMARK_', '')
        info['data'].to_parquet(PROCESSED / f'ccle_cohort_{pw_short}.parquet', index=False)
        info['thresholds'].to_parquet(PROCESSED / f'ccle_thresholds_{pw_short}.parquet', index=False)
    
    # ------------------------------------------------------------------
    # 4. Cross-dataset cohort intersection
    # ------------------------------------------------------------------
    print("\n[4/4] Finding cross-dataset cohorts...")
    common_cohorts = find_cross_dataset_cohorts(gdsc_selected, ccle_selected)
    
    # Save cohort manifest
    manifest = pd.DataFrame(common_cohorts, columns=['drug_class', 'pathway'])
    manifest.to_parquet(PROCESSED / 'cross_dataset_cohorts.parquet', index=False)
    
    # Summary table
    summary = []
    for pw, info in gdsc_selected.items():
        summary.append({
            'dataset': 'GDSC2',
            'pathway': pw,
            'n_cohorts': info['n_drug_classes'],
            'n_total_rows': len(info['data'])
        })
    for pw, info in ccle_selected.items():
        summary.append({
            'dataset': 'CCLE',
            'pathway': pw,
            'n_cohorts': info['n_drug_classes'],
            'n_total_rows': len(info['data'])
        })
    
    summary_df = pd.DataFrame(summary)
    print("\n" + "=" * 60)
    print("PHASE 2 COMPLETE")
    print("=" * 60)
    print(summary_df.to_string(index=False))
    print(f"\nCross-dataset cohorts: {len(common_cohorts)}")
    print(f"Saved to: {PROCESSED / 'cross_dataset_cohorts.parquet'}")
    print("\nNext: Phase 3 (Common Support & Propensity Modeling)")


if __name__ == '__main__':
    main()