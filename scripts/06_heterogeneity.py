#!/usr/bin/env python3
"""
Phase 6: Heterogeneous Treatment Effects.
Estimates CATE by tissue using CausalForestDML.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from src.causal.heterogeneity import run_cate_pipeline

PROCESSED = project_root / 'data' / 'processed'


def main():
    print("=" * 60)
    print("PHASE 6: Heterogeneous Treatment Effects")
    print("=" * 60)
    
    # ------------------------------------------------------------------
    # 1. Load cohorts and estimate CATE
    # ------------------------------------------------------------------
    print("\n[1/2] Estimating CATE for top cohorts...")
    
    all_cates = []
    
    for dataset in ['GDSC2', 'CCLE']:
        print(f"\n  {dataset}:")
        pathways = [
            'HALLMARK_APOPTOSIS',
            'HALLMARK_DNA_REPAIR',
            'HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION',
            'HALLMARK_KRAS_SIGNALING_UP',
            'HALLMARK_PI3K_AKT_MTOR_SIGNALING'
        ]
        
        for pw in pathways:
            pw_short = pw.replace('HALLMARK_', '')
            fpath = PROCESSED / f'{dataset.lower()}_cohort_{pw_short}.parquet'
            if not fpath.exists():
                continue
            
            df = pd.read_parquet(fpath)
            cate_df = run_cate_pipeline(df, dataset, top_n_cohorts=3)
            if not cate_df.empty:
                all_cates.append(cate_df)
    
    if not all_cates:
        print("ERROR: No CATE estimates generated.")
        return
    
    cate_full = pd.concat(all_cates, ignore_index=True)
    cate_full.to_parquet(PROCESSED / 'cate_estimates.parquet', index=False)
    
    # ------------------------------------------------------------------
    # 2. Build heatmap data
    # ------------------------------------------------------------------
    print("\n[2/2] Building heatmap data...")
    
    # Pivot: rows = drug_class × pathway, cols = tissue, values = CATE
    heatmap_data = []
    for (dataset, drug_class, pathway), group in cate_full.groupby(['dataset', 'drug_class', 'pathway']):
        row = {
            'dataset': dataset,
            'drug_class': drug_class,
            'pathway': pathway
        }
        for _, r in group.iterrows():
            row[r['tissue_type']] = r['cate_mean']
        heatmap_data.append(row)
    
    heatmap_df = pd.DataFrame(heatmap_data)
    heatmap_df.to_parquet(PROCESSED / 'cate_heatmap.parquet', index=False)
    
    # Summary
    print("\n" + "=" * 60)
    print("PHASE 6 COMPLETE")
    print("=" * 60)
    print(f"CATE records: {len(cate_full)}")
    print(f"Unique tissues with CATE: {cate_full['tissue_type'].nunique()}")
    print(f"Cohorts analyzed: {cate_full[['dataset','drug_class','pathway']].drop_duplicates().shape[0]}")
    
    print("\nTop positive CATEs (treatment most beneficial):")
    print(cate_full.nlargest(5, 'cate_mean')[['dataset', 'drug_class', 'pathway', 'tissue_type', 'cate_mean']].to_string(index=False))
    
    print("\nTop negative CATEs (treatment most harmful):")
    print(cate_full.nsmallest(5, 'cate_mean')[['dataset', 'drug_class', 'pathway', 'tissue_type', 'cate_mean']].to_string(index=False))
    
    print(f"\nSaved: {PROCESSED / 'cate_estimates.parquet'}")
    print(f"Saved: {PROCESSED / 'cate_heatmap.parquet'}")
    print("\nNext: Phase 6B (Counterfactual Analysis)")


if __name__ == '__main__':
    main()