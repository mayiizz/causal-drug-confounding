#!/usr/bin/env python3
"""
Phase 7: Validation & Cross-Dataset Reproducibility.
Pathway validation (A/B/C/D) and cross-dataset rank correlation.
Includes supplementary analysis: direct DR-ATE correlation across datasets.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from scipy.stats import spearmanr, kendalltau
from src.validation.pathway_validation import run_pathway_validation
from src.validation.cross_dataset import compute_cross_dataset_reproducibility

PROCESSED = project_root / 'data' / 'processed'


def main():
    print("=" * 60)
    print("PHASE 7: Validation & Cross-Dataset Reproducibility")
    print("=" * 60)
    
    # ------------------------------------------------------------------
    # 1. Pathway Validation
    # ------------------------------------------------------------------
    print("\n[1/3] Pathway validation (Model A/B/C/D)...")
    
    # Select a few key cohorts for deep validation
    test_cases = [
        ('GDSC2', 'EGFR_Inhibitor', 'HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION'),
        ('GDSC2', 'PI3K_Inhibitor', 'HALLMARK_PI3K_AKT_MTOR_SIGNALING'),
        ('CCLE', 'EGFR_Inhibitor', 'HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION'),
        ('CCLE', 'MEK_Inhibitor', 'HALLMARK_KRAS_SIGNALING_UP'),
    ]
    
    all_val_results = []
    
    for dataset, drug_class, pathway in test_cases:
        pw_short = pathway.replace('HALLMARK_', '')
        fpath = PROCESSED / f'{dataset.lower()}_cohort_{pw_short}.parquet'
        if not fpath.exists():
            continue
        
        df = pd.read_parquet(fpath)
        sub = df[df['drug_class'] == drug_class].copy()
        
        if len(sub) < 60:
            continue
        
        print(f"  {dataset} {drug_class} × {pw_short}...")
        val_df = run_pathway_validation(sub, drug_class, pathway, dataset)
        all_val_results.append(val_df)
    
    if all_val_results:
        val_full = pd.concat(all_val_results, ignore_index=True)
        val_full.to_parquet(PROCESSED / 'pathway_validation.parquet', index=False)
        
        print("\n  Pathway validation results:")
        for (dataset, drug_class, pathway), group in val_full.groupby(['dataset', 'drug_class', 'pathway']):
            print(f"\n    {dataset} {drug_class} × {pathway}:")
            print(group[['model', 'r2', 'rmse', 'n']].to_string(index=False))
            
            # Check if Model B (true pathway) performs best
            b_r2 = group[group['model'] == 'B_tissue_true_pathway']['r2'].values
            a_r2 = group[group['model'] == 'A_tissue_only']['r2'].values
            c_r2 = group[group['model'] == 'C_tissue_permuted_pathway']['r2'].values
            
            if len(b_r2) > 0 and len(a_r2) > 0 and len(c_r2) > 0:
                if b_r2[0] > a_r2[0] and b_r2[0] > c_r2[0]:
                    print("    ✅ Model B (true pathway) performs best")
                else:
                    print("    ⚠️  Model B does NOT perform best")
    
    # ------------------------------------------------------------------
    # 2. Cross-Dataset Reproducibility: |Naive − DR| Divergence
    # ------------------------------------------------------------------
    print("\n[2/3] Cross-dataset reproducibility (divergence |Naive − DR|)...")
    
    comparison = pd.read_parquet(PROCESSED / 'ate_comparison.parquet')
    gdsc_comp = comparison[comparison['dataset'] == 'GDSC2']
    ccle_comp = comparison[comparison['dataset'] == 'CCLE']
    
    repro = compute_cross_dataset_reproducibility(gdsc_comp, ccle_comp)
    
    print(f"\n  Common cohorts: {repro['n_common']}")
    print(f"  Spearman ρ: {repro['spearman_rho']:.3f} (p={repro['spearman_p']:.4f})")
    print(f"  Kendall τ:  {repro['kendall_tau']:.3f} (p={repro['kendall_p']:.4f})")
    
    # Save merged data for Figure 1
    if repro['merged_data'] is not None:
        repro['merged_data'].to_parquet(PROCESSED / 'cross_dataset_reproducibility.parquet', index=False)
    
    # ------------------------------------------------------------------
    # 3. SUPPLEMENTARY: Cross-Dataset Reproducibility of DR ATEs Directly
    # ------------------------------------------------------------------
    print("\n[3/3] SUPPLEMENTARY: Cross-dataset correlation of DR ATEs directly...")
    
    try:
        estimates = pd.read_parquet(PROCESSED / 'causal_estimates.parquet')
        dr_estimates = estimates[estimates['estimator'] == 'DR'].copy()
        
        # Pivot to wide format: one row per (drug_class, pathway), columns per dataset
        dr_wide = dr_estimates.pivot_table(
            index=['drug_class', 'pathway'],
            columns='dataset',
            values='ate',
            aggfunc='first'
        ).reset_index().dropna(subset=['GDSC2', 'CCLE'])
        
        if len(dr_wide) >= 5:
            rho, pval = spearmanr(dr_wide['GDSC2'], dr_wide['CCLE'])
            tau, tpval = kendalltau(dr_wide['GDSC2'], dr_wide['CCLE'])
            
            print(f"\n  Shared DR-ATE cohorts: {len(dr_wide)}")
            print(f"  Spearman ρ: {rho:.3f} (p={pval:.4f})")
            print(f"  Kendall τ:  {tau:.3f} (p={tpval:.4f})")
            
            # Save supplementary results
            supp_results = pd.DataFrame({
                'metric': ['spearman_rho', 'spearman_p', 'kendall_tau', 'kendall_p', 'n_cohorts'],
                'value': [rho, pval, tau, tpval, len(dr_wide)]
            })
            supp_results.to_csv(PROCESSED / 'cross_dataset_dr_ate_correlation.csv', index=False)
            print(f"  Saved: {PROCESSED / 'cross_dataset_dr_ate_correlation.csv'}")
        else:
            print("  Not enough shared cohorts for direct DR-ATE correlation.")
            
    except Exception as e:
        print(f"  Skipped supplementary analysis: {e}")
    
    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("PHASE 7 COMPLETE")
    print("=" * 60)
    
    print(f"\nSaved:")
    print(f"  {PROCESSED / 'pathway_validation.parquet'}")
    print(f"  {PROCESSED / 'cross_dataset_reproducibility.parquet'}")
    
    print("\nNext: Phase 8 (Power & Stability Analysis)")


if __name__ == '__main__':
    main()