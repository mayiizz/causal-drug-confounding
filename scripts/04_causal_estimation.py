#!/usr/bin/env python3
"""
Phase 4: Causal Estimation.
Estimates Naive, IPW, and DR ATEs with bootstrap CIs.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from src.causal.common_support import fit_propensity_model, trim_by_propensity
from src.causal.bootstrap import bootstrap_all_estimators

PROCESSED = project_root / 'data' / 'processed'


def main():
    print("=" * 60)
    print("PHASE 4: Causal Estimation")
    print("=" * 60)
    
    # ------------------------------------------------------------------
    # 1. Load common support report to filter viable cohorts
    # ------------------------------------------------------------------
    print("\n[1/4] Loading common support report...")
    cs_report = pd.read_parquet(PROCESSED / 'common_support_default.parquet')
    
    # Keep only cohorts with overlap >= 0.5
    viable = cs_report[cs_report['overlap_coefficient'] >= 0.5].copy()
    print(f"  Viable cohorts (overlap >= 0.5): {len(viable)}")
    print(f"  Excluded: {len(cs_report) - len(viable)}")
    
    # ------------------------------------------------------------------
    # 2. Prepare cohorts with propensity scores
    # ------------------------------------------------------------------
    print("\n[2/4] Preparing cohorts with propensity scores...")
    
    pathways = [
        'HALLMARK_APOPTOSIS',
        'HALLMARK_DNA_REPAIR',
        'HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION',
        'HALLMARK_KRAS_SIGNALING_UP',
        'HALLMARK_PI3K_AKT_MTOR_SIGNALING'
    ]
    
    all_results = []
    
    for dataset in ['GDSC2', 'CCLE']:
        print(f"\n  Processing {dataset}...")
        
        for pw in pathways:
            pw_short = pw.replace('HALLMARK_', '')
            fpath = PROCESSED / f'{dataset.lower()}_cohort_{pw_short}.parquet'
            if not fpath.exists():
                continue
            
            df = pd.read_parquet(fpath)
            
            # Get viable drug classes for this pathway
            viable_classes = viable[
                (viable['dataset'] == dataset) & 
                (viable['pathway'] == pw)
            ]['drug_class'].unique()
            
            for drug_class in viable_classes:
                sub = df[df['drug_class'] == drug_class].copy()
                if len(sub) < 60:
                    continue
                
                # Fit propensity model
                propensity, _, _ = fit_propensity_model(sub)
                sub['propensity'] = propensity
                
                # Trim by default bounds
                trimmed, _ = trim_by_propensity(sub, bounds=(0.10, 0.90))
                if len(trimmed) < 60:
                    continue
                
                # Bootstrap all estimators
                results = bootstrap_all_estimators(
                    trimmed, drug_class, pw, dataset,
                    n_bootstrap=200,  # 200 for speed; use 500 for final
                    random_state=42
                )
                all_results.extend(results)
                
                if len(results) > 0:
                    print(f"    {drug_class} × {pw_short}: "
                          f"naive={results[0]['ate']:.3f}, "
                          f"DR={results[-1]['ate']:.3f}")
    
    # ------------------------------------------------------------------
    # 3. Compile results
    # ------------------------------------------------------------------
    print("\n[3/4] Compiling results...")
    results_df = pd.DataFrame(all_results)
    
    if results_df.empty:
        print("ERROR: No estimates generated.")
        return
    
    results_df.to_parquet(PROCESSED / 'causal_estimates.parquet', index=False)
    
    # ------------------------------------------------------------------
    # 4. Summary and divergence analysis
    # ------------------------------------------------------------------
    print("\n[4/4] Summary statistics...")
    
    # Pivot for comparison
    pivot = results_df.pivot_table(
        index=['dataset', 'drug_class', 'pathway'],
        columns='estimator',
        values='ate'
    ).reset_index()
    
    pivot['naive_dr_divergence'] = (pivot['Naive'] - pivot['DR']).abs()
    pivot.to_parquet(PROCESSED / 'ate_comparison.parquet', index=False)
    
    print("\n" + "=" * 60)
    print("PHASE 4 COMPLETE")
    print("=" * 60)
    print(f"Total estimates: {len(results_df)}")
    print(f"Unique cohorts: {results_df[['dataset','drug_class','pathway']].drop_duplicates().shape[0]}")
    
    print("\n--- Estimator summary ---")
    summary = results_df.groupby(['dataset', 'estimator']).agg({
        'ate': ['mean', 'std', 'count'],
        'ci_lower': 'mean',
        'ci_upper': 'mean'
    }).round(3)
    print(summary.to_string())
    
    print("\n--- Top divergences (|Naive - DR|) ---")
    top_div = pivot.nlargest(10, 'naive_dr_divergence')[['dataset', 'drug_class', 'pathway', 'naive_dr_divergence']]
    print(top_div.to_string(index=False))
    
    # Pilot check
    print("\n--- Pilot benchmark check ---")
    pilot = results_df[
        (results_df['drug_class'] == 'EGFR_Inhibitor') &
        (results_df['pathway'] == 'HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION')
    ]
    if not pilot.empty:
        for _, row in pilot.iterrows():
            print(f"  {row['dataset']} {row['estimator']}: ATE={row['ate']:.3f} "
                  f"[{row['ci_lower']:.3f}, {row['ci_upper']:.3f}]")
    else:
        print("  EGFR × EMT not found in viable cohorts (check overlap filtering)")
    
    print(f"\nSaved: {PROCESSED / 'causal_estimates.parquet'}")
    print(f"Saved: {PROCESSED / 'ate_comparison.parquet'}")
    print("\nNext: Phase 5 (Robustness & Refutation)")


if __name__ == '__main__':
    main()