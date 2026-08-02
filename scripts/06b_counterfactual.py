#!/usr/bin/env python3
"""
Phase 6B: Counterfactual Analysis.
Individual-level outcomes under do(Treatment = 1) via DR CATE imputation.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
from src.causal.counterfactual import run_counterfactual_pipeline

PROCESSED = project_root / 'data' / 'processed'

PATHWAYS = [
    'HALLMARK_APOPTOSIS',
    'HALLMARK_DNA_REPAIR',
    'HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION',
    'HALLMARK_KRAS_SIGNALING_UP',
    'HALLMARK_PI3K_AKT_MTOR_SIGNALING',
]


def main():
    print("=" * 60)
    print("PHASE 6B: Counterfactual Analysis")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Load viable cohorts from common support
    # ------------------------------------------------------------------
    print("\n[1/3] Loading common support report...")
    cs_path = PROCESSED / 'common_support_default.parquet'
    if not cs_path.exists():
        print(f"ERROR: Missing {cs_path}. Run Phase 3 first.")
        return

    cs_report = pd.read_parquet(cs_path)
    viable = cs_report[cs_report['overlap_coefficient'] >= 0.5].copy()
    print(f"  Viable cohorts (overlap >= 0.5): {len(viable)}")
    print(f"  Excluded: {len(cs_report) - len(viable)}")

    # ------------------------------------------------------------------
    # 2. Counterfactual predictions
    # ------------------------------------------------------------------
    print("\n[2/3] Estimating individual counterfactuals under do(T=1)...")
    pred_df, summary_df = run_counterfactual_pipeline(
        viable, PATHWAYS, PROCESSED
    )

    if pred_df.empty:
        print("ERROR: No counterfactual predictions generated.")
        return

    pred_df.to_parquet(PROCESSED / 'counterfactual_predictions.parquet', index=False)
    summary_df.to_parquet(PROCESSED / 'counterfactual_summary.parquet', index=False)

    # ------------------------------------------------------------------
    # 3. Summary statistics
    # ------------------------------------------------------------------
    print("\n[3/3] Summary statistics...")
    global_row = summary_df[summary_df['level'] == 'global']
    if not global_row.empty:
        g = global_row.iloc[0]
        print(f"  N predictions: {int(g['n']):,} (valid={int(g['n_valid']):,})")
        print(f"  Mean delta_high:    {g['mean_delta_high']:.3f}")
        print(f"  Median delta_high:  {g['median_delta_high']:.3f}")
        print(f"  SD delta_high:      {g['sd_delta_high']:.3f}")
        print(f"  Mean delta_low:     {g['mean_delta_low']:.3f}")
        print(f"  Mean Y(1)-Y(0):     {g['mean_individual_effect']:.3f}")
        print(f"  Prop |delta_high|>1: {g['prop_large_shift']:.1%}")

    # Results-ready ranked tables
    top_tissues = (
        summary_df[summary_df['level'] == 'top_tissue']
        .sort_values('rank')
    )
    print("\n--- Top affected tissues (rank by mean |delta_high|, valid rows) ---")
    if not top_tissues.empty:
        cols = ['rank', 'dataset', 'tissue_type', 'n_valid', 'mean_delta_high',
                'mean_abs_delta_high', 'prop_large_shift']
        print(top_tissues[cols].to_string(index=False))
        top_tissues[cols].to_csv(
            PROCESSED / 'counterfactual_top_tissues.csv', index=False
        )

    top_classes = (
        summary_df[summary_df['level'] == 'top_drug_class']
        .sort_values('rank')
    )
    print("\n--- Top affected drug classes (rank by mean |delta_high|, valid rows) ---")
    if not top_classes.empty:
        cols = ['rank', 'dataset', 'drug_class', 'n_valid', 'mean_delta_high',
                'mean_abs_delta_high', 'prop_large_shift']
        print(top_classes[cols].to_string(index=False))
        top_classes[cols].to_csv(
            PROCESSED / 'counterfactual_top_drug_classes.csv', index=False
        )

    print("\n" + "=" * 60)
    print("PHASE 6B COMPLETE")
    print("=" * 60)
    print(f"Saved: {PROCESSED / 'counterfactual_predictions.parquet'}")
    print(f"Saved: {PROCESSED / 'counterfactual_summary.parquet'}")
    print(f"Saved: {PROCESSED / 'counterfactual_top_tissues.csv'}")
    print(f"Saved: {PROCESSED / 'counterfactual_top_drug_classes.csv'}")
    print("\nNext: Phase 7 (Validation & Cross-Dataset Reproducibility)")


if __name__ == '__main__':
    main()
