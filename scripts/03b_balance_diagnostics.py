#!/usr/bin/env python3
"""
Phase 3B: IPW post-weighting balance diagnostics (tissue SMD Love-plot inputs).

Runs after Phase 3 common support. Reuses the same propensity model,
trimming bounds (0.10–0.90), and stabilized IPW weights as the IPW estimator.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
from src.causal.balance import run_balance_pipeline

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
    print("PHASE 3B: IPW Balance Diagnostics (Tissue SMD)")
    print("=" * 60)

    print("\n[1/2] Loading viable cohorts from common support...")
    cs_path = PROCESSED / 'common_support_default.parquet'
    if not cs_path.exists():
        print(f"ERROR: Missing {cs_path}. Run Phase 3 first.")
        return

    cs = pd.read_parquet(cs_path)
    viable = cs[cs['overlap_coefficient'] >= 0.5].copy()
    print(f"  Viable cohorts (overlap >= 0.5): {len(viable)}")

    print("\n[2/2] Computing raw / trimmed / IPW-weighted tissue SMDs...")
    detail_df, summary_df = run_balance_pipeline(viable, PATHWAYS, PROCESSED)

    if detail_df.empty or summary_df.empty:
        print("ERROR: No balance diagnostics generated.")
        return

    detail_df.to_parquet(PROCESSED / 'balance_diagnostics.parquet', index=False)
    summary_df.to_parquet(PROCESSED / 'balance_summary.parquet', index=False)

    print("\n--- Balance summary (pooled) ---")
    print(
        f"  Mean of max |SMD|: "
        f"raw={summary_df['max_raw_smd'].mean():.3f}, "
        f"trim={summary_df['max_trimmed_smd'].mean():.3f}, "
        f"IPW={summary_df['max_weighted_smd'].mean():.3f}"
    )
    print(
        f"  Mean of mean |SMD|: "
        f"raw={summary_df['mean_raw_smd'].mean():.3f}, "
        f"trim={summary_df['mean_trimmed_smd'].mean():.3f}, "
        f"IPW={summary_df['mean_weighted_smd'].mean():.3f}"
    )
    print(
        f"  Cohorts with any tissue |SMD|>0.1 after IPW: "
        f"{(summary_df['n_above_0.1_after_weighting'] > 0).sum()}/{len(summary_df)}"
    )
    improved = (
        summary_df['max_weighted_smd'] < summary_df['max_raw_smd']
    ).sum()
    print(f"  Cohorts with max |SMD| improved vs raw: {improved}/{len(summary_df)}")

    print("\n" + "=" * 60)
    print("PHASE 3B COMPLETE")
    print("=" * 60)
    print(f"Saved: {PROCESSED / 'balance_diagnostics.parquet'}")
    print(f"Saved: {PROCESSED / 'balance_summary.parquet'}")
    print("\nNext: Phase 4 (Causal Estimation)")


if __name__ == '__main__':
    main()
