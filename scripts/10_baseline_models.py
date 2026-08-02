#!/usr/bin/env python3
"""
Phase 10: Predictive baselines (SVR + XGBoost).

Completely independent of causal estimators — predictive context only.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
from src.baselines.svr import run_svr_baseline
from src.baselines.xgboost_baseline import run_xgboost_baseline

PROCESSED = project_root / 'data' / 'processed'
OUTPUT = project_root / 'output'
OUTPUT.mkdir(exist_ok=True)


def main():
    print("=" * 60)
    print("PHASE 10: Predictive Baselines (SVR + XGBoost)")
    print("=" * 60)

    print("\n[1/2] SVR baseline...")
    try:
        svr_results, svr_preds = run_svr_baseline(PROCESSED)
        svr_results.to_parquet(PROCESSED / 'baseline_svr_results.parquet', index=False)
        svr_preds.to_parquet(PROCESSED / 'baseline_svr_predictions.parquet', index=False)
        print(f"  Saved: {PROCESSED / 'baseline_svr_results.parquet'}")
    except Exception as e:
        print(f"  SVR FAILED: {e}")
        svr_results, svr_preds = pd.DataFrame(), pd.DataFrame()

    print("\n[2/2] XGBoost baseline...")
    try:
        xgb_results, xgb_preds = run_xgboost_baseline(PROCESSED)
        xgb_results.to_parquet(PROCESSED / 'baseline_xgboost_results.parquet', index=False)
        xgb_preds.to_parquet(PROCESSED / 'baseline_xgboost_predictions.parquet', index=False)
        print(f"  Saved: {PROCESSED / 'baseline_xgboost_results.parquet'}")
    except Exception as e:
        print(f"  XGBoost FAILED: {e}")
        xgb_results, xgb_preds = pd.DataFrame(), pd.DataFrame()

    if not svr_results.empty or not xgb_results.empty:
        combined = pd.concat(
            [d for d in [svr_results, xgb_results] if not d.empty],
            ignore_index=True,
        )
        combined.to_csv(OUTPUT / 'table5_baseline_metrics.csv', index=False)
        print(f"  Saved: {OUTPUT / 'table5_baseline_metrics.csv'}")
        print("\n--- Baseline metrics ---")
        print(combined[[
            'dataset', 'model', 'rmse', 'mae', 'r2', 'pearson_r', 'spearman_rho', 'n_test'
        ]].to_string(index=False))

    print("\n" + "=" * 60)
    print("PHASE 10 COMPLETE")
    print("=" * 60)
    print("\nNext: re-run Phase 9 to refresh baseline figures, or run figure helpers.")


if __name__ == '__main__':
    main()
