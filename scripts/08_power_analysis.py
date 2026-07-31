#!/usr/bin/env python3
"""
Phase 8: Power and Stability Analysis.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from src.analysis.power import compute_power_metrics
from src.causal.common_support import fit_propensity_model, trim_by_propensity

PROCESSED = project_root / 'data' / 'processed'


def main():
    print("=" * 60)
    print("PHASE 8: Power & Stability Analysis")
    print("=" * 60)
    
    estimates = pd.read_parquet(PROCESSED / 'causal_estimates.parquet')
    
    all_metrics = []
    for _, row in estimates.iterrows():
        dataset = row['dataset']
        drug_class = row['drug_class']
        pathway = row['pathway']
        
        # Load cohort
        pw_short = pathway.replace('HALLMARK_', '')
        fpath = PROCESSED / f'{dataset.lower()}_cohort_{pw_short}.parquet'
        if not fpath.exists():
            continue
        
        df = pd.read_parquet(fpath)
        sub = df[df['drug_class'] == drug_class].copy()
        
        if len(sub) < 30:
            continue
        
        # Add propensity for metrics
        try:
            propensity, _, _ = fit_propensity_model(sub)
            sub['propensity'] = propensity
        except Exception:
            pass
        
        metrics = compute_power_metrics(sub, drug_class, pathway, dataset)
        metrics['ate'] = row['ate']
        metrics['se_boot'] = row.get('se_boot', np.nan)
        all_metrics.append(metrics)
    
    power_df = pd.DataFrame(all_metrics)
    power_df.to_parquet(PROCESSED / 'power_analysis.parquet', index=False)
    
    # Summary
    print("\n--- Power summary ---")
    print(f"Total cohorts: {len(power_df)}")
    print(f"Adequately powered: {power_df['power_adequate'].sum()}")
    print(f"Underpowered: {(~power_df['power_adequate']).sum()}")
    
    print("\n--- By dataset ---")
    summary = power_df.groupby('dataset').agg({
        'n_total': ['mean', 'min', 'max'],
        'ci_width': 'mean',
        'power_adequate': 'sum'
    }).round(1)
    print(summary.to_string())
    
    print(f"\nSaved: {PROCESSED / 'power_analysis.parquet'}")
    print("\nNext: Phase 9 (Visualization & Paper Artifacts)")


if __name__ == '__main__':
    main()