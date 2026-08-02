#!/usr/bin/env python3
"""
Phase 8: Power and Stability Analysis (enhanced PRD metrics).

Reuses common-support reports and bootstrap CIs from causal_estimates;
does not re-run bootstrapping.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from src.analysis.power import (
    build_cohort_power_record,
    compute_ess,
    summarize_power,
)
from src.causal.common_support import fit_propensity_model, trim_by_propensity
from src.causal.estimators import compute_stabilized_ipw_weights

PROCESSED = project_root / 'data' / 'processed'


def main():
    print("=" * 60)
    print("PHASE 8: Power & Stability Analysis")
    print("=" * 60)

    estimates = pd.read_parquet(PROCESSED / 'causal_estimates.parquet')
    cs_path = PROCESSED / 'common_support_default.parquet'
    if not cs_path.exists():
        print(f"ERROR: Missing {cs_path}. Run Phase 3 first.")
        return
    cs = pd.read_parquet(cs_path)

    # One row per cohort; prefer DR bootstrap CI width
    cohorts = estimates[['dataset', 'drug_class', 'pathway']].drop_duplicates()
    dr = estimates[estimates['estimator'] == 'DR'][
        ['dataset', 'drug_class', 'pathway', 'ci_lower', 'ci_upper', 'se_boot', 'n']
    ].copy()
    dr['ci_width'] = dr['ci_upper'] - dr['ci_lower']

    print(f"\n[1/2] Computing enhanced power metrics for {len(cohorts)} cohorts...")
    all_metrics = []

    for _, row in cohorts.iterrows():
        dataset = row['dataset']
        drug_class = row['drug_class']
        pathway = row['pathway']
        pw_short = pathway.replace('HALLMARK_', '')
        fpath = PROCESSED / f'{dataset.lower()}_cohort_{pw_short}.parquet'
        if not fpath.exists():
            continue

        df = pd.read_parquet(fpath)
        sub = df[df['drug_class'] == drug_class].copy()
        if len(sub) < 30:
            continue

        cs_row = cs[
            (cs['dataset'] == dataset) &
            (cs['drug_class'] == drug_class) &
            (cs['pathway'] == pathway)
        ]
        if cs_row.empty:
            n_original = len(sub)
            n_trimmed_cs = np.nan
            pct_ret = np.nan
            overlap = np.nan
        else:
            r0 = cs_row.iloc[0]
            n_original = r0['n_original']
            n_trimmed_cs = r0['n_trimmed']
            pct_ret = r0['percent_retained']
            overlap = r0['overlap_coefficient']

        dr_row = dr[
            (dr['dataset'] == dataset) &
            (dr['drug_class'] == drug_class) &
            (dr['pathway'] == pathway)
        ]
        ci_width = float(dr_row.iloc[0]['ci_width']) if not dr_row.empty else np.nan

        try:
            propensity, _, _ = fit_propensity_model(sub)
            sub = sub.copy()
            sub['propensity'] = propensity
            trimmed, _ = trim_by_propensity(sub, bounds=(0.10, 0.90))
            if len(trimmed) < 20 or trimmed['treatment'].nunique() < 2:
                trimmed = sub
            weights = compute_stabilized_ipw_weights(trimmed)
            ess = compute_ess(weights.values)
            prop_mean = float(trimmed['propensity'].mean())
            prop_std = float(trimmed['propensity'].std())
            n_treated = int(trimmed['treatment'].sum())
            n_control = int(len(trimmed) - n_treated)
            n_trim = len(trimmed)
            if pd.isna(n_trimmed_cs):
                n_trimmed_cs = n_trim
                pct_ret = 100.0 * n_trim / n_original if n_original else np.nan
        except Exception as e:
            print(f"  WARN {dataset} {drug_class} {pw_short}: {e}")
            ess = np.nan
            prop_mean = np.nan
            prop_std = np.nan
            n_treated = int(sub['treatment'].sum()) if 'treatment' in sub.columns else 0
            n_control = len(sub) - n_treated
            n_trim = int(n_trimmed_cs) if pd.notna(n_trimmed_cs) else len(sub)

        metrics = build_cohort_power_record(
            sub, drug_class, pathway, dataset,
            n_original=n_original,
            n_trimmed=n_trim,
            percent_retained=pct_ret,
            overlap_coefficient=overlap,
            ci_width=ci_width,
            propensity_mean=prop_mean,
            propensity_std=prop_std,
            ess=ess,
            n_treated=n_treated,
            n_control=n_control,
        )
        all_metrics.append(metrics)

    power_df = pd.DataFrame(all_metrics)
    if power_df.empty:
        print("ERROR: No power metrics generated.")
        return

    power_df.to_parquet(PROCESSED / 'power_analysis.parquet', index=False)
    summary_df = summarize_power(power_df)
    summary_df.to_parquet(PROCESSED / 'power_summary.parquet', index=False)

    print("\n--- Power summary by adequacy ---")
    print(
        power_df['adequacy_label']
        .value_counts()
        .reindex(['Excellent', 'Good', 'Adequate', 'Marginal', 'Underpowered'])
        .fillna(0)
        .astype(int)
        .to_string()
    )
    print("\n--- By dataset ---")
    print(
        power_df.groupby(['dataset', 'adequacy_label'])
        .size()
        .unstack(fill_value=0)
        .to_string()
    )

    print(f"\nSaved: {PROCESSED / 'power_analysis.parquet'}")
    print(f"Saved: {PROCESSED / 'power_summary.parquet'}")
    print("\nNext: Phase 9 (Visualization & Paper Artifacts)")


if __name__ == '__main__':
    main()
