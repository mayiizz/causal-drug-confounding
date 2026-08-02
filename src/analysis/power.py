"""Power and stability analysis per drug class.

Adequacy labels (deterministic)
-------------------------------
Evaluated on the trimmed analysis sample (default propensity 0.10–0.90)
plus overlap / ESS / bootstrap CI width when available:

Excellent : n_trimmed >= 200, n_treated >= 50, n_control >= 50,
            overlap >= 0.70, ESS >= 100
Good      : n_trimmed >= 150, n_treated >= 40, n_control >= 40,
            overlap >= 0.60, ESS >= 80
Adequate  : n_trimmed >= 100, n_treated >= 30, n_control >= 30,
            overlap >= 0.50, ESS >= 50
Marginal  : n_trimmed >= 60,  n_treated >= 20, n_control >= 20,
            overlap >= 0.40
Underpowered : otherwise

Missing overlap/ESS/CI do not upgrade a label; they only block higher tiers
when the metric is present and fails the threshold.
"""

import pandas as pd
import numpy as np


def compute_power_metrics(df, drug_class, pathway, dataset_name):
    """
    Legacy helper: sample size, CI width (if columns present), propensity spread.

    Prefer ``build_cohort_power_record`` for the enhanced PRD metrics.
    """
    n = len(df)
    n_treated = int(df['treatment'].sum())
    n_control = n - n_treated

    if 'ci_lower' in df.columns and 'ci_upper' in df.columns:
        ci_width = (df['ci_upper'] - df['ci_lower']).mean()
    else:
        ci_width = np.nan

    if 'propensity' in df.columns:
        prop_range = df['propensity'].max() - df['propensity'].min()
        prop_std = df['propensity'].std()
        prop_mean = df['propensity'].mean()
    else:
        prop_range = np.nan
        prop_std = np.nan
        prop_mean = np.nan

    return {
        'dataset': dataset_name,
        'drug_class': drug_class,
        'pathway': pathway,
        'n_total': n,
        'n_treated': n_treated,
        'n_control': n_control,
        'treated_ratio': n_treated / n if n > 0 else np.nan,
        'ci_width': ci_width,
        'propensity_mean': prop_mean,
        'propensity_range': prop_range,
        'propensity_std': prop_std,
        'power_adequate': n >= 100 and n_treated >= 30 and n_control >= 30,
    }


def compute_ess(weights):
    """Effective sample size: (sum w)^2 / sum(w^2)."""
    w = np.asarray(weights, dtype=float)
    w = w[np.isfinite(w)]
    if len(w) == 0 or np.sum(w ** 2) <= 0:
        return np.nan
    return float((np.sum(w) ** 2) / np.sum(w ** 2))


def assign_adequacy_label(n_trimmed, n_treated, n_control, overlap, ess,
                          ci_width=np.nan):
    """
    Deterministic adequacy label from PRD-oriented thresholds.

    CI width is informational for reporting; it does not downgrade labels
    unless provided and extremely wide (> 3.0) for Excellent/Good tiers.
    """
    def _ok(n_min, t_min, c_min, o_min, ess_min, max_ci=None):
        if n_trimmed < n_min or n_treated < t_min or n_control < c_min:
            return False
        if np.isfinite(overlap) and overlap < o_min:
            return False
        if np.isfinite(ess) and ess < ess_min:
            return False
        if max_ci is not None and np.isfinite(ci_width) and ci_width > max_ci:
            return False
        return True

    if _ok(200, 50, 50, 0.70, 100, max_ci=2.0):
        return 'Excellent'
    if _ok(150, 40, 40, 0.60, 80, max_ci=2.5):
        return 'Good'
    if _ok(100, 30, 30, 0.50, 50):
        return 'Adequate'
    if _ok(60, 20, 20, 0.40, 0):
        return 'Marginal'
    return 'Underpowered'


def build_cohort_power_record(
    df_raw,
    drug_class,
    pathway,
    dataset_name,
    n_original,
    n_trimmed,
    percent_retained,
    overlap_coefficient,
    ci_width,
    propensity_mean,
    propensity_std,
    ess,
    n_treated,
    n_control,
):
    """Assemble one enhanced power row for a cohort."""
    n_trim = int(n_trimmed) if pd.notna(n_trimmed) else len(df_raw)
    n_t = int(n_treated)
    n_c = int(n_control)
    n_tot = n_t + n_c if (n_t + n_c) > 0 else n_trim

    label = assign_adequacy_label(
        n_trim, n_t, n_c,
        overlap_coefficient if pd.notna(overlap_coefficient) else np.nan,
        ess if pd.notna(ess) else np.nan,
        ci_width=ci_width if pd.notna(ci_width) else np.nan,
    )

    return {
        'dataset': dataset_name,
        'drug_class': drug_class,
        'pathway': pathway,
        'n_original': int(n_original) if pd.notna(n_original) else len(df_raw),
        'n_trimmed': n_trim,
        'percent_retained': float(percent_retained) if pd.notna(percent_retained) else np.nan,
        'n_treated': n_t,
        'n_control': n_c,
        'treatment_proportion': n_t / n_tot if n_tot > 0 else np.nan,
        'ci_width': float(ci_width) if pd.notna(ci_width) else np.nan,
        'ess': float(ess) if pd.notna(ess) else np.nan,
        'propensity_mean': float(propensity_mean) if pd.notna(propensity_mean) else np.nan,
        'propensity_std': float(propensity_std) if pd.notna(propensity_std) else np.nan,
        'overlap_coefficient': float(overlap_coefficient) if pd.notna(overlap_coefficient) else np.nan,
        'adequacy_label': label,
        # Backward-compatible fields
        'n_total': n_trim,
        'treated_ratio': n_t / n_tot if n_tot > 0 else np.nan,
        'power_adequate': label in ('Excellent', 'Good', 'Adequate'),
    }


def summarize_power(power_df):
    """Cohort counts by adequacy label (and by dataset)."""
    if power_df is None or power_df.empty:
        return pd.DataFrame()

    rows = []
    order = ['Excellent', 'Good', 'Adequate', 'Marginal', 'Underpowered']

    # Global
    vc = power_df['adequacy_label'].value_counts()
    for lab in order:
        rows.append({
            'level': 'global',
            'dataset': 'ALL',
            'adequacy_label': lab,
            'n_cohorts': int(vc.get(lab, 0)),
            'pct_cohorts': float(vc.get(lab, 0) / len(power_df)),
        })

    for dataset, g in power_df.groupby('dataset'):
        vc = g['adequacy_label'].value_counts()
        for lab in order:
            rows.append({
                'level': 'dataset',
                'dataset': dataset,
                'adequacy_label': lab,
                'n_cohorts': int(vc.get(lab, 0)),
                'pct_cohorts': float(vc.get(lab, 0) / len(g)),
            })

    # Numeric means for Results
    rows.append({
        'level': 'metrics',
        'dataset': 'ALL',
        'adequacy_label': 'MEAN',
        'n_cohorts': len(power_df),
        'pct_cohorts': np.nan,
        'mean_n_trimmed': power_df['n_trimmed'].mean(),
        'mean_ess': power_df['ess'].mean(),
        'mean_ci_width': power_df['ci_width'].mean(),
        'mean_overlap': power_df['overlap_coefficient'].mean(),
        'mean_percent_retained': power_df['percent_retained'].mean(),
    })

    return pd.DataFrame(rows)
