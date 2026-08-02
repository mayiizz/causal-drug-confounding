"""
Post-weighting covariate balance diagnostics (Standardized Mean Difference).

For each tissue indicator, compute |SMD| at three stages:

1. Raw (before trimming / weighting)
2. After common-support trimming (default 0.10–0.90)
3. After IPW weighting on the trimmed sample

Weighted SMD uses the same unit-level IPW weights as ``ipw_ate`` /
``compute_stabilized_ipw_weights``:

    w_i = T_i / e_i           if T_i = 1
        = (1 - T_i) / (1 - e_i) if T_i = 0

    SMD_w = |m_t - m_c| / sqrt((v_t + v_c) / 2)

where m_*, v_* are weighted means and variances within arms.

Assumptions:
- Propensity e = P(T=1 | tissue) from the same logistic model as the pipeline.
- Tissue imbalance is assessed via one-hot tissue indicators.
- |SMD| < 0.1 is a conventional "adequate balance" threshold.
"""

import pandas as pd
import numpy as np
import warnings

from src.causal.common_support import (
    fit_propensity_model,
    trim_by_propensity,
    compute_smd,
)
from src.causal.estimators import compute_stabilized_ipw_weights

DEFAULT_BOUNDS = (0.10, 0.90)
DEFAULT_CLIP = (0.10, 0.90)
DEFAULT_MIN_N = 60


def _tissue_dummies(df, tissue_col='tissue_type'):
    """One-hot tissue indicators; columns are tissue names."""
    dummies = pd.get_dummies(df[tissue_col].astype(str), prefix='', prefix_sep='')
    # Drop all-zero columns if any
    dummies = dummies.loc[:, dummies.sum(axis=0) > 0]
    return dummies


def _smd_for_tissues(df, dummy_cols, treatment_col='treatment', weight_col=None):
    """
    Absolute SMD for each tissue indicator column present in df.

    Returns dict tissue -> smd (nan if undefined).
    """
    out = {}
    for col in dummy_cols:
        if col not in df.columns:
            out[col] = np.nan
            continue
        treated = df[df[treatment_col] == 1]
        control = df[df[treatment_col] == 0]
        if len(treated) == 0 or len(control) == 0:
            out[col] = np.nan
            continue
        # Need variation in at least one arm or pooled
        try:
            smd = compute_smd(
                df, col, treatment_col=treatment_col, weight_col=weight_col
            )
            out[col] = float(smd) if np.isfinite(smd) else np.nan
        except Exception:
            out[col] = np.nan
    return out


def assess_cohort_balance(df, drug_class, pathway, dataset_name,
                          tissue_col='tissue_type', treatment_col='treatment',
                          bounds=DEFAULT_BOUNDS, clip_bounds=DEFAULT_CLIP,
                          min_n=DEFAULT_MIN_N):
    """
    Tissue-level SMD at raw, trimmed, and IPW-weighted stages.

    Reuses ``fit_propensity_model``, ``trim_by_propensity``, and
    ``compute_stabilized_ipw_weights`` (same weights as IPW ATE).

    Returns
    -------
    detail_df : DataFrame
        One row per tissue.
    summary_row : dict
        Cohort-level summary metrics.
    """
    work = df.copy()
    if len(work) < min_n or work[treatment_col].nunique() < 2:
        return pd.DataFrame(), {}

    # Single propensity fit shared across stages (same model as pipeline)
    propensity, _, _ = fit_propensity_model(
        work, tissue_col=tissue_col, treatment_col=treatment_col
    )
    work['propensity'] = propensity

    # Tissue universe from the raw cohort
    dummies_raw = _tissue_dummies(work, tissue_col=tissue_col)
    tissue_names = list(dummies_raw.columns)
    if len(tissue_names) == 0:
        return pd.DataFrame(), {}

    raw = work.copy()
    for col in tissue_names:
        raw[col] = dummies_raw[col].values

    raw_smds = _smd_for_tissues(raw, tissue_names, treatment_col=treatment_col)

    # Trim with same bounds as main pipeline
    trimmed, _ = trim_by_propensity(work, propensity_col='propensity', bounds=bounds)
    if len(trimmed) < min_n or trimmed[treatment_col].nunique() < 2:
        warnings.warn(
            f"Balance: trim left unusable cohort "
            f"{dataset_name} {drug_class} {pathway}"
        )
        return pd.DataFrame(), {}

    dummies_trim = _tissue_dummies(trimmed, tissue_col=tissue_col)
    trim = trimmed.copy()
    for col in tissue_names:
        # Align to raw tissue set; missing tissue in trimmed → all zeros
        if col in dummies_trim.columns:
            trim[col] = dummies_trim[col].values
        else:
            trim[col] = 0

    trimmed_smds = _smd_for_tissues(trim, tissue_names, treatment_col=treatment_col)

    # IPW weights on trimmed sample (same formula as ipw_ate)
    trim = trim.copy()
    trim['ipw_weight'] = compute_stabilized_ipw_weights(
        trim,
        treatment_col=treatment_col,
        propensity_col='propensity',
        clip_bounds=clip_bounds,
    )
    weighted_smds = _smd_for_tissues(
        trim, tissue_names, treatment_col=treatment_col, weight_col='ipw_weight'
    )

    rows = []
    for tissue in tissue_names:
        rows.append({
            'dataset': dataset_name,
            'drug_class': drug_class,
            'pathway': pathway,
            'tissue': tissue,
            'raw_smd': raw_smds.get(tissue, np.nan),
            'trimmed_smd': trimmed_smds.get(tissue, np.nan),
            'weighted_smd': weighted_smds.get(tissue, np.nan),
        })

    detail = pd.DataFrame(rows)

    def _max_abs(series):
        s = series.dropna()
        return float(s.max()) if len(s) else np.nan

    def _mean_abs(series):
        s = series.dropna()
        return float(s.mean()) if len(s) else np.nan

    def _n_above(series, thr):
        s = series.dropna()
        return int((s > thr).sum())

    summary = {
        'dataset': dataset_name,
        'drug_class': drug_class,
        'pathway': pathway,
        'n_tissues': len(tissue_names),
        'n_raw': len(work),
        'n_trimmed': len(trim),
        'max_raw_smd': _max_abs(detail['raw_smd']),
        'max_trimmed_smd': _max_abs(detail['trimmed_smd']),
        'max_weighted_smd': _max_abs(detail['weighted_smd']),
        'mean_raw_smd': _mean_abs(detail['raw_smd']),
        'mean_trimmed_smd': _mean_abs(detail['trimmed_smd']),
        'mean_weighted_smd': _mean_abs(detail['weighted_smd']),
        'n_above_0.1_before': _n_above(detail['raw_smd'], 0.1),
        'n_above_0.1_after_trim': _n_above(detail['trimmed_smd'], 0.1),
        'n_above_0.1_after_weighting': _n_above(detail['weighted_smd'], 0.1),
        'n_above_0.05_before': _n_above(detail['raw_smd'], 0.05),
        'n_above_0.05_after_trim': _n_above(detail['trimmed_smd'], 0.05),
        'n_above_0.05_after_weighting': _n_above(detail['weighted_smd'], 0.05),
    }
    return detail, summary


def run_balance_pipeline(viable, pathways, processed_dir,
                         bounds=DEFAULT_BOUNDS, min_n=DEFAULT_MIN_N):
    """
    Run balance diagnostics for all eligible cohorts.

    Parameters
    ----------
    viable : DataFrame
        Common-support default report filtered to overlap >= 0.5 (or full set).
    pathways : list of str
    processed_dir : path-like
    """
    from pathlib import Path

    processed_dir = Path(processed_dir)
    details = []
    summaries = []
    n_ok, n_fail = 0, 0

    for dataset in ['GDSC2', 'CCLE']:
        print(f"\n  Processing {dataset}...")
        for pw in pathways:
            pw_short = pw.replace('HALLMARK_', '')
            fpath = processed_dir / f'{dataset.lower()}_cohort_{pw_short}.parquet'
            if not fpath.exists():
                continue

            df = pd.read_parquet(fpath)
            if viable is not None and len(viable):
                classes = viable[
                    (viable['dataset'] == dataset) & (viable['pathway'] == pw)
                ]['drug_class'].unique()
            else:
                classes = df['drug_class'].unique()

            for drug_class in classes:
                if drug_class == 'Other':
                    continue
                sub = df[df['drug_class'] == drug_class].copy()
                if len(sub) < min_n:
                    continue
                try:
                    detail, summary = assess_cohort_balance(
                        sub, drug_class, pw, dataset,
                        bounds=bounds, min_n=min_n,
                    )
                    if detail.empty:
                        continue
                    details.append(detail)
                    summaries.append(summary)
                    n_ok += 1
                    print(
                        f"    {drug_class} x {pw_short}: "
                        f"max |SMD| raw={summary['max_raw_smd']:.3f} -> "
                        f"trim={summary['max_trimmed_smd']:.3f} -> "
                        f"IPW={summary['max_weighted_smd']:.3f} | "
                        f"n>|0.1|: {summary['n_above_0.1_before']} -> "
                        f"{summary['n_above_0.1_after_trim']} -> "
                        f"{summary['n_above_0.1_after_weighting']}"
                    )
                except Exception as e:
                    n_fail += 1
                    warnings.warn(
                        f"Balance failed for {dataset} {drug_class} {pw}: {e}"
                    )
                    print(f"    FAILED {drug_class} x {pw_short}: {e}")

    print(f"\n  Balance cohorts succeeded: {n_ok}, failed: {n_fail}")
    detail_df = pd.concat(details, ignore_index=True) if details else pd.DataFrame()
    summary_df = pd.DataFrame(summaries) if summaries else pd.DataFrame()
    return detail_df, summary_df
