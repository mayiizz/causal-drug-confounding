"""
Individual-level counterfactual outcomes under do(Treatment = 0) and do(Treatment = 1).

This module does NOT produce another ATE. It imputes both potential outcomes
using the DR CATE from LinearDRLearner (tissue as X):

    tau(X)  = E[Y(1) - Y(0) | X]
    Y(1)    = Y_obs + (1 - T) * tau(X)   # do(high pathway)
    Y(0)    = Y_obs - T * tau(X)         # do(low pathway)
    delta_high = Y(1) - Y_obs
    delta_low  = Y(0) - Y_obs

Assumptions:
- Consistency and no unmeasured confounding given tissue type.
- The DR learner is refit per cohort (same pattern as Phase 4).

Validity:
- Non-finite observed/predicted values are flagged (not used in figures).
- Extreme |observed ln_IC50| above ABS_LN_IC50_MAX are flagged as invalid
  transformations (e.g. failed IC50→ln conversion), not silently dropped
  from the parquet.
- Extreme |tau| above ABS_TAU_MAX are flagged as unstable DR predictions.
"""

import pandas as pd
import numpy as np
import warnings

from src.causal.estimators import fit_linear_dr_learner

# Beyond ~99.9th percentile of observed ln_IC50 in these cohorts; values like
# 593 indicate invalid IC50 transformations, not biology.
ABS_LN_IC50_MAX = 15.0
# Extreme individual DR effects (rare; e.g. |tau|~85 in a few CCLE cohorts).
ABS_TAU_MAX = 25.0


def fit_dr_for_counterfactual(df, treatment_col='treatment', outcome_col='ln_ic50',
                              tissue_col='tissue_type'):
    """Fit shared LinearDRLearner for counterfactual imputation."""
    return fit_linear_dr_learner(
        df,
        treatment_col=treatment_col,
        outcome_col=outcome_col,
        tissue_col=tissue_col,
    )


def _validity_mask(y_obs, y1, y0, tau,
                   abs_ln_ic50_max=ABS_LN_IC50_MAX,
                   abs_tau_max=ABS_TAU_MAX):
    """
    Return (is_valid, reason) arrays.

    Reasons are empty string when valid; otherwise a short tag for auditing.
    """
    n = len(y_obs)
    reason = np.full(n, '', dtype=object)

    bad_obs = ~np.isfinite(y_obs)
    reason[bad_obs] = 'nonfinite_observed'

    extreme_obs = np.isfinite(y_obs) & (np.abs(y_obs) > abs_ln_ic50_max)
    reason[extreme_obs & (reason == '')] = 'extreme_observed_ln_ic50'

    bad_tau = ~np.isfinite(tau)
    reason[bad_tau & (reason == '')] = 'nonfinite_dr_tau'

    extreme_tau = np.isfinite(tau) & (np.abs(tau) > abs_tau_max)
    reason[extreme_tau & (reason == '')] = 'extreme_dr_tau'

    bad_pred = ~(np.isfinite(y1) & np.isfinite(y0))
    reason[bad_pred & (reason == '')] = 'nonfinite_potential_outcome'

    is_valid = reason == ''
    return is_valid, reason


def predict_counterfactuals(df, drug_class, pathway, dataset_name,
                            treatment_col='treatment', outcome_col='ln_ic50',
                            tissue_col='tissue_type', id_col='cell_line_id',
                            large_shift_threshold=1.0):
    """
    Estimate Y(0) and Y(1) via DR CATE imputation.

    Returns
    -------
    pd.DataFrame
        One row per cell line in the (trimmed) cohort, including validity flags.
        Invalid rows are retained for audit; figures should filter is_valid.
    """
    if len(df) < 60 or df[treatment_col].nunique() < 2:
        return pd.DataFrame()

    est, X, _ = fit_dr_for_counterfactual(
        df,
        treatment_col=treatment_col,
        outcome_col=outcome_col,
        tissue_col=tissue_col,
    )

    T = df[treatment_col].astype(float).values
    y_obs = df[outcome_col].astype(float).values
    tau = np.asarray(est.effect(X)).reshape(-1)

    # Both potential outcomes
    y1 = y_obs + (1.0 - T) * tau   # counterfactual_high / do(T=1)
    y0 = y_obs - T * tau           # counterfactual_low  / do(T=0)

    delta_high = y1 - y_obs        # (1-T)*tau
    delta_low = y0 - y_obs         # -T*tau
    individual_effect = y1 - y0    # tau

    is_valid, invalid_reason = _validity_mask(y_obs, y1, y0, tau)

    out = pd.DataFrame({
        'dataset': dataset_name,
        'drug_class': drug_class,
        'pathway': pathway,
        'cell_line_id': df[id_col].values if id_col in df.columns else np.arange(len(df)),
        'tissue_type': df[tissue_col].values,
        'treatment': T.astype(int),
        'observed_ln_ic50': y_obs,
        'counterfactual_high_ln_ic50': y1,
        'counterfactual_low_ln_ic50': y0,
        'delta_high': delta_high,
        'delta_low': delta_low,
        'abs_delta_high': np.abs(delta_high),
        'abs_delta_low': np.abs(delta_low),
        'individual_effect': individual_effect,
        'large_shift': np.abs(delta_high) > large_shift_threshold,
        'is_valid': is_valid,
        'invalid_reason': invalid_reason,
    })
    return out


def _agg_delta_stats(g):
    """Aggregate explicit delta_high stats on a group."""
    return {
        'n': len(g),
        'n_valid': int(g['is_valid'].sum()) if 'is_valid' in g.columns else len(g),
        'mean_delta_high': g['delta_high'].mean(),
        'median_delta_high': g['delta_high'].median(),
        'sd_delta_high': g['delta_high'].std(),
        'mean_delta_low': g['delta_low'].mean(),
        'mean_individual_effect': g['individual_effect'].mean(),
        'prop_large_shift': g['large_shift'].mean(),
        'mean_abs_delta_high': g['abs_delta_high'].mean(),
    }


def summarize_counterfactuals(pred_df, top_n=10):
    """
    Summarize counterfactual deltas for Results-section tables.

    Returns a long-format summary with `level` in
    {global, cohort, tissue, drug_class, top_tissue, top_drug_class,
    validity_audit}.

    top_tissue / top_drug_class rows include an explicit `rank` (1 = most
    affected by mean |delta_high| among valid rows).
    """
    if pred_df is None or pred_df.empty:
        return pd.DataFrame()

    valid = pred_df[pred_df['is_valid']].copy() if 'is_valid' in pred_df.columns else pred_df
    if valid.empty:
        valid = pred_df

    rows = []

    def _row(level, dataset='ALL', drug_class='ALL', pathway='ALL',
             tissue_type='ALL', rank=np.nan, g=None, **extra):
        if g is not None:
            stats = _agg_delta_stats(g)
        else:
            stats = {}
        out = {
            'level': level,
            'dataset': dataset,
            'drug_class': drug_class,
            'pathway': pathway,
            'tissue_type': tissue_type,
            'rank': rank,
            'n': stats.get('n', extra.get('n', 0)),
            'n_valid': stats.get('n_valid', extra.get('n_valid', 0)),
            'mean_delta_high': stats.get('mean_delta_high', np.nan),
            'median_delta_high': stats.get('median_delta_high', np.nan),
            'sd_delta_high': stats.get('sd_delta_high', np.nan),
            'mean_delta_low': stats.get('mean_delta_low', np.nan),
            'mean_individual_effect': stats.get('mean_individual_effect', np.nan),
            'prop_large_shift': stats.get('prop_large_shift', np.nan),
            'mean_abs_delta_high': stats.get('mean_abs_delta_high', np.nan),
        }
        out.update(extra)
        return out

    # Global
    rows.append(_row(
        'global', g=valid, n=len(pred_df), n_valid=int(valid['is_valid'].sum())
        if 'is_valid' in valid.columns else len(valid)
    ))
    rows[-1]['n'] = len(pred_df)
    rows[-1]['n_valid'] = int(pred_df['is_valid'].sum()) if 'is_valid' in pred_df.columns else len(pred_df)

    # Validity audit
    if 'invalid_reason' in pred_df.columns:
        n_invalid = int((~pred_df['is_valid']).sum())
        rows.append(_row(
            'validity_audit',
            n=len(pred_df),
            n_valid=int(pred_df['is_valid'].sum()),
            mean_delta_high=float(n_invalid),
        ))

    # Per cohort
    for (dataset, drug_class, pathway), g in valid.groupby(['dataset', 'drug_class', 'pathway']):
        rows.append(_row(
            'cohort', dataset=dataset, drug_class=drug_class, pathway=pathway, g=g
        ))

    # Tissues
    tissue_frames = []
    for (dataset, tissue), g in valid.groupby(['dataset', 'tissue_type']):
        s = _agg_delta_stats(g)
        s.update({'dataset': dataset, 'tissue_type': tissue})
        tissue_frames.append(s)
        rows.append(_row(
            'tissue', dataset=dataset, tissue_type=tissue, g=g
        ))

    tissue_stats = pd.DataFrame(tissue_frames).sort_values(
        'mean_abs_delta_high', ascending=False
    )
    for rank, r in enumerate(tissue_stats.head(top_n).itertuples(index=False), start=1):
        rows.append(_row(
            'top_tissue',
            dataset=r.dataset,
            tissue_type=r.tissue_type,
            rank=rank,
            n=int(r.n),
            n_valid=int(r.n_valid),
            mean_delta_high=r.mean_delta_high,
            median_delta_high=r.median_delta_high,
            sd_delta_high=r.sd_delta_high,
            mean_delta_low=r.mean_delta_low,
            mean_individual_effect=r.mean_individual_effect,
            prop_large_shift=r.prop_large_shift,
            mean_abs_delta_high=r.mean_abs_delta_high,
        ))

    # Drug classes
    class_frames = []
    for (dataset, drug_class), g in valid.groupby(['dataset', 'drug_class']):
        s = _agg_delta_stats(g)
        s.update({'dataset': dataset, 'drug_class': drug_class})
        class_frames.append(s)
        rows.append(_row(
            'drug_class', dataset=dataset, drug_class=drug_class, g=g
        ))

    class_stats = pd.DataFrame(class_frames).sort_values(
        'mean_abs_delta_high', ascending=False
    )
    for rank, r in enumerate(class_stats.head(top_n).itertuples(index=False), start=1):
        rows.append(_row(
            'top_drug_class',
            dataset=r.dataset,
            drug_class=r.drug_class,
            rank=rank,
            n=int(r.n),
            n_valid=int(r.n_valid),
            mean_delta_high=r.mean_delta_high,
            median_delta_high=r.median_delta_high,
            sd_delta_high=r.sd_delta_high,
            mean_delta_low=r.mean_delta_low,
            mean_individual_effect=r.mean_individual_effect,
            prop_large_shift=r.prop_large_shift,
            mean_abs_delta_high=r.mean_abs_delta_high,
        ))

    return pd.DataFrame(rows)


def run_counterfactual_pipeline(viable, pathways, processed_dir,
                                min_n=60, bounds=(0.10, 0.90)):
    """
    Run counterfactual prediction over all eligible cohorts.

    Parameters
    ----------
    viable : DataFrame
        Common-support rows with overlap_coefficient already filtered.
    pathways : list of str
        Hallmark pathway column names.
    processed_dir : Path
        Directory containing cohort parquet files.
    """
    from pathlib import Path
    from src.causal.common_support import fit_propensity_model, trim_by_propensity

    processed_dir = Path(processed_dir)
    all_preds = []
    n_ok, n_fail = 0, 0

    for dataset in ['GDSC2', 'CCLE']:
        print(f"\n  Processing {dataset}...")
        for pw in pathways:
            pw_short = pw.replace('HALLMARK_', '')
            fpath = processed_dir / f'{dataset.lower()}_cohort_{pw_short}.parquet'
            if not fpath.exists():
                print(f"    Skip missing cohort file: {fpath.name}")
                continue

            df = pd.read_parquet(fpath)
            viable_classes = viable[
                (viable['dataset'] == dataset) &
                (viable['pathway'] == pw)
            ]['drug_class'].unique()

            for drug_class in viable_classes:
                if drug_class == 'Other':
                    continue

                sub = df[df['drug_class'] == drug_class].copy()
                if len(sub) < min_n:
                    continue

                try:
                    propensity, _, _ = fit_propensity_model(sub)
                    sub['propensity'] = propensity
                    trimmed, _ = trim_by_propensity(sub, bounds=bounds)
                    if len(trimmed) < min_n or trimmed['treatment'].nunique() < 2:
                        continue

                    pred = predict_counterfactuals(
                        trimmed, drug_class, pw, dataset
                    )
                    if pred.empty:
                        continue

                    all_preds.append(pred)
                    n_ok += 1
                    n_valid = int(pred['is_valid'].sum())
                    print(
                        f"    {drug_class} x {pw_short}: "
                        f"n={len(pred)} (valid={n_valid}), "
                        f"mean_delta_high={pred.loc[pred['is_valid'], 'delta_high'].mean():.3f}, "
                        f"large_shift={pred.loc[pred['is_valid'], 'large_shift'].mean():.1%}"
                    )
                except Exception as e:
                    n_fail += 1
                    warnings.warn(
                        f"Counterfactual failed for {dataset} {drug_class} {pw}: {e}"
                    )
                    print(f"    FAILED {drug_class} x {pw_short}: {e}")
                    continue

    print(f"\n  Cohorts succeeded: {n_ok}, failed: {n_fail}")

    if not all_preds:
        return pd.DataFrame(), pd.DataFrame()

    pred_df = pd.concat(all_preds, ignore_index=True)

    n_invalid = int((~pred_df['is_valid']).sum())
    if n_invalid:
        print(f"\n  Validity: {n_invalid:,} / {len(pred_df):,} rows flagged invalid")
        print(pred_df.loc[~pred_df['is_valid'], 'invalid_reason'].value_counts().to_string())

    summary_df = summarize_counterfactuals(pred_df)
    return pred_df, summary_df
