"""
DR-based tissue permutation test.

Null hypothesis
---------------
Tissue assignment does not produce the observed causal (DR) effect.

Procedure (per cohort)
----------------------
1. Keep treatment T and outcome Y fixed.
2. Randomly permute tissue labels.
3. Refit propensity P(T=1 | tissue_perm).
4. Trim with the same propensity bounds as the main pipeline.
5. Refit EconML LinearDRLearner on the trimmed cohort.
6. Record the DR ATE.

The observed statistic uses the identical pipeline on unpermuted tissue.

This validates the primary Doubly Robust estimator, unlike the legacy
Naive ATE permutation in refutation.py.
"""

import pandas as pd
import numpy as np
import warnings

from src.causal.common_support import fit_propensity_model, trim_by_propensity
from src.causal.estimators import dr_ate_econml

# Publication default; override via CLI --n-permutations (e.g. 500, 1000).
DEFAULT_N_PERMUTATIONS = 500
DEFAULT_BOUNDS = (0.10, 0.90)
DEFAULT_MIN_N = 60
DEFAULT_SEED = 42


def _trim_for_dr(df, tissue_col='tissue_type', treatment_col='treatment',
                 bounds=DEFAULT_BOUNDS, min_n=DEFAULT_MIN_N):
    """
    Fit propensity on current tissue labels and trim.

    Returns trimmed DataFrame or None if the cohort is unusable.
    """
    work = df.copy()
    propensity, _, _ = fit_propensity_model(
        work, tissue_col=tissue_col, treatment_col=treatment_col
    )
    work['propensity'] = propensity
    trimmed, _ = trim_by_propensity(work, propensity_col='propensity', bounds=bounds)

    if len(trimmed) < min_n:
        return None
    if trimmed[treatment_col].nunique() < 2:
        return None
    return trimmed


def estimate_observed_dr_ate(df, treatment_col='treatment', outcome_col='ln_ic50',
                             tissue_col='tissue_type', bounds=DEFAULT_BOUNDS,
                             min_n=DEFAULT_MIN_N):
    """Observed DR ATE after propensity fit + trim (main-pipeline rules)."""
    trimmed = _trim_for_dr(
        df,
        tissue_col=tissue_col,
        treatment_col=treatment_col,
        bounds=bounds,
        min_n=min_n,
    )
    if trimmed is None:
        raise ValueError(
            f"Cohort unusable after trim (n<{min_n} or single treatment arm)."
        )
    result = dr_ate_econml(
        trimmed,
        treatment_col=treatment_col,
        outcome_col=outcome_col,
        tissue_col=tissue_col,
    )
    return float(result['ate']), len(trimmed)


def _one_permutation_dr_ate(df, tissues, rng, treatment_col, outcome_col,
                            tissue_col, bounds, min_n):
    """
    Single tissue permutation → propensity → trim → DR ATE.

    Returns float ATE or None if the permutation fails / is unusable.
    """
    perm_tissues = tissues.copy()
    rng.shuffle(perm_tissues)

    df_perm = df.copy()
    df_perm[tissue_col] = perm_tissues

    try:
        trimmed = _trim_for_dr(
            df_perm,
            tissue_col=tissue_col,
            treatment_col=treatment_col,
            bounds=bounds,
            min_n=min_n,
        )
        if trimmed is None:
            return None
        result = dr_ate_econml(
            trimmed,
            treatment_col=treatment_col,
            outcome_col=outcome_col,
            tissue_col=tissue_col,
        )
        ate = float(result['ate'])
        if not np.isfinite(ate):
            return None
        return ate
    except Exception:
        return None


def dr_permutation_test(df, treatment_col='treatment', outcome_col='ln_ic50',
                        tissue_col='tissue_type',
                        n_permutations=DEFAULT_N_PERMUTATIONS,
                        bounds=DEFAULT_BOUNDS, min_n=DEFAULT_MIN_N,
                        random_state=DEFAULT_SEED,
                        dataset=None, drug_class=None, pathway=None,
                        verbose=False):
    """
    Tissue-label permutation test for the DR ATE.

    Parameters
    ----------
    n_permutations : int
        Number of permutations (default 500; use CLI --n-permutations for 1000+).
    random_state : int
        Fixed seed for reproducibility.

    Returns
    -------
    summary : dict
        Cohort-level summary statistics including 95% null interval.
    null_df : pd.DataFrame
        One row per successful permutation with columns
        dataset, drug_class, pathway, permutation, dr_ate.
    """
    df = df.copy()
    df[tissue_col] = df[tissue_col].astype(str).astype(object)

    observed_dr_ate, n_obs_trimmed = estimate_observed_dr_ate(
        df,
        treatment_col=treatment_col,
        outcome_col=outcome_col,
        tissue_col=tissue_col,
        bounds=bounds,
        min_n=min_n,
    )

    tissues = df[tissue_col].to_numpy(dtype=object).copy()
    rng = np.random.RandomState(random_state)

    null_ates = []
    null_rows = []
    n_failed = 0
    n_requested = int(n_permutations)

    for i in range(n_requested):
        ate = _one_permutation_dr_ate(
            df, tissues, rng,
            treatment_col=treatment_col,
            outcome_col=outcome_col,
            tissue_col=tissue_col,
            bounds=bounds,
            min_n=min_n,
        )
        if ate is None:
            n_failed += 1
            if verbose:
                print(f"      perm {i+1}/{n_requested}: skipped (trim/DR failure)")
            continue

        null_ates.append(ate)
        null_rows.append({
            'dataset': dataset,
            'drug_class': drug_class,
            'pathway': pathway,
            'permutation': i,
            'dr_ate': ate,
        })
        if verbose and ((i + 1) % 100 == 0 or i == 0 or (i + 1) == n_requested):
            print(f"      perm {i+1}/{n_requested}: DR ATE={ate:.4f}")

    null_ates = np.asarray(null_ates, dtype=float)
    n_ok = len(null_ates)

    if n_ok == 0:
        warnings.warn(
            f"All {n_requested} permutations failed for "
            f"{dataset} {drug_class} {pathway}"
        )
        summary = {
            'dataset': dataset,
            'drug_class': drug_class,
            'pathway': pathway,
            'observed_dr_ate': observed_dr_ate,
            'null_mean': np.nan,
            'null_sd': np.nan,
            'empirical_p': np.nan,
            'z_score': np.nan,
            'percentile': np.nan,
            'null_ci_lower': np.nan,
            'null_ci_upper': np.nan,
            'n_permutations_requested': n_requested,
            'n_permutations': 0,
            'n_failed_permutations': n_failed,
            'n_obs_trimmed': n_obs_trimmed,
            'significant_0_05': False,
            # Backward-compatible aliases
            'permutation_p_value': np.nan,
            'observed_ate': observed_dr_ate,
        }
        return summary, pd.DataFrame(null_rows)

    null_mean = float(np.mean(null_ates))
    null_sd = float(np.std(null_ates, ddof=1)) if n_ok > 1 else 0.0
    null_ci_lower = float(np.percentile(null_ates, 2.5))
    null_ci_upper = float(np.percentile(null_ates, 97.5))

    # Two-sided empirical p-value
    empirical_p = float(np.mean(np.abs(null_ates) >= np.abs(observed_dr_ate)))
    if empirical_p == 0.0:
        # Discrete lower bound when observed is more extreme than all null draws
        empirical_p = 1.0 / (n_ok + 1)

    if null_sd > 1e-12:
        z_score = float((observed_dr_ate - null_mean) / null_sd)
    else:
        z_score = np.nan

    # Percentile of observed DR ATE within the null (empirical CDF)
    percentile = float(100.0 * np.mean(null_ates <= observed_dr_ate))

    summary = {
        'dataset': dataset,
        'drug_class': drug_class,
        'pathway': pathway,
        'observed_dr_ate': observed_dr_ate,
        'null_mean': null_mean,
        'null_sd': null_sd,
        'empirical_p': empirical_p,
        'z_score': z_score,
        'percentile': percentile,
        'null_ci_lower': null_ci_lower,
        'null_ci_upper': null_ci_upper,
        'n_permutations_requested': n_requested,
        'n_permutations': n_ok,
        'n_failed_permutations': n_failed,
        'n_obs_trimmed': n_obs_trimmed,
        'significant_0_05': bool(empirical_p < 0.05),
        # Backward-compatible aliases used by older Phase 5 prints / figures
        'permutation_p_value': empirical_p,
        'observed_ate': observed_dr_ate,
    }

    return summary, pd.DataFrame(null_rows)


def run_dr_permutation_suite(cohorts, n_permutations=DEFAULT_N_PERMUTATIONS,
                             bounds=DEFAULT_BOUNDS, min_n=DEFAULT_MIN_N,
                             random_state=DEFAULT_SEED, verbose=True):
    """
    Run DR permutation tests for a list of cohort dicts.

    Each cohort dict must contain:
        data, dataset, drug_class, pathway
    """
    summaries = []
    null_frames = []

    for item in cohorts:
        dataset = item['dataset']
        drug_class = item['drug_class']
        pathway = item['pathway']
        df = item['data']
        pw_short = pathway.replace('HALLMARK_', '')

        if verbose:
            print(
                f"  {dataset} {drug_class} x {pw_short} "
                f"(n={len(df)}, N_perm={n_permutations})..."
            )

        try:
            summary, null_df = dr_permutation_test(
                df,
                n_permutations=n_permutations,
                bounds=bounds,
                min_n=min_n,
                random_state=random_state,
                dataset=dataset,
                drug_class=drug_class,
                pathway=pathway,
                verbose=verbose,
            )
            summaries.append(summary)
            if not null_df.empty:
                null_frames.append(null_df)
            if verbose:
                print(
                    f"    observed_DR={summary['observed_dr_ate']:.4f}, "
                    f"null={summary['null_mean']:.4f}+/-{summary['null_sd']:.4f}, "
                    f"p={summary['empirical_p']:.4f}, "
                    f"z={summary['z_score']:.2f}, "
                    f"ok={summary['n_permutations']}/"
                    f"{summary['n_permutations'] + summary['n_failed_permutations']}"
                )
        except Exception as e:
            warnings.warn(
                f"DR permutation failed for {dataset} {drug_class} {pathway}: {e}"
            )
            if verbose:
                print(f"    FAILED: {e}")
            continue

    summary_df = pd.DataFrame(summaries) if summaries else pd.DataFrame()
    null_df = (
        pd.concat(null_frames, ignore_index=True) if null_frames else pd.DataFrame()
    )
    return summary_df, null_df
