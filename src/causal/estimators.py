"""Causal estimators: Naive, IPW, and Doubly Robust (AIPW)."""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import OneHotEncoder
import warnings


def naive_ate(df, treatment_col='treatment', outcome_col='ln_ic50'):
    """
    Baseline: simple difference in means.
    ATE = E[Y | T=1] - E[Y | T=0]
    """
    treated = df[df[treatment_col] == 1][outcome_col]
    control = df[df[treatment_col] == 0][outcome_col]
    
    ate = treated.mean() - control.mean()
    se = np.sqrt(treated.var() / len(treated) + control.var() / len(control))
    
    return {
        'ate': ate,
        'se': se,
        'n_treated': len(treated),
        'n_control': len(control)
    }


def compute_stabilized_ipw_weights(df, treatment_col='treatment',
                                   propensity_col='propensity',
                                   clip_bounds=(0.10, 0.90)):
    """
    Unit-level IPW weights matching the Hajek IPW estimator.

    w_i = T_i / e_i + (1 - T_i) / (1 - e_i)

    Propensity is clipped to clip_bounds (same as ipw_ate). The Hajek
    estimator normalizes within arm; for balance diagnostics these raw
    weights are used in weighted means / variances.
    """
    T = df[treatment_col].astype(float).values
    e = df[propensity_col].astype(float).values.clip(*clip_bounds)
    # Avoid exact 0/1 after clip edge cases
    e = np.clip(e, 1e-6, 1.0 - 1e-6)
    w = np.where(T == 1.0, 1.0 / e, 1.0 / (1.0 - e))
    return pd.Series(w, index=df.index, name='ipw_weight')


def ipw_ate(df, treatment_col='treatment', outcome_col='ln_ic50',
            propensity_col='propensity', clip_bounds=(0.10, 0.90)):
    """
    Stabilized IPW (Hajek estimator).
    ATE = (Σ w1 Y / Σ w1) - (Σ w0 Y / Σ w0)
    More stable than Horvitz-Thompson.
    """
    T = df[treatment_col].values
    Y = df[outcome_col].values
    e = df[propensity_col].values.clip(*clip_bounds)

    # Same weights as compute_stabilized_ipw_weights (arm-specific form)
    w1 = T / e
    w0 = (1 - T) / (1 - e)

    # Hajek estimator (normalized)
    ate = np.sum(w1 * Y) / np.sum(w1) - np.sum(w0 * Y) / np.sum(w0)

    # Robust SE via influence function
    mu1 = np.sum(w1 * Y) / np.sum(w1)
    mu0 = np.sum(w0 * Y) / np.sum(w0)

    # Influence function for Hajek
    inf1 = T * (Y - mu1) / e
    inf0 = (1 - T) * (Y - mu0) / (1 - e)
    psi = inf1 - inf0

    se = np.sqrt(np.var(psi) / len(df))

    return {
        'ate': ate,
        'se': se,
        'n_treated': int(T.sum()),
        'n_control': int(len(T) - T.sum())
    }


def fit_linear_dr_learner(df, treatment_col='treatment', outcome_col='ln_ic50',
                          tissue_col='tissue_type'):
    """
    Fit EconML LinearDRLearner with tissue one-hots as X.

    Returns
    -------
    est : LinearDRLearner
        Fitted doubly robust learner.
    X : ndarray
        Design matrix (tissue one-hots).
    encoder : OneHotEncoder
        Fitted encoder (for diagnostics / reuse).
    """
    try:
        from econml.dr import LinearDRLearner
    except ImportError:
        raise ImportError("econml not installed. Run: pip install econml")

    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    X = encoder.fit_transform(df[[tissue_col]])
    T = df[treatment_col].values
    Y = df[outcome_col].values

    est = LinearDRLearner(
        model_regression=LinearRegression(),
        model_propensity=LogisticRegression(max_iter=1000, C=1.0)
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        est.fit(Y, T, X=X)

    return est, X, encoder


def dr_ate_econml(df, treatment_col='treatment', outcome_col='ln_ic50',
                  tissue_col='tissue_type'):
    """
    Doubly Robust ATE via EconML LinearDRLearner.
    Uses tissue one-hot as covariates X.
    """
    est, X, _ = fit_linear_dr_learner(
        df, treatment_col=treatment_col, outcome_col=outcome_col,
        tissue_col=tissue_col
    )
    T = df[treatment_col].values

    # ATE averaged over the observed covariate distribution
    ate = est.ate(X=X)

    # Bootstrap SE (EconML doesn't always give analytic SE easily)
    # We'll compute in the bootstrap module
    return {
        'ate': float(ate),
        'se': np.nan,  # Computed via bootstrap
        'n_treated': int(T.sum()),
        'n_control': int(len(T) - T.sum())
    }


def compute_all_estimates(df, drug_class, pathway, dataset_name):
    """
    Run all three estimators on a single cohort.
    Returns list of result dicts.
    """
    results = []
    
    # Naive
    naive = naive_ate(df)
    results.append({
        'dataset': dataset_name,
        'drug_class': drug_class,
        'pathway': pathway,
        'estimator': 'Naive',
        'ate': naive['ate'],
        'se': naive['se'],
        'n_treated': naive['n_treated'],
        'n_control': naive['n_control']
    })
    
    # IPW (needs propensity)
    if 'propensity' in df.columns:
        ipw = ipw_ate(df)
        results.append({
            'dataset': dataset_name,
            'drug_class': drug_class,
            'pathway': pathway,
            'estimator': 'IPW',
            'ate': ipw['ate'],
            'se': ipw['se'],
            'n_treated': ipw['n_treated'],
            'n_control': ipw['n_control']
        })
    
    # DR
    try:
        dr = dr_ate_econml(df)
        results.append({
            'dataset': dataset_name,
            'drug_class': drug_class,
            'pathway': pathway,
            'estimator': 'DR',
            'ate': dr['ate'],
            'se': dr['se'],
            'n_treated': dr['n_treated'],
            'n_control': dr['n_control']
        })
    except Exception as e:
        warnings.warn(f"DR failed for {dataset_name} {drug_class} {pathway}: {e}")
    
    return results