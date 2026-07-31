"""Bootstrap confidence intervals for causal estimates."""

import numpy as np
import pandas as pd
from tqdm import tqdm


def bootstrap_ci(df, estimator_fn, n_bootstrap=500, ci=0.95, random_state=42):
    """
    Compute bootstrap confidence interval for an estimator.
    
    Parameters
    ----------
    df : pd.DataFrame
        Cohort data
    estimator_fn : callable
        Function that takes df and returns dict with 'ate' key
    n_bootstrap : int
        Number of bootstrap samples
    ci : float
        Confidence level (e.g., 0.95 for 95%)
    
    Returns
    -------
    dict with 'ate', 'ci_lower', 'ci_upper', 'se_boot'
    """
    rng = np.random.RandomState(random_state)
    n = len(df)
    
    ate_estimates = []
    
    for _ in range(n_bootstrap):
        # Resample with replacement
        boot_idx = rng.choice(n, size=n, replace=True)
        boot_df = df.iloc[boot_idx].copy()
        
        try:
            result = estimator_fn(boot_df)
            ate_estimates.append(result['ate'])
        except Exception:
            continue
    
    if len(ate_estimates) < n_bootstrap * 0.5:
        return {'ate': np.nan, 'ci_lower': np.nan, 'ci_upper': np.nan, 'se_boot': np.nan}
    
    ate_estimates = np.array(ate_estimates)
    alpha = (1 - ci) / 2
    
    return {
        'ate': np.median(ate_estimates),
        'ci_lower': np.percentile(ate_estimates, alpha * 100),
        'ci_upper': np.percentile(ate_estimates, (1 - alpha) * 100),
        'se_boot': np.std(ate_estimates)
    }


def bootstrap_all_estimators(df, drug_class, pathway, dataset_name,
                             n_bootstrap=500, random_state=42):
    """
    Bootstrap all three estimators for one cohort.
    """
    from src.causal.estimators import naive_ate, ipw_ate, dr_ate_econml
    
    results = []
    
    estimators = {
        'Naive': naive_ate,
        'IPW': ipw_ate,
        'DR': dr_ate_econml
    }
    
    for name, fn in estimators.items():
        # Skip IPW/DR if propensity missing
        if name in ['IPW', 'DR'] and 'propensity' not in df.columns:
            continue
        
        # Skip DR if econml fails on point estimate
        if name == 'DR':
            try:
                _ = fn(df)
            except Exception:
                continue
        
        boot = bootstrap_ci(df, fn, n_bootstrap=n_bootstrap, random_state=random_state)
        
        if not np.isnan(boot['ate']):
            results.append({
                'dataset': dataset_name,
                'drug_class': drug_class,
                'pathway': pathway,
                'estimator': name,
                'ate': boot['ate'],
                'ci_lower': boot['ci_lower'],
                'ci_upper': boot['ci_upper'],
                'se_boot': boot['se_boot'],
                'n': len(df)
            })
    
    return results