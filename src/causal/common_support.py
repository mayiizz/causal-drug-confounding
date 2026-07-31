"""Common support diagnostics and propensity score modeling."""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
import warnings


def fit_propensity_model(df, tissue_col='tissue_type', treatment_col='treatment'):
    """
    Fit logistic regression: P(treatment=1 | tissue_type).
    
    Returns
    -------
    propensity_scores : pd.Series
        Predicted propensity for each row
    model : LogisticRegression
        Fitted model
    encoder : OneHotEncoder
        Fitted encoder for tissues
    """
    # One-hot encode tissue
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    X = encoder.fit_transform(df[[tissue_col]])
    
    y = df[treatment_col].values
    
    # Fit logistic regression with regularization
    model = LogisticRegression(max_iter=1000, C=1.0, solver='lbfgs')
    model.fit(X, y)
    
    # Predict propensity
    propensity = model.predict_proba(X)[:, 1]
    
    return pd.Series(propensity, index=df.index), model, encoder


def compute_overlap_coefficient(treated_props, control_props, n_bins=100):
    """
    Compute overlap coefficient between treated and control propensity distributions.
    Overlap = sum(min(p_treated, p_control)) across bins.
    """
    # Create common bins
    all_props = np.concatenate([treated_props, control_props])
    bins = np.linspace(all_props.min(), all_props.max(), n_bins + 1)
    
    # Histograms
    hist_t, _ = np.histogram(treated_props, bins=bins, density=True)
    hist_c, _ = np.histogram(control_props, bins=bins, density=True)
    
    # Normalize to probabilities
    hist_t = hist_t / hist_t.sum() if hist_t.sum() > 0 else hist_t
    hist_c = hist_c / hist_c.sum() if hist_c.sum() > 0 else hist_c
    
    # Overlap coefficient
    overlap = np.sum(np.minimum(hist_t, hist_c))
    return overlap


def trim_by_propensity(df, propensity_col='propensity', bounds=(0.10, 0.90)):
    """
    Trim samples outside propensity bounds.
    
    Returns
    -------
    trimmed_df : pd.DataFrame
    report : dict
    """
    lower, upper = bounds
    mask = (df[propensity_col] >= lower) & (df[propensity_col] <= upper)
    
    trimmed = df[mask].copy()
    
    report = {
        'bounds': bounds,
        'n_original': len(df),
        'n_trimmed': len(trimmed),
        'n_dropped': len(df) - len(trimmed),
        'percent_retained': round(len(trimmed) / len(df) * 100, 2)
    }
    
    return trimmed, report


def compute_smd(df, covariate_col, treatment_col='treatment', weight_col=None):
    """
    Compute Standardized Mean Difference for a covariate.
    If weight_col provided, computes weighted SMD.
    """
    treated = df[df[treatment_col] == 1]
    control = df[df[treatment_col] == 0]
    
    if weight_col is not None:
        # Weighted mean and variance
        w_t = treated[weight_col].values
        w_c = control[weight_col].values
        
        mean_t = np.average(treated[covariate_col].values, weights=w_t)
        mean_c = np.average(control[covariate_col].values, weights=w_c)
        
        var_t = np.average((treated[covariate_col].values - mean_t) ** 2, weights=w_t)
        var_c = np.average((control[covariate_col].values - mean_c) ** 2, weights=w_c)
    else:
        mean_t = treated[covariate_col].mean()
        mean_c = control[covariate_col].mean()
        var_t = treated[covariate_col].var()
        var_c = control[covariate_col].var()
    
    pooled_std = np.sqrt((var_t + var_c) / 2)
    if pooled_std == 0:
        return 0.0
    
    smd = abs(mean_t - mean_c) / pooled_std
    return smd


def assess_common_support(df, drug_class, pathway, dataset_name,
                          tissue_col='tissue_type', treatment_col='treatment',
                          bounds_list=None):
    """
    Full common support assessment for one (drug_class, pathway, dataset) cohort.
    
    Returns
    -------
    list of dicts : one per bound configuration
    """
    if bounds_list is None:
        bounds_list = [(0.05, 0.95), (0.10, 0.90), (0.15, 0.85)]
    
    # Fit propensity model
    propensity, model, encoder = fit_propensity_model(df, tissue_col, treatment_col)
    df = df.copy()
    df['propensity'] = propensity
    
    treated_props = df[df[treatment_col] == 1]['propensity'].values
    control_props = df[df[treatment_col] == 0]['propensity'].values
    
    overlap = compute_overlap_coefficient(treated_props, control_props)
    
    # Compute SMD for tissue (using first tissue category as example)
    # For categorical, we compute SMD on propensity itself as a summary
    smd_before = compute_smd(df, 'propensity', treatment_col)
    
    results = []
    for bounds in bounds_list:
        trimmed, report = trim_by_propensity(df, bounds=bounds)
        
        if len(trimmed) > 0:
            smd_after = compute_smd(trimmed, 'propensity', treatment_col)
        else:
            smd_after = np.nan
        
        results.append({
            'dataset': dataset_name,
            'drug_class': drug_class,
            'pathway': pathway,
            'n_original': report['n_original'],
            'n_trimmed': report['n_trimmed'],
            'n_dropped': report['n_dropped'],
            'percent_retained': report['percent_retained'],
            'overlap_coefficient': round(overlap, 4),
            'smd_before': round(smd_before, 4),
            'smd_after': round(smd_after, 4),
            'propensity_lower': bounds[0],
            'propensity_upper': bounds[1]
        })
    
    return results


def run_common_support_pipeline(cohort_df, dataset_name, bounds_list=None):
    """
    Run common support assessment across all (drug_class, pathway) cohorts.
    
    Parameters
    ----------
    cohort_df : pd.DataFrame
        Must contain: drug_class, tissue_type, treatment, and pathway score columns
    dataset_name : str
        'GDSC2' or 'CCLE'
    
    Returns
    -------
    pd.DataFrame : common support report
    """
    all_results = []
    
    # Identify which pathway this cohort was binarized for
    # The treatment column exists; we need to know which pathway score was used
    # We'll infer from the data: group by drug_class and assess
    pathways = [c for c in cohort_df.columns if c.startswith('HALLMARK_')]
    
    for drug_class, group in cohort_df.groupby('drug_class'):
        if drug_class == 'Other':
            continue
            
        for pw in pathways:
            # Subset to rows with non-null pathway score
            sub = group[group[pw].notna()].copy()
            if len(sub) < 60:
                continue
            
            # Check if treatment column exists and is valid
            if 'treatment' not in sub.columns or sub['treatment'].nunique() < 2:
                continue
            
            try:
                results = assess_common_support(
                    sub, drug_class, pw, dataset_name, bounds_list=bounds_list
                )
                all_results.extend(results)
            except Exception as e:
                warnings.warn(f"Failed for {dataset_name} {drug_class} {pw}: {e}")
    
    if not all_results:
        return pd.DataFrame()
    
    return pd.DataFrame(all_results)