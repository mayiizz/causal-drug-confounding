"""Heterogeneous treatment effects via CausalForestDML."""

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import warnings


def estimate_cate_by_tissue(df, drug_class, pathway, dataset_name,
                            treatment_col='treatment', outcome_col='ln_ic50',
                            tissue_col='tissue_type'):
    """
    Estimate CATE for each tissue using CausalForestDML.
    Returns tissue-level CATE estimates.
    """
    try:
        from econml.dml import CausalForestDML
    except ImportError:
        raise ImportError("econml not installed. Run: pip install econml")
    
    sub = df.copy()
    
    # One-hot encode tissue
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    X = encoder.fit_transform(sub[[tissue_col]])
    feature_names = encoder.get_feature_names_out([tissue_col])
    
    T = sub[treatment_col].values
    Y = sub[outcome_col].values
    
    # Skip if insufficient variation
    if len(np.unique(T)) < 2 or len(sub) < 100:
        return pd.DataFrame()
    
    # Fit Causal Forest
    est = CausalForestDML(
        model_y='auto',
        model_t='auto',
        discrete_treatment=True,
        n_estimators=100,
        min_samples_leaf=10,
        max_depth=5,
        random_state=42
    )
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        est.fit(Y, T, X=X)
    
    # Predict CATE for each tissue (using tissue indicator as X)
    cates = est.effect(X)
    
    # Aggregate by tissue
    sub['cate'] = cates
    tissue_cate = sub.groupby(tissue_col).agg({
        'cate': ['mean', 'std', 'count']
    }).reset_index()
    tissue_cate.columns = ['tissue_type', 'cate_mean', 'cate_std', 'n']
    
    tissue_cate['drug_class'] = drug_class
    tissue_cate['pathway'] = pathway
    tissue_cate['dataset'] = dataset_name
    
    return tissue_cate


def run_cate_pipeline(cohort_df, dataset_name, top_n_cohorts=10):
    """
    Run CATE estimation on top divergent cohorts.
    """
    from src.causal.bootstrap import bootstrap_all_estimators
    from src.causal.common_support import fit_propensity_model, trim_by_propensity
    from src.causal.estimators import naive_ate, dr_ate_econml
    
    # Find top cohorts by |Naive - DR|
    results = []
    pathways = [c for c in cohort_df.columns if c.startswith('HALLMARK_')]
    
    for pw in pathways:
        for drug_class, group in cohort_df.groupby('drug_class'):
            if drug_class == 'Other':
                continue
            sub = group[group[pw].notna()].copy()
            if len(sub) < 60 or sub['treatment'].nunique() < 2:
                continue
            
            try:
                propensity, _, _ = fit_propensity_model(sub)
                sub['propensity'] = propensity
                trimmed, _ = trim_by_propensity(sub, bounds=(0.10, 0.90))
                
                naive = naive_ate(trimmed)
                dr = dr_ate_econml(trimmed)
                divergence = abs(naive['ate'] - dr['ate'])
                results.append({
                    'drug_class': drug_class,
                    'pathway': pw,
                    'divergence': divergence,
                    'data': trimmed
                })
            except Exception:
                continue
    
    # Select top N
    top = sorted(results, key=lambda x: x['divergence'], reverse=True)[:top_n_cohorts]
    
    all_cates = []
    for item in top:
        print(f"  {dataset_name} {item['drug_class']} × {item['pathway'].replace('HALLMARK_', '')}...")
        cate_df = estimate_cate_by_tissue(
            item['data'], item['drug_class'], item['pathway'], dataset_name
        )
        if not cate_df.empty:
            all_cates.append(cate_df)
    
    if all_cates:
        return pd.concat(all_cates, ignore_index=True)
    return pd.DataFrame()