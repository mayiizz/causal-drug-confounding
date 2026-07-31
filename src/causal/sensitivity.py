"""Sensitivity analysis across treatment thresholds and support bounds."""

import pandas as pd
import numpy as np
from src.causal.estimators import naive_ate, ipw_ate, dr_ate_econml
from src.causal.common_support import fit_propensity_model, trim_by_propensity


def sensitivity_grid(df, drug_class, pathway, dataset_name,
                     thresholds=[0.25, 0.33, 0.50],
                     support_bounds=[(0.05, 0.95), (0.10, 0.90), (0.15, 0.85)]):
    """
    Re-estimate causal effects across multiple thresholds and support bounds.
    Verify effect direction does not flip.
    """
    results = []
    pw_col = pathway
    
    for thresh in thresholds:
        # Re-binarize at this threshold
        scores = df[pw_col].dropna()
        cutoff = np.percentile(scores, thresh * 100)
        df_temp = df.copy()
        df_temp['treatment'] = (df_temp[pw_col] >= cutoff).astype(int)
        
        if df_temp['treatment'].nunique() < 2:
            continue
        
        for bounds in support_bounds:
            # Fit propensity and trim
            try:
                propensity, _, _ = fit_propensity_model(df_temp)
                df_temp['propensity'] = propensity
                trimmed, _ = trim_by_propensity(df_temp, bounds=bounds)
                
                if len(trimmed) < 30 or trimmed['treatment'].nunique() < 2:
                    continue
                
                # Estimate all three
                naive = naive_ate(trimmed)
                
                try:
                    ipw = ipw_ate(trimmed)
                except Exception:
                    ipw = {'ate': np.nan}
                
                try:
                    dr = dr_ate_econml(trimmed)
                except Exception:
                    dr = {'ate': np.nan}
                
                results.append({
                    'dataset': dataset_name,
                    'drug_class': drug_class,
                    'pathway': pathway,
                    'threshold': thresh,
                    'support_lower': bounds[0],
                    'support_upper': bounds[1],
                    'n_trimmed': len(trimmed),
                    'n_treated': int(trimmed['treatment'].sum()),
                    'naive_ate': naive['ate'],
                    'ipw_ate': ipw['ate'],
                    'dr_ate': dr['ate']
                })
            except Exception:
                continue
    
    return pd.DataFrame(results)