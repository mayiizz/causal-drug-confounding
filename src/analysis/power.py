"""Power and stability analysis per drug class."""

import pandas as pd
import numpy as np


def compute_power_metrics(df, drug_class, pathway, dataset_name):
    """
    Compute sample size, CI width, and retained support %.
    """
    n = len(df)
    n_treated = int(df['treatment'].sum())
    n_control = n - n_treated
    
    # CI width from bootstrap if available
    if 'ci_lower' in df.columns and 'ci_upper' in df.columns:
        ci_width = (df['ci_upper'] - df['ci_lower']).mean()
    else:
        ci_width = np.nan
    
    # Propensity range
    if 'propensity' in df.columns:
        prop_range = df['propensity'].max() - df['propensity'].min()
        prop_std = df['propensity'].std()
    else:
        prop_range = np.nan
        prop_std = np.nan
    
    return {
        'dataset': dataset_name,
        'drug_class': drug_class,
        'pathway': pathway,
        'n_total': n,
        'n_treated': n_treated,
        'n_control': n_control,
        'treated_ratio': n_treated / n if n > 0 else np.nan,
        'ci_width': ci_width,
        'propensity_range': prop_range,
        'propensity_std': prop_std,
        'power_adequate': n >= 100 and n_treated >= 30 and n_control >= 30
    }