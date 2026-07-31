"""Cross-dataset reproducibility: rank correlation of |Naive - DR|."""

import pandas as pd
import numpy as np
from scipy.stats import spearmanr, kendalltau


def compute_cross_dataset_reproducibility(gdsc_comparison, ccle_comparison):
    """
    Compute Spearman and Kendall correlation between datasets
    for |Naive - DR| divergence rankings.
    
    Parameters
    ----------
    gdsc_comparison : pd.DataFrame
        From ate_comparison.parquet, filtered to GDSC2
    ccle_comparison : pd.DataFrame
        From ate_comparison.parquet, filtered to CCLE
    
    Returns
    -------
    dict with correlation stats
    """
    # Merge on drug_class and pathway
    merged = gdsc_comparison.merge(
        ccle_comparison,
        on=['drug_class', 'pathway'],
        suffixes=('_gdsc', '_ccle')
    )
    
    if len(merged) < 5:
        return {
            'n_common': len(merged),
            'spearman_rho': np.nan,
            'spearman_p': np.nan,
            'kendall_tau': np.nan,
            'kendall_p': np.nan
        }
    
    # Rank by |Naive - DR| in each dataset
    merged['rank_gdsc'] = merged['naive_dr_divergence_gdsc'].rank()
    merged['rank_ccle'] = merged['naive_dr_divergence_ccle'].rank()
    
    rho, p_rho = spearmanr(merged['rank_gdsc'], merged['rank_ccle'])
    tau, p_tau = kendalltau(merged['rank_gdsc'], merged['rank_ccle'])
    
    return {
        'n_common': len(merged),
        'spearman_rho': rho,
        'spearman_p': p_rho,
        'kendall_tau': tau,
        'kendall_p': p_tau,
        'merged_data': merged
    }