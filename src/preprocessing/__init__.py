"""Binarize pathway scores into treatment (top tercile = 1, bottom 2/3 = 0)."""

import pandas as pd
import numpy as np


def binarize_pathway_treatment(df, pathway_col, drug_class_col='drug_class',
                               treatment_col='treatment', min_total=90):
    """
    For each drug_class, binarize pathway score into top tercile (1) vs bottom 2/3 (0).
    
    Returns
    -------
    binarized_df : pd.DataFrame
        Input df with added 'treatment' column
    threshold_df : pd.DataFrame
        One row per drug_class with threshold and counts
    """
    results = []
    thresholds = []
    
    for drug_class, group in df.groupby(drug_class_col):
        if drug_class == 'Other':
            continue  # Skip unmapped drugs
            
        scores = group[pathway_col].dropna()
        if len(scores) < min_total:
            continue
            
        # 66.67th percentile = top tercile cutoff
        threshold = np.percentile(scores, 100 * 2 / 3)
        
        group = group.copy()
        group[treatment_col] = (group[pathway_col] >= threshold).astype(int)
        
        n_treated = (group[treatment_col] == 1).sum()
        n_control = (group[treatment_col] == 0).sum()
        
        results.append(group)
        thresholds.append({
            'drug_class': drug_class,
            'pathway': pathway_col,
            'threshold': threshold,
            'n_total': len(group),
            'n_treated': int(n_treated),
            'n_control': int(n_control)
        })
    
    if not results:
        return pd.DataFrame(), pd.DataFrame()
    
    return pd.concat(results, ignore_index=True), pd.DataFrame(thresholds)


def binarize_all_pathways(df, pathway_cols=None, drug_class_col='drug_class'):
    """
    Binarize all pathway scores for all drug classes.
    
    Returns
    -------
    dict : {pathway_name: (binarized_df, threshold_df)}
    """
    if pathway_cols is None:
        pathway_cols = [c for c in df.columns if c.startswith('HALLMARK_')]
    
    all_results = {}
    for pw in pathway_cols:
        print(f"  Binarizing {pw}...")
        bin_df, thresh_df = binarize_pathway_treatment(df, pw, drug_class_col)
        all_results[pw] = (bin_df, thresh_df)
        if not thresh_df.empty:
            print(f"    {len(thresh_df)} drug classes, "
                  f"treated mean={thresh_df['n_treated'].mean():.1f}, "
                  f"control mean={thresh_df['n_control'].mean():.1f}")
    
    return all_results