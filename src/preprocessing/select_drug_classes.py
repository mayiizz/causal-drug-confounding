"""Filter drug classes by minimum sample size and overlap requirements."""

import pandas as pd


def select_drug_classes(bin_results, min_treated=30, min_control=30):
    """
    Filter drug classes that meet minimum sample size requirements.
    
    Parameters
    ----------
    bin_results : dict
        Output of binarize_all_pathways: {pathway: (df, thresholds)}
    
    Returns
    -------
    dict : {pathway: {'data': df, 'thresholds': df, 'n_drug_classes': int}}
    """
    selected = {}
    
    for pw, (bin_df, thresh_df) in bin_results.items():
        if bin_df.empty or thresh_df.empty:
            print(f"  {pw}: no valid cohorts")
            continue
            
        valid = thresh_df[
            (thresh_df['n_treated'] >= min_treated) & 
            (thresh_df['n_control'] >= min_control)
        ].copy()
        
        if len(valid) == 0:
            print(f"  {pw}: 0 drug classes pass filter")
            continue
            
        valid_classes = set(valid['drug_class'])
        filtered_df = bin_df[bin_df['drug_class'].isin(valid_classes)].copy()
        
        selected[pw] = {
            'data': filtered_df,
            'thresholds': valid,
            'n_drug_classes': len(valid_classes)
        }
        
        print(f"  {pw}: {len(valid_classes)} drug classes pass filter "
              f"(min {min_treated} treated, {min_control} control)")
    
    return selected


def find_cross_dataset_cohorts(gdsc_selected, ccle_selected):
    """
    Find (drug_class, pathway) pairs that pass filters in BOTH datasets.
    These are the cohorts used for cross-dataset validation (Phase 7).
    """
    common = []
    
    gdsc_pairs = set()
    for pw, info in gdsc_selected.items():
        for dc in info['thresholds']['drug_class']:
            gdsc_pairs.add((dc, pw))
    
    ccle_pairs = set()
    for pw, info in ccle_selected.items():
        for dc in info['thresholds']['drug_class']:
            ccle_pairs.add((dc, pw))
    
    common = sorted(gdsc_pairs & ccle_pairs)
    
    print(f"\nCross-dataset cohorts: {len(common)} (drug_class, pathway) pairs")
    for dc, pw in common[:10]:
        print(f"  {dc} × {pw}")
    if len(common) > 10:
        print(f"  ... and {len(common) - 10} more")
    
    return common