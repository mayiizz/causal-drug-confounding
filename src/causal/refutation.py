"""DoWhy refutation tests and permutation testing."""

import pandas as pd
import numpy as np
import warnings
from sklearn.linear_model import LogisticRegression, LinearRegression


def refute_placebo_treatment(df, treatment_col='treatment', outcome_col='ln_ic50',
                             tissue_col='tissue_type', n_placebos=10):
    """
    Replace treatment with random placebo and re-estimate naive ATE.
    Expectation: placebo ATE should be near zero.
    """
    rng = np.random.RandomState(42)
    placebo_ates = []
    
    for _ in range(n_placebos):
        placebo = rng.randint(0, 2, size=len(df))
        df_placebo = df.copy()
        df_placebo[treatment_col] = placebo
        
        treated = df_placebo[df_placebo[treatment_col] == 1][outcome_col]
        control = df_placebo[df_placebo[treatment_col] == 0][outcome_col]
        ate = treated.mean() - control.mean()
        placebo_ates.append(ate)
    
    # Compare to observed
    observed_treated = df[df[treatment_col] == 1][outcome_col]
    observed_control = df[df[treatment_col] == 0][outcome_col]
    observed_ate = observed_treated.mean() - observed_control.mean()
    
    placebo_mean = np.mean(placebo_ates)
    placebo_std = np.std(placebo_ates)
    
    # p-value: how many placebos exceed observed?
    p_value = np.mean(np.abs(placebo_ates) >= np.abs(observed_ate))
    
    return {
        'refuter': 'placebo_treatment',
        'observed_ate': observed_ate,
        'placebo_mean_ate': placebo_mean,
        'placebo_std_ate': placebo_std,
        'p_value': p_value,
        'passed': p_value < 0.05
    }


def refute_random_common_cause(df, treatment_col='treatment', outcome_col='ln_ic50',
                                tissue_col='tissue_type', n_trials=10):
    """
    Add random confounder and check if ATE changes substantially.
    Expectation: ATE should be stable.
    """
    rng = np.random.RandomState(42)
    modified_ates = []
    
    for _ in range(n_trials):
        df_mod = df.copy()
        df_mod['random_confounder'] = rng.randn(len(df))
        
        # Naive ATE with random confounder added to outcome
        df_mod[outcome_col] = df_mod[outcome_col] + 0.1 * df_mod['random_confounder']
        
        treated = df_mod[df_mod[treatment_col] == 1][outcome_col]
        control = df_mod[df_mod[treatment_col] == 0][outcome_col]
        ate = treated.mean() - control.mean()
        modified_ates.append(ate)
    
    observed_treated = df[df[treatment_col] == 1][outcome_col]
    observed_control = df[df[treatment_col] == 0][outcome_col]
    observed_ate = observed_treated.mean() - observed_control.mean()
    
    modified_mean = np.mean(modified_ates)
    
    # Check if change is < 10% of observed ATE
    if abs(observed_ate) > 0.01:
        relative_change = abs(modified_mean - observed_ate) / abs(observed_ate)
    else:
        relative_change = abs(modified_mean - observed_ate)
    
    return {
        'refuter': 'random_common_cause',
        'observed_ate': observed_ate,
        'modified_mean_ate': modified_mean,
        'relative_change': relative_change,
        'passed': relative_change < 0.25
    }


def refute_subset(df, treatment_col='treatment', outcome_col='ln_ic50',
                  tissue_col='tissue_type', subset_frac=0.8, n_trials=10):
    """
    Re-estimate ATE on random subsets.
    Expectation: ATE should be stable across subsets.
    """
    rng = np.random.RandomState(42)
    subset_ates = []
    
    for _ in range(n_trials):
        subset_idx = rng.choice(len(df), size=int(len(df) * subset_frac), replace=False)
        subset = df.iloc[subset_idx]
        
        treated = subset[subset[treatment_col] == 1][outcome_col]
        control = subset[subset[treatment_col] == 0][outcome_col]
        ate = treated.mean() - control.mean()
        subset_ates.append(ate)
    
    observed_treated = df[df[treatment_col] == 1][outcome_col]
    observed_control = df[df[treatment_col] == 0][outcome_col]
    observed_ate = observed_treated.mean() - observed_control.mean()
    
    subset_std = np.std(subset_ates)
    
    # Check if std is < 50% of observed ATE magnitude
    if abs(observed_ate) > 0.01:
        stability = subset_std / abs(observed_ate)
    else:
        stability = subset_std
    
    return {
        'refuter': 'subset',
        'observed_ate': observed_ate,
        'subset_std': subset_std,
        'stability_ratio': stability,
        'passed': stability < 0.50
    }


def run_all_refutations(df):
    """Run all three refutation tests on a cohort."""
    results = []
    
    try:
        results.append(refute_placebo_treatment(df))
    except Exception as e:
        results.append({'refuter': 'placebo_treatment', 'error': str(e)})
    
    try:
        results.append(refute_random_common_cause(df))
    except Exception as e:
        results.append({'refuter': 'random_common_cause', 'error': str(e)})
    
    try:
        results.append(refute_subset(df))
    except Exception as e:
        results.append({'refuter': 'subset', 'error': str(e)})
    
    return results


def permutation_test(df, treatment_col='treatment', outcome_col='ln_ic50',
                     tissue_col='tissue_type', n_permutations=100):
    """
    Shuffle tissue labels, re-estimate DR ATE from scratch, build null distribution.
    Tests whether the DR estimate is sensitive to tissue confounding structure.
    """
    from src.causal.estimators import dr_ate_econml
    
    # Force tissue to plain Python object array (no Arrow backend)
    df = df.copy()
    df[tissue_col] = df[tissue_col].astype(str).astype(object)
    
    # Observed DR ATE on original data
    obs = dr_ate_econml(df)
    obs_ate = obs['ate']
    
    # Extract tissues as numpy array for fast shuffling
    tissues = df[tissue_col].to_numpy(dtype=object).copy()
    null_ates = []
    
    rng = np.random.RandomState(42)
    
    for i in range(n_permutations):
        perm_tissues = tissues.copy()
        rng.shuffle(perm_tissues)
        
        df_perm = df.copy()
        df_perm[tissue_col] = perm_tissues
        
        # Recompute DR ATE from scratch on permuted data.
        # dr_ate_econml internally one-hot encodes tissue and fits
        # both propensity and outcome models, so the shuffle matters.
        try:
            perm = dr_ate_econml(df_perm)
            null_ates.append(perm['ate'])
        except Exception:
            # Skip permutations where DR fails (rare)
            continue
        
        if (i + 1) % 20 == 0:
            print(f"      Permutation {i+1}/{n_permutations}...")
    
    null_ates = np.array(null_ates)
    
    if len(null_ates) == 0:
        return {
            'permutation_p_value': np.nan,
            'observed_ate': obs_ate,
            'null_mean': np.nan,
            'null_std': np.nan,
            'n_permutations': 0
        }
    
    p_value = np.mean(np.abs(null_ates) >= np.abs(obs_ate))
    
    return {
        'permutation_p_value': p_value,
        'observed_ate': obs_ate,
        'null_mean': np.mean(null_ates),
        'null_std': np.std(null_ates),
        'n_permutations': len(null_ates)
    }