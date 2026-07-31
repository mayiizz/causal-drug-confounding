"""Pathway validation: compare true vs permuted vs unrelated pathway adjustments."""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


def fit_adjusted_model(df, adjust_cols, outcome_col='ln_ic50'):
    """
    Fit linear model: Y ~ treatment + adjustment covariates.
    Return R² and RMSE.
    """
    sub = df.dropna(subset=adjust_cols + [outcome_col, 'treatment'])
    if len(sub) < 30:
        return {'r2': np.nan, 'rmse': np.nan, 'n': 0}
    
    X = sub[['treatment'] + adjust_cols]
    y = sub[outcome_col].values
    
    model = LinearRegression()
    model.fit(X, y)
    
    y_pred = model.predict(X)
    
    return {
        'r2': r2_score(y, y_pred),
        'rmse': np.sqrt(mean_squared_error(y, y_pred)),
        'n': len(sub)
    }


def run_pathway_validation(df, drug_class, pathway, dataset_name,
                           tissue_col='tissue_type'):
    """
    Compare 4 models:
    A: adjust {tissue}
    B: adjust {tissue, true_pathway}
    C: adjust {tissue, permuted_pathway}
    D: adjust {tissue, unrelated_pathway}
    """
    df = df.copy()
    
    # True pathway score
    true_pw = pathway
    
    # Permuted pathway: shuffle the score IN-PLACE on df
    perm_pw = f"{pathway}_PERMUTED"
    df[perm_pw] = df[true_pw].sample(frac=1, random_state=42).values
    
    # Unrelated pathway: pick a different hallmark
    all_pws = [c for c in df.columns if c.startswith('HALLMARK_') and c != pathway and c != perm_pw]
    unrelated_pw = all_pws[0] if all_pws else None
    
    models = {
        'A_tissue_only': [tissue_col],
        'B_tissue_true_pathway': [tissue_col, true_pw],
        'C_tissue_permuted_pathway': [tissue_col, perm_pw],
    }
    
    if unrelated_pw:
        models['D_tissue_unrelated_pathway'] = [tissue_col, unrelated_pw]
    
    results = []
    for model_name, adjust_cols in models.items():
        # One-hot encode tissue
        sub = df.copy()
        tissue_dummies = pd.get_dummies(sub[tissue_col], prefix='tissue')
        sub = pd.concat([sub, tissue_dummies], axis=1)
        
        # Replace tissue_col with dummy columns
        final_adjust = [c for c in adjust_cols if c != tissue_col] + list(tissue_dummies.columns)
        
        res = fit_adjusted_model(sub, final_adjust)
        res.update({
            'dataset': dataset_name,
            'drug_class': drug_class,
            'pathway': pathway,
            'model': model_name
        })
        results.append(res)
    
    return pd.DataFrame(results)