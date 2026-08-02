"""Shared preprocessing for predictive baseline models (SVR / XGBoost).

Independent of the causal pipeline. Uses the same unified parquet tables
produced by Phase 1 (pathway scores, tissue, drug class, ln_ic50).
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr, spearmanr

ABS_LN_IC50_MAX = 15.0
RANDOM_STATE = 42
TEST_SIZE = 0.2

PATHWAY_COLS = [
    'HALLMARK_APOPTOSIS',
    'HALLMARK_DNA_REPAIR',
    'HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION',
    'HALLMARK_KRAS_SIGNALING_UP',
    'HALLMARK_PI3K_AKT_MTOR_SIGNALING',
]


def load_unified(processed_dir, dataset):
    """Load GDSC2 or CCLE unified parquet."""
    processed_dir = Path(processed_dir)
    name = 'gdsc2_unified.parquet' if dataset.upper() == 'GDSC2' else 'ccle_unified.parquet'
    path = processed_dir / name
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run Phase 1 first.")
    return pd.read_parquet(path)


def prepare_baseline_frame(df, abs_ln_ic50_max=ABS_LN_IC50_MAX):
    """
    Filter invalid outcomes and restrict to modeling columns.

    Features: tissue_type, drug_class, five Hallmark pathway scores.
    Target: ln_ic50.
    """
    work = df.copy()
    work = work[np.isfinite(work['ln_ic50'])]
    work = work[work['ln_ic50'].abs() <= abs_ln_ic50_max]
    work = work[work['drug_class'] != 'Other']
    cols = ['tissue_type', 'drug_class', 'ln_ic50'] + [
        c for c in PATHWAY_COLS if c in work.columns
    ]
    work = work[cols].dropna()
    return work.reset_index(drop=True)


def make_feature_preprocessor():
    """ColumnTransformer: one-hot tissue/drug_class + passthrough pathways."""
    categorical = ['tissue_type', 'drug_class']
    numeric = PATHWAY_COLS
    return ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical),
            ('num', 'passthrough', numeric),
        ]
    )


def train_test_matrices(df, test_size=TEST_SIZE, random_state=RANDOM_STATE):
    """Split into train/test feature frames and targets."""
    feature_cols = ['tissue_type', 'drug_class'] + PATHWAY_COLS
    X = df[feature_cols]
    y = df['ln_ic50'].values
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )


def regression_metrics(y_true, y_pred):
    """RMSE, MAE, R2, Pearson, Spearman."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[mask], y_pred[mask]
    if len(y_true) < 3:
        return {
            'rmse': np.nan, 'mae': np.nan, 'r2': np.nan,
            'pearson_r': np.nan, 'pearson_p': np.nan,
            'spearman_rho': np.nan, 'spearman_p': np.nan,
            'n_test': int(len(y_true)),
        }
    pearson_r, pearson_p = pearsonr(y_true, y_pred)
    spearman_rho, spearman_p = spearmanr(y_true, y_pred)
    return {
        'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred))),
        'mae': float(mean_absolute_error(y_true, y_pred)),
        'r2': float(r2_score(y_true, y_pred)),
        'pearson_r': float(pearson_r),
        'pearson_p': float(pearson_p),
        'spearman_rho': float(spearman_rho),
        'spearman_p': float(spearman_p),
        'n_test': int(len(y_true)),
    }


def build_model_pipeline(estimator):
    """Preprocessor + estimator pipeline."""
    return Pipeline([
        ('preprocess', make_feature_preprocessor()),
        ('model', estimator),
    ])
