"""XGBoost predictive baseline (context only; not causal)."""

import pandas as pd

from src.baselines.preprocessing import (
    load_unified,
    prepare_baseline_frame,
    train_test_matrices,
    make_feature_preprocessor,
    regression_metrics,
    RANDOM_STATE,
)
from sklearn.pipeline import Pipeline


def make_xgb_estimator():
    try:
        from xgboost import XGBRegressor
    except ImportError as e:
        raise ImportError(
            "xgboost is required for the XGBoost baseline. "
            "Install with: pip install xgboost"
        ) from e

    return XGBRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.08,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='reg:squarederror',
        n_jobs=4,
        random_state=RANDOM_STATE,
    )


def run_xgboost_baseline(processed_dir, datasets=('GDSC2', 'CCLE')):
    """Train XGBoost per dataset; return results and prediction frames."""
    rows = []
    preds = []

    for dataset in datasets:
        print(f"  XGBoost: {dataset}...")
        raw = load_unified(processed_dir, dataset)
        df = prepare_baseline_frame(raw)
        print(f"    n={len(df):,} after filters")

        X_train, X_test, y_train, y_test = train_test_matrices(df)
        pipe = Pipeline([
            ('preprocess', make_feature_preprocessor()),
            ('model', make_xgb_estimator()),
        ])
        pipe.fit(X_train, y_train)
        y_hat = pipe.predict(X_test)
        metrics = regression_metrics(y_test, y_hat)
        metrics.update({
            'dataset': dataset,
            'model': 'XGBoost',
            'n_train': len(y_train),
            'n_total': len(df),
        })
        rows.append(metrics)
        print(
            f"    RMSE={metrics['rmse']:.3f} R2={metrics['r2']:.3f} "
            f"Pearson={metrics['pearson_r']:.3f}"
        )
        preds.append(pd.DataFrame({
            'dataset': dataset,
            'model': 'XGBoost',
            'y_true': y_test,
            'y_pred': y_hat,
        }))

    return pd.DataFrame(rows), pd.concat(preds, ignore_index=True)
