"""SVR predictive baseline (context only; not causal)."""

from sklearn.svm import LinearSVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from src.baselines.preprocessing import (
    load_unified,
    prepare_baseline_frame,
    train_test_matrices,
    make_feature_preprocessor,
    regression_metrics,
    RANDOM_STATE,
)


def make_svr_estimator():
    """
    Linear SVR on standardized features (scalable to ~100k–300k rows).
    Nonlinear RBF SVR is intentionally avoided for runtime on full unified tables.
    """
    return Pipeline([
        ('scale', StandardScaler()),
        ('svr', LinearSVR(
            max_iter=5000,
            dual='auto',
            random_state=RANDOM_STATE,
            C=1.0,
        )),
    ])


def run_svr_baseline(processed_dir, datasets=('GDSC2', 'CCLE')):
    """
    Train SVR per dataset; return results DataFrame and prediction frames.
    """
    import pandas as pd

    rows = []
    preds = []

    for dataset in datasets:
        print(f"  SVR: {dataset}...")
        raw = load_unified(processed_dir, dataset)
        df = prepare_baseline_frame(raw)
        print(f"    n={len(df):,} after filters")

        X_train, X_test, y_train, y_test = train_test_matrices(df)
        pipe = Pipeline([
            ('preprocess', make_feature_preprocessor()),
            ('model', make_svr_estimator()),
        ])
        pipe.fit(X_train, y_train)
        y_hat = pipe.predict(X_test)
        metrics = regression_metrics(y_test, y_hat)
        metrics.update({
            'dataset': dataset,
            'model': 'SVR',
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
            'model': 'SVR',
            'y_true': y_test,
            'y_pred': y_hat,
        }))

    return pd.DataFrame(rows), pd.concat(preds, ignore_index=True)
