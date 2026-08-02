#!/usr/bin/env python3
"""
Phase 5: Robustness & Refutation.
DoWhy-style refuters, DR tissue permutation testing, and sensitivity analysis.

Refutation scope is configurable:
  --refutation-scope top   (default: top divergent cohorts)
  --refutation-scope all   (all eligible cohorts in ate_comparison)
"""

import sys
import argparse
import time
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from src.causal.refutation import run_all_refutations
from src.causal.permutation import run_dr_permutation_suite, DEFAULT_N_PERMUTATIONS
from src.causal.sensitivity import sensitivity_grid

PROCESSED = project_root / 'data' / 'processed'


def parse_args():
    parser = argparse.ArgumentParser(description='Phase 5: Robustness & Refutation')
    parser.add_argument(
        '--n-permutations',
        type=int,
        default=DEFAULT_N_PERMUTATIONS,
        help=f'Number of tissue permutations for DR null (default {DEFAULT_N_PERMUTATIONS})',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for permutation reproducibility',
    )
    parser.add_argument(
        '--refutation-scope',
        choices=['top', 'all'],
        default='top',
        help="Refutation cohorts: 'top' divergent only (default) or 'all' eligible",
    )
    parser.add_argument(
        '--top-n-per-dataset',
        type=int,
        default=5,
        help='When scope=top, number of divergent cohorts per dataset',
    )
    parser.add_argument(
        '--permutation-only',
        action='store_true',
        help='Run only DR tissue permutation (do not overwrite refutation/sensitivity)',
    )
    parser.add_argument(
        '--permutation-scope',
        choices=['top', 'all'],
        default='top',
        help="Permutation cohorts: 'top' divergent (default) or 'all' eligible in ate_comparison",
    )
    return parser.parse_args()


def _select_permutation_cohorts(comparison, scope, top_n_per_dataset):
    """Explicit coverage selector for DR permutation (does not silently expand)."""
    if scope == 'all':
        cols = comparison[['dataset', 'drug_class', 'pathway']].drop_duplicates()
        print(f"  Permutation scope=all: {len(cols)} eligible cohorts")
        return cols

    top_gdsc = comparison[comparison['dataset'] == 'GDSC2'].nlargest(
        top_n_per_dataset, 'naive_dr_divergence'
    )
    top_ccle = comparison[comparison['dataset'] == 'CCLE'].nlargest(
        top_n_per_dataset, 'naive_dr_divergence'
    )
    top = pd.concat([top_gdsc, top_ccle])
    print(
        f"  Permutation scope=top: {len(top)} cohorts "
        f"({top_n_per_dataset} per dataset by |Naive-DR|)"
    )
    return top


def _load_perm_cohort_frames(cohort_keys):
    perm_cohorts = []
    for _, row in cohort_keys.iterrows():
        dataset = row['dataset']
        drug_class = row['drug_class']
        pathway = row['pathway']
        pw_short = pathway.replace('HALLMARK_', '')
        fpath = PROCESSED / f'{dataset.lower()}_cohort_{pw_short}.parquet'
        if not fpath.exists():
            continue
        df = pd.read_parquet(fpath)
        sub = df[df['drug_class'] == drug_class].copy()
        if len(sub) < 60:
            continue
        perm_cohorts.append({
            'data': sub,
            'dataset': dataset,
            'drug_class': drug_class,
            'pathway': pathway,
        })
    return perm_cohorts


def _save_permutation_outputs(perm_df, null_df):
    if not perm_df.empty:
        out_cols = [
            'dataset', 'drug_class', 'pathway',
            'observed_dr_ate', 'null_mean', 'null_sd',
            'empirical_p', 'z_score', 'percentile',
            'null_ci_lower', 'null_ci_upper',
            'n_permutations_requested', 'n_permutations', 'n_failed_permutations',
            'n_obs_trimmed', 'significant_0_05',
            'permutation_p_value', 'observed_ate',
        ]
        save_cols = [c for c in out_cols if c in perm_df.columns]
        perm_df[save_cols].to_parquet(
            PROCESSED / 'permutation_results.parquet', index=False
        )
        # Publication CSV with expanded fields
        perm_df[save_cols].to_csv(
            project_root / 'output' / 'table6_permutation_results.csv', index=False
        )
    else:
        print("  WARNING: No permutation summaries generated.")

    if not null_df.empty:
        null_df.to_parquet(
            PROCESSED / 'permutation_null_distribution.parquet', index=False
        )
        print(f"  Null draws saved: {len(null_df):,}")
    else:
        print("  WARNING: No null distribution rows generated.")

    print("\n  DR permutation summary:")
    if not perm_df.empty and 'empirical_p' in perm_df.columns:
        sig = perm_df[perm_df['empirical_p'] < 0.05]
        print(f"    {len(sig)}/{len(perm_df)} cohorts significant at p < 0.05")
        print(
            f"    Mean null DR ATE: {perm_df['null_mean'].mean():.3f} "
            f"+/- {perm_df['null_sd'].mean():.3f}"
        )
        if 'n_permutations_requested' in perm_df.columns:
            print(
                f"    Requested permutations/cohort: "
                f"{int(perm_df['n_permutations_requested'].iloc[0])}"
            )
        print(
            f"    Successful permutations (sum): "
            f"{int(perm_df['n_permutations'].sum())}; "
            f"failed (sum): {int(perm_df['n_failed_permutations'].sum())}"
        )


def _select_refutation_cohorts(comparison, scope, top_n_per_dataset):
    if scope == 'all':
        cols = comparison[['dataset', 'drug_class', 'pathway']].drop_duplicates()
        print(f"  Refutation scope=all: {len(cols)} eligible cohorts")
        return cols

    top_gdsc = comparison[comparison['dataset'] == 'GDSC2'].nlargest(
        top_n_per_dataset, 'naive_dr_divergence'
    )
    top_ccle = comparison[comparison['dataset'] == 'CCLE'].nlargest(
        top_n_per_dataset, 'naive_dr_divergence'
    )
    top = pd.concat([top_gdsc, top_ccle])
    print(
        f"  Refutation scope=top: {len(top)} cohorts "
        f"({top_n_per_dataset} per dataset by |Naive-DR|)"
    )
    return top


def _build_refutation_summary(ref_df):
    if ref_df is None or ref_df.empty:
        return pd.DataFrame()

    rows = []
    for keys, g in ref_df.groupby(['dataset', 'drug_class', 'pathway']):
        dataset, drug_class, pathway = keys
        row = {
            'dataset': dataset,
            'drug_class': drug_class,
            'pathway': pathway,
            'placebo_result': np.nan,
            'random_common_cause_result': np.nan,
            'subset_result': np.nan,
            'placebo_passed': np.nan,
            'random_common_cause_passed': np.nan,
            'subset_passed': np.nan,
            'overall_refutation_status': 'FAIL',
            'success': False,
            'runtime_seconds': float(g['runtime_seconds'].sum())
            if 'runtime_seconds' in g.columns else np.nan,
            'n_refuters': len(g),
            'n_errors': int(g['error'].notna().sum()) if 'error' in g.columns else 0,
        }
        for _, r in g.iterrows():
            name = r.get('refuter')
            passed = r.get('passed')
            err = r.get('error') if 'error' in r.index else np.nan
            status = 'error' if (isinstance(err, str) or pd.notna(err)) and str(err) not in ('nan', '') else (
                'pass' if passed else 'fail'
            )
            if name == 'placebo_treatment':
                row['placebo_result'] = status
                row['placebo_passed'] = passed
            elif name == 'random_common_cause':
                row['random_common_cause_result'] = status
                row['random_common_cause_passed'] = passed
            elif name == 'subset':
                row['subset_result'] = status
                row['subset_passed'] = passed

        passes = [
            row['placebo_passed'],
            row['random_common_cause_passed'],
            row['subset_passed'],
        ]
        valid = [p for p in passes if pd.notna(p)]
        if valid and all(bool(p) for p in valid):
            row['overall_refutation_status'] = 'PASS'
            row['success'] = True
        elif valid and any(bool(p) for p in valid):
            row['overall_refutation_status'] = 'PARTIAL'
            row['success'] = False
        else:
            row['overall_refutation_status'] = 'FAIL'
            row['success'] = False
        rows.append(row)

    return pd.DataFrame(rows)


def main():
    args = parse_args()

    print("=" * 60)
    print("PHASE 5: Robustness & Refutation")
    print("=" * 60)
    print(f"  DR permutations per cohort: {args.n_permutations} (seed={args.seed})")
    print(f"  Refutation scope: {args.refutation_scope}")
    print(f"  Permutation scope: {args.permutation_scope}")
    print(f"  Permutation-only: {args.permutation_only}")

    print("\n[1/5] Loading estimates and selecting cohorts...")
    comparison = pd.read_parquet(PROCESSED / 'ate_comparison.parquet')
    perm_keys = _select_permutation_cohorts(
        comparison, args.permutation_scope, args.top_n_per_dataset
    )

    if not args.permutation_only:
        estimates = pd.read_parquet(PROCESSED / 'causal_estimates.parquet')  # noqa: F841
        refutation_cohorts = _select_refutation_cohorts(
            comparison, args.refutation_scope, args.top_n_per_dataset
        )
        # Sensitivity still uses fixed top-5/dataset divergent set (unchanged)
        top_gdsc = comparison[comparison['dataset'] == 'GDSC2'].nlargest(
            5, 'naive_dr_divergence'
        )
        top_ccle = comparison[comparison['dataset'] == 'CCLE'].nlargest(
            5, 'naive_dr_divergence'
        )
        top_cohorts = pd.concat([top_gdsc, top_ccle])

        print("\n[2/5] Running refutation tests...")
        all_refutations = []

        for _, row in refutation_cohorts.iterrows():
            dataset = row['dataset']
            drug_class = row['drug_class']
            pathway = row['pathway']
            pw_short = pathway.replace('HALLMARK_', '')

            fpath = PROCESSED / f'{dataset.lower()}_cohort_{pw_short}.parquet'
            if not fpath.exists():
                continue

            df = pd.read_parquet(fpath)
            sub = df[df['drug_class'] == drug_class].copy()
            if len(sub) < 60:
                continue

            print(f"  {dataset} {drug_class} x {pw_short}...")
            t0 = time.perf_counter()
            refs = run_all_refutations(sub)
            elapsed = time.perf_counter() - t0

            for r in refs:
                r['dataset'] = dataset
                r['drug_class'] = drug_class
                r['pathway'] = pathway
                r['runtime_seconds'] = elapsed / max(len(refs), 1)
                if 'error' not in r:
                    r['error'] = np.nan
                all_refutations.append(r)

        ref_df = pd.DataFrame(all_refutations)
        ref_df.to_parquet(PROCESSED / 'refutation_results.parquet', index=False)
        summary_df = _build_refutation_summary(ref_df)
        if not summary_df.empty:
            summary_df.to_parquet(PROCESSED / 'refutation_summary.parquet', index=False)

        print("\n  Refutation summary:")
        for refuter in ['placebo_treatment', 'random_common_cause', 'subset']:
            sub = ref_df[ref_df['refuter'] == refuter]
            if 'passed' in sub.columns:
                passed = sub['passed'].sum()
                total = sub['passed'].notna().sum()
                print(f"    {refuter}: {passed}/{total} passed")
        if not summary_df.empty:
            print(
                f"    Overall PASS/PARTIAL/FAIL: "
                f"{(summary_df['overall_refutation_status']=='PASS').sum()}/"
                f"{(summary_df['overall_refutation_status']=='PARTIAL').sum()}/"
                f"{(summary_df['overall_refutation_status']=='FAIL').sum()}"
            )
    else:
        print("\n[2/5] Skipping refutation (--permutation-only).")
        top_cohorts = None

    print("\n[3/5] Running DR tissue permutation tests...")
    print("  Null: tissue assignment does not produce the observed DR ATE.")
    print(
        f"  Coverage: permutation-scope={args.permutation_scope} "
        f"({len(perm_keys)} key rows selected)."
    )

    perm_cohorts = _load_perm_cohort_frames(perm_keys)
    print(f"  Cohorts loaded for permutation: {len(perm_cohorts)}")

    t_perm0 = time.perf_counter()
    perm_df, null_df = run_dr_permutation_suite(
        perm_cohorts,
        n_permutations=args.n_permutations,
        random_state=args.seed,
        verbose=True,
    )
    t_perm1 = time.perf_counter()
    print(f"  Permutation wall time: {t_perm1 - t_perm0:.1f}s")

    _save_permutation_outputs(perm_df, null_df)

    if not args.permutation_only:
        print("\n[4/5] Running sensitivity analysis...")
        sens_results = []
        for _, row in top_cohorts.head(6).iterrows():
            dataset = row['dataset']
            drug_class = row['drug_class']
            pathway = row['pathway']
            pw_short = pathway.replace('HALLMARK_', '')
            fpath = PROCESSED / f'{dataset.lower()}_cohort_{pw_short}.parquet'
            if not fpath.exists():
                continue
            df = pd.read_parquet(fpath)
            sub = df[df['drug_class'] == drug_class].copy()
            if len(sub) < 60:
                continue
            print(f"  {dataset} {drug_class} x {pw_short}...")
            sens = sensitivity_grid(sub, drug_class, pathway, dataset)
            if not sens.empty:
                sens_results.append(sens)

        if sens_results:
            sens_df = pd.concat(sens_results, ignore_index=True)
            sens_df.to_parquet(PROCESSED / 'sensitivity_results.parquet', index=False)
            print(f"    Grid configurations tested: {len(sens_df)}")
        else:
            print("    No sensitivity results generated.")
    else:
        print("\n[4/5] Skipping sensitivity (--permutation-only).")

    print("\n[5/5] Compiling tables...")
    print("\n" + "=" * 60)
    print("PHASE 5 COMPLETE")
    print("=" * 60)
    print(f"\nSaved:")
    if not args.permutation_only:
        print(f"  {PROCESSED / 'refutation_results.parquet'}")
        print(f"  {PROCESSED / 'refutation_summary.parquet'}")
        print(f"  {PROCESSED / 'sensitivity_results.parquet'}")
    print(f"  {PROCESSED / 'permutation_results.parquet'}")
    print(f"  {PROCESSED / 'permutation_null_distribution.parquet'}")
    print(f"  {project_root / 'output' / 'table6_permutation_results.csv'}")
    print("\nNext: Phase 6 (Heterogeneous Treatment Effects)")


if __name__ == '__main__':
    main()
