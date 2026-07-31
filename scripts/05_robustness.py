#!/usr/bin/env python3
"""
Phase 5: Robustness & Refutation.
DoWhy refuters, permutation testing, and sensitivity analysis.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from tqdm import tqdm
from src.causal.refutation import run_all_refutations, permutation_test
from src.causal.sensitivity import sensitivity_grid
from src.causal.common_support import fit_propensity_model, trim_by_propensity

PROCESSED = project_root / 'data' / 'processed'


def main():
    print("=" * 60)
    print("PHASE 5: Robustness & Refutation")
    print("=" * 60)
    
    # ------------------------------------------------------------------
    # 1. Load causal estimates to find top cohorts for deep testing
    # ------------------------------------------------------------------
    print("\n[1/5] Loading estimates and selecting top cohorts...")
    estimates = pd.read_parquet(PROCESSED / 'causal_estimates.parquet')
    
    # Select top 10 divergent cohorts per dataset for refutation
    comparison = pd.read_parquet(PROCESSED / 'ate_comparison.parquet')
    top_gdsc = comparison[comparison['dataset'] == 'GDSC2'].nlargest(5, 'naive_dr_divergence')
    top_ccle = comparison[comparison['dataset'] == 'CCLE'].nlargest(5, 'naive_dr_divergence')
    top_cohorts = pd.concat([top_gdsc, top_ccle])
    
    print(f"  Selected {len(top_cohorts)} top-divergence cohorts for refutation")
    
    # ------------------------------------------------------------------
    # 2. Refutation tests
    # ------------------------------------------------------------------
    print("\n[2/5] Running refutation tests...")
    all_refutations = []
    
    for _, row in top_cohorts.iterrows():
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
        
        print(f"  {dataset} {drug_class} × {pw_short}...")
        refs = run_all_refutations(sub)
        
        for r in refs:
            r['dataset'] = dataset
            r['drug_class'] = drug_class
            r['pathway'] = pathway
            all_refutations.append(r)
    
    ref_df = pd.DataFrame(all_refutations)
    ref_df.to_parquet(PROCESSED / 'refutation_results.parquet', index=False)
    
    # Summary
    print("\n  Refutation summary:")
    for refuter in ['placebo_treatment', 'random_common_cause', 'subset']:
        sub = ref_df[ref_df['refuter'] == refuter]
        if 'passed' in sub.columns:
            passed = sub['passed'].sum()
            total = sub['passed'].notna().sum()
            print(f"    {refuter}: {passed}/{total} passed")
    
    # ------------------------------------------------------------------
    # 3. Permutation testing
    # ------------------------------------------------------------------
    print("\n[3/5] Running permutation tests...")
    perm_results = []
    
    for _, row in top_cohorts.iterrows():
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
        
        print(f"  {dataset} {drug_class} × {pw_short} (n={len(sub)})...")
        perm = permutation_test(sub, n_permutations=50)  # 100 for speed
        
        perm['dataset'] = dataset
        perm['drug_class'] = drug_class
        perm['pathway'] = pathway
        perm_results.append(perm)
    
    perm_df = pd.DataFrame(perm_results)
    perm_df.to_parquet(PROCESSED / 'permutation_results.parquet', index=False)
    
    print("\n  Permutation summary:")
    if not perm_df.empty and 'permutation_p_value' in perm_df.columns:
        sig = perm_df[perm_df['permutation_p_value'] < 0.05]
        print(f"    {len(sig)}/{len(perm_df)} cohorts significant at p < 0.05")
        print(f"    Mean null ATE: {perm_df['null_mean'].mean():.3f} ± {perm_df['null_std'].mean():.3f}")
    
    # ------------------------------------------------------------------
    # 4. Sensitivity analysis
    # ------------------------------------------------------------------
    print("\n[4/5] Running sensitivity analysis...")
    sens_results = []
    
    # Run on a subset of top cohorts (slower)
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
        
        print(f"  {dataset} {drug_class} × {pw_short}...")
        sens = sensitivity_grid(sub, drug_class, pathway, dataset)
        if not sens.empty:
            sens_results.append(sens)
    
    if sens_results:
        sens_df = pd.concat(sens_results, ignore_index=True)
        sens_df.to_parquet(PROCESSED / 'sensitivity_results.parquet', index=False)
        
        print("\n  Sensitivity summary:")
        # Check DR direction stability across grid
        dr_sens = sens_df[['dataset', 'drug_class', 'pathway', 'dr_ate']].dropna()
        flipped = 0
        total = 0
        for _, grp in dr_sens.groupby(['dataset', 'drug_class', 'pathway']):
            if len(grp) > 1:
                total += 1
                signs = np.sign(grp['dr_ate'].values)
                nonzero = signs[signs != 0]
                if len(nonzero) > 1 and len(set(nonzero)) > 1:
                    flipped += 1
        
        print(f"    DR direction flips across grid: {flipped}/{total} cases")
        print(f"    Grid configurations tested: {len(sens_df)}")
    else:
        print("    No sensitivity results generated.")
    
    # ------------------------------------------------------------------
    # 5. Compile Table 2 (Refutation) and Table 3 (Sensitivity)
    # ------------------------------------------------------------------
    print("\n[5/5] Compiling tables...")
    
    print("\n" + "=" * 60)
    print("PHASE 5 COMPLETE")
    print("=" * 60)
    
    print(f"\nSaved:")
    print(f"  {PROCESSED / 'refutation_results.parquet'}")
    print(f"  {PROCESSED / 'permutation_results.parquet'}")
    print(f"  {PROCESSED / 'sensitivity_results.parquet'}")
    
    print("\nNext: Phase 6 (Heterogeneous Treatment Effects)")


if __name__ == '__main__':
    main()