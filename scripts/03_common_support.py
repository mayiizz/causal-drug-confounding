#!/usr/bin/env python3
"""
Phase 3: Common Support & Propensity Modeling.
Fits propensity models and trims by overlap for all cohorts.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
from src.causal.common_support import run_common_support_pipeline

PROCESSED = project_root / 'data' / 'processed'


def main():
    print("=" * 60)
    print("PHASE 3: Common Support & Propensity Modeling")
    print("=" * 60)
    
    # ------------------------------------------------------------------
    # 1. Load cohort data
    # ------------------------------------------------------------------
    print("\n[1/3] Loading cohort data...")
    pathways = [
        'HALLMARK_APOPTOSIS',
        'HALLMARK_DNA_REPAIR',
        'HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION',
        'HALLMARK_KRAS_SIGNALING_UP',
        'HALLMARK_PI3K_AKT_MTOR_SIGNALING'
    ]
    
    all_reports = []
    
    # ------------------------------------------------------------------
    # 2. Process GDSC2
    # ------------------------------------------------------------------
    print("\n[2/3] Assessing common support for GDSC2...")
    for pw in pathways:
        pw_short = pw.replace('HALLMARK_', '')
        fpath = PROCESSED / f'gdsc2_cohort_{pw_short}.parquet'
        if not fpath.exists():
            print(f"  Skipping {pw} (file not found)")
            continue
        
        df = pd.read_parquet(fpath)
        print(f"  Processing {pw} ({len(df):,} rows)...")
        report = run_common_support_pipeline(df, 'GDSC2')
        if not report.empty:
            all_reports.append(report)
            print(f"    {len(report)} records (default bounds)")
    
    # ------------------------------------------------------------------
    # 3. Process CCLE
    # ------------------------------------------------------------------
    print("\n[3/3] Assessing common support for CCLE...")
    for pw in pathways:
        pw_short = pw.replace('HALLMARK_', '')
        fpath = PROCESSED / f'ccle_cohort_{pw_short}.parquet'
        if not fpath.exists():
            print(f"  Skipping {pw} (file not found)")
            continue
        
        df = pd.read_parquet(fpath)
        print(f"  Processing {pw} ({len(df):,} rows)...")
        report = run_common_support_pipeline(df, 'CCLE')
        if not report.empty:
            all_reports.append(report)
            print(f"    {len(report)} records (default bounds)")
    
    # ------------------------------------------------------------------
    # 4. Compile and save
    # ------------------------------------------------------------------
    if all_reports:
        full_report = pd.concat(all_reports, ignore_index=True)
        
        # Save full report
        full_report.to_parquet(PROCESSED / 'common_support_report.parquet', index=False)
        
        # Save default bounds only (0.10-0.90) for downstream use
        default_report = full_report[
            (full_report['propensity_lower'] == 0.10) & 
            (full_report['propensity_upper'] == 0.90)
        ].copy()
        default_report.to_parquet(PROCESSED / 'common_support_default.parquet', index=False)
        
        # Print summary
        print("\n" + "=" * 60)
        print("PHASE 3 COMPLETE")
        print("=" * 60)
        print(f"Total assessments: {len(full_report)}")
        print(f"Default-bound cohorts: {len(default_report)}")
        
        print("\n--- Default bounds summary ---")
        summary = default_report.groupby('dataset').agg({
            'n_original': 'mean',
            'n_trimmed': 'mean',
            'percent_retained': 'mean',
            'overlap_coefficient': 'mean',
            'smd_before': 'mean',
            'smd_after': 'mean'
        }).round(3)
        print(summary.to_string())
        
        print(f"\nSaved:")
        print(f"  {PROCESSED / 'common_support_report.parquet'}")
        print(f"  {PROCESSED / 'common_support_default.parquet'}")
        
        # Flag problematic cohorts
        flagged = default_report[default_report['overlap_coefficient'] < 0.5]
        if len(flagged) > 0:
            print(f"\n⚠️  Flagged {len(flagged)} cohorts with overlap < 0.5:")
            for _, row in flagged.head(5).iterrows():
                print(f"  {row['dataset']} {row['drug_class']} {row['pathway']}: overlap={row['overlap_coefficient']:.3f}")
        
        print("\nNext: Phase 4 (Causal Estimation)")
    else:
        print("\nERROR: No common support assessments generated.")


if __name__ == '__main__':
    main()