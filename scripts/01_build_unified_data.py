#!/usr/bin/env python3
"""
Phase 1: Build unified GDSC2 and CCLE datasets.
Place this in scripts/ and run from project root.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.harmonize_tissue import harmonize_tissue, validate_tissue_coverage
from src.data.drug_class_map import map_drug_to_class, get_drug_class_stats
from src.data.pathway_scores import load_hallmark_gmt, compute_ssgsea_scores, compute_zscore_pathway_scores, HALLMARK_PATHWAYS
from src.data.loaders import build_gdsc2_unified, build_ccle_unified

# ------------------------------------------------------------------
# FILE PATHS - Update if your folder structure differs
# ------------------------------------------------------------------
RAW = project_root / 'data' / 'raw'
PROCESSED = project_root / 'data' / 'processed'
PROCESSED.mkdir(parents=True, exist_ok=True)

GDSC2_RESPONSE = RAW / 'GDSC2_fitted_dose_response_27Oct23.xlsx'
GDSC2_EXPRESSION = RAW / 'rnaseq_merged_20260323.zip'
CCLE_EXPRESSION = RAW / 'OmicsExpressionProteinCodingGenesTPMLogp1.csv'
CCLE_RESPONSE = RAW / 'secondary-screen-dose-response-curve-parameters.csv'
CCLE_SAMPLE_INFO = RAW / 'sample_info.csv'
MODEL_LIST = RAW / 'model_list_20260724.csv'
GMT_FILE = RAW / 'h.all.v2026.1.Hs.symbols.gmt'

# Choose: 'ssgsea' (recommended, slower) or 'zscore' (fast, fallback)
SCORING_METHOD = 'ssgsea'

# ------------------------------------------------------------------

def main():
    print("=" * 60)
    print("PHASE 1: Data Foundation & Harmonization")
    print("=" * 60)
    
    # Verify files exist
    required = {
        'GDSC2 response': GDSC2_RESPONSE,
        'GDSC2 expression': GDSC2_EXPRESSION,
        'CCLE expression': CCLE_EXPRESSION,
        'CCLE response': CCLE_RESPONSE,
        'Model list': MODEL_LIST,
        'Hallmark GMT': GMT_FILE,
    }
    missing = [name for name, path in required.items() if not path.exists()]
    if missing:
        print(f"\nMISSING FILES: {missing}")
        print("Please place all files in data/raw/ and try again.")
        return
    
    # 1. Load Hallmark pathways
    print("\n[1/6] Loading Hallmark gene sets...")
    gmt_dict = load_hallmark_gmt(str(GMT_FILE))
    print(f"  Loaded {len(gmt_dict)} pathways")
    
    # 2. Choose scoring function
    if SCORING_METHOD == 'ssgsea':
        pathway_fn = lambda expr, gmt: compute_ssgsea_scores(expr, gmt, HALLMARK_PATHWAYS)
    else:
        pathway_fn = lambda expr, gmt: compute_zscore_pathway_scores(expr, gmt, HALLMARK_PATHWAYS)
    print(f"  Scoring method: {SCORING_METHOD}")
    
    # 3. Build GDSC2
    print("\n[2/6] Building GDSC2 unified...")
    gdsc2_df = build_gdsc2_unified(
        response_file=str(GDSC2_RESPONSE),
        expression_file=str(GDSC2_EXPRESSION),
        model_list_file=str(MODEL_LIST),
        gmt_dict=gmt_dict,
        harmonize_fn=harmonize_tissue,
        drug_class_fn=map_drug_to_class,
        pathway_score_fn=pathway_fn,
        output_path=str(PROCESSED / 'gdsc2_unified.parquet')
    )
    
    # 4. Validate GDSC2
    print("\n[3/6] Validating GDSC2...")
    report = validate_tissue_coverage(gdsc2_df)
    print(f"  Coverage: {report['coverage_pct']}%")
    print(f"  Unique tissues: {report['unique_harmonized_tissues']}")
    print(f"  Top 5 tissues: {list(report['tissue_distribution'].keys())[:5]}")
    
    print("\n  Drug class distribution:")
    print(get_drug_class_stats(gdsc2_df).head(10).to_string())
    
    # 5. Build CCLE
    print("\n[4/6] Building CCLE unified...")
    ccle_df = build_ccle_unified(
        response_file=str(CCLE_RESPONSE),
        sample_info_file=str(CCLE_SAMPLE_INFO),
        expression_file=str(CCLE_EXPRESSION),
        model_list_file=str(MODEL_LIST),
        gmt_dict=gmt_dict,
        harmonize_fn=harmonize_tissue,
        drug_class_fn=map_drug_to_class,
        pathway_score_fn=pathway_fn,
        output_path=str(PROCESSED / 'ccle_unified.parquet'),
        audit_path=str(PROCESSED / 'ccle_response_correction_audit.parquet'),
    )
    
    # 6. Validate CCLE
    print("\n[5/6] Validating CCLE...")
    report = validate_tissue_coverage(ccle_df)
    print(f"  Coverage: {report['coverage_pct']}%")
    print(f"  Unique tissues: {report['unique_harmonized_tissues']}")
    print(f"  Top 5 tissues: {list(report['tissue_distribution'].keys())[:5]}")
    
    print("\n  Drug class distribution:")
    print(get_drug_class_stats(ccle_df).head(10).to_string())
    
    # 7. Pilot checks
    print("\n[6/6] Pilot checks...")
    from scipy.stats import f_oneway
    
    for name, df in [('GDSC2', gdsc2_df), ('CCLE', ccle_df)]:
        if 'HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION' not in df.columns:
            continue
        groups = [g['HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION'].dropna().values 
                  for _, g in df.groupby('tissue_type') if len(g) > 5]
        if len(groups) > 1:
            f_stat, p_val = f_oneway(*groups)
            n_total = sum(len(g) for g in groups)
            eta_sq = f_stat * (len(groups) - 1) / (f_stat * (len(groups) - 1) + (n_total - len(groups)))
            print(f"  {name}: Tissue -> EMT Eta^2 = {eta_sq:.3f}")
    
    # 8. Final summary
    print("\n" + "=" * 60)
    print("PHASE 1 COMPLETE")
    print("=" * 60)
    print(f"GDSC2: {gdsc2_df.shape[0]:,} rows x {gdsc2_df.shape[1]} cols")
    print(f"  -> {PROCESSED / 'gdsc2_unified.parquet'}")
    print(f"CCLE:  {ccle_df.shape[0]:,} rows x {ccle_df.shape[1]} cols")
    print(f"  -> {PROCESSED / 'ccle_unified.parquet'}")
    print("\nNext: Phase 2 (Preprocessing & Cohort Definition)")


if __name__ == '__main__':
    main()