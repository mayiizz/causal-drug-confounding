"""Loaders adapted to the user's exact files with flexible column detection."""

import pandas as pd
import numpy as np
from pathlib import Path
import zipfile


def _find_column(candidates, available_cols):
    """Find first matching column via substring search."""
    available_lower = {c.lower().replace('_', '').replace(' ', ''): c for c in available_cols}
    for cand in candidates:
        key = cand.lower().replace('_', '').replace(' ', '')
        if key in available_lower:
            return available_lower[key]
    return None


def load_model_list(filepath: str):
    df = pd.read_csv(filepath)
    df.columns = [c.strip() for c in df.columns]
    
    col_map = {}
    for c in df.columns:
        lc = c.lower().replace('_', '').replace(' ', '')
        if 'model' in lc and 'id' in lc:
            col_map['model_id'] = c
        elif lc == 'celllinename':
            col_map['cell_line_name'] = c
        elif lc == 'strippedcelllinename':
            col_map['stripped_name'] = c
        elif 'cosmic' in lc:
            col_map['cosmic_id'] = c
        elif 'sanger' in lc and 'model' in lc:
            col_map['sanger_model_id'] = c
        elif lc == 'lineage':
            col_map['lineage'] = c
        elif lc == 'lineagesubtype':
            col_map['lineage_subtype'] = c
        elif lc == 'primarydisease':
            col_map['primary_disease'] = c
    
    print(f"  Model list columns mapped: {col_map}")
    return df, col_map


def load_gdsc2_response(filepath: str) -> pd.DataFrame:
    df = pd.read_excel(filepath, engine='openpyxl')
    df.columns = [c.strip() for c in df.columns]
    print(f"  GDSC2 response columns: {df.columns.tolist()[:20]}")
    
    # Flexible column detection
    sanger_col = _find_column(['SANGER_MODEL_ID', 'SANGERMODELID', 'MODEL_ID'], df.columns)
    name_col = _find_column(['CELL_LINE_NAME', 'CELLLINENAME', 'CELL_LINE', 'CELLLINE'], df.columns)
    drug_id_col = _find_column(['DRUG_ID', 'DRUGID'], df.columns)
    drug_name_col = _find_column(['DRUG_NAME', 'DRUGNAME', 'COMPOUND_NAME', 'NAME'], df.columns)
    ic50_col = _find_column(['LN_IC50', 'LNIC50', 'LOG_IC50', 'IC50'], df.columns)
    cancer_type_col = _find_column(['CANCER_TYPE', 'CANCERTYPE', 'TISSUE', 'LINEAGE'], df.columns)
    
    required = {'drug_name': drug_name_col, 'ln_ic50': ic50_col}
    missing = [k for k, v in required.items() if v is None]
    if missing:
        raise ValueError(f"Could not find columns for {missing}. Available: {df.columns.tolist()}")
    
    result = pd.DataFrame({
        'sanger_model_id': df[sanger_col].astype(str).str.strip() if sanger_col else None,
        'cell_line_name': df[name_col].astype(str).str.strip() if name_col else None,
        'drug_id': df[drug_id_col].astype(str).str.strip() if drug_id_col else None,
        'drug_name': df[drug_name_col].astype(str).str.strip(),
        'ln_ic50': pd.to_numeric(df[ic50_col], errors='coerce'),
        'cancer_type': df[cancer_type_col].astype(str).str.strip() if cancer_type_col else None,
    })
    
    result = result.dropna(subset=['ln_ic50'])
    return result


def load_gdsc2_expression(filepath: str) -> pd.DataFrame:
    """Load GDSC2 RNA-seq from long-format CSV and pivot to genes x samples."""
    path = Path(filepath)
    extract_dir = path.parent / 'gdsc2_rnaseq_extracted'
    
    if not extract_dir.exists():
        with zipfile.ZipFile(path, 'r') as z:
            z.extractall(extract_dir)
    
    # Find the main data file (not the _rsem_tpm, _fpkm, _count variants which are same data)
    candidates = list(extract_dir.glob('rnaseq_merged_202*.csv'))
    if not candidates:
        candidates = list(extract_dir.glob('*.csv'))
    
    filepath = max(candidates, key=lambda p: p.stat().st_size)
    print(f"  GDSC2 expression file: {filepath.name} ({filepath.stat().st_size / 1024 / 1024:.1f} MB)")
    
    # Load long-format data
    print("  Loading long-format expression data...")
    df = pd.read_csv(filepath)
    print(f"  Raw shape: {df.shape}")
    print(f"  Columns: {list(df.columns)}")
    
    # Use rsem_tpm as the expression value
    value_col = 'rsem_tpm'
    if value_col not in df.columns:
        # Fallback to other TPM columns
        for col in ['rsem_tpm', 'htseq_fpkm', 'rsem_fpkm']:
            if col in df.columns:
                value_col = col
                break
    
    # Pivot: index = gene_symbol, columns = model_id, values = rsem_tpm
    print(f"  Pivoting to wide format using '{value_col}'...")
    wide = df.pivot_table(
        index='gene_symbol',
        columns='model_id',
        values=value_col,
        aggfunc='mean'  # In case of duplicates
    )
    
    # Clean up
    wide.index = wide.index.astype(str).str.upper().str.strip()
    wide.columns = wide.columns.astype(str).str.strip()
    
    # Drop rows with too many NaNs
    wide = wide.dropna(thresh=wide.shape[1] * 0.5)  # Keep genes present in >50% of samples
    
    print(f"  Final expression: {wide.shape[0]:,} genes x {wide.shape[1]:,} samples")
    print(f"  First 5 samples: {list(wide.columns[:5])}")
    
    return wide


def load_ccle_expression(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath, index_col=0)
    df.columns = [c.split(' ')[0].upper().strip() for c in df.columns]
    df.index = df.index.astype(str).str.strip()
    df = df.T
    return df


def load_ccle_response(filepath: str) -> pd.DataFrame:
    """Load DepMap PRISM secondary screen and log-transform IC50."""
    df = pd.read_csv(filepath, low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    print(f"  CCLE response columns: {df.columns.tolist()[:20]}")
    
    drug_col = _find_column(['DRUG_NAME', 'DRUGNAME', 'NAME', 'COMPOUND', 'DRUG'], df.columns)
    ic50_col = _find_column(['LN_IC50', 'LNIC50', 'LOG2_IC50', 'IC50', 'LOG_IC50', 'AUC'], df.columns)
    cell_col = _find_column(['MODEL_ID', 'DEPMAP_ID', 'CCLE_NAME', 'CELL_LINE_DISPLAY_NAME', 
                             'CELL_LINE_NAME', 'CELLLINENAME', 'MODELID'], df.columns)
    
    if drug_col is None or ic50_col is None:
        raise ValueError(f"Could not find drug/IC50 columns. Have: {df.columns.tolist()[:15]}")
    
    print(f"  Mapped: cell={cell_col}, drug={drug_col}, ic50={ic50_col}")
    
    result = pd.DataFrame({
        'cell_line_id': df[cell_col].astype(str).str.strip() if cell_col else 'unknown',
        'drug_name': df[drug_col].astype(str).str.strip(),
        'raw_ic50': pd.to_numeric(df[ic50_col], errors='coerce')
    })
    
    # =====================================================================
    # FIX: Clean raw IC50 BEFORE any transformation
    # =====================================================================
    n_raw = len(result)
    
    # 1. Drop NaN, inf, zero, and negative values
    result = result.dropna(subset=['raw_ic50'])
    result = result[np.isfinite(result['raw_ic50'])]
    result = result[result['raw_ic50'] > 0]
    n_clean = len(result)
    print(f"    Dropped {n_raw - n_clean} rows with invalid IC50 (NaN/inf/zero/negative)")
    
    # 2. Detect AUC (0-1) vs raw IC50 (>0)
    if result['raw_ic50'].max() <= 1.5 and result['raw_ic50'].min() >= 0:
        # AUC: convert to -log(AUC) so higher = more sensitive
        print(f"    Detected AUC scale (max={result['raw_ic50'].max():.3f}), converting to -log(AUC)")
        result['ln_ic50'] = -np.log(result['raw_ic50'].clip(lower=1e-6))
    else:
        # Raw IC50: take natural log
        print(f"    Detected IC50 scale (max={result['raw_ic50'].max():.1f}), converting to ln(IC50)")
        result['ln_ic50'] = np.log(result['raw_ic50'])
    
    # 3. Hard-cap extreme outliers in log-space
    #    GDSC2 range is [-2.7, +7.8]. Bounds [-15, 10] catch 1e299 placeholders
    #    while preserving legitimate resistant lines.
    n_before_cap = len(result)
    result = result[(result['ln_ic50'] >= -15) & (result['ln_ic50'] <= 10)]
    n_after_cap = len(result)
    print(f"    Capped {n_before_cap - n_after_cap} log-outliers outside [-15, 10]")
    
    result = result.dropna(subset=['ln_ic50'])
    print(f"  Final response: {len(result)} rows, ln_ic50 range: [{result['ln_ic50'].min():.2f}, {result['ln_ic50'].max():.2f}]")
    
    return result


def load_ccle_sample_info(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    df.columns = [c.strip() for c in df.columns]
    
    id_col = _find_column(['DEPMAP_ID', 'MODEL_ID', 'MODELID'], df.columns)
    name_col = _find_column(['STRIPPED_CELL_LINE_NAME', 'CELL_LINE_NAME', 'CELLLINENAME'], df.columns)
    lineage_col = _find_column(['LINEAGE', 'PRIMARY_DISEASE'], df.columns)
    
    keep = [c for c in [id_col, name_col, lineage_col] if c is not None]
    df = df[keep].copy()
    
    rename = {}
    if id_col: rename[id_col] = 'depmap_id'
    if name_col: rename[name_col] = 'stripped_name'
    if lineage_col: rename[lineage_col] = 'lineage'
    
    return df.rename(columns=rename)


def build_gdsc2_unified(response_file, expression_file, model_list_file,
                        gmt_dict, harmonize_fn, drug_class_fn, pathway_score_fn,
                        output_path=None):
    print("\n--- Building GDSC2 ---")
    
    resp = load_gdsc2_response(response_file)
    print(f"  Response: {len(resp):,} rows")
    
    expr = load_gdsc2_expression(expression_file)
    print(f"  Expression: {expr.shape[0]:,} genes x {expr.shape[1]:,} samples")
    
    # GDSC2 has CANCER_TYPE directly — use it as tissue
    if resp['cancer_type'].notna().any():
        print("  Using CANCER_TYPE as tissue source")
        resp['tissue_type_raw'] = resp['cancer_type']
        resp['tissue_type'] = harmonize_fn(resp['tissue_type_raw'])
    else:
        # Fallback: try model_list
        model_df, model_cols = load_model_list(model_list_file)
        if 'sanger_model_id' in model_cols:
            model_df['match_id'] = model_df[model_cols['sanger_model_id']].astype(str).str.strip()
            resp['match_id'] = resp['sanger_model_id']
            lineage_col = model_cols.get('lineage')
            if lineage_col:
                tissue_map = model_df[['match_id', lineage_col]].drop_duplicates()
                resp = resp.merge(tissue_map, on='match_id', how='left')
                resp['tissue_type_raw'] = resp[lineage_col]
                resp['tissue_type'] = harmonize_fn(resp['tissue_type_raw'])
    
    resp['drug_class'] = drug_class_fn(resp['drug_name'])
    
    print("  Computing pathway scores...")
    pw_scores = pathway_score_fn(expr, gmt_dict)
    pw_scores = pw_scores.reset_index()
    pw_scores['cell_line_id'] = pw_scores['cell_line_id'].astype(str).str.strip()
    
    # Try multiple matching strategies for expression
    resp['expr_key'] = resp['sanger_model_id'] if resp['sanger_model_id'].notna().any() else resp['cell_line_name']
    score_ids = set(pw_scores['cell_line_id'])
    
    for key_col in ['sanger_model_id', 'cell_line_name']:
        if key_col not in resp.columns:
            continue
        test_ids = set(resp[key_col].dropna().astype(str).str.strip())
        overlap = score_ids & test_ids
        print(f"  Overlap ({key_col}): {len(overlap)}")
        if len(overlap) > 50:
            resp['expr_key'] = resp[key_col].astype(str).str.strip()
            break
    
    unified = resp.merge(pw_scores, left_on='expr_key', right_on='cell_line_id', how='inner')
    
    unified['dataset_name'] = 'GDSC2'
    unified['cell_line_id'] = unified['expr_key']
    
    schema_cols = ['dataset_name', 'cell_line_id', 'tissue_type', 'drug_id', 'drug_class', 'ln_ic50']
    pw_cols = [c for c in unified.columns if c.startswith('HALLMARK_')]
    final_cols = [c for c in schema_cols + pw_cols if c in unified.columns]
    unified = unified[final_cols].copy()
    
    print(f"  Unified: {unified.shape[0]:,} rows x {unified.shape[1]} cols")
    print(f"  Tissues: {unified['tissue_type'].nunique()}")
    print(f"  Drug classes: {unified['drug_class'].nunique()}")
    
    if output_path:
        unified.to_parquet(output_path, index=False)
        print(f"  Saved: {output_path}")
    
    return unified


def build_ccle_unified(response_file, sample_info_file, expression_file, model_list_file,
                       gmt_dict, harmonize_fn, drug_class_fn, pathway_score_fn,
                       output_path=None):
    print("\n--- Building CCLE ---")
    
    resp = load_ccle_response(response_file)
    print(f"  Response: {len(resp):,} rows")
    
    expr = load_ccle_expression(expression_file)
    print(f"  Expression: {expr.shape[0]:,} genes x {expr.shape[1]:,} samples")
    
    model_df, model_cols = load_model_list(model_list_file)
    
    resp['match_key'] = resp['cell_line_id'].str.upper().str.strip()
    
    if 'model_id' in model_cols:
        model_df['match_key'] = model_df[model_cols['model_id']].astype(str).str.upper().str.strip()
    elif 'stripped_name' in model_cols:
        model_df['match_key'] = model_df[model_cols['stripped_name']].astype(str).str.upper().str.strip()
    
    lineage_col = model_cols.get('lineage')
    if lineage_col:
        tissue_map = model_df[['match_key', lineage_col]].drop_duplicates()
        resp = resp.merge(tissue_map, on='match_key', how='left')
        resp['tissue_type_raw'] = resp[lineage_col]
    else:
        sinfo = load_ccle_sample_info(sample_info_file)
        resp = resp.merge(sinfo[['depmap_id', 'lineage']], 
                         left_on='cell_line_id', right_on='depmap_id', how='left')
        resp['tissue_type_raw'] = resp['lineage']
    
    resp['tissue_type'] = harmonize_fn(resp['tissue_type_raw'])
    resp['drug_class'] = drug_class_fn(resp['drug_name'])
    
    print("  Computing pathway scores...")
    pw_scores = pathway_score_fn(expr, gmt_dict)
    pw_scores = pw_scores.reset_index()
    pw_scores['expr_id'] = pw_scores['cell_line_id'].astype(str).str.upper().str.strip()
    
    resp['expr_key'] = resp['cell_line_id'].str.upper().str.strip()
    
    score_ids = set(pw_scores['expr_id'])
    resp_ids = set(resp['expr_key'])
    overlap = score_ids & resp_ids
    print(f"  Expression/Response overlap: {len(overlap)} cell lines")
    
    unified = resp.merge(pw_scores, left_on='expr_key', right_on='expr_id', how='inner')
    
    unified['dataset_name'] = 'CCLE'
    unified['cell_line_id'] = unified['expr_key']
    unified['drug_id'] = unified['drug_name']
    
    schema_cols = ['dataset_name', 'cell_line_id', 'tissue_type', 'drug_id', 'drug_class', 'ln_ic50']
    pw_cols = [c for c in unified.columns if c.startswith('HALLMARK_')]
    final_cols = [c for c in schema_cols + pw_cols if c in unified.columns]
    unified = unified[final_cols].copy()
    
    print(f"  Unified: {unified.shape[0]:,} rows x {unified.shape[1]} cols")
    print(f"  Tissues: {unified['tissue_type'].nunique()}")
    print(f"  Drug classes: {unified['drug_class'].nunique()}")
    
    if output_path:
        unified.to_parquet(output_path, index=False)
        print(f"  Saved: {output_path}")
    
    return unified