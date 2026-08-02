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


# PRISM IC50 outside [exp(-ABS), exp(+ABS)] is treated as failed-curve extrapolation
# (Stage 1 audit |ln|>15; matches baseline ABS_LN_IC50_MAX). Not a silent SD clip.
CCLE_LN_IC50_ABS_MAX = 15.0

# Multi-screen PRISM replicates: one statistical unit = cell line × drug
CCLE_DUP_AGGREGATION_METHOD = 'median'


def aggregate_ccle_response_duplicates(
    kept: pd.DataFrame,
    method: str = CCLE_DUP_AGGREGATION_METHOD,
    audit_path: str = None,
) -> pd.DataFrame:
    """
    Collapse multi-screen PRISM replicates to one row per cell_line × drug.

    Default method = median of corrected ln(IC50):
      - Duplicates are screening-campaign replicates (HTS/MTS), not loader errors.
      - Median is robust to discordant campaigns and does not overweight any screen.
      - For causal IID analyses the unit must be one cell × one drug.

    Alternatives considered (not default):
      mean — sensitive to outlier screens
      highest R² — prefers one campaign; discards concordant information
      'best' replicate — ambiguous without a pre-specified quality hierarchy
    """
    if kept.empty:
        return kept

    work = kept.copy()
    if 'screen_id' not in work.columns:
        work['screen_id'] = ''
    if 'r2' not in work.columns:
        work['r2'] = np.nan

    work['cell_line_id'] = work['cell_line_id'].astype(str).str.strip()
    work['drug_name'] = work['drug_name'].astype(str).str.strip()

    method = (method or 'median').lower().strip()
    keys = ['cell_line_id', 'drug_name']

    def _join_screens(s):
        vals = sorted({str(x) for x in s if str(x) not in ('', 'nan', 'None')})
        return ';'.join(vals)

    def _join_vals(s):
        return ';'.join(f'{float(v):.6g}' for v in s)

    base = work.groupby(keys, sort=False).agg(
        number_of_replicates=('ln_ic50', 'size'),
        screen_ids=('screen_id', _join_screens),
        original_values=('ln_ic50', _join_vals),
        within_pair_variance=('ln_ic50', lambda s: float(np.var(s, ddof=1)) if len(s) > 1 else 0.0),
        mean_ln=('ln_ic50', 'mean'),
        median_ln=('ln_ic50', 'median'),
    )

    if method in ('highest_r2', 'best_r2', 'best'):
        tmp = work.assign_value(subset=['r2'], value=-np.inf)
        idx = tmp.groupby(keys, sort=False)['r2'].idxmax()
        best = work.loc[idx, keys + ['ln_ic50']].set_index(keys)['ln_ic50']
        all_nan_r2 = work.groupby(keys)['r2'].apply(lambda s: bool(s.isna().all()))
        agg_ln = best.reindex(base.index)
        fallback_idx = all_nan_r2[all_nan_r2].index
        if len(fallback_idx):
            agg_ln.loc[fallback_idx] = base.loc[fallback_idx, 'median_ln']
        base['aggregated_value'] = agg_ln.values
        base['aggregation_method'] = np.where(
            all_nan_r2.reindex(base.index).fillna(False),
            'median_fallback_no_r2', 'highest_r2'
        )
    elif method == 'mean':
        base['aggregated_value'] = base['mean_ln']
        base['aggregation_method'] = np.where(
            base['number_of_replicates'] == 1, 'identity', 'mean'
        )
    else:
        base['aggregated_value'] = base['median_ln']
        base['aggregation_method'] = np.where(
            base['number_of_replicates'] == 1, 'identity', 'median'
        )

    dup_audit = base.reset_index().rename(columns={
        'cell_line_id': 'cell_line',
        'drug_name': 'drug',
    })[[
        'cell_line', 'drug', 'number_of_replicates', 'screen_ids',
        'original_values', 'aggregated_value', 'aggregation_method',
        'within_pair_variance',
    ]]

    n_multi = int((dup_audit['number_of_replicates'] > 1).sum())
    n_multi_rows = int(dup_audit.loc[dup_audit['number_of_replicates'] > 1, 'number_of_replicates'].sum())
    print(f"  Duplicate aggregation ({method}): {n_multi:,} multi-screen pairs "
          f"({n_multi_rows:,} rows) → {len(dup_audit):,} unique cell×drug")

    if audit_path:
        path = Path(audit_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        dup_audit.to_csv(path, index=False)
        print(f"  Duplicate audit saved: {path}")

    aggregated = dup_audit.rename(columns={
        'cell_line': 'cell_line_id',
        'drug': 'drug_name',
        'aggregated_value': 'ln_ic50',
    })[['cell_line_id', 'drug_name', 'ln_ic50']].copy()
    aggregated['raw_ic50'] = np.exp(aggregated['ln_ic50'])
    aggregated['n_replicates'] = dup_audit['number_of_replicates'].values
    aggregated['aggregation_method'] = dup_audit['aggregation_method'].values
    return aggregated


def load_ccle_response(filepath: str, audit_path: str = None,
                       ln_abs_max: float = CCLE_LN_IC50_ABS_MAX,
                       aggregate_duplicates: bool = True,
                       dup_method: str = CCLE_DUP_AGGREGATION_METHOD,
                       duplicate_audit_path: str = None) -> pd.DataFrame:
    """
    Load DepMap PRISM secondary screen and log-transform IC50.

    Scientific invalidation (audit trail; no arbitrary ±SD clipping):
      1. Missing / non-parseable IC50
      2. Non-finite source IC50 (+inf / -inf) — failed dose-response fits
      3. Non-positive IC50 (≤0) — cannot take ln; prior 1e-6 floor was an artifact
      4. |ln(IC50)| > ln_abs_max — assay-extrapolated / biologically implausible

    Infinite values are handled before any summary statistics.

    After Stage 1B correction, multi-screen duplicates are aggregated to one
    row per cell line × drug (default: median ln(IC50); Stage 1C).
    """
    df = pd.read_csv(filepath, low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    print(f"  CCLE response columns: {df.columns.tolist()[:20]}")

    drug_col = _find_column(['DRUG_NAME', 'DRUGNAME', 'NAME', 'COMPOUND', 'DRUG'], df.columns)
    ic50_col = _find_column(['LN_IC50', 'LNIC50', 'LOG2_IC50', 'IC50', 'LOG_IC50', 'AUC'], df.columns)
    cell_col = _find_column(['MODEL_ID', 'DEPMAP_ID', 'CCLE_NAME', 'CELL_LINE_DISPLAY_NAME',
                             'CELL_LINE_NAME', 'CELLLINENAME', 'MODELID'], df.columns)
    screen_col = _find_column(['SCREEN_ID', 'SCREENID'], df.columns)
    r2_col = _find_column(['R2', 'R_SQUARED', 'RSQ'], df.columns)

    if drug_col is None or ic50_col is None:
        raise ValueError(f"Could not find drug/IC50 columns. Have: {df.columns.tolist()[:15]}")

    print(f"  Mapped: cell={cell_col}, drug={drug_col}, ic50={ic50_col}")

    raw = pd.to_numeric(df[ic50_col], errors='coerce')
    result = pd.DataFrame({
        'cell_line_id': df[cell_col].astype(str).str.strip() if cell_col else 'unknown',
        'drug_name': df[drug_col].astype(str).str.strip(),
        'raw_ic50': raw,
    })
    if screen_col:
        result['screen_id'] = df[screen_col].astype(str)
    if r2_col:
        result['r2'] = pd.to_numeric(df[r2_col], errors='coerce')

    # Scale detection on FINITE values only (inf must not poison max/min)
    finite_raw = raw[np.isfinite(raw)]
    if len(finite_raw) == 0:
        raise ValueError("No finite IC50/AUC values in CCLE response file")

    use_auc = bool(finite_raw.max() <= 1.5 and finite_raw.min() >= 0)
    if use_auc:
        print(f"  Detected AUC scale (finite max={finite_raw.max():.3f})")
        scale = 'AUC'
    else:
        print(f"  Detected IC50 scale (finite max={finite_raw.max():.3e}, "
              f"n_inf={int((~np.isfinite(raw) & raw.notna()).sum())})")
        scale = 'IC50'

    n = len(result)
    status = np.full(n, 'Kept', dtype=object)
    reason = np.full(n, 'valid_positive_finite_ic50', dtype=object)
    original_ln = np.full(n, np.nan)
    corrected_ln = np.full(n, np.nan)

    missing = raw.isna()
    nonfinite = raw.notna() & ~np.isfinite(raw)
    nonpositive = np.isfinite(raw) & (raw <= 0)
    positive_finite = np.isfinite(raw) & (raw > 0)

    status[missing] = 'Excluded'
    reason[missing] = 'missing_ic50'

    status[nonfinite] = 'Excluded'
    reason[nonfinite] = 'nonfinite_source_ic50'

    status[nonpositive] = 'Excluded'
    reason[nonpositive] = 'nonpositive_ic50'

    # Valid transform on positive finite only (no 1e-6 floor)
    if positive_finite.any():
        if use_auc:
            ln_vals = -np.log(raw[positive_finite].values)
        else:
            ln_vals = np.log(raw[positive_finite].values)
        original_ln[positive_finite] = ln_vals

        implausible = np.abs(ln_vals) > ln_abs_max
        # Map implausible back to full index
        pos_idx = np.flatnonzero(positive_finite.values)
        impl_idx = pos_idx[implausible]
        keep_idx = pos_idx[~implausible]

        status[impl_idx] = 'Excluded'
        reason[impl_idx] = f'implausible_abs_ln_gt_{ln_abs_max:g}'

        corrected_ln[keep_idx] = original_ln[keep_idx]
        status[keep_idx] = 'Kept'
        reason[keep_idx] = f'valid_{scale.lower()}_ln_transform'

    result['ln_ic50'] = corrected_ln
    result['response_status'] = status
    result['response_reason'] = reason

    audit = pd.DataFrame({
        'cell_line': result['cell_line_id'].values,
        'drug': result['drug_name'].values,
        'screen_id': result['screen_id'].values if 'screen_id' in result.columns else '',
        'original_ic50': raw.values,
        'original_ln_ic50': original_ln,
        'corrected_ln_ic50': corrected_ln,
        'status': status,
        'reason': reason,
        'r2': result['r2'].values if 'r2' in result.columns else np.nan,
        'scale_detected': scale,
        'row_index': np.arange(n),
    })

    if audit_path:
        audit_path = Path(audit_path)
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        csv_path = audit_path.with_suffix('.csv') if audit_path.suffix != '.csv' else audit_path
        audit.to_csv(csv_path, index=False)
        print(f"  Response audit CSV: {csv_path}")
        if audit_path.suffix == '.parquet':
            audit.to_parquet(audit_path, index=False)
            print(f"  Response audit parquet: {audit_path}")
        elif audit_path.suffix not in ('.csv', ''):
            audit.to_csv(audit_path, index=False)

    n_kept = int((status == 'Kept').sum())
    n_excl = int((status == 'Excluded').sum())
    print(f"  Status: Kept={n_kept:,}  Excluded={n_excl:,}")
    print(f"  Exclude reasons: {pd.Series(reason[status == 'Excluded']).value_counts().to_dict()}")

    kept = result.dropna(subset=['ln_ic50']).copy()
    print(f"  Post-correction (pre-aggregation): {len(kept):,} rows, "
          f"ln_ic50 range: [{kept['ln_ic50'].min():.2f}, {kept['ln_ic50'].max():.2f}]")

    if aggregate_duplicates:
        if duplicate_audit_path is None and audit_path:
            duplicate_audit_path = str(Path(audit_path).parent / 'duplicate_response_audit.csv')
        elif duplicate_audit_path is None:
            duplicate_audit_path = str(Path(filepath).resolve().parent.parent / 'processed' / 'duplicate_response_audit.csv')
        kept = aggregate_ccle_response_duplicates(
            kept, method=dup_method, audit_path=duplicate_audit_path
        )
        print(f"  Final response (1 row / cell×drug): {len(kept):,} rows, "
              f"ln_ic50 range: [{kept['ln_ic50'].min():.2f}, {kept['ln_ic50'].max():.2f}]")
    else:
        print(f"  Final response (duplicates retained): {len(kept):,} rows")

    return kept


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
                       output_path=None, audit_path=None, duplicate_audit_path=None):
    print("\n--- Building CCLE ---")
    
    if duplicate_audit_path is None and output_path:
        duplicate_audit_path = str(Path(output_path).parent / 'duplicate_response_audit.csv')
    resp = load_ccle_response(
        response_file,
        audit_path=audit_path,
        duplicate_audit_path=duplicate_audit_path,
    )
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