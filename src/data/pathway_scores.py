"""Pathway scoring from RNA-seq expression."""

import pandas as pd
import numpy as np
from pathlib import Path
import gseapy as gp

HALLMARK_PATHWAYS = [
    'HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION',
    'HALLMARK_PI3K_AKT_MTOR_SIGNALING',
    'HALLMARK_KRAS_SIGNALING_UP',
    'HALLMARK_DNA_REPAIR',
    'HALLMARK_APOPTOSIS',
]


def load_hallmark_gmt(gmt_path: str) -> dict:
    gmt_dict = {}
    with open(gmt_path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            gmt_dict[parts[0]] = parts[2:]
    return gmt_dict


def compute_ssgsea_scores(expression_df: pd.DataFrame, gmt_dict: dict, pathways: list = None):
    """
    expression_df: genes (index) x samples (columns)
    gseapy expects this exact orientation.
    """
    if pathways is None:
        pathways = HALLMARK_PATHWAYS
    
    available = {k: v for k, v in gmt_dict.items() if k in pathways}
    if not available:
        raise ValueError(f"No requested pathways found. Available: {list(gmt_dict.keys())[:5]}")
    
    outdir = './tmp_gsea'
    Path(outdir).mkdir(exist_ok=True)
    temp_gmt = Path(outdir) / 'selected_pathways.gmt'
    with open(temp_gmt, 'w') as f:
        for pw, genes in available.items():
            f.write(f"{pw}\tna\t" + "\t".join(genes) + "\n")
    
    # expression_df is already genes (index) x samples (columns) — correct for gseapy
    # Just ensure column names are strings
    expression_df = expression_df.copy()
    expression_df.columns = expression_df.columns.astype(str)
    expression_df.index = expression_df.index.astype(str)
    
    result = gp.ssgsea(
        data=expression_df,
        gene_sets=str(temp_gmt),
        outdir=outdir,
        sample_norm_method='rank',
        verbose=False,
        no_plot=True,
        processes=4
    )
    
    scores = result.res2d.pivot(index='Name', columns='Term', values='NES')
    scores.index.name = 'cell_line_id'
    return scores


def compute_zscore_pathway_scores(expression_df: pd.DataFrame, gmt_dict: dict, pathways: list = None):
    """Fallback fast z-score method."""
    if pathways is None:
        pathways = HALLMARK_PATHWAYS
    
    expr_t = expression_df.T  # samples x genes for z-scoring
    expr_z = pd.DataFrame(
        data=(expr_t.values - expr_t.mean(axis=0).values) / (expr_t.std(axis=0).values + 1e-8),
        index=expr_t.index,
        columns=expr_t.columns
    )
    
    scores = {}
    for pw in pathways:
        if pw not in gmt_dict:
            continue
        genes = [g for g in gmt_dict[pw] if g in expr_z.columns]
        if len(genes) == 0:
            continue
        scores[pw] = expr_z[genes].mean(axis=1)
    
    scores_df = pd.DataFrame(scores)
    scores_df.index.name = 'cell_line_id'
    return scores_df