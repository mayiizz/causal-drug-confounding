#!/usr/bin/env python3
"""
Phase 9: Generate all publication-ready figures and tables.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

PROCESSED = project_root / 'data' / 'processed'
OUTPUT = project_root / 'output'
OUTPUT.mkdir(exist_ok=True)

plt.style.use('seaborn-v0_8-whitegrid')


def figure1_cross_dataset_reproducibility():
    """Figure 1: Cross-dataset reproducibility scatter."""
    df = pd.read_parquet(PROCESSED / 'cross_dataset_reproducibility.parquet')
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    x = df['naive_dr_divergence_gdsc']
    y = df['naive_dr_divergence_ccle']
    
    ax.scatter(x, y, alpha=0.6, s=50, c='steelblue', edgecolors='white', linewidth=0.5)
    
    # Add diagonal
    lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]), 
            max(ax.get_xlim()[1], ax.get_ylim()[1])]
    ax.plot(lims, lims, 'k--', alpha=0.3, lw=1)
    
    ax.set_xlabel('|Naive − DR| (GDSC2)', fontsize=12)
    ax.set_ylabel('|Naive − DR| (CCLE)', fontsize=12)
    ax.set_title('Cross-Dataset Reproducibility of Causal Divergence', fontsize=13)
    
    # Add correlation text
    from scipy.stats import spearmanr
    rho, p = spearmanr(x, y)
    ax.text(0.05, 0.95, f'Spearman ρ = {rho:.3f}\np = {p:.3f}', 
            transform=ax.transAxes, fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    fig.savefig(OUTPUT / 'figure1_cross_dataset_reproducibility.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  Figure 1 saved.")


def figure2_pathway_validation():
    """Figure 2: Pathway validation A/B/C/D bar chart."""
    df = pd.read_parquet(PROCESSED / 'pathway_validation.parquet')
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    cases = df[['dataset', 'drug_class', 'pathway']].drop_duplicates()
    
    for idx, (_, case) in enumerate(cases.iterrows()):
        if idx >= 4:
            break
        
        sub = df[(df['dataset'] == case['dataset']) & 
                 (df['drug_class'] == case['drug_class']) &
                 (df['pathway'] == case['pathway'])]
        
        ax = axes[idx]
        colors = ['#1f77b4', '#2ca02c', '#d62728', '#ff7f0e']
        bars = ax.bar(sub['model'].str.replace('_', '\n'), sub['r2'], color=colors[:len(sub)])
        ax.set_ylabel('R²', fontsize=11)
        ax.set_title(f"{case['dataset']}\n{case['drug_class']} × {case['pathway'].replace('HALLMARK_', '')}", 
                     fontsize=10)
        ax.set_ylim(0, sub['r2'].max() * 1.2)
        
        # Highlight best
        best_idx = sub['r2'].idxmax()
        bars[sub.index.get_loc(best_idx)].set_edgecolor('black')
        bars[sub.index.get_loc(best_idx)].set_linewidth(2)
    
    plt.suptitle('Pathway Validation: Model A/B/C/D Comparison', fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig(OUTPUT / 'figure2_pathway_validation.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  Figure 2 saved.")


def figure3_cate_heatmap():
    """Figure 3: Drug Class × Tissue CATE heatmap."""
    cate = pd.read_parquet(PROCESSED / 'cate_estimates.parquet')
    
    # Pick one dataset and top drug classes
    gdsc_cate = cate[cate['dataset'] == 'GDSC2']
    if len(gdsc_cate) == 0:
        gdsc_cate = cate[cate['dataset'] == 'CCLE']
    
    # Pivot for heatmap
    pivot = gdsc_cate.pivot_table(
        index='drug_class', 
        columns='tissue_type', 
        values='cate_mean',
        aggfunc='mean'
    )
    
    # Select top tissues by variance
    tissue_var = pivot.var(axis=0).nlargest(15).index
    pivot = pivot[tissue_var]
    
    fig, ax = plt.subplots(figsize=(14, 8))
    im = ax.imshow(pivot.values, cmap='RdBu_r', aspect='auto', vmin=-2, vmax=2)
    
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha='right', fontsize=9)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=10)
    
    ax.set_xlabel('Tissue Type', fontsize=12)
    ax.set_ylabel('Drug Class', fontsize=12)
    ax.set_title('Conditional Average Treatment Effects by Tissue', fontsize=13)
    
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('CATE (log IC50)', fontsize=11)
    
    plt.tight_layout()
    fig.savefig(OUTPUT / 'figure3_cate_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  Figure 3 saved.")


def figure4_counterfactual_reranking():
    """Figure 4: Counterfactual reranking (simplified)."""
    estimates = pd.read_parquet(PROCESSED / 'causal_estimates.parquet')
    
    # Compare naive vs DR rankings
    dr = estimates[estimates['estimator'] == 'DR'][['dataset', 'drug_class', 'pathway', 'ate']]
    naive = estimates[estimates['estimator'] == 'Naive'][['dataset', 'drug_class', 'pathway', 'ate']]
    
    merged = dr.merge(naive, on=['dataset', 'drug_class', 'pathway'], suffixes=('_dr', '_naive'))
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    for idx, dataset in enumerate(['GDSC2', 'CCLE']):
        ax = axes[idx]
        sub = merged[merged['dataset'] == dataset]
        
        ax.scatter(sub['ate_naive'], sub['ate_dr'], alpha=0.6, c='steelblue', edgecolors='white')
        
        # Diagonal
        lims = [min(sub['ate_naive'].min(), sub['ate_dr'].min()) - 0.1,
                max(sub['ate_naive'].max(), sub['ate_dr'].max()) + 0.1]
        ax.plot(lims, lims, 'k--', alpha=0.3, lw=1)
        
        ax.set_xlabel('Naive ATE', fontsize=11)
        ax.set_ylabel('DR ATE', fontsize=11)
        ax.set_title(f'{dataset}: Naive vs Doubly Robust', fontsize=12)
        
        # Count rerankings (sign flips)
        flips = ((sub['ate_naive'] > 0) & (sub['ate_dr'] < 0)) | ((sub['ate_naive'] < 0) & (sub['ate_dr'] > 0))
        ax.text(0.05, 0.95, f'Sign flips: {flips.sum()}/{len(sub)}', 
                transform=ax.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('Counterfactual Reranking: Naive vs Causal Estimates', fontsize=14)
    plt.tight_layout()
    fig.savefig(OUTPUT / 'figure4_counterfactual_reranking.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  Figure 4 saved.")


def figure5_common_support():
    """Figure 5: Common support before/after trimming."""
    cs = pd.read_parquet(PROCESSED / 'common_support_default.parquet')
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    for idx, dataset in enumerate(['GDSC2', 'CCLE']):
        ax = axes[idx]
        sub = cs[cs['dataset'] == dataset]
        
        x = np.arange(len(sub))
        width = 0.35
        
        ax.bar(x - width/2, sub['n_original'], width, label='Original', alpha=0.8, color='steelblue')
        ax.bar(x + width/2, sub['n_trimmed'], width, label='After trimming', alpha=0.8, color='coral')
        
        ax.set_xlabel('Cohort index', fontsize=11)
        ax.set_ylabel('Sample size', fontsize=11)
        ax.set_title(f'{dataset}: Common Support', fontsize=12)
        ax.legend()
    
    plt.suptitle('Common Support Before and After Trimming', fontsize=14)
    plt.tight_layout()
    fig.savefig(OUTPUT / 'figure5_common_support.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  Figure 5 saved.")


def figure6_permutation_null():
    """Figure 6: Permutation null distribution."""
    perm = pd.read_parquet(PROCESSED / 'permutation_results.parquet')
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    plotted = 0
    for idx, row in perm.head(5).iterrows():
        if row['null_std'] < 1e-6:
            # Degenerate case: plot as a vertical spike
            ax.axvline(row['observed_ate'], color='red', linestyle='--', alpha=0.7, 
                      label=f"{row['drug_class'][:15]}... (null σ≈0)")
            plotted += 1
            continue
            
        null = np.random.normal(row['null_mean'], row['null_std'], 1000)
        ax.hist(null, bins=30, alpha=0.4, 
                label=f"{row['drug_class'][:15]}...")
        ax.axvline(row['observed_ate'], color='red', linestyle='--', alpha=0.7)
        plotted += 1
    
    ax.set_xlabel('ATE under tissue-shuffled null', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Permutation Null Distribution', fontsize=13)
    
    if plotted > 0:
        ax.legend(fontsize=9)
    
    plt.tight_layout()
    fig.savefig(OUTPUT / 'figure6_permutation_null.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  Figure 6 saved.")


def generate_tables():
    """Generate all tables as CSV."""
    print("\n  Generating tables...")
    
    # Table 1: Naive/IPW/DR estimates
    est = pd.read_parquet(PROCESSED / 'causal_estimates.parquet')
    est.to_csv(OUTPUT / 'table1_causal_estimates.csv', index=False)
    print("    Table 1: causal_estimates.csv")
    
    # Table 2: Refutation tests
    ref = pd.read_parquet(PROCESSED / 'refutation_results.parquet')
    ref.to_csv(OUTPUT / 'table2_refutation_tests.csv', index=False)
    print("    Table 2: refutation_tests.csv")
    
    # Table 3: Sensitivity analysis
    sens = pd.read_parquet(PROCESSED / 'sensitivity_results.parquet')
    sens.to_csv(OUTPUT / 'table3_sensitivity_analysis.csv', index=False)
    print("    Table 3: sensitivity_analysis.csv")
    
    # Table 4: Common support report
    cs = pd.read_parquet(PROCESSED / 'common_support_report.parquet')
    cs.to_csv(OUTPUT / 'table4_common_support.csv', index=False)
    print("    Table 4: common_support.csv")
    
    # Table 5: Power analysis
    power = pd.read_parquet(PROCESSED / 'power_analysis.parquet')
    power.to_csv(OUTPUT / 'table5_power_analysis.csv', index=False)
    print("    Table 5: power_analysis.csv")


def main():
    print("=" * 60)
    print("PHASE 9: Visualization & Paper Artifacts")
    print("=" * 60)
    
    print("\nGenerating figures...")
    figure1_cross_dataset_reproducibility()
    figure2_pathway_validation()
    figure3_cate_heatmap()
    figure4_counterfactual_reranking()
    figure5_common_support()
    figure6_permutation_null()
    
    generate_tables()
    
    print("\n" + "=" * 60)
    print("PHASE 9 COMPLETE")
    print("=" * 60)
    print(f"\nAll artifacts saved to: {OUTPUT}/")
    print("\nFigures:")
    for i in range(1, 7):
        print(f"  figure{i}_*.png")
    print("\nTables:")
    for i in range(1, 6):
        print(f"  table{i}_*.csv")
    
    print("\n" + "=" * 60)
    print("ALL PHASES COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    main()