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
    """Figure 4: Individual Y(0)/Y(1) counterfactuals and within-cohort reranking."""
    pred_path = PROCESSED / 'counterfactual_predictions.parquet'
    if not pred_path.exists():
        raise FileNotFoundError(
            f"Missing {pred_path}. Run Phase 6B "
            f"(python scripts/06b_counterfactual.py) before Phase 9."
        )

    raw = pd.read_parquet(pred_path)
    if raw.empty:
        raise ValueError("counterfactual_predictions.parquet is empty.")

    # Exclude invalid observations from plots only (kept in parquet for audit).
    # Invalid = non-finite values, extreme observed ln_IC50 (bad transforms),
    # or extreme DR tau. Valid biology is not removed.
    if 'is_valid' in raw.columns:
        n_invalid = int((~raw['is_valid']).sum())
        df = raw[raw['is_valid']].copy()
        print(f"    Figure 4: plotting {len(df):,} valid rows "
              f"(excluded {n_invalid:,} invalid)")
        if n_invalid and 'invalid_reason' in raw.columns:
            print("    Invalid reasons:")
            print(raw.loc[~raw['is_valid'], 'invalid_reason']
                  .value_counts().to_string().replace('\n', '\n      '))
    else:
        # Backward compatibility with older column names
        df = raw.copy()
        if 'observed_ln_ic50' not in df.columns and 'y_obs' in df.columns:
            df = df.rename(columns={
                'y_obs': 'observed_ln_ic50',
                'y_cf': 'counterfactual_high_ln_ic50',
                'delta': 'delta_high',
            })

    if df.empty:
        raise ValueError("No valid counterfactual rows available for Figure 4.")

    # Reranking definition: WITHIN COHORT
    # (dataset × drug_class × pathway). Not within drug class alone or dataset alone.
    # Rank by observed_ln_ic50 vs counterfactual_high_ln_ic50 (do(T=1)).
    df['rank_obs'] = df.groupby(
        ['dataset', 'drug_class', 'pathway']
    )['observed_ln_ic50'].rank(method='average')
    df['rank_cf_high'] = df.groupby(
        ['dataset', 'drug_class', 'pathway']
    )['counterfactual_high_ln_ic50'].rank(method='average')
    df['rank_change'] = df['rank_cf_high'] - df['rank_obs']

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Panel A: observed vs counterfactual_high scatter
    ax = axes[0, 0]
    sample = df.sample(n=min(5000, len(df)), random_state=42)
    ax.scatter(
        sample['observed_ln_ic50'],
        sample['counterfactual_high_ln_ic50'],
        alpha=0.25, s=12, c='steelblue', edgecolors='none'
    )
    lims = [
        min(sample['observed_ln_ic50'].min(), sample['counterfactual_high_ln_ic50'].min()),
        max(sample['observed_ln_ic50'].max(), sample['counterfactual_high_ln_ic50'].max()),
    ]
    ax.plot(lims, lims, 'k--', alpha=0.4, lw=1)
    ax.set_xlabel('Observed ln(IC50)', fontsize=11)
    ax.set_ylabel('Y(1): counterfactual high ln(IC50)', fontsize=11)
    ax.set_title('A. Observed vs do(High Pathway)', fontsize=12)

    # Panel B: delta_high distribution (all valid units)
    ax = axes[0, 1]
    deltas = df['delta_high']
    ax.hist(deltas, bins=50, alpha=0.75, color='coral', edgecolor='white',
            label=f'All valid (n={len(deltas):,})')
    ax.axvline(0, color='black', linestyle='--', lw=1, alpha=0.5)
    ax.axvline(deltas.mean(), color='steelblue', linestyle='-', lw=1.5,
               label=f'Mean={deltas.mean():.3f}')
    ax.set_xlabel('delta_high = Y(1) - Y_obs', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title('B. Delta to High Pathway', fontsize=12)
    ax.legend(fontsize=9)

    # Panel C: tissue-wise delta_high boxplots (top tissues by n)
    ax = axes[1, 0]
    tissue_n = df.groupby('tissue_type').size().sort_values(ascending=False)
    top_tissues = tissue_n.head(8).index.tolist()
    box_data = [df.loc[df['tissue_type'] == t, 'delta_high'].values for t in top_tissues]
    keep = [(t, d) for t, d in zip(top_tissues, box_data) if len(d) > 0]
    if keep:
        labels, data = zip(*keep)
        bp = ax.boxplot(data, tick_labels=[str(l)[:18] for l in labels],
                        patch_artist=True, showfliers=False)
        for patch in bp['boxes']:
            patch.set_facecolor('lightsteelblue')
        ax.axhline(0, color='black', linestyle='--', lw=0.8, alpha=0.5)
        ax.tick_params(axis='x', rotation=45, labelsize=8)
    ax.set_ylabel('delta_high ln(IC50)', fontsize=11)
    ax.set_title('C. Tissue-wise delta_high', fontsize=12)

    # Panel D: within-cohort reranking histogram
    ax = axes[1, 1]
    rc = df['rank_change']
    ax.hist(rc, bins=40, alpha=0.8, color='seagreen', edgecolor='white')
    ax.axvline(0, color='black', linestyle='--', lw=1, alpha=0.5)
    ax.set_xlabel('Within-cohort rank change (cf_high - obs)', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title(
        'D. Reranking within cohort\n(dataset x drug_class x pathway)',
        fontsize=11
    )
    moved = (rc.abs() > 0).mean() if len(rc) else 0
    ax.text(
        0.05, 0.95,
        f'Any rank move: {moved:.1%}\nMean |dRank|: {rc.abs().mean():.2f}',
        transform=ax.transAxes, fontsize=10, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    )

    plt.suptitle(
        'Counterfactual Analysis: Y(0) and Y(1) under pathway treatment',
        fontsize=14
    )
    plt.tight_layout()
    fig.savefig(OUTPUT / 'figure4_counterfactual_reranking.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  Figure 4 saved.")


def figure5_common_support():
    """
    Figure 5: Common support sample sizes + IPW tissue balance (Love-style).
    """
    cs_path = PROCESSED / 'common_support_default.parquet'
    bal_sum_path = PROCESSED / 'balance_summary.parquet'
    bal_det_path = PROCESSED / 'balance_diagnostics.parquet'

    if not cs_path.exists():
        raise FileNotFoundError(f"Missing {cs_path}. Run Phase 3 first.")

    cs = pd.read_parquet(cs_path)
    has_balance = bal_sum_path.exists() and bal_det_path.exists()

    if has_balance:
        summary = pd.read_parquet(bal_sum_path)
        detail = pd.read_parquet(bal_det_path)
        fig, axes = plt.subplots(2, 2, figsize=(14, 11))

        # Panel A/B: sample size before/after trim (existing)
        for idx, dataset in enumerate(['GDSC2', 'CCLE']):
            ax = axes[0, idx]
            sub = cs[cs['dataset'] == dataset]
            x = np.arange(len(sub))
            width = 0.35
            ax.bar(x - width / 2, sub['n_original'], width, label='Original',
                   alpha=0.8, color='steelblue')
            ax.bar(x + width / 2, sub['n_trimmed'], width, label='After trimming',
                   alpha=0.8, color='coral')
            ax.set_xlabel('Cohort index', fontsize=11)
            ax.set_ylabel('Sample size', fontsize=11)
            ax.set_title(f'{dataset}: Common Support', fontsize=12)
            ax.legend(fontsize=9)

        # Panel C: max |SMD| across cohorts at three stages
        ax = axes[1, 0]
        stages = ['max_raw_smd', 'max_trimmed_smd', 'max_weighted_smd']
        labels = ['Raw', 'Trimmed', 'IPW-weighted']
        data = [summary[s].dropna().values for s in stages]
        bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, showfliers=False)
        colors = ['#a6cee3', '#fdbf6f', '#b2df8a']
        for patch, c in zip(bp['boxes'], colors):
            patch.set_facecolor(c)
        ax.axhline(0.1, color='crimson', linestyle='--', lw=1, label='|SMD|=0.1')
        ax.axhline(0.05, color='gray', linestyle=':', lw=1, label='|SMD|=0.05')
        ax.set_ylabel('Max |SMD| across tissues', fontsize=11)
        ax.set_title('C. Cohort max |SMD| by stage', fontsize=12)
        ax.legend(fontsize=8)

        # Panel D: Love plot — mean |SMD| by tissue (top tissues by n)
        ax = axes[1, 1]
        tissue_n = detail.groupby('tissue').size().sort_values(ascending=False)
        top = tissue_n.head(12).index.tolist()
        love = (
            detail[detail['tissue'].isin(top)]
            .groupby('tissue', as_index=False)
            .agg(
                raw=('raw_smd', 'mean'),
                trimmed=('trimmed_smd', 'mean'),
                weighted=('weighted_smd', 'mean'),
            )
        )
        # Sort by raw imbalance
        love = love.sort_values('raw', ascending=True)
        y = np.arange(len(love))
        ax.scatter(love['raw'], y, marker='o', s=40, label='Raw', color='#1f78b4')
        ax.scatter(love['trimmed'], y, marker='s', s=40, label='Trimmed', color='#ff7f00')
        ax.scatter(love['weighted'], y, marker='D', s=40, label='IPW-weighted', color='#33a02c')
        ax.axvline(0.1, color='crimson', linestyle='--', lw=1, alpha=0.8)
        ax.axvline(0.05, color='gray', linestyle=':', lw=1, alpha=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels([str(t)[:22] for t in love['tissue']], fontsize=8)
        ax.set_xlabel('Mean |SMD| across cohorts', fontsize=11)
        ax.set_title('D. Love plot (tissue indicators)', fontsize=12)
        ax.legend(fontsize=8, loc='lower right')
        ax.set_xlim(left=0)

        plt.suptitle(
            'Figure 5: Common Support and IPW Covariate Balance',
            fontsize=14
        )
    else:
        print("    WARNING: balance_*.parquet missing; "
              "run Phase 3B for full Figure 5. Showing support only.")
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        for idx, dataset in enumerate(['GDSC2', 'CCLE']):
            ax = axes[idx]
            sub = cs[cs['dataset'] == dataset]
            x = np.arange(len(sub))
            width = 0.35
            ax.bar(x - width / 2, sub['n_original'], width, label='Original',
                   alpha=0.8, color='steelblue')
            ax.bar(x + width / 2, sub['n_trimmed'], width, label='After trimming',
                   alpha=0.8, color='coral')
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
    """Figure 6: Empirical DR null from tissue permutations (not a normal draw)."""
    results_path = PROCESSED / 'permutation_results.parquet'
    null_path = PROCESSED / 'permutation_null_distribution.parquet'

    if not results_path.exists():
        raise FileNotFoundError(
            f"Missing {results_path}. Run Phase 5 before Phase 9."
        )

    perm = pd.read_parquet(results_path)
    # Prefer observed_dr_ate; fall back to legacy observed_ate
    obs_col = 'observed_dr_ate' if 'observed_dr_ate' in perm.columns else 'observed_ate'
    p_col = 'empirical_p' if 'empirical_p' in perm.columns else 'permutation_p_value'

    has_empirical_null = null_path.exists()
    if has_empirical_null:
        null_df = pd.read_parquet(null_path)
    else:
        null_df = pd.DataFrame()
        print("    WARNING: permutation_null_distribution.parquet missing; "
              "falling back to summary-only plot.")

    # Plot up to 4 cohorts with the strongest |observed DR|
    plot_rows = perm.copy()
    plot_rows['_abs_obs'] = plot_rows[obs_col].abs()
    plot_rows = plot_rows.nlargest(min(4, len(plot_rows)), '_abs_obs')

    n_panels = len(plot_rows)
    if n_panels == 0:
        raise ValueError("permutation_results.parquet has no rows to plot.")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for ax in axes[n_panels:]:
        ax.axis('off')

    for ax, (_, row) in zip(axes, plot_rows.iterrows()):
        dataset = row['dataset']
        drug_class = row['drug_class']
        pathway = row['pathway']
        obs = row[obs_col]
        p_val = row[p_col] if p_col in row.index and pd.notna(row[p_col]) else np.nan
        label = f"{dataset} | {drug_class[:18]}\n{pathway.replace('HALLMARK_', '')}"

        cohort_null = pd.DataFrame()
        if has_empirical_null and not null_df.empty:
            cohort_null = null_df[
                (null_df['dataset'] == dataset) &
                (null_df['drug_class'] == drug_class) &
                (null_df['pathway'] == pathway)
            ]

        if not cohort_null.empty:
            vals = cohort_null['dr_ate'].values
            ax.hist(vals, bins=30, alpha=0.75, color='steelblue',
                    edgecolor='white', density=False,
                    label=f'Null (n={len(vals)})')
        elif pd.notna(row.get('null_mean')) and pd.notna(row.get('null_sd')):
            # Last-resort visual only if null draws missing
            fake = np.random.normal(row['null_mean'], max(row['null_sd'], 1e-6), 500)
            ax.hist(fake, bins=30, alpha=0.4, color='gray',
                    label='Approx null (no draws)')

        ax.axvline(obs, color='crimson', linestyle='--', lw=2,
                   label=f'Observed DR={obs:.3f}')
        ax.axvline(0, color='black', linestyle=':', lw=1, alpha=0.5)

        p_txt = f'p={p_val:.3f}' if pd.notna(p_val) else 'p=NA'
        z_txt = f"z={row['z_score']:.2f}" if 'z_score' in row.index and pd.notna(row.get('z_score')) else ''
        ax.set_title(f'{label}\n{p_txt}  {z_txt}', fontsize=10)
        ax.set_xlabel('DR ATE under tissue-permuted null', fontsize=10)
        ax.set_ylabel('Count', fontsize=10)
        ax.legend(fontsize=8, loc='best')

    plt.suptitle(
        'Figure 6: Empirical DR Permutation Null\n'
        '(tissue shuffled; T and Y fixed; LinearDRLearner refit)',
        fontsize=13
    )
    plt.tight_layout()
    fig.savefig(OUTPUT / 'figure6_permutation_null.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  Figure 6 saved.")


def figure7_power_adequacy():
    """Figure 7: Cohort power / adequacy labels."""
    path = PROCESSED / 'power_analysis.parquet'
    if not path.exists():
        print("    WARNING: power_analysis.parquet missing; skip Figure 7.")
        return
    df = pd.read_parquet(path)
    if 'adequacy_label' not in df.columns:
        print("    WARNING: adequacy_label missing; re-run Phase 8.")
        return

    order = ['Excellent', 'Good', 'Adequate', 'Marginal', 'Underpowered']
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = ['#1a9850', '#91cf60', '#fee08b', '#fc8d59', '#d73027']

    for ax, dataset in zip(axes, ['GDSC2', 'CCLE']):
        sub = df[df['dataset'] == dataset]
        counts = sub['adequacy_label'].value_counts().reindex(order).fillna(0)
        ax.bar(order, counts.values, color=colors, edgecolor='white')
        ax.set_title(f'{dataset}: Cohort Adequacy', fontsize=12)
        ax.set_ylabel('Number of cohorts', fontsize=11)
        ax.tick_params(axis='x', rotation=30)
        for i, v in enumerate(counts.values):
            ax.text(i, v + 0.5, str(int(v)), ha='center', fontsize=9)

    plt.suptitle('Figure 7: Power / Stability Adequacy Labels', fontsize=14)
    plt.tight_layout()
    fig.savefig(OUTPUT / 'figure7_power_adequacy.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  Figure 7 saved.")


def figure8_baseline_svr():
    """Figure 8: SVR predicted vs observed ln(IC50)."""
    pred_path = PROCESSED / 'baseline_svr_predictions.parquet'
    res_path = PROCESSED / 'baseline_svr_results.parquet'
    if not pred_path.exists():
        print("    WARNING: SVR baseline missing; run Phase 10. Skip Figure 8.")
        return
    preds = pd.read_parquet(pred_path)
    results = pd.read_parquet(res_path) if res_path.exists() else pd.DataFrame()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, dataset in zip(axes, ['GDSC2', 'CCLE']):
        sub = preds[preds['dataset'] == dataset]
        if sub.empty:
            ax.set_visible(False)
            continue
        sample = sub.sample(n=min(4000, len(sub)), random_state=42)
        ax.scatter(sample['y_true'], sample['y_pred'], alpha=0.2, s=8, c='steelblue')
        lims = [
            min(sample['y_true'].min(), sample['y_pred'].min()),
            max(sample['y_true'].max(), sample['y_pred'].max()),
        ]
        ax.plot(lims, lims, 'k--', alpha=0.4)
        ax.set_xlabel('Observed ln(IC50)', fontsize=11)
        ax.set_ylabel('Predicted ln(IC50)', fontsize=11)
        ax.set_title(f'SVR - {dataset}', fontsize=12)
        if not results.empty and dataset in set(results['dataset']):
            r = results[results['dataset'] == dataset].iloc[0]
            ax.text(
                0.05, 0.95,
                f"RMSE={r['rmse']:.3f}\nR2={r['r2']:.3f}\nPearson={r['pearson_r']:.3f}",
                transform=ax.transAxes, va='top', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            )
    plt.suptitle('Figure 8: SVR Predictive Baseline', fontsize=14)
    plt.tight_layout()
    fig.savefig(OUTPUT / 'figure8_baseline_svr.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  Figure 8 saved.")


def figure9_baseline_xgboost():
    """Figure 9: XGBoost predicted vs observed ln(IC50)."""
    pred_path = PROCESSED / 'baseline_xgboost_predictions.parquet'
    res_path = PROCESSED / 'baseline_xgboost_results.parquet'
    if not pred_path.exists():
        print("    WARNING: XGBoost baseline missing; run Phase 10. Skip Figure 9.")
        return
    preds = pd.read_parquet(pred_path)
    results = pd.read_parquet(res_path) if res_path.exists() else pd.DataFrame()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, dataset in zip(axes, ['GDSC2', 'CCLE']):
        sub = preds[preds['dataset'] == dataset]
        if sub.empty:
            ax.set_visible(False)
            continue
        sample = sub.sample(n=min(4000, len(sub)), random_state=42)
        ax.scatter(sample['y_true'], sample['y_pred'], alpha=0.2, s=8, c='darkorange')
        lims = [
            min(sample['y_true'].min(), sample['y_pred'].min()),
            max(sample['y_true'].max(), sample['y_pred'].max()),
        ]
        ax.plot(lims, lims, 'k--', alpha=0.4)
        ax.set_xlabel('Observed ln(IC50)', fontsize=11)
        ax.set_ylabel('Predicted ln(IC50)', fontsize=11)
        ax.set_title(f'XGBoost - {dataset}', fontsize=12)
        if not results.empty and dataset in set(results['dataset']):
            r = results[results['dataset'] == dataset].iloc[0]
            ax.text(
                0.05, 0.95,
                f"RMSE={r['rmse']:.3f}\nR2={r['r2']:.3f}\nPearson={r['pearson_r']:.3f}",
                transform=ax.transAxes, va='top', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            )
    plt.suptitle('Figure 9: XGBoost Predictive Baseline', fontsize=14)
    plt.tight_layout()
    fig.savefig(OUTPUT / 'figure9_baseline_xgboost.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  Figure 9 saved.")


def generate_tables():
    """Generate all tables as CSV."""
    print("\n  Generating tables...")

    est = pd.read_parquet(PROCESSED / 'causal_estimates.parquet')
    est.to_csv(OUTPUT / 'table1_causal_estimates.csv', index=False)
    print("    Table 1: causal_estimates.csv")

    ref = pd.read_parquet(PROCESSED / 'refutation_results.parquet')
    ref.to_csv(OUTPUT / 'table2_refutation_tests.csv', index=False)
    print("    Table 2: refutation_tests.csv")

    sens = pd.read_parquet(PROCESSED / 'sensitivity_results.parquet')
    sens.to_csv(OUTPUT / 'table3_sensitivity_analysis.csv', index=False)
    print("    Table 3: sensitivity_analysis.csv")

    cs = pd.read_parquet(PROCESSED / 'common_support_report.parquet')
    cs.to_csv(OUTPUT / 'table4_common_support.csv', index=False)
    print("    Table 4: common_support.csv")

    baseline_parts = []
    for name in ['baseline_svr_results.parquet', 'baseline_xgboost_results.parquet']:
        path = PROCESSED / name
        if path.exists():
            baseline_parts.append(pd.read_parquet(path))
    if baseline_parts:
        base = pd.concat(baseline_parts, ignore_index=True)
        base.to_csv(OUTPUT / 'table5_baseline_metrics.csv', index=False)
        print("    Table 5: baseline_metrics.csv (SVR/XGBoost)")
    else:
        print("    Table 5: baselines missing (run Phase 10)")

    power = pd.read_parquet(PROCESSED / 'power_analysis.parquet')
    power.to_csv(OUTPUT / 'table5_power_analysis.csv', index=False)
    print("    Also: table5_power_analysis.csv")


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
    figure7_power_adequacy()
    figure8_baseline_svr()
    figure9_baseline_xgboost()

    generate_tables()

    print("\n" + "=" * 60)
    print("PHASE 9 COMPLETE")
    print("=" * 60)
    print(f"\nAll artifacts saved to: {OUTPUT}/")
    print("\nFigures: figure1-figure9")
    print("Tables: table1-table5 (+ power supplemental)")
    print("\n" + "=" * 60)
    print("ALL PHASES COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    main()
