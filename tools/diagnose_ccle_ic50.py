#!/usr/bin/env python3
"""
Diagnostic: Auto-discover and inspect raw CCLE drug response data.
"""
import pandas as pd
import numpy as np
from pathlib import Path

RAW_DIR = Path("data/raw")

# Auto-discover CCLE response file
candidates = [
    "secondary-screen-dose-response-curve-parameters.csv",
    "secondary-screen-dose-response-curve-parameters.csv.gz",
    "CCLE_NP24.2009_Drug_data_2015.02.24.csv",
    "ccle_response.csv",
    "drug_response.csv",
]

# Also search for any file containing these keywords
for f in RAW_DIR.iterdir():
    name = f.name.lower()
    if any(k in name for k in ["secondary", "dose-response", "curve", "ccle_drug", "depmap_drug"]):
        if f.name not in candidates:
            candidates.insert(0, f.name)

ccle_path = None
for c in candidates:
    p = RAW_DIR / c
    if p.exists():
        ccle_path = p
        break

if ccle_path is None:
    print("ERROR: Could not find CCLE drug response file in data/raw/")
    print("Files found in data/raw/:")
    for f in RAW_DIR.iterdir():
        print(f"  {f.name}")
    raise SystemExit(1)

print("=" * 70)
print("CCLE IC50 DIAGNOSTIC")
print(f"Found file: {ccle_path}")
print("=" * 70)

# Load with appropriate kwargs
load_kwargs = {}
if ccle_path.suffix == ".csv":
    load_kwargs = {"low_memory": False}
elif ccle_path.suffix == ".gz":
    load_kwargs = {"compression": "gzip", "low_memory": False}

df = pd.read_csv(ccle_path, **load_kwargs)
print(f"\nRaw shape: {df.shape}")
print(f"Columns: {list(df.columns)}")

# Identify the IC50 column
ic50_candidates = [c for c in df.columns if "ic50" in c.lower()]
if not ic50_candidates:
    raise ValueError(f"No IC50-like column found. Columns: {list(df.columns)}")

ic50_col = ic50_candidates[0]
print(f"\nUsing IC50 column: {ic50_col}")

s = df[ic50_col]
print(f"\n--- Raw IC50 descriptive stats ---")
print(s.describe())

print(f"\n--- Finite check ---")
print(f"  Total rows: {len(s)}")
print(f"  Finite values: {np.isfinite(s).sum()}")
print(f"  +Inf: {np.isposinf(s).sum()}")
print(f"  -Inf: {np.isneginf(s).sum()}")
print(f"  NaN: {s.isna().sum()}")
print(f"  Zero: {(s == 0).sum()}")
print(f"  Negative: {(s < 0).sum()}")

finite_vals = s[np.isfinite(s)]
print(f"\n--- Finite value distribution ---")
print(f"  Min: {finite_vals.min()}")
print(f"  Max: {finite_vals.max()}")
print(f"  Median: {finite_vals.median()}")
print(f"  Mean: {finite_vals.mean()}")

# Heuristic: is this already log-transformed?
print(f"\n--- Scale heuristic ---")
print(f"  Values between -20 and +20: {((finite_vals > -20) & (finite_vals < 20)).mean():.2%}")
print(f"  Values > 100: {(finite_vals > 100).mean():.2%}")
print(f"  Values < 1: {(finite_vals < 1).mean():.2%}")

# Show extreme values
print(f"\n--- Top 10 largest IC50 values ---")
print(df.nlargest(10, ic50_col)[[ic50_col, 'name', 'ccle_name', 'depmap_id']].to_string())

print(f"\n--- Rows with inf ---")
inf_rows = df[np.isinf(df[ic50_col])]
print(f"Count: {len(inf_rows)}")
if len(inf_rows) > 0:
    print(inf_rows[[ic50_col, 'name', 'ccle_name', 'depmap_id']].head(10).to_string())

# Check per-drug summary
print(f"\n--- Per-drug IC50 summary (top 10 by mean) ---")
drug_summary = df.groupby('name')[ic50_col].agg(['count', 'mean', 'median', 'max', 'min']).sort_values('mean', ascending=False)
print(drug_summary.head(10).to_string())

print("\n" + "=" * 70)
print("DIAGNOSTIC COMPLETE")
print("=" * 70)