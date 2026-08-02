# DATA.md — Raw Data Acquisition Guide

This document explains **exactly** how to obtain and place the raw inputs required by Phase 01 (`scripts/01_build_unified_data.py`).

All raw files must live under:

```text
data/raw/
```

Filenames must match **exactly** (case-sensitive on Linux/macOS).

The `data/` directory is gitignored. After cloning, create the folders and download files yourself:

```bash
mkdir -p data/raw data/processed
```

---

## Required files (checklist)

| # | Expected filename | Role | Required by |
|---|-------------------|------|-------------|
| 1 | `GDSC2_fitted_dose_response_27Oct23.xlsx` | GDSC2 `LN_IC50` drug response | Phase 01 (GDSC2) |
| 2 | `rnaseq_merged_20260323.zip` | GDSC2 / Sanger RNA-seq (TPM) | Phase 01 (GDSC2) |
| 3 | `OmicsExpressionProteinCodingGenesTPMLogp1.csv` | CCLE / DepMap expression | Phase 01 (CCLE) |
| 4 | `secondary-screen-dose-response-curve-parameters.csv` | PRISM secondary-screen response | Phase 01 (CCLE) |
| 5 | `sample_info.csv` | CCLE cell-line annotations | Phase 01 (CCLE) |
| 6 | `model_list_20260724.csv` | Cell Model Passports model metadata | Phase 01 (both) |
| 7 | `h.all.v2026.1.Hs.symbols.gmt` | MSigDB Hallmark gene sets | Phase 01 (pathways) |

Phase 01 hard checks list items 1-4 and 6-7. Item **5** (`sample_info.csv`) is still required for `build_ccle_unified` and must be present.

---

## 1. GDSC2 fitted dose-response

- **Expected filename:** `GDSC2_fitted_dose_response_27Oct23.xlsx`
- **Expected location:** `data/raw/GDSC2_fitted_dose_response_27Oct23.xlsx`
- **Version used in this project:** release dated **27 Oct 2023** (encoded in filename)
- **Official sources:**
  - https://www.cancerrxgene.org/downloads/bulk_download
  - https://cellmodelpassports.sanger.ac.uk/
  - https://ftp.sanger.ac.uk/pub/project/cancerrxgene/releases/
- **What to download:** GDSC2 fitted dose-response / IC50 spreadsheet matching the Oct 2023 release naming.
- **Notes:** Must contain `LN_IC50`, cell-line / Sanger model IDs, and drug identifiers used by the loaders.

---

## 2. GDSC2 / Cell Model Passports RNA-seq

- **Expected filename:** `rnaseq_merged_20260323.zip`
- **Expected location:** `data/raw/rnaseq_merged_20260323.zip`
- **Version used:** merge dated **2026-03-23** (encoded in filename)
- **Official source:** https://cellmodelpassports.sanger.ac.uk/
- **Notes:** The loader accepts the zip or an extracted long-format CSV. Prefer the zip at the path above for a clean reproduce.

---

## 3. CCLE / DepMap expression

- **Expected filename:** `OmicsExpressionProteinCodingGenesTPMLogp1.csv`
- **Expected location:** `data/raw/OmicsExpressionProteinCodingGenesTPMLogp1.csv`
- **Official source:** https://depmap.org/portal/download/all/
- **Version:** Use the DepMap release that provides this standard filename. Record the DepMap release ID (e.g. 23Q2 / 24Q2) for manuscript methods.
- **Notes:** Multi-GB file; ensure disk space.

---

## 4. PRISM secondary-screen dose-response

- **Expected filename:** `secondary-screen-dose-response-curve-parameters.csv`
- **Expected location:** `data/raw/secondary-screen-dose-response-curve-parameters.csv`
- **Official source:** https://depmap.org/portal/download/all/
- **Notes:** CCLE-arm drug-response table. Phase 01 applies documented validity filters and median aggregation of duplicate cell x drug pairs (`src/data/loaders.py`).

---

## 5. CCLE sample info

- **Expected filename:** `sample_info.csv`
- **Expected location:** `data/raw/sample_info.csv`
- **Official source:** https://depmap.org/portal/download/all/
- **Notes:** Tissue / lineage annotations for CCLE models.

---

## 6. Cell Model Passports model list

- **Expected filename:** `model_list_20260724.csv`
- **Expected location:** `data/raw/model_list_20260724.csv`
- **Version used:** export dated **2026-07-24**
- **Official source:** https://cellmodelpassports.sanger.ac.uk/
- **Notes:** Model ID / tissue harmonization bridging GDSC2 and CCLE.

---

## 7. MSigDB Hallmark GMT

- **Expected filename:** `h.all.v2026.1.Hs.symbols.gmt`
- **Expected location:** `data/raw/h.all.v2026.1.Hs.symbols.gmt`
- **Version used:** MSigDB **v2026.1** Human symbols Hallmark collection
- **Official source:** https://www.gsea-msigdb.org/gsea/msigdb/
- **Notes:** Phase 01 scores a curated Hallmark subset via ssGSEA (`gseapy`).

---

## Directory layout after download

```text
data/
├── raw/
│   ├── GDSC2_fitted_dose_response_27Oct23.xlsx
│   ├── rnaseq_merged_20260323.zip
│   ├── OmicsExpressionProteinCodingGenesTPMLogp1.csv
│   ├── secondary-screen-dose-response-curve-parameters.csv
│   ├── sample_info.csv
│   ├── model_list_20260724.csv
│   └── h.all.v2026.1.Hs.symbols.gmt
└── processed/          # created automatically by the pipeline
```

Verify:

```bash
python -c "from pathlib import Path; r=Path('data/raw'); files=['GDSC2_fitted_dose_response_27Oct23.xlsx','rnaseq_merged_20260323.zip','OmicsExpressionProteinCodingGenesTPMLogp1.csv','secondary-screen-dose-response-curve-parameters.csv','sample_info.csv','model_list_20260724.csv','h.all.v2026.1.Hs.symbols.gmt'];
[print(('OK' if (r/f).exists() else 'MISSING')+': '+f) for f in files]"
```

---

## Preprocessing assumptions (frozen scientific code)

1. Tissue harmonization (`src/data/harmonize_tissue.py`).
2. Drug classes (`src/data/drug_class_map.py`); `Other` excluded from causal cohorts.
3. Pathway scores: ssGSEA on selected Hallmarks.
4. Treatment: top tercile within `drug_class`.
5. Outcome: `ln_ic50` (higher = more resistant); CCLE filters and duplicate median aggregation as coded.
6. Datasets analyzed separately (never pooled ATEs).

---

## Licensing of data

Upstream datasets follow Sanger / DepMap / Broad / MSigDB terms. This repository MIT license covers **code only**, not redistributed data.
