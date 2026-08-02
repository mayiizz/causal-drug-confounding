# tools/ - Developer utilities

Scripts in this folder are **not** part of the publication pipeline (`run_pipeline.py`).

| File | Purpose |
|------|---------|
| `diagnose_ccle_ic50.py` | Ad-hoc CCLE/PRISM IC50 diagnostics used during Stage 1 data QA |

Run only if debugging data issues:

```bash
python tools/diagnose_ccle_ic50.py
```