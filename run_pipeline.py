#!/usr/bin/env python3
"""Publication entry point: run Phases 01-10 in order (including 03b and 06b).

Does not modify scientific scripts - only orchestrates them via subprocess.
Stops immediately if any phase exits non-zero.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SCRIPTS = ROOT / "scripts"
PYTHON = sys.executable

PHASES = [
    ("01", "01_build_unified_data.py", ["data/processed/gdsc2_unified.parquet", "data/processed/ccle_unified.parquet"]),
    ("02", "02_preprocess_cohorts.py", ["data/processed/cross_dataset_cohorts.parquet"]),
    ("03", "03_common_support.py", ["data/processed/common_support_default.parquet"]),
    ("03b", "03b_balance_diagnostics.py", ["data/processed/balance_summary.parquet"]),
    ("04", "04_causal_estimation.py", ["data/processed/causal_estimates.parquet", "data/processed/ate_comparison.parquet"]),
    ("05", "05_robustness.py", ["data/processed/permutation_results.parquet", "data/processed/sensitivity_results.parquet"]),
    ("06", "06_heterogeneity.py", ["data/processed/cate_estimates.parquet"]),
    ("06b", "06b_counterfactual.py", ["data/processed/counterfactual_predictions.parquet"]),
    ("07", "07_validation.py", ["data/processed/pathway_validation.parquet"]),
    ("08", "08_power_analysis.py", ["data/processed/power_analysis.parquet"]),
    ("09", "09_generate_figures.py", ["output/figure1_cross_dataset_reproducibility.png", "output/table1_causal_estimates.csv"]),
    ("10", "10_baseline_models.py", ["output/table5_baseline_metrics.csv"]),
]


def parse_args():
    p = argparse.ArgumentParser(description="Run the full causal-drug-confounding pipeline.")
    p.add_argument("--from", dest="from_phase", default=None, help="Start at this phase id (e.g. 04 or 03b)")
    p.add_argument("--only", dest="only_phase", default=None, help="Run only this phase id")
    p.add_argument("--skip-existing", action="store_true", help="Skip a phase if marker outputs already exist")
    p.add_argument("--dry-run", action="store_true", help="Print the plan without executing")
    return p.parse_args()


def _index_of(phase_id):
    for i, (pid, *_rest) in enumerate(PHASES):
        if pid == phase_id:
            return i
    raise SystemExit("Unknown phase id: %s. Valid: %s" % (phase_id, [p[0] for p in PHASES]))


def _markers_exist(markers):
    return all((ROOT / m).exists() for m in markers)


def run_phase(phase_id, script_name):
    script = SCRIPTS / script_name
    if not script.exists():
        raise SystemExit("Missing script: %s" % script)

    print("\n" + "=" * 72)
    print("PHASE %s: %s" % (phase_id, script_name))
    print("=" * 72)
    print("  Command: %s %s" % (PYTHON, script.relative_to(ROOT)))
    sys.stdout.flush()

    t0 = time.perf_counter()
    result = subprocess.run([PYTHON, str(script)], cwd=str(ROOT))
    elapsed = time.perf_counter() - t0

    if result.returncode != 0:
        print("\nERROR: Phase %s failed with exit code %s after %.1fs" % (phase_id, result.returncode, elapsed))
        print("Pipeline stopped. Fix the error and re-run (optionally with --from).")
        raise SystemExit(result.returncode)

    print("\nPHASE %s completed in %.1fs" % (phase_id, elapsed))
    return elapsed


def main():
    args = parse_args()
    print("=" * 72)
    print("CAUSAL DRUG CONFOUNDING - FULL PIPELINE")
    print("=" * 72)
    print("  Root: %s" % ROOT)
    print("  Python: %s" % PYTHON)
    print("  Phases: %s" % " -> ".join(p[0] for p in PHASES))

    if args.only_phase:
        selected = [PHASES[_index_of(args.only_phase)]]
    else:
        start = _index_of(args.from_phase) if args.from_phase else 0
        selected = PHASES[start:]

    print("  Planned: %s" % " -> ".join(p[0] for p in selected))
    if args.dry_run:
        for pid, script, markers in selected:
            print("  - %s: scripts/%s  markers=%s" % (pid, script, markers))
        print("Dry run only - no phases executed.")
        return

    if selected[0][0] == "01":
        raw = ROOT / "data" / "raw"
        if not raw.exists():
            print("\nWARNING: %s does not exist. Create it and add files listed in DATA.md." % raw)

    totals = {}
    for pid, script, markers in selected:
        if args.skip_existing and _markers_exist(markers):
            print("\nSKIP Phase %s: marker outputs already exist (%s)" % (pid, ", ".join(markers)))
            continue
        totals[pid] = run_phase(pid, script)

    print("\n" + "=" * 72)
    print("PIPELINE COMPLETE")
    print("=" * 72)
    for pid, sec in totals.items():
        print("  Phase %3s: %.1f min" % (pid, sec / 60.0))
    print("  Total executed: %.1f min" % (sum(totals.values()) / 60.0))
    print("\nSee OUTPUTS.md for figures and tables.")


if __name__ == "__main__":
    main()