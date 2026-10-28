# run_all.py
# ---------------------------------------------------------------
# MINE Framework — Full Pipeline Orchestrator
# Executes all phases in sequence. Aborts on first failure.
# ---------------------------------------------------------------

import subprocess
import sys
import time

steps = [
    # (label, script_path)
    ("Phase 2 — Homogeneous Ensemble Generation",       "phase2_ensemble/run_phase2.py"),
    ("Phases 3-4 — Dynamic Regressor Selection",        "phase3_4_dynamic/run_phase3_4.py"),
    ("Phases 3-4 — Results Extraction to CSV",          "phase3_4_dynamic/results_to_csv.py"),
]

def run_pipeline():
    total_start = time.time()
    print("\n" + "=" * 60)
    print("  MINE Framework — Full Pipeline")
    print("=" * 60)

    for i, (name, script) in enumerate(steps, start=1):
        print(f"\n[{i}/{len(steps)}] {name}")
        print("-" * 60)
        step_start = time.time()

        result = subprocess.run(
            [sys.executable, script],
            check=False
        )

        elapsed = time.time() - step_start

        if result.returncode != 0:
            print(f"\n[ERROR] Step '{name}' failed with return code {result.returncode}.")
            print("Pipeline aborted.")
            sys.exit(result.returncode)

        print(f"[OK] Completed in {elapsed:.1f}s")

    total_elapsed = time.time() - total_start
    print("\n" + "=" * 60)
    print(f"  Pipeline completed successfully in {total_elapsed:.1f}s")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    run_pipeline()
