#!/usr/bin/env python3
"""
Master Reproduction Script for Synthetic Lethality QSP Project
==============================================================

Runs the complete analysis pipeline for ATM-deficient CLL drug screening
via DDR pathway modeling.

Usage:
    cd github_QSP_SL/
    python scripts/run_all.py
"""

import sys
import subprocess
from pathlib import Path

# Project root
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


def run_script(script_name: str) -> bool:
    """Run a script and return True if successful."""
    script_path = ROOT / "scripts" / script_name
    print(f"\n{'='*70}")
    print(f"Running: {script_name}")
    print(f"{'='*70}")
    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(ROOT),
        capture_output=False
    )
    if result.returncode != 0:
        print(f"WARNING: {script_name} exited with code {result.returncode}")
        return False
    print(f"Completed: {script_name}")
    return True


def main():
    print("=" * 70)
    print("SYNTHETIC LETHALITY QSP - FULL REPRODUCTION PIPELINE")
    print("ATM-deficient CLL Drug Screening via DDR Pathway Modeling")
    print("=" * 70)

    # Verify we're in the repo root
    if not (ROOT / "src" / "enhanced_ddr_qsp_model.py").exists():
        print("ERROR: Cannot find src/enhanced_ddr_qsp_model.py")
        print("Please run this script from the repository root:")
        print("  python scripts/run_all.py")
        sys.exit(1)

    steps = [
        ("run_phase2_optimization.py", "Phase 2: Parameter optimization"),
        ("run_sensitivity_analysis.py", "Phase 2: Sensitivity analysis"),
        ("run_dose_response.py", "Phase 2: Concentration-response curves"),
        ("generate_figures.py", "Generate publication figures"),
    ]

    results = {}
    for script, description in steps:
        print(f"\n>>> Step: {description}")
        success = run_script(script)
        results[description] = "OK" if success else "FAILED"

    # Summary
    print("\n" + "=" * 70)
    print("PIPELINE SUMMARY")
    print("=" * 70)
    for step, status in results.items():
        marker = "[OK]" if status == "OK" else "[FAIL]"
        print(f"  {marker} {step}")

    failed = sum(1 for s in results.values() if s != "OK")
    if failed:
        print(f"\n{failed} step(s) had issues. Check output above for details.")
    else:
        print("\nAll steps completed successfully!")
        print(f"Results are in: {ROOT / 'data'}")
        print(f"Figures are in: {ROOT / 'figures'}")

    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
