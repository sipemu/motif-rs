#!/usr/bin/env python3
"""Pipeline orchestration: generate data, run both implementations, compare results.

Usage:
    uv sync && python validation/run_all.py
"""

import subprocess
import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def run_step(name: str, script: str) -> bool:
    """Run a pipeline step."""
    print(f"\n{'=' * 60}")
    print(f"  Step: {name}")
    print(f"{'=' * 60}\n")

    result = subprocess.run(
        [sys.executable, os.path.join(SCRIPT_DIR, script)],
        cwd=os.path.dirname(SCRIPT_DIR),
    )
    if result.returncode != 0:
        print(f"\n  FAILED: {name}")
        return False
    return True


def main():
    print("motif-rs Validation Pipeline")
    print("=" * 60)

    steps = [
        ("Generate test data", "generate_data.py"),
        ("Run stumpy reference", "run_stumpy.py"),
        ("Run motif-rs", "run_rust.py"),
        ("Compare results", "compare_results.py"),
        ("Performance benchmarks", "benchmark.py"),
    ]

    for name, script in steps:
        if not run_step(name, script):
            print(f"\nPipeline failed at: {name}")
            sys.exit(1)

    print(f"\n{'=' * 60}")
    print("  Validation pipeline complete!")
    print(f"{'=' * 60}")
    print(f"\n  Reports:")
    print(f"    validation/results/comparison_report.md")
    print(f"    validation/results/benchmark_report.md")


if __name__ == "__main__":
    main()
