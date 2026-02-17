#!/usr/bin/env python3
"""Compare motif-rs results against stumpy reference.

Loads results from results/stumpy/ and results/rust/, computes comparison
metrics, and generates a comparison report at results/comparison_report.md.
"""

import json
import os
from dataclasses import dataclass

import numpy as np

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
STUMPY_DIR = os.path.join(RESULTS_DIR, "stumpy")
RUST_DIR = os.path.join(RESULTS_DIR, "rust")
REPORT_PATH = os.path.join(RESULTS_DIR, "comparison_report.md")

INF_SENTINEL = 1e300


@dataclass
class ComparisonResult:
    name: str
    signal_type: str
    n: int
    m: int
    mad: float
    max_abs_diff: float
    max_diff_idx: int
    correlation: float
    stumpy_time: float
    rust_time: float
    n_inf_match: int
    n_total: int
    status: str


def classify_status(mad: float, correlation: float) -> str:
    """Categorize result quality."""
    if mad < 1e-6 and correlation > 0.999999:
        return "Excellent"
    elif mad < 1e-4 and correlation > 0.9999:
        return "Good"
    elif mad < 1e-2 and correlation > 0.99:
        return "Acceptable"
    else:
        return "CONCERN"


def compare_profiles(stumpy_profile: list, rust_profile: list) -> dict:
    """Compare two profile arrays, handling infinity sentinels."""
    sp = np.array(stumpy_profile)
    rp = np.array(rust_profile)

    assert len(sp) == len(rp), f"Length mismatch: stumpy={len(sp)} vs rust={len(rp)}"

    # Identify infinity sentinels
    sp_inf = sp > INF_SENTINEL
    rp_inf = rp > INF_SENTINEL
    both_inf = sp_inf & rp_inf
    n_inf_match = int(both_inf.sum())

    # Compare only finite values
    finite_mask = ~sp_inf & ~rp_inf
    sp_finite = sp[finite_mask]
    rp_finite = rp[finite_mask]

    if len(sp_finite) == 0:
        return {
            "mad": 0.0,
            "max_abs_diff": 0.0,
            "max_diff_idx": 0,
            "correlation": 1.0,
            "n_inf_match": n_inf_match,
            "n_total": len(sp),
        }

    diffs = np.abs(sp_finite - rp_finite)
    mad = float(np.mean(diffs))
    max_abs_diff = float(np.max(diffs))

    # Find the original index of max diff
    finite_indices = np.where(finite_mask)[0]
    max_diff_idx = int(finite_indices[np.argmax(diffs)])

    # Pearson correlation
    if np.std(sp_finite) < 1e-15 or np.std(rp_finite) < 1e-15:
        correlation = 1.0 if np.allclose(sp_finite, rp_finite) else 0.0
    else:
        correlation = float(np.corrcoef(sp_finite, rp_finite)[0, 1])

    return {
        "mad": mad,
        "max_abs_diff": max_abs_diff,
        "max_diff_idx": max_diff_idx,
        "correlation": correlation,
        "n_inf_match": n_inf_match,
        "n_total": len(sp),
    }


def compare_one(name: str) -> ComparisonResult | None:
    """Compare a single test case."""
    stumpy_path = os.path.join(STUMPY_DIR, f"{name}.json")
    rust_path = os.path.join(RUST_DIR, f"{name}.json")

    if not os.path.exists(stumpy_path):
        print(f"  {name}: no stumpy results, skipping")
        return None
    if not os.path.exists(rust_path):
        print(f"  {name}: no rust results, skipping")
        return None

    with open(stumpy_path) as f:
        stumpy_data = json.load(f)
    with open(rust_path) as f:
        rust_data = json.load(f)

    metrics = compare_profiles(stumpy_data["profile"], rust_data["profile"])
    status = classify_status(metrics["mad"], metrics["correlation"])

    signal_type = stumpy_data.get("algorithm", "batch").split(".")[-1]
    n = stumpy_data.get("n", stumpy_data.get("n_initial", 0))
    m = stumpy_data["m"]

    result = ComparisonResult(
        name=name,
        signal_type=signal_type,
        n=n,
        m=m,
        mad=metrics["mad"],
        max_abs_diff=metrics["max_abs_diff"],
        max_diff_idx=metrics["max_diff_idx"],
        correlation=metrics["correlation"],
        stumpy_time=stumpy_data.get("elapsed_s", 0),
        rust_time=rust_data.get("elapsed_s", 0),
        n_inf_match=metrics["n_inf_match"],
        n_total=metrics["n_total"],
        status=status,
    )

    print(f"  {name}: MAD={result.mad:.2e}, corr={result.correlation:.8f}, status={result.status}")
    return result


def generate_report(results: list[ComparisonResult]):
    """Generate markdown comparison report."""
    report = []
    report.append("# motif-rs vs stumpy: Validation Comparison Report\n")
    report.append(f"Generated from `python validation/compare_results.py`\n")

    # Summary table
    report.append("## Results Summary\n")
    report.append("| Test Case | Algorithm | MAD (profile) | Max Abs Diff | Correlation | Status |")
    report.append("|-----------|-----------|---------------|--------------|-------------|--------|")

    for r in results:
        report.append(
            f"| {r.name} | {r.signal_type} | {r.mad:.2e} | {r.max_abs_diff:.2e} "
            f"| {r.correlation:.8f} | {r.status} |"
        )

    # Performance comparison
    report.append("\n## Performance Comparison\n")
    report.append("| Test Case | stumpy (s) | motif-rs (s) | Speedup |")
    report.append("|-----------|------------|--------------|---------|")
    for r in results:
        if r.stumpy_time > 0 and r.rust_time > 0:
            speedup = r.stumpy_time / r.rust_time
            report.append(
                f"| {r.name} | {r.stumpy_time:.3f} | {r.rust_time:.3f} | {speedup:.1f}x |"
            )

    # Detailed notes
    report.append("\n## Numerical Notes\n")
    report.append(
        "- Floating point precision differences of ~4.2e-8 are expected for identical/linear "
        "subsequences due to IEEE 754 arithmetic in the distance formula"
    )
    report.append(
        "- Infinity sentinels (1e308) are used in JSON serialization since JSON does not "
        "support IEEE 754 infinity values"
    )
    report.append(
        "- MAD (Mean Absolute Difference) < 1e-6 with correlation > 0.999999 is classified "
        "as 'Excellent'"
    )

    # Quality tiers
    report.append("\n## Quality Tiers\n")
    report.append("| Tier | MAD Threshold | Correlation Threshold |")
    report.append("|------|---------------|-----------------------|")
    report.append("| Excellent | < 1e-6 | > 0.999999 |")
    report.append("| Good | < 1e-4 | > 0.9999 |")
    report.append("| Acceptable | < 1e-2 | > 0.99 |")
    report.append("| Concern | >= 1e-2 | <= 0.99 |")

    report_text = "\n".join(report) + "\n"

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(REPORT_PATH, "w") as f:
        f.write(report_text)
    print(f"\n  Report written to {REPORT_PATH}")

    return report_text


def compare_all():
    print("Comparing results...\n")

    test_cases = ["sine_wave", "square_wave", "mixed_signal", "streaming_sine"]
    results = []

    for name in test_cases:
        result = compare_one(name)
        if result:
            results.append(result)

    if results:
        generate_report(results)
    else:
        print("  No results to compare. Run run_stumpy.py and run_rust.py first.")

    # Check for concerns
    concerns = [r for r in results if r.status == "CONCERN"]
    if concerns:
        print(f"\n  WARNING: {len(concerns)} test case(s) flagged as CONCERN")
        for c in concerns:
            print(f"    - {c.name}: MAD={c.mad:.2e}, corr={c.correlation:.8f}")
        return False

    return True


if __name__ == "__main__":
    success = compare_all()
    exit(0 if success else 1)
