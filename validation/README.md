# motif-rs Validation

Systematic comparison of motif-rs against [stumpy](https://stumpy.readthedocs.io/) to ensure implementation correctness.

## Purpose

Validate that motif-rs produces numerically equivalent results to the established stumpy Python library across a range of signal types and algorithm modes (batch STOMP, streaming STAMPI).

## Methodology

### Test Signals

| Signal | Type | Parameters | Description |
|--------|------|------------|-------------|
| Sine wave | Batch | n=10000, m=100 | `sin(t) + 0.1 * noise`, seed=42 |
| Square wave | Batch | n=10000, m=100 | `sign(sin(t)) + 0.05 * noise`, seed=123 |
| Mixed signal | Batch | n=10000, m=100 | `sin(t) + 0.5*sin(3t) + 0.3*cos(7t) + noise`, seed=456 |
| Streaming sine | STAMPI | n_initial=200, n_stream=100, m=50 | `sin(t) + 0.05 * noise`, seed=789 |

### Comparison Metrics

- **MAD** (Mean Absolute Difference): Average element-wise absolute difference
- **Max Absolute Difference**: Worst-case element-wise difference
- **Pearson Correlation**: Correlation between profile vectors (should be ~1.0)
- Infinity sentinels (`1e308`) are matched separately

### Quality Tiers

| Tier | MAD Threshold | Correlation Threshold |
|------|---------------|-----------------------|
| Excellent | < 1e-6 | > 0.999999 |
| Good | < 1e-4 | > 0.9999 |
| Acceptable | < 1e-2 | > 0.99 |
| Concern | >= 1e-2 | <= 0.99 |

## Results Summary

| Test Case | Signal Type | MAD (profile) | Correlation | Status |
|-----------|------------|---------------|-------------|--------|
| Sine wave | Batch STOMP | < 1e-6 | > 0.999999 | Excellent |
| Square wave | Batch STOMP | < 1e-6 | > 0.999999 | Excellent |
| Mixed signal | Batch STOMP | < 1e-6 | > 0.999999 | Excellent |
| Streaming sine | STAMPI grow | < 1e-6 | > 0.999999 | Excellent |

*Results confirmed by `cargo test` golden integration tests (29 tests passing).*

## Performance Benchmarks

Performance comparison across multiple time series sizes (5 iterations, median reported):

### Batch STOMP (m=100)

| n | stumpy (s) | motif-rs (s) | Speedup |
|--:|-----------:|------------:|--------:|
| 1,000 | 0.0105 | 0.0081 | **1.3x** |
| 5,000 | 0.0465 | 0.1734 | 0.3x |
| 10,000 | 0.0941 | 0.6833 | 0.1x |
| 25,000 | 0.3015 | 4.3339 | 0.1x |
| 50,000 | 0.8642 | 17.2696 | 0.1x |

### Streaming STAMPI (m=50, +200 points)

| n_initial | stumpy (s) | motif-rs (s) | Speedup |
|----------:|-----------:|------------:|--------:|
| 500 | 0.0487 | 0.0044 | **11.2x** |
| 1,000 | 0.0533 | 0.0115 | **4.6x** |
| 2,000 | 0.0661 | 0.0366 | **1.8x** |
| 5,000 | 0.1024 | 0.1963 | 0.5x |

*Run `python validation/benchmark.py` to regenerate. Full results at `validation/results/benchmark_report.md`.*

**Benchmark methodology:**
- **stumpy**: numba JIT-compiled parallel STOMP. Warmup run before measurement to exclude JIT compilation.
- **motif-rs**: single-threaded STOMP with O(1) QT recurrence. Timing measured internally with `std::time::Instant` (excludes JSON I/O overhead).

## Numerical Notes

- Floating point precision differences of ~4.2e-8 are expected for identical/linear subsequences. This is inherent to IEEE 754 arithmetic in the z-normalized Euclidean distance formula: `d = sqrt(2*m*(1 - r))` where small differences in `r` near 1.0 are amplified.
- Both implementations use population standard deviation (not sample), matching stumpy's convention.
- Exclusion zone: `ceil(m/4)`, matching stumpy's default.
- Left profile at the first `ceil(m/4)` indices is infinity (no valid left neighbor exists within the exclusion zone).

## How to Reproduce

```bash
# Install dependencies
uv sync

# Run the full validation pipeline
python validation/run_all.py

# Or run individual steps:
python validation/generate_data.py      # Generate test signals
python validation/run_stumpy.py         # Run stumpy reference
python validation/run_rust.py           # Run motif-rs
python validation/compare_results.py    # Compare & generate report
python validation/benchmark.py         # Performance benchmarks
```

Generated reports:
- `validation/results/comparison_report.md` — correctness comparison
- `validation/results/benchmark_report.md` — performance benchmarks

## Structure

```
validation/
├── README.md                    # This file
├── generate_data.py             # Generate test time series
├── run_stumpy.py                # Run stumpy reference implementation
├── run_rust.py                  # Run motif-rs (via validation_runner binary)
├── compare_results.py           # Compare & generate reports
├── benchmark.py                 # Performance benchmarks
├── run_all.py                   # Pipeline orchestration
├── data/                        # Generated time series
└── results/
    ├── stumpy/                  # stumpy outputs
    ├── rust/                    # motif-rs outputs
    ├── comparison_report.md     # Generated comparison report
    ├── benchmark_report.md      # Generated benchmark report
    └── benchmark_raw.json       # Raw benchmark timing data
```
