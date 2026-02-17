# motif-rs vs stumpy: Performance Benchmark Report

Methodology: 5 iterations per configuration, reporting median. stumpy timed after JIT warmup; motif-rs timed internally (excludes I/O).

## Batch STOMP (m=100)

| n | stumpy (s) | motif-rs (s) | Speedup |
|--:|----------:|------------:|--------:|
| 1,000 | 0.0098 | 0.0012 | **8.0x** |
| 5,000 | 0.0495 | 0.0082 | **6.0x** |
| 10,000 | 0.1198 | 0.0197 | **6.1x** |
| 25,000 | 0.3074 | 0.0779 | **3.9x** |
| 50,000 | 0.8822 | 0.2739 | **3.2x** |

## Streaming STAMPI (m=50, +200 points)

| n_initial | stumpy (s) | motif-rs (s) | Speedup |
|----------:|----------:|------------:|--------:|
| 500 | 0.0474 | 0.0038 | **12.4x** |
| 1,000 | 0.0547 | 0.0066 | **8.3x** |
| 2,000 | 0.0675 | 0.0124 | **5.4x** |
| 5,000 | 0.1021 | 0.0292 | **3.5x** |

## AAMP — Non-normalized Euclidean (m=100)

| n | stumpy (s) | motif-rs (s) | Speedup |
|--:|----------:|------------:|--------:|
| 1,000 | 0.0094 | 0.0017 | **5.7x** |
| 5,000 | 0.0906 | 0.0098 | **9.2x** |
| 10,000 | 0.2391 | 0.0295 | **8.1x** |
| 25,000 | 1.2969 | 0.1967 | **6.6x** |

## AB-Join (m=100, n_a = n_b = n)

| n | stumpy (s) | motif-rs (s) | Speedup |
|--:|----------:|------------:|--------:|
| 1,000 | 0.0105 | 0.0021 | **5.0x** |
| 5,000 | 0.0572 | 0.0195 | **2.9x** |
| 10,000 | 0.1132 | 0.0784 | **1.4x** |
| 25,000 | 0.3906 | 0.2970 | **1.3x** |

## Top-k Nearest Neighbors (m=100, k=3)

| n | stumpy (s) | motif-rs (s) | Speedup |
|--:|----------:|------------:|--------:|
| 1,000 | 0.0156 | 0.0021 | **7.4x** |
| 5,000 | 0.0847 | 0.0170 | **5.0x** |
| 10,000 | 0.1470 | 0.0480 | **3.1x** |
| 25,000 | 0.3985 | 0.2237 | **1.8x** |

## Notes

- **stumpy** uses numba JIT-compiled parallel STOMP. First run triggers compilation; all measurements taken after warmup.
- **motif-rs** uses single-threaded STOMP with O(1) QT recurrence updates. Timing measured internally with `std::time::Instant` (excludes JSON I/O).
- Streaming benchmarks measure the full pipeline: initial batch computation + incremental updates for each new point.
- Both implementations produce numerically equivalent results (MAD < 1e-10, see comparison_report.md).
