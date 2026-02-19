# motif-rs vs stumpy: Performance Benchmark Report

Methodology: 5 iterations per configuration, reporting median. stumpy timed after JIT warmup; motif-rs timed internally (excludes I/O).

## Batch STOMP (m=100)

| n | stumpy (s) | motif-rs (s) | Speedup |
|--:|----------:|------------:|--------:|
| 1,000 | 0.0106 | 0.0013 | **8.0x** |
| 5,000 | 0.0664 | 0.0099 | **6.7x** |
| 10,000 | 0.1120 | 0.0198 | **5.6x** |
| 25,000 | 0.3117 | 0.0864 | **3.6x** |
| 50,000 | 0.9176 | 0.3066 | **3.0x** |

## Streaming STAMPI (m=50, +200 points)

| n_initial | stumpy (s) | motif-rs (s) | Speedup |
|----------:|----------:|------------:|--------:|
| 500 | 0.0482 | 0.0065 | **7.4x** |
| 1,000 | 0.0546 | 0.0062 | **8.8x** |
| 2,000 | 0.0674 | 0.0118 | **5.7x** |
| 5,000 | 0.1056 | 0.0293 | **3.6x** |

## AAMP — Non-normalized Euclidean (m=100)

| n | stumpy (s) | motif-rs (s) | Speedup |
|--:|----------:|------------:|--------:|
| 1,000 | 0.0187 | 0.0017 | **11.2x** |
| 5,000 | 0.0971 | 0.0096 | **10.2x** |
| 10,000 | 0.2441 | 0.0299 | **8.2x** |
| 25,000 | 1.2615 | 0.2056 | **6.1x** |

## AB-Join (m=100, n_a = n_b = n)

| n | stumpy (s) | motif-rs (s) | Speedup |
|--:|----------:|------------:|--------:|
| 1,000 | 0.0104 | 0.0012 | **8.4x** |
| 5,000 | 0.0574 | 0.0092 | **6.2x** |
| 10,000 | 0.1065 | 0.0191 | **5.6x** |
| 25,000 | 0.3705 | 0.0930 | **4.0x** |

## Top-k Nearest Neighbors (m=100, k=3)

| n | stumpy (s) | motif-rs (s) | Speedup |
|--:|----------:|------------:|--------:|
| 1,000 | 0.0131 | 0.0018 | **7.3x** |
| 5,000 | 0.0702 | 0.0179 | **3.9x** |
| 10,000 | 0.1368 | 0.0442 | **3.1x** |
| 25,000 | 0.3861 | 0.2061 | **1.9x** |

## Snippets (m=100, k=3)

| n | stumpy (s) | motif-rs (s) | Speedup |
|--:|----------:|------------:|--------:|
| 1,000 | 0.0297 | 0.0009 | **32.2x** |
| 5,000 | 0.6802 | 0.0227 | **30.0x** |
| 10,000 | 2.7497 | 0.0569 | **48.3x** |
| 25,000 | 17.2441 | 0.3156 | **54.6x** |

## MPdist (m=100, n_a = n_b = n)

| n | stumpy (s) | motif-rs (s) | Speedup |
|--:|----------:|------------:|--------:|
| 1,000 | 0.0213 | 0.0015 | **14.6x** |
| 5,000 | 0.1028 | 0.0096 | **10.7x** |
| 10,000 | 0.2158 | 0.0212 | **10.2x** |

## STIMP — Pan Matrix Profile (min_m=10, max_m=100, step=1)

| n | stumpy (s) | motif-rs (s) | Speedup |
|--:|----------:|------------:|--------:|
| 1,000 | 1.2395 | 0.0314 | **39.5x** |
| 5,000 | 5.3374 | 0.6866 | **7.8x** |
| 10,000 | 11.2118 | 1.5844 | **7.1x** |

## MASS — Distance Profile (query_len=100)

| n | stumpy (s) | motif-rs (s) | Speedup |
|--:|----------:|------------:|--------:|
| 1,000 | 0.0019 | 0.0001 | **15.9x** |
| 5,000 | 0.0036 | 0.0011 | **3.5x** |
| 10,000 | 0.0035 | 0.0020 | **1.8x** |
| 25,000 | 0.0042 | 0.0019 | **2.2x** |
| 50,000 | 0.0057 | 0.0040 | **1.4x** |

## Chains — STOMP + ALLC (m=100)

| n | stumpy (s) | motif-rs (s) | Speedup |
|--:|----------:|------------:|--------:|
| 1,000 | 0.0143 | 0.0015 | **9.9x** |
| 5,000 | 0.0565 | 0.0080 | **7.1x** |
| 10,000 | 0.1171 | 0.0191 | **6.1x** |
| 25,000 | 0.3411 | 0.0829 | **4.1x** |

## MSTUMP — Multi-Dimensional Matrix Profile (d=3, m=20)

| n | stumpy (s) | motif-rs (s) | Speedup |
|--:|----------:|------------:|--------:|
| 1,000 | 0.1782 | 0.0028 | **62.8x** |
| 5,000 | 1.8673 | 0.0379 | **49.3x** |
| 10,000 | 5.6137 | 0.1495 | **37.6x** |
| 25,000 | 30.6977 | 0.9532 | **32.2x** |
| 50,000 | 117.4517 | 3.6105 | **32.5x** |

## Notes

- **stumpy** uses numba JIT-compiled parallel STOMP. First run triggers compilation; all measurements taken after warmup.
- **motif-rs** uses single-threaded STOMP with O(1) QT recurrence updates. Timing measured internally with `std::time::Instant` (excludes JSON I/O).
- Streaming benchmarks measure the full pipeline: initial batch computation + incremental updates for each new point.
- Both implementations produce numerically equivalent results (MAD < 1e-10, see comparison_report.md).
