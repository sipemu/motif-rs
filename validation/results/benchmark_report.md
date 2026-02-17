# motif-rs vs stumpy: Performance Benchmark Report

Methodology: 5 iterations per configuration, reporting median. stumpy timed after JIT warmup; motif-rs timed internally (excludes I/O).

## Batch STOMP (m=100)

| n | stumpy (s) | motif-rs (s) | Speedup |
|--:|----------:|------------:|--------:|
| 1,000 | 0.0104 | 0.0013 | **7.8x** |
| 5,000 | 0.0586 | 0.0091 | **6.5x** |
| 10,000 | 0.1058 | 0.0214 | **4.9x** |
| 25,000 | 0.3068 | 0.0831 | **3.7x** |
| 50,000 | 0.9161 | 0.2870 | **3.2x** |

## Streaming STAMPI (m=50, +200 points)

| n_initial | stumpy (s) | motif-rs (s) | Speedup |
|----------:|----------:|------------:|--------:|
| 500 | 0.0483 | 0.0042 | **11.6x** |
| 1,000 | 0.0549 | 0.0061 | **9.0x** |
| 2,000 | 0.0706 | 0.0114 | **6.2x** |
| 5,000 | 0.1149 | 0.0301 | **3.8x** |

## AAMP — Non-normalized Euclidean (m=100)

| n | stumpy (s) | motif-rs (s) | Speedup |
|--:|----------:|------------:|--------:|
| 1,000 | 0.0111 | 0.0012 | **9.1x** |
| 5,000 | 0.0923 | 0.0138 | **6.7x** |
| 10,000 | 0.2785 | 0.0372 | **7.5x** |
| 25,000 | 1.3268 | 0.1967 | **6.7x** |

## AB-Join (m=100, n_a = n_b = n)

| n | stumpy (s) | motif-rs (s) | Speedup |
|--:|----------:|------------:|--------:|
| 1,000 | 0.0107 | 0.0014 | **7.8x** |
| 5,000 | 0.0509 | 0.0089 | **5.7x** |
| 10,000 | 0.1120 | 0.0202 | **5.5x** |
| 25,000 | 0.3839 | 0.0982 | **3.9x** |

## Top-k Nearest Neighbors (m=100, k=3)

| n | stumpy (s) | motif-rs (s) | Speedup |
|--:|----------:|------------:|--------:|
| 1,000 | 0.0129 | 0.0020 | **6.4x** |
| 5,000 | 0.0737 | 0.0165 | **4.5x** |
| 10,000 | 0.1440 | 0.0486 | **3.0x** |
| 25,000 | 0.3967 | 0.2267 | **1.7x** |

## Snippets (m=100, k=3)

| n | stumpy (s) | motif-rs (s) | Speedup |
|--:|----------:|------------:|--------:|
| 1,000 | 0.0325 | 0.0009 | **36.5x** |
| 5,000 | 0.6933 | 0.0174 | **39.8x** |
| 10,000 | 2.7850 | 0.0635 | **43.8x** |
| 25,000 | 17.5260 | 0.3186 | **55.0x** |

## MPdist (m=100, n_a = n_b = n)

| n | stumpy (s) | motif-rs (s) | Speedup |
|--:|----------:|------------:|--------:|
| 1,000 | 0.0213 | 0.0012 | **17.8x** |
| 5,000 | 0.1133 | 0.0097 | **11.7x** |
| 10,000 | 0.2360 | 0.0212 | **11.1x** |

## STIMP — Pan Matrix Profile (min_m=10, max_m=100, step=5)

| n | stumpy (s) | motif-rs (s) | Speedup |
|--:|----------:|------------:|--------:|
| 1,000 | 0.0004 | 0.0117 | **0.0x** |
| 5,000 | 0.0007 | 0.1772 | **0.0x** |
| 10,000 | 0.0009 | 0.3488 | **0.0x** |

## Notes

- **stumpy** uses numba JIT-compiled parallel STOMP. First run triggers compilation; all measurements taken after warmup.
- **motif-rs** uses single-threaded STOMP with O(1) QT recurrence updates. Timing measured internally with `std::time::Instant` (excludes JSON I/O).
- Streaming benchmarks measure the full pipeline: initial batch computation + incremental updates for each new point.
- Both implementations produce numerically equivalent results (MAD < 1e-10, see comparison_report.md).
