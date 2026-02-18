# motif-rs vs stumpy: Performance Benchmark Report

Methodology: 5 iterations per configuration, reporting median. stumpy timed after JIT warmup; motif-rs timed internally (excludes I/O).

## Batch STOMP (m=100)

| n | stumpy (s) | motif-rs (s) | Speedup |
|--:|----------:|------------:|--------:|
| 1,000 | 0.0104 | 0.0014 | **7.7x** |
| 5,000 | 0.0551 | 0.0094 | **5.8x** |
| 10,000 | 0.1005 | 0.0206 | **4.9x** |
| 25,000 | 0.2978 | 0.0788 | **3.8x** |
| 50,000 | 0.8920 | 0.2712 | **3.3x** |

## Streaming STAMPI (m=50, +200 points)

| n_initial | stumpy (s) | motif-rs (s) | Speedup |
|----------:|----------:|------------:|--------:|
| 500 | 0.0484 | 0.0043 | **11.2x** |
| 1,000 | 0.0541 | 0.0063 | **8.6x** |
| 2,000 | 0.0671 | 0.0130 | **5.1x** |
| 5,000 | 0.1043 | 0.0293 | **3.6x** |

## AAMP — Non-normalized Euclidean (m=100)

| n | stumpy (s) | motif-rs (s) | Speedup |
|--:|----------:|------------:|--------:|
| 1,000 | 0.0104 | 0.0013 | **8.0x** |
| 5,000 | 0.0840 | 0.0109 | **7.7x** |
| 10,000 | 0.2504 | 0.0336 | **7.5x** |
| 25,000 | 1.4084 | 0.1972 | **7.1x** |

## AB-Join (m=100, n_a = n_b = n)

| n | stumpy (s) | motif-rs (s) | Speedup |
|--:|----------:|------------:|--------:|
| 1,000 | 0.0103 | 0.0014 | **7.5x** |
| 5,000 | 0.0566 | 0.0089 | **6.4x** |
| 10,000 | 0.1150 | 0.0234 | **4.9x** |
| 25,000 | 0.4036 | 0.1010 | **4.0x** |

## Top-k Nearest Neighbors (m=100, k=3)

| n | stumpy (s) | motif-rs (s) | Speedup |
|--:|----------:|------------:|--------:|
| 1,000 | 0.0161 | 0.0019 | **8.3x** |
| 5,000 | 0.0761 | 0.0176 | **4.3x** |
| 10,000 | 0.1509 | 0.0506 | **3.0x** |
| 25,000 | 0.3974 | 0.2412 | **1.6x** |

## Snippets (m=100, k=3)

| n | stumpy (s) | motif-rs (s) | Speedup |
|--:|----------:|------------:|--------:|
| 1,000 | 0.0267 | 0.0004 | **61.0x** |
| 5,000 | 0.6870 | 0.0171 | **40.2x** |
| 10,000 | 2.8453 | 0.0583 | **48.8x** |
| 25,000 | 17.3819 | 0.3153 | **55.1x** |

## MPdist (m=100, n_a = n_b = n)

| n | stumpy (s) | motif-rs (s) | Speedup |
|--:|----------:|------------:|--------:|
| 1,000 | 0.0291 | 0.0017 | **17.5x** |
| 5,000 | 0.1059 | 0.0092 | **11.6x** |
| 10,000 | 0.2237 | 0.0218 | **10.2x** |

## STIMP — Pan Matrix Profile (min_m=10, max_m=100, step=1)

| n | stumpy (s) | motif-rs (s) | Speedup |
|--:|----------:|------------:|--------:|
| 1,000 | 1.2317 | 0.0319 | **38.6x** |
| 5,000 | 5.4519 | 0.6050 | **9.0x** |
| 10,000 | 11.6819 | 1.6240 | **7.2x** |

## MASS — Distance Profile (query_len=100)

| n | stumpy (s) | motif-rs (s) | Speedup |
|--:|----------:|------------:|--------:|
| 1,000 | 0.0015 | 0.0002 | **8.3x** |
| 5,000 | 0.0034 | 0.0014 | **2.4x** |
| 10,000 | 0.0030 | 0.0020 | **1.5x** |
| 25,000 | 0.0039 | 0.0040 | **1.0x** |
| 50,000 | 0.0063 | 0.0041 | **1.5x** |

## Chains — STOMP + ALLC (m=100)

| n | stumpy (s) | motif-rs (s) | Speedup |
|--:|----------:|------------:|--------:|
| 1,000 | 0.0112 | 0.0014 | **7.9x** |
| 5,000 | 0.0649 | 0.0091 | **7.1x** |
| 10,000 | 0.1124 | 0.0183 | **6.1x** |
| 25,000 | 0.3551 | 0.0873 | **4.1x** |

## Notes

- **stumpy** uses numba JIT-compiled parallel STOMP. First run triggers compilation; all measurements taken after warmup.
- **motif-rs** uses single-threaded STOMP with O(1) QT recurrence updates. Timing measured internally with `std::time::Instant` (excludes JSON I/O).
- Streaming benchmarks measure the full pipeline: initial batch computation + incremental updates for each new point.
- Both implementations produce numerically equivalent results (MAD < 1e-10, see comparison_report.md).
