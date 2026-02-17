# motif-rs vs stumpy: Performance Benchmark Report

Methodology: 5 iterations per configuration, reporting median. stumpy timed after JIT warmup; motif-rs timed internally (excludes I/O).

## Batch STOMP (m=100)

| n | stumpy (s) | motif-rs (s) | Speedup |
|--:|----------:|------------:|--------:|
| 1,000 | 0.0104 | 0.0013 | **7.7x** |
| 5,000 | 0.0484 | 0.0078 | **6.2x** |
| 10,000 | 0.1040 | 0.0178 | **5.8x** |
| 25,000 | 0.3030 | 0.0783 | **3.9x** |
| 50,000 | 0.8577 | 0.2688 | **3.2x** |

## Streaming STAMPI (m=50, +200 points)

| n_initial | stumpy (s) | motif-rs (s) | Speedup |
|----------:|----------:|------------:|--------:|
| 500 | 0.0483 | 0.0042 | **11.5x** |
| 1,000 | 0.0558 | 0.0087 | **6.4x** |
| 2,000 | 0.0689 | 0.0118 | **5.8x** |
| 5,000 | 0.1099 | 0.0296 | **3.7x** |

## AAMP — Non-normalized Euclidean (m=100)

| n | stumpy (s) | motif-rs (s) | Speedup |
|--:|----------:|------------:|--------:|
| 1,000 | 0.0150 | 0.0014 | **10.8x** |
| 5,000 | 0.0876 | 0.0092 | **9.5x** |
| 10,000 | 0.2343 | 0.0337 | **6.9x** |
| 25,000 | 1.2416 | 0.1935 | **6.4x** |

## AB-Join (m=100, n_a = n_b = n)

| n | stumpy (s) | motif-rs (s) | Speedup |
|--:|----------:|------------:|--------:|
| 1,000 | 0.0117 | 0.0014 | **8.6x** |
| 5,000 | 0.0584 | 0.0072 | **8.1x** |
| 10,000 | 0.1135 | 0.0230 | **4.9x** |
| 25,000 | 0.3720 | 0.1014 | **3.7x** |

## Top-k Nearest Neighbors (m=100, k=3)

| n | stumpy (s) | motif-rs (s) | Speedup |
|--:|----------:|------------:|--------:|
| 1,000 | 0.0149 | 0.0019 | **7.8x** |
| 5,000 | 0.0903 | 0.0208 | **4.3x** |
| 10,000 | 0.1533 | 0.0475 | **3.2x** |
| 25,000 | 0.3724 | 0.1979 | **1.9x** |

## Snippets (m=100, k=3)

| n | stumpy (s) | motif-rs (s) | Speedup |
|--:|----------:|------------:|--------:|
| 1,000 | 0.0288 | 0.0008 | **35.3x** |
| 5,000 | 0.7006 | 0.0144 | **48.7x** |
| 10,000 | 2.7948 | 0.0556 | **50.3x** |
| 25,000 | 17.6913 | 0.3097 | **57.1x** |

## Notes

- **stumpy** uses numba JIT-compiled parallel STOMP. First run triggers compilation; all measurements taken after warmup.
- **motif-rs** uses single-threaded STOMP with O(1) QT recurrence updates. Timing measured internally with `std::time::Instant` (excludes JSON I/O).
- Streaming benchmarks measure the full pipeline: initial batch computation + incremental updates for each new point.
- Both implementations produce numerically equivalent results (MAD < 1e-10, see comparison_report.md).
