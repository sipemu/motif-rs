# motif-rs vs stumpy: Performance Benchmark Report

Methodology: 5 iterations per configuration, reporting median. stumpy timed after JIT warmup; motif-rs timed internally (excludes I/O).

## Batch STOMP (m=100)

| n | stumpy (s) | motif-rs (s) | Speedup |
|--:|----------:|------------:|--------:|
| 1,000 | 0.0107 | 0.0015 | **7.1x** |
| 5,000 | 0.0573 | 0.0078 | **7.3x** |
| 10,000 | 0.0920 | 0.0185 | **5.0x** |
| 25,000 | 0.2944 | 0.0777 | **3.8x** |
| 50,000 | 0.8632 | 0.2755 | **3.1x** |

## Streaming STAMPI (m=50, +200 points)

| n_initial | stumpy (s) | motif-rs (s) | Speedup |
|----------:|----------:|------------:|--------:|
| 500 | 0.0484 | 0.0038 | **12.8x** |
| 1,000 | 0.0534 | 0.0068 | **7.8x** |
| 2,000 | 0.0671 | 0.0115 | **5.8x** |
| 5,000 | 0.1086 | 0.0293 | **3.7x** |

## Notes

- **stumpy** uses numba JIT-compiled parallel STOMP. First run triggers compilation; all measurements taken after warmup.
- **motif-rs** uses single-threaded STOMP with O(1) QT recurrence updates. Timing measured internally with `std::time::Instant` (excludes JSON I/O).
- Streaming benchmarks measure the full pipeline: initial batch computation + incremental updates for each new point.
- Both implementations produce numerically equivalent results (MAD < 1e-10, see comparison_report.md).
