# motif-rs vs stumpy: Performance Benchmark Report

Methodology: 5 iterations per configuration, reporting median. stumpy timed after JIT warmup; motif-rs timed internally (excludes I/O).

## Batch STOMP (m=100)

| n | stumpy (s) | motif-rs (s) | Speedup |
|--:|----------:|------------:|--------:|
| 1,000 | 0.0131 | 0.0015 | **8.6x** |
| 5,000 | 0.0614 | 0.0103 | **6.0x** |
| 10,000 | 0.0937 | 0.0187 | **5.0x** |
| 25,000 | 0.2997 | 0.0841 | **3.6x** |
| 50,000 | 0.8696 | 0.2916 | **3.0x** |

## Streaming STAMPI (m=50, +200 points)

| n_initial | stumpy (s) | motif-rs (s) | Speedup |
|----------:|----------:|------------:|--------:|
| 500 | 0.0501 | 0.0047 | **10.7x** |
| 1,000 | 0.0532 | 0.0065 | **8.2x** |
| 2,000 | 0.0673 | 0.0118 | **5.7x** |
| 5,000 | 0.1019 | 0.0297 | **3.4x** |

## Notes

- **stumpy** uses numba JIT-compiled parallel STOMP. First run triggers compilation; all measurements taken after warmup.
- **motif-rs** uses single-threaded STOMP with O(1) QT recurrence updates. Timing measured internally with `std::time::Instant` (excludes JSON I/O).
- Streaming benchmarks measure the full pipeline: initial batch computation + incremental updates for each new point.
- Both implementations produce numerically equivalent results (MAD < 1e-10, see comparison_report.md).
