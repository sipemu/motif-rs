# Performance

Benchmarked against stumpy (Numba JIT, parallel) on sine + noise signals, 5 iterations (median). motif-rs is faster across every feature and scale:

## Batch STOMP (z-normalized, m=100)

| n | stumpy | motif-rs | Speedup |
|--:|-------:|---------:|--------:|
| 1,000 | 0.011s | 0.001s | **8.0x** |
| 5,000 | 0.066s | 0.010s | **6.7x** |
| 10,000 | 0.112s | 0.020s | **5.6x** |
| 25,000 | 0.312s | 0.086s | **3.6x** |
| 50,000 | 0.918s | 0.307s | **3.0x** |

## MSTUMP (multi-dimensional, d=3, m=20)

| n | stumpy | motif-rs | Speedup |
|--:|-------:|---------:|--------:|
| 1,000 | 0.178s | 0.003s | **62.8x** |
| 5,000 | 1.867s | 0.038s | **49.3x** |
| 10,000 | 5.614s | 0.150s | **37.6x** |
| 25,000 | 30.698s | 0.953s | **32.2x** |
| 50,000 | 117.452s | 3.611s | **32.5x** |

## Snippets (m=100, k=3)

| n | stumpy | motif-rs | Speedup |
|--:|-------:|---------:|--------:|
| 1,000 | 0.030s | 0.001s | **32.2x** |
| 5,000 | 0.680s | 0.023s | **30.0x** |
| 10,000 | 2.750s | 0.057s | **48.3x** |
| 25,000 | 17.244s | 0.316s | **54.6x** |

## STIMP (pan matrix profile, m=10..100)

| n | stumpy | motif-rs | Speedup |
|--:|-------:|---------:|--------:|
| 1,000 | 1.240s | 0.031s | **39.5x** |
| 5,000 | 5.337s | 0.687s | **7.8x** |
| 10,000 | 11.212s | 1.584s | **7.1x** |

## MPdist (m=100, n_a = n_b)

| n | stumpy | motif-rs | Speedup |
|--:|-------:|---------:|--------:|
| 1,000 | 0.021s | 0.002s | **14.6x** |
| 5,000 | 0.103s | 0.010s | **10.7x** |
| 10,000 | 0.216s | 0.021s | **10.2x** |

## AAMP (non-normalized Euclidean, m=100)

| n | stumpy | motif-rs | Speedup |
|--:|-------:|---------:|--------:|
| 1,000 | 0.019s | 0.002s | **11.2x** |
| 5,000 | 0.097s | 0.010s | **10.2x** |
| 10,000 | 0.244s | 0.030s | **8.2x** |
| 25,000 | 1.262s | 0.206s | **6.1x** |

## AB-Join (m=100, n_a = n_b)

| n | stumpy | motif-rs | Speedup |
|--:|-------:|---------:|--------:|
| 1,000 | 0.010s | 0.001s | **8.4x** |
| 5,000 | 0.057s | 0.009s | **6.2x** |
| 10,000 | 0.107s | 0.019s | **5.6x** |
| 25,000 | 0.371s | 0.093s | **4.0x** |

## Chains (m=100)

| n | stumpy | motif-rs | Speedup |
|--:|-------:|---------:|--------:|
| 1,000 | 0.014s | 0.002s | **9.9x** |
| 5,000 | 0.057s | 0.008s | **7.1x** |
| 10,000 | 0.117s | 0.019s | **6.1x** |
| 25,000 | 0.341s | 0.083s | **4.1x** |

## Top-k Nearest Neighbors (m=100, k=3)

| n | stumpy | motif-rs | Speedup |
|--:|-------:|---------:|--------:|
| 1,000 | 0.013s | 0.002s | **7.3x** |
| 5,000 | 0.070s | 0.018s | **3.9x** |
| 10,000 | 0.137s | 0.044s | **3.1x** |
| 25,000 | 0.386s | 0.206s | **1.9x** |

## Streaming STAMPI (m=50, +200 points)

| n_initial | stumpy | motif-rs | Speedup |
|----------:|-------:|---------:|--------:|
| 500 | 0.048s | 0.007s | **7.4x** |
| 1,000 | 0.055s | 0.006s | **8.8x** |
| 2,000 | 0.067s | 0.012s | **5.7x** |
| 5,000 | 0.106s | 0.029s | **3.6x** |

## MASS (query_len=100)

| n | stumpy | motif-rs | Speedup |
|--:|-------:|---------:|--------:|
| 1,000 | 0.002s | 0.0001s | **15.9x** |
| 5,000 | 0.004s | 0.001s | **3.5x** |
| 10,000 | 0.004s | 0.002s | **1.8x** |
| 25,000 | 0.004s | 0.002s | **2.2x** |
| 50,000 | 0.006s | 0.004s | **1.4x** |

## Key Optimizations

Diagonal traversal with O(1) QT recurrence (eliminates per-row FFT calls), correlation-domain inner loop (deferred sqrt), precomputed inverse standard deviations, hardware FMA via `f64::mul_add`, AoS cache-line accumulator, 4-wide diagonal grouping, parallel diagonal partitioning via rayon, and symmetry exploitation. Built with fat LTO, single codegen unit, and `target-cpu=native`.
