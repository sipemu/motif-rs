# motif-rs

[![CI](https://github.com/sipemu/motif-rs/actions/workflows/ci.yml/badge.svg)](https://github.com/sipemu/motif-rs/actions/workflows/ci.yml)
[![MIT licensed](https://img.shields.io/badge/license-MIT-blue.svg)](./LICENSE)

A high-performance matrix profile library for time series analysis, validated against [stumpy](https://github.com/TDAmeritrade/stumpy).

The [matrix profile](https://www.cs.ucr.edu/~eamonn/MatrixProfile.html) stores the distance between every subsequence of a time series and its nearest neighbor, enabling discovery of motifs (recurring patterns), discords (anomalies), regime changes, and more.

## Features

- **Batch STOMP** — diagonal traversal with O(1) QT recurrence updates per element
- **Streaming STAMPI** — incremental matrix profile with grow and egress (sliding window) modes
- **AB-Join** — cross-series comparison (find shared patterns between two time series)
- **Top-k neighbors** — store the k nearest neighbors per subsequence, not just the best
- **Motif discovery** — extract the top-k most repeated patterns
- **Discord detection** — find the top-k most anomalous subsequences
- **FLUSS segmentation** — detect regime changes via the Corrected Arc Curve
- **Snippets** — extract k representative subsequences that best summarize a time series
- **MPdist** — scalar distance between two time series based on matrix profile
- **SCRUMP** — approximate matrix profile via PreSCRIMP diagonal sampling
- **Ostinato** — consensus motif discovery across multiple time series
- **STIMP** — pan matrix profile across a range of window sizes
- **AAMP** — non-normalized (absolute) Euclidean distance matrix profile
- **Parallel computation** — load-balanced diagonal partitioning via [Rayon](https://github.com/rayon-rs/rayon)
- **Left/right matrix profiles** — directional nearest-neighbor distances
- **Extensible metric trait** — `DistanceMetric` trait with static dispatch via monomorphization

## Performance

Benchmarked against stumpy (Numba JIT, parallel) on sine + noise signals, 5 iterations (median). motif-rs is faster across every feature and scale:

### Batch STOMP (z-normalized, m=100)

| n | stumpy | motif-rs | Speedup |
|--:|-------:|---------:|--------:|
| 1,000 | 0.010s | 0.001s | **7.7x** |
| 5,000 | 0.048s | 0.008s | **6.2x** |
| 10,000 | 0.104s | 0.018s | **5.8x** |
| 25,000 | 0.303s | 0.078s | **3.9x** |
| 50,000 | 0.858s | 0.269s | **3.2x** |

### Snippets (m=100, k=3)

| n | stumpy | motif-rs | Speedup |
|--:|-------:|---------:|--------:|
| 1,000 | 0.029s | 0.001s | **35.3x** |
| 5,000 | 0.701s | 0.014s | **48.7x** |
| 10,000 | 2.795s | 0.056s | **50.3x** |
| 25,000 | 17.691s | 0.310s | **57.1x** |

### AAMP (non-normalized Euclidean, m=100)

| n | stumpy | motif-rs | Speedup |
|--:|-------:|---------:|--------:|
| 1,000 | 0.015s | 0.001s | **10.8x** |
| 5,000 | 0.088s | 0.009s | **9.5x** |
| 10,000 | 0.234s | 0.034s | **6.9x** |
| 25,000 | 1.242s | 0.194s | **6.4x** |

### AB-Join (m=100, n_a = n_b)

| n | stumpy | motif-rs | Speedup |
|--:|-------:|---------:|--------:|
| 1,000 | 0.012s | 0.001s | **8.6x** |
| 5,000 | 0.058s | 0.007s | **8.1x** |
| 10,000 | 0.114s | 0.023s | **4.9x** |
| 25,000 | 0.372s | 0.101s | **3.7x** |

### Top-k Nearest Neighbors (m=100, k=3)

| n | stumpy | motif-rs | Speedup |
|--:|-------:|---------:|--------:|
| 1,000 | 0.015s | 0.002s | **7.8x** |
| 5,000 | 0.090s | 0.021s | **4.3x** |
| 10,000 | 0.153s | 0.048s | **3.2x** |
| 25,000 | 0.372s | 0.198s | **1.9x** |

### Streaming STAMPI (m=50, +200 points)

| n_initial | stumpy | motif-rs | Speedup |
|----------:|-------:|---------:|--------:|
| 500 | 0.048s | 0.004s | **11.5x** |
| 1,000 | 0.056s | 0.009s | **6.4x** |
| 2,000 | 0.069s | 0.012s | **5.8x** |
| 5,000 | 0.110s | 0.030s | **3.7x** |

Key optimizations: correlation-domain inner loop (deferred sqrt), precomputed inverse standard deviations, hardware FMA via `f64::mul_add`, AoS cache-line accumulator, 4-wide diagonal grouping, and unsafe bounds elision. Built with fat LTO, single codegen unit, and `target-cpu=native`.

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
motif-rs = { git = "https://github.com/sipemu/motif-rs" }
```

## Quick Start

### Batch Computation

```rust
use motif_rs::{EuclideanEngine, MatrixProfileConfig};

let ts = vec![1.0, 2.0, 3.0, 2.0, 1.0, 2.0, 3.0, 2.0, 1.0];
let engine = EuclideanEngine::new(MatrixProfileConfig::new(4));
let mp = engine.compute(&ts);

println!("Distances: {:?}", mp.profile);
println!("Indices:   {:?}", mp.profile_index);
```

### Motif and Discord Discovery

```rust
use motif_rs::{EuclideanEngine, MatrixProfileConfig};

let ts: Vec<f64> = (0..1000).map(|i| (i as f64 * 0.1).sin()).collect();
let engine = EuclideanEngine::new(MatrixProfileConfig::new(50));
let mp = engine.compute(&ts);

// Top-3 motifs (most repeated patterns)
let motifs = motif_rs::find_motifs(&mp, 3);
for m in &motifs {
    println!("Motif: indices ({}, {}), distance {:.4}", m.idx_a, m.idx_b, m.distance);
}

// Top-3 discords (most anomalous subsequences)
let discords = motif_rs::find_discords(&mp, 3);
for d in &discords {
    println!("Discord: index {}, distance {:.4}", d.idx, d.distance);
}
```

### AB-Join (Cross-Series Comparison)

```rust
use motif_rs::{EuclideanEngine, MatrixProfileConfig};

let ts_a: Vec<f64> = (0..500).map(|i| (i as f64 * 0.1).sin()).collect();
let ts_b: Vec<f64> = (0..500).map(|i| (i as f64 * 0.13).cos()).collect();

let engine = EuclideanEngine::new(MatrixProfileConfig::new(30));
let (join_a, join_b) = engine.ab_join(&ts_a, &ts_b);
// join_a: for each A subsequence, its nearest neighbor in B
// join_b: for each B subsequence, its nearest neighbor in A
```

### Regime Change Detection (FLUSS)

```rust
use motif_rs::{EuclideanEngine, MatrixProfileConfig};

let ts: Vec<f64> = (0..1000).map(|i| (i as f64 * 0.1).sin()).collect();
let engine = EuclideanEngine::new(MatrixProfileConfig::new(25));
let mp = engine.compute(&ts);

let seg = motif_rs::fluss(&mp, 2); // detect 2 regime boundaries
println!("Boundaries: {:?}", seg.regime_boundaries);
// seg.cac contains the Corrected Arc Curve (low values = boundaries)
```

### Snippets (Time Series Summarization)

```rust
use motif_rs::find_snippets;

let ts: Vec<f64> = (0..1000).map(|i| (i as f64 * 0.1).sin()).collect();
let result = find_snippets(&ts, 50, 3); // m=50, k=3 snippets

for (i, idx) in result.indices.iter().enumerate() {
    println!("Snippet {i}: starts at {idx}, covers {:.1}%", result.fractions[i] * 100.0);
}
// result.regimes[j] = which snippet each position is closest to
```

### Top-k Nearest Neighbors

```rust
use motif_rs::{EuclideanEngine, MatrixProfileConfig};

let ts: Vec<f64> = (0..500).map(|i| (i as f64 * 0.1).sin()).collect();
let engine = EuclideanEngine::new(MatrixProfileConfig::new(30));
let topk = engine.compute_topk(&ts, 3); // k=3 nearest neighbors
// topk.distances[i] = [d1, d2, d3] sorted ascending for subsequence i
```

### AAMP (Non-normalized Distance)

```rust
use motif_rs::{AampEngine, MatrixProfileConfig};

// AAMP uses raw Euclidean distance — amplitude and offset matter
let ts: Vec<f64> = (0..500).map(|i| (i as f64 * 0.1).sin()).collect();
let engine = AampEngine::new(MatrixProfileConfig::new(30));
let mp = engine.compute(&ts);
```

### MPdist (Time Series Distance)

```rust
use motif_rs::{EuclideanEngine, MatrixProfileConfig, ZNormalizedEuclidean};

let ts_a: Vec<f64> = (0..500).map(|i| (i as f64 * 0.1).sin()).collect();
let ts_b: Vec<f64> = (0..600).map(|i| (i as f64 * 0.13).cos()).collect();

// Scalar distance between two time series (robust to length differences)
let dist = motif_rs::mpdist::<ZNormalizedEuclidean>(&ts_a, &ts_b, 30, None);
println!("MPdist: {dist:.4}");
```

### SCRUMP (Approximate Matrix Profile)

```rust
use motif_rs::{EuclideanEngine, MatrixProfileConfig};

let ts: Vec<f64> = (0..5000).map(|i| (i as f64 * 0.1).sin()).collect();
let engine = EuclideanEngine::new(MatrixProfileConfig::new(50));

// Approximate profile using 25% of diagonals (faster, less accurate)
let mp_approx = engine.scrump(&ts, 0.25);

// percentage=1.0 computes the exact profile (same as engine.compute)
let mp_exact = engine.scrump(&ts, 1.0);
```

### Ostinato (Consensus Motif)

```rust
use motif_rs::{ZNormalizedEuclidean, ostinato};

let ts1: Vec<f64> = (0..300).map(|i| (i as f64 * 0.1).sin()).collect();
let ts2: Vec<f64> = (0..300).map(|i| (i as f64 * 0.1).sin() + 0.5).collect();
let ts3: Vec<f64> = (0..300).map(|i| (i as f64 * 0.1).sin() * 1.2).collect();

// Find the motif that appears across all time series
let result = ostinato::<ZNormalizedEuclidean>(&[&ts1, &ts2, &ts3], 25);
println!("Consensus motif in series {}, index {}, radius {:.4}",
    result.ts_index, result.subsequence_index, result.radius);
```

### STIMP (Pan Matrix Profile)

```rust
use motif_rs::{ZNormalizedEuclidean, stimp};

let ts: Vec<f64> = (0..500).map(|i| (i as f64 * 0.1).sin()).collect();

// Compute matrix profiles across window sizes 10 to 50
let pan = stimp::<ZNormalizedEuclidean>(&ts, 10, 50, Some(5), None);
for (i, m) in pan.windows.iter().enumerate() {
    println!("m={m}: min normalized distance = {:.4}",
        pan.profiles[i].iter().cloned().fold(f64::INFINITY, f64::min));
}
```

### Streaming (Grow Mode)

```rust
use motif_rs::{EuclideanEngine, MatrixProfileConfig};

let initial_ts: Vec<f64> = (0..100).map(|i| (i as f64 * 0.1).sin()).collect();
let engine = EuclideanEngine::new(MatrixProfileConfig::new(10));

let mut stream = engine.streaming(&initial_ts, false); // grow mode
stream.update(0.42);
stream.update(0.87);
let mp = stream.profile();
```

### Streaming (Egress Mode)

```rust
use motif_rs::{EuclideanEngine, MatrixProfileConfig};

let initial_ts: Vec<f64> = (0..200).map(|i| (i as f64 * 0.1).sin()).collect();
let engine = EuclideanEngine::new(MatrixProfileConfig::new(10));

// Fixed-size window: oldest points dropped as new ones arrive
let mut stream = engine.streaming(&initial_ts, true);
stream.update(0.42);
```

## Output

`MatrixProfile` contains six arrays, each of length `n - m + 1`:

| Field | Description |
|-------|-------------|
| `profile` | Nearest-neighbor distance for each subsequence |
| `profile_index` | Index of the nearest neighbor |
| `left_profile` | Distance to nearest neighbor with a smaller index |
| `left_profile_index` | Index of the left nearest neighbor |
| `right_profile` | Distance to nearest neighbor with a larger index |
| `right_profile_index` | Index of the right nearest neighbor |

## Cargo Features

| Feature | Default | Description |
|---------|---------|-------------|
| `parallel` | yes | Parallel diagonal computation via Rayon |

```bash
# With parallelism (default)
cargo build --release

# Without parallelism
cargo build --release --no-default-features
```

## Building

```bash
cargo build --release
cargo test
```

For best performance, ensure `.cargo/config.toml` targets your CPU:

```toml
[target.x86_64-unknown-linux-gnu]
rustflags = ["-C", "target-cpu=native"]
```

## Examples

Self-contained examples covering each feature (see `examples/`):

```bash
cargo run --release --example basic_matrix_profile
cargo run --release --example motif_discovery
cargo run --release --example anomaly_detection
cargo run --release --example segmentation
cargo run --release --example ab_join
cargo run --release --example streaming
cargo run --release --example aamp
```

## Validation

All features are validated against stumpy via golden integration tests. Each test loads reference data generated by stumpy and compares output at epsilon < 1e-6:

```bash
cargo test                           # run all tests (99 total)
cargo test --test features_golden_test  # AAMP, AB-join, top-k, motifs, discords, FLUSS, snippets, MPdist, SCRUMP, ostinato, STIMP
cargo test --test stomp_golden_test     # batch STOMP
cargo test --test stampi_golden_test    # streaming STAMPI
```

To regenerate golden data or run the stumpy comparison benchmarks:

```bash
pip install stumpy numpy
python scripts/generate_golden_features.py   # regenerate reference data
python validation/benchmark.py               # run performance benchmarks
```

## Dependencies

- [realfft](https://crates.io/crates/realfft) — FFT-based sliding dot product
- [rayon](https://crates.io/crates/rayon) — parallel computation (optional)

## License

MIT License
