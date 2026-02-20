# motif-rs

[![CI](https://github.com/sipemu/motif-rs/actions/workflows/ci.yml/badge.svg)](https://github.com/sipemu/motif-rs/actions/workflows/ci.yml)
[![Crates.io](https://img.shields.io/crates/v/motif-rs.svg)](https://crates.io/crates/motif-rs)
[![Documentation](https://docs.rs/motif-rs/badge.svg)](https://docs.rs/motif-rs)
[![codecov](https://codecov.io/gh/sipemu/motif-rs/branch/main/graph/badge.svg)](https://codecov.io/gh/sipemu/motif-rs)
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
- **Time series chains** — discover evolving patterns via bidirectional nearest-neighbor links (ATSC/ALLC)
- **MASS** — fast z-normalized distance profile for a query subsequence
- **Pattern matching** — find all occurrences of a query pattern in a time series
- **Multi-dimensional matrix profile (MSTUMP)** — compute matrix profiles across multiple co-evolving time series
- **Subspace selection** — identify which dimensions best characterize a motif pair
- **MDL dimensionality** — find optimal number of dimensions via Minimum Description Length
- **Multi-dimensional motifs (mmotifs)** — discover motifs that span multiple dimensions with automatic subspace selection
- **AAMP** — non-normalized (absolute) Euclidean distance matrix profile
- **P-norm** — Minkowski p-norm distance (Manhattan, Euclidean, or arbitrary p≥1) for self-join, AB-join, and MPdist
- **Configurable thresholds** — `sigma_threshold` for constant-subsequence detection (matches stumpy's `config.STUMPY_STDDEV_THRESHOLD`)
- **Parallel computation** — load-balanced diagonal partitioning via [Rayon](https://github.com/rayon-rs/rayon)
- **Left/right matrix profiles** — directional nearest-neighbor distances
- **Extensible metric trait** — `DistanceMetric` trait with static dispatch via monomorphization

## Performance

See [validation/PERFORMANCE.md](validation/PERFORMANCE.md) for detailed benchmark tables against stumpy across all features and scales.

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

### P-norm (Minkowski Distance)

```rust
use motif_rs::{stomp_pnorm, ab_join_pnorm, mpdist_pnorm, MatrixProfileConfig};

let ts: Vec<f64> = (0..500).map(|i| (i as f64 * 0.1).sin()).collect();
let config = MatrixProfileConfig::new(30);

// Manhattan distance (p=1)
let mp_manhattan = stomp_pnorm(&ts, &config, 1.0);

// Cubic norm (p=3)
let mp_cubic = stomp_pnorm(&ts, &config, 3.0);

// p=2.0 automatically delegates to the optimized AAMP path
let mp_euclidean = stomp_pnorm(&ts, &config, 2.0);

// P-norm AB-join
let ts_b: Vec<f64> = (0..600).map(|i| (i as f64 * 0.13).cos()).collect();
let (join_a, join_b) = ab_join_pnorm(&ts, &ts_b, 30, 1.5);

// P-norm MPdist
let dist = mpdist_pnorm(&ts, &ts_b, 30, 1.5, None);
```

Also available via `Engine` methods: `engine.compute_pnorm(&ts, p)`, `engine.ab_join_pnorm(&ts_a, &ts_b, p)`, and `engine.mpdist_pnorm(&ts_a, &ts_b, p, percentage)`.

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

### Time Series Chains (Evolving Patterns)

```rust
use motif_rs::{EuclideanEngine, MatrixProfileConfig, allc, atsc};

let ts: Vec<f64> = (0..500).map(|i| (i as f64 * 0.1).sin()).collect();
let engine = EuclideanEngine::new(MatrixProfileConfig::new(25));
let mp = engine.compute(&ts);

// Discover all chains (evolving patterns linked by bidirectional nearest neighbors)
let result = allc(&mp);
println!("Longest chain: {} links, indices: {:?}",
    result.longest.len(), result.longest.indices);
println!("Found {} unique chains", result.chains.len());

// Or trace a single chain from a specific anchor index
let chain = atsc(&mp, 0);
println!("Chain from index 0: {:?}", chain.indices);
```

### Pattern Matching (MASS / match)

```rust
use motif_rs::{mass, find_matches};

let ts: Vec<f64> = (0..1000).map(|i| (i as f64 * 0.1).sin()).collect();
let query = &ts[100..130]; // extract a 30-point query

// MASS: compute the z-normalized distance profile for the query
let distance_profile = mass(query, &ts);
println!("Best match distance: {:.4}", distance_profile.iter().cloned().fold(f64::INFINITY, f64::min));

// find_matches: extract all occurrences below a threshold
let matches = find_matches(query, &ts, None, None); // auto threshold + default exclusion zone
for m in &matches {
    println!("Match at index {}, distance {:.4}", m.index, m.distance);
}

// With explicit threshold and exclusion zone
let matches = find_matches(query, &ts, Some(0.5), Some(8));
```

### Multi-Dimensional Matrix Profile (MSTUMP)

```rust
use motif_rs::{mstump, subspace, mdl, mmotifs};

// Three co-evolving time series (e.g., x/y/z accelerometer axes)
let ts0: Vec<f64> = (0..300).map(|i| (i as f64 * 0.1).sin()).collect();
let ts1: Vec<f64> = (0..300).map(|i| (i as f64 * 0.2).sin()).collect();
let ts2: Vec<f64> = (0..300).map(|i| (i as f64 * 0.3).sin()).collect();
let ts_refs: Vec<&[f64]> = vec![&ts0, &ts1, &ts2];
let m = 20;

// Compute multi-dimensional matrix profile
let profile = mstump(&ts_refs, m);
// profile.profile[k][j] = best (k+1)-dimensional average distance at position j
// profile.profile_index[k][j] = nearest neighbor index

// Find which dimensions matter most for a motif pair
let idx = 0; // query position
let nn = profile.profile_index[0][idx]; // its nearest neighbor
let dims = subspace(&ts_refs, m, idx, nn, 2); // best 2 dimensions
println!("Best 2 dimensions for motif: {:?}", dims);

// MDL: find optimal number of dimensions
let d = ts_refs.len();
let subseq_idx: Vec<usize> = vec![idx; d];
let nn_idx: Vec<usize> = (0..d).map(|k| profile.profile_index[k][idx]).collect();
let (bit_sizes, subspaces) = mdl(&ts_refs, m, &subseq_idx, &nn_idx);
let optimal_k = bit_sizes.iter().enumerate()
    .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
    .map(|(i, _)| i + 1).unwrap();
println!("Optimal dimensionality: {}", optimal_k);

// Discover multi-dimensional motifs automatically
let motifs = mmotifs(&ts_refs, m, &profile, 3);
for motif in &motifs {
    println!("Motif: ({}, {}), k={}, dims={:?}, dist={:.4}",
        motif.idx, motif.nn_idx, motif.k, motif.dimensions, motif.distance);
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
cargo run --release --example basic_matrix_profile  # batch STOMP computation
cargo run --release --example motif_discovery        # find repeated patterns
cargo run --release --example anomaly_detection      # discord (anomaly) detection
cargo run --release --example segmentation           # FLUSS regime change detection
cargo run --release --example ab_join                # cross-series comparison
cargo run --release --example streaming              # incremental STAMPI updates
cargo run --release --example aamp                   # non-normalized Euclidean
cargo run --release --example topk                   # k nearest neighbors per subsequence
cargo run --release --example snippets               # time series summarization
cargo run --release --example mpdist                 # scalar distance between series
cargo run --release --example scrump                 # approximate matrix profile
cargo run --release --example ostinato               # consensus motif across series
cargo run --release --example stimp                  # pan matrix profile (multi-window)
cargo run --release --example chains                 # evolving pattern discovery
cargo run --release --example mass_matching          # MASS distance profile + pattern matching
cargo run --release --example multidimensional       # MSTUMP, subspace, MDL, mmotifs
cargo run --release --example pnorm                  # Minkowski p-norm distances
```

## Validation

All features are validated against stumpy via golden integration tests. Each test loads reference data generated by stumpy and compares output at epsilon < 1e-6:

```bash
cargo test                           # run all tests (149 total)
cargo test --test features_golden_test  # AAMP, AB-join, top-k, motifs, discords, FLUSS, snippets, MPdist, SCRUMP, ostinato, STIMP, chains, MASS/match, MSTUMP, subspace, MDL, mmotifs, p-norm
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

MIT License — see [LICENSE](LICENSE).

For dependency licenses and academic attribution, see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
