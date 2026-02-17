# motif-rs

A high-performance matrix profile library for time series analysis, written in Rust.

**3-8x faster than [stumpy](https://github.com/TDAmeritrade/stumpy)** (Python/Numba) across all tested sizes.

## What is a Matrix Profile?

The [matrix profile](https://www.cs.ucr.edu/~eamonn/MatrixProfile.html) is a data structure that stores the z-normalized Euclidean distance between every subsequence of a time series and its nearest neighbor. It enables efficient discovery of:

- **Motifs** — recurring patterns (subsequences with the smallest profile values)
- **Discords** — anomalies (subsequences with the largest profile values)
- **Shapelets**, **chains**, **regimes**, and more

## Performance

Benchmarked against stumpy (Numba JIT, parallel) on random walk data, m=100:

| n | stumpy | motif-rs | Speedup |
|--:|-------:|---------:|--------:|
| 1,000 | 0.013s | 0.002s | **8.6x** |
| 5,000 | 0.061s | 0.010s | **6.0x** |
| 10,000 | 0.094s | 0.019s | **5.0x** |
| 25,000 | 0.300s | 0.084s | **3.6x** |
| 50,000 | 0.870s | 0.292s | **3.0x** |

Key optimizations: correlation-domain inner loop (deferred sqrt), precomputed inverse standard deviations, hardware FMA via `f64::mul_add`, AoS cache-line accumulator, 4-wide diagonal grouping, and unsafe bounds elision. Built with fat LTO, single codegen unit, and `target-cpu=native`.

## Quick Start

Add to your `Cargo.toml`:

```toml
[dependencies]
motif-rs = { git = "https://github.com/sipemu/motif-rs" }
```

### Batch computation

```rust
use motif_rs::{EuclideanEngine, MatrixProfileConfig};

let ts = vec![1.0, 2.0, 3.0, 2.0, 1.0, 2.0, 3.0, 2.0, 1.0];
let engine = EuclideanEngine::new(MatrixProfileConfig::new(4));
let mp = engine.compute(&ts);

// Nearest-neighbor distances
println!("{:?}", mp.profile);
// Nearest-neighbor indices
println!("{:?}", mp.profile_index);
```

### Streaming (incremental updates)

```rust
use motif_rs::{EuclideanEngine, MatrixProfileConfig};

let initial_ts: Vec<f64> = (0..100).map(|i| (i as f64 * 0.1).sin()).collect();
let engine = EuclideanEngine::new(MatrixProfileConfig::new(10));

// Grow mode: time series extends with each new point
let mut stream = engine.streaming(&initial_ts, false);
stream.update(0.42);
stream.update(0.87);

let mp = stream.profile();
println!("Profile length: {}", mp.profile.len());
```

### Egress mode (sliding window)

```rust
use motif_rs::{EuclideanEngine, MatrixProfileConfig};

let initial_ts: Vec<f64> = (0..200).map(|i| (i as f64 * 0.1).sin()).collect();
let engine = EuclideanEngine::new(MatrixProfileConfig::new(10));

// Egress mode: fixed-size window, oldest points dropped
let mut stream = engine.streaming(&initial_ts, true);
stream.update(0.42); // window slides forward
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

## Configuration

```rust
use motif_rs::MatrixProfileConfig;

let mut config = MatrixProfileConfig::new(100); // subsequence length m=100
config.ignore_trivial = true;      // apply exclusion zone (default: true)
config.exclusion_zone_denom = 4;   // zone = ceil(m/4) (default: 4, matches stumpy)
```

## Features

| Feature | Default | Description |
|---------|---------|-------------|
| `parallel` | yes | Parallel diagonal computation via [Rayon](https://github.com/rayon-rs/rayon) |

```bash
# With parallelism (default)
cargo build --release

# Without parallelism
cargo build --release --no-default-features
```

## Algorithms

- **STOMP** (Scalable Time series Ordered-search Matrix Profile) — diagonal traversal with O(1) QT recurrence updates per element
- **STAMPI** — streaming/incremental matrix profile supporting grow and egress modes

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

## Validation

Results are validated against stumpy with MAD < 1e-10 across sine, square, mixed, and streaming signals:

```bash
pip install stumpy numpy
python validation/run_all.py
python validation/compare_results.py
```

## License

MIT
