//! Validation runner: reads JSON input from stdin, computes matrix profile, writes JSON to stdout.
//!
//! Input format:
//! ```json
//! { "ts": [...], "m": 100, "signal_type": "sine", ... }
//! ```
//!
//! For streaming signals (signal_type == "streaming"):
//! ```json
//! { "ts_initial": [...], "ts_stream": [...], "m": 50, "signal_type": "streaming", ... }
//! ```

use motif_rs::algorithms::stampi::Stampi;
use motif_rs::{EuclideanEngine, MatrixProfileConfig, ZNormalizedEuclidean};
use std::io::{self, Read};
use std::time::Instant;

fn main() {
    let mut input = String::new();
    io::stdin().read_to_string(&mut input).unwrap();

    let data: serde_json::Value = serde_json::from_str(&input).unwrap();
    let signal_type = data["signal_type"].as_str().unwrap_or("batch");
    let m = data["m"].as_u64().unwrap() as usize;

    let result = if signal_type == "streaming" {
        run_streaming(&data, m)
    } else {
        run_batch(&data, m)
    };

    println!("{}", serde_json::to_string(&result).unwrap());
}

fn sanitize_profile(profile: &[f64]) -> Vec<serde_json::Value> {
    profile
        .iter()
        .map(|&v| {
            if v.is_infinite() {
                serde_json::Value::from(1e308)
            } else {
                serde_json::Value::from(v)
            }
        })
        .collect()
}

fn run_batch(data: &serde_json::Value, m: usize) -> serde_json::Value {
    let ts: Vec<f64> = data["ts"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect();

    let engine = EuclideanEngine::new(MatrixProfileConfig::new(m));

    let start = Instant::now();
    let mp = engine.compute(&ts);
    let compute_s = start.elapsed().as_secs_f64();

    serde_json::json!({
        "name": data["name"],
        "algorithm": "motif-rs::stomp",
        "m": m,
        "n": ts.len(),
        "compute_s": compute_s,
        "profile": sanitize_profile(&mp.profile),
        "profile_index": mp.profile_index,
    })
}

fn run_streaming(data: &serde_json::Value, m: usize) -> serde_json::Value {
    let ts_initial: Vec<f64> = data["ts_initial"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect();
    let ts_stream: Vec<f64> = data["ts_stream"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect();

    let config = MatrixProfileConfig::new(m);

    let start = Instant::now();
    let mut stampi = Stampi::<ZNormalizedEuclidean>::new(&ts_initial, config, false);
    for &val in &ts_stream {
        stampi.update(val);
    }
    let compute_s = start.elapsed().as_secs_f64();

    let mp = stampi.profile();

    serde_json::json!({
        "name": data["name"],
        "algorithm": "motif-rs::stampi",
        "m": m,
        "n_initial": ts_initial.len(),
        "n_stream": ts_stream.len(),
        "compute_s": compute_s,
        "profile": sanitize_profile(&mp.profile),
        "profile_index": mp.profile_index,
    })
}
