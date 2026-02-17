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
use motif_rs::{AampEngine, EuclideanEngine, MatrixProfileConfig, ZNormalizedEuclidean};
use std::io::{self, Read};
use std::time::Instant;

fn main() {
    let mut input = String::new();
    io::stdin().read_to_string(&mut input).unwrap();

    let data: serde_json::Value = serde_json::from_str(&input).unwrap();
    let signal_type = data["signal_type"].as_str().unwrap_or("batch");
    let m = data["m"].as_u64().unwrap() as usize;

    let result = match signal_type {
        "streaming" => run_streaming(&data, m),
        "aamp" => run_aamp(&data, m),
        "ab_join" => run_ab_join(&data, m),
        "topk" => run_topk(&data, m),
        _ => run_batch(&data, m),
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
    let ts = parse_ts(data, "ts");
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

fn parse_ts(data: &serde_json::Value, key: &str) -> Vec<f64> {
    data[key]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect()
}

fn run_aamp(data: &serde_json::Value, m: usize) -> serde_json::Value {
    let ts = parse_ts(data, "ts");
    let engine = AampEngine::new(MatrixProfileConfig::new(m));

    let start = Instant::now();
    let mp = engine.compute(&ts);
    let compute_s = start.elapsed().as_secs_f64();

    serde_json::json!({
        "name": data["name"],
        "algorithm": "motif-rs::aamp",
        "m": m,
        "n": ts.len(),
        "compute_s": compute_s,
        "profile": sanitize_profile(&mp.profile),
    })
}

fn run_ab_join(data: &serde_json::Value, m: usize) -> serde_json::Value {
    let ts_a = parse_ts(data, "ts_a");
    let ts_b = parse_ts(data, "ts_b");
    let engine = EuclideanEngine::new(MatrixProfileConfig::new(m));

    let start = Instant::now();
    let (join_a, join_b) = engine.ab_join(&ts_a, &ts_b);
    let compute_s = start.elapsed().as_secs_f64();

    serde_json::json!({
        "name": data["name"],
        "algorithm": "motif-rs::ab_join",
        "m": m,
        "n_a": ts_a.len(),
        "n_b": ts_b.len(),
        "compute_s": compute_s,
        "profile_a": sanitize_profile(&join_a.distances),
        "profile_b": sanitize_profile(&join_b.distances),
    })
}

fn run_topk(data: &serde_json::Value, m: usize) -> serde_json::Value {
    let ts = parse_ts(data, "ts");
    let k = data["k"].as_u64().unwrap() as usize;
    let engine = EuclideanEngine::new(MatrixProfileConfig::new(m));

    let start = Instant::now();
    let _topk = engine.compute_topk(&ts, k);
    let compute_s = start.elapsed().as_secs_f64();

    serde_json::json!({
        "name": data["name"],
        "algorithm": "motif-rs::topk",
        "m": m,
        "k": k,
        "n": ts.len(),
        "compute_s": compute_s,
    })
}

fn run_streaming(data: &serde_json::Value, m: usize) -> serde_json::Value {
    let ts_initial = parse_ts(data, "ts_initial");
    let ts_stream = parse_ts(data, "ts_stream");

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
