use motif_rs::{MatrixProfileConfig, ZNormalizedEuclidean};
use motif_rs::algorithms::stampi::Stampi;
use motif_rs::algorithms::stomp::stomp;
use serde::Deserialize;
use std::fs;

#[derive(Deserialize)]
struct StreamingGoldenData {
    ts_initial: Vec<f64>,
    ts_stream: Vec<f64>,
    ts_full: Vec<f64>,
    m: usize,
    streaming_profile: Vec<f64>,
    #[allow(dead_code)]
    streaming_profile_index: Vec<i64>,
    #[allow(dead_code)]
    streaming_left_profile: Vec<f64>,
    #[allow(dead_code)]
    streaming_left_profile_index: Vec<i64>,
    batch_profile: Vec<f64>,
    #[allow(dead_code)]
    batch_profile_index: Vec<i64>,
}

const EPSILON: f64 = 1e-6;

fn load_streaming_golden() -> StreamingGoldenData {
    let path = "tests/golden_data/streaming_sine_mp.json";
    let data = fs::read_to_string(path).unwrap_or_else(|_| {
        panic!("Golden data file not found: {path}. Run: python scripts/generate_golden_data.py")
    });
    serde_json::from_str(&data).unwrap()
}

/// Treat values > 1e300 as infinity (JSON uses 1e308 sentinel for inf).
fn is_sentinel_inf(v: f64) -> bool {
    v.is_infinite() || v > 1e300
}

fn assert_profile_match(name: &str, rust_profile: &[f64], ref_profile: &[f64], epsilon: f64) {
    assert_eq!(
        rust_profile.len(),
        ref_profile.len(),
        "{name}: length mismatch: rust={} vs ref={}",
        rust_profile.len(),
        ref_profile.len()
    );

    let mut max_diff = 0.0_f64;
    let mut max_diff_idx = 0;

    for (i, (r, s)) in rust_profile.iter().zip(ref_profile).enumerate() {
        // Both infinite (or sentinel) → match
        if is_sentinel_inf(*r) && is_sentinel_inf(*s) {
            continue;
        }
        let diff = (r - s).abs();
        if diff > max_diff {
            max_diff = diff;
            max_diff_idx = i;
        }
    }

    assert!(
        max_diff < epsilon,
        "{name}: max diff = {max_diff:.2e} at index {max_diff_idx} \
         (rust={}, ref={}), epsilon={epsilon:.0e}",
        rust_profile[max_diff_idx],
        ref_profile[max_diff_idx],
    );

    eprintln!("  {name}: max_diff = {max_diff:.2e} (epsilon = {epsilon:.0e})");
}

#[test]
fn test_stampi_grow_matches_batch_on_full_series() {
    let golden = load_streaming_golden();
    let config = MatrixProfileConfig::new(golden.m);

    // Build streaming profile
    let mut stampi =
        Stampi::<ZNormalizedEuclidean>::new(&golden.ts_initial, config.clone(), false);
    for &val in &golden.ts_stream {
        stampi.update(val);
    }

    // Compare streaming against batch STOMP on the full series
    let batch_mp = stomp::<ZNormalizedEuclidean>(&golden.ts_full, &config);

    assert_profile_match(
        "streaming_vs_batch/profile",
        &stampi.profile().profile,
        &batch_mp.profile,
        EPSILON,
    );
}

#[test]
fn test_batch_vs_stumpy_on_full_series() {
    let golden = load_streaming_golden();
    let config = MatrixProfileConfig::new(golden.m);

    let mp = stomp::<ZNormalizedEuclidean>(&golden.ts_full, &config);

    assert_profile_match(
        "batch_vs_stumpy/profile",
        &mp.profile,
        &golden.batch_profile,
        EPSILON,
    );
}

#[test]
fn test_streaming_profile_vs_stumpy_streaming() {
    let golden = load_streaming_golden();
    let config = MatrixProfileConfig::new(golden.m);

    let mut stampi =
        Stampi::<ZNormalizedEuclidean>::new(&golden.ts_initial, config.clone(), false);
    for &val in &golden.ts_stream {
        stampi.update(val);
    }

    // Compare against stumpy's streaming result
    assert_profile_match(
        "streaming_vs_stumpy_streaming/profile",
        &stampi.profile().profile,
        &golden.streaming_profile,
        EPSILON,
    );
}
