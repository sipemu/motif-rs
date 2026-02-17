use motif_rs::{EuclideanEngine, MatrixProfileConfig};
use serde::Deserialize;
use std::fs;

#[derive(Deserialize)]
struct GoldenData {
    ts: Vec<f64>,
    m: usize,
    profile: Vec<f64>,
    #[allow(dead_code)]
    profile_index: Vec<i64>,
    left_profile: Vec<f64>,
    #[allow(dead_code)]
    left_profile_index: Vec<i64>,
}

const EPSILON: f64 = 1e-6;

fn load_golden(filename: &str) -> GoldenData {
    let path = format!("tests/golden_data/{filename}");
    let data = fs::read_to_string(&path).unwrap_or_else(|_| {
        panic!(
            "Golden data file not found: {path}. Run: python scripts/generate_golden_data.py"
        )
    });
    serde_json::from_str(&data).unwrap()
}

/// Treat values > 1e300 as infinity (JSON uses 1e308 sentinel for inf).
fn is_sentinel_inf(v: f64) -> bool {
    v.is_infinite() || v > 1e300
}

fn assert_profile_match(name: &str, rust_profile: &[f64], stumpy_profile: &[f64], epsilon: f64) {
    assert_eq!(
        rust_profile.len(),
        stumpy_profile.len(),
        "{name}: profile length mismatch: rust={} vs stumpy={}",
        rust_profile.len(),
        stumpy_profile.len()
    );

    let mut max_diff = 0.0_f64;
    let mut max_diff_idx = 0;

    for (i, (r, s)) in rust_profile.iter().zip(stumpy_profile).enumerate() {
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
         (rust={}, stumpy={}), epsilon={epsilon:.0e}",
        rust_profile[max_diff_idx],
        stumpy_profile[max_diff_idx],
    );

    eprintln!("  {name}: max_diff = {max_diff:.2e} (epsilon = {epsilon:.0e})");
}

fn run_golden_test(filename: &str) {
    let golden = load_golden(filename);
    eprintln!(
        "Testing {filename}: n={}, m={}",
        golden.ts.len(),
        golden.m
    );

    let engine = EuclideanEngine::new(MatrixProfileConfig::new(golden.m));
    let mp = engine.compute(&golden.ts);

    assert_profile_match(
        &format!("{filename}/profile"),
        &mp.profile,
        &golden.profile,
        EPSILON,
    );
    assert_profile_match(
        &format!("{filename}/left_profile"),
        &mp.left_profile,
        &golden.left_profile,
        EPSILON,
    );
}

#[test]
fn test_stomp_vs_stumpy_sine_wave() {
    run_golden_test("sine_wave_mp.json");
}

#[test]
fn test_stomp_vs_stumpy_square_wave() {
    run_golden_test("square_wave_mp.json");
}

#[test]
fn test_stomp_vs_stumpy_mixed_signal() {
    run_golden_test("mixed_signal_mp.json");
}
