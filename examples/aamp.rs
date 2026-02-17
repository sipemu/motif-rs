//! AAMP: Non-normalized (Absolute) Euclidean Distance Matrix Profile.
//!
//! This example demonstrates AAMP, which computes the matrix profile
//! using raw Euclidean distance (no z-normalization). This is useful
//! when the amplitude and offset of patterns matter, not just shape.
//!
//! Run with: cargo run --release --example aamp

use motif_rs::{AampEngine, EuclideanEngine, MatrixProfileConfig};

fn main() {
    let m = 30;

    // Create a signal where amplitude matters:
    // - Region A (0-199): small sine wave (amplitude 1)
    // - Region B (200-399): large sine wave (amplitude 5), same frequency
    // - Region C (400-599): small sine wave (amplitude 1), same frequency
    let n = 600;
    let mut ts = Vec::with_capacity(n);
    for i in 0..n {
        let t = i as f64;
        let amplitude = if (200..400).contains(&i) { 5.0 } else { 1.0 };
        ts.push(amplitude * (t * std::f64::consts::TAU / 50.0).sin());
    }

    let config = MatrixProfileConfig::new(m);

    // Z-normalized: ignores amplitude — A, B, C all look the same
    let engine_norm = EuclideanEngine::new(config.clone());
    let mp_norm = engine_norm.compute(&ts);

    // AAMP: respects amplitude — A≈C but B is different
    let engine_aamp = AampEngine::new(config);
    let mp_aamp = engine_aamp.compute(&ts);

    println!("AAMP vs Z-Normalized Euclidean");
    println!("===============================");
    println!("Time series: sine wave with amplitude change at indices 200-400");
    println!("Subsequence length: {m}\n");

    // Compare profiles at key positions
    let positions = [50, 100, 250, 350, 500];
    println!(
        "{:>6}  {:>12}  {:>12}  {:>12}  {:>12}",
        "Index", "Z-Norm Dist", "Z-Norm NN", "AAMP Dist", "AAMP NN"
    );
    println!("{:-<66}", "");

    for &pos in &positions {
        if pos < mp_norm.profile.len() && pos < mp_aamp.profile.len() {
            println!(
                "{pos:>6}  {dist_n:>12.4}  {nn_n:>12}  {dist_a:>12.4}  {nn_a:>12}",
                dist_n = mp_norm.profile[pos],
                nn_n = mp_norm.profile_index[pos],
                dist_a = mp_aamp.profile[pos],
                nn_a = mp_aamp.profile_index[pos],
            );
        }
    }

    println!("\nKey insight:");
    println!("  Z-normalized: subsequences at 50 and 250 have SMALL distance");
    println!("    (same shape, amplitude is normalized away)");
    println!("  AAMP: subsequences at 50 and 250 have LARGE distance");
    println!("    (different amplitudes: 1.0 vs 5.0)");

    // Show that z-norm finds region A similar to region B
    let znorm_50_dist = mp_norm.profile[50];
    let znorm_50_nn = mp_norm.profile_index[50];

    // Show that AAMP finds region A similar to region C (not B)
    let aamp_50_dist = mp_aamp.profile[50];
    let aamp_50_nn = mp_aamp.profile_index[50];

    println!("\nSubsequence at index 50 (amplitude=1):");
    println!("  Z-normalized NN: index {znorm_50_nn} (dist={znorm_50_dist:.4})");
    println!("  AAMP NN:         index {aamp_50_nn} (dist={aamp_50_dist:.4})");

    let aamp_nn_region = if aamp_50_nn >= 400 {
        "Region C (amplitude=1) ✓"
    } else if aamp_50_nn >= 200 {
        "Region B (amplitude=5)"
    } else {
        "Region A (amplitude=1) ✓"
    };
    println!("  AAMP matched → {aamp_nn_region}");
}
