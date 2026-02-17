//! AB-Join: comparing two time series.
//!
//! This example mirrors stumpy's AB-join tutorial.
//! The AB-join finds, for each subsequence in series A, its nearest
//! neighbor in series B (and vice versa). This is useful for finding
//! shared patterns between different signals.
//!
//! Run with: cargo run --release --example ab_join

use motif_rs::{EuclideanEngine, MatrixProfileConfig};

fn main() {
    let m = 30;

    // Series A: sine wave with two embedded Gaussian pulses
    let n_a = 400;
    let mut ts_a = Vec::with_capacity(n_a);
    for i in 0..n_a {
        let t = i as f64;
        let mut val = (t * std::f64::consts::TAU / 80.0).sin() * 0.5;
        // Gaussian pulse at index 100 and 300
        for &center in &[100.0, 300.0] {
            val += 2.0 * (-(t - center).powi(2) / 50.0).exp();
        }
        ts_a.push(val);
    }

    // Series B: different base signal with ONE matching Gaussian pulse
    let n_b = 300;
    let mut ts_b = Vec::with_capacity(n_b);
    for i in 0..n_b {
        let t = i as f64;
        let mut val = (t * std::f64::consts::TAU / 60.0).cos() * 0.5;
        // Matching Gaussian pulse at index 150
        val += 2.0 * (-(t - 150.0).powi(2) / 50.0).exp();
        ts_b.push(val);
    }

    // Compute AB-join
    let engine = EuclideanEngine::new(MatrixProfileConfig::new(m));
    let (join_a, join_b) = engine.ab_join(&ts_a, &ts_b);

    println!("AB-Join: Comparing Two Time Series");
    println!("===================================");
    println!("Series A length: {n_a}");
    println!("Series B length: {n_b}");
    println!("Subsequence length: {m}");
    println!(
        "Profile A length: {} (each A subsequence → best match in B)",
        join_a.distances.len()
    );
    println!(
        "Profile B length: {} (each B subsequence → best match in A)",
        join_b.distances.len()
    );

    // Best match: A subsequence most similar to something in B
    let (best_a_idx, best_a_dist) = join_a
        .distances
        .iter()
        .enumerate()
        .filter(|(_, d)| d.is_finite())
        .min_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap();
    let best_a_nn = join_a.indices[best_a_idx];

    println!("\nBest A→B match:");
    println!("  A subsequence at index {best_a_idx} ≈ B subsequence at index {best_a_nn}");
    println!("  Distance: {best_a_dist:.6}");

    // Best match: B subsequence most similar to something in A
    let (best_b_idx, best_b_dist) = join_b
        .distances
        .iter()
        .enumerate()
        .filter(|(_, d)| d.is_finite())
        .min_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap();
    let best_b_nn = join_b.indices[best_b_idx];

    println!("\nBest B→A match:");
    println!("  B subsequence at index {best_b_idx} ≈ A subsequence at index {best_b_nn}");
    println!("  Distance: {best_b_dist:.6}");

    // Show the most different patterns
    let (worst_a_idx, worst_a_dist) = join_a
        .distances
        .iter()
        .enumerate()
        .filter(|(_, d)| d.is_finite())
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap();

    println!("\nMost different A subsequence (no match in B):");
    println!("  A index {worst_a_idx}, distance: {worst_a_dist:.6}");
}
