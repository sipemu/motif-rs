//! Basic Matrix Profile computation with motif-rs.
//!
//! This example mirrors stumpy's "The Matrix Profile" tutorial.
//! It computes the z-normalized Euclidean distance matrix profile
//! using the STOMP algorithm.
//!
//! Run with: cargo run --release --example basic_matrix_profile

use motif_rs::{EuclideanEngine, MatrixProfileConfig};

fn main() {
    // Generate a synthetic time series: two copies of a sine pattern
    // separated by noise, so the matrix profile reveals the repeated motif.
    let n = 500;
    let m = 50; // subsequence length

    let mut ts = Vec::with_capacity(n);
    for i in 0..n {
        let t = i as f64;
        // Base signal: sine wave with period 100
        let base = (t * std::f64::consts::TAU / 100.0).sin();
        // Add a small amount of noise
        let noise = ((t * 7.3).sin() * (t * 13.7).cos()) * 0.05;
        ts.push(base + noise);
    }

    // Compute the matrix profile
    let engine = EuclideanEngine::new(MatrixProfileConfig::new(m));
    let mp = engine.compute(&ts);

    // The matrix profile has n - m + 1 entries
    println!("Time series length: {n}");
    println!("Subsequence length: {m}");
    println!("Matrix profile length: {}", mp.profile.len());

    // Find the minimum distance (best motif pair)
    let (min_idx, min_dist) = mp
        .profile
        .iter()
        .enumerate()
        .min_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap();
    let nn_idx = mp.profile_index[min_idx];

    println!("\nBest matching pair:");
    println!("  Subsequence at index {min_idx}");
    println!("  Nearest neighbor at index {nn_idx}");
    println!("  Distance: {min_dist:.6}");

    // Find the maximum distance (most anomalous subsequence)
    let (max_idx, max_dist) = mp
        .profile
        .iter()
        .enumerate()
        .filter(|(_, d)| d.is_finite())
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap();

    println!("\nMost anomalous subsequence:");
    println!("  Index: {max_idx}");
    println!("  Distance: {max_dist:.6}");

    // Print a summary of the profile distribution
    let finite_dists: Vec<f64> = mp
        .profile
        .iter()
        .copied()
        .filter(|d| d.is_finite())
        .collect();
    let mean = finite_dists.iter().sum::<f64>() / finite_dists.len() as f64;
    let std = (finite_dists.iter().map(|d| (d - mean).powi(2)).sum::<f64>()
        / finite_dists.len() as f64)
        .sqrt();

    println!("\nProfile statistics:");
    println!("  Mean distance: {mean:.6}");
    println!("  Std deviation: {std:.6}");
    println!("  Min distance:  {min_dist:.6}");
    println!("  Max distance:  {max_dist:.6}");
}
