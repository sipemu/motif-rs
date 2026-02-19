//! Top-k Nearest Neighbors Matrix Profile.
//!
//! Instead of storing only the single nearest neighbor per subsequence,
//! the top-k matrix profile retains the k closest matches. This is
//! useful for finding repeated patterns that occur more than twice.
//!
//! Run with: cargo run --release --example topk

use motif_rs::{EuclideanEngine, MatrixProfileConfig};

fn main() {
    let m = 30;
    let k = 5;

    // Create a signal with a pattern that repeats ~6 times
    let mut ts = Vec::with_capacity(600);
    for i in 0..600 {
        let t = i as f64;
        // Noise + Gaussian pulse every 100 points
        let mut val = ((t * 7.1).sin() * (t * 11.3).cos()) * 0.05;
        for rep in 0..6 {
            let center = 50.0 + rep as f64 * 100.0;
            val += 2.0 * (-(t - center).powi(2) / 30.0).exp();
        }
        ts.push(val);
    }

    let engine = EuclideanEngine::new(MatrixProfileConfig::new(m));

    // Standard matrix profile: only 1-nearest neighbor
    let mp = engine.compute(&ts);

    // Top-k: retains k nearest neighbors per subsequence
    let topk = engine.compute_topk(&ts, k);

    println!("Top-k Nearest Neighbors (k={k})");
    println!("================================");
    println!("Time series length: {}", ts.len());
    println!("Subsequence length: {m}");
    println!("Profile length: {}", topk.distances.len());

    // Find the subsequence with the smallest 1-NN distance (best motif)
    let (best_idx, _) = mp
        .profile
        .iter()
        .enumerate()
        .filter(|(_, d)| d.is_finite())
        .min_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap();

    println!("\nBest motif at index {best_idx}:");
    println!("  {:-<50}", "");
    for j in 0..k {
        println!(
            "  {}-NN: index {:>4}, distance {:.6}",
            j + 1,
            topk.indices[best_idx][j],
            topk.distances[best_idx][j]
        );
    }

    // Show how many neighbors are within 2x of the 1-NN distance
    let threshold = topk.distances[best_idx][0] * 2.0;
    let close_count = topk.distances[best_idx]
        .iter()
        .filter(|d| **d <= threshold)
        .count();
    println!("\n  {close_count} of {k} neighbors within 2x of best distance");
    println!("  (pattern repeats ~6 times, so expect ~5 close neighbors)");
}
