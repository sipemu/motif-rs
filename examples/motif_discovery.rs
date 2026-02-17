//! Motif Discovery: finding repeated patterns in time series.
//!
//! This example mirrors stumpy's "Motif Discovery" tutorial.
//! A motif is a pair of subsequences that are very similar to each other.
//! After computing the matrix profile, we extract the top-k motifs.
//!
//! Run with: cargo run --release --example motif_discovery

use motif_rs::{EuclideanEngine, MatrixProfileConfig};

fn main() {
    // Create a signal with 3 embedded motifs (repeated patterns):
    // - Pattern A (sharp pulse) appears at indices ~50 and ~300
    // - Pattern B (double bump) appears at indices ~150 and ~400
    // - Background is smooth sine wave + noise
    let n = 500;
    let m = 30;

    let mut ts = Vec::with_capacity(n);
    for i in 0..n {
        let t = i as f64;
        let mut val = (t * std::f64::consts::TAU / 200.0).sin() * 0.3;

        // Pattern A: sharp pulse
        for &center in &[50.0, 300.0] {
            let d = (t - center).abs();
            if d < 15.0 {
                val += 2.0 * (-d * d / 20.0).exp();
            }
        }

        // Pattern B: double bump
        for &center in &[150.0, 400.0] {
            let d1 = (t - (center - 5.0)).abs();
            let d2 = (t - (center + 5.0)).abs();
            if d1 < 15.0 || d2 < 15.0 {
                val += 1.5 * (-d1 * d1 / 10.0).exp() + 1.5 * (-d2 * d2 / 10.0).exp();
            }
        }

        // Small noise
        val += ((t * 7.1).sin() * (t * 11.3).cos()) * 0.02;
        ts.push(val);
    }

    // Compute matrix profile
    let engine = EuclideanEngine::new(MatrixProfileConfig::new(m));
    let mp = engine.compute(&ts);

    // Extract top-3 motifs
    let motifs = motif_rs::find_motifs(&mp, 3);

    println!("Motif Discovery");
    println!("===============");
    println!("Time series length: {n}");
    println!("Subsequence length: {m}");
    println!("Found {} motifs:\n", motifs.len());

    for (i, motif) in motifs.iter().enumerate() {
        println!(
            "  Motif #{}: indices ({}, {}), distance = {:.6}",
            i + 1,
            motif.idx_a,
            motif.idx_b,
            motif.distance
        );
    }

    // Also demonstrate top-k nearest neighbors
    println!("\n\nTop-k Matrix Profile (k=3)");
    println!("==========================");
    let topk = engine.compute_topk(&ts, 3);

    // Show the top-k distances for the first motif's position
    if let Some(motif) = motifs.first() {
        let idx = motif.idx_a;
        println!("Distances for subsequence at index {idx}:");
        for (j, d) in topk.distances[idx].iter().enumerate() {
            println!(
                "  {}-nearest neighbor: distance = {:.6}, index = {}",
                j + 1,
                d,
                topk.indices[idx][j]
            );
        }
    }
}
