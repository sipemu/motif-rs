//! Time Series Segmentation with FLUSS.
//!
//! This example mirrors stumpy's "Semantic Segmentation" tutorial.
//! FLUSS detects regime changes by analyzing the arc structure of the
//! matrix profile index — positions where few nearest-neighbor arcs
//! cross indicate semantic boundaries.
//!
//! Run with: cargo run --release --example segmentation

use motif_rs::{EuclideanEngine, MatrixProfileConfig};

fn main() {
    // Create a signal with two distinct regimes:
    // - Regime 1 (indices 0-499): sine wave with period 40
    // - Regime 2 (indices 500-999): sawtooth wave with period 25
    let n = 1000;
    let m = 25;

    let mut ts = Vec::with_capacity(n);
    for i in 0..n {
        let val = if i < 500 {
            // Regime 1: smooth sine
            (i as f64 * std::f64::consts::TAU / 40.0).sin()
        } else {
            // Regime 2: sawtooth
            let phase = (i - 500) % 25;
            phase as f64 / 25.0 * 2.0 - 1.0
        };
        // Small noise
        let noise = ((i as f64 * 7.1).sin() * (i as f64 * 11.3).cos()) * 0.03;
        ts.push(val + noise);
    }

    // Step 1: Compute matrix profile
    println!("Time Series Segmentation with FLUSS");
    println!("====================================");
    println!("Time series length: {n}");
    println!("Subsequence length: {m}");
    println!("Expected regime change at index: 500\n");

    let engine = EuclideanEngine::new(MatrixProfileConfig::new(m));
    let mp = engine.compute(&ts);

    // Step 2: Run FLUSS to detect regime boundaries
    let num_regimes = 1;
    let seg = motif_rs::fluss(&mp, num_regimes);

    println!("FLUSS Results:");
    println!("  CAC length: {}", seg.cac.len());
    println!("  Detected boundaries: {:?}", seg.regime_boundaries);

    // Find the CAC minimum (strongest boundary signal) in the valid region
    let excl_zone = 5 * m;
    let (min_idx, min_val) = seg
        .cac
        .iter()
        .enumerate()
        .filter(|(i, _)| *i >= excl_zone && *i < seg.cac.len() - excl_zone)
        .min_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap();

    println!("\n  CAC minimum: index {min_idx}, value {min_val:.4}");

    let error = (min_idx as isize - 500).unsigned_abs();
    println!("  Distance from true boundary: {error} indices");

    // Show the CAC curve around the boundary
    println!("\nCAC values near the regime change:");
    let region_start = 450.min(seg.cac.len());
    let region_end = 550.min(seg.cac.len());
    for i in (region_start..region_end).step_by(10) {
        let bar_len = ((1.0 - seg.cac[i]) * 40.0) as usize;
        let bar: String = std::iter::repeat_n('█', bar_len).collect();
        println!("  [{i:4}] {:.3} {bar}", seg.cac[i]);
    }

    // Show edge nullification
    println!("\nEdge nullification (first/last {excl_zone} positions set to 1.0):");
    println!("  CAC[0] = {:.4}", seg.cac[0]);
    println!("  CAC[{}] = {:.4}", excl_zone - 1, seg.cac[excl_zone - 1]);
    println!("  CAC[{}] = {:.4}", excl_zone, seg.cac[excl_zone]);
}
