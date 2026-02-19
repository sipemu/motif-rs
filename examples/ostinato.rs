//! Ostinato: Consensus Motif Discovery Across Multiple Time Series.
//!
//! Ostinato finds the subsequence that appears most consistently across
//! all input time series — the "consensus motif". It minimizes the maximum
//! nearest-neighbor distance across all series, finding the pattern that
//! every series has in common.
//!
//! Run with: cargo run --release --example ostinato

use motif_rs::{ostinato, ZNormalizedEuclidean};

fn main() {
    let m = 25;

    // Three sensor signals, each with a common embedded pulse pattern
    // plus different baseline behaviors
    let n = 300;

    // Series 1: slow sine baseline + pulse
    let ts1: Vec<f64> = (0..n)
        .map(|i| {
            let t = i as f64;
            let base = (t * std::f64::consts::TAU / 100.0).sin();
            let pulse = 2.0 * (-(t - 150.0).powi(2) / 40.0).exp();
            base + pulse
        })
        .collect();

    // Series 2: fast sine baseline + same pulse shape at different position
    let ts2: Vec<f64> = (0..n)
        .map(|i| {
            let t = i as f64;
            let base = (t * std::f64::consts::TAU / 30.0).sin() * 0.5;
            let pulse = 2.0 * (-(t - 100.0).powi(2) / 40.0).exp();
            base + pulse
        })
        .collect();

    // Series 3: linear trend + same pulse shape
    let ts3: Vec<f64> = (0..n)
        .map(|i| {
            let t = i as f64;
            let base = t / n as f64 - 0.5;
            let pulse = 2.0 * (-(t - 200.0).powi(2) / 40.0).exp();
            base + pulse
        })
        .collect();

    let ts_list: Vec<&[f64]> = vec![&ts1, &ts2, &ts3];
    let result = ostinato::<ZNormalizedEuclidean>(&ts_list, m);

    println!("Ostinato: Consensus Motif");
    println!("=========================");
    println!("Number of time series: {}", ts_list.len());
    println!("Each series length: {n}");
    println!("Subsequence length: {m}\n");

    println!("Consensus motif found:");
    println!("  Source series:  {}", result.ts_index);
    println!("  Start index:    {}", result.subsequence_index);
    println!("  Radius:         {:.6}", result.radius);
    println!("  (radius = max distance to nearest neighbor in any series)");
}
