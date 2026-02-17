//! Anomaly Detection (Discord Discovery) with the Matrix Profile.
//!
//! This example mirrors stumpy's "Anomaly Detection" tutorial.
//! A discord is a subsequence whose nearest neighbor is unusually far away,
//! making it the most unusual/anomalous pattern in the series.
//!
//! Run with: cargo run --release --example anomaly_detection

use motif_rs::{EuclideanEngine, MatrixProfileConfig};

fn main() {
    // Simulate a machine sensor signal:
    // - Normal operation: smooth periodic pattern
    // - Anomaly 1 at index ~200: sudden amplitude spike
    // - Anomaly 2 at index ~600: frequency change
    let n = 1000;
    let m = 50;

    let mut ts = Vec::with_capacity(n);
    for i in 0..n {
        let t = i as f64;
        let mut val = (t * std::f64::consts::TAU / 80.0).sin();

        // Anomaly 1: amplitude spike at index 200
        if (180..230).contains(&i) {
            val *= 3.0;
        }

        // Anomaly 2: frequency change at index 600
        if (580..650).contains(&i) {
            val = (t * std::f64::consts::TAU / 20.0).sin();
        }

        // Background noise
        val += ((t * 7.3).sin() * (t * 13.7).cos()) * 0.05;
        ts.push(val);
    }

    // Compute matrix profile
    let engine = EuclideanEngine::new(MatrixProfileConfig::new(m));
    let mp = engine.compute(&ts);

    // Find top-3 discords (anomalies)
    let discords = motif_rs::find_discords(&mp, 3);

    println!("Anomaly Detection (Discord Discovery)");
    println!("=====================================");
    println!("Time series length: {n}");
    println!("Subsequence length: {m}");
    println!("\nDetected {} anomalies:\n", discords.len());

    for (i, discord) in discords.iter().enumerate() {
        println!(
            "  Anomaly #{}: index {}, distance = {:.4}",
            i + 1,
            discord.idx,
            discord.distance
        );

        // Classify the anomaly based on known injection points
        let desc = if (150..260).contains(&discord.idx) {
            "amplitude spike (injected at ~200)"
        } else if (530..680).contains(&discord.idx) {
            "frequency change (injected at ~600)"
        } else {
            "unknown"
        };
        println!("    → {desc}");
    }

    // Show the profile statistics around the anomalies
    println!("\nProfile values near anomalies:");
    for discord in &discords {
        let start = discord.idx.saturating_sub(2);
        let end = (discord.idx + 3).min(mp.profile.len());
        let neighbors: Vec<String> = (start..end)
            .map(|j| format!("{:.2}", mp.profile[j]))
            .collect();
        println!("  idx {}: [..., {}]", discord.idx, neighbors.join(", "));
    }
}
