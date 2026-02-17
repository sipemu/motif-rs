//! Streaming Matrix Profile with STAMPI.
//!
//! This example mirrors stumpy's streaming tutorial.
//! STAMPI incrementally updates the matrix profile as new data arrives,
//! without recomputing from scratch. This enables real-time monitoring
//! of time series data streams.
//!
//! Run with: cargo run --release --example streaming

use motif_rs::{EuclideanEngine, MatrixProfileConfig};

fn main() {
    let m = 20;

    // Start with an initial batch of data (normal periodic signal)
    let n_initial = 200;
    let mut ts: Vec<f64> = (0..n_initial)
        .map(|i| (i as f64 * std::f64::consts::TAU / 40.0).sin())
        .collect();

    // Create the streaming engine
    let engine = EuclideanEngine::new(MatrixProfileConfig::new(m));
    let mut streamer = engine.streaming(&ts, false);

    let initial_mp = streamer.profile();
    let initial_min = initial_mp
        .profile
        .iter()
        .copied()
        .filter(|d| d.is_finite())
        .fold(f64::INFINITY, f64::min);

    println!("Streaming Matrix Profile (STAMPI)");
    println!("==================================");
    println!("Subsequence length: {m}");
    println!("Initial series length: {n_initial}");
    println!("Initial profile length: {}", initial_mp.profile.len());
    println!("Initial min distance: {initial_min:.6}");

    // Stream in 100 more normal points
    println!("\n--- Streaming 100 normal points ---");
    for i in 0..100 {
        let t = (n_initial + i) as f64;
        let val = (t * std::f64::consts::TAU / 40.0).sin();
        streamer.update(val);
        ts.push(val);
    }

    let mp_after_normal = streamer.profile();
    let min_after_normal = mp_after_normal
        .profile
        .iter()
        .copied()
        .filter(|d| d.is_finite())
        .fold(f64::INFINITY, f64::min);
    let max_after_normal = mp_after_normal
        .profile
        .iter()
        .copied()
        .filter(|d| d.is_finite())
        .fold(f64::NEG_INFINITY, f64::max);

    println!("Profile length: {}", mp_after_normal.profile.len());
    println!("Min distance: {min_after_normal:.6} (normal — motifs still present)");
    println!("Max distance: {max_after_normal:.6}");

    // Stream in an anomaly: sudden amplitude change
    println!("\n--- Streaming 50 anomalous points (3x amplitude) ---");
    for i in 0..50 {
        let t = (n_initial + 100 + i) as f64;
        let val = (t * std::f64::consts::TAU / 40.0).sin() * 3.0; // 3x amplitude
        streamer.update(val);
        ts.push(val);
    }

    let mp_after_anomaly = streamer.profile();
    let max_after_anomaly = mp_after_anomaly
        .profile
        .iter()
        .copied()
        .filter(|d| d.is_finite())
        .fold(f64::NEG_INFINITY, f64::max);

    // Find the discord (most anomalous point)
    let discords = motif_rs::find_discords(mp_after_anomaly, 1);
    if let Some(discord) = discords.first() {
        println!("Profile length: {}", mp_after_anomaly.profile.len());
        println!("Max distance: {max_after_anomaly:.6} (increased due to anomaly)");
        println!(
            "Top discord at index {} (distance {:.4})",
            discord.idx, discord.distance
        );
        let expected_region = n_initial + 100 - m..n_initial + 100 + 50;
        if expected_region.contains(&discord.idx) {
            println!("  → Correctly detected in the anomalous region!");
        }
    }

    // Stream in normal data again to show recovery
    println!("\n--- Streaming 100 more normal points ---");
    for i in 0..100 {
        let t = (n_initial + 150 + i) as f64;
        let val = (t * std::f64::consts::TAU / 40.0).sin();
        streamer.update(val);
        ts.push(val);
    }

    let final_mp = streamer.profile();
    println!("Final profile length: {}", final_mp.profile.len());
    println!("Total points processed: {}", ts.len());

    // Verify streaming result matches batch computation
    let batch_mp = engine.compute(&ts);
    let mut max_diff = 0.0_f64;
    for (s, b) in final_mp.profile.iter().zip(&batch_mp.profile) {
        if s.is_finite() && b.is_finite() {
            max_diff = max_diff.max((s - b).abs());
        }
    }
    println!("\nStreaming vs batch max difference: {max_diff:.2e} (should be ~0)");
}
