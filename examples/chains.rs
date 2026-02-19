//! Time Series Chains: Discovering Evolving Patterns.
//!
//! A time series chain is a sequence of subsequences linked by
//! bidirectional nearest-neighbor relationships. Chains reveal
//! patterns that gradually evolve over time (e.g., a machine
//! slowly degrading).
//!
//! Run with: cargo run --release --example chains

use motif_rs::{allc, atsc, EuclideanEngine, MatrixProfileConfig};

fn main() {
    let m = 20;

    // Create a signal with a slowly evolving pattern:
    // A sine wave whose frequency gradually increases over time
    let n = 500;
    let ts: Vec<f64> = (0..n)
        .map(|i| {
            let t = i as f64;
            // Frequency increases linearly from period=80 to period=40
            let freq = 1.0 / (80.0 - t * 40.0 / n as f64);
            (t * std::f64::consts::TAU * freq).sin() + ((t * 7.1).sin() * (t * 11.3).cos()) * 0.02
        })
        .collect();

    let engine = EuclideanEngine::new(MatrixProfileConfig::new(m));
    let mp = engine.compute(&ts);

    // Discover all chains
    let result = allc(&mp);

    println!("Time Series Chains");
    println!("===================");
    println!("Time series length: {n}");
    println!("Subsequence length: {m}");
    println!("Total chains found: {}", result.chains.len());
    println!(
        "Longest chain length: {} links",
        result.longest.indices.len()
    );

    // Show the longest chain
    println!("\nLongest chain (evolving pattern):");
    println!("  Indices: {:?}", result.longest.indices);

    if result.longest.indices.len() >= 2 {
        // Show distances between consecutive chain links
        println!("\n  Consecutive distances:");
        for w in result.longest.indices.windows(2) {
            let dist = mp.profile[w[0]];
            println!("    {} → {}: distance = {:.4}", w[0], w[1], dist);
        }
    }

    // Trace a chain from a specific anchor
    let anchor = 0;
    let chain = atsc(&mp, anchor);
    println!("\nChain anchored at index {anchor}:");
    println!("  Length: {} links", chain.indices.len());
    println!("  Indices: {:?}", chain.indices);
}
