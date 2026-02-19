//! Snippets: Time Series Summarization.
//!
//! Snippets extract k representative subsequences that together best
//! cover (summarize) the entire time series. Each position in the series
//! is assigned to its closest snippet, and the fraction of the series
//! covered by each snippet is reported.
//!
//! Run with: cargo run --release --example snippets

use motif_rs::find_snippets;

fn main() {
    let m = 50;
    let k = 3;

    // Create a signal with three distinct regimes:
    // - Region 1 (0-299): slow sine
    // - Region 2 (300-599): fast sine
    // - Region 3 (600-899): sawtooth
    let n = 900;
    let mut ts = Vec::with_capacity(n);
    for i in 0..n {
        let t = i as f64;
        let val = if i < 300 {
            (t * std::f64::consts::TAU / 100.0).sin()
        } else if i < 600 {
            (t * std::f64::consts::TAU / 25.0).sin()
        } else {
            let phase = (i - 600) % 30;
            phase as f64 / 30.0 * 2.0 - 1.0
        };
        let noise = ((t * 7.1).sin() * (t * 11.3).cos()) * 0.02;
        ts.push(val + noise);
    }

    let result = find_snippets(&ts, m, k);

    println!("Snippets: Time Series Summarization");
    println!("====================================");
    println!("Time series length: {n}");
    println!("Subsequence length: {m}");
    println!("Number of snippets: {k}\n");

    for (i, &idx) in result.indices.iter().enumerate() {
        let fraction_pct = result.fractions[i] * 100.0;
        let region = if idx < 300 {
            "slow sine"
        } else if idx < 600 {
            "fast sine"
        } else {
            "sawtooth"
        };
        println!(
            "  Snippet {}: starts at index {idx:>4}, covers {fraction_pct:>5.1}% ({region})",
            i + 1
        );
    }

    // Verify total coverage is ~100%
    let total: f64 = result.fractions.iter().sum();
    println!("\n  Total coverage: {:.1}%", total * 100.0);

    // Show regime assignments around boundaries
    println!("\nRegime assignments (which snippet each position is closest to):");
    for &pos in &[0, 150, 299, 300, 450, 599, 600, 750] {
        if pos < result.regimes.len() {
            println!("  Position {pos:>4} → snippet {}", result.regimes[pos]);
        }
    }
}
