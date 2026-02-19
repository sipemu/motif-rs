//! STIMP: Pan Matrix Profile Across Window Sizes.
//!
//! STIMP computes matrix profiles across a range of window sizes,
//! producing a "pan matrix profile" — a 2D view that reveals which
//! window size best captures the dominant patterns. Profiles are
//! normalized by 1/sqrt(2m) for cross-window comparability.
//!
//! Run with: cargo run --release --example stimp

use motif_rs::{stimp, ZNormalizedEuclidean};

fn main() {
    let ts: Vec<f64> = (0..500)
        .map(|i| {
            let t = i as f64;
            // Multi-scale signal: period-40 sine + period-100 sine
            (t * std::f64::consts::TAU / 40.0).sin()
                + 0.5 * (t * std::f64::consts::TAU / 100.0).sin()
                + ((t * 7.1).sin() * (t * 11.3).cos()) * 0.03
        })
        .collect();

    let min_m = 10;
    let max_m = 80;
    let step = 5;

    let pan = stimp::<ZNormalizedEuclidean>(&ts, min_m, max_m, Some(step), None);

    println!("STIMP: Pan Matrix Profile");
    println!("=========================");
    println!("Time series length: {}", ts.len());
    println!("Window range: {min_m}..{max_m} (step {step})");
    println!("Number of profiles: {}\n", pan.windows.len());

    // For each window size, find the minimum normalized distance
    println!("  {:>5}  {:>12}  {:>8}", "m", "min(norm_dist)", "motif_idx");
    println!("  {:-<30}", "");

    let mut best_m = 0;
    let mut best_dist = f64::INFINITY;

    for (i, &m) in pan.windows.iter().enumerate() {
        let (min_idx, min_dist) = pan.profiles[i]
            .iter()
            .enumerate()
            .filter(|(_, d)| d.is_finite())
            .min_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap_or((0, &f64::INFINITY));

        if *min_dist < best_dist {
            best_dist = *min_dist;
            best_m = m;
        }

        println!("  {m:>5}  {min_dist:>12.6}  {min_idx:>8}");
    }

    println!("\n  Best window size: m={best_m} (lowest normalized distance = {best_dist:.6})");
}
