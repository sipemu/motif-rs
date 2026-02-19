//! SCRUMP: Approximate Matrix Profile via Diagonal Sampling.
//!
//! SCRUMP (PreSCRIMP) computes an approximate matrix profile by sampling
//! a fraction of diagonals. This gives a speed/accuracy tradeoff: lower
//! percentages are faster but less accurate. At 100%, it produces the
//! exact result.
//!
//! Run with: cargo run --release --example scrump

use motif_rs::{EuclideanEngine, MatrixProfileConfig};

fn main() {
    let n = 5000;
    let m = 50;

    let ts: Vec<f64> = (0..n)
        .map(|i| {
            let t = i as f64;
            (t * std::f64::consts::TAU / 80.0).sin() + ((t * 7.1).sin() * (t * 11.3).cos()) * 0.05
        })
        .collect();

    let engine = EuclideanEngine::new(MatrixProfileConfig::new(m));

    // Compute exact profile for reference
    let exact = engine.compute(&ts);

    println!("SCRUMP: Approximate Matrix Profile");
    println!("===================================");
    println!("Time series length: {n}");
    println!("Subsequence length: {m}\n");

    // Try different sampling percentages
    for &pct in &[0.01, 0.05, 0.10, 0.25, 0.50, 1.0] {
        let approx = engine.scrump(&ts, pct);

        // Compute error vs exact
        let mut sum_err = 0.0;
        let mut max_err = 0.0_f64;
        let mut count = 0;
        for (a, e) in approx.profile.iter().zip(&exact.profile) {
            if a.is_finite() && e.is_finite() {
                let err = (a - e).abs();
                sum_err += err;
                max_err = max_err.max(err);
                count += 1;
            }
        }
        let mean_err = if count > 0 {
            sum_err / count as f64
        } else {
            0.0
        };

        println!(
            "  {pct:>4.0}% diagonals: mean_err={mean_err:.6}, max_err={max_err:.6}",
            pct = pct * 100.0,
        );
    }

    println!("\n  At 100%, SCRUMP delegates to exact STOMP (error = 0).");
}
