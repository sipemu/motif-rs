//! MPdist: Scalar Distance Between Time Series.
//!
//! MPdist provides a single distance value between two time series based
//! on the matrix profile. Unlike DTW or Euclidean distance, it is robust
//! to partial matches and length differences — two series are "close" if
//! they share common subsequence patterns.
//!
//! Run with: cargo run --release --example mpdist

use motif_rs::ZNormalizedEuclidean;

fn main() {
    let m = 30;

    // Series A: sine wave
    let ts_a: Vec<f64> = (0..400)
        .map(|i| (i as f64 * std::f64::consts::TAU / 50.0).sin())
        .collect();

    // Series B: same sine wave (should be close to A)
    let ts_b: Vec<f64> = (0..500)
        .map(|i| (i as f64 * std::f64::consts::TAU / 50.0).sin())
        .collect();

    // Series C: different frequency (should be far from A)
    let ts_c: Vec<f64> = (0..400)
        .map(|i| (i as f64 * std::f64::consts::TAU / 15.0).sin())
        .collect();

    // Series D: sawtooth wave (should be far from A)
    let ts_d: Vec<f64> = (0..400)
        .map(|i| {
            let phase = i % 50;
            phase as f64 / 50.0 * 2.0 - 1.0
        })
        .collect();

    let dist_ab = motif_rs::mpdist::<ZNormalizedEuclidean>(&ts_a, &ts_b, m, None);
    let dist_ac = motif_rs::mpdist::<ZNormalizedEuclidean>(&ts_a, &ts_c, m, None);
    let dist_ad = motif_rs::mpdist::<ZNormalizedEuclidean>(&ts_a, &ts_d, m, None);

    println!("MPdist: Time Series Distance");
    println!("============================");
    println!("Subsequence length: {m}\n");
    println!("  A: sine (period=50, n=400)");
    println!("  B: sine (period=50, n=500) — same pattern, different length");
    println!("  C: sine (period=15, n=400) — different frequency");
    println!("  D: sawtooth (period=50, n=400) — different shape\n");

    println!("  MPdist(A, B) = {dist_ab:.6}  (same pattern → small)");
    println!("  MPdist(A, C) = {dist_ac:.6}  (different freq → larger)");
    println!("  MPdist(A, D) = {dist_ad:.6}  (different shape → larger)");

    println!("\n  A≈B: {}", if dist_ab < dist_ac { "yes" } else { "no" });
    println!("  A≈C: {}", if dist_ac < dist_ab { "yes" } else { "no" });
}
