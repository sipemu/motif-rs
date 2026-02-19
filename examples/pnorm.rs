//! P-norm: Minkowski Distance Matrix Profile.
//!
//! The p-norm generalizes distance computation beyond Euclidean (p=2):
//! - p=1: Manhattan distance (sum of absolute differences)
//! - p=2: Euclidean distance (delegates to optimized AAMP path)
//! - p=3+: higher-order norms (emphasize larger differences)
//!
//! Also supports AB-join and MPdist with arbitrary p.
//!
//! Run with: cargo run --release --example pnorm

use motif_rs::{ab_join_pnorm, mpdist_pnorm, stomp_pnorm, MatrixProfileConfig};

fn main() {
    let m = 30;
    let config = MatrixProfileConfig::new(m);

    // Signal with an outlier spike — different norms weight it differently
    let n = 500;
    let mut ts = Vec::with_capacity(n);
    for i in 0..n {
        let t = i as f64;
        let mut val = (t * std::f64::consts::TAU / 50.0).sin();
        // Large spike at index 250
        if i == 250 {
            val += 10.0;
        }
        ts.push(val);
    }

    println!("P-norm: Minkowski Distance Matrix Profile");
    println!("==========================================");
    println!("Time series length: {n}");
    println!("Subsequence length: {m}\n");

    // Compare different p values
    println!("Self-join with different p values:");
    println!(
        "  {:>5}  {:>12}  {:>12}  {:>10}",
        "p", "min_dist", "max_dist", "discord_idx"
    );
    println!("  {:-<48}", "");

    for &p in &[1.0, 2.0, 3.0, 5.0] {
        let mp = stomp_pnorm(&ts, &config, p);

        let (min_idx, min_dist) = mp
            .profile
            .iter()
            .enumerate()
            .filter(|(_, d)| d.is_finite())
            .min_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap();
        let _ = min_idx;

        let (max_idx, max_dist) = mp
            .profile
            .iter()
            .enumerate()
            .filter(|(_, d)| d.is_finite())
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap();

        println!("  {p:>5.1}  {min_dist:>12.4}  {max_dist:>12.4}  {max_idx:>10}");
    }

    println!("\n  Higher p emphasizes the spike more (larger max distance).");

    // AB-join with p-norm
    let ts_b: Vec<f64> = (0..400)
        .map(|i| (i as f64 * std::f64::consts::TAU / 50.0).sin())
        .collect();

    println!("\nAB-join (p=1.5):");
    let (join_a, join_b) = ab_join_pnorm(&ts, &ts_b, m, 1.5);

    let best_a = join_a
        .distances
        .iter()
        .filter(|d| d.is_finite())
        .cloned()
        .fold(f64::INFINITY, f64::min);
    let best_b = join_b
        .distances
        .iter()
        .filter(|d| d.is_finite())
        .cloned()
        .fold(f64::INFINITY, f64::min);
    println!("  Best A→B distance: {best_a:.6}");
    println!("  Best B→A distance: {best_b:.6}");

    // MPdist with p-norm
    println!("\nMPdist with different p values:");
    for &p in &[1.0, 2.0, 3.0] {
        let dist = mpdist_pnorm(&ts, &ts_b, m, p, None);
        println!("  MPdist(p={p:.0}) = {dist:.6}");
    }
}
