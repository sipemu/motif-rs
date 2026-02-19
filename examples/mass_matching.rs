//! MASS and Pattern Matching.
//!
//! MASS (Mueen's Algorithm for Similarity Search) computes the
//! z-normalized distance between a query subsequence and every
//! position in a time series. find_matches() then extracts all
//! occurrences below a distance threshold.
//!
//! Run with: cargo run --release --example mass_matching

use motif_rs::{find_matches, mass};

fn main() {
    // Create a signal with a distinctive pattern embedded 4 times
    let n = 1000;
    let mut ts = vec![0.0; n];

    // Base: low-amplitude noise
    for (i, val) in ts.iter_mut().enumerate() {
        let t = i as f64;
        *val = ((t * 7.1).sin() * (t * 11.3).cos()) * 0.1;
    }

    // Embed a Gaussian pulse at 4 positions
    let pulse_centers = [100, 350, 600, 850];
    for &center in &pulse_centers {
        for j in 0..50 {
            let offset = j as f64 - 25.0;
            ts[center + j] += 2.0 * (-offset * offset / 30.0).exp();
        }
    }

    // Use the first occurrence as the query
    let query = &ts[100..150];

    // MASS: compute distance profile
    let distances = mass(query, &ts);

    println!("MASS and Pattern Matching");
    println!("=========================");
    println!("Time series length: {n}");
    println!("Query length: {}", query.len());
    println!("Distance profile length: {}", distances.len());

    // Find the top-4 matches manually from the distance profile
    let mut indexed: Vec<(usize, f64)> = distances
        .iter()
        .enumerate()
        .filter(|(_, d)| d.is_finite())
        .map(|(i, &d)| (i, d))
        .collect();
    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

    println!("\nTop matches from MASS distance profile:");
    for (rank, &(idx, dist)) in indexed.iter().take(6).enumerate() {
        println!("  #{}: index {idx:>4}, distance {dist:.6}", rank + 1);
    }

    // find_matches: automatic threshold + exclusion zone
    let matches = find_matches(query, &ts, None, None);

    println!("\nfind_matches (auto threshold, default exclusion zone):");
    println!("  Found {} matches", matches.len());
    for m in &matches {
        let near_pulse = pulse_centers
            .iter()
            .any(|&c| (m.index as isize - c as isize).unsigned_abs() < 30);
        let label = if near_pulse { "pulse" } else { "other" };
        println!(
            "    index {:>4}, distance {:.6} ({label})",
            m.index, m.distance
        );
    }

    // With explicit threshold
    let strict_matches = find_matches(query, &ts, Some(0.5), None);
    println!(
        "\nfind_matches (threshold=0.5): {} matches",
        strict_matches.len()
    );
}
