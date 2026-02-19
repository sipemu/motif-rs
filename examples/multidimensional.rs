//! Multi-Dimensional Matrix Profile (MSTUMP).
//!
//! MSTUMP computes matrix profiles across multiple co-evolving time
//! series (e.g., x/y/z accelerometer axes). It also supports subspace
//! selection (which dimensions matter for a motif), MDL-based optimal
//! dimensionality, and multi-dimensional motif discovery.
//!
//! Run with: cargo run --release --example multidimensional

use motif_rs::{mdl, mmotifs, mstump, subspace};

fn main() {
    let m = 20;
    let n = 300;

    // Three co-evolving time series with a shared pattern in dims 0 and 1
    // Dim 0: sine with embedded pulse
    let ts0: Vec<f64> = (0..n)
        .map(|i| {
            let t = i as f64;
            let base = (t * std::f64::consts::TAU / 50.0).sin();
            // Pulse pattern at indices 50 and 200
            let p1 = 2.0 * (-(t - 50.0).powi(2) / 20.0).exp();
            let p2 = 2.0 * (-(t - 200.0).powi(2) / 20.0).exp();
            base + p1 + p2
        })
        .collect();

    // Dim 1: cosine with same pulse pattern (co-evolving)
    let ts1: Vec<f64> = (0..n)
        .map(|i| {
            let t = i as f64;
            let base = (t * std::f64::consts::TAU / 50.0).cos();
            let p1 = 1.5 * (-(t - 50.0).powi(2) / 20.0).exp();
            let p2 = 1.5 * (-(t - 200.0).powi(2) / 20.0).exp();
            base + p1 + p2
        })
        .collect();

    // Dim 2: independent noise (not part of the motif)
    let ts2: Vec<f64> = (0..n)
        .map(|i| {
            let t = i as f64;
            (t * 7.1).sin() * (t * 11.3).cos() * 0.5
        })
        .collect();

    let ts_refs: Vec<&[f64]> = vec![&ts0, &ts1, &ts2];

    // Step 1: Compute multi-dimensional matrix profile
    let profile = mstump(&ts_refs, m);
    let n_subs = n - m + 1;

    println!("Multi-Dimensional Matrix Profile (MSTUMP)");
    println!("==========================================");
    println!("Dimensions: {}", ts_refs.len());
    println!("Time series length: {n}");
    println!("Subsequence length: {m}");
    println!("Profile length: {n_subs}\n");

    // Show best match for each dimensionality
    println!("Best match by dimensionality:");
    for k in 0..ts_refs.len() {
        let (best_idx, best_dist) = profile.profile[k]
            .iter()
            .enumerate()
            .filter(|(_, d)| d.is_finite())
            .min_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap();
        let nn = profile.profile_index[k][best_idx];
        println!(
            "  k={} dims: index {best_idx} ↔ {nn}, avg distance = {best_dist:.6}",
            k + 1
        );
    }

    // Step 2: Subspace selection — which dimensions matter?
    let idx = 40; // near the first pulse
    let nn = profile.profile_index[0][idx];
    println!("\nSubspace selection for motif at index {idx} (NN={nn}):");
    for num_dims in 1..=ts_refs.len() {
        let dims = subspace(&ts_refs, m, idx, nn, num_dims);
        println!("  Best {num_dims} dimension(s): {dims:?}");
    }

    // Step 3: MDL — find optimal number of dimensions
    let d = ts_refs.len();
    let subseq_idx: Vec<usize> = vec![idx; d];
    let nn_idx: Vec<usize> = (0..d).map(|k| profile.profile_index[k][idx]).collect();
    let (bit_sizes, subspaces) = mdl(&ts_refs, m, &subseq_idx, &nn_idx);

    println!("\nMDL (Minimum Description Length):");
    for (k, bits) in bit_sizes.iter().enumerate() {
        println!("  k={}: {bits:.0} bits, dims={:?}", k + 1, &subspaces[k]);
    }
    let optimal_k = bit_sizes
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i + 1)
        .unwrap();
    println!("  Optimal dimensionality: {optimal_k}");

    // Step 4: Multi-dimensional motif discovery
    let motifs = mmotifs(&ts_refs, m, &profile, 3);

    println!("\nMulti-dimensional motifs:");
    for (i, motif) in motifs.iter().enumerate() {
        println!(
            "  Motif #{}: ({}, {}), k={}, dims={:?}, distance={:.6}",
            i + 1,
            motif.idx,
            motif.nn_idx,
            motif.k,
            motif.dimensions,
            motif.distance
        );
    }
}
