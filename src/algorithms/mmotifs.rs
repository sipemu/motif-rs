use crate::algorithms::common::apply_exclusion_zone;
use crate::algorithms::mdl::mdl;
use crate::algorithms::mstump::MultiDimensionalProfile;
use crate::algorithms::subspace::subspace;

/// A multi-dimensional motif discovered by `mmotifs()`.
#[derive(Debug, Clone)]
pub struct MultiDimensionalMotif {
    /// Index of the first motif occurrence.
    pub idx: usize,
    /// Index of the second motif occurrence (nearest neighbor).
    pub nn_idx: usize,
    /// The dimension indices that form the motif's subspace (most relevant first).
    pub dimensions: Vec<usize>,
    /// MDL bit sizes for each dimensionality (length d).
    pub mdl_bit_sizes: Vec<f64>,
    /// Optimal number of dimensions (argmin of MDL bit sizes + 1).
    pub k: usize,
    /// The 1D profile distance (from the best single dimension).
    pub distance: f64,
}

/// Find multi-dimensional motifs in a multi-dimensional time series.
///
/// Uses the multi-dimensional matrix profile from `mstump()` to identify motif
/// candidates, then applies MDL to determine optimal dimensionality and subspace
/// selection for each motif.
///
/// # Algorithm
/// 1. Use the 1D profile (row 0 of `profile`) to find motif candidates
/// 2. For each candidate:
///    - Get the motif pair from the profile index
///    - Run MDL to determine optimal number of dimensions
///    - Run subspace selection to identify which dimensions
/// 3. Apply exclusion zone and repeat for additional motifs
///
/// # Arguments
/// * `ts` - Multi-dimensional time series (one slice per dimension, all same length)
/// * `m` - Subsequence length
/// * `profile` - Multi-dimensional profile from `mstump()`
/// * `max_motifs` - Maximum number of motifs to find
///
/// # Returns
/// Up to `max_motifs` motifs, sorted by their 1D profile distance (ascending).
pub fn mmotifs(
    ts: &[&[f64]],
    m: usize,
    profile: &MultiDimensionalProfile,
    max_motifs: usize,
) -> Vec<MultiDimensionalMotif> {
    assert_eq!(
        ts.len(),
        profile.d,
        "Number of dimensions in ts ({}) must match profile ({})",
        ts.len(),
        profile.d
    );
    assert_eq!(m, profile.m, "Subsequence length mismatch");

    let ez = (m as f64 / 4.0).ceil() as usize;

    // Work with the 1D profile (row 0) to find motif candidates
    let mut working_profile = profile.profile[0].clone();
    let mut motifs = Vec::with_capacity(max_motifs);

    for _ in 0..max_motifs {
        // Find position with minimum 1D distance
        let (best_idx, best_dist) = match working_profile
            .iter()
            .enumerate()
            .filter(|(_, d)| d.is_finite())
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        {
            Some((idx, &dist)) => (idx, dist),
            None => break,
        };

        let nn_idx = profile.profile_index[0][best_idx];

        // Per-k NN indices from the multi-dimensional profile
        // Each row k of mstump has its own best NN for the motif pair
        let subseq_idx_arr: Vec<usize> = vec![best_idx; profile.d];
        let nn_idx_per_k: Vec<usize> = (0..profile.d)
            .map(|k| profile.profile_index[k][best_idx])
            .collect();

        // Compute MDL to find optimal dimensionality
        let (bit_sizes, _subspaces) = mdl(ts, m, &subseq_idx_arr, &nn_idx_per_k);

        let k = bit_sizes
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i + 1)
            .unwrap_or(1);

        // Get the subspace dimensions using row-0 NN
        let dimensions = subspace(ts, m, best_idx, nn_idx, k);

        motifs.push(MultiDimensionalMotif {
            idx: best_idx,
            nn_idx,
            dimensions,
            mdl_bit_sizes: bit_sizes,
            k,
            distance: best_dist,
        });

        // Apply exclusion zone to prevent overlapping motifs
        apply_exclusion_zone(&mut working_profile, best_idx, ez);
        apply_exclusion_zone(&mut working_profile, nn_idx, ez);
    }

    motifs
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algorithms::mstump::mstump;

    #[test]
    fn test_mmotifs_basic() {
        let n = 100;
        let m = 10;

        let ts0: Vec<f64> = (0..n).map(|i| (i as f64 * 0.2).sin()).collect();
        let ts1: Vec<f64> = (0..n).map(|i| (i as f64 * 0.3).cos()).collect();

        let ts_refs: [&[f64]; 2] = [&ts0, &ts1];
        let profile = mstump(&ts_refs, m);

        let motifs = mmotifs(&ts_refs, m, &profile, 3);

        assert!(!motifs.is_empty(), "Should find at least one motif");

        for (i, motif) in motifs.iter().enumerate() {
            assert!(
                motif.k >= 1 && motif.k <= 2,
                "motif[{i}].k = {} out of range [1, 2]",
                motif.k
            );
            assert_eq!(
                motif.dimensions.len(),
                motif.k,
                "motif[{i}] dimensions length should match k"
            );
            assert!(
                motif.distance >= 0.0,
                "motif[{i}].distance = {} is negative",
                motif.distance
            );
            assert_ne!(
                motif.idx, motif.nn_idx,
                "motif[{i}] should have distinct indices"
            );
        }
    }

    #[test]
    fn test_mmotifs_sorted_by_distance() {
        let n = 100;
        let m = 8;

        let ts0: Vec<f64> = (0..n).map(|i| (i as f64 * 0.2).sin()).collect();
        let ts1: Vec<f64> = (0..n).map(|i| (i as f64 * 0.3).cos()).collect();
        let ts2: Vec<f64> = (0..n).map(|i| (i as f64 * 0.15).sin() + 0.5).collect();

        let ts_refs: [&[f64]; 3] = [&ts0, &ts1, &ts2];
        let profile = mstump(&ts_refs, m);

        let motifs = mmotifs(&ts_refs, m, &profile, 5);

        for w in motifs.windows(2) {
            assert!(
                w[0].distance <= w[1].distance + 1e-10,
                "Motifs not sorted: {} > {}",
                w[0].distance,
                w[1].distance
            );
        }
    }

    #[test]
    fn test_mmotifs_exclusion_zone() {
        let n = 100;
        let m = 10;
        let ez = (m as f64 / 4.0).ceil() as usize;

        let ts0: Vec<f64> = (0..n).map(|i| (i as f64 * 0.2).sin()).collect();
        let ts1: Vec<f64> = (0..n).map(|i| (i as f64 * 0.3).cos()).collect();

        let ts_refs: [&[f64]; 2] = [&ts0, &ts1];
        let profile = mstump(&ts_refs, m);
        let motifs = mmotifs(&ts_refs, m, &profile, 5);

        // No two motif idx values should be within exclusion zone of each other
        for i in 0..motifs.len() {
            for j in (i + 1)..motifs.len() {
                let diff = motifs[i].idx.abs_diff(motifs[j].idx);
                assert!(
                    diff > ez,
                    "Motifs {} and {} indices too close: {} and {} (diff={diff}, ez={ez})",
                    i,
                    j,
                    motifs[i].idx,
                    motifs[j].idx
                );
            }
        }
    }
}
