use crate::algorithms::common::apply_exclusion_zone;
use crate::core::matrix_profile::MatrixProfile;

/// A discovered motif (recurring pattern).
///
/// A motif is a pair of subsequences with the smallest distance in the matrix profile.
/// Multiple motifs are found greedily with exclusion zone elimination.
#[derive(Debug, Clone)]
pub struct Motif {
    /// Index of the first occurrence.
    pub idx_a: usize,
    /// Index of the nearest-neighbor match.
    pub idx_b: usize,
    /// Distance between the two subsequences.
    pub distance: f64,
}

/// A discovered discord (anomaly).
///
/// A discord is a subsequence whose nearest neighbor is unusually far away,
/// indicating it is unlike any other pattern in the time series.
#[derive(Debug, Clone)]
pub struct Discord {
    /// Index of the anomalous subsequence.
    pub idx: usize,
    /// Distance to its nearest neighbor (high = anomalous).
    pub distance: f64,
}

/// Find the top-k motifs (most similar recurring patterns) in a matrix profile.
///
/// Uses greedy extraction with exclusion zone elimination: find the smallest
/// distance, record it, exclude both the motif and its match from future
/// consideration, and repeat.
///
/// # Arguments
/// * `mp` - A computed matrix profile
/// * `k` - Number of motifs to find
///
/// # Returns
/// Up to `k` motifs, sorted by distance (ascending). May return fewer than `k`
/// if the profile doesn't contain enough finite-distance entries.
pub fn find_motifs(mp: &MatrixProfile, k: usize) -> Vec<Motif> {
    let mut profile = mp.profile.clone();
    let ez = mp.exclusion_zone;
    let mut motifs = Vec::with_capacity(k);

    for _ in 0..k {
        // Find the index with the smallest distance
        let (best_idx, &best_dist) = match profile
            .iter()
            .enumerate()
            .filter(|(_, d)| d.is_finite())
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        {
            Some(pair) => pair,
            None => break,
        };

        let match_idx = mp.profile_index[best_idx];

        motifs.push(Motif {
            idx_a: best_idx,
            idx_b: match_idx,
            distance: best_dist,
        });

        // Exclude both the motif and its match from future consideration
        apply_exclusion_zone(&mut profile, best_idx, ez);
        apply_exclusion_zone(&mut profile, match_idx, ez);
    }

    motifs
}

/// Find the top-k discords (most anomalous subsequences) in a matrix profile.
///
/// Uses greedy extraction with exclusion zone elimination: find the largest
/// finite distance, record it, exclude it from future consideration, and repeat.
///
/// # Arguments
/// * `mp` - A computed matrix profile
/// * `k` - Number of discords to find
///
/// # Returns
/// Up to `k` discords, sorted by distance (descending). May return fewer than `k`
/// if the profile doesn't contain enough finite-distance entries.
pub fn find_discords(mp: &MatrixProfile, k: usize) -> Vec<Discord> {
    let mut profile = mp.profile.clone();
    let ez = mp.exclusion_zone;
    let mut discords = Vec::with_capacity(k);

    for _ in 0..k {
        // Find the index with the largest finite distance
        let (worst_idx, &worst_dist) = match profile
            .iter()
            .enumerate()
            .filter(|(_, d)| d.is_finite())
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        {
            Some(pair) => pair,
            None => break,
        };

        discords.push(Discord {
            idx: worst_idx,
            distance: worst_dist,
        });

        // Exclude this discord from future consideration
        apply_exclusion_zone(&mut profile, worst_idx, ez);
    }

    discords
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algorithms::stomp::stomp;
    use crate::core::matrix_profile::MatrixProfileConfig;
    use crate::metrics::euclidean::ZNormalizedEuclidean;

    #[test]
    fn test_find_motifs_basic() {
        // Create a signal where a distinctive pattern appears exactly twice,
        // far apart, with chaotic noise in between (each noise subsequence is unique).
        //
        // Pattern: [0, 1, 0, -1, 0, 1, 0, -1] at indices 0 and 40
        // Noise: pseudo-random values that create unique shapes
        let m = 8;
        let n = 56;
        let mut ts = vec![0.0; n];

        // Place pattern at index 0
        let pattern = [0.0, 1.0, 0.0, -1.0, 0.0, 1.0, 0.0, -1.0];
        ts[0..8].copy_from_slice(&pattern);

        // Fill noise (indices 8..40): use varied non-repeating values
        // Each group of m values has a distinct shape that won't match the pattern
        for (i, val) in ts.iter_mut().enumerate().take(40).skip(8) {
            // Mix of quadratic and varying terms so no subsequence matches another
            *val = (i as f64).powi(2) * 0.01 + (i as f64 * 1.7).sin() * 3.0;
        }

        // Place same pattern at index 40
        ts[40..48].copy_from_slice(&pattern);

        // More noise after second pattern
        for (i, val) in ts.iter_mut().enumerate().take(n).skip(48) {
            *val = -(i as f64).powi(2) * 0.01 + (i as f64 * 2.3).cos() * 5.0;
        }

        let config = MatrixProfileConfig::new(m);
        let mp = stomp::<ZNormalizedEuclidean>(&ts, &config);
        let motifs = find_motifs(&mp, 3);

        assert!(!motifs.is_empty());
        let top = &motifs[0];
        assert!(
            top.distance < 1e-4,
            "Top motif distance should be very small, got {}",
            top.distance
        );
        let pair = (top.idx_a.min(top.idx_b), top.idx_a.max(top.idx_b));
        assert_eq!(pair, (0, 40), "Top motif should be at (0, 40)");
    }

    #[test]
    fn test_find_discords_basic() {
        // Sine wave with an anomaly injected at index 25
        let mut ts: Vec<f64> = (0..100).map(|i| (i as f64 * 0.2).sin()).collect();
        // Inject anomaly: spike at index 25
        ts[25] = 10.0;
        ts[26] = -10.0;

        let config = MatrixProfileConfig::new(8);
        let mp = stomp::<ZNormalizedEuclidean>(&ts, &config);
        let discords = find_discords(&mp, 3);

        assert!(!discords.is_empty());
        // The top discord should be near the anomaly
        let top = &discords[0];
        let near_anomaly = (20..=30).contains(&top.idx);
        assert!(
            near_anomaly,
            "Top discord at index {} should be near anomaly at 25",
            top.idx
        );
    }

    #[test]
    fn test_motifs_decreasing_distance() {
        let ts: Vec<f64> = (0..200).map(|i| (i as f64 * 0.15).sin()).collect();
        let config = MatrixProfileConfig::new(10);
        let mp = stomp::<ZNormalizedEuclidean>(&ts, &config);
        let motifs = find_motifs(&mp, 5);

        for w in motifs.windows(2) {
            assert!(
                w[0].distance <= w[1].distance,
                "Motifs should be sorted by distance: {} > {}",
                w[0].distance,
                w[1].distance
            );
        }
    }

    #[test]
    fn test_discords_decreasing_distance() {
        let ts: Vec<f64> = (0..200).map(|i| (i as f64 * 0.15).sin()).collect();
        let config = MatrixProfileConfig::new(10);
        let mp = stomp::<ZNormalizedEuclidean>(&ts, &config);
        let discords = find_discords(&mp, 5);

        for w in discords.windows(2) {
            assert!(
                w[0].distance >= w[1].distance,
                "Discords should be sorted by distance: {} < {}",
                w[0].distance,
                w[1].distance
            );
        }
    }

    #[test]
    fn test_motifs_empty_profile() {
        // All infinite profile
        let mp = MatrixProfile::new(10, 4, 1);
        let motifs = find_motifs(&mp, 5);
        assert!(motifs.is_empty());
    }

    #[test]
    fn test_discords_empty_profile() {
        let mp = MatrixProfile::new(10, 4, 1);
        let discords = find_discords(&mp, 5);
        assert!(discords.is_empty());
    }
}
