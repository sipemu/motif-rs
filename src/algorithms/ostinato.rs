use crate::algorithms::ab_join::ab_join;
use crate::core::distance_metric::DistanceMetric;

/// Result of consensus motif search across multiple time series.
#[derive(Debug, Clone)]
pub struct ConsensusMotif {
    /// Maximum nearest-neighbor distance across all series (the "radius").
    pub radius: f64,
    /// Index of the "central" time series containing the consensus motif.
    pub ts_index: usize,
    /// Starting index of the consensus motif subsequence within its series.
    pub subsequence_index: usize,
}

/// Evaluate a single candidate series: compute the max NN distance across all
/// other series for each subsequence, and return the best (minimum max-radius).
fn evaluate_candidate<M: DistanceMetric>(
    ts_list: &[&[f64]],
    c: usize,
    m: usize,
) -> ConsensusMotif {
    let n_c = ts_list[c].len() - m + 1;
    let mut max_radius = vec![0.0_f64; n_c];

    for (o, _) in ts_list.iter().enumerate().filter(|&(o, _)| o != c) {
        let (jp_c, _) = ab_join::<M>(ts_list[c], ts_list[o], m);
        for (mr, &d) in max_radius.iter_mut().zip(&jp_c.distances) {
            *mr = mr.max(d);
        }
    }

    let (subsequence_index, &radius) = max_radius
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .unwrap();

    ConsensusMotif {
        radius,
        ts_index: c,
        subsequence_index,
    }
}

/// Find the consensus motif across multiple time series.
///
/// The consensus motif is the subsequence (from any series) whose maximum
/// nearest-neighbor distance to all other series is minimized. This is the
/// "most representative" pattern across a collection of time series.
///
/// # Algorithm
/// For each candidate series and each subsequence within it:
/// 1. Compute AB-join against every other series
/// 2. For each subsequence, track the maximum NN distance across all other series
/// 3. The consensus motif is the subsequence with the minimum such max-distance
///
/// # Arguments
/// * `ts_list` - Slice of time series references
/// * `m` - Subsequence length
///
/// # Panics
/// Panics if `ts_list.len() < 2`, any series is shorter than `m`, or `m < 2`.
///
/// # References
/// Kamgar et al., "Matrix Profile XV: Exploiting Time Series Consensus Motifs
/// to Find Structure in Time Series Sets", 2019.
pub fn ostinato<M: DistanceMetric>(ts_list: &[&[f64]], m: usize) -> ConsensusMotif {
    assert!(ts_list.len() >= 2, "Need at least 2 time series");
    assert!(m >= 2, "Subsequence length must be >= 2");
    for (i, ts) in ts_list.iter().enumerate() {
        assert!(
            ts.len() >= m,
            "Time series {i} is shorter than m ({} < {m})",
            ts.len()
        );
    }

    let mut best = ConsensusMotif {
        radius: f64::INFINITY,
        ts_index: 0,
        subsequence_index: 0,
    };

    for c in 0..ts_list.len() {
        let candidate = evaluate_candidate::<M>(ts_list, c, m);
        if candidate.radius < best.radius {
            best = candidate;
        }
    }

    best
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metrics::euclidean::ZNormalizedEuclidean;

    #[test]
    fn test_ostinato_identical_series() {
        // All series identical → consensus motif should have radius ~0
        let ts: Vec<f64> = (0..100).map(|i| (i as f64 * 0.2).sin()).collect();
        let ts_list: Vec<&[f64]> = vec![&ts, &ts, &ts];

        let result = ostinato::<ZNormalizedEuclidean>(&ts_list, 10);
        assert!(
            result.radius < 1e-4,
            "Identical series should give radius ~0, got {}",
            result.radius
        );
    }

    #[test]
    fn test_ostinato_shared_pattern() {
        // Three series with a shared sine pattern: consensus should find it
        let n = 200;
        let m = 20;

        let ts0: Vec<f64> = (0..n).map(|i| (i as f64 * 0.2).sin()).collect();
        let ts1: Vec<f64> = (0..n)
            .map(|i| (i as f64 * 0.2).sin() + 0.1 * (i as f64 * 0.7).cos())
            .collect();
        let ts2: Vec<f64> = (0..n)
            .map(|i| (i as f64 * 0.2).sin() + 0.05 * (i as f64 * 1.1).sin())
            .collect();

        let ts_list: Vec<&[f64]> = vec![&ts0, &ts1, &ts2];
        let result = ostinato::<ZNormalizedEuclidean>(&ts_list, m);

        assert!(result.radius.is_finite(), "Radius should be finite");
        assert!(
            result.radius < 2.0,
            "Shared sine pattern should give small radius, got {}",
            result.radius
        );
        assert!(result.ts_index < 3);
        assert!(result.subsequence_index < n - m + 1);
    }

    #[test]
    fn test_ostinato_different_lengths() {
        let ts0: Vec<f64> = (0..100).map(|i| (i as f64 * 0.2).sin()).collect();
        let ts1: Vec<f64> = (0..150).map(|i| (i as f64 * 0.2).sin()).collect();
        let ts2: Vec<f64> = (0..120).map(|i| (i as f64 * 0.2).sin()).collect();

        let ts_list: Vec<&[f64]> = vec![&ts0, &ts1, &ts2];
        let result = ostinato::<ZNormalizedEuclidean>(&ts_list, 10);

        assert!(result.radius.is_finite());
        assert!(
            result.radius < 1e-4,
            "Similar series, got {}",
            result.radius
        );
    }

    #[test]
    #[should_panic(expected = "Need at least 2")]
    fn test_ostinato_too_few_series() {
        let ts: Vec<f64> = (0..100).map(|i| (i as f64 * 0.2).sin()).collect();
        ostinato::<ZNormalizedEuclidean>(&[&ts], 10);
    }
}
