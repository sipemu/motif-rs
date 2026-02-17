use crate::algorithms::ab_join::ab_join;
use crate::core::distance_metric::DistanceMetric;

/// Compute MPdist: a scalar distance between two time series based on the
/// matrix profile.
///
/// MPdist is defined as the `k`-th smallest value of the concatenated AB-join
/// distance profiles (A→B and B→A). This makes it robust to length differences
/// and partial matches.
///
/// # Arguments
/// * `ts_a` - First time series
/// * `ts_b` - Second time series
/// * `m` - Subsequence length
/// * `percentage` - Percentile to use (default: 0.05, matching stumpy).
///   `k = min(ceil(percentage * (len_a + len_b)), n_subs_total - 1)` (0-indexed).
///
/// # References
/// Gharghabi et al., "Matrix Profile XII: MPdist", 2018.
pub fn mpdist<M: DistanceMetric>(
    ts_a: &[f64],
    ts_b: &[f64],
    m: usize,
    percentage: Option<f64>,
) -> f64 {
    assert!(ts_a.len() >= m, "Time series A must be >= m");
    assert!(ts_b.len() >= m, "Time series B must be >= m");
    assert!(m >= 2, "Subsequence length must be >= 2");

    let n_a = ts_a.len() - m + 1;
    let n_b = ts_b.len() - m + 1;

    let (jp_a, jp_b) = ab_join::<M>(ts_a, ts_b, m);

    // Concatenate both distance profiles (matching stumpy: P_ABBA)
    let mut p_abba: Vec<f64> = Vec::with_capacity(n_a + n_b);
    p_abba.extend_from_slice(&jp_a.distances);
    p_abba.extend_from_slice(&jp_b.distances);

    if p_abba.is_empty() {
        return f64::INFINITY;
    }

    // Compute k (0-indexed) matching stumpy's formula:
    // k = min(ceil(percentage * (n_A + n_B)), total_subs - 1)
    // where n_A, n_B are RAW time series lengths (not subsequence counts)
    let p = percentage.unwrap_or(0.05);
    assert!((0.0..=1.0).contains(&p), "percentage must be in [0.0, 1.0]");
    let k = ((p * (ts_a.len() + ts_b.len()) as f64).ceil() as usize)
        .min(p_abba.len().saturating_sub(1));

    // O(n) partial sort to find the k-th smallest value
    let (_, kth, _) = p_abba.select_nth_unstable_by(k, |a, b| a.partial_cmp(b).unwrap());

    let result = *kth;

    // If the k-th value is infinite, fall back to the largest finite value
    if !result.is_finite() {
        let n_finite = p_abba.iter().filter(|v| v.is_finite()).count();
        if n_finite == 0 {
            return f64::INFINITY;
        }
        let k_fallback = n_finite - 1;
        let (_, kth_fb, _) =
            p_abba.select_nth_unstable_by(k_fallback, |a, b| a.partial_cmp(b).unwrap());
        return *kth_fb;
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metrics::euclidean::ZNormalizedEuclidean;

    #[test]
    fn test_mpdist_identical_series() {
        let ts: Vec<f64> = (0..100).map(|i| (i as f64 * 0.2).sin()).collect();
        let d = mpdist::<ZNormalizedEuclidean>(&ts, &ts, 10, None);
        assert!(d < 1e-4, "MPdist of identical series should be ~0, got {d}");
    }

    #[test]
    fn test_mpdist_different_series() {
        let ts_a: Vec<f64> = (0..100).map(|i| (i as f64 * 0.2).sin()).collect();
        let ts_b: Vec<f64> = (0..100).map(|i| (i as f64 * 0.5).cos()).collect();
        let d = mpdist::<ZNormalizedEuclidean>(&ts_a, &ts_b, 10, None);
        assert!(d.is_finite(), "MPdist should be finite");
        assert!(d >= 0.0, "MPdist should be non-negative");
    }

    #[test]
    fn test_mpdist_different_lengths() {
        let ts_a: Vec<f64> = (0..80).map(|i| (i as f64 * 0.2).sin()).collect();
        let ts_b: Vec<f64> = (0..120).map(|i| (i as f64 * 0.2).sin()).collect();
        let d = mpdist::<ZNormalizedEuclidean>(&ts_a, &ts_b, 10, None);
        assert!(
            d < 1e-4,
            "MPdist of similar series should be small, got {d}"
        );
    }

    #[test]
    fn test_mpdist_custom_percentage() {
        let ts_a: Vec<f64> = (0..100).map(|i| (i as f64 * 0.2).sin()).collect();
        let ts_b: Vec<f64> = (0..100).map(|i| (i as f64 * 0.5).cos()).collect();
        let d1 = mpdist::<ZNormalizedEuclidean>(&ts_a, &ts_b, 10, Some(0.05));
        let d2 = mpdist::<ZNormalizedEuclidean>(&ts_a, &ts_b, 10, Some(0.5));
        // Higher percentile should give higher or equal distance
        assert!(
            d2 >= d1 - 1e-10,
            "Higher percentile should give >= distance"
        );
    }

    #[test]
    fn test_mpdist_symmetry() {
        let ts_a: Vec<f64> = (0..80).map(|i| (i as f64 * 0.2).sin()).collect();
        let ts_b: Vec<f64> = (0..100).map(|i| (i as f64 * 0.3).cos()).collect();
        let d_ab = mpdist::<ZNormalizedEuclidean>(&ts_a, &ts_b, 10, Some(0.05));
        let d_ba = mpdist::<ZNormalizedEuclidean>(&ts_b, &ts_a, 10, Some(0.05));
        assert!(
            (d_ab - d_ba).abs() < 1e-6,
            "MPdist should be symmetric: {d_ab} vs {d_ba}"
        );
    }
}
