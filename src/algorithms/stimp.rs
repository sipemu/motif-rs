use crate::algorithms::scrump::scrump;
use crate::algorithms::stomp::stomp;
use crate::core::distance_metric::DistanceMetric;
use crate::core::matrix_profile::MatrixProfileConfig;

/// Pan matrix profile result: profiles computed across a range of window sizes.
///
/// Each entry corresponds to one window size. The profiles are normalized by
/// `1 / sqrt(2 * m)` so that distances are comparable across different window sizes.
#[derive(Debug, Clone)]
pub struct PanMatrixProfile {
    /// Normalized distance profile per window size (each has length `n - m + 1`).
    pub profiles: Vec<Vec<f64>>,
    /// Index profile per window size.
    pub indices: Vec<Vec<usize>>,
    /// The window sizes used.
    pub windows: Vec<usize>,
}

/// Compute the pan matrix profile across a range of window sizes.
///
/// For each window size from `min_m` to `max_m` (inclusive, with the given step),
/// computes either an exact STOMP or an approximate SCRUMP matrix profile and
/// normalizes it for cross-window-size comparability.
///
/// # Arguments
/// * `ts` - Time series
/// * `min_m` - Minimum window size (must be >= 2)
/// * `max_m` - Maximum window size (must be <= ts.len())
/// * `step` - Step between window sizes (default: 1)
/// * `percentage` - If `Some(p)` with `p < 1.0`, use SCRUMP approximation at each
///   window size. If `None` or `Some(1.0)`, use exact STOMP.
///
/// # References
/// Madrid et al., "Matrix Profile XX: Finding and Visualizing Time Series Motifs
/// of All Lengths using the Matrix Profile", 2019.
pub fn stimp<M: DistanceMetric>(
    ts: &[f64],
    min_m: usize,
    max_m: usize,
    step: Option<usize>,
    percentage: Option<f64>,
) -> PanMatrixProfile {
    assert!(min_m >= 2, "min_m must be >= 2");
    assert!(max_m >= min_m, "max_m must be >= min_m");
    assert!(
        max_m <= ts.len(),
        "max_m ({max_m}) must be <= ts.len() ({})",
        ts.len()
    );

    let step = step.unwrap_or(1).max(1);

    let mut profiles = Vec::new();
    let mut indices = Vec::new();
    let mut windows = Vec::new();

    let mut m = min_m;
    while m <= max_m {
        let config = MatrixProfileConfig::new(m);

        let mp = match percentage {
            Some(p) if p < 1.0 => scrump::<M>(ts, &config, p),
            _ => stomp::<M>(ts, &config),
        };

        // Pan matrix profile normalization: divide by sqrt(2 * m)
        // This makes distances comparable across window sizes
        let norm = 1.0 / (2.0 * m as f64).sqrt();
        let normalized_profile: Vec<f64> = mp.profile.iter().map(|&d| d * norm).collect();

        profiles.push(normalized_profile);
        indices.push(mp.profile_index);
        windows.push(m);

        m += step;
    }

    PanMatrixProfile {
        profiles,
        indices,
        windows,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metrics::euclidean::ZNormalizedEuclidean;

    #[test]
    fn test_stimp_basic() {
        let ts: Vec<f64> = (0..200).map(|i| (i as f64 * 0.2).sin()).collect();
        let pan = stimp::<ZNormalizedEuclidean>(&ts, 10, 30, Some(5), None);

        // Window sizes: 10, 15, 20, 25, 30
        assert_eq!(pan.windows, vec![10, 15, 20, 25, 30]);
        assert_eq!(pan.profiles.len(), 5);
        assert_eq!(pan.indices.len(), 5);

        // Each profile should have the correct length
        for (i, &m) in pan.windows.iter().enumerate() {
            let expected_len = ts.len() - m + 1;
            assert_eq!(
                pan.profiles[i].len(),
                expected_len,
                "Profile {i} (m={m}) has wrong length"
            );
            assert_eq!(
                pan.indices[i].len(),
                expected_len,
                "Indices {i} (m={m}) has wrong length"
            );
        }
    }

    #[test]
    fn test_stimp_with_scrump() {
        let ts: Vec<f64> = (0..200).map(|i| (i as f64 * 0.2).sin()).collect();
        let pan = stimp::<ZNormalizedEuclidean>(&ts, 10, 20, Some(5), Some(0.5));

        assert_eq!(pan.windows, vec![10, 15, 20]);
        assert_eq!(pan.profiles.len(), 3);
    }

    #[test]
    fn test_stimp_normalized_values_finite() {
        let ts: Vec<f64> = (0..200).map(|i| (i as f64 * 0.2).sin()).collect();
        let pan = stimp::<ZNormalizedEuclidean>(&ts, 10, 20, Some(5), None);

        for (wi, profile) in pan.profiles.iter().enumerate() {
            for (j, &d) in profile.iter().enumerate() {
                assert!(
                    d.is_finite(),
                    "Profile[{wi}][{j}] (m={}) should be finite, got {d}",
                    pan.windows[wi]
                );
                assert!(d >= 0.0, "Normalized profile should be non-negative: {d}");
            }
        }
    }

    #[test]
    fn test_stimp_single_window() {
        let ts: Vec<f64> = (0..100).map(|i| (i as f64 * 0.2).sin()).collect();
        let pan = stimp::<ZNormalizedEuclidean>(&ts, 15, 15, None, None);

        assert_eq!(pan.windows, vec![15]);
        assert_eq!(pan.profiles.len(), 1);
    }

    #[test]
    fn test_stimp_step_one() {
        let ts: Vec<f64> = (0..100).map(|i| (i as f64 * 0.2).sin()).collect();
        let pan = stimp::<ZNormalizedEuclidean>(&ts, 10, 13, Some(1), None);

        assert_eq!(pan.windows, vec![10, 11, 12, 13]);
    }

    #[test]
    #[should_panic(expected = "min_m must be >= 2")]
    fn test_stimp_min_m_too_small() {
        let ts = vec![1.0; 50];
        stimp::<ZNormalizedEuclidean>(&ts, 1, 10, None, None);
    }
}
