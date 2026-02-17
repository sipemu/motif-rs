use crate::algorithms::common::sliding_dot_product;
use crate::algorithms::stomp::stomp;
use crate::core::distance_metric::DistanceMetric;
use crate::core::matrix_profile::{MatrixProfile, MatrixProfileConfig};

/// Compute an approximate matrix profile using the PreSCRIMP algorithm.
///
/// Samples a fraction of diagonals to produce an approximate profile much faster
/// than exact STOMP. At `percentage >= 1.0`, delegates to exact STOMP.
///
/// # Arguments
/// * `ts` - Time series
/// * `config` - Matrix profile configuration (subsequence length, exclusion zone, etc.)
/// * `percentage` - Fraction of diagonals to sample, in (0.0, 1.0].
///   At 1.0, returns exact STOMP result.
///
/// # References
/// Zhu et al., "Matrix Profile XI: SCRIMP++", 2018.
pub fn scrump<M: DistanceMetric>(
    ts: &[f64],
    config: &MatrixProfileConfig,
    percentage: f64,
) -> MatrixProfile {
    assert!(
        percentage > 0.0,
        "percentage must be > 0.0, got {percentage}"
    );

    // At 100% or more, just use exact STOMP
    if percentage >= 1.0 {
        return stomp::<M>(ts, config);
    }

    let m = config.m;
    let n = ts.len();
    assert!(n >= m, "Time series length must be >= subsequence length");
    assert!(m >= 2, "Subsequence length must be >= 2");

    let n_subs = n - m + 1;
    let exclusion_zone = config.exclusion_zone();
    let ctx = M::precompute(ts, m);
    let mut mp = MatrixProfile::new(n_subs, m, exclusion_zone);

    // PreSCRIMP: sample random diagonal indices and compute MASS distance profiles
    // for those query subsequences, then walk the diagonal to propagate matches.
    //
    // Number of diagonals to sample (excluding trivial match diagonals)
    let n_diags = n_subs.saturating_sub(exclusion_zone + 1);
    let n_samples = ((percentage * n_diags as f64).ceil() as usize)
        .max(1)
        .min(n_diags);

    // Deterministic sampling: evenly spaced diagonal indices
    // This avoids requiring an RNG dependency and gives reproducible results.
    let step = if n_samples >= n_diags {
        1
    } else {
        n_diags / n_samples
    };

    for s in 0..n_samples {
        let diag_idx = exclusion_zone + 1 + s * step;
        if diag_idx >= n_subs {
            break;
        }

        // Compute the distance profile for the subsequence at diag_idx
        let dp = if M::supports_qt_optimization() {
            let query = &ts[diag_idx..diag_idx + m];
            let qts = sliding_dot_product(query, ts);
            (0..n_subs)
                .map(|j| M::qt_to_distance(qts[j], diag_idx, j, m, &ctx))
                .collect::<Vec<f64>>()
        } else {
            M::distance_profile(ts, diag_idx, m, &ctx)
        };

        // Apply exclusion zone around the query index and update the profile
        for (j, &d) in dp.iter().enumerate() {
            if j.abs_diff(diag_idx) <= exclusion_zone {
                continue;
            }
            mp.update(diag_idx, d, j);
            mp.update(j, d, diag_idx);
        }

        // Walk the diagonal: propagate along the diagonal using QT recurrence
        // This is the PreSCRIMP optimization — each sampled diagonal gives O(n) updates
        if M::supports_qt_optimization() {
            // Walk forward along the diagonal from (diag_idx, 0)
            // Diagonal k = diag_idx: pairs (i, j) = (i, i - diag_idx) for i >= diag_idx
            // But we use (i, j) where j = i + k or i = j + k depending on direction
            // Here: query is at diag_idx, diagonal k = diag_idx means pairs where |i - j| = k
            // Walk: (diag_idx+1, 1), (diag_idx+2, 2), ...
            let k = diag_idx;
            if k < n_subs {
                let mut qt = ts[k..k + m]
                    .iter()
                    .zip(&ts[0..m])
                    .map(|(a, b)| a * b)
                    .sum::<f64>();

                for p in 1..(n_subs - k).min(n_subs) {
                    let i = p + k;
                    let j = p;
                    if i >= n_subs {
                        break;
                    }
                    qt = qt - ts[i - 1] * ts[j - 1] + ts[i + m - 1] * ts[j + m - 1];
                    let d = M::qt_to_distance(qt, i, j, m, &ctx);
                    if j.abs_diff(i) > exclusion_zone {
                        mp.update(i, d, j);
                        mp.update(j, d, i);
                    }
                }
            }
        }
    }

    mp
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algorithms::stomp::stomp;
    use crate::metrics::euclidean::ZNormalizedEuclidean;

    #[test]
    fn test_scrump_full_percentage_matches_stomp() {
        let ts: Vec<f64> = (0..200).map(|i| (i as f64 * 0.2).sin()).collect();
        let config = MatrixProfileConfig::new(10);

        let exact = stomp::<ZNormalizedEuclidean>(&ts, &config);
        let approx = scrump::<ZNormalizedEuclidean>(&ts, &config, 1.0);

        // At percentage=1.0, should delegate to exact STOMP
        for (i, (e, a)) in exact.profile.iter().zip(approx.profile.iter()).enumerate() {
            assert!(
                (e - a).abs() < 1e-10,
                "Mismatch at {i}: exact={e}, approx={a}"
            );
        }
    }

    #[test]
    fn test_scrump_approximate_is_upper_bound() {
        let ts: Vec<f64> = (0..300).map(|i| (i as f64 * 0.2).sin()).collect();
        let config = MatrixProfileConfig::new(15);

        let exact = stomp::<ZNormalizedEuclidean>(&ts, &config);
        let approx = scrump::<ZNormalizedEuclidean>(&ts, &config, 0.25);

        // Approximate profile should be >= exact (it's an upper bound since
        // we only sample some diagonals)
        for (i, (e, a)) in exact.profile.iter().zip(approx.profile.iter()).enumerate() {
            assert!(
                *a >= *e - 1e-6,
                "Approx should be upper bound at {i}: exact={e}, approx={a}"
            );
        }
    }

    #[test]
    fn test_scrump_reasonable_approximation() {
        let ts: Vec<f64> = (0..500).map(|i| (i as f64 * 0.2).sin()).collect();
        let config = MatrixProfileConfig::new(20);

        let exact = stomp::<ZNormalizedEuclidean>(&ts, &config);
        let approx = scrump::<ZNormalizedEuclidean>(&ts, &config, 0.5);

        // With 50% sampling + diagonal propagation, many values should be close
        let mut close_count = 0;
        let total = exact.profile.len();
        for (e, a) in exact.profile.iter().zip(approx.profile.iter()) {
            if (*a - *e).abs() < 0.1 || *a < *e + 0.1 {
                close_count += 1;
            }
        }
        let close_fraction = close_count as f64 / total as f64;
        assert!(
            close_fraction > 0.5,
            "At 50% sampling, at least 50% of values should be close: {close_fraction:.2}"
        );
    }

    #[test]
    fn test_scrump_all_finite_at_full() {
        let ts: Vec<f64> = (0..200).map(|i| (i as f64 * 0.2).sin()).collect();
        let config = MatrixProfileConfig::new(10);

        let mp = scrump::<ZNormalizedEuclidean>(&ts, &config, 1.0);
        for (i, &d) in mp.profile.iter().enumerate() {
            assert!(d.is_finite(), "Profile[{i}] should be finite, got {d}");
        }
    }
}
