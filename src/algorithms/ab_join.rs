use crate::algorithms::common::sliding_dot_product;
use crate::core::distance_metric::DistanceMetric;
use crate::core::matrix_profile::{JoinAccumulator, JoinAccumulatorDist, JoinProfile};

/// Minimum number of subsequences before dispatching to parallel AB-join.
#[cfg(feature = "parallel")]
const MIN_PARALLEL_SUBS: usize = 256;

/// Compute the AB-join between two time series.
///
/// Returns two `JoinProfile`s:
/// - The first contains nearest neighbors for each subsequence in `ts_a` (searching in `ts_b`)
/// - The second contains nearest neighbors for each subsequence in `ts_b` (searching in `ts_a`)
///
/// Unlike self-join, AB-join:
/// - Traverses ALL diagonals (no exclusion zone)
/// - Produces a rectangular distance matrix (n_a × n_b)
/// - Has two separate output profiles
pub fn ab_join<M: DistanceMetric>(
    ts_a: &[f64],
    ts_b: &[f64],
    m: usize,
) -> (JoinProfile, JoinProfile) {
    assert!(ts_a.len() >= m, "Time series A must be >= m");
    assert!(ts_b.len() >= m, "Time series B must be >= m");
    assert!(m >= 2, "Subsequence length must be >= 2");

    let n_a = ts_a.len() - m + 1;
    let n_b = ts_b.len() - m + 1;

    let ctx_a = M::precompute(ts_a, m);
    let ctx_b = M::precompute(ts_b, m);

    let mut jp_a = JoinProfile::new(n_a, m);
    let mut jp_b = JoinProfile::new(n_b, m);

    if M::supports_correlation_domain() && M::supports_ab_join() {
        #[cfg(feature = "parallel")]
        if n_a.min(n_b) >= MIN_PARALLEL_SUBS {
            ab_join_corr_parallel::<M>(
                ts_a, ts_b, m, n_a, n_b, &ctx_a, &ctx_b, &mut jp_a, &mut jp_b,
            );
        } else {
            ab_join_corr::<M>(
                ts_a, ts_b, m, n_a, n_b, &ctx_a, &ctx_b, &mut jp_a, &mut jp_b,
            );
        }
        #[cfg(not(feature = "parallel"))]
        ab_join_corr::<M>(
            ts_a, ts_b, m, n_a, n_b, &ctx_a, &ctx_b, &mut jp_a, &mut jp_b,
        );
    } else if M::supports_qt_optimization() && M::supports_ab_join() {
        #[cfg(feature = "parallel")]
        if n_a.min(n_b) >= MIN_PARALLEL_SUBS {
            ab_join_qt_parallel::<M>(
                ts_a, ts_b, m, n_a, n_b, &ctx_a, &ctx_b, &mut jp_a, &mut jp_b,
            );
        } else {
            ab_join_qt::<M>(
                ts_a, ts_b, m, n_a, n_b, &ctx_a, &ctx_b, &mut jp_a, &mut jp_b,
            );
        }
        #[cfg(not(feature = "parallel"))]
        ab_join_qt::<M>(
            ts_a, ts_b, m, n_a, n_b, &ctx_a, &ctx_b, &mut jp_a, &mut jp_b,
        );
    } else {
        ab_join_naive::<M>(
            ts_a, ts_b, m, n_a, n_b, &ctx_a, &ctx_b, &mut jp_a, &mut jp_b,
        );
    }

    (jp_a, jp_b)
}

/// Correlation-domain AB-join (serial).
#[allow(clippy::too_many_arguments)]
fn ab_join_corr<M: DistanceMetric>(
    ts_a: &[f64],
    ts_b: &[f64],
    m: usize,
    n_a: usize,
    n_b: usize,
    ctx_a: &M::Context,
    ctx_b: &M::Context,
    jp_a: &mut JoinProfile,
    jp_b: &mut JoinProfile,
) {
    let (mean_a, msi_a, mean_b, msi_b, _has_constant) = M::correlation_data_ab(ctx_a, ctx_b);
    let m_f = m as f64;

    // Positive diagonals: qt_first_pos[j] = dot(T_A[0..m], T_B[j..j+m])
    let qt_first_pos = sliding_dot_product(&ts_a[0..m], ts_b);
    // Negative diagonals: qt_first_neg[i] = dot(T_B[0..m], T_A[i..i+m])
    let qt_first_neg = sliding_dot_product(&ts_b[0..m], ts_a);

    let mut acc_a = JoinAccumulator::new(n_a);
    let mut acc_b = JoinAccumulator::new(n_b);

    // Positive diagonals: k = 0..n_b-1, start at (i=0, j=k)
    for k in 0..n_b {
        let diag_len = n_a.min(n_b - k);
        let mut qt = qt_first_pos[k];

        let neg_r = corr_neg_r(qt, m_f, mean_a[0], mean_b[k], msi_a[0], msi_b[k]);
        acc_a.update(0, neg_r, k);
        acc_b.update(k, neg_r, 0);

        for p in 1..diag_len {
            let i = p;
            let j = p + k;
            qt = qt - ts_a[i - 1] * ts_b[j - 1] + ts_a[i + m - 1] * ts_b[j + m - 1];
            let neg_r = corr_neg_r(qt, m_f, mean_a[i], mean_b[j], msi_a[i], msi_b[j]);
            acc_a.update(i, neg_r, j);
            acc_b.update(j, neg_r, i);
        }
    }

    // Negative diagonals: k = 1..n_a-1, start at (i=k, j=0)
    for k in 1..n_a {
        let diag_len = n_b.min(n_a - k);
        let mut qt = qt_first_neg[k];

        let neg_r = corr_neg_r(qt, m_f, mean_a[k], mean_b[0], msi_a[k], msi_b[0]);
        acc_a.update(k, neg_r, 0);
        acc_b.update(0, neg_r, k);

        for p in 1..diag_len {
            let i = p + k;
            let j = p;
            qt = qt - ts_a[i - 1] * ts_b[j - 1] + ts_a[i + m - 1] * ts_b[j + m - 1];
            let neg_r = corr_neg_r(qt, m_f, mean_a[i], mean_b[j], msi_a[i], msi_b[j]);
            acc_a.update(i, neg_r, j);
            acc_b.update(j, neg_r, i);
        }
    }

    let two_m = 2.0 * m_f;
    acc_a.write_to_join_profile(jp_a, |nc| (two_m * (1.0 + nc)).max(0.0).sqrt());
    acc_b.write_to_join_profile(jp_b, |nc| (two_m * (1.0 + nc)).max(0.0).sqrt());
}

/// Compute negated Pearson correlation for AB-join.
#[inline(always)]
fn corr_neg_r(qt: f64, m_f: f64, mean_i: f64, mean_j: f64, msi_i: f64, msi_j: f64) -> f64 {
    if msi_i == 0.0 && msi_j == 0.0 {
        -1.0 // both constant → perfect match
    } else if msi_i == 0.0 || msi_j == 0.0 {
        0.0 // one constant → max distance
    } else {
        (m_f * mean_i).mul_add(mean_j, -qt) * msi_i * msi_j
    }
}

/// Parallel correlation-domain AB-join.
#[cfg(feature = "parallel")]
#[allow(clippy::too_many_arguments)]
fn ab_join_corr_parallel<M: DistanceMetric>(
    ts_a: &[f64],
    ts_b: &[f64],
    m: usize,
    n_a: usize,
    n_b: usize,
    ctx_a: &M::Context,
    ctx_b: &M::Context,
    jp_a: &mut JoinProfile,
    jp_b: &mut JoinProfile,
) {
    use rayon::prelude::*;

    let (mean_a, msi_a, mean_b, msi_b, _has_constant) = M::correlation_data_ab(ctx_a, ctx_b);
    let m_f = m as f64;

    let qt_first_pos = sliding_dot_product(&ts_a[0..m], ts_b);
    let qt_first_neg = sliding_dot_product(&ts_b[0..m], ts_a);

    let n_threads = rayon::current_num_threads();

    // Process positive diagonals in parallel
    let chunk_size_pos = n_b.div_ceil(n_threads);
    let pos_results: Vec<(JoinAccumulator, JoinAccumulator)> = (0..n_threads)
        .into_par_iter()
        .map(|t| {
            let start_k = t * chunk_size_pos;
            let end_k = (start_k + chunk_size_pos).min(n_b);
            let mut acc_a = JoinAccumulator::new(n_a);
            let mut acc_b = JoinAccumulator::new(n_b);

            for k in start_k..end_k {
                let diag_len = n_a.min(n_b - k);
                let mut qt = qt_first_pos[k];

                let neg_r = corr_neg_r(qt, m_f, mean_a[0], mean_b[k], msi_a[0], msi_b[k]);
                acc_a.update(0, neg_r, k);
                acc_b.update(k, neg_r, 0);

                for p in 1..diag_len {
                    let i = p;
                    let j = p + k;
                    qt = qt - ts_a[i - 1] * ts_b[j - 1] + ts_a[i + m - 1] * ts_b[j + m - 1];
                    let neg_r = corr_neg_r(qt, m_f, mean_a[i], mean_b[j], msi_a[i], msi_b[j]);
                    acc_a.update(i, neg_r, j);
                    acc_b.update(j, neg_r, i);
                }
            }

            (acc_a, acc_b)
        })
        .collect();

    // Process negative diagonals in parallel
    let n_neg = n_a.saturating_sub(1);
    let chunk_size_neg = n_neg.div_ceil(n_threads);
    let neg_results: Vec<(JoinAccumulator, JoinAccumulator)> = (0..n_threads)
        .into_par_iter()
        .map(|t| {
            let start_k = t * chunk_size_neg + 1;
            let end_k = (start_k + chunk_size_neg).min(n_a);
            let mut acc_a = JoinAccumulator::new(n_a);
            let mut acc_b = JoinAccumulator::new(n_b);

            for k in start_k..end_k {
                let diag_len = n_b.min(n_a - k);
                let mut qt = qt_first_neg[k];

                let neg_r = corr_neg_r(qt, m_f, mean_a[k], mean_b[0], msi_a[k], msi_b[0]);
                acc_a.update(k, neg_r, 0);
                acc_b.update(0, neg_r, k);

                for p in 1..diag_len {
                    let i = p + k;
                    let j = p;
                    qt = qt - ts_a[i - 1] * ts_b[j - 1] + ts_a[i + m - 1] * ts_b[j + m - 1];
                    let neg_r = corr_neg_r(qt, m_f, mean_a[i], mean_b[j], msi_a[i], msi_b[j]);
                    acc_a.update(i, neg_r, j);
                    acc_b.update(j, neg_r, i);
                }
            }

            (acc_a, acc_b)
        })
        .collect();

    // Merge all results
    let mut combined_a = JoinAccumulator::new(n_a);
    let mut combined_b = JoinAccumulator::new(n_b);

    for (acc_a, acc_b) in pos_results.iter().chain(neg_results.iter()) {
        combined_a.merge(acc_a);
        combined_b.merge(acc_b);
    }

    let two_m = 2.0 * m_f;
    combined_a.write_to_join_profile(jp_a, |nc| (two_m * (1.0 + nc)).max(0.0).sqrt());
    combined_b.write_to_join_profile(jp_b, |nc| (two_m * (1.0 + nc)).max(0.0).sqrt());
}

/// QT-optimized AB-join (serial, for non-correlation-domain metrics like AAMP).
#[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
fn ab_join_qt<M: DistanceMetric>(
    ts_a: &[f64],
    ts_b: &[f64],
    m: usize,
    n_a: usize,
    n_b: usize,
    ctx_a: &M::Context,
    ctx_b: &M::Context,
    jp_a: &mut JoinProfile,
    jp_b: &mut JoinProfile,
) {
    let qt_first_pos = sliding_dot_product(&ts_a[0..m], ts_b);
    let qt_first_neg = sliding_dot_product(&ts_b[0..m], ts_a);

    let mut acc_a = JoinAccumulatorDist::new(n_a);
    let mut acc_b = JoinAccumulatorDist::new(n_b);

    // Positive diagonals
    for k in 0..n_b {
        let diag_len = n_a.min(n_b - k);
        let mut qt = qt_first_pos[k];

        let d = M::qt_to_distance_ab(qt, 0, k, m, ctx_a, ctx_b);
        acc_a.update(0, d, k);
        acc_b.update(k, d, 0);

        for p in 1..diag_len {
            let i = p;
            let j = p + k;
            qt = qt - ts_a[i - 1] * ts_b[j - 1] + ts_a[i + m - 1] * ts_b[j + m - 1];
            let d = M::qt_to_distance_ab(qt, i, j, m, ctx_a, ctx_b);
            acc_a.update(i, d, j);
            acc_b.update(j, d, i);
        }
    }

    // Negative diagonals
    for k in 1..n_a {
        let diag_len = n_b.min(n_a - k);
        let mut qt = qt_first_neg[k];

        let d = M::qt_to_distance_ab(qt, k, 0, m, ctx_a, ctx_b);
        acc_a.update(k, d, 0);
        acc_b.update(0, d, k);

        for p in 1..diag_len {
            let i = p + k;
            let j = p;
            qt = qt - ts_a[i - 1] * ts_b[j - 1] + ts_a[i + m - 1] * ts_b[j + m - 1];
            let d = M::qt_to_distance_ab(qt, i, j, m, ctx_a, ctx_b);
            acc_a.update(i, d, j);
            acc_b.update(j, d, i);
        }
    }

    acc_a.write_to_join_profile(jp_a);
    acc_b.write_to_join_profile(jp_b);
}

/// Parallel QT-optimized AB-join.
#[cfg(feature = "parallel")]
#[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
fn ab_join_qt_parallel<M: DistanceMetric>(
    ts_a: &[f64],
    ts_b: &[f64],
    m: usize,
    n_a: usize,
    n_b: usize,
    ctx_a: &M::Context,
    ctx_b: &M::Context,
    jp_a: &mut JoinProfile,
    jp_b: &mut JoinProfile,
) {
    use rayon::prelude::*;

    let qt_first_pos = sliding_dot_product(&ts_a[0..m], ts_b);
    let qt_first_neg = sliding_dot_product(&ts_b[0..m], ts_a);

    let n_threads = rayon::current_num_threads();

    let chunk_size_pos = n_b.div_ceil(n_threads);
    let pos_results: Vec<(JoinAccumulatorDist, JoinAccumulatorDist)> = (0..n_threads)
        .into_par_iter()
        .map(|t| {
            let start_k = t * chunk_size_pos;
            let end_k = (start_k + chunk_size_pos).min(n_b);
            let mut acc_a = JoinAccumulatorDist::new(n_a);
            let mut acc_b = JoinAccumulatorDist::new(n_b);

            for k in start_k..end_k {
                let diag_len = n_a.min(n_b - k);
                let mut qt = qt_first_pos[k];

                let d = M::qt_to_distance_ab(qt, 0, k, m, ctx_a, ctx_b);
                acc_a.update(0, d, k);
                acc_b.update(k, d, 0);

                for p in 1..diag_len {
                    let i = p;
                    let j = p + k;
                    qt = qt - ts_a[i - 1] * ts_b[j - 1] + ts_a[i + m - 1] * ts_b[j + m - 1];
                    let d = M::qt_to_distance_ab(qt, i, j, m, ctx_a, ctx_b);
                    acc_a.update(i, d, j);
                    acc_b.update(j, d, i);
                }
            }

            (acc_a, acc_b)
        })
        .collect();

    let n_neg = n_a.saturating_sub(1);
    let chunk_size_neg = n_neg.div_ceil(n_threads);
    let neg_results: Vec<(JoinAccumulatorDist, JoinAccumulatorDist)> = (0..n_threads)
        .into_par_iter()
        .map(|t| {
            let start_k = t * chunk_size_neg + 1;
            let end_k = (start_k + chunk_size_neg).min(n_a);
            let mut acc_a = JoinAccumulatorDist::new(n_a);
            let mut acc_b = JoinAccumulatorDist::new(n_b);

            for k in start_k..end_k {
                let diag_len = n_b.min(n_a - k);
                let mut qt = qt_first_neg[k];

                let d = M::qt_to_distance_ab(qt, k, 0, m, ctx_a, ctx_b);
                acc_a.update(k, d, 0);
                acc_b.update(0, d, k);

                for p in 1..diag_len {
                    let i = p + k;
                    let j = p;
                    qt = qt - ts_a[i - 1] * ts_b[j - 1] + ts_a[i + m - 1] * ts_b[j + m - 1];
                    let d = M::qt_to_distance_ab(qt, i, j, m, ctx_a, ctx_b);
                    acc_a.update(i, d, j);
                    acc_b.update(j, d, i);
                }
            }

            (acc_a, acc_b)
        })
        .collect();

    let mut combined_a = JoinAccumulatorDist::new(n_a);
    let mut combined_b = JoinAccumulatorDist::new(n_b);

    for (acc_a, acc_b) in pos_results.iter().chain(neg_results.iter()) {
        combined_a.merge(acc_a);
        combined_b.merge(acc_b);
    }

    combined_a.write_to_join_profile(jp_a);
    combined_b.write_to_join_profile(jp_b);
}

/// Naive AB-join for metrics without QT optimization.
#[allow(clippy::too_many_arguments)]
fn ab_join_naive<M: DistanceMetric>(
    ts_a: &[f64],
    ts_b: &[f64],
    m: usize,
    n_a: usize,
    n_b: usize,
    ctx_a: &M::Context,
    ctx_b: &M::Context,
    jp_a: &mut JoinProfile,
    jp_b: &mut JoinProfile,
) {
    // Build a combined time series for distance computation
    // This is a fallback; for proper cross-series naive distance we compute directly
    for i in 0..n_a {
        for j in 0..n_b {
            // Compute cross-distance directly
            let qt: f64 = ts_a[i..i + m]
                .iter()
                .zip(&ts_b[j..j + m])
                .map(|(a, b)| a * b)
                .sum();
            let d = if M::supports_ab_join() {
                M::qt_to_distance_ab(qt, i, j, m, ctx_a, ctx_b)
            } else {
                // Compute Euclidean distance directly
                let sq_sum: f64 = ts_a[i..i + m]
                    .iter()
                    .zip(&ts_b[j..j + m])
                    .map(|(a, b)| (a - b) * (a - b))
                    .sum();
                sq_sum.sqrt()
            };

            if d < jp_a.distances[i] {
                jp_a.distances[i] = d;
                jp_a.indices[i] = j;
            }
            if d < jp_b.distances[j] {
                jp_b.distances[j] = d;
                jp_b.indices[j] = i;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metrics::absolute::AbsoluteEuclidean;
    use crate::metrics::euclidean::ZNormalizedEuclidean;

    #[test]
    fn test_ab_join_identical_series() {
        // AB-join of a series with itself should yield distance ≈ 0 everywhere
        let ts: Vec<f64> = (0..50).map(|i| (i as f64 * 0.2).sin()).collect();
        let m = 8;
        let (jp_a, jp_b) = ab_join::<ZNormalizedEuclidean>(&ts, &ts, m);

        for (i, &d) in jp_a.distances.iter().enumerate() {
            assert!(
                d < 1e-6,
                "AB-join of identical series: d[{i}] should be ~0, got {d}"
            );
        }
        for (j, &d) in jp_b.distances.iter().enumerate() {
            assert!(
                d < 1e-6,
                "AB-join of identical series: d[{j}] should be ~0, got {d}"
            );
        }
    }

    #[test]
    fn test_ab_join_different_lengths() {
        let ts_a: Vec<f64> = (0..30).map(|i| (i as f64 * 0.3).sin()).collect();
        let ts_b: Vec<f64> = (0..50).map(|i| (i as f64 * 0.3).sin()).collect();
        let m = 6;

        let (jp_a, jp_b) = ab_join::<ZNormalizedEuclidean>(&ts_a, &ts_b, m);

        assert_eq!(jp_a.distances.len(), ts_a.len() - m + 1);
        assert_eq!(jp_b.distances.len(), ts_b.len() - m + 1);

        // Since ts_b contains ts_a as a prefix, all A subsequences should have near-zero matches
        for (i, &d) in jp_a.distances.iter().enumerate() {
            assert!(d < 1e-4, "AB-join: d_a[{i}] should be small, got {d}");
        }
    }

    #[test]
    fn test_ab_join_aamp() {
        let ts_a = vec![1.0, 2.0, 3.0, 2.0, 1.0, 2.0, 3.0, 2.0];
        let ts_b = vec![1.0, 2.0, 3.0, 2.0, 5.0, 6.0, 7.0, 6.0];
        let m = 4;

        let (jp_a, jp_b) = ab_join::<AbsoluteEuclidean>(&ts_a, &ts_b, m);

        assert_eq!(jp_a.distances.len(), ts_a.len() - m + 1);
        assert_eq!(jp_b.distances.len(), ts_b.len() - m + 1);

        // [1,2,3,2] at ts_a[0] matches [1,2,3,2] at ts_b[0] exactly
        assert!(
            jp_a.distances[0] < 1e-6,
            "AAMP AB-join: d_a[0] should be ~0, got {}",
            jp_a.distances[0]
        );
    }

    #[test]
    fn test_ab_join_symmetry() {
        // ab_join(A, B) and ab_join(B, A) should produce swapped results
        let ts_a: Vec<f64> = (0..30).map(|i| (i as f64 * 0.2).sin()).collect();
        let ts_b: Vec<f64> = (0..30).map(|i| (i as f64 * 0.3).cos()).collect();
        let m = 6;

        let (jp_a_fwd, jp_b_fwd) = ab_join::<ZNormalizedEuclidean>(&ts_a, &ts_b, m);
        let (jp_b_rev, jp_a_rev) = ab_join::<ZNormalizedEuclidean>(&ts_b, &ts_a, m);

        let eps = 1e-6;
        for i in 0..jp_a_fwd.distances.len() {
            assert!(
                (jp_a_fwd.distances[i] - jp_a_rev.distances[i]).abs() < eps,
                "AB-join symmetry mismatch at A[{i}]: {} vs {}",
                jp_a_fwd.distances[i],
                jp_a_rev.distances[i]
            );
        }
        for j in 0..jp_b_fwd.distances.len() {
            assert!(
                (jp_b_fwd.distances[j] - jp_b_rev.distances[j]).abs() < eps,
                "AB-join symmetry mismatch at B[{j}]: {} vs {}",
                jp_b_fwd.distances[j],
                jp_b_rev.distances[j]
            );
        }
    }

    #[test]
    fn test_ab_join_all_finite() {
        let ts_a: Vec<f64> = (0..40).map(|i| (i as f64 * 0.2).sin()).collect();
        let ts_b: Vec<f64> = (0..40).map(|i| (i as f64 * 0.3).cos()).collect();
        let m = 8;

        let (jp_a, jp_b) = ab_join::<ZNormalizedEuclidean>(&ts_a, &ts_b, m);

        for (i, &d) in jp_a.distances.iter().enumerate() {
            assert!(d.is_finite(), "jp_a.distances[{i}] is not finite: {d}");
        }
        for (j, &d) in jp_b.distances.iter().enumerate() {
            assert!(d.is_finite(), "jp_b.distances[{j}] is not finite: {d}");
        }
    }
}
