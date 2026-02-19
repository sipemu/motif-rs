use crate::core::matrix_profile::{
    JoinAccumulatorDist, JoinProfile, MatrixProfile, MatrixProfileConfig,
};
use crate::metrics::absolute::AbsoluteEuclidean;

/// Minimum number of subsequences before dispatching to parallel p-norm STOMP.
#[cfg(feature = "parallel")]
const MIN_PARALLEL_SUBS: usize = 256;

/// Compute the matrix profile using Minkowski p-norm distance.
///
/// For `p == 2.0`, delegates to the optimized `stomp::<AbsoluteEuclidean>` path.
/// For other values of `p`, uses a diagonal traversal with O(1) recurrence:
///
/// ```text
/// D^p(i+1, j+1) = D^p(i,j) - |T[i]-T[j]|^p + |T[i+m]-T[j+m]|^p
/// ```
///
/// Special-cases `p == 1.0` (Manhattan distance) to use `abs()` instead of `powf()`.
pub fn stomp_pnorm(ts: &[f64], config: &MatrixProfileConfig, p: f64) -> MatrixProfile {
    assert!(p >= 1.0, "p-norm requires p >= 1.0");

    let m = config.m;
    let n = ts.len();
    assert!(n >= m, "Time series length must be >= subsequence length");
    assert!(m >= 2, "Subsequence length must be >= 2");

    // For p=2, delegate to optimized AAMP path
    if (p - 2.0).abs() < f64::EPSILON {
        return crate::algorithms::stomp::stomp::<AbsoluteEuclidean>(ts, config);
    }

    let n_subs = n - m + 1;
    let exclusion_zone = config.exclusion_zone();
    let mut mp = MatrixProfile::new(n_subs, m, exclusion_zone);

    #[cfg(feature = "parallel")]
    if n_subs >= MIN_PARALLEL_SUBS {
        stomp_pnorm_parallel(ts, m, n_subs, exclusion_zone, p, &mut mp);
        return mp;
    }

    stomp_pnorm_serial(ts, m, n_subs, exclusion_zone, p, &mut mp);
    mp
}

/// Compute the p-norm distance^p sum for a subsequence pair.
#[inline]
fn pnorm_init(ts: &[f64], i: usize, j: usize, m: usize, p: f64) -> f64 {
    if p == 1.0 {
        ts[i..i + m]
            .iter()
            .zip(&ts[j..j + m])
            .map(|(a, b)| (a - b).abs())
            .sum()
    } else {
        ts[i..i + m]
            .iter()
            .zip(&ts[j..j + m])
            .map(|(a, b)| (a - b).abs().powf(p))
            .sum()
    }
}

/// Convert accumulated D^p sum to final distance.
#[inline]
fn dp_to_distance(dp_sum: f64, p: f64) -> f64 {
    if p == 1.0 {
        dp_sum.max(0.0)
    } else {
        dp_sum.max(0.0).powf(1.0 / p)
    }
}

/// Compute term |a - b|^p.
#[inline]
fn pnorm_term(a: f64, b: f64, p: f64) -> f64 {
    if p == 1.0 {
        (a - b).abs()
    } else {
        (a - b).abs().powf(p)
    }
}

/// Serial diagonal p-norm STOMP.
fn stomp_pnorm_serial(
    ts: &[f64],
    m: usize,
    n_subs: usize,
    exclusion_zone: usize,
    p: f64,
    mp: &mut MatrixProfile,
) {
    let first_k = exclusion_zone + 1;

    for k in first_k..n_subs {
        let diag_len = n_subs - k;

        // O(m) initialization for first position on this diagonal
        let mut dp_sum = pnorm_init(ts, 0, k, m, p);
        let d = dp_to_distance(dp_sum, p);
        mp.update(0, d, k);
        mp.update(k, d, 0);

        // O(1) recurrence for remaining positions
        for i in 1..diag_len {
            let j = i + k;
            dp_sum = dp_sum - pnorm_term(ts[i - 1], ts[j - 1], p)
                + pnorm_term(ts[i + m - 1], ts[j + m - 1], p);
            let d = dp_to_distance(dp_sum, p);
            mp.update(i, d, j);
            mp.update(j, d, i);
        }
    }
}

/// Parallel diagonal p-norm STOMP with load-balanced chunking.
#[cfg(feature = "parallel")]
fn stomp_pnorm_parallel(
    ts: &[f64],
    m: usize,
    n_subs: usize,
    exclusion_zone: usize,
    p: f64,
    mp: &mut MatrixProfile,
) {
    use rayon::prelude::*;

    let first_k = exclusion_zone + 1;
    let n_threads = rayon::current_num_threads();
    let ranges = crate::algorithms::stomp::compute_diagonal_ranges(first_k, n_subs, n_threads);

    let results: Vec<MatrixProfile> = ranges
        .into_par_iter()
        .map(|(start_k, end_k)| {
            let mut local_mp = MatrixProfile::new(n_subs, m, exclusion_zone);

            for k in start_k..end_k {
                let diag_len = n_subs - k;
                let mut dp_sum = pnorm_init(ts, 0, k, m, p);
                let d = dp_to_distance(dp_sum, p);
                local_mp.update(0, d, k);
                local_mp.update(k, d, 0);

                for i in 1..diag_len {
                    let j = i + k;
                    dp_sum = dp_sum - pnorm_term(ts[i - 1], ts[j - 1], p)
                        + pnorm_term(ts[i + m - 1], ts[j + m - 1], p);
                    let d = dp_to_distance(dp_sum, p);
                    local_mp.update(i, d, j);
                    local_mp.update(j, d, i);
                }
            }

            local_mp
        })
        .collect();

    for result in &results {
        mp.merge(result);
    }
}

/// Compute the AB-join between two time series using Minkowski p-norm distance.
///
/// For `p == 2.0`, delegates to the optimized `ab_join::<AbsoluteEuclidean>` path.
/// For other values of `p`, uses diagonal traversal with O(1) recurrence over
/// both positive and negative diagonals.
///
/// Returns two `JoinProfile`s:
/// - First: nearest neighbors for each subsequence in `ts_a` (searching in `ts_b`)
/// - Second: nearest neighbors for each subsequence in `ts_b` (searching in `ts_a`)
pub fn ab_join_pnorm(ts_a: &[f64], ts_b: &[f64], m: usize, p: f64) -> (JoinProfile, JoinProfile) {
    assert!(p >= 1.0, "p-norm requires p >= 1.0");
    assert!(ts_a.len() >= m, "Time series A must be >= m");
    assert!(ts_b.len() >= m, "Time series B must be >= m");
    assert!(m >= 2, "Subsequence length must be >= 2");

    // For p=2, delegate to optimized AAMP path
    if (p - 2.0).abs() < f64::EPSILON {
        return crate::algorithms::ab_join::ab_join::<AbsoluteEuclidean>(ts_a, ts_b, m);
    }

    let n_a = ts_a.len() - m + 1;
    let n_b = ts_b.len() - m + 1;
    let mut jp_a = JoinProfile::new(n_a, m);
    let mut jp_b = JoinProfile::new(n_b, m);

    #[cfg(feature = "parallel")]
    if n_a.min(n_b) >= MIN_PARALLEL_SUBS {
        ab_join_pnorm_parallel(ts_a, ts_b, m, n_a, n_b, p, &mut jp_a, &mut jp_b);
        return (jp_a, jp_b);
    }

    ab_join_pnorm_serial(ts_a, ts_b, m, n_a, n_b, p, &mut jp_a, &mut jp_b);
    (jp_a, jp_b)
}

/// Compute p-norm distance^p sum for a cross-series subsequence pair.
#[inline]
fn pnorm_init_ab(ts_a: &[f64], ts_b: &[f64], i: usize, j: usize, m: usize, p: f64) -> f64 {
    if p == 1.0 {
        ts_a[i..i + m]
            .iter()
            .zip(&ts_b[j..j + m])
            .map(|(a, b)| (a - b).abs())
            .sum()
    } else {
        ts_a[i..i + m]
            .iter()
            .zip(&ts_b[j..j + m])
            .map(|(a, b)| (a - b).abs().powf(p))
            .sum()
    }
}

/// Serial AB-join with p-norm diagonal traversal.
#[allow(clippy::too_many_arguments)]
fn ab_join_pnorm_serial(
    ts_a: &[f64],
    ts_b: &[f64],
    m: usize,
    n_a: usize,
    n_b: usize,
    p: f64,
    jp_a: &mut JoinProfile,
    jp_b: &mut JoinProfile,
) {
    let mut acc_a = JoinAccumulatorDist::new(n_a);
    let mut acc_b = JoinAccumulatorDist::new(n_b);

    // Positive diagonals: k = 0..n_b, pairs (i, i+k) with i in A, i+k in B
    for k in 0..n_b {
        let diag_len = n_a.min(n_b - k);

        let mut dp_sum = pnorm_init_ab(ts_a, ts_b, 0, k, m, p);
        let d = dp_to_distance(dp_sum, p);
        acc_a.update(0, d, k);
        acc_b.update(k, d, 0);

        for i in 1..diag_len {
            let j = i + k;
            dp_sum = dp_sum - pnorm_term(ts_a[i - 1], ts_b[j - 1], p)
                + pnorm_term(ts_a[i + m - 1], ts_b[j + m - 1], p);
            let d = dp_to_distance(dp_sum, p);
            acc_a.update(i, d, j);
            acc_b.update(j, d, i);
        }
    }

    // Negative diagonals: k = 1..n_a, pairs (i+k, i) with i+k in A, i in B
    for k in 1..n_a {
        let diag_len = n_b.min(n_a - k);

        let mut dp_sum = pnorm_init_ab(ts_a, ts_b, k, 0, m, p);
        let d = dp_to_distance(dp_sum, p);
        acc_a.update(k, d, 0);
        acc_b.update(0, d, k);

        for i in 1..diag_len {
            let a_idx = i + k;
            let b_idx = i;
            dp_sum = dp_sum - pnorm_term(ts_a[a_idx - 1], ts_b[b_idx - 1], p)
                + pnorm_term(ts_a[a_idx + m - 1], ts_b[b_idx + m - 1], p);
            let d = dp_to_distance(dp_sum, p);
            acc_a.update(a_idx, d, b_idx);
            acc_b.update(b_idx, d, a_idx);
        }
    }

    acc_a.write_to_join_profile(jp_a);
    acc_b.write_to_join_profile(jp_b);
}

/// Parallel AB-join with p-norm diagonal traversal.
#[cfg(feature = "parallel")]
#[allow(clippy::too_many_arguments)]
fn ab_join_pnorm_parallel(
    ts_a: &[f64],
    ts_b: &[f64],
    m: usize,
    n_a: usize,
    n_b: usize,
    p: f64,
    jp_a: &mut JoinProfile,
    jp_b: &mut JoinProfile,
) {
    use rayon::prelude::*;

    let n_threads = rayon::current_num_threads();

    // Positive diagonals
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
                let mut dp_sum = pnorm_init_ab(ts_a, ts_b, 0, k, m, p);
                let d = dp_to_distance(dp_sum, p);
                acc_a.update(0, d, k);
                acc_b.update(k, d, 0);

                for i in 1..diag_len {
                    let j = i + k;
                    dp_sum = dp_sum - pnorm_term(ts_a[i - 1], ts_b[j - 1], p)
                        + pnorm_term(ts_a[i + m - 1], ts_b[j + m - 1], p);
                    let d = dp_to_distance(dp_sum, p);
                    acc_a.update(i, d, j);
                    acc_b.update(j, d, i);
                }
            }

            (acc_a, acc_b)
        })
        .collect();

    // Negative diagonals
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
                let mut dp_sum = pnorm_init_ab(ts_a, ts_b, k, 0, m, p);
                let d = dp_to_distance(dp_sum, p);
                acc_a.update(k, d, 0);
                acc_b.update(0, d, k);

                for i in 1..diag_len {
                    let a_idx = i + k;
                    let b_idx = i;
                    dp_sum = dp_sum - pnorm_term(ts_a[a_idx - 1], ts_b[b_idx - 1], p)
                        + pnorm_term(ts_a[a_idx + m - 1], ts_b[b_idx + m - 1], p);
                    let d = dp_to_distance(dp_sum, p);
                    acc_a.update(a_idx, d, b_idx);
                    acc_b.update(b_idx, d, a_idx);
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algorithms::ab_join::ab_join;
    use crate::algorithms::stomp::stomp;

    #[test]
    fn test_pnorm_p2_matches_aamp() {
        let ts: Vec<f64> = (0..100).map(|i| (i as f64 * 0.3).sin()).collect();
        let config = MatrixProfileConfig::new(10);

        let mp_pnorm = stomp_pnorm(&ts, &config, 2.0);
        let mp_aamp = stomp::<AbsoluteEuclidean>(&ts, &config);

        // p=2 should delegate exactly to AAMP, so profiles should be bit-identical
        assert_eq!(mp_pnorm.profile, mp_aamp.profile);
        assert_eq!(mp_pnorm.profile_index, mp_aamp.profile_index);
    }

    #[test]
    fn test_pnorm_p1_manhattan() {
        // Hand-computed: ts = [1, 2, 3, 2, 1, 2, 3, 2], m = 3
        // Manhattan distance between [1,2,3] and [2,3,2] = |1-2|+|2-3|+|3-2| = 3
        // Manhattan distance between [1,2,3] and [1,2,3] (at idx 4) — wait, idx 4 is [1,2,3]
        // Actually ts[4..7] = [1,2,3], so d(0,4) = 0
        let ts = vec![1.0, 2.0, 3.0, 2.0, 1.0, 2.0, 3.0, 2.0];
        let config = MatrixProfileConfig::new(3);
        let mp = stomp_pnorm(&ts, &config, 1.0);

        // [1,2,3] at idx=0 and [1,2,3] at idx=4 → Manhattan distance = 0
        assert!(mp.profile[0] < 1e-10, "Expected ~0, got {}", mp.profile[0]);
        assert_eq!(mp.profile_index[0], 4);
    }

    #[test]
    fn test_pnorm_p3_hand_computed() {
        // ts = [0, 1, 2, 3, 0, 1, 2, 3], m = 2
        // d_3([0,1], [3,0]) = (|0-3|^3 + |1-0|^3)^(1/3) = (27+1)^(1/3) = 28^(1/3)
        // d_3([0,1], [0,1]) at idx 4 = 0
        let ts = vec![0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 2.0, 3.0];
        let config = MatrixProfileConfig::new(2);
        let mp = stomp_pnorm(&ts, &config, 3.0);

        // [0,1] at idx=0 and [0,1] at idx=4 → distance = 0
        assert!(mp.profile[0] < 1e-10, "Expected ~0, got {}", mp.profile[0]);
    }

    #[test]
    fn test_pnorm_recurrence_vs_naive() {
        // Compare diagonal recurrence against O(n*m) naive for p=1.5
        let ts: Vec<f64> = (0..50).map(|i| (i as f64 * 0.3).sin()).collect();
        let m = 8;
        let p = 1.5;
        let config = MatrixProfileConfig::new(m);
        let mp = stomp_pnorm(&ts, &config, p);
        let n_subs = ts.len() - m + 1;
        let exclusion_zone = config.exclusion_zone();

        // Naive computation
        for i in 0..n_subs {
            let mut best_d = f64::INFINITY;
            let mut best_j = 0;
            for j in 0..n_subs {
                if i.abs_diff(j) <= exclusion_zone {
                    continue;
                }
                let d: f64 = ts[i..i + m]
                    .iter()
                    .zip(&ts[j..j + m])
                    .map(|(a, b)| (a - b).abs().powf(p))
                    .sum::<f64>()
                    .powf(1.0 / p);
                if d < best_d {
                    best_d = d;
                    best_j = j;
                }
            }
            assert!(
                (mp.profile[i] - best_d).abs() < 1e-10,
                "Mismatch at i={}: recurrence={}, naive={} (best_j={})",
                i,
                mp.profile[i],
                best_d,
                best_j
            );
        }
    }

    #[test]
    fn test_pnorm_ab_join_p2_matches_aamp() {
        let ts_a: Vec<f64> = (0..50).map(|i| (i as f64 * 0.2).sin()).collect();
        let ts_b: Vec<f64> = (0..60).map(|i| (i as f64 * 0.3).cos()).collect();
        let m = 8;

        let (jp_a_pnorm, jp_b_pnorm) = ab_join_pnorm(&ts_a, &ts_b, m, 2.0);
        let (jp_a_aamp, jp_b_aamp) = ab_join::<AbsoluteEuclidean>(&ts_a, &ts_b, m);

        assert_eq!(jp_a_pnorm.distances, jp_a_aamp.distances);
        assert_eq!(jp_a_pnorm.indices, jp_a_aamp.indices);
        assert_eq!(jp_b_pnorm.distances, jp_b_aamp.distances);
        assert_eq!(jp_b_pnorm.indices, jp_b_aamp.indices);
    }

    #[test]
    fn test_pnorm_ab_join_vs_naive() {
        // Compare AB-join diagonal recurrence against O(n_a*n_b*m) naive for p=1.5
        let ts_a: Vec<f64> = (0..30).map(|i| (i as f64 * 0.2).sin()).collect();
        let ts_b: Vec<f64> = (0..40).map(|i| (i as f64 * 0.3).cos()).collect();
        let m = 6;
        let p = 1.5;

        let (jp_a, jp_b) = ab_join_pnorm(&ts_a, &ts_b, m, p);
        let n_a = ts_a.len() - m + 1;
        let n_b = ts_b.len() - m + 1;

        // Naive: for each i in A, find nearest j in B
        for i in 0..n_a {
            let mut best_d = f64::INFINITY;
            for j in 0..n_b {
                let d: f64 = ts_a[i..i + m]
                    .iter()
                    .zip(&ts_b[j..j + m])
                    .map(|(a, b)| (a - b).abs().powf(p))
                    .sum::<f64>()
                    .powf(1.0 / p);
                if d < best_d {
                    best_d = d;
                }
            }
            assert!(
                (jp_a.distances[i] - best_d).abs() < 1e-10,
                "AB jp_a mismatch at i={}: got {}, expected {}",
                i,
                jp_a.distances[i],
                best_d
            );
        }

        // Naive: for each j in B, find nearest i in A
        for j in 0..n_b {
            let mut best_d = f64::INFINITY;
            for i in 0..n_a {
                let d: f64 = ts_a[i..i + m]
                    .iter()
                    .zip(&ts_b[j..j + m])
                    .map(|(a, b)| (a - b).abs().powf(p))
                    .sum::<f64>()
                    .powf(1.0 / p);
                if d < best_d {
                    best_d = d;
                }
            }
            assert!(
                (jp_b.distances[j] - best_d).abs() < 1e-10,
                "AB jp_b mismatch at j={}: got {}, expected {}",
                j,
                jp_b.distances[j],
                best_d
            );
        }
    }
}
