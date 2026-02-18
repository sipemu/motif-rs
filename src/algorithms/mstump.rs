use crate::algorithms::common::sliding_dot_product;
#[cfg(feature = "parallel")]
use crate::algorithms::stomp::compute_diagonal_ranges;
use crate::core::matrix_profile::RollingStats;

/// Minimum number of subsequences before dispatching to parallel MSTUMP.
#[cfg(feature = "parallel")]
const MIN_PARALLEL_SUBS: usize = 256;

/// Multi-dimensional matrix profile result.
///
/// Computed by `mstump()`, which evaluates z-normalized Euclidean distances
/// across multiple dimensions simultaneously.
#[derive(Debug, Clone)]
pub struct MultiDimensionalProfile {
    /// Distance profiles: shape (d, n_subs).
    /// Row k contains the best (k+1)-dimensional cumulative average distance at each position.
    pub profile: Vec<Vec<f64>>,
    /// Index profiles: shape (d, n_subs).
    /// Nearest-neighbor index for each (dimension-count k, position).
    pub profile_index: Vec<Vec<usize>>,
    /// Number of dimensions.
    pub d: usize,
    /// Subsequence length.
    pub m: usize,
}

/// Read-only context shared across diagonal processing.
struct MstumpCtx<'a> {
    ts: &'a [&'a [f64]],
    stats: &'a [RollingStats],
    qt_first: &'a [Vec<f64>],
    d: usize,
    m_f: f64,
    sqrt_2m: f64,
    n_subs: usize,
}

/// Mutable state for profile accumulation.
struct MstumpAcc {
    profile: Vec<Vec<f64>>,
    profile_index: Vec<Vec<usize>>,
    dists: Vec<f64>,
    qt: Vec<f64>,
}

impl MstumpAcc {
    fn new(d: usize, n_subs: usize) -> Self {
        Self {
            profile: vec![vec![f64::INFINITY; n_subs]; d],
            profile_index: vec![vec![0usize; n_subs]; d],
            dists: vec![0.0; d],
            qt: vec![0.0; d],
        }
    }

    /// Merge another accumulator into this one (take element-wise minimum).
    fn merge(&mut self, other: &MstumpAcc) {
        let d = self.profile.len();
        let n_subs = self.profile[0].len();
        for k in 0..d {
            for j in 0..n_subs {
                if other.profile[k][j] < self.profile[k][j] {
                    self.profile[k][j] = other.profile[k][j];
                    self.profile_index[k][j] = other.profile_index[k][j];
                }
            }
        }
    }
}

/// Compute the multi-dimensional matrix profile using MSTUMP.
///
/// Uses diagonal traversal with QT recurrence (like 1D STOMP) instead of
/// per-row MASS calls, reducing complexity from O(d·n²·log n) to O(d·n²).
///
/// For `d` dimensions, computes a `(d, n_subs)` matrix where row `k` contains
/// the best `(k+1)`-dimensional cumulative average z-normalized Euclidean distance
/// at each position. The nearest neighbor at each row may differ.
///
/// # Arguments
/// * `ts` - Slice of time series slices, one per dimension (all same length)
/// * `m` - Subsequence length
///
/// # Panics
/// - If `ts` is empty
/// - If time series have different lengths
/// - If `n < 2*m` or `m < 2`
pub fn mstump(ts: &[&[f64]], m: usize) -> MultiDimensionalProfile {
    let d = ts.len();
    assert!(d >= 1, "Need at least one dimension");
    let n = ts[0].len();
    for (i, t) in ts.iter().enumerate() {
        assert_eq!(
            t.len(),
            n,
            "Dimension {i} has length {}, expected {n}",
            t.len()
        );
    }
    assert!(
        n >= 2 * m,
        "Time series length ({n}) must be >= 2*m ({})",
        2 * m
    );
    assert!(m >= 2, "Subsequence length must be >= 2");

    let n_subs = n - m + 1;
    let ez = (m as f64 / 4.0).ceil() as usize;
    let m_f = m as f64;

    // Precompute rolling stats for each dimension
    let stats: Vec<RollingStats> = ts.iter().map(|t| RollingStats::compute(t, m)).collect();

    // One FFT per dimension for initial QT (vs n_subs * d FFTs in the old approach)
    let qt_first: Vec<Vec<f64>> = (0..d)
        .map(|dim| sliding_dot_product(&ts[dim][0..m], ts[dim]))
        .collect();

    let cx = MstumpCtx {
        ts,
        stats: &stats,
        qt_first: &qt_first,
        d,
        m_f,
        sqrt_2m: (2.0 * m_f).sqrt(),
        n_subs,
    };

    let mut acc = MstumpAcc::new(d, n_subs);

    #[cfg(feature = "parallel")]
    if n_subs >= MIN_PARALLEL_SUBS {
        mstump_diagonal_parallel(&cx, ez, &mut acc);
    } else {
        mstump_diagonal(&cx, ez, &mut acc);
    }
    #[cfg(not(feature = "parallel"))]
    mstump_diagonal(&cx, ez, &mut acc);

    MultiDimensionalProfile {
        profile: acc.profile,
        profile_index: acc.profile_index,
        d,
        m,
    }
}

/// Process one (i, j) pair: compute per-dimension distances, sort, cumulative-average,
/// and update both sides of the profile.
#[inline(always)]
fn update_position(cx: &MstumpCtx<'_>, acc: &mut MstumpAcc, i: usize, j: usize) {
    let d = cx.d;

    // Compute per-dimension z-normalized Euclidean distances
    for dim in 0..d {
        let si = cx.stats[dim].m_sigma_inv[i];
        let sj = cx.stats[dim].m_sigma_inv[j];

        acc.dists[dim] = if si == 0.0 && sj == 0.0 {
            // Both constant subsequences -> identical
            0.0
        } else if si == 0.0 || sj == 0.0 {
            // One constant, one not -> maximally different
            cx.sqrt_2m
        } else {
            // neg_r = (m*mu_i*mu_j - QT) * m_sigma_inv_i * m_sigma_inv_j
            let neg_r = (cx.m_f * cx.stats[dim].mean[i])
                .mul_add(cx.stats[dim].mean[j], -acc.qt[dim])
                * si
                * sj;
            (2.0 * cx.m_f * (1.0 + neg_r)).max(0.0).sqrt()
        };
    }

    // Sort distances ascending (for small d, sort_unstable is optimal)
    acc.dists[..d].sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());

    // Cumulative average: update both (i, j) and (j, i) -- symmetric
    let mut cum_sum = 0.0;
    for k in 0..d {
        cum_sum += acc.dists[k];
        let cum_avg = cum_sum / (k + 1) as f64;

        if cum_avg < acc.profile[k][i] {
            acc.profile[k][i] = cum_avg;
            acc.profile_index[k][i] = j;
        }
        if cum_avg < acc.profile[k][j] {
            acc.profile[k][j] = cum_avg;
            acc.profile_index[k][j] = i;
        }
    }
}

/// Serial diagonal-traversal MSTUMP.
///
/// For each diagonal k (beyond the exclusion zone), walks along the diagonal
/// maintaining per-dimension QT values with O(1) recurrence updates.
fn mstump_diagonal(cx: &MstumpCtx<'_>, ez: usize, acc: &mut MstumpAcc) {
    let d = cx.d;
    let m = cx.m_f as usize;

    for k in (ez + 1)..cx.n_subs {
        let diag_len = cx.n_subs - k;

        // Initialize per-dimension QT from qt_first
        for (dim, qf) in cx.qt_first.iter().enumerate() {
            acc.qt[dim] = qf[k];
        }

        // p = 0: use qt_first directly
        update_position(cx, acc, 0, k);

        // p = 1..diag_len: QT recurrence per dimension
        for p in 1..diag_len {
            let j = p + k;
            for dim in 0..d {
                acc.qt[dim] = (-cx.ts[dim][p - 1]).mul_add(cx.ts[dim][j - 1], acc.qt[dim]);
                acc.qt[dim] = cx.ts[dim][p + m - 1].mul_add(cx.ts[dim][j + m - 1], acc.qt[dim]);
            }
            update_position(cx, acc, p, j);
        }
    }
}

/// Parallel diagonal-traversal MSTUMP with load-balanced chunking.
///
/// Partitions diagonals across threads (same strategy as 1D parallel STOMP),
/// each thread maintains its own accumulator, merged at the end.
#[cfg(feature = "parallel")]
fn mstump_diagonal_parallel(cx: &MstumpCtx<'_>, ez: usize, acc: &mut MstumpAcc) {
    use rayon::prelude::*;

    let d = cx.d;
    let m = cx.m_f as usize;
    let n_threads = rayon::current_num_threads();
    let ranges = compute_diagonal_ranges(ez + 1, cx.n_subs, n_threads);

    let results: Vec<MstumpAcc> = ranges
        .into_par_iter()
        .map(|(start_k, end_k)| {
            let mut local = MstumpAcc::new(d, cx.n_subs);

            for k in start_k..end_k {
                for (dim, qf) in cx.qt_first.iter().enumerate() {
                    local.qt[dim] = qf[k];
                }
                update_position(cx, &mut local, 0, k);

                for p in 1..(cx.n_subs - k) {
                    let j = p + k;
                    for dim in 0..d {
                        local.qt[dim] =
                            (-cx.ts[dim][p - 1]).mul_add(cx.ts[dim][j - 1], local.qt[dim]);
                        local.qt[dim] =
                            cx.ts[dim][p + m - 1].mul_add(cx.ts[dim][j + m - 1], local.qt[dim]);
                    }
                    update_position(cx, &mut local, p, j);
                }
            }

            local
        })
        .collect();

    // Merge thread-local results
    for result in &results {
        acc.merge(result);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algorithms::stomp::stomp;
    use crate::core::matrix_profile::MatrixProfileConfig;
    use crate::metrics::euclidean::ZNormalizedEuclidean;

    #[test]
    fn test_mstump_single_dimension() {
        // With d=1, P[0] should match the regular 1D matrix profile
        let ts: Vec<f64> = (0..100).map(|i| (i as f64 * 0.2).sin()).collect();
        let m = 10;

        let config = MatrixProfileConfig::new(m);
        let mp_1d = stomp::<ZNormalizedEuclidean>(&ts, &config);

        let ts_refs: [&[f64]; 1] = [&ts];
        let mdp = mstump(&ts_refs, m);

        assert_eq!(mdp.d, 1);
        assert_eq!(mdp.profile.len(), 1);
        assert_eq!(mdp.profile[0].len(), mp_1d.profile.len());

        // The distances should match within epsilon (different code paths may differ slightly)
        for (i, (a, b)) in mdp.profile[0].iter().zip(&mp_1d.profile).enumerate() {
            if a.is_infinite() && b.is_infinite() {
                continue;
            }
            assert!(
                (a - b).abs() < 1e-6,
                "Mismatch at {i}: mstump={a}, stomp={b}"
            );
        }
    }

    #[test]
    fn test_mstump_profile_nondecreasing() {
        // For each position i, P[k][i] should be non-decreasing in k
        // because cumulative average of sorted ascending sequence is non-decreasing
        let n = 100;
        let m = 10;
        let ts0: Vec<f64> = (0..n).map(|i| (i as f64 * 0.2).sin()).collect();
        let ts1: Vec<f64> = (0..n).map(|i| (i as f64 * 0.3).cos()).collect();
        let ts2: Vec<f64> = (0..n).map(|i| (i as f64 * 0.15).sin() + 0.5).collect();

        let ts_refs: [&[f64]; 3] = [&ts0, &ts1, &ts2];
        let mdp = mstump(&ts_refs, m);

        assert_eq!(mdp.d, 3);
        let n_subs = n - m + 1;
        for j in 0..n_subs {
            for k in 1..3 {
                assert!(
                    mdp.profile[k][j] >= mdp.profile[k - 1][j] - 1e-10,
                    "Profile not non-decreasing at position {j}: P[{}]={}, P[{}]={}",
                    k - 1,
                    mdp.profile[k - 1][j],
                    k,
                    mdp.profile[k][j]
                );
            }
        }
    }

    #[test]
    fn test_mstump_distances_nonnegative() {
        let n = 80;
        let m = 8;
        let ts0: Vec<f64> = (0..n).map(|i| (i as f64 * 0.3).sin()).collect();
        let ts1: Vec<f64> = (0..n).map(|i| (i as f64 * 0.5).cos()).collect();

        let ts_refs: [&[f64]; 2] = [&ts0, &ts1];
        let mdp = mstump(&ts_refs, m);

        for k in 0..mdp.d {
            for (j, &v) in mdp.profile[k].iter().enumerate() {
                assert!(
                    v >= 0.0 || v.is_infinite(),
                    "Negative distance at P[{k}][{j}] = {v}"
                );
            }
        }
    }

    #[test]
    fn test_mstump_output_shapes() {
        let n = 60;
        let m = 8;
        let d = 3;
        let ts0: Vec<f64> = (0..n).map(|i| (i as f64 * 0.2).sin()).collect();
        let ts1: Vec<f64> = (0..n).map(|i| (i as f64 * 0.3).cos()).collect();
        let ts2: Vec<f64> = (0..n).map(|i| (i as f64 * 0.1).sin() + 1.0).collect();

        let ts_refs: [&[f64]; 3] = [&ts0, &ts1, &ts2];
        let mdp = mstump(&ts_refs, m);

        let n_subs = n - m + 1;
        assert_eq!(mdp.d, d);
        assert_eq!(mdp.m, m);
        assert_eq!(mdp.profile.len(), d);
        assert_eq!(mdp.profile_index.len(), d);
        for k in 0..d {
            assert_eq!(mdp.profile[k].len(), n_subs);
            assert_eq!(mdp.profile_index[k].len(), n_subs);
        }
    }
}
