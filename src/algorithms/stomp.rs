use crate::algorithms::common::{apply_exclusion_zone, sliding_dot_product};
use crate::core::distance_metric::DistanceMetric;
use crate::core::matrix_profile::{MatrixProfile, MatrixProfileConfig, ProfileAccumulator};

/// Minimum number of subsequences before dispatching to parallel STOMP.
/// Below this threshold, thread-dispatch overhead exceeds parallelism gains.
#[cfg(feature = "parallel")]
const MIN_PARALLEL_SUBS: usize = 256;

/// Compute the matrix profile using the STOMP algorithm.
///
/// STOMP exploits the relationship between consecutive dot products:
/// `QT[i][j] = QT[i-1][j-1] - T[j-1]*T[i-1] + T[j+m-1]*T[i+m-1]`
///
/// This allows O(1) updates per element instead of O(m), giving O(n^2) total
/// instead of O(n^2 * m) for the naive approach.
///
/// Two paths:
/// - **Fast path**: When `M::supports_qt_optimization()` is true (e.g., Euclidean),
///   uses incremental QT updates and `qt_to_distance`.
/// - **Naive path**: For metrics without QT support, calls `distance_profile()` per row.
pub fn stomp<M: DistanceMetric>(ts: &[f64], config: &MatrixProfileConfig) -> MatrixProfile {
    let m = config.m;
    let n = ts.len();
    assert!(n >= m, "Time series length must be >= subsequence length");
    assert!(m >= 2, "Subsequence length must be >= 2");

    let n_subs = n - m + 1;
    let exclusion_zone = config.exclusion_zone();
    let ctx = M::precompute(ts, m);
    let mut mp = MatrixProfile::new(n_subs, m, exclusion_zone);

    if M::supports_qt_optimization() {
        if M::supports_correlation_domain() {
            #[cfg(feature = "parallel")]
            if n_subs >= MIN_PARALLEL_SUBS {
                stomp_diagonal_corr_parallel::<M>(ts, m, n_subs, exclusion_zone, &ctx, &mut mp);
            } else {
                stomp_diagonal_corr::<M>(ts, m, n_subs, exclusion_zone, &ctx, &mut mp);
            }
            #[cfg(not(feature = "parallel"))]
            stomp_diagonal_corr::<M>(ts, m, n_subs, exclusion_zone, &ctx, &mut mp);
        } else {
            #[cfg(feature = "parallel")]
            if n_subs >= MIN_PARALLEL_SUBS {
                stomp_diagonal_parallel::<M>(ts, m, n_subs, exclusion_zone, &ctx, &mut mp);
            } else {
                stomp_diagonal::<M>(ts, m, n_subs, exclusion_zone, &ctx, &mut mp);
            }
            #[cfg(not(feature = "parallel"))]
            stomp_diagonal::<M>(ts, m, n_subs, exclusion_zone, &ctx, &mut mp);
        }
    } else {
        #[cfg(feature = "parallel")]
        if n_subs >= MIN_PARALLEL_SUBS {
            stomp_naive_parallel::<M>(ts, m, n_subs, exclusion_zone, &ctx, &mut mp);
        } else {
            stomp_naive::<M>(ts, m, n_subs, exclusion_zone, &ctx, &mut mp);
        }
        #[cfg(not(feature = "parallel"))]
        stomp_naive::<M>(ts, m, n_subs, exclusion_zone, &ctx, &mut mp);
    }

    mp
}

/// Diagonal-traversal STOMP: processes diagonals of the distance matrix.
///
/// Each diagonal `k` contains pairs `(i, j)` where `j = i + k`. The QT recurrence
/// is applied along the diagonal: `QT[p] = QT[p-1] - T[p-1]*T[p+k-1] + T[p+m-1]*T[p+k+m-1]`.
///
/// Advantages over row-wise traversal:
/// - Only ONE FFT call total (for `qt_first`)
/// - Each diagonal is independent → naturally parallel
/// - Better cache locality (sequential access along diagonals)
/// - Exclusion zone handled by skipping diagonals `k <= exclusion_zone`
#[allow(dead_code, clippy::needless_range_loop)]
fn stomp_diagonal<M: DistanceMetric>(
    ts: &[f64],
    m: usize,
    n_subs: usize,
    exclusion_zone: usize,
    ctx: &M::Context,
    mp: &mut MatrixProfile,
) {
    let qt_first = sliding_dot_product(&ts[0..m], ts);

    for k in (exclusion_zone + 1)..n_subs {
        let diag_len = n_subs - k;
        let mut qt = qt_first[k];

        let d = M::qt_to_distance(qt, 0, k, m, ctx);
        mp.update(0, d, k);
        mp.update(k, d, 0);

        for i in 1..diag_len {
            let j = i + k;
            qt = qt - ts[i - 1] * ts[j - 1] + ts[i + m - 1] * ts[j + m - 1];
            let d = M::qt_to_distance(qt, i, j, m, ctx);
            mp.update(i, d, j);
            mp.update(j, d, i);
        }
    }
}

/// Partition diagonals into load-balanced chunks.
///
/// Returns `Vec<(start_k, end_k)>` ranges where each chunk has approximately equal
/// total work. Diagonal `k` has length `n_subs - k`, so earlier diagonals are longer.
/// Uses binary search over an analytical cumulative-work formula (same approach as
/// stumpy's `_get_array_ranges`).
#[cfg(feature = "parallel")]
pub fn compute_diagonal_ranges(
    first_diag: usize,
    n_subs: usize,
    n_chunks: usize,
) -> Vec<(usize, usize)> {
    let n_diags = n_subs.saturating_sub(first_diag);
    if n_diags == 0 || n_chunks == 0 {
        return vec![];
    }
    let n_chunks = n_chunks.min(n_diags);

    // Cumulative work for the first `i` diagonals (starting from first_diag):
    //   cumwork(i) = sum_{j=0}^{i-1} (n_diags - j) = i*n_diags - i*(i-1)/2
    let cumwork = |i: usize| -> usize { i * n_diags - i * i.saturating_sub(1) / 2 };
    let total_work = cumwork(n_diags);

    let mut ranges = Vec::with_capacity(n_chunks);
    let mut prev = 0usize;

    for c in 1..=n_chunks {
        let target = if c == n_chunks {
            n_diags
        } else {
            let threshold = (c as f64 * total_work as f64 / n_chunks as f64).round() as usize;
            let mut lo = prev;
            let mut hi = n_diags;
            while lo < hi {
                let mid = lo + (hi - lo) / 2;
                if cumwork(mid) >= threshold {
                    hi = mid;
                } else {
                    lo = mid + 1;
                }
            }
            lo
        };

        if target > prev {
            ranges.push((first_diag + prev, first_diag + target));
        }
        prev = target;
    }

    ranges
}

/// Parallel diagonal-traversal STOMP with load-balanced chunking.
///
/// Pre-allocates exactly `n_threads` MatrixProfile accumulators (instead of the
/// ~128-256 created by rayon's `fold`/`reduce`), with weight-balanced diagonal
/// partitioning so each thread gets approximately equal total work.
#[allow(dead_code, clippy::needless_range_loop)]
#[cfg(feature = "parallel")]
fn stomp_diagonal_parallel<M: DistanceMetric>(
    ts: &[f64],
    m: usize,
    n_subs: usize,
    exclusion_zone: usize,
    ctx: &M::Context,
    mp: &mut MatrixProfile,
) {
    use rayon::prelude::*;

    let qt_first = sliding_dot_product(&ts[0..m], ts);
    let n_threads = rayon::current_num_threads();
    let ranges = compute_diagonal_ranges(exclusion_zone + 1, n_subs, n_threads);

    let results: Vec<MatrixProfile> = ranges
        .into_par_iter()
        .map(|(start_k, end_k)| {
            let mut local_mp = MatrixProfile::new(n_subs, m, exclusion_zone);
            for k in start_k..end_k {
                let mut qt = qt_first[k];

                let d = M::qt_to_distance(qt, 0, k, m, ctx);
                local_mp.update(0, d, k);
                local_mp.update(k, d, 0);

                for i in 1..(n_subs - k) {
                    let j = i + k;
                    qt = qt - ts[i - 1] * ts[j - 1] + ts[i + m - 1] * ts[j + m - 1];
                    let d = M::qt_to_distance(qt, i, j, m, ctx);
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

// ---------------------------------------------------------------------------
// Correlation-domain inner loop helpers
//
// All use:
// - f64::mul_add() for FMA fusion (single-instruction multiply-add)
// - Hoisted p=0 (no branch in the inner loop)
// - Unsafe get_unchecked for bounds-check elimination
// - Multi-diagonal grouping (4 adjacent diagonals) for SIMD opportunities
// ---------------------------------------------------------------------------

/// Read-only context shared by all correlation-domain inner loop helpers.
struct CorrCtx<'a> {
    ts: &'a [f64],
    m: usize,
    n_subs: usize,
    qt_first: &'a [f64],
    mean: &'a [f64],
    m_sigma_inv: &'a [f64],
    m_mean: &'a [f64],
}

/// Process a single diagonal (branchless, no constant-subsequence checks).
///
/// Uses FMA, hoisted p=0, and unchecked indexing.
#[inline(always)]
fn diagonal_single_branchless(cx: &CorrCtx<'_>, k: usize, acc: &mut ProfileAccumulator) {
    let CorrCtx {
        ts,
        m,
        n_subs,
        qt_first,
        mean,
        m_sigma_inv,
        m_mean,
    } = *cx;
    debug_assert!(k < n_subs);
    let diag_len = n_subs - k;

    // SAFETY: All indices are in bounds:
    // - qt_first[k]: k < n_subs = qt_first.len()
    // - mean/m_sigma_inv/m_mean[0..n_subs-1]: p < diag_len <= n_subs, j = p+k < n_subs
    // - ts[p-1], ts[j-1]: p >= 1 so p-1 >= 0; j < n_subs so j-1 < n_subs-1 < n
    // - ts[p+m-1], ts[j+m-1]: p+m-1 < (n_subs-k)+m-1 = n-k <= n; j+m-1 < n_subs+m-1 = n
    unsafe {
        let qt_init = *qt_first.get_unchecked(k);

        // p = 0: use qt_first directly, no recurrence
        let neg_r = (*m_mean.get_unchecked(0)).mul_add(*mean.get_unchecked(k), -qt_init)
            * *m_sigma_inv.get_unchecked(0)
            * *m_sigma_inv.get_unchecked(k);
        acc.update_right(0, neg_r, k);
        acc.update_left(k, neg_r, 0);

        // p = 1..diag_len: QT recurrence + neg_r, no branches
        let mut qt = qt_init;
        for p in 1..diag_len {
            let j = p + k;
            qt = (-*ts.get_unchecked(p - 1)).mul_add(*ts.get_unchecked(j - 1), qt);
            qt = (*ts.get_unchecked(p + m - 1)).mul_add(*ts.get_unchecked(j + m - 1), qt);

            let neg_r = (*m_mean.get_unchecked(p)).mul_add(*mean.get_unchecked(j), -qt)
                * *m_sigma_inv.get_unchecked(p)
                * *m_sigma_inv.get_unchecked(j);
            acc.update_right(p, neg_r, j);
            acc.update_left(j, neg_r, p);
        }
    }
}

/// Process 4 adjacent diagonals simultaneously (branchless).
///
/// Diagonals k, k+1, k+2, k+3 share `ts[p-1]` and `ts[p+m-1]`, while
/// `ts[j-1..j+2]` and `ts[j+m-1..j+m+2]` are consecutive loads —
/// enabling the compiler to use packed AVX2 operations.
#[inline(always)]
#[allow(clippy::needless_range_loop)]
fn diagonal_group4_branchless(cx: &CorrCtx<'_>, k: usize, acc: &mut ProfileAccumulator) {
    let CorrCtx {
        ts,
        m,
        n_subs,
        qt_first,
        mean,
        m_sigma_inv,
        m_mean,
    } = *cx;
    debug_assert!(k + 3 < n_subs);
    let min_diag_len = n_subs - (k + 3);

    // SAFETY: Same bounds reasoning as diagonal_single_branchless, extended to k+3.
    // j_max = p + k + 3 < n_subs for p < min_diag_len. ts[j_max+m-1] < n.
    unsafe {
        // p = 0: initialize from qt_first
        let mut qt = [
            *qt_first.get_unchecked(k),
            *qt_first.get_unchecked(k + 1),
            *qt_first.get_unchecked(k + 2),
            *qt_first.get_unchecked(k + 3),
        ];

        let mm0 = *m_mean.get_unchecked(0);
        let si0 = *m_sigma_inv.get_unchecked(0);
        for d in 0..4usize {
            let kd = k + d;
            let neg_r =
                mm0.mul_add(*mean.get_unchecked(kd), -qt[d]) * si0 * *m_sigma_inv.get_unchecked(kd);
            acc.update_right(0, neg_r, kd);
            acc.update_left(kd, neg_r, 0);
        }

        // p = 1..min_diag_len: 4-wide recurrence + neg_r
        for p in 1..min_diag_len {
            let j_base = p + k;
            let neg_a = -*ts.get_unchecked(p - 1);
            let c = *ts.get_unchecked(p + m - 1);
            let mm = *m_mean.get_unchecked(p);
            let si = *m_sigma_inv.get_unchecked(p);

            // Inner loop over 4 lanes — consecutive loads from ts/mean/m_sigma_inv
            // enable the compiler to use packed AVX2 FMA.
            for d in 0..4usize {
                let j = j_base + d;
                qt[d] = neg_a.mul_add(*ts.get_unchecked(j - 1), qt[d]);
                qt[d] = c.mul_add(*ts.get_unchecked(j + m - 1), qt[d]);
                let neg_r =
                    mm.mul_add(*mean.get_unchecked(j), -qt[d]) * si * *m_sigma_inv.get_unchecked(j);
                acc.update_right(p, neg_r, j);
                acc.update_left(j, neg_r, p);
            }
        }

        // Tail: each diagonal may extend 0-3 elements beyond min_diag_len
        for d in 0..4usize {
            let kd = k + d;
            let diag_len = n_subs - kd;
            let mut qt_d = qt[d];
            for p in min_diag_len..diag_len {
                let j = p + kd;
                qt_d = (-*ts.get_unchecked(p - 1)).mul_add(*ts.get_unchecked(j - 1), qt_d);
                qt_d = (*ts.get_unchecked(p + m - 1)).mul_add(*ts.get_unchecked(j + m - 1), qt_d);
                let neg_r = (*m_mean.get_unchecked(p)).mul_add(*mean.get_unchecked(j), -qt_d)
                    * *m_sigma_inv.get_unchecked(p)
                    * *m_sigma_inv.get_unchecked(j);
                acc.update_right(p, neg_r, j);
                acc.update_left(j, neg_r, p);
            }
        }
    }
}

/// Process a single diagonal with constant-subsequence handling.
///
/// Uses FMA, hoisted p=0, and unchecked indexing, but checks m_sigma_inv
/// for zeros (constant subsequences).
#[inline(always)]
fn diagonal_single_branching(cx: &CorrCtx<'_>, k: usize, acc: &mut ProfileAccumulator) {
    let CorrCtx {
        ts,
        m,
        n_subs,
        qt_first,
        mean,
        m_sigma_inv,
        m_mean,
    } = *cx;
    debug_assert!(k < n_subs);
    let diag_len = n_subs - k;

    unsafe {
        let qt_init = *qt_first.get_unchecked(k);

        // p = 0
        let si = *m_sigma_inv.get_unchecked(0);
        let sj = *m_sigma_inv.get_unchecked(k);
        let neg_r = if si == 0.0 && sj == 0.0 {
            -1.0
        } else if si == 0.0 || sj == 0.0 {
            0.0
        } else {
            (*m_mean.get_unchecked(0)).mul_add(*mean.get_unchecked(k), -qt_init) * si * sj
        };
        acc.update_right(0, neg_r, k);
        acc.update_left(k, neg_r, 0);

        let mut qt = qt_init;
        for p in 1..diag_len {
            let j = p + k;
            qt = (-*ts.get_unchecked(p - 1)).mul_add(*ts.get_unchecked(j - 1), qt);
            qt = (*ts.get_unchecked(p + m - 1)).mul_add(*ts.get_unchecked(j + m - 1), qt);

            let si = *m_sigma_inv.get_unchecked(p);
            let sj = *m_sigma_inv.get_unchecked(j);
            let neg_r = if si == 0.0 && sj == 0.0 {
                -1.0
            } else if si == 0.0 || sj == 0.0 {
                0.0
            } else {
                (*m_mean.get_unchecked(p)).mul_add(*mean.get_unchecked(j), -qt) * si * sj
            };
            acc.update_right(p, neg_r, j);
            acc.update_left(j, neg_r, p);
        }
    }
}

/// Process a range of diagonals using the branchless path with 4-wide grouping.
#[inline(always)]
fn process_diags_branchless(
    cx: &CorrCtx<'_>,
    start_k: usize,
    end_k: usize,
    acc: &mut ProfileAccumulator,
) {
    let mut k = start_k;
    // Process groups of 4 adjacent diagonals
    while k + 4 <= end_k {
        diagonal_group4_branchless(cx, k, acc);
        k += 4;
    }
    // Remaining 0-3 diagonals
    while k < end_k {
        diagonal_single_branchless(cx, k, acc);
        k += 1;
    }
}

/// Process a range of diagonals using the branching path (handles constants).
#[inline(always)]
fn process_diags_branching(
    cx: &CorrCtx<'_>,
    start_k: usize,
    end_k: usize,
    acc: &mut ProfileAccumulator,
) {
    for k in start_k..end_k {
        diagonal_single_branching(cx, k, acc);
    }
}

// ---------------------------------------------------------------------------
// Public correlation-domain STOMP entry points
// ---------------------------------------------------------------------------

/// Correlation-domain diagonal STOMP (serial).
///
/// Inner loop optimizations:
/// - Negated Pearson correlations (defers sqrt to O(n) final pass)
/// - `f64::mul_add()` for FMA fusion on QT recurrence and neg_r computation
/// - Hoisted p=0 (no branch in inner loop)
/// - Unchecked indexing (bounds verified by invariants, checked in debug)
/// - 4-wide diagonal grouping (consecutive loads enable AVX2 auto-vectorization)
/// - AoS `ProfileAccumulator` (48-byte cache-line-friendly entries)
/// - Specialized `update_right`/`update_left` (no direction-check branches)
fn stomp_diagonal_corr<M: DistanceMetric>(
    ts: &[f64],
    m: usize,
    n_subs: usize,
    exclusion_zone: usize,
    ctx: &M::Context,
    mp: &mut MatrixProfile,
) {
    let qt_first = sliding_dot_product(&ts[0..m], ts);
    let (mean, m_sigma_inv, has_constant) = M::correlation_data(ctx);
    let m_f = m as f64;
    let m_mean: Vec<f64> = mean.iter().map(|&mu| m_f * mu).collect();

    let cx = CorrCtx {
        ts,
        m,
        n_subs,
        qt_first: &qt_first,
        mean,
        m_sigma_inv,
        m_mean: &m_mean,
    };
    let mut acc = ProfileAccumulator::new(n_subs);
    let first_k = exclusion_zone + 1;

    if !has_constant {
        process_diags_branchless(&cx, first_k, n_subs, &mut acc);
    } else {
        process_diags_branching(&cx, first_k, n_subs, &mut acc);
    }

    let two_m = 2.0 * m_f;
    acc.write_to_matrix_profile(mp, |nc| (two_m * (1.0 + nc)).max(0.0).sqrt());
}

/// Parallel correlation-domain diagonal STOMP with load-balanced chunking.
///
/// Same optimizations as `stomp_diagonal_corr`, partitioned across threads.
#[cfg(feature = "parallel")]
fn stomp_diagonal_corr_parallel<M: DistanceMetric>(
    ts: &[f64],
    m: usize,
    n_subs: usize,
    exclusion_zone: usize,
    ctx: &M::Context,
    mp: &mut MatrixProfile,
) {
    use rayon::prelude::*;

    let qt_first = sliding_dot_product(&ts[0..m], ts);
    let (mean, m_sigma_inv, has_constant) = M::correlation_data(ctx);
    let m_f = m as f64;
    let m_mean: Vec<f64> = mean.iter().map(|&mu| m_f * mu).collect();

    let cx = CorrCtx {
        ts,
        m,
        n_subs,
        qt_first: &qt_first,
        mean,
        m_sigma_inv,
        m_mean: &m_mean,
    };
    let n_threads = rayon::current_num_threads();
    let ranges = compute_diagonal_ranges(exclusion_zone + 1, n_subs, n_threads);

    let results: Vec<ProfileAccumulator> = ranges
        .into_par_iter()
        .map(|(start_k, end_k)| {
            let mut acc = ProfileAccumulator::new(n_subs);
            if !has_constant {
                process_diags_branchless(&cx, start_k, end_k, &mut acc);
            } else {
                process_diags_branching(&cx, start_k, end_k, &mut acc);
            }
            acc
        })
        .collect();

    let mut combined = ProfileAccumulator::new(n_subs);
    for result in &results {
        combined.merge(result);
    }

    let two_m = 2.0 * m_f;
    combined.write_to_matrix_profile(mp, |nc| (two_m * (1.0 + nc)).max(0.0).sqrt());
}

/// Compute the matrix profile using the original row-wise STOMP algorithm.
///
/// Exposed for benchmarking comparison against the diagonal traversal.
/// Uses `stomp_qt` (serial) or `stomp_qt_parallel` (parallel) internally.
pub fn stomp_rowwise<M: DistanceMetric>(ts: &[f64], config: &MatrixProfileConfig) -> MatrixProfile {
    let m = config.m;
    let n = ts.len();
    assert!(n >= m, "Time series length must be >= subsequence length");
    assert!(m >= 2, "Subsequence length must be >= 2");

    let n_subs = n - m + 1;
    let exclusion_zone = config.exclusion_zone();
    let ctx = M::precompute(ts, m);
    let mut mp = MatrixProfile::new(n_subs, m, exclusion_zone);

    #[cfg(feature = "parallel")]
    if n_subs >= MIN_PARALLEL_SUBS {
        stomp_qt_parallel::<M>(ts, m, n_subs, exclusion_zone, &ctx, &mut mp);
    } else {
        stomp_qt::<M>(ts, m, n_subs, exclusion_zone, &ctx, &mut mp);
    }
    #[cfg(not(feature = "parallel"))]
    stomp_qt::<M>(ts, m, n_subs, exclusion_zone, &ctx, &mut mp);

    mp
}

/// QT-optimized STOMP path (row-wise traversal).
#[allow(dead_code)]
fn stomp_qt<M: DistanceMetric>(
    ts: &[f64],
    m: usize,
    n_subs: usize,
    exclusion_zone: usize,
    ctx: &M::Context,
    mp: &mut MatrixProfile,
) {
    // First row: compute QT_first[j] = dot(T[0..m], T[j..j+m]) for all j
    let qt_first = sliding_dot_product(&ts[0..m], ts);

    // Process first row: distance profile for subsequence 0
    let mut dist_profile: Vec<f64> = (0..n_subs)
        .map(|j| M::qt_to_distance(qt_first[j], 0, j, m, ctx))
        .collect();
    apply_exclusion_zone(&mut dist_profile, 0, exclusion_zone);

    // Update MP from first row (both directions for symmetry)
    for (j, &d) in dist_profile.iter().enumerate() {
        if d < mp.profile[j] {
            mp.update(j, d, 0);
        }
        if d < mp.profile[0] {
            mp.update(0, d, j);
        }
    }

    // Subsequent rows: use QT recurrence
    let mut qt = qt_first.clone();

    for i in 1..n_subs {
        // QT update: shift and update
        // QT_new[j] = QT_old[j-1] - T[j-1]*T[i-1] + T[j+m-1]*T[i+m-1]
        // Process right-to-left to avoid overwriting needed values
        for j in (1..n_subs).rev() {
            qt[j] = qt[j - 1] - ts[j - 1] * ts[i - 1] + ts[j + m - 1] * ts[i + m - 1];
        }
        // QT[0] for this row needs to be recomputed from QT_first
        qt[0] = qt_first[i];

        // Convert QT to distances
        for (j, dp) in dist_profile.iter_mut().enumerate() {
            *dp = M::qt_to_distance(qt[j], i, j, m, ctx);
        }
        apply_exclusion_zone(&mut dist_profile, i, exclusion_zone);

        // Update MP with symmetry exploitation
        for (j, &d) in dist_profile.iter().enumerate() {
            mp.update(i, d, j);
            mp.update(j, d, i);
        }
    }
}

/// Naive STOMP path for metrics without QT optimization.
fn stomp_naive<M: DistanceMetric>(
    ts: &[f64],
    m: usize,
    n_subs: usize,
    exclusion_zone: usize,
    ctx: &M::Context,
    mp: &mut MatrixProfile,
) {
    for i in 0..n_subs {
        let mut dist_profile = M::distance_profile(ts, i, m, ctx);
        apply_exclusion_zone(&mut dist_profile, i, exclusion_zone);

        for (j, &d) in dist_profile.iter().enumerate() {
            mp.update(i, d, j);
        }
    }
}

/// Parallel QT-optimized STOMP (row-wise): chunks rows across threads.
#[allow(dead_code)]
#[cfg(feature = "parallel")]
fn stomp_qt_parallel<M: DistanceMetric>(
    ts: &[f64],
    m: usize,
    n_subs: usize,
    exclusion_zone: usize,
    ctx: &M::Context,
    mp: &mut MatrixProfile,
) {
    use rayon::prelude::*;

    // Compute QT_first once, shared read-only across all threads
    let qt_first = sliding_dot_product(&ts[0..m], ts);

    let n_threads = rayon::current_num_threads();
    let chunk_size = n_subs.div_ceil(n_threads);

    let result = (0..n_threads)
        .into_par_iter()
        .map(|t| {
            let start = t * chunk_size;
            let end = (start + chunk_size).min(n_subs);
            if start >= n_subs {
                return MatrixProfile::new(n_subs, m, exclusion_zone);
            }

            let mut local_mp = MatrixProfile::new(n_subs, m, exclusion_zone);

            // Thread 0 reuses qt_first; others compute fresh sliding dot product
            let mut qt = if start == 0 {
                qt_first.clone()
            } else {
                sliding_dot_product(&ts[start..start + m], ts)
            };
            let mut dist_profile = vec![0.0; n_subs];

            for i in start..end {
                if i > start {
                    // QT recurrence (right-to-left to avoid overwriting)
                    for j in (1..n_subs).rev() {
                        qt[j] = qt[j - 1] - ts[j - 1] * ts[i - 1] + ts[j + m - 1] * ts[i + m - 1];
                    }
                    qt[0] = qt_first[i];
                }

                for (j, dp) in dist_profile.iter_mut().enumerate() {
                    *dp = M::qt_to_distance(qt[j], i, j, m, ctx);
                }
                apply_exclusion_zone(&mut dist_profile, i, exclusion_zone);

                for (j, &d) in dist_profile.iter().enumerate() {
                    local_mp.update(i, d, j);
                    local_mp.update(j, d, i);
                }
            }
            local_mp
        })
        .reduce(
            || MatrixProfile::new(n_subs, m, exclusion_zone),
            |mut a, b| {
                a.merge(&b);
                a
            },
        );

    mp.merge(&result);
}

/// Parallel naive STOMP: each row is fully independent, trivially parallel.
#[cfg(feature = "parallel")]
fn stomp_naive_parallel<M: DistanceMetric>(
    ts: &[f64],
    m: usize,
    n_subs: usize,
    exclusion_zone: usize,
    ctx: &M::Context,
    mp: &mut MatrixProfile,
) {
    use rayon::prelude::*;

    let result = (0..n_subs)
        .into_par_iter()
        .fold(
            || MatrixProfile::new(n_subs, m, exclusion_zone),
            |mut local_mp, i| {
                let mut dp = M::distance_profile(ts, i, m, ctx);
                apply_exclusion_zone(&mut dp, i, exclusion_zone);
                for (j, &d) in dp.iter().enumerate() {
                    local_mp.update(i, d, j);
                }
                local_mp
            },
        )
        .reduce(
            || MatrixProfile::new(n_subs, m, exclusion_zone),
            |mut a, b| {
                a.merge(&b);
                a
            },
        );

    mp.merge(&result);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metrics::euclidean::ZNormalizedEuclidean;

    #[test]
    fn test_stomp_tiny_repeating() {
        // A simple repeating pattern: distances should be small for similar subsequences
        let ts = vec![1.0, 2.0, 3.0, 2.0, 1.0, 2.0, 3.0, 2.0];
        let config = MatrixProfileConfig::new(4);
        let mp = stomp::<ZNormalizedEuclidean>(&ts, &config);

        // Subsequences [1,2,3,2] at index 0 and [1,2,3,2] at index 4 are identical
        // Their z-normalized distance should be 0
        assert!(
            mp.profile[0] < 1e-6,
            "Identical subsequence distance should be ~0, got {}",
            mp.profile[0]
        );
        assert!(
            mp.profile[4] < 1e-6,
            "Identical subsequence distance should be ~0, got {}",
            mp.profile[4]
        );
    }

    #[test]
    fn test_stomp_linear() {
        // Linearly increasing: all subsequences have same shape → all distances ≈ 0
        let ts: Vec<f64> = (0..20).map(|i| i as f64).collect();
        let config = MatrixProfileConfig::new(4);
        let mp = stomp::<ZNormalizedEuclidean>(&ts, &config);

        for (i, &d) in mp.profile.iter().enumerate() {
            assert!(
                d < 1e-6,
                "Linear series: all distances should be ~0, got {d} at index {i}"
            );
        }
    }

    #[test]
    fn test_stomp_symmetry() {
        // If mp[i] says nearest neighbor is j, then mp[j] should say i (or have same distance)
        let ts = vec![1.0, 3.0, 2.0, 4.0, 1.5, 3.5, 2.5, 1.0, 3.0, 2.0, 4.0, 1.0];
        let config = MatrixProfileConfig::new(3);
        let mp = stomp::<ZNormalizedEuclidean>(&ts, &config);

        for i in 0..mp.profile.len() {
            let j = mp.profile_index[i];
            let d_ij = mp.profile[i];
            // The distance should be consistent
            let ctx = ZNormalizedEuclidean::precompute(&ts, 3);
            let d_check = ZNormalizedEuclidean::distance(&ts, i, j, 3, &ctx);
            assert!(
                (d_ij - d_check).abs() < 1e-9,
                "Distance mismatch at i={i}: profile says {d_ij}, direct says {d_check}"
            );
        }
    }

    #[test]
    fn test_stomp_known_motif() {
        // Construct a series with an obvious motif pair and noise elsewhere
        // Pattern: [0, 1, 0, -1] appears at positions 0 and 10
        let mut ts = vec![0.0; 20];
        // Place pattern at index 0
        ts[0] = 0.0;
        ts[1] = 1.0;
        ts[2] = 0.0;
        ts[3] = -1.0;
        // Fill middle with different pattern
        for (i, val) in ts.iter_mut().enumerate().take(10).skip(4) {
            *val = (i as f64) * 0.5;
        }
        // Place same pattern at index 10
        ts[10] = 0.0;
        ts[11] = 1.0;
        ts[12] = 0.0;
        ts[13] = -1.0;
        // Fill rest
        for (i, val) in ts.iter_mut().enumerate().take(20).skip(14) {
            *val = -(i as f64) * 0.3;
        }

        let config = MatrixProfileConfig::new(4);
        let mp = stomp::<ZNormalizedEuclidean>(&ts, &config);

        // The motif pair (0, 10) should find each other
        assert_eq!(mp.profile_index[0], 10, "Index 0 should match index 10");
        assert_eq!(mp.profile_index[10], 0, "Index 10 should match index 0");
        assert!(mp.profile[0] < 1e-6, "Motif pair distance should be ~0");
    }

    #[test]
    fn test_diagonal_matches_rowwise() {
        // Compare diagonal (stomp) vs row-wise (stomp_rowwise) on a non-trivial series
        let ts: Vec<f64> = (0..100).map(|i| (i as f64 * 0.3).sin()).collect();
        let config = MatrixProfileConfig::new(10);

        let mp_diag = stomp::<ZNormalizedEuclidean>(&ts, &config);
        let mp_row = stomp_rowwise::<ZNormalizedEuclidean>(&ts, &config);

        // FMA (mul_add) in the diagonal path changes rounding vs the rowwise path,
        // so we use a relaxed epsilon. Results are still well within 1e-6.
        let eps = 1e-7;
        let close = |a: f64, b: f64| -> bool {
            if a.is_infinite() && b.is_infinite() && a.signum() == b.signum() {
                return true;
            }
            (a - b).abs() < eps
        };
        for (i, (&pd, &pr)) in mp_diag
            .profile
            .iter()
            .zip(mp_row.profile.iter())
            .enumerate()
        {
            assert!(
                close(pd, pr),
                "profile mismatch at {i}: diag={pd}, row={pr}"
            );
        }
        for (i, (&ld, &lr)) in mp_diag
            .left_profile
            .iter()
            .zip(mp_row.left_profile.iter())
            .enumerate()
        {
            assert!(
                close(ld, lr),
                "left_profile mismatch at {i}: diag={ld}, row={lr}"
            );
        }
        for (i, (&rd, &rr)) in mp_diag
            .right_profile
            .iter()
            .zip(mp_row.right_profile.iter())
            .enumerate()
        {
            assert!(
                close(rd, rr),
                "right_profile mismatch at {i}: diag={rd}, row={rr}"
            );
        }
    }

    #[test]
    fn test_diagonal_small_series() {
        // Edge case: n=5, m=3 → n_subs=3, exclusion_zone=1
        // Only diagonal k=2 is processed (k > exclusion_zone=1)
        let ts = vec![1.0, 2.0, 3.0, 1.0, 2.0];
        let config = MatrixProfileConfig::new(3);
        let mp = stomp::<ZNormalizedEuclidean>(&ts, &config);

        assert_eq!(mp.profile.len(), 3);
        // With exclusion_zone=1 and n_subs=3, only pair (0,2) is valid
        // Subsequences [1,2,3] and [3,1,2] — different shapes, non-zero distance
        assert!(mp.profile[0].is_finite(), "Should find a match for index 0");
        assert!(mp.profile[2].is_finite(), "Should find a match for index 2");
    }

    #[test]
    fn test_diagonal_exclusion_zone_skips_correctly() {
        // Verify that no nearest-neighbor match falls within the exclusion zone
        let ts: Vec<f64> = (0..50).map(|i| (i as f64 * 0.7).cos()).collect();
        let config = MatrixProfileConfig::new(8);
        let mp = stomp::<ZNormalizedEuclidean>(&ts, &config);

        let exclusion_zone = config.exclusion_zone();
        for (i, (&d, &j)) in mp.profile.iter().zip(mp.profile_index.iter()).enumerate() {
            if d.is_finite() {
                let gap = j.abs_diff(i);
                assert!(
                    gap > exclusion_zone,
                    "Match at i={i}, j={j} (gap={gap}) violates exclusion_zone={exclusion_zone}"
                );
            }
        }
    }
}
