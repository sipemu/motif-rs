use crate::algorithms::common::sliding_dot_product;
use crate::core::distance_metric::DistanceMetric;
use crate::core::matrix_profile::{JoinAccumulator, JoinAccumulatorDist, JoinProfile};

/// Minimum number of subsequences before dispatching to parallel AB-join.
#[cfg(feature = "parallel")]
pub const MIN_PARALLEL_SUBS: usize = 256;

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

// ---------------------------------------------------------------------------
// Correlation-domain AB-join inner loop helpers
//
// Mirrors the self-join STOMP optimizations from stomp.rs:
// - f64::mul_add() for FMA fusion on QT recurrence and neg_r
// - Hoisted p=0 (no branch in inner loop)
// - Unsafe get_unchecked for bounds-check elimination
// - 4-wide diagonal grouping for SIMD opportunities
// - Pre-computed m_mean_a/b for FMA-friendly neg_r computation
// - Branchless/branching dispatch based on has_constant
// ---------------------------------------------------------------------------

/// Read-only context shared by all AB-join correlation-domain inner loop helpers.
struct CorrCtxAB<'a> {
    ts_a: &'a [f64],
    ts_b: &'a [f64],
    m: usize,
    n_a: usize,
    n_b: usize,
    qt_first_pos: &'a [f64],
    qt_first_neg: &'a [f64],
    mean_a: &'a [f64],
    mean_b: &'a [f64],
    m_sigma_inv_a: &'a [f64],
    m_sigma_inv_b: &'a [f64],
    m_mean_a: &'a [f64],
    m_mean_b: &'a [f64],
}

// --- Positive diagonal helpers (k = 0..n_b, step p: i=p, j=p+k) ---

/// Process a single positive diagonal (branchless, no constant-subsequence checks).
#[inline(always)]
fn ab_diag_pos_single_branchless(
    cx: &CorrCtxAB<'_>,
    k: usize,
    acc_a: &mut JoinAccumulator,
    acc_b: &mut JoinAccumulator,
) {
    let diag_len = cx.n_a.min(cx.n_b - k);
    // SAFETY: All indices are in bounds:
    // - qt_first_pos[k]: k < n_b = qt_first_pos.len()
    // - mean_a/msi_a/m_mean_a[p]: p < diag_len <= n_a
    // - mean_b/msi_b[j]: j = p+k < n_a+n_b <= ts_b.len(), actually j < n_b
    // - ts_a[p-1..p+m-1]: p >= 1, p+m-1 < n_a+m-1 = ts_a.len()
    // - ts_b[j-1..j+m-1]: j >= 1, j+m-1 < n_b+m-1 = ts_b.len()
    unsafe {
        let qt_init = *cx.qt_first_pos.get_unchecked(k);
        let neg_r = (*cx.m_mean_a.get_unchecked(0)).mul_add(*cx.mean_b.get_unchecked(k), -qt_init)
            * *cx.m_sigma_inv_a.get_unchecked(0)
            * *cx.m_sigma_inv_b.get_unchecked(k);
        acc_a.update(0, neg_r, k);
        acc_b.update(k, neg_r, 0);

        let mut qt = qt_init;
        for p in 1..diag_len {
            let j = p + k;
            qt = (-*cx.ts_a.get_unchecked(p - 1)).mul_add(*cx.ts_b.get_unchecked(j - 1), qt);
            qt = (*cx.ts_a.get_unchecked(p + cx.m - 1))
                .mul_add(*cx.ts_b.get_unchecked(j + cx.m - 1), qt);
            let neg_r = (*cx.m_mean_a.get_unchecked(p)).mul_add(*cx.mean_b.get_unchecked(j), -qt)
                * *cx.m_sigma_inv_a.get_unchecked(p)
                * *cx.m_sigma_inv_b.get_unchecked(j);
            acc_a.update(p, neg_r, j);
            acc_b.update(j, neg_r, p);
        }
    }
}

/// Process a single positive diagonal with constant-subsequence handling.
#[inline(always)]
fn ab_diag_pos_single_branching(
    cx: &CorrCtxAB<'_>,
    k: usize,
    acc_a: &mut JoinAccumulator,
    acc_b: &mut JoinAccumulator,
) {
    let diag_len = cx.n_a.min(cx.n_b - k);
    unsafe {
        let qt_init = *cx.qt_first_pos.get_unchecked(k);
        let si = *cx.m_sigma_inv_a.get_unchecked(0);
        let sj = *cx.m_sigma_inv_b.get_unchecked(k);
        let neg_r = if si == 0.0 && sj == 0.0 {
            -1.0
        } else if si == 0.0 || sj == 0.0 {
            0.0
        } else {
            (*cx.m_mean_a.get_unchecked(0)).mul_add(*cx.mean_b.get_unchecked(k), -qt_init) * si * sj
        };
        acc_a.update(0, neg_r, k);
        acc_b.update(k, neg_r, 0);

        let mut qt = qt_init;
        for p in 1..diag_len {
            let j = p + k;
            qt = (-*cx.ts_a.get_unchecked(p - 1)).mul_add(*cx.ts_b.get_unchecked(j - 1), qt);
            qt = (*cx.ts_a.get_unchecked(p + cx.m - 1))
                .mul_add(*cx.ts_b.get_unchecked(j + cx.m - 1), qt);
            let si = *cx.m_sigma_inv_a.get_unchecked(p);
            let sj = *cx.m_sigma_inv_b.get_unchecked(j);
            let neg_r = if si == 0.0 && sj == 0.0 {
                -1.0
            } else if si == 0.0 || sj == 0.0 {
                0.0
            } else {
                (*cx.m_mean_a.get_unchecked(p)).mul_add(*cx.mean_b.get_unchecked(j), -qt) * si * sj
            };
            acc_a.update(p, neg_r, j);
            acc_b.update(j, neg_r, p);
        }
    }
}

/// Process 4 adjacent positive diagonals simultaneously (branchless).
///
/// Diagonals k, k+1, k+2, k+3 share `ts_a[p-1]` and `ts_a[p+m-1]`, while
/// `ts_b[j-1..j+2]` and `ts_b[j+m-1..j+m+2]` are consecutive loads —
/// enabling the compiler to use packed AVX2 operations.
#[inline(always)]
#[allow(clippy::needless_range_loop)]
fn ab_diag_pos_group4_branchless(
    cx: &CorrCtxAB<'_>,
    k: usize,
    acc_a: &mut JoinAccumulator,
    acc_b: &mut JoinAccumulator,
) {
    let min_diag_len = cx.n_a.min(cx.n_b - k - 3);
    unsafe {
        let mut qt = [
            *cx.qt_first_pos.get_unchecked(k),
            *cx.qt_first_pos.get_unchecked(k + 1),
            *cx.qt_first_pos.get_unchecked(k + 2),
            *cx.qt_first_pos.get_unchecked(k + 3),
        ];

        // p = 0: shared A-side values
        let mm0 = *cx.m_mean_a.get_unchecked(0);
        let si0 = *cx.m_sigma_inv_a.get_unchecked(0);
        for d in 0..4usize {
            let j = k + d;
            let neg_r = mm0.mul_add(*cx.mean_b.get_unchecked(j), -qt[d])
                * si0
                * *cx.m_sigma_inv_b.get_unchecked(j);
            acc_a.update(0, neg_r, j);
            acc_b.update(j, neg_r, 0);
        }

        // p = 1..min_diag_len: 4-wide recurrence + neg_r
        for p in 1..min_diag_len {
            let j_base = p + k;
            let neg_a = -*cx.ts_a.get_unchecked(p - 1);
            let c = *cx.ts_a.get_unchecked(p + cx.m - 1);
            let mm = *cx.m_mean_a.get_unchecked(p);
            let si = *cx.m_sigma_inv_a.get_unchecked(p);

            for d in 0..4usize {
                let j = j_base + d;
                qt[d] = neg_a.mul_add(*cx.ts_b.get_unchecked(j - 1), qt[d]);
                qt[d] = c.mul_add(*cx.ts_b.get_unchecked(j + cx.m - 1), qt[d]);
                let neg_r = mm.mul_add(*cx.mean_b.get_unchecked(j), -qt[d])
                    * si
                    * *cx.m_sigma_inv_b.get_unchecked(j);
                acc_a.update(p, neg_r, j);
                acc_b.update(j, neg_r, p);
            }
        }

        // Tail: each diagonal may extend 0-3 elements beyond min_diag_len
        for d in 0..4usize {
            let kd = k + d;
            let diag_len = cx.n_a.min(cx.n_b - kd);
            let mut qt_d = qt[d];
            for p in min_diag_len..diag_len {
                let j = p + kd;
                qt_d =
                    (-*cx.ts_a.get_unchecked(p - 1)).mul_add(*cx.ts_b.get_unchecked(j - 1), qt_d);
                qt_d = (*cx.ts_a.get_unchecked(p + cx.m - 1))
                    .mul_add(*cx.ts_b.get_unchecked(j + cx.m - 1), qt_d);
                let neg_r = (*cx.m_mean_a.get_unchecked(p))
                    .mul_add(*cx.mean_b.get_unchecked(j), -qt_d)
                    * *cx.m_sigma_inv_a.get_unchecked(p)
                    * *cx.m_sigma_inv_b.get_unchecked(j);
                acc_a.update(p, neg_r, j);
                acc_b.update(j, neg_r, p);
            }
        }
    }
}

// --- Negative diagonal helpers (k = 1..n_a, step p: i=p+k, j=p) ---
// Shared (4-wide): ts_b[p-1], ts_b[p+m-1], m_mean_b[p], msi_b[p]
// Varying: ts_a[i-1], ts_a[i+m-1], mean_a[i], msi_a[i]

/// Process a single negative diagonal (branchless).
#[inline(always)]
fn ab_diag_neg_single_branchless(
    cx: &CorrCtxAB<'_>,
    k: usize,
    acc_a: &mut JoinAccumulator,
    acc_b: &mut JoinAccumulator,
) {
    let diag_len = cx.n_b.min(cx.n_a - k);
    unsafe {
        let qt_init = *cx.qt_first_neg.get_unchecked(k);
        // p=0: i=k, j=0. Use m_mean_b[0] * mean_a[k] form (equivalent, enables shared hoisting).
        let neg_r = (*cx.m_mean_b.get_unchecked(0)).mul_add(*cx.mean_a.get_unchecked(k), -qt_init)
            * *cx.m_sigma_inv_a.get_unchecked(k)
            * *cx.m_sigma_inv_b.get_unchecked(0);
        acc_a.update(k, neg_r, 0);
        acc_b.update(0, neg_r, k);

        let mut qt = qt_init;
        for p in 1..diag_len {
            let i = p + k;
            qt = (-*cx.ts_a.get_unchecked(i - 1)).mul_add(*cx.ts_b.get_unchecked(p - 1), qt);
            qt = (*cx.ts_a.get_unchecked(i + cx.m - 1))
                .mul_add(*cx.ts_b.get_unchecked(p + cx.m - 1), qt);
            let neg_r = (*cx.m_mean_b.get_unchecked(p)).mul_add(*cx.mean_a.get_unchecked(i), -qt)
                * *cx.m_sigma_inv_a.get_unchecked(i)
                * *cx.m_sigma_inv_b.get_unchecked(p);
            acc_a.update(i, neg_r, p);
            acc_b.update(p, neg_r, i);
        }
    }
}

/// Process a single negative diagonal with constant-subsequence handling.
#[inline(always)]
fn ab_diag_neg_single_branching(
    cx: &CorrCtxAB<'_>,
    k: usize,
    acc_a: &mut JoinAccumulator,
    acc_b: &mut JoinAccumulator,
) {
    let diag_len = cx.n_b.min(cx.n_a - k);
    unsafe {
        let qt_init = *cx.qt_first_neg.get_unchecked(k);
        let si = *cx.m_sigma_inv_a.get_unchecked(k);
        let sj = *cx.m_sigma_inv_b.get_unchecked(0);
        let neg_r = if si == 0.0 && sj == 0.0 {
            -1.0
        } else if si == 0.0 || sj == 0.0 {
            0.0
        } else {
            (*cx.m_mean_b.get_unchecked(0)).mul_add(*cx.mean_a.get_unchecked(k), -qt_init) * si * sj
        };
        acc_a.update(k, neg_r, 0);
        acc_b.update(0, neg_r, k);

        let mut qt = qt_init;
        for p in 1..diag_len {
            let i = p + k;
            qt = (-*cx.ts_a.get_unchecked(i - 1)).mul_add(*cx.ts_b.get_unchecked(p - 1), qt);
            qt = (*cx.ts_a.get_unchecked(i + cx.m - 1))
                .mul_add(*cx.ts_b.get_unchecked(p + cx.m - 1), qt);
            let si = *cx.m_sigma_inv_a.get_unchecked(i);
            let sj = *cx.m_sigma_inv_b.get_unchecked(p);
            let neg_r = if si == 0.0 && sj == 0.0 {
                -1.0
            } else if si == 0.0 || sj == 0.0 {
                0.0
            } else {
                (*cx.m_mean_b.get_unchecked(p)).mul_add(*cx.mean_a.get_unchecked(i), -qt) * si * sj
            };
            acc_a.update(i, neg_r, p);
            acc_b.update(p, neg_r, i);
        }
    }
}

/// Process 4 adjacent negative diagonals simultaneously (branchless).
///
/// Diagonals k, k+1, k+2, k+3: j=p is shared across all lanes.
/// Shared: `ts_b[p-1]`, `ts_b[p+m-1]`, `m_mean_b[p]`, `msi_b[p]`.
/// Varying: `ts_a[i-1]`, `ts_a[i+m-1]`, `mean_a[i]`, `msi_a[i]` — consecutive loads.
#[inline(always)]
#[allow(clippy::needless_range_loop)]
fn ab_diag_neg_group4_branchless(
    cx: &CorrCtxAB<'_>,
    k: usize,
    acc_a: &mut JoinAccumulator,
    acc_b: &mut JoinAccumulator,
) {
    let min_diag_len = cx.n_b.min(cx.n_a - k - 3);
    unsafe {
        let mut qt = [
            *cx.qt_first_neg.get_unchecked(k),
            *cx.qt_first_neg.get_unchecked(k + 1),
            *cx.qt_first_neg.get_unchecked(k + 2),
            *cx.qt_first_neg.get_unchecked(k + 3),
        ];

        // p = 0: i = k+d, j = 0. Shared B-side: m_mean_b[0], msi_b[0].
        let mm0 = *cx.m_mean_b.get_unchecked(0);
        let si0 = *cx.m_sigma_inv_b.get_unchecked(0);
        for d in 0..4usize {
            let i = k + d;
            let neg_r = mm0.mul_add(*cx.mean_a.get_unchecked(i), -qt[d])
                * *cx.m_sigma_inv_a.get_unchecked(i)
                * si0;
            acc_a.update(i, neg_r, 0);
            acc_b.update(0, neg_r, i);
        }

        // p = 1..min_diag_len: 4-wide recurrence
        for p in 1..min_diag_len {
            let i_base = p + k;
            let neg_b = -*cx.ts_b.get_unchecked(p - 1);
            let c_b = *cx.ts_b.get_unchecked(p + cx.m - 1);
            let mm = *cx.m_mean_b.get_unchecked(p);
            let si_b = *cx.m_sigma_inv_b.get_unchecked(p);

            for d in 0..4usize {
                let i = i_base + d;
                qt[d] = neg_b.mul_add(*cx.ts_a.get_unchecked(i - 1), qt[d]);
                qt[d] = c_b.mul_add(*cx.ts_a.get_unchecked(i + cx.m - 1), qt[d]);
                let neg_r = mm.mul_add(*cx.mean_a.get_unchecked(i), -qt[d])
                    * *cx.m_sigma_inv_a.get_unchecked(i)
                    * si_b;
                acc_a.update(i, neg_r, p);
                acc_b.update(p, neg_r, i);
            }
        }

        // Tail: each diagonal may extend 0-3 elements beyond min_diag_len
        for d in 0..4usize {
            let kd = k + d;
            let diag_len = cx.n_b.min(cx.n_a - kd);
            let mut qt_d = qt[d];
            for p in min_diag_len..diag_len {
                let i = p + kd;
                qt_d =
                    (-*cx.ts_a.get_unchecked(i - 1)).mul_add(*cx.ts_b.get_unchecked(p - 1), qt_d);
                qt_d = (*cx.ts_a.get_unchecked(i + cx.m - 1))
                    .mul_add(*cx.ts_b.get_unchecked(p + cx.m - 1), qt_d);
                let neg_r = (*cx.m_mean_b.get_unchecked(p))
                    .mul_add(*cx.mean_a.get_unchecked(i), -qt_d)
                    * *cx.m_sigma_inv_a.get_unchecked(i)
                    * *cx.m_sigma_inv_b.get_unchecked(p);
                acc_a.update(i, neg_r, p);
                acc_b.update(p, neg_r, i);
            }
        }
    }
}

// --- Range processing with 4-wide grouping ---

/// Process a range of positive diagonals using the branchless path with 4-wide grouping.
#[inline(always)]
fn process_ab_pos_branchless(
    cx: &CorrCtxAB<'_>,
    start_k: usize,
    end_k: usize,
    acc_a: &mut JoinAccumulator,
    acc_b: &mut JoinAccumulator,
) {
    let mut k = start_k;
    while k + 4 <= end_k {
        ab_diag_pos_group4_branchless(cx, k, acc_a, acc_b);
        k += 4;
    }
    while k < end_k {
        ab_diag_pos_single_branchless(cx, k, acc_a, acc_b);
        k += 1;
    }
}

/// Process a range of positive diagonals using the branching path (handles constants).
#[inline(always)]
fn process_ab_pos_branching(
    cx: &CorrCtxAB<'_>,
    start_k: usize,
    end_k: usize,
    acc_a: &mut JoinAccumulator,
    acc_b: &mut JoinAccumulator,
) {
    for k in start_k..end_k {
        ab_diag_pos_single_branching(cx, k, acc_a, acc_b);
    }
}

/// Process a range of negative diagonals using the branchless path with 4-wide grouping.
#[inline(always)]
fn process_ab_neg_branchless(
    cx: &CorrCtxAB<'_>,
    start_k: usize,
    end_k: usize,
    acc_a: &mut JoinAccumulator,
    acc_b: &mut JoinAccumulator,
) {
    let mut k = start_k;
    while k + 4 <= end_k {
        ab_diag_neg_group4_branchless(cx, k, acc_a, acc_b);
        k += 4;
    }
    while k < end_k {
        ab_diag_neg_single_branchless(cx, k, acc_a, acc_b);
        k += 1;
    }
}

/// Process a range of negative diagonals using the branching path (handles constants).
#[inline(always)]
fn process_ab_neg_branching(
    cx: &CorrCtxAB<'_>,
    start_k: usize,
    end_k: usize,
    acc_a: &mut JoinAccumulator,
    acc_b: &mut JoinAccumulator,
) {
    for k in start_k..end_k {
        ab_diag_neg_single_branching(cx, k, acc_a, acc_b);
    }
}

// ---------------------------------------------------------------------------
// Correlation-domain AB-join entry points
// ---------------------------------------------------------------------------

/// Correlation-domain AB-join (serial).
///
/// Inner loop optimizations (mirroring self-join STOMP):
/// - Negated Pearson correlations (defers sqrt to O(n) final pass)
/// - `f64::mul_add()` for FMA fusion on QT recurrence and neg_r
/// - Hoisted p=0 (no branch in inner loop)
/// - Unchecked indexing (bounds verified by invariants)
/// - 4-wide diagonal grouping (consecutive loads enable AVX2 auto-vectorization)
/// - Pre-computed `m_mean` arrays for FMA-friendly neg_r
/// - Branchless/branching dispatch based on `has_constant`
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
    let (mean_a, msi_a, mean_b, msi_b, has_constant) = M::correlation_data_ab(ctx_a, ctx_b);
    let m_f = m as f64;

    let qt_first_pos = sliding_dot_product(&ts_a[0..m], ts_b);
    let qt_first_neg = sliding_dot_product(&ts_b[0..m], ts_a);

    let m_mean_a: Vec<f64> = mean_a.iter().map(|&mu| m_f * mu).collect();
    let m_mean_b: Vec<f64> = mean_b.iter().map(|&mu| m_f * mu).collect();

    let cx = CorrCtxAB {
        ts_a,
        ts_b,
        m,
        n_a,
        n_b,
        qt_first_pos: &qt_first_pos,
        qt_first_neg: &qt_first_neg,
        mean_a,
        mean_b,
        m_sigma_inv_a: msi_a,
        m_sigma_inv_b: msi_b,
        m_mean_a: &m_mean_a,
        m_mean_b: &m_mean_b,
    };
    let mut acc_a = JoinAccumulator::new(n_a);
    let mut acc_b = JoinAccumulator::new(n_b);

    if !has_constant {
        process_ab_pos_branchless(&cx, 0, n_b, &mut acc_a, &mut acc_b);
        process_ab_neg_branchless(&cx, 1, n_a, &mut acc_a, &mut acc_b);
    } else {
        process_ab_pos_branching(&cx, 0, n_b, &mut acc_a, &mut acc_b);
        process_ab_neg_branching(&cx, 1, n_a, &mut acc_a, &mut acc_b);
    }

    let two_m = 2.0 * m_f;
    acc_a.write_to_join_profile(jp_a, |nc| (two_m * (1.0 + nc)).max(0.0).sqrt());
    acc_b.write_to_join_profile(jp_b, |nc| (two_m * (1.0 + nc)).max(0.0).sqrt());
}

/// Partition AB-join diagonals into load-balanced chunks.
///
/// Uses a unified numbering: d = 0..n_b are positive diagonals, d = n_b..(n_b+n_a-1)
/// are negative diagonals. Returns ranges in this unified space.
#[cfg(feature = "parallel")]
fn compute_ab_diagonal_ranges(n_a: usize, n_b: usize, n_chunks: usize) -> Vec<(usize, usize)> {
    use crate::algorithms::common::balanced_ranges;

    let total_diags = n_b + n_a.saturating_sub(1);
    if total_diags == 0 || n_chunks == 0 {
        return vec![];
    }

    // Cumulative work via prefix sums
    let mut cum_work = Vec::with_capacity(total_diags + 1);
    cum_work.push(0usize);
    for d in 0..total_diags {
        let w = if d < n_b {
            n_a.min(n_b - d)
        } else {
            let k = d - n_b + 1;
            n_b.min(n_a - k)
        };
        cum_work.push(cum_work[d] + w);
    }

    balanced_ranges(&cum_work, total_diags, n_chunks)
}

/// Parallel correlation-domain AB-join with load-balanced chunking.
///
/// Single parallel pass over all diagonals (positive + negative combined),
/// using work-balanced partitioning to equalize load across threads.
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

    let (mean_a, msi_a, mean_b, msi_b, has_constant) = M::correlation_data_ab(ctx_a, ctx_b);
    let m_f = m as f64;

    let qt_first_pos = sliding_dot_product(&ts_a[0..m], ts_b);
    let qt_first_neg = sliding_dot_product(&ts_b[0..m], ts_a);

    let m_mean_a: Vec<f64> = mean_a.iter().map(|&mu| m_f * mu).collect();
    let m_mean_b: Vec<f64> = mean_b.iter().map(|&mu| m_f * mu).collect();

    let cx = CorrCtxAB {
        ts_a,
        ts_b,
        m,
        n_a,
        n_b,
        qt_first_pos: &qt_first_pos,
        qt_first_neg: &qt_first_neg,
        mean_a,
        mean_b,
        m_sigma_inv_a: msi_a,
        m_sigma_inv_b: msi_b,
        m_mean_a: &m_mean_a,
        m_mean_b: &m_mean_b,
    };

    let n_threads = rayon::current_num_threads();
    let ranges = compute_ab_diagonal_ranges(n_a, n_b, n_threads);

    let results: Vec<(JoinAccumulator, JoinAccumulator)> = ranges
        .into_par_iter()
        .map(|(start_d, end_d)| {
            let mut acc_a = JoinAccumulator::new(n_a);
            let mut acc_b = JoinAccumulator::new(n_b);

            // Positive diagonals in [start_d, min(end_d, n_b))
            let pos_end = end_d.min(n_b);
            if start_d < pos_end {
                if !has_constant {
                    process_ab_pos_branchless(&cx, start_d, pos_end, &mut acc_a, &mut acc_b);
                } else {
                    process_ab_pos_branching(&cx, start_d, pos_end, &mut acc_a, &mut acc_b);
                }
            }

            // Negative diagonals: unified d >= n_b maps to neg diagonal k = d - n_b + 1
            if end_d > n_b {
                let neg_start_k = if start_d >= n_b { start_d - n_b + 1 } else { 1 };
                let neg_end_k = end_d - n_b + 1;
                if !has_constant {
                    process_ab_neg_branchless(&cx, neg_start_k, neg_end_k, &mut acc_a, &mut acc_b);
                } else {
                    process_ab_neg_branching(&cx, neg_start_k, neg_end_k, &mut acc_a, &mut acc_b);
                }
            }

            (acc_a, acc_b)
        })
        .collect();

    // Merge all thread-local results
    let mut combined_a = JoinAccumulator::new(n_a);
    let mut combined_b = JoinAccumulator::new(n_b);

    for (acc_a, acc_b) in &results {
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
