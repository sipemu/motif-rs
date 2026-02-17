use crate::algorithms::common::sliding_dot_product;
use crate::core::distance_metric::DistanceMetric;
use crate::core::matrix_profile::MatrixProfileConfig;

/// Minimum number of subsequences before dispatching to parallel top-k.
#[cfg(feature = "parallel")]
const MIN_PARALLEL_SUBS: usize = 256;

/// Matrix profile with top-k nearest neighbors per subsequence.
#[derive(Debug, Clone)]
pub struct TopKMatrixProfile {
    /// `distances[i]` = k nearest distances for subsequence i (sorted ascending).
    pub distances: Vec<Vec<f64>>,
    /// `indices[i]` = corresponding neighbor indices.
    pub indices: Vec<Vec<usize>>,
    /// Number of neighbors stored per subsequence.
    pub k: usize,
    /// Subsequence length used.
    pub m: usize,
    /// Exclusion zone radius used.
    pub exclusion_zone: usize,
}

/// Flat-array accumulator for top-k nearest neighbors.
///
/// Stores `n * k` entries in contiguous flat arrays. Position `i`'s k neighbors
/// are at `[i*k..(i+1)*k]`, sorted ascending by distance.
pub(crate) struct TopKAccumulator {
    distances: Vec<f64>,
    indices: Vec<usize>,
    n: usize,
    k: usize,
}

impl TopKAccumulator {
    pub fn new(n: usize, k: usize) -> Self {
        Self {
            distances: vec![f64::INFINITY; n * k],
            indices: vec![0; n * k],
            n,
            k,
        }
    }

    /// Update the top-k list for position `idx` with a new candidate.
    ///
    /// Quick-reject: if the candidate distance is >= the k-th best, skip entirely.
    /// Otherwise, insert in sorted position using linear insertion sort (k is small).
    #[inline(always)]
    pub fn update(&mut self, idx: usize, dist: f64, neighbor: usize) {
        let base = idx * self.k;
        let worst = base + self.k - 1;

        // Quick reject: candidate doesn't beat the worst of k best
        if dist >= self.distances[worst] {
            return;
        }

        // Find insertion position (linear scan, k is small)
        let mut pos = worst;
        while pos > base && dist < self.distances[pos - 1] {
            // Shift right
            self.distances[pos] = self.distances[pos - 1];
            self.indices[pos] = self.indices[pos - 1];
            pos -= 1;
        }

        self.distances[pos] = dist;
        self.indices[pos] = neighbor;
    }

    /// Merge another accumulator into this one. For each position, merges two
    /// sorted-k lists and keeps the k smallest.
    #[cfg(feature = "parallel")]
    pub fn merge(&mut self, other: &Self) {
        debug_assert_eq!(self.n, other.n);
        debug_assert_eq!(self.k, other.k);

        let k = self.k;
        // Temporary buffer for two-pointer merge
        let mut tmp_dists = vec![0.0; k];
        let mut tmp_idxs = vec![0usize; k];

        for i in 0..self.n {
            let base = i * k;
            let a_dists = &self.distances[base..base + k];
            let b_dists = &other.distances[base..base + k];
            let a_idxs = &self.indices[base..base + k];
            let b_idxs = &other.indices[base..base + k];

            // Two-pointer merge of two sorted lists, take k smallest
            let mut ai = 0;
            let mut bi = 0;
            for out in 0..k {
                if ai < k && (bi >= k || a_dists[ai] <= b_dists[bi]) {
                    tmp_dists[out] = a_dists[ai];
                    tmp_idxs[out] = a_idxs[ai];
                    ai += 1;
                } else if bi < k {
                    tmp_dists[out] = b_dists[bi];
                    tmp_idxs[out] = b_idxs[bi];
                    bi += 1;
                }
            }

            self.distances[base..base + k].copy_from_slice(&tmp_dists);
            self.indices[base..base + k].copy_from_slice(&tmp_idxs);
        }
    }

    /// Convert to a TopKMatrixProfile.
    pub fn into_topk_profile(self, m: usize, exclusion_zone: usize) -> TopKMatrixProfile {
        let k = self.k;
        let mut distances = Vec::with_capacity(self.n);
        let mut indices = Vec::with_capacity(self.n);

        for i in 0..self.n {
            let base = i * k;
            distances.push(self.distances[base..base + k].to_vec());
            indices.push(self.indices[base..base + k].to_vec());
        }

        TopKMatrixProfile {
            distances,
            indices,
            k,
            m,
            exclusion_zone,
        }
    }
}

/// Compute the top-k matrix profile using diagonal STOMP.
///
/// This is a separate code path from the standard k=1 STOMP to avoid any
/// performance regression for the common case.
pub fn stomp_topk<M: DistanceMetric>(
    ts: &[f64],
    config: &MatrixProfileConfig,
    k: usize,
) -> TopKMatrixProfile {
    let m = config.m;
    let n = ts.len();
    assert!(n >= m, "Time series length must be >= subsequence length");
    assert!(m >= 2, "Subsequence length must be >= 2");
    assert!(k >= 1, "k must be >= 1");

    let n_subs = n - m + 1;
    let exclusion_zone = config.exclusion_zone();
    let ctx = M::precompute(ts, m);

    if M::supports_qt_optimization() {
        #[cfg(feature = "parallel")]
        if n_subs >= MIN_PARALLEL_SUBS {
            return stomp_topk_diagonal_parallel::<M>(ts, m, n_subs, exclusion_zone, k, &ctx);
        }
        stomp_topk_diagonal::<M>(ts, m, n_subs, exclusion_zone, k, &ctx)
    } else {
        stomp_topk_naive::<M>(ts, m, n_subs, exclusion_zone, k, &ctx)
    }
}

/// Diagonal-traversal top-k STOMP (serial).
#[allow(clippy::needless_range_loop)]
fn stomp_topk_diagonal<M: DistanceMetric>(
    ts: &[f64],
    m: usize,
    n_subs: usize,
    exclusion_zone: usize,
    k: usize,
    ctx: &M::Context,
) -> TopKMatrixProfile {
    let qt_first = sliding_dot_product(&ts[0..m], ts);
    let mut acc = TopKAccumulator::new(n_subs, k);

    for diag_k in (exclusion_zone + 1)..n_subs {
        let diag_len = n_subs - diag_k;
        let mut qt = qt_first[diag_k];

        let d = M::qt_to_distance(qt, 0, diag_k, m, ctx);
        acc.update(0, d, diag_k);
        acc.update(diag_k, d, 0);

        for p in 1..diag_len {
            let i = p;
            let j = p + diag_k;
            qt = qt - ts[i - 1] * ts[j - 1] + ts[i + m - 1] * ts[j + m - 1];
            let d = M::qt_to_distance(qt, i, j, m, ctx);
            acc.update(i, d, j);
            acc.update(j, d, i);
        }
    }

    acc.into_topk_profile(m, exclusion_zone)
}

/// Parallel diagonal-traversal top-k STOMP.
#[cfg(feature = "parallel")]
#[allow(clippy::needless_range_loop)]
fn stomp_topk_diagonal_parallel<M: DistanceMetric>(
    ts: &[f64],
    m: usize,
    n_subs: usize,
    exclusion_zone: usize,
    k: usize,
    ctx: &M::Context,
) -> TopKMatrixProfile {
    use rayon::prelude::*;

    let qt_first = sliding_dot_product(&ts[0..m], ts);
    let n_threads = rayon::current_num_threads();
    let ranges =
        crate::algorithms::stomp::compute_diagonal_ranges(exclusion_zone + 1, n_subs, n_threads);

    let results: Vec<TopKAccumulator> = ranges
        .into_par_iter()
        .map(|(start_k, end_k)| {
            let mut acc = TopKAccumulator::new(n_subs, k);

            for diag_k in start_k..end_k {
                let diag_len = n_subs - diag_k;
                let mut qt = qt_first[diag_k];

                let d = M::qt_to_distance(qt, 0, diag_k, m, ctx);
                acc.update(0, d, diag_k);
                acc.update(diag_k, d, 0);

                for p in 1..diag_len {
                    let i = p;
                    let j = p + diag_k;
                    qt = qt - ts[i - 1] * ts[j - 1] + ts[i + m - 1] * ts[j + m - 1];
                    let d = M::qt_to_distance(qt, i, j, m, ctx);
                    acc.update(i, d, j);
                    acc.update(j, d, i);
                }
            }

            acc
        })
        .collect();

    let mut combined = TopKAccumulator::new(n_subs, k);
    for result in &results {
        combined.merge(result);
    }

    combined.into_topk_profile(m, exclusion_zone)
}

/// Naive top-k STOMP for metrics without QT optimization.
fn stomp_topk_naive<M: DistanceMetric>(
    ts: &[f64],
    m: usize,
    n_subs: usize,
    exclusion_zone: usize,
    k: usize,
    ctx: &M::Context,
) -> TopKMatrixProfile {
    let mut acc = TopKAccumulator::new(n_subs, k);

    for i in 0..n_subs {
        let dp = M::distance_profile(ts, i, m, ctx);
        for (j, &d) in dp.iter().enumerate() {
            if i.abs_diff(j) > exclusion_zone {
                acc.update(i, d, j);
            }
        }
    }

    acc.into_topk_profile(m, exclusion_zone)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metrics::euclidean::ZNormalizedEuclidean;

    #[test]
    fn test_topk_k1_matches_standard() {
        use crate::algorithms::stomp::stomp;

        let ts: Vec<f64> = (0..100).map(|i| (i as f64 * 0.2).sin()).collect();
        let config = MatrixProfileConfig::new(10);

        let mp = stomp::<ZNormalizedEuclidean>(&ts, &config);
        let topk = stomp_topk::<ZNormalizedEuclidean>(&ts, &config, 1);

        let eps = 1e-6;
        for (i, (std_d, topk_d)) in mp.profile.iter().zip(topk.distances.iter()).enumerate() {
            assert!(
                (std_d - topk_d[0]).abs() < eps,
                "k=1 mismatch at {i}: standard={std_d}, topk={}",
                topk_d[0]
            );
        }
    }

    #[test]
    fn test_topk_distances_sorted() {
        let ts: Vec<f64> = (0..100).map(|i| (i as f64 * 0.3).sin()).collect();
        let config = MatrixProfileConfig::new(8);
        let topk = stomp_topk::<ZNormalizedEuclidean>(&ts, &config, 5);

        for (i, dists) in topk.distances.iter().enumerate() {
            for w in dists.windows(2) {
                assert!(
                    w[0] <= w[1],
                    "Distances not sorted at {i}: {} > {}",
                    w[0],
                    w[1]
                );
            }
        }
    }

    #[test]
    fn test_topk_basic_properties() {
        let ts: Vec<f64> = (0..50).map(|i| (i as f64 * 0.2).sin()).collect();
        let config = MatrixProfileConfig::new(6);
        let k = 3;
        let topk = stomp_topk::<ZNormalizedEuclidean>(&ts, &config, k);

        assert_eq!(topk.k, k);
        assert_eq!(topk.m, 6);
        assert_eq!(topk.distances.len(), ts.len() - 6 + 1);
        assert_eq!(topk.indices.len(), ts.len() - 6 + 1);

        for dists in &topk.distances {
            assert_eq!(dists.len(), k);
        }
        for idxs in &topk.indices {
            assert_eq!(idxs.len(), k);
        }
    }

    #[test]
    fn test_topk_exclusion_zone() {
        let ts: Vec<f64> = (0..50).map(|i| (i as f64 * 0.7).cos()).collect();
        let config = MatrixProfileConfig::new(8);
        let topk = stomp_topk::<ZNormalizedEuclidean>(&ts, &config, 3);
        let ez = config.exclusion_zone();

        for (i, (dists, idxs)) in topk.distances.iter().zip(topk.indices.iter()).enumerate() {
            for (&d, &j) in dists.iter().zip(idxs.iter()) {
                if d.is_finite() {
                    let gap = i.abs_diff(j);
                    assert!(
                        gap > ez,
                        "Top-k match at i={i}, j={j} (gap={gap}) violates exclusion_zone={ez}"
                    );
                }
            }
        }
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn test_topk_accumulator_merge() {
        let mut a = TopKAccumulator::new(3, 2);
        let mut b = TopKAccumulator::new(3, 2);

        // Position 0: a has [1.0, 3.0], b has [2.0, 4.0] → merged [1.0, 2.0]
        a.update(0, 1.0, 10);
        a.update(0, 3.0, 30);
        b.update(0, 2.0, 20);
        b.update(0, 4.0, 40);

        a.merge(&b);
        assert!((a.distances[0] - 1.0).abs() < 1e-10);
        assert!((a.distances[1] - 2.0).abs() < 1e-10);
    }
}
