/// Standard normal quantile function (probit / inverse CDF).
///
/// Uses Peter Acklam's rational approximation with relative error < 1.15e-9.
pub(crate) fn norm_ppf(p: f64) -> f64 {
    #[allow(clippy::excessive_precision)]
    const A: [f64; 6] = [
        -3.969683028665376e+01,
        2.209460984245205e+02,
        -2.759285104469687e+02,
        1.383577518672690e+02,
        -3.066479806614716e+01,
        2.506628277459239e+00,
    ];
    #[allow(clippy::excessive_precision)]
    const B: [f64; 5] = [
        -5.447609879822406e+01,
        1.615858368580409e+02,
        -1.556989798598866e+02,
        6.680131188771972e+01,
        -1.328068155288572e+01,
    ];
    #[allow(clippy::excessive_precision)]
    const C: [f64; 6] = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e+00,
        -2.549732539343734e+00,
        4.374664141464968e+00,
        2.938163982698783e+00,
    ];
    #[allow(clippy::excessive_precision)]
    const D: [f64; 4] = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e+00,
        3.754408661907416e+00,
    ];

    const P_LOW: f64 = 0.02425;
    const P_HIGH: f64 = 1.0 - P_LOW;

    if p < P_LOW {
        let q = (-2.0 * p.ln()).sqrt();
        (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    } else if p <= P_HIGH {
        let q = p - 0.5;
        let r = q * q;
        (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q
            / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
    } else {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -(((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    }
}

/// Compute bin edges from standard normal quantiles.
///
/// With `n_bit=8` (default), generates `2^n_bit - 1 = 255` bin edges from
/// `norm.ppf(1/256)` to `norm.ppf(255/256)`, creating 256 possible bin values (0-255).
/// This matches stumpy's `_inverse_norm(n_bit=8)`.
pub(crate) fn normal_quantile_bins(n_bit: usize) -> Vec<f64> {
    let n = 1_usize << n_bit; // 2^n_bit = 256 for n_bit=8
    (1..n).map(|i| norm_ppf(i as f64 / n as f64)).collect()
}

/// Discretize values into bin indices using binary search.
///
/// Equivalent to `numpy.searchsorted(bins, data, side='left')`.
/// Returns indices in `0..=bins.len()` (i.e., `n_bits + 1` possible values).
pub(crate) fn discretize(data: &[f64], bins: &[f64]) -> Vec<usize> {
    data.iter()
        .map(|&v| bins.partition_point(|&b| b < v))
        .collect()
}

/// Z-normalize a slice, returning a new vector.
///
/// Returns all zeros if the standard deviation is below threshold (constant subsequence).
pub(crate) fn z_normalize(data: &[f64]) -> Vec<f64> {
    let n = data.len() as f64;
    let mean = data.iter().sum::<f64>() / n;
    let var = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
    let std = var.sqrt();

    if std < 1e-7 {
        vec![0.0; data.len()]
    } else {
        data.iter().map(|&x| (x - mean) / std).collect()
    }
}

/// Compute dimension ordering and discretized subsequences for a motif pair.
///
/// Accepts per-dimension nearest neighbor indices (each dimension can use a different NN).
///
/// Returns `(sorted_dimension_indices, disc_q_sorted, disc_c_sorted)` where:
/// - `sorted_dimension_indices`: dimensions sorted by discretized L2 distance (ascending)
/// - `disc_q_sorted[k]`: discretized z-normalized subsequence at `subseq_idx` for the k-th best dimension
/// - `disc_c_sorted[k]`: discretized z-normalized subsequence at `nn_idx[k]` for the k-th best dimension
pub(crate) fn dimension_order_multi(
    ts: &[&[f64]],
    m: usize,
    subseq_idx: usize,
    nn_idx: &[usize],
) -> (Vec<usize>, Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let d = ts.len();
    assert_eq!(nn_idx.len(), d, "nn_idx must have one entry per dimension");
    let n_bit = 8;
    let bins = normal_quantile_bins(n_bit);

    let mut disc_q = Vec::with_capacity(d);
    let mut disc_c = Vec::with_capacity(d);
    let mut dim_dists: Vec<(f64, usize)> = Vec::with_capacity(d);

    for (dim, ts_dim) in ts.iter().enumerate() {
        let q = z_normalize(&ts_dim[subseq_idx..subseq_idx + m]);
        let c = z_normalize(&ts_dim[nn_idx[dim]..nn_idx[dim] + m]);

        let dq: Vec<f64> = discretize(&q, &bins).iter().map(|&x| x as f64).collect();
        let dc: Vec<f64> = discretize(&c, &bins).iter().map(|&x| x as f64).collect();

        let dist: f64 = dq.iter().zip(&dc).map(|(a, b)| (a - b).powi(2)).sum();

        dim_dists.push((dist, dim));
        disc_q.push(dq);
        disc_c.push(dc);
    }

    // Sort by discretized distance (ascending = most similar first)
    dim_dists.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    let order: Vec<usize> = dim_dists.iter().map(|&(_, dim)| dim).collect();

    // Reorder discretized arrays to match sorted dimension order
    let disc_q_sorted: Vec<Vec<f64>> = order.iter().map(|&dim| disc_q[dim].clone()).collect();
    let disc_c_sorted: Vec<Vec<f64>> = order.iter().map(|&dim| disc_c[dim].clone()).collect();

    (order, disc_q_sorted, disc_c_sorted)
}

/// Convenience wrapper: compute dimension ordering using a single NN index for all dimensions.
pub(crate) fn dimension_order(
    ts: &[&[f64]],
    m: usize,
    subseq_idx: usize,
    nn_idx: usize,
) -> (Vec<usize>, Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let nn_idx_arr = vec![nn_idx; ts.len()];
    dimension_order_multi(ts, m, subseq_idx, &nn_idx_arr)
}

/// Select the k-dimensional subspace that best characterizes a motif pair.
///
/// For each dimension, computes the discretized L2 distance between the
/// z-normalized subsequences at `subseq_idx` and `nn_idx`. Returns the `k`
/// dimension indices with the smallest discretized distances (most relevant first).
///
/// Uses 8-bit discretization with standard normal quantile bin edges,
/// matching stumpy's default behavior.
///
/// # Arguments
/// * `ts` - Multi-dimensional time series (one slice per dimension)
/// * `m` - Subsequence length
/// * `subseq_idx` - Index of the first motif occurrence
/// * `nn_idx` - Index of the second motif occurrence (nearest neighbor)
/// * `k` - Number of dimensions to select
///
/// # Returns
/// A vector of `k` dimension indices, sorted by relevance (most relevant first).
pub fn subspace(ts: &[&[f64]], m: usize, subseq_idx: usize, nn_idx: usize, k: usize) -> Vec<usize> {
    let d = ts.len();
    assert!(d >= 1, "Need at least one dimension");
    assert!(k >= 1 && k <= d, "k must be in [1, {d}]");

    let (order, _, _) = dimension_order(ts, m, subseq_idx, nn_idx);
    order[..k].to_vec()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_norm_ppf_symmetry() {
        // norm_ppf(0.5) should be 0 (median of standard normal)
        assert!((norm_ppf(0.5)).abs() < 1e-10);

        // norm_ppf(p) + norm_ppf(1-p) should be 0 (symmetry)
        for &p in &[0.1, 0.25, 0.4, 0.05, 0.01] {
            let sum = norm_ppf(p) + norm_ppf(1.0 - p);
            assert!(
                sum.abs() < 1e-8,
                "norm_ppf({p}) + norm_ppf({}) = {sum}",
                1.0 - p
            );
        }
    }

    #[test]
    fn test_norm_ppf_known_values() {
        // Known quantiles of the standard normal distribution
        assert!((norm_ppf(0.5) - 0.0).abs() < 1e-8);
        assert!((norm_ppf(0.975) - 1.95996398).abs() < 1e-5);
        assert!((norm_ppf(0.025) - (-1.95996398)).abs() < 1e-5);
        assert!((norm_ppf(0.84134) - 1.0).abs() < 1e-3);
    }

    #[test]
    fn test_normal_quantile_bins() {
        let bins = normal_quantile_bins(8);
        assert_eq!(bins.len(), 255); // 2^8 - 1 = 255 bin edges

        // Bins should be strictly increasing
        for w in bins.windows(2) {
            assert!(w[1] > w[0], "Bins not increasing: {} >= {}", w[0], w[1]);
        }

        // Symmetric around 0 (bin[i] + bin[254-i] ≈ 0)
        for i in 0..127 {
            assert!(
                (bins[i] + bins[254 - i]).abs() < 1e-8,
                "Bins not symmetric: {} + {} = {}",
                bins[i],
                bins[254 - i],
                bins[i] + bins[254 - i]
            );
        }

        // Middle bin should be 0 (median of N(0,1))
        assert!(
            bins[127].abs() < 1e-8,
            "Middle bin should be ~0: {}",
            bins[127]
        );
    }

    #[test]
    fn test_discretize_basic() {
        let bins = vec![-1.0, 0.0, 1.0];
        let data = vec![-2.0, -0.5, 0.0, 0.5, 2.0];
        let result = discretize(&data, &bins);
        // -2.0 < -1.0 → bin 0
        // -1.0 <= -0.5 < 0.0 → bin 1
        // 0.0 == 0.0 → bin 1 (partition_point(b < 0.0) = 1)
        // Wait: partition_point(|&b| b < v) where bins = [-1.0, 0.0, 1.0]
        // For v = 0.0: b < 0.0 gives [true, false, false] → partition_point = 1
        // Hmm, -1.0 < 0.0 is true, 0.0 < 0.0 is false → partition_point = 1
        assert_eq!(result, vec![0, 1, 1, 2, 3]);
    }

    #[test]
    fn test_z_normalize() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let normed = z_normalize(&data);
        assert_eq!(normed.len(), 5);

        // Mean should be ~0
        let mean: f64 = normed.iter().sum::<f64>() / normed.len() as f64;
        assert!(mean.abs() < 1e-10, "Mean = {mean}");

        // Population std should be ~1
        let var: f64 = normed.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / normed.len() as f64;
        assert!((var.sqrt() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_z_normalize_constant() {
        let data = vec![5.0; 10];
        let normed = z_normalize(&data);
        for &v in &normed {
            assert!((v - 0.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_subspace_returns_correct_k() {
        let n = 100;
        let m = 10;
        let ts0: Vec<f64> = (0..n).map(|i| (i as f64 * 0.2).sin()).collect();
        let ts1: Vec<f64> = (0..n).map(|i| (i as f64 * 0.3).cos()).collect();
        let ts2: Vec<f64> = (0..n).map(|i| (i as f64 * 0.15).sin() + 0.5).collect();

        let ts_refs: [&[f64]; 3] = [&ts0, &ts1, &ts2];

        for k in 1..=3 {
            let dims = subspace(&ts_refs, m, 5, 50, k);
            assert_eq!(dims.len(), k, "subspace should return {k} dimensions");
        }
    }

    #[test]
    fn test_subspace_identical_subsequences() {
        // If two positions have identical subsequences in dim 0 but different in dim 1,
        // dim 0 should be ranked first
        let n = 50;
        let m = 5;
        let mut ts0 = vec![0.0; n];
        let mut ts1 = vec![0.0; n];

        // Place identical pattern in dim 0 at indices 5 and 30
        let pattern = [1.0, 2.0, 3.0, 4.0, 5.0];
        ts0[5..10].copy_from_slice(&pattern);
        ts0[30..35].copy_from_slice(&pattern);

        // Place different patterns in dim 1
        ts1[5..10].copy_from_slice(&[5.0, 4.0, 3.0, 2.0, 1.0]);
        ts1[30..35].copy_from_slice(&[1.0, 1.0, 1.0, 1.0, 1.0]);

        let ts_refs: [&[f64]; 2] = [&ts0, &ts1];
        let dims = subspace(&ts_refs, m, 5, 30, 1);
        assert_eq!(
            dims[0], 0,
            "Dim 0 should be most relevant (identical subsequences)"
        );
    }
}
