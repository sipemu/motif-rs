use crate::algorithms::subspace::{discretize, normal_quantile_bins, z_normalize};

/// Compute MDL bit sizes for a multi-dimensional matrix profile.
///
/// For each `k` from 0 to `d-1`, uses the motif pair at `(subseq_idx[k], nn_idx[k])`
/// to compute the MDL bit cost for the best `(k+1)`-dimensional subspace.
///
/// This matches stumpy's `mdl(T, m, subseq_idx, nn_idx)` calling convention,
/// where each `k` iteration uses a different motif pair.
///
/// # Arguments
/// * `ts` - Multi-dimensional time series (one slice per dimension)
/// * `m` - Subsequence length
/// * `subseq_idx` - Per-k subsequence indices (length d)
/// * `nn_idx` - Per-k nearest neighbor indices (length d)
///
/// # Returns
/// `(bit_sizes, subspaces)` where:
/// - `bit_sizes[k]` is the MDL cost for the best `(k+1)`-dimensional subspace
/// - `subspaces[k]` contains the `(k+1)` dimension indices selected
pub fn mdl(
    ts: &[&[f64]],
    m: usize,
    subseq_idx: &[usize],
    nn_idx: &[usize],
) -> (Vec<f64>, Vec<Vec<usize>>) {
    let d = ts.len();
    assert!(d >= 1, "Need at least one dimension");
    assert_eq!(subseq_idx.len(), d, "subseq_idx must have one entry per k");
    assert_eq!(nn_idx.len(), d, "nn_idx must have one entry per k");

    let n_bit: usize = 8;
    let bins = normal_quantile_bins(n_bit);

    let mut bit_sizes = vec![0.0f64; d];
    let mut subspaces = vec![Vec::new(); d];

    for k in 0..d {
        // Z-normalize and discretize all dimensions at this k's motif pair
        let mut disc_subseqs: Vec<Vec<usize>> = Vec::with_capacity(d);
        let mut disc_neighbors: Vec<Vec<usize>> = Vec::with_capacity(d);

        for ts_dim in ts {
            let sub = z_normalize(&ts_dim[subseq_idx[k]..subseq_idx[k] + m]);
            let nei = z_normalize(&ts_dim[nn_idx[k]..nn_idx[k] + m]);
            disc_subseqs.push(discretize(&sub, &bins));
            disc_neighbors.push(discretize(&nei, &bins));
        }

        // Per-dimension L2 norm of discretized difference (for subspace selection)
        let d_norms: Vec<f64> = (0..d)
            .map(|dim| {
                disc_subseqs[dim]
                    .iter()
                    .zip(&disc_neighbors[dim])
                    .map(|(&a, &b)| {
                        let diff = a as f64 - b as f64;
                        diff * diff
                    })
                    .sum::<f64>()
                    .sqrt()
            })
            .collect();

        // Select best (k+1) dimensions by sorting on discretized distance
        let mut indices: Vec<usize> = (0..d).collect();
        indices.sort_by(|&a, &b| d_norms[a].partial_cmp(&d_norms[b]).unwrap());
        let s: Vec<usize> = indices[..k + 1].to_vec();

        // Compute MDL bit size using stumpy's formula:
        //   bit_size = n_bit * (2*d*m - sub_dims*m)
        //            + sub_dims*m * log2(n_val)
        //            + n_val * n_bit
        let sub_dims = s.len(); // k + 1

        // Count unique values in the residual across selected dimensions
        let mut residuals: Vec<i32> = Vec::with_capacity(sub_dims * m);
        for &dim in &s {
            for j in 0..m {
                residuals.push(disc_subseqs[dim][j] as i32 - disc_neighbors[dim][j] as i32);
            }
        }
        residuals.sort();
        residuals.dedup();
        let n_val = residuals.len();

        let bit_size = n_bit as f64 * (2 * d * m - sub_dims * m) as f64
            + sub_dims as f64 * m as f64 * (n_val as f64).log2()
            + n_val as f64 * n_bit as f64;

        bit_sizes[k] = bit_size;
        subspaces[k] = s;
    }

    (bit_sizes, subspaces)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mdl_basic() {
        // MDL should return d values
        let n = 100;
        let m = 10;
        let ts0: Vec<f64> = (0..n).map(|i| (i as f64 * 0.2).sin()).collect();
        let ts1: Vec<f64> = (0..n).map(|i| (i as f64 * 0.3).cos()).collect();
        let ts2: Vec<f64> = (0..n).map(|i| (i as f64 * 0.15).sin() + 0.5).collect();

        let ts_refs: [&[f64]; 3] = [&ts0, &ts1, &ts2];
        let subseq_idx = [5, 5, 5];
        let nn_idx = [50, 50, 50];
        let (bit_sizes, subspaces) = mdl(&ts_refs, m, &subseq_idx, &nn_idx);

        assert_eq!(bit_sizes.len(), 3);
        assert_eq!(subspaces.len(), 3);
        // All bit sizes should be positive
        for (k, &bs) in bit_sizes.iter().enumerate() {
            assert!(bs > 0.0, "bit_sizes[{k}] = {bs} is not positive");
        }
        // Subspace[k] should have k+1 dimensions
        for (k, s) in subspaces.iter().enumerate() {
            assert_eq!(s.len(), k + 1, "subspace[{k}] should have {}", k + 1);
        }
    }

    #[test]
    fn test_mdl_identical_pair() {
        // If the subsequences are identical across all dimensions,
        // residual has only one unique value (0), so log2(1)=0 for that term
        let n = 50;
        let m = 5;
        let period = 10;
        let ts0: Vec<f64> = (0..n)
            .map(|i| (i as f64 * 2.0 * std::f64::consts::PI / period as f64).sin())
            .collect();
        let ts1: Vec<f64> = (0..n)
            .map(|i| (i as f64 * 2.0 * std::f64::consts::PI / period as f64).cos())
            .collect();

        let ts_refs: [&[f64]; 2] = [&ts0, &ts1];
        let subseq_idx = [0, 0];
        let nn_idx = [period, period];
        let (bit_sizes, _) = mdl(&ts_refs, m, &subseq_idx, &nn_idx);

        // Identical subsequences: residual is all 0, n_val=1, log2(1)=0
        // bit_size[0] = 8*(2*2*5 - 1*5) + 1*5*0 + 1*8 = 8*15 + 8 = 128
        for (k, &bs) in bit_sizes.iter().enumerate() {
            assert!(bs > 0.0, "bit_sizes[{k}] = {bs} is not positive");
        }
    }
}
