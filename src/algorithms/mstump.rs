use crate::algorithms::common::sliding_dot_product;
use crate::core::matrix_profile::RollingStats;

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

/// Compute the z-normalized distance profile for a query against a time series
/// using precomputed rolling statistics.
fn mass_precomputed(query: &[f64], ts: &[f64], stats: &RollingStats, m: usize) -> Vec<f64> {
    let n_subs = ts.len() - m + 1;
    let m_f = m as f64;

    let mu_q: f64 = query.iter().sum::<f64>() / m_f;
    let sum_sq_q: f64 = query.iter().map(|x| x * x).sum::<f64>();
    let var_q = (sum_sq_q / m_f - mu_q * mu_q).max(0.0);
    let sigma_q = var_q.sqrt();

    let qt = sliding_dot_product(query, ts);

    let mut profile = vec![f64::INFINITY; n_subs];

    if sigma_q < 1e-15 {
        for (i, d) in profile.iter_mut().enumerate() {
            if stats.std[i] < 1e-15 {
                *d = 0.0;
            } else {
                *d = (2.0 * m_f).sqrt();
            }
        }
    } else {
        for (i, d) in profile.iter_mut().enumerate() {
            if stats.std[i] < 1e-15 {
                *d = (2.0 * m_f).sqrt();
            } else {
                let r = (qt[i] - m_f * mu_q * stats.mean[i]) / (m_f * sigma_q * stats.std[i]);
                let r_clamped = r.clamp(-1.0, 1.0);
                *d = (2.0 * m_f * (1.0 - r_clamped)).max(0.0).sqrt();
            }
        }
    }

    profile
}

/// Compute the multi-dimensional matrix profile using MSTUMP.
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

    // Precompute rolling stats for each dimension
    let stats: Vec<RollingStats> = ts.iter().map(|t| RollingStats::compute(t, m)).collect();

    let mut profile = vec![vec![f64::INFINITY; n_subs]; d];
    let mut profile_index = vec![vec![0usize; n_subs]; d];

    // For each query position, compute multi-dimensional distance profile
    for i in 0..n_subs {
        // Compute per-dimension distance profiles using MASS
        let dist_profiles: Vec<Vec<f64>> = (0..d)
            .map(|dim| {
                let query = &ts[dim][i..i + m];
                mass_precomputed(query, ts[dim], &stats[dim], m)
            })
            .collect();

        // For each target position, sort across dimensions and compute cumulative average
        for j in 0..n_subs {
            // Exclusion zone: skip trivial self-matches
            if i.abs_diff(j) <= ez {
                continue;
            }

            // Collect distances from all dimensions at target j
            let mut dists: Vec<f64> = dist_profiles.iter().map(|dp| dp[j]).collect();

            // Sort ascending (best dimensions first)
            dists.sort_by(|a, b| a.partial_cmp(b).unwrap());

            // Cumulative average: row k = mean of top (k+1) dimensions
            let mut cum_sum = 0.0;
            for k in 0..d {
                cum_sum += dists[k];
                let cum_avg = cum_sum / (k + 1) as f64;

                if cum_avg < profile[k][i] {
                    profile[k][i] = cum_avg;
                    profile_index[k][i] = j;
                }
            }
        }
    }

    MultiDimensionalProfile {
        profile,
        profile_index,
        d,
        m,
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
