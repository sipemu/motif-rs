use crate::algorithms::common::sliding_dot_product;
use crate::core::distance_metric::DistanceMetric;
use crate::core::matrix_profile::RollingStats;
use crate::metrics::euclidean::ZNormalizedEuclidean;

/// Result of time series snippet extraction.
#[derive(Debug, Clone)]
pub struct SnippetsResult {
    /// Indices into the time series of each snippet's starting position.
    pub indices: Vec<usize>,
    /// The distance profile for each selected snippet (one per snippet, length n_subs).
    pub profiles: Vec<Vec<f64>>,
    /// The fraction of the time series each snippet represents.
    pub fractions: Vec<f64>,
    /// The area under the combined coverage curve at each selection step.
    pub areas: Vec<f64>,
    /// Per-position assignment: index into `indices` of the nearest snippet.
    pub regimes: Vec<usize>,
}

/// Compute a z-normalized distance profile for the subsequence starting at `idx`.
fn compute_distance_profile(ts: &[f64], idx: usize, m: usize, ctx: &RollingStats) -> Vec<f64> {
    let query = &ts[idx..idx + m];
    let qts = sliding_dot_product(query, ts);
    let n_subs = ts.len() - m + 1;
    (0..n_subs)
        .map(|j| ZNormalizedEuclidean::qt_to_distance(qts[j], idx, j, m, ctx))
        .collect()
}

/// Extract `k` representative subsequences (snippets) that best summarize the time series.
///
/// Implements the snippets algorithm (Imani et al., "Matrix Profile XIII: Time Series
/// Snippets", 2019) with the default `percentage=1.0` behavior matching `stumpy.snippets`:
///
/// 1. Divide T into non-overlapping candidate windows of length `m`
/// 2. Compute z-normalized distance profiles for each candidate
/// 3. Greedily select `k` snippets that minimize the combined distance profile area
///
/// # Arguments
/// * `ts` - The time series
/// * `m` - Subsequence length
/// * `k` - Number of snippets to extract
///
/// # Panics
/// Panics if `ts.len() < 2*m`, `k == 0`, or `k > s` (number of candidate windows).
pub fn find_snippets(ts: &[f64], m: usize, k: usize) -> SnippetsResult {
    assert!(m > 0, "Subsequence length must be > 0");
    let n = ts.len();
    assert!(n >= 2 * m, "Time series must be at least 2*m long");
    let n_subs = n - m + 1;

    // Number of non-overlapping candidate windows
    let s = n_subs / m;
    assert!(k > 0, "Must request at least 1 snippet");
    assert!(
        k <= s,
        "Cannot extract more snippets than candidate windows (k={k} > s={s})"
    );

    // Candidate indices: 0, m, 2m, ..., (s-1)*m
    let candidate_indices: Vec<usize> = (0..s).map(|i| i * m).collect();

    // Precompute rolling statistics
    let ctx = ZNormalizedEuclidean::precompute(ts, m);

    // Compute distance profiles for all candidates
    let profiles: Vec<Vec<f64>> = candidate_indices
        .iter()
        .map(|&idx| compute_distance_profile(ts, idx, m, &ctx))
        .collect();

    // Greedy snippet selection
    let mut result_indices = Vec::with_capacity(k);
    let mut result_profiles = Vec::with_capacity(k);
    let mut result_areas = Vec::with_capacity(k);
    let mut used = vec![false; s];

    // Q tracks the element-wise minimum distance across selected snippets.
    // Starts at infinity so the first snippet's area = sum(D[best]).
    let mut q = vec![f64::INFINITY; n_subs];

    for _ in 0..k {
        let mut best_area = f64::INFINITY;
        let mut best_candidate = 0;

        for c in 0..s {
            if used[c] {
                continue;
            }
            let area: f64 = q
                .iter()
                .zip(&profiles[c])
                .map(|(&qi, &di)| qi.min(di))
                .sum();
            if area < best_area {
                best_area = area;
                best_candidate = c;
            }
        }

        used[best_candidate] = true;
        result_indices.push(candidate_indices[best_candidate]);
        result_profiles.push(profiles[best_candidate].clone());
        result_areas.push(best_area);

        // Update Q with the selected snippet's profile
        for (qi, &di) in q.iter_mut().zip(&profiles[best_candidate]) {
            *qi = qi.min(di);
        }
    }

    // Compute regimes: per-position nearest snippet assignment
    let mut regimes = vec![0usize; n_subs];
    for j in 0..n_subs {
        let mut best_dist = f64::INFINITY;
        let mut best_snippet = 0;
        for (si, profile) in result_profiles.iter().enumerate() {
            if profile[j] < best_dist {
                best_dist = profile[j];
                best_snippet = si;
            }
        }
        regimes[j] = best_snippet;
    }

    // Compute fractions: fraction of positions each snippet is nearest
    let mut counts = vec![0usize; k];
    for &r in &regimes {
        counts[r] += 1;
    }
    let fractions: Vec<f64> = counts.iter().map(|&c| c as f64 / n_subs as f64).collect();

    SnippetsResult {
        indices: result_indices,
        profiles: result_profiles,
        fractions,
        areas: result_areas,
        regimes,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_snippets_single() {
        // Repeating sine wave: one snippet should capture almost everything
        let n = 500;
        let m = 50;
        let ts: Vec<f64> = (0..n)
            .map(|i| (i as f64 * 2.0 * std::f64::consts::PI / 50.0).sin())
            .collect();

        let result = find_snippets(&ts, m, 1);
        assert_eq!(result.indices.len(), 1);
        assert_eq!(result.profiles.len(), 1);
        assert_eq!(result.profiles[0].len(), n - m + 1);
        assert_eq!(result.fractions.len(), 1);
        assert_eq!(result.areas.len(), 1);
        assert_eq!(result.regimes.len(), n - m + 1);

        // Single snippet covers everything
        assert!((result.fractions[0] - 1.0).abs() < 1e-10);
        assert!(result.regimes.iter().all(|&r| r == 0));
    }

    #[test]
    fn test_snippets_fractions_sum_to_one() {
        let n = 600;
        let m = 50;
        let k = 3;

        // Multi-regime signal
        let mut ts = vec![0.0; n];
        for (i, v) in ts.iter_mut().enumerate().take(200) {
            *v = (i as f64 * 2.0 * std::f64::consts::PI / 50.0).sin();
        }
        for (i, v) in ts.iter_mut().enumerate().skip(200).take(200) {
            let phase = (i - 200) % 30;
            *v = (phase as f64 / 30.0) * 2.0 - 1.0;
        }
        for (i, v) in ts.iter_mut().enumerate().skip(400) {
            *v = (i as f64 * 2.0 * std::f64::consts::PI / 25.0).cos() * 0.5;
        }

        let result = find_snippets(&ts, m, k);
        assert_eq!(result.indices.len(), k);

        let frac_sum: f64 = result.fractions.iter().sum();
        assert!(
            (frac_sum - 1.0).abs() < 1e-10,
            "Fractions should sum to 1.0, got {frac_sum}"
        );
    }

    #[test]
    fn test_snippets_decreasing_areas() {
        // Areas should be non-increasing: each added snippet reduces coverage area
        let n = 500;
        let m = 50;
        let k = 3;
        let ts: Vec<f64> = (0..n)
            .map(|i| (i as f64 * 0.1).sin() + (i as f64 * 0.03).cos())
            .collect();

        let result = find_snippets(&ts, m, k);
        for w in result.areas.windows(2) {
            assert!(
                w[0] >= w[1] - 1e-10,
                "Areas should be non-increasing: {} > {}",
                w[0],
                w[1]
            );
        }
    }

    #[test]
    #[should_panic(expected = "Cannot extract more snippets")]
    fn test_snippets_k_too_large() {
        let ts: Vec<f64> = (0..150).map(|i| (i as f64 * 0.1).sin()).collect();
        // n=150, m=50, n_subs=101, s=101/50=2, but k=5 > 2
        find_snippets(&ts, 50, 5);
    }

    #[test]
    #[should_panic(expected = "Time series must be at least 2*m long")]
    fn test_snippets_ts_too_short() {
        let ts = vec![1.0; 50];
        find_snippets(&ts, 50, 1);
    }
}
