use crate::algorithms::common::{apply_exclusion_zone, sliding_dot_product};
use crate::core::matrix_profile::RollingStats;

/// A single match result from pattern matching.
#[derive(Debug, Clone)]
pub struct Match {
    /// Index of the matching subsequence in the time series.
    pub index: usize,
    /// Z-normalized Euclidean distance between query and this subsequence.
    pub distance: f64,
}

/// Compute the z-normalized distance profile for a query against a time series.
///
/// Implements the MASS (Mueen's Algorithm for Similarity Search) algorithm:
/// 1. Compute rolling statistics (mean, std) for the time series
/// 2. Compute the sliding dot product between query and time series
/// 3. Convert QT values to z-normalized Euclidean distances
///
/// # Arguments
/// * `query` - The query subsequence
/// * `ts` - The time series to search
///
/// # Returns
/// A distance profile of length `ts.len() - query.len() + 1`.
///
/// # Panics
/// Panics if `ts.len() < query.len()` or `query.len() == 0`.
pub fn mass(query: &[f64], ts: &[f64]) -> Vec<f64> {
    let m = query.len();
    assert!(m > 0, "Query must be non-empty");
    assert!(
        ts.len() >= m,
        "Time series must be at least as long as query"
    );

    let n_subs = ts.len() - m + 1;

    // Rolling stats for the time series
    let stats = RollingStats::compute(ts, m);

    // Query stats
    let m_f = m as f64;
    let mu_q: f64 = query.iter().sum::<f64>() / m_f;
    let sum_sq_q: f64 = query.iter().map(|x| x * x).sum::<f64>();
    let var_q = (sum_sq_q / m_f - mu_q * mu_q).max(0.0);
    let sigma_q = var_q.sqrt();

    // Sliding dot product
    let qt = sliding_dot_product(query, ts);

    // Convert to distances
    let mut profile = vec![f64::INFINITY; n_subs];

    if sigma_q < 1e-15 {
        // Query is constant
        for (i, d) in profile.iter_mut().enumerate() {
            if stats.std[i] < 1e-15 {
                // Both constant -> distance 0
                *d = 0.0;
            } else {
                // One constant, one not -> sqrt(2*m)
                *d = (2.0 * m_f).sqrt();
            }
        }
    } else {
        for (i, d) in profile.iter_mut().enumerate() {
            if stats.std[i] < 1e-15 {
                // Subsequence is constant, query is not -> sqrt(2*m)
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

/// Find all subsequences in a time series that match a query within a distance threshold.
///
/// Implements the stumpy.match algorithm:
/// 1. Compute distance profile via MASS
/// 2. Determine threshold (default: `max(mean(D) - 2*std(D), min(D))`)
/// 3. Iteratively extract matches: find the minimum distance below threshold,
///    record it, apply exclusion zone, repeat
///
/// # Arguments
/// * `query` - The query subsequence
/// * `ts` - The time series to search
/// * `max_distance` - Maximum distance threshold. If `None`, uses stumpy's default.
/// * `exclusion_zone` - Exclusion zone radius. If `None`, uses `m / 4`.
///
/// # Returns
/// Matches sorted by distance (ascending).
pub fn find_matches(
    query: &[f64],
    ts: &[f64],
    max_distance: Option<f64>,
    exclusion_zone: Option<usize>,
) -> Vec<Match> {
    let m = query.len();
    let dp = mass(query, ts);
    let n_subs = dp.len();

    let ez = exclusion_zone.unwrap_or((m as f64 / 4.0).ceil() as usize);

    // Determine threshold
    let max_dist = match max_distance {
        Some(d) => d,
        None => {
            // stumpy default: max(mean(D) - 2*std(D), min(D))
            let finite_vals: Vec<f64> = dp.iter().copied().filter(|d| d.is_finite()).collect();
            if finite_vals.is_empty() {
                return Vec::new();
            }
            let n_f = finite_vals.len() as f64;
            let mean_d = finite_vals.iter().sum::<f64>() / n_f;
            let var_d = finite_vals
                .iter()
                .map(|d| (d - mean_d).powi(2))
                .sum::<f64>()
                / n_f;
            let std_d = var_d.sqrt();
            let min_d = finite_vals.iter().copied().fold(f64::INFINITY, f64::min);
            (mean_d - 2.0 * std_d).max(min_d)
        }
    };

    // Greedy extraction with exclusion zone
    let mut working = dp;
    let mut matches = Vec::new();

    loop {
        // Find minimum finite distance
        let (best_idx, best_dist) = working
            .iter()
            .enumerate()
            .filter(|(_, d)| d.is_finite())
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, &d)| (i, d))
            .unwrap_or((0, f64::INFINITY));

        if !best_dist.is_finite() || best_dist > max_dist {
            break;
        }

        matches.push(Match {
            index: best_idx,
            distance: best_dist,
        });

        apply_exclusion_zone(&mut working, best_idx, ez.max(1).min(n_subs));
    }

    matches
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mass_self_match() {
        // The distance profile for a query extracted from the time series
        // should have a near-zero entry at the extraction point
        let ts: Vec<f64> = (0..200)
            .map(|i| (i as f64 * 2.0 * std::f64::consts::PI / 50.0).sin())
            .collect();
        let query = &ts[50..80]; // m=30

        let dp = mass(query, &ts);
        assert_eq!(dp.len(), ts.len() - 30 + 1);

        // Distance at index 50 should be ~0
        assert!(
            dp[50] < 1e-6,
            "Self-match distance should be ~0, got {}",
            dp[50]
        );
    }

    #[test]
    fn test_mass_constant_query() {
        let ts: Vec<f64> = (0..100).map(|i| (i as f64 * 0.1).sin()).collect();
        let query = vec![5.0; 10]; // constant query

        let dp = mass(&query, &ts);
        assert_eq!(dp.len(), 91);
        // All distances should be sqrt(2*m) = sqrt(20) since no ts subsequence is constant
        let expected = (2.0 * 10.0_f64).sqrt();
        for (i, &d) in dp.iter().enumerate() {
            assert!(
                (d - expected).abs() < 1e-6,
                "Constant query vs non-constant ts[{i}]: expected {expected}, got {d}"
            );
        }
    }

    #[test]
    fn test_mass_distances_non_negative() {
        let ts: Vec<f64> = (0..300)
            .map(|i| (i as f64 * 0.1).sin() + (i as f64 * 0.03).cos())
            .collect();
        let query = &ts[10..30];
        let dp = mass(query, &ts);
        for (i, &d) in dp.iter().enumerate() {
            assert!(d >= 0.0, "Distance at {i} is negative: {d}");
        }
    }

    #[test]
    fn test_find_matches_basic() {
        // Sine wave: query from one period should match all other periods
        let n = 500;
        let period = 50;
        let ts: Vec<f64> = (0..n)
            .map(|i| (i as f64 * 2.0 * std::f64::consts::PI / period as f64).sin())
            .collect();
        let query = &ts[0..period];

        let matches = find_matches(query, &ts, Some(0.5), None);
        assert!(
            !matches.is_empty(),
            "Should find at least one match in periodic signal"
        );
        // First match should have very small distance
        assert!(
            matches[0].distance < 1e-6,
            "Best match should be near-exact, got {}",
            matches[0].distance
        );
    }

    #[test]
    fn test_find_matches_sorted() {
        let ts: Vec<f64> = (0..300).map(|i| (i as f64 * 0.15).sin()).collect();
        let query = &ts[50..70];

        let matches = find_matches(query, &ts, None, None);
        for w in matches.windows(2) {
            assert!(
                w[0].distance <= w[1].distance + 1e-10,
                "Matches should be sorted by distance: {} > {}",
                w[0].distance,
                w[1].distance
            );
        }
    }

    #[test]
    fn test_find_matches_with_exclusion_zone() {
        let ts: Vec<f64> = (0..200).map(|i| (i as f64 * 0.15).sin()).collect();
        let query = &ts[50..60]; // m=10
        let ez = 3;

        let matches = find_matches(query, &ts, Some(2.0), Some(ez));

        // No two match indices should be within the exclusion zone
        for i in 0..matches.len() {
            for j in (i + 1)..matches.len() {
                let diff = matches[i].index.abs_diff(matches[j].index);
                assert!(
                    diff > ez,
                    "Matches at {} and {} are within exclusion zone (diff={diff}, ez={ez})",
                    matches[i].index,
                    matches[j].index
                );
            }
        }
    }

    #[test]
    fn test_mass_empty_result() {
        let ts = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let query = &ts[0..3];
        let dp = mass(query, &ts);
        assert_eq!(dp.len(), 3);
    }
}
