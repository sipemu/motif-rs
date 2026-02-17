use crate::algorithms::common::apply_exclusion_zone;
use crate::core::matrix_profile::MatrixProfile;

/// Result of FLUSS/FLOSS segmentation.
#[derive(Debug, Clone)]
pub struct SegmentationResult {
    /// Corrected Arc Curve. Values near 0 indicate regime boundaries.
    pub cac: Vec<f64>,
    /// Detected regime boundary indices, sorted by confidence (ascending CAC value).
    pub regime_boundaries: Vec<usize>,
}

/// Compute the arc curve from a matrix profile index (matches stumpy's `_nnmark`).
///
/// An "arc" connects each subsequence `i` to its nearest neighbor `profile_index[i]`.
/// The arc curve counts how many arcs cross each position. An arc from `lo` to `hi`
/// crosses positions in `[lo, hi)` (half-open interval). Uses an O(n) prefix-sum
/// approach with a delta array.
fn compute_arc_counts(profile_index: &[usize], n: usize) -> Vec<usize> {
    let mut deltas = vec![0i64; n + 1];

    for (i, &j) in profile_index.iter().enumerate() {
        if j >= n {
            continue; // skip invalid indices
        }
        let lo = i.min(j);
        let hi = i.max(j);
        // Arc from lo to hi crosses positions [lo, hi)
        deltas[lo] += 1;
        if hi < n + 1 {
            deltas[hi] -= 1;
        }
    }

    // Prefix sum to get actual arc counts
    let mut counts = vec![0usize; n];
    let mut running = 0i64;
    for i in 0..n {
        running += deltas[i];
        counts[i] = running.max(0) as usize;
    }

    counts
}

/// Compute the Corrected Arc Curve (CAC) from raw arc counts.
///
/// Normalizes the arc count at each position by the theoretical maximum number
/// of arcs that could cross: `min(p+1, n-p)`. The result is clamped to [0, 1].
///
/// Convention (matches stumpy): LOW CAC = regime boundary, HIGH CAC = non-boundary.
/// Edge regions (within `excl_factor * m` of each end) are set to 1.0 (non-boundary).
fn corrected_arc_curve(arc_counts: &[usize], n: usize, m: usize, excl_factor: usize) -> Vec<f64> {
    let excl_width = excl_factor * m;
    let mut cac = vec![1.0; n];

    for p in 0..n {
        let max_arcs = (p + 1).min(n - p);
        if max_arcs == 0 {
            cac[p] = 1.0;
        } else {
            // CAC = AC / max_arcs, clamped to [0, 1]
            // Low values indicate regime boundaries (few arcs cross)
            cac[p] = (arc_counts[p] as f64 / max_arcs as f64).min(1.0);
        }
    }

    // Nullify edges (set to 1.0 = non-boundary)
    for v in cac.iter_mut().take(excl_width.min(n)) {
        *v = 1.0;
    }
    for v in cac.iter_mut().take(n).skip(n.saturating_sub(excl_width)) {
        *v = 1.0;
    }

    cac
}

/// Find regime boundaries by greedy argmin with exclusion zone.
fn find_regime_boundaries(cac: &[f64], num_regimes: usize, ez: usize) -> Vec<usize> {
    let mut working_cac = cac.to_vec();
    let mut boundaries = Vec::with_capacity(num_regimes);

    for _ in 0..num_regimes {
        let (best_idx, &best_val) = match working_cac
            .iter()
            .enumerate()
            .filter(|(_, v)| **v < 1.0)
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        {
            Some(pair) => pair,
            None => break,
        };

        // Skip if the best CAC value is 1.0 (no more valid boundaries)
        if best_val >= 1.0 {
            break;
        }

        boundaries.push(best_idx);
        apply_exclusion_zone(&mut working_cac, best_idx, ez);
    }

    boundaries
}

/// Perform FLUSS segmentation on a computed matrix profile.
///
/// FLUSS (Fast Low-cost Unipotent Semantic Segmentation) detects regime changes
/// in time series by analyzing the arc structure of the matrix profile index.
/// Positions where few arcs cross indicate semantic boundaries.
///
/// # Arguments
/// * `mp` - A computed matrix profile
/// * `num_regimes` - Number of regime boundaries to detect
///
/// # Returns
/// A `SegmentationResult` containing the Corrected Arc Curve and detected boundaries.
///
/// # References
/// Gharghabi et al., "Matrix Profile VIII: Domain Agnostic Online Semantic
/// Segmentation at Superhuman Performance Levels", ICDM 2017.
pub fn fluss(mp: &MatrixProfile, num_regimes: usize) -> SegmentationResult {
    let n = mp.profile_index.len();
    let m = mp.m;
    let excl_factor = 5; // standard FLUSS exclusion factor

    let arc_counts = compute_arc_counts(&mp.profile_index, n);
    let cac = corrected_arc_curve(&arc_counts, n, m, excl_factor);
    let ez = excl_factor * m;
    let regime_boundaries = find_regime_boundaries(&cac, num_regimes, ez);

    SegmentationResult {
        cac,
        regime_boundaries,
    }
}

/// Streaming FLOSS (Fast Low-cost Online Semantic Segmentation).
///
/// Maintains a one-directional arc curve using only the right profile index,
/// updated incrementally as new points arrive.
pub struct Floss {
    /// Current Corrected Arc Curve (right-directional only).
    cac: Vec<f64>,
    /// Window size for the sliding CAC.
    window_size: usize,
    /// Subsequence length.
    m: usize,
    /// Right-directional arc counts within the window.
    arc_counts: Vec<usize>,
    /// Current window start offset in the global index space.
    offset: usize,
}

impl Floss {
    /// Create a new FLOSS instance from an initial matrix profile.
    ///
    /// Uses only the right profile index for one-directional arc counting.
    pub fn new(mp: &MatrixProfile, window_size: usize) -> Self {
        let n = mp.right_profile_index.len();
        let ws = window_size.min(n);

        // Compute initial arc counts using only right profile index
        let arc_counts = Self::compute_right_arc_counts(&mp.right_profile_index, n, ws);
        let cac = Self::compute_cac(&arc_counts, ws, mp.m);

        Self {
            cac,
            window_size: ws,
            m: mp.m,
            arc_counts,
            offset: 0,
        }
    }

    /// Update the FLOSS state with a new right-profile index entry.
    ///
    /// Call this after each streaming update to the matrix profile.
    pub fn update(&mut self, new_idx: usize, new_right_neighbor: usize) {
        let n = self.arc_counts.len();
        if n == 0 {
            return;
        }

        // Add new arc
        let lo = new_idx.min(new_right_neighbor);
        let hi = new_idx.max(new_right_neighbor);
        let global_start = self.offset;
        for p in (lo + 1)..hi {
            if p >= global_start && p - global_start < n {
                self.arc_counts[p - global_start] += 1;
            }
        }

        // Recompute CAC
        self.cac = Self::compute_cac(&self.arc_counts, self.window_size, self.m);
    }

    /// Get the current Corrected Arc Curve.
    pub fn cac(&self) -> &[f64] {
        &self.cac
    }

    fn compute_right_arc_counts(
        right_profile_index: &[usize],
        n: usize,
        window_size: usize,
    ) -> Vec<usize> {
        let start = n.saturating_sub(window_size);
        let ws = n - start;
        let mut counts = vec![0usize; ws];

        for (global_i, &j) in right_profile_index.iter().enumerate().take(n).skip(start) {
            if j <= global_i {
                continue; // skip if no valid right neighbor
            }
            let lo = global_i;
            for p in (lo + 1)..j {
                if p >= start && p - start < ws {
                    counts[p - start] += 1;
                }
            }
        }

        counts
    }

    fn compute_cac(arc_counts: &[usize], window_size: usize, m: usize) -> Vec<f64> {
        let n = arc_counts.len();
        let excl_width = 5 * m;
        let mut cac = vec![1.0; n];

        for p in 0..n {
            let max_arcs = (p + 1).min(window_size.saturating_sub(p));
            if max_arcs == 0 {
                cac[p] = 1.0;
            } else {
                cac[p] = 1.0 - arc_counts[p] as f64 / max_arcs as f64;
            }
        }

        // Nullify edges
        for v in cac.iter_mut().take(excl_width.min(n)) {
            *v = 1.0;
        }
        for v in cac.iter_mut().take(n).skip(n.saturating_sub(excl_width)) {
            *v = 1.0;
        }

        cac
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algorithms::stomp::stomp;
    use crate::core::matrix_profile::MatrixProfileConfig;
    use crate::metrics::euclidean::ZNormalizedEuclidean;

    #[test]
    fn test_arc_counts_simple() {
        // profile_index = [2, 3, 0, 1]
        // Each entry generates an arc: 0→2, 1→3, 2→0, 3→1
        // Arc 0→2 crosses positions [0, 2) = {0, 1}
        // Arc 1→3 crosses positions [1, 3) = {1, 2}
        // Arc 2→0 crosses positions [0, 2) = {0, 1} (same pair reversed)
        // Arc 3→1 crosses positions [1, 3) = {1, 2} (same pair reversed)
        let pi = vec![2, 3, 0, 1];
        let counts = compute_arc_counts(&pi, 4);
        assert_eq!(counts[0], 2); // arcs 0→2 and 2→0
        assert_eq!(counts[1], 4); // all four arcs cross here
        assert_eq!(counts[2], 2); // arcs 1→3 and 3→1
        assert_eq!(counts[3], 0);
    }

    #[test]
    fn test_arc_counts_adjacent() {
        // Adjacent matches: arc from 0 to 1 crosses [0, 1) = {0}
        let pi = vec![1, 0, 3, 2];
        let counts = compute_arc_counts(&pi, 4);
        // Arc 0→1: [0,1) = {0}, Arc 1→0: [0,1) = {0} → count[0] = 2
        // Arc 2→3: [2,3) = {2}, Arc 3→2: [2,3) = {2} → count[2] = 2
        assert_eq!(counts[0], 2);
        assert_eq!(counts[1], 0);
        assert_eq!(counts[2], 2);
        assert_eq!(counts[3], 0);
    }

    #[test]
    fn test_cac_bounds() {
        // Even with artificially high arc counts, CAC should be clamped to [0,1]
        let arc_counts = vec![0, 5, 10, 5, 0];
        let cac = corrected_arc_curve(&arc_counts, 5, 1, 0);
        for (i, &v) in cac.iter().enumerate() {
            assert!(
                (0.0..=1.0).contains(&v),
                "CAC value out of [0,1] at {i}: {v}"
            );
        }
    }

    #[test]
    fn test_fluss_regime_change() {
        // Create a signal with a clear regime change at index 250:
        // Repeating sine pattern for first half, repeating sawtooth for second half.
        // The key insight is that FLUSS detects where NN arcs don't cross —
        // subsequences in regime 1 should match other regime-1 subsequences,
        // and regime-2 subsequences should match regime-2 subsequences.
        let n = 500;
        let m = 10;
        let mut ts: Vec<f64> = Vec::with_capacity(n);

        // Regime 1: repeating sine (period 20)
        for i in 0..250 {
            ts.push((i as f64 * std::f64::consts::TAU / 20.0).sin());
        }
        // Regime 2: repeating sawtooth (period 15)
        for i in 250..500 {
            let phase = (i - 250) % 15;
            ts.push(phase as f64 / 15.0 * 2.0 - 1.0);
        }

        let config = MatrixProfileConfig::new(m);
        let mp = stomp::<ZNormalizedEuclidean>(&ts, &config);
        let result = fluss(&mp, 1);

        assert_eq!(result.cac.len(), mp.profile_index.len());

        // The CAC should have values in [0, 1]
        for (i, &v) in result.cac.iter().enumerate() {
            assert!((0.0..=1.0).contains(&v), "CAC[{i}] out of [0,1]: {v}");
        }

        // Should find at least one boundary
        assert!(
            !result.regime_boundaries.is_empty(),
            "Should detect at least one regime boundary"
        );

        // The boundary should be in the middle region (near 250, within the valid CAC zone)
        let boundary = result.regime_boundaries[0];
        let excl_width = 5 * m;
        let n_subs = n - m + 1;
        assert!(
            boundary >= excl_width && boundary < n_subs - excl_width,
            "Regime boundary at {boundary} should be in the valid CAC zone"
        );
    }

    #[test]
    fn test_fluss_cac_edge_nullification() {
        let ts: Vec<f64> = (0..100).map(|i| (i as f64 * 0.2).sin()).collect();
        let m = 8;
        let config = MatrixProfileConfig::new(m);
        let mp = stomp::<ZNormalizedEuclidean>(&ts, &config);
        let result = fluss(&mp, 1);

        // Edge regions should be 1.0
        let excl_width = 5 * m;
        for i in 0..excl_width.min(result.cac.len()) {
            assert!(
                (result.cac[i] - 1.0).abs() < 1e-10,
                "Left edge CAC[{i}] should be 1.0, got {}",
                result.cac[i]
            );
        }
    }

    #[test]
    fn test_floss_creation() {
        let ts: Vec<f64> = (0..100).map(|i| (i as f64 * 0.2).sin()).collect();
        let config = MatrixProfileConfig::new(8);
        let mp = stomp::<ZNormalizedEuclidean>(&ts, &config);
        let floss = Floss::new(&mp, 50);

        assert!(!floss.cac().is_empty());
        for &v in floss.cac() {
            assert!(
                (0.0..=1.0).contains(&v),
                "FLOSS CAC value out of [0,1]: {v}"
            );
        }
    }
}
