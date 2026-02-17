/// Configuration for matrix profile computation.
#[derive(Debug, Clone)]
pub struct MatrixProfileConfig {
    /// Subsequence length.
    pub m: usize,
    /// Whether to apply an exclusion zone around trivial matches.
    pub ignore_trivial: bool,
    /// Exclusion zone denominator: zone = ceil(m / exclusion_zone_denom).
    /// Default is 4 to match stumpy.
    pub exclusion_zone_denom: usize,
}

impl MatrixProfileConfig {
    pub fn new(m: usize) -> Self {
        Self {
            m,
            ignore_trivial: true,
            exclusion_zone_denom: 4,
        }
    }

    /// Compute the exclusion zone radius.
    pub fn exclusion_zone(&self) -> usize {
        if self.ignore_trivial {
            (self.m as f64 / self.exclusion_zone_denom as f64).ceil() as usize
        } else {
            0
        }
    }
}

/// The matrix profile result.
#[derive(Debug, Clone)]
pub struct MatrixProfile {
    /// Nearest-neighbor distances for each subsequence.
    pub profile: Vec<f64>,
    /// Index of the nearest neighbor for each subsequence.
    pub profile_index: Vec<usize>,
    /// Left nearest-neighbor distances (neighbors with smaller index).
    pub left_profile: Vec<f64>,
    /// Index of the left nearest neighbor.
    pub left_profile_index: Vec<usize>,
    /// Right nearest-neighbor distances (neighbors with larger index).
    pub right_profile: Vec<f64>,
    /// Index of the right nearest neighbor.
    pub right_profile_index: Vec<usize>,
    /// Subsequence length used.
    pub m: usize,
    /// Exclusion zone radius used.
    pub exclusion_zone: usize,
}

impl MatrixProfile {
    /// Create a new matrix profile initialized to infinity distances.
    pub fn new(n_subs: usize, m: usize, exclusion_zone: usize) -> Self {
        Self {
            profile: vec![f64::INFINITY; n_subs],
            profile_index: vec![0; n_subs],
            left_profile: vec![f64::INFINITY; n_subs],
            left_profile_index: vec![0; n_subs],
            right_profile: vec![f64::INFINITY; n_subs],
            right_profile_index: vec![0; n_subs],
            m,
            exclusion_zone,
        }
    }

    /// Merge another matrix profile into this one, taking element-wise minimums.
    ///
    /// Used for combining thread-local results in parallel STOMP.
    pub fn merge(&mut self, other: &MatrixProfile) {
        debug_assert_eq!(self.profile.len(), other.profile.len());
        for i in 0..self.profile.len() {
            if other.profile[i] < self.profile[i] {
                self.profile[i] = other.profile[i];
                self.profile_index[i] = other.profile_index[i];
            }
            if other.left_profile[i] < self.left_profile[i] {
                self.left_profile[i] = other.left_profile[i];
                self.left_profile_index[i] = other.left_profile_index[i];
            }
            if other.right_profile[i] < self.right_profile[i] {
                self.right_profile[i] = other.right_profile[i];
                self.right_profile_index[i] = other.right_profile_index[i];
            }
        }
    }

    /// Update the profile at `idx` if `distance` is smaller than the current value.
    /// `neighbor_idx` is the index of the matching subsequence.
    #[inline]
    pub fn update(&mut self, idx: usize, distance: f64, neighbor_idx: usize) {
        if distance < self.profile[idx] {
            self.profile[idx] = distance;
            self.profile_index[idx] = neighbor_idx;
        }
        // Update directional profiles
        if neighbor_idx < idx && distance < self.left_profile[idx] {
            self.left_profile[idx] = distance;
            self.left_profile_index[idx] = neighbor_idx;
        }
        if neighbor_idx > idx && distance < self.right_profile[idx] {
            self.right_profile[idx] = distance;
            self.right_profile_index[idx] = neighbor_idx;
        }
    }
}

/// Rolling mean and standard deviation for all subsequences of length `m`.
///
/// Computed via a single-pass sliding window over cumulative sums and
/// sums-of-squares, matching stumpy's numerical approach.
#[derive(Debug, Clone)]
pub struct RollingStats {
    pub mean: Vec<f64>,
    pub std: Vec<f64>,
    /// Precomputed `1 / (sqrt(m) * sigma)` for each subsequence.
    /// Zero for constant subsequences (sigma == 0).
    /// Enables replacing division with multiplication in the inner loop:
    /// `r = (QT - m*mu_i*mu_j) * m_sigma_inv[i] * m_sigma_inv[j]`
    pub m_sigma_inv: Vec<f64>,
    /// Whether any subsequence is constant (has sigma == 0).
    pub has_constant: bool,
}

impl RollingStats {
    /// Compute rolling statistics for subsequences of length `m`.
    ///
    /// Uses the same approach as stumpy: cumulative sums for numerical consistency.
    pub fn compute(ts: &[f64], m: usize) -> Self {
        assert!(m > 0, "Subsequence length must be > 0");
        assert!(ts.len() >= m, "Time series must be at least as long as m");

        let n = ts.len();
        let n_subs = n - m + 1;

        // Cumulative sums for mean
        let mut cumsum = vec![0.0; n + 1];
        let mut cumsum_sq = vec![0.0; n + 1];
        for i in 0..n {
            cumsum[i + 1] = cumsum[i] + ts[i];
            cumsum_sq[i + 1] = cumsum_sq[i] + ts[i] * ts[i];
        }

        let mut mean = vec![0.0; n_subs];
        let mut std = vec![0.0; n_subs];
        let mut m_sigma_inv = vec![0.0; n_subs];
        let mut has_constant = false;

        let m_f = m as f64;
        let sqrt_m = m_f.sqrt();
        for i in 0..n_subs {
            let sum = cumsum[i + m] - cumsum[i];
            let sum_sq = cumsum_sq[i + m] - cumsum_sq[i];
            let mu = sum / m_f;
            // Variance via E[X^2] - E[X]^2, clamped to 0 for numerical stability
            let var = (sum_sq / m_f - mu * mu).max(0.0);
            let sigma = var.sqrt();
            mean[i] = mu;
            std[i] = sigma;
            if sigma < 1e-15 {
                m_sigma_inv[i] = 0.0;
                has_constant = true;
            } else {
                m_sigma_inv[i] = 1.0 / (sqrt_m * sigma);
            }
        }

        Self {
            mean,
            std,
            m_sigma_inv,
            has_constant,
        }
    }

    /// Extend rolling statistics by one new subsequence after appending a point.
    pub fn extend(&mut self, ts: &[f64], m: usize) {
        let n = ts.len();
        assert!(n >= m);
        let start = n - m;
        let m_f = m as f64;

        let sum: f64 = ts[start..n].iter().sum();
        let sum_sq: f64 = ts[start..n].iter().map(|x| x * x).sum();
        let mu = sum / m_f;
        let var = (sum_sq / m_f - mu * mu).max(0.0);
        let sigma = var.sqrt();

        self.mean.push(mu);
        self.std.push(sigma);
        if sigma < 1e-15 {
            self.m_sigma_inv.push(0.0);
            self.has_constant = true;
        } else {
            self.m_sigma_inv.push(1.0 / (m_f.sqrt() * sigma));
        }
    }
}

/// A single entry in the AoS profile accumulator.
///
/// Stores negated Pearson correlations (lower = better match) for the overall,
/// left, and right nearest neighbors. All 6 fields pack into 48 bytes,
/// fitting within a single 64-byte cache line.
#[derive(Clone, Copy)]
#[repr(C)]
pub(crate) struct AccEntry {
    pub neg_corr: f64,
    pub right_neg_corr: f64,
    pub left_neg_corr: f64,
    pub index: usize,
    pub right_index: usize,
    pub left_index: usize,
}

/// Array-of-Structs accumulator for diagonal STOMP.
///
/// Used only during computation; converted to `MatrixProfile` at the end.
/// The AoS layout ensures all fields for one index share a cache line,
/// reducing cache misses from 6 (SoA) to 1 per `update` call.
pub(crate) struct ProfileAccumulator {
    pub entries: Vec<AccEntry>,
}

impl ProfileAccumulator {
    pub fn new(n: usize) -> Self {
        Self {
            entries: vec![
                AccEntry {
                    neg_corr: f64::INFINITY,
                    right_neg_corr: f64::INFINITY,
                    left_neg_corr: f64::INFINITY,
                    index: 0,
                    right_index: 0,
                    left_index: 0,
                };
                n
            ],
        }
    }

    /// Update overall + right profile for `idx` (neighbor has larger index).
    ///
    /// In diagonal traversal with `j = i + k` (k > 0), calling
    /// `update_right(i, neg_r, j)` is correct because j > i.
    /// Eliminates the direction-check branch from the generic `update`.
    #[inline(always)]
    pub fn update_right(&mut self, idx: usize, neg_corr: f64, neighbor: usize) {
        let e = unsafe { self.entries.get_unchecked_mut(idx) };
        if neg_corr < e.neg_corr {
            e.neg_corr = neg_corr;
            e.index = neighbor;
        }
        if neg_corr < e.right_neg_corr {
            e.right_neg_corr = neg_corr;
            e.right_index = neighbor;
        }
    }

    /// Update overall + left profile for `idx` (neighbor has smaller index).
    ///
    /// In diagonal traversal with `j = i + k` (k > 0), calling
    /// `update_left(j, neg_r, i)` is correct because i < j.
    #[inline(always)]
    pub fn update_left(&mut self, idx: usize, neg_corr: f64, neighbor: usize) {
        let e = unsafe { self.entries.get_unchecked_mut(idx) };
        if neg_corr < e.neg_corr {
            e.neg_corr = neg_corr;
            e.index = neighbor;
        }
        if neg_corr < e.left_neg_corr {
            e.left_neg_corr = neg_corr;
            e.left_index = neighbor;
        }
    }

    /// Merge another accumulator into this one, taking element-wise minimums.
    #[cfg(feature = "parallel")]
    pub fn merge(&mut self, other: &Self) {
        for (a, b) in self.entries.iter_mut().zip(other.entries.iter()) {
            if b.neg_corr < a.neg_corr {
                a.neg_corr = b.neg_corr;
                a.index = b.index;
            }
            if b.left_neg_corr < a.left_neg_corr {
                a.left_neg_corr = b.left_neg_corr;
                a.left_index = b.left_index;
            }
            if b.right_neg_corr < a.right_neg_corr {
                a.right_neg_corr = b.right_neg_corr;
                a.right_index = b.right_index;
            }
        }
    }

    /// Convert negated correlations to distances and write into a `MatrixProfile`.
    ///
    /// The `convert` closure applies `d = sqrt(2*m*(1 + neg_corr))` (or similar)
    /// once per element — an O(n) pass that replaces the O(n^2) per-element sqrt
    /// from the inner loop.
    pub fn write_to_matrix_profile(&self, mp: &mut MatrixProfile, convert: impl Fn(f64) -> f64) {
        for (i, e) in self.entries.iter().enumerate() {
            mp.profile[i] = convert(e.neg_corr);
            mp.profile_index[i] = e.index;
            mp.left_profile[i] = convert(e.left_neg_corr);
            mp.left_profile_index[i] = e.left_index;
            mp.right_profile[i] = convert(e.right_neg_corr);
            mp.right_profile_index[i] = e.right_index;
        }
    }
}

/// Result of an AB-join: nearest-neighbor profile for one series against another.
#[derive(Debug, Clone)]
pub struct JoinProfile {
    /// Nearest-neighbor distances for each subsequence.
    pub distances: Vec<f64>,
    /// Index of the nearest neighbor in the *other* series.
    pub indices: Vec<usize>,
    /// Subsequence length used.
    pub m: usize,
}

impl JoinProfile {
    /// Create a new join profile initialized to infinity distances.
    pub fn new(n_subs: usize, m: usize) -> Self {
        Self {
            distances: vec![f64::INFINITY; n_subs],
            indices: vec![0; n_subs],
            m,
        }
    }
}

/// Accumulator for AB-join computation.
///
/// Simpler than `ProfileAccumulator` since there's no left/right distinction
/// in cross-series comparison — just overall nearest neighbor.
pub(crate) struct JoinAccumulator {
    pub neg_corrs: Vec<f64>,
    pub indices: Vec<usize>,
}

impl JoinAccumulator {
    pub fn new(n: usize) -> Self {
        Self {
            neg_corrs: vec![f64::INFINITY; n],
            indices: vec![0; n],
        }
    }

    #[inline(always)]
    pub fn update(&mut self, idx: usize, neg_corr: f64, neighbor: usize) {
        unsafe {
            let curr = self.neg_corrs.get_unchecked_mut(idx);
            if neg_corr < *curr {
                *curr = neg_corr;
                *self.indices.get_unchecked_mut(idx) = neighbor;
            }
        }
    }

    #[cfg(feature = "parallel")]
    pub fn merge(&mut self, other: &Self) {
        for i in 0..self.neg_corrs.len() {
            if other.neg_corrs[i] < self.neg_corrs[i] {
                self.neg_corrs[i] = other.neg_corrs[i];
                self.indices[i] = other.indices[i];
            }
        }
    }

    /// Convert to a JoinProfile, applying a distance conversion function.
    pub fn write_to_join_profile(&self, jp: &mut JoinProfile, convert: impl Fn(f64) -> f64) {
        for (i, (&nc, &idx)) in self.neg_corrs.iter().zip(self.indices.iter()).enumerate() {
            jp.distances[i] = convert(nc);
            jp.indices[i] = idx;
        }
    }
}

/// Accumulator for AB-join with raw distances (no correlation domain).
pub(crate) struct JoinAccumulatorDist {
    pub distances: Vec<f64>,
    pub indices: Vec<usize>,
}

impl JoinAccumulatorDist {
    pub fn new(n: usize) -> Self {
        Self {
            distances: vec![f64::INFINITY; n],
            indices: vec![0; n],
        }
    }

    #[inline(always)]
    pub fn update(&mut self, idx: usize, dist: f64, neighbor: usize) {
        if dist < self.distances[idx] {
            self.distances[idx] = dist;
            self.indices[idx] = neighbor;
        }
    }

    #[cfg(feature = "parallel")]
    pub fn merge(&mut self, other: &Self) {
        for i in 0..self.distances.len() {
            if other.distances[i] < self.distances[i] {
                self.distances[i] = other.distances[i];
                self.indices[i] = other.indices[i];
            }
        }
    }

    pub fn write_to_join_profile(&self, jp: &mut JoinProfile) {
        for i in 0..self.distances.len() {
            jp.distances[i] = self.distances[i];
            jp.indices[i] = self.indices[i];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rolling_stats_simple() {
        // ts = [1, 2, 3, 4, 5], m = 3
        // Subsequences: [1,2,3], [2,3,4], [3,4,5]
        // Means: 2, 3, 4
        // Stds: sqrt(2/3), sqrt(2/3), sqrt(2/3)
        let ts = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = RollingStats::compute(&ts, 3);

        assert_eq!(stats.mean.len(), 3);
        assert!((stats.mean[0] - 2.0).abs() < 1e-10);
        assert!((stats.mean[1] - 3.0).abs() < 1e-10);
        assert!((stats.mean[2] - 4.0).abs() < 1e-10);

        let expected_std = (2.0_f64 / 3.0).sqrt();
        for s in &stats.std {
            assert!((s - expected_std).abs() < 1e-10);
        }
    }

    #[test]
    fn test_rolling_stats_constant() {
        let ts = vec![5.0; 10];
        let stats = RollingStats::compute(&ts, 4);
        for mu in &stats.mean {
            assert!((mu - 5.0).abs() < 1e-10);
        }
        for s in &stats.std {
            assert!(*s < 1e-10);
        }
    }

    #[test]
    fn test_rolling_stats_extend() {
        let mut ts = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let mut stats = RollingStats::compute(&ts, 3);
        assert_eq!(stats.mean.len(), 3);

        // Append a point and extend
        ts.push(6.0);
        stats.extend(&ts, 3);
        assert_eq!(stats.mean.len(), 4);
        assert!((stats.mean[3] - 5.0).abs() < 1e-10); // mean of [4,5,6]
    }

    #[test]
    fn test_matrix_profile_update() {
        let mut mp = MatrixProfile::new(5, 3, 1);

        mp.update(0, 1.5, 3);
        assert!((mp.profile[0] - 1.5).abs() < 1e-10);
        assert_eq!(mp.profile_index[0], 3);
        assert!((mp.right_profile[0] - 1.5).abs() < 1e-10);

        // Smaller distance should replace
        mp.update(0, 0.5, 2);
        assert!((mp.profile[0] - 0.5).abs() < 1e-10);
        assert_eq!(mp.profile_index[0], 2);

        // Larger distance should not replace
        mp.update(0, 2.0, 4);
        assert!((mp.profile[0] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_matrix_profile_merge() {
        let mut a = MatrixProfile::new(3, 4, 1);
        let mut b = MatrixProfile::new(3, 4, 1);

        // Set up a: profile = [1.0, 3.0, 2.0]
        a.update(0, 1.0, 2); // right neighbor
        a.update(1, 3.0, 0); // left neighbor
        a.update(2, 2.0, 0); // left neighbor

        // Set up b: profile = [2.0, 1.0, 1.5]
        b.update(0, 2.0, 1); // right neighbor
        b.update(1, 1.0, 2); // right neighbor
        b.update(2, 1.5, 0); // left neighbor

        a.merge(&b);

        // Element-wise minimums: [1.0, 1.0, 1.5]
        assert!((a.profile[0] - 1.0).abs() < 1e-10);
        assert_eq!(a.profile_index[0], 2); // from a
        assert!((a.profile[1] - 1.0).abs() < 1e-10);
        assert_eq!(a.profile_index[1], 2); // from b
        assert!((a.profile[2] - 1.5).abs() < 1e-10);
        assert_eq!(a.profile_index[2], 0); // from b

        // Left profiles: a had [inf, 3.0, 2.0], b had [inf, inf, 1.5] → [inf, 3.0, 1.5]
        assert!(a.left_profile[0].is_infinite());
        assert!((a.left_profile[1] - 3.0).abs() < 1e-10);
        assert!((a.left_profile[2] - 1.5).abs() < 1e-10);

        // Right profiles: a had [1.0, inf, inf], b had [2.0, 1.0, inf] → [1.0, 1.0, inf]
        assert!((a.right_profile[0] - 1.0).abs() < 1e-10);
        assert!((a.right_profile[1] - 1.0).abs() < 1e-10);
        assert!(a.right_profile[2].is_infinite());
    }

    #[test]
    fn test_exclusion_zone() {
        let config = MatrixProfileConfig::new(8);
        assert_eq!(config.exclusion_zone(), 2); // ceil(8/4) = 2

        let config = MatrixProfileConfig::new(10);
        assert_eq!(config.exclusion_zone(), 3); // ceil(10/4) = 3

        let mut config = MatrixProfileConfig::new(10);
        config.ignore_trivial = false;
        assert_eq!(config.exclusion_zone(), 0);
    }
}
