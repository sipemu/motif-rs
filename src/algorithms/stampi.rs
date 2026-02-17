use crate::algorithms::common::apply_exclusion_zone;
use crate::algorithms::stomp::stomp;
use crate::core::distance_metric::DistanceMetric;
use crate::core::matrix_profile::{MatrixProfile, MatrixProfileConfig};

/// Streaming matrix profile computation using the STAMPI algorithm.
///
/// Supports two modes:
/// - **Grow mode** (`egress=false`): The time series grows unboundedly. Each new point
///   extends the profile by one entry and updates existing entries.
/// - **Egress mode** (`egress=true`): Fixed-size sliding window. Old points are removed
///   as new ones arrive.
pub struct Stampi<M: DistanceMetric> {
    /// The growing time series.
    ts: Vec<f64>,
    /// Current matrix profile.
    mp: MatrixProfile,
    /// Configuration.
    config: MatrixProfileConfig,
    /// Precomputed metric context.
    ctx: M::Context,
    /// Whether to use egress (sliding window) mode.
    egress: bool,
    /// Window size for egress mode.
    window_size: usize,
}

impl<M: DistanceMetric> Stampi<M> {
    /// Create a new streaming matrix profile from an initial time series.
    ///
    /// Computes the full batch matrix profile on the initial data, then allows
    /// incremental updates via `update()`.
    pub fn new(initial_ts: &[f64], config: MatrixProfileConfig, egress: bool) -> Self {
        assert!(
            initial_ts.len() >= config.m,
            "Initial time series must be at least as long as m"
        );

        let mp = stomp::<M>(initial_ts, &config);
        let ctx = M::precompute(initial_ts, config.m);
        let window_size = initial_ts.len();

        Self {
            ts: initial_ts.to_vec(),
            mp,
            config,
            ctx,
            egress,
            window_size,
        }
    }

    /// Append a new point and update the matrix profile.
    ///
    /// In grow mode: extends the time series and profile.
    /// In egress mode: slides the window, removing the oldest point.
    pub fn update(&mut self, new_val: f64) {
        if self.egress {
            self.update_egress(new_val);
        } else {
            self.update_grow(new_val);
        }
    }

    /// Grow mode: append point, extend profile, update distances.
    fn update_grow(&mut self, new_val: f64) {
        let m = self.config.m;
        let exclusion_zone = self.config.exclusion_zone();

        // Append the new point
        self.ts.push(new_val);

        // Update the metric context with the new subsequence
        M::update_context(&mut self.ctx, &self.ts, m);

        let n = self.ts.len();
        let new_n_subs = n - m + 1;
        let new_idx = new_n_subs - 1;

        // Compute distance profile for the new subsequence against all others
        let mut dist_profile = M::distance_profile(&self.ts, new_idx, m, &self.ctx);
        apply_exclusion_zone(&mut dist_profile, new_idx, exclusion_zone);

        // Extend the matrix profile arrays by one
        self.mp.profile.push(f64::INFINITY);
        self.mp.profile_index.push(0);
        self.mp.left_profile.push(f64::INFINITY);
        self.mp.left_profile_index.push(0);
        self.mp.right_profile.push(f64::INFINITY);
        self.mp.right_profile_index.push(0);

        // Update: new subsequence vs all existing, and vice versa
        for (j, &d) in dist_profile.iter().enumerate() {
            // Update the new entry
            self.mp.update(new_idx, d, j);
            // Update existing entries if new subsequence is closer
            self.mp.update(j, d, new_idx);
        }
    }

    /// Egress mode: slide window, remove oldest point, add new point.
    /// Delegates to batch stomp() which automatically uses parallelization when available.
    fn update_egress(&mut self, new_val: f64) {
        self.ts.push(new_val);

        let excess = self.ts.len().saturating_sub(self.window_size);
        if excess > 0 {
            self.ts.drain(0..excess);
        }

        self.ctx = M::precompute(&self.ts, self.config.m);
        self.mp = stomp::<M>(&self.ts, &self.config);
    }

    /// Get a reference to the current matrix profile.
    pub fn profile(&self) -> &MatrixProfile {
        &self.mp
    }

    /// Get a reference to the current time series.
    pub fn time_series(&self) -> &[f64] {
        &self.ts
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metrics::euclidean::ZNormalizedEuclidean;

    #[test]
    fn test_stampi_grow_matches_batch() {
        // Build a time series incrementally and compare against batch STOMP
        let full_ts: Vec<f64> = (0..30).map(|i| (i as f64 * 0.5).sin()).collect();
        let m = 4;
        let config = MatrixProfileConfig::new(m);

        // Start with first 10 points
        let initial = &full_ts[..10];
        let mut stampi = Stampi::<ZNormalizedEuclidean>::new(initial, config.clone(), false);

        // Feed remaining points one at a time
        for &val in &full_ts[10..] {
            stampi.update(val);
        }

        // Compare against batch STOMP on the full series
        let batch_mp = stomp::<ZNormalizedEuclidean>(&full_ts, &config);

        assert_eq!(stampi.profile().profile.len(), batch_mp.profile.len());
        for i in 0..batch_mp.profile.len() {
            assert!(
                (stampi.profile().profile[i] - batch_mp.profile[i]).abs() < 1e-9,
                "Mismatch at index {i}: streaming={}, batch={}",
                stampi.profile().profile[i],
                batch_mp.profile[i]
            );
        }
    }

    #[test]
    fn test_stampi_grow_single_update() {
        let ts = vec![1.0, 2.0, 3.0, 2.0, 1.0, 2.0, 3.0];
        let config = MatrixProfileConfig::new(3);

        // Batch on first 6 points
        let mut stampi = Stampi::<ZNormalizedEuclidean>::new(&ts[..6], config.clone(), false);
        let old_len = stampi.profile().profile.len();

        // Add one point
        stampi.update(ts[6]);
        assert_eq!(stampi.profile().profile.len(), old_len + 1);

        // Compare against batch on full series
        let batch_mp = stomp::<ZNormalizedEuclidean>(&ts, &config);
        for i in 0..batch_mp.profile.len() {
            assert!(
                (stampi.profile().profile[i] - batch_mp.profile[i]).abs() < 1e-9,
                "Mismatch at index {i}: streaming={}, batch={}",
                stampi.profile().profile[i],
                batch_mp.profile[i]
            );
        }
    }
}
