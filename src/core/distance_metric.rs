/// Trait for distance metrics used in matrix profile computation.
///
/// Designed for static polymorphism: algorithms are generic over `M: DistanceMetric`,
/// enabling monomorphization, inlining, and autovectorization in the inner loop.
///
/// The associated `Context` type holds precomputed statistics (e.g., rolling means/stds
/// for Euclidean). This avoids recomputing per-pair statistics in O(n^2) calls.
pub trait DistanceMetric: Clone + Send + Sync {
    /// Precomputed context for the metric (e.g., rolling statistics).
    type Context: Clone + Send + Sync;

    /// Precompute context from a time series and subsequence length.
    fn precompute(ts: &[f64], m: usize) -> Self::Context;

    /// Compute distance between subsequences starting at indices `i` and `j`.
    fn distance(ts: &[f64], i: usize, j: usize, m: usize, ctx: &Self::Context) -> f64;

    /// Compute the distance profile for subsequence at `idx` against all others.
    ///
    /// Default implementation loops over `distance()`. Metrics can override with
    /// batch-optimized versions (e.g., using sliding dot product for Euclidean).
    fn distance_profile(ts: &[f64], idx: usize, m: usize, ctx: &Self::Context) -> Vec<f64> {
        let n_subs = ts.len() - m + 1;
        (0..n_subs)
            .map(|j| Self::distance(ts, idx, j, m, ctx))
            .collect()
    }

    /// Whether this metric supports the QT (dot product) incremental update optimization.
    ///
    /// When true, STOMP can use O(1) QT updates per element instead of recomputing
    /// full dot products. The compiler eliminates the dead branch via monomorphization.
    fn supports_qt_optimization() -> bool {
        false
    }

    /// Convert a dot product value to a distance, given precomputed context.
    ///
    /// Only meaningful when `supports_qt_optimization()` returns true.
    /// For Euclidean: `d = sqrt(2*m*(1 - (QT - m*mu_i*mu_j) / (m*sigma_i*sigma_j)))`.
    fn qt_to_distance(_qt: f64, _i: usize, _j: usize, _m: usize, _ctx: &Self::Context) -> f64 {
        unimplemented!("qt_to_distance not supported for this metric")
    }

    /// Whether this metric supports the correlation-domain optimization for diagonal STOMP.
    ///
    /// When true, the inner loop stores negated Pearson correlations instead of distances,
    /// deferring the sqrt to a single O(n) pass at the end. Combined with precomputed
    /// inverse standard deviations, this eliminates division, sqrt, and clamping from
    /// the ~O(n^2) inner loop.
    fn supports_correlation_domain() -> bool {
        false
    }

    /// Extract correlation-domain data from the precomputed context.
    ///
    /// Returns `(mean, m_sigma_inv, has_constant)` where:
    /// - `mean[i]` = mean of subsequence i
    /// - `m_sigma_inv[i]` = `1 / (sqrt(m) * sigma_i)` (0 for constant subsequences)
    /// - `has_constant` = whether any subsequence is constant
    ///
    /// Only called when `supports_correlation_domain()` is true.
    fn correlation_data(_ctx: &Self::Context) -> (&[f64], &[f64], bool) {
        unreachable!("correlation_data not supported for this metric")
    }

    /// Update context incrementally after appending a new point to the time series.
    ///
    /// Used by streaming algorithms (STAMPI) to avoid full recomputation.
    fn update_context(ctx: &mut Self::Context, ts: &[f64], m: usize);

    // --- AB-Join methods (defaulted so existing metrics compile without changes) ---

    /// Whether this metric supports AB-join (cross-series comparison).
    fn supports_ab_join() -> bool {
        false
    }

    /// Convert a dot product to a distance for AB-join, using two separate contexts.
    ///
    /// Only meaningful when `supports_ab_join()` returns true.
    fn qt_to_distance_ab(
        _qt: f64,
        _i: usize,
        _j: usize,
        _m: usize,
        _ctx_a: &Self::Context,
        _ctx_b: &Self::Context,
    ) -> f64 {
        unimplemented!("qt_to_distance_ab not supported for this metric")
    }

    /// Extract correlation-domain data for AB-join from two separate contexts.
    ///
    /// Returns `(mean_a, m_sigma_inv_a, mean_b, m_sigma_inv_b, has_constant)`
    ///
    /// Only called when both `supports_correlation_domain()` and `supports_ab_join()` are true.
    fn correlation_data_ab<'a>(
        _ctx_a: &'a Self::Context,
        _ctx_b: &'a Self::Context,
    ) -> (&'a [f64], &'a [f64], &'a [f64], &'a [f64], bool) {
        unreachable!("correlation_data_ab not supported for this metric")
    }
}
