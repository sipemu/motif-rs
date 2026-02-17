pub mod algorithms;
pub mod core;
pub mod metrics;

pub use crate::algorithms::ab_join::ab_join;
pub use crate::algorithms::fluss::{fluss, Floss, SegmentationResult};
pub use crate::algorithms::motifs::{find_discords, find_motifs, Discord, Motif};
pub use crate::algorithms::snippets::{find_snippets, SnippetsResult};
pub use crate::algorithms::topk::TopKMatrixProfile;
pub use crate::core::distance_metric::DistanceMetric;
pub use crate::core::matrix_profile::{
    JoinProfile, MatrixProfile, MatrixProfileConfig, RollingStats,
};
pub use crate::metrics::absolute::AbsoluteEuclidean;
pub use crate::metrics::euclidean::ZNormalizedEuclidean;

use crate::algorithms::stampi::Stampi;
use crate::algorithms::stomp::stomp;

/// High-level facade for matrix profile computation, generic over distance metric.
///
/// # Examples
///
/// ```
/// use motif_rs::{EuclideanEngine, MatrixProfileConfig};
///
/// let ts = vec![1.0, 2.0, 3.0, 2.0, 1.0, 2.0, 3.0, 2.0, 1.0];
/// let engine = EuclideanEngine::new(MatrixProfileConfig::new(4));
/// let mp = engine.compute(&ts);
/// assert_eq!(mp.profile.len(), ts.len() - 4 + 1);
/// ```
pub struct Engine<M: DistanceMetric> {
    config: MatrixProfileConfig,
    _metric: std::marker::PhantomData<M>,
}

impl<M: DistanceMetric> Engine<M> {
    /// Create a new engine with the given configuration.
    pub fn new(config: MatrixProfileConfig) -> Self {
        Self {
            config,
            _metric: std::marker::PhantomData,
        }
    }

    /// Compute the full matrix profile for a time series (batch STOMP).
    pub fn compute(&self, ts: &[f64]) -> MatrixProfile {
        stomp::<M>(ts, &self.config)
    }

    /// Create a streaming matrix profile from an initial time series.
    ///
    /// - `egress=false`: grow mode — time series extends unboundedly.
    /// - `egress=true`: egress mode — fixed-size sliding window.
    pub fn streaming(&self, initial_ts: &[f64], egress: bool) -> Stampi<M> {
        Stampi::<M>::new(initial_ts, self.config.clone(), egress)
    }

    /// Compute the AB-join between two time series.
    ///
    /// Returns two `JoinProfile`s: one for each series against the other.
    pub fn ab_join(&self, ts_a: &[f64], ts_b: &[f64]) -> (JoinProfile, JoinProfile) {
        crate::algorithms::ab_join::ab_join::<M>(ts_a, ts_b, self.config.m)
    }

    /// Compute the top-k matrix profile for a time series.
    ///
    /// Stores the k nearest neighbors for each subsequence, rather than just the best one.
    pub fn compute_topk(&self, ts: &[f64], k: usize) -> TopKMatrixProfile {
        crate::algorithms::topk::stomp_topk::<M>(ts, &self.config, k)
    }

    /// Extract `k` representative snippets that best summarize the time series.
    ///
    /// Uses z-normalized Euclidean distance profiles regardless of the engine's metric.
    pub fn snippets(&self, ts: &[f64], k: usize) -> SnippetsResult {
        crate::algorithms::snippets::find_snippets(ts, self.config.m, k)
    }
}

/// Convenience type alias for the most common use case.
pub type EuclideanEngine = Engine<ZNormalizedEuclidean>;

/// Convenience type alias for non-normalized (absolute) Euclidean distance.
pub type AampEngine = Engine<AbsoluteEuclidean>;
