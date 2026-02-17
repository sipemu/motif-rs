pub mod algorithms;
pub mod core;
pub mod metrics;

pub use crate::core::distance_metric::DistanceMetric;
pub use crate::core::matrix_profile::{MatrixProfile, MatrixProfileConfig, RollingStats};
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
}

/// Convenience type alias for the most common use case.
pub type EuclideanEngine = Engine<ZNormalizedEuclidean>;
