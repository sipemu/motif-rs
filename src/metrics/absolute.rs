use crate::core::distance_metric::DistanceMetric;

/// Precomputed context for non-normalized (absolute) Euclidean distance.
///
/// Stores running sum-of-squares for each subsequence, computed via cumulative
/// sums in O(n). The distance formula is:
/// `d(i,j) = sqrt(sum_sq[i] + sum_sq[j] - 2*QT_ij)`
#[derive(Debug, Clone)]
pub struct AampContext {
    /// `sum_sq[i]` = sum of squares of `ts[i..i+m]`.
    pub sum_sq: Vec<f64>,
}

impl AampContext {
    /// Compute running sum-of-squares via cumulative sum.
    pub fn compute(ts: &[f64], m: usize) -> Self {
        assert!(m > 0, "Subsequence length must be > 0");
        assert!(ts.len() >= m, "Time series must be at least as long as m");

        let n = ts.len();
        let n_subs = n - m + 1;

        // Cumulative sum of squares
        let mut cumsum_sq = vec![0.0; n + 1];
        for i in 0..n {
            cumsum_sq[i + 1] = cumsum_sq[i] + ts[i] * ts[i];
        }

        let sum_sq: Vec<f64> = (0..n_subs)
            .map(|i| cumsum_sq[i + m] - cumsum_sq[i])
            .collect();

        Self { sum_sq }
    }

    /// Extend context by one new subsequence after appending a point.
    pub fn extend(&mut self, ts: &[f64], m: usize) {
        let n = ts.len();
        assert!(n >= m);
        let start = n - m;
        let sum_sq: f64 = ts[start..n].iter().map(|x| x * x).sum();
        self.sum_sq.push(sum_sq);
    }
}

/// Non-normalized (absolute) Euclidean distance metric (AAMP).
///
/// Unlike `ZNormalizedEuclidean`, this metric does NOT z-normalize subsequences.
/// It computes raw Euclidean distances, making it suitable for time series where
/// amplitude matters (e.g., sensor data with meaningful absolute values).
///
/// Distance formula: `d(i,j) = sqrt(sum_sq[i] + sum_sq[j] - 2*QT_ij)`
///
/// The QT recurrence is identical to z-normalized STOMP (just dot products),
/// so we get O(1) updates per element via the QT-diagonal path.
#[derive(Debug, Clone)]
pub struct AbsoluteEuclidean;

impl DistanceMetric for AbsoluteEuclidean {
    type Context = AampContext;

    fn precompute(ts: &[f64], m: usize) -> Self::Context {
        AampContext::compute(ts, m)
    }

    fn distance(ts: &[f64], i: usize, j: usize, m: usize, ctx: &Self::Context) -> f64 {
        let qt: f64 = ts[i..i + m]
            .iter()
            .zip(&ts[j..j + m])
            .map(|(a, b)| a * b)
            .sum();
        Self::qt_to_distance(qt, i, j, m, ctx)
    }

    fn supports_qt_optimization() -> bool {
        true
    }

    fn qt_to_distance(qt: f64, i: usize, j: usize, _m: usize, ctx: &Self::Context) -> f64 {
        // d = sqrt(sum_sq[i] + sum_sq[j] - 2*QT)
        // Clamp to 0 for numerical stability (tiny negative values from rounding)
        (ctx.sum_sq[i] + ctx.sum_sq[j] - 2.0 * qt).max(0.0).sqrt()
    }

    fn supports_correlation_domain() -> bool {
        false
    }

    fn update_context(ctx: &mut Self::Context, ts: &[f64], m: usize) {
        ctx.extend(ts, m);
    }

    fn supports_ab_join() -> bool {
        true
    }

    fn qt_to_distance_ab(
        qt: f64,
        i: usize,
        j: usize,
        _m: usize,
        ctx_a: &Self::Context,
        ctx_b: &Self::Context,
    ) -> f64 {
        (ctx_a.sum_sq[i] + ctx_b.sum_sq[j] - 2.0 * qt)
            .max(0.0)
            .sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_aamp_context_compute() {
        // ts = [1, 2, 3, 4], m = 2
        // sum_sq[0] = 1^2 + 2^2 = 5
        // sum_sq[1] = 2^2 + 3^2 = 13
        // sum_sq[2] = 3^2 + 4^2 = 25
        let ts = vec![1.0, 2.0, 3.0, 4.0];
        let ctx = AampContext::compute(&ts, 2);
        assert_eq!(ctx.sum_sq.len(), 3);
        assert!((ctx.sum_sq[0] - 5.0).abs() < 1e-10);
        assert!((ctx.sum_sq[1] - 13.0).abs() < 1e-10);
        assert!((ctx.sum_sq[2] - 25.0).abs() < 1e-10);
    }

    #[test]
    fn test_aamp_distance_identical() {
        let ts = vec![1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0];
        let m = 3;
        let ctx = AbsoluteEuclidean::precompute(&ts, m);
        // [1,2,3] at index 0 and [1,2,3] at index 5 — identical
        let d = AbsoluteEuclidean::distance(&ts, 0, 5, m, &ctx);
        assert!(d.abs() < 1e-10, "Identical subsequences → d=0, got {d}");
    }

    #[test]
    fn test_aamp_distance_hand_computed() {
        // ts = [1, 2, 3, 4], m = 2
        // d(0,1) = ||[1,2]-[2,3]|| = sqrt(1+1) = sqrt(2)
        let ts = vec![1.0, 2.0, 3.0, 4.0];
        let m = 2;
        let ctx = AbsoluteEuclidean::precompute(&ts, m);
        let d = AbsoluteEuclidean::distance(&ts, 0, 1, m, &ctx);
        let expected = 2.0_f64.sqrt();
        assert!((d - expected).abs() < 1e-10, "Expected {expected}, got {d}");
    }

    #[test]
    fn test_aamp_qt_to_distance() {
        // ts = [1, 2, 3, 4], m = 2
        // QT(0,2) = 1*3 + 2*4 = 11
        // sum_sq[0] = 5, sum_sq[2] = 25
        // d = sqrt(5 + 25 - 22) = sqrt(8) = 2*sqrt(2)
        let ts = vec![1.0, 2.0, 3.0, 4.0];
        let m = 2;
        let ctx = AbsoluteEuclidean::precompute(&ts, m);
        let qt = 1.0 * 3.0 + 2.0 * 4.0; // 11
        let d = AbsoluteEuclidean::qt_to_distance(qt, 0, 2, m, &ctx);
        let expected = 8.0_f64.sqrt();
        assert!((d - expected).abs() < 1e-10, "Expected {expected}, got {d}");
    }

    #[test]
    fn test_aamp_context_extend() {
        let mut ts = vec![1.0, 2.0, 3.0, 4.0];
        let m = 2;
        let mut ctx = AampContext::compute(&ts, m);
        assert_eq!(ctx.sum_sq.len(), 3);

        ts.push(5.0);
        ctx.extend(&ts, m);
        assert_eq!(ctx.sum_sq.len(), 4);
        // sum_sq[3] = 4^2 + 5^2 = 41
        assert!((ctx.sum_sq[3] - 41.0).abs() < 1e-10);
    }

    #[test]
    fn test_aamp_stomp_integration() {
        use crate::algorithms::stomp::stomp;
        use crate::core::matrix_profile::MatrixProfileConfig;

        // Repeating pattern — should find near-zero distances
        let ts = vec![1.0, 2.0, 3.0, 2.0, 1.0, 2.0, 3.0, 2.0, 1.0];
        let config = MatrixProfileConfig::new(4);
        let mp = stomp::<AbsoluteEuclidean>(&ts, &config);

        assert_eq!(mp.profile.len(), ts.len() - 4 + 1);
        // [1,2,3,2] at index 0 and 4 are identical
        assert!(
            mp.profile[0] < 1e-6,
            "Identical subsequence distance should be ~0, got {}",
            mp.profile[0]
        );
    }
}
