use crate::core::distance_metric::DistanceMetric;
use crate::core::matrix_profile::RollingStats;

/// Z-normalized Euclidean distance metric.
///
/// Distance formula: `d = sqrt(2 * m * (1 - r))` where
/// `r = (QT - m * mu_i * mu_j) / (m * sigma_i * sigma_j)`.
///
/// Edge cases:
/// - Both subsequences constant (sigma_i == 0 && sigma_j == 0) → d = 0
/// - One subsequence constant → d = sqrt(2*m)
/// - `r` is clamped to [-1, 1] for numerical stability
#[derive(Debug, Clone)]
pub struct ZNormalizedEuclidean;

impl DistanceMetric for ZNormalizedEuclidean {
    type Context = RollingStats;

    fn precompute(ts: &[f64], m: usize) -> Self::Context {
        RollingStats::compute(ts, m)
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

    fn qt_to_distance(qt: f64, i: usize, j: usize, m: usize, ctx: &Self::Context) -> f64 {
        let msi = ctx.m_sigma_inv[i];
        let msj = ctx.m_sigma_inv[j];
        let m_f = m as f64;

        // Both constant → identical after z-normalization → distance 0
        if msi == 0.0 && msj == 0.0 {
            return 0.0;
        }
        // One constant → maximally different from any non-constant subsequence
        if msi == 0.0 || msj == 0.0 {
            return (2.0 * m_f).sqrt();
        }

        // r = (QT - m*mu_i*mu_j) * m_sigma_inv[i] * m_sigma_inv[j]
        // where m_sigma_inv = 1/(sqrt(m)*sigma), so product = 1/(m*sigma_i*sigma_j)
        let r = (qt - m_f * ctx.mean[i] * ctx.mean[j]) * msi * msj;
        let r_clamped = r.clamp(-1.0, 1.0);
        (2.0 * m_f * (1.0 - r_clamped)).max(0.0).sqrt()
    }

    fn supports_correlation_domain() -> bool {
        true
    }

    fn correlation_data(ctx: &Self::Context) -> (&[f64], &[f64], bool) {
        (&ctx.mean, &ctx.m_sigma_inv, ctx.has_constant)
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
        m: usize,
        ctx_a: &Self::Context,
        ctx_b: &Self::Context,
    ) -> f64 {
        let msi_a = ctx_a.m_sigma_inv[i];
        let msi_b = ctx_b.m_sigma_inv[j];
        let m_f = m as f64;

        if msi_a == 0.0 && msi_b == 0.0 {
            return 0.0;
        }
        if msi_a == 0.0 || msi_b == 0.0 {
            return (2.0 * m_f).sqrt();
        }

        let r = (qt - m_f * ctx_a.mean[i] * ctx_b.mean[j]) * msi_a * msi_b;
        let r_clamped = r.clamp(-1.0, 1.0);
        (2.0 * m_f * (1.0 - r_clamped)).max(0.0).sqrt()
    }

    fn correlation_data_ab<'a>(
        ctx_a: &'a Self::Context,
        ctx_b: &'a Self::Context,
    ) -> (&'a [f64], &'a [f64], &'a [f64], &'a [f64], bool) {
        (
            &ctx_a.mean,
            &ctx_a.m_sigma_inv,
            &ctx_b.mean,
            &ctx_b.m_sigma_inv,
            ctx_a.has_constant || ctx_b.has_constant,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_distance_identical_subsequences() {
        // Distance between a subsequence and itself should be 0
        let ts = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let m = 4;
        let ctx = ZNormalizedEuclidean::precompute(&ts, m);
        let d = ZNormalizedEuclidean::distance(&ts, 0, 0, m, &ctx);
        assert!(d.abs() < 1e-6, "Self-distance should be 0, got {d}");
    }

    #[test]
    fn test_distance_shifted_linear() {
        // [1,2,3,4] vs [3,4,5,6] — same shape (linear), just shifted → d ≈ 0
        let ts = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let m = 4;
        let ctx = ZNormalizedEuclidean::precompute(&ts, m);
        let d = ZNormalizedEuclidean::distance(&ts, 0, 2, m, &ctx);
        assert!(
            d < 1e-6,
            "Shifted linear sequences should have d≈0, got {d}"
        );
    }

    #[test]
    fn test_distance_constant_both() {
        let ts = vec![5.0; 10];
        let m = 4;
        let ctx = ZNormalizedEuclidean::precompute(&ts, m);
        let d = ZNormalizedEuclidean::distance(&ts, 0, 3, m, &ctx);
        assert!(d.abs() < 1e-10, "Two constant subsequences → d=0, got {d}");
    }

    #[test]
    fn test_distance_one_constant() {
        // [5,5,5,5] vs [1,2,3,4] — one constant
        let ts = vec![5.0, 5.0, 5.0, 5.0, 1.0, 2.0, 3.0, 4.0];
        let m = 4;
        let ctx = ZNormalizedEuclidean::precompute(&ts, m);
        let d = ZNormalizedEuclidean::distance(&ts, 0, 4, m, &ctx);
        let expected = (2.0 * 4.0_f64).sqrt();
        assert!(
            (d - expected).abs() < 1e-10,
            "One constant → d=sqrt(2m)={expected}, got {d}"
        );
    }

    #[test]
    fn test_qt_to_distance_hand_computed() {
        // Manual computation:
        // ts = [1, 2, 3, 4], m = 2
        // Subseqs: [1,2] (mu=1.5, std=0.5), [2,3] (mu=2.5, std=0.5), [3,4] (mu=3.5, std=0.5)
        // QT(0,1) = 1*2 + 2*3 = 8
        // r = (8 - 2*1.5*2.5) / (2*0.5*0.5) = (8-7.5)/0.5 = 1.0
        // d = sqrt(2*2*(1-1)) = 0
        let ts = vec![1.0, 2.0, 3.0, 4.0];
        let m = 2;
        let ctx = ZNormalizedEuclidean::precompute(&ts, m);
        let qt = 1.0 * 2.0 + 2.0 * 3.0; // = 8
        let d = ZNormalizedEuclidean::qt_to_distance(qt, 0, 1, m, &ctx);
        // Tolerance relaxed from 1e-10 to 1e-7: precomputed m_sigma_inv introduces
        // ~1 ULP error in r near 1.0, amplified by sqrt(1-r) to ~3e-8.
        assert!(d < 1e-7, "Hand-computed: d should be ~0, got {d}");
    }

    #[test]
    fn test_qt_to_distance_anticorrelated() {
        // ts = [1, 2, 4, 3], m = 2
        // Subseqs: [1,2] (mu=1.5, std=0.5), [2,4] (mu=3, std=1), [4,3] (mu=3.5, std=0.5)
        // [1,2] z-norm: [-1, 1], [4,3] z-norm: [1, -1] — perfectly anticorrelated
        // QT(0,2) = 1*4 + 2*3 = 10
        // r = (10 - 2*1.5*3.5) / (2*0.5*0.5) = (10 - 10.5) / 0.5 = -1.0
        // d = sqrt(2*2*(1-(-1))) = sqrt(8) = 2*sqrt(2)
        let ts = vec![1.0, 2.0, 4.0, 3.0];
        let m = 2;
        let ctx = ZNormalizedEuclidean::precompute(&ts, m);
        let qt = 1.0 * 4.0 + 2.0 * 3.0; // = 10
        let d = ZNormalizedEuclidean::qt_to_distance(qt, 0, 2, m, &ctx);
        let expected = (8.0_f64).sqrt(); // 2*sqrt(2)
        assert!(
            (d - expected).abs() < 1e-10,
            "Anticorrelated: expected {expected}, got {d}"
        );
    }
}
