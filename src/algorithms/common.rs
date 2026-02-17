use realfft::RealFftPlanner;

/// Size threshold (n * m) above which we dispatch to the FFT path.
/// Below this, the naive O(n*m) loop wins due to lower constant overhead.
const FFT_THRESHOLD: usize = 256 * 1024;

/// Compute the sliding dot product between a query subsequence `q` and time series `ts`.
///
/// Returns a vector of length `ts.len() - q.len() + 1` where element `i` is
/// `dot(q, ts[i..i+m])`.
///
/// Adaptively dispatches to an FFT-based O(n log n) implementation for large
/// inputs, falling back to the naive O(n*m) loop for small inputs.
pub fn sliding_dot_product(q: &[f64], ts: &[f64]) -> Vec<f64> {
    let m = q.len();
    let n = ts.len();
    assert!(n >= m, "Time series shorter than query");
    if n * m > FFT_THRESHOLD {
        sliding_dot_product_fft(q, ts)
    } else {
        sliding_dot_product_naive(q, ts)
    }
}

/// Naive O(n*m) sliding dot product.
pub fn sliding_dot_product_naive(q: &[f64], ts: &[f64]) -> Vec<f64> {
    let m = q.len();
    assert!(ts.len() >= m, "Time series shorter than query");
    let n_subs = ts.len() - m + 1;

    (0..n_subs)
        .map(|i| q.iter().zip(&ts[i..i + m]).map(|(a, b)| a * b).sum())
        .collect()
}

/// FFT-based O(n log n) sliding dot product via cross-correlation.
///
/// Uses real-to-complex FFT to compute the convolution of the reversed query
/// with the time series, then extracts the dot-product values.
pub fn sliding_dot_product_fft(q: &[f64], ts: &[f64]) -> Vec<f64> {
    let m = q.len();
    let n = ts.len();
    assert!(n >= m, "Time series shorter than query");
    let n_subs = n - m + 1;
    let conv_len = n + m - 1;
    let fft_len = conv_len.next_power_of_two();

    let mut planner = RealFftPlanner::<f64>::new();
    let fft_forward = planner.plan_fft_forward(fft_len);
    let fft_inverse = planner.plan_fft_inverse(fft_len);

    // Reverse query into zero-padded buffer
    let mut q_padded = vec![0.0; fft_len];
    for i in 0..m {
        q_padded[i] = q[m - 1 - i];
    }

    // Zero-pad time series
    let mut ts_padded = vec![0.0; fft_len];
    ts_padded[..n].copy_from_slice(ts);

    // Forward FFT both
    let mut q_spectrum = fft_forward.make_output_vec();
    let mut ts_spectrum = fft_forward.make_output_vec();
    fft_forward.process(&mut q_padded, &mut q_spectrum).unwrap();
    fft_forward.process(&mut ts_padded, &mut ts_spectrum).unwrap();

    // Element-wise complex multiply
    for (q_val, ts_val) in q_spectrum.iter_mut().zip(ts_spectrum.iter()) {
        *q_val *= ts_val;
    }

    // Inverse FFT
    let mut result = vec![0.0; fft_len];
    fft_inverse.process(&mut q_spectrum, &mut result).unwrap();

    // realfft inverse is unnormalized — divide by fft_len
    let norm = 1.0 / fft_len as f64;

    // Extract dot products: convolution result at indices [m-1 .. m-1+n_subs]
    result[m - 1..m - 1 + n_subs]
        .iter()
        .map(|&x| x * norm)
        .collect()
}

/// Apply an exclusion zone around index `idx`, setting entries within the zone to infinity.
///
/// The zone covers indices `[idx - zone, idx + zone]` (clamped to bounds).
#[inline]
pub fn apply_exclusion_zone(profile: &mut [f64], idx: usize, zone: usize) {
    let start = idx.saturating_sub(zone);
    let end = (idx + zone + 1).min(profile.len());
    for val in &mut profile[start..end] {
        *val = f64::INFINITY;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sliding_dot_product_simple() {
        // q = [1, 2], ts = [1, 2, 3, 4]
        // dot([1,2], [1,2]) = 5
        // dot([1,2], [2,3]) = 8
        // dot([1,2], [3,4]) = 11
        let q = vec![1.0, 2.0];
        let ts = vec![1.0, 2.0, 3.0, 4.0];
        let result = sliding_dot_product(&q, &ts);
        assert_eq!(result.len(), 3);
        assert!((result[0] - 5.0).abs() < 1e-10);
        assert!((result[1] - 8.0).abs() < 1e-10);
        assert!((result[2] - 11.0).abs() < 1e-10);
    }

    #[test]
    fn test_sliding_dot_product_single() {
        let q = vec![3.0, 4.0, 5.0];
        let ts = vec![3.0, 4.0, 5.0];
        let result = sliding_dot_product(&q, &ts);
        assert_eq!(result.len(), 1);
        assert!((result[0] - 50.0).abs() < 1e-10); // 9 + 16 + 25
    }

    #[test]
    fn test_fft_vs_naive_equivalence() {
        for (n, m) in [(100, 10), (1000, 50), (5000, 100)] {
            let ts: Vec<f64> = (0..n).map(|i| (i as f64 * 0.1).sin()).collect();
            let q = &ts[0..m];
            let naive = sliding_dot_product_naive(q, &ts);
            let fft = sliding_dot_product_fft(q, &ts);
            assert_eq!(naive.len(), fft.len());
            for (i, (a, b)) in naive.iter().zip(fft.iter()).enumerate() {
                assert!(
                    (a - b).abs() < 1e-6,
                    "Mismatch at {i} (n={n}, m={m}): naive={a}, fft={b}"
                );
            }
        }
    }

    #[test]
    fn test_fft_simple() {
        // Force FFT path on the same simple case
        let q = vec![1.0, 2.0];
        let ts = vec![1.0, 2.0, 3.0, 4.0];
        let result = sliding_dot_product_fft(&q, &ts);
        assert_eq!(result.len(), 3);
        assert!((result[0] - 5.0).abs() < 1e-10);
        assert!((result[1] - 8.0).abs() < 1e-10);
        assert!((result[2] - 11.0).abs() < 1e-10);
    }

    #[test]
    fn test_exclusion_zone_middle() {
        let mut profile = vec![1.0; 10];
        apply_exclusion_zone(&mut profile, 5, 2);
        // Indices 3,4,5,6,7 should be inf
        for i in 0..10 {
            if (3..=7).contains(&i) {
                assert!(profile[i].is_infinite());
            } else {
                assert!((profile[i] - 1.0).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_exclusion_zone_edge() {
        let mut profile = vec![1.0; 5];
        apply_exclusion_zone(&mut profile, 0, 2);
        // Indices 0,1,2 should be inf
        assert!(profile[0].is_infinite());
        assert!(profile[1].is_infinite());
        assert!(profile[2].is_infinite());
        assert!((profile[3] - 1.0).abs() < 1e-10);
        assert!((profile[4] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_exclusion_zone_zero() {
        let mut profile = vec![1.0; 5];
        apply_exclusion_zone(&mut profile, 2, 0);
        // Only index 2 should be inf
        assert!((profile[0] - 1.0).abs() < 1e-10);
        assert!((profile[1] - 1.0).abs() < 1e-10);
        assert!(profile[2].is_infinite());
        assert!((profile[3] - 1.0).abs() < 1e-10);
        assert!((profile[4] - 1.0).abs() < 1e-10);
    }
}
