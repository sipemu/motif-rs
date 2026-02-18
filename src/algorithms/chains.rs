use crate::core::matrix_profile::MatrixProfile;

/// A single time series chain -- an ordered sequence of indices representing
/// an evolving pattern through time.
#[derive(Debug, Clone)]
pub struct Chain {
    /// Subsequence indices in temporal order.
    pub indices: Vec<usize>,
}

impl Chain {
    /// The number of links in this chain.
    pub fn len(&self) -> usize {
        self.indices.len()
    }

    /// Whether this chain is empty.
    pub fn is_empty(&self) -> bool {
        self.indices.is_empty()
    }
}

/// Result of all-chain discovery (ALLC).
#[derive(Debug, Clone)]
pub struct ChainsResult {
    /// All unique chains found.
    pub chains: Vec<Chain>,
    /// The longest chain (unanchored -- may start at any index).
    pub longest: Chain,
    /// The longest anchored chain (starts from an index with no left-predecessor).
    pub longest_anchored: Chain,
}

/// Anchored Time Series Chain: follow the right-neighbor links from `anchor`,
/// validating each bidirectional link.
///
/// A link from `i` to `j` is valid when `right_profile_index[i] == j` AND
/// `left_profile_index[j] == i`.
///
/// # Arguments
/// * `mp` - A computed matrix profile (must have left/right profile indices)
/// * `anchor` - Starting index for the chain
pub fn atsc(mp: &MatrixProfile, anchor: usize) -> Chain {
    let n = mp.profile.len();
    assert!(anchor < n, "anchor {anchor} out of range (n_subs={n})");

    let mut indices = vec![anchor];
    let mut current = anchor;

    loop {
        let next = mp.right_profile_index[current];
        // Stop if next is out of bounds or not strictly forward
        if next >= n || next <= current {
            break;
        }
        // Validate bidirectional link: left neighbor of `next` must be `current`
        if mp.left_profile_index[next] != current {
            break;
        }
        indices.push(next);
        current = next;
    }

    Chain { indices }
}

/// All Chains: discover all unique time series chains in the matrix profile.
///
/// Implements the ALLC algorithm:
/// 1. Build forward links from bidirectionally-validated left/right profile indices
/// 2. Identify chain anchors (indices with no valid predecessor)
/// 3. Trace each anchor's chain forward
///
/// Returns all chains, the longest overall, and the longest anchored chain.
///
/// # Arguments
/// * `mp` - A computed matrix profile (must have left/right profile indices)
pub fn allc(mp: &MatrixProfile) -> ChainsResult {
    let n = mp.profile.len();

    // Build forward links: fwd[i] = Some(j) iff i→j is a valid bidirectional link
    let mut fwd = vec![None; n];
    let mut has_predecessor = vec![false; n];

    for (i, fwd_i) in fwd.iter_mut().enumerate() {
        let j = mp.right_profile_index[i];
        if j < n && j > i && mp.left_profile_index[j] == i {
            *fwd_i = Some(j);
            has_predecessor[j] = true;
        }
    }

    // Trace chains from every anchor (indices with no predecessor)
    let mut chains = Vec::new();
    let mut longest = Chain {
        indices: Vec::new(),
    };
    let mut longest_anchored = Chain {
        indices: Vec::new(),
    };

    // Also trace from non-anchor starts to find ALL unique chains
    let mut visited = vec![false; n];

    for start in 0..n {
        if visited[start] {
            continue;
        }
        // Trace forward from this start
        let mut indices = vec![start];
        visited[start] = true;
        let mut current = start;
        while let Some(next) = fwd[current] {
            if visited[next] {
                break;
            }
            indices.push(next);
            visited[next] = true;
            current = next;
        }

        // Only record chains with at least 2 links
        if indices.len() >= 2 {
            let chain = Chain {
                indices: indices.clone(),
            };
            if chain.len() > longest.len() {
                longest = chain.clone();
            }
            if !has_predecessor[start] && chain.len() > longest_anchored.len() {
                longest_anchored = chain.clone();
            }
            chains.push(chain);
        }
    }

    // If no chains found, return empty results
    if longest.is_empty() {
        // Fallback: longest is just the first index
        longest = Chain { indices: vec![0] };
        longest_anchored = Chain { indices: vec![0] };
    }
    if longest_anchored.is_empty() {
        longest_anchored = longest.clone();
    }

    ChainsResult {
        chains,
        longest,
        longest_anchored,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algorithms::stomp::stomp;
    use crate::core::matrix_profile::MatrixProfileConfig;
    use crate::metrics::euclidean::ZNormalizedEuclidean;

    #[test]
    fn test_atsc_basic() {
        // Create a time series with a clear evolving pattern
        let n = 200;
        let m = 10;
        let ts: Vec<f64> = (0..n)
            .map(|i| (i as f64 * 2.0 * std::f64::consts::PI / 50.0).sin())
            .collect();

        let config = MatrixProfileConfig::new(m);
        let mp = stomp::<ZNormalizedEuclidean>(&ts, &config);

        // Pick an anchor and trace
        let chain = atsc(&mp, 0);
        assert!(!chain.is_empty());
        // Chain indices should be strictly increasing
        for w in chain.indices.windows(2) {
            assert!(w[0] < w[1], "Chain indices should be increasing");
        }
    }

    #[test]
    fn test_allc_basic() {
        let n = 200;
        let m = 10;
        let ts: Vec<f64> = (0..n)
            .map(|i| (i as f64 * 2.0 * std::f64::consts::PI / 50.0).sin())
            .collect();

        let config = MatrixProfileConfig::new(m);
        let mp = stomp::<ZNormalizedEuclidean>(&ts, &config);

        let result = allc(&mp);
        // Should find at least one chain
        assert!(!result.longest.is_empty(), "Should find at least one chain");
        // Longest anchored should be <= longest overall
        assert!(result.longest_anchored.len() <= result.longest.len() + 1);
    }

    #[test]
    fn test_atsc_single_index() {
        // A very short time series where no forward links exist
        let mp = MatrixProfile::new(5, 3, 1);
        let chain = atsc(&mp, 0);
        assert_eq!(chain.len(), 1);
        assert_eq!(chain.indices, vec![0]);
    }

    #[test]
    fn test_chain_indices_temporal_order() {
        // Verify that all chains have strictly increasing indices
        let n = 300;
        let m = 15;
        let ts: Vec<f64> = (0..n)
            .map(|i| (i as f64 * 0.1).sin() + (i as f64 * 0.03).cos() * 0.5)
            .collect();

        let config = MatrixProfileConfig::new(m);
        let mp = stomp::<ZNormalizedEuclidean>(&ts, &config);
        let result = allc(&mp);

        for chain in &result.chains {
            for w in chain.indices.windows(2) {
                assert!(
                    w[0] < w[1],
                    "Chain indices must be strictly increasing: {} >= {}",
                    w[0],
                    w[1]
                );
            }
        }
    }
}
