#!/usr/bin/env python3
"""Generate golden reference data for new motif-rs features using stumpy.

Usage:
    pip install stumpy numpy
    python scripts/generate_golden_features.py

Generates JSON files in tests/golden_data/ for:
- AAMP (non-normalized Euclidean distance)
- AB-Join (cross-series comparison)
- Top-k nearest neighbors
- Motif discovery
- Discord (anomaly) detection
- FLUSS segmentation
"""

import json
import os

import numpy as np
import stumpy


def save_golden(filename: str, data: dict) -> None:
    """Save golden data as JSON with high precision."""
    os.makedirs("tests/golden_data", exist_ok=True)
    filepath = os.path.join("tests/golden_data", filename)

    def sanitize(val):
        """Replace inf/nan with JSON-safe sentinel values."""
        if isinstance(val, (float, np.floating)):
            if np.isinf(val):
                return 1e308
            if np.isnan(val):
                return None
        if isinstance(val, (np.integer,)):
            return int(val)
        return val

    serializable = {}
    for key, value in data.items():
        if isinstance(value, np.ndarray):
            flat = value.tolist()
            if isinstance(flat, list) and len(flat) > 0 and isinstance(flat[0], list):
                # 2D array — sanitize each inner list
                serializable[key] = [[sanitize(v) for v in row] for row in flat]
            else:
                serializable[key] = [sanitize(v) for v in flat]
        elif isinstance(value, list):
            if len(value) > 0 and isinstance(value[0], list):
                serializable[key] = [[sanitize(v) for v in row] for row in value]
            else:
                serializable[key] = [sanitize(v) for v in value]
        else:
            serializable[key] = sanitize(value)

    with open(filepath, "w") as f:
        json.dump(serializable, f)
    size_kb = os.path.getsize(filepath) / 1024
    print(f"  Saved {filepath} ({size_kb:.0f} KB)")


def generate_aamp():
    """Non-normalized (absolute) Euclidean distance — AAMP."""
    print("Generating aamp_sine_wave.json...")
    n = 1000
    m = 50
    np.random.seed(42)

    t = np.linspace(0, 8 * np.pi, n)
    ts = np.sin(t) + 0.1 * np.random.randn(n)

    # stumpy with normalize=False gives non-normalized Euclidean distance
    result = stumpy.stump(ts, m, normalize=False)

    save_golden("aamp_sine_wave.json", {
        "description": "AAMP on sine wave with noise, n=1000, m=50",
        "ts": ts,
        "m": m,
        "n": n,
        "profile": result[:, 0].astype(float),
        "profile_index": result[:, 1].astype(int),
    })


def generate_ab_join():
    """AB-join between two different time series."""
    print("Generating ab_join_sine_square.json...")
    n_a = 500
    n_b = 600
    m = 30
    np.random.seed(100)

    t_a = np.linspace(0, 6 * np.pi, n_a)
    ts_a = np.sin(t_a) + 0.1 * np.random.randn(n_a)

    t_b = np.linspace(0, 8 * np.pi, n_b)
    ts_b = np.sin(t_b) + 0.5 * np.cos(3 * t_b) + 0.1 * np.random.randn(n_b)

    # AB-join: compare T_A against T_B
    result = stumpy.stump(ts_a, m, ts_b, ignore_trivial=False)

    # result columns: [distance, index_in_B, left_idx, right_idx]
    # This gives the profile for T_A (nearest neighbor in T_B)
    profile_a = result[:, 0].astype(float)
    index_a = result[:, 1].astype(int)

    # For the B profile, we need to do the reverse
    result_b = stumpy.stump(ts_b, m, ts_a, ignore_trivial=False)
    profile_b = result_b[:, 0].astype(float)
    index_b = result_b[:, 1].astype(int)

    save_golden("ab_join_sine_square.json", {
        "description": "AB-join: sine vs mixed signal, n_a=500, n_b=600, m=30",
        "ts_a": ts_a,
        "ts_b": ts_b,
        "m": m,
        "n_a": n_a,
        "n_b": n_b,
        "profile_a": profile_a,
        "index_a": index_a,
        "profile_b": profile_b,
        "index_b": index_b,
    })


def generate_topk():
    """Top-k nearest neighbors."""
    print("Generating topk_sine_wave.json...")
    n = 1000
    m = 50
    k = 3
    np.random.seed(42)

    t = np.linspace(0, 8 * np.pi, n)
    ts = np.sin(t) + 0.1 * np.random.randn(n)

    result = stumpy.stump(ts, m, k=k)
    # With k>1, result shape is (n_subs, 2*k+2) or similar
    # Columns: [P1..Pk, I1..Ik, left_I, right_I]
    # Actually for k>1: shape = (n_subs, 2*k + 2)
    # First k columns: distances, next k: indices, then left_I, right_I

    n_subs = n - m + 1

    # Extract top-k distances and indices
    distances = result[:, :k].astype(float).tolist()
    indices = result[:, k:2*k].astype(int).tolist()

    save_golden("topk_sine_wave.json", {
        "description": f"Top-{k} on sine wave with noise, n={n}, m={m}",
        "ts": ts,
        "m": m,
        "n": n,
        "k": k,
        "distances": distances,
        "indices": indices,
    })


def generate_motifs():
    """Motif discovery using stumpy.motifs()."""
    print("Generating motifs_synthetic.json...")
    n = 1000
    m = 50
    np.random.seed(42)

    t = np.linspace(0, 8 * np.pi, n)
    ts = np.sin(t) + 0.1 * np.random.randn(n)

    result = stumpy.stump(ts, m)
    mp = result[:, 0].astype(float)

    # Find top-3 motifs
    motif_distances, motif_indices = stumpy.motifs(
        ts, mp, max_motifs=3, max_matches=2, min_neighbors=1
    )

    # motif_distances: shape (max_motifs, max_matches) or ragged
    # motif_indices: shape (max_motifs, max_matches) or ragged
    # Convert to flat list of (idx_a, idx_b, distance) per motif
    motifs_list = []
    for i in range(len(motif_indices)):
        idxs = motif_indices[i]
        dists = motif_distances[i]
        valid = [j for j in range(len(idxs)) if idxs[j] >= 0 and not np.isnan(idxs[j])]
        if len(valid) >= 2:
            motifs_list.append({
                "idx_a": int(idxs[0]),
                "idx_b": int(idxs[1]),
                "distance": float(dists[1]) if len(dists) > 1 else float(dists[0]),
            })

    save_golden("motifs_synthetic.json", {
        "description": f"Motifs on sine wave with noise, n={n}, m={m}",
        "ts": ts,
        "m": m,
        "n": n,
        "profile": mp,
        "profile_index": result[:, 1].astype(int),
        "motifs": motifs_list,
    })


def generate_discords():
    """Discord (anomaly) detection."""
    print("Generating discords_synthetic.json...")
    n = 1000
    m = 50
    np.random.seed(42)

    t = np.linspace(0, 8 * np.pi, n)
    ts = np.sin(t) + 0.1 * np.random.randn(n)

    # Inject anomalies
    ts[250:260] = 5.0  # spike
    ts[600:610] = -5.0  # negative spike

    result = stumpy.stump(ts, m)
    mp = result[:, 0].astype(float)
    mp_idx = result[:, 1].astype(int)

    # Discords are the positions with highest matrix profile values
    # Get top-3 discords using greedy exclusion zone extraction
    ez = int(np.ceil(m / 4))
    working_mp = mp.copy()
    discords = []
    for _ in range(3):
        finite_mask = np.isfinite(working_mp)
        if not finite_mask.any():
            break
        idx = np.argmax(np.where(finite_mask, working_mp, -np.inf))
        discords.append({
            "idx": int(idx),
            "distance": float(working_mp[idx]),
        })
        # Apply exclusion zone
        start = max(0, idx - ez)
        end = min(len(working_mp), idx + ez + 1)
        working_mp[start:end] = -np.inf

    save_golden("discords_synthetic.json", {
        "description": f"Discords on sine+anomaly, n={n}, m={m}",
        "ts": ts,
        "m": m,
        "n": n,
        "profile": mp,
        "profile_index": mp_idx,
        "discords": discords,
    })


def generate_fluss():
    """FLUSS segmentation."""
    print("Generating fluss_regime_change.json...")
    n = 1000
    m = 20
    np.random.seed(42)

    # Two regimes: sine wave (0..500), sawtooth (500..1000)
    ts = np.zeros(n)
    for i in range(500):
        ts[i] = np.sin(i * 2 * np.pi / 50) + 0.05 * np.random.randn()
    for i in range(500, 1000):
        phase = (i - 500) % 30
        ts[i] = (phase / 30.0) * 2 - 1 + 0.05 * np.random.randn()

    result = stumpy.stump(ts, m)
    mp_idx = result[:, 1].astype(int)

    # FLUSS: I = profile index, L = subsequence length for CAC, n_regimes
    L = m  # Window length for ideal arc curve
    cac, regime_locs = stumpy.fluss(mp_idx, L, n_regimes=1, excl_factor=5)

    save_golden("fluss_regime_change.json", {
        "description": f"FLUSS on sine→sawtooth regime change at 500, n={n}, m={m}",
        "ts": ts,
        "m": m,
        "n": n,
        "L": int(L),
        "profile_index": mp_idx,
        "cac": cac.astype(float),
        "regime_locs": [int(x) for x in regime_locs],
    })


if __name__ == "__main__":
    print("Generating golden reference data for new features using stumpy...\n")
    generate_aamp()
    generate_ab_join()
    generate_topk()
    generate_motifs()
    generate_discords()
    generate_fluss()
    print("\nDone! Run 'cargo test' to validate against these references.")
