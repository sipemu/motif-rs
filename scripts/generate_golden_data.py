#!/usr/bin/env python3
"""Generate golden reference data using stumpy for validation of motif-rs.

Usage:
    pip install stumpy numpy
    python scripts/generate_golden_data.py

Generates JSON files in tests/golden_data/ containing:
- Time series data
- Matrix profile (P) and matrix profile index (I)
- Left/right profiles and indices
- Parameters (n, m)
"""

import json
import os

import numpy as np
import stumpy


def compute_left_profile(ts, m, left_indices):
    """Compute left profile distances from left neighbor indices.

    stumpy.stump returns left/right neighbor indices, not distances.
    We compute the actual z-normalized Euclidean distances here.
    """
    n_subs = len(left_indices)
    left_profile = np.full(n_subs, np.inf)

    for i in range(n_subs):
        j = int(left_indices[i])
        if j < 0:
            continue  # No left neighbor
        # Compute z-normalized Euclidean distance
        sub_i = ts[i : i + m]
        sub_j = ts[j : j + m]
        std_i = np.std(sub_i)
        std_j = np.std(sub_j)
        if std_i < 1e-15 and std_j < 1e-15:
            left_profile[i] = 0.0
        elif std_i < 1e-15 or std_j < 1e-15:
            left_profile[i] = np.sqrt(2.0 * m)
        else:
            zi = (sub_i - np.mean(sub_i)) / std_i
            zj = (sub_j - np.mean(sub_j)) / std_j
            left_profile[i] = np.sqrt(np.sum((zi - zj) ** 2))

    return left_profile


def save_golden(filename: str, data: dict) -> None:
    """Save golden data as JSON with high precision."""
    os.makedirs("tests/golden_data", exist_ok=True)
    filepath = os.path.join("tests/golden_data", filename)

    def sanitize(val):
        """Replace inf/nan with JSON-safe sentinel values."""
        if isinstance(val, float):
            if np.isinf(val):
                return 1e308  # Large but finite sentinel for infinity
            if np.isnan(val):
                return None
        return val

    # Convert numpy arrays to lists with full precision, sanitize all lists
    serializable = {}
    for key, value in data.items():
        if isinstance(value, np.ndarray):
            serializable[key] = [sanitize(v) for v in value.tolist()]
        elif isinstance(value, list):
            serializable[key] = [sanitize(v) for v in value]
        else:
            serializable[key] = value

    with open(filepath, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"  Saved {filepath} ({len(json.dumps(serializable))} bytes)")


def generate_sine_wave():
    """Sine wave with embedded motif pair."""
    print("Generating sine_wave_mp.json...")
    n = 10000
    m = 100
    np.random.seed(42)

    t = np.linspace(0, 20 * np.pi, n)
    ts = np.sin(t) + 0.1 * np.random.randn(n)

    result = stumpy.stump(ts, m)
    left_indices = result[:, 2].astype(int)
    left_profile = compute_left_profile(ts, m, left_indices)

    save_golden("sine_wave_mp.json", {
        "description": "Sine wave with noise, n=10000, m=100",
        "ts": ts,
        "m": m,
        "n": n,
        "profile": result[:, 0].astype(float),
        "profile_index": result[:, 1].astype(int),
        "left_profile": left_profile,
        "left_profile_index": left_indices,
    })


def generate_square_wave():
    """Square wave signal."""
    print("Generating square_wave_mp.json...")
    n = 10000
    m = 100
    np.random.seed(123)

    t = np.linspace(0, 40 * np.pi, n)
    ts = np.sign(np.sin(t)) + 0.05 * np.random.randn(n)

    result = stumpy.stump(ts, m)
    left_indices = result[:, 2].astype(int)
    left_profile = compute_left_profile(ts, m, left_indices)

    save_golden("square_wave_mp.json", {
        "description": "Square wave with noise, n=10000, m=100",
        "ts": ts,
        "m": m,
        "n": n,
        "profile": result[:, 0].astype(float),
        "profile_index": result[:, 1].astype(int),
        "left_profile": left_profile,
        "left_profile_index": left_indices,
    })


def generate_mixed_signal():
    """Mixed frequency signal."""
    print("Generating mixed_signal_mp.json...")
    n = 10000
    m = 100
    np.random.seed(456)

    t = np.linspace(0, 10 * np.pi, n)
    ts = np.sin(t) + 0.5 * np.sin(3 * t) + 0.3 * np.cos(7 * t) + 0.1 * np.random.randn(n)

    result = stumpy.stump(ts, m)
    left_indices = result[:, 2].astype(int)
    left_profile = compute_left_profile(ts, m, left_indices)

    save_golden("mixed_signal_mp.json", {
        "description": "Mixed frequency signal with noise, n=10000, m=100",
        "ts": ts,
        "m": m,
        "n": n,
        "profile": result[:, 0].astype(float),
        "profile_index": result[:, 1].astype(int),
        "left_profile": left_profile,
        "left_profile_index": left_indices,
    })


def generate_streaming_sine():
    """Streaming (STAMPI) reference: build incrementally and capture final state."""
    print("Generating streaming_sine_mp.json...")
    m = 50
    np.random.seed(789)

    # Initial series
    n_initial = 200
    t_init = np.linspace(0, 4 * np.pi, n_initial)
    ts_initial = np.sin(t_init) + 0.05 * np.random.randn(n_initial)

    # Additional points to stream in
    n_stream = 100
    t_stream = np.linspace(4 * np.pi, 6 * np.pi, n_stream + 1)[1:]  # exclude overlap
    ts_stream = np.sin(t_stream) + 0.05 * np.random.randn(n_stream)

    # Build streaming profile using stumpy's stumpi
    stream = stumpy.stumpi(ts_initial, m, egress=False)
    for val in ts_stream:
        stream.update(val)

    # Full series for batch comparison
    ts_full = np.concatenate([ts_initial, ts_stream])
    batch_result = stumpy.stump(ts_full, m)

    save_golden("streaming_sine_mp.json", {
        "description": "Streaming sine wave, n_initial=200, n_stream=100, m=50",
        "ts_initial": ts_initial,
        "ts_stream": ts_stream,
        "ts_full": ts_full,
        "m": m,
        "n_initial": n_initial,
        "n_stream": n_stream,
        # Streaming final state
        "streaming_profile": stream.P_.tolist(),
        "streaming_profile_index": stream.I_.tolist(),
        "streaming_left_profile": stream.left_P_.tolist(),
        "streaming_left_profile_index": stream.left_I_.tolist(),
        # Batch reference on full series
        "batch_profile": batch_result[:, 0].astype(float),
        "batch_profile_index": batch_result[:, 1].astype(int),
    })


if __name__ == "__main__":
    print("Generating golden reference data using stumpy...\n")
    generate_sine_wave()
    generate_square_wave()
    generate_mixed_signal()
    generate_streaming_sine()
    print("\nDone! Run 'cargo test' to validate against these references.")
