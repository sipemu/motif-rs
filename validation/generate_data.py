#!/usr/bin/env python3
"""Generate test time series for validation.

Creates deterministic test signals of varying complexity for comparing
motif-rs against the stumpy reference implementation.
"""

import json
import os

import numpy as np

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def save_series(name: str, ts: np.ndarray, m: int, **extra):
    """Save a time series to JSON."""
    os.makedirs(DATA_DIR, exist_ok=True)
    data = {"name": name, "ts": ts.tolist(), "m": m, "n": len(ts), **extra}
    path = os.path.join(DATA_DIR, f"{name}.json")
    with open(path, "w") as f:
        json.dump(data, f)
    print(f"  Generated {path} (n={len(ts)}, m={m})")


def generate_all():
    print("Generating validation test data...\n")

    # Batch test signals (n=10000, m=100)
    n, m = 10000, 100

    np.random.seed(42)
    t = np.linspace(0, 20 * np.pi, n)
    save_series("sine_wave", np.sin(t) + 0.1 * np.random.randn(n), m, signal_type="sine")

    np.random.seed(123)
    t = np.linspace(0, 40 * np.pi, n)
    save_series(
        "square_wave",
        np.sign(np.sin(t)) + 0.05 * np.random.randn(n),
        m,
        signal_type="square",
    )

    np.random.seed(456)
    t = np.linspace(0, 10 * np.pi, n)
    save_series(
        "mixed_signal",
        np.sin(t) + 0.5 * np.sin(3 * t) + 0.3 * np.cos(7 * t) + 0.1 * np.random.randn(n),
        m,
        signal_type="mixed",
    )

    # Streaming test signal (n_initial=200, n_stream=100, m=50)
    m_stream = 50
    np.random.seed(789)
    n_initial, n_stream = 200, 100
    t_init = np.linspace(0, 4 * np.pi, n_initial)
    ts_initial = np.sin(t_init) + 0.05 * np.random.randn(n_initial)
    t_stream = np.linspace(4 * np.pi, 6 * np.pi, n_stream + 1)[1:]
    ts_stream = np.sin(t_stream) + 0.05 * np.random.randn(n_stream)

    save_series(
        "streaming_sine",
        np.concatenate([ts_initial, ts_stream]),
        m_stream,
        signal_type="streaming",
        n_initial=n_initial,
        n_stream=n_stream,
        ts_initial=ts_initial.tolist(),
        ts_stream=ts_stream.tolist(),
    )

    print("\nDone.")


if __name__ == "__main__":
    generate_all()
