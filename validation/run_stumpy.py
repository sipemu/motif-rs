#!/usr/bin/env python3
"""Run stumpy reference implementation on validation data.

Loads test time series from data/ and computes matrix profiles using stumpy,
saving results to results/stumpy/.
"""

import json
import os
import time

import numpy as np
import stumpy

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results", "stumpy")


def sanitize(val):
    """Replace inf/nan with JSON-safe values."""
    if isinstance(val, float):
        if np.isinf(val):
            return 1e308
        if np.isnan(val):
            return None
    return val


def run_batch(name: str):
    """Run stumpy batch STOMP on a test signal."""
    path = os.path.join(DATA_DIR, f"{name}.json")
    with open(path) as f:
        data = json.load(f)

    ts = np.array(data["ts"])
    m = data["m"]

    start = time.perf_counter()
    result = stumpy.stump(ts, m)
    elapsed = time.perf_counter() - start

    profile = result[:, 0].astype(float)
    profile_index = result[:, 1].astype(int)

    out = {
        "name": name,
        "algorithm": "stumpy.stump",
        "version": stumpy.__version__,
        "elapsed_s": elapsed,
        "m": m,
        "n": len(ts),
        "profile": [sanitize(v) for v in profile.tolist()],
        "profile_index": profile_index.tolist(),
    }

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, f"{name}.json")
    with open(out_path, "w") as f:
        json.dump(out, f)
    print(f"  {name}: stumpy.stump in {elapsed:.3f}s (n={len(ts)}, m={m})")


def run_streaming(name: str):
    """Run stumpy streaming STAMPI on a test signal."""
    path = os.path.join(DATA_DIR, f"{name}.json")
    with open(path) as f:
        data = json.load(f)

    ts_initial = np.array(data["ts_initial"])
    ts_stream = np.array(data["ts_stream"])
    m = data["m"]

    start = time.perf_counter()
    stream = stumpy.stumpi(ts_initial, m, egress=False)
    for val in ts_stream:
        stream.update(val)
    elapsed = time.perf_counter() - start

    out = {
        "name": name,
        "algorithm": "stumpy.stumpi",
        "version": stumpy.__version__,
        "elapsed_s": elapsed,
        "m": m,
        "n_initial": len(ts_initial),
        "n_stream": len(ts_stream),
        "profile": [sanitize(v) for v in stream.P_.tolist()],
        "profile_index": stream.I_.tolist(),
    }

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, f"{name}.json")
    with open(out_path, "w") as f:
        json.dump(out, f)
    print(f"  {name}: stumpy.stumpi in {elapsed:.3f}s")


def run_all():
    print("Running stumpy reference implementation...\n")
    for name in ["sine_wave", "square_wave", "mixed_signal"]:
        run_batch(name)
    run_streaming("streaming_sine")
    print("\nDone.")


if __name__ == "__main__":
    run_all()
