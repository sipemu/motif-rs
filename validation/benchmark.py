#!/usr/bin/env python3
"""Performance benchmarks comparing motif-rs against stumpy.

Runs both implementations at multiple time series sizes, measures
computation time (excluding I/O overhead), and generates a benchmark report.

Methodology:
- stumpy: time only the stump()/stumpi() call, with a warmup run to trigger
  numba JIT compilation before measuring.
- motif-rs: the validation_runner binary reports internal timing via
  `compute_s` (measured with std::time::Instant, excludes JSON parse/serialize).
- Multiple iterations per size, reporting median to reduce noise.
"""

import json
import os
import subprocess
import statistics
import time

import numpy as np
import stumpy

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
BINARY = os.path.join(PROJECT_ROOT, "target", "release", "examples", "validation_runner")

# Benchmark parameters
BATCH_SIZES = [1_000, 5_000, 10_000, 25_000, 50_000]
BATCH_M = 100
STREAMING_SIZES = [500, 1_000, 2_000, 5_000]
STREAMING_M = 50
STREAMING_N_STREAM = 200
N_ITERATIONS = 5

# Feature benchmark parameters
FEATURE_SIZES = [1_000, 5_000, 10_000, 25_000]
FEATURE_M = 100
TOPK_K = 3
SNIPPETS_K = 3
SNIPPETS_SIZES = [1_000, 5_000, 10_000, 25_000]


def generate_signal(n: int, seed: int = 42) -> np.ndarray:
    """Generate a sine + noise test signal."""
    np.random.seed(seed)
    t = np.linspace(0, 20 * np.pi, n)
    return np.sin(t) + 0.1 * np.random.randn(n)


def warmup_stumpy():
    """Run a small stumpy computation to trigger numba JIT compilation."""
    ts = generate_signal(500, seed=0)
    stumpy.stump(ts, 50)
    # Also warm up stumpi
    stream = stumpy.stumpi(ts[:200], 50, egress=False)
    for v in ts[200:300]:
        stream.update(v)


def bench_stumpy_batch(ts: np.ndarray, m: int) -> float:
    """Benchmark stumpy.stump, return elapsed seconds."""
    start = time.perf_counter()
    stumpy.stump(ts, m)
    return time.perf_counter() - start


def bench_stumpy_streaming(ts_initial: np.ndarray, ts_stream: np.ndarray, m: int) -> float:
    """Benchmark stumpy.stumpi, return elapsed seconds."""
    start = time.perf_counter()
    stream = stumpy.stumpi(ts_initial, m, egress=False)
    for v in ts_stream:
        stream.update(v)
    return time.perf_counter() - start


def bench_rust_batch(ts: np.ndarray, m: int) -> float:
    """Benchmark motif-rs batch STOMP, return compute_s from binary."""
    input_data = json.dumps({"name": "bench", "ts": ts.tolist(), "m": m, "signal_type": "batch"})
    result = subprocess.run(
        [BINARY], input=input_data, capture_output=True, text=True, timeout=300
    )
    if result.returncode != 0:
        raise RuntimeError(f"validation_runner failed: {result.stderr}")
    out = json.loads(result.stdout)
    return out["compute_s"]


def bench_rust_streaming(
    ts_initial: np.ndarray, ts_stream: np.ndarray, m: int
) -> float:
    """Benchmark motif-rs streaming STAMPI, return compute_s from binary."""
    input_data = json.dumps({
        "name": "bench",
        "ts_initial": ts_initial.tolist(),
        "ts_stream": ts_stream.tolist(),
        "m": m,
        "signal_type": "streaming",
    })
    result = subprocess.run(
        [BINARY], input=input_data, capture_output=True, text=True, timeout=300
    )
    if result.returncode != 0:
        raise RuntimeError(f"validation_runner failed: {result.stderr}")
    out = json.loads(result.stdout)
    return out["compute_s"]


def bench_stumpy_aamp(ts: np.ndarray, m: int) -> float:
    """Benchmark stumpy.stump with normalize=False (AAMP), return elapsed seconds."""
    start = time.perf_counter()
    stumpy.stump(ts, m, normalize=False)
    return time.perf_counter() - start


def bench_stumpy_ab_join(ts_a: np.ndarray, ts_b: np.ndarray, m: int) -> float:
    """Benchmark stumpy AB-join, return elapsed seconds."""
    start = time.perf_counter()
    stumpy.stump(ts_a, m, ts_b, ignore_trivial=False)
    return time.perf_counter() - start


def bench_stumpy_topk(ts: np.ndarray, m: int, k: int) -> float:
    """Benchmark stumpy top-k, return elapsed seconds."""
    start = time.perf_counter()
    stumpy.stump(ts, m, k=k)
    return time.perf_counter() - start


def bench_rust_aamp(ts: np.ndarray, m: int) -> float:
    """Benchmark motif-rs AAMP, return compute_s from binary."""
    input_data = json.dumps({
        "name": "bench", "ts": ts.tolist(), "m": m,
        "signal_type": "aamp",
    })
    result = subprocess.run(
        [BINARY], input=input_data, capture_output=True, text=True, timeout=300
    )
    if result.returncode != 0:
        raise RuntimeError(f"validation_runner failed: {result.stderr}")
    out = json.loads(result.stdout)
    return out["compute_s"]


def bench_rust_ab_join(ts_a: np.ndarray, ts_b: np.ndarray, m: int) -> float:
    """Benchmark motif-rs AB-join, return compute_s from binary."""
    input_data = json.dumps({
        "name": "bench", "ts_a": ts_a.tolist(), "ts_b": ts_b.tolist(), "m": m,
        "signal_type": "ab_join",
    })
    result = subprocess.run(
        [BINARY], input=input_data, capture_output=True, text=True, timeout=300
    )
    if result.returncode != 0:
        raise RuntimeError(f"validation_runner failed: {result.stderr}")
    out = json.loads(result.stdout)
    return out["compute_s"]


def bench_rust_topk(ts: np.ndarray, m: int, k: int) -> float:
    """Benchmark motif-rs top-k, return compute_s from binary."""
    input_data = json.dumps({
        "name": "bench", "ts": ts.tolist(), "m": m, "k": k,
        "signal_type": "topk",
    })
    result = subprocess.run(
        [BINARY], input=input_data, capture_output=True, text=True, timeout=300
    )
    if result.returncode != 0:
        raise RuntimeError(f"validation_runner failed: {result.stderr}")
    out = json.loads(result.stdout)
    return out["compute_s"]


def bench_stumpy_snippets(ts: np.ndarray, m: int, k: int) -> float:
    """Benchmark stumpy.snippets, return elapsed seconds."""
    start = time.perf_counter()
    stumpy.snippets(ts, m, k)
    return time.perf_counter() - start


def bench_rust_snippets(ts: np.ndarray, m: int, k: int) -> float:
    """Benchmark motif-rs snippets, return compute_s from binary."""
    input_data = json.dumps({
        "name": "bench", "ts": ts.tolist(), "m": m, "k": k,
        "signal_type": "snippets",
    })
    result = subprocess.run(
        [BINARY], input=input_data, capture_output=True, text=True, timeout=300
    )
    if result.returncode != 0:
        raise RuntimeError(f"validation_runner failed: {result.stderr}")
    out = json.loads(result.stdout)
    return out["compute_s"]


MPDIST_SIZES = [1_000, 5_000, 10_000]
MPDIST_M = 100
STIMP_SIZES = [1_000, 5_000, 10_000]
STIMP_MIN_M = 10
STIMP_MAX_M = 100
STIMP_STEP = 1
MASS_SIZES = [1_000, 5_000, 10_000, 25_000, 50_000]
MASS_QUERY_LEN = 100
CHAINS_SIZES = [1_000, 5_000, 10_000, 25_000]
CHAINS_M = 100
MSTUMP_SIZES = [1_000, 5_000, 10_000]
MSTUMP_M = 20
MSTUMP_D = 3


def bench_stumpy_mpdist(ts_a: np.ndarray, ts_b: np.ndarray, m: int) -> float:
    """Benchmark stumpy.mpdist, return elapsed seconds."""
    start = time.perf_counter()
    stumpy.mpdist(ts_a, ts_b, m)
    return time.perf_counter() - start


def bench_rust_mpdist(ts_a: np.ndarray, ts_b: np.ndarray, m: int) -> float:
    """Benchmark motif-rs mpdist, return compute_s from binary."""
    input_data = json.dumps({
        "name": "bench", "ts_a": ts_a.tolist(), "ts_b": ts_b.tolist(), "m": m,
        "signal_type": "mpdist",
    })
    result = subprocess.run(
        [BINARY], input=input_data, capture_output=True, text=True, timeout=300
    )
    if result.returncode != 0:
        raise RuntimeError(f"validation_runner failed: {result.stderr}")
    out = json.loads(result.stdout)
    return out["compute_s"]


def bench_stumpy_stimp(ts: np.ndarray, min_m: int, max_m: int) -> float:
    """Benchmark stumpy.stimp, return elapsed seconds.

    stumpy.stimp is lazy — the constructor returns immediately and profiles are
    computed on-demand. We must call update() for all window sizes and access
    PAN_ to force full computation.
    """
    n_windows = max_m - min_m + 1
    start = time.perf_counter()
    pan = stumpy.stimp(ts, min_m=min_m, max_m=max_m)
    for _ in range(n_windows):
        pan.update()
    _ = pan.PAN_
    return time.perf_counter() - start


def bench_rust_stimp(ts: np.ndarray, min_m: int, max_m: int, step: int) -> float:
    """Benchmark motif-rs stimp, return compute_s from binary."""
    input_data = json.dumps({
        "name": "bench", "ts": ts.tolist(), "m": min_m,
        "min_m": min_m, "max_m": max_m, "step": step,
        "signal_type": "stimp",
    })
    result = subprocess.run(
        [BINARY], input=input_data, capture_output=True, text=True, timeout=600
    )
    if result.returncode != 0:
        raise RuntimeError(f"validation_runner failed: {result.stderr}")
    out = json.loads(result.stdout)
    return out["compute_s"]


def bench_stumpy_mass(query: np.ndarray, ts: np.ndarray) -> float:
    """Benchmark stumpy.mass, return elapsed seconds."""
    start = time.perf_counter()
    stumpy.mass(query, ts)
    return time.perf_counter() - start


def bench_rust_mass(query: np.ndarray, ts: np.ndarray) -> float:
    """Benchmark motif-rs mass, return compute_s from binary."""
    input_data = json.dumps({
        "name": "bench", "ts": ts.tolist(), "query": query.tolist(),
        "m": len(query), "signal_type": "mass",
    })
    result = subprocess.run(
        [BINARY], input=input_data, capture_output=True, text=True, timeout=300
    )
    if result.returncode != 0:
        raise RuntimeError(f"validation_runner failed: {result.stderr}")
    out = json.loads(result.stdout)
    return out["compute_s"]


def bench_stumpy_chains(ts: np.ndarray, m: int) -> float:
    """Benchmark stumpy stump + allc, return elapsed seconds."""
    start = time.perf_counter()
    result = stumpy.stump(ts, m)
    left_idx = result[:, 2].astype(int)
    right_idx = result[:, 3].astype(int)
    stumpy.allc(left_idx, right_idx)
    return time.perf_counter() - start


def bench_rust_chains(ts: np.ndarray, m: int) -> float:
    """Benchmark motif-rs chains (stomp + allc), return compute_s from binary."""
    input_data = json.dumps({
        "name": "bench", "ts": ts.tolist(), "m": m,
        "signal_type": "chains",
    })
    result = subprocess.run(
        [BINARY], input=input_data, capture_output=True, text=True, timeout=300
    )
    if result.returncode != 0:
        raise RuntimeError(f"validation_runner failed: {result.stderr}")
    out = json.loads(result.stdout)
    return out["compute_s"]


def generate_multi_dim_signal(n: int, d: int, seed: int = 42) -> np.ndarray:
    """Generate a d-dimensional sine + noise test signal."""
    np.random.seed(seed)
    T = np.empty((d, n))
    t = np.linspace(0, 20 * np.pi, n)
    for i in range(d):
        T[i] = np.sin((i + 1) * t) + 0.1 * np.random.randn(n)
    return T


def bench_stumpy_mstump(T: np.ndarray, m: int) -> float:
    """Benchmark stumpy.mstump, return elapsed seconds."""
    start = time.perf_counter()
    stumpy.mstump(T, m)
    return time.perf_counter() - start


def bench_rust_mstump(T: np.ndarray, m: int) -> float:
    """Benchmark motif-rs mstump, return compute_s from binary."""
    input_data = json.dumps({
        "name": "bench",
        "ts": [T[i].tolist() for i in range(T.shape[0])],
        "m": m,
        "signal_type": "mstump",
    })
    result = subprocess.run(
        [BINARY], input=input_data, capture_output=True, text=True, timeout=300
    )
    if result.returncode != 0:
        raise RuntimeError(f"validation_runner failed: {result.stderr}")
    out = json.loads(result.stdout)
    return out["compute_s"]


def run_mstump_benchmarks() -> list[dict]:
    """Run MSTUMP benchmarks at multiple sizes."""
    results = []
    for n in MSTUMP_SIZES:
        T = generate_multi_dim_signal(n, MSTUMP_D)
        stumpy_times = []
        rust_times = []

        for _ in range(N_ITERATIONS):
            stumpy_times.append(bench_stumpy_mstump(T, MSTUMP_M))
            rust_times.append(bench_rust_mstump(T, MSTUMP_M))

        stumpy_median = statistics.median(stumpy_times)
        rust_median = statistics.median(rust_times)
        speedup = stumpy_median / rust_median if rust_median > 0 else float("inf")

        result = {
            "type": "mstump", "n": n, "m": MSTUMP_M, "d": MSTUMP_D,
            "stumpy_median_s": stumpy_median, "rust_median_s": rust_median,
            "speedup": speedup,
        }
        results.append(result)
        print(
            f"  MSTUMP d={MSTUMP_D} n={n:>6}: stumpy={stumpy_median:.4f}s  motif-rs={rust_median:.4f}s  "
            f"speedup={speedup:.1f}x"
        )
    return results


def run_mass_benchmarks() -> list[dict]:
    """Run MASS benchmarks at multiple sizes."""
    results = []
    for n in MASS_SIZES:
        ts = generate_signal(n)
        query = ts[100:100 + MASS_QUERY_LEN]
        stumpy_times = []
        rust_times = []

        for _ in range(N_ITERATIONS):
            stumpy_times.append(bench_stumpy_mass(query, ts))
            rust_times.append(bench_rust_mass(query, ts))

        stumpy_median = statistics.median(stumpy_times)
        rust_median = statistics.median(rust_times)
        speedup = stumpy_median / rust_median if rust_median > 0 else float("inf")

        result = {
            "type": "mass", "n": n, "m": MASS_QUERY_LEN,
            "stumpy_median_s": stumpy_median, "rust_median_s": rust_median,
            "speedup": speedup,
        }
        results.append(result)
        print(
            f"  MASS n={n:>6}: stumpy={stumpy_median:.4f}s  motif-rs={rust_median:.4f}s  "
            f"speedup={speedup:.1f}x"
        )
    return results


def run_chains_benchmarks() -> list[dict]:
    """Run chains benchmarks at multiple sizes (includes STOMP + ALLC)."""
    results = []
    for n in CHAINS_SIZES:
        ts = generate_signal(n)
        stumpy_times = []
        rust_times = []

        for _ in range(N_ITERATIONS):
            stumpy_times.append(bench_stumpy_chains(ts, CHAINS_M))
            rust_times.append(bench_rust_chains(ts, CHAINS_M))

        stumpy_median = statistics.median(stumpy_times)
        rust_median = statistics.median(rust_times)
        speedup = stumpy_median / rust_median if rust_median > 0 else float("inf")

        result = {
            "type": "chains", "n": n, "m": CHAINS_M,
            "stumpy_median_s": stumpy_median, "rust_median_s": rust_median,
            "speedup": speedup,
        }
        results.append(result)
        print(
            f"  Chains n={n:>6}: stumpy={stumpy_median:.4f}s  motif-rs={rust_median:.4f}s  "
            f"speedup={speedup:.1f}x"
        )
    return results


def run_mpdist_benchmarks() -> list[dict]:
    """Run MPdist benchmarks at multiple sizes."""
    results = []
    for n in MPDIST_SIZES:
        ts_a = generate_signal(n, seed=42)
        ts_b = generate_signal(n, seed=99)
        stumpy_times = []
        rust_times = []

        for _ in range(N_ITERATIONS):
            stumpy_times.append(bench_stumpy_mpdist(ts_a, ts_b, MPDIST_M))
            rust_times.append(bench_rust_mpdist(ts_a, ts_b, MPDIST_M))

        stumpy_median = statistics.median(stumpy_times)
        rust_median = statistics.median(rust_times)
        speedup = stumpy_median / rust_median if rust_median > 0 else float("inf")

        result = {
            "type": "mpdist", "n": n, "m": MPDIST_M,
            "stumpy_median_s": stumpy_median, "rust_median_s": rust_median,
            "speedup": speedup,
        }
        results.append(result)
        print(
            f"  MPdist n={n:>6}: stumpy={stumpy_median:.4f}s  motif-rs={rust_median:.4f}s  "
            f"speedup={speedup:.1f}x"
        )
    return results


def run_stimp_benchmarks() -> list[dict]:
    """Run STIMP benchmarks at multiple sizes."""
    results = []
    for n in STIMP_SIZES:
        ts = generate_signal(n)
        stumpy_times = []
        rust_times = []

        for _ in range(N_ITERATIONS):
            stumpy_times.append(bench_stumpy_stimp(ts, STIMP_MIN_M, STIMP_MAX_M))
            rust_times.append(bench_rust_stimp(ts, STIMP_MIN_M, STIMP_MAX_M, STIMP_STEP))

        stumpy_median = statistics.median(stumpy_times)
        rust_median = statistics.median(rust_times)
        speedup = stumpy_median / rust_median if rust_median > 0 else float("inf")

        result = {
            "type": "stimp", "n": n,
            "min_m": STIMP_MIN_M, "max_m": STIMP_MAX_M, "step": STIMP_STEP,
            "stumpy_median_s": stumpy_median, "rust_median_s": rust_median,
            "speedup": speedup,
        }
        results.append(result)
        print(
            f"  STIMP n={n:>6}: stumpy={stumpy_median:.4f}s  motif-rs={rust_median:.4f}s  "
            f"speedup={speedup:.1f}x"
        )
    return results


def run_snippets_benchmarks() -> list[dict]:
    """Run snippets benchmarks at multiple sizes."""
    results = []
    for n in SNIPPETS_SIZES:
        ts = generate_signal(n)
        stumpy_times = []
        rust_times = []

        for _ in range(N_ITERATIONS):
            stumpy_times.append(bench_stumpy_snippets(ts, FEATURE_M, SNIPPETS_K))
            rust_times.append(bench_rust_snippets(ts, FEATURE_M, SNIPPETS_K))

        stumpy_median = statistics.median(stumpy_times)
        rust_median = statistics.median(rust_times)
        speedup = stumpy_median / rust_median if rust_median > 0 else float("inf")

        result = {
            "type": "snippets", "n": n, "m": FEATURE_M, "k": SNIPPETS_K,
            "stumpy_median_s": stumpy_median, "rust_median_s": rust_median,
            "speedup": speedup,
        }
        results.append(result)
        print(
            f"  Snippets n={n:>6}: stumpy={stumpy_median:.4f}s  motif-rs={rust_median:.4f}s  "
            f"speedup={speedup:.1f}x"
        )
    return results


def run_aamp_benchmarks() -> list[dict]:
    """Run AAMP benchmarks at multiple sizes."""
    results = []
    for n in FEATURE_SIZES:
        ts = generate_signal(n)
        stumpy_times = []
        rust_times = []

        for _ in range(N_ITERATIONS):
            stumpy_times.append(bench_stumpy_aamp(ts, FEATURE_M))
            rust_times.append(bench_rust_aamp(ts, FEATURE_M))

        stumpy_median = statistics.median(stumpy_times)
        rust_median = statistics.median(rust_times)
        speedup = stumpy_median / rust_median if rust_median > 0 else float("inf")

        result = {
            "type": "aamp", "n": n, "m": FEATURE_M,
            "stumpy_median_s": stumpy_median, "rust_median_s": rust_median,
            "speedup": speedup,
        }
        results.append(result)
        print(
            f"  AAMP n={n:>6}: stumpy={stumpy_median:.4f}s  motif-rs={rust_median:.4f}s  "
            f"speedup={speedup:.1f}x"
        )
    return results


def run_ab_join_benchmarks() -> list[dict]:
    """Run AB-join benchmarks at multiple sizes."""
    results = []
    for n in FEATURE_SIZES:
        ts_a = generate_signal(n, seed=42)
        ts_b = generate_signal(n, seed=99)
        stumpy_times = []
        rust_times = []

        for _ in range(N_ITERATIONS):
            stumpy_times.append(bench_stumpy_ab_join(ts_a, ts_b, FEATURE_M))
            rust_times.append(bench_rust_ab_join(ts_a, ts_b, FEATURE_M))

        stumpy_median = statistics.median(stumpy_times)
        rust_median = statistics.median(rust_times)
        speedup = stumpy_median / rust_median if rust_median > 0 else float("inf")

        result = {
            "type": "ab_join", "n": n, "m": FEATURE_M,
            "stumpy_median_s": stumpy_median, "rust_median_s": rust_median,
            "speedup": speedup,
        }
        results.append(result)
        print(
            f"  AB-Join n={n:>6}: stumpy={stumpy_median:.4f}s  motif-rs={rust_median:.4f}s  "
            f"speedup={speedup:.1f}x"
        )
    return results


def run_topk_benchmarks() -> list[dict]:
    """Run top-k benchmarks at multiple sizes."""
    results = []
    for n in FEATURE_SIZES:
        ts = generate_signal(n)
        stumpy_times = []
        rust_times = []

        for _ in range(N_ITERATIONS):
            stumpy_times.append(bench_stumpy_topk(ts, FEATURE_M, TOPK_K))
            rust_times.append(bench_rust_topk(ts, FEATURE_M, TOPK_K))

        stumpy_median = statistics.median(stumpy_times)
        rust_median = statistics.median(rust_times)
        speedup = stumpy_median / rust_median if rust_median > 0 else float("inf")

        result = {
            "type": "topk", "n": n, "m": FEATURE_M, "k": TOPK_K,
            "stumpy_median_s": stumpy_median, "rust_median_s": rust_median,
            "speedup": speedup,
        }
        results.append(result)
        print(
            f"  Top-{TOPK_K} n={n:>6}: stumpy={stumpy_median:.4f}s  motif-rs={rust_median:.4f}s  "
            f"speedup={speedup:.1f}x"
        )
    return results


def run_batch_benchmarks() -> list[dict]:
    """Run batch STOMP benchmarks at multiple sizes."""
    results = []
    for n in BATCH_SIZES:
        ts = generate_signal(n)
        stumpy_times = []
        rust_times = []

        for i in range(N_ITERATIONS):
            stumpy_times.append(bench_stumpy_batch(ts, BATCH_M))
            rust_times.append(bench_rust_batch(ts, BATCH_M))

        stumpy_median = statistics.median(stumpy_times)
        rust_median = statistics.median(rust_times)
        speedup = stumpy_median / rust_median if rust_median > 0 else float("inf")

        result = {
            "type": "batch",
            "n": n,
            "m": BATCH_M,
            "stumpy_median_s": stumpy_median,
            "rust_median_s": rust_median,
            "speedup": speedup,
            "stumpy_times": stumpy_times,
            "rust_times": rust_times,
        }
        results.append(result)
        print(
            f"  STOMP n={n:>6}: stumpy={stumpy_median:.4f}s  motif-rs={rust_median:.4f}s  "
            f"speedup={speedup:.1f}x"
        )

    return results


def run_streaming_benchmarks() -> list[dict]:
    """Run streaming STAMPI benchmarks at multiple initial sizes."""
    results = []
    for n_init in STREAMING_SIZES:
        ts = generate_signal(n_init + STREAMING_N_STREAM)
        ts_initial = ts[:n_init]
        ts_stream = ts[n_init:]
        stumpy_times = []
        rust_times = []

        for i in range(N_ITERATIONS):
            stumpy_times.append(bench_stumpy_streaming(ts_initial, ts_stream, STREAMING_M))
            rust_times.append(bench_rust_streaming(ts_initial, ts_stream, STREAMING_M))

        stumpy_median = statistics.median(stumpy_times)
        rust_median = statistics.median(rust_times)
        speedup = stumpy_median / rust_median if rust_median > 0 else float("inf")

        result = {
            "type": "streaming",
            "n_initial": n_init,
            "n_stream": STREAMING_N_STREAM,
            "m": STREAMING_M,
            "stumpy_median_s": stumpy_median,
            "rust_median_s": rust_median,
            "speedup": speedup,
            "stumpy_times": stumpy_times,
            "rust_times": rust_times,
        }
        results.append(result)
        print(
            f"  STAMPI n_init={n_init:>5} +{STREAMING_N_STREAM}: "
            f"stumpy={stumpy_median:.4f}s  motif-rs={rust_median:.4f}s  "
            f"speedup={speedup:.1f}x"
        )

    return results


def generate_report(
    batch_results: list[dict],
    streaming_results: list[dict],
    aamp_results: list[dict] | None = None,
    ab_join_results: list[dict] | None = None,
    topk_results: list[dict] | None = None,
    snippets_results: list[dict] | None = None,
    mpdist_results: list[dict] | None = None,
    stimp_results: list[dict] | None = None,
    mass_results: list[dict] | None = None,
    chains_results: list[dict] | None = None,
    mstump_results: list[dict] | None = None,
):
    """Generate markdown benchmark report."""
    report = []
    report.append("# motif-rs vs stumpy: Performance Benchmark Report\n")
    report.append(
        f"Methodology: {N_ITERATIONS} iterations per configuration, reporting median. "
        f"stumpy timed after JIT warmup; motif-rs timed internally (excludes I/O).\n"
    )

    # Batch STOMP
    report.append("## Batch STOMP (m=100)\n")
    report.append("| n | stumpy (s) | motif-rs (s) | Speedup |")
    report.append("|--:|----------:|------------:|--------:|")
    for r in batch_results:
        report.append(
            f"| {r['n']:,} | {r['stumpy_median_s']:.4f} | {r['rust_median_s']:.4f} "
            f"| **{r['speedup']:.1f}x** |"
        )

    # Streaming STAMPI
    report.append(f"\n## Streaming STAMPI (m=50, +{STREAMING_N_STREAM} points)\n")
    report.append("| n_initial | stumpy (s) | motif-rs (s) | Speedup |")
    report.append("|----------:|----------:|------------:|--------:|")
    for r in streaming_results:
        report.append(
            f"| {r['n_initial']:,} | {r['stumpy_median_s']:.4f} | {r['rust_median_s']:.4f} "
            f"| **{r['speedup']:.1f}x** |"
        )

    # AAMP
    if aamp_results:
        report.append(f"\n## AAMP — Non-normalized Euclidean (m={FEATURE_M})\n")
        report.append("| n | stumpy (s) | motif-rs (s) | Speedup |")
        report.append("|--:|----------:|------------:|--------:|")
        for r in aamp_results:
            report.append(
                f"| {r['n']:,} | {r['stumpy_median_s']:.4f} | {r['rust_median_s']:.4f} "
                f"| **{r['speedup']:.1f}x** |"
            )

    # AB-Join
    if ab_join_results:
        report.append(f"\n## AB-Join (m={FEATURE_M}, n_a = n_b = n)\n")
        report.append("| n | stumpy (s) | motif-rs (s) | Speedup |")
        report.append("|--:|----------:|------------:|--------:|")
        for r in ab_join_results:
            report.append(
                f"| {r['n']:,} | {r['stumpy_median_s']:.4f} | {r['rust_median_s']:.4f} "
                f"| **{r['speedup']:.1f}x** |"
            )

    # Top-k
    if topk_results:
        report.append(f"\n## Top-k Nearest Neighbors (m={FEATURE_M}, k={TOPK_K})\n")
        report.append("| n | stumpy (s) | motif-rs (s) | Speedup |")
        report.append("|--:|----------:|------------:|--------:|")
        for r in topk_results:
            report.append(
                f"| {r['n']:,} | {r['stumpy_median_s']:.4f} | {r['rust_median_s']:.4f} "
                f"| **{r['speedup']:.1f}x** |"
            )

    # Snippets
    if snippets_results:
        report.append(f"\n## Snippets (m={FEATURE_M}, k={SNIPPETS_K})\n")
        report.append("| n | stumpy (s) | motif-rs (s) | Speedup |")
        report.append("|--:|----------:|------------:|--------:|")
        for r in snippets_results:
            report.append(
                f"| {r['n']:,} | {r['stumpy_median_s']:.4f} | {r['rust_median_s']:.4f} "
                f"| **{r['speedup']:.1f}x** |"
            )

    # MPdist
    if mpdist_results:
        report.append(f"\n## MPdist (m={MPDIST_M}, n_a = n_b = n)\n")
        report.append("| n | stumpy (s) | motif-rs (s) | Speedup |")
        report.append("|--:|----------:|------------:|--------:|")
        for r in mpdist_results:
            report.append(
                f"| {r['n']:,} | {r['stumpy_median_s']:.4f} | {r['rust_median_s']:.4f} "
                f"| **{r['speedup']:.1f}x** |"
            )

    # STIMP
    if stimp_results:
        report.append(f"\n## STIMP — Pan Matrix Profile (min_m={STIMP_MIN_M}, max_m={STIMP_MAX_M}, step={STIMP_STEP})\n")
        report.append("| n | stumpy (s) | motif-rs (s) | Speedup |")
        report.append("|--:|----------:|------------:|--------:|")
        for r in stimp_results:
            report.append(
                f"| {r['n']:,} | {r['stumpy_median_s']:.4f} | {r['rust_median_s']:.4f} "
                f"| **{r['speedup']:.1f}x** |"
            )

    # MASS
    if mass_results:
        report.append(f"\n## MASS — Distance Profile (query_len={MASS_QUERY_LEN})\n")
        report.append("| n | stumpy (s) | motif-rs (s) | Speedup |")
        report.append("|--:|----------:|------------:|--------:|")
        for r in mass_results:
            report.append(
                f"| {r['n']:,} | {r['stumpy_median_s']:.4f} | {r['rust_median_s']:.4f} "
                f"| **{r['speedup']:.1f}x** |"
            )

    # Chains
    if chains_results:
        report.append(f"\n## Chains — STOMP + ALLC (m={CHAINS_M})\n")
        report.append("| n | stumpy (s) | motif-rs (s) | Speedup |")
        report.append("|--:|----------:|------------:|--------:|")
        for r in chains_results:
            report.append(
                f"| {r['n']:,} | {r['stumpy_median_s']:.4f} | {r['rust_median_s']:.4f} "
                f"| **{r['speedup']:.1f}x** |"
            )

    # MSTUMP
    if mstump_results:
        report.append(f"\n## MSTUMP — Multi-Dimensional Matrix Profile (d={MSTUMP_D}, m={MSTUMP_M})\n")
        report.append("| n | stumpy (s) | motif-rs (s) | Speedup |")
        report.append("|--:|----------:|------------:|--------:|")
        for r in mstump_results:
            report.append(
                f"| {r['n']:,} | {r['stumpy_median_s']:.4f} | {r['rust_median_s']:.4f} "
                f"| **{r['speedup']:.1f}x** |"
            )

    # Notes
    report.append("\n## Notes\n")
    report.append(
        "- **stumpy** uses numba JIT-compiled parallel STOMP. First run triggers compilation; "
        "all measurements taken after warmup."
    )
    report.append(
        "- **motif-rs** uses single-threaded STOMP with O(1) QT recurrence updates. "
        "Timing measured internally with `std::time::Instant` (excludes JSON I/O)."
    )
    report.append(
        "- Streaming benchmarks measure the full pipeline: initial batch computation + "
        "incremental updates for each new point."
    )
    report.append(
        "- Both implementations produce numerically equivalent results "
        "(MAD < 1e-10, see comparison_report.md)."
    )

    report_text = "\n".join(report) + "\n"

    os.makedirs(RESULTS_DIR, exist_ok=True)
    report_path = os.path.join(RESULTS_DIR, "benchmark_report.md")
    with open(report_path, "w") as f:
        f.write(report_text)
    print(f"\n  Report written to {report_path}")

    # Also save raw data
    raw_data = {
        "batch": batch_results,
        "streaming": streaming_results,
    }
    if aamp_results:
        raw_data["aamp"] = aamp_results
    if ab_join_results:
        raw_data["ab_join"] = ab_join_results
    if topk_results:
        raw_data["topk"] = topk_results
    if snippets_results:
        raw_data["snippets"] = snippets_results
    if mpdist_results:
        raw_data["mpdist"] = mpdist_results
    if stimp_results:
        raw_data["stimp"] = stimp_results
    if mass_results:
        raw_data["mass"] = mass_results
    if chains_results:
        raw_data["chains"] = chains_results
    if mstump_results:
        raw_data["mstump"] = mstump_results

    raw_path = os.path.join(RESULTS_DIR, "benchmark_raw.json")
    with open(raw_path, "w") as f:
        json.dump(raw_data, f, indent=2)
    print(f"  Raw data written to {raw_path}")

    return report_text


def main():
    print("motif-rs Performance Benchmarks")
    print("=" * 60)

    # Check binary exists
    if not os.path.exists(BINARY):
        print(f"\n  Building motif-rs validation runner...")
        result = subprocess.run(
            [
                "cargo", "build", "--release",
                "--example", "validation_runner",
                "--features", "validation",
            ],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print(f"  ERROR: Failed to build: {result.stderr}")
            return

    # Warm up stumpy JIT
    print("\n  Warming up stumpy (JIT compilation)...")
    warmup_stumpy()
    # Also warm up AAMP path
    ts_warmup = generate_signal(500, seed=0)
    stumpy.stump(ts_warmup, 50, normalize=False)

    # Batch benchmarks
    print(f"\nBatch STOMP benchmarks (m={BATCH_M}, {N_ITERATIONS} iterations):\n")
    batch_results = run_batch_benchmarks()

    # Streaming benchmarks
    print(f"\nStreaming STAMPI benchmarks (m={STREAMING_M}, +{STREAMING_N_STREAM} points, "
          f"{N_ITERATIONS} iterations):\n")
    streaming_results = run_streaming_benchmarks()

    # AAMP benchmarks
    print(f"\nAAMP benchmarks (m={FEATURE_M}, {N_ITERATIONS} iterations):\n")
    aamp_results = run_aamp_benchmarks()

    # AB-Join benchmarks
    print(f"\nAB-Join benchmarks (m={FEATURE_M}, {N_ITERATIONS} iterations):\n")
    ab_join_results = run_ab_join_benchmarks()

    # Top-k benchmarks
    print(f"\nTop-{TOPK_K} benchmarks (m={FEATURE_M}, {N_ITERATIONS} iterations):\n")
    topk_results = run_topk_benchmarks()

    # Snippets benchmarks
    print(f"\nSnippets benchmarks (m={FEATURE_M}, k={SNIPPETS_K}, {N_ITERATIONS} iterations):\n")
    snippets_results = run_snippets_benchmarks()

    # MPdist benchmarks
    print(f"\nMPdist benchmarks (m={MPDIST_M}, {N_ITERATIONS} iterations):\n")
    mpdist_results = run_mpdist_benchmarks()

    # STIMP benchmarks
    print(f"\nSTIMP benchmarks (min_m={STIMP_MIN_M}, max_m={STIMP_MAX_M}, step={STIMP_STEP}, {N_ITERATIONS} iterations):\n")
    stimp_results = run_stimp_benchmarks()

    # MASS benchmarks
    print(f"\nMASS benchmarks (query_len={MASS_QUERY_LEN}, {N_ITERATIONS} iterations):\n")
    mass_results = run_mass_benchmarks()

    # Chains benchmarks
    print(f"\nChains benchmarks (m={CHAINS_M}, {N_ITERATIONS} iterations):\n")
    chains_results = run_chains_benchmarks()

    # MSTUMP benchmarks
    print(f"\nMSTUMP benchmarks (d={MSTUMP_D}, m={MSTUMP_M}, {N_ITERATIONS} iterations):\n")
    # Warm up stumpy mstump path
    T_warmup = generate_multi_dim_signal(500, MSTUMP_D, seed=0)
    stumpy.mstump(T_warmup, MSTUMP_M)
    mstump_results = run_mstump_benchmarks()

    # Generate report
    print()
    generate_report(
        batch_results, streaming_results, aamp_results, ab_join_results,
        topk_results, snippets_results, mpdist_results, stimp_results,
        mass_results, chains_results, mstump_results,
    )


if __name__ == "__main__":
    main()
