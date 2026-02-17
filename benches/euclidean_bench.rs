use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use motif_rs::algorithms::common::{
    sliding_dot_product, sliding_dot_product_fft, sliding_dot_product_naive,
};
use motif_rs::algorithms::stomp::stomp_rowwise;
use motif_rs::{
    AampEngine, EuclideanEngine, MatrixProfileConfig, RollingStats, ZNormalizedEuclidean,
};

fn bench_sliding_dot_product(c: &mut Criterion) {
    let mut group = c.benchmark_group("sliding_dot_product");
    for n in [1_000, 5_000, 10_000] {
        let ts: Vec<f64> = (0..n).map(|i| (i as f64 * 0.1).sin()).collect();
        let m = 100;
        let q: Vec<f64> = ts[0..m].to_vec();
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| sliding_dot_product(black_box(&q), black_box(&ts)))
        });
    }
    group.finish();
}

fn bench_sdp_naive_vs_fft(c: &mut Criterion) {
    let mut group = c.benchmark_group("sdp_naive_vs_fft");
    let m = 100;
    for n in [500, 1_000, 2_000, 5_000, 10_000] {
        let ts: Vec<f64> = (0..n).map(|i| (i as f64 * 0.1).sin()).collect();
        let q: Vec<f64> = ts[0..m].to_vec();
        group.bench_with_input(BenchmarkId::new("naive", n), &n, |b, _| {
            b.iter(|| sliding_dot_product_naive(black_box(&q), black_box(&ts)))
        });
        group.bench_with_input(BenchmarkId::new("fft", n), &n, |b, _| {
            b.iter(|| sliding_dot_product_fft(black_box(&q), black_box(&ts)))
        });
    }
    group.finish();
}

fn bench_rolling_stats(c: &mut Criterion) {
    let mut group = c.benchmark_group("rolling_stats");
    for n in [1_000, 5_000, 10_000] {
        let ts: Vec<f64> = (0..n).map(|i| (i as f64 * 0.1).sin()).collect();
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| RollingStats::compute(black_box(&ts), 100))
        });
    }
    group.finish();
}

fn bench_stomp(c: &mut Criterion) {
    let mut group = c.benchmark_group("stomp");
    group.sample_size(10);
    for n in [1_000, 5_000, 10_000] {
        let ts: Vec<f64> = (0..n).map(|i| (i as f64 * 0.1).sin()).collect();
        let engine = EuclideanEngine::new(MatrixProfileConfig::new(100));
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| engine.compute(black_box(&ts)))
        });
    }
    group.finish();
}

fn bench_streaming_update(c: &mut Criterion) {
    let ts: Vec<f64> = (0..1_000).map(|i| (i as f64 * 0.1).sin()).collect();
    let engine = EuclideanEngine::new(MatrixProfileConfig::new(100));
    let mut stampi = engine.streaming(&ts, false);

    c.bench_function("streaming_update", |b| {
        let mut val = 1000.0_f64;
        b.iter(|| {
            val += 0.1;
            stampi.update(black_box(val.sin()));
        })
    });
}

fn bench_stomp_rowwise_vs_diagonal(c: &mut Criterion) {
    let mut group = c.benchmark_group("stomp_rowwise_vs_diagonal");
    group.sample_size(10);
    for n in [1_000, 5_000, 10_000] {
        let ts: Vec<f64> = (0..n).map(|i| (i as f64 * 0.1).sin()).collect();
        let config = MatrixProfileConfig::new(100);
        let engine = EuclideanEngine::new(config.clone());
        group.bench_with_input(BenchmarkId::new("diagonal", n), &n, |b, _| {
            b.iter(|| engine.compute(black_box(&ts)))
        });
        group.bench_with_input(BenchmarkId::new("rowwise", n), &n, |b, _| {
            b.iter(|| stomp_rowwise::<ZNormalizedEuclidean>(black_box(&ts), black_box(&config)))
        });
    }
    group.finish();
}

#[cfg(feature = "parallel")]
fn bench_stomp_thread_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("stomp_thread_scaling");
    group.sample_size(10);

    let n = 10_000;
    let ts: Vec<f64> = (0..n).map(|i| (i as f64 * 0.1).sin()).collect();
    let config = MatrixProfileConfig::new(100);

    for threads in [1, 2, 4, 8] {
        group.bench_with_input(
            BenchmarkId::new("threads", threads),
            &threads,
            |b, &threads| {
                let pool = rayon::ThreadPoolBuilder::new()
                    .num_threads(threads)
                    .build()
                    .unwrap();
                let engine = EuclideanEngine::new(config.clone());
                b.iter(|| pool.install(|| engine.compute(black_box(&ts))));
            },
        );
    }
    group.finish();
}

fn bench_aamp(c: &mut Criterion) {
    let mut group = c.benchmark_group("aamp");
    group.sample_size(10);
    for n in [1_000, 5_000, 10_000] {
        let ts: Vec<f64> = (0..n).map(|i| (i as f64 * 0.1).sin()).collect();
        let engine = AampEngine::new(MatrixProfileConfig::new(100));
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| engine.compute(black_box(&ts)))
        });
    }
    group.finish();
}

fn bench_aamp_vs_euclidean(c: &mut Criterion) {
    let mut group = c.benchmark_group("aamp_vs_euclidean");
    group.sample_size(10);
    let n = 10_000;
    let ts: Vec<f64> = (0..n).map(|i| (i as f64 * 0.1).sin()).collect();
    let config = MatrixProfileConfig::new(100);

    group.bench_function("euclidean", |b| {
        let engine = EuclideanEngine::new(config.clone());
        b.iter(|| engine.compute(black_box(&ts)))
    });
    group.bench_function("aamp", |b| {
        let engine = AampEngine::new(config.clone());
        b.iter(|| engine.compute(black_box(&ts)))
    });
    group.finish();
}

fn bench_ab_join(c: &mut Criterion) {
    let mut group = c.benchmark_group("ab_join");
    group.sample_size(10);
    for n in [1_000, 5_000, 10_000] {
        let ts_a: Vec<f64> = (0..n).map(|i| (i as f64 * 0.1).sin()).collect();
        let ts_b: Vec<f64> = (0..n).map(|i| (i as f64 * 0.13).cos()).collect();
        let engine = EuclideanEngine::new(MatrixProfileConfig::new(100));
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| engine.ab_join(black_box(&ts_a), black_box(&ts_b)))
        });
    }
    group.finish();
}

fn bench_topk(c: &mut Criterion) {
    let mut group = c.benchmark_group("topk");
    group.sample_size(10);
    let n = 5_000;
    let ts: Vec<f64> = (0..n).map(|i| (i as f64 * 0.1).sin()).collect();
    let engine = EuclideanEngine::new(MatrixProfileConfig::new(100));

    for k in [1, 3, 5, 10] {
        group.bench_with_input(BenchmarkId::from_parameter(k), &k, |b, &k| {
            b.iter(|| engine.compute_topk(black_box(&ts), black_box(k)))
        });
    }
    group.finish();
}

fn bench_motifs_discords(c: &mut Criterion) {
    // Pre-compute the matrix profile once, then benchmark just extraction
    let n = 10_000;
    let ts: Vec<f64> = (0..n).map(|i| (i as f64 * 0.1).sin()).collect();
    let engine = EuclideanEngine::new(MatrixProfileConfig::new(100));
    let mp = engine.compute(&ts);

    let mut group = c.benchmark_group("motifs_discords");
    group.bench_function("find_motifs_k3", |b| {
        b.iter(|| motif_rs::find_motifs(black_box(&mp), 3))
    });
    group.bench_function("find_discords_k3", |b| {
        b.iter(|| motif_rs::find_discords(black_box(&mp), 3))
    });
    group.finish();
}

fn bench_fluss(c: &mut Criterion) {
    // Pre-compute the matrix profile once, then benchmark just FLUSS
    let n = 10_000;
    let ts: Vec<f64> = (0..n).map(|i| (i as f64 * 0.1).sin()).collect();
    let engine = EuclideanEngine::new(MatrixProfileConfig::new(100));
    let mp = engine.compute(&ts);

    c.bench_function("fluss_n10000", |b| {
        b.iter(|| motif_rs::fluss(black_box(&mp), 3))
    });
}

criterion_group!(
    benches,
    bench_sliding_dot_product,
    bench_sdp_naive_vs_fft,
    bench_rolling_stats,
    bench_stomp,
    bench_streaming_update,
    bench_stomp_rowwise_vs_diagonal,
    bench_aamp,
    bench_aamp_vs_euclidean,
    bench_ab_join,
    bench_topk,
    bench_motifs_discords,
    bench_fluss,
);

#[cfg(feature = "parallel")]
criterion_group!(parallel_benches, bench_stomp_thread_scaling);

#[cfg(feature = "parallel")]
criterion_main!(benches, parallel_benches);

#[cfg(not(feature = "parallel"))]
criterion_main!(benches);
