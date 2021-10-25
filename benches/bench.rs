use std::ops::AddAssign;
use std::time::{Duration, Instant};

use criterion::{criterion_group, criterion_main, Bencher, Criterion};
use rand::distributions::Uniform;
use rand::prelude::*;

use ips4o::{is_sorted_by, Sorter, Storage};

fn bench_sort(c: &mut Criterion) {
    fn bench<S>(b: &mut Bencher, init: impl Fn() -> S, sort: impl Fn(&mut [u32], &mut S)) {
        b.iter_custom(|iterations| {
            let mut sorter = init();
            let count = 1_000_000;
            let mut iter = SmallRng::from_seed([0u8; 32]).sample_iter(Uniform::new(0u32, 100));
            let mut buffer = vec![0; count];

            let mut duration = Duration::default();
            for _ in 0..iterations {
                buffer.fill_with(|| iter.next().unwrap());
                let start = Instant::now();
                sort(&mut buffer, &mut sorter);
                duration.add_assign(start.elapsed());
                assert!(is_sorted_by(&buffer, &|a, b| a.lt(b)));
            }

            duration
        });
    }

    c.bench_function("ips4o", |b| {
        bench(
            b,
            || (Storage::default(), Sorter::with_small_rng()),
            |values, (storage, sorter)| {
                sorter.sort(values, storage, &|a, b| a.lt(b));
            },
        );
    });

    c.bench_function("sort_unstable", |b| {
        bench(
            b,
            || (),
            |values, _| {
                values.sort_unstable();
            },
        );
    });

    c.bench_function("sort", |b| {
        bench(
            b,
            || (),
            |values, _| {
                values.sort();
            },
        );
    });
}

criterion_group!(
    name = benches;
    config = Criterion::default().warm_up_time(Duration::from_secs(100)).measurement_time(Duration::from_secs(100));
    targets = bench_sort
);
criterion_main!(benches);
