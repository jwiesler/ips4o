use std::ops::AddAssign;
use std::time::{Duration, Instant};

use criterion::{criterion_group, criterion_main, Criterion};
use rand::distributions::Uniform;
use rand::prelude::*;

use ips4o::{is_sorted_by, Sorter, Storage};

fn bench_sort(c: &mut Criterion) {
    let mut rng = SmallRng::from_seed([0u8; 32]);
    let values = rng
        .sample_iter(Uniform::new(0u32, 10))
        .take(10000000)
        .collect::<Vec<_>>();

    c.bench_function("ips4o", |b| {
        b.iter_custom(|iterations| {
            let mut storage = Storage::default();
            let mut sorter = Sorter::with_small_rng();
            let mut buffer = vec![0; values.len()];

            let mut duration = Duration::default();
            for _ in 0..iterations {
                buffer.copy_from_slice(&values);
                let start = Instant::now();
                sorter.sort(&mut buffer, &mut storage, &|a, b| a.lt(b));
                duration.add_assign(start.elapsed());
                assert!(is_sorted_by(&buffer, &|a, b| a.lt(b)));
            }

            duration
        })
    });

    c.bench_function("sort", |b| {
        b.iter_custom(|iterations| {
            let mut buffer = vec![0; values.len()];

            let mut duration = Duration::default();
            for _ in 0..iterations {
                buffer.copy_from_slice(&values);
                let start = Instant::now();
                buffer.sort();
                duration.add_assign(start.elapsed());
            }

            duration
        })
    });

    c.bench_function("sort_unstable", |b| {
        b.iter_custom(|iterations| {
            let mut buffer = vec![0; values.len()];

            let mut duration = Duration::default();
            for _ in 0..iterations {
                buffer.copy_from_slice(&values);
                let start = Instant::now();
                buffer.sort_unstable();
                duration.add_assign(start.elapsed());
            }

            duration
        })
    });
}

criterion_group!(benches, bench_sort);
criterion_main!(benches);
