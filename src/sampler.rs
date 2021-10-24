use rand::distributions::Uniform;
use rand::prelude::*;

pub struct Sampler<R: Rng>(R);

impl Sampler<SmallRng> {
    pub fn with_small_rng() -> Self {
        Self::new(SmallRng::from_rng(thread_rng()).unwrap())
    }
}

impl<R: Rng> Sampler<R> {
    pub fn new(rng: R) -> Sampler<R> {
        Sampler(rng)
    }

    pub fn select_n<T>(&mut self, mut values: &mut [T], n: usize) {
        assert!(n <= values.len());
        for _ in 0..n {
            let index = Uniform::new(0, values.len()).sample(&mut self.0);
            values.swap(0, index);
            values = &mut values[1..];
        }
    }
}

#[cfg(test)]
mod test {
    use std::fmt::Debug;

    use rand::rngs::mock::StepRng;

    use super::*;

    fn test_with_rng<R: Rng, T: PartialEq + Debug>(
        rng: R,
        values: &mut [T],
        n: usize,
        expected: &[T],
    ) {
        let mut sampler = Sampler::new(rng);
        sampler.select_n(values, n);
        assert_eq!(values, expected, "sampling {} values", n);
    }

    #[test]
    fn test_sampler() {
        test_with_rng(StepRng::new(u64::MAX / 2, 1), &mut [0, 1, 2], 1, &[1, 0, 2]);
        test_with_rng(
            StepRng::new(u64::MAX / 2, u64::MAX / 2),
            &mut [0, 1, 2],
            2,
            &[1, 2, 0],
        );
    }
}
