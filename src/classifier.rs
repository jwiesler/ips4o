use crate::blocks::Buffers;
use crate::insertion_sort::is_sorted_by;
use crate::{Sortable, EQUAL_BUCKETS_THRESHOLD, LOG_MAX_BUCKETS, MAX_BUCKETS};
use std::convert::TryInto;

const CLASSIFIER_STORAGE_LENGTH: usize = MAX_BUCKETS / 2;

pub type Storage<T> = [T; CLASSIFIER_STORAGE_LENGTH];

pub struct ClassifierStorage<T> {
    pub tree: Storage<T>,
    pub splitter: Storage<T>,
}

impl<T: Default + Copy> Default for ClassifierStorage<T> {
    #[inline]
    fn default() -> Self {
        Self {
            tree: [T::default(); CLASSIFIER_STORAGE_LENGTH],
            splitter: [T::default(); CLASSIFIER_STORAGE_LENGTH],
        }
    }
}

struct Tree<'a, T, F> {
    tree: &'a mut Storage<T>,
    is_less: &'a F,
    log_buckets: u32,
}

impl<'a, T, F> Tree<'a, T, F>
where
    T: Copy,
    F: Fn(&T, &T) -> bool,
{
    fn new(
        storage: &'a mut Storage<T>,
        sorted_splitters: &Storage<T>,
        is_less: &'a F,
        log_buckets: u32,
    ) -> Self {
        debug_assert!(log_buckets >= 1);
        debug_assert!(log_buckets <= LOG_MAX_BUCKETS + 1);
        let num_buckets = 1usize << log_buckets;
        let num_splitters = num_buckets - 1;
        let splitters = &sorted_splitters[..num_splitters];

        debug_assert!(is_sorted_by(splitters, is_less));
        debug_assert!(num_buckets <= storage.len());

        let mut ret = Self {
            tree: storage,
            is_less,
            log_buckets,
        };
        ret.build(1, splitters);
        ret
    }

    fn build(&mut self, position: usize, sorted_splitters: &[T]) {
        let mid = sorted_splitters.len() / 2;
        self.tree[position] = sorted_splitters[mid];
        if 2 * position < (1 << self.log_buckets) {
            let (left, right) = sorted_splitters.split_at(mid);
            self.build(2 * position, left);
            self.build(2 * position + 1, right);
        }
    }

    #[inline]
    fn classify(&self, value: &T) -> usize {
        let mut b = 1;
        for _ in 0..self.log_buckets {
            b = 2 * b + ((self.is_less)(&self.tree[b], value)) as usize;
        }
        b
    }

    #[inline]
    fn classify_all<const LOG_BUCKETS: u32, const N: usize>(&self, values: &[T; N]) -> [u32; N] {
        debug_assert_eq!(LOG_BUCKETS, self.log_buckets);

        let mut indices = [1u32; N];
        for _ in 0..LOG_BUCKETS {
            for i in 0..N {
                let value = &values[i];
                let index = indices[i];
                indices[i] = 2 * index + ((self.is_less)(&self.tree[index as usize], value)) as u32;
            }
        }

        indices
    }
}

pub struct ClassifierInfo<'a, T> {
    pub sorted_samples: &'a [T],
    pub num_buckets: usize,
    pub step: usize,
}

pub struct Classifier<'a, T, F> {
    tree: Tree<'a, T, F>,
    sorted_splitters: &'a Storage<T>,
    num_buckets: usize,
}

impl<'a, T: Sortable, F> Classifier<'a, T, F>
where
    F: Fn(&T, &T) -> bool,
{
    pub fn new(
        storage: &'a mut Storage<T>,
        sorted_splitters: &'a mut Storage<T>,
        is_less: &'a F,
        log_buckets: u32,
    ) -> Self {
        debug_assert!(log_buckets <= LOG_MAX_BUCKETS + 1);
        let num_buckets = 1 << log_buckets;

        // Duplicate the last splitter
        let num_splitters = num_buckets - 1;
        sorted_splitters[num_splitters] = sorted_splitters[num_buckets - 1];
        Self {
            tree: Tree::new(storage, sorted_splitters, is_less, log_buckets),
            sorted_splitters,
            num_buckets,
        }
    }

    pub fn num_buckets(&self) -> usize {
        self.num_buckets
    }

    pub fn from_sorted_samples(
        samples: &[T],
        storage: &'a mut ClassifierStorage<T>,
        is_less: &'a F,
        num_buckets: usize,
        step: usize,
    ) -> (Classifier<'a, T, F>, bool) {
        debug_assert!(is_sorted_by(samples, is_less));

        let mut splitter = step - 1;
        let mut offset = 0;
        storage.splitter[offset] = samples[splitter];
        for _ in 2..num_buckets {
            splitter += step;
            if is_less(&storage.splitter[offset], &samples[splitter]) {
                offset += 1;
                storage.splitter[offset] = samples[splitter];
            }
        }

        // Check for duplicate splitters
        let splitter_count = offset + 1;
        let max_splitters = num_buckets - 1;
        let use_equal_buckets = (max_splitters - splitter_count) >= EQUAL_BUCKETS_THRESHOLD;

        // Fill the array to the next power of two
        let log_buckets = splitter_count.log2() + 1;
        let num_buckets = 1 << log_buckets;
        debug_assert!(num_buckets <= storage.splitter.len());
        debug_assert!(splitter_count < num_buckets);
        storage.splitter[splitter_count + 1..num_buckets].fill(samples[splitter]);

        (
            Classifier::new(
                &mut storage.tree,
                &mut storage.splitter,
                is_less,
                log_buckets,
            ),
            use_equal_buckets,
        )
    }

    #[inline]
    pub fn classify<const EQUAL_BUCKETS: bool>(&self, value: &T) -> usize {
        let index = self.tree.classify(value);
        let bucket = index - self.num_buckets;
        if EQUAL_BUCKETS {
            let equal_to_splitter = !(self.tree.is_less)(value, &self.sorted_splitters[bucket]);
            2 * index + equal_to_splitter as usize - 2 * self.num_buckets
        } else {
            bucket
        }
    }

    // TODO why is this different from the other classify used?
    #[inline]
    fn classify_all<const EQUAL_BUCKETS: bool, const LOG_BUCKETS: u32, const N: usize>(
        &self,
        values: &[T; N],
    ) -> [u32; N] {
        let mut indices = self.tree.classify_all::<LOG_BUCKETS, N>(values);
        if EQUAL_BUCKETS {
            for i in 0..N {
                let value = &values[i];
                let index = indices[i];
                let equal_to_splitter = !(self.tree.is_less)(
                    value,
                    // TODO here
                    &self.sorted_splitters[index as usize - self.num_buckets / 2],
                );
                indices[i] = 2 * index + (equal_to_splitter as u32);
            }
        }
        for value in indices.iter_mut() {
            *value -= self.num_buckets as u32;
        }
        indices
    }

    fn classify_locally_inner_buckets<const EQUAL_BUCKETS: bool, const LOG_BUCKETS: u32>(
        &self,
        values: &mut [T],
        blocks: &mut Buffers<T>,
        bucket_sizes: &mut [usize; MAX_BUCKETS],
    ) -> usize {
        const BATCH_SIZE: usize = 16;
        let n = values.len();

        let mut write = 0;
        let mut insert_into_bucket = |values: &mut [T], offset: usize, bucket_index: usize| {
            let mut bucket = blocks.bucket_mut(bucket_index);

            if bucket.is_full() {
                let block = bucket.take();
                debug_assert_eq!(block.len(), T::BUFFER_SIZE);
                values[write..write + block.len()].copy_from_slice(block);
                write += block.len();
                bucket_sizes[bucket_index] += block.len();
            }
            bucket.push(values[offset]);
        };

        let mut i = 0;

        if n > BATCH_SIZE {
            let cutoff = n - BATCH_SIZE;
            while i <= cutoff {
                let batch = (&values[i..i + BATCH_SIZE]).try_into().unwrap();
                let bucket_indices =
                    self.classify_all::<EQUAL_BUCKETS, LOG_BUCKETS, BATCH_SIZE>(batch);

                for (j, bucket) in bucket_indices.iter().cloned().enumerate() {
                    insert_into_bucket(values, i + j, bucket as usize);
                }

                i += BATCH_SIZE;
            }
        }

        for i in i..n {
            let [bucket_index] = self.classify_all::<EQUAL_BUCKETS, LOG_BUCKETS, 1>(
                (&values[i..i + 1]).try_into().unwrap(),
            );

            insert_into_bucket(values, i, bucket_index as usize);
        }

        write
    }

    fn classify_locally_inner<const EQUAL_BUCKETS: bool>(
        &self,
        values: &mut [T],
        blocks: &mut Buffers<T>,
        bucket_sizes: &mut [usize; MAX_BUCKETS],
    ) -> usize {
        macro_rules! gen {
            ($log_buckets:ident, $values:expr, $blocks:expr, $block_sizes:expr, $($i:literal),*) => {
                match $log_buckets {
                    $(
                        $i => self.classify_locally_inner_buckets::<EQUAL_BUCKETS, $i>($values, $blocks, $block_sizes),
                    )+
                    e => unreachable!("missing case for log_buckets {}", e),
                }
            }
        }

        let log_buckets = self.tree.log_buckets;
        gen!(log_buckets, values, blocks, bucket_sizes, 1, 2, 3, 4, 5, 6)
    }

    pub fn classify_locally(
        &self,
        values: &mut [T],
        buffers: &mut Buffers<T>,
        bucket_starts: &mut [usize; MAX_BUCKETS + 1],
        equal_buckets: bool,
    ) -> usize {
        let mut bucket_sizes = [0usize; MAX_BUCKETS];
        let my_first_empty_position = if equal_buckets {
            self.classify_locally_inner::<true>(values, buffers, &mut bucket_sizes)
        } else {
            self.classify_locally_inner::<false>(values, buffers, &mut bucket_sizes)
        };

        // Calculate bucket starts
        let mut sum = 0;
        bucket_starts[0] = 0;
        for i in 0..self.num_buckets {
            // Add the partially filled buffers
            bucket_sizes[i] += buffers.bucket_mut(i).len();

            sum += bucket_sizes[i];
            bucket_starts[i + 1] = sum;
        }

        debug_assert_eq!(sum, values.len());

        my_first_empty_position
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_tree() {
        let mut storage = [0; CLASSIFIER_STORAGE_LENGTH];
        let mut splitter = [0; CLASSIFIER_STORAGE_LENGTH];
        splitter[0..3].copy_from_slice(&[0, 1, 2]);
        let is_less = |a: &u32, b: &u32| a.lt(b);
        let tree = Tree::new(&mut storage, &splitter, &is_less, 2);
        assert_eq!(&tree.tree[1..4], &[1, 0, 2]);

        splitter[0..15].copy_from_slice(&[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]);
        let tree = Tree::new(&mut storage, &splitter, &is_less, 4);
        assert_eq!(
            &tree.tree[1..16],
            &[7, 3, 11, 1, 5, 9, 13, 0, 2, 4, 6, 8, 10, 12, 14]
        );
    }
}
