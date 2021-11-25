#![feature(int_log)]

use std::cmp::Ordering;
use std::marker::PhantomData;
use std::ops::{BitAnd, Not};

use rand::prelude::*;

use crate::blocks::Buffers;
pub use crate::blocks::BLOCK_SIZE_BYTES;
use crate::bucket_pointers::{BucketPointer, BucketPointers};
use crate::classifier::{Classifier, ClassifierInfo, ClassifierStorage};
pub use crate::definitions::Buffer;
pub use crate::insertion_sort::is_sorted_by;
use crate::permute::{Overflow, Permute, SwapBuffers};
use crate::sampler::Sampler;

mod blocks;
mod bucket_pointers;
mod classifier;
mod definitions;
mod insertion_sort;
mod permute;
mod sampler;

pub trait Sortable: Copy + Default + std::fmt::Debug + Eq {
    const BUFFER_SIZE: usize;

    type Buffer: AsRef<[Self]> + AsMut<[Self]> + Default + Copy;

    fn align_to_next_block(offset: usize) -> usize {
        (offset + Self::BUFFER_SIZE - 1).bitand((Self::BUFFER_SIZE - 1).not())
    }
}

const BASE_CASE_SIZE: usize = 32;
const LOG_MAX_BUCKETS: u32 = 6;
const MAX_BUCKETS: usize = 1 << (LOG_MAX_BUCKETS + 1);
const SINGLE_LEVEL_THRESHOLD: usize = BASE_CASE_SIZE * (1 << LOG_MAX_BUCKETS);
const TWO_LEVEL_THRESHOLD: usize = SINGLE_LEVEL_THRESHOLD * (1 << LOG_MAX_BUCKETS);
const EQUAL_BUCKETS_THRESHOLD: usize = 5;

fn log_buckets(n: usize) -> u32 {
    if n <= SINGLE_LEVEL_THRESHOLD {
        // Only one more level until the base case, reduce the number of buckets
        1.max((n / BASE_CASE_SIZE).log2())
    } else if n <= TWO_LEVEL_THRESHOLD {
        // Only two more levels until we reach the base case, split the buckets
        // evenly
        1.max(((n / BASE_CASE_SIZE).log2() + 1) / 2)
    } else {
        // Use the maximum number of buckets
        LOG_MAX_BUCKETS
    }
}

fn oversampling_factor(n: usize) -> f64 {
    let v = 0.2 * n.log2() as f64;
    if v < 1.0 {
        1.0
    } else {
        v
    }
}

pub struct Sorter<T, R> {
    sampler: Sampler<R>,
    _marker: PhantomData<T>,
}

pub struct Storage<T: Sortable> {
    classifier: ClassifierStorage<T>,
    buffers: Buffers<T>,
    bucket_pointers: BucketPointers<T>,
    swap: SwapBuffers<T>,
    overflow: T::Buffer,
}

impl<T: Sortable> Default for Storage<T> {
    #[inline]
    fn default() -> Self {
        Self {
            bucket_pointers: [BucketPointer::default(); MAX_BUCKETS],
            classifier: ClassifierStorage::default(),
            buffers: Buffers::default(),
            swap: SwapBuffers::<T>::default(),
            overflow: T::Buffer::default(),
        }
    }
}

impl<T: Sortable> Sorter<T, SmallRng> {
    pub fn with_small_rng() -> Self {
        Self::new(SmallRng::from_rng(thread_rng()).unwrap())
    }
}

impl<T: Sortable, R: Rng> Sorter<T, R> {
    pub fn new(rng: R) -> Self {
        Self {
            sampler: Sampler::new(rng),
            _marker: Default::default(),
        }
    }

    #[allow(unused)]
    pub fn sort<F>(&mut self, values: &mut [T], storage: &mut Storage<T>, is_less: &F)
    where
        F: Fn(&T, &T) -> bool,
    {
        if values.len() <= 2 * BASE_CASE_SIZE {
            insertion_sort::sort(values, is_less);
        } else {
            self.sample_sort(values, storage, is_less);
        }
    }

    fn build_classifier<'a, F>(
        &mut self,
        values: &'a mut [T],
        storage: &mut Storage<T>,
        is_less: &F,
    ) -> ClassifierInfo<'a, T>
    where
        F: Fn(&T, &T) -> bool,
    {
        let n = values.len();
        let log_buckets = log_buckets(n);
        let num_buckets = 1 << log_buckets;
        let step = 1.max(oversampling_factor(n) as usize);
        let num_samples = (step * num_buckets - 1).min(n / 2);

        self.sampler.select_n(values, num_samples);

        let samples = &mut values[0..num_samples];
        self.sort(samples, storage, is_less);

        ClassifierInfo {
            sorted_samples: samples,
            num_buckets,
            step,
        }
    }

    fn cleanup_margins<F>(
        values: &mut [T],
        buffers: &mut Buffers<T>,
        bucket_starts: &[usize; MAX_BUCKETS + 1],
        bucket_pointers: &mut BucketPointers<T>,
        num_buckets: usize,
        overflow: &mut Overflow<T>,
        overflow_bucket: Option<usize>,
        is_less: &F,
    ) where
        F: Fn(&T, &T) -> bool,
    {
        let is_last_level = values.len() <= SINGLE_LEVEL_THRESHOLD;

        for i in 0..num_buckets {
            // Get bucket information
            let start = bucket_starts[i];
            let end = bucket_starts[i + 1];
            let write = bucket_pointers[i].write();

            let head_range = start..T::align_to_next_block(start).min(end);

            // Invariant:
            //            head            tail  overflow
            //            <-->               <><->
            // ----|----|----|----|----|----|----|------
            //           ][     bucket i      ][
            // ----------><-------------------><--------
            //       start^                 end^ ^write
            //                               ^write'
            // - head is empty
            // - head.len() < block size
            // - tail might be filled
            // - tail.len() < block size
            //
            // Cleanup:
            // - Overflow: Tail is empty, write to head and rest to tail
            //   => head always gets filled since overflow.len() == block size
            // - Final block written: Move excess to head
            // - Insert remaining elements from bucket's buffer

            let (head_range, tail_range) = if Some(i) == overflow_bucket && overflow.is_used() {
                // Overflow

                // Overflow buffer has been written => write pointer must be at end of bucket
                // This gives end <= write
                debug_assert_eq!(T::align_to_next_block(end), write);

                let tail_range = (write - T::BUFFER_SIZE)..end;

                // There must be space for at least block size elements
                debug_assert!(T::BUFFER_SIZE <= tail_range.len() + head_range.len());

                let overflow_data = overflow.take().as_mut();
                let (head_src, tail_src) = overflow_data.split_at(head_range.len());

                // Fill head
                values[head_range].copy_from_slice(head_src);

                // Write remaining elements into tail
                let tail_offset = tail_range.start + tail_src.len();
                let tail = &mut values[tail_range.start..tail_offset];
                tail.copy_from_slice(tail_src);
                (tail_offset..tail_range.end, 0..0)
            } else if end < write && T::BUFFER_SIZE < end - start {
                // Final block has been written
                debug_assert_eq!(T::align_to_next_block(end), write);

                let overflow_range = end..write;
                let len = overflow_range.len();

                // Must fit, no other empty space left
                debug_assert!(len <= head_range.len());

                // Write to head
                // TODO can they even overlap?
                values.copy_within(overflow_range, head_range.start);

                ((head_range.start + len)..head_range.end, 0..0)
            } else {
                let tail_range = if end < write { 0..0 } else { write..end };
                debug_assert_eq!(
                    (head_range.len() + tail_range.len()) % T::BUFFER_SIZE,
                    (end - start) % T::BUFFER_SIZE
                );
                (head_range, tail_range)
            };

            // Write elements from buffer
            let buffer = buffers.bucket_mut(i);
            let (head, tail) = buffer.as_slice().split_at(head_range.len());
            values[head_range].copy_from_slice(head);
            let tail = &tail[..tail_range.len()];
            values[tail_range].copy_from_slice(tail);

            // Perform final base case sort here, while the data is still cached
            // TODO this might be very big for a bad bucket distribution
            if is_last_level || (end - start <= 2 * BASE_CASE_SIZE) {
                insertion_sort::sort(&mut values[start..end], is_less);
            }
        }

        debug_assert!(!overflow.is_used());
    }

    fn partition<F>(
        &mut self,
        values: &mut [T],
        bucket_starts: &mut [usize; MAX_BUCKETS + 1],
        storage: &mut Storage<T>,
        is_less: &F,
    ) -> (usize, bool)
    where
        F: Fn(&T, &T) -> bool,
    {
        let (classifier, equal_buckets) = {
            let ClassifierInfo {
                sorted_samples,
                num_buckets,
                step,
            } = self.build_classifier(values, storage, is_less);
            Classifier::from_sorted_samples(
                sorted_samples,
                &mut storage.classifier,
                is_less,
                num_buckets,
                step,
            )
        };
        let Storage {
            buffers,
            bucket_pointers,
            swap,
            overflow,
            ..
        } = storage;
        Self::classify(
            values,
            &classifier,
            buffers,
            bucket_starts,
            bucket_pointers,
            equal_buckets,
        );

        let overflow_bucket = {
            let mut overflow = None;
            for bucket in (0..classifier.num_buckets()).rev() {
                if bucket_starts[bucket + 1] - bucket_starts[bucket] > T::BUFFER_SIZE {
                    overflow = Some(bucket);
                    break;
                }
            }
            overflow
        };

        let permute = Permute {
            classifier: &classifier,
            values,
            bucket_pointers,
        };
        let mut overflow = Overflow::new(overflow);
        if equal_buckets {
            permute.permute_blocks::<true>(swap, &mut overflow);
        } else {
            permute.permute_blocks::<false>(swap, &mut overflow);
        }

        Self::cleanup_margins(
            values,
            buffers,
            bucket_starts,
            bucket_pointers,
            classifier.num_buckets(),
            &mut overflow,
            overflow_bucket,
            is_less,
        );

        (classifier.num_buckets(), equal_buckets)
    }

    fn classify<F>(
        values: &mut [T],
        classifier: &Classifier<T, F>,
        buffers: &mut Buffers<T>,
        bucket_starts: &mut [usize; MAX_BUCKETS + 1],
        bucket_pointers: &mut BucketPointers<T>,
        equal_buckets: bool,
    ) where
        F: Fn(&T, &T) -> bool,
    {
        buffers.reset();
        let first_empty_position =
            classifier.classify_locally(values, buffers, bucket_starts, equal_buckets);

        // IPS4OML_ASSUME_NOT(bucket_start_[num_buckets_] != end_ - begin_);
        // => all values are in a bucket

        for bucket in 0..classifier.num_buckets() {
            // No buffer overflow with align_to_next_block on the last bucket end
            // since we write to the overflow buffer in the permutation instead
            let start = bucket_starts[bucket];
            let stop = bucket_starts[bucket + 1];

            bucket_pointers[bucket] = BucketPointer::new(start, stop, first_empty_position);
        }
    }

    fn sample_sort<F>(&mut self, values: &mut [T], storage: &mut Storage<T>, is_less: &F)
    where
        F: Fn(&T, &T) -> bool,
    {
        let mut bucket_starts = [0usize; MAX_BUCKETS + 1];
        let (num_buckets, equal_buckets) =
            self.partition(values, &mut bucket_starts, storage, is_less);

        // Final base cases were executed in cleanup step, so we're done here
        if values.len() <= SINGLE_LEVEL_THRESHOLD {
            debug_assert!(is_sorted_by(values, is_less));
            return;
        }

        let mut recurse = |bucket: usize| {
            let start = bucket_starts[bucket];
            let stop = bucket_starts[bucket + 1];
            if stop - start > 2 * BASE_CASE_SIZE {
                self.sort(&mut values[start..stop], storage, is_less);
            } else {
                debug_assert!(is_sorted_by(&values[start..stop], is_less));
            }
        };

        // Recurse
        for i in (0..num_buckets).step_by(1 + equal_buckets as usize) {
            recurse(i);
        }
        // TODO Last bucket twice? Or is num_buckets odd in this case?
        if equal_buckets {
            recurse(num_buckets - 1);
        }
    }
}

#[inline]
pub fn sort_by_less<T: Sortable, F>(values: &mut [T], is_less: F)
where
    F: Fn(&T, &T) -> bool,
{
    let mut storage = Storage::default();
    let mut sorter = Sorter::with_small_rng();
    sorter.sort(values, &mut storage, &is_less);
}

#[inline]
pub fn sort_by<T: Sortable, F>(values: &mut [T], cmp: F)
where
    F: Fn(&T, &T) -> Ordering,
{
    sort_by_less(values, |a, b| cmp(a, b) == Ordering::Less)
}

#[allow(unused)]
#[inline]
pub fn sort<T: Sortable + Ord>(values: &mut [T]) {
    sort_by_less(values, |a, b| a.lt(b))
}

#[cfg(test)]
mod test {
    use crate::insertion_sort::is_sorted_by;
    use crate::sort_by_less;

    use std::time::SystemTime;

    #[test]
    fn test_sort() {
        let mut values = (0u32..100000000).rev().collect::<Vec<_>>();
        let time = SystemTime::now();
        // values.sort_unstable();
        let is_less = |a: &u32, b: &u32| a.lt(b);
        sort_by_less(&mut values, is_less);
        dbg!(time.elapsed().unwrap());
        assert!(is_sorted_by(&values, &is_less));
    }
}
