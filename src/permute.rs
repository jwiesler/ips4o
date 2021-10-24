use crate::bucket_pointers::WriteBlock;
use crate::classifier::Classifier;
use crate::{BucketPointers, Sortable};

pub type SwapBuffers<T> = [<T as Sortable>::Buffer; 2];

struct Swap<'a, T: Sortable> {
    buffers: &'a mut SwapBuffers<T>,
    current: bool,
}

impl<'a, T: Sortable> Swap<'a, T> {
    #[inline]
    fn new(buffers: &'a mut SwapBuffers<T>) -> Self {
        Self {
            buffers,
            current: false,
        }
    }

    #[inline]
    fn current_mut(&mut self) -> &mut T::Buffer {
        &mut self.buffers[self.current as usize]
    }

    #[inline]
    fn buffers_mut(&mut self) -> (&mut T::Buffer, &mut T::Buffer) {
        let Swap {
            buffers: [a, b],
            current,
        } = *self;
        if current {
            (b, a)
        } else {
            (a, b)
        }
    }

    #[inline]
    fn swap_buffers(&mut self) {
        self.current = !self.current;
    }
}

pub struct Overflow<'a, T: Sortable> {
    buffer: &'a mut T::Buffer,
    used: bool,
}

impl<'a, T: Sortable> Overflow<'a, T> {
    #[inline]
    pub fn new(buffer: &'a mut T::Buffer) -> Self {
        Self {
            buffer,
            used: false,
        }
    }

    #[inline]
    pub fn is_used(&self) -> bool {
        self.used
    }

    #[inline]
    pub fn take(&mut self) -> &mut T::Buffer {
        self.used = false;
        &mut self.buffer
    }

    #[inline]
    pub fn insert(&mut self, src: &[T]) {
        debug_assert!(!self.used);
        let buffer = self.buffer.as_mut();
        debug_assert_eq!(src.len(), buffer.len());
        buffer.copy_from_slice(src);
        self.used = true;
    }
}

pub struct Permute<'a, T, F> {
    pub classifier: &'a Classifier<'a, T, F>,
    pub values: &'a mut [T],
    pub bucket_pointers: &'a mut BucketPointers<T>,
}

impl<'a, T: Sortable, F> Permute<'a, T, F>
where
    F: Fn(&T, &T) -> bool,
{
    fn classify_and_read_block<const EQUAL_BUCKETS: bool>(
        &mut self,
        bucket: usize,
        buffer: &mut T::Buffer,
    ) -> Option<usize> {
        self.bucket_pointers[bucket].decrement_read().map(|read| {
            let block = &self.values[read..read + T::BUFFER_SIZE];
            buffer.as_mut().copy_from_slice(block);
            let first_value = &block[0];
            self.classifier.classify::<EQUAL_BUCKETS>(first_value)
        })
    }

    fn swap_block<const EQUAL_BUCKETS: bool>(
        &mut self,
        target_bucket: usize,
        (current_swap, other_swap): (&mut T::Buffer, &mut T::Buffer),
        overflow: &mut Overflow<T>,
        max_offset: usize,
    ) -> Option<usize> {
        let ptr = &mut self.bucket_pointers[target_bucket];
        loop {
            let write = ptr.increment_write();

            match write {
                WriteBlock::Empty(write) => {
                    // Destination block is empty

                    if write >= max_offset {
                        // Out-of-bounds; write to overflow buffer instead
                        overflow.insert(current_swap.as_mut());
                    } else {
                        // Write block
                        let block = &mut self.values[write..write + T::BUFFER_SIZE];
                        block.copy_from_slice(current_swap.as_ref());
                    }

                    break None;
                }
                WriteBlock::Occupied(write) => {
                    let block = &mut self.values[write..write + T::BUFFER_SIZE];
                    let new_target = self.classifier.classify::<EQUAL_BUCKETS>(&block[0]);
                    if new_target != target_bucket {
                        other_swap.as_mut().copy_from_slice(block);
                        block.copy_from_slice(current_swap.as_ref());
                        break Some(new_target);
                    }
                }
            }
        }
    }

    pub fn permute_blocks<const EQUAL_BUCKETS: bool>(
        mut self,
        buffers: &mut SwapBuffers<T>,
        overflow: &mut Overflow<T>,
    ) {
        let max_offset = T::align_to_next_block(self.values.len()) - T::BUFFER_SIZE;
        let num_buckets = self.classifier.num_buckets();
        for bucket in 0..num_buckets {
            let mut swap = Swap::<T>::new(buffers);
            while let Some(mut target_bucket) =
                self.classify_and_read_block::<EQUAL_BUCKETS>(bucket, swap.current_mut())
            {
                while let Some(new_target) = self.swap_block::<EQUAL_BUCKETS>(
                    target_bucket,
                    swap.buffers_mut(),
                    overflow,
                    max_offset,
                ) {
                    swap.swap_buffers();
                    target_bucket = new_target;
                }
            }
        }
    }
}
