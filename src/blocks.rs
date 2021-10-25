use crate::{Sortable, MAX_BUCKETS};

pub const BLOCK_SIZE_BYTES: usize = 1024;

pub struct Buffer<'a, T: Sortable> {
    storage: &'a mut T::Buffer,
    index: &'a mut usize,
}

impl<'a, T: Sortable> Buffer<'a, T> {
    #[inline]
    pub fn push(&mut self, value: T) {
        debug_assert_ne!(*self.index, T::BUFFER_SIZE);
        unsafe { *self.storage.as_mut().get_unchecked_mut(*self.index) = value };
        *self.index += 1;
    }

    pub fn len(&self) -> usize {
        *self.index
    }

    #[inline]
    pub fn is_full(&self) -> bool {
        *self.index == T::BUFFER_SIZE
    }

    #[inline]
    pub fn take(&mut self) -> &[T] {
        let result = &self.storage.as_ref()[..*self.index];
        *self.index = 0;
        result
    }

    #[inline]
    pub fn as_slice(&self) -> &[T] {
        self.storage.as_ref()
    }
}

pub struct Buffers<T: Sortable> {
    storage: Vec<T::Buffer>,
    indices: [usize; MAX_BUCKETS],
}

impl<T: Sortable> Default for Buffers<T> {
    fn default() -> Self {
        // TODO 2?
        Self {
            storage: vec![T::Buffer::default(); 2 * MAX_BUCKETS],
            indices: [0; MAX_BUCKETS],
        }
    }
}

impl<T: Sortable> Buffers<T> {
    #[inline]
    pub fn bucket_mut(&mut self, bucket: usize) -> Buffer<T> {
        Buffer {
            storage: &mut self.storage[bucket],
            index: &mut self.indices[bucket],
        }
    }

    pub fn reset(&mut self) {
        self.indices.fill(0);
    }
}
