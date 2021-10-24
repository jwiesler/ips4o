use crate::{Sortable, MAX_BUCKETS};
use std::marker::PhantomData;

#[derive(Debug, Default, Copy, Clone)]
pub struct BucketPointer<T> {
    // points to the block after the next block to read
    read: usize,
    // points to the next block to write to
    write: usize,

    _marker: PhantomData<T>,
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum WriteBlock {
    Occupied(usize),
    Empty(usize),
}

impl<T: Sortable> BucketPointer<T> {
    #[inline]
    pub fn new(start: usize, stop: usize, first_empty_position: usize) -> Self {
        // No buffer overflow with align_to_next_block on the last bucket end
        // since we write to the overflow buffer in the permutation instead
        let start = T::align_to_next_block(start);
        let stop = T::align_to_next_block(stop);

        // TODO why?
        let read = if first_empty_position <= start {
            start
        } else if stop <= first_empty_position {
            stop
        } else {
            first_empty_position
        };

        BucketPointer {
            read,
            write: start,
            _marker: PhantomData,
        }
    }

    #[inline]
    pub fn increment_write(&mut self) -> WriteBlock {
        let write = self.write;
        self.write += T::BUFFER_SIZE;
        if self.read <= write {
            WriteBlock::Empty(write)
        } else {
            WriteBlock::Occupied(write)
        }
    }

    pub fn write(&self) -> usize {
        self.write
    }

    #[inline]
    pub fn decrement_read(&mut self) -> Option<usize> {
        if self.read <= self.write {
            None
        } else {
            self.read -= T::BUFFER_SIZE;
            Some(self.read)
        }
    }
}

pub type BucketPointers<T> = [BucketPointer<T>; MAX_BUCKETS];

#[cfg(test)]
mod test {
    use crate::bucket_pointers::{BucketPointer, WriteBlock};
    use crate::Sortable;

    #[derive(Debug, Default, Copy, Clone, Eq, PartialEq)]
    struct BufferSizeMock {}

    impl Sortable for BufferSizeMock {
        const BUFFER_SIZE: usize = 4;
        type Buffer = [BufferSizeMock; 1];
    }

    #[test]
    fn test_pointers() {
        let mut ptr = BucketPointer::<BufferSizeMock>::new(1, 10, 100);
        assert_eq!(
            ptr.read,
            12,
            "read pointer of {:?} with first empty position {}",
            1..10,
            100
        );
        assert_eq!(
            ptr.write,
            4,
            "write pointer of {:?} with first empty position {}",
            1..10,
            100
        );

        assert_eq!(ptr.increment_write(), WriteBlock::Occupied(4));
        assert_eq!(ptr.decrement_read(), Some(8));
        assert_eq!(ptr.increment_write(), WriteBlock::Empty(8));
        assert_eq!(ptr.decrement_read(), None);
    }
}
