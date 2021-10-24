#[derive(Debug, Copy, Clone)]
pub struct Buffer<T, const N: usize>([T; N])
where
    [T; N]: Sized;

impl<T: Default + Copy, const N: usize> Default for Buffer<T, N>
where
    [T; N]: Sized,
{
    #[inline(always)]
    fn default() -> Self {
        Self([T::default(); N])
    }
}

impl<T, const N: usize> AsRef<[T]> for Buffer<T, N>
where
    [T; N]: Sized,
{
    #[inline(always)]
    fn as_ref(&self) -> &[T] {
        self.0.as_ref()
    }
}

impl<T, const N: usize> AsMut<[T]> for Buffer<T, N>
where
    [T; N]: Sized,
{
    #[inline(always)]
    fn as_mut(&mut self) -> &mut [T] {
        self.0.as_mut()
    }
}

/// This would be doable with more const_evaluable support without a macro
/// Probably only performant for types with `BLOCK_SIZE_BYTES % size_of::<T> == 0`
#[macro_export]
macro_rules! define_sortable {
    ($t:ty) => {
        impl $crate::Sortable for $t {
            const BUFFER_SIZE: usize = $crate::BLOCK_SIZE_BYTES / std::mem::size_of::<$t>();
            type Buffer = $crate::Buffer<$t, { Self::BUFFER_SIZE }>;
        }
    };
}

define_sortable!(u32);
define_sortable!(u64);
define_sortable!(u16);
define_sortable!(u8);
define_sortable!(i32);
define_sortable!(i64);
define_sortable!(i16);
define_sortable!(i8);
