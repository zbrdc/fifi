//! Sets of device buffers a caller keeps between calls of the same shape.
//!
//! Returning a device buffer to the allocator synchronizes the device whenever more than one
//! stream is live (see `context::alloc_cache`), and a fresh allocation queries free memory first.
//! A caller that runs the same shapes call after call keeps its finished sets here instead and
//! takes back the one whose sizes match. The caller owns the ordering: a set goes back only after
//! every stream that used it has been drained.

use std::cell::RefCell;

use super::{CudaContext, DeviceBuffer};
use crate::EngError;

/// At most `cap` sets, oldest dropped first.
#[derive(Debug)]
pub(crate) struct SetPool<S> {
    sets: RefCell<Vec<S>>,
    cap: usize,
}

impl<S> SetPool<S> {
    pub(crate) fn new(cap: usize) -> Self {
        Self {
            sets: RefCell::new(Vec::with_capacity(cap)),
            cap,
        }
    }

    /// Removes and returns the newest set that `fits` accepts.
    pub(crate) fn take(&self, fits: impl Fn(&S) -> bool) -> Option<S> {
        let mut sets = self.sets.borrow_mut();
        let at = sets.iter().rposition(fits)?;
        Some(sets.remove(at))
    }

    /// Keeps `set`, dropping the oldest one when the pool is full (a zero cap keeps nothing).
    pub(crate) fn put(&self, set: S) {
        let mut sets = self.sets.borrow_mut();
        if sets.len() >= self.cap {
            if self.cap == 0 {
                return;
            }
            sets.remove(0);
        }
        sets.push(set);
    }
}

impl<const N: usize> SetPool<[DeviceBuffer; N]> {
    /// The set of buffers of exactly `sizes` bytes: a kept one, or fresh ones when the device
    /// keeps `headroom` bytes free around them. `None` when it does not, or when the allocator
    /// refuses the plan.
    pub(crate) fn take_or_allocate(
        &self,
        context: &CudaContext,
        sizes: [usize; N],
        headroom: u64,
    ) -> Result<Option<[DeviceBuffer; N]>, EngError> {
        let fits = |set: &[DeviceBuffer; N]| set.iter().zip(sizes).all(|(b, n)| b.bytes() == n);
        if let Some(set) = self.take(fits) {
            return Ok(Some(set));
        }
        let bytes = sizes.iter().sum::<usize>() as u64;
        if (context.memory_info()?.free as u64).saturating_sub(bytes) < headroom {
            return Ok(None);
        }
        let mut set = Vec::with_capacity(N);
        for n in sizes {
            match context.allocate(n) {
                Ok(buffer) => set.push(buffer),
                Err(EngError::MemoryPlan(_)) => return Ok(None),
                Err(error) => return Err(error),
            }
        }
        if (context.memory_info()?.free as u64) < headroom {
            return Ok(None);
        }
        Ok(set.try_into().ok())
    }
}

#[cfg(test)]
mod tests {
    use super::SetPool;

    #[test]
    fn a_set_comes_back_by_its_sizes_and_the_oldest_leaves_a_full_pool() {
        let pool = SetPool::new(2);
        pool.put([4, 8]);
        pool.put([16, 8]);
        assert_eq!(pool.take(|s| s[0] == 16), Some([16, 8]));
        assert_eq!(pool.take(|s| s[0] == 16), None, "taken sets leave the pool");
        pool.put([16, 8]);
        pool.put([32, 8]);
        assert_eq!(
            pool.take(|s| s[0] == 4),
            None,
            "the oldest set left the full pool"
        );
        assert_eq!(
            pool.take(|s| s[1] == 8),
            Some([32, 8]),
            "the newest match first"
        );
        let none = SetPool::new(0);
        none.put([1]);
        assert_eq!(none.take(|_| true), None);
    }
}
