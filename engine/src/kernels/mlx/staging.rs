use std::ffi::c_void;

use crate::EngError;

/// Host bytes copied into storage that every native element type can read in place.
///
/// `mlx_array_set_data` reads its source through typed pointers such as `float*`, `uint32_t*`, and
/// `bfloat16_t*`, so the source must have the natural alignment of the element type. A `Vec<u8>`
/// guarantees an alignment of 1 and relies on the allocator for more. A `Vec<u64>` allocation is
/// aligned for `u64` by construction. That covers every dtype the adapter stages, including `I64`.
///
/// The words hold the source bytes in memory order, so the pointer addresses exactly the bytes
/// that were copied. Zero-length input still gets one aligned word, so the pointer never dangles.
pub(crate) struct StagedBytes {
    words: Vec<u64>,
}

impl StagedBytes {
    pub(super) fn copy_of(bytes: &[u8]) -> Result<Self, EngError> {
        let mut words = reserve(bytes.len())?;
        for chunk in bytes.chunks(size_of::<u64>()) {
            let mut word = [0u8; size_of::<u64>()];
            word[..chunk.len()].copy_from_slice(chunk);
            words.push(u64::from_ne_bytes(word));
        }
        Ok(Self::finish(words))
    }

    /// Copies the values in memory order, as `copy_of` would copy their native-endian bytes.
    pub(super) fn from_f32(values: &[f32]) -> Result<Self, EngError> {
        let length = values
            .len()
            .checked_mul(size_of::<f32>())
            .ok_or_else(|| EngError::BadRequest("native staging size overflows".into()))?;
        let mut words = reserve(length)?;
        for pair in values.chunks(2) {
            let mut word = [0u8; size_of::<u64>()];
            let (low, high) = word.split_at_mut(size_of::<f32>());
            low.copy_from_slice(&pair[0].to_ne_bytes());
            if let Some(second) = pair.get(1) {
                high.copy_from_slice(&second.to_ne_bytes());
            }
            words.push(u64::from_ne_bytes(word));
        }
        Ok(Self::finish(words))
    }

    /// The address of the first staged byte, aligned for `u64` and valid while `self` lives.
    pub(super) fn as_ptr(&self) -> *const c_void {
        self.words.as_ptr().cast()
    }

    fn finish(mut words: Vec<u64>) -> Self {
        if words.is_empty() {
            words.push(0);
        }
        Self { words }
    }

    /// The first `len` staged bytes, for tests that compare contents.
    #[cfg(test)]
    fn prefix(&self, len: usize) -> Vec<u8> {
        self.words
            .iter()
            .flat_map(|word| word.to_ne_bytes())
            .take(len)
            .collect()
    }
}

/// Reserves the exact word count for `length` bytes, or reports a memory-plan failure.
fn reserve(length: usize) -> Result<Vec<u64>, EngError> {
    let count = length
        .checked_add(size_of::<u64>() - 1)
        .ok_or_else(|| EngError::BadRequest("native staging size overflows".into()))?
        / size_of::<u64>();
    let mut words = Vec::new();
    words
        .try_reserve_exact(count.max(1))
        .map_err(|_| EngError::MemoryPlan("native staging allocation failed".into()))?;
    Ok(words)
}

#[cfg(test)]
mod tests {
    use super::StagedBytes;
    use crate::EngError;

    fn assert_aligned(staged: &StagedBytes) {
        assert!(!staged.as_ptr().is_null());
        assert!(staged.as_ptr().cast::<u64>().is_aligned());
    }

    #[test]
    fn copies_every_length_from_unaligned_sources_byte_for_byte() {
        let source: Vec<u8> = (0..40u8).collect();
        for offset in 0..8 {
            for length in 0..=17 {
                let slice = &source[offset..offset + length];
                let staged = StagedBytes::copy_of(slice).unwrap();
                assert_aligned(&staged);
                assert_eq!(
                    staged.prefix(length),
                    slice,
                    "offset {offset}, length {length}"
                );
            }
        }
    }

    #[test]
    fn empty_input_still_yields_an_aligned_pointer() {
        assert_aligned(&StagedBytes::copy_of(&[]).unwrap());
        assert_aligned(&StagedBytes::from_f32(&[]).unwrap());
    }

    #[test]
    fn float_staging_matches_the_native_endian_bytes() {
        for count in 0..=5 {
            let values: Vec<f32> = (0..count).map(|index| index as f32 - 1.5).collect();
            let expected: Vec<u8> = values
                .iter()
                .flat_map(|value| value.to_ne_bytes())
                .collect();
            let staged = StagedBytes::from_f32(&values).unwrap();
            assert_aligned(&staged);
            assert_eq!(staged.prefix(expected.len()), expected, "{count} values");
        }
    }

    #[test]
    fn an_impossible_reservation_reports_an_error_without_aborting() {
        // The word count passes the size check but exceeds the allocation limit.
        assert!(matches!(
            super::reserve(usize::MAX / 2),
            Err(EngError::MemoryPlan(_))
        ));
        assert!(matches!(
            super::reserve(usize::MAX),
            Err(EngError::BadRequest(_))
        ));
    }
}
