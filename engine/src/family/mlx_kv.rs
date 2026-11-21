//! Capacity planning for MLX attention buffers with a separate visible prefix.

/// Keep short sessions small and bound the cumulative copies while growing.
/// The final allocation can end at an unaligned context limit.
pub(super) fn compact_capacity(visible: u32, limit: u32) -> u32 {
    if visible == 0 {
        return 0;
    }
    visible
        .max(256)
        .checked_next_power_of_two()
        .unwrap_or(u32::MAX)
        .min(limit)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compact_capacity_stays_empty_then_doubles_up_to_the_limit() {
        for (visible, expected) in [
            (0, 0),
            (1, 256),
            (256, 256),
            (257, 512),
            (512, 512),
            (513, 1024),
            (4096, 4096),
            (4097, 8192),
            (8193, 10_000),
        ] {
            assert_eq!(compact_capacity(visible, 10_000), expected);
        }
        assert_eq!(compact_capacity(1, 128), 128);
    }
}
