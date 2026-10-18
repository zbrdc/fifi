//! Deterministic synthetic data. The Python reference uses the same generator.

/// Xorshift32 stream seeded from a tensor seed.
pub struct Stream(u32);

impl Stream {
    pub fn new(seed: u32) -> Self {
        let state = seed.wrapping_mul(2_654_435_761).wrapping_add(1);
        Self(if state == 0 { 1 } else { state })
    }

    pub fn next_u32(&mut self) -> u32 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        self.0 = x;
        x
    }

    /// A value in [-0.5, 0.5) with 24 significant bits.
    pub fn unit(&mut self) -> f32 {
        (self.next_u32() >> 8) as f32 / 16_777_216.0 - 0.5
    }
}

pub fn words_le(seed: u32, count: usize) -> Vec<u8> {
    let mut stream = Stream::new(seed);
    let mut bytes = Vec::with_capacity(count * 4);
    for _ in 0..count {
        bytes.extend_from_slice(&stream.next_u32().to_le_bytes());
    }
    bytes
}

fn floats_le(seed: u32, count: usize, map: impl Fn(f32) -> f32) -> Vec<u8> {
    let mut stream = Stream::new(seed);
    let mut bytes = Vec::with_capacity(count * 4);
    for _ in 0..count {
        bytes.extend_from_slice(&map(stream.unit()).to_le_bytes());
    }
    bytes
}

/// Activation values: `2 * unit`.
pub fn activation_le(seed: u32, count: usize) -> Vec<u8> {
    floats_le(seed, count, |u| u * 2.0)
}

/// Dense matrix values: `unit`.
pub fn dense_le(seed: u32, count: usize) -> Vec<u8> {
    floats_le(seed, count, |u| u)
}

/// Norm weights: `1 + 0.25 * unit`.
pub fn norm_le(seed: u32, count: usize) -> Vec<u8> {
    floats_le(seed, count, |u| 1.0 + 0.25 * u)
}

/// Quantization scales: `2^-6 * (1 + unit)`.
pub fn scales_le(seed: u32, count: usize) -> Vec<u8> {
    floats_le(seed, count, |u| 0.015_625 * (1.0 + u))
}

/// Quantization biases: `-7.5 * scale` for the same seed as the scales.
pub fn biases_le(seed: u32, count: usize) -> Vec<u8> {
    floats_le(seed, count, |u| -7.5 * (0.015_625 * (1.0 + u)))
}

#[cfg(test)]
mod tests {
    use super::{Stream, words_le};

    #[test]
    fn zero_seed_uses_state_one() {
        assert_eq!(Stream::new(0).0, 1);
    }

    #[test]
    fn seed_one_matches_reference_words() {
        let mut stream = Stream::new(1);
        let actual: Vec<u32> = (0..4).map(|_| stream.next_u32()).collect();

        assert_eq!(
            actual,
            vec![0x5121_2772, 0xCC24_EF35, 0x698E_9D54, 0xFEAC_4886]
        );
    }

    #[test]
    fn maximum_seed_matches_reference_words() {
        let mut stream = Stream::new(u32::MAX);
        let actual: Vec<u32> = (0..4).map(|_| stream.next_u32()).collect();

        assert_eq!(
            actual,
            vec![0x5155_64F1, 0x44AC_7894, 0xBC6D_342B, 0x47E3_72BF]
        );
    }

    #[test]
    fn unit_values_stay_in_half_open_interval() {
        let mut stream = Stream::new(u32::MAX);
        for _ in 0..4096 {
            assert!((-0.5..0.5).contains(&stream.unit()));
        }
    }

    #[test]
    fn words_le_encodes_first_word_little_endian() {
        assert_eq!(words_le(1, 1), vec![0x72, 0x27, 0x21, 0x51]);
    }
}

#[cfg(test)]
mod float_reference_tests {
    use super::{activation_le, biases_le, dense_le, norm_le, scales_le};

    #[test]
    fn activation_matches_reference_bytes() {
        assert_eq!(
            activation_le(1, 4),
            vec![
                0x64, 0x7b, 0xbb, 0xbe, 0xde, 0x49, 0x18, 0x3f, 0x18, 0x8b, 0x33, 0xbe, 0x90, 0x58,
                0x7d, 0x3f
            ]
        );
    }

    #[test]
    fn dense_matches_reference_bytes() {
        assert_eq!(
            dense_le(1, 4),
            vec![
                0x64, 0x7b, 0x3b, 0xbe, 0xde, 0x49, 0x98, 0x3e, 0x18, 0x8b, 0xb3, 0xbd, 0x90, 0x58,
                0xfd, 0x3e
            ]
        );
    }

    #[test]
    fn norm_matches_reference_bytes() {
        assert_eq!(
            norm_le(1, 4),
            vec![
                0x4a, 0x48, 0x74, 0x3f, 0x9e, 0x84, 0x89, 0x3f, 0xa7, 0x63, 0x7a, 0x3f, 0x89, 0xd5,
                0x8f, 0x3f
            ]
        );
    }

    #[test]
    fn scales_match_reference_bytes() {
        assert_eq!(
            scales_le(1, 4),
            vec![
                0x27, 0x21, 0x51, 0x3c, 0x78, 0x12, 0xa6, 0x3c, 0x9d, 0x8e, 0x69, 0x3c, 0x24, 0x56,
                0xbf, 0x3c
            ]
        );
    }

    #[test]
    fn biases_match_reference_bytes() {
        assert_eq!(
            biases_le(1, 4),
            vec![
                0x15, 0x0f, 0xc4, 0xbd, 0x50, 0xb1, 0x1b, 0xbe, 0xb3, 0xf5, 0xda, 0xbd, 0xc2, 0x60,
                0x33, 0xbe
            ]
        );
    }
}
