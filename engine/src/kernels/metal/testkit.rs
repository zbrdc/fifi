//! Deterministic test data of the Metal kernels, the generator of the CUDA testkit.

/// Deterministic xorshift64* generator for the test data of the kernels. The kernel test
/// modules draw from it instead of keeping local copies; one seed always gives one sequence, so
/// a failing case replays.
pub(crate) struct Rng(u64);

impl Rng {
    /// The all-zero state is a fixed point of xorshift, so seed 0 maps to a fixed non-zero
    /// state.
    pub(crate) fn new(seed: u64) -> Self {
        Self(if seed == 0 {
            0x9e37_79b9_7f4a_7c15
        } else {
            seed
        })
    }

    fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }

    /// A uniform value in [0, 1): the top 53 bits of `next_u64` over 2^53. Every result is an
    /// exact f64 and 1.0 is never drawn.
    fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// An approximately standard normal value: the sum of 12 `unit` draws minus 6 (mean 0,
    /// variance 1, support [-6, 6]). The Irwin-Hall sum is enough for test inputs and keeps the
    /// sequence a pure function of the seed.
    pub(crate) fn normal(&mut self) -> f64 {
        (0..12).map(|_| self.unit()).sum::<f64>() - 6.0
    }
}
