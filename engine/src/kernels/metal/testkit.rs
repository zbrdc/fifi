//! Deterministic test data and the shared fixtures of the Metal kernels' device tests: the
//! generator of the CUDA testkit, a context, and the padded buffers with their sentinels.

use crate::kernels::metal::{DeviceBuffer, DeviceRange, MetalContext};

/// The sentinel bytes around the used bytes of every buffer of a device test.
const PAD: usize = 1024;
/// A NaN with a payload: an element a call must not write keeps it, and no arithmetic of a
/// kernel turns it into another pattern by accident.
pub(crate) const SENT_BITS: u32 = 0x7fc0_dead;
const SENT: [u8; 4] = SENT_BITS.to_le_bytes();

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

/// A Metal context for a test. The process has two adapter leases, so the tests that hold one
/// run one after the other (`--test-threads=1`).
pub(crate) struct Rig {
    pub(crate) ctx: MetalContext,
}

impl Rig {
    pub(crate) fn new() -> Self {
        Self {
            ctx: MetalContext::new().expect("a Metal context"),
        }
    }
}

/// `bytes` bytes of the sentinel pattern.
pub(crate) fn sentinel(bytes: usize) -> Vec<u8> {
    SENT.iter().copied().cycle().take(bytes).collect()
}

/// The little-endian bytes of f32 values.
pub(crate) fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// The f32 values of little-endian bytes; a partial float at the end is left out.
pub(crate) fn floats(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect()
}

/// Device bytes between two sentinels; `extra` shifts the first data byte past the allocation
/// start to test alignments.
pub(crate) struct Padded {
    buf: DeviceBuffer,
    lead: usize,
    len: usize,
}

impl Padded {
    pub(crate) fn new(ctx: &MetalContext, data: &[u8], extra: usize) -> Self {
        let lead = PAD + extra;
        let mut bytes = sentinel(lead);
        bytes.extend_from_slice(data);
        bytes.extend_from_slice(&sentinel(PAD));
        Self {
            buf: ctx.upload(&bytes).expect("an upload"),
            lead,
            len: data.len(),
        }
    }

    pub(crate) fn range(&self, skip: usize, bytes: usize) -> DeviceRange<'_> {
        self.buf.range(self.lead + skip, bytes).expect("a range")
    }

    pub(crate) fn all(&self) -> DeviceRange<'_> {
        self.range(0, self.len)
    }

    /// The data bytes; the sentinels around them must be intact.
    pub(crate) fn read(&self, what: &str) -> Vec<u8> {
        let all = self
            .buf
            .read(0, self.lead + self.len + PAD)
            .expect("a read");
        let (lead, rest) = all.split_at(self.lead);
        let (data, trail) = rest.split_at(self.len);
        assert!(
            lead == sentinel(self.lead),
            "{what}: the sentinel before the used bytes changed"
        );
        assert!(
            trail == sentinel(PAD),
            "{what}: the sentinel after the used bytes changed"
        );
        data.to_vec()
    }
}
