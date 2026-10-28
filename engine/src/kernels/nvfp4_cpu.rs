#![allow(unsafe_code)]
//! CPU kernel for the routed experts of a NVFP4 mixture-of-experts layer (Nemotron 3.5 Super).
//!
//! WHAT: [`ExpertPool::routed_experts`] computes, for one latent activation row, the weighted
//! sum of the selected experts' feed-forward blocks, `sum_e w_e * down_e(relu2(up_e(x)))`, with
//! both projections read straight from NVFP4 bytes (`[rows, cols/2]` packed E2M1 nibbles, one
//! E4M3FN scale byte per 16 columns, one F32 per-tensor multiplier). [`ExpertPool::routed_experts_batch`]
//! does the same for 2..=8 rows and reads an expert shared by several rows once. The weights are
//! borrowed `&[u8]` views (a read-only mmap of the checkpoint): nothing is copied or locked in memory.
//!
//! WHY a CPU kernel: the 31 CPU-resident MoE layers stream about 68 MB each per decoded token, so
//! the work is bound by memory bandwidth, not arithmetic. The kernel therefore spends its effort
//! on keeping every core's load stream sequential and busy, and on dequantizing cheaply:
//! an E2M1 code indexes a 16-entry f32 register table with one `vpermps`, the E4M3 group scale
//! becomes an f32 through an exact f16 bit trick (no table), and activations stay f32, so the
//! result is as accurate as the engine's host reference (no activation quantization).
//!
//! Numerics: `row = wss2 * sum_g scale_g * (sum_{i in g} e2m1_i * x_i)`. The engine reference rounds
//! each weight and each product separately and sums strictly in column order; this kernel sums
//! 16 lanes per group and 16 lanes across groups, which is at least as accurate (checked
//! against an f64 reference in the tests). Results do not depend on the thread count or on how
//! chunks were scheduled: every output element is produced by exactly one thread in a fixed order.
//! Expert contributions are accumulated in ascending expert index, like the reference.
//!
//! Errors mirror `kernels::nvfp4::WeightBlock::matvec`: non-finite activations, negative or NaN
//! group scales, non-finite weight scales and non-finite projections are refused. Scale bytes are
//! checked inside the streaming pass (one running maximum), so a refusal can arrive after the
//! work was done; `out` is unspecified after any error. The scale-product check is conservative:
//! it refuses a matrix when `6 * max_scale * weight_scale_2` overflows f32, which the reference only
//! does when a nibble of magnitude 6 sits in a group with that maximal scale.
//!
//! Threading: [`ExpertPool`] owns `threads - 1` persistent workers; the calling thread is the last
//! worker, so a call never pays a hand-off for its own share. A call is a single job split into
//! chunks (an expert times a block of up-projection rows, then a block of output rows over all
//! experts) claimed from one atomic counter, so a slow or descheduled thread only delays the
//! chunk it holds. Idle workers wait per [`WaitPolicy`]: spin, park, or spin for a window then park.
//!
//! Unsafe scope: this file holds the x86-64 intrinsics (runtime-detected AVX-512 and AVX2 paths,
//! a portable scalar fallback), the raw-pointer hand-off of one job to the workers, and the aligned
//! scratch buffers. Each `unsafe` block carries its own SAFETY argument. The repo's third scoped
//! unsafe module, confined to this file by the `unsafe_is_confined_to_the_avx2_module` test.
//! The proposed unsafe scope still needs approval. It is not a general policy for unsafe code.
//! Weight views borrow read-only `MappedFile` data. This module maps no files.

use std::ops::Range;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, PoisonError};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// Elements sharing one E4M3FN scale byte.
pub const GROUP_SIZE: usize = 16;
/// Most activation rows one call may carry (speculative verify).
pub const MAX_BATCH_ROWS: usize = 8;
/// Elements per permuted-layout block: two groups.
const BLOCK: usize = 2 * GROUP_SIZE;
/// Largest E4M3FN code that is a finite nonnegative scale (0x7f is NaN, 0x80.. are negative).
const MAX_VALID_SCALE_CODE: u8 = 0x7e;
/// Activation rows one micro-kernel pass handles; a larger batch runs several passes.
const PASS_ROWS: usize = 4;
/// Output rows of the up projection per chunk. A multiple of [`BLOCK`], so chunks write disjoint
/// whole blocks of the permuted intermediate vector (no false sharing).
const UP_CHUNK_ROWS: usize = 32;
/// Output rows of the down projection per chunk.
const DOWN_CHUNK_ROWS: usize = 8;

/// Failure of a kernel call. Variants separate caller mistakes from weight-data refusals.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Nvfp4CpuError {
    /// A dimension, length or row count does not fit the expert geometry.
    Shape(String),
    /// A non-finite activation, a malformed selection or a bad routing weight.
    Input(String),
    /// A weight scale or projection result the reference would refuse.
    Weights(String),
    /// A worker failed (panic inside a chunk) or the pool could not start.
    Pool(String),
}

impl std::fmt::Display for Nvfp4CpuError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let (kind, message) = match self {
            Self::Shape(m) => ("shape", m),
            Self::Input(m) => ("input", m),
            Self::Weights(m) => ("weights", m),
            Self::Pool(m) => ("pool", m),
        };
        write!(f, "NVFP4 CPU experts ({kind}): {message}")
    }
}

impl std::error::Error for Nvfp4CpuError {}

fn shape_err(message: impl Into<String>) -> Nvfp4CpuError {
    Nvfp4CpuError::Shape(message.into())
}

fn input_err(message: impl Into<String>) -> Nvfp4CpuError {
    Nvfp4CpuError::Input(message.into())
}

fn weights_err(message: impl Into<String>) -> Nvfp4CpuError {
    Nvfp4CpuError::Weights(message.into())
}

// ---------------------------------------------------------------------------------------------
// Weight views
// ---------------------------------------------------------------------------------------------

/// One row-major NVFP4 matrix borrowed from the checkpoint mapping: `rows` x `cols`, packed two
/// E2M1 codes per byte (even column in the low nibble), one E4M3FN scale byte per 16 columns.
#[derive(Clone, Copy, Debug)]
pub struct Nvfp4Matrix<'a> {
    packed: &'a [u8],
    scales: &'a [u8],
    weight_scale_2: f32,
    rows: usize,
    cols: usize,
}

impl<'a> Nvfp4Matrix<'a> {
    /// Checks that the byte counts match the shape exactly, as the reference does.
    pub fn new(
        packed: &'a [u8],
        scales: &'a [u8],
        weight_scale_2: f32,
        rows: usize,
        cols: usize,
    ) -> Result<Self, Nvfp4CpuError> {
        if rows == 0 || cols == 0 || !cols.is_multiple_of(GROUP_SIZE) {
            return Err(shape_err(
                "matrix dimensions must be positive and group aligned",
            ));
        }
        let elements = rows
            .checked_mul(cols)
            .ok_or_else(|| shape_err("matrix element count overflows"))?;
        if packed.len() != elements / 2 || scales.len() != elements / GROUP_SIZE {
            return Err(shape_err("stored bytes differ from the matrix shape"));
        }
        if !weight_scale_2.is_finite() || weight_scale_2 <= 0.0 {
            return Err(weights_err("weight_scale_2 must be finite and positive"));
        }
        Ok(Self {
            packed,
            scales,
            weight_scale_2,
            rows,
            cols,
        })
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn cols(&self) -> usize {
        self.cols
    }

    pub fn packed(&self) -> &'a [u8] {
        self.packed
    }

    pub fn scales(&self) -> &'a [u8] {
        self.scales
    }

    pub fn weight_scale_2(&self) -> f32 {
        self.weight_scale_2
    }

    /// Single-threaded `y = W x` with the same contract as `WeightBlock::matvec` of the host
    /// reference (natural-order `x`, finite activations, valid scales). The expert kernels use
    /// the same row dots; this entry point serves tests and small dense uses.
    pub fn matvec(&self, isa: Isa, x: &[f32], y: &mut [f32]) -> Result<(), Nvfp4CpuError> {
        self.matvec_activation(isa, &Activation::new(x)?, y)
    }

    /// [`Self::matvec`] for an activation already validated and permuted, so one row can be
    /// applied to many matrices without repeating that work.
    pub fn matvec_activation(
        &self,
        isa: Isa,
        x: &Activation,
        y: &mut [f32],
    ) -> Result<(), Nvfp4CpuError> {
        if x.len() != self.cols || y.len() != self.rows {
            return Err(shape_err(
                "activation or output length differs from the matrix shape",
            ));
        }
        if !isa.is_supported() {
            return Err(Nvfp4CpuError::Pool(format!(
                "{isa:?} is not supported by this CPU"
            )));
        }
        let mut max_code = 0u8;
        let mut res = [RowResult::default(); UP_CHUNK_ROWS];
        for first in (0..self.rows).step_by(UP_CHUNK_ROWS) {
            let rows = first..(first + UP_CHUNK_ROWS).min(self.rows);
            dot_rows(
                isa,
                self,
                rows.clone(),
                [x.as_slice()],
                &mut res,
                &mut max_code,
            );
            for (offset, row) in rows.enumerate() {
                y[row] = res[offset][0];
            }
        }
        check_scale_extent(max_code, self.weight_scale_2)
    }

    fn row_packed(&self, row: usize) -> &'a [u8] {
        let bytes = self.cols / 2;
        &self.packed[row * bytes..(row + 1) * bytes]
    }

    fn row_scales(&self, row: usize) -> &'a [u8] {
        let groups = self.cols / GROUP_SIZE;
        &self.scales[row * groups..(row + 1) * groups]
    }
}

/// The two projections of one routed expert: `up` is `[inter, latent]`, `down` is `[latent, inter]`.
#[derive(Clone, Copy, Debug)]
pub struct ExpertView<'a> {
    pub up: Nvfp4Matrix<'a>,
    pub down: Nvfp4Matrix<'a>,
}

/// All routed experts of one MoE layer, indexed by expert id, with the geometry validated once.
pub struct LayerExperts<'a> {
    views: Vec<ExpertView<'a>>,
    latent: usize,
    inter: usize,
}

impl<'a> LayerExperts<'a> {
    pub fn new(views: Vec<ExpertView<'a>>) -> Result<Self, Nvfp4CpuError> {
        let first = views
            .first()
            .ok_or_else(|| shape_err("a layer needs experts"))?;
        let (inter, latent) = (first.up.rows, first.up.cols);
        for view in &views {
            if view.up.rows != inter
                || view.up.cols != latent
                || view.down.rows != latent
                || view.down.cols != inter
            {
                return Err(shape_err("experts disagree on [inter, latent] geometry"));
            }
        }
        Ok(Self {
            views,
            latent,
            inter,
        })
    }

    pub fn len(&self) -> usize {
        self.views.len()
    }

    /// The projections of expert `index` (panics when out of range, like slice indexing).
    pub fn view(&self, index: usize) -> ExpertView<'a> {
        self.views[index]
    }

    pub fn is_empty(&self) -> bool {
        self.views.is_empty()
    }

    /// Width of the activation row and of the output row.
    pub fn latent(&self) -> usize {
        self.latent
    }

    /// Width of the expert intermediate vector.
    pub fn inter(&self) -> usize {
        self.inter
    }
}

/// One routed expert chosen for a row, with the router weight already scaled and normalized.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ExpertPick {
    pub expert: u32,
    pub weight: f32,
}

/// An activation row validated (finite) and stored in the kernels' permuted, cache-line-aligned layout.
pub struct Activation {
    data: AlignedBuf,
}

impl Activation {
    pub fn new(x: &[f32]) -> Result<Self, Nvfp4CpuError> {
        if x.is_empty() || !x.len().is_multiple_of(GROUP_SIZE) {
            return Err(shape_err(
                "activation length must be a positive multiple of 16",
            ));
        }
        if x.iter().any(|v| !v.is_finite()) {
            return Err(input_err("activation values must be finite"));
        }
        let mut data = AlignedBuf::new();
        data.ensure(x.len());
        permute_row(x, data.as_mut_slice());
        Ok(Self { data })
    }

    pub fn len(&self) -> usize {
        self.data.len
    }

    pub fn is_empty(&self) -> bool {
        self.data.len == 0
    }

    /// The permuted elements (see the layout note above `perm_index`).
    pub fn as_slice(&self) -> &[f32] {
        self.data.as_slice()
    }
}

// ---------------------------------------------------------------------------------------------
// Instruction-set selection
// ---------------------------------------------------------------------------------------------

/// Which implementation of the row dot runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Isa {
    /// Portable Rust, the reference for the vector paths.
    Scalar,
    /// 256-bit AVX2 with FMA and F16C.
    Avx2,
    /// 512-bit AVX-512 (F, BW, VL) with F16C.
    Avx512,
}

impl Isa {
    /// The best implementation this CPU supports.
    pub fn detect() -> Self {
        [Self::Avx512, Self::Avx2]
            .into_iter()
            .find(|isa| isa.is_supported())
            .unwrap_or(Self::Scalar)
    }

    pub fn is_supported(self) -> bool {
        match self {
            Self::Scalar => true,
            #[cfg(target_arch = "x86_64")]
            Self::Avx2 => {
                std::is_x86_feature_detected!("avx2")
                    && std::is_x86_feature_detected!("fma")
                    && std::is_x86_feature_detected!("f16c")
            }
            #[cfg(target_arch = "x86_64")]
            Self::Avx512 => {
                std::is_x86_feature_detected!("avx512f")
                    && std::is_x86_feature_detected!("avx512bw")
                    && std::is_x86_feature_detected!("avx512vl")
                    && std::is_x86_feature_detected!("f16c")
            }
            #[cfg(not(target_arch = "x86_64"))]
            _ => false,
        }
    }
}

// ---------------------------------------------------------------------------------------------
// E4M3FN and E2M1 tables (scalar path and scale validation)
// ---------------------------------------------------------------------------------------------

/// E2M1 magnitudes for codes 0..8; bit 3 of a code is the sign.
const E2M1: [f32; 16] = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
];

/// E4M3FN values by bit pattern, built as `kernels::nvfp4::E4M3_VALUES` is.
const E4M3: [f32; 256] = e4m3_table();

const fn e4m3_table() -> [f32; 256] {
    let mut values = [0.0; 256];
    let mut code = 0u32;
    while code < 256 {
        let exponent = (code >> 3) & 15;
        let fraction = code & 7;
        let magnitude = if exponent == 0 {
            fraction as f32 / 512.0
        } else if exponent == 15 && fraction == 7 {
            f32::NAN
        } else {
            f32::from_bits(((exponent + 120) << 23) | (fraction << 20))
        };
        values[code as usize] = if code & 128 == 0 {
            magnitude
        } else {
            -magnitude
        };
        code += 1;
    }
    values
}

/// Refuses a matrix whose largest scale code is invalid or overflows once multiplied out.
fn check_scale_extent(max_code: u8, weight_scale_2: f32) -> Result<(), Nvfp4CpuError> {
    if max_code > MAX_VALID_SCALE_CODE {
        return Err(weights_err("weight_scale must be finite and nonnegative"));
    }
    let scale = E4M3[usize::from(max_code)] * weight_scale_2;
    if !scale.is_finite() || !(6.0 * scale).is_finite() {
        return Err(weights_err(
            "weight_scale product or decoded weight is not finite",
        ));
    }
    Ok(())
}

// ---------------------------------------------------------------------------------------------
// Permuted activation layout
// ---------------------------------------------------------------------------------------------
//
// A vector of length k (a multiple of 16) is stored in blocks of 32: the 16 even-index elements
// of the block, then its 16 odd-index elements. A weight byte holds elements (2i, 2i+1), so one
// 16-byte load yields 16 even codes and 16 odd codes whose lanes line up with those halves. A
// trailing group of 16 (k % 32 == 16) is stored as 8 even then 8 odd elements.

/// Position of natural element `j` of a `k`-vector in the permuted layout.
fn perm_index(k: usize, j: usize) -> usize {
    let start = j & !(BLOCK - 1);
    let half = if k - start >= BLOCK {
        GROUP_SIZE
    } else {
        GROUP_SIZE / 2
    };
    let p = j - start;
    start + if p & 1 == 0 { p >> 1 } else { half + (p >> 1) }
}

fn permute_row(src: &[f32], dst: &mut [f32]) {
    let k = src.len();
    for (j, &v) in src.iter().enumerate() {
        dst[perm_index(k, j)] = v;
    }
}

// ---------------------------------------------------------------------------------------------
// Row dots: dot(row, x_0..x_NX) for a range of matrix rows, per instruction set
// ---------------------------------------------------------------------------------------------

/// One result row: lane `n` is the dot with activation vector `n` of the pass.
type RowResult = [f32; PASS_ROWS];

/// Dots `rows` of `m` against `NX` permuted activation vectors; `res[i][n]` is row `rows.start+i`
/// dotted with vector `n`, scaled by every group scale and the weight multiplier. Folds the largest
/// scale code it read into `max_code`. `res` may be longer than `rows`.
fn dot_rows<const NX: usize>(
    isa: Isa,
    m: &Nvfp4Matrix<'_>,
    rows: Range<usize>,
    xs: [&[f32]; NX],
    res: &mut [RowResult],
    max_code: &mut u8,
) {
    assert!(NX >= 1 && NX <= PASS_ROWS && rows.end <= m.rows && res.len() >= rows.len());
    assert!(xs.iter().all(|x| x.len() == m.cols));
    match isa {
        #[cfg(target_arch = "x86_64")]
        Isa::Avx512 => {
            // SAFETY: Avx512 only reaches here after `Isa::is_supported` held at pool construction.
            unsafe { avx512::dot_rows(m, rows, xs, res, max_code) }
        }
        #[cfg(target_arch = "x86_64")]
        Isa::Avx2 => {
            // SAFETY: Avx2 only reaches here after `Isa::is_supported` held at pool construction.
            unsafe { avx2::dot_rows(m, rows, xs, res, max_code) }
        }
        _ => scalar::dot_rows(m, rows, xs, res, max_code),
    }
}

/// Runtime-arity front end: `xs.len()` (1..=[`PASS_ROWS`]) selects the const-generic kernel.
fn dot_rows_n(
    isa: Isa,
    m: &Nvfp4Matrix<'_>,
    rows: Range<usize>,
    xs: &[&[f32]],
    res: &mut [RowResult],
    max_code: &mut u8,
) {
    match *xs {
        [a] => dot_rows(isa, m, rows, [a], res, max_code),
        [a, b] => dot_rows(isa, m, rows, [a, b], res, max_code),
        [a, b, c] => dot_rows(isa, m, rows, [a, b, c], res, max_code),
        [a, b, c, d] => dot_rows(isa, m, rows, [a, b, c, d], res, max_code),
        _ => unreachable!("pass of {} activation rows", xs.len()),
    }
}

mod scalar {
    use super::*;

    /// Portable row dot, group by group: 16 products summed in f32, then one scale multiply.
    pub(super) fn dot_rows<const NX: usize>(
        m: &Nvfp4Matrix<'_>,
        rows: Range<usize>,
        xs: [&[f32]; NX],
        res: &mut [RowResult],
        max_code: &mut u8,
    ) {
        let k = m.cols;
        let factor = m.weight_scale_2;
        for (out, row) in res.iter_mut().zip(rows) {
            let packed = m.row_packed(row);
            let scales = m.row_scales(row);
            let mut acc = [0.0f32; NX];
            for (g, (&code, bytes)) in scales.iter().zip(packed.as_chunks::<8>().0).enumerate() {
                *max_code = (*max_code).max(code);
                let scale = E4M3[usize::from(code)];
                let (even, odd) = group_offsets(k, g);
                for (a, x) in acc.iter_mut().zip(&xs) {
                    let mut sum = 0.0f32;
                    for (j, &byte) in bytes.iter().enumerate() {
                        sum += E2M1[usize::from(byte & 15)] * x[even + j];
                        sum += E2M1[usize::from(byte >> 4)] * x[odd + j];
                    }
                    *a += sum * scale;
                }
            }
            for (o, a) in out.iter_mut().zip(&acc) {
                *o = a * factor;
            }
        }
    }
}

/// Offsets of group `g`'s even-element and odd-element halves (8 floats each) in a permuted
/// `k`-vector.
fn group_offsets(k: usize, g: usize) -> (usize, usize) {
    let start = (g / 2) * BLOCK;
    if k - start < BLOCK {
        (start, start + GROUP_SIZE / 2)
    } else if g.is_multiple_of(2) {
        (start, start + GROUP_SIZE)
    } else {
        (start + GROUP_SIZE / 2, start + GROUP_SIZE + GROUP_SIZE / 2)
    }
}

#[cfg(target_arch = "x86_64")]
mod avx512 {
    use super::*;

    /// Scale codes of 16 groups to f32 divided by 256: the byte `0eeeemmm` placed at bits 7..13 of an
    /// f16 is the E4M3FN value / 256 exactly (bias 15 vs 7 gives 2^-8, subnormals line up too).
    /// Codes above 0x7e are caught by the caller's running maximum, not here.
    #[target_feature(enable = "avx512f,avx512bw,avx512vl,f16c")]
    #[inline]
    fn scales_div256(codes: __m128i) -> __m512 {
        _mm512_cvtph_ps(_mm256_slli_epi16::<7>(_mm256_cvtepu8_epi16(codes)))
    }

    /// `vpermps` index vectors that copy scale lane `2b` into lanes 0..8 and lane `2b+1` into 8..16.
    #[target_feature(enable = "avx512f,avx512bw,avx512vl,f16c")]
    #[inline]
    fn block_selectors() -> [__m512i; 8] {
        let mut out = [_mm512_setzero_si512(); 8];
        for (b, slot) in out.iter_mut().enumerate() {
            let lo = 2 * b as i32;
            let hi = lo + 1;
            *slot = _mm512_setr_epi32(
                lo, lo, lo, lo, lo, lo, lo, lo, hi, hi, hi, hi, hi, hi, hi, hi,
            );
        }
        out
    }

    /// Loads 16 floats at `offset` from each activation vector.
    ///
    /// # Safety
    /// `offset + 16 <= xs[n].len()` for every `n`.
    #[target_feature(enable = "avx512f,avx512bw,avx512vl,f16c")]
    #[inline]
    unsafe fn load_x<const NX: usize>(xs: &[&[f32]; NX], offset: usize) -> [__m512; NX] {
        let mut out = [_mm512_setzero_ps(); NX];
        for (v, x) in out.iter_mut().zip(xs) {
            debug_assert!(offset + 16 <= x.len());
            // SAFETY: the caller guarantees the 16 floats are inside `x`; loadu has no alignment need.
            *v = unsafe { _mm512_loadu_ps(x.as_ptr().add(offset)) };
        }
        out
    }

    /// Accumulates one 32-element block (two groups) of one row into `acc`.
    ///
    /// # Safety
    /// 16 packed bytes are readable at `bytes`, and `blk * BLOCK + 32 <= xs[n].len()`.
    #[target_feature(enable = "avx512f,avx512bw,avx512vl,f16c")]
    #[inline]
    unsafe fn block<const NX: usize>(
        bytes: *const u8,
        xs: &[&[f32]; NX],
        blk: usize,
        scales: __m512,
        tab: __m512,
        acc: &mut [__m512; NX],
    ) {
        // SAFETY: forwarded from the caller's contract.
        let (raw, xe, xo) = unsafe {
            (
                _mm_loadu_si128(bytes as *const __m128i),
                load_x::<NX>(xs, blk * BLOCK),
                load_x::<NX>(xs, blk * BLOCK + GROUP_SIZE),
            )
        };
        let idx = _mm512_cvtepu8_epi32(raw);
        let wl = _mm512_permutexvar_ps(_mm512_and_si512(idx, _mm512_set1_epi32(15)), tab);
        let wh = _mm512_permutexvar_ps(_mm512_srli_epi32::<4>(idx), tab);
        for n in 0..NX {
            let t = _mm512_fmadd_ps(wh, xo[n], _mm512_mul_ps(wl, xe[n]));
            acc[n] = _mm512_fmadd_ps(t, scales, acc[n]);
        }
    }

    /// Row-by-row AVX-512 dot, see [`super::dot_rows`]. Every unsafe load below is bounded by the
    /// row slices (checked by `row_packed`/`row_scales`) and the x length (asserted on entry).
    #[target_feature(enable = "avx512f,avx512bw,avx512vl,f16c")]
    pub(super) fn dot_rows<const NX: usize>(
        m: &Nvfp4Matrix<'_>,
        rows: Range<usize>,
        xs: [&[f32]; NX],
        res: &mut [RowResult],
        max_code: &mut u8,
    ) {
        let k = m.cols;
        assert!(xs.iter().all(|x| x.len() >= k));
        // SAFETY: E2M1 holds exactly 16 floats, the width of the load.
        let tab = unsafe { _mm512_loadu_ps(E2M1.as_ptr()) };
        let sel = block_selectors();
        let mut maxv = _mm_setzero_si128();
        let factor = m.weight_scale_2 * 256.0;
        let groups = k / GROUP_SIZE;
        for (out, row) in res.iter_mut().zip(rows) {
            let packed = m.row_packed(row).as_ptr();
            let scales = m.row_scales(row).as_ptr();
            let mut acc = [_mm512_setzero_ps(); NX];
            let mut g = 0usize;
            // Whole 16-group super-blocks: one scale load converts 16 scales at once.
            while g + 16 <= groups {
                // SAFETY: g + 16 <= groups, so 16 scale bytes at `g` are inside the row.
                let codes = unsafe { _mm_loadu_si128(scales.add(g) as *const __m128i) };
                maxv = _mm_max_epu8(maxv, codes);
                let s16 = scales_div256(codes);
                for (b, &select) in sel.iter().enumerate() {
                    let sv = _mm512_permutexvar_ps(select, s16);
                    // SAFETY: groups g+2b and g+2b+1 are < groups: their 16 packed bytes start at
                    // SAFETY: byte 8*(g+2b) and the block's 32 floats at 32*(g/2+b) are in bounds.
                    unsafe {
                        block::<NX>(
                            packed.add(8 * (g + 2 * b)),
                            &xs,
                            g / 2 + b,
                            sv,
                            tab,
                            &mut acc,
                        )
                    };
                }
                g += 16;
            }
            if g < groups {
                let rest = groups - g;
                // SAFETY: the masked load reads only the `rest` bytes left in the row.
                let codes = unsafe {
                    _mm_maskz_loadu_epi8(
                        ((1u32 << rest) - 1) as __mmask16,
                        scales.add(g) as *const i8,
                    )
                };
                maxv = _mm_max_epu8(maxv, codes);
                let s16 = scales_div256(codes);
                for (b, &select) in sel.iter().enumerate().take(rest / 2) {
                    let sv = _mm512_permutexvar_ps(select, s16);
                    // SAFETY: full block b of the remainder, bounds as in the main loop.
                    unsafe {
                        block::<NX>(
                            packed.add(8 * (g + 2 * b)),
                            &xs,
                            g / 2 + b,
                            sv,
                            tab,
                            &mut acc,
                        )
                    };
                }
                if rest % 2 == 1 {
                    // One trailing group: 8 bytes, even/odd halves share a vector, one scale lane.
                    let last = g + rest - 1;
                    // SAFETY: `last` < groups, so 8 packed bytes at 8*last and the 16 floats at
                    // SAFETY: 16*last (permuted start of the trailing group) are in bounds.
                    let (raw, xt) = unsafe {
                        (
                            _mm_loadl_epi64(packed.add(8 * last) as *const __m128i),
                            load_x::<NX>(&xs, last * GROUP_SIZE),
                        )
                    };
                    let idx8 = _mm256_cvtepu8_epi32(raw);
                    let idx = _mm512_inserti64x4::<1>(
                        _mm512_castsi256_si512(_mm256_and_si256(idx8, _mm256_set1_epi32(15))),
                        _mm256_srli_epi32::<4>(idx8),
                    );
                    let w = _mm512_permutexvar_ps(idx, tab);
                    let sv = _mm512_permutexvar_ps(_mm512_set1_epi32((rest - 1) as i32), s16);
                    for n in 0..NX {
                        acc[n] = _mm512_fmadd_ps(_mm512_mul_ps(w, xt[n]), sv, acc[n]);
                    }
                }
            }
            for n in 0..NX {
                out[n] = _mm512_reduce_add_ps(acc[n]) * factor;
            }
        }
        let mut lanes = [0u8; 16];
        // SAFETY: `lanes` is 16 bytes, the width of the store.
        unsafe { _mm_storeu_si128(lanes.as_mut_ptr() as *mut __m128i, maxv) };
        *max_code = (*max_code).max(lanes.into_iter().max().unwrap_or(0));
    }
}

#[cfg(target_arch = "x86_64")]
mod avx2 {
    use super::*;

    /// E2M1 values of 8 nibble codes: magnitude lookup in both signs, sign bit (code bit 3) picks one.
    #[target_feature(enable = "avx2,fma,f16c")]
    #[inline]
    fn signed_lookup(pos: __m256, neg: __m256, code: __m256i) -> __m256 {
        let sign = _mm256_castsi256_ps(_mm256_slli_epi32::<28>(code));
        _mm256_blendv_ps(
            _mm256_permutevar8x32_ps(pos, code),
            _mm256_permutevar8x32_ps(neg, code),
            sign,
        )
    }

    #[target_feature(enable = "avx2,fma,f16c")]
    #[inline]
    fn hsum(v: __m256) -> f32 {
        let s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps::<1>(v));
        let s = _mm_add_ps(s, _mm_movehl_ps(s, s));
        _mm_cvtss_f32(_mm_add_ss(s, _mm_shuffle_ps::<1>(s, s)))
    }

    /// Row-by-row AVX2 dot, see [`super::dot_rows`]: one 16-element group per step, the 16-entry
    /// code table split into a positive and a negative 8-entry `vpermps` selected by the sign bit.
    #[target_feature(enable = "avx2,fma,f16c")]
    pub(super) fn dot_rows<const NX: usize>(
        m: &Nvfp4Matrix<'_>,
        rows: Range<usize>,
        xs: [&[f32]; NX],
        res: &mut [RowResult],
        max_code: &mut u8,
    ) {
        let k = m.cols;
        assert!(xs.iter().all(|x| x.len() >= k));
        // SAFETY: E2M1 has 16 floats; the load reads the first 8.
        let pos = unsafe { _mm256_loadu_ps(E2M1.as_ptr()) };
        let neg = _mm256_sub_ps(_mm256_setzero_ps(), pos);
        let mask15 = _mm256_set1_epi32(15);
        let mut maxv = _mm_setzero_si128();
        let factor = m.weight_scale_2 * 256.0;
        let groups = k / GROUP_SIZE;
        for (out, row) in res.iter_mut().zip(rows) {
            let packed = m.row_packed(row);
            let scales = m.row_scales(row);
            let mut acc = [_mm256_setzero_ps(); NX];
            for (g0, scale_chunk) in scales.chunks(8).enumerate() {
                let mut raw = [0u8; 8];
                raw[..scale_chunk.len()].copy_from_slice(scale_chunk);
                // SAFETY: `raw` is 8 bytes, the width of the load.
                let codes = unsafe { _mm_loadl_epi64(raw.as_ptr() as *const __m128i) };
                maxv = _mm_max_epu8(maxv, codes);
                let s8 = _mm256_cvtph_ps(_mm_slli_epi16::<7>(_mm_cvtepu8_epi16(codes)));
                for i in 0..scale_chunk.len() {
                    let group = g0 * 8 + i;
                    debug_assert!(group < groups);
                    let bytes: &[u8; 8] = packed[8 * group..8 * group + 8]
                        .try_into()
                        .expect("group slice is 8 bytes");
                    // SAFETY: `bytes` is 8 bytes, the width of the load.
                    let raw = unsafe { _mm_loadl_epi64(bytes.as_ptr() as *const __m128i) };
                    let idx = _mm256_cvtepu8_epi32(raw);
                    let wl = signed_lookup(pos, neg, _mm256_and_si256(idx, mask15));
                    let wh = signed_lookup(pos, neg, _mm256_srli_epi32::<4>(idx));
                    let sv = _mm256_permutevar8x32_ps(s8, _mm256_set1_epi32(i as i32));
                    let (even, odd) = group_offsets(k, group);
                    for n in 0..NX {
                        // SAFETY: even+8 and odd+8 are <= k by the permuted layout; x.len() >= k.
                        let (xe, xo) = unsafe {
                            (
                                _mm256_loadu_ps(xs[n].as_ptr().add(even)),
                                _mm256_loadu_ps(xs[n].as_ptr().add(odd)),
                            )
                        };
                        let t = _mm256_fmadd_ps(wh, xo, _mm256_mul_ps(wl, xe));
                        acc[n] = _mm256_fmadd_ps(t, sv, acc[n]);
                    }
                }
            }
            for n in 0..NX {
                out[n] = hsum(acc[n]) * factor;
            }
        }
        let mut lanes = [0u8; 16];
        // SAFETY: `lanes` is 16 bytes, the width of the store.
        unsafe { _mm_storeu_si128(lanes.as_mut_ptr() as *mut __m128i, maxv) };
        *max_code = (*max_code).max(lanes.into_iter().max().unwrap_or(0));
    }
}

// ---------------------------------------------------------------------------------------------
// Aligned scratch
// ---------------------------------------------------------------------------------------------

#[repr(C, align(64))]
#[derive(Clone, Copy)]
struct Line([f32; 16]);

/// Cache-line-aligned f32 storage, so vector loads of the permuted vectors never split lines.
struct AlignedBuf {
    lines: Vec<Line>,
    len: usize,
}

impl AlignedBuf {
    fn new() -> Self {
        Self {
            lines: Vec::new(),
            len: 0,
        }
    }

    fn ensure(&mut self, len: usize) {
        let needed = len.div_ceil(16);
        if self.lines.len() < needed {
            self.lines.resize(needed, Line([0.0; 16]));
        }
        self.len = len;
    }

    fn as_mut_ptr(&mut self) -> *mut f32 {
        self.lines.as_mut_ptr() as *mut f32
    }

    fn as_slice(&self) -> &[f32] {
        // SAFETY: as in `as_mut_slice`, for a shared borrow.
        unsafe { std::slice::from_raw_parts(self.lines.as_ptr() as *const f32, self.len) }
    }

    fn as_mut_slice(&mut self) -> &mut [f32] {
        // SAFETY: `Line` is repr(C) around [f32; 16], so the lines are `lines.len() * 16` contiguous
        // SAFETY: f32 values and `len` never exceeds that after `ensure`.
        unsafe { std::slice::from_raw_parts_mut(self.as_mut_ptr(), self.len) }
    }
}

// ---------------------------------------------------------------------------------------------
// One call: planning, chunk execution
// ---------------------------------------------------------------------------------------------

/// One distinct expert of a call and the rows that selected it.
struct Work {
    expert: usize,
    hits: usize,
    token: [u8; MAX_BATCH_ROWS],
    weight: [f32; MAX_BATCH_ROWS],
    /// Offset of this expert's first intermediate vector in the shared intermediate buffer.
    u_offset: usize,
    max_up: AtomicU32,
    max_down: AtomicU32,
}

/// Everything the chunks of one call read and write. Lives on the calling thread's stack; workers
/// reach it through the type-erased pointer of [`JobCore`] only while the call is waiting.
struct Ctx<'a> {
    isa: Isa,
    layer: &'a LayerExperts<'a>,
    works: &'a [Work],
    tokens: usize,
    x_perm: *const f32,
    u_perm: *mut f32,
    out: *mut f32,
    up_chunks: usize,
    up_total: usize,
    nonfinite: AtomicBool,
}

impl Ctx<'_> {
    /// Runs chunk `i`: indices below `up_total` are up-projection chunks, the rest down chunks.
    fn run_chunk(&self, i: usize) {
        if i < self.up_total {
            self.up_chunk(i / self.up_chunks, i % self.up_chunks);
        } else {
            self.down_chunk(i - self.up_total);
        }
    }

    fn x_row(&self, token: usize) -> &[f32] {
        let latent = self.layer.latent;
        // SAFETY: x_perm holds `tokens * latent` floats written before the job was published and
        // SAFETY: never written while it runs; `token < tokens` by planning.
        unsafe { std::slice::from_raw_parts(self.x_perm.add(token * latent), latent) }
    }

    fn u_row(&self, work: &Work, hit: usize) -> &[f32] {
        let inter = self.layer.inter;
        // SAFETY: the vector was fully written by the up chunks of `work`, which completed (acquire on
        // SAFETY: the phase counter) before any down chunk runs, and is not written afterwards.
        unsafe { std::slice::from_raw_parts(self.u_perm.add(work.u_offset + hit * inter), inter) }
    }

    fn up_chunk(&self, w: usize, block: usize) {
        let work = &self.works[w];
        let view = &self.layer.views[work.expert];
        let inter = self.layer.inter;
        let rows = block * UP_CHUNK_ROWS..((block + 1) * UP_CHUNK_ROWS).min(inter);
        let mut max_code = 0u8;
        let mut res = [RowResult::default(); UP_CHUNK_ROWS];
        for first in (0..work.hits).step_by(PASS_ROWS) {
            let count = (work.hits - first).min(PASS_ROWS);
            let xs: [&[f32]; PASS_ROWS] = std::array::from_fn(|n| {
                self.x_row(usize::from(work.token[first + n.min(count - 1)]))
            });
            dot_rows_n(
                self.isa,
                &view.up,
                rows.clone(),
                &xs[..count],
                &mut res,
                &mut max_code,
            );
            for (lanes, row) in res.iter().zip(rows.clone()) {
                let at = perm_index(inter, row);
                for (n, &v) in lanes.iter().enumerate().take(count) {
                    if !v.is_finite() {
                        self.nonfinite.store(true, Ordering::Relaxed);
                    }
                    let u = v.max(0.0);
                    // SAFETY: row blocks are multiples of BLOCK, so this chunk exclusively owns the
                    // SAFETY: permuted positions [rows.start, rows.end) of every vector of `work`.
                    unsafe {
                        *self.u_perm.add(work.u_offset + (first + n) * inter + at) = u * u;
                    }
                }
            }
        }
        work.max_up
            .fetch_max(u32::from(max_code), Ordering::Relaxed);
    }

    fn down_chunk(&self, block: usize) {
        let latent = self.layer.latent;
        let rows = block * DOWN_CHUNK_ROWS..((block + 1) * DOWN_CHUNK_ROWS).min(latent);
        // Per-row sums stay in this frame and reach `out` once at the end: adjacent chunks share
        // a cache line of `out`, and writing through it for every expert made the line bounce.
        let mut sums = [[0.0f32; DOWN_CHUNK_ROWS]; MAX_BATCH_ROWS];
        let mut res = [RowResult::default(); DOWN_CHUNK_ROWS];
        for work in self.works {
            let view = &self.layer.views[work.expert];
            let mut max_code = 0u8;
            for first in (0..work.hits).step_by(PASS_ROWS) {
                let count = (work.hits - first).min(PASS_ROWS);
                let xs: [&[f32]; PASS_ROWS] =
                    std::array::from_fn(|n| self.u_row(work, first + n.min(count - 1)));
                dot_rows_n(
                    self.isa,
                    &view.down,
                    rows.clone(),
                    &xs[..count],
                    &mut res,
                    &mut max_code,
                );
                for n in 0..count {
                    let token = usize::from(work.token[first + n]);
                    let weight = work.weight[first + n];
                    for (sum, lanes) in sums[token].iter_mut().zip(&res) {
                        *sum += lanes[n] * weight;
                    }
                }
            }
            work.max_down
                .fetch_max(u32::from(max_code), Ordering::Relaxed);
        }
        for (token, sum) in sums.iter().enumerate().take(self.tokens) {
            // SAFETY: this chunk exclusively owns out[token][rows] (disjoint row blocks).
            let dst = unsafe {
                std::slice::from_raw_parts_mut(
                    self.out.add(token * latent + rows.start),
                    rows.len(),
                )
            };
            dst.copy_from_slice(&sum[..rows.len()]);
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Pool
// ---------------------------------------------------------------------------------------------

/// How an idle worker waits for the next call.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WaitPolicy {
    /// Busy-wait forever (yielding the CPU now and then). Fastest wake-up, burns a core when idle,
    /// and degrades badly when the host is oversubscribed.
    Spin,
    /// Block on a condition variable at once. Each call pays the scheduler's wake-up latency.
    Park,
    /// Busy-wait for this long after a call finishes, then park. Layers of one token follow each
    /// other within microseconds, so workers stay hot across a token and sleep between tokens.
    SpinThenPark(Duration),
}

/// Pool settings.
#[derive(Clone, Copy, Debug)]
pub struct PoolConfig {
    /// Threads that compute a call, the calling thread included (`threads - 1` are spawned).
    pub threads: usize,
    pub wait: WaitPolicy,
    /// Force an instruction set; `None` picks the best supported one.
    pub isa: Option<Isa>,
    /// Called once on each spawned worker, with its index in `1..threads`, before it takes work.
    /// The engine can pin the thread to a core here; the module itself needs no OS binding.
    pub on_worker_start: Option<fn(usize)>,
}

impl Default for PoolConfig {
    fn default() -> Self {
        Self {
            threads: 12,
            wait: WaitPolicy::SpinThenPark(Duration::from_micros(500)),
            isa: None,
            on_worker_start: None,
        }
    }
}

#[repr(align(64))]
struct Padded<T>(T);

/// One call as the workers see it: a counter of chunks and a type-erased pointer to the context.
struct JobCore {
    ctx: *const (),
    run: unsafe fn(*const (), usize),
    up_total: usize,
    total: usize,
    next: Padded<AtomicUsize>,
    up_done: Padded<AtomicUsize>,
    done: Padded<AtomicUsize>,
    panicked: AtomicBool,
}

// SAFETY: `ctx` is only dereferenced by `work` after a successful claim of a chunk index below
// SAFETY: `total`, and the call that owns the pointee does not return before `done == total`, so the
// SAFETY: pointee outlives every dereference. The remaining fields are atomics or immutable.
unsafe impl Send for JobCore {}
// SAFETY: see above; all shared mutation goes through atomics or disjoint chunk-owned memory.
unsafe impl Sync for JobCore {}

/// Spins with a pause hint, then yields the CPU, until `ready` holds.
fn wait_until(ready: impl Fn() -> bool) {
    let mut spins = 0u32;
    while !ready() {
        spins += 1;
        if spins < 4096 {
            std::hint::spin_loop();
        } else {
            std::thread::yield_now();
        }
    }
}

impl JobCore {
    /// Claims and runs chunks until none is left. Down chunks wait for every up chunk first.
    fn work(&self) {
        loop {
            let i = self.next.0.fetch_add(1, Ordering::Relaxed);
            if i >= self.total {
                return;
            }
            let is_up = i < self.up_total;
            if !is_up {
                wait_until(|| self.up_done.0.load(Ordering::Acquire) == self.up_total);
            }
            // SAFETY: chunk `i < total` was claimed above, so the context is alive (see the Send impl).
            let result = catch_unwind(AssertUnwindSafe(|| unsafe { (self.run)(self.ctx, i) }));
            if result.is_err() {
                self.panicked.store(true, Ordering::Relaxed);
            }
            if is_up {
                self.up_done.0.fetch_add(1, Ordering::Release);
            }
            self.done.0.fetch_add(1, Ordering::Release);
        }
    }
}

/// Entry the workers call for one chunk.
///
/// # Safety
/// `ctx` must point to a live `Ctx` and `i` must be a claimed chunk index of that job.
unsafe fn run_chunk_erased(ctx: *const (), i: usize) {
    // SAFETY: guaranteed by the caller.
    let ctx = unsafe { &*(ctx as *const Ctx<'_>) };
    ctx.run_chunk(i);
}

struct Slot {
    epoch: u64,
    job: Option<Arc<JobCore>>,
    parked: usize,
    shutdown: bool,
}

struct Shared {
    slot: Mutex<Slot>,
    wake: Condvar,
    /// Mirrors `Slot::epoch` so spinning workers can poll without the lock.
    epoch: AtomicU64,
    wait: WaitPolicy,
}

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(PoisonError::into_inner)
}

impl Shared {
    fn publish(&self, job: Option<Arc<JobCore>>, shutdown: bool) {
        let mut slot = lock(&self.slot);
        slot.epoch += 1;
        slot.job = job;
        slot.shutdown = shutdown;
        self.epoch.store(slot.epoch, Ordering::Release);
        let parked = slot.parked;
        drop(slot);
        if parked > 0 {
            self.wake.notify_all();
        }
    }

    fn spin_window(&self) -> Option<Duration> {
        match self.wait {
            WaitPolicy::Spin => Some(Duration::MAX),
            WaitPolicy::Park => None,
            WaitPolicy::SpinThenPark(window) => Some(window),
        }
    }

    /// Busy-waits for a new epoch up to the policy's window; true when one arrived. The wait yields
    /// the CPU now and then, so spinning workers cannot starve the calling thread (or any other
    /// runnable thread) when threads outnumber the cores available to the pool.
    fn spin_for_epoch(&self, seen: u64) -> bool {
        let Some(window) = self.spin_window() else {
            return false;
        };
        let start = Instant::now();
        let mut spins = 0u32;
        while self.epoch.load(Ordering::Acquire) == seen {
            spins += 1;
            if spins.is_multiple_of(64) && start.elapsed() >= window {
                return false;
            }
            if spins.is_multiple_of(4096) {
                std::thread::yield_now();
            }
            std::hint::spin_loop();
        }
        true
    }

    /// Blocks until the epoch moves past `seen`; returns the new epoch, its job and the shutdown flag.
    fn next_job(&self, seen: u64) -> (u64, Option<Arc<JobCore>>, bool) {
        self.spin_for_epoch(seen);
        let mut slot = lock(&self.slot);
        while slot.epoch == seen {
            slot.parked += 1;
            slot = self.wake.wait(slot).unwrap_or_else(PoisonError::into_inner);
            slot.parked -= 1;
        }
        (slot.epoch, slot.job.clone(), slot.shutdown)
    }
}

fn worker_main(shared: Arc<Shared>, index: usize, on_start: Option<fn(usize)>) {
    if let Some(hook) = on_start {
        hook(index);
    }
    let mut seen = 0u64;
    loop {
        let (epoch, job, shutdown) = shared.next_job(seen);
        if shutdown {
            return;
        }
        seen = epoch;
        if let Some(job) = job {
            job.work();
        }
    }
}

struct Scratch {
    x_perm: AlignedBuf,
    u_perm: AlignedBuf,
    works: Vec<Work>,
    hits: Vec<(u32, u8, f32)>,
}

/// A persistent set of worker threads plus the scratch buffers of the call in flight.
///
/// The pool is `Sync`; concurrent calls serialize on the scratch lock. Dropping it joins the
/// workers. It holds no reference to any weight view between calls.
pub struct ExpertPool {
    shared: Arc<Shared>,
    workers: Vec<JoinHandle<()>>,
    threads: usize,
    isa: Isa,
    scratch: Mutex<Scratch>,
}

impl ExpertPool {
    pub fn new(config: PoolConfig) -> Result<Self, Nvfp4CpuError> {
        let isa = config.isa.unwrap_or_else(Isa::detect);
        if !isa.is_supported() {
            return Err(Nvfp4CpuError::Pool(format!(
                "{isa:?} is not supported by this CPU"
            )));
        }
        let threads = config.threads.max(1);
        let shared = Arc::new(Shared {
            slot: Mutex::new(Slot {
                epoch: 0,
                job: None,
                parked: 0,
                shutdown: false,
            }),
            wake: Condvar::new(),
            epoch: AtomicU64::new(0),
            wait: config.wait,
        });
        let mut workers = Vec::with_capacity(threads - 1);
        for index in 1..threads {
            let worker_shared = Arc::clone(&shared);
            let spawned = std::thread::Builder::new()
                .name(format!("nvfp4-expert-{index}"))
                .spawn(move || worker_main(worker_shared, index, config.on_worker_start));
            match spawned {
                Ok(handle) => workers.push(handle),
                Err(error) => {
                    let mut pool = Self::assemble(shared, workers, threads, isa);
                    pool.shutdown();
                    return Err(Nvfp4CpuError::Pool(format!(
                        "spawn worker {index}: {error}"
                    )));
                }
            }
        }
        Ok(Self::assemble(shared, workers, threads, isa))
    }

    fn assemble(
        shared: Arc<Shared>,
        workers: Vec<JoinHandle<()>>,
        threads: usize,
        isa: Isa,
    ) -> Self {
        Self {
            shared,
            workers,
            threads,
            isa,
            scratch: Mutex::new(Scratch {
                x_perm: AlignedBuf::new(),
                u_perm: AlignedBuf::new(),
                works: Vec::new(),
                hits: Vec::new(),
            }),
        }
    }

    fn shutdown(&mut self) {
        self.shared.publish(None, true);
        for handle in self.workers.drain(..) {
            let _ = handle.join();
        }
    }

    /// Threads that compute a call (workers plus the caller).
    pub fn threads(&self) -> usize {
        self.threads
    }

    pub fn isa(&self) -> Isa {
        self.isa
    }

    /// One activation row: `out = sum_e w_e * down_e(relu2(up_e(x)))` over `picks`.
    pub fn routed_experts(
        &self,
        layer: &LayerExperts<'_>,
        x: &[f32],
        picks: &[ExpertPick],
        out: &mut [f32],
    ) -> Result<(), Nvfp4CpuError> {
        self.routed_experts_batch(layer, x, &[picks], out)
    }

    /// 1..=[`MAX_BATCH_ROWS`] rows; row `t` of `xs`/`out` (each `latent` wide) uses `picks[t]`.
    /// An expert chosen by several rows is read from memory once.
    pub fn routed_experts_batch<P: AsRef<[ExpertPick]>>(
        &self,
        layer: &LayerExperts<'_>,
        xs: &[f32],
        picks: &[P],
        out: &mut [f32],
    ) -> Result<(), Nvfp4CpuError> {
        let tokens = picks.len();
        let latent = layer.latent;
        if tokens == 0 || tokens > MAX_BATCH_ROWS {
            return Err(shape_err(format!(
                "row count {tokens} outside 1..={MAX_BATCH_ROWS}"
            )));
        }
        if xs.len() != tokens * latent || out.len() != tokens * latent {
            return Err(shape_err(
                "activation or output length differs from rows x latent",
            ));
        }
        if xs.iter().any(|v| !v.is_finite()) {
            return Err(input_err("activation values must be finite"));
        }
        let mut scratch = lock(&self.scratch);
        let scratch = &mut *scratch;
        plan(layer, picks, scratch)?;
        scratch.x_perm.ensure(tokens * latent);
        for (src, dst) in xs
            .chunks_exact(latent)
            .zip(scratch.x_perm.as_mut_slice().chunks_exact_mut(latent))
        {
            permute_row(src, dst);
        }
        let hits_total: usize = scratch.works.iter().map(|w| w.hits).sum();
        scratch.u_perm.ensure(hits_total * layer.inter);
        let up_chunks = layer.inter.div_ceil(UP_CHUNK_ROWS);
        let up_total = scratch.works.len() * up_chunks;
        let ctx = Ctx {
            isa: self.isa,
            layer,
            works: &scratch.works,
            tokens,
            x_perm: scratch.x_perm.as_mut_ptr(),
            u_perm: scratch.u_perm.as_mut_ptr(),
            out: out.as_mut_ptr(),
            up_chunks,
            up_total,
            nonfinite: AtomicBool::new(false),
        };
        let panicked = self.execute(&ctx, up_total + latent.div_ceil(DOWN_CHUNK_ROWS), up_total);
        if panicked {
            return Err(Nvfp4CpuError::Pool("a chunk panicked".into()));
        }
        finish(&ctx, layer)
    }

    /// Publishes the job, works on it from the calling thread and waits for every chunk.
    fn execute(&self, ctx: &Ctx<'_>, total: usize, up_total: usize) -> bool {
        let job = Arc::new(JobCore {
            ctx: ctx as *const Ctx<'_> as *const (),
            run: run_chunk_erased,
            up_total,
            total,
            next: Padded(AtomicUsize::new(0)),
            up_done: Padded(AtomicUsize::new(0)),
            done: Padded(AtomicUsize::new(0)),
            panicked: AtomicBool::new(false),
        });
        if self.threads > 1 {
            self.shared.publish(Some(Arc::clone(&job)), false);
        }
        job.work();
        wait_until(|| job.done.0.load(Ordering::Acquire) == total);
        job.panicked.load(Ordering::Relaxed)
    }
}

impl Drop for ExpertPool {
    fn drop(&mut self) {
        self.shutdown();
    }
}

/// Validates the picks and fills `scratch.works`: one entry per distinct expert, ascending.
fn plan<P: AsRef<[ExpertPick]>>(
    layer: &LayerExperts<'_>,
    picks: &[P],
    scratch: &mut Scratch,
) -> Result<(), Nvfp4CpuError> {
    scratch.hits.clear();
    for (token, row) in picks.iter().enumerate() {
        let row = row.as_ref();
        if row.is_empty() {
            return Err(input_err(
                "selection must contain one nonempty row per token",
            ));
        }
        for (position, pick) in row.iter().enumerate() {
            let repeated = row[..position].iter().any(|p| p.expert == pick.expert);
            if !pick.weight.is_finite() || pick.weight < 0.0 || repeated {
                return Err(input_err("invalid selection"));
            }
            if pick.expert as usize >= layer.views.len() {
                return Err(input_err(format!("expert {} is not loaded", pick.expert)));
            }
            scratch.hits.push((pick.expert, token as u8, pick.weight));
        }
    }
    // Ascending expert, then token: the reference accumulates hit experts in index order.
    scratch
        .hits
        .sort_by_key(|&(expert, token, _)| (expert, token));
    scratch.works.clear();
    let mut u_offset = 0usize;
    for &(expert, token, weight) in &scratch.hits {
        match scratch.works.last_mut() {
            Some(work) if work.expert == expert as usize => {
                work.token[work.hits] = token;
                work.weight[work.hits] = weight;
                work.hits += 1;
            }
            _ => {
                let mut work = Work {
                    expert: expert as usize,
                    hits: 1,
                    token: [0; MAX_BATCH_ROWS],
                    weight: [0.0; MAX_BATCH_ROWS],
                    u_offset,
                    max_up: AtomicU32::new(0),
                    max_down: AtomicU32::new(0),
                };
                work.token[0] = token;
                work.weight[0] = weight;
                scratch.works.push(work);
            }
        }
        u_offset += layer.inter;
    }
    Ok(())
}

/// Post-run checks: weight scales seen by the streams, then finite intermediate and output values.
fn finish(ctx: &Ctx<'_>, layer: &LayerExperts<'_>) -> Result<(), Nvfp4CpuError> {
    for work in ctx.works {
        let view = &layer.views[work.expert];
        check_scale_extent(
            work.max_up.load(Ordering::Relaxed) as u8,
            view.up.weight_scale_2,
        )?;
        check_scale_extent(
            work.max_down.load(Ordering::Relaxed) as u8,
            view.down.weight_scale_2,
        )?;
    }
    if ctx.nonfinite.load(Ordering::Relaxed) {
        return Err(weights_err("expert up projection is not finite"));
    }
    // SAFETY: all chunks completed (acquire on `done`), so reading `out` here races with nothing.
    let out = unsafe { std::slice::from_raw_parts(ctx.out, ctx.tokens * layer.latent) };
    if out.iter().any(|v| !v.is_finite()) {
        return Err(weights_err("routed expert output is not finite"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 64-bit LCG; draws keep the top 53 bits and scale by 2^53 (a `>> 11` draw divided by
    /// `u64::MAX` would sit in the first 0.05% of its range).
    struct Lcg(u64);

    impl Lcg {
        fn next_u53(&mut self) -> u64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            self.0 >> 11
        }

        fn unit(&mut self) -> f64 {
            self.next_u53() as f64 / (1u64 << 53) as f64
        }

        fn f32_in(&mut self, lo: f32, hi: f32) -> f32 {
            lo + (hi - lo) * self.unit() as f32
        }

        fn below(&mut self, n: usize) -> usize {
            (self.next_u53() % n as u64) as usize
        }
    }

    struct Owned {
        packed: Vec<u8>,
        scales: Vec<u8>,
        wss2: f32,
        rows: usize,
        cols: usize,
    }

    impl Owned {
        /// Random codes; scale codes drawn from `lo..=hi` (valid codes are 0..=0x7e).
        fn random(rng: &mut Lcg, rows: usize, cols: usize, lo: u8, hi: u8) -> Self {
            let packed = (0..rows * cols / 2).map(|_| rng.below(256) as u8).collect();
            let span = usize::from(hi - lo) + 1;
            let scales = (0..rows * cols / GROUP_SIZE)
                .map(|_| lo + rng.below(span) as u8)
                .collect();
            Self {
                packed,
                scales,
                wss2: rng.f32_in(1e-3, 1.0),
                rows,
                cols,
            }
        }

        fn view(&self) -> Nvfp4Matrix<'_> {
            Nvfp4Matrix::new(&self.packed, &self.scales, self.wss2, self.rows, self.cols).unwrap()
        }

        fn weight(&self, row: usize, col: usize) -> f64 {
            let byte = self.packed[(row * self.cols + col) / 2];
            let code = if col.is_multiple_of(2) {
                byte & 15
            } else {
                byte >> 4
            };
            let scale =
                E4M3[usize::from(self.scales[row * self.cols / GROUP_SIZE + col / GROUP_SIZE])];
            f64::from(E2M1[usize::from(code)]) * f64::from(scale) * f64::from(self.wss2)
        }

        /// Exact-ish f64 matvec and the per-row sum of |w x| (the error scale of a dot product).
        fn matvec_f64(&self, x: &[f32]) -> (Vec<f64>, Vec<f64>) {
            let mut y = vec![0.0; self.rows];
            let mut mag = vec![0.0; self.rows];
            for r in 0..self.rows {
                for (c, &xv) in x.iter().enumerate() {
                    let term = self.weight(r, c) * f64::from(xv);
                    y[r] += term;
                    mag[r] += term.abs();
                }
            }
            (y, mag)
        }

        /// The engine reference's arithmetic: rounded scale, rounded weight, sequential f32 sum.
        fn matvec_reference_f32(&self, x: &[f32]) -> Vec<f32> {
            let per_row = self.cols / GROUP_SIZE;
            (0..self.rows)
                .map(|r| {
                    let mut sum = 0.0f32;
                    for g in 0..per_row {
                        let scale = E4M3[usize::from(self.scales[r * per_row + g])] * self.wss2;
                        for (c, &xv) in x.iter().enumerate().skip(g * GROUP_SIZE).take(GROUP_SIZE) {
                            let byte = self.packed[(r * self.cols + c) / 2];
                            let code = if c.is_multiple_of(2) {
                                byte & 15
                            } else {
                                byte >> 4
                            };
                            sum += E2M1[usize::from(code)] * scale * xv;
                        }
                    }
                    sum
                })
                .collect()
        }
    }

    fn supported_isas() -> Vec<Isa> {
        [Isa::Scalar, Isa::Avx2, Isa::Avx512]
            .into_iter()
            .filter(|isa| isa.is_supported())
            .collect()
    }

    fn random_x(rng: &mut Lcg, k: usize) -> Vec<f32> {
        (0..k).map(|_| rng.f32_in(-2.0, 2.0)).collect()
    }

    /// Runs the row-dot kernel over all rows of `m` for each activation (natural order).
    fn kernel_matvec(isa: Isa, m: &Owned, xs: &[Vec<f32>]) -> (Vec<Vec<f32>>, u8) {
        let permuted: Vec<Vec<f32>> = xs
            .iter()
            .map(|x| {
                let mut p = vec![0.0; x.len()];
                permute_row(x, &mut p);
                p
            })
            .collect();
        let refs: Vec<&[f32]> = permuted.iter().map(Vec::as_slice).collect();
        let mut out = vec![vec![0.0f32; m.rows]; xs.len()];
        let mut max_code = 0u8;
        let view = m.view();
        for first in (0..m.rows).step_by(UP_CHUNK_ROWS) {
            let rows = first..(first + UP_CHUNK_ROWS).min(m.rows);
            let mut res = vec![RowResult::default(); rows.len()];
            dot_rows_n(isa, &view, rows.clone(), &refs, &mut res, &mut max_code);
            for (i, row) in rows.enumerate() {
                for (n, o) in out.iter_mut().enumerate() {
                    o[row] = res[i][n];
                }
            }
        }
        (out, max_code)
    }

    #[test]
    fn rng_draws_spread_over_the_requested_range() {
        let mut rng = Lcg(7);
        let draws: Vec<f32> = (0..4000).map(|_| rng.f32_in(-1.0, 1.0)).collect();
        let (min, max) = draws
            .iter()
            .fold((f32::MAX, f32::MIN), |(a, b), &v| (a.min(v), b.max(v)));
        assert!(min < -0.98 && max > 0.98, "draws span {min}..{max}");
    }

    #[test]
    fn permuted_layout_is_a_bijection_for_every_group_aligned_length() {
        for k in (16..=300).step_by(16) {
            let mut seen = vec![false; k];
            for j in 0..k {
                let at = perm_index(k, j);
                assert!(at < k && !seen[at], "k={k} j={j} -> {at}");
                seen[at] = true;
            }
        }
    }

    #[test]
    fn e4m3_table_matches_the_engine_construction_on_spot_codes() {
        assert_eq!(E4M3[0x38], 1.0);
        assert_eq!(E4M3[0x7e], 448.0);
        assert_eq!(E4M3[0x01], 2.0f32.powi(-9));
        assert!(E4M3[0x7f].is_nan() && E4M3[0xff].is_nan());
        assert!(E4M3[0x80].is_sign_negative() && E4M3[0x80] == 0.0);
    }

    #[test]
    fn every_valid_scale_code_decodes_exactly_on_every_isa() {
        // One row of 16 weights equal to 1.0 (code 2) against x = 1: the dot is 16 * scale exactly.
        for isa in supported_isas() {
            for code in 0..=MAX_VALID_SCALE_CODE {
                let m = Owned {
                    packed: vec![0x22; 8],
                    scales: vec![code],
                    wss2: 1.0,
                    rows: 1,
                    cols: 16,
                };
                let (out, max_code) = kernel_matvec(isa, &m, &[vec![1.0; 16]]);
                assert_eq!(
                    out[0][0],
                    16.0 * E4M3[usize::from(code)],
                    "{isa:?} code {code:#x}"
                );
                assert_eq!(max_code, code);
            }
        }
    }

    #[test]
    fn every_nibble_value_lands_on_its_own_column_on_every_isa() {
        // Row r puts code `c` only at column `col`; x is 1 at that column: dot must be E2M1[c].
        for isa in supported_isas() {
            for cols in [16usize, 32, 48, 288] {
                for code in 0..16u8 {
                    for col in [0, 1, 2, 15, cols / 2, cols - 2, cols - 1] {
                        let mut packed = vec![0u8; cols / 2];
                        packed[col / 2] = if col.is_multiple_of(2) {
                            code
                        } else {
                            code << 4
                        };
                        let m = Owned {
                            packed,
                            scales: vec![0x38; cols / GROUP_SIZE],
                            wss2: 1.0,
                            rows: 1,
                            cols,
                        };
                        let mut x = vec![0.0; cols];
                        x[col] = 1.0;
                        let (out, _) = kernel_matvec(isa, &m, &[x]);
                        assert_eq!(
                            out[0][0],
                            E2M1[usize::from(code)],
                            "{isa:?} cols {cols} col {col} code {code}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn dots_match_an_f64_reference_within_dot_product_error_bounds() {
        let mut rng = Lcg(11);
        let mut worst = [0.0f64; 3];
        for isa in supported_isas() {
            for &cols in &[16usize, 32, 48, 80, 144, 256, 272, 1024, 2688, 4112] {
                for &rows in &[1usize, 3, 17, 33] {
                    for (lo, hi) in [(0x30u8, 0x44u8), (0x00, MAX_VALID_SCALE_CODE)] {
                        let m = Owned::random(&mut rng, rows, cols, lo, hi);
                        let xs = [
                            random_x(&mut rng, cols),
                            random_x(&mut rng, cols),
                            random_x(&mut rng, cols),
                        ];
                        let (out, _) = kernel_matvec(isa, &m, &xs);
                        for (x, y) in xs.iter().zip(&out) {
                            let (truth, mag) = m.matvec_f64(x);
                            let theirs = m.matvec_reference_f32(x);
                            for r in 0..rows {
                                if mag[r] == 0.0 {
                                    assert_eq!(y[r], 0.0);
                                    continue;
                                }
                                let ours_err = (f64::from(y[r]) - truth[r]).abs() / mag[r];
                                let ref_err = (f64::from(theirs[r]) - truth[r]).abs() / mag[r];
                                worst[0] = worst[0].max(ours_err);
                                worst[1] = worst[1].max(ref_err);
                                assert!(
                                    ours_err <= 4e-6,
                                    "{isa:?} {rows}x{cols}: err {ours_err:e}"
                                );
                            }
                        }
                        worst[2] += 1.0;
                    }
                }
            }
        }
        println!(
            "worst error / sum|wx|: kernel {:e}, f32 reference {:e}",
            worst[0], worst[1]
        );
        assert!(
            worst[0] <= worst[1].max(1e-6) * 2.0,
            "kernel is less accurate than the reference path"
        );
    }

    #[test]
    fn instruction_sets_agree_with_each_other_to_rounding() {
        let mut rng = Lcg(5);
        let m = Owned::random(&mut rng, 40, 2688, 0x28, 0x4c);
        let xs = [random_x(&mut rng, 2688)];
        let (scalar, _) = kernel_matvec(Isa::Scalar, &m, &xs);
        let (_, mag) = m.matvec_f64(&xs[0]);
        for isa in supported_isas() {
            let (out, _) = kernel_matvec(isa, &m, &xs);
            for r in 0..40 {
                let diff = f64::from(out[0][r] - scalar[0][r]).abs() / mag[r];
                assert!(diff <= 1e-5, "{isa:?} row {r}: {diff:e}");
            }
        }
    }

    // ----- layer level ------------------------------------------------------------------------

    struct OwnedLayer {
        experts: Vec<(Owned, Owned)>,
        latent: usize,
    }

    impl OwnedLayer {
        fn random(rng: &mut Lcg, experts: usize, latent: usize, inter: usize) -> Self {
            let experts = (0..experts)
                .map(|_| {
                    (
                        Owned::random(rng, inter, latent, 0x30, 0x44),
                        Owned::random(rng, latent, inter, 0x30, 0x44),
                    )
                })
                .collect();
            Self { experts, latent }
        }

        fn layer(&self) -> LayerExperts<'_> {
            let views = self
                .experts
                .iter()
                .map(|(up, down)| ExpertView {
                    up: up.view(),
                    down: down.view(),
                })
                .collect();
            LayerExperts::new(views).unwrap()
        }

        /// f64 pipeline: relu2 between the projections, experts summed in ascending index.
        fn reference_f64(&self, x: &[f32], picks: &[ExpertPick]) -> Vec<f64> {
            let mut sorted = picks.to_vec();
            sorted.sort_by_key(|p| p.expert);
            let mut out = vec![0.0f64; self.latent];
            for pick in sorted {
                let (up, down) = &self.experts[pick.expert as usize];
                let (u, _) = up.matvec_f64(x);
                let u: Vec<f32> = u
                    .iter()
                    .map(|&v| (v.max(0.0) * v.max(0.0)) as f32)
                    .collect();
                let (y, _) = down.matvec_f64(&u);
                for (o, v) in out.iter_mut().zip(y) {
                    *o += v * f64::from(pick.weight);
                }
            }
            out
        }
    }

    fn pool(threads: usize, wait: WaitPolicy, isa: Isa) -> ExpertPool {
        ExpertPool::new(PoolConfig {
            threads,
            wait,
            isa: Some(isa),
            on_worker_start: None,
        })
        .unwrap()
    }

    fn picks_for(rng: &mut Lcg, experts: usize, count: usize) -> Vec<ExpertPick> {
        let mut chosen: Vec<u32> = Vec::new();
        while chosen.len() < count {
            let e = rng.below(experts) as u32;
            if !chosen.contains(&e) {
                chosen.push(e);
            }
        }
        chosen
            .into_iter()
            .map(|expert| ExpertPick {
                expert,
                weight: rng.f32_in(0.05, 1.5),
            })
            .collect()
    }

    fn assert_close(out: &[f32], truth: &[f64], what: &str) {
        let scale = truth.iter().fold(0.0f64, |m, v| m.max(v.abs())).max(1e-30);
        for (i, (&o, &t)) in out.iter().zip(truth).enumerate() {
            let err = (f64::from(o) - t).abs() / scale;
            assert!(
                err <= 2e-5,
                "{what}[{i}]: {o} vs {t} (err {err:e} of max {scale:e})"
            );
        }
    }

    #[test]
    fn routed_experts_match_the_f64_pipeline_on_every_isa_and_thread_count() {
        let mut rng = Lcg(21);
        // 96 = three blocks; 80 = two blocks and a trailing group; 48 likewise.
        for (latent, inter) in [(64usize, 96usize), (48, 80), (32, 16), (128, 272)] {
            let owned = OwnedLayer::random(&mut rng, 9, latent, inter);
            let layer = owned.layer();
            let x = random_x(&mut rng, latent);
            let picks = picks_for(&mut rng, 9, 5);
            let truth = owned.reference_f64(&x, &picks);
            let mut first: Option<Vec<f32>> = None;
            for isa in supported_isas() {
                for threads in [1usize, 2, 5] {
                    let p = pool(threads, WaitPolicy::Park, isa);
                    let mut out = vec![0.0f32; latent];
                    p.routed_experts(&layer, &x, &picks, &mut out).unwrap();
                    assert_close(
                        &out,
                        &truth,
                        &format!("{isa:?} t{threads} {latent}x{inter}"),
                    );
                    if isa == Isa::detect() {
                        match &first {
                            None => first = Some(out),
                            Some(f) => assert_eq!(f, &out, "result depends on the thread count"),
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn batch_rows_equal_single_row_calls_bit_for_bit() {
        let mut rng = Lcg(33);
        let (latent, inter) = (64usize, 96usize);
        let owned = OwnedLayer::random(&mut rng, 12, latent, inter);
        let layer = owned.layer();
        for isa in supported_isas() {
            let p = pool(3, WaitPolicy::Park, isa);
            for rows in 1..=MAX_BATCH_ROWS {
                let xs: Vec<f32> = (0..rows).flat_map(|_| random_x(&mut rng, latent)).collect();
                // Rows share experts on purpose (12 experts, 4 picks each).
                let picks: Vec<Vec<ExpertPick>> =
                    (0..rows).map(|_| picks_for(&mut rng, 12, 4)).collect();
                let mut batch = vec![0.0f32; rows * latent];
                p.routed_experts_batch(&layer, &xs, &picks, &mut batch)
                    .unwrap();
                for t in 0..rows {
                    let mut single = vec![0.0f32; latent];
                    p.routed_experts(
                        &layer,
                        &xs[t * latent..(t + 1) * latent],
                        &picks[t],
                        &mut single,
                    )
                    .unwrap();
                    assert_eq!(
                        &batch[t * latent..(t + 1) * latent],
                        single.as_slice(),
                        "{isa:?} rows {rows} row {t}"
                    );
                }
            }
        }
    }

    #[test]
    fn weight_views_at_odd_addresses_give_identical_results() {
        // Checkpoint tensors start at arbitrary offsets of the mapping; no alignment may be assumed.
        let mut rng = Lcg(19);
        let owned = OwnedLayer::random(&mut rng, 3, 64, 96);
        let x = random_x(&mut rng, 64);
        let picks = picks_for(&mut rng, 3, 2);
        let mut aligned = [0.0f32; 64];
        let p = pool(2, WaitPolicy::Park, Isa::detect());
        p.routed_experts(&owned.layer(), &x, &picks, &mut aligned)
            .unwrap();
        for shift in [1usize, 3, 7, 13] {
            let shifted = |bytes: &[u8]| {
                let mut buffer = vec![0u8; bytes.len() + 16];
                buffer[shift..shift + bytes.len()].copy_from_slice(bytes);
                buffer
            };
            let buffers: Vec<[Vec<u8>; 4]> = owned
                .experts
                .iter()
                .map(|(up, down)| {
                    [
                        shifted(&up.packed),
                        shifted(&up.scales),
                        shifted(&down.packed),
                        shifted(&down.scales),
                    ]
                })
                .collect();
            let views = owned
                .experts
                .iter()
                .zip(&buffers)
                .map(|((up, down), b)| ExpertView {
                    up: Nvfp4Matrix::new(
                        &b[0][shift..shift + up.packed.len()],
                        &b[1][shift..shift + up.scales.len()],
                        up.wss2,
                        up.rows,
                        up.cols,
                    )
                    .unwrap(),
                    down: Nvfp4Matrix::new(
                        &b[2][shift..shift + down.packed.len()],
                        &b[3][shift..shift + down.scales.len()],
                        down.wss2,
                        down.rows,
                        down.cols,
                    )
                    .unwrap(),
                })
                .collect();
            let layer = LayerExperts::new(views).unwrap();
            let mut out = [0.0f32; 64];
            p.routed_experts(&layer, &x, &picks, &mut out).unwrap();
            assert_eq!(out, aligned, "shift {shift}");
        }
    }

    #[test]
    fn zero_scales_and_zero_weights_give_zero_output() {
        let mut rng = Lcg(2);
        let mut owned = OwnedLayer::random(&mut rng, 3, 32, 64);
        for (up, down) in &mut owned.experts {
            up.scales.fill(0);
            down.scales.fill(0);
        }
        let layer = owned.layer();
        let p = pool(2, WaitPolicy::Park, Isa::detect());
        let mut out = vec![1.0f32; 32];
        p.routed_experts(
            &layer,
            &random_x(&mut rng, 32),
            &picks_for(&mut rng, 3, 2),
            &mut out,
        )
        .unwrap();
        assert!(out.iter().all(|&v| v == 0.0));
    }

    // ----- refusals ---------------------------------------------------------------------------

    fn small_layer(rng: &mut Lcg) -> OwnedLayer {
        OwnedLayer::random(rng, 4, 32, 64)
    }

    fn expect_err(result: Result<(), Nvfp4CpuError>, kind: &str) {
        let error = result.expect_err("call should be refused");
        let text = format!("{error:?}");
        assert!(text.starts_with(kind), "{text} should be a {kind} error");
    }

    #[test]
    fn non_finite_activations_are_refused() {
        let mut rng = Lcg(3);
        let owned = small_layer(&mut rng);
        let layer = owned.layer();
        let p = pool(2, WaitPolicy::Park, Isa::detect());
        let picks = [ExpertPick {
            expert: 1,
            weight: 1.0,
        }];
        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let mut x = random_x(&mut rng, 32);
            x[7] = bad;
            expect_err(
                p.routed_experts(&layer, &x, &picks, &mut [0.0; 32]),
                "Input",
            );
        }
    }

    #[test]
    fn invalid_scale_codes_are_refused_wherever_they_sit() {
        let mut rng = Lcg(4);
        // 0x7f and 0xff are NaN, 0x80 is -0 (the reference refuses a negative sign), 0xb8 is -1.
        for code in [0x7fu8, 0xff, 0x80, 0xb8] {
            for (in_up, at) in [(true, 0usize), (true, 4095), (false, 0), (false, 2047)] {
                let mut owned = small_layer(&mut rng);
                let target = if in_up {
                    &mut owned.experts[2].0
                } else {
                    &mut owned.experts[2].1
                };
                let at = at % target.scales.len();
                target.scales[at] = code;
                let layer = owned.layer();
                let p = pool(2, WaitPolicy::Park, Isa::detect());
                let picks = [ExpertPick {
                    expert: 2,
                    weight: 1.0,
                }];
                let x = random_x(&mut rng, 32);
                expect_err(
                    p.routed_experts(&layer, &x, &picks, &mut [0.0; 32]),
                    "Weights",
                );
                // The same bad scale in an expert nobody picked is not read and not refused.
                let picks = [ExpertPick {
                    expert: 1,
                    weight: 1.0,
                }];
                p.routed_experts(&layer, &x, &picks, &mut [0.0; 32])
                    .unwrap();
            }
        }
    }

    #[test]
    fn weight_scale_two_and_scale_products_are_checked() {
        let mut rng = Lcg(6);
        let packed = vec![0x22u8; 8];
        let scales = vec![0x38u8];
        for bad in [0.0f32, -1.0, f32::NAN, f32::INFINITY] {
            assert!(
                Nvfp4Matrix::new(&packed, &scales, bad, 1, 16).is_err(),
                "wss2 {bad}"
            );
        }
        // 448 * 1e38 overflows f32: the matrix is constructible but every use is refused.
        let mut owned = small_layer(&mut rng);
        owned.experts[0].0.scales.fill(MAX_VALID_SCALE_CODE);
        owned.experts[0].0.wss2 = 1e38;
        let layer = owned.layer();
        let p = pool(2, WaitPolicy::Park, Isa::detect());
        let picks = [ExpertPick {
            expert: 0,
            weight: 1.0,
        }];
        expect_err(
            p.routed_experts(&layer, &random_x(&mut rng, 32), &picks, &mut [0.0; 32]),
            "Weights",
        );
    }

    #[test]
    fn decoded_weight_overflow_is_refused_even_when_the_scale_itself_is_finite() {
        // 448 * 7e35 = 3.1e38 fits f32, 6 * that does not: the reference refuses the decoded weight.
        // A tiny activation keeps the projection finite, so only the scale check can refuse it.
        let mut rng = Lcg(10);
        let mut owned = small_layer(&mut rng);
        owned.experts[0].1.scales.fill(MAX_VALID_SCALE_CODE);
        owned.experts[0].1.wss2 = 7e35;
        let layer = owned.layer();
        let p = pool(2, WaitPolicy::Park, Isa::detect());
        let picks = [ExpertPick {
            expert: 0,
            weight: 1.0,
        }];
        expect_err(
            p.routed_experts(&layer, &[1e-6f32; 32], &picks, &mut [0.0; 32]),
            "Weights",
        );
    }

    #[test]
    fn overflowing_up_projection_is_refused_even_though_relu_would_hide_it() {
        // Every code is -6 and the scales are huge but valid: the dot overflows to -inf, relu2 turns that
        // into 0 and the output stays finite, yet the reference refuses a non-finite up projection.
        let mut rng = Lcg(17);
        let mut owned = small_layer(&mut rng);
        owned.experts[2].0.packed.fill(0xff);
        owned.experts[2].0.scales.fill(MAX_VALID_SCALE_CODE);
        owned.experts[2].0.wss2 = 1e30;
        let layer = owned.layer();
        let p = pool(2, WaitPolicy::Park, Isa::detect());
        let picks = [ExpertPick {
            expert: 2,
            weight: 1.0,
        }];
        expect_err(
            p.routed_experts(&layer, &[1e6f32; 32], &picks, &mut [0.0; 32]),
            "Weights",
        );
    }

    #[test]
    fn invalid_scale_codes_are_refused_inside_whole_super_blocks_too() {
        // 512 columns = 32 groups = two full 16-group super-blocks per row (the small layers never reach them).
        let mut rng = Lcg(18);
        for at in [0usize, 7, 16, 31, 100, 511] {
            let mut owned = OwnedLayer::random(&mut rng, 2, 512, 64);
            owned.experts[1].0.scales[at] = 0x7f;
            let layer = owned.layer();
            let p = pool(2, WaitPolicy::Park, Isa::detect());
            let x = random_x(&mut rng, 512);
            let bad = [ExpertPick {
                expert: 1,
                weight: 1.0,
            }];
            expect_err(
                p.routed_experts(&layer, &x, &bad, &mut [0.0; 512]),
                "Weights",
            );
            let good = [ExpertPick {
                expert: 0,
                weight: 1.0,
            }];
            p.routed_experts(&layer, &x, &good, &mut [0.0; 512])
                .unwrap();
        }
    }

    #[test]
    fn overflowing_output_is_refused() {
        // Valid scales and finite weights, but the down projection sums past f32::MAX.
        let mut rng = Lcg(16);
        let mut owned = small_layer(&mut rng);
        owned.experts[1].1.scales.fill(0x70);
        owned.experts[1].1.wss2 = 1e30;
        owned.experts[1].0.scales.fill(0x50);
        owned.experts[1].0.wss2 = 1.0;
        let layer = owned.layer();
        let p = pool(2, WaitPolicy::Park, Isa::detect());
        let picks = [ExpertPick {
            expert: 1,
            weight: 1.0,
        }];
        let x = vec![1.0e3f32; 32];
        expect_err(
            p.routed_experts(&layer, &x, &picks, &mut [0.0; 32]),
            "Weights",
        );
    }

    #[test]
    fn malformed_calls_are_refused_with_a_shape_or_input_error() {
        let mut rng = Lcg(8);
        let owned = small_layer(&mut rng);
        let layer = owned.layer();
        let p = pool(2, WaitPolicy::Park, Isa::detect());
        let x = random_x(&mut rng, 32);
        let ok = ExpertPick {
            expert: 0,
            weight: 1.0,
        };
        let mut out = vec![0.0f32; 32];
        expect_err(p.routed_experts(&layer, &x[..31], &[ok], &mut out), "Shape");
        expect_err(p.routed_experts(&layer, &x, &[ok], &mut out[..31]), "Shape");
        expect_err(p.routed_experts(&layer, &x, &[], &mut out), "Input");
        expect_err(p.routed_experts(&layer, &x, &[ok, ok], &mut out), "Input");
        for weight in [-0.5f32, f32::NAN, f32::INFINITY] {
            expect_err(
                p.routed_experts(&layer, &x, &[ExpertPick { expert: 0, weight }], &mut out),
                "Input",
            );
        }
        expect_err(
            p.routed_experts(
                &layer,
                &x,
                &[ExpertPick {
                    expert: 4,
                    weight: 1.0,
                }],
                &mut out,
            ),
            "Input",
        );
        let nine: Vec<Vec<ExpertPick>> = (0..9).map(|_| vec![ok]).collect();
        let xs = vec![0.5f32; 9 * 32];
        expect_err(
            p.routed_experts_batch(&layer, &xs, &nine, &mut vec![0.0; 9 * 32]),
            "Shape",
        );
        let none: [Vec<ExpertPick>; 0] = [];
        expect_err(p.routed_experts_batch(&layer, &[], &none, &mut []), "Shape");
    }

    #[test]
    fn matrix_and_layer_constructors_check_geometry() {
        let packed = vec![0u8; 8];
        let scales = vec![0x38u8];
        assert!(Nvfp4Matrix::new(&packed, &scales, 1.0, 1, 16).is_ok());
        assert!(
            Nvfp4Matrix::new(&packed, &scales, 1.0, 1, 24).is_err(),
            "cols not group aligned"
        );
        assert!(Nvfp4Matrix::new(&packed[..7], &scales, 1.0, 1, 16).is_err());
        assert!(Nvfp4Matrix::new(&packed, &[], 1.0, 1, 16).is_err());
        assert!(Nvfp4Matrix::new(&packed, &scales, 1.0, 0, 16).is_err());
        let a = Nvfp4Matrix::new(&packed, &scales, 1.0, 1, 16).unwrap();
        let view = ExpertView { up: a, down: a };
        // up [1, 16] with down [1, 16]: down.rows (1) must equal latent (16).
        assert!(LayerExperts::new(vec![view]).is_err());
        assert!(LayerExperts::new(vec![]).is_err());
    }

    #[test]
    fn nonfinite_projection_is_refused_not_silently_clamped() {
        // Huge-but-valid scales make the up projection overflow f32 on a large activation.
        let mut rng = Lcg(9);
        let mut owned = small_layer(&mut rng);
        owned.experts[0].0.scales.fill(MAX_VALID_SCALE_CODE);
        owned.experts[0].0.wss2 = 1e30;
        let layer = owned.layer();
        let p = pool(2, WaitPolicy::Park, Isa::detect());
        let picks = [ExpertPick {
            expert: 0,
            weight: 1.0,
        }];
        let x = vec![1.0e6f32; 32];
        expect_err(
            p.routed_experts(&layer, &x, &picks, &mut [0.0; 32]),
            "Weights",
        );
    }

    // ----- pool -------------------------------------------------------------------------------

    #[test]
    fn every_wait_policy_serves_many_calls_and_shuts_down() {
        let mut rng = Lcg(12);
        let owned = OwnedLayer::random(&mut rng, 6, 64, 96);
        let layer = owned.layer();
        let x = random_x(&mut rng, 64);
        let picks = picks_for(&mut rng, 6, 3);
        let truth = owned.reference_f64(&x, &picks);
        for wait in [
            WaitPolicy::Park,
            WaitPolicy::SpinThenPark(Duration::from_micros(200)),
            WaitPolicy::Spin,
        ] {
            let p = pool(3, wait, Isa::detect());
            for round in 0..40 {
                if round % 10 == 9 {
                    // Let the workers park (or keep spinning) between bursts.
                    std::thread::sleep(Duration::from_millis(2));
                }
                let mut out = vec![0.0f32; 64];
                p.routed_experts(&layer, &x, &picks, &mut out).unwrap();
                assert_close(&out, &truth, &format!("{wait:?} round {round}"));
            }
            drop(p);
        }
    }

    #[test]
    fn concurrent_callers_serialize_without_corruption() {
        let mut rng = Lcg(14);
        let owned = OwnedLayer::random(&mut rng, 6, 64, 96);
        let layer = owned.layer();
        let p = pool(3, WaitPolicy::Park, Isa::detect());
        let cases: Vec<(Vec<f32>, Vec<ExpertPick>)> = (0..4)
            .map(|_| (random_x(&mut rng, 64), picks_for(&mut rng, 6, 3)))
            .collect();
        std::thread::scope(|scope| {
            for (x, picks) in &cases {
                let (owned, layer, p) = (&owned, &layer, &p);
                scope.spawn(move || {
                    let truth = owned.reference_f64(x, picks);
                    for _ in 0..20 {
                        let mut out = vec![0.0f32; 64];
                        p.routed_experts(layer, x, picks, &mut out).unwrap();
                        assert_close(&out, &truth, "concurrent");
                    }
                });
            }
        });
    }

    #[test]
    fn more_threads_than_chunks_is_fine() {
        let mut rng = Lcg(15);
        let owned = OwnedLayer::random(&mut rng, 2, 32, 32);
        let layer = owned.layer();
        let x = random_x(&mut rng, 32);
        let picks = picks_for(&mut rng, 2, 1);
        let p = pool(16, WaitPolicy::Park, Isa::detect());
        let mut out = vec![0.0f32; 32];
        p.routed_experts(&layer, &x, &picks, &mut out).unwrap();
        assert_close(
            &out,
            &owned.reference_f64(&x, &picks),
            "16 threads, 2 chunks",
        );
    }
}
