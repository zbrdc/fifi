//! Safe launch wrapper of the fused residual add, RMSNorm and fp16 pack: `h += delta` in place
//! (optional), then `norm = h * rsqrt(mean(h^2) + eps) * weight` rounded to nearest even into
//! the matmul input `xf`, and, when asked, the f32 row `norm_out` that `xf` rounds.
//!
//! [`CudaContext::add_norm_pack`] queues one kernel on a [`DeviceStream`] and returns. The call
//! allocates nothing, copies nothing and never
//! synchronizes; a fault in the queued work surfaces at the next synchronize of the stream and
//! latches the backend there. The caller owns the order: the writer of `delta` must be queued on
//! the same stream before this call, and `delta` must not share bytes with `h`.
//!
//! Layouts: `h`, `delta`, `norm_out` are f32 `[m][n]`, token-major; `weight` is f32 `[n]`. `xf` is
//! the layout `pack` writes: the fixed-size fragment-order buffer (1,048,592 bytes, of which the
//! kernel writes the data lanes of the `m` rows and the zero tail entry the padding lanes read; the
//! bytes no consumer reads are left untouched) for `m <= 16`, fp16 `[m][n]` row-major above.
//! `norm_out` must be 16-byte aligned because the router reads it as its input row.
//!
//! Numerics: the f32 sum of squares runs in a fixed order (no atomics), so a launch is
//! deterministic. Outside the claimed domain (finite inputs, `|h| <= 2^40`) a row whose sum of
//! squares overflows f32 gives `rstd = 0` and therefore finite zeros in `norm_out` and `xf`, not
//! NaN, and nothing flags it; a non-finite input gives a non-finite output, which the router and
//! the argmax flag.
//!
//! Every argument is checked here, before any native call, in this order: the terminal flag, the
//! shape and `eps`, size and alignment (`require_fit`), disjointness (`require_apart`), then
//! `raw_range` for each range, all before `launch_on`. So on a latched context every call answers
//! the latch error and the native entry never sees a bad argument (it would latch the backend).

use std::ffi::c_int;

use crate::EngError;
use crate::kernels::cuda::context::{Claim, require_apart, require_fit};
use crate::kernels::cuda::native::resident::add_norm_pack::{
    AddNormInts, AddNormRaw, add_norm_pack_launch,
};
use crate::kernels::cuda::native::xf_bytes;
use crate::kernels::cuda::{CudaContext, DeviceRange, DeviceStream};

/// The most tokens one call takes.
pub(crate) const MAX_TOKENS: usize = 32_768;
/// The widest row one call takes.
pub(crate) const MAX_WIDTH: usize = 32_768;
/// `n` is a multiple of this.
pub(crate) const NORM_WIDTH_UNIT: usize = 256;
/// Limits as C integers, for the checks on the converted values.
const MAX_TOKENS_C: c_int = MAX_TOKENS as c_int;
const MAX_WIDTH_C: c_int = MAX_WIDTH as c_int;
const UNIT_C: c_int = NORM_WIDTH_UNIT as c_int;

/// One fused add, norm and pack call. All ranges are device ranges of this context.
///
/// - `h`: f32 `[m][n]`, 4-byte aligned; read, and written in place with `h + delta` when `delta`
///   is given. It is the new residual.
/// - `delta`: `None`, or f32 `[m][n]`, 4-byte aligned, read only. It may share bytes with
///   `weight` and with nothing else.
/// - `weight`: f32 `[n]`, 4-byte aligned, read only.
/// - `xf`: [`xf_bytes`]`(m, n)` bytes (1,048,592 for `m <= 16`, `m * n * 2` above), 16-byte
///   aligned; the data lanes of the `m` rows and the zero tail entry are written (`m <= 16`), the
///   whole buffer above.
/// - `norm_out`: `None`, or f32 `[m][n]`, 16-byte aligned; the value that `xf` rounds.
/// - `eps`: finite and above 0 (GGUF `layer_norm_rms_epsilon`).
pub(crate) struct AddNormCall<'b> {
    pub(crate) m: usize,
    pub(crate) n: usize,
    pub(crate) eps: f32,
    pub(crate) h: DeviceRange<'b>,
    pub(crate) delta: Option<DeviceRange<'b>>,
    pub(crate) weight: DeviceRange<'b>,
    pub(crate) xf: DeviceRange<'b>,
    pub(crate) norm_out: Option<DeviceRange<'b>>,
}

/// A `BadRequest` that names the rule that fired.
fn rule(tag: &str, detail: &str) -> EngError {
    EngError::BadRequest(format!("add_norm_pack: {tag}: {detail}"))
}

/// `value` as a C integer in `low..=high`, or the refusal for `tag`. The range is tested on the
/// converted value, so a `usize` that wraps into range is refused too.
fn ranged(tag: &str, value: usize, low: c_int, high: c_int) -> Result<c_int, EngError> {
    let refused = || rule(tag, &format!("{tag} is {value}, not {low} to {high}"));
    let int = c_int::try_from(value).map_err(|_| refused())?;
    if (low..=high).contains(&int) {
        Ok(int)
    } else {
        Err(refused())
    }
}

/// Checks `m` and `n` and converts them once, so that no conversion can fail inside the launch
/// closure. Each refusal carries the tag `add_norm_pack: <rule>`.
pub(in crate::kernels::cuda) fn check_shape(m: usize, n: usize) -> Result<AddNormInts, EngError> {
    let m = ranged("m", m, 1, MAX_TOKENS_C)?;
    let n = ranged("n", n, UNIT_C, MAX_WIDTH_C)?;
    if n % UNIT_C != 0 {
        return Err(rule(
            "n",
            &format!("n is {n}, not a multiple of {NORM_WIDTH_UNIT}"),
        ));
    }
    Ok(AddNormInts { m, n })
}

/// Checks `eps`: finite and above 0.
pub(crate) fn check_eps(eps: f32) -> Result<(), EngError> {
    if eps.is_finite() && eps > 0.0 {
        Ok(())
    } else {
        Err(rule("eps", "eps is not finite and above 0"))
    }
}

/// Bytes of `h`, `delta` and `norm_out`: `m` rows of `n` f32, or `None` on overflow.
pub(crate) fn row_bytes(m: usize, n: usize) -> Option<usize> {
    m.checked_mul(n)?.checked_mul(size_of::<f32>())
}

/// The bytes each range must hold; only built for a shape that passed [`check_shape`].
struct Sizes {
    row: usize,
    weight: usize,
    xf: usize,
}

fn sizes(m: usize, n: usize) -> Result<Sizes, EngError> {
    let overflow = || rule("shape", "the buffer sizes overflow");
    Ok(Sizes {
        row: row_bytes(m, n).ok_or_else(overflow)?,
        weight: n.checked_mul(size_of::<f32>()).ok_or_else(overflow)?,
        xf: xf_bytes(m, n).ok_or_else(overflow)?,
    })
}

/// Size and alignment of each range, one `require_fit` per range: `xf` and `norm_out` 16, the
/// rest 4.
fn check_fit(call: &AddNormCall<'_>, need: &Sizes) -> Result<(), EngError> {
    require_fit("h", &call.h, need.row, 4)?;
    if let Some(delta) = &call.delta {
        require_fit("delta", delta, need.row, 4)?;
    }
    require_fit("weight", &call.weight, need.weight, 4)?;
    require_fit("xf", &call.xf, need.xf, 16)?;
    if let Some(out) = &call.norm_out {
        require_fit("norm_out", out, need.row, 16)?;
    }
    Ok(())
}

/// Disjointness of the touched extents: the written ranges (`h`, `xf`, `norm_out`) pairwise (3
/// pairs) and each against the read ranges (`delta`, `weight`; up to 6 pairs). `delta` and
/// `weight` may overlap each other. `h` counts as written even without `delta`, as in the shim.
fn check_apart(call: &AddNormCall<'_>, need: &Sizes) -> Result<(), EngError> {
    let mut written = vec![
        Claim::of("h", &call.h, need.row),
        Claim::of("xf", &call.xf, need.xf),
    ];
    written.extend(
        call.norm_out
            .as_ref()
            .map(|out| Claim::of("norm_out", out, need.row)),
    );
    let mut read = vec![Claim::of("weight", &call.weight, need.weight)];
    read.extend(
        call.delta
            .as_ref()
            .map(|delta| Claim::of("delta", delta, need.row)),
    );
    for (i, claim) in written.iter().enumerate() {
        for other in written[i + 1..].iter().chain(read.iter()) {
            require_apart(claim, other)?;
        }
    }
    Ok(())
}

impl CudaContext {
    /// Queues the fused residual add, RMSNorm and fp16 pack of `call` on `stream`: `h += delta` in
    /// place when `delta` is given, the f32 normalised row into `norm_out` when given, and the fp16
    /// matmul input into `xf` (fragment order for `m <= 16`, where the kernel also writes the zero
    /// tail entry the padding lanes read and leaves the unread bytes untouched; row-major and fully
    /// written above). Returns when the work is queued; the outputs are valid after the stream
    /// drained.
    ///
    /// The caller owns the order: queue the writer of `delta` on `stream` before this call, and
    /// never pass the same bytes as `delta` and `h`. A finite row whose sum of squares overflows
    /// f32 gives zeros, not NaN (see the module doc).
    ///
    /// Returns the latch error on a terminal backend, and `BadRequest`, before any native call
    /// and with the backend unchanged, for a stream or buffer of another context, a shape or
    /// `eps` the kernel refuses, a range that is too short or misaligned, and a written range
    /// that overlaps another range of the call.
    pub(crate) fn add_norm_pack(
        &self,
        stream: &DeviceStream,
        call: AddNormCall<'_>,
    ) -> Result<(), EngError> {
        self.ensure_usable()?;
        let ints = check_shape(call.m, call.n)?;
        check_eps(call.eps)?;
        let need = sizes(call.m, call.n)?;
        check_fit(&call, &need)?;
        check_apart(&call, &need)?;
        let h = self.raw_range(&call.h)?;
        let weight = self.raw_range(&call.weight)?;
        let xf = self.raw_range(&call.xf)?;
        let delta = call.delta.as_ref().map(|r| self.raw_range(r)).transpose()?;
        let norm_out = call
            .norm_out
            .as_ref()
            .map(|r| self.raw_range(r))
            .transpose()?;
        let eps = call.eps;
        self.launch_on(stream, move |raw| {
            add_norm_pack_launch(
                raw,
                AddNormRaw {
                    h,
                    weight,
                    xf,
                    delta,
                    norm_out,
                    ints,
                    eps,
                },
            )
        })
    }
}

#[cfg(all(test, feature = "native-cuda"))]
mod tests;
