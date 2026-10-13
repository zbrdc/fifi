//! The MoE combine of the resident decode step: the residual plus the picks summed in pick order
//! plus the optional gated shared term, in one launch (decision 0047, stage 1, RD-4e).
//!
//! Design guidance only, no code copied. Ideas from MegaCapybara v1.35 static RE,
//! the MegaCapybara static-RE lessons, C6 (a small separate combine kernel in
//! stage 1; fusion into the expert kernel is RD-6b) and C8 (sum the picks in a fixed order in
//! registers so that the result does not depend on scheduling).
//!
//! [`CudaContext::moe_combine`] queues one kernel on a [`DeviceStream`] and returns. It allocates
//! nothing, copies nothing and never synchronizes; a fault in the queued work surfaces at the next
//! synchronize of the stream. The caller owns the order: `down`, `weights`, the shared pair and
//! `res` must hold their data before the call is queued (an event or the same stream), and `out`
//! is read only behind the stream.
//!
//! For every token `t` and column `j`, each operation rounded alone (no fused multiply-add) and in
//! this order: `s = w[t][0] * down[t][0][j]` (the sum starts at the first product, so a `-0`
//! stays `-0` as in ggml); `s = s + w[t][i] * down[t][i][j]` for `i = 1 .. top_k - 1`;
//! `s = s + shared_gate[t] * shared[t][j]` when the shared pair is given; and
//! `out[t][j] = res[t][j] + s`. This is the association of the pinned llama.cpp graph,
//! `res + (S + g * shared)` with `S` the pick-order sum.
//!
//! The weights and the gate are applied as RD-5 wrote them: the router has already applied the
//! softmax, the renormalization and the sigmoid of the shared gate (and, for a family that has
//! one, the expert-weights scale). The combine applies no second sigmoid, no softmax and no
//! renormalization, tests no weight for zero and clamps nothing.
//!
//! NaN is preserved by IEEE propagation. A faulty router row has ids -1, weights 0 and gate 0;
//! `matmul_expert` writes NaN to every `down` row of a bad id, and `0 * NaN` is NaN, so the whole
//! output row is NaN. The limit: the combine trusts the NaN of `matmul_expert`. A faulty row whose
//! `down` rows are finite (a caller that skipped `matmul_expert`) gives `res`.
//!
//! In place: [`CombineOut::InPlace`] writes `res`. Each thread reads and writes only its own four
//! columns, so the result equals the out-of-place one byte for byte.
//!
//! Every refusal is a `BadRequest` and happens before any native call, because a rejection from the
//! native layer latches the backend terminal. In order: the backend is usable; the stream and every
//! range belong to this context; the shape (see [`check_shape`]); each range holds what the call
//! uses and starts at its alignment; the range the call writes overlaps no input. The native entry
//! and the C entry repeat these checks. A refused call changes nothing.

use std::ffi::c_int;

use crate::EngError;
use crate::kernels::cuda::context::{
    Claim, CudaContext, DeviceRange, DeviceStream, require_apart, require_fit,
};
use crate::kernels::cuda::native::resident::moe_combine::{CombineRaw, moe_combine_launch};

const MAX_TOKENS: usize = 4096;
const MAX_N: usize = 65536;
const MAX_TOP_K: usize = 32;
const MAX_LD_DOWN: usize = 1 << 26;

/// The sizes of one combine call. Qwen3.6 is `m` 1, `n` 2048 (the hidden size), `top_k` 8 and
/// `ld_down` 2048.
#[derive(Debug, Clone, Copy)]
pub(crate) struct CombineShape {
    /// Token rows, 1 to 4096.
    pub(crate) m: usize,
    /// Width of a row, a multiple of 4 from 4 to 65536.
    pub(crate) n: usize,
    /// Picks per token, 1 to 32.
    pub(crate) top_k: usize,
    /// Floats from the start of one `down` row to the next: `n` to 2^26, a multiple of 4.
    pub(crate) ld_down: usize,
}

/// The gated shared-expert term: a gate and a row, given together or not at all.
pub(crate) struct SharedTerm<'b> {
    /// f32 `[m]`, 4-byte aligned: RD-5's `shared_gate`, already a sigmoid, used as given.
    pub(crate) gate: DeviceRange<'b>,
    /// f32 `[m][n]`, 16-byte aligned: the shared expert's down output.
    pub(crate) row: DeviceRange<'b>,
}

/// Where the sum goes.
pub(crate) enum CombineOut<'b> {
    /// Into `res`.
    InPlace,
    /// Into this f32 `[m][n]` range, 16-byte aligned, which shares no byte with an input.
    To(DeviceRange<'b>),
}

/// One combine call.
pub(crate) struct CombineCall<'b> {
    /// The sizes of the call.
    pub(crate) shape: CombineShape,
    /// f32 `[m][n]`, 16-byte aligned: the residual stream.
    pub(crate) res: DeviceRange<'b>,
    /// f32 rows `t * top_k + i` of `n` values at a stride of `ld_down` floats, 16-byte aligned:
    /// the `y` rows of `top_k` `matmul_expert` calls in pick (rank) order.
    pub(crate) down: DeviceRange<'b>,
    /// f32 `[m][top_k]`, 4-byte aligned, rank order: RD-5's `weights`, used as given.
    pub(crate) weights: DeviceRange<'b>,
    /// The gated shared term, or `None` for a model without a shared expert.
    pub(crate) shared: Option<SharedTerm<'b>>,
    /// Where the result goes.
    pub(crate) out: CombineOut<'b>,
}

/// The bytes each range of a call must hold.
struct Need {
    rows: usize,
    down: usize,
    weights: usize,
    gate: usize,
}

fn overflow() -> EngError {
    EngError::BadRequest("the CUDA combine sizes overflow".to_owned())
}

fn product(factors: &[usize]) -> Result<usize, EngError> {
    factors
        .iter()
        .try_fold(1usize, |acc, factor| acc.checked_mul(*factor))
        .ok_or_else(overflow)
}

/// The bytes of `res`, of `out` and of the shared row: `4 * m * n`. Returns `BadRequest` when the
/// product overflows.
pub(crate) fn combine_row_bytes(shape: &CombineShape) -> Result<usize, EngError> {
    product(&[shape.m, shape.n, 4])
}

/// The bytes of `down`: `4 * ((m * top_k - 1) * ld_down + n)`, the last row without its stride
/// padding. Returns `BadRequest` when the product overflows or `m * top_k` is 0.
pub(crate) fn combine_down_bytes(shape: &CombineShape) -> Result<usize, EngError> {
    let last_row = product(&[shape.m, shape.top_k])?
        .checked_sub(1)
        .ok_or_else(overflow)?;
    product(&[last_row, shape.ld_down])?
        .checked_add(shape.n)
        .and_then(|floats| floats.checked_mul(4))
        .ok_or_else(overflow)
}

fn needs(shape: &CombineShape) -> Result<Need, EngError> {
    Ok(Need {
        rows: combine_row_bytes(shape)?,
        down: combine_down_bytes(shape)?,
        weights: product(&[shape.m, shape.top_k, 4])?,
        gate: product(&[shape.m, 4])?,
    })
}

/// `BadRequest` unless `m` is in `1..=4096`, `n` a multiple of 4 in `4..=65536`, `top_k` in
/// `1..=32` and `ld_down` a multiple of 4 in `n..=2^26`. Pure: no device is needed.
pub(crate) fn check_shape(shape: &CombineShape) -> Result<(), EngError> {
    let CombineShape {
        m,
        n,
        top_k,
        ld_down,
    } = *shape;
    let inside = (1..=MAX_TOKENS).contains(&m)
        && (4..=MAX_N).contains(&n)
        && n.is_multiple_of(4)
        && (1..=MAX_TOP_K).contains(&top_k)
        && (n..=MAX_LD_DOWN).contains(&ld_down)
        && ld_down.is_multiple_of(4);
    if inside {
        return Ok(());
    }
    Err(EngError::BadRequest(format!(
        "the CUDA combine takes m 1 to {MAX_TOKENS}, n a multiple of 4 from 4 to {MAX_N}, top_k 1 \
         to {MAX_TOP_K} and ld_down a multiple of 4 from n to {MAX_LD_DOWN}; got m {m}, n {n}, \
         top_k {top_k}, ld_down {ld_down}"
    )))
}

fn check_fit(call: &CombineCall<'_>, need: &Need) -> Result<(), EngError> {
    require_fit("res", &call.res, need.rows, 16)?;
    require_fit("down", &call.down, need.down, 16)?;
    require_fit("weights", &call.weights, need.weights, 4)?;
    if let Some(shared) = &call.shared {
        require_fit("shared_gate", &shared.gate, need.gate, 4)?;
        require_fit("shared", &shared.row, need.rows, 16)?;
    }
    if let CombineOut::To(out) = &call.out {
        require_fit("out", out, need.rows, 16)?;
    }
    Ok(())
}

/// The range the launch writes (`out`, or `res` in place) against every input, the written one
/// named first in the refusal.
fn check_apart(call: &CombineCall<'_>, need: &Need) -> Result<(), EngError> {
    let res = Claim::of("res", &call.res, need.rows);
    let mut inputs = vec![
        Claim::of("down", &call.down, need.down),
        Claim::of("weights", &call.weights, need.weights),
    ];
    if let Some(shared) = &call.shared {
        inputs.push(Claim::of("shared_gate", &shared.gate, need.gate));
        inputs.push(Claim::of("shared", &shared.row, need.rows));
    }
    let written = match &call.out {
        CombineOut::To(out) => {
            let written = Claim::of("out", out, need.rows);
            require_apart(&written, &res)?;
            written
        }
        CombineOut::InPlace => res,
    };
    inputs
        .iter()
        .try_for_each(|input| require_apart(&written, input))
}

fn to_int(name: &str, value: usize) -> Result<c_int, EngError> {
    c_int::try_from(value)
        .map_err(|_| EngError::BadRequest(format!("the CUDA combine {name} exceeds i32")))
}

impl CudaContext {
    /// Queues `out = res + ((w_0 d_0 + ... + w_(k-1) d_(k-1)) + g * shared)` on `stream`, with
    /// the weights and the gate as given. See the module doc for the sequence and the NaN rule.
    ///
    /// Returns `BadRequest` for a stream or buffer of another adapter, a shape outside
    /// [`check_shape`], a range that is short or misaligned, and an `out` that overlaps an input.
    pub(crate) fn moe_combine(
        &self,
        stream: &DeviceStream,
        call: CombineCall<'_>,
    ) -> Result<(), EngError> {
        self.ensure_usable()?;
        let raw = self.plan_combine(&call)?;
        self.launch_on(stream, |stream_raw| moe_combine_launch(stream_raw, &raw))
    }

    /// Every check of a combine call, then the native view of its ranges and ints.
    fn plan_combine<'b>(&self, call: &CombineCall<'b>) -> Result<CombineRaw<'b>, EngError> {
        let raw = |range: &DeviceRange<'b>| self.raw_range(range);
        let res = raw(&call.res)?;
        let out = match &call.out {
            CombineOut::To(out) => raw(out)?,
            CombineOut::InPlace => res,
        };
        let (down, weights) = (raw(&call.down)?, raw(&call.weights)?);
        let (shared_gate, shared) = match &call.shared {
            Some(term) => (Some(raw(&term.gate)?), Some(raw(&term.row)?)),
            None => (None, None),
        };
        check_shape(&call.shape)?;
        let need = needs(&call.shape)?;
        check_fit(call, &need)?;
        check_apart(call, &need)?;
        let CombineShape {
            m,
            n,
            top_k,
            ld_down,
        } = call.shape;
        Ok(CombineRaw {
            res,
            down,
            weights,
            shared_gate,
            shared,
            out,
            m: to_int("m", m)?,
            n: to_int("n", n)?,
            top_k: to_int("top_k", top_k)?,
            ld_down: to_int("ld_down", ld_down)?,
        })
    }
}

#[cfg(test)]
mod tests;
