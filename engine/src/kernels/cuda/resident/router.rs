//! The device router of the resident decode step: logits against the F32 router matrix, softmax,
//! the top `k` probabilities, weights renormalized over the selected set, the sigmoid of the
//! shared-expert gate (when the model has one) and a status bit for a faulty row.
//!
//! Routing runs on the device so that the expert ids never reach the host; the expert kernel
//! reads them from device memory.
//!
//! [`CudaContext::router`] queues two kernels on a [`DeviceStream`] and returns. It allocates
//! nothing, copies nothing and never synchronizes; a fault in the queued work surfaces at the next
//! synchronize of the stream and latches the backend there. The caller owns the order: `x` must
//! hold its data before the call is queued, and `ids`, `weights`, `shared_gate` and `status` are
//! read only behind the stream (an event, or a synchronize).
//!
//! The rule is the one of `route_experts` with `Router::SoftmaxTopK { renorm: true }`:
//! `p = softmax(x . W^T)` over all `n_expert` logits, rank by `p` descending and then by the
//! lower expert index, `weights = p / (sum of the selected p in rank order)`, and
//! `shared_gate = sigmoid(dot(x, w_shared))`. The device does the exp and the sums in f32 with a
//! fixed order, so the values equal the host rule within the class R1 tolerance, not bit for bit.
//! The order of the logits does not depend on the launch: a call of 8 or more rows (a prefill
//! chunk) shares each load of the router matrix across a tile of tokens, a smaller one (decode)
//! uses one warp per dot product, and the two give the same logit bits for every row (but for the
//! sign of an exact zero when `hidden / 4` is not a multiple of 512, see `router.cu`).
//!
//! Fault convention: a row is faulty when a logit is not finite, when the softmax denominator or
//! the selected sum is not a finite positive number, or when the gate dot is not finite. The
//! faulty row gets ids -1 in every slot (an id outside `[0, n_expert)` makes the expert kernel
//! write NaN), weights 0, gate 0, and [`ROUTER_STATUS_BIT`] is OR-ed into `status`. The router
//! never clears `status`; the step start zeroes it. A router fault read after an invalid step
//! state (status mask 4) is not a model fault.
//!
//! Every range is a checked [`DeviceRange`]. Before any native call the wrapper proves that the
//! stream and every buffer belong to this context, the shape limits, that each range holds the
//! bytes the call uses and is aligned, and that the ranges the launch writes overlap no other
//! range of the call. `native::resident::router` and the C entry repeat these checks, so a
//! rejection is always a `BadRequest` here and never a terminal adapter failure. A refused call
//! changes nothing.
#![cfg_attr(
    not(test),
    expect(dead_code, reason = "staged for the resident decode step")
)]

use crate::EngError;
use crate::kernels::cuda::context::{
    Claim, CudaContext, DeviceRange, DeviceStream, require_apart, require_fit,
};
use crate::kernels::cuda::native::resident::RawRange;
use crate::kernels::cuda::native::resident::router::{
    RouterInts, RouterLogitsRaw, RouterRaw, RouterSigmoidRaw, router_launch, router_logits_launch,
    router_sigmoid_launch,
};

/// The bit of `status` that the router sets when any row is faulty.
pub(crate) const ROUTER_STATUS_BIT: i32 = 1;

const MAX_ROWS: usize = 4096;
const MAX_HIDDEN: usize = 65536;
const MAX_EXPERTS: usize = 256;
const MAX_TOP_K: usize = 32;

/// The sizes of one router call. Qwen3.6 is `hidden` 2048, `n_expert` 256, `top_k` 8, `m` 1.
#[derive(Debug, Clone, Copy)]
pub(crate) struct RouterShape {
    /// Token rows, 1 to 4096.
    pub(crate) m: usize,
    /// Width of a row of `x` and of the router matrix, a multiple of 4 from 4 to 65536.
    pub(crate) hidden: usize,
    /// Experts, 1 to 256.
    pub(crate) n_expert: usize,
    /// Experts chosen per row, 1 to the smaller of `n_expert` and 32.
    pub(crate) top_k: usize,
}

/// The router weights, both f32 and 16-byte aligned.
pub(crate) struct RouterWeights<'b> {
    /// `[n_expert][hidden]`, expert-major (GGUF `ffn_gate_inp`).
    pub(crate) w: DeviceRange<'b>,
    /// `[hidden]` (GGUF `ffn_gate_inp_shexp`), or `None` for a model without a shared expert.
    /// Given together with `shared_gate`, or not at all.
    pub(crate) w_shared: Option<DeviceRange<'b>>,
}

/// What the router writes.
pub(crate) struct RouterOut<'b> {
    /// i32 `[m][top_k]` in rank order; -1 in every slot of a faulty row.
    pub(crate) ids: DeviceRange<'b>,
    /// f32 `[m][top_k]`, `p_i / sum(p)` over the selected set; 0 on a faulty row.
    pub(crate) weights: DeviceRange<'b>,
    /// f32 `[m]`, `sigmoid(dot(x, w_shared))`; 0 on a faulty row. `Some` exactly when
    /// `w_shared` is.
    pub(crate) shared_gate: Option<DeviceRange<'b>>,
    /// One device i32 word; [`ROUTER_STATUS_BIT`] is OR-ed in when any row is faulty. At least 4
    /// bytes, 4-aligned.
    pub(crate) status: DeviceRange<'b>,
}

/// One router call.
pub(crate) struct RouterCall<'b> {
    /// The sizes of the call.
    pub(crate) shape: RouterShape,
    /// f32 `[m][hidden]`, 16-byte aligned: the normalized row, not its fp16 pack.
    pub(crate) x: DeviceRange<'b>,
    /// The router weights.
    pub(crate) weights: RouterWeights<'b>,
    /// f32 scratch `[m][n_expert + 1]` ([`router_logits_bytes`] bytes); the last column holds the
    /// shared-gate dot.
    pub(crate) logits: DeviceRange<'b>,
    /// The outputs.
    pub(crate) out: RouterOut<'b>,
}

/// The bytes each range of a call must hold.
struct Bytes {
    x: usize,
    w: usize,
    w_shared: usize,
    logits: usize,
    picks: usize,
    gate: usize,
}

fn overflow() -> EngError {
    EngError::BadRequest("the CUDA router sizes overflow".to_owned())
}

fn bytes_of(shape: &RouterShape) -> Result<Bytes, EngError> {
    let product = |factors: &[usize]| {
        factors
            .iter()
            .try_fold(1usize, |acc, factor| acc.checked_mul(*factor))
            .ok_or_else(overflow)
    };
    let columns = shape.n_expert.checked_add(1).ok_or_else(overflow)?;
    Ok(Bytes {
        x: product(&[shape.m, shape.hidden, 4])?,
        w: product(&[shape.n_expert, shape.hidden, 4])?,
        w_shared: product(&[shape.hidden, 4])?,
        logits: product(&[shape.m, columns, 4])?,
        picks: product(&[shape.m, shape.top_k, 4])?,
        gate: product(&[shape.m, 4])?,
    })
}

/// The bytes of the logits scratch for `shape`: `4 * m * (n_expert + 1)`. 1,028 at Qwen3.6 with
/// `m` 1. Returns `BadRequest` when the product overflows.
pub(crate) fn router_logits_bytes(shape: RouterShape) -> Result<usize, EngError> {
    bytes_of(&shape).map(|bytes| bytes.logits)
}

fn check_shape(shape: &RouterShape) -> Result<(), EngError> {
    let RouterShape {
        m,
        hidden,
        n_expert,
        top_k,
    } = *shape;
    let inside = (1..=MAX_ROWS).contains(&m)
        && hidden.is_multiple_of(4)
        && (4..=MAX_HIDDEN).contains(&hidden)
        && (1..=MAX_EXPERTS).contains(&n_expert)
        && (1..=n_expert.min(MAX_TOP_K)).contains(&top_k);
    if inside {
        return Ok(());
    }
    Err(EngError::BadRequest(format!(
        "the CUDA router takes m 1 to {MAX_ROWS}, hidden a multiple of 4 up to {MAX_HIDDEN}, \
         n_expert 1 to {MAX_EXPERTS} and top_k 1 to min(n_expert, {MAX_TOP_K}); \
         got m {m}, hidden {hidden}, n_expert {n_expert}, top_k {top_k}"
    )))
}

fn check_pair(call: &RouterCall<'_>) -> Result<(), EngError> {
    if call.weights.w_shared.is_some() == call.out.shared_gate.is_some() {
        return Ok(());
    }
    Err(EngError::BadRequest(
        "the CUDA router needs w_shared and shared_gate together or neither".to_owned(),
    ))
}

fn check_fit(call: &RouterCall<'_>, need: &Bytes) -> Result<(), EngError> {
    require_fit("x", &call.x, need.x, 16)?;
    require_fit("w", &call.weights.w, need.w, 16)?;
    if let Some(w_shared) = &call.weights.w_shared {
        require_fit("w_shared", w_shared, need.w_shared, 16)?;
    }
    require_fit("logits", &call.logits, need.logits, 4)?;
    require_fit("ids", &call.out.ids, need.picks, 4)?;
    require_fit("weights", &call.out.weights, need.picks, 4)?;
    if let Some(gate) = &call.out.shared_gate {
        require_fit("shared_gate", gate, need.gate, 4)?;
    }
    require_fit("status", &call.out.status, 4, 4)
}

/// Every range the launch writes (and the scratch) against the other outputs and the inputs, the
/// written one named first in the refusal.
fn check_apart<'b>(call: &RouterCall<'b>, need: &Bytes) -> Result<(), EngError> {
    let mut outputs = vec![
        Claim::of("ids", &call.out.ids, need.picks),
        Claim::of("weights", &call.out.weights, need.picks),
    ];
    if let Some(gate) = &call.out.shared_gate {
        outputs.push(Claim::of("shared_gate", gate, need.gate));
    }
    outputs.push(Claim::of("logits", &call.logits, need.logits));
    outputs.push(Claim::of("status", &call.out.status, 4));
    let mut inputs = vec![
        Claim::of("x", &call.x, need.x),
        Claim::of("w", &call.weights.w, need.w),
    ];
    if let Some(w_shared) = &call.weights.w_shared {
        inputs.push(Claim::of("w_shared", w_shared, need.w_shared));
    }
    for (index, output) in outputs.iter().enumerate() {
        for input in &inputs {
            require_apart(output, input)?;
        }
        for other in &outputs[index + 1..] {
            require_apart(output, other)?;
        }
    }
    Ok(())
}

fn to_int(name: &str, value: usize) -> Result<std::ffi::c_int, EngError> {
    std::ffi::c_int::try_from(value)
        .map_err(|_| EngError::BadRequest(format!("the CUDA router {name} exceeds i32")))
}

impl CudaContext {
    /// Every check of a router call, then the native view of its ranges and ints.
    fn plan_router<'b>(&self, call: &RouterCall<'b>) -> Result<RouterRaw<'b>, EngError> {
        let raw = |range: &DeviceRange<'b>| self.raw_range(range);
        let optional = |range: &Option<DeviceRange<'b>>| -> Result<Option<RawRange<'b>>, EngError> {
            range.as_ref().map(raw).transpose()
        };
        let ranges = (
            raw(&call.x)?,
            raw(&call.weights.w)?,
            optional(&call.weights.w_shared)?,
            raw(&call.logits)?,
        );
        let outs = (
            raw(&call.out.ids)?,
            raw(&call.out.weights)?,
            optional(&call.out.shared_gate)?,
            raw(&call.out.status)?,
        );
        check_shape(&call.shape)?;
        check_pair(call)?;
        let need = bytes_of(&call.shape)?;
        check_fit(call, &need)?;
        check_apart(call, &need)?;
        let ints = RouterInts {
            m: to_int("m", call.shape.m)?,
            hidden: to_int("hidden", call.shape.hidden)?,
            n_expert: to_int("n_expert", call.shape.n_expert)?,
            top_k: to_int("top_k", call.shape.top_k)?,
        };
        Ok(RouterRaw {
            x: ranges.0,
            w: ranges.1,
            w_shared: ranges.2,
            logits: ranges.3,
            ids: outs.0,
            weights: outs.1,
            shared_gate: outs.2,
            status: outs.3,
            ints,
        })
    }

    /// Queues the router of `call` on `stream`: the logits kernel, then the selection kernel (a
    /// warp per token row: softmax, the top `top_k` by probability then lower index,
    /// renormalization, the gate, the fault check). Returns when both are queued; it does not
    /// wait. The logits kernel is chosen by `shape.m`: below 8 rows (decode) a warp per row of the
    /// router matrix and per token, and one for `w_shared`; from 8 rows (a prefill chunk) a tiled
    /// kernel in which a warp takes 4 rows of the matrix against 8 tokens. Both sum every dot
    /// product in one fixed order, so the logits do not depend on the choice.
    ///
    /// The caller guarantees that `x` holds its data in stream order, that the device memory of
    /// every range stays alive until the stream has drained, and that nothing else reads or writes
    /// the outputs or the scratch until then. A faulty row is reported by ids -1, weights 0, gate 0
    /// and [`ROUTER_STATUS_BIT`] in `status`, not by an error.
    ///
    /// Returns `BadRequest` for a stream or buffer of another adapter, a shape outside the limits
    /// of [`RouterShape`], `w_shared` and `shared_gate` that are not given together, a range that
    /// is too short or misaligned (x, w and w_shared 16; the rest 4), and an output, the scratch
    /// or `status` that overlaps another range of the call.
    pub(crate) fn router(
        &self,
        stream: &DeviceStream,
        call: RouterCall<'_>,
    ) -> Result<(), EngError> {
        self.ensure_usable()?;
        let args = self.plan_router(&call)?;
        self.launch_on(stream, move |raw| router_launch(raw, args))
    }
}

/// Split router dot product. Input and weight rows are BF16-precision F32 storage;
/// output is tightly packed F32 [m,n_expert], for shared BF16 publication next.
pub(crate) struct RouterLogitsCall<'a> {
    pub(crate) shape: RouterShape,
    pub(crate) x: DeviceRange<'a>,
    pub(crate) w: DeviceRange<'a>,
    pub(crate) logits: DeviceRange<'a>,
}

/// Sigmoid top-k with selected-only F32 normalization. No shared gate or bias.
/// Logits must already have passed the caller's shared BF16 publication.
pub(crate) struct RouterSigmoidCall<'a> {
    pub(crate) shape: RouterShape,
    pub(crate) logits: DeviceRange<'a>,
    pub(crate) ids: DeviceRange<'a>,
    pub(crate) weights: DeviceRange<'a>,
    pub(crate) status: DeviceRange<'a>,
}

fn split_plan(shape: RouterShape) -> Result<(Bytes, RouterInts), EngError> {
    check_shape(&shape)?;
    let mut need = bytes_of(&shape)?;
    need.logits = need.logits.checked_sub(need.gate).ok_or_else(overflow)?;
    let ints = RouterInts {
        m: to_int("m", shape.m)?,
        hidden: to_int("hidden", shape.hidden)?,
        n_expert: to_int("n_expert", shape.n_expert)?,
        top_k: to_int("top_k", shape.top_k)?,
    };
    Ok((need, ints))
}

/// Tightly packed split logits, excluding the legacy shared-gate scratch column.
pub(crate) fn router_packed_logits_bytes(shape: RouterShape) -> Result<usize, EngError> {
    split_plan(shape).map(|(need, _)| need.logits)
}

fn check_split_regions(
    context: &CudaContext,
    regions: &[(&'static str, &DeviceRange<'_>, usize, usize, bool)],
) -> Result<(), EngError> {
    for (index, &(name, range, bytes, align, written)) in regions.iter().enumerate() {
        context.raw_range(range)?;
        require_fit(name, range, bytes, align)?;
        for &(other_name, other, other_bytes, _, other_written) in &regions[..index] {
            if written || other_written {
                require_apart(
                    &Claim::of(name, range, bytes),
                    &Claim::of(other_name, other, other_bytes),
                )?;
            }
        }
    }
    Ok(())
}

impl CudaContext {
    /// Enqueues all router dots at once, reusing the legacy decode/tiled F32 kernels.
    /// No allocation, copy, synchronization, shared gate or BF16 conversion occurs here.
    pub(crate) fn router_logits_rows(
        &self,
        stream: &DeviceStream,
        call: RouterLogitsCall<'_>,
    ) -> Result<(), EngError> {
        self.ensure_usable()?;
        let (need, ints) = split_plan(call.shape)?;
        check_split_regions(
            self,
            &[
                ("x", &call.x, need.x, 16, false),
                ("w", &call.w, need.w, 16, false),
                ("logits", &call.logits, need.logits, 4, true),
            ],
        )?;
        let io = RouterLogitsRaw {
            x: self.raw_range(&call.x)?,
            w: self.raw_range(&call.w)?,
            logits: self.raw_range(&call.logits)?,
            ints,
        };
        self.launch_on(stream, move |raw| router_logits_launch(raw, io))
    }

    /// Enqueues F32 sigmoid selection for every row. All raw logits are checked;
    /// faulty rows set status bit0, IDs -1 and zero gates. Status is never cleared.
    pub(crate) fn router_sigmoid_select_rows(
        &self,
        stream: &DeviceStream,
        call: RouterSigmoidCall<'_>,
    ) -> Result<(), EngError> {
        self.ensure_usable()?;
        let (need, ints) = split_plan(call.shape)?;
        check_split_regions(
            self,
            &[
                ("logits", &call.logits, need.logits, 4, false),
                ("ids", &call.ids, need.picks, 4, true),
                ("weights", &call.weights, need.picks, 4, true),
                ("status", &call.status, 4, 4, true),
            ],
        )?;
        let io = RouterSigmoidRaw {
            logits: self.raw_range(&call.logits)?,
            ids: self.raw_range(&call.ids)?,
            weights: self.raw_range(&call.weights)?,
            status: self.raw_range(&call.status)?,
            ints,
        };
        self.launch_on(stream, move |raw| router_sigmoid_launch(raw, io))
    }
}

#[cfg(all(test, feature = "native-cuda"))]
mod split_tests;
#[cfg(all(test, feature = "native-cuda"))]
mod tests;
