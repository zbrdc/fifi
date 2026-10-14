//! The `mlx_compile` seam of the adapter (packet MX-COMPILE-1a): the compiled DeltaNet and
//! device-routed MoE decode steps behind the engine setting `qwen36.compile` (default false),
//! and the teardown clear of the thread-local MLX compile cache (MLX keeps one cache per
//! thread, so the clear empties the dropping thread's cache). The stage 1 probe
//! (`compile_probe_gated_delta`) and the measurement harness around it stay test-only.
//!
//! Each compiled step replays its block's uncompiled op sequence call for call — including every
//! cast of packet Q36-BF16-a (the `bb` cast to F32, the F32 recurrence state and ring, the
//! `to_act` before the out projection, the ONE `mlx_minimum` clamp of the expert ids) — so a
//! compiled run and an uncompiled run issue the same MLX calls in the same order in both
//! activation dtypes. The activation dtype and the scale dtype of every packed weight are part of
//! the compile cache key (`kernels::compile_key`), so the F32 and BF16 decoders keep distinct
//! traces, and one adapter context holds at most 4 compiled programs (`ProgramRegistry`, a
//! field of the adapter state that dies with it), admitted before any native call.
//!
//! Every wrapper runs under the dispatch guard, validates before the first native call
//! (`check_operand` and the F32 rule on every input, `check_owner` plus the logical shapes on
//! every packed weight), frees its temporaries on every path, and marks the adapter terminal on
//! any native failure, like every resident operation. There is NO fallback to the uncompiled
//! path: after the engine turned the setting on, a failure of the compiled step is a native
//! failure (decision 0032/0023), because a silent fallback would change the numerics between
//! two tokens of one request. The fault hooks of the qualification build (call failure, null
//! output, output mismatch) select the two block calls like every other resident operation.

use super::super::ResidentPackedWeight;
use super::{
    Context, ffi,
    guard::{DispatchGuard, clear_message, failure, mark_failed},
    handles::{Array, native_dtype},
    resident_ops::{
        GATED_DELTA_SOURCE, check_f32_operand, check_operand, native_output_valid, produce,
    },
};
use crate::{
    EngError,
    kernels::{QuantizationMode, compile_key},
    tensor::{Dtype, ExecutionTensor},
};

/// The fixed compile-cache key of the stage 1 probe closure, kept distinct from every block
/// key, so a probe run cannot collide with a block run in one thread. Test builds only, like
/// its only caller.
#[cfg(test)]
const PROBE_FUN_ID: u64 = 0x0036_5147_a5a0_0001;

/// Checks one float dtype of the compile seam: the blocks run in F32 or BF16 only.
fn check_float(label: &'static str, dtype: Dtype) -> Result<(), EngError> {
    if matches!(dtype, Dtype::F32 | Dtype::Bf16) {
        return Ok(());
    }
    Err(EngError::BadRequest(format!(
        "{label} activation dtype must be F32 or BF16"
    )))
}

/// The quantization of one packed weight as the three ints the native seam bakes: the mode (0
/// affine, 1 mxfp4, 2 mxfp8) and the group size and bits, each -1 when the mode has none. The
/// load-time compile plan reads the same triple for its cache keys.
pub(crate) fn quant_triple(weight: &ResidentPackedWeight) -> Result<[i32; 3], EngError> {
    let spec = weight.quantization();
    let (mode, group, bits) = match spec.mode {
        QuantizationMode::Affine => (0i32, spec.group_size, spec.bits),
        QuantizationMode::Mxfp4 | QuantizationMode::Mxfp8 => (
            if spec.mode == QuantizationMode::Mxfp4 {
                1
            } else {
                2
            },
            0,
            0,
        ),
    };
    let native = |name: &'static str, value: u32| {
        i32::try_from(value)
            .map_err(|_| EngError::BadRequest(format!("compile seam {name} exceeds i32")))
    };
    Ok([
        mode,
        if group == 0 {
            -1
        } else {
            native("group size", group)?
        },
        if bits == 0 { -1 } else { native("bits", bits)? },
    ])
}

/// The quantization triples of `weights`, one `[mode, group, bits]` per packed weight, in the
/// caller's weight order: both the native payload and the cache key read them.
/// The quantization triples of `weights`, one `[mode, group, bits]` per packed weight, in the
/// caller's weight order: the native payload, the cache key, and the load-time warm-up plan read
/// them.
pub(crate) fn quant_triples<const N: usize>(
    weights: [&ResidentPackedWeight; N],
) -> Result<[[i32; 3]; N], EngError> {
    let mut triples = [[0i32; 3]; N];
    for (slot, weight) in triples.iter_mut().zip(weights) {
        *slot = quant_triple(weight)?;
    }
    Ok(triples)
}

/// The flat input list and the baked constants of one compiled DeltaNet decode step (packet
/// MX-COMPILE-1a). Every tensor must belong to this adapter.
pub(crate) struct CompileDeltaArgs<'a> {
    /// The layer input `[1, H]`.
    pub(crate) x: &'a ExecutionTensor,
    /// The recurrent state `[VH, VD, KD]`.
    pub(crate) rec: &'a ExecutionTensor,
    /// The convolution ring `[CD, CK - 1]`.
    pub(crate) ring: &'a ExecutionTensor,
    /// qkv, z, b, a, out.
    pub(crate) packed: [&'a ResidentPackedWeight; 5],
    /// conv_taps, conv_last, neg_exp_a_log, dt_bias, norm.
    pub(crate) plain: [&'a ExecutionTensor; 5],
    /// zero, qk_eps, sqrt_kd.
    pub(crate) consts: [&'a ExecutionTensor; 3],
    /// The hidden width `H`.
    pub(crate) hidden: u32,
    /// kh, kd, vh, vd, cd, taps.
    pub(crate) dims: [u32; 6],
    /// The RMS norm epsilon.
    pub(crate) eps: f32,
    /// The activation dtype of the step (packet Q36-BF16-a): the dtype of the layer input and of
    /// the block output (`to_act` before the out projection), and part of the compile cache key.
    pub(crate) act: Dtype,
}

/// The outputs of one compiled DeltaNet decode step, all lazy: the same three tensors that the
/// uncompiled `delta::step` returns.
pub(crate) struct CompiledDelta {
    /// Layer output `[1, H]`.
    pub(crate) out: ExecutionTensor,
    /// New recurrent state `[VH, VD, KD]`.
    pub(crate) rec: ExecutionTensor,
    /// New convolution ring `[CD, CK - 1]`.
    pub(crate) ring: ExecutionTensor,
}

/// The flat input list and the baked constants of one compiled device-routed MoE decode step
/// (packet MX-COMPILE-1a). Every tensor must belong to this adapter.
pub(crate) struct CompileMoeArgs<'a> {
    /// The layer input `[1, H]`.
    pub(crate) x: &'a ExecutionTensor,
    /// The transposed router weight `[H, E]`.
    pub(crate) router_t: &'a ExecutionTensor,
    /// The shared gate, up, down weights.
    pub(crate) shared: [&'a ResidentPackedWeight; 3],
    /// The transposed shared gate scalar `[H, 1]`.
    pub(crate) shared_scalar_t: &'a ExecutionTensor,
    /// The expert gate, up, down stacks.
    pub(crate) stacks: [&'a ResidentPackedWeight; 3],
    /// The hidden width `H`.
    pub(crate) hidden: u32,
    /// The expert count `E`.
    pub(crate) experts: u32,
    /// The routing width `k`.
    pub(crate) top_k: u32,
    /// Whether the gates renormalize over the selection.
    pub(crate) renorm: bool,
    /// The activation dtype of the step (packet Q36-BF16-a): the dtype of the layer input and of
    /// the block output (the final `to_act`), and part of the compile cache key.
    pub(crate) act: Dtype,
}

/// The outputs of one compiled device-routed MoE decode step, all lazy.
pub(crate) struct CompiledMoe {
    /// Layer output `[1, H]`.
    pub(crate) out: ExecutionTensor,
    /// The chosen experts `[k]` (U32) in rank order.
    pub(crate) ids: ExecutionTensor,
    /// The gates `[k, 1, 1]` (F32).
    pub(crate) gates: ExecutionTensor,
    /// The sum of the whole probability row `[1]` (F32).
    pub(crate) health: ExecutionTensor,
}

/// Checks one flat input list of a compiled block before admission. Every tensor must belong
/// to `adapter`. `x` must carry the activation dtype of the step exactly, and the adapter must
/// take that dtype — BF16 needs `allow_bf16`, the rule of the uncompiled `cast`. The F32
/// operands must be F32 exactly, because their bytes reach the traced kernels raw (the F32 rule
/// of the uncompiled `gated_delta`: the Metal kernel reads plain F32 bytes). The packed payloads
/// keep their stored dtype.
fn check_block_inputs(
    label: &'static str,
    adapter: &super::State,
    x: &ExecutionTensor,
    act: Dtype,
    f32_operands: &[&ExecutionTensor],
    packed: &[&ResidentPackedWeight],
) -> Result<(), EngError> {
    check_operand(label, adapter, x)?;
    if !adapter.accepts_float(act) {
        return Err(EngError::BadRequest(format!(
            "{label}: the adapter does not take {act:?} operands"
        )));
    }
    if x.dtype() != act {
        return Err(EngError::BadRequest(format!(
            "{label}: the layer input carries {:?}, the activation dtype is {act:?}",
            x.dtype()
        )));
    }
    for operand in f32_operands {
        check_f32_operand(label, adapter, operand)?;
    }
    for weight in packed {
        for part in [weight.payload(), weight.scales()]
            .into_iter()
            .chain(weight.biases())
        {
            if part.owner_key() != adapter.owner {
                return Err(EngError::BadRequest(format!(
                    "{label}: weight belongs to another native owner"
                )));
            }
        }
    }
    Ok(())
}

/// Checks the logical shape of one packed weight of a compiled block against `want`, a
/// rank-and-dimension match, before admission: a stack that disagrees with the baked
/// `experts` or a projection of the wrong width would otherwise fail inside the traced closure
/// (a terminal Backend) or gather out of range on the device.
fn check_logical(
    label: &'static str,
    what: &str,
    weight: &ResidentPackedWeight,
    want: &[u32],
) -> Result<(), EngError> {
    if weight.logical_shape() != want {
        return Err(EngError::BadRequest(format!(
            "{label}: the {what} weight holds the logical shape {:?}, the block needs {want:?}",
            weight.logical_shape()
        )));
    }
    Ok(())
}

/// A checked i32 dimension of a compile seam call.
fn native_dim(label: &'static str, value: u32) -> Result<i32, EngError> {
    i32::try_from(value).map_err(|_| EngError::BadRequest(format!("{label} dimension exceeds i32")))
}

impl Context {
    /// Stage 1 of the compile seam, kept as the measurement probe it was: runs the gated delta
    /// kernel (decision 0050) through one `mlx_compile`d closure and returns `y` `[B, T, Hv, Dv]`
    /// and the new state `[B, Hv, Dv, Dk]`, both lazy, exactly as [`Context::gated_delta`] does.
    /// The closure is compiled with the fixed cache key [`PROBE_FUN_ID`] and `shapeless = false`,
    /// so the second call hits the compile cache and the first call is the compile. A native
    /// failure is terminal, as for every resident operation. Test builds only: the probe
    /// answered the first compile question and no production path calls it.
    #[cfg(test)]
    pub(in crate::kernels::mlx) fn compile_probe_gated_delta(
        &mut self,
        q: &ExecutionTensor,
        k: &ExecutionTensor,
        v: &ExecutionTensor,
        g: &ExecutionTensor,
        beta: &ExecutionTensor,
        recurrent: &ExecutionTensor,
    ) -> Result<(ExecutionTensor, ExecutionTensor), EngError> {
        let label = "gated delta compile probe";
        let _dispatch = DispatchGuard::enter();
        self.check_ready()?;
        let adapter = self.state.as_mut().unwrap();
        for operand in [q, k, v, g, beta, recurrent] {
            check_operand(label, adapter, operand)?;
        }
        let [qs, ks, vs, gs, bs, ss] = [q, k, v, g, beta, recurrent].map(ExecutionTensor::shape);
        let layout = crate::kernels::gated_delta::layout(qs, ks, vs, gs, bs, ss)?;
        let _inputs = [
            q.share(),
            k.share(),
            v.share(),
            g.share(),
            beta.share(),
            recurrent.share(),
        ];
        clear_message();
        let stream = adapter.stream.0;
        let (mut y, mut next) = (Array::null(), Array::null());
        let built = produce(&mut y, label, |slot| unsafe {
            ffi::mlx_compile_probe(
                slot,
                &mut next.0,
                q.storage.native_array(),
                k.storage.native_array(),
                v.storage.native_array(),
                g.storage.native_array(),
                beta.storage.native_array(),
                recurrent.storage.native_array(),
                layout.dims.as_ptr(),
                GATED_DELTA_SOURCE.as_ptr().cast(),
                PROBE_FUN_ID as usize,
                stream,
            )
        });
        let checked = built.and_then(|()| {
            if native_output_valid(y.0, &layout.y_shape, Dtype::F32, layout.y_byte_len)
                && native_output_valid(next.0, ss, Dtype::F32, layout.state_byte_len)
            {
                Ok(())
            } else {
                Err(failure(label))
            }
        });
        if checked.is_err() {
            mark_failed();
        }
        checked?;
        Ok((
            adapter.resident(y, Dtype::F32, layout.y_shape.to_vec(), layout.y_byte_len),
            adapter.resident(next, Dtype::F32, ss.to_vec(), layout.state_byte_len),
        ))
    }

    /// Runs one DeltaNet decode step (`delta.rs` `step`) through one `mlx_compile`d closure and
    /// returns the layer output, the new recurrent state, and the new ring, all lazy — the same
    /// three tensors, shapes, and dtypes that the uncompiled step returns.
    ///
    /// Inputs: `x` `[1, H]` in the activation dtype, the F32 recurrent state `rec`
    /// `[VH, VD, KD]`, the F32 convolution `ring` `[CD, CK - 1]`, the five packed weights
    /// (qkv, z, b, a, out) with their stored scale and bias dtypes and logical shapes
    /// `[CD, H]`, `[VH*VD, H]` (three), and `[H, VH*VD]`, the five plain tensors
    /// (conv_taps, conv_last, neg_exp_a_log, dt_bias, norm), and the three rank-0 constants
    /// (zero, qk_eps, sqrt_kd). Every tensor must belong to this adapter; `check_operand`, the
    /// F32 rule, and `check_owner` plus the logical shapes reject anything else before the
    /// first native call.
    ///
    /// Baked constants: the hidden width, the six block dimensions, `kw` and `kw2`, the RMS norm
    /// epsilon, the activation dtype, and per packed weight the scale dtype and the quantization
    /// triple. They form the compile cache key (`compile_key::delta_fun_id`), so every layer with
    /// the same constants shares one compiled program, and the per-adapter registry admits the
    /// key BEFORE any native call: a fifth distinct key of one adapter is `BadRequest`.
    ///
    /// Dtype rules (packet Q36-BF16-a): the block output carries the activation dtype (the block
    /// ends in `to_act`); the recurrence state and the ring stay F32; the closure mirrors every
    /// cast of the uncompiled step, so both activation dtypes build the same computation as
    /// their uncompiled counterparts.
    ///
    /// No fallback: a failure is a native failure. The call runs under the dispatch guard, every
    /// temporary is freed on every path, a native error makes the adapter terminal (decision
    /// 0032/0023), the adapter stays thread-confined like the whole resident interface, and the
    /// qualification fault hooks select the call (call failure, null output, output mismatch)
    /// like `gated_delta`.
    pub(in crate::kernels::mlx) fn compiled_delta_step(
        &mut self,
        args: &CompileDeltaArgs<'_>,
    ) -> Result<CompiledDelta, EngError> {
        let label = "compiled delta block";
        let _dispatch = DispatchGuard::enter();
        self.check_ready()?;
        check_float(label, args.act)?;
        let adapter = self.state.as_mut().unwrap();
        // The checked dimension set: every dimension, every product (vh * vd, kh * kd, 2 * kw),
        // and the fused-projection identity cd == 2 * kw + vh * vd fit an i32, before any
        // native call (the pure rule of `compile_key::checked_delta_dims`, Linux-tested).
        let [kh, kd, vh, vd, cd, taps, kw, kw2] = compile_key::checked_delta_dims(args.dims)?;
        let vh_vd = vh
            .checked_mul(vd)
            .ok_or_else(|| EngError::BadRequest("compile seam vh * vd exceeds u32".into()))?;
        if args.x.shape() != [1, args.hidden]
            || args.rec.shape() != [vh, vd, kd]
            || args.ring.shape() != [cd, taps]
        {
            return Err(EngError::BadRequest(
                "compile seam inputs differ from the delta block shapes".into(),
            ));
        }
        // The logical shapes of the five projections: qkv writes the fused [CD] projection, z
        // writes [VH * VD], b and a write one value per value head [VH], and out reads [VH * VD]
        // back to [H].
        for (what, weight, want) in [
            ("qkv", &args.packed[0], [cd, args.hidden]),
            ("z", &args.packed[1], [vh_vd, args.hidden]),
            ("b", &args.packed[2], [vh, args.hidden]),
            ("a", &args.packed[3], [vh, args.hidden]),
            ("out", &args.packed[4], [args.hidden, vh_vd]),
        ] {
            check_logical(label, what, weight, &want)?;
        }
        let f32_operands: Vec<&ExecutionTensor> = [args.rec, args.ring]
            .into_iter()
            .chain(args.plain.iter().copied())
            .chain(args.consts.iter().copied())
            .collect();
        check_block_inputs(
            label,
            adapter,
            args.x,
            args.act,
            &f32_operands,
            &args.packed,
        )?;
        let dims = [
            native_dim(label, args.hidden)?,
            native_dim(label, kh)?,
            native_dim(label, kd)?,
            native_dim(label, vh)?,
            native_dim(label, vd)?,
            native_dim(label, cd)?,
            native_dim(label, taps)?,
            native_dim(label, kw)?,
            native_dim(label, kw2)?,
        ];
        let triples = quant_triples(args.packed)?;
        let quant: Vec<i32> = triples.iter().flat_map(|t| t.iter().copied()).collect();
        let fun_id = compile_key::delta_fun_id(&compile_key::DeltaBaked {
            hidden: args.hidden,
            dims: [kh, kd, vh, vd, cd, taps, kw, kw2],
            eps_bits: args.eps.to_bits(),
            act: compile_key::dtype_key(args.act),
            scale_keys: args
                .packed
                .each_ref()
                .map(|weight| compile_key::dtype_key(weight.scales().dtype())),
            quant: triples,
        });
        // The per-adapter program registry decides before any native call: a fifth distinct key
        // of one adapter is a setting mistake, not a reason to grow the MLX compile cache.
        adapter.programs.admit(fun_id)?;
        let mut keep: Vec<_> = [args.x, args.rec, args.ring]
            .into_iter()
            .chain(args.plain.iter().copied())
            .chain(args.consts.iter().copied())
            .map(|tensor| tensor.share())
            .collect();
        for weight in args.packed {
            keep.push(weight.payload().share());
            keep.push(weight.scales().share());
            if let Some(biases) = weight.biases() {
                keep.push(biases.share());
            }
        }
        let mut inputs: Vec<ffi::mlx_array> = Vec::with_capacity(26);
        inputs.push(args.x.storage.native_array());
        inputs.push(args.rec.storage.native_array());
        inputs.push(args.ring.storage.native_array());
        for weight in args.packed {
            inputs.push(weight.payload().storage.native_array());
            inputs.push(weight.scales().storage.native_array());
            inputs.push(
                weight
                    .biases()
                    .map(|biases| biases.storage.native_array())
                    .unwrap_or(ffi::NULL_ARRAY),
            );
        }
        for tensor in args.plain.iter().chain(args.consts.iter()) {
            inputs.push(tensor.storage.native_array());
        }
        clear_message();
        let stream = adapter.stream.0;
        let (mut out_a, mut rec_a, mut ring_a) = (Array::null(), Array::null(), Array::null());
        let started = std::time::Instant::now();
        let built = produce(&mut out_a, label, |slot| unsafe {
            ffi::mlx_compile_delta_block(
                slot,
                &mut rec_a.0,
                &mut ring_a.0,
                inputs.as_ptr(),
                inputs.len(),
                dims.as_ptr(),
                quant.as_ptr(),
                args.eps,
                native_dtype(args.act),
                GATED_DELTA_SOURCE.as_ptr().cast(),
                fun_id as usize,
                stream,
            )
        });
        super::super::compile_stats::record(true, fun_id, started.elapsed().as_secs_f64());
        // The layer output carries the activation dtype (the block ends in `to_act`); the
        // recurrence state and the ring stay F32 (packet Q36-BF16-a).
        let hidden_bytes = u64::from(args.hidden) * args.act.byte_width();
        let state_bytes = u64::from(vh) * u64::from(vd) * u64::from(kd) * 4;
        let ring_bytes = u64::from(cd) * u64::from(taps) * 4;
        let checked = built.and_then(|()| {
            if native_output_valid(out_a.0, &[1, args.hidden], args.act, hidden_bytes)
                && native_output_valid(rec_a.0, &[vh, vd, kd], Dtype::F32, state_bytes)
                && native_output_valid(ring_a.0, &[cd, taps], Dtype::F32, ring_bytes)
            {
                Ok(())
            } else {
                Err(failure(label))
            }
        });
        if checked.is_err() {
            mark_failed();
        }
        checked?;
        Ok(CompiledDelta {
            out: adapter.resident(out_a, args.act, vec![1, args.hidden], hidden_bytes),
            rec: adapter.resident(rec_a, Dtype::F32, vec![vh, vd, kd], state_bytes),
            ring: adapter.resident(ring_a, Dtype::F32, vec![cd, taps], ring_bytes),
        })
    }

    /// Runs one device-routed MoE decode step (`moe.rs` `step_device`, `route_device`, and
    /// `combine`) through one `mlx_compile`d closure and returns the layer output with the route
    /// (ids, gates, health), all lazy — the same four tensors, shapes, and dtypes that the
    /// uncompiled step returns (the ids are the once-clamped ids, wrapped by the caller with
    /// `BoundedIndices::from_clamped`).
    ///
    /// Inputs: `x` `[1, H]` in the activation dtype, the F32 `router_t` `[H, E]`, the three
    /// shared packed weights, the F32 `shared_scalar_t` `[H, 1]`, and the three expert stacks
    /// with the logical shapes `[E, I, H]` (gate, up) and `[E, H, I]` (down). Every tensor must
    /// belong to this adapter; `check_operand`, the F32 rule, and `check_owner` plus the expert
    /// count and the widths of the logical shapes reject anything else before the first native
    /// call (an out-of-range device gather is not bounds-checked by MLX).
    ///
    /// Baked constants: the hidden width, the expert count, the routing width, the renormalization
    /// flag, the clamp bound `experts - 1` (the ONE `mlx_minimum` clamp per layer), the activation
    /// dtype, and per packed weight the scale dtype and the quantization triple. They form the
    /// compile cache key (`compile_key::moe_fun_id`); the per-adapter registry admits it BEFORE
    /// any native call, and a fifth distinct key of one adapter is `BadRequest`.
    ///
    /// Dtype rules (packet Q36-BF16-a): the block output carries the activation dtype (the final
    /// `to_act`); the router arithmetic, the gates, and the health row stay F32; the ids stay
    /// U32.
    ///
    /// No fallback: a failure is a native failure, with the same discipline as
    /// [`Context::compiled_delta_step`] (dispatch guard, ownership checks, temporaries freed on
    /// every path, terminal on a native error, thread confinement, and the qualification fault
    /// hooks).
    pub(in crate::kernels::mlx) fn compiled_moe_step(
        &mut self,
        args: &CompileMoeArgs<'_>,
    ) -> Result<CompiledMoe, EngError> {
        let label = "compiled moe block";
        let _dispatch = DispatchGuard::enter();
        self.check_ready()?;
        check_float(label, args.act)?;
        let adapter = self.state.as_mut().unwrap();
        if args.top_k == 0 || args.top_k > args.experts || args.experts == 0 {
            return Err(EngError::BadRequest(
                "compile seam moe top-k must be between 1 and the expert count".into(),
            ));
        }
        if args.x.shape() != [1, args.hidden]
            || args.router_t.shape() != [args.hidden, args.experts]
            || args.shared_scalar_t.shape() != [args.hidden, 1]
        {
            return Err(EngError::BadRequest(
                "compile seam inputs differ from the moe block shapes".into(),
            ));
        }
        // The expert stacks must hold exactly the baked expert count and the hidden width where
        // the block reads it: the gate and up stacks read `[1, H]` (their last dimension), the
        // down stack writes `[1, H]` (its middle dimension), the shared weights the same at rank
        // 2. A stack that disagrees would gather past its own last expert or die in a reshape
        // inside the traced closure.
        for (what, stack, reads_hidden_last) in [
            ("gate stack", &args.stacks[0], true),
            ("up stack", &args.stacks[1], true),
            ("down stack", &args.stacks[2], false),
        ] {
            let shape = stack.logical_shape();
            let ok = match (shape, reads_hidden_last) {
                ([experts, _, width], true) => *experts == args.experts && *width == args.hidden,
                ([experts, hidden_width, _], false) => {
                    *experts == args.experts && *hidden_width == args.hidden
                }
                _ => false,
            };
            if !ok {
                return Err(EngError::BadRequest(format!(
                    "{label}: the {what} holds the logical shape {shape:?}, the block needs \
                     [{}, ., {}] around the hidden width",
                    args.experts, args.hidden
                )));
            }
        }
        // The shared weights must carry the hidden width where the block reads or writes it: the
        // shared gate and up weights read `[1, H]` (their last dimension), the shared down
        // weight writes `[1, H]` (its first dimension).
        for (what, weight, hidden_last) in [
            ("shared gate", &args.shared[0], true),
            ("shared up", &args.shared[1], true),
            ("shared down", &args.shared[2], false),
        ] {
            let shape = weight.logical_shape();
            let ok = match (shape, hidden_last) {
                ([_, width], true) => *width == args.hidden,
                ([hidden_width, _], false) => *hidden_width == args.hidden,
                _ => false,
            };
            if !ok {
                return Err(EngError::BadRequest(format!(
                    "{label}: the {what} weight holds the logical shape {shape:?}, the block \
                     needs the hidden width {} where it reads or writes it",
                    args.hidden
                )));
            }
        }
        let weights: Vec<&ResidentPackedWeight> = args
            .shared
            .iter()
            .chain(args.stacks.iter())
            .copied()
            .collect();
        check_block_inputs(
            label,
            adapter,
            args.x,
            args.act,
            &[args.router_t, args.shared_scalar_t],
            &weights,
        )?;
        let dims = [
            native_dim(label, args.hidden)?,
            native_dim(label, args.experts)?,
            native_dim(label, args.top_k)?,
            i32::from(args.renorm),
        ];
        let shared_triples = quant_triples(args.shared)?;
        let stack_triples = quant_triples(args.stacks)?;
        let quant: Vec<i32> = shared_triples
            .iter()
            .chain(stack_triples.iter())
            .flat_map(|triple| triple.iter().copied())
            .collect();
        let shared_scales = args
            .shared
            .each_ref()
            .map(|weight| compile_key::dtype_key(weight.scales().dtype()));
        let stack_scales = args
            .stacks
            .each_ref()
            .map(|weight| compile_key::dtype_key(weight.scales().dtype()));
        let [sg, su, sd] = shared_triples;
        let [xg, xu, xd] = stack_triples;
        let fun_id = compile_key::moe_fun_id(&compile_key::MoeBaked {
            hidden: args.hidden,
            experts: args.experts,
            top_k: args.top_k,
            renorm: u32::from(args.renorm),
            clamp: args
                .experts
                .checked_sub(1)
                .ok_or_else(|| EngError::BadRequest("compile seam moe needs experts".into()))?,
            act: compile_key::dtype_key(args.act),
            scale_keys: [
                shared_scales[0],
                shared_scales[1],
                shared_scales[2],
                stack_scales[0],
                stack_scales[1],
                stack_scales[2],
            ],
            quant: [sg, su, sd, xg, xu, xd],
        });
        // The per-adapter program registry decides before any native call, exactly as for the
        // delta block.
        adapter.programs.admit(fun_id)?;
        let mut inputs: Vec<ffi::mlx_array> = Vec::with_capacity(21);
        inputs.push(args.x.storage.native_array());
        inputs.push(args.router_t.storage.native_array());
        // The native body reads the shared scalar at index 11, between the shared weights and the
        // expert stacks (the input order of `compile_moe_body` in guard.cpp).
        for weight in args.shared {
            inputs.push(weight.payload().storage.native_array());
            inputs.push(weight.scales().storage.native_array());
            inputs.push(
                weight
                    .biases()
                    .map(|biases| biases.storage.native_array())
                    .unwrap_or(ffi::NULL_ARRAY),
            );
        }
        inputs.push(args.shared_scalar_t.storage.native_array());
        for weight in args.stacks {
            inputs.push(weight.payload().storage.native_array());
            inputs.push(weight.scales().storage.native_array());
            inputs.push(
                weight
                    .biases()
                    .map(|biases| biases.storage.native_array())
                    .unwrap_or(ffi::NULL_ARRAY),
            );
        }
        let mut keep: Vec<_> = [args.x, args.router_t, args.shared_scalar_t]
            .into_iter()
            .map(|tensor| tensor.share())
            .collect();
        for weight in args.shared.into_iter().chain(args.stacks.iter().copied()) {
            keep.push(weight.payload().share());
            keep.push(weight.scales().share());
            if let Some(biases) = weight.biases() {
                keep.push(biases.share());
            }
        }
        clear_message();
        let stream = adapter.stream.0;
        let (mut out_a, mut ids_a, mut gates_a, mut health_a) =
            (Array::null(), Array::null(), Array::null(), Array::null());
        let started = std::time::Instant::now();
        let built = produce(&mut out_a, label, |slot| unsafe {
            ffi::mlx_compile_moe_block(
                slot,
                &mut ids_a.0,
                &mut gates_a.0,
                &mut health_a.0,
                inputs.as_ptr(),
                inputs.len(),
                dims.as_ptr(),
                quant.as_ptr(),
                native_dtype(args.act),
                fun_id as usize,
                stream,
            )
        });
        super::super::compile_stats::record(false, fun_id, started.elapsed().as_secs_f64());
        // The layer output carries the activation dtype (the block ends in `to_act`); the route
        // stays exactly the uncompiled route: U32 ids, F32 gates, F32 health.
        let hidden_bytes = u64::from(args.hidden) * args.act.byte_width();
        let k_bytes = u64::from(args.top_k) * 4;
        let checked = built.and_then(|()| {
            if native_output_valid(out_a.0, &[1, args.hidden], args.act, hidden_bytes)
                && native_output_valid(ids_a.0, &[args.top_k], Dtype::U32, k_bytes)
                && native_output_valid(gates_a.0, &[args.top_k, 1, 1], Dtype::F32, k_bytes)
                && native_output_valid(health_a.0, &[1], Dtype::F32, 4)
            {
                Ok(())
            } else {
                Err(failure(label))
            }
        });
        if checked.is_err() {
            mark_failed();
        }
        checked?;
        Ok(CompiledMoe {
            out: adapter.resident(out_a, args.act, vec![1, args.hidden], hidden_bytes),
            ids: adapter.resident(ids_a, Dtype::U32, vec![args.top_k], k_bytes),
            gates: adapter.resident(gates_a, Dtype::F32, vec![args.top_k, 1, 1], k_bytes),
            health: adapter.resident(health_a, Dtype::F32, vec![1], 4),
        })
    }

    /// Clears the calling thread's MLX compile cache and counts it, so the first compiled call
    /// after the clear is a clean compile. The MLX compile cache is thread-local, so the clear
    /// covers the adapters of THIS thread. The measurement harness of the test builds uses this
    /// between runs; the adapter teardown clears through [`teardown_clear`] on its own path.
    #[cfg(test)]
    pub(in crate::kernels::mlx) fn compile_clear_cache(&mut self) -> Result<(), EngError> {
        use super::guard::check;
        let _dispatch = DispatchGuard::enter();
        self.check_ready()?;
        clear_message();
        let status = unsafe { ffi::mlx_compile_clear_cache() };
        super::super::compile_stats::counted_clear();
        check(status, "compile cache clear")
    }
}

/// Reads the process-wide counts of compiled closure-body entries per block kind from the
/// native seam: the authoritative trace counts, because the body of a compiled closure runs
/// once per trace (and on every call when MLX returns the uncompiled function), while a cache
/// hit replays the compiled tape without entering the body. `compile_stats` reads them at every
/// counter read, so a retrace that the Rust-side key comparison cannot see (a shape or dtype
/// drift under one key, `MLX_DISABLE_COMPILE`, a second engine thread's own thread-local
/// cache) still shows in `traces`.
pub(in crate::kernels::mlx) fn native_trace_counts() -> (u64, u64) {
    let (mut delta, mut moe) = (0u64, 0u64);
    // Safe on any thread: the counts are static atomics in the native seam and the pointers
    // point at the two locals above.
    unsafe { ffi::mlx_compile_trace_counts(&mut delta, &mut moe) };
    (delta, moe)
}

/// Zeroes the native closure-body entry counts. Test builds only: `compile_stats::reset` calls
/// this with its own reset, so a test that reads `traces` sees its own traces and not the body
/// entries of the tests before it.
#[cfg(test)]
pub(in crate::kernels::mlx) fn native_trace_reset() {
    unsafe { ffi::mlx_compile_reset_trace_counts() };
}

/// Clears the calling thread's MLX compile cache at adapter teardown (packet MX-COMPILE-1a
/// item 4). `Context::drop` calls this after its synchronize completed every pending evaluation
/// and the retire list is drained, and before the stream and the device are released; the
/// caller already holds the dispatch guard. The MLX compile cache is thread-local, so the clear
/// covers this adapter's thread and its compiled programs die with it. The clear is idempotent
/// — it empties a possibly empty cache — and a failure changes no completion state of the dying
/// adapter, so it only goes uncounted: the programs of the cache stay until the thread exits.
pub(in crate::kernels::mlx) fn teardown_clear() {
    if unsafe { ffi::mlx_compile_clear_cache() } == 0 {
        super::super::compile_stats::counted_clear();
    }
}
