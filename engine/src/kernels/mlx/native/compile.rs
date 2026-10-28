//! The `mlx_compile` seam of the adapter (packet MX-COMPILE-1a): the compiled DeltaNet and
//! device-routed MoE decode steps behind the engine setting `qwen36.compile` (default false),
//! and the teardown clear of the thread-local MLX compile cache (MLX keeps one cache per
//! thread, so the clear empties the dropping thread's cache). The stage 1 probe
//! (`compile_probe_gated_delta`) and the measurement harness around it stay test-only.
//!
//! Each B=1 compiled step replays the uncompiled op sequence call for call, including every
//! cast of packet Q36-BF16-a (the `bb` cast to F32, the F32 recurrence state and ring, the
//! `to_act` before the out projection, the ONE `mlx_minimum` clamp of the expert ids) — so a
//! B=1 compiled run and an uncompiled run issue the same MLX calls in the same order in both
//! activation dtypes. Activation and packed-weight scale dtypes belong to the compile cache key
//! (`kernels::compile_key`), keeping the F32 and BF16 decoders on distinct baked keys. One
//! adapter context admits at most 4 keys before any native call; its `ProgramRegistry` dies
//! with it. B>1 MoE vectorizes its sequence. Staged B>1 DeltaNet batches
//! five projections and keeps one recurrence/ring core per slot; taps<=1 stays B1-only.
//! Each block key permits B=1/2/4/8 where supported: one context admits at most sixteen key/batch
//! pairs. This is not a per-thread or global bound: other contexts can admit other keys, and
//! cache clears or other threads can cause further closure-body entries.
//!
//! Every wrapper runs under the dispatch guard, validates before the first native call
//! (`check_operand` and the F32 rule on every input, `check_owner` plus the logical shapes on
//! every packed weight), and marks the adapter terminal on any native failure. The rows C
//! wrapper releases partial lazy outputs before returning a failure. Legacy B1 instead returns
//! partial handles to Rust, whose terminal policy retains them without further native calls;
//! successful handles with invalid metadata follow that retention policy too.
//! There is NO fallback to the uncompiled
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
/// Cache key of the tiny-add-chain compile probe (per-kernel floor calibration).
#[cfg(all(test, feature = "native-fault-qualification"))]
const ADD_CHAIN_PROBE_FUN_ID: u64 = 0x0036_5147_a5a0_0003;
/// Cache key of the Flint layer-subgraph compile probe (distinct from the gated-delta probe).
#[cfg(all(test, feature = "native-fault-qualification"))]
const FLINT_LAYER_PROBE_FUN_ID: u64 = 0x0036_5147_a5a0_0002;

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
    /// The layer input: `[1, H]` for the step seam or `[B, H]` for staged rows.
    pub(crate) x: &'a ExecutionTensor,
    /// Slot zero's recurrent state `[VH, VD, KD]`.
    pub(crate) rec: &'a ExecutionTensor,
    /// Slot zero's convolution ring `[CD, CK - 1]`.
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

/// Staged multirow DeltaNet outputs with separate per-slot F32 state/ring tensors.
#[cfg_attr(not(test), allow(dead_code))] // The resident caller stays solo until preparation.
pub(crate) struct CompiledDeltaRows {
    pub(crate) out: ExecutionTensor,
    pub(crate) states: Vec<(ExecutionTensor, ExecutionTensor)>,
}

/// The flat input list and the baked constants of one compiled device-routed MoE decode step
/// (packet MX-COMPILE-1a). Every tensor must belong to this adapter.
pub(crate) struct CompileMoeArgs<'a> {
    /// The layer input `[B, H]`, with `B` in 1, 2, 4, or 8.
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
    /// Layer output `[B, H]`.
    pub(crate) out: ExecutionTensor,
    /// The chosen experts `[B, k]` (U32) in per-row rank order; `[k]` for B=1.
    pub(crate) ids: ExecutionTensor,
    /// The gates `[B, k, 1, 1]` (F32); `[k, 1, 1]` for B=1.
    pub(crate) gates: ExecutionTensor,
    /// The sums of the whole probability rows `[B]` (F32).
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
        let got = self.compiled_delta_impl(args, None)?;
        let (rec, ring) = got
            .states
            .into_iter()
            .next()
            .expect("validated B1 state count");
        Ok(CompiledDelta {
            out: got.out,
            rec,
            ring,
        })
    }

    /// Staged B2/4/8 seam: `args.rec`/`ring` are slot zero; `extra_states` supplies the rest
    /// in row order. All states are live closure operands. Only projections are batched.
    /// B specializes the nonshapeless input signature, never the baked id or registry policy.
    #[cfg_attr(not(test), allow(dead_code))] // No resident wiring or implicit multirow warm-up.
    pub(in crate::kernels::mlx) fn compiled_delta_rows(
        &mut self,
        args: &CompileDeltaArgs<'_>,
        extra_states: &[(&ExecutionTensor, &ExecutionTensor)],
    ) -> Result<CompiledDeltaRows, EngError> {
        self.compiled_delta_impl(args, Some(extra_states))
    }

    fn compiled_delta_impl(
        &mut self,
        args: &CompileDeltaArgs<'_>,
        extra_states: Option<&[(&ExecutionTensor, &ExecutionTensor)]>,
    ) -> Result<CompiledDeltaRows, EngError> {
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
        let batch = match (extra_states, args.x.shape()) {
            (None, _) => 1,
            (Some(extra), &[batch @ (2 | 4 | 8), hidden])
                if hidden == args.hidden && extra.len() == batch as usize - 1 =>
            {
                batch
            }
            _ => {
                return Err(EngError::BadRequest(
                    "compile seam delta requires B1 or B in 2, 4, or 8 with B-1 extra states"
                        .into(),
                ));
            }
        };
        // shortcut: one-tap rings may retain full B, keep solo until private copies are qualified.
        if batch > 1 && taps <= 1 {
            return Err(EngError::BadRequest(
                "compile seam multirow delta requires taps > 1 for independent ring backing".into(),
            ));
        }
        let states: Vec<_> = std::iter::once((args.rec, args.ring))
            .chain(extra_states.unwrap_or(&[]).iter().copied())
            .collect();
        if (batch == 1 && args.x.shape() != [1, args.hidden])
            || states
                .iter()
                .any(|(rec, ring)| rec.shape() != [vh, vd, kd] || ring.shape() != [cd, taps])
        {
            return Err(EngError::BadRequest(
                "compile seam inputs differ from the delta block shapes".into(),
            ));
        }
        // Keep the legacy B1 validation path. Multirow operands must match the full closure
        // signature before admission, including the shared plain tensors and scalar constants.
        if batch > 1
            && (args.plain[0].shape() != [cd, taps]
                || args.plain[1].shape() != [cd]
                || args.plain[2].shape() != [vh]
                || args.plain[3].shape() != [vh]
                || args.plain[4].shape() != [vd]
                || args.consts.iter().any(|tensor| !tensor.shape().is_empty()))
        {
            return Err(EngError::BadRequest(
                "compile seam inputs differ from the delta plain operand shapes".into(),
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
        let f32_operands: Vec<&ExecutionTensor> = states
            .iter()
            .flat_map(|&(rec, ring)| [rec, ring])
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
        let mut inputs: Vec<ffi::mlx_array> = Vec::with_capacity(26 + 2 * (batch as usize - 1));
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
        for &(rec, ring) in &states[1..] {
            inputs.push(rec.storage.native_array());
            inputs.push(ring.storage.native_array());
            keep.push(rec.share());
            keep.push(ring.share());
        }
        clear_message();
        let stream = adapter.stream.0;
        let (mut out_a, mut rec_a, mut ring_a) = (Array::null(), Array::null(), Array::null());
        let mut row_outputs = if batch == 1 {
            Vec::new()
        } else {
            vec![ffi::NULL_ARRAY; 1 + 2 * batch as usize]
        };
        let started = std::time::Instant::now();
        let built = produce(&mut out_a, label, |slot| unsafe {
            if batch == 1 {
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
            } else {
                let status = ffi::mlx_compile_delta_rows(
                    row_outputs.as_mut_ptr(),
                    row_outputs.len(),
                    inputs.as_ptr(),
                    inputs.len(),
                    dims.as_ptr(),
                    quant.as_ptr(),
                    args.eps,
                    native_dtype(args.act),
                    GATED_DELTA_SOURCE.as_ptr().cast(),
                    fun_id as usize,
                    stream,
                );
                *slot = row_outputs[0];
                row_outputs[0] = ffi::NULL_ARRAY;
                status
            }
        });
        super::super::compile_stats::record(true, fun_id, started.elapsed().as_secs_f64());
        // The layer output carries the activation dtype (the block ends in `to_act`); the
        // recurrence state and the ring stay F32 (packet Q36-BF16-a).
        let hidden_bytes = u64::from(batch) * u64::from(args.hidden) * args.act.byte_width();
        let state_bytes = u64::from(vh) * u64::from(vd) * u64::from(kd) * 4;
        let ring_bytes = u64::from(cd) * u64::from(taps) * 4;
        let state_arrays: Vec<_> = if batch == 1 {
            vec![rec_a, ring_a]
        } else {
            row_outputs.into_iter().skip(1).map(Array).collect()
        };
        let checked = built.and_then(|()| {
            if native_output_valid(out_a.0, &[batch, args.hidden], args.act, hidden_bytes)
                && state_arrays.as_chunks::<2>().0.iter().all(|pair| {
                    native_output_valid(pair[0].0, &[vh, vd, kd], Dtype::F32, state_bytes)
                        && native_output_valid(pair[1].0, &[cd, taps], Dtype::F32, ring_bytes)
                })
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
        let mut arrays = state_arrays.into_iter();
        let states = (0..batch)
            .map(|_| {
                (
                    adapter.resident(
                        arrays.next().expect("validated rec count"),
                        Dtype::F32,
                        vec![vh, vd, kd],
                        state_bytes,
                    ),
                    adapter.resident(
                        arrays.next().expect("validated ring count"),
                        Dtype::F32,
                        vec![cd, taps],
                        ring_bytes,
                    ),
                )
            })
            .collect();
        Ok(CompiledDeltaRows {
            out: adapter.resident(out_a, args.act, vec![batch, args.hidden], hidden_bytes),
            states,
        })
    }

    /// Runs one device-routed MoE decode step (`moe.rs` `step_device`, `route_device`, and
    /// `combine`) through one `mlx_compile`d closure and returns the layer output with the route
    /// (ids, gates, health), all lazy. B=1 retains the uncompiled step's shapes and op sequence.
    /// B=2/4/8 vectorizes routing, shared projections, and gathered expert products by row.
    /// The ids are once-clamped; callers can wrap them with `BoundedIndices::from_clamped`.
    ///
    /// Inputs: `x` `[B, H]` in the activation dtype, the F32 `router_t` `[H, E]`, the three
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
    /// any native call, and a fifth distinct key of one adapter is `BadRequest`. B is not baked;
    /// the nonshapeless MLX cache specializes input shapes under the same key.
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
        let batch = match args.x.shape() {
            &[batch @ (1 | 2 | 4 | 8), hidden] if hidden == args.hidden => batch,
            _ => {
                return Err(EngError::BadRequest(
                    "compile seam moe input must have shape [B, H] with B in 1, 2, 4, or 8".into(),
                ));
            }
        };
        if args.router_t.shape() != [args.hidden, args.experts]
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
        // B is bounded by 8 and each width is u32, so these byte counts fit u64.
        let hidden_bytes = u64::from(batch) * u64::from(args.hidden) * args.act.byte_width();
        let k_bytes = u64::from(batch) * u64::from(args.top_k) * 4;
        let health_bytes = u64::from(batch) * 4;
        let out_shape = vec![batch, args.hidden];
        let ids_shape = if batch == 1 {
            vec![args.top_k]
        } else {
            vec![batch, args.top_k]
        };
        let gates_shape = if batch == 1 {
            vec![args.top_k, 1, 1]
        } else {
            vec![batch, args.top_k, 1, 1]
        };
        let checked = built.and_then(|()| {
            if native_output_valid(out_a.0, &out_shape, args.act, hidden_bytes)
                && native_output_valid(ids_a.0, &ids_shape, Dtype::U32, k_bytes)
                && native_output_valid(gates_a.0, &gates_shape, Dtype::F32, k_bytes)
                && native_output_valid(health_a.0, &[batch], Dtype::F32, health_bytes)
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
            out: adapter.resident(out_a, args.act, out_shape, hidden_bytes),
            ids: adapter.resident(ids_a, Dtype::U32, ids_shape, k_bytes),
            gates: adapter.resident(gates_a, Dtype::F32, gates_shape, k_bytes),
            health: adapter.resident(health_a, Dtype::F32, vec![batch], health_bytes),
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
        check(status, "compile cache clear")?;
        super::super::compile_stats::counted_clear();
        Ok(())
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

impl Context {
    /// Graph-cache timing probe (test builds only): compiles the fixed-shape
    /// rows=1 Flint layer-subgraph stand-in with the fixed cache key
    /// [`FLINT_LAYER_PROBE_FUN_ID`] (shapeless false, so a same-shape replay is
    /// a cache hit) and applies it once to the 10 operands; the output is lazy.
    #[cfg(all(test, feature = "native-fault-qualification"))]
    #[allow(clippy::too_many_arguments)]
    pub(in crate::kernels::mlx) fn compile_probe_flint_layer(
        &mut self,
        x: &ExecutionTensor,
        wq: &ExecutionTensor,
        wk: &ExecutionTensor,
        wv: &ExecutionTensor,
        wo: &ExecutionTensor,
        wg: &ExecutionTensor,
        wu: &ExecutionTensor,
        wd: &ExecutionTensor,
        gamma1: &ExecutionTensor,
        gamma2: &ExecutionTensor,
        eps: f32,
    ) -> Result<ExecutionTensor, EngError> {
        let label = "flint layer compile probe";
        let _dispatch = DispatchGuard::enter();
        self.check_ready()?;
        let state = self.state.as_mut().unwrap();
        for operand in [x, wq, wk, wv, wo, wg, wu, wd, gamma1, gamma2] {
            check_operand(label, state, operand)?;
        }
        let inputs = [
            x.storage.native_array(),
            wq.storage.native_array(),
            wk.storage.native_array(),
            wv.storage.native_array(),
            wo.storage.native_array(),
            wg.storage.native_array(),
            wu.storage.native_array(),
            wd.storage.native_array(),
            gamma1.storage.native_array(),
            gamma2.storage.native_array(),
        ];
        clear_message();
        let stream = state.stream.0;
        let mut out = Array::null();
        let built = produce(&mut out, label, |slot| unsafe {
            ffi::mlx_compile_flint_layer_probe(
                slot,
                inputs.as_ptr(),
                inputs.len(),
                eps,
                FLINT_LAYER_PROBE_FUN_ID as usize,
                stream,
            )
        });
        if built.is_err() {
            mark_failed();
        }
        built?;
        let shape = vec![1, 2048];
        let byte_len = 2048 * 4;
        if !native_output_valid(out.0, &shape, Dtype::F32, byte_len) {
            mark_failed();
            return Err(failure(label));
        }
        Ok(state.resident(out, Dtype::F32, shape, byte_len))
    }
}

impl Context {
    /// Test-only per-kernel floor calibration: compiles a chain of `m`
    /// back-to-back adds on `x` with the fixed cache key
    /// [`ADD_CHAIN_PROBE_FUN_ID`] (shapeless false) and applies it once; the
    /// output is lazy. Same-shape replays are cache hits.
    #[cfg(all(test, feature = "native-fault-qualification"))]
    pub(in crate::kernels::mlx) fn compile_probe_add_chain(
        &mut self,
        x: &ExecutionTensor,
        m: u32,
    ) -> Result<ExecutionTensor, EngError> {
        let label = "add chain compile probe";
        let _dispatch = DispatchGuard::enter();
        self.check_ready()?;
        let state = self.state.as_mut().unwrap();
        check_operand(label, state, x)?;
        let inputs = [x.storage.native_array()];
        clear_message();
        let stream = state.stream.0;
        let mut out = Array::null();
        let built = produce(&mut out, label, |slot| unsafe {
            ffi::mlx_compile_add_chain_probe(
                slot,
                inputs.as_ptr(),
                inputs.len(),
                m,
                ADD_CHAIN_PROBE_FUN_ID as usize,
                stream,
            )
        });
        if built.is_err() {
            mark_failed();
        }
        built?;
        let shape = x.storage.shape().to_vec();
        let byte_len = x.storage.byte_len();
        if !native_output_valid(out.0, &shape, Dtype::F32, byte_len) {
            mark_failed();
            return Err(failure(label));
        }
        Ok(state.resident(out, Dtype::F32, shape, byte_len))
    }
}

/// The flat input list and baked constants of one compiled Flint decode tail (graph-cache
/// slice 1). Every tensor and weight must belong to this adapter.
pub(crate) struct FlintTailArgs<'a> {
    /// The residual input `[rows, hidden]`: the layer input before the attention.
    pub(crate) x: &'a ExecutionTensor,
    /// The attention output `[rows, hidden]`: the out projection's product, before any
    /// residual add (the closure's first op adds it to `x`).
    pub(crate) attended: &'a ExecutionTensor,
    /// The post-attention RMS norm weight `[hidden]`.
    pub(crate) gamma: &'a ExecutionTensor,
    /// The dense SwiGLU gate, up, down weights.
    pub(crate) packed: [&'a ResidentPackedWeight; 3],
    /// The hidden width `H`.
    pub(crate) hidden: u32,
    /// The RMS norm epsilon.
    pub(crate) eps: f32,
}

impl Context {
    /// The Flint decode tail (graph-cache slice 1) as one `mlx_compile`d closure: the
    /// post-attention residual add, the RMS norm, and the dense SwiGLU FFN, replayed from the
    /// thread-local MLX compile cache. The closure body calls the same `inference_mlx_*`
    /// entries the eager tail calls, in the same order, so a replay is bit-identical to the
    /// eager op sequence at the same shapes and dtypes. The compile cache key
    /// (`kernels::compile_key::flint_tail_fun_id`) covers the row count, the hidden width, the
    /// eps bits, the activation dtype, and the scale dtype and quantization triple of each
    /// packed weight, and the per-adapter registry admits it before any native call. There is
    /// NO fallback to the uncompiled path once the caller routed here (decision 0032/0023): a
    /// failure of the compiled call is a native failure.
    pub(in crate::kernels::mlx) fn flint_compiled_tail(
        &mut self,
        args: &FlintTailArgs<'_>,
    ) -> Result<ExecutionTensor, EngError> {
        let label = "flint compiled tail";
        let _dispatch = DispatchGuard::enter();
        self.check_ready()?;
        let state = self.state.as_mut().unwrap();
        let (x, attended, gamma) = (args.x, args.attended, args.gamma);
        let packed = args.packed;
        let hidden = args.hidden;
        // The tail runs in the layer's activation dtype: F32 in test fixtures, BF16 on the
        // served Flint path (the BF16 embedding policy plus `allow_bf16`). Every plain
        // operand carries that dtype, the closure replays the same native entries the eager
        // tail calls at that dtype, and the output keeps it.
        let act = x.dtype();
        check_float(label, act)?;
        check_operand(label, state, x)?;
        check_operand(label, state, attended)?;
        check_operand(label, state, gamma)?;
        for weight in packed {
            for part in [weight.payload(), weight.scales()]
                .into_iter()
                .chain(weight.biases())
            {
                if part.owner_key() != state.owner {
                    return Err(EngError::BadRequest(format!(
                        "{label}: weight belongs to another native owner"
                    )));
                }
            }
        }
        if attended.dtype() != act || gamma.dtype() != act {
            return Err(EngError::BadRequest(format!(
                "{label}: the attention output and the norm must carry the activation dtype {act:?}"
            )));
        }
        let rows = match x.shape() {
            [rows, width] if *width == hidden => *rows,
            _ => {
                return Err(EngError::BadRequest(format!(
                    "{label}: the tail input must be [rows, {hidden}]"
                )));
            }
        };
        if attended.shape() != x.shape() {
            return Err(EngError::BadRequest(format!(
                "{label}: the attention output must match the tail input shape"
            )));
        }
        if gamma.shape().len() != 1 || gamma.shape()[0] != hidden {
            return Err(EngError::BadRequest(format!(
                "{label}: the post-attention norm must hold the width {hidden}"
            )));
        }
        // gate/up project hidden -> ffn and down projects ffn -> hidden; the FFN width only
        // shapes the closure body's intermediates, so the check pins the consistency, not the
        // value.
        let [gate, up, down] = packed;
        let consistent = matches!(
            (gate.logical_shape(), up.logical_shape(), down.logical_shape()),
            ([g_out, g_in], [u_out, u_in], [d_out, d_in])
                if *g_in == hidden
                    && *u_in == hidden
                    && *g_out == *u_out
                    && *d_in == *g_out
                    && *d_out == hidden
        );
        if !consistent {
            return Err(EngError::BadRequest(format!(
                "{label}: gate/up must project {hidden} and down must project back"
            )));
        }
        // Affine packed weights only: the closure transport cannot carry an absent bias
        // (the NULL placeholder a bias-free mode would need is not a valid vector element
        // for the compile shim), and the pinned Flint loader is affine anyway.
        let eps = args.eps;
        let triples = quant_triples(packed)?;
        for triple in &triples {
            if triple[0] != 0 {
                return Err(EngError::BadRequest(format!(
                    "{label}: the compiled tail takes affine packed weights only"
                )));
            }
        }
        let fun_id = compile_key::flint_tail_fun_id(&compile_key::FlintTailBaked {
            rows,
            hidden,
            eps_bits: eps.to_bits(),
            act: compile_key::dtype_key(act),
            scale_keys: packed
                .each_ref()
                .map(|weight| compile_key::dtype_key(weight.scales().dtype())),
            quant: triples,
        });
        // The per-adapter program registry decides before any native call, exactly as for the
        // delta and MoE blocks.
        state.programs.admit(fun_id)?;
        let mut inputs: Vec<ffi::mlx_array> = Vec::with_capacity(12);
        inputs.push(x.storage.native_array());
        inputs.push(attended.storage.native_array());
        inputs.push(gamma.storage.native_array());
        for weight in packed {
            inputs.push(weight.payload().storage.native_array());
            inputs.push(weight.scales().storage.native_array());
            inputs.push(
                weight
                    .biases()
                    .map(|biases| biases.storage.native_array())
                    .unwrap_or(ffi::NULL_ARRAY),
            );
        }
        let mut keep: Vec<_> = [x, attended, gamma]
            .into_iter()
            .map(|tensor| tensor.share())
            .collect();
        for weight in packed {
            keep.push(weight.payload().share());
            keep.push(weight.scales().share());
            if let Some(biases) = weight.biases() {
                keep.push(biases.share());
            }
        }
        let group: [i32; 3] = [triples[0][1], triples[1][1], triples[2][1]];
        let bits: [i32; 3] = [triples[0][2], triples[1][2], triples[2][2]];
        let mode: [i32; 3] = [triples[0][0], triples[1][0], triples[2][0]];
        clear_message();
        let stream = state.stream.0;
        let mut out = Array::null();
        let built = produce(&mut out, label, |slot| unsafe {
            ffi::mlx_compile_flint_tail(
                slot,
                inputs.as_ptr(),
                inputs.len(),
                eps,
                group.as_ptr(),
                bits.as_ptr(),
                mode.as_ptr(),
                fun_id as usize,
                stream,
            )
        });
        if built.is_err() {
            mark_failed();
        }
        built?;
        let shape = vec![rows, hidden];
        let byte_len = u64::from(rows)
            .checked_mul(u64::from(hidden))
            .and_then(|elements| elements.checked_mul(act.byte_width()))
            .ok_or_else(|| EngError::BadRequest(format!("{label} output bytes exceed u64")))?;
        if !native_output_valid(out.0, &shape, act, byte_len) {
            mark_failed();
            return Err(failure(label));
        }
        Ok(state.resident(out, act, shape, byte_len))
    }
}
