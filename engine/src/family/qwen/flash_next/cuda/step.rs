//! The Flash-Next decode step on the device: one token, one row, queued on one owned stream.
//!
//! WHAT: [`FlashStep`] owns the two arenas the state and step plans describe (the persistent state and the
//! activations of one token), borrows the trunk weights the loader uploaded and the context and stream of
//! the caller, and runs the layers in the site order of the host reference
//! (`family/qwen/flash_next/step.rs`): per layer the attention hyper-connection read, the mixer
//! (Gated DeltaNet or QSA, `mixer.rs`), the apply, the MLP read, the MoE block and its apply; after
//! the last layer the final mixer, `lm_head` and the logits. The launch order is the phase order of
//! `step_plan::Phase`.
//!
//! WHY a struct over views: every tensor of the token lives in the step arena at an offset the
//! planner chose by lifetime, so tensors of different phases share addresses. The launches must
//! keep the phase order of `step_plan::Phase`.
//!
//! Reused: the kernel wrappers of the resident decode (hyper, `gdn_conv`, `gdn_scan_seq`, the
//! router's strided dot, `attn_prep`, the `qsa_*` kernels, `exl3_linear`), Qwen3.6's EXL3 MoE block
//! (`run_exl3_moe`, shared with this family) and `CudaWeights` keyed by pack name. New: the
//! op lists, the Flash sites and the glue between them.
//!
//! MoE: [`FlashStep::moe_block`] is the seam. It runs the layer's MoE through Qwen3.6's block with
//! the routed experts as an all-resident bank (`Exl3Bank`), which is what the reduced-layer parity
//! runs use; the split block puts the partial slot bank behind the same call.
//!
//! One row only: the step plan allows up to 8, the shared MoE block takes one.
//!
//! The PLE block is NOT part of this step. The model applies a per-token update to the residual
//! stream before the attention read of the PLE layer; the caller does it on
//! the host between two calls, with the stream read back and written again. A range of layers that
//! would run the PLE layer after an earlier layer of the same call therefore has the wrong stream
//! from the PLE layer on, and [`FlashStep::run_layers`] refuses it: call it with `0..ple_layer`,
//! apply the update, then call it with `ple_layer..layers` (or [`FlashStep::run_layer`] one layer
//! at a time, which does not check). A range that starts at the PLE layer is the caller's promise
//! that the update was applied. The layer and head tests of `step_tests.rs` run without PLE on both sides, on
//! purpose, through `run_layers_without_ple`, a test-only entry point.
//!
//! Errors: after any error in a launch call the step is poisoned (its stream or state may be half
//! written) and every later call except [`FlashStep::reset`] refuses, as the host reference does.
//! [`FlashStep::finish_token`] turns a nonzero status word or EXL3 flag into an error, after the
//! drain; a step that finished a token with a fault is poisoned too.

// Staged: the executor is the first caller of the whole step.
#![cfg_attr(not(test), allow(dead_code, reason = "staged for the executor"))]

use crate::family::engine_owner::Handle;
use crate::family::exl3::launch::{Launch, MoeDims, RouteBufs};
use crate::kernels::cuda::resident::attn_prep::MAX_DECODE_ROWS;
use std::cell::Cell;

use crate::{
    EngError,
    family::{
        exl3::{
            moe::{Exl3MoeBufs, Exl3MoeWeights, Scratch, run_exl3_moe},
            prefill::{Exl3Scratch, Output, Projector, write_dense_jobs},
        },
        exl3_pack::layer_name,
        gguf_model::Role,
        prefill::arena::{ArenaHooks, allocate_zeroed},
        project::cuda::{
            CudaWeight, CudaWeights,
            exl3::{Exl3Io, Exl3JobsIo, Exl3Weight},
        },
        qwen::flash_next::{
            cuda::{
                moe_block::{SplitMoe, SplitMoeTrunk},
                state::{FlashGeometry, QsaCacheStorage, StatePlan, StateSlot, WORD_STEP_PARAMS},
                step_plan::{PREFILL_TILE_ROWS, Site, Slot, StepOptions, StepPlan},
            },
            load::model_shape,
            moe::context_budget::KvFormat,
            step::{GgufDecodePrecision, ModelShape},
        },
    },
    kernels::cuda::{
        CudaContext, DeviceArena, DeviceBuffer, DeviceRange, DeviceStream, NativeWeight,
        prefill::gdn_scan_seq::GdnForm,
        resident::{
            dense_gemv::{DenseDtype, DenseGemvCall, DenseGemvShape},
            exl3::Exl3Job,
            hyper::{
                HyperApplyCall, HyperDtype, HyperReadCall, HyperScratch, HyperShape, HyperWeights,
            },
            matvec_gguf_f32::DenseF32Call,
            matvec_q::MatvecCall,
            quantize_q8_1::{QuantizeCall, q8_1_bytes},
            step_params::{OFFSET_POS, OFFSET_STATUS},
        },
    },
    weights::GgufType,
};

pub(super) const F32: usize = size_of::<f32>();

pub(super) fn refuse(detail: String) -> EngError {
    EngError::BadRequest(format!("Flash-Next step: {detail}"))
}

/// One staged native GGUF decode projection with caller-owned, stream-ordered scratch.
pub(super) struct NativeProjection<'a> {
    pub(super) weight: &'a NativeWeight,
    pub(super) input: DeviceRange<'a>,
    pub(super) output: DeviceRange<'a>,
    pub(super) scratch: DeviceRange<'a>,
}

pub(super) fn project_native(
    ctx: &CudaContext,
    stream: &DeviceStream,
    call: NativeProjection<'_>,
) -> Result<(), EngError> {
    if call.weight.experts().is_some() {
        return Err(refuse(
            "native projection requires a matrix, not an expert stack".into(),
        ));
    }
    ctx.validate_native_projection(call.weight, call.input, call.scratch, call.output)?;
    ctx.quantize_q8_1(
        stream,
        QuantizeCall {
            x: call.input,
            y: call.scratch,
            k: call.weight.k(),
            ncols: 1,
        },
    )?;
    ctx.matvec_native(stream, call.weight, call.scratch, call.output, 1)
}

/// One exact native GGUF decode projection ([`GgufDecodePrecision::Exact`]): the f32 activation
/// against the exactly decoded weights of any type the window decoders serve. No scratch.
pub(super) fn project_exact(
    ctx: &CudaContext,
    stream: &DeviceStream,
    weight: &NativeWeight,
    input: DeviceRange<'_>,
    output: DeviceRange<'_>,
) -> Result<(), EngError> {
    if weight.experts().is_some() {
        return Err(refuse(
            "exact projection requires a matrix, not an expert stack".into(),
        ));
    }
    ctx.matvec_gguf_f32(
        stream,
        DenseF32Call {
            ty: weight.quant(),
            w: weight.buffer().range(0, weight.buffer().bytes())?,
            x: input,
            y: output,
            k: weight.k(),
            rows: weight.n_rows(),
            ncols: 1,
        },
    )
}

/// Test-only experiment: retain the native Q6_K weights and use the existing dequantizing
/// tensor-core GEMM, which rounds activations and decoded weights to FP16 rather than Q8_1.
#[cfg(test)]
pub(super) fn project_q6_f16(
    ctx: &CudaContext,
    stream: &DeviceStream,
    call: NativeProjection<'_>,
) -> Result<(), EngError> {
    use crate::kernels::cuda::{prefill::gemm_dq::GemmDqTypedCall, resident::matvec_q::QuantType};
    if call.weight.quant() != QuantType::Q6K || call.weight.experts().is_some() {
        return Err(refuse(
            "FP16 DeltaNet experiment requires a Q6_K matrix".into(),
        ));
    }
    ctx.gemm_dq_codebook(
        stream,
        QuantType::Q6K,
        GemmDqTypedCall {
            w: call
                .weight
                .buffer()
                .range(0, call.weight.buffer().bytes())?,
            x: call.input,
            y: call.output,
            k: call.weight.k(),
            rows: call.weight.n_rows(),
            ncols: 1,
            x_col_stride: 0,
            y_col_stride: 0,
            x_f32: true,
        },
    )
}

/// Test-only native Q6_K projection using F32 activations and accumulation.
#[cfg(test)]
pub(super) fn project_q6_f32(
    ctx: &CudaContext,
    stream: &DeviceStream,
    call: NativeProjection<'_>,
) -> Result<(), EngError> {
    use crate::kernels::cuda::resident::{matvec_q::QuantType, q6_f32::Q6F32Call};
    if call.weight.quant() != QuantType::Q6K || call.weight.experts().is_some() {
        return Err(refuse(
            "F32 native projection experiment requires a Q6_K matrix".into(),
        ));
    }
    ctx.q6_f32(
        stream,
        Q6F32Call {
            w: call
                .weight
                .buffer()
                .range(0, call.weight.buffer().bytes())?,
            x: call.input,
            y: call.output,
            rows: call.weight.n_rows(),
            k: call.weight.k(),
        },
    )
}

/// The pack name of the hyper-connection mixer that feeds `lm_head`.
/// Activation rows one native matvec launch takes.
const MATVEC_COLUMNS: usize = 8;

pub(super) const MIXER: &str = "model.language_model.hyper_connection_mixer";

/// The step of one token over a window of layers `first_layer..end_layer` (all 48 for a full
/// model, a few for a parity window).
pub(super) struct FlashStep<'a> {
    pub(super) ctx: Handle<'a, CudaContext>,
    pub(super) stream: Handle<'a, DeviceStream>,
    pub(super) weights: Handle<'a, CudaWeights>,
    pub(super) geometry: FlashGeometry,
    pub(super) model: ModelShape,
    pub(super) state_plan: StatePlan,
    pub(super) step_plan: StepPlan,
    pub(super) options: StepOptions,
    /// The arithmetic of the native GGUF decode projections of a decode owner
    /// ([`Self::new_gguf_decode`]): [`GgufDecodePrecision::default`] until
    /// [`Self::set_gguf_decode_precision`]. Any other step never reads it.
    gguf_decode: GgufDecodePrecision,
    /// True when the QSA attention caches are f32 ([`QsaCacheStorage::F32`]): the mixer writes the key
    /// and value rows in f32 and reads them with `qsa_attn_exact`, so nothing is rounded at the
    /// store. The GGUF decode owner's choice ([`FlashStep::new_gguf_decode`], from
    /// [`gguf_qsa_cache_storage`]); every other step has fp16 caches.
    pub(super) qsa_cache_f32: bool,
    state: DeviceArena,
    scratch: DeviceArena,
    /// f32 `[hidden]` zeros: the residual the shared MoE block adds to, since the hyper apply adds
    /// the block output to the stream instead.
    zeros: DeviceBuffer,
    pub(super) layers: std::ops::Range<usize>,
    /// The q8_1 pack of the activation the native projections read: one row, padded.
    xq: DeviceBuffer,
    /// The scan's rules: the Hugging Face head order of the EXL3 pack, or the tiled order of a
    /// GGUF.
    pub(super) gdn_form: GdnForm,
    /// The plan's row count: every launch takes `m` rows with `m <= rows` (decode is 1, a chunk
    /// is its token count). The slots hold this many rows.
    pub(super) rows: usize,
    /// The shared owner also serves decode: only an open prefill window uses its new launches.
    pub(super) prefill_window: Cell<bool>,
    /// Set by any error of a launch call; cleared by [`Self::reset`].
    poisoned: Cell<bool>,
}

/// What the GGUF decode owner adds to the step: the exact scratch plan's `max_k` and the
/// storage of the QSA attention caches.
#[derive(Debug, Clone, Copy)]
pub(super) struct GgufDecode {
    pub(super) max_k: usize,
    pub(super) qsa_cache: QsaCacheStorage,
}

/// The bytes of the Q8_1 pack of the widest activation a native projection reads: the hidden
/// width, the attention and DeltaNet output widths and the expert widths; at least 64 KiB. The
/// step uploads one at construction; a figure for the step's payload counts it.
pub(super) fn activation_pack_bytes(geometry: &FlashGeometry) -> Result<usize, EngError> {
    let widest = [
        geometry.hidden,
        geometry.qsa.q_width(),
        geometry.delta.value_width(),
        geometry.inter,
        geometry.shared_inter,
    ]
    .into_iter()
    .max()
    .unwrap_or(0);
    Ok(q8_1_bytes(widest, MATVEC_COLUMNS)
        .ok_or_else(|| refuse("the activation pack size overflows".into()))?
        .max(64 << 10))
}

/// The storage of the QSA attention caches of the GGUF decode owner
/// ([`FlashStep::new_gguf_decode`]): fp16, unless `FIFI_FLASH_GGUF_QSA_CACHE` is `f32`, which
/// rounds nothing at the store and takes twice the cache bytes. An opt-in setting until the f32
/// caches are qualified, so that every step the owner builds follows it; any other value is
/// refused.
pub(super) fn gguf_qsa_cache_storage() -> Result<QsaCacheStorage, EngError> {
    let value = match std::env::var("FIFI_FLASH_GGUF_QSA_CACHE") {
        Ok(value) => Some(value),
        Err(std::env::VarError::NotPresent) => None,
        Err(error) => {
            return Err(EngError::BadRequest(format!(
                "FIFI_FLASH_GGUF_QSA_CACHE is not valid UTF-8: {error}"
            )));
        }
    };
    parse_qsa_cache_storage(value.as_deref()).map_err(|rule| {
        EngError::BadRequest(format!(
            "FIFI_FLASH_GGUF_QSA_CACHE {rule}, got {:?}",
            value.as_deref().unwrap_or("")
        ))
    })
}

/// The storage the setting names: unset or `f16` is fp16, `f32` is f32. Anything else breaks the
/// rule the error names.
pub(super) fn parse_qsa_cache_storage(
    value: Option<&str>,
) -> Result<QsaCacheStorage, &'static str> {
    match value {
        None | Some("f16") => Ok(QsaCacheStorage::F16),
        Some("f32") => Ok(QsaCacheStorage::F32),
        Some(_) => Err("must be f16 or f32"),
    }
}

impl<'a> FlashStep<'a> {
    /// The step over `layers` of the model in `model_config`, with caches for `positions`
    /// positions. The state is zeroed: a new sequence.
    pub(super) fn new(
        ctx: impl Into<Handle<'a, CudaContext>>,
        stream: impl Into<Handle<'a, DeviceStream>>,
        weights: impl Into<Handle<'a, CudaWeights>>,
        model_config: &[u8],
        positions: usize,
        rows: usize,
        layers: std::ops::Range<usize>,
    ) -> Result<Self, EngError> {
        Self::build(
            ctx,
            stream,
            weights,
            model_shape(model_config)?,
            positions,
            rows,
            layers,
            false,
            None,
        )
    }

    /// Runs the DeltaNet scan in `form` (the tiled head order of a GGUF).
    pub(in crate::family::qwen::flash_next) fn with_gdn_form(mut self, form: GdnForm) -> Self {
        self.gdn_form = form;
        self
    }

    /// The step over a model whose shape the caller already has (a GGUF, which carries no
    /// `config.json`): `kinds` is true for the QSA layers.
    pub(in crate::family::qwen::flash_next) fn from_shape(
        ctx: impl Into<Handle<'a, CudaContext>>,
        stream: impl Into<Handle<'a, DeviceStream>>,
        weights: impl Into<Handle<'a, CudaWeights>>,
        shape: (ModelShape, Vec<bool>),
        positions: usize,
        rows: usize,
        layers: std::ops::Range<usize>,
    ) -> Result<Self, EngError> {
        Self::build(
            ctx, stream, weights, shape, positions, rows, layers, false, None,
        )
    }

    /// The shared-prefill owner, with dense and routed scratch from the shared plan.
    pub(super) fn new_prefill(
        ctx: impl Into<Handle<'a, CudaContext>>,
        stream: impl Into<Handle<'a, DeviceStream>>,
        weights: impl Into<Handle<'a, CudaWeights>>,
        model_config: &[u8],
        positions: usize,
        rows: usize,
        layers: std::ops::Range<usize>,
    ) -> Result<Self, EngError> {
        Self::build(
            ctx,
            stream,
            weights,
            model_shape(model_config)?,
            positions,
            rows,
            layers,
            true,
            None,
        )
    }

    /// Construct a single-token GGUF decode owner with its exact scratch plan. The QSA caches
    /// are stored as [`gguf_qsa_cache_storage`] says: fp16 unless the setting opts into f32.
    #[expect(clippy::too_many_arguments)]
    pub(super) fn new_gguf_decode(
        ctx: impl Into<Handle<'a, CudaContext>>,
        stream: impl Into<Handle<'a, DeviceStream>>,
        weights: impl Into<Handle<'a, CudaWeights>>,
        model: ModelShape,
        kinds: &[bool],
        positions: usize,
        max_k: usize,
        layers: std::ops::Range<usize>,
    ) -> Result<Self, EngError> {
        Self::new_gguf_decode_in(
            ctx,
            stream,
            weights,
            model,
            kinds,
            positions,
            max_k,
            gguf_qsa_cache_storage()?,
            layers,
        )
    }

    /// [`Self::new_gguf_decode`] with the QSA caches stored in `qsa_cache`, whatever the setting
    /// says: fp16, or [`QsaCacheStorage::F32`], which rounds nothing at the store and takes twice
    /// the cache bytes. The f32 rows entry of the attention preparation takes at most
    /// [`MAX_DECODE_ROWS`] rows a step.
    #[expect(clippy::too_many_arguments)]
    pub(super) fn new_gguf_decode_in(
        ctx: impl Into<Handle<'a, CudaContext>>,
        stream: impl Into<Handle<'a, DeviceStream>>,
        weights: impl Into<Handle<'a, CudaWeights>>,
        model: ModelShape,
        kinds: &[bool],
        positions: usize,
        max_k: usize,
        qsa_cache: QsaCacheStorage,
        layers: std::ops::Range<usize>,
    ) -> Result<Self, EngError> {
        Self::build(
            ctx,
            stream,
            weights,
            (model, kinds.to_vec()),
            positions,
            1,
            layers,
            false,
            Some(GgufDecode { max_k, qsa_cache }),
        )
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "one constructor for the two arena lists"
    )]
    fn build(
        ctx: impl Into<Handle<'a, CudaContext>>,
        stream: impl Into<Handle<'a, DeviceStream>>,
        weights: impl Into<Handle<'a, CudaWeights>>,
        (model, kinds): (ModelShape, Vec<bool>),
        positions: usize,
        rows: usize,
        layers: std::ops::Range<usize>,
        shared_prefill: bool,
        gguf: Option<GgufDecode>,
    ) -> Result<Self, EngError> {
        let (ctx, stream, weights) = (ctx.into(), stream.into(), weights.into());
        let geometry = FlashGeometry::new(&model, &kinds)?;
        if layers.start >= layers.end || layers.end > geometry.layers.len() {
            return Err(refuse(format!(
                "layers {layers:?} are outside the model's {}",
                geometry.layers.len()
            )));
        }
        let qsa_cache = gguf.map_or(QsaCacheStorage::F16, |decode| decode.qsa_cache);
        if qsa_cache == QsaCacheStorage::F32 && rows > MAX_DECODE_ROWS {
            return Err(refuse(format!(
                "f32 QSA caches take at most {MAX_DECODE_ROWS} rows a step, got {rows}"
            )));
        }
        let state_plan = StatePlan::with_qsa_cache(&geometry, positions, KvFormat::F16, qsa_cache)?;
        let options = StepOptions::for_sm_count(ctx.sm_count());
        let step_plan = if let Some(decode) = gguf {
            StepPlan::gguf_decode(&geometry, positions, &options, decode.max_k)?
        } else if shared_prefill {
            StepPlan::new_prefill(&geometry, rows, positions, &options)?
        } else {
            StepPlan::new(&geometry, rows, positions, &options)?
        };
        let state = ctx.arena(&state_plan.arena_tensors())?;
        let tensors = step_plan.arena_tensors();
        let scratch = if shared_prefill {
            allocate_zeroed(&ctx, &stream, &tensors, false, ArenaHooks::default())?
        } else {
            ctx.arena(&tensors)?
        };
        let zeros = ctx.upload(&vec![0u8; rows * geometry.hidden * F32])?;
        let xq = ctx.upload(&vec![0u8; activation_pack_bytes(&geometry)?])?;
        let step = Self {
            ctx,
            stream,
            weights,
            geometry,
            model,
            state_plan,
            step_plan,
            options,
            gguf_decode: gguf.map_or(GgufDecodePrecision::Q8, |_| GgufDecodePrecision::default()),
            qsa_cache_f32: qsa_cache == QsaCacheStorage::F32,
            state,
            scratch,
            zeros,
            layers,
            xq,
            gdn_form: GdnForm::FLASH_NEXT,
            rows,
            prefill_window: Cell::new(false),
            poisoned: Cell::new(false),
        };
        step.reset()?;
        Ok(step)
    }

    #[cfg(test)]
    pub(super) fn scratch_bytes_for_test(&self) -> usize {
        self.scratch.bytes()
    }

    /// Selects the arithmetic of the native GGUF decode projections ([`GgufDecodePrecision`]).
    /// The scratch of both is planned either way, so an owner may switch between tokens.
    pub(super) fn set_gguf_decode_precision(&mut self, precision: GgufDecodePrecision) {
        self.gguf_decode = precision;
    }

    /// The arithmetic in force for the native GGUF decode projections.
    pub(super) fn gguf_decode_precision(&self) -> GgufDecodePrecision {
        self.gguf_decode
    }

    /// True for a step built with the GGUF decode description ([`Self::new_gguf_decode`]): the
    /// plan holds the decode scratch slot, and a native weight is projected on the owner's terms
    /// (`project_native`, `project_exact`), one row. Any other step projects a native weight with
    /// its own q8_1 pack ([`Self::project_native`]), as many rows as it holds.
    pub(super) fn gguf_decode_step(&self) -> bool {
        self.step_plan.index(Slot::GgufDecodeQ8).is_some()
    }

    /// Zeroes the persistent state (the recurrent states, the rings, the caches) and clears the
    /// poison: a new sequence. The device has no memset wrapper yet, so the zeros go over in chunks
    /// of 16 MiB; a state of 6 GiB (a 262,144-position context) takes about half a second.
    ///
    /// With the prefix-resume switch armed, only the fixed state is zeroed (the recurrent
    /// states, the conv rings, the PLE ring, the words and the raw-key ring): the QSA planes and
    /// the indexer keys stay in place for a restore to rewind the valid length onto (see
    /// [`StatePlan::is_position_state`] for why a fresh sequence never observes them).
    pub(super) fn reset(&self) -> Result<(), EngError> {
        self.prefill_window.set(false);
        let zeros = vec![0u8; 1 << 24];
        let ranges: Vec<(usize, usize)> = if crate::family::qwen::flash_next::cuda::resume::on() {
            self.state_plan
                .slots()
                .iter()
                .filter(|&&(slot, _)| !StatePlan::is_position_state(slot))
                .filter_map(|&(slot, _)| self.state_plan.placement(slot))
                .collect()
        } else {
            vec![(0, self.state.bytes())]
        };
        for (start, bytes) in ranges {
            let mut at = start;
            let end = start
                .checked_add(bytes)
                .ok_or_else(|| refuse("state arena offset overflows".into()))?;
            while at < end {
                let len = zeros.len().min(end - at);
                self.state
                    .buffer()
                    .write_at(&self.stream, at, &zeros[..len])?;
                at += len;
            }
        }
        self.poisoned.set(false);
        Ok(())
    }

    /// Reads `slot` of the persistent state back as raw little-endian bytes, behind the queued
    /// work. The prefix-resume snapshot is built from these.
    pub(super) fn read_state_bytes(&self, slot: StateSlot) -> Result<Vec<u8>, EngError> {
        let range = self.state_slot(slot)?;
        let (_, bytes) = self
            .state_plan
            .placement(slot)
            .ok_or_else(|| refuse(format!("the state has no slot {slot:?}")))?;
        let mut out = vec![0u8; bytes];
        range
            .buffer()
            .read_at(&self.stream, range.offset(), &mut out)?;
        Ok(out)
    }

    /// Writes `slot` of the persistent state from raw bytes, ordered before the next launch.
    /// A prefix restore installs its snapshot through these.
    pub(super) fn write_state_bytes(
        &mut self,
        slot: StateSlot,
        bytes: &[u8],
    ) -> Result<(), EngError> {
        let range = self.state_slot(slot)?;
        let (_, want) = self
            .state_plan
            .placement(slot)
            .ok_or_else(|| refuse(format!("the state has no slot {slot:?}")))?;
        if bytes.len() != want {
            return Err(refuse(format!(
                "the state slot {slot:?} holds {want} bytes, not {}",
                bytes.len()
            )));
        }
        range.buffer().write_at(&self.stream, range.offset(), bytes)
    }

    /// Refuses while the step is poisoned.
    fn ensure_usable(&self) -> Result<(), EngError> {
        if self.poisoned.get() {
            return Err(refuse(
                "an earlier call failed; the step needs a reset".to_owned(),
            ));
        }
        Ok(())
    }

    /// Poisons the step when `result` is an error.
    fn poison_on<T>(&self, result: Result<T, EngError>) -> Result<T, EngError> {
        if result.is_err() {
            self.poisoned.set(true);
        }
        result
    }

    /// True when an error has poisoned the step.
    pub(super) fn is_poisoned(&self) -> bool {
        self.poisoned.get()
    }

    /// The tensor of `slot` in the step arena.
    pub(super) fn slot(&self, slot: Slot) -> Result<DeviceRange<'_>, EngError> {
        let index = self
            .step_plan
            .index(slot)
            .ok_or_else(|| refuse(format!("the plan has no slot {slot:?}")))?;
        self.scratch.tensor(index)
    }

    /// The tensor of `slot` in the state arena.
    pub(super) fn state_slot(&self, slot: StateSlot) -> Result<DeviceRange<'_>, EngError> {
        let index = self
            .state_plan
            .index(slot)
            .ok_or_else(|| refuse(format!("the state has no slot {slot:?}")))?;
        self.state.tensor(index)
    }

    /// The `StepParams` block in the words tensor, and one of its words.
    pub(super) fn params(&self) -> Result<DeviceRange<'_>, EngError> {
        let words = self.state_slot(StateSlot::Words)?;
        words.buffer().range(
            words.offset() + WORD_STEP_PARAMS,
            crate::kernels::cuda::resident::step_params::STEP_PARAMS_BYTES,
        )
    }

    pub(super) fn pos_word(&self) -> Result<DeviceRange<'_>, EngError> {
        let params = self.params()?;
        params.buffer().range(params.offset() + OFFSET_POS, 4)
    }

    pub(super) fn status_word(&self) -> Result<DeviceRange<'_>, EngError> {
        let params = self.params()?;
        params.buffer().range(params.offset() + OFFSET_STATUS, 4)
    }

    /// The device buffer and offset the token start writes: the words block of the state arena.
    pub(super) fn words_buffer(&self) -> Result<(&DeviceBuffer, usize), EngError> {
        let words = self.state_slot(StateSlot::Words)?;
        Ok((self.state.buffer(), words.offset()))
    }

    /// The offset of `slot` in the step arena buffer.
    pub(super) fn slot_offset(&self, slot: Slot) -> Result<usize, EngError> {
        Ok(self.slot(slot)?.offset())
    }

    /// Reads every copied slot of the persistent state (see [`StatePlan::is_position_state"])
    /// as raw bytes, behind the queued work: the prefix-resume snapshot.
    pub(super) fn read_fixed_state(&self) -> Result<Vec<(StateSlot, Vec<u8>)>, EngError> {
        self.state_plan
            .slots()
            .iter()
            .filter(|&&(slot, _)| !StatePlan::is_position_state(slot))
            .map(|&(slot, _)| Ok((slot, self.read_state_bytes(slot)?)))
            .collect()
    }

    /// The bytes [`Self::read_fixed_state`] will produce, without reading them: the host charge
    /// of one snapshot's fixed part.
    pub(super) fn fixed_state_bytes(&self) -> u64 {
        self.state_plan
            .slots()
            .iter()
            .filter(|&&(slot, _)| !StatePlan::is_position_state(slot))
            .map(|&(_, bytes)| bytes as u64)
            .sum()
    }

    /// Installs a snapshot's copied slots, ordered before the next launch.
    pub(super) fn write_fixed_state(
        &mut self,
        parts: &[(StateSlot, Vec<u8>)],
    ) -> Result<(), EngError> {
        for (slot, bytes) in parts {
            self.write_state_bytes(*slot, bytes)?;
        }
        Ok(())
    }

    // ------------------------------------------------------------------ weights

    pub(super) fn exl3(&self, prefix: &str) -> Result<&Exl3Weight, EngError> {
        match self.weights.get(prefix) {
            Some(CudaWeight::Exl3 { weight, .. }) => Ok(weight),
            _ => Err(refuse(format!("{prefix} is not an uploaded EXL3 matrix"))),
        }
    }

    /// An F32 tensor uploaded as such (a vector widened at load).
    pub(super) fn float(&self, name: &str) -> Result<DeviceRange<'_>, EngError> {
        match self.weights.get(name) {
            Some(CudaWeight::Raw {
                ty: GgufType::F32,
                buffer,
                ..
            }) => buffer.range(0, buffer.bytes()),
            _ => Err(refuse(format!("{name} is not an uploaded F32 tensor"))),
        }
    }

    /// A tensor kept as stored: F16 or BF16 with its dtype, or the Q8_0 blocks of a hyper matrix
    /// with `Q80AndF16`, which only the hyper-connection kernels read.
    pub(super) fn half(&self, name: &str) -> Result<(DeviceRange<'_>, HyperDtype), EngError> {
        half_of(&self.weights, name)
    }

    /// [`Self::half`] for a step that borrows its weights, so the range outlives the step.
    pub(super) fn half_borrowed(
        &self,
        name: &str,
    ) -> Result<(DeviceRange<'a>, HyperDtype), EngError> {
        let weights = self
            .weights
            .borrowed()
            .ok_or_else(|| refuse("the step shares its weights, which it cannot lend".into()))?;
        half_of(weights, name)
    }

    pub(super) fn shared_prefill(&self) -> bool {
        self.prefill_window.get() && self.step_plan.index(Slot::PrefillPerm).is_some()
    }

    pub(super) fn prefill_projector<'b>(
        &'b self,
        launch: &'b Launch<'b>,
        m: usize,
    ) -> Result<Projector<'b>, EngError> {
        self.check_rows(m)?;
        Ok(Projector {
            launch,
            rows: m,
            tile_rows: PREFILL_TILE_ROWS,
            scratch: Exl3Scratch {
                xh: self.slot(Slot::Exl3Xh)?,
                z: self.slot(Slot::Exl3Z)?,
                flag: self.slot(Slot::Exl3Flag)?,
                jobs: self.slot(Slot::Exl3Jobs)?,
            },
        })
    }

    // ------------------------------------------------------------------ launches

    /// One EXL3 projection of the row `x` into `out`. The scratch is the plan's: rotated inputs,
    /// the split-K partial and the sticky arithmetic flag, which the step reads once per token.
    pub(super) fn project(
        &self,
        prefix: &str,
        x: DeviceRange<'_>,
        out: DeviceRange<'_>,
        m: usize,
    ) -> Result<(), EngError> {
        self.check_rows(m)?;
        match self.weights.get(prefix) {
            Some(CudaWeight::Native { weight, .. }) => {
                if !self.gguf_decode_step() {
                    return self.project_native(weight, x, out, m);
                }
                if m != 1 || self.shared_prefill() {
                    return Err(refuse(format!(
                        "{prefix}: native projection supports decode only"
                    )));
                }
                let call = NativeProjection {
                    weight,
                    input: x,
                    output: out,
                    scratch: self.slot(Slot::GgufDecodeQ8)?,
                };
                #[cfg(test)]
                if (prefix.ends_with("linear_attn.in_proj_qkv")
                    || prefix.ends_with("linear_attn.in_proj_z"))
                    && std::env::var_os("FIFI_FLASH_GGUF_TEST_F32_DELTA_IN").is_some()
                {
                    use crate::kernels::cuda::resident::matvec_q::QuantType;
                    match weight.quant() {
                        QuantType::Q6K => {
                            eprintln!("GGUF test-only Delta input: {prefix} Q6_K F32");
                            return project_q6_f32(&self.ctx, &self.stream, call);
                        }
                        QuantType::Q80 => {
                            eprintln!(
                                "GGUF test-only Delta input: {prefix} Q8_0 native Q8 fallback"
                            );
                        }
                        _ => {
                            return Err(refuse(format!(
                                "{prefix}: unsupported Delta input probe type"
                            )));
                        }
                    }
                }
                #[cfg(test)]
                if prefix.ends_with("linear_attn.out_proj") {
                    let f16 = std::env::var_os("FIFI_FLASH_GGUF_TEST_F16_DELTA_OUT").is_some();
                    let f32 = std::env::var_os("FIFI_FLASH_GGUF_TEST_F32_DELTA_OUT").is_some();
                    if f16 && f32 {
                        return Err(refuse(
                            "select only one DeltaNet precision experiment".into(),
                        ));
                    }
                    if f16 {
                        return project_q6_f16(&self.ctx, &self.stream, call);
                    }
                    if f32 {
                        return project_q6_f32(&self.ctx, &self.stream, call);
                    }
                }
                #[cfg(test)]
                if prefix.ends_with("self_attn.o_proj")
                    && std::env::var_os("FIFI_FLASH_GGUF_TEST_F32_QSA_OUT").is_some()
                {
                    eprintln!("GGUF test-only QSA output: {prefix} Q6_K F32");
                    return project_q6_f32(&self.ctx, &self.stream, call);
                }
                #[cfg(test)]
                if prefix.ends_with("self_attn.q_proj")
                    && std::env::var_os("FIFI_FLASH_GGUF_TEST_F32_QSA_IN").is_some()
                {
                    return project_q6_f32(&self.ctx, &self.stream, call);
                }
                #[cfg(all(test, feature = "native-cuda", target_os = "linux"))]
                if crate::family::qwen::flash_next::load::gguf::native_f16::enabled() {
                    return crate::family::qwen::flash_next::load::gguf::native_f16::project(
                        &self.ctx,
                        &self.stream,
                        weight,
                        x,
                        out,
                    );
                }
                return match self.gguf_decode {
                    GgufDecodePrecision::Q8 => self.ctx.project_1row_prefix(
                        &self.stream,
                        prefix,
                        weight,
                        x,
                        call.scratch,
                        out,
                    ),
                    GgufDecodePrecision::Exact => {
                        project_exact(&self.ctx, &self.stream, weight, x, out)
                    }
                };
            }
            None => return self.project_dense_pair(prefix, x, out, m),
            Some(_) => {}
        }
        if prefix.ends_with("self_attn.indexer.index_qk_proj")
            && self.step_plan.index(Slot::GgufDecodeQ8).is_some()
        {
            return self.project_gguf_indexer(prefix, x, out, m);
        }
        let weight = self.exl3(prefix)?;
        if self.shared_prefill() {
            let launch = Launch {
                ctx: &self.ctx,
                stream: &self.stream,
            };
            return self
                .prefill_projector(&launch, m)?
                .project(weight, x, Output::Rows(out));
        }
        if m == 1 {
            let view = weight.view()?;
            let io = Exl3Io {
                x,
                xh: self.slot(Slot::Exl3Xh)?,
                z: self.slot(Slot::Exl3Z)?,
                out,
                flag: self.slot(Slot::Exl3Flag)?,
                rows: m,
                fused: false,
            };
            return self.ctx.exl3_linear(&self.stream, view.linear_call(&io));
        }
        // The job-table path: the out of many rows is strided (`out_ld` per row), which the
        // contiguous single-row call cannot express.
        let [_, n] = weight.shape;
        self.exl3_jobs(prefix, x, out, n, 0, m)
    }

    /// One native matvec: the activation is quantized to q8_1 into the step's pack, then the
    /// matvec of the weight's type reads the stored blocks. One row only.
    fn project_native(
        &self,
        weight: &crate::kernels::cuda::NativeWeight,
        x: DeviceRange<'_>,
        out: DeviceRange<'_>,
        m: usize,
    ) -> Result<(), EngError> {
        let (k, rows) = (weight.k(), weight.n_rows());
        let w = weight.buffer().range(0, weight.buffer().bytes())?;
        // The matvec takes up to eight activation rows a launch; a longer piece runs in groups.
        for first in (0..m).step_by(MATVEC_COLUMNS) {
            let n = MATVEC_COLUMNS.min(m - first);
            let pack =
                q8_1_bytes(k, n).ok_or_else(|| refuse("the activation pack overflows".into()))?;
            let x_q8 = self.xq.range(0, pack)?;
            let x_rows = x
                .buffer()
                .range(x.offset() + first * k * F32, n * k * F32)?;
            self.ctx.quantize_q8_1(
                &self.stream,
                QuantizeCall {
                    x: x_rows,
                    y: x_q8,
                    k,
                    ncols: n,
                },
            )?;
            self.ctx.matvec_q(
                &self.stream,
                MatvecCall {
                    ty: weight.quant(),
                    w,
                    x_q8,
                    y: out
                        .buffer()
                        .range(out.offset() + first * rows * F32, n * rows * F32)?,
                    k,
                    rows,
                    ncols: n,
                },
            )?;
        }
        Ok(())
    }

    /// A projection stored as two half-precision tensors under `prefix.q` and `prefix.k` (the
    /// GGUF indexer: queries then the key): two dense matvecs into the first and the last
    /// columns of `out`.
    fn project_dense_pair(
        &self,
        prefix: &str,
        x: DeviceRange<'_>,
        out: DeviceRange<'_>,
        m: usize,
    ) -> Result<(), EngError> {
        let cols = self.geometry.hidden;
        let mut parts = Vec::with_capacity(2);
        for suffix in ["q", "k"] {
            let name = format!("{prefix}.{suffix}");
            let Some(CudaWeight::Raw { ty, buffer, .. }) = self.weights.get(&name) else {
                return Err(refuse(format!(
                    "{prefix} is not an uploaded matrix or dense pair"
                )));
            };
            let dtype = match ty {
                GgufType::F16 => DenseDtype::F16,
                GgufType::Bf16 => DenseDtype::Bf16,
                other => return Err(refuse(format!("{name} is {other:?}, not a half tensor"))),
            };
            parts.push((buffer, dtype, buffer.bytes() / 2 / cols));
        }
        // One row of the output holds the queries then the key: both matvecs write their
        // columns into rows `total` floats wide.
        let total: usize = parts.iter().map(|part| part.2).sum();
        if out.bytes() < m * total * F32 {
            return Err(refuse(format!(
                "{prefix} has {total} output columns for {m} rows, the slot holds {} floats",
                out.bytes() / F32
            )));
        }
        let mut at = 0usize;
        for (buffer, dtype, rows) in parts {
            let shape = DenseGemvShape {
                ld_y: total,
                ..DenseGemvShape::packed(m, rows, cols)
            };
            self.ctx.dense_gemv(
                &self.stream,
                DenseGemvCall {
                    shape,
                    dtype,
                    w: buffer.range(0, buffer.bytes())?,
                    x,
                    y: out
                        .buffer()
                        .range(out.offset() + at * F32, (m - 1) * total * F32 + rows * F32)?,
                },
            )?;
            at += rows;
        }
        Ok(())
    }

    /// GGUF stores the QSA indexer's query and key as one uploaded half matrix.
    fn project_gguf_indexer(
        &self,
        prefix: &str,
        x: DeviceRange<'_>,
        out: DeviceRange<'_>,
        m: usize,
    ) -> Result<(), EngError> {
        use crate::kernels::cuda::resident::dense_gemv::{
            DenseDtype, DenseGemvCall, DenseGemvShape,
        };
        if m != 1 || self.shared_prefill() {
            return Err(refuse(format!(
                "{prefix}: GGUF indexer supports decode only"
            )));
        }
        let Some(CudaWeight::Raw { ty, buffer, .. }) = self.weights.get(prefix) else {
            return Err(refuse(format!("{prefix}: missing GGUF indexer matrix")));
        };
        let dtype = match ty {
            GgufType::F16 => DenseDtype::F16,
            GgufType::Bf16 => DenseDtype::Bf16,
            other => {
                return Err(refuse(format!(
                    "{prefix}: unsupported indexer type {other:?}"
                )));
            }
        };
        let rows = self.geometry.qsa.idx_proj();
        let cols = self.geometry.hidden;
        let bytes = rows
            .checked_mul(cols)
            .and_then(|v| v.checked_mul(2))
            .ok_or_else(|| refuse(format!("{prefix}: indexer shape overflows")))?;
        if buffer.bytes() != bytes {
            return Err(refuse(format!("{prefix}: indexer matrix has wrong size")));
        }
        self.ctx.dense_gemv(
            &self.stream,
            DenseGemvCall {
                shape: DenseGemvShape::packed(1, rows, cols),
                dtype,
                w: buffer.range(0, bytes)?,
                x,
                y: out,
            },
        )
    }

    /// The projection of `m` rows into `n` floats at column `col_off` of a `ld`-wide row
    /// (`DeltaProj`/`AttnProj`): one row writes its columns as today, many rows run the job
    /// table.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn project_cols(
        &self,
        prefix: &str,
        x: DeviceRange<'_>,
        row: DeviceRange<'_>,
        col_off: usize,
        n: usize,
        ld: usize,
        m: usize,
    ) -> Result<(), EngError> {
        if self.shared_prefill() {
            let launch = Launch {
                ctx: &self.ctx,
                stream: &self.stream,
            };
            return self.prefill_projector(&launch, m)?.project(
                self.exl3(prefix)?,
                x,
                Output::Columns {
                    dst: row,
                    ld,
                    col_off,
                },
            );
        }
        if m == 1 {
            let out = Self::columns(&row, col_off, n)?;
            return self.project(prefix, x, out, 1);
        }
        self.check_rows(m)?;
        self.exl3_jobs(prefix, x, row, ld, col_off, m)
    }

    /// The job-table projection behind [`Self::project`] and [`Self::project_cols`].
    fn exl3_jobs(
        &self,
        prefix: &str,
        x: DeviceRange<'_>,
        out: DeviceRange<'_>,
        out_ld: usize,
        out_col: usize,
        m: usize,
    ) -> Result<(), EngError> {
        let weight = self.exl3(prefix)?;
        let members = weight.members()?;
        let jobs = self.slot(Slot::Exl3Jobs)?;
        let io = Exl3JobsIo {
            x,
            xh: self.slot(Slot::Exl3Xh)?,
            z: self.slot(Slot::Exl3Z)?,
            out,
            flag: self.slot(Slot::Exl3Flag)?,
            jobs,
            jobs_count: 1,
            max_rows: m,
            total_rows: m,
            out_ld,
            out_col,
        };
        self.ctx.exl3_jobs(&self.stream, members.jobs_call(&io))
    }

    /// The input projections `parts` as one grouped launch: part `(prefix, col, n)` writes the
    /// `n` floats of column `col` of `row`, exactly as one [`Self::project`] call per part would;
    /// only the kernel grouping changes. The scratch slices follow group order, so the plan sizes
    /// them for the member sums ([`exl3_scratch`]); the flag is shared, as in every group this
    /// engine launches.
    pub(super) fn project_group(
        &self,
        parts: &[(&str, usize, usize)],
        x: DeviceRange<'_>,
        row: &DeviceRange<'_>,
        m: usize,
    ) -> Result<(), EngError> {
        let native_count = parts
            .iter()
            .filter(|(prefix, _, _)| {
                matches!(self.weights.get(prefix), Some(CudaWeight::Native { .. }))
            })
            .count();
        if self.gguf_decode_step() && native_count != 0 {
            if native_count != parts.len() {
                return Err(refuse("mixed native and EXL3 projection group".into()));
            }
            self.check_rows(m)?;
            if m != 1 || self.shared_prefill() {
                return Err(refuse(
                    "native projection group supports decode only".into(),
                ));
            }
            let scratch = self.slot(Slot::GgufDecodeQ8)?;
            // Every member reads x again after the previous output is written.
            if std::ptr::eq(row.buffer(), x.buffer())
                && row.offset() < x.offset() + x.bytes()
                && x.offset() < row.offset() + row.bytes()
            {
                return Err(refuse("native group row overlaps its input".into()));
            }
            let mut calls = Vec::with_capacity(parts.len());
            for (index, (prefix, col, n)) in parts.iter().enumerate() {
                let Some(CudaWeight::Native { weight, .. }) = self.weights.get(prefix) else {
                    unreachable!();
                };
                if weight.experts().is_some() || weight.n_rows() != *n {
                    return Err(refuse(format!(
                        "{prefix}: invalid native group matrix width or experts"
                    )));
                }
                let end = col
                    .checked_add(*n)
                    .and_then(|v| v.checked_mul(F32))
                    .ok_or_else(|| refuse("native group output range overflows".into()))?;
                if end > row.bytes() || std::ptr::eq(row.buffer(), weight.buffer()) {
                    return Err(refuse(
                        "native group output lies outside its row or aliases a weight".into(),
                    ));
                }
                if parts[..index].iter().any(|(_, other_col, other_n)| {
                    col.checked_add(*n).is_none_or(|end| {
                        other_col
                            .checked_add(*other_n)
                            .is_none_or(|other_end| *col < other_end && *other_col < end)
                    })
                }) {
                    return Err(refuse("native group outputs overlap".into()));
                }
                let output = Self::columns(row, *col, *n)?;
                self.ctx
                    .validate_native_projection(weight, x, scratch, output)?;
                calls.push(NativeProjection {
                    weight,
                    input: x,
                    output,
                    scratch,
                });
            }
            #[cfg(test)]
            if parts.len() == 2
                && parts[0].0.ends_with("self_attn.k_proj")
                && parts[1].0.ends_with("self_attn.v_proj")
                && std::env::var_os("FIFI_FLASH_GGUF_TEST_F32_QSA_IN").is_some()
            {
                use crate::kernels::cuda::resident::matvec_q::QuantType;
                if calls
                    .iter()
                    .any(|call| call.weight.quant() != QuantType::Q6K)
                {
                    return Err(refuse("QSA input precision probe requires Q6_K K/V".into()));
                }
                for call in calls {
                    project_q6_f32(&self.ctx, &self.stream, call)?;
                }
                return Ok(());
            }
            #[cfg(all(test, feature = "native-cuda", target_os = "linux"))]
            if crate::family::qwen::flash_next::load::gguf::native_f16::enabled() {
                for call in calls {
                    crate::family::qwen::flash_next::load::gguf::native_f16::project(
                        &self.ctx,
                        &self.stream,
                        call.weight,
                        call.input,
                        call.output,
                    )?;
                }
                return Ok(());
            }
            if self.gguf_decode == GgufDecodePrecision::Exact {
                for call in &calls {
                    project_exact(
                        &self.ctx,
                        &self.stream,
                        call.weight,
                        call.input,
                        call.output,
                    )?;
                }
                return Ok(());
            }
            let k = calls[0].weight.k();
            if calls.iter().all(|call| call.weight.k() == k) {
                // The whole group reads the same row; one quantization serves every matvec.
                self.ctx.quantize_q8_1(
                    &self.stream,
                    QuantizeCall {
                        x,
                        y: scratch,
                        k,
                        ncols: 1,
                    },
                )?;
                for call in calls {
                    self.ctx
                        .matvec_native(&self.stream, call.weight, scratch, call.output, 1)?;
                }
            } else {
                for call in calls {
                    project_native(&self.ctx, &self.stream, call)?;
                }
            }
            return Ok(());
        }
        if let Some((first, _, _)) = parts.first()
            && matches!(self.weights.get(first), Some(CudaWeight::Native { .. }))
        {
            // Native weights have no grouped launch: one projection each.
            for &(prefix, col, n) in parts {
                self.project(prefix, x, Self::columns(row, col, n)?, m)?;
            }
            return Ok(());
        }
        let (xh, z, flag) = (
            self.slot(Slot::Exl3Xh)?,
            self.slot(Slot::Exl3Z)?,
            self.slot(Slot::Exl3Flag)?,
        );
        let mut xh_at = xh.offset();
        let mut z_at = z.offset();
        // The grouped launch takes one width per group (exl3.cu group_args: every call's k and n
        // equal the first's), so the guards below refuse a mixer change that mixes widths.
        let group_n = parts.first().map(|&(_, _, n)| n);
        // The views and their io outlive the calls that borrow them: collect the members first,
        // then build the calls in a second pass.
        let mut members = Vec::with_capacity(parts.len());
        let mut group_k: Option<usize> = None;
        for (prefix, col, n) in parts {
            if Some(*n) != group_n {
                return Err(refuse(format!(
                    "{prefix} is {n} wide, the group takes one width ({:?})",
                    group_n
                )));
            }
            let view = self.exl3(prefix)?.view()?;
            let [k, wide] = view.shape;
            if group_k.is_some_and(|gk| gk != k) {
                return Err(refuse(format!(
                    "{prefix} reads a {k} wide input, the group takes one input width ({:?})",
                    group_k
                )));
            }
            group_k = Some(k);
            if wide != *n {
                return Err(refuse(format!(
                    "{prefix} is {wide} wide, the mixer row expects {n}"
                )));
            }
            members.push((
                view,
                Exl3Io {
                    x,
                    xh: xh.buffer().range(xh_at, k * F32)?,
                    z: z.buffer().range(z_at, n * F32)?,
                    out: Self::columns(row, *col, *n)?,
                    flag,
                    rows: m,
                    fused: false,
                },
            ));
            xh_at += k * F32;
            z_at += n * F32;
        }
        let calls: Vec<_> = members
            .iter()
            .map(|(view, io)| view.linear_call(io))
            .collect();
        self.ctx.exl3_grouped_step(&self.stream, &calls)
    }

    /// `width` floats of `row` from float column `col`.
    pub(super) fn columns<'r>(
        row: &DeviceRange<'r>,
        col: usize,
        width: usize,
    ) -> Result<DeviceRange<'r>, EngError> {
        row.buffer().range(row.offset() + col * F32, width * F32)
    }

    /// Whether a hyper site's weights pair other than by one dtype: its vectors (`norm`, `inject`)
    /// F16 and its two matrices Q8_0, which the kernels of dtype `Q80AndF16` read as stored.
    fn stored_q80(vectors: HyperDtype, matrices: [HyperDtype; 2], inject: HyperDtype) -> bool {
        let [down, up] = matrices;
        inject == vectors
            && down == up
            && down == HyperDtype::Q80AndF16
            && vectors == HyperDtype::F16
    }

    fn hyper_shape(&self, m: usize) -> HyperShape {
        HyperShape {
            m,
            branches: self.geometry.branches,
            hidden: self.geometry.hidden,
            rank: self.geometry.rank,
        }
    }

    /// The read of the hyper-connection site under `prefix`: the mixed row of `site` and, with an
    /// inject projection, its gate.
    pub(super) fn hyper_read(&self, site: Site, prefix: &str, m: usize) -> Result<(), EngError> {
        let inject = site != Site::Final;
        let (norm, dtype) = self.half(&format!("{prefix}.hc_norm.weight"))?;
        let (down, down_dtype) = self.half(&format!("{prefix}.input_mix_weight_down.weight"))?;
        let (up, up_dtype) = self.half(&format!("{prefix}.input_mix_weight_up.weight"))?;
        let (inject_weight, inject_dtype) = if inject {
            let (range, found) = self.half(&format!("{prefix}.block_inject_weight.weight"))?;
            (Some(range), found)
        } else {
            (None, dtype)
        };
        if [down_dtype, up_dtype, inject_dtype] != [dtype; 3]
            && !Self::stored_q80(dtype, [down_dtype, up_dtype], inject_dtype)
        {
            return Err(refuse(format!(
                "the weights of {prefix} are not of one dtype: norm {dtype:?}, down {down_dtype:?}, \
                 up {up_dtype:?}, inject {inject_dtype:?}"
            )));
        }
        let dtype = down_dtype;
        self.ctx.hyper_read(
            &self.stream,
            HyperReadCall {
                shape: self.hyper_shape(m),
                eps: self.model.hyper.eps(),
                x: self.slot(Slot::Stream)?,
                weights: HyperWeights {
                    dtype,
                    norm,
                    down,
                    up,
                    inject: inject_weight,
                },
                scratch: HyperScratch {
                    normed: self.slot(Slot::HyperNormed(site))?,
                    low: self.slot(Slot::HyperLow(site))?,
                    mix: self.slot(Slot::HyperMix(site))?,
                },
                mixed: self.slot(Slot::Mixed(site))?,
                gate: if inject {
                    Some(self.slot(Slot::Gate(site))?)
                } else {
                    None
                },
            },
        )
    }

    /// `Stream += gate * block` for the site's `m` rows.
    pub(super) fn hyper_apply(&self, site: Site, m: usize) -> Result<(), EngError> {
        self.ctx.hyper_apply(
            &self.stream,
            HyperApplyCall {
                shape: self.hyper_shape(m),
                x: self.slot(Slot::Stream)?,
                gate: self.slot(Slot::Gate(site))?,
                block: self.slot(Slot::BlockOut(site))?,
            },
        )
    }

    // ------------------------------------------------------------------ the MoE seam

    pub(in crate::family::qwen::flash_next) fn moe_dims(&self, m: usize) -> MoeDims {
        MoeDims {
            rows: m,
            hidden: self.geometry.hidden,
            n_expert: self.geometry.experts,
            top_k: self.geometry.top_k,
            inter: self.geometry.inter,
            shared_inter: self.geometry.shared_inter,
        }
    }

    /// The pack keys of the MoE tensors of `layer` in the uploaded set, by role. The routed banks
    /// are the three `Exl3Bank` entries named by [`bank_key`].
    pub(super) fn moe_roles(layer: usize) -> Vec<(Role, String)> {
        let at = |suffix: &str| layer_name(layer, suffix);
        vec![
            (Role::FfnGateInp, at("mlp.gate.weight")),
            (Role::FfnGateInpShexp, at("mlp.shared_expert_gate.weight")),
            (Role::FfnGateExps, bank_key(layer, "gate_proj")),
            (Role::FfnUpExps, bank_key(layer, "up_proj")),
            (Role::FfnDownExps, bank_key(layer, "down_proj")),
            (Role::FfnGateShexp, at("mlp.shared_expert.gate_proj")),
            (Role::FfnUpShexp, at("mlp.shared_expert.up_proj")),
            (Role::FfnDownShexp, at("mlp.shared_expert.down_proj")),
        ]
    }

    /// The buffers of the MoE block, views of the plan's slots: the block reads `Mixed(Mlp)` and
    /// writes `BlockOut(Mlp)` (no residual: the hyper apply adds it to the stream).
    pub(super) fn moe_bufs(&self) -> Result<Exl3MoeBufs<'_>, EngError> {
        self.moe_bufs_at(0)
    }

    /// The buffers of row `r` of the plan's multi-row slots: the split block runs one row at a
    /// time (M1D lifts it), so a chunk loops it per row.
    fn moe_bufs_at(&self, r: usize) -> Result<Exl3MoeBufs<'_>, EngError> {
        let g = &self.geometry;
        let at = |slot: Slot, width: usize| -> Result<DeviceRange<'_>, EngError> {
            let range = self.slot(slot)?;
            range
                .buffer()
                .range(range.offset() + r * width * F32, width * F32)
        };
        let shared_gu = at(Slot::SharedGu, 2 * g.shared_inter)?;
        let shared_act = at(Slot::SharedAct, g.shared_inter)?;
        let expert_gu = at(Slot::ExpertGu, g.top_k * 2 * g.inter)?;
        let expert_act = at(Slot::ExpertAct, g.top_k * g.inter)?;
        let expert_down = at(Slot::ExpertDown, g.top_k * g.hidden)?;
        let ids = self.slot(Slot::TopkId)?;
        let ids = ids
            .buffer()
            .range(ids.offset() + r * g.top_k * F32, g.top_k * F32)?;
        Ok(Exl3MoeBufs {
            route: RouteBufs {
                resid_in: self.zeros.range(r * g.hidden * F32, g.hidden * F32)?,
                resid_out: at(Slot::BlockOut(Site::Mlp), g.hidden)?,
                norm_row: at(Slot::Mixed(Site::Mlp), g.hidden)?,
                logits: at(Slot::RouterLogits, g.experts + 1)?,
                ids,
                weights: at(Slot::TopkWeight, g.top_k)?,
                shared_gate: at(Slot::SharedGate, 1)?,
                status: self.status_word()?,
            },
            expert_gu,
            expert_act,
            expert_down,
            shared_gu,
            shared_act,
            shared_out: at(Slot::SharedOut, g.hidden)?,
            scratch: Scratch {
                xh: self.slot(Slot::Exl3Xh)?,
                z: self.slot(Slot::Exl3Z)?,
                flag: self.slot(Slot::Exl3Flag)?,
            },
        })
    }

    /// The buffers of the MoE block over all `m` rows at once: every slot spans `m` rows and
    /// the residual input is the plan's zero rows. The batched block
    /// of the prefill pass; `m` must match the plan's row count.
    pub(super) fn moe_bufs_rows(&self, m: usize) -> Result<Exl3MoeBufs<'_>, EngError> {
        let g = &self.geometry;
        let at = |slot: Slot, width: usize| -> Result<DeviceRange<'_>, EngError> {
            let range = self.slot(slot)?;
            range.buffer().range(range.offset(), width * m * F32)
        };
        let ids = self.slot(Slot::TopkId)?;
        let ids = ids.buffer().range(ids.offset(), g.top_k * m * F32)?;
        Ok(Exl3MoeBufs {
            route: RouteBufs {
                resid_in: self.zeros.range(0, self.zeros.bytes())?,
                resid_out: at(Slot::BlockOut(Site::Mlp), g.hidden)?,
                norm_row: at(Slot::Mixed(Site::Mlp), g.hidden)?,
                logits: at(Slot::RouterLogits, g.experts + 1)?,
                ids,
                weights: at(Slot::TopkWeight, g.top_k)?,
                shared_gate: at(Slot::SharedGate, 1)?,
                status: self.status_word()?,
            },
            expert_gu: at(Slot::ExpertGu, g.top_k * 2 * g.inter)?,
            expert_act: at(Slot::ExpertAct, g.top_k * g.inter)?,
            expert_down: at(Slot::ExpertDown, g.top_k * g.hidden)?,
            shared_gu: at(Slot::SharedGu, 2 * g.shared_inter)?,
            shared_act: at(Slot::SharedAct, g.shared_inter)?,
            shared_out: at(Slot::SharedOut, g.hidden)?,
            scratch: Scratch {
                xh: self.slot(Slot::Exl3Xh)?,
                z: self.slot(Slot::Exl3Z)?,
                flag: self.slot(Slot::Exl3Flag)?,
            },
        })
    }

    /// The MoE block of `layer` at the phases `Route` to `Combine`, run by `moe`: the output lands
    /// in `BlockOut(Mlp)`.
    pub(super) fn moe_block(
        &self,
        layer: usize,
        pos: usize,
        moe: &mut dyn MoeBlock,
        m: usize,
    ) -> Result<(), EngError> {
        let launch = Launch {
            ctx: &self.ctx,
            stream: &self.stream,
        };
        if self.shared_prefill() {
            return moe.run_prefill(
                &self.prefill_projector(&launch, m)?,
                layer,
                &self.moe_bufs_rows(m)?,
                &self.moe_dims(m),
                self.slot(Slot::PrefillPerm)?,
            );
        }
        // Outside a prefill window every block runs one row at a time. A multi-row
        // step here cannot occur (the driver opens the window for every chunked
        // prefill), so refuse it instead of looping.
        if m > 1 {
            return Err(EngError::BadRequest(format!(
                "a multi-row MoE step ({m} rows) outside a prefill window"
            )));
        }
        let dims = self.moe_dims(1);
        for r in 0..m {
            moe.run(&launch, layer, pos, &self.moe_bufs_at(r)?, &dims)?;
        }
        Ok(())
    }

    // ------------------------------------------------------------------ layers and the tail

    /// One layer at position `pos`, in the site order of the host reference. The PLE update, if
    /// `layer` is the PLE layer, is the caller's to apply before this call.
    pub(super) fn run_layer(
        &self,
        layer: usize,
        pos: usize,
        m: usize,
        moe: &mut dyn MoeBlock,
    ) -> Result<(), EngError> {
        self.ensure_usable()?;
        let result = (|| {
            let profile = super::driver::chunk::prof::on();
            let mut t = std::time::Instant::now();
            let mut tick = |slot: usize| {
                if profile {
                    super::driver::chunk::prof::add(slot, t.elapsed());
                    t = std::time::Instant::now();
                }
            };
            self.hyper_read(Site::Attn, &layer_name(layer, "attn_hyper_connection"), m)?;
            self.mixer(layer, pos, m)?;
            tick(9);
            self.hyper_apply(Site::Attn, m)?;
            self.hyper_read(Site::Mlp, &layer_name(layer, "mlp_hyper_connection"), m)?;
            tick(8);
            self.moe_block(layer, pos, moe, m)?;
            tick(10);
            let r = self.hyper_apply(Site::Mlp, m);
            tick(8);
            r
        })();
        self.poison_on(result)
    }

    /// The final mixer and `lm_head`: the logits land in the `Logits` slot.
    pub(super) fn head(&self, m: usize) -> Result<(), EngError> {
        self.ensure_usable()?;
        let result = (|| {
            self.hyper_read(Site::Final, MIXER, m)?;
            self.project(
                "lm_head",
                self.slot(Slot::Mixed(Site::Final))?,
                self.slot(Slot::Logits)?,
                m,
            )
        })();
        self.poison_on(result)
    }

    /// `begin_token` plus the dense job table of an `m`-row chunk: the job-table projections of
    /// `m > 1` rows read it. The table write waits for the stream, so it runs once per chunk.
    pub(super) fn begin_chunk(&self, m: usize) -> Result<(), EngError> {
        self.begin_token()?;
        self.check_rows(m)?;
        if self.shared_prefill() {
            let launch = Launch {
                ctx: &self.ctx,
                stream: &self.stream,
            };
            return write_dense_jobs(
                &launch,
                &self.prefill_projector(&launch, m)?.scratch,
                m,
                PREFILL_TILE_ROWS,
            );
        }
        // A chunk of at most MAX_STEP_ROWS rows is one dense job of member 0: the kernel's
        // job row cap (`jobs_row_cap`, 256, exl3.cu) is wider than any planned chunk.
        let job = Exl3Job {
            member: 0,
            row0: 0,
            rows: m as u32,
        };
        let table = self.slot(Slot::Exl3Jobs)?;
        table
            .buffer()
            .write_at(&self.stream, table.offset(), &Exl3Job::table(&[job]))
    }

    /// Clears the sticky EXL3 flag and the status word: the first launch of a token.
    pub(super) fn begin_token(&self) -> Result<(), EngError> {
        self.ensure_usable()?;
        let result = (|| {
            let flag = self.slot(Slot::Exl3Flag)?;
            flag.buffer()
                .write_at(&self.stream, flag.offset(), &[0u8; 4])?;
            let status = self.status_word()?;
            status
                .buffer()
                .write_at(&self.stream, status.offset(), &[0u8; 4])
        })();
        self.poison_on(result)
    }

    /// Reads the status word and the EXL3 flag behind the queued work (so it drains the stream's
    /// work up to here) and turns a fault into an error: a nonzero status word or EXL3 flag, or a
    /// failed read, poisons the step. Call it after the last launch of a token, `head` included.
    pub(super) fn finish_token(&self) -> Result<(), EngError> {
        let (status, flag) = self.poison_on(self.health())?;
        if status != 0 || flag != 0 {
            self.poisoned.set(true);
            return Err(EngError::ModelOutput(format!(
                "Flash-Next step: the token finished with status word {status} and EXL3 flag {flag}"
            )));
        }
        Ok(())
    }

    /// The layer of the PLE block.
    pub(super) fn ple_layer(&self) -> usize {
        self.model.ple_layer
    }

    /// Refuses a range that has the PLE layer after its first layer.
    fn check_ple(&self, range: &std::ops::Range<usize>) -> Result<(), EngError> {
        let ple = self.ple_layer();
        if range.start < ple && ple < range.end {
            return Err(refuse(format!(
                "layers {}..{} run the PLE layer {ple} after layer {}; apply the PLE update on the \
                 host and call {}..{ple} and {ple}..{}",
                range.start, range.end, range.start, range.start, range.end
            )));
        }
        Ok(())
    }

    /// Queues `range` of the window's layers for the token at `pos`. Refuses a range outside the
    /// window, and a range that crosses the PLE layer (see the module doc): the real path is two
    /// calls, `0..ple_layer` and `ple_layer..layers`, with the host update between them.
    pub(super) fn run_layers(
        &self,
        range: std::ops::Range<usize>,
        pos: usize,
        m: usize,
        moe: &mut dyn MoeBlock,
    ) -> Result<(), EngError> {
        if range.start >= range.end
            || range.start < self.layers.start
            || range.end > self.layers.end
        {
            return Err(refuse(format!(
                "layers {}..{} are not inside the step's window {}..{}",
                range.start, range.end, self.layers.start, self.layers.end
            )));
        }
        self.check_ple(&range)?;
        self.run_range(range, pos, m, moe)
    }

    /// `run_layers` over the whole window, for the tests, which run a window across the PLE layer
    /// without the update on both sides, host reference and device.
    #[cfg(test)]
    pub(super) fn run_layers_without_ple(
        &self,
        pos: usize,
        m: usize,
        moe: &mut dyn MoeBlock,
    ) -> Result<(), EngError> {
        self.run_range(self.layers.clone(), pos, m, moe)
    }

    fn run_range(
        &self,
        range: std::ops::Range<usize>,
        pos: usize,
        m: usize,
        moe: &mut dyn MoeBlock,
    ) -> Result<(), EngError> {
        self.check_rows(m)?;
        for layer in range {
            self.run_layer(layer, pos, m, moe)?;
        }
        Ok(())
    }

    /// Refuses a row count the plan does not hold.
    fn check_rows(&self, m: usize) -> Result<(), EngError> {
        if m == 0 || m > self.rows {
            return Err(refuse(format!(
                "{m} rows, the step's plan holds 1 to {}",
                self.rows
            )));
        }
        Ok(())
    }

    /// `values` floats of `range` read back behind the queued work.
    pub(super) fn read_f32(
        &self,
        range: DeviceRange<'_>,
        values: usize,
    ) -> Result<Vec<f32>, EngError> {
        let mut bytes = vec![0u8; values * F32];
        range
            .buffer()
            .read_at(&self.stream, range.offset(), &mut bytes)?;
        Ok(bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|word| f32::from_le_bytes(*word))
            .collect())
    }

    /// The status word and the EXL3 arithmetic flag, read back: `(status, flag)`.
    pub(super) fn health(&self) -> Result<(i32, u32), EngError> {
        let mut word = [0u8; 4];
        let status = self.status_word()?;
        status
            .buffer()
            .read_at(&self.stream, status.offset(), &mut word)?;
        let status = i32::from_le_bytes(word);
        let flag = self.slot(Slot::Exl3Flag)?;
        flag.buffer()
            .read_at(&self.stream, flag.offset(), &mut word)?;
        Ok((status, u32::from_le_bytes(word)))
    }
}

/// The key of the bank of every routed expert's `part` of `layer` in the uploaded set.
pub(super) fn bank_key(layer: usize, part: &str) -> String {
    format!("layer {layer} routed {part} bank")
}

/// The MoE block of a layer, as the step calls it between the MLP hyper read and its apply.
///
/// `run` queues the block for one row on `launch.stream`: it reads `bufs.route.norm_row` (the
/// hyper-mixed row, f32 `[hidden]`), writes the weighted picks plus the gated shared expert (no
/// residual) to `bufs.route.resid_out`, and returns once that output is complete and ordered on the
/// stream. `bufs.route.resid_in` is a zero row for the combine. It may wait on the stream and copy
/// (a partial expert bank does on a miss); the step does not rely on it being non-blocking.
pub(super) trait MoeBlock {
    /// `pos` is the token's position: one decode step is every layer's
    /// run at the same position. The disk-rung miss counter keys on it.
    fn run(
        &mut self,
        launch: &Launch<'_>,
        layer: usize,
        pos: usize,
        bufs: &Exl3MoeBufs<'_>,
        dims: &MoeDims,
    ) -> Result<(), EngError>;

    fn run_prefill(
        &mut self,
        _projector: &Projector<'_>,
        _layer: usize,
        _bufs: &Exl3MoeBufs<'_>,
        _dims: &MoeDims,
        _perm: DeviceRange<'_>,
    ) -> Result<(), EngError> {
        Err(refuse("this MoE block has no shared prefill path".into()))
    }
}

/// The stand-in until the split block lands: Qwen3.6's EXL3 MoE block over routed experts that
/// are all resident, as three `Exl3Bank`s per layer in the uploaded set (keys from [`bank_key`]).
pub(super) struct AllResidentMoe<'w> {
    pub(super) weights: &'w CudaWeights,
}

impl MoeBlock for AllResidentMoe<'_> {
    fn run(
        &mut self,
        launch: &Launch<'_>,
        layer: usize,
        _pos: usize,
        bufs: &Exl3MoeBufs<'_>,
        dims: &MoeDims,
    ) -> Result<(), EngError> {
        let roles = FlashStep::moe_roles(layer);
        let weights = Exl3MoeWeights::resolve_by(self.weights, layer, dims, |role| {
            roles
                .iter()
                .find(|(found, _)| *found == role)
                .map(|(_, name)| name.as_str())
                .ok_or_else(|| refuse(format!("layer {layer} has no key for {role:?}")))
        })?;
        // Flash-Next runs the captured block; the q36 expert bank never reaches it (the
        // bank is a q36 member and the eager read path is the resolver's).
        run_exl3_moe(launch, &weights, bufs, dims, true, None, false)
    }
}

/// The trunk weights of `layer`'s MoE block that the split block needs, borrowed from the uploaded
/// set: the router, the shared-expert gate row and the three shared-expert matrices.
pub(super) fn split_trunk(
    weights: &CudaWeights,
    layer: usize,
) -> Result<SplitMoeTrunk<'_>, EngError> {
    let at = |suffix: &str| layer_name(layer, suffix);
    let raw = |name: String| match weights.get(&name) {
        Some(CudaWeight::Raw {
            ty: GgufType::F32,
            buffer,
            ..
        }) => Ok(buffer),
        _ => Err(refuse(format!("{name} is not an uploaded F32 tensor"))),
    };
    let matrix = |name: String| match weights.get(&name) {
        Some(CudaWeight::Exl3 { weight, .. }) => Ok(weight),
        _ => Err(refuse(format!("{name} is not an uploaded EXL3 matrix"))),
    };
    Ok(SplitMoeTrunk {
        router: raw(at("mlp.gate.weight"))?,
        router_shared: raw(at("mlp.shared_expert_gate.weight"))?,
        shared_gate: matrix(at("mlp.shared_expert.gate_proj"))?,
        shared_up: matrix(at("mlp.shared_expert.up_proj"))?,
        shared_down: matrix(at("mlp.shared_expert.down_proj"))?,
    })
}

/// The split block as the step's MoE block. The bank has one row per layer of a contiguous window
/// of the model that starts at `first_layer`: a bank over a window of layers is indexed from its
/// own first row, a bank over the whole model from 0.
pub(super) struct SplitMoeAt<'w> {
    pub(super) inner: SplitMoe<'w>,
    first_layer: usize,
    rows: usize,
}

impl<'w> SplitMoeAt<'w> {
    /// `layers` are the model layers of the bank's rows, in row order: consecutive layers.
    pub(super) fn new(inner: SplitMoe<'w>, layers: &[usize]) -> Result<Self, EngError> {
        let Some(&first_layer) = layers.first() else {
            return Err(refuse("a split block over no layers".to_owned()));
        };
        if layers
            .iter()
            .enumerate()
            .any(|(row, &layer)| layer != first_layer + row)
        {
            return Err(refuse(format!(
                "the split block maps rows to layers by offset, the layers {layers:?} are not consecutive"
            )));
        }
        Ok(Self {
            inner,
            first_layer,
            rows: layers.len(),
        })
    }

    /// The bank behind the split block (its counters and slot row).
    pub(super) fn bank(&self) -> &super::bank::device::DeviceBank {
        self.inner.bank()
    }

    /// Re-plans the prefill ring of the inner block against the bank as decode left it.
    pub(super) fn begin_prefill(&mut self) -> Result<(), EngError> {
        self.inner.begin_prefill()
    }

    /// Closes the prefill window; the decode steps after it use the bank's miss path again.
    pub(super) fn end_prefill(&mut self) -> Result<(), EngError> {
        self.inner.end_prefill()
    }

    /// Resets the prefill ring's per-chunk cursors.
    pub(super) fn begin_chunk(&mut self) {
        self.inner.begin_chunk();
    }
}

impl MoeBlock for SplitMoeAt<'_> {
    fn run(
        &mut self,
        launch: &Launch<'_>,
        layer: usize,
        pos: usize,
        bufs: &Exl3MoeBufs<'_>,
        dims: &MoeDims,
    ) -> Result<(), EngError> {
        let row = layer
            .checked_sub(self.first_layer)
            .filter(|&row| row < self.rows)
            .ok_or_else(|| {
                refuse(format!(
                    "layer {layer} is outside the bank's layers {}..{}",
                    self.first_layer,
                    self.first_layer + self.rows
                ))
            })?;
        self.inner.run(launch, row, pos, bufs, dims)
    }

    fn run_prefill(
        &mut self,
        projector: &Projector<'_>,
        layer: usize,
        bufs: &Exl3MoeBufs<'_>,
        dims: &MoeDims,
        perm: DeviceRange<'_>,
    ) -> Result<(), EngError> {
        let row = layer
            .checked_sub(self.first_layer)
            .filter(|&row| row < self.rows)
            .ok_or_else(|| {
                refuse(format!(
                    "layer {layer} is outside the bank's layers {}..{}",
                    self.first_layer,
                    self.first_layer + self.rows
                ))
            })?;
        self.inner.run_prefill(projector, row, bufs, dims, perm)
    }
}

/// A tensor of `weights` kept as stored: F16 or BF16 with its dtype, or the Q8_0 blocks of a
/// hyper matrix with `Q80AndF16`, which only the hyper-connection kernels read.
fn half_of<'w>(
    weights: &'w CudaWeights,
    name: &str,
) -> Result<(DeviceRange<'w>, HyperDtype), EngError> {
    match weights.get(name) {
        Some(CudaWeight::Raw { ty, buffer, .. }) => {
            let dtype = match ty {
                GgufType::F16 => HyperDtype::F16,
                GgufType::Bf16 => HyperDtype::Bf16,
                GgufType::Q80 => HyperDtype::Q80AndF16,
                other => return Err(refuse(format!("{name} is stored as {other:?}"))),
            };
            Ok((buffer.range(0, buffer.bytes())?, dtype))
        }
        _ => Err(refuse(format!("{name} is not an uploaded half tensor"))),
    }
}
