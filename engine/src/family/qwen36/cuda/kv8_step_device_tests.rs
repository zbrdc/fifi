//! KV8-B B5: the device tests of the Int8G32 step on the 3090 (one test per process, through
//! the baton). Each test's device phase is a plain `Result` function (only `?` inside; no
//! assert, expect or unwrap while a CUDA owner is alive) that returns host data; the caller
//! drops every owner simply by receiving them, and only THEN asserts (the isolation rule of
//! `native_device_rig.rs`).
//!
//! The full-step tests (a), (b) and (e) build the step the way `r5_fmt.rs` and the gate's
//! `build_step` do — the pinned GGUF, a Native load, `StepDims`, `StepResources` and
//! `QwenResidentStep::new` — with `StepOptions { kv_mode: Int8G32, layout: Native, .. }`
//! (`start.rs` still hard-codes Fp16 until B4 lands, so the options are constructed here).
//! The rig tests (c) and (d) reuse the mixer-block rig of `native_mixer_device_tests.rs` for
//! ONE full-attention layer: (c) proves the four planes equal the phase A host reference
//! quantizer applied to the kernel's own f32 rows, (d) drives a bad `pos` word through the
//! step's real Int8G32 op list (the full step's host path cannot inject one: `write_start`
//! rewrites `StepParams` from the book mirror before every token, so the poisoned word of the
//! production path — `ResidentState::poison_device` of the prefill packet — is injected here
//! as the same 4-byte write, into the real `Mixers::run_block`).

use super::common_tests::{dims as step_dims, inventory};
use super::mixer::{BlockIo, MixerParts, MixerViews, Mixers};
use super::mixer_weights::{DenseWeight, LayerWeights, MixerWeights};
use super::native_device_rig::{Rng, assert_finite, device_present, f16 as f16_value, raw_values};
use super::{
    dims::{KvMode, StepOptions},
    state::{ResidentState, StepField},
    state_layout::{Plane, Word},
    step::{CaptureConfig, QwenResidentStep},
    step_plan::{KernelScratch, Slot, StepPlan},
};
use crate::EngError;
use crate::family::gguf_model::{Role, open_qwen3_6_35b_a3b};
use crate::family::project::cuda::{
    CudaProjector, CudaWeights, LoadOptions, Reserve, WeightLayout,
};
use crate::kernels::cuda::resident::step_params::{OFFSET_POS, StepParams};
use crate::kernels::cuda::{ArenaTensor, CudaContext, DeviceStream, QuantTensor};
use crate::weights::GgufType;

/// The sampler vocabulary and the padded head width of the pinned model (`r5_fmt.rs`'s
/// constants): the rows are head-wide, the ids live in the sampler vocabulary.
const SAMPLER_VOCAB: usize = 248_077;
const HEAD_ROWS: usize = 248_320;
const EOS: [u32; 2] = [248_046, 248_044];
/// The context of the full-step tests: P5 (612 tokens) plus 128 generated rows fits, and the
/// reset test's 64 + 32 tokens restart at 0 inside it.
const CONTEXT_TOKENS: usize = 1024;
/// The KV capacity of the rig tests (c) and (d): small planes read back whole, with stale
/// cells left past the appends.
const RIG_CELLS: usize = 16;
/// Clean appends of the rig tests: (c) fills this many cells, (d) leaves 13 untouched cells
/// after its 3.
const RIG_APPENDS: usize = 9;
const RIG_CLEAN_APPENDS: usize = 3;

fn required_dir() -> String {
    std::env::var("Q36_CUDA_MODEL_DIR")
        .unwrap_or_else(|_| "/media/dan/nvme/models/qwen3.6-35b-a3b".to_string())
}

/// The id list of an oracle prompt (`start_health_tests.rs`'s `p5_ids` shape).
fn prompt_ids(name: &str) -> Vec<u32> {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../qualification/cuda-rd/oracle/prompts/")
        .join(format!("{name}.ids.json"));
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("{}: {error}", path.display()));
    let value: serde_json::Value = serde_json::from_str(&text).expect("the ids json");
    value["ids"]
        .as_array()
        .expect("ids")
        .iter()
        .map(|id| id.as_u64().expect("id") as u32)
        .collect()
}

/// The pinned checkpoint loaded once, in the Native layout the Int8G32 step requires (the
/// `r5_fmt.rs` path; `start.rs` stays Fp16 until B4).
fn load_full_native() -> Result<(CudaProjector, CudaWeights), EngError> {
    let dir = required_dir();
    let gguf_path = std::path::Path::new(&dir).join("Qwen3.6-35B-A3B-Q4_K_M.gguf");
    let gguf = open_qwen3_6_35b_a3b(&gguf_path, None)?;
    let projector = CudaProjector::new()?;
    let device = projector
        .load_model(
            &gguf,
            LoadOptions {
                reserve: Reserve::default(),
                drop_page_cache: None,
                layout: WeightLayout::Native,
            },
        )?
        .device;
    Ok((projector, device))
}

/// The full-model step options of the device tests: the Int8G32 KV mode (or Fp16, the
/// control of the reported row) over the Native layout.
fn full_options(kv_mode: KvMode) -> StepOptions {
    StepOptions {
        kv_mode,
        context_tokens: CONTEXT_TOKENS,
        delta_slots: 1,
        forked_experts: false,
        tiered: None,
        layout: WeightLayout::Native,
    }
}

/// The rules of a full-model step (`r5_fmt.rs`'s own numbers).
fn rules() -> Result<super::rules::QwenRules, EngError> {
    super::rules::QwenRules::new(EOS.to_vec(), SAMPLER_VOCAB, HEAD_ROWS)
}

// ---------------------------------------------------------------------------------------------
// The host reference quantizer of phase A (kv8_append/tests.rs), written from the frozen
// format, not from the kernel.
// ---------------------------------------------------------------------------------------------

/// The host reference quantizer of one group of 32 values: `(scale bits, codes)`. A group
/// whose STORED scale is 0 — all-zero, or an amax below what an fp16 scale can carry — has
/// codes 0.
fn quant_group(x: &[f32]) -> (u16, Vec<i8>) {
    let amax = x.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
    let bits = f16_bits(amax / 127.0);
    let s = f16_value(bits);
    let codes = if s == 0.0 {
        vec![0; x.len()]
    } else {
        x.iter()
            .map(|&v| {
                let r = (v / s).round_ties_even().clamp(-127.0, 127.0);
                r as i32 as i8
            })
            .collect()
    };
    (bits, codes)
}

/// The host reference quantizer of one f32 row block (`[n_kv_head][head_dim]`): codes
/// `i8 [n_kv_head][head_dim]`, scales `u16 [n_kv_head][head_dim / 32]`.
fn quant_row(row: &[f32], n_kv_head: usize, head_dim: usize) -> (Vec<i8>, Vec<u16>) {
    let groups = head_dim / 32;
    let mut codes = vec![0i8; n_kv_head * head_dim];
    let mut scales = vec![0u16; n_kv_head * groups];
    for kvh in 0..n_kv_head {
        for g in 0..groups {
            let at = kvh * head_dim + g * 32;
            let (bits, group_codes) = quant_group(&row[at..at + 32]);
            codes[at..at + 32].copy_from_slice(&group_codes);
            scales[kvh * groups + g] = bits;
        }
    }
    (codes, scales)
}

/// f32 to IEEE-754 binary16 bits, round-to-nearest-even, matching `__float2half` (a local
/// reimplementation: the resident testkit that holds the engine's encoder is private to
/// `kernels::cuda`, the same rule `native_device_rig.rs` follows for its decoder).
fn f16_bits(x: f32) -> u16 {
    let bits = x.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32;
    let frac = bits & 0x007f_ffff;
    if exp == 0xff {
        // Infinity and the canonical NaN (the quantizer's amax / 127 of finite rows never
        // reaches this arm; the pin covers it anyway).
        return sign | if frac == 0 { 0x7c00 } else { 0x7e00 };
    }
    let mut e = exp - 127 + 15;
    if e >= 0x1f {
        return sign | 0x7c00; // overflow rounds to infinity
    }
    if e <= 0 {
        if e < -10 || exp == 0 {
            return sign; // rounds to zero (every f32 subnormal included)
        }
        // Subnormal half: the 24-bit mantissa shifted into units of 2^-24, half-even.
        let m = 0x0080_0000 | frac;
        let shift = (14 - e) as u32;
        let mask = (1u32 << shift) - 1;
        let half = 1u32 << (shift - 1);
        let mut q = m >> shift;
        let rest = m & mask;
        if rest > half || (rest == half && (q & 1) == 1) {
            q += 1;
        }
        return sign | q as u16;
    }
    let mut keep = frac >> 13;
    let dropped = frac & 0x1fff;
    if dropped > 0x1000 || (dropped == 0x1000 && (keep & 1) == 1) {
        keep += 1;
        if keep == 0x400 {
            keep = 0;
            e += 1;
            if e >= 0x1f {
                return sign | 0x7c00;
            }
        }
    }
    sign | ((e as u16) << 10) | keep as u16
}

// ---------------------------------------------------------------------------------------------
// (a) capture on equals capture off
// ---------------------------------------------------------------------------------------------

/// (a) The Int8G32 step with capture off and on over 32 tokens of P5: the captured replays'
/// logits rows are bitwise the eager step's, and the device argmax agrees.
#[test]
#[ignore = "the 3090 through the baton (KV8-B B5): loads the real model; run by name"]
fn int8_step_capture_on_equals_capture_off_over_32_p5_tokens() {
    if !device_present("int8_step_capture_on_equals_capture_off_over_32_p5_tokens") {
        return;
    }
    let ids: Vec<u32> = prompt_ids("P5")[..32].to_vec();
    let rows = match capture_on_off_rows(&ids) {
        Ok(rows) => rows,
        Err(error) => panic!("the device phase failed: {error:?}"),
    };
    let CaptureOnOff {
        eager_rows,
        eager_toks,
        cap_rows,
        cap_toks,
    } = rows;
    // Host data only from here (the isolation split).
    assert_eq!(eager_rows.len(), 32, "the eager step read 32 rows");
    assert_eq!(cap_rows.len(), 32, "the captured step read 32 rows");
    for (at, (eager, cap)) in eager_rows.iter().zip(&cap_rows).enumerate() {
        assert_finite(&format!("eager row {at}"), &row_values(eager));
        assert_finite(&format!("captured row {at}"), &row_values(cap));
    }
    for (at, ((eager, cap), (et, ct))) in eager_rows
        .iter()
        .zip(&cap_rows)
        .zip(eager_toks.iter().zip(&cap_toks))
        .enumerate()
    {
        assert_eq!(eager, cap, "captured row {at} must equal the eager row");
        assert_eq!(et, ct, "captured argmax {at} must equal the eager argmax");
    }
}

/// What (a)'s device phase hands back: both steps' logits rows and argmax tokens.
struct CaptureOnOff {
    eager_rows: Vec<Vec<u8>>,
    eager_toks: Vec<u32>,
    cap_rows: Vec<Vec<u8>>,
    cap_toks: Vec<u32>,
}

/// The device phase of (a): one load, two steps (separate arenas and states), the eager step
/// first (its tokens all run eagerly), then the capture step (token 0 is the warm-up, token 1
/// captures, the rest replay), so the two never contend for the shared matmul workspace.
fn capture_on_off_rows(ids: &[u32]) -> Result<CaptureOnOff, EngError> {
    let (projector, weights) = load_full_native()?;
    let context = projector.context();
    let dims = step_dims();
    let inventory = inventory();
    let options = full_options(KvMode::Int8G32);
    let res_eager =
        super::resources::StepResources::new(context, dims.clone(), &inventory, &options)?;
    let mut eager = QwenResidentStep::new(context, &res_eager, &weights, &options, rules()?)?;
    let res_cap = super::resources::StepResources::new(context, dims, &inventory, &options)?;
    let mut cap = QwenResidentStep::new(context, &res_cap, &weights, &options, rules()?)?;
    cap.set_capture_config(CaptureConfig { capture: true })?;
    let mut eager_rows = Vec::with_capacity(ids.len());
    let mut cap_rows = Vec::with_capacity(ids.len());
    let (mut eager_toks, mut cap_toks) = (Vec::new(), Vec::new());
    for &id in ids {
        eager_toks.push(eager.run_token(id, true)?);
        eager_rows.push(eager.row_bytes().to_vec());
    }
    for &id in ids {
        cap_toks.push(cap.run_token(id, true)?);
        cap_rows.push(cap.row_bytes().to_vec());
    }
    drop(cap);
    drop(eager);
    drop(weights);
    drop(projector);
    Ok(CaptureOnOff {
        eager_rows,
        eager_toks,
        cap_rows,
        cap_toks,
    })
}

// ---------------------------------------------------------------------------------------------
// (b) the reset invariant
// ---------------------------------------------------------------------------------------------

/// (b) The reset invariant (design item 5): 64 tokens of prompt X, `reset_state`, then 32
/// tokens of P5 are bitwise a FRESH session's 32 tokens of P5. The reset leaves the stale
/// Int8G32 codes and scales of prompt X in every cell >= the new kv_len; the rows still match
/// because `attn_decode_q8` reads cells `[0, kv_len)` only and every cell below kv_len was
/// appended in the new session (`pos` restarts at 0).
#[test]
#[ignore = "the 3090 through the baton (KV8-B B5): loads the real model; run by name"]
fn int8_step_reset_leaves_a_fresh_sessions_rows() {
    if !device_present("int8_step_reset_leaves_a_fresh_sessions_rows") {
        return;
    }
    let x: Vec<u32> = prompt_ids("P4")[..64].to_vec();
    let p5: Vec<u32> = prompt_ids("P5")[..32].to_vec();
    let rows = match reset_rows(&x, &p5) {
        Ok(rows) => rows,
        Err(error) => panic!("the device phase failed: {error:?}"),
    };
    let (after_reset_rows, fresh_rows) = (rows.after_reset, rows.fresh);
    // Host data only from here (the isolation split).
    assert_eq!(after_reset_rows.len(), 32, "the reset session read 32 rows");
    assert_eq!(fresh_rows.len(), 32, "the fresh session read 32 rows");
    for (at, row) in after_reset_rows.iter().chain(&fresh_rows).enumerate() {
        assert_finite(&format!("row {at}"), &row_values(row));
    }
    for (at, (after, fresh)) in after_reset_rows.iter().zip(&fresh_rows).enumerate() {
        assert_eq!(
            after, fresh,
            "row {at} after the reset must equal the fresh session's row"
        );
    }
}

/// What (b)'s device phase hands back: the 32 P5 rows of the session that reset (after its
/// 64 tokens of prompt X) and of the fresh session.
struct ResetPair {
    after_reset: Vec<Vec<u8>>,
    fresh: Vec<Vec<u8>>,
}

/// The device phase of (b): one load, two steps. The first runs prompt X, resets and runs
/// P5; the second is the fresh session that runs P5 only.
fn reset_rows(x: &[u32], p5: &[u32]) -> Result<ResetPair, EngError> {
    let (projector, weights) = load_full_native()?;
    let context = projector.context();
    let dims = step_dims();
    let inventory = inventory();
    let options = full_options(KvMode::Int8G32);
    let res_after =
        super::resources::StepResources::new(context, dims.clone(), &inventory, &options)?;
    let mut after_reset = QwenResidentStep::new(context, &res_after, &weights, &options, rules()?)?;
    let res_fresh = super::resources::StepResources::new(context, dims, &inventory, &options)?;
    let mut fresh = QwenResidentStep::new(context, &res_fresh, &weights, &options, rules()?)?;
    let mut after_rows = Vec::with_capacity(p5.len());
    let mut fresh_rows = Vec::with_capacity(p5.len());
    for &id in x {
        after_reset.run_token(id, false)?;
    }
    after_reset.reset_state()?;
    for &id in p5 {
        after_reset.run_token(id, true)?;
        after_rows.push(after_reset.row_bytes().to_vec());
    }
    for &id in p5 {
        fresh.run_token(id, true)?;
        fresh_rows.push(fresh.row_bytes().to_vec());
    }
    drop(fresh);
    drop(after_reset);
    drop(weights);
    drop(projector);
    Ok(ResetPair {
        after_reset: after_rows,
        fresh: fresh_rows,
    })
}

// ---------------------------------------------------------------------------------------------
// (c) and (d): the one-layer rig
// ---------------------------------------------------------------------------------------------

/// The four planes of one attention layer in read order.
const PLANES: [Plane; 4] = [Plane::K, Plane::KScales, Plane::V, Plane::VScales];

/// The four whole planes of `ordinal`'s KV pool, read back as bytes.
fn read_planes(
    stream: &DeviceStream,
    state: &ResidentState,
    ordinal: usize,
) -> Result<[Vec<u8>; 4], EngError> {
    let mut planes: [Vec<u8>; 4] = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
    for (out, plane) in planes.iter_mut().zip(PLANES) {
        let range = state.kv_plane(ordinal, plane)?;
        let mut bytes = vec![0u8; range.bytes()];
        range.buffer().read_at(stream, range.offset(), &mut bytes)?;
        *out = bytes;
    }
    Ok(planes)
}

/// The rig-wide pieces every append runs against: the stream, the mixers, the unaliased
/// arena, the plan, the one full-attention layer's index, its residual slot index and the
/// hidden width.
struct Rig<'r, 'm> {
    stream: &'r DeviceStream,
    mixers: &'r Mixers<'m>,
    arena: &'r crate::kernels::cuda::DeviceArena,
    plan: &'r StepPlan,
    layer: usize,
    resid_index: usize,
    hidden: usize,
}

/// One clean append of the rig at cell `t`: the real step's own call order (start block,
/// enqueue, block, read-back, commit), with a fresh residual per token. Returns the f32 k
/// and v rows the block's prep wrote into KRows/VRows.
fn rig_append(
    rig: &Rig<'_, '_>,
    state: &mut ResidentState,
    t: usize,
) -> Result<(Vec<f32>, Vec<f32>), EngError> {
    state.write_step_params(rig.stream, &StepParams::new(t as i32, t as i32)?)?;
    let admitted = state.enqueue_rows(1)?;
    let mut rng = Rng::new(0x1234_0000 + t as u64);
    let resid: Vec<f32> = (0..rig.hidden)
        .map(|_| (rng.below(200) as f32 - 100.0) / 500.0)
        .collect();
    let resid_bytes: Vec<u8> = resid.iter().flat_map(|v| v.to_le_bytes()).collect();
    let view = rig.arena.tensor(rig.resid_index)?;
    view.buffer()
        .write_at(rig.stream, view.offset(), &resid_bytes)?;
    rig.mixers.run_block(
        rig.stream,
        state,
        rig.layer,
        BlockIo {
            resid: view,
            admit_pos: admitted.admit_pos,
        },
    )?;
    rig.stream.synchronize()?;
    let read = |slot: Slot| -> Result<Vec<f32>, EngError> {
        let at = rig
            .plan
            .index(slot)
            .ok_or_else(|| EngError::Checkpoint(format!("no slot {slot:?}")))?;
        let view = rig.arena.tensor(at)?;
        let mut bytes = vec![0u8; view.bytes()];
        view.buffer()
            .read_at(rig.stream, view.offset(), &mut bytes)?;
        Ok(bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect())
    };
    let rows = (read(Slot::KRows)?, read(Slot::VRows)?);
    state.commit_step(1, 0)?;
    Ok(rows)
}

/// What the rig tests read back: the plane bytes, the f32 rows of every append and the
/// attention geometry.
struct RigData {
    planes: [Vec<u8>; 4],
    k_rows: Vec<Vec<f32>>,
    v_rows: Vec<Vec<f32>>,
    heads: usize,
    head_dim: usize,
    cells: usize,
    /// The status word, the attention core row and the mixer output row of the poisoned runs
    /// (empty vectors before the first poisoned run).
    poisoned: Vec<PoisonOutcome>,
}

/// One poisoned run's read-back.
struct PoisonOutcome {
    status: i32,
    core: Vec<f32>,
    mix_out: Vec<f32>,
    planes: [Vec<u8>; 4],
}

/// Builds the one-layer Int8G32 rig and runs its device phase: `clean` appends first (their
/// rows read back), then one poisoned run per word of `poison_words` (the `pos` field only,
/// exactly the write `poison_device` performs), with the planes read after the clean appends
/// and after every poisoned run.
fn rig_phase(clean: usize, poison_words: &[i32]) -> Result<RigData, EngError> {
    let context = CudaContext::new()?;
    let stream = context.stream()?;
    let dims = step_dims();
    let options = StepOptions {
        kv_mode: KvMode::Int8G32,
        context_tokens: RIG_CELLS,
        delta_slots: 1,
        forked_experts: false,
        tiered: None,
        layout: WeightLayout::Native,
    };
    let scratch = KernelScratch::for_dims(&dims, 1, KvMode::Int8G32)?;
    let plan = StepPlan::new(&dims, 1, &options, &scratch)?;
    // Widen every lifetime to the whole step (the `native_mixer_device_tests.rs` rule): the
    // dead KRows/VRows slots stay readable after the block, and no slot shares bytes.
    let planned = plan.arena_tensors();
    let end = planned.iter().map(|t| t.last).max().unwrap_or(0);
    let unaliased: Vec<ArenaTensor> = planned
        .into_iter()
        .map(|t| ArenaTensor {
            first: 0,
            last: end,
            ..t
        })
        .collect();
    let arena = context.arena(&unaliased)?;
    let inventory_value = inventory();
    let table = super::layer_table::LayerTable::new(&dims, &inventory_value)?;
    let geometry = super::mixer_geometry::MixerGeometry::new(&dims)?;
    let mut state = ResidentState::allocate(&context, &stream, &dims, &options)?;
    state.write_step_params(&stream, &StepParams::new(0, 0)?)?;
    // The first full-attention layer and its ordinal among the attention layers.
    let mut found = None;
    for l in 0..table.len() {
        if table.slot(l)?.kind == crate::family::AttentionKind::Full {
            found = Some(l);
            break;
        }
    }
    let layer = found.ok_or_else(|| EngError::Checkpoint("no attention layer".into()))?;
    let ordinal = table.slot(layer)?.ordinal;
    // The rig weights of `native_mixer_device_tests.rs`: Q4_K dense sources uploaded as
    // native blocks, raw f32 values for the norms.
    let kind = crate::family::AttentionKind::Full;
    let mut native: Vec<crate::kernels::cuda::NativeWeight> = Vec::new();
    let mut dense_specs: Vec<(Role, usize)> = Vec::new();
    for spec in geometry.dense_specs(kind)? {
        let mut rng = Rng::new(0x5eed_0000 + spec.rows as u64);
        let source = super::native_device_rig::q4k_rows(spec.rows, spec.k, &mut rng);
        let tensor = QuantTensor {
            name: "rig",
            ty: GgufType::Q4K,
            n_rows: spec.rows,
            k: spec.k,
            source_len: source.len(),
        };
        let reader = &mut |at: usize, buf: &mut [u8]| {
            buf.copy_from_slice(&source[at..at + buf.len()]);
            Ok(())
        };
        dense_specs.push((spec.role, native.len()));
        native.push(context.upload_native(tensor, 1 << 16, reader)?);
    }
    let dense: Vec<(Role, DenseWeight<'_>)> = dense_specs
        .into_iter()
        .map(|(role, at)| (role, DenseWeight::Native(&native[at])))
        .collect();
    let mut raw_specs: Vec<(Role, crate::kernels::cuda::DeviceBuffer, usize)> = Vec::new();
    for spec in geometry.raw_specs(kind)? {
        let values = raw_values(&spec.role, spec.values);
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let buffer = context.upload(&bytes)?;
        raw_specs.push((spec.role, buffer, values.len() * 4));
    }
    let mut raw: Vec<(Role, crate::kernels::cuda::DeviceRange<'_>)> = Vec::new();
    for (role, buffer, bytes) in &raw_specs {
        raw.push((*role, buffer.range(0, *bytes)?));
    }
    let mut entries: Vec<Option<LayerWeights<'_>>> = (0..table.len()).map(|_| None).collect();
    entries[layer] = Some(LayerWeights::for_tests(dense, raw));
    let weights = MixerWeights::from_layers(entries);
    let parts = MixerParts {
        geometry,
        table: &table,
        weights,
        views: MixerViews::new(
            &plan,
            &arena,
            &geometry,
            WeightLayout::Native,
            KvMode::Int8G32,
        )?,
        layout: WeightLayout::Native,
        kv_mode: KvMode::Int8G32,
    };
    let mixers = Mixers::new(&context, parts)?;
    let resid_slot = if layer.is_multiple_of(2) {
        Slot::ResidA
    } else {
        Slot::ResidB
    };
    let resid_index = plan
        .index(resid_slot)
        .ok_or_else(|| EngError::Checkpoint("no resid slot".into()))?;
    let rig = Rig {
        stream: &stream,
        mixers: &mixers,
        arena: &arena,
        plan: &plan,
        layer,
        resid_index,
        hidden: dims.shape.hidden,
    };
    // The clean appends, then the pristine planes, then the poisoned runs.
    let mut k_rows = Vec::with_capacity(clean);
    let mut v_rows = Vec::with_capacity(clean);
    for t in 0..clean {
        let (k, v) = rig_append(&rig, &mut state, t)?;
        k_rows.push(k);
        v_rows.push(v);
    }
    let planes = read_planes(&stream, &state, ordinal)?;
    let mut poisoned = Vec::with_capacity(poison_words.len());
    for &word in poison_words {
        // The poisoned run: the `pos` word only (kv_len keeps the valid word of the last
        // committed token, so a NaN output comes from the refused prep, not from the decode's
        // own kv_len guard), then the real block and its read-back.
        let params = state.word(Word::StepParams)?;
        params
            .buffer()
            .write_at(&stream, params.offset() + OFFSET_POS, &word.to_le_bytes())?;
        let resid_view = arena.tensor(resid_index)?;
        mixers.run_block(
            &stream,
            &state,
            layer,
            BlockIo {
                resid: resid_view,
                admit_pos: clean,
            },
        )?;
        stream.synchronize()?;
        let read = |slot: Slot| -> Result<Vec<f32>, EngError> {
            let at = plan
                .index(slot)
                .ok_or_else(|| EngError::Checkpoint(format!("no slot {slot:?}")))?;
            let view = arena.tensor(at)?;
            let mut bytes = vec![0u8; view.bytes()];
            view.buffer().read_at(&stream, view.offset(), &mut bytes)?;
            Ok(bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| f32::from_le_bytes(*c))
                .collect())
        };
        let status_range = state.step_field(StepField::Status)?;
        let mut status_bytes = [0u8; 4];
        status_range
            .buffer()
            .read_at(&stream, status_range.offset(), &mut status_bytes)?;
        poisoned.push(PoisonOutcome {
            status: i32::from_le_bytes(status_bytes),
            core: read(Slot::AttnCore)?,
            mix_out: read(Slot::MixOut)?,
            planes: read_planes(&stream, &state, ordinal)?,
        });
    }
    let cells = state.layout().kv.capacity;
    drop(mixers);
    drop(state);
    drop(arena);
    drop(stream);
    drop(context);
    Ok(RigData {
        planes,
        k_rows,
        v_rows,
        heads: dims.shape.attn.heads_kv,
        head_dim: dims.shape.attn.head_dim,
        cells,
        poisoned,
    })
}

/// (c) The planes against the host quantizer: after `RIG_APPENDS` appends of the one-layer
/// rig, the four planes equal the phase A host reference quantizer applied to the kernel's
/// own f32 rows (read back from KRows/VRows), bitwise, cell for cell, and the cells past the
/// appends are still the allocation's zeros.
#[test]
fn int8_planes_equal_the_host_quantizer_of_the_kernels_own_rows() {
    if !device_present("int8_planes_equal_the_host_quantizer_of_the_kernels_own_rows") {
        return;
    }
    let data = match rig_phase(RIG_APPENDS, &[]) {
        Ok(data) => data,
        Err(error) => panic!("the device phase failed: {error:?}"),
    };
    let (heads, head_dim, cells) = (data.heads, data.head_dim, data.cells);
    let groups = head_dim / 32;
    // Host data only from here (the isolation split).
    for (t, (k, v)) in data.k_rows.iter().zip(&data.v_rows).enumerate() {
        assert_finite(&format!("k rows {t}"), k);
        assert_finite(&format!("v rows {t}"), v);
    }
    // The expected planes: the head-major layout of `state_layout.rs` (`codes
    // [head][cell][head_dim]`, `scales [head][cell][groups]`), the appended cells quantized
    // from the read-back rows, the rest the zeroed allocation.
    let mut want: [Vec<u8>; 4] = [
        vec![0u8; data.planes[0].len()],
        vec![0u8; data.planes[1].len()],
        vec![0u8; data.planes[2].len()],
        vec![0u8; data.planes[3].len()],
    ];
    for t in 0..data.k_rows.len() {
        let (k_codes, k_scales) = quant_row(&data.k_rows[t], heads, head_dim);
        let (v_codes, v_scales) = quant_row(&data.v_rows[t], heads, head_dim);
        for kvh in 0..heads {
            let codes_at = (kvh * cells + t) * head_dim;
            want[0][codes_at..codes_at + head_dim].copy_from_slice(
                &k_codes[kvh * head_dim..][..head_dim]
                    .iter()
                    .map(|&c| c as u8)
                    .collect::<Vec<_>>(),
            );
            let scales_at = (kvh * cells + t) * groups;
            for (g, bits) in k_scales[kvh * groups..][..groups].iter().enumerate() {
                want[1][(scales_at + g) * 2..(scales_at + g) * 2 + 2]
                    .copy_from_slice(&bits.to_le_bytes());
            }
            want[2][codes_at..codes_at + head_dim].copy_from_slice(
                &v_codes[kvh * head_dim..][..head_dim]
                    .iter()
                    .map(|&c| c as u8)
                    .collect::<Vec<_>>(),
            );
            for (g, bits) in v_scales[kvh * groups..][..groups].iter().enumerate() {
                want[3][(scales_at + g) * 2..(scales_at + g) * 2 + 2]
                    .copy_from_slice(&bits.to_le_bytes());
            }
        }
    }
    let names = ["K codes", "K scales", "V codes", "V scales"];
    for (name, (got, want)) in names.iter().zip(data.planes.iter().zip(&want)) {
        assert_eq!(got.len(), want.len(), "{name} plane size");
        assert!(got == want, "{name} plane differs from the host quantizer");
    }
}

/// (d) A bad `pos` word poisons the step: the rows entry writes no row (its outputs are NaN),
/// `kv8_append` writes no plane byte, and the status word reports mask 4 — for `pos` = -1
/// (the saturated advance / `poison_device` word) and `pos` = cells (one past the last valid
/// cell).
#[test]
fn a_bad_pos_word_poisons_the_int8_step() {
    if !device_present("a_bad_pos_word_poisons_the_int8_step") {
        return;
    }
    let data = match rig_phase(RIG_CLEAN_APPENDS, &[-1, RIG_CELLS as i32]) {
        Ok(data) => data,
        Err(error) => panic!("the device phase failed: {error:?}"),
    };
    // Host data only from here (the isolation split).
    assert_eq!(data.poisoned.len(), 2, "the two poisoned runs");
    for (label, outcome) in [
        ("pos = -1", &data.poisoned[0]),
        ("pos = cells", &data.poisoned[1]),
    ] {
        assert_eq!(
            outcome.status, 4,
            "{label}: the status word must carry exactly mask 4"
        );
        assert!(
            outcome.core.iter().all(|v| v.is_nan()),
            "{label}: the attention core row must be all NaN"
        );
        assert!(
            outcome.mix_out.iter().all(|v| v.is_nan()),
            "{label}: the mixer output row must be all NaN"
        );
        for (name, (before, after)) in [
            ("K codes", (&data.planes[0], &outcome.planes[0])),
            ("K scales", (&data.planes[1], &outcome.planes[1])),
            ("V codes", (&data.planes[2], &outcome.planes[2])),
            ("V scales", (&data.planes[3], &outcome.planes[3])),
        ] {
            assert!(
                before == after,
                "{label}: the {name} plane must be unchanged"
            );
        }
    }
}

// ---------------------------------------------------------------------------------------------
// (e) the reported row
// ---------------------------------------------------------------------------------------------

/// (e) The REPORTED row (never an assertion on the numbers): the Int8G32 and Fp16 steps on
/// P5, teacher-forced on the Fp16 greedy tokens over 128 tokens. Prints the top-1 agreement
/// and the max |Δlogit| of the argmax.
#[test]
#[ignore = "the 3090 through the baton (KV8-B B5): loads the real model; run by name"]
fn int8_vs_fp16_step_row_on_p5_is_reported() {
    if !device_present("int8_vs_fp16_step_row_on_p5_is_reported") {
        return;
    }
    let report = match quality_row() {
        Ok(report) => report,
        Err(error) => panic!("the device phase failed: {error:?}"),
    };
    // Host data only from here (the isolation split). The finiteness of every compared row
    // is asserted; the agreement numbers are REPORTED, never gated.
    for (at, row) in report.fp16_rows.iter().enumerate() {
        assert_finite(&format!("fp16 row {at}"), &row_values(row));
    }
    for (at, row) in report.int8_rows.iter().enumerate() {
        assert_finite(&format!("int8 row {at}"), &row_values(row));
    }
    let mut agree = 0;
    let mut max_delta = 0.0f32;
    for (at, (fp16, int8)) in report.fp16_rows.iter().zip(&report.int8_rows).enumerate() {
        let id = report.greedy[at] as usize;
        let (fp16, int8) = (row_values(fp16), row_values(int8));
        if report.int8_toks[at] as usize == id {
            agree += 1;
        }
        let delta = (int8[id] - fp16[id]).abs();
        assert!(
            delta.is_finite(),
            "row {at}: the argmax delta must be finite"
        );
        if delta > max_delta {
            max_delta = delta;
        }
    }
    let total = report.fp16_rows.len();
    println!(
        "KV8-B B5 (e): int8g32 vs fp16 on P5 over {total} teacher-forced tokens: \
         top-1 agreement {agree}/{total} ({:.2}%), max |dlogit| at the argmax {max_delta:.6}",
        100.0 * agree as f32 / total as f32
    );
}

/// What (e)'s device phase hands back: both steps' rows, the Fp16 greedy ids and the Int8G32
/// step's argmax tokens.
struct QualityRow {
    fp16_rows: Vec<Vec<u8>>,
    int8_rows: Vec<Vec<u8>>,
    greedy: Vec<u32>,
    int8_toks: Vec<u32>,
}

/// The device phase of (e): the Fp16 step streams P5 greedily for 128 rows, then the Int8G32
/// step runs the same prompt teacher-forced on the Fp16 greedy ids for its 128 rows.
fn quality_row() -> Result<QualityRow, EngError> {
    let (projector, weights) = load_full_native()?;
    let context = projector.context();
    let dims = step_dims();
    let inventory_value = inventory();
    let prompt = prompt_ids("P5");
    let steps = 128;
    let last = *prompt
        .last()
        .ok_or_else(|| EngError::BadRequest("an empty prompt".into()))?;
    // The Fp16 step first, free-run greedy: the prefill runs every prompt token but the last
    // (the gate's `prefill` shape), then the last prompt token seeds the 128 rows.
    let fp16_options = full_options(KvMode::Fp16);
    let res_fp16 = super::resources::StepResources::new(
        context,
        dims.clone(),
        &inventory_value,
        &fp16_options,
    )?;
    let mut fp16 = QwenResidentStep::new(context, &res_fp16, &weights, &fp16_options, rules()?)?;
    for &id in &prompt[..prompt.len() - 1] {
        fp16.run_token(id, false)?;
    }
    let mut greedy = Vec::with_capacity(steps);
    let mut fp16_rows = Vec::with_capacity(steps);
    let mut feed = last;
    for _ in 0..steps {
        let chosen = fp16.run_token(feed, true)?;
        fp16_rows.push(fp16.row_bytes().to_vec());
        greedy.push(chosen);
        feed = chosen;
    }
    drop(fp16);
    drop(res_fp16);
    // The Int8G32 step, teacher-forced on the Fp16 greedy ids.
    let int8_options = full_options(KvMode::Int8G32);
    let res_int8 =
        super::resources::StepResources::new(context, dims, &inventory_value, &int8_options)?;
    let mut int8 = QwenResidentStep::new(context, &res_int8, &weights, &int8_options, rules()?)?;
    for &id in &prompt[..prompt.len() - 1] {
        int8.run_token(id, false)?;
    }
    let mut int8_rows = Vec::with_capacity(steps);
    let mut int8_toks = Vec::with_capacity(steps);
    let mut feed = last;
    for &forced in &greedy {
        let chosen = int8.run_token(feed, true)?;
        int8_rows.push(int8.row_bytes().to_vec());
        int8_toks.push(chosen);
        feed = forced;
    }
    drop(int8);
    drop(res_int8);
    drop(weights);
    drop(projector);
    Ok(QualityRow {
        fp16_rows,
        int8_rows,
        greedy,
        int8_toks,
    })
}

/// One logits row as f32 values.
fn row_values(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect()
}

/// CPU pins of the local half encoder: the exact bits of known values, and a round trip
/// through the rig's decoder for a spread of magnitudes (the quantizer's amax / 127 range).
#[test]
fn the_local_half_encoder_round_trips() {
    assert_eq!(f16_bits(0.0), 0x0000);
    assert_eq!(f16_bits(-0.0), 0x8000);
    assert_eq!(f16_bits(1.0), 0x3c00);
    assert_eq!(f16_bits(-2.0), 0xc000);
    assert_eq!(f16_bits(65504.0), 0x7bff);
    assert_eq!(
        f16_bits(65520.0),
        0x7c00,
        "the overflow half-way point rounds to infinity"
    );
    assert_eq!(f16_bits(f32::INFINITY), 0x7c00);
    assert_eq!(f16_bits(f32::NAN) & 0x7e00, 0x7e00, "NaN stays a NaN");
    assert_eq!(f16_bits(5.96e-8), 0x0001, "the smallest subnormal");
    let mut x = 1e-6f32;
    while x < 1e5 {
        let bits = f16_bits(x);
        let decoded = f16_value(bits);
        // The decoded value is the nearest half to x, so |decoded - x| is at most half the
        // local grid step (a subnormal step below the smallest normal, an ulp above).
        let step = if bits & 0x7c00 == 0 {
            5.96e-8
        } else {
            decoded.abs() * 9.77e-4
        };
        assert!(
            (decoded - x).abs() <= step,
            "{x}: {decoded} (bits {bits:#06x})"
        );
        x *= 1.37;
    }
}
