//! Device test of the NATIVE mixer block of MATVEC-PAR S3b (packet test 6), in the MSG-73
//! isolation split: the device phase (`run_block`) returns `Result` with only `?` inside - no
//! assert, expect or unwrap - and hands back host `Vec`s; every device owner drops when it
//! returns, and only THEN do the asserts run. The Q8 slots are ZEROED before the run so a
//! skipped producer (m5) leaves a known-zero pack, not stale memory.
//!
//! The block-end `NormRow` is NOT compared across layouts (MSG-73 F3: it is the epilogue norm
//! of resid + MixOut, and MixOut differs between the layouts by design; the producer equality
//! is packet test 10). The bounds instead: the input hops take `A` from the PROLOGUE norm
//! computed on the host from resid and AttnNorm; the output hop takes `A` from the read-back
//! core row; the whole-block envelope multiplies the output single hop by BLOCK_HOPS = 6
//! (derived: the block quantizes activations at three stages - the in-pack, the core-out pack,
//! the epilogue pack - each contributing its input-side and output-side quantization envelope
//! at most once, the gated rule's gates and decays bounding propagation at 1).
use super::super::host::HostShape;
use super::common_tests::{dims as step_dims, inventory};
use super::dims::StepDims;
use super::mixer::{BlockIo, MixerParts, MixerViews, Mixers};
use super::mixer_weights::{DenseWeight, LayerWeights, MixerWeights};
use super::native_device_rig::{Rng, assert_finite, device_present, raw_values, row_l1};
use super::step_plan::{KernelScratch, Slot, StepPlan};
use crate::EngError;
use crate::family::project::cuda::WeightLayout;
use crate::family::{QWEN3_6_35B_A3B, gguf_model::Role};
use crate::kernels::cuda::resident::step_params::StepParams;
use crate::kernels::cuda::{ArenaTensor, CudaContext, DeviceBuffer, DeviceStream, QuantTensor};
use crate::weights::GgufType;

const F32: usize = size_of::<f32>();
/// Derived in the module doc: three quantized activation stages, input and output side each.
const BLOCK_HOPS: f64 = 6.0;

/// The block's readable outputs, host data only.
struct BlockRun {
    resid_out: Vec<f32>,
    /// The mixer output row (the epilogue's delta input), for the per-side identity check (m8).
    mix_out: Vec<f32>,
    proj: Vec<f32>,
    /// The read-back core row (DeltaCore or AttnCore): the output hop's own input amplitude.
    core: Vec<f32>,
}

/// The device phase (the isolation rule: only `?` inside). One mixer block of `layer` in
/// `layout` from the same Q4_K sources - uploaded as tiles and as native blocks, OWNED in this
/// scope (no leaks: MSG-73 F4).
fn tiled_index(specs: &[super::mixer_geometry::DenseSpec], role: Role) -> usize {
    specs
        .iter()
        .position(|s| s.role == role)
        .expect("role in specs")
}

fn run_block(
    ctx: &CudaContext,
    stream: &DeviceStream,
    layer: usize,
    layout: WeightLayout,
    resid: &[f32],
    sources: &[(Role, Vec<u8>)],
) -> Result<BlockRun, EngError> {
    let inv = inventory();
    let shape = HostShape::from_spec(&QWEN3_6_35B_A3B, 64)?;
    let dims = StepDims::new(&shape, &inv)?;
    let options = super::dims::StepOptions {
        kv_mode: super::dims::KvMode::Fp16,
        context_tokens: 8,
        delta_slots: 1,
        forked_experts: false,
        tiered: None,
        layout,
    };
    let scratch = KernelScratch::for_dims(&dims, 1, super::dims::KvMode::Fp16)?;
    let plan = StepPlan::new(&dims, 1, &options, &scratch)?;
    // The rig reads the projection row AFTER the whole block, so no slot may share bytes: the
    // plan hands DeltaProj's bytes to DeltaCore, the out pack and MixOut once DELTA_RULE ends,
    // and a read-back of the aliased slot sees their bytes (q8 packs read as f32 hold NaNs).
    // Every lifetime is widened to the whole step; the views still come from `arena.tensor`.
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
    let arena = ctx.arena(&unaliased)?;
    let table = super::layer_table::LayerTable::new(&dims, &inv)?;
    let geometry = super::mixer_geometry::MixerGeometry::new(&dims)?;
    let mut state = super::state::ResidentState::allocate(ctx, stream, &dims, &options)?;
    state.write_step_params(stream, &StepParams::new(0, 0)?)?;
    let resid_slot = if layer.is_multiple_of(2) {
        Slot::ResidA
    } else {
        Slot::ResidB
    };
    let resid_view = arena.tensor(
        plan.index(resid_slot)
            .ok_or_else(|| EngError::Checkpoint("no resid slot".into()))?,
    )?;
    resid_view.buffer().write_at(
        stream,
        resid_view.offset(),
        &resid
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>(),
    )?;
    // Zero the Q8 packs before the run: a skipped producer (m5) leaves a KNOWN-zero pack.
    for slot in [Slot::Q8Mix, Slot::Q8DeltaOut, Slot::Q8AttnOut] {
        if let Some(at) = plan.index(slot) {
            let view = arena.tensor(at)?;
            view.buffer()
                .write_at(stream, view.offset(), &vec![0u8; view.bytes()])?;
        }
    }
    let kind = table.slot(layer)?.kind;
    let mut tiled: Vec<crate::kernels::cuda::TiledWeight> = Vec::new();
    let mut native: Vec<crate::kernels::cuda::NativeWeight> = Vec::new();
    let mut native_by_role: Vec<(Role, usize)> = Vec::new();
    // The rig mirrors the loader for BOTH layouts. Enumerate the canonical unfused roles
    // (Tiled never fuses) for the Tiled arm, and when the flag fuses Native uploads,
    // row-concatenate a complete sibling group under the qkv name exactly as the loader
    // does - the fused op list then finds one wide native weight and never asks for the
    // siblings, whose native uploads are skipped.
    let fuse = crate::family::project::cuda::gdn_fuse_inproj() && layout == WeightLayout::Native;
    let specs: Vec<_> = geometry
        .dense_specs(WeightLayout::Tiled, kind)?
        .into_iter()
        .collect::<Vec<_>>();
    let has = |role: Role| specs.iter().any(|s| s.role == role);
    for spec in &specs {
        let source = &sources
            .iter()
            .find(|(r, _)| *r == spec.role)
            .ok_or_else(|| EngError::Checkpoint(format!("no source for {:?}", spec.role)))?
            .1;
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
        let t_at = tiled.len();
        tiled.push(ctx.upload_tiled(tensor, 1 << 16, reader)?);
        let group: &[Role] = &[Role::AttnGate, Role::SsmBeta, Role::SsmAlpha];
        // The loader fuses a group only when every member is present; a partial set
        // uploads per-role. The rig mirrors that: the sibling skip applies only when
        // the complete group actually formed (qkv present + all siblings present).
        let complete_group = has(Role::AttnQkv) && group.iter().all(|r| has(*r));
        if fuse && spec.role == Role::AttnQkv && complete_group {
            let mut fused = source.clone();
            let mut rows = spec.rows;
            for role in group {
                let sibling = &sources
                    .iter()
                    .find(|(r, _)| *r == *role)
                    .ok_or_else(|| EngError::Checkpoint(format!("no source for {:?}", role)))?
                    .1;
                rows += specs
                    .iter()
                    .find(|s| s.role == *role)
                    .expect("sibling spec")
                    .rows;
                fused.extend_from_slice(sibling);
            }
            let fused_tensor = QuantTensor {
                name: "rig",
                ty: GgufType::Q4K,
                n_rows: rows,
                k: spec.k,
                source_len: fused.len(),
            };
            let fused_reader = &mut |at: usize, buf: &mut [u8]| {
                buf.copy_from_slice(&fused[at..at + buf.len()]);
                Ok(())
            };
            native_by_role.push((spec.role, native.len()));
            native.push(ctx.upload_native(fused_tensor, 1 << 16, fused_reader)?);
        } else if fuse && group.contains(&spec.role) && complete_group {
            // Sibling of a COMPLETE fused group: uploaded only in the Tiled arm; a
            // partial group uploads per-role, like the loader.
            continue;
        } else {
            native_by_role.push((spec.role, native.len()));
            native.push(ctx.upload_native(tensor, 1 << 16, reader)?);
        }
        let _ = t_at;
    }
    let dense: Vec<(Role, DenseWeight<'_>)> = specs
        .iter()
        .filter_map(|spec| {
            let t_at = tiled_index(&specs, spec.role);
            let weight = match layout {
                WeightLayout::Tiled => DenseWeight::Tiled(&tiled[t_at]),
                WeightLayout::Exl3 => panic!("GGUF fixture cannot fabricate EXL3 components"),
                WeightLayout::Native => {
                    // Under the fused ops the siblings have no native weight; they are
                    // simply absent from the map (nothing asks for them).
                    let n_at = native_by_role
                        .iter()
                        .find(|(r, _)| *r == spec.role)
                        .map(|(_, at)| *at)?;
                    DenseWeight::Native(&native[n_at])
                }
            };
            Some((spec.role, weight))
        })
        .collect();
    let mut raw_specs: Vec<(Role, DeviceBuffer, usize)> = Vec::new();
    for spec in geometry.raw_specs(kind)? {
        let values = raw_values(&spec.role, spec.values);
        raw_specs.push((
            spec.role,
            ctx.upload(
                &values
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect::<Vec<_>>(),
            )?,
            values.len() * F32,
        ));
    }
    let mut raw: Vec<(Role, crate::kernels::cuda::DeviceRange<'_>)> = Vec::new();
    for (role, buf, bytes) in &raw_specs {
        raw.push((*role, buf.range(0, *bytes)?));
    }
    let mut entries: Vec<Option<LayerWeights<'_>>> = (0..table.len()).map(|_| None).collect();
    entries[layer] = Some(LayerWeights::for_tests(dense, raw));
    let weights = MixerWeights::from_layers(entries);
    let parts = MixerParts {
        geometry,
        table: &table,
        weights,
        views: MixerViews::new(&plan, &arena, &geometry, layout, super::dims::KvMode::Fp16)?,
        layout,
        kv_mode: super::dims::KvMode::Fp16,
        paged: None,
    };
    let mixers = Mixers::new(ctx, parts)?;
    mixers.run_block(
        stream,
        &state,
        layer,
        BlockIo {
            resid: resid_view,
            admit_pos: 0,
        },
    )?;
    stream.synchronize()?;
    let read = |slot: Slot| -> Result<Vec<f32>, EngError> {
        let view = arena.tensor(
            plan.index(slot)
                .ok_or_else(|| EngError::Checkpoint(format!("no slot {slot:?}")))?,
        )?;
        let mut bytes = vec![0u8; view.bytes()];
        view.buffer().read_at(stream, view.offset(), &mut bytes)?;
        Ok(bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect())
    };
    let (proj_slot, core_slot) = if kind == crate::family::AttentionKind::DeltaNet {
        (Slot::DeltaProj, Slot::DeltaCore)
    } else {
        (Slot::AttnProj, Slot::AttnCore)
    };
    Ok(BlockRun {
        resid_out: read(resid_slot)?,
        mix_out: read(Slot::MixOut)?,
        proj: read(proj_slot)?,
        core: read(core_slot)?,
    })
}

/// The host prologue norm: `x = rstd * h * w` in f64 from the rig's own inputs (the bound's A
/// for the input hops; the kernel's f32 arithmetic differs only at reduction rounding).
fn host_prologue_norm(resid: &[f32], weight: &[f32]) -> Vec<f64> {
    let sum: f64 = resid
        .iter()
        .zip(weight)
        .map(|(h, w)| (h * w) as f64 * (h * w) as f64)
        .sum();
    let rstd = 1.0 / (sum / resid.len() as f64 + 1e-6).sqrt();
    resid
        .iter()
        .zip(weight)
        .map(|(h, w)| rstd * (*h * w) as f64)
        .collect()
}

/// The bound of one hop (the rig's derivation; `a` is THAT hop's input amplitude).
fn hop_bound(l1: f64, a: f64) -> f64 {
    a * l1 * ((0.5 + 127.0 / 2048.0) / 127.0 + 1.0 / 1024.0 + 1.0 / 262_144.0)
}

#[test]
fn native_mixer_block_matches_the_tiled_block() {
    if !device_present("native_mixer_block_matches_the_tiled_block") {
        return;
    }
    let ctx = CudaContext::new().expect("context");
    let stream = ctx.stream().expect("stream");
    let dims = step_dims();
    let inventory = inventory();
    let table = super::layer_table::LayerTable::new(&dims, &inventory).expect("table");
    let geometry = super::mixer_geometry::MixerGeometry::new(&dims).expect("geometry");
    let mut rng = Rng::new(0x5eed_b10c);
    let resid: Vec<f32> = (0..dims.shape.hidden)
        .map(|_| (rng.below(200) as f32 - 100.0) / 500.0)
        .collect();
    assert_finite("the residual input", &resid);
    let norm_weight: Vec<f32> = (0..dims.shape.hidden)
        .map(|i| 0.5 + (i % 97) as f32 / 194.0)
        .collect();
    let layers: Vec<usize> = [
        crate::family::AttentionKind::DeltaNet,
        crate::family::AttentionKind::Full,
    ]
    .iter()
    .map(|&kind| {
        (0..table.len())
            .find(|&l| table.slot(l).expect("slot").kind == kind)
            .expect("a layer")
    })
    .collect();
    // The device phase for both layers first; every device owner drops before any assert.
    let mut runs = Vec::new();
    for layer in layers {
        let kind = table.slot(layer).expect("slot").kind;
        let mut sources: Vec<(Role, Vec<u8>)> = Vec::new();
        for spec in geometry
            .dense_specs(WeightLayout::Tiled, kind)
            .expect("specs")
        {
            sources.push((spec.role, {
                let mut rng = Rng::new(0x5eed_0000 + spec.rows as u64);
                super::native_device_rig::q4k_rows(spec.rows, spec.k, &mut rng)
            }));
        }
        let (tiled, native) = (
            run_block(&ctx, &stream, layer, WeightLayout::Tiled, &resid, &sources).expect("tiled"),
            run_block(&ctx, &stream, layer, WeightLayout::Native, &resid, &sources)
                .expect("native"),
        );
        runs.push((kind, sources, tiled, native));
    }
    drop(stream);
    drop(ctx);
    // Host data only from here (the isolation split).
    for (kind, sources, tiled, native) in runs {
        for run in [&tiled, &native] {
            assert_finite("resid_out", &run.resid_out);
            assert_finite("mix_out", &run.mix_out);
            assert_finite("proj", &run.proj);
        }
        // m8's killer, per side: the residual must have absorbed THIS side's own MixOut (the
        // epilogue's delta add) within rounding - a dropped delta (m8) leaves resid_out equal
        // to the input residual and fails on both layouts.
        for (name, run) in [("tiled", &tiled), ("native", &native)] {
            let mix_signal = run
                .mix_out
                .iter()
                .fold(0.0f64, |m, v| m.max(v.abs() as f64));
            assert!(mix_signal > 0.0, "{name}: MixOut must be non-zero");
            for (i, ((got, mix), input)) in run
                .resid_out
                .iter()
                .zip(&run.mix_out)
                .zip(&resid)
                .enumerate()
            {
                let want = *input as f64 + *mix as f64;
                let got = *got as f64;
                // The epilogue adds in f32: one rounding of the exact sum, at most half an f32
                // ulp (2^-24 relative); the slack is twice that. A dropped delta (m8) misses by
                // the whole |mix|.
                let slack = f32::EPSILON as f64 * want.abs().max(1.0);
                assert!(
                    (got - want).abs() <= slack,
                    "{name} resid {i}: {got} vs {want} (the delta add of the epilogue)"
                );
            }
        }
        // The input hops: every fused column within the derived single-hop bound with A from
        // the HOST prologue norm (AttnNorm of the rig's residual).
        let x = host_prologue_norm(&resid, &norm_weight);
        let a_in = x.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        let cols: Vec<(Role, usize)> = match kind {
            crate::family::AttentionKind::DeltaNet => vec![
                (Role::AttnQkv, 0),
                (Role::AttnGate, geometry.delta.z_off),
                (Role::SsmBeta, geometry.delta.beta_off),
                (Role::SsmAlpha, geometry.delta.alpha_off),
            ],
            _ => vec![
                (Role::AttnQ, 0),
                (Role::AttnK, geometry.attn.k_off),
                (Role::AttnV, geometry.attn.v_off),
            ],
        };
        // m5 (the native input pack skipped) zeroes every native projection, so the per-row
        // check below fails on any row whose tiled signal exceeds its own bound. The bound is a
        // worst case (l1 * A) and the signal a sum with random signs, so this is asserted per
        // row, not as max signal over max bound: some row must carry at least 3x its bound.
        let mut best_ratio = 0.0f64;
        for (role, col) in cols {
            let (rows, k) = geometry
                .dense_specs(WeightLayout::Tiled, kind)
                .expect("specs")
                .into_iter()
                .find(|s| s.role == role)
                .map(|s| (s.rows, s.k))
                .expect("a dense role");
            let source = &sources
                .iter()
                .find(|(r, _)| *r == role)
                .expect("a source")
                .1;
            for r in 0..rows {
                let diff = (tiled.proj[col + r] - native.proj[col + r]).abs() as f64;
                let bound = hop_bound(row_l1(source, r, k), a_in);
                assert!((diff <= bound), "{role:?} row {r}: {diff} > {bound}");
                best_ratio = best_ratio.max(tiled.proj[col + r].abs() as f64 / bound);
            }
        }
        assert!(
            best_ratio > 3.0,
            "{kind:?}: no projection row carries 3x its bound (best {best_ratio}); a zeroed \
             native pack (m5) could hide inside the input-hop bounds"
        );
        // The output hop: the mixer output row within BLOCK_HOPS x the single-hop bound with A
        // from the READ-BACK core row. A validity check only: with 4,096 inputs per row (value_width
        // and q_width), the worst-case bound exceeds the random-sign signal, so no dominance is
        // asserted here; m5 is killed at the input hop above.
        let (out_role, out_k) = match kind {
            crate::family::AttentionKind::DeltaNet => (Role::SsmOut, geometry.delta.value_width),
            _ => (Role::AttnOutput, geometry.attn.q_width),
        };
        let source = &sources
            .iter()
            .find(|(r, _)| *r == out_role)
            .expect("a source")
            .1;
        let a_core = tiled.core.iter().fold(0.0f64, |m, v| m.max(v.abs() as f64));
        for i in 0..dims.shape.hidden {
            let bound = BLOCK_HOPS * hop_bound(row_l1(source, i, out_k), a_core);
            let diff = (tiled.mix_out[i] - native.mix_out[i]).abs() as f64;
            assert!(diff <= bound, "mix_out {i}: {diff} > {bound}");
        }
    }
}
