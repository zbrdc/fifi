//! Explicit one-row artifact capture, not a CUDA graph or a numerical accuracy gate.
//!
//! `FIFI_EXL3_CHECKPOINT` selects the EXL3 checkpoint. `FIFI_EXL3_MIXER_CASE`
//! selects a directory containing exactly one hidden-size row in `input-f32.bin`.
//! `FIFI_EXL3_MIXER_OUT` must name a new directory. Files contain raw LE F32.
//! This runs only the first full-attention mixer at position zero with fresh FP16 KV.
//! It does not qualify MoE, a Transformer, a model, throughput, or batched execution.

use std::{
    fs::{self, File, OpenOptions},
    io::{Read, Write},
    path::Path,
};

use super::{
    common_tests::{dims, inventory},
    dims::{KvMode, StepDims, StepOptions},
    exl3_source,
    layer_table::LayerTable,
    mixer::{BlockIo, MixerParts, MixerViews, Mixers},
    mixer_geometry::MixerGeometry,
    mixer_weights::MixerWeights,
    native_device_rig::{assert_finite, device_present, f32_buf},
    resources::f32_bytes,
    state::{ResidentState, StepField},
    step_plan::{KernelScratch, Slot, StepPlan},
};
use crate::{
    EngError,
    family::{
        AttentionKind,
        project::cuda::{CudaProjector, Reserve, WeightLayout},
    },
    kernels::cuda::{ArenaTensor, DeviceRange, DeviceStream, resident::step_params::StepParams},
    weights::exl3::Checkpoint,
};
use serde_json::json;

/// Host data only; every device owner has dropped before the caller checks these values.
struct Capture {
    stages: Vec<(&'static str, Vec<u8>)>,
    exl3_flag: u32,
    step_status: u32,
}

/// Use the native rigs' stream-ordered readback contract, including nonzero arena offsets.
fn read_range(stream: &DeviceStream, range: DeviceRange<'_>) -> Result<Vec<u8>, EngError> {
    let mut bytes = vec![0; range.bytes()];
    range.buffer().read_at(stream, range.offset(), &mut bytes)?;
    Ok(bytes)
}

fn read_word(stream: &DeviceStream, range: DeviceRange<'_>) -> Result<u32, EngError> {
    let mut bytes = [0; 4];
    range.buffer().read_at(stream, range.offset(), &mut bytes)?;
    Ok(u32::from_le_bytes(bytes))
}

/// Device phase: fallible operations only, with no assertions or panic-based error handling.
fn capture(
    checkpoint: &Checkpoint,
    dims: &StepDims,
    table: &LayerTable,
    geometry: MixerGeometry,
    layer: usize,
    input: &[f32],
) -> Result<Capture, EngError> {
    let options = StepOptions {
        kv_mode: KvMode::Fp16,
        context_tokens: 1,
        delta_slots: 1,
        forked_experts: false,
        tiered: None,
        layout: WeightLayout::Exl3,
    };
    let scratch = KernelScratch::for_dims(dims, 1, options.kv_mode)?;
    let plan = StepPlan::exl3_mixers(dims, &options, &scratch)?;
    // Keep all intermediates disjoint until readback, as the native mixer rig does.
    // The normal plan reuses dead storage; widening lives changes no block operation.
    let planned = plan.arena_tensors();
    let end = planned.iter().map(|tensor| tensor.last).max().unwrap_or(0);
    let unaliased: Vec<ArenaTensor> = planned
        .into_iter()
        .map(|tensor| ArenaTensor {
            first: 0,
            last: end,
            ..tensor
        })
        .collect();
    let projector = CudaProjector::new()?;
    let ctx = projector.context();
    let stream = ctx.stream()?;
    let arena = ctx.arena(&unaliased)?;
    // Keep the captured F32 residual independent of the plan's legacy residual storage.
    let residual = f32_buf(ctx, input)?;
    let resid = residual.range(0, f32_bytes(geometry.hidden)?)?;
    let mut state = ResidentState::allocate(ctx, &stream, dims, &options)?;
    state.write_step_params(&stream, &StepParams::new(0, 0)?)?;
    let loaded = exl3_source::load_attention(
        &projector,
        checkpoint,
        table,
        layer,
        &geometry,
        Reserve::default(),
    )?;
    let weights = MixerWeights::new(&geometry, table, &loaded, &[layer], options.layout)?;
    let view =
        |slot| {
            arena.tensor(plan.index(slot).ok_or_else(|| {
                EngError::BadRequest(format!("EXL3 capture: missing slot {slot:?}"))
            })?)
        };
    let stages = [
        ("attn-proj-f32.bin", Slot::AttnProj),
        ("attn-q-f32.bin", Slot::AttnQ),
        ("attn-core-f32.bin", Slot::AttnCore),
        ("mix-out-f32.bin", Slot::MixOut),
        ("postnorm-f32.bin", Slot::NormRow),
    ];
    // A skipped producer must not accidentally expose finite uninitialized storage.
    for (_, slot) in stages {
        let range = view(slot)?;
        range
            .buffer()
            .write_at(&stream, range.offset(), &vec![0xff; range.bytes()])?;
    }
    let mixers = Mixers::new(
        ctx,
        MixerParts {
            geometry,
            table,
            weights,
            views: MixerViews::new(&plan, &arena, &geometry, options.layout, options.kv_mode)?,
            layout: options.layout,
            kv_mode: options.kv_mode,
            paged: None,
        },
    )?;
    // Match the existing native block rig: initialized params and run_block, no invented
    // transaction, epoch, token commit, or whole-step publication for this partial block.
    if let Err(error) = mixers.run_block(
        &stream,
        &state,
        layer,
        BlockIo {
            resid,
            admit_pos: 0,
        },
    ) {
        state.poison();
        stream.synchronize().map_err(|sync| {
            EngError::BadRequest(format!(
                "EXL3 capture block failed: {error:?}; synchronization also failed: {sync:?}"
            ))
        })?;
        return Err(error);
    }
    stream.synchronize().inspect_err(|_| state.poison())?;
    let exl3_flag = read_word(&stream, view(Slot::Exl3Flag)?)?;
    let step_status = read_word(&stream, state.step_field(StepField::Status)?)?;
    if exl3_flag != 0 || step_status != 0 {
        state.poison();
    }
    let mut outputs = Vec::new();
    for (name, slot) in stages {
        outputs.push((name, read_range(&stream, view(slot)?)?));
    }
    outputs.push(("residual-out-f32.bin", read_range(&stream, resid)?));
    // These eager, single-stream borrows retain loaded weights, arena, residual and state
    // through the final ordered read. No graph pin or leaked device owner is needed.
    Ok(Capture {
        stages: outputs,
        exl3_flag,
        step_status,
    })
}

fn write_new(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)?
        .write_all(bytes)
}

fn write_manifest(output: &Path, bytes: &[u8]) -> std::io::Result<()> {
    let mut temporary = tempfile::NamedTempFile::new_in(output)?;
    temporary.write_all(bytes)?;
    temporary
        .persist_noclobber(output.join("manifest.json"))
        .map_err(|error| error.error)?;
    Ok(())
}

#[test]
fn manifest_publication_refuses_missing_parent_and_overwrite() {
    let output = tempfile::tempdir().expect("capture directory");
    let missing = output.path().join("missing");
    assert!(write_manifest(&missing, b"{}").is_err());
    assert!(!missing.exists());
    write_manifest(output.path(), b"{\"complete\":true}").expect("publish manifest");
    assert!(write_manifest(output.path(), b"replacement").is_err());
    assert_eq!(
        fs::read(output.path().join("manifest.json")).expect("original manifest"),
        b"{\"complete\":true}"
    );
    assert_eq!(
        fs::read_dir(output.path()).expect("capture files").count(),
        1
    );
}

#[cfg(target_os = "linux")]
#[test]
fn manifest_publication_leaves_no_marker_after_a_short_write() {
    const CHILD_DIR: &str = "FIFI_EXL3_SHORT_WRITE_CHILD_DIR";
    const COMPLETE: &str = "EXL3_SHORT_WRITE_CHECK_COMPLETE";
    if let Some(output) = std::env::var_os(CHILD_DIR) {
        let output = Path::new(&output);
        let error = write_manifest(output, &[b'{'; 4096]).expect_err("file size limit");
        assert_eq!(error.kind(), std::io::ErrorKind::FileTooLarge);
        assert!(
            !output.join("manifest.json").exists(),
            "failed write exposed manifest: {:?}",
            fs::metadata(output.join("manifest.json")).map(|m| m.len())
        );
        assert_eq!(fs::read_dir(output).expect("capture files").count(), 0);
        println!("{COMPLETE}");
        return;
    }
    let output = tempfile::tempdir().expect("capture directory");
    let name = std::thread::current().name().expect("test name").to_owned();
    // Limit only the fresh child; SIGXFSZ must become a recoverable write error.
    let child = std::process::Command::new("/bin/sh")
        .args([
            "-c",
            "trap '' XFSZ; ulimit -f 1; exec \"$@\"",
            "exl3-short-write",
        ])
        .arg(std::env::current_exe().expect("libtest executable"))
        .args(["--exact", &name, "--nocapture", "--test-threads=1"])
        .env(CHILD_DIR, output.path())
        .output()
        .expect("isolated file-size-limited test");
    assert!(child.status.success(), "{child:?}");
    assert!(String::from_utf8_lossy(&child.stdout).contains(COMPLETE));
}

#[test]
#[ignore = "requires explicit GPU admission, EXL3 checkpoint, input row and new capture directory"]
fn real_exl3_attention_mixer_capture() {
    let checkpoint_path = std::env::var_os("FIFI_EXL3_CHECKPOINT").expect("FIFI_EXL3_CHECKPOINT");
    let case_path = std::env::var_os("FIFI_EXL3_MIXER_CASE").expect("FIFI_EXL3_MIXER_CASE");
    let output_path = std::env::var_os("FIFI_EXL3_MIXER_OUT").expect("FIFI_EXL3_MIXER_OUT");
    let output = Path::new(&output_path);
    let dims = dims();
    let inventory = inventory();
    let table = LayerTable::new(&dims, &inventory).expect("layer table");
    let geometry = MixerGeometry::new(&dims).expect("mixer geometry");
    let mut layer = None;
    for at in 0..table.len() {
        if table.slot(at).expect("layer slot").kind == AttentionKind::Full {
            layer = Some(at);
            break;
        }
    }
    let layer = layer.expect("first full-attention layer");
    let expected = f32_bytes(geometry.hidden).expect("input row byte count");
    let mut input_bytes = Vec::new();
    File::open(Path::new(&case_path).join("input-f32.bin"))
        .expect("open input-f32.bin")
        .take(u64::try_from(expected).expect("input byte limit") + 1)
        .read_to_end(&mut input_bytes)
        .expect("read bounded input row");
    assert_eq!(
        input_bytes.len(),
        expected,
        "input must contain exactly one hidden-size F32 row"
    );
    let input: Vec<_> = input_bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect();
    assert_finite("input-f32.bin", &input);
    // create_dir refuses existing files, directories and symlinks. Never reuse a capture.
    fs::create_dir(output).expect("output directory must be new and its parent must exist");
    let checkpoint = Checkpoint::open(Path::new(&checkpoint_path)).expect("EXL3 checkpoint");
    assert!(
        device_present("real_exl3_attention_mixer_capture"),
        "CUDA device is required"
    );
    let result =
        capture(&checkpoint, &dims, &table, geometry, layer, &input).expect("capture mixer");
    // Isolation split: no device owner remains when these host assertions execute.
    assert_eq!(result.exl3_flag, 0, "EXL3 arithmetic flag");
    assert_eq!(result.step_status, 0, "resident step status");
    let mut stage_manifest = Vec::new();
    for (name, bytes) in &result.stages {
        assert!(
            bytes.len().is_multiple_of(4),
            "{name}: incomplete F32 storage"
        );
        let values: Vec<_> = bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect();
        let expected_values = match *name {
            "attn-proj-f32.bin" => geometry.attn.ld,
            "attn-q-f32.bin" | "attn-core-f32.bin" => geometry.attn.q_width,
            _ => geometry.hidden,
        };
        assert_eq!(values.len(), expected_values, "{name}: stage length");
        assert_finite(name, &values);
        let meaning = match *name {
            "attn-proj-f32.bin" => "raw Q/gate, K and V projections before head norm and RoPE",
            "attn-q-f32.bin" => "Q after head norm and RoPE",
            "attn-core-f32.bin" => "attention result after sigmoid output gate",
            "mix-out-f32.bin" => "attention output projection before residual add",
            "postnorm-f32.bin" => "post-attention norm of the updated residual",
            _ => "updated F32 residual after attention output add",
        };
        stage_manifest
            .push(json!({"file": name, "shape": [1, expected_values], "meaning": meaning}));
    }
    write_new(&output.join("input-f32.bin"), &input_bytes).expect("write input copy");
    for (name, bytes) in &result.stages {
        write_new(&output.join(name), bytes).expect("write stage");
    }
    let manifest = json!({
        "scope": "capture/finite/flag checks only; no numerical accuracy PASS",
        "checkpoint": Path::new(&checkpoint_path),
        "case": Path::new(&case_path),
        "layer": layer, "kind": "full_attention", "rows": 1, "position": 0,
        "kv_len": 1, "kv_mode": "Fp16", "fresh_state": true,
        "dtype": "F32", "byte_order": "little", "input": "input-f32.bin",
        "geometry": {
            "hidden": geometry.hidden, "eps": geometry.eps,
            "heads_q": geometry.attn.heads_q, "heads_kv": geometry.attn.heads_kv,
            "head_dim": geometry.attn.head_dim, "n_rot": geometry.attn.n_rot,
            "rope_base": geometry.attn.rope_base, "scale": geometry.attn.scale,
            "q_rows": geometry.attn.q_rows, "kv_rows": geometry.attn.kv_rows,
            "k_off": geometry.attn.k_off, "v_off": geometry.attn.v_off,
            "projection_ld": geometry.attn.ld, "q_width": geometry.attn.q_width
        },
        "projection_layout": "per query head [query | gate], then K, then V",
        "norm_weights": "source converted to F32, then 1 added in F32",
        "postnorm": "RMSNorm(residual_input + mix_out) * effective post_attention_layernorm",
        "exl3_flag": result.exl3_flag, "step_status": result.step_status,
        "stages": stage_manifest
    });
    // Publish the completion marker only after its bytes are complete, without clobbering.
    write_manifest(
        output,
        &serde_json::to_vec_pretty(&manifest).expect("manifest JSON"),
    )
    .expect("write manifest");
}
