//! Test-only DeltaNet boundary reads. Recomputations reuse Q8 scratch when selected;
//! the live core and mixer output are checked for preservation.
use super::*;
use crate::kernels::cuda::DeviceBuffer;

pub(super) struct Reference {
    input: Vec<f32>,
    core: Vec<f32>,
    output: Vec<f32>,
}

impl Reference {
    pub(super) fn input(&self) -> &[f32] {
        &self.input
    }
    pub(super) fn core(&self) -> &[f32] {
        &self.core
    }
    pub(super) fn output(&self) -> &[f32] {
        &self.output
    }

    pub(super) fn load(layer: usize, capture: &Path) -> Self {
        let text = std::fs::read_to_string(capture).unwrap();
        let rows: Vec<serde_json::Value> = text
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect();
        let row = |suffix: &str| {
            let name = format!("blk.{layer}.{suffix}");
            let mut found = rows.iter().filter(|row| row["name"] == name);
            let first = found.next().unwrap_or_else(|| panic!("missing {name}"));
            assert!(found.next().is_none(), "duplicate {name}");
            first
        };
        let floats = |value: &serde_json::Value, field: &str| -> Vec<f32> {
            serde_json::from_value(value[field].clone()).unwrap()
        };
        let input = floats(row("attn_qkv.weight"), "input");
        assert_eq!(input, floats(row("attn_gate.weight"), "input"));
        let out = row("ssm_out.weight");
        Self {
            input,
            core: floats(out, "input"),
            output: floats(out, "output"),
        }
    }
}

#[test]
#[should_panic(expected = "missing blk.0.attn_qkv.weight")]
fn missing_delta_input_is_rejected() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("missing.jsonl");
    std::fs::write(&path, "{\"name\":\"blk.0.attn_gate.weight\"}\n").unwrap();
    let _ = Reference::load(0, &path);
}

fn report(position: usize, layer: usize, stage: &str, actual: &[f32], expected: &[f32]) {
    let error = relative_l2(actual, expected);
    let maximum = max_abs(actual, expected);
    assert!(error.is_finite() && maximum.is_finite());
    eprintln!(
        "DeltaNet diagnostic: position={position} layer={layer} stage={stage} relative_l2={error:.9} max_abs={maximum:.9}"
    );
}

pub(super) fn before(step: &FlashStep<'_>, reference: &Reference, position: usize, layer: usize) {
    assert_eq!(reference.input.len(), step.geometry.hidden);
    let input = step
        .read_f32(
            step.slot(Slot::Mixed(Site::Attn)).unwrap(),
            reference.input.len(),
        )
        .unwrap();
    report(position, layer, "mixer_input", &input, &reference.input);
}

fn benchmark_projection(step: &FlashStep<'_>, name: &str, core: &DeviceBuffer, host: &[f32]) {
    use crate::{
        family::{
            qwen::flash_next::cuda::step::{project_q6_f16, project_q6_f32},
            step::rows::NativeRow,
        },
        kernels::cuda::resident::quantize_q8_1::q8_1_bytes,
    };
    let Some(CudaWeight::Native { weight, .. }) = step.weights.get(name) else {
        panic!("benchmark requires the native DeltaNet output weight");
    };
    let q8 = step
        .ctx
        .allocate(q8_1_bytes(weight.k(), 1).unwrap())
        .unwrap();
    let output = step.ctx.allocate(host.len() * 4).unwrap();
    let launch = |mode| {
        let call = NativeRow {
            weight,
            input: core.range(0, core.bytes()).unwrap(),
            output: output.range(0, output.bytes()).unwrap(),
        };
        match mode {
            "q8" => call
                .quantized(&step.ctx, &step.stream, q8.range(0, q8.bytes()).unwrap())
                .unwrap(),
            "f16" => project_q6_f16(&step.ctx, &step.stream, call).unwrap(),
            "f32" => project_q6_f32(&step.ctx, &step.stream, call).unwrap(),
            _ => unreachable!(),
        }
    };
    for mode in ["q8", "f16", "f32", "f32", "f16", "q8"] {
        for _ in 0..5 {
            launch(mode);
        }
        let start = step.ctx.event_timing().unwrap();
        let end = step.ctx.event_timing().unwrap();
        step.stream.record_timing(&start).unwrap();
        for _ in 0..100 {
            launch(mode);
        }
        step.stream.record_timing(&end).unwrap();
        step.stream.synchronize().unwrap();
        let ms_per_call = start.elapsed_ms_since(&end).unwrap() / 100.0;
        let actual = step
            .read_f32(output.range(0, output.bytes()).unwrap(), host.len())
            .unwrap();
        let error = relative_l2(&actual, host);
        let bound = match mode {
            "q8" => 0.03,
            "f16" => 0.001,
            "f32" => 0.00001,
            _ => unreachable!(),
        };
        assert!(error < bound);
        eprintln!(
            "DeltaNet projection benchmark: mode={mode} rows={} k={} repeat=100 ms_per_call={ms_per_call:.6} host_relative_l2={error:.9}",
            weight.n_rows(),
            weight.k()
        );
    }
}

pub(super) fn after(step: &FlashStep<'_>, reference: &Reference, position: usize, layer: usize) {
    let delta = step.geometry.delta;
    assert_eq!(reference.core.len(), delta.value_width());
    assert_eq!(reference.output.len(), step.geometry.hidden);
    let name = layer_name(layer, "linear_attn.out_proj");
    let tiled = matches!(step.weights.get(&name), Some(CudaWeight::Native { weight, .. }) if !delta.value_dim.is_multiple_of(weight.quant().block_size()));
    let out_slot = step.slot(Slot::BlockOut(Site::Attn)).unwrap();
    let output = step.read_f32(out_slot, reference.output.len()).unwrap();
    report(
        position,
        layer,
        "actual_mixer_output",
        &output,
        &reference.output,
    );
    if tiled {
        let core_slot = step.slot(Slot::GgufDeltaTile).unwrap();
        let core = step.read_f32(core_slot, reference.core.len()).unwrap();
        report(position, layer, "actual_tiled_core", &core, &reference.core);
        // Recompute into separate allocations, then verify the live core and output survived.
        let bytes: Vec<_> = reference
            .core
            .iter()
            .flat_map(|v| v.to_ne_bytes())
            .collect();
        let host_core = step.ctx.upload(&bytes).unwrap();
        let result = step.ctx.allocate(reference.output.len() * 4).unwrap();
        step.project(
            &name,
            host_core.range(0, bytes.len()).unwrap(),
            result.range(0, result.bytes()).unwrap(),
            1,
        )
        .unwrap();
        let recomputed = step
            .read_f32(
                result.range(0, result.bytes()).unwrap(),
                reference.output.len(),
            )
            .unwrap();
        report(
            position,
            layer,
            "host_core_recomputed_output",
            &recomputed,
            &reference.output,
        );
        if (position, layer) == (1, 0)
            && std::env::var_os("FIFI_FLASH_GGUF_TEST_DELTA_BENCH").is_some()
        {
            benchmark_projection(step, &name, &host_core, &reference.output);
        }
        if (position, layer) == (1, 0)
            && let Some(path) = std::env::var_os("FIFI_FLASH_GGUF_Q8_DIAG_OUT")
        {
            use std::io::Write;
            let mut file = std::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(path)
                .unwrap();
            for value in &recomputed {
                file.write_all(&value.to_le_bytes()).unwrap();
            }
            file.flush().unwrap();
        }
        assert_eq!(
            step.read_f32(core_slot, reference.core.len()).unwrap(),
            core,
            "diagnostic changed the live DeltaNet core"
        );
        assert_eq!(
            step.read_f32(out_slot, reference.output.len()).unwrap(),
            output,
            "diagnostic changed the live mixer output"
        );
    } else {
        eprintln!(
            "DeltaNet diagnostic: position={position} layer={layer} stage=core skipped (native weight not in tiled-input form)"
        );
    }
}
