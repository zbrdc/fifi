//! Test-only QSA boundary measurements. Recomputed projections never replace live state.
use super::*;
use crate::kernels::cuda::DeviceRange;

pub(super) struct Reference {
    input: Vec<f32>,
    projections: Vec<Vec<f32>>,
    core: Vec<f32>,
    output: Vec<f32>,
}

impl Reference {
    pub(super) fn load(layer: usize, capture: &Path) -> Self {
        let text = std::fs::read_to_string(capture).unwrap();
        let rows: Vec<serde_json::Value> = text
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect();
        let row = |suffix: &str| {
            let name = format!("blk.{layer}.{suffix}");
            let mut found = rows.iter().filter(|row| row["name"] == name);
            let first = found.next();
            assert!(found.next().is_none(), "duplicate {name}");
            first
        };
        let q = row("attn_q.weight").expect("QSA query capture missing");
        let floats = |value: &serde_json::Value, field: &str| -> Vec<f32> {
            serde_json::from_value(value[field].clone()).unwrap()
        };
        let input = floats(q, "input");
        let mut projections = vec![floats(q, "output")];
        for name in [
            "attn_k.weight",
            "attn_v.weight",
            "indexer.q_proj.weight",
            "indexer.k_proj.weight",
        ] {
            let value = row(name).expect("QSA capture is incomplete");
            assert_eq!(
                input,
                floats(value, "input"),
                "QSA inputs differ for {name}"
            );
            projections.push(floats(value, "output"));
        }
        let output = row("attn_output.weight").expect("QSA output capture missing");
        Self {
            input,
            projections,
            core: floats(output, "input"),
            output: floats(output, "output"),
        }
    }
}

#[test]
#[should_panic(expected = "QSA query capture missing")]
fn missing_qsa_query_is_rejected() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("missing.jsonl");
    std::fs::write(&path, "{\"name\":\"blk.3.attn_k.weight\"}\n").unwrap();
    let _ = Reference::load(3, &path);
}

fn report(position: usize, layer: usize, stage: &str, actual: &[f32], expected: &[f32]) {
    let error = relative_l2(actual, expected);
    let maximum = max_abs(actual, expected);
    assert!(error.is_finite() && maximum.is_finite());
    eprintln!(
        "QSA diagnostic: position={position} layer={layer} stage={stage} relative_l2={error:.9} max_abs={maximum:.9}"
    );
}

/// Same decode projections as QSA, using a diagnostic-only output buffer.
fn projections(step: &FlashStep<'_>, layer: usize, x: DeviceRange<'_>) -> Vec<f32> {
    let qsa = step.geometry.qsa;
    let q = 2 * qsa.q_width();
    let k = qsa.kv_width();
    let index = q + 2 * k;
    let ld = qsa.proj_ld();
    assert_eq!(index + qsa.idx_proj(), ld);
    let buffer = step.ctx.allocate(ld * 4).unwrap();
    let row = buffer.range(0, ld * 4).unwrap();
    let at = |suffix: &str| layer_name(layer, suffix);
    step.project_cols(&at("self_attn.q_proj"), x, row, 0, q, ld, 1)
        .unwrap();
    step.project_group(
        &[
            (&at("self_attn.k_proj"), q, k),
            (&at("self_attn.v_proj"), q + k, k),
        ],
        x,
        &row,
        1,
    )
    .unwrap();
    step.project_cols(
        &at("self_attn.indexer.index_qk_proj"),
        x,
        row,
        index,
        qsa.idx_proj(),
        ld,
        1,
    )
    .unwrap();
    step.read_f32(row, ld).unwrap()
}

fn projection_errors(
    step: &FlashStep<'_>,
    reference: &Reference,
    label: (usize, usize, &str),
    actual: &[f32],
) {
    let (position, layer, kind) = label;
    let qsa = step.geometry.qsa;
    let widths = [
        2 * qsa.q_width(),
        qsa.kv_width(),
        qsa.kv_width(),
        qsa.idx_heads * qsa.idx_dim,
        qsa.idx_dim,
    ];
    let names = ["q", "k", "v", "index_q", "index_k"];
    assert_eq!(actual.len(), qsa.proj_ld());
    let mut offset = 0;
    for ((name, width), want) in names.into_iter().zip(widths).zip(&reference.projections) {
        assert_eq!(want.len(), width, "QSA {name} metadata width");
        report(
            position,
            layer,
            &format!("{kind}_{name}"),
            &actual[offset..offset + width],
            want,
        );
        offset += width;
    }
    assert_eq!(offset, actual.len());
}

/// Recompute raw projections with actual and host inputs; neither changes the live input/state.
pub(super) fn before(step: &FlashStep<'_>, reference: &Reference, position: usize, layer: usize) {
    let x = step.slot(Slot::Mixed(Site::Attn)).unwrap();
    assert_eq!(reference.input.len(), step.geometry.hidden);
    let actual = step.read_f32(x, reference.input.len()).unwrap();
    report(position, layer, "mixer_input", &actual, &reference.input);
    let recomputed = projections(step, layer, x);
    projection_errors(
        step,
        reference,
        (position, layer, "actual_input_recomputed"),
        &recomputed,
    );
    let bytes: Vec<_> = reference
        .input
        .iter()
        .flat_map(|v| v.to_ne_bytes())
        .collect();
    let forced = step.ctx.upload(&bytes).unwrap();
    let forced = forced.range(0, bytes.len()).unwrap();
    let recomputed_host = projections(step, layer, forced);
    projection_errors(
        step,
        reference,
        (position, layer, "host_input_recomputed"),
        &recomputed_host,
    );
    assert_eq!(
        step.read_f32(x, reference.input.len()).unwrap(),
        actual,
        "diagnostic changed the live mixer input"
    );
}

/// Capture the surviving core and block output before the hyper-connection consumes them.
pub(super) fn after(step: &FlashStep<'_>, reference: &Reference, position: usize, layer: usize) {
    assert_eq!(reference.core.len(), step.geometry.qsa.q_width());
    assert_eq!(reference.output.len(), step.geometry.hidden);
    let core = step
        .read_f32(step.slot(Slot::AttnCore).unwrap(), reference.core.len())
        .unwrap();
    let output = step
        .read_f32(
            step.slot(Slot::BlockOut(Site::Attn)).unwrap(),
            reference.output.len(),
        )
        .unwrap();
    report(position, layer, "actual_core", &core, &reference.core);
    report(
        position,
        layer,
        "actual_mixer_output",
        &output,
        &reference.output,
    );
}

/// Execute one QSA mixer on the supplied input and this step's existing cache.
fn matched_history_step(
    step: &FlashStep<'_>,
    position: usize,
    token: u32,
    input: &[f32],
    reference: Option<&Reference>,
) -> (Vec<f32>, Vec<f32>, Vec<u8>) {
    step.begin_token().unwrap();
    let (words, offset) = step.words_buffer().unwrap();
    words
        .write_at(
            &step.stream,
            offset,
            &StepParams::new(token as i32, position as i32)
                .unwrap()
                .to_bytes(),
        )
        .unwrap();
    let mixed = step.slot(Slot::Mixed(Site::Attn)).unwrap();
    assert_eq!(input.len(), step.geometry.hidden);
    let bytes: Vec<_> = input.iter().flat_map(|value| value.to_ne_bytes()).collect();
    mixed
        .buffer()
        .write_at(&step.stream, mixed.offset(), &bytes)
        .unwrap();
    assert_eq!(
        step.read_f32(mixed, input.len()).unwrap(),
        input,
        "mixer input changed on upload"
    );
    if let Some(reference) = reference {
        assert_eq!(input, reference.input);
        before(step, reference, position, 3);
    }
    step.mixer(3, position, 1).unwrap();
    if let Some(reference) = reference {
        after(step, reference, position, 3);
    }
    let core = step
        .read_f32(
            step.slot(Slot::AttnCore).unwrap(),
            step.geometry.qsa.q_width(),
        )
        .unwrap();
    let output = step
        .read_f32(step.slot(Slot::BlockOut(Site::Attn)).unwrap(), input.len())
        .unwrap();
    assert!(core.iter().chain(&output).all(|v| v.is_finite()));
    let scratch = step.slot(Slot::GgufDecodeQ8).unwrap();
    let q8_len = crate::kernels::cuda::resident::quantize_q8_1::q8_1_bytes(core.len(), 1).unwrap();
    let mut q8 = vec![0; q8_len];
    scratch
        .buffer()
        .read_at(&step.stream, scratch.offset(), &mut q8)
        .unwrap();
    step.finish_token().unwrap();
    (core, output, q8)
}

/// Independently project the captured host core without modifying the QSA cache.
fn host_core_projection(step: &FlashStep<'_>, core: &[f32]) -> serde_json::Value {
    use crate::family::{qwen::flash_next::cuda::step::project_q6_f32, step::rows::NativeRow};
    let name = layer_name(3, "self_attn.o_proj");
    let Some(CudaWeight::Native { weight, .. }) = step.weights.get(&name) else {
        panic!("missing native QSA output weight");
    };
    assert_eq!(weight.k(), core.len());
    let input = step
        .ctx
        .upload(
            &core
                .iter()
                .flat_map(|v| v.to_ne_bytes())
                .collect::<Vec<_>>(),
        )
        .unwrap();
    let output = step.ctx.allocate(weight.n_rows() * 4).unwrap();
    let x = input.range(0, input.bytes()).unwrap();
    let y = output.range(0, output.bytes()).unwrap();
    let scratch = step.slot(Slot::GgufDecodeQ8).unwrap();
    step.project(&name, x, y, 1).unwrap();
    let q8_output = step.read_f32(y, weight.n_rows()).unwrap();
    let mut q8 =
        vec![0; crate::kernels::cuda::resident::quantize_q8_1::q8_1_bytes(core.len(), 1).unwrap()];
    scratch
        .buffer()
        .read_at(&step.stream, scratch.offset(), &mut q8)
        .unwrap();
    project_q6_f32(
        &step.ctx,
        &step.stream,
        NativeRow {
            weight,
            input: x,
            output: y,
        },
    )
    .unwrap();
    let f32_output = step.read_f32(y, weight.n_rows()).unwrap();
    assert!(q8_output.iter().chain(&f32_output).all(|v| v.is_finite()));
    serde_json::json!({"q8":q8_output,"q8_bytes":q8,"f32":f32_output})
}

#[test]
#[ignore = "requires pinned GGUF and three-position host fixtures and an admitted CUDA device"]
fn pinned_layer_three_qsa_matched_input_history() {
    let path = std::env::var_os("FIFI_FLASH_GGUF").expect("verified first shard");
    assert_pinned_file(
        &path,
        "9d5644b371ffac921427c8021cc7242dab8e3ba37d7b48d1e79fe3770fbf9915",
    );
    let output_path = std::env::var_os("FIFI_FLASH_GGUF_QSA_MATCHED_OUT").expect("new output path");
    let (captures, _) = prompt_captures();
    let references: Vec<_> = captures
        .iter()
        .map(|pos| Reference::load(3, &pos[3]))
        .collect();
    let model = GgufModel::open(Path::new(&path)).unwrap().unwrap();
    assert!(model.config().kinds()[3], "fixture requires QSA layer 3");
    let ctx = CudaContext::new().unwrap();
    let stream = ctx.stream().unwrap();
    let weights =
        CudaWeights::from_native(model.upload_layer_trunk(3, &ctx, &stream).unwrap()).unwrap();
    let shape = model.config().model_shape().unwrap();
    let max_k = model.map().max_quantized_input_width().unwrap();
    let step = FlashStep::new_gguf_decode(
        &ctx,
        &stream,
        &weights,
        shape,
        model.config().kinds(),
        3,
        max_k,
        3..4,
    )
    .unwrap();
    let control = FlashStep::new_gguf_decode(
        &ctx,
        &stream,
        &weights,
        shape,
        model.config().kinds(),
        3,
        max_k,
        3..4,
    )
    .unwrap();
    let mut rows = Vec::new();
    for (position, token) in [9707u32, 11, 1879].into_iter().enumerate() {
        let reference = &references[position];
        let (core, output, q8) =
            matched_history_step(&step, position, token, &reference.input, Some(reference));
        let core_error = relative_l2(&core, &reference.core);
        let output_error = relative_l2(&output, &reference.output);
        assert!(core_error.is_finite() && output_error.is_finite());
        eprintln!(
            "matched host-input QSA history: position={position} core_l2={core_error:.9} output_l2={output_error:.9}; diagnostic only"
        );
        rows.push(
            serde_json::json!({"position":position,"token":token,"core":core,"output":output,
            "core_q8_bytes":q8,"host_core":reference.core,"host_output":reference.output,
            "core_relative_l2":core_error,"output_relative_l2":output_error}),
        );
    }
    let zeros = vec![0.0; references[0].input.len()];
    let _ = matched_history_step(&control, 0, 9707, &zeros, None);
    let (_, control_output, _) = matched_history_step(&control, 1, 11, &references[1].input, None);
    let matched_output: Vec<f32> = serde_json::from_value(rows[1]["output"].clone()).unwrap();
    let history_effect = relative_l2(&control_output, &matched_output);
    assert!(
        history_effect.is_finite() && history_effect > 1e-5,
        "zero-history control failed to distinguish cache contents"
    );
    eprintln!(
        "matched QSA zero-history negative control: position=1 output_relative_l2={history_effect:.9}"
    );
    let host_core_controls: Vec<_> = references
        .iter()
        .map(|reference| host_core_projection(&step, &reference.core))
        .collect();
    let receipt = serde_json::json!({"layer":3,"history":"device_generated_from_host_inputs",
        "positions":rows,"host_core_controls":host_core_controls,"zero_history_position1":control_output,
        "zero_history_relative_l2":history_effect});
    let file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(output_path)
        .unwrap();
    serde_json::to_writer(file, &receipt).unwrap();
    eprintln!(
        "matched QSA history diagnostic complete; not propagated execution or parity qualification"
    );
}
