//! Real-checkpoint tests of the resident decode step.
use super::{Gemma, VOCAB};

/// Loads the pinned checkpoint and runs one decode step.
#[test]
#[ignore = "requires the pinned Humanizer file and a device"]
fn decodes_one_token_against_the_pinned_checkpoint() {
    let Ok(path) = std::env::var("GEMMA4_Q8_FILE") else {
        return;
    };
    let header = std::env::var("GEMMA4_Q8_HEADER_SHA256").ok();
    let load_start = std::time::Instant::now();
    let mut model: Gemma =
        super::load(std::path::Path::new(&path), header.as_deref()).expect("load");
    eprintln!("load_seconds={:.3}", load_start.elapsed().as_secs_f64());
    let step_start = std::time::Instant::now();
    model.decode(2, 0).expect("decode");
    eprintln!("decode_seconds={:.3}", step_start.elapsed().as_secs_f64());
    let mut logits = vec![0u8; VOCAB * 4];
    model
        .ctx
        .download(&model.logits_capped.buf, &mut logits)
        .expect("read logits");
    let values: Vec<f32> = logits
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect();
    assert!(values.iter().all(|v| v.is_finite()), "non-finite logits");
    let peak = values
        .iter()
        .fold(0.0f32, |acc, v| if v.abs() > acc { v.abs() } else { acc });
    assert!(peak <= 30.0 + 1e-3, "softcap exceeded: {peak}");
}

/// Measures the decode matvec against the pinned gate matrix: one 15,360-row
/// width read per launch. Set GEMMA4_BANDWIDTH_PROBE to the launch count.
#[test]
#[ignore = "requires the pinned Humanizer file, a device and GEMMA4_BANDWIDTH_PROBE"]
fn measures_matvec_bandwidth() {
    use crate::kernels::cuda::resident::matvec_q::{MatvecCall, QuantType};
    let Ok(path) = std::env::var("GEMMA4_Q8_FILE") else {
        return;
    };
    let Ok(reps) = std::env::var("GEMMA4_BANDWIDTH_PROBE") else {
        return;
    };
    let reps: usize = reps.parse().expect("probe count");
    let header = std::env::var("GEMMA4_Q8_HEADER_SHA256").ok();
    let model: Gemma = super::load(std::path::Path::new(&path), header.as_deref()).expect("load");
    // The embedding is the one matrix that stays in the checkpoint's native Q8_0 layout,
    // so it is what the vendored matvec probe can measure.
    let embed = &model.embed;
    let out = &model.logits;
    let q8 = &model.scratch.q8_hidden;
    let started = std::time::Instant::now();
    for _ in 0..reps {
        model
            .ctx
            .matvec_q(
                &model.stream,
                MatvecCall {
                    ty: QuantType::Q80,
                    w: embed.buf.range(0, embed.bytes()).expect("embed"),
                    x_q8: q8.range(0, q8.bytes()).expect("q8"),
                    y: out.buf.range(0, out.bytes()).expect("out"),
                    k: embed.k,
                    rows: embed.rows,
                    ncols: 1,
                },
            )
            .expect("matvec");
    }
    model.stream.synchronize().expect("sync");
    let seconds = started.elapsed().as_secs_f64();
    let bytes = reps * embed.bytes();
    eprintln!(
        "bandwidth reps={reps} seconds={seconds:.3} gb_per_s={:.1}",
        bytes as f64 / seconds / 1e9
    );
}

/// Measures the matvec at an arbitrary shape, so a wide-k shape and a Mellum-like
/// 2304-wide shape can be compared at equal bytes. Set GEMMA4_PROBE_K,
/// GEMMA4_PROBE_ROWS and GEMMA4_BANDWIDTH_PROBE.
#[test]
#[ignore = "requires a device and GEMMA4_PROBE_K / GEMMA4_PROBE_ROWS"]
fn measures_matvec_bandwidth_at_a_shape() {
    use crate::kernels::cuda::resident::matvec_q::{MatvecCall, QuantType};
    use crate::kernels::cuda::resident::quantize_q8_1::q8_1_bytes;
    let (Ok(k), Ok(rows), Ok(reps)) = (
        std::env::var("GEMMA4_PROBE_K"),
        std::env::var("GEMMA4_PROBE_ROWS"),
        std::env::var("GEMMA4_BANDWIDTH_PROBE"),
    ) else {
        return;
    };
    let k: usize = k.parse().expect("k");
    let rows: usize = rows.parse().expect("rows");
    let reps: usize = reps.parse().expect("reps");
    let ctx = crate::kernels::cuda::CudaContext::with_admission(
        &crate::family::project::cuda::served_admission(),
    )
    .expect("context");
    let stream = ctx.stream().expect("stream");
    let weight_bytes = rows * (k / 32) * 34;
    let w = ctx.upload(&vec![0u8; weight_bytes]).expect("weights");
    let q8 = ctx
        .upload(&vec![0u8; q8_1_bytes(k, 1).expect("q8 size")])
        .expect("q8");
    let out = ctx.upload(&vec![0u8; rows * 4]).expect("out");
    let started = std::time::Instant::now();
    for _ in 0..reps {
        ctx.matvec_q(
            &stream,
            MatvecCall {
                ty: QuantType::Q80,
                w: w.range(0, weight_bytes).expect("w range"),
                x_q8: q8
                    .range(0, q8_1_bytes(k, 1).expect("q8 size"))
                    .expect("q8 range"),
                y: out.range(0, rows * 4).expect("y range"),
                k,
                rows,
                ncols: 1,
            },
        )
        .expect("matvec");
    }
    stream.synchronize().expect("sync");
    let seconds = started.elapsed().as_secs_f64();
    let bytes = reps * weight_bytes;
    eprintln!(
        "shape k={k} rows={rows} reps={reps} mb={:.1} seconds={seconds:.3} gb_per_s={:.1}",
        weight_bytes as f64 * reps as f64 / 1e6,
        bytes as f64 / seconds / 1e9
    );
}

/// The residual rotation must feed every layer its predecessor finished output.
/// The earlier `(current + 1) % 4` advance broke exactly this and no test noticed.
#[test]
fn each_layer_reads_the_previous_layers_output() {
    let mut current = 1usize;
    for layer in 0..48 {
        let (input, r1, r2, out) = super::step::layer_buffers(current);
        assert_eq!(input, current);
        assert_ne!(out, input);
        assert!(r1 != input && r2 != input && r1 != out && r2 != out);
        current = out;
        let (next_input, _, _, _) = super::step::layer_buffers(current);
        assert_eq!(next_input, out, "layer {layer} output is not read next");
    }
}

/// This trace downloads five ids and five log probabilities, never a logits row.
#[test]
#[ignore = "requires the pinned Humanizer file and an exclusive device lease"]
fn traces_first_position_top_five() {
    let path = std::env::var("GEMMA4_Q8_FILE").expect("checkpoint path required");
    let header = std::env::var("GEMMA4_Q8_HEADER_SHA256").expect("header pin required");
    let mut model = super::load(std::path::Path::new(&path), Some(&header)).expect("load");
    let prompt: serde_json::Value = serde_json::from_str(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/testdata/gemma4/parity-prompt.json"
    )))
    .expect("pinned prompt");
    let ids = prompt["ids"].as_array().expect("ids");
    for (pos, token) in ids.iter().enumerate() {
        let token = u32::try_from(token.as_u64().expect("token")).expect("u32 token");
        if pos + 1 == ids.len() {
            model.decode(token, pos).expect("last prefill token");
        } else {
            model.decode_hidden(token, pos).expect("prefill token");
        }
    }
    let trace = model
        .ctx
        .top_logits(
            &model.stream,
            model.logits_capped.buf.range(0, VOCAB * 4).expect("logits"),
            VOCAB,
        )
        .expect("top five");
    eprintln!(
        "GEMMA4_TOP5={}",
        serde_json::to_string(&trace).expect("json")
    );
}
