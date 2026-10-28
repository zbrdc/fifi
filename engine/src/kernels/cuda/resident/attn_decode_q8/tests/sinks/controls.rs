use super::*;
use crate::kernels::cuda::CapturePlan;

const HEADS: usize = 16;
const DIM: usize = 128;
const CELLS: usize = 129;
const LENGTHS: [i32; 12] = [1, 10, 33, 65, 100, 129, 17, 0, -1, 130, i32::MAX, 10];

struct Refusal {
    label: &'static str,
    rejected: bool,
    unchanged: bool,
    recovered: bool,
    diagnostic: String,
}

struct Report {
    refusals: Vec<Refusal>,
    nodes: usize,
    eager: Vec<Vec<f32>>,
    replay: Vec<Vec<f32>>,
}

fn phase() -> Result<Report, EngError> {
    let rig = Rig::new()?;
    let other = Rig::new()?;
    let (ctx, stream) = (&rig.ctx, &rig.stream);
    let spec = Spec::other(DIM, 0, 0, CELLS, false);
    let part_len = attn_q8_part_bytes(1, 2, DIM, spec.split.max_splits)?;
    let out_len = HEADS * DIM * 4;
    let q = upload_f32(ctx, &vec![0.25; HEADS * DIM])?;
    let k_codes = upload_bytes(ctx, &le_i8(&vec![1; CELLS * 2 * DIM]))?;
    let v_codes = upload_bytes(ctx, &le_i8(&vec![3; CELLS * 2 * DIM]))?;
    let k_scales = upload_bytes(ctx, &le16(&vec![f16_bits(0.125); CELLS * 2 * DIM / 32]))?;
    let v_scales = upload_bytes(ctx, &le16(&vec![f16_bits(0.25); CELLS * 2 * DIM / 32]))?;
    let word = upload_bytes(ctx, &10_i32.to_le_bytes())?;
    let nan_part = part_content(part_len, PartFill::Nan);
    let part = upload_bytes(ctx, &nan_part)?;
    let out = upload_filled(ctx, 0xff, out_len)?;
    let mut values = logits(HEADS);
    values.push(0.0);
    let sinks = upload_f32(ctx, &values)?;
    let foreign = upload_f32(&other.ctx, &values)?;
    let valid = sinks.range(0, HEADS * 4)?;
    let queue = |sink: DeviceRange<'_>| {
        ctx.attn_decode_q8_sinks(
            stream,
            AttnDecodeQ8Call {
                shape: spec.shape(),
                span: spec.span(),
                split: spec.split,
                scale: spec.scale(),
                q: whole(&q)?,
                proj: None,
                k_codes: whole(&k_codes)?,
                k_scales: whole(&k_scales)?,
                v_codes: whole(&v_codes)?,
                v_scales: whole(&v_scales)?,
                kv_len: whole(&word)?,
                part: whole(&part)?,
                out: whole(&out)?,
            },
            sink,
        )
    };
    queue(valid)?;
    stream.synchronize()?;
    let baseline = read_bytes(ctx, &out, out_len)?;
    let partials = read_bytes(ctx, &part, part_len)?;
    let bad = [
        ("short", sinks.range(0, (HEADS - 1) * 4)?),
        ("misaligned", sinks.range(1, HEADS * 4)?),
        ("foreign", foreign.range(0, HEADS * 4)?),
        ("out alias", out.range(0, HEADS * 4)?),
        ("part alias", part.range(0, HEADS * 4)?),
    ];
    let mut refusals = Vec::new();
    for (label, range) in bad {
        let result = queue(range);
        let diagnostic = format!("{result:?}");
        let rejected = matches!(result, Err(EngError::BadRequest(_)));
        stream.synchronize()?;
        let unchanged = read_bytes(ctx, &out, out_len)? == baseline
            && read_bytes(ctx, &part, part_len)? == partials;
        queue(valid)?;
        stream.synchronize()?;
        let recovered = read_bytes(ctx, &out, out_len)? == baseline;
        refusals.push(Refusal {
            label,
            rejected,
            unchanged,
            recovered,
            diagnostic,
        });
    }
    let reset = |len: i32| -> Result<(), EngError> {
        word.write_at(stream, 0, &len.to_le_bytes())?;
        part.write_at(stream, 0, &nan_part)?;
        out.write_at(stream, 0, &vec![0xff; out_len])
    };
    let mut eager = Vec::new();
    for len in LENGTHS {
        reset(len)?;
        queue(valid)?;
        stream.synchronize()?;
        eager.push(testkit::floats(&read_bytes(ctx, &out, out_len)?));
    }
    reset(10)?;
    let (graph, ()) = ctx.capture(
        stream,
        CapturePlan {
            pins: &[],
            holds: Vec::new(),
        },
        || queue(valid),
    )?;
    let nodes = graph.nodes();
    let mut replay = Vec::new();
    for len in LENGTHS {
        reset(len)?;
        graph.launch(stream)?;
        stream.synchronize()?;
        replay.push(testkit::floats(&read_bytes(ctx, &out, out_len)?));
    }
    Ok(Report {
        refusals,
        nodes,
        eager,
        replay,
    })
}

#[test]
#[ignore = "requires an admitted CUDA correctness run; never silently skips a missing device"]
fn split_decode_sink_q8_refusals_and_device_word_capture() {
    let report = phase().expect("sink controls; every device owner dropped");
    assert_eq!(report.refusals.len(), 5);
    for row in report.refusals {
        assert!(
            row.rejected && row.unchanged && row.recovered,
            "{}: rejected={} unchanged={} recovered={} {}",
            row.label,
            row.rejected,
            row.unchanged,
            row.recovered,
            row.diagnostic
        );
    }
    assert_eq!(report.nodes, 2, "split-KV capture retains two kernels");
    assert_eq!(report.eager.len(), LENGTHS.len());
    assert_eq!(report.replay.len(), LENGTHS.len());
    let scale = Spec::other(DIM, 0, 0, CELLS, false).scale();
    for ((len, eager), replay) in LENGTHS.into_iter().zip(report.eager).zip(report.replay) {
        assert_eq!(
            eager.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            replay.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            "kv_len={len}"
        );
        assert_eq!(replay.len(), HEADS * DIM);
        if len <= 0 || len as usize > CELLS {
            assert!(replay.iter().all(|x| x.is_nan()), "invalid kv_len={len}");
            continue;
        }
        for (h, sink) in logits(HEADS).into_iter().enumerate() {
            let real = f64::from(len) * (4.0 * f64::from(scale)).exp();
            let want = 0.75 * real / (real + f64::from(sink).exp());
            check_head(&replay[h * DIM..(h + 1) * DIM], &vec![want; DIM]);
        }
    }
}
