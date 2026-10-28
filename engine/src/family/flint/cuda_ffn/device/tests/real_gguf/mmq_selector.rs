//! Backend-selection witness, not a replacement for S3's original-F64 accuracy gate.
//! The loader's explicit mode is used for rollback probes; tests never mutate process env.
use super::*;
use crate::kernels::cuda::prefill::mmq_nvfp4::{Nvfp4DenseCall, Nvfp4DenseShape};
use std::io::Seek;

#[test]
fn mmq_policy_defaults_off_and_refuses_ambiguous_values() {
    assert!(!mmq_enabled(None).unwrap());
    assert!(!mmq_enabled(Some("0")).unwrap());
    assert!(mmq_enabled(Some("1")).unwrap());
    for value in ["", "true", "01", " 1", "2"] {
        assert!(mmq_enabled(Some(value)).is_err());
    }
}

struct Raw {
    bytes: DeviceBuffer,
    factor: f32,
    shape: [usize; 2],
}

impl Raw {
    fn load(ctx: &CudaContext, file: &File, source: &MatrixSource) -> Self {
        let data = gguf::raw(file, source.data.clone()).unwrap();
        let scale = gguf::raw(file, source.scale.clone()).unwrap();
        let bound = source.bind(&data, &scale).unwrap();
        Self {
            bytes: ctx.upload(bound.data).unwrap(),
            factor: bound.tensor_scale,
            shape: bound.shape,
        }
    }
}

fn project(frame: &mut Frame<'_, '_>, x: usize, weight: &Raw, rows: usize) -> usize {
    let ctx = frame.ctx;
    let [n, k] = weight.shape;
    let sizes = Nvfp4DenseShape::new(k, n, rows).unwrap();
    let plan = ctx
        .mmq_nvfp4_dense_plan(
            weight.bytes.range(0, weight.bytes.bytes()).unwrap(),
            [k, n, rows],
            weight.factor,
        )
        .unwrap();
    let normalized = frame.allocate(RowShape { rows, columns: k }).unwrap();
    let scales = frame.allocate(RowShape { rows, columns: 1 }).unwrap();
    let out_shape = RowShape { rows, columns: n };
    let raw = frame.allocate(out_shape).unwrap();
    let restored = frame.allocate(out_shape).unwrap();
    let q8 = frame.keep(ctx.allocate(sizes.needs.y).unwrap(), sizes.needs.y);
    let fixup_bytes = ctx.mmq_fixup_bytes().unwrap();
    let fixup = frame.keep(ctx.allocate(fixup_bytes).unwrap(), fixup_bytes);
    ctx.row_pow2(
        frame.stream,
        RowPow2Call {
            x: frame.range(x).unwrap(),
            out: frame.range(normalized).unwrap(),
            scales: frame.range(scales).unwrap(),
            flag: frame.flag().unwrap(),
            rows,
            columns: k,
            mode: RowPow2Mode::Normalize,
        },
    )
    .unwrap();
    ctx.mmq_nvfp4_dense_input(
        frame.stream,
        &plan,
        frame.range(normalized).unwrap(),
        frame.range(q8).unwrap(),
    )
    .unwrap();
    ctx.mmq_nvfp4_dense(
        frame.stream,
        Nvfp4DenseCall {
            plan: &plan,
            q8_1: frame.range(q8).unwrap(),
            dst: frame.range(raw).unwrap(),
            fixup: frame.range(fixup).unwrap(),
        },
    )
    .unwrap();
    ctx.row_pow2(
        frame.stream,
        RowPow2Call {
            x: frame.range(raw).unwrap(),
            out: frame.range(restored).unwrap(),
            scales: frame.range(scales).unwrap(),
            flag: frame.flag().unwrap(),
            rows,
            columns: n,
            mode: RowPow2Mode::Restore,
        },
    )
    .unwrap();
    let published = frame.publish(restored, out_shape).unwrap();
    // This explicit reference driver does not retain plans in the frame.
    frame.stream.synchronize().unwrap();
    published
}

fn expert(frame: &mut Frame<'_, '_>, x: usize, weights: &[Raw; 3], rows: usize) -> usize {
    let shape = RowShape {
        rows,
        columns: weights[0].shape[0],
    };
    let gate = project(frame, x, &weights[0], rows);
    let up = project(frame, x, &weights[1], rows);
    let probability = frame.sigmoid(gate, shape).unwrap();
    let silu = frame
        .multiply(gate, probability, shape, shape.columns)
        .unwrap();
    let hidden = frame.multiply(silu, up, shape, shape.columns).unwrap();
    project(frame, hidden, &weights[2], rows)
}

fn expected(
    ctx: &CudaContext,
    stream: &DeviceStream,
    input: &DeviceBuffer,
    weights: &FfnWeights<'_>,
    raw: &[Raw; 3],
) -> DeviceBuffer {
    let rows = input.bytes() / (WIDTH * 4);
    let block = weights.block(rows).unwrap();
    let mut frame = Frame::new(ctx, stream).unwrap();
    let shape = RowShape {
        rows,
        columns: WIDTH,
    };
    let x = frame
        .input(input.range(0, input.bytes()).unwrap(), shape)
        .unwrap();
    if let FfnWeights::Moe {
        router, experts, ..
    } = weights
    {
        let routed = route(&mut frame, *router, x, block).unwrap();
        let selected = routed
            .grouping
            .order
            .iter()
            .map(|&p| experts[routed.ids[p] as usize])
            .collect::<Vec<_>>();
        let plan = ExpertPlan::new(ctx, &selected).unwrap();
        let tokens = frame.upload_indices(&routed.grouping.tokens).unwrap();
        let inverse = frame.upload_indices(&routed.grouping.inverse).unwrap();
        let pairs = RowShape {
            rows: block.pairs,
            columns: WIDTH,
        };
        let gathered = frame.gather(x, tokens, pairs, rows).unwrap();
        let sorted = frame.expert(&plan, gathered).unwrap();
        let down = frame.gather(sorted, inverse, pairs, block.pairs).unwrap();
        let shared = expert(&mut frame, x, raw, rows);
        let result = frame.combine(down, routed.gates, shared, block).unwrap();
        frame.finish(result).unwrap()
    } else {
        let result = expert(&mut frame, x, raw, rows);
        frame.finish(result).unwrap()
    }
}

#[derive(Deserialize)]
struct Activation {
    layer: u32,
    rows: usize,
    model_sha256: String,
    trace_matches_untraced: bool,
    boundaries: Vec<Boundary>,
}
#[derive(Deserialize)]
struct Boundary {
    name: String,
    bits: Vec<u32>,
}

#[test]
#[ignore = "requires admitted CUDA, pinned GGUF/native activation, FLINT_CUDA_FFN_MMQ=1"]
fn explicit_mmq_selection_reaches_dense_and_shared_ffn_not_frame() {
    assert!(device_present(
        "explicit_mmq_selection_reaches_dense_and_shared_ffn_not_frame"
    ));
    assert_eq!(std::env::var("FLINT_CUDA_FFN_MMQ").unwrap(), "1");
    let mut file = File::open(std::env::var("FIFI_FLINT_GGUF").unwrap()).unwrap();
    crate::weights::verify_sha256(
        &mut BufReader::new(&file),
        22_863_084_160,
        gguf::MODEL_SHA256,
    )
    .unwrap();
    file.rewind().unwrap();
    let header = GgufHeader::parse(
        &mut BufReader::new(&file),
        file.metadata().unwrap().len(),
        64 << 20,
    )
    .unwrap();
    let bytes = std::fs::read(std::env::var("FIFI_FLINT_MMQ_INPUT").unwrap()).unwrap();
    assert_eq!(
        format!("{:x}", Sha256::digest(&bytes)),
        "88bcdf501fe655c749344bf669f507df3b2fe8edfd40b6e1f2c3b8e64cb480d8"
    );
    let saved: Activation = serde_json::from_slice(&bytes).unwrap();
    assert_eq!((saved.layer, saved.rows), (0, 5));
    assert_eq!(saved.model_sha256, gguf::MODEL_SHA256);
    assert!(saved.trace_matches_untraced);
    let matches = saved
        .boundaries
        .iter()
        .filter(|b| b.name == "post_norm")
        .collect::<Vec<_>>();
    assert_eq!(matches.len(), 1);
    let bits = &matches[0].bits;
    assert_eq!(bits.len(), saved.rows * WIDTH);
    assert!(
        bits.iter()
            .all(|&b| b & 0xffff == 0 && f32::from_bits(b).is_finite())
    );
    let ctx = CudaContext::new().unwrap();
    let stream = ctx.stream().unwrap();
    let input = ctx
        .upload(
            &bits
                .iter()
                .flat_map(|b| b.to_ne_bytes())
                .collect::<Vec<_>>(),
        )
        .unwrap();
    for layer in [0, 2] {
        let ffn = Ffn::load(&ctx, &file, &header, layer).unwrap();
        let stems = ["gate", "up", "down"].map(|p| {
            format!(
                "blk.{layer}.ffn_{p}{}",
                if layer == 0 { "" } else { "_shexp" }
            )
        });
        let sources = gguf::scalar_sources(
            &header,
            stems.each_ref().map(String::as_str),
            if layer == 0 { 6144 } else { 1024 },
        )
        .unwrap();
        let raw = [&sources.gate, &sources.up, &sources.down].map(|s| Raw::load(&ctx, &file, s));
        let check = |weights: FfnWeights<'_>| {
            let wanted = expected(&ctx, &stream, &input, &weights, &raw);
            let baseline = forward(
                &ctx,
                &stream,
                FfnCall {
                    weights: &weights,
                    x: input.range(0, input.bytes()).unwrap(),
                    rows: saved.rows,
                },
            )
            .unwrap();
            let wanted = values(&stream, &wanted).unwrap();
            let baseline = values(&stream, &baseline).unwrap();
            assert_ne!(
                wanted, baseline,
                "selection fixture must distinguish MMQ from Frame layer={layer}"
            );
            let actual = ffn.run(&ctx, &stream, &input).unwrap();
            assert_eq!(
                values(&stream, &actual).unwrap(),
                wanted,
                "MMQ_SELECTOR_NOT_FRAME layer={layer}"
            );
            let rollback = Ffn::load_with_mmq(&ctx, &file, &header, layer, false).unwrap();
            match &rollback {
                Ffn::Dense(e) => assert!(e.mmq.is_none()),
                Ffn::Moe {
                    experts, shared, ..
                } => {
                    assert!(shared.mmq.is_none());
                    assert!(experts.iter().all(|e| e.mmq.is_none()));
                }
            }
            assert_eq!(
                values(&stream, &rollback.run(&ctx, &stream, &input).unwrap()).unwrap(),
                baseline,
                "explicit Frame rollback layer={layer}"
            );
            let mut broken = Ffn::load_with_mmq(&ctx, &file, &header, layer, true).unwrap();
            let e = match &mut broken {
                Ffn::Dense(e) => e,
                Ffn::Moe { shared, .. } => shared,
            };
            e.mmq.as_mut().unwrap()[0] = ctx.upload(&[0; 16]).unwrap();
            assert!(
                matches!(
                    broken.run(&ctx, &stream, &input),
                    Err(EngError::BadRequest(_))
                ),
                "MMQ error must not silently fall back to Frame"
            );
            assert_eq!(
                values(&stream, &rollback.run(&ctx, &stream, &input).unwrap()).unwrap(),
                baseline,
                "range refusal must leave Frame rollback healthy layer={layer}"
            );
        };
        match &ffn {
            Ffn::Dense(e) => check(FfnWeights::Dense(e.view(&ctx).unwrap())),
            Ffn::Moe {
                router,
                experts,
                shared,
            } => {
                let views = experts
                    .iter()
                    .map(|e| e.view(&ctx).unwrap())
                    .collect::<Vec<_>>();
                check(FfnWeights::Moe {
                    router: router.range(0, router.bytes()).unwrap(),
                    experts: &views,
                    shared: shared.view(&ctx).unwrap(),
                });
            }
        }
        eprintln!(
            "MMQ_SELECTOR_OK layer={layer} rows={} SIX_FFN_SHAPES_ONLY NO_MODEL_OR_SPEED_CLAIM",
            saved.rows
        );
    }
}
