use super::super::s2_nvfp4_dense::bench_cell;
use super::*;

const ROWS: [usize; 3] = [512, 1024, 2048];

fn bench_weight(
    weight: &Weight,
    saved: &[f32],
    provenance: &str,
    measured: bool,
) -> Result<(), EngError> {
    let ctx = CudaContext::new()?;
    let stream = ctx.stream()?;
    let raw = ctx.upload(&weight.raw)?;
    let packed = ctx.upload(&weight.packed)?;
    let scales = ctx.upload(&weight.scales)?;
    let view = ctx.nvfp4_weight_range(
        packed.range(0, packed.bytes())?,
        scales.range(0, scales.bytes())?,
        weight.factor,
        [weight.n, weight.k],
    )?;
    let expected_saved = weight.expected(saved);
    for rows in ROWS {
        let input = repeat_rows(saved, weight.k, rows);
        let x = ctx.upload(
            &input
                .iter()
                .flat_map(|x| x.to_ne_bytes())
                .collect::<Vec<_>>(),
        )?;
        let old = ctx.nvfp4_grouped_plan(&vec![view; rows])?;
        let mmq = ctx.mmq_nvfp4_dense_plan(
            raw.range(0, raw.bytes())?,
            [weight.k, weight.n, rows],
            weight.factor,
        )?;
        let mut calibration = None;
        let run = |candidate| {
            let empty = if calibration.is_none() {
                let host = std::time::Instant::now();
                let begin = ctx.event_timing()?;
                let end = ctx.event_timing()?;
                stream.record_timing(&begin)?;
                stream.record_timing(&end)?;
                Some((begin, end, host.elapsed().as_nanos()))
            } else {
                None
            };
            let mut frame = Frame::new(&ctx, &stream)?;
            let input = frame.input(
                x.range(0, x.bytes())?,
                RowShape {
                    rows,
                    columns: weight.k,
                },
            )?;
            let output = if candidate {
                frame.bench_mmq_project(&mmq, input, rows, [weight.n, weight.k])?
            } else {
                frame.project(&old, input, rows, [weight.n, weight.k])?
            };
            let output = frame.finish(output)?;
            if let Some((begin, end, before)) = empty {
                // Read only after the normal projection completion; this is the first warm-up, not a sample.
                let host = std::time::Instant::now();
                let ms = begin.elapsed_ms_since(&end)?;
                drop((begin, end));
                calibration = Some((ms, before + host.elapsed().as_nanos()));
            }
            Ok(output)
        };
        let cell = bench_cell(&stream, run, rows, measured)?;
        let expected = repeat_rows(&expected_saved, weight.n, rows);
        check_rows(
            &cell.output[0],
            &expected,
            weight.n,
            "prefill-frame-original-f64",
        );
        check_rows(
            &cell.output[1],
            &expected,
            weight.n,
            "prefill-mmq-original-f64",
        );
        let current = cell.output[0]
            .iter()
            .map(|&x| f64::from(x))
            .collect::<Vec<_>>();
        check_rows(&cell.output[1], &current, weight.n, "prefill-mmq-frame");
        assert_eq!(cell.rows, rows);
        assert_eq!(cell.samples_ns.len(), if measured { 30 } else { 0 });
        assert!(cell.samples_ns.iter().flatten().all(|&ns| ns > 0));
        let (empty_gpu_ms, empty_host_ns) = calibration.expect("completed first warm-up");
        assert!(empty_gpu_ms.is_finite() && empty_gpu_ms >= 0.0);
        eprintln!(
            "PREFILL_MICROBENCH_RECORD {}",
            serde_json::json!({
                "tensor":weight.name, "M":rows, "K":weight.k, "N":weight.n,
                "mode":if measured { "timing" } else { "dry" },
                "input":provenance, "saved_rows":SAVED_ROWS, "replay":"cyclic whole rows",
                "pair_order":["Frame.project", "Frame.mmq_project"],
                "samples_ns":cell.samples_ns, "warmup_pairs":if measured { 8 } else { 1 },
                "metric":"completed production projection wall: allocation, row_pow2, GEMM, MMQ input quant/fixup, BF16 and health; excludes immutable weight/input uploads, plan construction, readback and output destruction",
                "old_small_m_difference":"S2 MMQ preallocated q8/fixup and omitted row_pow2; this table uses the actual production method on both arms",
                "empty_event_pair_gpu_ms":empty_gpu_ms, "empty_event_pair_host_ns":empty_host_ns,
                "calibration":"one empty pair before first warm-up, read after its existing finish; no new sync",
                "all_rows_original_f64_and_frame_checks_passed":true,
                "model_qualification":false, "default_flip":false
            })
        );
    }
    Ok(())
}

fn timing_mode(mode: &str) -> bool {
    match mode {
        "dry" => false,
        "timing" => true,
        _ => panic!("FIFI_FLINT_MMQ_BENCH_MODE must be dry or timing"),
    }
}

#[test]
fn mode_requires_explicit_dry_or_timing() {
    assert!(!timing_mode("dry"));
    assert!(timing_mode("timing"));
    for invalid in ["", "Timing", "garbage"] {
        assert!(std::panic::catch_unwind(|| timing_mode(invalid)).is_err());
    }
}

#[test]
#[ignore = "requires admitted CUDA and pinned real tensors; timing mode requires exclusive coordinator GO"]
fn nvfp4_dense_shared_prefill_microbench() {
    let mode = std::env::var("FIFI_FLINT_MMQ_BENCH_MODE").expect("explicit dry or timing mode");
    let measured = timing_mode(&mode);
    assert!(device_present("nvfp4_dense_shared_prefill_microbench"));
    let mut tensors = 0;
    with_cases(|weight, input, provenance| {
        bench_weight(weight, input, provenance, measured)
            .expect("completed prefill projection benchmark");
        tensors += 1;
    });
    assert_eq!(tensors, 6);
    eprintln!(
        "PREFILL_MICROBENCH_END observed_tensors={tensors} observed_cells={} mode={mode}",
        tensors * ROWS.len()
    );
}
