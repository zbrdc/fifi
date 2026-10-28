use super::*;
use crate::kernels::cuda::prefill::kv8_dequant::{Kv8DequantCall, Kv8DequantShape};

const ROWS: usize = 50;
const M: usize = 16;
const STAGE_ROWS: usize = 8;
const SLOTS: usize = ROWS.div_ceil(STAGE_ROWS);
const STAGE_FILL: u8 = 0x6b;

struct SegmentResult {
    actual: Vec<f32>,
    reference: Vec<f32>,
    part_only: bool,
    neutral: bool,
    canaries: bool,
}

// Keep every owner inside the Result phase; host assertions run only after it returns.
fn segmented_phase(gated: bool, stage_rows: usize) -> Result<Vec<SegmentResult>, EngError> {
    let slots = ROWS.div_ceil(stage_rows);
    let mut input = data(M, ROWS, 913, 0.0, 0.0);
    input.q.fill(0.0);
    input.k.fill(0);
    input.proj.fill(0.0);
    let mut v_codes = vec![0u8; 2 * ROWS * D];
    for row in 0..ROWS {
        for head in 0..2 {
            for d in 0..D {
                let code = (row % 23) as i8 - 11 + (head * 3 + d % 5) as i8;
                v_codes[(head * ROWS + row) * D + d] = code as u8;
                input.v[(row * 2 + head) * D + d] = f16_bits(f32::from(code) * 0.25);
            }
        }
    }
    let fx = Fx::with_data(
        qwen(M, ROWS, 34),
        CpfAttnSplit {
            max_splits: slots,
            min_keys: CPF_ATTN_MIN_KEYS,
        },
        gated,
        true,
        &input,
    )?;
    fx.run()?;
    let reference = fx.out_values();
    let k_codes = fx.ctx.upload(&vec![0u8; 2 * ROWS * D])?;
    let v_codes = fx.ctx.upload(&v_codes)?;
    let scales = fx
        .ctx
        .upload(&le16(&vec![f16_bits(0.25); 2 * ROWS * (D / 32)]))?;
    let stage_bytes = stage_rows * 2 * D * 2;
    let stage = [
        fx.ctx.upload(&vec![STAGE_FILL; stage_bytes + 16])?,
        fx.ctx.upload(&vec![STAGE_FILL; stage_bytes + 16])?,
    ];
    let call = || -> Result<AttnPrefillCall<'_>, EngError> {
        let mut call = fx.call()?;
        call.shape.cache_rows = stage_rows;
        call.k_cache = stage[0].range(0, stage_bytes)?;
        call.v_cache = stage[1].range(0, stage_bytes)?;
        Ok(call)
    };
    let mut results = Vec::new();
    for pass in 0..2 {
        fx.reset_outputs()?;
        for order in 0..slots {
            let index = if pass == 0 { order } else { slots - 1 - order };
            let origin = index * stage_rows;
            let rows = stage_rows.min(ROWS - origin);
            for (codes, output) in [(&k_codes, &stage[0]), (&v_codes, &stage[1])] {
                fx.ctx.kv8_dequant_segment(
                    &fx.stream,
                    Kv8DequantCall {
                        shape: Kv8DequantShape {
                            capacity: ROWS,
                            rows,
                            n_kv_head: 2,
                            head_dim: D,
                        },
                        codes: codes.range(0, codes.bytes())?,
                        scales: scales.range(0, scales.bytes())?,
                        out: output.range(0, stage_bytes)?,
                    },
                    origin,
                )?;
            }
            fx.ctx.attn_prefill_part(
                &fx.stream,
                call()?,
                AttnPrefillSegment {
                    key_origin: origin,
                    key_rows: rows,
                    part_index: index,
                },
            )?;
        }
        let out_bytes = M * 16 * D * 4;
        let part_only = fx.bytes(OUT)[..out_bytes].iter().all(|&b| b == OUT_FILL);
        fx.ctx.attn_prefill_combine(&fx.stream, call()?)?;
        fx.stream.synchronize()?;
        let part_bytes = fx.base[PART].2;
        let part = fx.bytes(PART);
        let floats = testkit::floats(&part[..part_bytes]);
        let cells = M.div_ceil(CPF_ATTN_Q_TILE) * 2 * slots;
        let mut neutral = true;
        if (slots - 1) * stage_rows >= fx.shape.p0 as usize + CPF_ATTN_Q_TILE {
            for head in 0..2 {
                let cell = head * slots + slots - 1;
                for row in 0..64 {
                    neutral &= floats[cell * 128 + row * 2] == f32::NEG_INFINITY;
                    neutral &= floats[cell * 128 + row * 2 + 1] == 0.0;
                }
                let start = cells * 128 + cell * 64 * D;
                neutral &= floats[start..start + 64 * D].iter().all(|&v| v == 0.0);
            }
        }
        let mut canaries = part[part_bytes..].iter().all(|&b| b == PART_FILL)
            && fx.bytes(OUT)[out_bytes..].iter().all(|&b| b == OUT_FILL)
            && fx.status_word() == 0;
        for buffer in &stage {
            let mut bytes = vec![0; buffer.bytes()];
            buffer.read_at(&fx.stream, 0, &mut bytes)?;
            canaries &= bytes[stage_bytes..].iter().all(|&b| b == STAGE_FILL);
        }
        results.push(SegmentResult {
            actual: fx.out_values(),
            reference: reference.clone(),
            part_only,
            neutral,
            canaries,
        });
    }
    Ok(results)
}

#[test]
#[ignore = "native GPU correctness; precompiled binary and GPU lock required"]
fn attn_prefill_segmented_stage_matches_full_prefix_and_gates_once() {
    if !testkit::device_present("attn_prefill_segmented_stage_matches_full_prefix_and_gates_once") {
        return;
    }
    for (gated, stage_rows) in [(false, 8), (true, 8), (false, 40), (true, 40)] {
        let results = segmented_phase(gated, stage_rows).expect("same-stream staged attention");
        for result in results {
            let expected_first = if gated {
                -8.25_f32 / 35.0
            } else {
                -16.5 / 35.0
            };
            assert!((result.reference[0] - expected_first).abs() <= 2e-6);
            assert!(result.part_only, "part-only entry wrote output");
            assert!(
                result.neutral,
                "future segment did not initialize every neutral row"
            );
            assert!(result.canaries, "stage/part/output/status boundary changed");
            assert_eq!(result.actual.len(), result.reference.len());
            for (actual, reference) in result.actual.iter().zip(&result.reference) {
                assert!(actual.is_finite() && reference.is_finite());
                assert!(
                    (actual - reference).abs() <= 2e-6,
                    "{actual} != {reference}"
                );
            }
        }
    }
}

fn refusal_phase() -> Result<Vec<bool>, EngError> {
    let mut fx = Fx::new(
        qwen(M, ROWS, 34),
        CpfAttnSplit {
            max_splits: SLOTS,
            min_keys: CPF_ATTN_MIN_KEYS,
        },
        17,
        true,
        true,
    )?;
    fx.shape.cache_rows = STAGE_ROWS;
    let segment = AttnPrefillSegment {
        key_origin: 0,
        key_rows: STAGE_ROWS,
        part_index: 0,
    };
    let mut refused = Vec::new();
    for span in [
        CpfAttnSpan { window: 1, ring: 0 },
        CpfAttnSpan {
            window: 16,
            ring: 32,
        },
    ] {
        fx.span = span;
        refused.push(matches!(
            fx.ctx.attn_prefill_part(&fx.stream, fx.call()?, segment),
            Err(EngError::BadRequest(_))
        ));
        refused.push(matches!(
            fx.ctx.attn_prefill_combine(&fx.stream, fx.call()?),
            Err(EngError::BadRequest(_))
        ));
    }
    fx.span = CpfAttnSpan::FULL;
    fx.ctx.attn_prefill_part(&fx.stream, fx.call()?, segment)?;
    fx.stream.synchronize()?;
    fx.ctx.ensure_usable()?;
    Ok(refused)
}

#[test]
#[ignore = "native GPU correctness; precompiled binary and GPU lock required"]
fn attn_prefill_segmented_rejects_window_and_ring_without_poisoning() {
    if !testkit::device_present("attn_prefill_segmented_rejects_window_and_ring_without_poisoning")
    {
        return;
    }
    assert_eq!(
        refusal_phase().expect("checked refusal then valid part"),
        vec![true; 4]
    );
}
