use super::*;

mod controls;

fn logits(heads: usize) -> Vec<f32> {
    let values = [-7.0, -1.5, 0.0, 1.25, 3.0, 7.0, 18.0, f32::NEG_INFINITY];
    (0..heads).map(|h| values[h % GROUP]).collect()
}

fn publish_cell(p: &mut Problem, pos: i64, head: usize) {
    let s = &p.spec;
    let seed =
        (pos as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15) ^ ((head as u64) << 32) ^ 0x81a7_aa19;
    let mut rng = Rng::new(seed);
    let cell = head * s.cells + s.cell_of(pos);
    for group in 0..s.head_dim / 32 {
        let zero = rng.below(17) == 0;
        let k_scale = if zero {
            0
        } else {
            f16_bits((rng.below(31) + 1) as f32 / 1024.0)
        };
        let v_scale = if zero {
            0
        } else {
            f16_bits((rng.below(31) + 1) as f32 / 512.0)
        };
        p.k_scales[cell * (s.head_dim / 32) + group] = k_scale;
        p.v_scales[cell * (s.head_dim / 32) + group] = v_scale;
        for d in 0..32 {
            let at = cell * s.head_dim + group * 32 + d;
            p.k_codes[at] = if zero {
                0
            } else {
                (rng.below(255) as i32 - 127) as i8
            };
            p.v_codes[at] = if zero {
                0
            } else {
                (rng.below(191) as i32 - 95 + (pos / 519) as i32 * 3).clamp(-127, 127) as i8
            };
        }
    }
}

fn cases() -> Vec<Problem> {
    let shapes = [
        (128, 1, 127, 0, 0, 1, false),
        (256, 4, 385, 0, 0, 4, true),
        (128, 7, 1037, 128, 519, 64, true),
        (256, 7, 1037, 512, 519, 128, false),
        (256, 16, 1555, 496, 519, 8, true),
    ];
    shapes
        .into_iter()
        .map(|(dim, m, len, window, ring, splits, gate)| {
            let mut spec = Spec::other(dim, window, ring, 519, gate);
            spec.m = m;
            spec.kv_len = len;
            spec.split = AttnSplit {
                max_splits: splits,
                min_keys: 64,
            };
            let mut p = problem(&spec, false);
            // Publish distinct logical generations; untouched cells keep -128 codes and NaN scales.
            for pos in spec.visible(0).start..spec.visible(m - 1).end {
                for h in 0..spec.n_kv_head {
                    publish_cell(&mut p, pos, h);
                }
            }
            p
        })
        .collect()
}

fn reference_sink(p: &Problem, row: (usize, usize), sink: f32) -> Vec<f64> {
    let (t, h) = row;
    let s = &p.spec;
    let d = s.head_dim;
    let q = &p.q[(t * s.n_head + h) * d..][..d];
    let scores: Vec<f64> = s
        .visible(t)
        .map(|pos| {
            let cell = s.cell_of(pos);
            f64::from(s.scale())
                * q.iter()
                    .enumerate()
                    .map(|(dim, &qi)| {
                        f64::from(qi) * value(&p.k_codes, &p.k_scales, cell, dim, h / GROUP, s)
                    })
                    .sum::<f64>()
        })
        .collect();
    let top = scores.iter().copied().fold(f64::from(sink), f64::max);
    let real: f64 = scores.iter().map(|score| (score - top).exp()).sum();
    let fraction = real / (real + (f64::from(sink) - top).exp());
    reference(p, t, h)
        .into_iter()
        .map(|x| x * fraction)
        .collect()
}

fn launch_sink(rig: &Rig, p: &Problem, sinks: Option<&[f32]>) -> Result<Vec<f32>, EngError> {
    let (ctx, s) = (&rig.ctx, &p.spec);
    let part_len = attn_q8_part_bytes(s.m, s.n_kv_head, s.head_dim, s.split.max_splits)?;
    let out_len = s.m * s.n_head * s.head_dim * 4;
    let q = upload_f32(ctx, &p.q)?;
    let proj = s.gate.then(|| upload_f32(ctx, &p.proj)).transpose()?;
    let k_codes = upload_bytes(ctx, &le_i8(&p.k_codes))?;
    let k_scales = upload_bytes(ctx, &le16(&p.k_scales))?;
    let v_codes = upload_bytes(ctx, &le_i8(&p.v_codes))?;
    let v_scales = upload_bytes(ctx, &le16(&p.v_scales))?;
    let word = upload_bytes(ctx, &s.kv_len.to_le_bytes())?;
    let part = upload_bytes(ctx, &part_content(part_len, PartFill::Nan))?;
    let out = upload_filled(ctx, 0xff, out_len)?;
    let sink = sinks.map(|values| upload_f32(ctx, values)).transpose()?;
    let call = AttnDecodeQ8Call {
        shape: s.shape(),
        span: s.span(),
        split: s.split,
        scale: s.scale(),
        q: whole(&q)?,
        proj: proj.as_ref().map(whole).transpose()?,
        k_codes: whole(&k_codes)?,
        k_scales: whole(&k_scales)?,
        v_codes: whole(&v_codes)?,
        v_scales: whole(&v_scales)?,
        kv_len: whole(&word)?,
        part: whole(&part)?,
        out: whole(&out)?,
    };
    if let Some(sink) = &sink {
        ctx.attn_decode_q8_sinks(&rig.stream, call, whole(sink)?)?;
    } else {
        ctx.attn_decode_q8(&rig.stream, call)?;
    }
    rig.stream.synchronize()?;
    Ok(testkit::floats(&read_bytes(ctx, &out, out_len)?))
}

fn device_phase(problems: &[Problem]) -> Result<Vec<[Vec<f32>; 4]>, EngError> {
    let rig = Rig::new()?;
    problems
        .iter()
        .map(|p| {
            let sinks = logits(p.spec.n_head);
            let neutral = vec![f32::NEG_INFINITY; p.spec.n_head];
            Ok([
                launch_sink(&rig, p, Some(&sinks))?,
                launch_sink(&rig, p, Some(&sinks))?,
                launch_sink(&rig, p, None)?,
                launch_sink(&rig, p, Some(&neutral))?,
            ])
        })
        .collect()
}

fn check_head(actual: &[f32], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len());
    let magnitude = expected.iter().map(|x| x.abs()).fold(0.0_f64, f64::max);
    let exponent = ((magnitude.to_bits() >> 52) & 0x7ff) as i32 - 1023;
    let bound = 2.0 * 2.0_f64.powi(exponent.max(-126) - 7);
    for (d, (&got, &want)) in actual.iter().zip(expected).enumerate() {
        assert!(
            got.is_finite() && want.is_finite(),
            "dim={d} got={got} want={want}"
        );
        assert!(
            (f64::from(got) - want).abs() <= bound,
            "dim={d} got={got:e} want={want:e} two_bf16_ulp={bound:e}"
        );
    }
}

#[test]
#[ignore = "requires an admitted CUDA correctness run; never silently skips a missing device"]
fn split_decode_sink_q8_matches_fp64_and_preserves_neutral_bits() {
    let problems = cases();
    let observed = device_phase(&problems).expect("Q8 sink device phase; all owners dropped");
    let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    for (p, [actual, repeat, legacy, neutral]) in problems.iter().zip(observed) {
        let s = &p.spec;
        assert_eq!(bits(&actual), bits(&repeat), "repeat {}", s.label());
        assert_eq!(bits(&legacy), bits(&neutral), "neutral {}", s.label());
        let sinks = logits(s.n_head);
        for t in 0..s.m {
            for (h, &sink) in sinks.iter().enumerate() {
                let at = (t * s.n_head + h) * s.head_dim;
                let expected = reference_sink(p, (t, h), sink);
                check_head(&actual[at..at + s.head_dim], &expected);
            }
        }
        assert_ne!(
            bits(&actual),
            bits(&legacy),
            "omitting sinks must matter: {}",
            s.label()
        );
        println!("SINK_Q8 {} calls=4", s.label());
    }
}
