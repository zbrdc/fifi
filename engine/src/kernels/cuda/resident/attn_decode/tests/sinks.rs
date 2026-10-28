use super::*;

mod controls;

fn logits(heads: usize) -> Vec<f32> {
    let values = [-7.0, -1.5, 0.0, 1.25, 3.0, 7.0, 18.0, f32::NEG_INFINITY];
    (0..heads).map(|h| values[h % GROUP]).collect()
}

fn cases(pool: &Pool) -> Vec<Problem> {
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
            let mut p = problem(&spec, pool, false);
            // Publish distinct logical generations; untouched physical rows remain poisoned.
            for pos in spec.visible(0).start..spec.visible(m - 1).end {
                for h in 0..spec.n_kv_head {
                    let seed = (pos as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15)
                        ^ ((h as u64) << 32)
                        ^ 0x51a7_aa19;
                    let mut rng = Rng::new(seed);
                    let at = (spec.row_of(pos) * spec.n_kv_head + h) * dim;
                    for d in 0..dim {
                        p.k[at + d] = f16_bits((rng.normal() * 0.8) as f32);
                        p.v[at + d] =
                            f16_bits((rng.normal() * 0.6 + (pos / 519) as f64 * 0.125) as f32);
                    }
                }
            }
            p
        })
        .collect()
}

fn reference_sink(p: &Problem, pool: &Pool, row: (usize, usize), sink: f32) -> Vec<f64> {
    let (t, h) = row;
    let s = &p.spec;
    let d = s.head_dim;
    let q = &p.q[(t * s.n_head + h) * d..][..d];
    let scores: Vec<f64> = s
        .visible(t)
        .map(|pos| {
            let at = (s.row_of(pos) * s.n_kv_head + h / GROUP) * d;
            f64::from(s.scale())
                * q.iter()
                    .zip(&p.k[at..])
                    .map(|(&qi, &ki)| f64::from(qi) * pool.value[usize::from(ki)])
                    .sum::<f64>()
        })
        .collect();
    let top = scores.iter().copied().fold(f64::from(sink), f64::max);
    let real: f64 = scores.iter().map(|score| (score - top).exp()).sum();
    let fraction = real / (real + (f64::from(sink) - top).exp());
    // The ordinary chronological FP64 SDPA has the same zero-valued extra denominator term.
    reference(p, pool, t, h)
        .into_iter()
        .map(|x| x * fraction)
        .collect()
}

fn launch_sink(rig: &Rig, p: &Problem, sinks: Option<&[f32]>) -> Result<Vec<f32>, EngError> {
    let (ctx, s) = (&rig.ctx, &p.spec);
    let part_len = attn_part_bytes(s.m, s.n_kv_head, s.head_dim, s.split.max_splits)?;
    let out_len = s.m * s.n_head * s.head_dim * 4;
    let q = upload_f32(ctx, &p.q)?;
    let proj = s.gate.then(|| upload_f32(ctx, &p.proj)).transpose()?;
    let k = upload_bytes(ctx, &le_bytes(&p.k))?;
    let v = upload_bytes(ctx, &le_bytes(&p.v))?;
    let word = upload_bytes(ctx, &s.kv_len.to_le_bytes())?;
    let part = upload_bytes(ctx, &part_content(part_len, PartFill::Nan))?;
    let out = upload_filled(ctx, 0xff, out_len)?;
    let sink = sinks.map(|values| upload_f32(ctx, values)).transpose()?;
    let call = AttnDecodeCall {
        shape: s.shape(),
        span: s.span(),
        split: s.split,
        scale: s.scale(),
        q: try_whole(&q)?,
        proj: proj.as_ref().map(try_whole).transpose()?,
        k_cache: try_whole(&k)?,
        v_cache: try_whole(&v)?,
        kv_len: try_whole(&word)?,
        part: try_whole(&part)?,
        out: try_whole(&out)?,
    };
    if let Some(sink) = &sink {
        ctx.attn_decode_sinks(&rig.stream, call, try_whole(sink)?)?;
    } else {
        ctx.attn_decode(&rig.stream, call)?;
    }
    rig.stream.synchronize()?;
    Ok(testkit::floats(&read_bytes(ctx, &out, out_len)?))
}

fn device_phase(problems: &[Problem]) -> Result<Vec<[Vec<f32>; 4]>, EngError> {
    let rig = Rig::try_new()?;
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
fn split_decode_sink_f16_matches_fp64_and_preserves_neutral_bits() {
    let pool = Pool::new();
    let problems = cases(&pool);
    let observed = device_phase(&problems).expect("F16 sink device phase; all owners dropped");
    let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    for (p, [actual, repeat, legacy, neutral]) in problems.iter().zip(observed) {
        let s = &p.spec;
        assert_eq!(bits(&actual), bits(&repeat), "repeat {}", s.label());
        assert_eq!(bits(&legacy), bits(&neutral), "neutral {}", s.label());
        let sinks = logits(s.n_head);
        for t in 0..s.m {
            for (h, &sink) in sinks.iter().enumerate() {
                let at = (t * s.n_head + h) * s.head_dim;
                let expected = reference_sink(p, &pool, (t, h), sink);
                check_head(&actual[at..at + s.head_dim], &expected);
            }
        }
        assert_ne!(
            bits(&actual),
            bits(&legacy),
            "omitting sinks must matter: {}",
            s.label()
        );
        println!("SINK_F16 {} calls=4", s.label());
    }
}
