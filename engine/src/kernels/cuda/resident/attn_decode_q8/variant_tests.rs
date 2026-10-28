//! Gated full-64/non-power-of-two coverage and checked cursor rejection.
use super::{
    AttnQ8Shape,
    paged::{PagedQ8Input, PagedQ8Page},
};
use crate::{
    EngError,
    kernels::cuda::{CudaContext, DeviceBuffer, device_present},
};

const D: usize = 128;
const KV: usize = 2;
const HEADS: usize = KV * 8;
const SCALE: f32 = 0.75;
const ABS: f64 = 1e-5;
const E3: f64 = 1.0 / 131_072.0;

#[derive(Clone, Copy)]
struct Spec {
    cells: usize,
    groups: usize,
    keys: usize,
}

fn local(spec: Spec, group: usize) -> usize {
    (spec.keys - group * spec.cells).min(spec.cells)
}

fn quant_scale(index: usize) -> (u16, f64) {
    match index % 3 {
        0 => (0x3800, 0.5),
        1 => (0x3c00, 1.0),
        _ => (0x4000, 2.0),
    }
}

fn key_code(group: usize, kv: usize, dim: usize) -> i8 {
    if dim < 4 {
        ((group / [1, 3, 9, 27][dim] + kv) % 3) as i8 - 1
    } else {
        0
    }
}

fn value_code(group: usize, kv: usize, dim: usize) -> i8 {
    ((group + kv + dim) % 13) as i8 - 6
}

fn query_value(head: usize, dim: usize) -> f32 {
    match dim {
        0 => (head + 1) as f32 / 8.0,
        1 => -0.5,
        2 => 0.25,
        3 => -0.125,
        _ => 0.0,
    }
}

fn gate_value(head: usize, dim: usize) -> f32 {
    ((dim % 9) as f32 - 4.0) / 8.0 + head as f32 / 16.0
}

fn page_bytes(spec: Spec, group: usize) -> Vec<u8> {
    let codes = KV * spec.cells * D;
    let scales = KV * spec.cells * (D / 32) * 2;
    let v = codes + scales;
    let vs = v + codes;
    let mut bytes = vec![0; 2 * (codes + scales)];
    for kv in 0..KV {
        for row in 0..spec.cells {
            let live = row < local(spec, group);
            for dim in 0..D {
                let offset = (kv * spec.cells + row) * D + dim;
                bytes[offset] = if live {
                    key_code(group, kv, dim) as u8
                } else {
                    127
                };
                bytes[v + offset] = if live {
                    value_code(group, kv, dim) as u8
                } else {
                    127
                };
            }
            for quant in 0..D / 32 {
                let offset = ((kv * spec.cells + row) * (D / 32) + quant) * 2;
                let k = if live {
                    quant_scale(group + kv).0
                } else {
                    0x7e00
                };
                let val = if live {
                    quant_scale(group + kv + quant).0
                } else {
                    0x7e00
                };
                bytes[codes + offset..codes + offset + 2].copy_from_slice(&k.to_le_bytes());
                bytes[vs + offset..vs + offset + 2].copy_from_slice(&val.to_le_bytes());
            }
        }
    }
    bytes
}

fn page(buffer: &DeviceBuffer, spec: Spec, group: usize) -> Result<PagedQ8Page<'_>, EngError> {
    let codes = KV * spec.cells * D;
    let scales = KV * spec.cells * (D / 32) * 2;
    Ok(PagedQ8Page {
        group,
        k_codes: buffer.range(0, codes)?,
        k_scales: buffer.range(codes, scales)?,
        v_codes: buffer.range(codes + scales, codes)?,
        v_scales: buffer.range(2 * codes + scales, scales)?,
    })
}

// Independent f64 group softmax, not a decode of the uploaded bytes or GPU scratch.
fn reference(spec: Spec) -> Vec<f64> {
    let mut values = vec![0.0; HEADS * D];
    for head in 0..HEADS {
        let kv = head / 8;
        let scores: Vec<f64> = (0..spec.groups)
            .map(|group| {
                (0..4)
                    .map(|dim| {
                        f64::from(query_value(head, dim))
                            * f64::from(SCALE)
                            * f64::from(key_code(group, kv, dim))
                            * quant_scale(group + kv).1
                    })
                    .sum()
            })
            .collect();
        let maximum = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let weights: Vec<f64> = scores
            .iter()
            .enumerate()
            .map(|(group, score)| local(spec, group) as f64 * (score - maximum).exp())
            .collect();
        let denominator: f64 = weights.iter().sum();
        for dim in 0..D {
            let numerator: f64 = weights
                .iter()
                .enumerate()
                .map(|(group, weight)| {
                    weight
                        * f64::from(value_code(group, kv, dim))
                        * quant_scale(group + kv + dim / 32).1
                })
                .sum();
            let gate = 1.0 / (1.0 + (-f64::from(gate_value(head, dim))).exp());
            values[head * D + dim] = numerator / denominator * gate;
        }
    }
    values
}

struct Observed {
    bytes: Vec<u8>,
    expected: Vec<f64>,
    rejected: usize,
    unchanged: bool,
}

fn run(spec: Spec, order_checks: bool) -> Result<Observed, EngError> {
    let pages: Vec<Vec<u8>> = (0..spec.groups).map(|g| page_bytes(spec, g)).collect();
    let q: Vec<u8> = (0..HEADS)
        .flat_map(|head| (0..D).flat_map(move |dim| query_value(head, dim).to_le_bytes()))
        .collect();
    let mut projection = vec![f32::NAN; HEADS * 2 * D];
    for head in 0..HEADS {
        for dim in 0..D {
            projection[head * 2 * D + D + dim] = gate_value(head, dim);
        }
    }
    let projection: Vec<u8> = projection.iter().flat_map(|v| v.to_le_bytes()).collect();
    let expected = reference(spec);
    let mut bytes = vec![0xff; HEADS * D * 4];
    let mut rejected = 0;
    let mut unchanged = true;
    {
        let ctx = CudaContext::new()?;
        let stream = ctx.stream()?;
        let q = ctx.upload(&q)?;
        let proj = ctx.upload(&projection)?;
        let gpu: Vec<DeviceBuffer> = pages
            .iter()
            .map(|p| ctx.upload(p))
            .collect::<Result<_, _>>()?;
        let out = ctx.upload(&bytes)?;
        let shape = AttnQ8Shape {
            m: 1,
            n_head: HEADS,
            n_kv_head: KV,
            head_dim: D,
            cells: spec.keys,
            ld: 0,
        };
        let mut scratch = ctx.paged_q8_scratch(shape, spec.cells)?;
        let mut pass = scratch.begin(
            &ctx,
            &stream,
            PagedQ8Input {
                q: q.range(0, q.bytes())?,
                visible_keys: spec.keys,
                scale: SCALE,
            },
        )?;
        if order_checks {
            let before = stream.enqueued();
            rejected += usize::from(matches!(
                pass.push(page(&gpu[1], spec, 1)?),
                Err(EngError::BadRequest(_))
            ));
            unchanged &= stream.enqueued() == before;
        }
        for (group, buffer) in gpu.iter().enumerate() {
            pass.push(page(buffer, spec, group)?)?;
            if order_checks && group == 0 {
                let before = stream.enqueued();
                rejected += usize::from(matches!(
                    pass.push(page(buffer, spec, group)?),
                    Err(EngError::BadRequest(_))
                ));
                unchanged &= stream.enqueued() == before;
            }
        }
        if order_checks {
            let before = stream.enqueued();
            rejected += usize::from(matches!(
                pass.push(page(&gpu[0], spec, spec.groups)?),
                Err(EngError::BadRequest(_))
            ));
            unchanged &= stream.enqueued() == before;
        }
        pass.finish(
            Some(proj.range(0, proj.bytes())?),
            out.range(0, out.bytes())?,
        )?;
        stream.synchronize()?;
        ctx.download(&out, &mut bytes)?;
    }
    Ok(Observed {
        bytes,
        expected,
        rejected,
        unchanged,
    })
}

fn metrics(got: &Observed) -> (bool, f64, f64) {
    let actual: Vec<f64> = got
        .bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f64::from(f32::from_le_bytes([b[0], b[1], b[2], b[3]])))
        .collect();
    let mut ok = actual.len() == got.expected.len();
    let mut maximum = 0.0_f64;
    let mut square = 0.0;
    for head in 0..HEADS {
        let bound = E3
            * got.expected[head * D..(head + 1) * D]
                .iter()
                .map(|v| v.abs())
                .fold(1.0_f64, f64::max);
        for dim in 0..D {
            let i = head * D + dim;
            let error = if actual[i].is_finite() {
                (actual[i] - got.expected[i]).abs()
            } else {
                f64::INFINITY
            };
            ok &= actual[i].is_finite() && error <= ABS && error <= bound;
            maximum = maximum.max(error);
            square += error * error;
        }
    }
    (ok, maximum, (square / actual.len() as f64).sqrt())
}

#[test]
fn checked_gated_d128_full64_and_non_power_of_two_pages_match_f64() {
    if !device_present("checked_gated_d128_full64_and_non_power_of_two_pages_match_f64") {
        return;
    }
    for spec in [
        Spec {
            cells: 64,
            groups: 64,
            keys: 4096,
        },
        Spec {
            cells: 192,
            groups: 3,
            keys: 575,
        },
    ] {
        let got = run(spec, false).expect("normal checked gated variants");
        let (ok, max, rms) = metrics(&got);
        eprintln!(
            "CHECKED_VARIANT cells={} groups={} keys={} max={max} rms={rms}",
            spec.cells, spec.groups, spec.keys
        );
        assert!(ok, "gated variant must meet frozen absolute and E3 bounds");
    }
}

#[test]
fn checked_group_order_rejections_do_not_enqueue_or_advance() {
    if !device_present("checked_group_order_rejections_do_not_enqueue_or_advance") {
        return;
    }
    let got = run(
        Spec {
            cells: 64,
            groups: 2,
            keys: 65,
        },
        true,
    )
    .expect("normal group rejections");
    let (ok, max, rms) = metrics(&got);
    eprintln!(
        "CHECKED_GROUP_ORDER rejected={} unchanged={} max={max} rms={rms}",
        got.rejected, got.unchanged
    );
    assert_eq!(got.rejected, 3);
    assert!(got.unchanged);
    assert!(ok);
}

fn invalid_begin_run() -> Result<Observed, EngError> {
    let spec = Spec {
        cells: 64,
        groups: 2,
        keys: 65,
    };
    let pages: Vec<Vec<u8>> = (0..spec.groups).map(|g| page_bytes(spec, g)).collect();
    let q_bytes: Vec<u8> = (0..HEADS)
        .flat_map(|head| (0..D).flat_map(move |dim| query_value(head, dim).to_le_bytes()))
        .collect();
    let q_len = q_bytes.len();
    let mut padded = q_bytes;
    padded.extend_from_slice(&[0u8; 16]);
    let mut projection = vec![f32::NAN; HEADS * 2 * D];
    for head in 0..HEADS {
        for dim in 0..D {
            projection[head * 2 * D + D + dim] = gate_value(head, dim);
        }
    }
    let projection: Vec<u8> = projection.iter().flat_map(|v| v.to_le_bytes()).collect();
    let expected = reference(spec);
    let mut bytes = vec![0xff; HEADS * D * 4];
    let mut rejected = 0;
    let unchanged;
    {
        let ctx = CudaContext::new()?;
        let foreign = CudaContext::new()?;
        let a = ctx.stream()?;
        let b = ctx.stream()?;
        let foreign_stream = foreign.stream()?;
        let q = ctx.upload(&padded)?;
        let foreign_q = foreign.upload(&padded)?;
        let proj = ctx.upload(&projection)?;
        let gpu: Vec<DeviceBuffer> = pages
            .iter()
            .map(|p| ctx.upload(p))
            .collect::<Result<_, _>>()?;
        let out = ctx.upload(&bytes)?;
        let shape = AttnQ8Shape {
            m: 1,
            n_head: HEADS,
            n_kv_head: KV,
            head_dim: D,
            cells: 2 * spec.cells,
            ld: 0,
        };
        let mut scratch = ctx.paged_q8_scratch(shape, spec.cells)?;
        let good_q = q.range(0, q_len)?;
        let short_q = q.range(0, q_len - 4)?;
        let unaligned_q = q.range(4, q_len)?;
        let wrong_q = foreign_q.range(0, q_len)?;
        let owners = (ctx.live_bytes(), ctx.live_owners());
        let seq = (a.enqueued(), b.enqueued(), foreign_stream.enqueued());
        for (visible, scale, input_q) in [
            (0, SCALE, good_q),
            (2 * spec.cells + 1, SCALE, good_q),
            (usize::MAX, SCALE, good_q),
            (spec.keys, 0.0, good_q),
            (spec.keys, -1.0, good_q),
            (spec.keys, f32::NAN, good_q),
            (spec.keys, f32::INFINITY, good_q),
            (spec.keys, 1e30, good_q),
            (spec.keys, SCALE, short_q),
            (spec.keys, SCALE, unaligned_q),
            (spec.keys, SCALE, wrong_q),
        ] {
            rejected += usize::from(matches!(
                scratch.begin(
                    &ctx,
                    &a,
                    PagedQ8Input {
                        q: input_q,
                        visible_keys: visible,
                        scale,
                    }
                ),
                Err(EngError::BadRequest(_))
            ));
        }
        rejected += usize::from(matches!(
            scratch.begin(
                &ctx,
                &foreign_stream,
                PagedQ8Input {
                    q: good_q,
                    visible_keys: spec.keys,
                    scale: SCALE,
                }
            ),
            Err(EngError::BadRequest(_))
        ));
        // Invalid begins must not bind A; the first valid begin binds B even when aborted.
        scratch.begin(
            &ctx,
            &b,
            PagedQ8Input {
                q: good_q,
                visible_keys: spec.keys,
                scale: SCALE,
            },
        )?;
        rejected += usize::from(matches!(
            scratch.begin(
                &ctx,
                &a,
                PagedQ8Input {
                    q: good_q,
                    visible_keys: spec.keys,
                    scale: SCALE,
                }
            ),
            Err(EngError::BadRequest(_))
        ));
        rejected += usize::from(matches!(
            scratch.begin(
                &ctx,
                &foreign_stream,
                PagedQ8Input {
                    q: good_q,
                    visible_keys: spec.keys,
                    scale: SCALE,
                }
            ),
            Err(EngError::BadRequest(_))
        ));
        unchanged = seq == (a.enqueued(), b.enqueued(), foreign_stream.enqueued())
            && owners == (ctx.live_bytes(), ctx.live_owners());
        let mut pass = scratch.begin(
            &ctx,
            &b,
            PagedQ8Input {
                q: good_q,
                visible_keys: spec.keys,
                scale: SCALE,
            },
        )?;
        for (group, buffer) in gpu.iter().enumerate() {
            pass.push(page(buffer, spec, group)?)?;
        }
        pass.finish(
            Some(proj.range(0, proj.bytes())?),
            out.range(0, out.bytes())?,
        )?;
        b.synchronize()?;
        ctx.download(&out, &mut bytes)?;
    }
    Ok(Observed {
        bytes,
        expected,
        rejected,
        unchanged,
    })
}

#[test]
fn invalid_begins_preserve_affinity_enqueues_owners_and_recovery() {
    if !device_present("invalid_begins_preserve_affinity_enqueues_owners_and_recovery") {
        return;
    }
    let got = invalid_begin_run().expect("normal begin rejection matrix and recovery");
    let (ok, max, rms) = metrics(&got);
    eprintln!(
        "CHECKED_BEGIN rejected={} unchanged={} max={max} rms={rms}",
        got.rejected, got.unchanged
    );
    assert_eq!(got.rejected, 14);
    assert!(got.unchanged);
    assert!(ok);
}
