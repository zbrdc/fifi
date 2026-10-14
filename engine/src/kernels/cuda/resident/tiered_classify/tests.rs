//! Device tests of the tiered classification wrapper at seam 1, against the host oracle
//! `classify_reference` (TCAP C2, test seam approved by the user 2026-10-13). Inputs are built here with
//! `testkit::Rng`: no oracle data, no environment variable, no `#[ignore]`. The mailbox is plain device
//! memory here (the mapped memory and the server are C1 and C5). Run with `--test-threads=1`.
//!
//! Every device phase is a function that creates its own context, stream and buffers, has only `?` inside,
//! drops every owner when it returns and hands back plain host records; the tests assert on those records
//! afterwards (an assertion that panics while a context is alive would latch the backend for the rest of a
//! one-process run).

use super::{
    CTRL_BYTES, ClassifyCall, ClassifyShape, LOG_BYTES_PER_PICK, REQUEST_BYTES, check_shape,
};
use crate::EngError;
use crate::family::qwen36::tiered::classify::{Classified, classify_reference, word_after};
use crate::kernels::cuda::DeviceBuffer;
use crate::kernels::cuda::context::{CudaContext, DeviceRange, DeviceStream};
use crate::kernels::cuda::testkit::{self, Rng};

const POISON: u8 = 0x55;
const UNTOUCHED: u8 = 0x77;
/// Entries on each side of the slot row: a slot the classification must never read (a valid one, 0).
const GUARD: usize = 4;

fn bytes_i32(values: &[i32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn bytes_f32(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn i32s(bytes: &[u8]) -> Vec<i32> {
    let (chunks, _) = bytes.as_chunks::<4>();
    chunks.iter().map(|c| i32::from_le_bytes(*c)).collect()
}

fn u64_at(bytes: &[u8], word: usize) -> u64 {
    u64::from_le_bytes(bytes[word * 8..word * 8 + 8].try_into().expect("8 bytes"))
}

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|v| v.to_bits()).collect()
}

fn filled(ctx: &CudaContext, byte: u8, len: usize) -> Result<DeviceBuffer, EngError> {
    ctx.upload(&vec![byte; len])
}

fn read(ctx: &CudaContext, buffer: &DeviceBuffer, len: usize) -> Result<Vec<u8>, EngError> {
    let mut bytes = vec![0u8; len];
    ctx.download(buffer, &mut bytes)?;
    Ok(bytes)
}

/// `bytes` bytes of `buffer` from `offset`.
fn at(buffer: &DeviceBuffer, offset: usize, bytes: usize) -> Result<DeviceRange<'_>, EngError> {
    buffer.range(offset, bytes)
}

/// One layer's inputs.
#[derive(Clone)]
struct Layer {
    layer: usize,
    ids: Vec<i32>,
    weights: Vec<f32>,
    slot_row: Vec<i32>,
    norm_row: Vec<f32>,
}

/// One launch of a script: the inputs, a word the host sets first (a bit another kernel would have set), and
/// what the oracle expects of the layer and of the token's word after it.
struct Planned {
    input: Layer,
    set_word: Option<i32>,
    want: Classified,
    want_word: i32,
}

/// What the device wrote for one layer, decoded.
struct Observed {
    classified: Classified,
    count: u64,
    expect_seq: u64,
    seq_counter: u64,
    request: Vec<u8>,
    row: Vec<f32>,
}

/// The device buffers of one tiered session (one token across layers): the inputs are rewritten per launch,
/// `ctrl`, `word` and the mailbox persist.
struct Rig<'c> {
    ctx: &'c CudaContext,
    stream: &'c DeviceStream,
    shape: ClassifyShape,
    ids: DeviceBuffer,
    weights: DeviceBuffer,
    slot_row: DeviceBuffer,
    norm_row: DeviceBuffer,
    slot_ids_out: DeviceBuffer,
    weights_out: DeviceBuffer,
    log_out: DeviceBuffer,
    ctrl: DeviceBuffer,
    word: DeviceBuffer,
    mbox_req: DeviceBuffer,
    mbox_row: DeviceBuffer,
}

impl<'c> Rig<'c> {
    fn new(
        ctx: &'c CudaContext,
        stream: &'c DeviceStream,
        shape: ClassifyShape,
    ) -> Result<Self, EngError> {
        let (top, row) = (shape.top_k * 4, shape.hidden * 4);
        Ok(Self {
            ctx,
            stream,
            shape,
            ids: filled(ctx, 0, top)?,
            weights: filled(ctx, 0, top)?,
            // The interior is the slot row; the guard entries on both sides hold a VALID slot (0), so a kernel
            // that read one of them for an id of -2 or n_expert would classify the pick as a hit.
            slot_row: ctx.upload(&bytes_i32(&vec![0; shape.n_expert + 2 * GUARD]))?,
            norm_row: filled(ctx, 0, row)?,
            slot_ids_out: filled(ctx, POISON, top)?,
            weights_out: filled(ctx, POISON, top)?,
            log_out: filled(ctx, POISON, shape.top_k * LOG_BYTES_PER_PICK)?,
            ctrl: filled(ctx, 0, CTRL_BYTES)?,
            word: filled(ctx, 0xAA, 4)?,
            mbox_req: filled(ctx, UNTOUCHED, REQUEST_BYTES)?,
            mbox_row: filled(ctx, UNTOUCHED, row)?,
        })
    }

    fn launch(&self, input: &Layer) -> Result<Observed, EngError> {
        let s = ClassifyShape {
            layer: input.layer,
            ..self.shape
        };
        let write = |buffer: &DeviceBuffer, offset: usize, bytes: &[u8]| {
            buffer.write_at(self.stream, offset, bytes)
        };
        write(&self.ids, 0, &bytes_i32(&input.ids))?;
        write(&self.weights, 0, &bytes_f32(&input.weights))?;
        write(&self.slot_row, GUARD * 4, &bytes_i32(&input.slot_row))?;
        write(&self.norm_row, 0, &bytes_f32(&input.norm_row))?;
        for (buffer, len) in [
            (&self.slot_ids_out, s.top_k * 4),
            (&self.weights_out, s.top_k * 4),
            (&self.log_out, s.top_k * LOG_BYTES_PER_PICK),
        ] {
            write(buffer, 0, &vec![POISON; len])?;
        }
        let call = ClassifyCall {
            shape: s,
            ids: at(&self.ids, 0, s.top_k * 4)?,
            weights: at(&self.weights, 0, s.top_k * 4)?,
            slot_row: at(&self.slot_row, GUARD * 4, s.n_expert * 4)?,
            norm_row: at(&self.norm_row, 0, s.hidden * 4)?,
            slot_ids_out: at(&self.slot_ids_out, 0, s.top_k * 4)?,
            weights_out: at(&self.weights_out, 0, s.top_k * 4)?,
            log_out: at(&self.log_out, 0, s.top_k * LOG_BYTES_PER_PICK)?,
            ctrl: at(&self.ctrl, 0, CTRL_BYTES)?,
            word: at(&self.word, 0, 4)?,
            mbox_req: at(&self.mbox_req, 0, REQUEST_BYTES)?,
            mbox_row: at(&self.mbox_row, 0, s.hidden * 4)?,
        };
        self.ctx.tiered_classify(self.stream, call)?;
        self.stream.synchronize()?;
        self.observe()
    }

    /// Sets the tiered word, as the wait-add kernel's bits would (CPU_FAILED 1, WAIT_TIMEOUT 2).
    fn set_word(&self, value: i32) -> Result<(), EngError> {
        self.word.write_at(self.stream, 0, &value.to_le_bytes())
    }

    fn observe(&self) -> Result<Observed, EngError> {
        let rd = |buffer: &DeviceBuffer, len| read(self.ctx, buffer, len);
        let top = self.shape.top_k;
        let log_words = i32s(&rd(&self.log_out, top * LOG_BYTES_PER_PICK)?);
        let log: Vec<(i32, f32)> = log_words
            .as_chunks::<2>()
            .0
            .iter()
            .map(|pair| (pair[0], f32::from_bits(pair[1] as u32)))
            .collect();
        let request = rd(&self.mbox_req, REQUEST_BYTES)?;
        let count = (u64_at(&request, 1) >> 32) as usize;
        let ids = i32s(&request[16..48]);
        let weights = testkit::floats(&request[48..80]);
        let misses = (0..count.min(8)).map(|k| (ids[k], weights[k])).collect();
        let ctrl = rd(&self.ctrl, CTRL_BYTES)?;
        let classified = Classified {
            slot_ids: i32s(&rd(&self.slot_ids_out, top * 4)?),
            weights: testkit::floats(&rd(&self.weights_out, top * 4)?),
            log,
            misses,
            word: i32s(&rd(&self.word, 4)?)[0],
        };
        Ok(Observed {
            classified,
            count: u64_at(&ctrl, 0),
            expect_seq: u64_at(&ctrl, 1),
            seq_counter: u64_at(&ctrl, 2),
            request,
            row: testkit::floats(&rd(&self.mbox_row, self.shape.hidden * 4)?),
        })
    }
}

/// THE device phase of a script: a fresh context, stream and rig, the initial state, then every planned launch in
/// order; everything is dropped before the records come back.
fn drive(shape: ClassifyShape, script: &[Planned]) -> Result<(Observed, Vec<Observed>), EngError> {
    let ctx = CudaContext::new()?;
    let stream = ctx.stream()?;
    let rig = Rig::new(&ctx, &stream, shape)?;
    let initial = rig.observe()?;
    let mut seen = Vec::with_capacity(script.len());
    for planned in script {
        if let Some(value) = planned.set_word {
            rig.set_word(value)?;
        }
        seen.push(rig.launch(&planned.input)?);
    }
    Ok((initial, seen))
}

/// Plans one launch: the oracle's answer for the layer, and the token's word after it.
fn plan(shape: &ClassifyShape, input: Layer, set_word: Option<i32>, word: &mut i32) -> Planned {
    let want = classify_reference(&input.slot_row, shape.slots, &input.ids, &input.weights)
        .expect("equal lengths");
    if let Some(value) = set_word {
        *word = value;
    }
    *word = word_after(input.layer, *word, want.word);
    Planned {
        input,
        set_word,
        want,
        want_word: *word,
    }
}

/// A random slot row: `slots` experts resident in distinct slots, the rest -1.
fn random_slot_row(rng: &mut Rng, n_expert: usize, slots: usize) -> Vec<i32> {
    let mut row = vec![-1; n_expert];
    let mut free: Vec<i32> = (0..slots as i32).collect();
    while !free.is_empty() {
        let expert = rng.below(n_expert as u64) as usize;
        if row[expert] == -1 {
            let at = rng.below(free.len() as u64) as usize;
            row[expert] = free.swap_remove(at);
        }
    }
    row
}

/// `count` distinct experts drawn from `pool` (from every expert when `pool` is empty).
fn distinct(rng: &mut Rng, n_expert: usize, count: usize, pool: &[usize]) -> Vec<i32> {
    let mut ids: Vec<i32> = Vec::new();
    while ids.len() < count {
        let id = if pool.is_empty() {
            rng.below(n_expert as u64) as usize
        } else {
            pool[rng.below(pool.len() as u64) as usize]
        } as i32;
        if !ids.contains(&id) {
            ids.push(id);
        }
    }
    ids
}

/// The categories of [`random_layer`], one in twelve each.
const CATEGORIES: usize = 12;

/// Random picks of one layer, by `route % 12`: 0 valid, 1 whole fault, 2 a fault among valid picks, 3 a pick
/// twice, 4 an id past the layer, 5 an id below -1, 6 a NaN weight, 7 a +Inf weight, 8 a -Inf weight, 9 a
/// corrupt table entry (at or past the bank, or below -1), 10 resident experts only, 11 missing experts only.
fn random_layer(rng: &mut Rng, shape: &ClassifyShape, route: usize) -> Layer {
    let (n, top) = (shape.n_expert, shape.top_k);
    let mut slot_row = random_slot_row(rng, n, shape.slots.min(n / 2));
    let resident: Vec<usize> = (0..n).filter(|&e| slot_row[e] >= 0).collect();
    let missing: Vec<usize> = (0..n).filter(|&e| slot_row[e] < 0).collect();
    let category = route % CATEGORIES;
    let pool: &[usize] = match category {
        10 => &resident,
        11 => &missing,
        _ => &[],
    };
    let mut ids = distinct(rng, n, top, pool);
    let mut weights: Vec<f32> = ids
        .iter()
        .map(|&id| 0.05 + (id % 7) as f32 * 0.03)
        .collect();
    let victim = (route / CATEGORIES) % top;
    match category {
        1 => {
            ids.iter_mut().for_each(|id| *id = -1);
            weights.iter_mut().for_each(|w| *w = 0.0);
        }
        2 => ids[victim] = -1,
        3 => ids[top - 1] = ids[0],
        4 => ids[victim] = n as i32,
        5 => ids[victim] = -2,
        6 => weights[victim] = f32::NAN,
        7 => weights[victim] = f32::INFINITY,
        8 => weights[victim] = f32::NEG_INFINITY,
        9 => {
            let expert = ids[victim] as usize;
            slot_row[expert] = if route % 24 == 9 {
                shape.slots as i32
            } else {
                -2
            };
        }
        _ => {}
    }
    let norm_row = (0..shape.hidden).map(|_| rng.normal() as f32).collect();
    Layer {
        layer: (route / 3) % 40,
        ids,
        weights,
        slot_row,
        norm_row,
    }
}

/// The seeded script of `routes` routes: every seventh route first sets a bit another kernel would have set.
fn seeded_script(shape: &ClassifyShape, seed: u64, routes: usize) -> Vec<Planned> {
    let mut rng = Rng::new(seed);
    let mut word = 0xAAAA_AAAAu32 as i32;
    (0..routes)
        .map(|route| {
            let input = random_layer(&mut rng, shape, route);
            let set_word = (route % 7 == 3).then(|| word | (1 << (route % 2)));
            plan(shape, input, set_word, &mut word)
        })
        .collect()
}

/// Seam 1: the kernel equals the host oracle bit for bit over seeded routes of two shapes, refused routes
/// included, and publishes exactly when the layer has a miss.
#[test]
fn tiered_classify_matches_the_reference_bit_for_bit() {
    if !testkit::device_present("tiered_classify_matches_the_reference_bit_for_bit") {
        return;
    }
    let shapes = [
        (
            ClassifyShape {
                n_expert: 16,
                slots: 8,
                top_k: 8,
                hidden: 256,
                layer: 0,
            },
            0x7CA9_0101u64,
        ),
        (
            ClassifyShape {
                n_expert: 256,
                slots: 31,
                top_k: 8,
                hidden: 2048,
                layer: 0,
            },
            0x7CA9_0102,
        ),
    ];
    for (shape, seed) in shapes {
        let script = seeded_script(&shape, seed, 2_880);
        let (initial, seen) = drive(shape, &script).expect("the device phase");
        let (mut seq, mut published) = (0u64, 0usize);
        let (mut previous_req, mut previous_row) = (initial.request, initial.row);
        let (mut all_hit, mut all_miss) = (0usize, 0usize);
        for (route, (planned, got)) in script.iter().zip(&seen).enumerate() {
            let (input, want) = (&planned.input, &planned.want);
            let at = format!(
                "shape {} route {route} layer {}",
                shape.n_expert, input.layer
            );
            assert_eq!(got.classified.slot_ids, want.slot_ids, "{at}: slot ids");
            assert_eq!(
                bits(&got.classified.weights),
                bits(&want.weights),
                "{at}: weights"
            );
            assert!(
                got.classified.log.len() == want.log.len()
                    && got
                        .classified
                        .log
                        .iter()
                        .zip(&want.log)
                        .all(|(g, w)| g.0 == w.0 && g.1.to_bits() == w.1.to_bits()),
                "{at}: log"
            );
            assert_eq!(got.classified.word, planned.want_word, "{at}: tiered word");
            assert_eq!(got.count, want.misses.len() as u64, "{at}: miss count");
            let clean = want.word == 0 && !input.ids.contains(&-1);
            all_hit += usize::from(clean && want.misses.is_empty());
            all_miss += usize::from(clean && want.misses.len() == shape.top_k);
            if want.misses.is_empty() {
                assert_eq!(
                    got.request, previous_req,
                    "{at}: no miss, the request line is untouched"
                );
                assert_eq!(
                    bits(&got.row),
                    bits(&previous_row),
                    "{at}: no miss, the row is untouched"
                );
                assert_eq!(got.seq_counter, seq, "{at}: the sequence does not move");
                continue;
            }
            seq += 1;
            published += 1;
            assert_eq!(
                (got.seq_counter, got.expect_seq),
                (seq, seq),
                "{at}: sequence"
            );
            assert_eq!(u64_at(&got.request, 0), seq, "{at}: published sequence");
            assert_eq!(
                u64_at(&got.request, 1) & 0xFFFF_FFFF,
                input.layer as u64,
                "{at}: layer"
            );
            assert_eq!(
                got.classified.misses.len(),
                want.misses.len(),
                "{at}: miss list length"
            );
            assert!(
                got.classified
                    .misses
                    .iter()
                    .zip(&want.misses)
                    .all(|(g, w)| g.0 == w.0 && g.1.to_bits() == w.1.to_bits()),
                "{at}: miss list in pick order"
            );
            let tail_ids = i32s(&got.request[16 + 4 * want.misses.len()..48]);
            let tail_weights = bits(&testkit::floats(
                &got.request[48 + 4 * want.misses.len()..80],
            ));
            assert!(
                tail_ids.iter().all(|&id| id == -1),
                "{at}: unused ids are -1"
            );
            assert!(
                tail_weights.iter().all(|&w| w == 0),
                "{at}: unused weights are +0.0"
            );
            assert_eq!(
                bits(&got.row),
                bits(&input.norm_row),
                "{at}: the published row"
            );
            previous_req = got.request.clone();
            previous_row = got.row.clone();
        }
        let n = shape.n_expert;
        assert!(
            published > 100,
            "shape {n}: the run published {published} times"
        );
        assert!(
            all_hit >= 10,
            "shape {n}: only {all_hit} rows were resident only"
        );
        assert!(
            all_miss >= 10,
            "shape {n}: only {all_miss} rows were missing only"
        );
    }
}

/// A scripted token: a layer with a miss, a layer with none, a layer with a miss again. The sequence counts
/// the publications only, and the request line carries the layer.
#[test]
fn tiered_classify_publishes_only_with_misses_and_numbers_its_requests() {
    if !testkit::device_present(
        "tiered_classify_publishes_only_with_misses_and_numbers_its_requests",
    ) {
        return;
    }
    let shape = ClassifyShape {
        n_expert: 16,
        slots: 8,
        top_k: 4,
        hidden: 256,
        layer: 0,
    };
    let mut rng = Rng::new(0x7CA9_0103);
    let resident_row: Vec<i32> = (0..16).map(|e| if e < 8 { e } else { -1 }).collect();
    let mut word = 0;
    let mut layer = |layer: usize, ids: [i32; 4]| {
        let input = Layer {
            layer,
            ids: ids.to_vec(),
            weights: vec![0.4, 0.3, 0.2, 0.1],
            slot_row: resident_row.clone(),
            norm_row: (0..256).map(|_| rng.normal() as f32).collect(),
        };
        plan(&shape, input, None, &mut word)
    };
    let script = [
        layer(0, [0, 1, 2, 3]),
        layer(1, [9, 1, 12, 3]),
        layer(2, [4, 5, 6, 7]),
        layer(3, [15, 1, 2, 3]),
    ];
    let (_, seen) = drive(shape, &script).expect("the device phase");
    let (all_hit, first, none_again, second) = (&seen[0], &seen[1], &seen[2], &seen[3]);
    assert_eq!((all_hit.count, all_hit.seq_counter), (0, 0));
    assert!(
        all_hit.request.iter().all(|&b| b == UNTOUCHED),
        "the request line is untouched"
    );
    assert!(
        all_hit.row.iter().all(|v| v.to_bits() == 0x7777_7777),
        "the row is untouched"
    );
    assert_eq!(
        (first.count, first.seq_counter, first.expect_seq),
        (2, 1, 1)
    );
    assert_eq!(u64_at(&first.request, 0), 1);
    assert_eq!(first.classified.misses, vec![(9, 0.4), (12, 0.2)]);
    assert_eq!(bits(&first.row), bits(&script[1].input.norm_row));
    assert_eq!(
        (none_again.count, none_again.seq_counter),
        (0, 1),
        "the sequence waits for the next miss"
    );
    assert_eq!(none_again.request, first.request, "the old request stays");
    assert_eq!(
        (second.count, second.seq_counter, second.expect_seq),
        (1, 2, 2)
    );
    assert_eq!(u64_at(&second.request, 0), 2);
    assert_eq!(
        u64_at(&second.request, 1),
        (1u64 << 32) | 3,
        "count 1 in the high word, layer 3 in the low"
    );
}

/// The tiered word clears at layer 0 and ORs after: a refused route in layer 1 is still there after a clean
/// layer 2, a bit another kernel set is kept by a later refused layer, and layer 0 of the next token starts over.
#[test]
fn the_tiered_word_clears_at_layer_zero_and_ors_after() {
    if !testkit::device_present("the_tiered_word_clears_at_layer_zero_and_ors_after") {
        return;
    }
    let shape = ClassifyShape {
        n_expert: 16,
        slots: 8,
        top_k: 4,
        hidden: 256,
        layer: 0,
    };
    let row: Vec<i32> = (0..16).map(|e| if e < 8 { e } else { -1 }).collect();
    let mut word = 0xAAAA_AAAAu32 as i32;
    let mut step = |layer: usize, ids: [i32; 4], set_word: Option<i32>| {
        let input = Layer {
            layer,
            ids: ids.to_vec(),
            weights: vec![0.4, 0.3, 0.2, 0.1],
            slot_row: row.clone(),
            norm_row: vec![1.0; 256],
        };
        plan(&shape, input, set_word, &mut word)
    };
    let script = [
        step(0, [0, 1, 2, 3], None),
        step(1, [0, 1, 1, 3], None),
        step(2, [0, 1, 2, 3], None),
        step(3, [0, 1, 1, 3], Some(1)),
        step(0, [0, 1, 2, 3], None),
    ];
    let (_, seen) = drive(shape, &script).expect("the device phase");
    let words: Vec<i32> = seen.iter().map(|o| o.classified.word).collect();
    assert_eq!(
        words[0], 0,
        "the 0xAAAAAAAA of the start is cleared by layer 0"
    );
    assert_eq!(words[1], 4, "a pick twice");
    assert_eq!(words[2], 4, "a clean layer keeps the bit");
    assert_eq!(
        words[3],
        1 | 4,
        "a refused route ORs onto a bit another kernel set, it does not replace it"
    );
    assert_eq!(words[4], 0, "layer 0 of the next token starts over");
}

/// The spare buffers of the refusal phase: eleven allocations of 4,096 bytes, in the order of the call's ranges.
struct Spare(Vec<DeviceBuffer>);

/// One edit of a call: the index of the range, its offset in its allocation and its length (`None`: exact).
type Edit = (usize, usize, Option<usize>);

/// The call of the refusal phase: every range exact and aligned except the `edits`.
fn call<'b>(
    spare: &'b Spare,
    shape: ClassifyShape,
    edits: &[Edit],
) -> Result<ClassifyCall<'b>, EngError> {
    let (top, row) = (shape.top_k * 4, shape.hidden * 4);
    let sized = [
        top,
        top,
        shape.n_expert * 4,
        row,
        top,
        top,
        shape.top_k * LOG_BYTES_PER_PICK,
        CTRL_BYTES,
        4,
        REQUEST_BYTES,
        row,
    ];
    let r = |i: usize| {
        let edit = edits.iter().find(|e| e.0 == i);
        let (offset, bytes) = edit.map_or((0, sized[i]), |e| (e.1, e.2.unwrap_or(sized[i])));
        at(&spare.0[i], offset, bytes)
    };
    Ok(ClassifyCall {
        shape,
        ids: r(0)?,
        weights: r(1)?,
        slot_row: r(2)?,
        norm_row: r(3)?,
        slot_ids_out: r(4)?,
        weights_out: r(5)?,
        log_out: r(6)?,
        ctrl: r(7)?,
        word: r(8)?,
        mbox_req: r(9)?,
        mbox_row: r(10)?,
    })
}

/// The cases of the refusal test: a name, the edits, and the text the refusal must contain (`None`: accepted).
fn refusal_cases() -> Vec<(&'static str, Vec<Edit>, Option<&'static str>)> {
    vec![
        ("base", vec![], None),
        (
            "short slot_row",
            vec![(2, 0, Some(60))],
            Some("the CUDA slot_row range holds 60 bytes but the call needs 64"),
        ),
        (
            "short ids",
            vec![(0, 0, Some(12))],
            Some("the CUDA ids range holds 12 bytes but the call needs 16"),
        ),
        (
            "short ctrl",
            vec![(7, 0, Some(32))],
            Some("the CUDA ctrl range holds 32 bytes but the call needs 64"),
        ),
        (
            "short word",
            vec![(8, 0, Some(2))],
            Some("the CUDA word range holds 2 bytes but the call needs 4"),
        ),
        (
            "misaligned ids",
            vec![(0, 2, None)],
            Some("the CUDA ids range starts at offset 2, which is not a multiple of 4"),
        ),
        (
            "misaligned mbox_req",
            vec![(9, 4, None)],
            Some("the CUDA mbox_req range starts at offset 4, which is not a multiple of 8"),
        ),
        (
            "misaligned mbox_row",
            vec![(10, 8, None)],
            Some("the CUDA mbox_row range starts at offset 8, which is not a multiple of 16"),
        ),
    ]
}

/// The device phase of the refusal test: each case is one call; the result is the error text, or `None` when the
/// call went through. Then the two overlap cases, which need ranges in the same allocation.
fn refusal_phase(shape: ClassifyShape) -> Result<Vec<Option<String>>, EngError> {
    let ctx = CudaContext::new()?;
    let stream = ctx.stream()?;
    let spare = Spare(
        (0..11)
            .map(|_| filled(&ctx, 0, 4096))
            .collect::<Result<_, _>>()?,
    );
    let text = |result: Result<(), EngError>| result.err().map(|e| format!("{e:?}"));
    let mut out = Vec::new();
    for (_, edits, _) in refusal_cases() {
        out.push(text(
            ctx.tiered_classify(&stream, call(&spare, shape, &edits)?),
        ));
    }
    // The request line placed 8 bytes into the ids allocation (the ids are 16 bytes), and the mailbox row over the normed row.
    let mut over_ids = call(&spare, shape, &[])?;
    over_ids.mbox_req = at(&spare.0[0], 8, REQUEST_BYTES)?;
    out.push(text(ctx.tiered_classify(&stream, over_ids)));
    let mut over_row = call(&spare, shape, &[])?;
    over_row.mbox_row = at(&spare.0[3], 0, shape.hidden * 4)?;
    out.push(text(ctx.tiered_classify(&stream, over_row)));
    stream.synchronize()?;
    Ok(out)
}

/// The wrapper refuses a short range, a misaligned range and a written range over another range before any
/// launch, each with its own text, and accepts the exact call.
#[test]
fn the_wrapper_refuses_bad_ranges_with_their_own_text() {
    if !testkit::device_present("the_wrapper_refuses_bad_ranges_with_their_own_text") {
        return;
    }
    let shape = ClassifyShape {
        n_expert: 16,
        slots: 8,
        top_k: 4,
        hidden: 256,
        layer: 0,
    };
    let got = refusal_phase(shape).expect("the device phase");
    let mut cases: Vec<(&str, Option<&str>)> = refusal_cases()
        .into_iter()
        .map(|(name, _, text)| (name, text))
        .collect();
    cases.push((
        "mbox_req over ids",
        Some("the CUDA mbox_req and ids ranges overlap"),
    ));
    cases.push((
        "mbox_row over norm_row",
        Some("the CUDA mbox_row and norm_row ranges overlap"),
    ));
    assert_eq!(got.len(), cases.len());
    for (text, (name, want)) in got.iter().zip(cases) {
        match (text, want) {
            (None, None) => {}
            (Some(text), Some(want)) => assert!(text.contains(want), "{name}: {text}"),
            (text, want) => panic!("{name}: got {text:?}, wanted {want:?}"),
        }
    }
}

/// Host only: the shape limits, each side of each one, with the text that names the values.
#[test]
fn check_shape_refuses_every_limit_and_names_the_values() {
    let good = ClassifyShape {
        n_expert: 256,
        slots: 31,
        top_k: 8,
        hidden: 2048,
        layer: 39,
    };
    assert!(check_shape(&good).is_ok());
    let edges = [
        (
            "n_expert 0",
            ClassifyShape {
                n_expert: 0,
                ..good
            },
        ),
        (
            "n_expert 4097",
            ClassifyShape {
                n_expert: 4097,
                ..good
            },
        ),
        ("slots 0", ClassifyShape { slots: 0, ..good }),
        ("slots past n_expert", ClassifyShape { slots: 257, ..good }),
        ("top_k 0", ClassifyShape { top_k: 0, ..good }),
        ("top_k 9", ClassifyShape { top_k: 9, ..good }),
        ("hidden 0", ClassifyShape { hidden: 0, ..good }),
        ("hidden 6", ClassifyShape { hidden: 6, ..good }),
        (
            "hidden 65540",
            ClassifyShape {
                hidden: 65540,
                ..good
            },
        ),
        (
            "layer 1024",
            ClassifyShape {
                layer: 1024,
                ..good
            },
        ),
        (
            "one expert, two slots",
            ClassifyShape {
                n_expert: 1,
                slots: 2,
                ..good
            },
        ),
        ("hidden 3", ClassifyShape { hidden: 3, ..good }),
    ];
    for (name, shape) in edges {
        let text = format!("{:?}", check_shape(&shape).expect_err(name));
        assert!(
            text.contains(&format!("n_expert {}", shape.n_expert)),
            "{name}: {text}"
        );
        assert!(
            text.contains(&format!("hidden {}", shape.hidden)),
            "{name}: {text}"
        );
    }
    let edge_ok = [
        ClassifyShape {
            n_expert: 4096,
            slots: 4096,
            top_k: 8,
            hidden: 65536,
            layer: 1023,
        },
        ClassifyShape {
            n_expert: 1,
            slots: 1,
            top_k: 1,
            hidden: 4,
            layer: 0,
        },
    ];
    for shape in edge_ok {
        assert!(check_shape(&shape).is_ok(), "{shape:?}");
    }
}
