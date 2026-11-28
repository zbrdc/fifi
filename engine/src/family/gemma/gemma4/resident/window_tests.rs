//! Device tests of the batched window: the kernels that make a row of a window the same bits as
//! the row alone, and the model step itself against solo runs. Run with `--ignored` on a device
//! with `--test-threads=1`; the model test reads the Q8_0 file from `BOLT_Q8_FILE`.

use super::bolt::{
    ATTN_SPLIT_MAX, ATTN_SPLIT_MIN_KEYS, Bolt, HEAD_COLUMNS, VOCAB, attn_part_bytes,
};

/// The cells of the caches these tests build themselves; the model under test plans its own.
const CELLS: usize = 32_768;
use crate::kernels::cuda::resident::attn_decode_gemma::{
    AttnDecodeGemmaCall, AttnDecodeGemmaSplit,
};
use crate::kernels::cuda::resident::kv8_append::{Kv8CodeScale, Kv8Data, Kv8Shape};
use crate::kernels::cuda::resident::matvec_q::{MatvecCall, QuantType};
use crate::kernels::cuda::resident::quantize_q8_1::{QuantizeCall, q8_1_bytes};
use crate::kernels::cuda::resident::rope_gemma::RopeGemmaCall;
use crate::kernels::cuda::{CudaContext, DeviceBuffer, DeviceStream};

/// A small deterministic generator, so a failing run reproduces.
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.0 >> 33
    }

    fn byte(&mut self) -> u8 {
        self.next() as u8
    }

    /// A float in [-1, 1).
    fn unit(&mut self) -> f32 {
        (self.next() % 2_000_001) as f32 / 1_000_000.0 - 1.0
    }
}

fn context() -> (CudaContext, DeviceStream) {
    let ctx = CudaContext::with_admission(&crate::family::project::cuda::served_admission())
        .expect("context");
    let stream = ctx.stream().expect("stream");
    (ctx, stream)
}

fn floats(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn download(ctx: &CudaContext, buffer: &DeviceBuffer, bytes: usize) -> Vec<u8> {
    let mut out = vec![0u8; bytes];
    ctx.download(buffer, &mut out).expect("download");
    out
}

/// Activations that stress the quantizer and the sums: ordinary values, one column scaled up,
/// one all zero, one near-constant (near ties), one with a few huge entries.
fn activation_columns(rng: &mut Lcg, columns: usize, k: usize) -> Vec<f32> {
    let mut x = Vec::with_capacity(columns * k);
    for column in 0..columns {
        for index in 0..k {
            let base = rng.unit();
            x.push(match column % 5 {
                0 => base,
                1 => base * 400.0,
                2 => 0.0,
                3 => 0.5 + base * 1e-4,
                _ => {
                    if index % 97 == 0 {
                        base * 5_000.0
                    } else {
                        base * 0.01
                    }
                }
            });
        }
    }
    x
}

/// The head of a window runs the vendored Q8_0 matvec. Column j of a launch of up to
/// `HEAD_COLUMNS` columns must equal the lone column; the wider launches are reported.
#[test]
#[ignore = "requires a device; use --test-threads=1"]
fn the_head_matvec_columns_equal_a_lone_column_up_to_the_head_width() {
    let (ctx, stream) = context();
    let mut rng = Lcg(11);
    let k = 2560;
    for rows in [4096usize, VOCAB] {
        // 4,096 random rows repeated down the matrix keep the host fill cheap; the block scales
        // are fixed to finite halves so no row sums to NaN.
        let row_bytes = (k / 32) * 34;
        let total = rows * row_bytes;
        let mut weights: Vec<u8> = (0..rows.min(4096) * row_bytes)
            .map(|_| rng.byte())
            .collect();
        for block in weights.as_chunks_mut::<34>().0 {
            block[0..2].copy_from_slice(&0x2c00u16.to_le_bytes());
        }
        while weights.len() < total {
            let more = weights.len().min(total - weights.len());
            weights.extend_from_within(..more);
        }
        let w = ctx.upload(&weights).expect("weights");
        let x = ctx
            .upload(&floats(&activation_columns(&mut rng, 8, k)))
            .expect("activations");
        let q8_row = q8_1_bytes(k, 1).expect("row bytes");
        let q8 = ctx.upload(&vec![0u8; q8_row * 8]).expect("q8");
        ctx.quantize_q8_1(
            &stream,
            QuantizeCall {
                x: x.range(0, 8 * k * 4).expect("x"),
                y: q8.range(0, q8_row * 8).expect("y"),
                k,
                ncols: 8,
            },
        )
        .expect("quantize");
        let launch = |first: usize, columns: usize, y: &DeviceBuffer| {
            ctx.matvec_q(
                &stream,
                MatvecCall {
                    ty: QuantType::Q80,
                    w: w.range(0, w.bytes()).expect("w"),
                    x_q8: q8
                        .range(first * q8_row, q8_1_bytes(k, columns).expect("bytes"))
                        .expect("x"),
                    y: y.range(0, rows * columns * 4).expect("y"),
                    k,
                    rows,
                    ncols: columns,
                },
            )
            .expect("launch");
            stream.synchronize().expect("sync");
        };
        let single = ctx.upload(&vec![0u8; rows * 4]).expect("single");
        let wide = ctx.upload(&vec![0u8; rows * 8 * 4]).expect("wide");
        let alone: Vec<Vec<u8>> = (0..8)
            .map(|column| {
                launch(column, 1, &single);
                download(&ctx, &single, rows * 4)
            })
            .collect();
        for columns in 2..=8 {
            launch(0, columns, &wide);
            let got = download(&ctx, &wide, rows * 8 * 4);
            let mut differing = Vec::new();
            for column in 0..columns {
                if got[column * rows * 4..(column + 1) * rows * 4] != alone[column][..] {
                    differing.push(column);
                }
            }
            eprintln!(
                "head rows {rows}: {columns} columns, columns differing from alone {differing:?}"
            );
            if columns <= HEAD_COLUMNS {
                assert!(
                    differing.is_empty(),
                    "rows {rows}: a {columns}-column launch changes columns {differing:?}"
                );
            }
        }
    }
}

struct Planes {
    k_codes: DeviceBuffer,
    k_scales: DeviceBuffer,
    v_codes: DeviceBuffer,
    v_scales: DeviceBuffer,
}

fn random_planes(ctx: &CudaContext, head: usize, seed: u64) -> (Planes, Vec<Vec<u8>>) {
    let mut rng = Lcg(seed);
    let codes = 2 * CELLS * head;
    let scales = 2 * CELLS * (head / 32);
    let host: Vec<Vec<u8>> = vec![
        (0..codes).map(|_| rng.byte()).collect(),
        (0..scales)
            .flat_map(|_| (0x2800u16 + (rng.next() % 0x400) as u16).to_le_bytes())
            .collect(),
        (0..codes).map(|_| rng.byte()).collect(),
        (0..scales)
            .flat_map(|_| (0x2800u16 + (rng.next() % 0x400) as u16).to_le_bytes())
            .collect(),
    ];
    let planes = Planes {
        k_codes: ctx.upload(&host[0]).expect("k codes"),
        k_scales: ctx.upload(&host[1]).expect("k scales"),
        v_codes: ctx.upload(&host[2]).expect("v codes"),
        v_scales: ctx.upload(&host[3]).expect("v scales"),
    };
    (planes, host)
}

/// Per-slot rotation, cache append and attention in a window against the same launches on one
/// row alone, at both head widths, with the split attention off and on, at positions that cross
/// the sliding window and the split thresholds. The appended cells and the context rows must
/// match byte for byte.
#[test]
#[ignore = "requires a device; use --test-threads=1"]
#[allow(
    clippy::needless_range_loop,
    reason = "the slot indexes several parallel per-slot collections"
)]
fn per_slot_rope_append_and_attention_equal_the_solo_launches() {
    let (ctx, stream) = context();
    let positions = [0usize, 1, 37, 511, 512, 513, 4000, 30_000];
    let rows = positions.len();
    let part = ctx
        .upload(&vec![0u8; attn_part_bytes(ATTN_SPLIT_MAX, 8)])
        .expect("part");
    for head in [256usize, 512] {
        let global = head == 512;
        let (q_heads, kv_heads) = (8usize, 2usize);
        let q_row = q_heads * head;
        let kv_row = kv_heads * head;
        let mut rng = Lcg(head as u64);
        let q_host: Vec<f32> = (0..rows * q_row).map(|_| rng.unit()).collect();
        let k_host: Vec<f32> = (0..rows * kv_row).map(|_| rng.unit()).collect();
        let v_host: Vec<f32> = (0..rows * kv_row).map(|_| rng.unit()).collect();
        let factors_host: Vec<f32> = (0..256).map(|i| 1.0 + i as f32 / 64.0).collect();
        let factors = ctx.upload(&floats(&factors_host)).expect("factors");
        let shape = Kv8Shape::linear(1, kv_heads, head, CELLS, Kv8CodeScale::StoredHalf);
        for armed in [false, true] {
            // The window: one q/k/v buffer of all rows, one planes set and word pair per slot.
            let q_win = ctx.upload(&floats(&q_host)).expect("q");
            let k_win = ctx.upload(&floats(&k_host)).expect("k");
            let v_win = ctx.upload(&floats(&v_host)).expect("v");
            let out_win = ctx.upload(&vec![0u8; rows * q_row * 4]).expect("out");
            let status = ctx.upload(&[0u8; 4]).expect("status");
            let mut win_planes = Vec::new();
            let mut solo_planes = Vec::new();
            for slot in 0..rows {
                let (win, _) = random_planes(&ctx, head, 100 + slot as u64);
                let (solo, _) = random_planes(&ctx, head, 100 + slot as u64);
                win_planes.push(win);
                solo_planes.push(solo);
            }
            let words: Vec<(DeviceBuffer, DeviceBuffer)> = positions
                .iter()
                .map(|&pos| {
                    (
                        ctx.upload(&(pos as i32).to_le_bytes()).expect("pos"),
                        ctx.upload(&(pos as i32 + 1).to_le_bytes()).expect("kv_len"),
                    )
                })
                .collect();
            let split = |armed: bool| {
                armed.then(|| AttnDecodeGemmaSplit {
                    part: part
                        .range(0, attn_part_bytes(ATTN_SPLIT_MAX, 8))
                        .expect("part"),
                    max_splits: ATTN_SPLIT_MAX,
                    min_keys: ATTN_SPLIT_MIN_KEYS,
                })
            };
            // One slot of the window, or the lone row of a solo run: rotate, append, attend.
            let run = |q: &DeviceBuffer,
                       k: &DeviceBuffer,
                       v: &DeviceBuffer,
                       out: &DeviceBuffer,
                       row: usize,
                       planes: &Planes,
                       slot: usize| {
                ctx.rope_gemma(
                    &stream,
                    RopeGemmaCall {
                        q: q.range(row * q_row * 4, q_row * 4).expect("q"),
                        k: k.range(row * kv_row * 4, kv_row * 4).expect("k"),
                        pos: words[slot].0.range(0, 4).expect("pos"),
                        factors: global.then(|| factors.range(0, 256 * 4).expect("factors")),
                        rows_q: 1,
                        rows_k: 1,
                        n_q_head: q_heads,
                        n_kv_head: kv_heads,
                        head_dim: head,
                        global_mode: global,
                    },
                )
                .expect("rope");
                ctx.kv8_append(
                    &stream,
                    shape,
                    Kv8Data {
                        k: k.range(row * kv_row * 4, kv_row * 4).expect("k"),
                        v: v.range(row * kv_row * 4, kv_row * 4).expect("v"),
                        k_codes: planes.k_codes.range(0, planes.k_codes.bytes()).expect("kc"),
                        k_scales: planes
                            .k_scales
                            .range(0, planes.k_scales.bytes())
                            .expect("ks"),
                        v_codes: planes.v_codes.range(0, planes.v_codes.bytes()).expect("vc"),
                        v_scales: planes
                            .v_scales
                            .range(0, planes.v_scales.bytes())
                            .expect("vs"),
                    },
                    words[slot].0.range(0, 4).expect("pos"),
                    status.range(0, 4).expect("status"),
                )
                .expect("append");
                ctx.attn_decode_gemma(
                    &stream,
                    AttnDecodeGemmaCall {
                        out: out.range(row * q_row * 4, q_row * 4).expect("out"),
                        status: None,
                        q: q.range(row * q_row * 4, q_row * 4).expect("q"),
                        k_codes: planes.k_codes.range(0, planes.k_codes.bytes()).expect("kc"),
                        k_scales: planes
                            .k_scales
                            .range(0, planes.k_scales.bytes())
                            .expect("ks"),
                        v_codes: planes.v_codes.range(0, planes.v_codes.bytes()).expect("vc"),
                        v_scales: planes
                            .v_scales
                            .range(0, planes.v_scales.bytes())
                            .expect("vs"),
                        kv_len: words[slot].1.range(0, 4).expect("kv_len"),
                        m: 1,
                        n_q_head: q_heads,
                        n_kv_head: kv_heads,
                        head_dim: head,
                        cells: CELLS,
                        window: if global { 0 } else { 512 },
                        scale: 1.0,
                        split: split(armed),
                        ring: 0,
                    },
                )
                .expect("attention");
            };
            for slot in 0..rows {
                run(
                    &q_win,
                    &k_win,
                    &v_win,
                    &out_win,
                    slot,
                    &win_planes[slot],
                    slot,
                );
            }
            stream.synchronize().expect("window drain");
            let win_out = download(&ctx, &out_win, rows * q_row * 4);
            for slot in 0..rows {
                // The solo run: the row alone in its own buffers.
                let q = ctx
                    .upload(&floats(&q_host[slot * q_row..(slot + 1) * q_row]))
                    .expect("q");
                let k = ctx
                    .upload(&floats(&k_host[slot * kv_row..(slot + 1) * kv_row]))
                    .expect("k");
                let v = ctx
                    .upload(&floats(&v_host[slot * kv_row..(slot + 1) * kv_row]))
                    .expect("v");
                let out = ctx.upload(&vec![0u8; q_row * 4]).expect("out");
                run(&q, &k, &v, &out, 0, &solo_planes[slot], slot);
                stream.synchronize().expect("solo drain");
                assert_eq!(
                    download(&ctx, &out, q_row * 4),
                    win_out[slot * q_row * 4..(slot + 1) * q_row * 4],
                    "head {head} split {armed} slot {slot} position {}: context row",
                    positions[slot]
                );
                // The appended cell of every plane, per kv head.
                let pos = positions[slot];
                for kv_head in 0..kv_heads {
                    let code_at = (kv_head * CELLS + pos) * head;
                    let scale_at = (kv_head * CELLS + pos) * (head / 32) * 2;
                    for (name, win, solo, at, len) in [
                        (
                            "k codes",
                            &win_planes[slot].k_codes,
                            &solo_planes[slot].k_codes,
                            code_at,
                            head,
                        ),
                        (
                            "v codes",
                            &win_planes[slot].v_codes,
                            &solo_planes[slot].v_codes,
                            code_at,
                            head,
                        ),
                        (
                            "k scales",
                            &win_planes[slot].k_scales,
                            &solo_planes[slot].k_scales,
                            scale_at,
                            head / 16,
                        ),
                        (
                            "v scales",
                            &win_planes[slot].v_scales,
                            &solo_planes[slot].v_scales,
                            scale_at,
                            head / 16,
                        ),
                    ] {
                        let (mut a, mut b) = (vec![0u8; len], vec![0u8; len]);
                        win.read_at(&stream, at, &mut a).expect("window cell");
                        solo.read_at(&stream, at, &mut b).expect("solo cell");
                        assert_eq!(a, b, "head {head} slot {slot} {name} kv head {kv_head}");
                    }
                }
            }
            eprintln!("head {head} split {armed}: {rows} slots equal their solo launches");
        }
    }
}

/// The deterministic prompt of slot `slot`: ids that the vocabulary holds, different per slot.
fn prompt_of(slot: usize, len: usize) -> Vec<u32> {
    let mut rng = Lcg(900 + slot as u64);
    (0..len)
        .map(|_| 100 + (rng.next() % 20_000) as u32)
        .collect()
}

fn argmax(row: &[f32]) -> u32 {
    row.iter()
        .enumerate()
        .fold((0usize, f32::MIN), |(best, value), (index, &other)| {
            if other > value {
                (index, other)
            } else {
                (best, value)
            }
        })
        .0 as u32
}

fn bits(row: &[f32]) -> Vec<u32> {
    row.iter().map(|v| v.to_bits()).collect()
}

/// Feeds the prompt cache-only like the serving path: every token but the last, which stays the
/// pending decode input.
fn prefill(model: &mut Bolt, seq: &super::bolt::BoltSeq, prompt: &[u32]) {
    let (head, _) = prompt.split_at(prompt.len() - 1);
    let mut pos = 0;
    for chunk in head.chunks(Bolt::WINDOW_ROWS) {
        model
            .prefill_batch(seq, chunk, pos, false)
            .expect("prefill");
        pos += chunk.len();
    }
}

/// Leaves other tokens' cells in the planes, as a pooled session does after a finished job.
fn dirty(model: &mut Bolt, seq: &super::bolt::BoltSeq, slot: usize) {
    let junk = prompt_of(slot + 50, 301);
    prefill(model, seq, &junk);
    let mut token = junk[300];
    for pos in 300..312 {
        model.decode(seq, token, pos).expect("decode");
        token = argmax(&model.capped_logits().expect("logits"));
    }
}

const STEPS: usize = 20;

/// One slot's solo run: the logits bits and the greedy token of each of its steps.
fn solo_run(model: &mut Bolt, seq: &super::bolt::BoltSeq, prompt: &[u32]) -> Vec<(Vec<u32>, u32)> {
    prefill(model, seq, prompt);
    let mut token = *prompt.last().expect("prompt");
    let mut out = Vec::new();
    for step in 0..STEPS {
        model
            .decode(seq, token, prompt.len() - 1 + step)
            .expect("decode");
        let logits = model.capped_logits().expect("logits");
        token = argmax(&logits);
        out.push((bits(&logits), token));
    }
    out
}

/// A window of B sessions equals B solo runs, step for step and bit for bit, for windows of
/// every width, with sessions that join late and leave early, at different positions, on
/// sessions that held other data before.
#[test]
#[ignore = "requires the Q8_0 Bolt file in BOLT_Q8_FILE and a device; use --test-threads=1"]
fn a_window_of_sessions_equals_the_solo_runs() {
    let path = std::env::var("BOLT_Q8_FILE").expect("BOLT_Q8_FILE names the Q8_0 Bolt file");
    // The cache is planned for a full window of sessions: a lone-session plan fills the card.
    let file = crate::family::gguf_select::GgufFile {
        path: std::path::PathBuf::from(&path),
        bytes: std::fs::metadata(&path).expect("the Bolt file").len(),
    };
    let mut model = super::bolt::load_choosing(
        &[file],
        crate::family::gemma::gemma4::config::MaxSeqs::Count(Bolt::WINDOW_ROWS),
        &|_| {},
    )
    .expect("load");
    model
        .reserve_window(Bolt::WINDOW_ROWS)
        .expect("window buffers");
    let lengths = [5usize, 17, 1, 130, 40, 2, 9, 77];
    let mut seqs: Vec<_> = (0..Bolt::WINDOW_ROWS)
        .enumerate()
        .map(|(slot, _)| {
            // Sessions of different sizes share a window: each row reads its own cells. Slot 3
            // starts at the sliding window and grows after its prompt, as a served session does.
            let cells = match slot {
                3 => 512,
                _ if slot % 2 == 0 => 2048,
                _ => 1536,
            };
            model.new_seq_cells(cells).expect("sequence")
        })
        .collect();
    for width in [2usize, 3, 4, 5, 8] {
        let prompts: Vec<Vec<u32>> = (0..width)
            .map(|slot| prompt_of(slot, lengths[slot]))
            .collect();
        // The references first, each on a sequence that held other data.
        let mut solo = Vec::new();
        for slot in 0..width {
            dirty(&mut model, &seqs[slot], slot);
            solo.push(solo_run(&mut model, &seqs[slot], &prompts[slot]));
        }
        // The window run: slot s joins at step s % 3 and leaves after STEPS - s % 4 steps.
        for slot in 0..width {
            dirty(&mut model, &seqs[slot], slot + 20);
            prefill(&mut model, &seqs[slot], &prompts[slot]);
        }
        // The grown session keeps the cells it fed and runs in larger planes than its solo run.
        if width > 3 {
            model
                .make_room(&mut seqs[3], prompts[3].len() - 1, 800)
                .expect("the session grows");
        }
        let join: Vec<usize> = (0..width).map(|slot| slot % 3).collect();
        let last: Vec<usize> = (0..width).map(|slot| STEPS - slot % 4).collect();
        let mut token: Vec<u32> = prompts.iter().map(|p| *p.last().expect("prompt")).collect();
        let mut local = vec![0usize; width];
        let mut widest = 0;
        for global in 0..STEPS + 3 {
            let live: Vec<usize> = (0..width)
                .filter(|&slot| global >= join[slot] && local[slot] < last[slot])
                .collect();
            if live.is_empty() {
                continue;
            }
            widest = widest.max(live.len());
            let refs: Vec<&super::bolt::BoltSeq> = live.iter().map(|&slot| &seqs[slot]).collect();
            let tokens: Vec<u32> = live.iter().map(|&slot| token[slot]).collect();
            let positions: Vec<usize> = live
                .iter()
                .map(|&slot| prompts[slot].len() - 1 + local[slot])
                .collect();
            let rows = if live.len() == 1 {
                model
                    .decode(refs[0], tokens[0], positions[0])
                    .expect("solo path");
                vec![model.capped_logits().expect("logits")]
            } else {
                model
                    .decode_rows(&refs, &tokens, &positions)
                    .expect("window")
            };
            for (&slot, row) in live.iter().zip(&rows) {
                let (want_bits, want_token) = &solo[slot][local[slot]];
                assert_eq!(
                    &bits(row),
                    want_bits,
                    "width {width} slot {slot} step {} (global {global}, window of {})",
                    local[slot],
                    live.len()
                );
                token[slot] = argmax(row);
                assert_eq!(token[slot], *want_token);
                local[slot] += 1;
            }
        }
        eprintln!(
            "window width {width}: widest live window {widest}, every row equals its solo run"
        );
        assert!(widest >= width.min(2));
    }
}

/// One word per row: the hash of the logits bits, so a long run keeps no rows.
fn row_hash(row: &[f32]) -> u64 {
    row.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, v| {
        (h ^ u64::from(v.to_bits())).wrapping_mul(0x0100_0000_01b3)
    })
}

/// A sequence that grows while it sits in a window of others gives its row and leaves theirs as
/// they are: three sessions decode together for 1600 positions, one on cells of one granule that
/// grows through the window (past the sliding window of the local layers), one that stays on its
/// one granule and leaves the window at 900, one sized for the run from the start. Every row of
/// every step equals the row of the same session run alone on planes sized for the whole run.
#[test]
#[ignore = "requires the Q8_0 Bolt file in BOLT_Q8_FILE and a device; use --test-threads=1"]
fn a_sequence_grown_inside_a_window_leaves_every_row_as_its_solo_run() {
    use crate::family::cell_pool::CELL_GRANULE;
    let path = std::env::var("BOLT_Q8_FILE").expect("BOLT_Q8_FILE names the Q8_0 Bolt file");
    let file = crate::family::gguf_select::GgufFile {
        path: std::path::PathBuf::from(&path),
        bytes: std::fs::metadata(&path).expect("the Bolt file").len(),
    };
    let mut model = super::bolt::load_choosing(
        &[file],
        crate::family::gemma::gemma4::config::MaxSeqs::Count(Bolt::WINDOW_ROWS),
        &|_| {},
    )
    .expect("load");
    model.reserve_window(3).expect("window buffers");
    let steps = [1600usize, 900, 1600];
    let token_of =
        |slot: usize, pos: usize| 2 + ((pos * (17 + 6 * slot) + slot * 101) % 5000) as u32;
    // The solo runs, on planes sized for the whole run.
    let mut solo: Vec<Vec<u64>> = Vec::new();
    for (slot, &run) in steps.iter().enumerate() {
        let seq = model
            .new_seq_cells(2 * CELL_GRANULE)
            .expect("whole-run sequence");
        let mut rows = Vec::new();
        for pos in 0..run {
            model
                .decode(&seq, token_of(slot, pos), pos)
                .expect("solo step");
            rows.push(row_hash(&model.capped_logits().expect("logits")));
        }
        solo.push(rows);
    }
    // The window run: slot 0 grows through the window, slot 1 never does, slot 2 never needs to.
    let mut seqs = [
        model.new_seq_cells(CELL_GRANULE).expect("small sequence"),
        model.new_seq_cells(CELL_GRANULE).expect("small sequence"),
        model
            .new_seq_cells(2 * CELL_GRANULE)
            .expect("whole-run sequence"),
    ];
    let mut growths = 0;
    for pos in 0..steps[0] {
        let live: Vec<usize> = (0..3).filter(|&slot| pos < steps[slot]).collect();
        for &slot in &live {
            let before = seqs[slot].cells();
            model
                .make_room(&mut seqs[slot], pos, pos + 1)
                .expect("room");
            growths += usize::from(seqs[slot].cells() != before);
        }
        let refs: Vec<&super::bolt::BoltSeq> = live.iter().map(|&slot| &seqs[slot]).collect();
        let tokens: Vec<u32> = live.iter().map(|&slot| token_of(slot, pos)).collect();
        let positions = [pos; 3];
        let positions = &positions[..live.len()];
        let rows = model
            .decode_rows(&refs, &tokens, positions)
            .expect("window step");
        for (&slot, row) in live.iter().zip(&rows) {
            let want = solo[slot].get(pos).copied();
            assert_eq!(
                Some(row_hash(row)),
                want,
                "slot {slot} position {pos} (window of {}, slot 0 holds {} cells)",
                live.len(),
                seqs[0].cells()
            );
        }
    }
    assert!(growths >= 1, "a sequence grew inside the window");
    assert!(seqs[0].cells() >= steps[0]);
    assert_eq!(
        seqs[1].cells(),
        CELL_GRANULE,
        "the other small session never grew"
    );
}

/// A window of sessions that has gone past the sliding layers' ring gives, at every step and bit
/// for bit, the logits of the same window of sessions that keep every position in linear planes.
/// Windows are the default decode at four or more sessions, and the other window test never
/// writes a ring cell twice: here the prompts run past the ring's 519 cells (fed 700, 530, 900
/// and 1,100 positions), so every row's append wraps, with windows of four, two and three rows
/// and sessions that join and leave.
#[test]
#[ignore = "requires the Q8_0 Bolt file in BOLT_Q8_FILE and a device; use --test-threads=1"]
fn a_window_past_the_ring_equals_the_linear_control_window() {
    let path = std::env::var("BOLT_Q8_FILE").expect("BOLT_Q8_FILE names the Q8_0 Bolt file");
    let file = crate::family::gguf_select::GgufFile {
        path: std::path::PathBuf::from(&path),
        bytes: std::fs::metadata(&path).expect("the Bolt file").len(),
    };
    let mut model = super::bolt::load_choosing(
        &[file],
        crate::family::gemma::gemma4::config::MaxSeqs::Count(4),
        &|_| {},
    )
    .expect("load");
    model.reserve_window(4).expect("window buffers");
    let lengths = [700usize, 530, 900, 1_100];
    let ring: Vec<_> = (0..4)
        .map(|_| model.new_seq_cells(1_536).expect("ring session"))
        .collect();
    model.set_linear_local_layers(true);
    let linear: Vec<_> = (0..4)
        .map(|_| model.new_seq_cells(1_536).expect("linear session"))
        .collect();
    model.set_linear_local_layers(false);
    let prompts: Vec<Vec<u32>> = (0..4).map(|slot| prompt_of(slot, lengths[slot])).collect();
    for slot in 0..4 {
        prefill(&mut model, &ring[slot], &prompts[slot]);
        prefill(&mut model, &linear[slot], &prompts[slot]);
    }
    let mut token: Vec<u32> = prompts.iter().map(|p| *p.last().expect("prompt")).collect();
    let mut at: Vec<usize> = prompts.iter().map(|p| p.len() - 1).collect();
    // Which sessions are live at each step: four for twenty steps, then two, then three.
    let schedule: Vec<Vec<usize>> = (0..20)
        .map(|_| vec![0, 1, 2, 3])
        .chain((0..6).map(|_| vec![0, 2]))
        .chain((0..6).map(|_| vec![1, 2, 3]))
        .collect();
    for (step, live) in schedule.iter().enumerate() {
        let tokens: Vec<u32> = live.iter().map(|&slot| token[slot]).collect();
        let positions: Vec<usize> = live.iter().map(|&slot| at[slot]).collect();
        let ring_refs: Vec<&super::bolt::BoltSeq> = live.iter().map(|&slot| &ring[slot]).collect();
        let linear_refs: Vec<&super::bolt::BoltSeq> =
            live.iter().map(|&slot| &linear[slot]).collect();
        let got = model
            .decode_rows(&ring_refs, &tokens, &positions)
            .expect("ring window");
        let want = model
            .decode_rows(&linear_refs, &tokens, &positions)
            .expect("linear window");
        for (index, &slot) in live.iter().enumerate() {
            assert_eq!(
                bits(&got[index]),
                bits(&want[index]),
                "step {step} slot {slot} at position {}",
                at[slot]
            );
            token[slot] = argmax(&got[index]);
            at[slot] += 1;
        }
    }
    assert!(
        at.iter().all(|&position| position > 519 + 20),
        "every row wrapped the ring"
    );
}
