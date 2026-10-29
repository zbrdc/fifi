//! Real-checkpoint oracle gates and the decode measurement of the resident path.
//!
//! The three real-checkpoint tests are ignored. They need the Flash-Next artifact on the Mac, and
//! they run through the resident-decode qualification harness. The teacher-forced and end-to-end
//! tests reuse the frozen tolerances and the `Gate` of `flash_next/mlx.rs`. The decode test
//! measures and writes its report; the runner applies the token-match and decode-run rules. The
//! other tests check the helpers and run in the normal Mac suite.

use super::super::hyper::HyperShape;
use super::super::load::FlashWeights;
use super::super::mlx::MlxProjector;
use super::super::mlx::tests::{
    COS_BLOCK, COS_END, Gate, PROMPT, REL_BLOCK, REL_END, env_path, open_checkpoint, oracle, top5,
    write_report,
};
use super::super::moe::MoeShape;
use super::super::step::{ModelShape, ModelState, step as host_step};
use super::super::stored::{StackLayout, read_slice};
use super::ops::{Ctx, dim, f32_tensor, linear, values};
use super::slots::{ExpertSlots, POOL_SETS, SetLayout};
use super::state::{LayerState, ResidentState};
use super::trace::{RouteTrace, TRACE_SCHEMA, TraceMode, TraceStep, check_step, tsv_columns};
use super::weights::{RAttn, RHyper, RLayer, RMoe, RPle, ResidentWeights};
use super::{ResidentDecoder, StepMode, deltanet, hyper, moe, ple, qsa};
use crate::{
    EngError,
    kernels::{HostPages, MlxKernels, ResidentPackedWeight},
    tensor::{ExecutionTensor, Tensor},
    weights::{ShardIo, StreamedCheckpoint},
};
use std::collections::BTreeSet;
use std::io::{BufWriter, Read, Seek, SeekFrom, Write};
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

/// Generated tokens in the decode run.
const GENERATED: usize = 512;
/// Host tokens compared with the resident tokens.
const MATCH: usize = 32;
/// Generated-token counts after which the decode test records the process RSS.
const RSS_AT: [usize; 5] = [16, 128, 256, 384, 512];
/// Column order of each `stepNNNN` line. The resident-decode qualification harness parses the same order.
const STEP_COLUMNS: &str =
    "mode,input,output,seconds,syncs,flushes,retired_peak,peak,active,cache,read_bytes,read_ns";

/// Read bytes and read nanoseconds of every shard reader of one checkpoint.
#[derive(Debug, Default)]
struct ReadCounter {
    bytes: AtomicU64,
    nanos: AtomicU64,
}

impl ReadCounter {
    /// The bytes and nanoseconds counted so far.
    fn snapshot(&self) -> (u64, u64) {
        (
            self.bytes.load(Ordering::Relaxed),
            self.nanos.load(Ordering::Relaxed),
        )
    }
}

/// A shard reader that adds each read's bytes and time to a shared counter. Seeks are not counted.
struct Counted<R> {
    inner: R,
    counter: Arc<ReadCounter>,
}

impl<R: Read> Read for Counted<R> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let started = Instant::now();
        let n = self.inner.read(buf)?;
        let nanos = u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX);
        let bytes = u64::try_from(n).unwrap_or(u64::MAX);
        self.counter.bytes.fetch_add(bytes, Ordering::Relaxed);
        self.counter.nanos.fetch_add(nanos, Ordering::Relaxed);
        Ok(n)
    }
}

impl<R: Seek> Seek for Counted<R> {
    fn seek(&mut self, pos: SeekFrom) -> std::io::Result<u64> {
        self.inner.seek(pos)
    }
}

/// Opens the checkpoint in `dir` as `mlx.rs` `open_checkpoint` does, with every shard reader counted.
fn open_counted(dir: &Path) -> (StreamedCheckpoint, Arc<ReadCounter>) {
    let index = std::fs::read(dir.join("model.safetensors.index.json")).unwrap();
    let value: serde_json::Value = serde_json::from_slice(&index).unwrap();
    let names: BTreeSet<String> = value["weight_map"]
        .as_object()
        .unwrap()
        .values()
        .filter_map(|v| v.as_str().map(str::to_string))
        .collect();
    let counter = Arc::new(ReadCounter::default());
    let shards: Vec<(String, Box<dyn ShardIo + Send>, u64)> = names
        .into_iter()
        .map(|name| {
            let file = std::fs::File::open(dir.join(&name)).unwrap();
            let len = file.metadata().unwrap().len();
            let reader = Counted {
                inner: file,
                counter: Arc::clone(&counter),
            };
            (name, Box::new(reader) as Box<dyn ShardIo + Send>, len)
        })
        .collect();
    let ckpt = StreamedCheckpoint::open(&index, shards, 64 << 20).unwrap();
    (ckpt, counter)
}

/// The process-wide synchronization count in qualification builds (`weights.rs` `syncs`).
fn syncs(kernels: &MlxKernels) -> Option<u64> {
    #[cfg(feature = "native-fault-qualification")]
    {
        Some(kernels.qualification_sync_count())
    }
    #[cfg(not(feature = "native-fault-qualification"))]
    {
        let _ = kernels;
        None
    }
}

/// Forced retire flushes (process total) and the peak retired bytes since the last reset, in
/// qualification builds.
fn retire_stats(kernels: &MlxKernels) -> Option<(u64, u64)> {
    #[cfg(feature = "native-fault-qualification")]
    {
        Some(kernels.qualification_retire_stats())
    }
    #[cfg(not(feature = "native-fault-qualification"))]
    {
        let _ = kernels;
        None
    }
}

/// Resets the peak retired bytes in qualification builds. Other builds do nothing.
fn reset_retire_peak(kernels: &mut MlxKernels) {
    #[cfg(feature = "native-fault-qualification")]
    {
        kernels.qualification_reset_retire_peak();
    }
    #[cfg(not(feature = "native-fault-qualification"))]
    {
        let _ = kernels;
    }
}

/// Page imports that MLX served with its copy path (process total), in qualification builds.
fn page_copies(kernels: &MlxKernels) -> Option<u64> {
    #[cfg(feature = "native-fault-qualification")]
    {
        Some(kernels.qualification_page_copies())
    }
    #[cfg(not(feature = "native-fault-qualification"))]
    {
        let _ = kernels;
        None
    }
}

/// An expert page pool sized as `ResidentDecoder::new` sizes it, for the block checks.
fn expert_pool(
    ckpt: &StreamedCheckpoint,
    weights: &ResidentWeights,
) -> Result<ExpertSlots, EngError> {
    let top_k = u32::try_from(weights.shape.moe.top_k())
        .map_err(|_| EngError::BadRequest("top-k exceeds u32".into()))?;
    let page = HostPages::page_size();
    let mut set_bytes = 0;
    for layer in &weights.layers {
        set_bytes = set_bytes.max(SetLayout::new(ckpt, &layer.moe, top_k, page)?.end);
    }
    ExpertSlots::new(POOL_SETS, set_bytes)
}

/// A counter value, or `-` when the build has no counter.
fn opt(value: Option<u64>) -> String {
    match value {
        Some(v) => v.to_string(),
        None => "-".to_string(),
    }
}

/// Comma-separated values.
fn join<T: ToString>(values: &[T]) -> String {
    values
        .iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join(",")
}

/// The process resident set size in bytes, from `ps` (KiB). Call it outside timed regions.
fn rss_bytes() -> u64 {
    let pid = std::process::id().to_string();
    let out = std::process::Command::new("ps")
        .args(["-o", "rss=", "-p", &pid])
        .output()
        .unwrap();
    let kib: u64 = String::from_utf8_lossy(&out.stdout).trim().parse().unwrap();
    kib * 1024
}

/// The greedy token, lowest index on ties (the `top5` and device `argmax` rule), and its margin
/// over the next largest logit.
fn host_greedy(logits: &[f32]) -> (u32, f32) {
    let mut best = 0;
    for (i, &v) in logits.iter().enumerate() {
        if v > logits[best] {
            best = i;
        }
    }
    let second = logits
        .iter()
        .enumerate()
        .filter(|&(i, _)| i != best)
        .map(|(_, &v)| v)
        .fold(f32::NEG_INFINITY, f32::max);
    (best as u32, logits[best] - second)
}

/// `after - before`, element by element: the PLE update from the block's `hyper + update`.
fn minus(after: &[f32], before: &[f32]) -> Vec<f32> {
    after.iter().zip(before).map(|(a, b)| a - b).collect()
}

/// Layer `layer` of every token's recorded rows, joined in token order (the host `layer_out` order).
fn layer_column(rows: &[Vec<Vec<f32>>], layer: usize) -> Vec<f32> {
    rows.iter()
        .flat_map(|token| token.get(layer).cloned().unwrap_or_default())
        .collect()
}

/// Stored bytes of the experts of one token: `top_k` slices of `gate`, `up`, and `down` in every
/// layer, with scales and biases. It reads slice 0 of each stack once.
fn expert_bytes_per_token(ckpt: &StreamedCheckpoint, weights: &ResidentWeights) -> u64 {
    let top_k = weights.shape.moe.top_k() as u64;
    let mut total = 0;
    for layer in &weights.layers {
        for stack in [&layer.moe.gate, &layer.moe.up, &layer.moe.down] {
            let slice = read_slice(ckpt, stack, 0).unwrap();
            let bytes = slice.payload().byte_len()
                + slice.scales().byte_len()
                + slice.biases().map_or(0, Tensor::byte_len);
            total += top_k * bytes;
        }
    }
    total
}

/// What every teacher-forced layer check reads.
struct Source<'a> {
    ckpt: &'a StreamedCheckpoint,
    shape: &'a ModelShape,
    dir: &'a Path,
}

/// Imports host values as a resident F32 tensor of `shape`.
fn import(ctx: &mut Ctx<'_>, shape: Vec<u32>, data: &[f32]) -> Result<ExecutionTensor, EngError> {
    ctx.kernels.import_tensor(&f32_tensor(shape, data)?)
}

/// The `[C, H]` shape of a hyper residual.
fn residual_shape(shape: &HyperShape) -> Result<Vec<u32>, EngError> {
    Ok(vec![dim(shape.branches())?, dim(shape.hidden())?])
}

/// The token start of `token`, and its `[C * H]` hyper values.
fn block_start(
    ctx: &mut Ctx<'_>,
    ckpt: &StreamedCheckpoint,
    weights: &ResidentWeights,
    history: &[u32],
    token: u32,
) -> Result<(ple::TokenStart, Vec<f32>), EngError> {
    let start = ple::import_token(ctx, ckpt, weights, history, token)?;
    let row = values(ctx.kernels, &start.hyper)?;
    Ok((start, row))
}

/// The hyper read of an oracle `[C * H]` row: `mixed` `[H]` and `gate` `[C]` (empty without inject).
fn block_hyper(
    ctx: &mut Ctx<'_>,
    shape: &HyperShape,
    w: &RHyper,
    row: &[f32],
) -> Result<(Vec<f32>, Vec<f32>), EngError> {
    let h = import(ctx, residual_shape(shape)?, row)?;
    let read = hyper::read(ctx, shape, w, &h)?;
    let mixed = values(ctx.kernels, &read.mixed)?;
    let gate = match &read.gate {
        Some(gate) => values(ctx.kernels, gate)?,
        None => Vec::new(),
    };
    Ok((mixed, gate))
}

/// One attention step on an oracle `mixed` row. It returns the output and the replacement state.
fn block_attention(
    ctx: &mut Ctx<'_>,
    shape: &ModelShape,
    attn: &RAttn,
    state: LayerState,
    pos: u32,
    x: &[f32],
) -> Result<(Vec<f32>, LayerState), EngError> {
    let x = import(ctx, vec![1, dim(x.len())?], x)?;
    match (attn, state) {
        (RAttn::Delta(w), LayerState::Delta { rec, ring }) => {
            let s = deltanet::step(ctx, &shape.delta, w, &rec, &ring, &x)?;
            let out = values(ctx.kernels, &s.out)?;
            Ok((
                out,
                LayerState::Delta {
                    rec: s.rec,
                    ring: s.ring,
                },
            ))
        }
        (RAttn::Qsa(w), cache @ LayerState::Qsa { .. }) => {
            let s = qsa::step(ctx, &shape.qsa, w, &cache, pos, &x)?;
            let out = values(ctx.kernels, &s.out)?;
            Ok((out, s.state))
        }
        _ => Err(EngError::BadRequest(
            "attention state does not match its layer".into(),
        )),
    }
}

/// One paged MoE step on an oracle `mixed` row: the input import, the router readback, and the
/// output readback. The output readback evaluates the only consumer of the step's page set, so the
/// set is released after it (the release contract of the page pool). After an error the step's
/// graph is dropped unevaluated, so the release is safe as well.
fn block_moe(
    ctx: &mut Ctx<'_>,
    ckpt: &StreamedCheckpoint,
    shape: &MoeShape,
    w: &RMoe,
    slots: &mut ExpertSlots,
    x: &[f32],
) -> Result<Vec<f32>, EngError> {
    let x = import(ctx, vec![1, dim(x.len())?], x)?;
    let result = moe::step_paged(ctx, ckpt, shape, w, slots, &x)
        .and_then(|(out, _)| values(ctx.kernels, &out));
    slots.release_lent();
    result
}

/// The PLE block on an oracle `[C * H]` row. It returns the update and the new ring.
fn block_ple(
    ctx: &mut Ctx<'_>,
    shape: &ModelShape,
    w: &RPle,
    emb: &ExecutionTensor,
    row: &[f32],
    ring: &ExecutionTensor,
) -> Result<(Vec<f32>, ExecutionTensor), EngError> {
    let hyper_in = import(ctx, residual_shape(&shape.hyper)?, row)?;
    let out = ple::block(ctx, w, shape, emb, &hyper_in, ring)?;
    let after = values(ctx.kernels, &out.hyper)?;
    Ok((minus(&after, row), out.ring))
}

/// The mixer read of an oracle `[C * H]` row: `mixed` `[H]`.
fn block_mixer(
    ctx: &mut Ctx<'_>,
    shape: &HyperShape,
    w: &RHyper,
    row: &[f32],
) -> Result<Vec<f32>, EngError> {
    let h = import(ctx, residual_shape(shape)?, row)?;
    let mixed = hyper::mixer_read(ctx, shape, w, &h)?;
    values(ctx.kernels, &mixed)
}

/// The head on an oracle `mixed` row: logits `[V]`.
fn block_head(
    ctx: &mut Ctx<'_>,
    w: &ResidentPackedWeight,
    mixed: &[f32],
) -> Result<Vec<f32>, EngError> {
    let x = import(ctx, vec![1, dim(mixed.len())?], mixed)?;
    let logits = linear(ctx, w, &x)?;
    values(ctx.kernels, &logits)
}

/// Checks one resident layer over the five prompt tokens, teacher-forced from the oracle rows.
/// It mirrors `check_layer` in `mlx.rs`: the same labels, order, and six checks per token.
fn check_layer(
    ctx: &mut Ctx<'_>,
    src: &Source<'_>,
    slots: &mut ExpertSlots,
    index: usize,
    layer: &RLayer,
    state: LayerState,
    gate: &mut Gate,
) {
    let dir = src.dir;
    let attn_in = oracle(
        dir,
        &format!("read_in.layers.{index}.attn_hyper_connection"),
    );
    let attn_mixed = oracle(dir, &format!("mixed.layers.{index}.attn_hyper_connection"));
    let attn_gate = oracle(dir, &format!("gate.layers.{index}.attn_hyper_connection"));
    let attn_out = oracle(dir, &format!("attn_out.L{index:02}"));
    let mlp_in = oracle(dir, &format!("read_in.layers.{index}.mlp_hyper_connection"));
    let mlp_mixed = oracle(dir, &format!("mixed.layers.{index}.mlp_hyper_connection"));
    let mlp_gate = oracle(dir, &format!("gate.layers.{index}.mlp_hyper_connection"));
    let moe_out = oracle(dir, &format!("moe_out.L{index:02}"));
    let hyper_shape = &src.shape.hyper;
    let mut state = Some(state);
    for (t, attn_row) in attn_in.iter().enumerate() {
        let label = format!("L{index:02} t{t}");
        let Some((mixed, read_gate)) = gate.take(
            &format!("{label} attn hyper"),
            block_hyper(ctx, hyper_shape, &layer.attn_hyper, attn_row),
        ) else {
            return;
        };
        gate.check(
            &format!("{label} attn hyper mixed"),
            &mixed,
            &attn_mixed[t],
            COS_BLOCK,
            REL_BLOCK,
        );
        gate.check(
            &format!("{label} attn hyper gate"),
            &read_gate,
            &attn_gate[t],
            COS_BLOCK,
            REL_BLOCK,
        );
        let Some(current) = state.take() else {
            return;
        };
        let Some((out, next)) = gate.take(
            &format!("{label} attn block"),
            block_attention(
                ctx,
                src.shape,
                &layer.attn,
                current,
                t as u32,
                &attn_mixed[t],
            ),
        ) else {
            return;
        };
        state = Some(next);
        gate.check(
            &format!("{label} attn out"),
            &out,
            &attn_out[t],
            COS_BLOCK,
            REL_BLOCK,
        );
        let Some((mixed, read_gate)) = gate.take(
            &format!("{label} mlp hyper"),
            block_hyper(ctx, hyper_shape, &layer.mlp_hyper, &mlp_in[t]),
        ) else {
            return;
        };
        gate.check(
            &format!("{label} mlp hyper mixed"),
            &mixed,
            &mlp_mixed[t],
            COS_BLOCK,
            REL_BLOCK,
        );
        gate.check(
            &format!("{label} mlp hyper gate"),
            &read_gate,
            &mlp_gate[t],
            COS_BLOCK,
            REL_BLOCK,
        );
        let Some(out) = gate.take(
            &format!("{label} mlp block"),
            block_moe(
                ctx,
                src.ckpt,
                &src.shape.moe,
                &layer.moe,
                slots,
                &mlp_mixed[t],
            ),
        ) else {
            return;
        };
        gate.check(
            &format!("{label} mlp out"),
            &out,
            &moe_out[t],
            COS_BLOCK,
            REL_BLOCK,
        );
    }
}

#[test]
fn counted_reader_adds_every_byte_it_reads() {
    let counter = Arc::new(ReadCounter::default());
    let mut reader = Counted {
        inner: std::io::Cursor::new(vec![7u8; 10]),
        counter: Arc::clone(&counter),
    };
    reader.seek(SeekFrom::Start(3)).unwrap();
    let mut head = [0u8; 4];
    reader.read_exact(&mut head).unwrap();
    let mut rest = Vec::new();
    reader.read_to_end(&mut rest).unwrap();
    assert_eq!(head, [7; 4], "read_exact bytes");
    assert_eq!(rest.len(), 3, "bytes after the first read");
    assert_eq!(
        counter.snapshot().0,
        7,
        "counted bytes: 10 minus the 3 that the seek skipped"
    );
}

#[test]
fn host_greedy_takes_the_lowest_index_on_ties() {
    assert_eq!(
        host_greedy(&[1.0, 3.0, 3.0, 2.0]),
        (1, 0.0),
        "tie of index 1 and 2"
    );
    assert_eq!(
        host_greedy(&[0.5, 2.0, 1.25]),
        (1, 0.75),
        "margin over the next logit"
    );
}

#[test]
fn ple_update_subtracts_the_input_residual() {
    assert_eq!(minus(&[3.0, 1.5], &[1.0, 2.0]), [2.0, -0.5]);
}

#[test]
fn layer_column_joins_one_layer_over_tokens() {
    let rows = [
        vec![vec![1.0, 2.0], vec![3.0, 4.0]],
        vec![vec![5.0, 6.0], vec![7.0, 8.0]],
    ];
    assert_eq!(layer_column(&rows, 0), [1.0, 2.0, 5.0, 6.0], "layer 0");
    assert_eq!(layer_column(&rows, 1), [3.0, 4.0, 7.0, 8.0], "layer 1");
}

#[test]
fn block_moe_returns_every_page_set_after_its_readback() {
    let _serial = crate::native_test_lock();
    let ckpt = super::fixtures::checkpoint();
    let flash = FlashWeights::load(&ckpt, &super::fixtures::config()).unwrap();
    let mut kernels = MlxKernels::try_new().unwrap();
    let weights = ResidentWeights::import(&mut kernels, &flash).unwrap();
    let mut slots = expert_pool(&ckpt, &weights).unwrap();
    let hidden = weights.shape.hyper.hidden();
    let x: Vec<f32> = (0..hidden)
        .map(|i| f32::from(u8::try_from(i % 7).unwrap()) * 0.125 - 0.375)
        .collect();
    let mut ctx = Ctx {
        kernels: &mut kernels,
        consts: &weights.consts,
    };
    let mut outputs = Vec::new();
    for _ in 0..POOL_SETS + 2 {
        let result = block_moe(
            &mut ctx,
            &ckpt,
            &weights.shape.moe,
            &weights.layers[0].moe,
            &mut slots,
            &x,
        );
        assert!(result.is_ok(), "{result:?}");
        outputs.extend(result);
    }
    assert!(
        outputs.windows(2).all(|pair| pair[0] == pair[1]),
        "the same row gives the same output"
    );
    // One more synchronization frees the residents of the released sets.
    import(&mut ctx, vec![1], &[0.0]).unwrap();
    let deadline = Instant::now() + std::time::Duration::from_secs(1);
    let in_use = loop {
        let in_use: Vec<usize> = (0..POOL_SETS)
            .filter(|&index| slots.pages(index).unwrap().in_use())
            .collect();
        if in_use.is_empty() || Instant::now() >= deadline {
            break in_use;
        }
        std::thread::sleep(std::time::Duration::from_millis(1));
    };
    assert!(
        in_use.is_empty(),
        "sets still lent after the readbacks: {in_use:?}"
    );
}

#[test]
#[ignore = "needs the Flash-Next artifact and oracle dump on the Mac"]
fn resident_blocks_match_the_oracle_teacher_forced() {
    let _serial = crate::native_test_lock();
    let artifact = env_path("FLASH_NEXT_ARTIFACT");
    let oracle_dir = env_path("FLASH_NEXT_ORACLE");
    let report_path = env_path("FLASH_NEXT_RESIDENT_BLOCKS_REPORT");
    let dir = oracle_dir.as_path();
    let ckpt = open_checkpoint(&artifact);
    let config = std::fs::read(artifact.join("config.json")).unwrap();
    let flash = FlashWeights::load(&ckpt, &config).unwrap();
    let mut kernels = MlxKernels::try_new().unwrap();
    let weights = ResidentWeights::import(&mut kernels, &flash).unwrap();
    drop(flash);
    let mut slots = expert_pool(&ckpt, &weights).unwrap();
    // Zero layer states and PLE ring in the `state.rs` layouts.
    let mut zero = ResidentState::zero(&mut kernels, &weights).unwrap();
    let mut ple_ring = zero.ple_ring().share();
    let layer_states = zero.begin();
    let shape = weights.shape;
    let hidden = shape.hyper.hidden();
    let mut ctx = Ctx {
        kernels: &mut kernels,
        consts: &weights.consts,
    };
    let mut gate = Gate::default();

    let embed_in = oracle(dir, "read_in.layers.0.attn_hyper_connection");
    let mut history: Vec<u32> = Vec::new();
    let mut embs = Vec::new();
    for (t, (&token, want)) in PROMPT.iter().zip(&embed_in).enumerate() {
        let label = format!("embed t{t}");
        let result = block_start(&mut ctx, &ckpt, &weights, &history, token);
        if let Some((start, row)) = gate.take(&label, result) {
            gate.check(
                &label,
                row.get(..hidden).unwrap_or_default(),
                &want[..hidden],
                COS_BLOCK,
                REL_BLOCK,
            );
            history = start.history;
            embs.push(start.emb);
        }
    }

    assert_eq!(weights.layers.len(), 48, "layer count");
    let src = Source {
        ckpt: &ckpt,
        shape: &shape,
        dir,
    };
    for (index, (layer, state)) in weights.layers.iter().zip(layer_states).enumerate() {
        check_layer(&mut ctx, &src, &mut slots, index, layer, state, &mut gate);
    }

    let ple_in = oracle(dir, "layer_out.L00");
    let ple_want = oracle(dir, "ple_out");
    for (t, (hyper_row, emb)) in ple_in.iter().zip(&embs).enumerate() {
        let label = format!("ple t{t}");
        let result = block_ple(&mut ctx, &shape, &weights.ple, emb, hyper_row, &ple_ring);
        if let Some((update, ring)) = gate.take(&label, result) {
            ple_ring = ring;
            gate.check(&label, &update, &ple_want[t], COS_BLOCK, REL_BLOCK);
        }
    }

    let mixer_in = oracle(dir, "read_in.hyper_connection_mixer");
    let mixer_mixed = oracle(dir, "mixed.hyper_connection_mixer");
    let logits = oracle(dir, "logits");
    for (t, hin) in mixer_in.iter().enumerate() {
        let label = format!("mixer t{t}");
        let result = block_mixer(&mut ctx, &shape.hyper, &weights.mixer, hin);
        if let Some(mixed) = gate.take(&format!("{label} hyper"), result) {
            gate.check(
                &format!("{label} mixed"),
                &mixed,
                &mixer_mixed[t],
                COS_BLOCK,
                REL_BLOCK,
            );
        }
        let result = block_head(&mut ctx, &weights.lm_head, &mixer_mixed[t]);
        if let Some(got) = gate.take(&format!("{label} logits"), result) {
            gate.check(
                &format!("{label} logits"),
                &got,
                &logits[t],
                COS_BLOCK,
                REL_BLOCK,
            );
        }
    }

    gate.report.push_str(&format!("checks={}\n", gate.checks));
    write_report(&report_path, &gate);
    assert_eq!(gate.checks, 1460, "teacher-forced check count");
    assert!(gate.failures.is_empty(), "{:?}", gate.failures);
}

#[test]
#[ignore = "needs the Flash-Next artifact and oracle dump on the Mac"]
fn resident_prompt_matches_the_oracle_end_to_end() {
    let _serial = crate::native_test_lock();
    let artifact = env_path("FLASH_NEXT_ARTIFACT");
    let oracle_dir = env_path("FLASH_NEXT_ORACLE");
    let report_path = env_path("FLASH_NEXT_RESIDENT_PROMPT_REPORT");
    let dir = oracle_dir.as_path();
    let ckpt = open_checkpoint(&artifact);
    let config = std::fs::read(artifact.join("config.json")).unwrap();
    let flash = FlashWeights::load(&ckpt, &config).unwrap();
    let mut decoder = ResidentDecoder::new(MlxKernels::try_new().unwrap(), &ckpt, &flash).unwrap();
    drop(flash);
    let mut state = decoder.zero_state().unwrap();
    let mut gate = Gate::default();

    let mut results = Vec::new();
    for (t, &token) in PROMPT.iter().enumerate() {
        let before = syncs(&decoder.kernels);
        let started = Instant::now();
        let result = decoder.step(&mut state, token, StepMode::Record);
        let seconds = started.elapsed().as_secs_f64();
        let used = syncs(&decoder.kernels)
            .zip(before)
            .map(|(after, before)| after - before);
        gate.report.push_str(&format!(
            "token{t}_seconds={seconds:.3}\nrecord_syncs_t{t}={}\n",
            opt(used)
        ));
        match gate.take(&format!("token{t}"), result) {
            Some(output) => results.push(output),
            None => {
                write_report(&report_path, &gate);
                panic!("token {t} failed: {:?}", gate.failures);
            }
        }
    }

    let rows: Vec<Vec<Vec<f32>>> = results
        .iter()
        .map(|r| r.layer_out.clone().unwrap_or_default())
        .collect();
    for layer in 0..48usize {
        let want = oracle(dir, &format!("layer_out.L{layer:02}")).concat();
        let got = layer_column(&rows, layer);
        gate.check(
            &format!("layer_out L{layer:02}"),
            &got,
            &want,
            COS_END,
            REL_END,
        );
    }

    let oracle_logits = oracle(dir, "logits");
    let rust_logits = results[4].logits.clone().unwrap_or_default();
    let want_logits = &oracle_logits[4];
    gate.check("logits t4", &rust_logits, want_logits, COS_END, REL_END);
    let rust_top = top5(&rust_logits);
    let oracle_top = top5(want_logits);
    gate.report.push_str(&format!(
        "rust_top5={rust_top:?}\noracle_top5={oracle_top:?}\n"
    ));
    let rust_ids: Vec<usize> = rust_top.iter().map(|(id, _)| *id).collect();
    let oracle_ids: Vec<usize> = oracle_top.iter().map(|(id, _)| *id).collect();
    if rust_ids != oracle_ids {
        gate.failures.push("top5 ids".to_string());
    }
    if rust_ids.first() != Some(&11751) {
        gate.failures.push("argmax".to_string());
    }
    for (id, want) in &oracle_top {
        let got = rust_logits.get(*id).copied().unwrap_or(f32::NAN);
        let diff = (got - *want).abs();
        if diff.is_nan() || diff > 0.1 {
            gate.failures.push(format!("top5 value id {id}"));
        }
    }
    let device = results[4].token;
    gate.report
        .push_str(&format!("device_token_t4={}\n", opt(device.map(u64::from))));
    if device != Some(11751) {
        gate.failures.push("device argmax".to_string());
    }

    write_report(&report_path, &gate);
    assert!(gate.failures.is_empty(), "{:?}", gate.failures);
}

#[test]
#[ignore = "needs the Flash-Next artifact on the Mac"]
fn resident_decode_reports_speed_memory_and_host_tokens() {
    let _serial = crate::native_test_lock();
    let artifact = env_path("FLASH_NEXT_ARTIFACT");
    let report_path = env_path("FLASH_NEXT_RESIDENT_DECODE_REPORT");
    let cache_limit: u64 = std::env::var("FLASH_NEXT_CACHE_LIMIT")
        .expect("FLASH_NEXT_CACHE_LIMIT is not set")
        .parse()
        .expect("FLASH_NEXT_CACHE_LIMIT is not a byte count");
    let build = if cfg!(feature = "native-fault-qualification") {
        "q"
    } else {
        "p"
    };
    let config = std::fs::read(artifact.join("config.json")).unwrap();
    let (ckpt, counter) = open_counted(&artifact);
    let mut lines = format!(
        "build={build}\nprompt={}\ngenerated_tokens={GENERATED}\ncache_limit_bytes={cache_limit}\n",
        join(&PROMPT)
    );

    // Load and import.
    let started = Instant::now();
    let flash = FlashWeights::load(&ckpt, &config).unwrap();
    let load_seconds = started.elapsed().as_secs_f64();
    let mut kernels = MlxKernels::try_new().unwrap();
    let cache_previous = kernels.set_cache_limit(cache_limit).unwrap();
    let before_import = syncs(&kernels);
    let started = Instant::now();
    let mut decoder = ResidentDecoder::new(kernels, &ckpt, &flash).unwrap();
    let import_seconds = started.elapsed().as_secs_f64();
    let after_import = syncs(&decoder.kernels);
    drop(flash);
    let mut state = decoder.zero_state().unwrap();
    let after_zero = syncs(&decoder.kernels);
    lines.push_str(&format!("pool_bytes={}\n", decoder.pool_bytes()));
    let import_syncs = after_import
        .zip(before_import)
        .map(|(after, before)| after - before);
    let zero_syncs = after_zero
        .zip(after_import)
        .map(|(after, before)| after - before);
    lines.push_str(&format!(
        "cache_limit_previous={cache_previous}\nload_seconds={load_seconds}\n\
         import_seconds={import_seconds}\nimport_syncs={}\nzero_state_syncs={}\n\
         weights_bytes={}\nstate_bytes_start={}\nexpert_bytes_per_token={}\n\
         step_columns={STEP_COLUMNS}\n",
        opt(import_syncs),
        opt(zero_syncs),
        decoder.weights.byte_len().unwrap(),
        state.byte_len().unwrap(),
        expert_bytes_per_token(&ckpt, &decoder.weights),
    ));

    // Four `Prompt` steps, then 512 `Generate` steps. Only `step` is timed.
    let mut resident: Vec<u32> = Vec::with_capacity(GENERATED);
    let mut next = 0;
    let copies_before = page_copies(&decoder.kernels);
    for n in 0..PROMPT.len() - 1 + GENERATED {
        let input = PROMPT.get(n).copied().unwrap_or(next);
        let (mode, name) = if n + 1 < PROMPT.len() {
            (StepMode::Prompt, "prompt")
        } else {
            (StepMode::Generate, "generate")
        };
        decoder.kernels.reset_peak_memory().unwrap();
        reset_retire_peak(&mut decoder.kernels);
        let s0 = syncs(&decoder.kernels);
        let f0 = retire_stats(&decoder.kernels);
        let (b0, n0) = counter.snapshot();
        let started = Instant::now();
        let result = decoder.step(&mut state, input, mode);
        let seconds = started.elapsed().as_secs_f64();
        let (b1, n1) = counter.snapshot();
        let s1 = syncs(&decoder.kernels);
        let f1 = retire_stats(&decoder.kernels);
        let out = match result {
            Ok(out) => out,
            Err(error) => {
                lines.push_str(&format!("step_error=step{n:04} {error:?}\n"));
                std::fs::write(&report_path, &lines).unwrap();
                panic!("step {n} failed: {error:?}");
            }
        };
        let memory = decoder.kernels.memory_stats().unwrap();
        let used = s1.zip(s0).map(|(after, before)| after - before);
        let flushes = f1.zip(f0).map(|(after, before)| after.0 - before.0);
        let retired_peak = f1.map(|f| f.1);
        lines.push_str(&format!(
            "step{n:04}={name},{input},{},{seconds:.6},{},{},{},{},{},{},{},{}\n",
            opt(out.token.map(u64::from)),
            opt(used),
            opt(flushes),
            opt(retired_peak),
            memory.peak,
            memory.active,
            memory.cache,
            b1 - b0,
            n1 - n0,
        ));
        if let Some(token) = out.token {
            resident.push(token);
            next = token;
            if RSS_AT.contains(&resident.len()) {
                lines.push_str(&format!("rss_t{}={}\n", resident.len(), rss_bytes()));
            }
        }
    }
    let copies = page_copies(&decoder.kernels)
        .zip(copies_before)
        .map(|(after, before)| after - before);
    lines.push_str(&format!(
        "page_copies={}\nstate_bytes_end={}\nresident_tokens={}\n",
        opt(copies),
        state.byte_len().unwrap(),
        join(&resident)
    ));
    std::fs::write(&report_path, &lines).unwrap();

    // The host path on the same tokens. Two leases never coexist.
    drop(state);
    drop(decoder);
    let flash = FlashWeights::load(&ckpt, &config).unwrap();
    let shape = *flash.shape();
    let view = flash.view();
    let mut proj = MlxProjector::new(&ckpt).unwrap();
    let mut host = ModelState::new(&shape, &view);
    let mut host_tokens: Vec<u32> = Vec::with_capacity(MATCH);
    let mut margins: Vec<String> = Vec::with_capacity(MATCH);
    let mut host_next = 0;
    let started = Instant::now();
    for n in 0..PROMPT.len() - 1 + MATCH {
        let input = PROMPT.get(n).copied().unwrap_or(host_next);
        let output = host_step(&mut proj, &shape, &view, &mut host, input, false).unwrap();
        if n + 1 >= PROMPT.len() {
            let (token, margin) = host_greedy(&output.logits);
            host_tokens.push(token);
            margins.push(format!("{margin:.6}"));
            host_next = token;
        }
    }
    lines.push_str(&format!(
        "host_tokens={}\nhost_margins={}\nhost_seconds={}\n",
        join(&host_tokens),
        margins.join(","),
        started.elapsed().as_secs_f64()
    ));
    std::fs::write(&report_path, &lines).unwrap();
    assert_eq!(resident.len(), GENERATED, "generated tokens");
    assert_eq!(host_tokens.len(), MATCH, "host tokens");
}

/// Generated tokens in the profile window.
const PROFILE_TOKENS: usize = 64;

/// Splits the resident decode time per generated token. Qualification builds only: it reads the
/// call timing, the retire counters, and the counted reader.
#[cfg(feature = "native-fault-qualification")]
#[test]
#[ignore = "needs the Flash-Next artifact on the Mac"]
fn resident_decode_profile() {
    let _serial = crate::native_test_lock();
    let artifact = env_path("FLASH_NEXT_ARTIFACT");
    let report_path = env_path("FLASH_NEXT_RESIDENT_PROFILE_REPORT");
    let cache_limit: u64 = std::env::var("FLASH_NEXT_CACHE_LIMIT")
        .expect("FLASH_NEXT_CACHE_LIMIT is not set")
        .parse()
        .expect("FLASH_NEXT_CACHE_LIMIT is not a byte count");
    let config = std::fs::read(artifact.join("config.json")).unwrap();
    let (ckpt, counter) = open_counted(&artifact);
    let flash = FlashWeights::load(&ckpt, &config).unwrap();
    let mut kernels = MlxKernels::try_new().unwrap();
    kernels.set_cache_limit(cache_limit).unwrap();
    let mut decoder = ResidentDecoder::new(kernels, &ckpt, &flash).unwrap();
    drop(flash);
    let mut state = decoder.zero_state().unwrap();
    let mut next = 0;
    for &token in &PROMPT[..PROMPT.len() - 1] {
        decoder.step(&mut state, token, StepMode::Prompt).unwrap();
    }
    let mut input = PROMPT[PROMPT.len() - 1];
    decoder.kernels.qualification_reset_timing();
    let copies_before = decoder.kernels.qualification_page_copies();
    let s0 = decoder.kernels.qualification_sync_count();
    let (b0, n0) = counter.snapshot();
    let started = Instant::now();
    for _ in 0..PROFILE_TOKENS {
        let out = decoder.step(&mut state, input, StepMode::Generate).unwrap();
        next = out.token.unwrap();
        input = next;
    }
    let seconds = started.elapsed().as_secs_f64();
    let (b1, n1) = counter.snapshot();
    let s1 = decoder.kernels.qualification_sync_count();
    let copies = decoder.kernels.qualification_page_copies() - copies_before;
    let t = decoder.kernels.qualification_timing();
    let per = |ns: u64| ns as f64 / 1e9 / PROFILE_TOKENS as f64;
    let step = seconds / PROFILE_TOKENS as f64;
    let read = per(n1 - n0);
    let import = per(t.import_ns);
    let readback = per(t.readback_ns);
    let evaluate = per(t.evaluate_ns);
    let other = step - read - import - readback - evaluate;
    let lines = format!(
        "tokens={PROFILE_TOKENS}\nlast_token={next}\nstep_seconds={step:.6}\n\
         tokens_per_second={:.4}\nread_seconds={read:.6}\nread_bytes={}\n\
         import_seconds={import:.6}\nimport_calls={}\nreadback_seconds={readback:.6}\n\
         readback_calls={}\nevaluate_seconds={evaluate:.6}\nevaluate_calls={}\n\
         other_seconds={other:.6}\nsyncs={}\npool_bytes={}\npage_copies={copies}\n",
        1.0 / step,
        (b1 - b0) / PROFILE_TOKENS as u64,
        t.import_calls as f64 / PROFILE_TOKENS as f64,
        t.readback_calls as f64 / PROFILE_TOKENS as f64,
        t.evaluate_calls as f64 / PROFILE_TOKENS as f64,
        (s1 - s0) as f64 / PROFILE_TOKENS as f64,
        decoder.pool_bytes(),
    );
    std::fs::write(&report_path, &lines).unwrap();
    assert_eq!(s1 - s0, 51 * PROFILE_TOKENS as u64, "{lines}");
    assert_eq!(copies, 0, "no page import takes the copy path");
}

/// The reference resident tokens: the first 32 greedy tokens after `PROMPT`.
const C5_TOKENS: [u32; MATCH] = [
    11751, 13, 11751, 369, 279, 6511, 314, 9338, 13, 271, 27775, 24660, 383, 279, 1414, 3294, 11,
    1092, 369, 279, 6511, 314, 9338, 30, 271, 248068, 198, 760, 1156, 369, 9859, 728,
];

/// One prompt of the trace prompts file.
struct TracePrompt {
    id: u32,
    tokens: Vec<u32>,
    generate: usize,
}

/// Reads `prompts[*].{id, tokens, generate}` of the prompts file. Other fields are notes.
fn trace_prompts(bytes: &[u8]) -> Vec<TracePrompt> {
    let value: serde_json::Value = serde_json::from_slice(bytes).unwrap();
    let as_u32 = |v: &serde_json::Value| u32::try_from(v.as_u64().unwrap()).unwrap();
    let prompts: Vec<TracePrompt> = value["prompts"]
        .as_array()
        .expect("the prompts file has no prompts array")
        .iter()
        .map(|p| TracePrompt {
            id: as_u32(&p["id"]),
            tokens: p["tokens"].as_array().unwrap().iter().map(as_u32).collect(),
            generate: usize::try_from(p["generate"].as_u64().unwrap()).unwrap(),
        })
        .collect();
    let ids: BTreeSet<u32> = prompts.iter().map(|p| p.id).collect();
    assert_eq!(ids.len(), prompts.len(), "prompt ids are unique");
    assert!(
        prompts
            .iter()
            .all(|p| !p.tokens.is_empty() && p.generate > 0),
        "every prompt has tokens and generates at least one token"
    );
    prompts
}

/// Lower-case hex SHA-256 of `bytes`.
fn sha256_hex(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

/// The engine commit of this checkout, with `+dirty` for tracked changes, or `unknown`.
fn engine_commit() -> String {
    let git = |args: &[&str]| {
        std::process::Command::new("git")
            .arg("-C")
            .arg(env!("CARGO_MANIFEST_DIR"))
            .args(args)
            .output()
            .ok()
            .filter(|out| out.status.success())
            .map(|out| String::from_utf8_lossy(&out.stdout).trim().to_string())
    };
    match (
        git(&["rev-parse", "HEAD"]),
        git(&["status", "--porcelain", "--untracked-files=no"]),
    ) {
        (Some(commit), Some(status)) if status.is_empty() => commit,
        (Some(commit), Some(_)) => format!("{commit}+dirty"),
        _ => "unknown".to_string(),
    }
}

/// `sysctl -n vm.swapusage`, or `unavailable`. Call it outside timed regions.
fn swap_usage() -> String {
    std::process::Command::new("sysctl")
        .args(["-n", "vm.swapusage"])
        .output()
        .ok()
        .filter(|out| out.status.success())
        .map_or_else(
            || "unavailable".to_string(),
            |out| String::from_utf8_lossy(&out.stdout).trim().to_string(),
        )
}

/// Stored bytes of one expert in each layer: one slice of `gate`, `up`, and `down` with scales and
/// biases, from the checkpoint headers (there is no fixed expert size). No reads.
fn expert_bytes_per_layer(ckpt: &StreamedCheckpoint, weights: &ResidentWeights) -> Vec<usize> {
    let page = HostPages::page_size();
    weights
        .layers
        .iter()
        .map(|layer| {
            [&layer.moe.gate, &layer.moe.up, &layer.moe.down]
                .into_iter()
                .map(|stack| {
                    let layout = StackLayout::new(ckpt, stack, 1, 0, page).unwrap();
                    layout.payload.slice_len
                        + layout.scales.slice_len
                        + layout.biases.map_or(0, |range| range.slice_len)
                })
                .sum()
        })
        .collect()
}

/// Records the exact resident routing of every step of the prompts file and checks the reference
/// tokens on the way. Each prompt starts from a zero state; its prompt tokens run in `Prompt` mode
/// except the last, which runs in `Generate` mode as in the decode test, and then `generate - 1`
/// more greedy tokens follow. End of sequence does not stop a prompt.
///
/// Environment: `FLASH_NEXT_ARTIFACT`, `FLASH_NEXT_CACHE_LIMIT`, `FLASH_NEXT_TRACE_PROMPTS` (the
/// prompts file), `FLASH_NEXT_TRACE_DIR` (output directory, outside Git), optional
/// `FLASH_NEXT_TRACE_ONLY` (comma-separated prompt ids), optional `FLASH_NEXT_TRACE_FULL=1` (also
/// writes `probs.bin`, little-endian F32 `[rows, experts]` in trace row order), and optional
/// `FLASH_NEXT_ARTIFACT_MANIFEST` (a manifest file of the artifact, for example its `SHA256SUMS`;
/// the header records its SHA-256 as `artifact_manifest_sha256`, or `unset` without it, and the
/// value can then be filled from the `SHA256SUMS` of the artifact).
#[test]
#[ignore = "needs the Flash-Next artifact on the Mac"]
fn resident_route_trace() {
    let _serial = crate::native_test_lock();
    let artifact = env_path("FLASH_NEXT_ARTIFACT");
    let prompts_path = env_path("FLASH_NEXT_TRACE_PROMPTS");
    let out_dir = env_path("FLASH_NEXT_TRACE_DIR");
    let cache_limit: u64 = std::env::var("FLASH_NEXT_CACHE_LIMIT")
        .expect("FLASH_NEXT_CACHE_LIMIT is not set")
        .parse()
        .expect("FLASH_NEXT_CACHE_LIMIT is not a byte count");
    let full = std::env::var("FLASH_NEXT_TRACE_FULL").is_ok_and(|v| v == "1");
    let only: Option<BTreeSet<u32>> = std::env::var("FLASH_NEXT_TRACE_ONLY").ok().map(|list| {
        list.split(',')
            .map(|id| {
                id.trim()
                    .parse()
                    .expect("FLASH_NEXT_TRACE_ONLY holds prompt ids")
            })
            .collect()
    });
    let prompts_bytes = std::fs::read(&prompts_path).unwrap();
    let prompts: Vec<TracePrompt> = trace_prompts(&prompts_bytes)
        .into_iter()
        .filter(|p| only.as_ref().is_none_or(|ids| ids.contains(&p.id)))
        .collect();
    assert!(!prompts.is_empty(), "no prompt selected");
    // The reference check compares `PROMPT` with `C5_TOKENS`, so a full run must contain it.
    assert!(
        only.is_some()
            || prompts
                .iter()
                .any(|p| p.tokens == PROMPT && p.generate >= MATCH),
        "the prompts file has no `PROMPT` entry with at least {MATCH} generated tokens"
    );
    std::fs::create_dir_all(&out_dir).unwrap();
    let index = std::fs::read(artifact.join("model.safetensors.index.json")).unwrap();
    let config = std::fs::read(artifact.join("config.json")).unwrap();
    let manifest: Option<Vec<u8>> = std::env::var("FLASH_NEXT_ARTIFACT_MANIFEST")
        .ok()
        .map(|path| std::fs::read(&path).unwrap_or_else(|error| panic!("{path}: {error}")));
    let (ckpt, counter) = open_counted(&artifact);
    let flash = FlashWeights::load(&ckpt, &config).unwrap();
    let mut kernels = MlxKernels::try_new().unwrap();
    kernels.set_cache_limit(cache_limit).unwrap();
    let mut decoder = ResidentDecoder::new(kernels, &ckpt, &flash).unwrap();
    drop(flash);
    let shape = decoder.weights.shape;
    let layers = decoder.weights.layers.len();
    // Every step must stay inside the resident QSA context: fail before the run, not after an hour.
    for prompt in &prompts {
        let last = prompt.tokens.len() - 1 + prompt.generate - 1;
        let pos = u32::try_from(last).unwrap();
        assert!(
            qsa::check_cache(last, pos).is_ok(),
            "prompt {}: position {last} passes the resident QSA context limit",
            prompt.id
        );
    }
    let mut meta = format!(
        "schema={TRACE_SCHEMA}\nprompts={}\nfull={full}\nswap_before={}\npool_bytes={}\n",
        join(&prompts.iter().map(|p| p.id).collect::<Vec<_>>()),
        swap_usage(),
        decoder.pool_bytes(),
    );
    let mut trace_file = BufWriter::new(std::fs::File::create(out_dir.join("trace.tsv")).unwrap());
    let mut tokens_file =
        BufWriter::new(std::fs::File::create(out_dir.join("tokens.tsv")).unwrap());
    let mut probs_file =
        full.then(|| BufWriter::new(std::fs::File::create(out_dir.join("probs.bin")).unwrap()));
    let header = [
        format!("schema={TRACE_SCHEMA}"),
        format!("engine_commit={}", engine_commit()),
        format!("index_sha256={}", sha256_hex(&index)),
        format!("config_sha256={}", sha256_hex(&config)),
        format!(
            "artifact_manifest_sha256={}",
            manifest.as_deref().map_or("unset".to_string(), sha256_hex)
        ),
        format!("prompts_sha256={}", sha256_hex(&prompts_bytes)),
        format!("cache_limit_bytes={cache_limit}"),
        format!("layers={layers}"),
        format!("experts={}", shape.moe.experts()),
        format!("top_k={}", shape.moe.top_k()),
        format!("renorm={}", shape.moe.renorm()),
        format!(
            "expert_bytes={}",
            join(&expert_bytes_per_layer(&ckpt, &decoder.weights))
        ),
    ];
    for line in header {
        writeln!(trace_file, "# {line}").unwrap();
    }
    writeln!(trace_file, "{}", tsv_columns(shape.moe.top_k())).unwrap();
    writeln!(
        tokens_file,
        "prompt\tpos\tmode\tinput\toutput\tseconds\tread_bytes\tsyncs"
    )
    .unwrap();

    decoder.route_trace = Some(RouteTrace::default());
    let mut state = decoder.zero_state().unwrap();
    let started_run = Instant::now();
    let (mut steps, mut generated_total) = (0usize, 0usize);
    for prompt in &prompts {
        decoder.reset(&mut state).unwrap();
        let is_c5 = prompt.tokens == PROMPT;
        let mut generated: Vec<u32> = Vec::with_capacity(prompt.generate);
        let mut next = 0;
        for n in 0..prompt.tokens.len() - 1 + prompt.generate {
            let input = prompt.tokens.get(n).copied().unwrap_or(next);
            let (mode, trace_mode) = if n + 1 < prompt.tokens.len() {
                (StepMode::Prompt, TraceMode::Prompt)
            } else {
                (StepMode::Generate, TraceMode::Generate)
            };
            let step = TraceStep {
                prompt: prompt.id,
                pos: state.pos(),
                mode: trace_mode,
            };
            decoder.route_trace.as_mut().unwrap().begin(step);
            let s0 = syncs(&decoder.kernels);
            let (b0, _) = counter.snapshot();
            let started = Instant::now();
            let result = decoder.step(&mut state, input, mode);
            let seconds = started.elapsed().as_secs_f64();
            let (b1, _) = counter.snapshot();
            let s1 = syncs(&decoder.kernels);
            let rows = decoder.route_trace.as_mut().unwrap().take();
            let checked = result
                .map_err(|error| format!("{error:?}"))
                .and_then(|out| check_step(&rows, &shape.moe, layers, step).map(|()| out));
            let out = match checked {
                Ok(out) => out,
                Err(message) => {
                    meta.push_str(&format!("error=prompt {} step {n}: {message}\n", prompt.id));
                    std::fs::write(out_dir.join("meta.txt"), &meta).unwrap();
                    panic!("prompt {} step {n}: {message}", prompt.id);
                }
            };
            for row in &rows {
                writeln!(trace_file, "{}", row.tsv().unwrap()).unwrap();
                if let Some(file) = probs_file.as_mut() {
                    for p in &row.probs {
                        file.write_all(&p.to_le_bytes()).unwrap();
                    }
                }
            }
            let output = out.token.map(u64::from);
            writeln!(
                tokens_file,
                "{}\t{}\t{}\t{input}\t{}\t{seconds:.6}\t{}\t{}",
                prompt.id,
                step.pos,
                if mode == StepMode::Prompt { "p" } else { "g" },
                opt(output),
                b1 - b0,
                opt(s1.zip(s0).map(|(after, before)| after - before)),
            )
            .unwrap();
            steps += 1;
            if mode == StepMode::Generate {
                let token = out.token.expect("a generated step returns a token");
                generated.push(token);
                next = token;
                // `PROMPT` repeats `C5_TOKENS`. Checked as soon as they exist.
                if is_c5 && generated.len() == MATCH {
                    assert_eq!(generated, C5_TOKENS, "the first 32 reference tokens");
                    meta.push_str("c5_tokens_match=true\n");
                }
            }
        }
        assert!(
            !is_c5 || generated.len() >= MATCH,
            "the C5 prompt generated {} tokens, fewer than {MATCH}",
            generated.len()
        );
        generated_total += generated.len();
        meta.push_str(&format!(
            "prompt{}_tokens={}\nprompt{}_generated={}\n",
            prompt.id,
            prompt.tokens.len(),
            prompt.id,
            join(&generated)
        ));
        trace_file.flush().unwrap();
        tokens_file.flush().unwrap();
    }
    meta.push_str(&format!(
        "steps={steps}\ngenerated_tokens={generated_total}\nrun_seconds={:.3}\n\
         swap_after={}\nrss_end={}\n",
        started_run.elapsed().as_secs_f64(),
        swap_usage(),
        rss_bytes(),
    ));
    if let Some(file) = probs_file.as_mut() {
        file.flush().unwrap();
    }
    std::fs::write(out_dir.join("meta.txt"), &meta).unwrap();
}
