//! The trunk executor of the lineage on the device: a chunk of token rows through the visits of
//! a model's layer list (the looped model's 56, the routed-expert model's 40). The walk and the
//! chunk mixers are in [`prefill`]; this module holds the step, its state and the one-row
//! mixers a decode step (a chunk of one row) takes.
//!
//! For the looped model the trunk is the CPU reference's
//! (`crate::family::qwen::sensenova::step::token`), visit for visit: mHC
//! visits keep both streams, route twice (attn, ffn) through the router table, and end with
//! `cur = 0.5 * (a + b)`; a visit whose physical layer first enters the mHC range resets the
//! streams from `cur` (a loop revisit of that physical layer resets them again, as the
//! vendor graph does). Plain visits residual-add the branch (fused into the post norm) and
//! the FFN output (a two-buffer rotation, because `row_add` writes a distinct `out`).
//!
//! A loop revisit (the same physical weights) owns its own state, so the GDN recurrent
//! state, the conv rings and the attention caches are per-VISIT, sized once in [`open`].
//! The embedding row is gathered by the caller (the Flash-Next pattern: the table stays on
//! the host) and handed to [`LoopStep::decode`] with the position; the logits row stays on
//! the device.
//!
//! Every buffer is allocated once in [`LoopStep::open`], and a decode step works in the first
//! row of the chunk's scratch; a step enqueues its launch list and
//! returns without synchronizing (the caller drains via [`LoopStep::stream`]), so the same
//! list is what a later capture slice replays.

use std::rc::Rc;

use crate::EngError;
use crate::family::project::cuda::CudaWeights;
use crate::family::session::Kept;
use crate::family::step::recipe::Layer;
#[cfg(test)]
use crate::family::step::recipe::{Forward, Mixer};
use crate::family::step::replay::Replay;
use crate::family::step::rows::Map;
use crate::family::step::scratch::{Count, OnDevice, Zeroed};
use crate::family::step::state::{Slots, State};
use crate::family::step::tail::Tail;
use crate::family::step::{attend, delta, keep};
use crate::kernels::cuda::resident::embed::EmbedTable;
use crate::kernels::cuda::resident::quantize_q8_1::q8_1_bytes;
use crate::kernels::cuda::resident::step_params::{
    OFFSET_KV_LEN, OFFSET_POS, OFFSET_STATUS, STEP_PARAMS_BYTES, StepParams, ensure_rows,
};
use crate::kernels::cuda::{CudaContext, DeviceBuffer, DeviceRange, DeviceStream, NativeWeight};

use crate::family::qwen::recipe::{Blocks, Model};
use crate::family::qwen::sensenova::step::yarn_inv_freq;
use crate::family::qwen::sensenova::weights::LoopShape;
use crate::family::qwen::sensenova::{LoopPlan, RopePlan};
use crate::family::step::moe;

mod prefill;

pub(crate) use prefill::PREFILL_ROWS;

const F32: usize = size_of::<f32>();
const ATTN_SCALE: f32 = 0.0625; // 1/sqrt(256)
const MAX_SPLITS: usize = 64;

/// A piece of the trunk's carried state, for the parity rigs; the visit index is what
/// names a row of their tables, through the derived `Debug` alone.
#[cfg(all(test, feature = "native-cuda"))]
#[expect(
    dead_code,
    reason = "read by the derived Debug, which dead code analysis ignores"
)]
#[derive(Debug)]
pub(crate) enum Carried {
    Hidden,
    K(usize),
    V(usize),
    Rec(usize),
    Ring(usize),
}

/// The trunk executor: the resolved weights, the per-visit state and the scratch rows.
/// Built once per engine; [`Self::reset`] starts a new session on the same buffers, and `decode`
/// runs one token.
pub(crate) struct LoopStep<'a> {
    /// The recorded decode step; first, so it drops before the buffers and the stream it uses.
    replay: Replay<usize>,
    /// The token tail, when the embedding table is on the device, and the record of the step
    /// that uses it: another list of launches than the step from an embedding row records.
    tail: Option<Tail>,
    tail_replay: Replay<usize>,
    /// The ids the recorded pick ranges over: a pick over another count records again.
    tail_vocab: usize,
    ctx: Rc<CudaContext>,
    stream: DeviceStream,
    weights: &'a CudaWeights,
    /// One layer per trunk visit: a loop revisit names the tensors of its physical layer again
    /// and carries its own state.
    layers: Vec<Layer>,
    delta_dims: delta::Dims,
    attn_dims: attend::Dims,
    head: &'a NativeWeight,
    output_norm: &'a DeviceBuffer,
    shape: LoopShape,
    family: &'static str,
    routing: Routing,
    /// The widths and the rows of a routed feed-forward, for a model that has one.
    moe: Option<(moe::Dims, moe::Scratch)>,
    cache_rows: usize,
    /// The projections' route: the f32 decode matvec over f32 rows, or the matvec over q8_1.
    dq: bool,
    /// The same choice for chunks of several rows.
    exact_chunks: bool,
    /// GDN state and conv rings, one slot per trunk visit (an attention visit's slot is
    /// unused; direct visit indexing keeps the mapping arithmetic-free).
    state: State,
    // scratch rows
    params: DeviceBuffer,
    ones_router: DeviceBuffer,
    inv_freq: DeviceBuffer,
    /// The scratch rows of a chunk, sized for `PREFILL_ROWS` rows; a decode step works in
    /// the first row of each.
    rows: prefill::Rows,
}

/// How the two-stream range of a model mixes its streams.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Routing {
    pub(crate) pre_scale: f32,
    pub(crate) iterations: usize,
    pub(crate) fixed_identity: bool,
}

fn native<'a>(
    weights: &'a CudaWeights,
    family: &'static str,
    name: &str,
) -> Result<&'a NativeWeight, EngError> {
    crate::family::step::rows::native(weights, family, name)
}

/// An F32 tensor the loader copied as read: the conv, `ssm_a`, the biases and every norm.
fn side<'a>(
    weights: &'a CudaWeights,
    family: &'static str,
    name: &str,
) -> Result<&'a DeviceBuffer, EngError> {
    crate::family::step::rows::raw(weights, family, name)
}

pub(super) fn floats(
    buffer: &DeviceBuffer,
    at: usize,
    len: usize,
) -> Result<DeviceRange<'_>, EngError> {
    buffer.range(at * F32, len * F32)
}

pub(super) fn bytes_range(buffer: &DeviceBuffer, len: usize) -> Result<DeviceRange<'_>, EngError> {
    buffer.range(0, len)
}

#[cfg(all(test, feature = "native-cuda"))]
fn q8_row(k: usize) -> Result<usize, EngError> {
    q8_1_bytes(k, 1).ok_or_else(|| EngError::BadRequest("a q8 row size overflows".to_owned()))
}

/// `FIFI_SENSENOVA_DECODE_GEMM=dq` puts the step's projections through the f32 decode matvec
/// (`CudaContext::matvec_dq_native`) over the f32 rows: every weight decoded per value, no
/// q8_1 step. The default keeps the matvec over the q8_1 rows. The route is opt-in and
/// default-off; a weight that is not Q4_K or Q6_K, or a `k` that is not a multiple of 256,
/// stops the step with the matvec's `BadRequest`, and nothing runs in its place. Read once:
/// every projection launch would otherwise take an env lock.
fn decode_dq() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("FIFI_SENSENOVA_DECODE_GEMM").is_ok_and(|v| v == "dq"))
}

/// One weight shape's sweep at one column: its description, then the microseconds of each group
/// on the matvec route and on the dq route.
#[cfg(all(test, feature = "native-cuda"))]
type ShapeSweep = (String, Vec<f64>, Vec<f64>);

pub(super) fn le_f32(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn qkvz_row(shape: &LoopShape) -> usize {
    2 * shape.delta_key_heads * shape.delta_key_dim
        + 2 * shape.delta_value_heads * shape.delta_value_dim
}

fn attn_row(shape: &LoopShape) -> usize {
    shape.heads_q * 2 * shape.head_dim + 2 * shape.heads_kv * shape.head_dim
}

/// The looped model of the lineage: the visits of `plan` over dense blocks with fused delta
/// inputs.
pub(crate) fn looped(shape: &LoopShape, plan: &LoopPlan, rope: &RopePlan) -> Model {
    Model {
        family: "sensenova",
        shape: *shape,
        visits: plan.visits(),
        blocks: Blocks {
            split_delta_inputs: false,
            experts: false,
        },
        experts: None,
        rope: rope.clone(),
    }
}

impl<'a> LoopStep<'a> {
    /// Builds the executor of the looped model over a completed device load, as the device
    /// tests open it. `cache_rows` is the session's attention cache depth; the GDN state and
    /// the conv rings follow the shape.
    #[cfg(test)]
    pub(crate) fn open(
        ctx: Rc<CudaContext>,
        weights: &'a CudaWeights,
        shape: &LoopShape,
        plan: &LoopPlan,
        rope: &RopePlan,
        cache_rows: usize,
    ) -> Result<Self, EngError> {
        let model = looped(shape, plan, rope);
        Self::open_model(ctx, weights, &model, plan.routing(), cache_rows)
    }

    /// The device bytes each cached position of `model` takes: the key and value planes of its
    /// attention visits. Counted from the state's own declaration, so it is what an open
    /// allocates per row.
    pub(crate) fn bytes_per_position(model: &Model) -> Result<u64, EngError> {
        let shape = &model.shape;
        let (_, slots) = crate::family::qwen::recipe::layers(&model.visits, model.blocks);
        let counted = |rows: usize| -> Result<u64, EngError> {
            let count = Count::default();
            State::new(
                &count,
                Slots {
                    delta: slots.delta,
                    ring: shape.conv_dim() * (shape.conv_kernel - 1),
                    record: shape.delta_value_heads * shape.delta_key_dim * shape.delta_value_dim,
                    attend: slots.attend,
                    plane: rows * shape.heads_kv * shape.head_dim * 2,
                    rows,
                },
            )?;
            Ok(count.total())
        };
        Ok(counted(2)? - counted(1)?)
    }

    /// The executor with the deepest cache the device admits, up to `wanted` positions. It opens
    /// the step once with a cache of one position, so that everything else the step holds is
    /// allocated, reads the free memory, and gives the cache what is left after `keep_free`.
    /// A model's trained context is `wanted`; the device decides how much of it is served.
    pub(crate) fn open_planned(
        ctx: Rc<CudaContext>,
        weights: &'a CudaWeights,
        model: &Model,
        routing: Routing,
        wanted: usize,
        keep_free: u64,
    ) -> Result<Self, EngError> {
        if wanted == 0 {
            return Err(EngError::BadRequest(format!(
                "{}: requested context length must be at least 1",
                model.family
            )));
        }
        let per_position = Self::bytes_per_position(model)?;
        let probe = Self::open_model(Rc::clone(&ctx), weights, model, routing, 1)?;
        let free = u64::try_from(ctx.memory_info()?.free).unwrap_or(u64::MAX);
        drop(probe);
        let affordable = free
            .saturating_sub(keep_free)
            .checked_div(per_position)
            .unwrap_or(if free >= keep_free { u64::MAX } else { 0 });
        let floor = wanted.min(prefill::PREFILL_ROWS);
        let rows = wanted.min(usize::try_from(affordable).unwrap_or(usize::MAX));
        if rows < floor {
            return Err(EngError::MemoryPlan(format!(
                "{}: the device has {free} bytes free after the step's scratch; with {keep_free} kept free that is {affordable} cached positions of {per_position} bytes, fewer than the minimum {floor}",
                model.family
            )));
        }
        Self::open_model(ctx, weights, model, routing, rows)
    }

    /// The positions the step's cache holds.
    pub(crate) fn cache_rows(&self) -> usize {
        self.cache_rows
    }

    /// The executor of any model of the lineage over a completed device load.
    pub(crate) fn open_model(
        ctx: Rc<CudaContext>,
        weights: &'a CudaWeights,
        model: &Model,
        routing: Routing,
        cache_rows: usize,
    ) -> Result<Self, EngError> {
        let (shape, rope, family) = (&model.shape, &model.rope, model.family);
        let h = shape.hidden;
        let rec_floats = shape.delta_value_heads * shape.delta_key_dim * shape.delta_value_dim;
        let ring_floats = shape.conv_dim() * (shape.conv_kernel - 1);
        let cache_bytes = cache_rows * shape.heads_kv * shape.head_dim * 2;
        // One Phys per PHYSICAL layer (a loop revisit reads the same weights through its
        // own visit state); `recurrent` is block-indexed, so the mixer test goes through
        // `block(p)` exactly as the load template does.
        let map = Map {
            family,
            weights,
            derived: None,
        };
        let attn_dims = attend::Dims {
            heads: shape.heads_q,
            kv_heads: shape.heads_kv,
            head_dim: shape.head_dim,
            scale: ATTN_SCALE,
            eps: shape.eps,
            rot_dims: rope.dims,
            freq_base: rope.freq_base,
            row_splits: MAX_SPLITS,
        };
        let dims = delta::Dims {
            key_heads: shape.delta_key_heads,
            key_dim: shape.delta_key_dim,
            value_heads: shape.delta_value_heads,
            value_dim: shape.delta_value_dim,
            conv_kernel: shape.conv_kernel,
            conv_dim: shape.conv_dim(),
        };
        let (layers, slots) = crate::family::qwen::recipe::layers(&model.visits, model.blocks);
        let moe = match model.experts {
            Some((experts, used)) => {
                let dims = moe::Dims {
                    hidden: h,
                    inner: shape.feed_forward,
                    experts,
                    used,
                    gated_shared: true,
                };
                let scratch =
                    moe::Scratch::new(&Zeroed(OnDevice(&ctx)), dims, prefill::PREFILL_ROWS)?;
                Some((dims, scratch))
            }
            None => None,
        };
        layers
            .iter()
            .try_for_each(|layer| layer.check(&map, dims, Some(attn_dims)))?;
        let step = Self {
            stream: ctx.stream()?,
            layers,
            delta_dims: dims,
            attn_dims,
            head: native(weights, family, "output.weight")?,
            output_norm: side(weights, family, "output_norm.weight")?,
            state: State::new(
                &Zeroed(OnDevice(&ctx)),
                Slots {
                    delta: slots.delta,
                    ring: ring_floats,
                    record: rec_floats,
                    attend: slots.attend,
                    plane: cache_bytes,
                    rows: cache_rows,
                },
            )?,
            params: ctx.upload(&[0u8; STEP_PARAMS_BYTES])?,
            ones_router: ctx.upload(&le_f32(&vec![1.0; 2 * h]))?,
            inv_freq: ctx.upload(&le_f32(&yarn_inv_freq(rope)))?,
            replay: Replay::default(),
            tail: None,
            tail_replay: Replay::default(),
            tail_vocab: 0,
            rows: prefill::open_rows(&ctx, shape, ctx.sm_count() as usize)?,
            shape: *shape,
            family,
            routing,
            moe,
            cache_rows,
            dq: decode_dq(),
            exact_chunks: prefill::exact_gemm(),
            ctx,
            weights,
        };
        step.check_geometry()?;
        Ok(step)
    }

    /// The pinned-model geometry the kernels rely on; a different Loop shape needs this
    /// list revisited, so it fails closed here and not inside a launch.
    fn check_geometry(&self) -> Result<(), EngError> {
        let s = &self.shape;
        let refuse = |what: &str| {
            Err(EngError::BadRequest(format!(
                "the Loop shape is not the pinned geometry: {what}"
            )))
        };
        if s.head_dim != 256 {
            return refuse("head_dim is not 256");
        }
        if s.heads_q != 4 * s.heads_kv && s.heads_q != 8 * s.heads_kv {
            return refuse("the attention is not 4 or 8 query heads per kv head");
        }
        if s.delta_key_dim != 128 || s.delta_value_dim != 128 || s.conv_kernel != 4 {
            return refuse("the GDN geometry is not kd=vd=128, ck=4");
        }
        if !s.hidden.is_multiple_of(256) || !s.feed_forward.is_multiple_of(32) {
            return refuse("hidden or feed_forward breaks the norm or swiglu units");
        }
        Ok(())
    }

    /// One token: `embed_row` is the gathered embedding row of the drawn token at `pos`.
    /// Enqueues the trunk and the head; the caller drains and reads [`Self::logits_row`].
    pub(crate) fn decode(&mut self, embed_row: &[f32], pos: usize) -> Result<(), EngError> {
        self.decode_inner(embed_row, pos, None)
    }

    /// The parity rig's twin: also snapshots the residual row after every visit.
    #[cfg(all(test, feature = "native-cuda"))]
    pub(crate) fn decode_traced(
        &mut self,
        embed_row: &[f32],
        pos: usize,
        trace: &mut Vec<Vec<f32>>,
    ) -> Result<(), EngError> {
        self.decode_inner(embed_row, pos, Some(trace))
    }

    fn decode_inner(
        &mut self,
        embed_row: &[f32],
        pos: usize,
        trace: Option<&mut Vec<Vec<f32>>>,
    ) -> Result<(), EngError> {
        let h = self.shape.hidden;
        if embed_row.len() != h {
            return Err(EngError::BadRequest(
                "the embed row length is not hidden".to_owned(),
            ));
        }
        ensure_rows(pos, 1, self.cache_rows)?;
        if trace.is_some() {
            self.walk(&le_f32(embed_row), 1, pos, trace)?;
            return self.head(0, self.dq);
        }
        self.feed(&le_f32(embed_row), 1, pos)?;
        let mut replay = std::mem::take(&mut self.replay);
        let left = replay.run(&self.ctx, &self.stream, || {
            self.visits(1, pos, None)?;
            self.head(0, self.dq)?;
            Ok(self.rows.core.cur.get())
        });
        self.replay = replay;
        self.rows.core.cur.set(left?);
        Ok(())
    }

    /// Drops the recorded steps: the next steps record again.
    pub(super) fn forget_records(&mut self) {
        self.replay = Replay::default();
        self.tail_replay = Replay::default();
    }

    /// Gives the step the embedding table of the file, on the device: a decode step can then
    /// take a token id and leave the id of the largest logit.
    pub(crate) fn set_embedding(&mut self, table: EmbedTable) -> Result<(), EngError> {
        self.tail = Some(Tail::new(&self.ctx, table)?);
        self.tail_replay = Replay::default();
        Ok(())
    }

    /// The context the step runs on.
    pub(crate) fn context(&self) -> &CudaContext {
        &self.ctx
    }

    /// Whether [`Self::decode_token`] is available.
    pub(crate) fn takes_tokens(&self) -> bool {
        self.tail.is_some()
    }

    /// One decode step from a token id at `pos`, wholly on the device: the embedding row is
    /// gathered there, the step runs, and the id of the largest of the first `vocab` logits is
    /// what the host reads (a tie goes to the lower id). The logits row stays on the device as
    /// after [`Self::decode`].
    pub(crate) fn decode_token(
        &mut self,
        token: u32,
        pos: usize,
        vocab: usize,
    ) -> Result<u32, EngError> {
        let h = self.shape.hidden;
        if token as usize >= self.shape.vocab || vocab == 0 || vocab > self.shape.vocab {
            return Err(EngError::BadRequest(format!(
                "{}: token {token} or a pick over {vocab} ids is outside the vocabulary of {}",
                self.family, self.shape.vocab
            )));
        }
        ensure_rows(pos, 1, self.cache_rows)?;
        let Some(tail) = &self.tail else {
            return Err(EngError::BadRequest(format!(
                "{}: the step has no embedding table on the device",
                self.family
            )));
        };
        let params = StepParams::new(token as i32, pos as i32)?;
        self.params
            .write_at_enqueue(&self.stream, 0, &params.to_bytes())?;
        if self.tail_vocab != vocab {
            (self.tail_replay, self.tail_vocab) = (Replay::default(), vocab);
        }
        let mut replay = std::mem::take(&mut self.tail_replay);
        let left = replay.run(&self.ctx, &self.stream, || {
            let row = self.rows.core.resid[0].range(0, h * 4)?;
            tail.gather(&self.ctx, &self.stream, &self.params, row)?;
            self.visits(1, pos, None)?;
            self.head(0, self.dq)?;
            let logits = &self.rows.core.logits;
            tail.pick(&self.ctx, &self.stream, logits, vocab, &self.params)?;
            Ok(self.rows.core.cur.get())
        });
        self.tail_replay = replay;
        self.rows.core.cur.set(left?);
        tail.picked(&self.stream)
    }

    /// Whether the last step from a token id replayed its recorded step.
    #[cfg(test)]
    pub(crate) fn token_step_replayed(&self) -> bool {
        self.tail_replay.recorded()
    }

    /// Whether the last decode step replayed the recorded step.
    #[cfg(test)]
    pub(crate) fn replayed(&self) -> bool {
        self.replay.recorded()
    }

    /// Puts every projection of this card on the decoding routes, or back on the quantized
    /// ones: the anatomy test compares the routes from one carried state.
    #[cfg(all(test, feature = "native-cuda"))]
    pub(crate) fn set_exact(&mut self, on: bool) {
        self.set_exact_routes(on, on);
    }

    /// The decoding routes for one-row steps and for chunks, each on its own: an accuracy run
    /// tells which of the two a distance from the reference comes from.
    #[cfg(all(test, feature = "native-cuda"))]
    pub(crate) fn set_exact_routes(&mut self, rows: bool, chunks: bool) {
        (self.dq, self.exact_chunks) = (rows, chunks);
        self.forget_records();
    }

    /// One chunk at `p0` with row 0 of every visit's mixer output and residual pushed to `trace`.
    #[cfg(all(test, feature = "native-cuda"))]
    pub(crate) fn chunk_traced(
        &mut self,
        embed_rows: &[f32],
        p0: usize,
        trace: &mut Vec<Vec<f32>>,
    ) -> Result<(), EngError> {
        let m = embed_rows.len() / self.shape.hidden;
        self.forget_records();
        self.walk(&le_f32(embed_rows), m, p0, Some(trace))
    }

    /// Queues the head on row `row` of the chunk just walked, on the decoding route when the
    /// one-row step takes it: the accuracy probe reads every row of a chunk this way.
    #[cfg(all(test, feature = "native-cuda"))]
    pub(crate) fn head_of_row(&self, row: usize) -> Result<(), EngError> {
        self.head(row, self.dq)
    }

    /// The logits row (device floats); valid after the stream drained.
    #[cfg(test)]
    pub(crate) fn logits_row(&self) -> &DeviceBuffer {
        &self.rows.core.logits
    }

    /// Device buffers for logits readback and host-equal shortlist sampling.
    pub(crate) fn shortlist_io(&self) -> crate::family::step::stepper::ShortlistIo<'_> {
        self.rows
            .core
            .shortlist_io(&self.ctx, &self.stream, self.shape.vocab)
    }

    /// Clears the carried state for a new session on the same card: the GDN state, the conv
    /// rings and the attention planes return to the zeros of [`LoopStep::open`] (the planes too,
    /// so a session that ended on a non-finite value leaves nothing a masked read could meet),
    /// and the residual cursor returns to its start. Only the carried state is cleared: a step
    /// overwrites every scratch row before its first read. The fills are enqueued on the step's
    /// stream, behind a step that is still in flight.
    pub(crate) fn reset(&mut self) -> Result<(), EngError> {
        self.state.reset(&self.stream)?;
        self.rows.core.cur.set(0);
        Ok(())
    }

    /// Starts a new sequence on the rows the planes hold: the GDN state and the conv rings
    /// return to zeros and the planes stay. A step reads no cache row at or above its position,
    /// so the new sequence has the bits it has after [`Self::reset`], and a sequence that
    /// returns to a kept state finds its rows below that position.
    pub(crate) fn reset_running(&mut self) -> Result<(), EngError> {
        keep::reset_running(&self.state, &self.stream)?;
        self.rows.core.cur.set(0);
        Ok(())
    }

    /// The bytes of one [`Kept`] of this step.
    pub(crate) fn kept_bytes(&self) -> u64 {
        keep::bytes(&self.state)
    }

    /// The GDN state and the conv rings after `rows` positions, once the queued steps ran.
    pub(crate) fn keep(&self, rows: usize) -> Result<Kept, EngError> {
        keep::keep(&self.state, &self.stream, rows)
    }

    /// Returns to a kept state. The copies land in the buffers the recorded steps name, so both
    /// records stay valid: the next decode step replays.
    pub(crate) fn put_back(&mut self, kept: &Kept) -> Result<(), EngError> {
        keep::put_back(&self.state, &self.stream, kept)?;
        self.rows.core.cur.set(0);
        Ok(())
    }

    /// Fills every cache plane from row `from` on with the two bytes `word`, for the test that
    /// shows a step reads no row at or above its position.
    #[cfg(all(test, feature = "native-cuda"))]
    pub(crate) fn fill_planes(&self, from: usize, word: [u8; 2]) -> Result<(), EngError> {
        let row = self.shape.heads_kv * self.shape.head_dim * 2;
        let fill: Vec<u8> = word.repeat((self.cache_rows - from) * row / 2);
        for buffer in self.state.buffers().skip(self.state.running().count()) {
            buffer.write_at(&self.stream, from * row, &fill)?;
        }
        Ok(())
    }

    /// The stored block and head weight bytes one decode step reads: every visit's dense
    /// weights (the loop layers once per revisit) and the head.
    #[cfg(all(test, feature = "native-cuda"))]
    pub(super) fn step_weight_bytes(&self) -> usize {
        let trunk: usize = self.sweep_weights().iter().map(|w| w.bytes()).sum();
        trunk
    }

    /// Device microbenchmark, not a gate: every matvec the decode walk issues, at `ncols`
    /// activation columns, so the weights are read once for `ncols` tokens the way a batched
    /// step would read them. Returns the microseconds of one sweep for each of `groups`
    /// groups: a group queues `sweeps` sweeps (about 1,825 host launches for five) and
    /// synchronizes once, and its elapsed time is divided by `sweeps`. The first group warms
    /// the card up and is dropped. The inputs are zero and the outputs are never read. The
    /// figure leaves out the step's drain points and its norm and mHC glue kernels. It does
    /// include any device idle where the host launch stream falls behind inside a group,
    /// which is most likely for the small matrices. Above four columns `matvec_native` takes
    /// the vendored MMVQ geometry that the decode, at one column, never takes. With `dq`, every
    /// weight goes through the f32 decode matvec over one f32 row instead, whatever `ncols`
    /// says: the launch the step issues on its dq route.
    #[cfg(all(test, feature = "native-cuda"))]
    pub(super) fn sweep_micros(
        &self,
        ncols: usize,
        sweeps: usize,
        groups: usize,
        dq: bool,
    ) -> Result<Vec<f64>, EngError> {
        self.sweep_list_micros(&self.sweep_weights(), ncols, sweeps, groups, dq)
    }

    /// The same sweep for each distinct weight shape (k, rows, quant) at one column, both
    /// routes: the shape, then the microseconds of each group on the matvec route and on the
    /// dq route.
    #[cfg(all(test, feature = "native-cuda"))]
    pub(super) fn sweep_shape_micros(
        &self,
        sweeps: usize,
        groups: usize,
    ) -> Result<Vec<ShapeSweep>, EngError> {
        use crate::kernels::cuda::resident::matvec_q::QuantType;
        let mut shapes: Vec<(usize, usize, QuantType, Vec<&'a NativeWeight>)> = Vec::new();
        for w in self.sweep_weights() {
            let key = (w.k(), w.n_rows(), w.quant());
            match shapes.iter_mut().find(|s| (s.0, s.1, s.2) == key) {
                Some(s) => s.3.push(w),
                None => shapes.push((key.0, key.1, key.2, vec![w])),
            }
        }
        let mut out = Vec::with_capacity(shapes.len());
        for (k, rows, quant, weights) in shapes {
            let shape = format!("k {k} rows {rows} {quant:?}, {} weights", weights.len());
            let matvec = self.sweep_list_micros(&weights, 1, sweeps, groups, false)?;
            let dq = self.sweep_list_micros(&weights, 1, sweeps, groups, true)?;
            out.push((shape, matvec, dq));
        }
        Ok(out)
    }

    /// Every weight the decode walk's projections read, in walk order.
    #[cfg(all(test, feature = "native-cuda"))]
    fn sweep_weights(&self) -> Vec<&'a NativeWeight> {
        let map = Map {
            family: self.family,
            weights: self.weights,
            derived: None,
        };
        let mut weights = Vec::new();
        for layer in &self.layers {
            match &layer.mixer {
                Mixer::Delta { names, .. } => {
                    let resolved = names.resolve(&map, self.delta_dims).expect("a delta layer");
                    weights.extend(resolved.projections());
                }
                Mixer::Attend { stem, .. } => {
                    let resolved = attend::resolve(&map, stem, self.attn_dims).expect("a layer");
                    weights.extend(resolved.projections());
                }
                Mixer::Latent { .. } | Mixer::Relative { .. } => {}
            }
            if let Forward::Dense(names) = &layer.forward {
                weights.extend(names.iter().map(|n| map.native(n).expect("a projection")));
            }
        }
        weights.push(self.head);
        weights
    }

    #[cfg(all(test, feature = "native-cuda"))]
    fn sweep_list_micros(
        &self,
        weights: &[&NativeWeight],
        ncols: usize,
        sweeps: usize,
        groups: usize,
        dq: bool,
    ) -> Result<Vec<f64>, EngError> {
        let (k_max, rows_max) = weights
            .iter()
            .fold((0, 0), |(k, n), w| (k.max(w.k()), n.max(w.n_rows())));
        let x = self.ctx.upload(&vec![0u8; ncols * q8_row(k_max)?])?;
        let x_f32 = self.ctx.upload(&le_f32(&vec![0.0; k_max]))?;
        let y = self.ctx.upload(&le_f32(&vec![0.0; ncols * rows_max]))?;
        let mut out = Vec::with_capacity(groups);
        for _ in 0..=groups {
            let t0 = std::time::Instant::now();
            for _ in 0..sweeps {
                for w in weights {
                    let y_w = y.range(0, ncols * w.n_rows() * F32)?;
                    if dq {
                        self.ctx.matvec_dq_native(
                            &self.stream,
                            w,
                            x_f32.range(0, w.k() * F32)?,
                            y_w,
                        )?;
                    } else {
                        let x_w = x.range(0, ncols * q8_row(w.k())?)?;
                        self.ctx.matvec_native(&self.stream, w, x_w, y_w, ncols)?;
                    }
                }
            }
            self.stream.synchronize()?;
            out.push(t0.elapsed().as_secs_f64() * 1e6 / sweeps as f64);
        }
        out.remove(0);
        Ok(out)
    }

    /// The carried state for the parity rigs, as (piece, buffer, byte offset, byte length):
    /// the hidden row, every attention visit's K and V planes over the first `rows` rows
    /// (fp16), and every GDN visit's record and conv ring (f32).
    #[cfg(all(test, feature = "native-cuda"))]
    pub(crate) fn carried(
        &self,
        rows: usize,
    ) -> Result<Vec<(Carried, &DeviceBuffer, usize, usize)>, EngError> {
        let shape = &self.shape;
        let mut out = Vec::new();
        for (il, layer) in self.layers.iter().enumerate() {
            if let Mixer::Attend { slot, .. } = &layer.mixer {
                let plane = rows * shape.heads_kv * shape.head_dim * 2;
                let cache = self.state.cache(*slot)?;
                out.push((Carried::K(il), cache.k, 0, plane));
                out.push((Carried::V(il), cache.v, 0, plane));
            } else {
                let rec = shape.delta_value_heads * shape.delta_key_dim * shape.delta_value_dim;
                let ring = shape.conv_dim() * (shape.conv_kernel - 1);
                let (rings, records) = self.state.delta_buffers()?;
                out.push((Carried::Rec(il), records, il * rec * 4, rec * 4));
                out.push((Carried::Ring(il), rings, il * ring * 4, ring * 4));
            }
        }
        out.push((
            Carried::Hidden,
            &self.rows.core.resid[self.rows.core.cur.get()],
            0,
            shape.hidden * 4,
        ));
        Ok(out)
    }

    /// The step's stream: a read enqueued on it lands behind the step's kernels.
    #[cfg(test)]
    pub(crate) fn stream(&self) -> &DeviceStream {
        &self.stream
    }

    fn params_word(&self, offset: usize) -> Result<DeviceRange<'_>, EngError> {
        self.params.range(offset, 4)
    }
}
