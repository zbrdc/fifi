//! Whole-token transaction, driven by the shared walk over the Inkling recipe: the walk mixes,
//! seals and adds on the depth site, whose two planes alternate per pass (the plane a rewind
//! leaves is the bank of the next pass) and whose mark is the sealed prelude, and each visit's
//! relative attention and stacked routed experts run on the visit's arena. Sealed prelude, previous-pass bank, and each
//! visit's K/V rows and convolution state never cross PCIe during the chain. One coalesced readback
//! holds the logits and every status word. A committed token leaves the device
//! owning the rows; the host keeps cursors until an explicit event calls
//! [`TokenWorkspace::materialize`].
mod admission;
mod shadow;
use super::*;
use crate::family::project::cuda::{CudaWeight, CudaWeights};
use crate::family::step::depth::{self as step_depth, Depth, DepthSite};
use crate::family::step::recipe::{Layer, Model, Walk};
use crate::family::step::rows::StepIo;
use crate::family::step::scratch::{Core, OnDevice, Widths};
use crate::family::step::state::{Slots, State};
use crate::family::step::walk;
use crate::family::step::walk::Half;
use crate::kernels::cuda::resident::depth_attention::{MixCall, MixShape};
use crate::kernels::cuda::resident::word_rows::MAX_WORD_ROWS;
use crate::weights::GgufType;

/// Declines the whole-token path; the first decline of each reason is logged once per process.
fn declined<T>(reason: &'static str) -> Result<Option<T>, EngError> {
    static SEEN: std::sync::Mutex<Vec<&'static str>> = std::sync::Mutex::new(Vec::new());
    let mut seen = SEEN
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if !seen.contains(&reason) {
        seen.push(reason);
        eprintln!("Inkling CUDA declined whole resident decode token: {reason}");
    }
    Ok(None)
}

struct Slot {
    w: Workspace,
    /// Journal copy of the arena status words that run up to `layout.log`.
    receipt: Region,
    status_bytes: usize,
}

/// What the host needs to adopt rows read back from the device.
pub(in super::super) struct Cursor<'a> {
    pub(super) states: &'a mut [VisitState],
    /// Committed rows: `0..position` are in each ring.
    pub(super) position: usize,
    /// Rows kept per convolution history.
    pub(super) past: usize,
    /// K/V ring capacity per visit slot.
    pub(super) capacities: &'a [usize],
}

/// What the shared walk drives the chain with beside the slots: the recipe, the sealed rows and
/// the plane of each visit, the walk's rows (the depth mixes land in `input`, the visits write
/// `core.mix_out` and `core.forward_out`), the depth queries by recipe name, and the depth
/// site's health word with its receipt in the journal.
struct Chain {
    layers: Vec<Layer>,
    blocks: Vec<usize>,
    planes: Vec<usize>,
    core: Core,
    state: State,
    queries: CudaWeights,
    input: DeviceBuffer,
    health: DeviceBuffer,
    receipt: Region,
}

impl Chain {
    fn open(f: &CpuForward, ctx: &CudaContext, receipt: Region) -> Result<Self, EngError> {
        use crate::family::inkling2::recipe;
        let h = f.hidden();
        let layers = recipe::layers(&f.spec);
        if layers.len() != f.visits.len() {
            return Err(bad("the recipe and the visits differ in length"));
        }
        // The rows the recipe names: the first row of a layer's attention queries and its MLP
        // query, once per weight layer.
        let mut queries = Vec::new();
        let mut seen = Vec::new();
        for visit in &f.visits {
            let layer = visit.weight_layer;
            if seen.contains(&layer) {
                continue;
            }
            seen.push(layer);
            let mut rows = vec![(
                "ffn",
                f.weights
                    .vec(&format!("model.llm.attn_res.mlp_queries.{layer}"), h)?,
            )];
            if let Some(query) = layer.checked_sub(1) {
                let name = format!("model.llm.attn_res.attn_queries.{query}");
                rows.push(("attn", f.weights.row(&name, 0, h)?));
            }
            for (half, row) in rows {
                let mut bytes = Vec::new();
                append(&mut bytes, &row);
                queries.push(CudaWeight::Raw {
                    name: format!("blk.{layer}.{half}_res_query.weight"),
                    ty: GgufType::F32,
                    buffer: ctx.upload(&bytes)?,
                });
            }
        }
        let device = OnDevice(ctx);
        let widths = Widths {
            rows: 1,
            hidden: h,
            streams: 1,
            proj: h,
            conv: h,
            read: h,
            product: h,
            vocab: 1,
            fixup: WORD,
        };
        let none = Slots {
            delta: 0,
            ring: 0,
            record: 0,
            attend: 0,
            plane: 0,
            rows: 0,
        };
        Ok(Self {
            blocks: step_depth::blocks(&layers),
            planes: step_depth::planes(&layers, 2),
            layers,
            core: Core::new(&device, widths)?,
            state: State::new(&device, none)?,
            queries: CudaWeights::from_weights(queries)?,
            input: ctx.allocate(h * WORD)?,
            health: ctx.allocate(WORD)?,
            receipt,
        })
    }
}

/// One token's step for the shared walk: the depth site over the pass planes, and each visit's
/// relative attention and stacked routed experts on its arena.
struct ChainStep<'a> {
    f: &'a CpuForward,
    cuda: &'a GgufCuda,
    slots: &'a [Slot],
    chain: &'a Chain,
    depth: &'a [DeviceBuffer; 2],
    prelude: &'a DeviceBuffer,
    params: &'a DeviceBuffer,
    stream: &'a DeviceStream,
}

impl ChainStep<'_> {
    fn depth(&self) -> Depth<'_> {
        let h = self.f.hidden();
        Depth {
            width: h,
            capacity: self.depth[0].bytes() / (h * WORD) - 1,
            eps: MIX_EPS,
            states: &self.depth[0],
            bank: Some(&self.depth[1]),
            mark: Some(self.prelude),
            health: &self.chain.health,
        }
    }

    /// Visit `visit` on its plane, its bank the other plane, reading and writing the walk's rows.
    fn execution(&self, visit: usize) -> Result<Execution<'_>, EngError> {
        let slot = self
            .slots
            .get(visit)
            .ok_or_else(|| bad("the walk visits past the token chain"))?;
        let plane = *self
            .chain
            .planes
            .get(visit)
            .ok_or_else(|| bad("the walk visits past the token chain"))?;
        let depth = self.depth();
        let (d, b) = (depth.plane(plane)?, depth.bank(plane)?);
        let row = self.f.hidden() * WORD;
        let core = &self.chain.core;
        Ok(Execution {
            plan: &slot.w.plan,
            layout: &slot.w.layout,
            weights: &slot.w.weights,
            arena: &slot.w.arena,
            stream: self.stream,
            context: self.cuda.context(),
            shared: Some(Shared {
                depth: d.range(0, d.bytes())?,
                bank: b.range(0, b.bytes())?,
                pos: self.params.range(WORD, WORD)?,
                walk: Some(WalkRows {
                    input: self.chain.input.range(0, row)?,
                    mix_out: core.mix_out.range(0, row)?,
                    forward_out: core.forward_out.range(0, row)?,
                }),
            }),
        })
    }
}

impl Model for ChainStep<'_> {
    fn family(&self) -> &'static str {
        "inkling2"
    }

    fn layers(&self) -> &[Layer] {
        &self.chain.layers
    }

    fn io(&self, exact: bool) -> StepIo<'_> {
        StepIo {
            family: "inkling2",
            ctx: self.cuda.context(),
            stream: self.stream,
            weights: &self.chain.queries,
            derived: None,
            mmq_q8: &self.chain.core.read_q8,
            fixup: &self.chain.core.fixup,
            exact,
        }
    }

    fn core(&self) -> &Core {
        &self.chain.core
    }

    fn state(&self) -> &State {
        &self.chain.state
    }

    fn hidden(&self) -> usize {
        self.f.hidden()
    }

    fn eps(&self) -> f32 {
        self.f.math.rms_norm_eps
    }

    fn depth_bank(
        &self,
        visit: usize,
        half: Half,
    ) -> Result<Option<step_depth::BankRead<'_>>, EngError> {
        if half == Half::Mixer {
            self.execution(visit)?.bank_read()
        } else {
            Ok(None)
        }
    }

    fn relative(&self, visit: usize, _slot: usize, _n: usize, p0: usize) -> Result<(), EngError> {
        let ex = self.execution(visit)?;
        ex.attention(self.cuda, p0, ex.layout.attention_input)
    }

    fn stacked_routed(&self, visit: usize, _n: usize, _p0: usize) -> Result<(), EngError> {
        self.execution(visit)?.forward(self.cuda, false)
    }

    fn depth_site(&self) -> Result<DepthSite<'_>, EngError> {
        Ok(DepthSite {
            depth: self.depth(),
            blocks: &self.chain.blocks,
            planes: &self.chain.planes,
            input: &self.chain.input,
        })
    }
}

pub(in super::super) struct TokenWorkspace {
    slots: Vec<Slot>,
    chain: Chain,
    stream: DeviceStream,
    depth: [DeviceBuffer; 2],
    /// The depth site's mark: the states as the first pass's first visit found them.
    prelude: DeviceBuffer,
    embedding: crate::kernels::cuda::resident::embed::EmbedTable,
    constants: DeviceBuffer,
    params: DeviceBuffer,
    scratch: DeviceBuffer,
    journal: DeviceBuffer,
    pinned: PinnedBuf,
    logits: Region,
    health: Region,
    /// Per slot: the arenas hold the histories of the slot's last committed token, so the next
    /// chain token skips the boot upload. Cleared when the host materializes rows.
    warm: Vec<bool>,
}

impl TokenWorkspace {
    fn layout(f: &CpuForward) -> Vec<crate::family::inkling2::recipe::Visit> {
        use crate::family::inkling2::recipe;
        recipe::token_layout(&recipe::layers(&f.spec))
    }

    fn eligible(cuda: &GgufCuda, v: LayerVisit, pos: usize, banked: bool) -> bool {
        if !banked {
            return cuda.resident_plain_supported(v, pos);
        }
        match v.attention {
            Attention::Local => cuda.resident_local_decode_supported(v, pos),
            Attention::Global => cuda.resident_global_supported(v, pos),
        }
    }

    pub(in super::super) fn open(
        f: &CpuForward,
        cuda: &GgufCuda,
        states: &[VisitState],
    ) -> Result<Option<Self>, EngError> {
        Self::open_limited(f, cuda, states, usize::MAX)
    }

    fn open_limited(
        f: &CpuForward,
        cuda: &GgufCuda,
        states: &[VisitState],
        limit: usize,
    ) -> Result<Option<Self>, EngError> {
        let h = f.hidden();
        let vocab = f.spec.unpadded_vocab_size as usize;
        if f.draft_window_enabled() || !f.taps.is_empty() {
            return Ok(None);
        }
        if !Self::valid_geometry(f, states) {
            return declined("the loop geometry is not one this workspace runs");
        }
        if !cuda.resident_head_supported([vocab, h]) {
            return declined("the output head has no resident GEMV");
        }
        let geometry = Self::layout(f);
        if f.visits.iter().any(|&v| {
            states[v.state_slot].position() != f.position || states[v.state_slot].is_resident()
        }) {
            // A resident state has no host rows to boot from; the host path materializes first.
            return Ok(None);
        }
        if f.visits
            .iter()
            .zip(&geometry)
            .any(|(&v, g)| !Self::eligible(cuda, v, f.position, g.3.is_some()))
        {
            return declined("a visit's attention has no resident decode at this position");
        }
        let Weights::Gguf(weights) = &f.weights else {
            return Ok(None);
        };
        let ctx = cuda.context();
        if !Self::fits(f, cuda, &geometry, limit.min(ctx.memory_info()?.free))? {
            return declined("a layer has no resident kernel or the workspace does not fit");
        }
        let mut end = 0;
        let logits = reserve(&mut end, vocab);
        let health = reserve(&mut end, 4);
        let (mut slots, mut depth_bytes, mut cold_bytes) = (Vec::new(), 0, 8);
        for (&visit, (_, _, blocks, bank, _)) in f.visits.iter().zip(geometry) {
            let Some(w) = Workspace::open_geometry(f, cuda, visit, blocks, bank)? else {
                return declined("a visit workspace declined");
            };
            depth_bytes = depth_bytes.max(w.layout.depth.bytes);
            cold_bytes += w.layout.histories.iter().map(|r| r.bytes).sum::<usize>();
            let status_bytes = w.layout.log.start - w.layout.attention_status.start;
            let receipt = reserve(&mut end, status_bytes / WORD);
            slots.push(Slot {
                w,
                receipt,
                status_bytes,
            });
        }
        let mut c = Vec::new();
        for name in [
            "model.llm.embed_norm.weight",
            "model.llm.attn_res.output_query",
            "model.llm.norm.weight",
        ] {
            append(&mut c, &f.weights.vec(name, h)?);
        }
        let constants = ctx.upload(&c)?;
        let chain = Chain::open(f, ctx, reserve(&mut end, 1))?;
        // The device word pair: the token id and the position.
        let params = ctx.upload(&[0; 8])?;
        let depth = [ctx.allocate(depth_bytes)?, ctx.allocate(depth_bytes)?];
        let prelude = ctx.allocate(depth_bytes)?;
        let journal = ctx.allocate(end)?;
        let pinned = ctx.pinned_buf(end.max(cold_bytes))?;
        let scratch = ctx.allocate(2 * h * WORD)?;
        let Some(embedding) = weights.resident_embedding(ctx)? else {
            return declined("the embedding table has no device gather or no room");
        };
        let stream = ctx.stream()?;
        if ctx.memory_info()?.free
            < cuda
                .token_growth_reserve(&f.spec, f.context)
                .unwrap_or(usize::MAX)
        {
            return declined("free device memory is under the expert growth reserve");
        }
        Ok(Some(Self {
            warm: vec![false; slots.len()],
            slots,
            chain,
            stream,
            depth,
            prelude,
            embedding,
            constants,
            params,
            scratch,
            journal,
            pinned,
            logits,
            health,
        }))
    }

    pub(in super::super) fn ready(
        &self,
        cuda: &GgufCuda,
        states: &[VisitState],
        pos: usize,
    ) -> bool {
        self.slots.iter().all(|s| {
            let v = s.w.plan.visit;
            states[v.state_slot].position() == pos
                && Self::eligible(cuda, v, pos, s.w.plan.shape.banked)
        })
    }

    pub(in super::super) fn run(
        &mut self,
        f: &CpuForward,
        cuda: &GgufCuda,
        states: &[VisitState],
        id: u32,
    ) -> Result<Vec<f32>, EngError> {
        let ctx = cuda.context();
        let stream = &self.stream;
        let h = f.hidden();
        let row = h * WORD;
        let pos = i32::try_from(f.position).map_err(|_| bad("resident position overflows"))?;
        let mut boot = Vec::new();
        let mut input = Vec::new();
        input.extend(id.to_ne_bytes());
        input.extend(pos.to_ne_bytes());
        for (slot, warm) in self.slots.iter().zip(&mut self.warm) {
            let w = &slot.w;
            let s = &states[w.plan.visit.state_slot];
            if std::mem::take(warm) && s.is_resident() {
                continue;
            }
            if s.is_resident() {
                return Err(bad("resident token rows have no device lineage"));
            }
            let at = input.len();
            for (region, history) in w
                .layout
                .histories
                .into_iter()
                .zip(s.resident_conv_histories())
            {
                let start = input.len();
                let past = w.plan.shape.widths.taps - 1;
                if history.len() > past
                    || history
                        .iter()
                        .any(|v| v.len() * WORD * past != region.bytes)
                {
                    return Err(bad("resident token history geometry differs"));
                }
                for v in history {
                    append(&mut input, v);
                }
                input.resize(start + region.bytes, 0);
            }
            boot.push((w, at, input.len() - at));
        }
        self.pinned.write(0, &input)?;
        let params = &self.params;
        let head = |i| self.health.device(&self.journal)?.slice(i * WORD, WORD);
        let queued = (|| {
            ctx.copy_from_pinned(stream, &mut self.pinned, 0, params.range(0, 8)?)?;
            for (w, at, bytes) in boot {
                ctx.copy_from_pinned(
                    stream,
                    &mut self.pinned,
                    at,
                    w.arena.range(w.layout.histories[0].start, bytes)?,
                )?;
            }
            // One launch clears the token's health words and every visit's status words before
            // any chain, and one gathers the status words into the journal after the last.
            let statuses = self
                .slots
                .iter()
                .map(|s| {
                    s.w.arena
                        .range(s.w.layout.attention_status.start, s.status_bytes)
                })
                .collect::<Result<Vec<_>, EngError>>()?;
            let cleared: Vec<_> = std::iter::once(self.health.device(&self.journal))
                .chain(std::iter::once(self.chain.health.range(0, WORD)))
                .chain(statuses.iter().copied().map(Ok))
                .collect::<Result<_, EngError>>()?;
            for rows in cleared.chunks(MAX_WORD_ROWS) {
                ctx.zero_word_rows(stream, rows)?;
            }
            let raw = self.scratch.range(0, row)?;
            self.embedding
                .gather(ctx, stream, [params.range(0, WORD)?, raw, head(0)?], 1)?;
            ctx.enqueue_row_weighted_norm(
                stream,
                [
                    raw,
                    self.constants.range(0, row)?,
                    self.depth[0].range(0, row)?,
                    head(1)?,
                ],
                h,
                f.math.rms_norm_eps,
            )?;
            let step = ChainStep {
                f,
                cuda,
                slots: &self.slots,
                chain: &self.chain,
                depth: &self.depth,
                prelude: &self.prelude,
                params,
                stream,
            };
            walk::visits(&Walk(&step), 1, f.position, |_, _| Ok(()))?;
            let receipts = self
                .slots
                .iter()
                .zip(&statuses)
                .map(|(s, &status)| Ok((status, s.receipt.device(&self.journal)?)))
                .chain(std::iter::once(Ok((
                    self.chain.health.range(0, WORD)?,
                    self.chain.receipt.device(&self.journal)?,
                ))))
                .collect::<Result<Vec<_>, EngError>>()?;
            for pairs in receipts.chunks(MAX_WORD_ROWS) {
                ctx.copy_word_rows(stream, pairs)?;
            }
            let last = self.slots.last().ok_or_else(|| bad("empty token chain"))?;
            let plane = *self
                .chain
                .planes
                .last()
                .ok_or_else(|| bad("empty token chain"))?;
            let shape = last.w.plan.shape;
            let blocks = shape.blocks + usize::from(shape.seal);
            let normed = self.scratch.range(row, row)?;
            ctx.depth_attention_mix(
                stream,
                MixCall {
                    state: self.depth[plane].range(0, row * (blocks + 1))?,
                    query: self.constants.range(row, row)?,
                    out: raw,
                    health: head(2)?,
                },
                MixShape {
                    first_block: 0,
                    blocks,
                    width: h,
                    eps: MIX_EPS,
                },
            )?;
            ctx.enqueue_row_weighted_norm(
                stream,
                [raw, self.constants.range(row * 2, row)?, normed, head(3)?],
                h,
                f.math.rms_norm_eps,
            )?;
            cuda.enqueue_resident_head(
                stream,
                normed,
                self.logits.device(&self.journal)?,
                f.math.final_logit_softcapping,
            )?;
            ctx.copy_to_pinned(
                stream,
                self.journal.range(0, self.journal.bytes())?,
                &mut self.pinned,
                0,
            )
        })();
        self.pinned.synchronize_after_enqueue(stream, queued)?;
        let check = |span: Region| -> Result<(), EngError> {
            if span.read(&self.pinned)?.iter().any(|x| x.to_bits() != 0) {
                return Err(bad("resident token reported invalid device state"));
            }
            Ok(())
        };
        check(self.health)?;
        check(self.chain.receipt)?;
        for s in &self.slots {
            check(Region {
                start: s.receipt.start,
                bytes: s.status_bytes,
            })?;
        }
        let logits = self.logits.read(&self.pinned)?;
        if logits.iter().any(|v| !v.is_finite()) {
            return Err(bad("resident head produced nonfinite logits"));
        }
        Ok(logits)
    }

    /// Makes the device the owner of the committed rows. Call only after `run` returned its
    /// logits, so every status word and the fence passed before a host cursor moves.
    pub(in super::super) fn commit(
        &mut self,
        f: &CpuForward,
        cuda: &GgufCuda,
        states: &mut [VisitState],
    ) {
        for (index, s) in self.slots.iter().enumerate() {
            states[s.w.plan.visit.state_slot].advance_resident();
            cuda.publish_local_decode(s.w.plan.visit, f.position);
            self.warm[index] = true;
        }
    }

    /// Reads the K/V rows and the four convolution histories of every resident visit back
    /// into the host state. Explicit events only: a snapshot, the host-orchestrated path, or a
    /// prefill after a chain token. `restore` and `reset` replace the sequence instead.
    pub(in super::super) fn materialize(
        &mut self,
        cuda: &GgufCuda,
        cursor: Cursor<'_>,
    ) -> Result<(), EngError> {
        let Cursor {
            states,
            position,
            past,
            capacities,
        } = cursor;
        let arenas: Vec<usize> = self
            .slots
            .iter()
            .map(|s| s.w.plan.visit.state_slot)
            .collect();
        if states
            .iter()
            .enumerate()
            .any(|(slot, state)| state.is_resident() && !arenas.contains(&slot))
        {
            return Err(bad("resident visit rows have no token arena"));
        }
        let Some((_, attention)) = cuda.resident_attention() else {
            return Err(bad("resident visit rows have no device attention"));
        };
        for s in &self.slots {
            let slot = s.w.plan.visit.state_slot;
            if !states[slot].is_resident() {
                continue;
            }
            let (keys, values) = attention.download_ring(&self.stream, slot, position)?;
            let rows = position.min(past);
            let mut histories = Vec::with_capacity(s.w.layout.histories.len());
            for region in s.w.layout.histories {
                let width = region.bytes / WORD / past;
                let mut bytes = vec![0u8; region.bytes];
                s.w.arena.read_at(&self.stream, region.start, &mut bytes)?;
                let plane: Vec<Vec<f32>> = bytes
                    .chunks_exact(width * WORD)
                    .take(rows)
                    .map(|row| {
                        row.as_chunks::<WORD>()
                            .0
                            .iter()
                            .map(|w| f32::from_ne_bytes(*w))
                            .collect()
                    })
                    .collect();
                histories.push(plane);
            }
            let [k, v, attn, mlp] = <[Vec<Vec<f32>>; 4]>::try_from(histories).unwrap();
            states[slot].adopt_resident(
                (keys, values),
                [k, v, attn, mlp],
                (past, capacities[slot]),
            );
        }
        self.warm.fill(false);
        Ok(())
    }
}

#[cfg(test)]
mod materialize_tests;
#[cfg(test)]
mod refusal_tests;
#[cfg(test)]
mod shadow_tests;
#[cfg(test)]
mod tap_tests;
#[cfg(test)]
mod tests;
