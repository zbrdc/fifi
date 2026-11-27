//! One opt-in cross-pass local decode visit. The host owns request state at the
//! boundary; within the visit, one stream orders depth mixing, attention, expert
//! routing and the residual updates without a host wait between them.

use crate::family::inkling2::switches::Switch;
use std::sync::Once;

use super::{
    CpuForward, LayerVisit, MIX_EPS, MessageBank, Role, Stream, VisitState, Weights, Work, bad,
    control_names,
};
use crate::{
    EngError,
    family::inkling2::{
        Attention,
        cuda_projection::gguf::{
            ConvRows, GgufCuda, LocalDecodeWidths, PrepWeights, ResidentLocalDecode,
            resident_mlp::ResidentMlpRequest,
        },
    },
    kernels::cuda::{
        CudaContext, DeviceBuffer, DeviceRange, DeviceStream, PinnedBuf,
        resident::{
            dense_gemv::DenseDtype,
            inkling_conv::ConvCall,
            inkling_local_prep::PrepMath,
            inkling_stream::{
                AddCall, MAX_BLOCKS, MAX_MIX_JOBS, MixCall, MixJob, MixNorm, MixShape, SealCall,
                StreamShape,
            },
        },
    },
};

#[cfg(test)]
mod pair;
mod token;
#[cfg(test)]
pub(super) use pair::Pair;
pub(super) use token::TokenWorkspace;

const WORD: usize = size_of::<f32>();

#[derive(Clone, Copy)]
struct Region {
    start: usize,
    bytes: usize,
}

impl Region {
    fn row(self, width: usize) -> Self {
        Self {
            bytes: width * WORD,
            ..self
        }
    }

    fn end(self) -> usize {
        self.start + self.bytes
    }

    fn device(self, arena: &DeviceBuffer) -> Result<DeviceRange<'_>, EngError> {
        arena.range(self.start, self.bytes)
    }

    fn read(self, transfer: &PinnedBuf) -> Result<Vec<f32>, EngError> {
        let mut row = vec![0.0; self.bytes / WORD];
        transfer.read_f32(self.start, &mut row)?;
        Ok(row)
    }
}

fn reserve(end: &mut usize, words: usize) -> Region {
    let span = Region {
        start: *end,
        bytes: words * WORD,
    };
    *end = span.end();
    span
}

struct Controls {
    names: [String; 3],
    dtypes: [DenseDtype; 2],
    rank: usize,
    caps: [f32; 2],
    scale: f32,
}

#[derive(Clone, Copy)]
struct Shape {
    widths: LocalDecodeWidths,
    global: bool,
    banked: bool,
    blocks: usize,
    bank_blocks: usize,
    bank_first: usize,
    seal: bool,
}

struct Plan {
    visit: LayerVisit,
    shape: Shape,
    math: PrepMath,
    controls: Controls,
}

/// Contiguous upload (position, streams, histories, zeroed status) followed by the
/// compact readback log and scratch. All rows passed to a kernel are exact ranges.
struct Layout {
    pos: Region,
    depth: Region,
    bank: Region,
    histories: [Region; 4],
    attention_status: Region,
    glue_health: Region,
    mlp_health: Region,
    mlp_status: Region,
    log: Region,
    mlp_raw: Region,
    attention_input: Region,
    global_scratch: Option<([Region; 2], [Region; 2])>,
    messages: [Region; 4],
    attention_out: Region,
    mlp_input: Region,
    mlp_conv: Region,
    /// The decode glue's rows: the normed attention input a mix job writes and the normed MLP
    /// input another writes (empty when the glue is off; they never exist then).
    attention_normed: Region,
    mlp_normed: Region,
    bytes: usize,
}

impl Layout {
    fn new(shape: Shape, status_words: usize) -> Self {
        let h = shape.widths.hidden;
        let kv = shape.widths.kv_channels;
        let past = shape.widths.taps - 1;
        let mut end = 0;
        let mut take = |words| reserve(&mut end, words);
        Self {
            pos: take(1),
            depth: take((shape.blocks + 1 + usize::from(shape.seal)) * h),
            bank: take((shape.bank_blocks + 1) * h),
            histories: [kv, kv, h, h].map(|width| take(width * past)),
            attention_status: take(status_words),
            glue_health: take(1),
            mlp_health: take(1),
            mlp_status: take(1),
            log: take(4 * kv),
            mlp_raw: take(h),
            attention_input: take(h),
            global_scratch: shape
                .global
                .then(|| ([h; 2].map(&mut take), [kv / 128 * 4; 2].map(&mut take))),
            messages: {
                // Plain projection and expert kernels require their normalized
                // FP32 inputs on 16-byte boundaries (Mini rows have width 2304).
                let span = take(4 * h + 3);
                let aligned = span.start.next_multiple_of(16);
                std::array::from_fn(|i| Region {
                    start: aligned + i * h * WORD,
                    bytes: h * WORD,
                })
            },
            attention_out: take(h),
            mlp_input: take(h),
            mlp_conv: take(h),
            attention_normed: Self::take_normed(&mut take, h, Switch::DecodeGlue.on()),
            mlp_normed: Self::take_normed(&mut take, h, Switch::DecodeGlue.on()),
            bytes: end,
        }
    }

    /// A row a mix job norms into, 16-byte aligned as the projection inputs are; or no row.
    fn take_normed(take: &mut impl FnMut(usize) -> Region, h: usize, glue: bool) -> Region {
        if !glue {
            return take(0);
        }
        let span = take(h + 3);
        Region {
            start: span.start.next_multiple_of(16),
            bytes: h * WORD,
        }
    }

    fn receipts(&self) -> Region {
        Region {
            start: self.attention_status.start,
            bytes: self.mlp_raw.end() - self.attention_status.start,
        }
    }
}

struct Constants {
    device: DeviceBuffer,
    attention_query: Option<Region>,
    global_roles: Option<([Region; 2], [Region; 2])>,
    bank_queries: Option<[Region; 4]>,
    mlp_query: Region,
    attention: [Region; 6],
    mlp: [Region; 3],
}

impl Constants {
    fn load(
        forward: &CpuForward,
        cuda: &GgufCuda,
        plan: &Plan,
        layer: &str,
    ) -> Result<Self, EngError> {
        let mut bytes = Vec::new();
        let mut pack = |row: &[f32]| {
            let span = Region {
                start: bytes.len(),
                bytes: std::mem::size_of_val(row),
            };
            append(&mut bytes, row);
            span
        };
        let h = plan.shape.widths.hidden;
        let physical = plan.visit.weight_layer;
        let attention_query = physical
            .checked_sub(1)
            .map(|query| {
                forward
                    .weights
                    .row(&format!("model.llm.attn_res.attn_queries.{query}"), 0, h)
                    .map(|row| pack(&row))
            })
            .transpose()?;
        let global_roles = if plan.shape.global {
            let query = physical
                .checked_sub(1)
                .ok_or_else(|| bad("banked global depth needs an attention query"))?;
            let name = format!("model.llm.attn_res.attn_queries.{query}");
            let [k, v] = [1, 2].map(|i| forward.weights.row(&name, i, h).map(|row| pack(&row)));
            let [kg, vg] = ["k", "v"].map(|r| {
                forward
                    .weights
                    .vec(&format!("{layer}.{r}_input_norm.weight"), h)
                    .map(|row| pack(&row))
            });
            Some(([k?, v?], [kg?, vg?]))
        } else {
            None
        };
        let bank_queries = if plan.shape.banked {
            let [q, k, v, m] = [Role::Query, Role::Key, Role::Value, Role::Mlp]
                .map(|role| forward.control_query(physical, role).map(|row| pack(&row)));
            Some([q?, k?, v?, m?])
        } else {
            None
        };
        let mlp_query = pack(
            &forward
                .weights
                .vec(&format!("model.llm.attn_res.mlp_queries.{physical}"), h)?,
        );
        let read = |(name, words): (String, usize)| forward.weights.vec(&name, words);
        let [gamma, q_scale, rel_scale, k_conv, v_conv, attn_conv] =
            cuda.resident_local_decode_weight_rows(plan.visit, layer)?;
        let attention = [gamma, q_scale, rel_scale, k_conv, v_conv, attn_conv]
            .map(|request| read(request).map(|row| pack(&row)));
        let mlp = [
            (format!("{layer}.mlp_norm.weight"), h),
            (
                format!("{layer}.mlp.gate.bias"),
                forward.spec.n_routed_experts as usize,
            ),
            (
                format!("{layer}.mlp_sconv.weight"),
                h * plan.shape.widths.taps,
            ),
        ]
        .map(|request| read(request).map(|row| pack(&row)));
        let [a, b, c, d, e, f] = attention;
        let [gamma, bias, conv] = mlp;
        Ok(Self {
            bank_queries,
            attention_query,
            global_roles,
            mlp_query,
            attention: [a?, b?, c?, d?, e?, f?],
            mlp: [gamma?, bias?, conv?],
            device: cuda.context().upload(&bytes)?,
        })
    }

    fn row(&self, span: Region) -> Result<DeviceRange<'_>, EngError> {
        span.device(&self.device)
    }
}

pub(super) struct Workspace {
    plan: Plan,
    layout: Layout,
    weights: Constants,
    arena: DeviceBuffer,
    stream: DeviceStream,
    transfer: PinnedBuf,
    encoded: Vec<u8>,
}

impl CpuForward {
    pub(super) fn try_resident_visit(
        &self,
        visit: LayerVisit,
        state: &mut VisitState,
        depth: &mut Stream,
        bank: &MessageBank,
    ) -> Result<bool, EngError> {
        if visit.attention != Attention::Local
            || !Switch::ResidentVisit.on()
            || self.draft_window_enabled()
        {
            return Ok(false);
        }
        let Weights::Gguf(weights) = &self.weights else {
            return Ok(false);
        };
        let Some(cuda) = weights.cuda() else {
            return Ok(false);
        };
        let mut workspace = self.resident_visit.borrow_mut();
        if workspace
            .as_ref()
            .is_some_and(|w| w.plan.visit.state_slot != visit.state_slot)
            || !cuda.resident_local_decode_supported(visit, state.position())
        {
            return Ok(false);
        }
        if workspace.is_none() {
            match Workspace::open(self, cuda, visit, (depth.block_count(), bank)) {
                Ok(Some(prepared)) => *workspace = Some(prepared),
                Ok(None) | Err(EngError::MemoryPlan(_)) => return Ok(false),
                Err(error) => return Err(error),
            }
        }
        let workspace = workspace
            .as_mut()
            .expect("the visit workspace was admitted");
        let position = state.position();
        let _span = self.phase_span(Work::ResidentVisit);
        let result = workspace.run(cuda, state, (depth, bank));
        let output = result.inspect_err(|_| self.invalidate_device_attention())?;
        cuda.publish_local_decode(visit, position);
        state.publish_resident_visit(
            output.history,
            (
                self.math.sconv_kernel_size - 1,
                self.kv_capacity(Attention::Local),
            ),
        );
        depth.publish_resident_current(output.current, workspace.plan.shape.seal);
        static USED: Once = Once::new();
        USED.call_once(|| eprintln!("Inkling CUDA used resident decode visit: slot {}, weight layer {}, one completion fence", visit.state_slot, visit.weight_layer));
        Ok(true)
    }
}

impl Workspace {
    fn open(
        forward: &CpuForward,
        cuda: &GgufCuda,
        visit: LayerVisit,
        (blocks, bank): (usize, &MessageBank),
    ) -> Result<Option<Self>, EngError> {
        Self::open_geometry(
            forward,
            cuda,
            visit,
            blocks,
            Some((bank.stream.block_count(), bank.first_block)),
        )
    }

    fn open_geometry(
        forward: &CpuForward,
        cuda: &GgufCuda,
        visit: LayerVisit,
        blocks: usize,
        bank: Option<(usize, usize)>,
    ) -> Result<Option<Self>, EngError> {
        let Some(widths) = cuda.resident_local_decode_widths(visit) else {
            return Ok(None);
        };
        let shape = Shape {
            widths,
            global: visit.attention == Attention::Global,
            blocks,
            banked: bank.is_some(),
            bank_blocks: bank.map_or(0, |b| b.0),
            bank_first: bank.map_or(0, |b| b.1),
            seal: visit
                .weight_layer
                .is_multiple_of(forward.spec.n_layers_per_attn_res_block),
        };
        if widths.taps != forward.math.sconv_kernel_size
            || shape.blocks + usize::from(shape.seal) > MAX_BLOCKS
            || shape.bank_blocks > MAX_BLOCKS
            || shape.bank_first > shape.bank_blocks
        {
            return Ok(None);
        }
        let Weights::Gguf(weights) = &forward.weights else {
            return Ok(None);
        };
        let layer = format!("model.llm.layers.{}", visit.weight_layer);
        let Some(dtypes) = weights.resident_expert_dtypes(&format!("{layer}.mlp"))? else {
            return Ok(None);
        };
        let names = control_names(&layer, Role::Mlp);
        let rank = forward.spec.core.gate_rank as usize;
        if bank.is_some()
            && !cuda.resident_mlp_supported(
                visit.weight_layer as u32,
                names.each_ref().map(String::as_str),
                [rank, widths.hidden],
            )
        {
            return Ok(None);
        }
        if bank.is_none()
            && !cuda.resident_plain_mlp_supported(visit.weight_layer as u32, widths.hidden)
        {
            return Ok(None);
        }
        let Some(status_words) = cuda.resident_local_decode_status_words(visit, bank.is_some())
        else {
            return Ok(None);
        };
        let plan = Plan {
            visit,
            shape,
            math: PrepMath {
                qk_limit: forward.math.qk_l2_clip_threshold,
                rel_limit: forward.math.rel_l2_clip_threshold,
                eps: forward.math.rms_norm_eps,
            },
            controls: Controls {
                names,
                dtypes,
                rank,
                caps: [forward.math.mlp_gate_softcap, forward.math.mlp_up_softcap],
                scale: forward.math.route_scale,
            },
        };
        let layout = Layout::new(shape, status_words);
        let context = cuda.context();
        let transfer = context.pinned_buf(layout.mlp_raw.end())?;
        let arena = context.allocate(layout.bytes)?;
        let stream = context.stream()?;
        let weights = Constants::load(forward, cuda, &plan, &layer)?;
        if bank.is_some() {
            cuda.prepare_local_decode(visit)?;
        } else {
            cuda.prepare_token_decode(visit, false)?;
        }
        Ok(Some(Self {
            plan,
            layout,
            weights,
            arena,
            stream,
            transfer,
            encoded: Vec::new(),
        }))
    }

    fn run(
        &mut self,
        cuda: &GgufCuda,
        state: &VisitState,
        (depth, bank): (&Stream, &MessageBank),
    ) -> Result<Output, EngError> {
        let position = state.position();
        self.encode(state, Some(depth), bank, true)?;
        self.transfer.write(0, &self.encoded)?;
        let context = cuda.context();
        let layout = &self.layout;
        let queued = (|| {
            context.copy_from_pinned(
                &self.stream,
                &mut self.transfer,
                0,
                self.arena.range(0, self.encoded.len())?,
            )?;
            let visit = Execution {
                plan: &self.plan,
                layout,
                weights: &self.weights,
                arena: &self.arena,
                shared: None,
                stream: &self.stream,
                context,
            };
            visit.enqueue(cuda, position)?;
            let hidden = self.plan.shape.widths.hidden;
            for span in [
                layout.receipts(),
                layout.depth.row(hidden),
                layout.histories[2].row(hidden),
            ] {
                context.copy_to_pinned(
                    &self.stream,
                    span.device(&self.arena)?,
                    &mut self.transfer,
                    span.start,
                )?;
            }
            Ok(())
        })();
        self.transfer
            .synchronize_after_enqueue(&self.stream, queued)?;
        self.readback(&self.transfer, 0)
    }

    fn encode(
        &mut self,
        state: &VisitState,
        depth: Option<&Stream>,
        bank: &MessageBank,
        bootstrap_history: bool,
    ) -> Result<(), EngError> {
        let shape = self.plan.shape;
        if (
            depth.map_or(shape.blocks, Stream::block_count),
            bank.stream.block_count(),
            bank.first_block,
        ) != (shape.blocks, shape.bank_blocks, shape.bank_first)
        {
            return Err(bad("resident visit stream geometry changed"));
        }
        let bytes = &mut self.encoded;
        bytes.clear();
        let position = i32::try_from(state.position())
            .map_err(|_| bad("resident visit position overflows"))?;
        bytes.extend_from_slice(&position.to_ne_bytes());
        if let Some(depth) = depth {
            depth.append_device_state(bytes);
        }
        bytes.resize(self.layout.depth.end(), 0);
        if depth.is_some() {
            bank.stream.append_device_state(bytes);
        } else {
            bytes.resize(self.layout.bank.end(), 0);
        }
        if !bootstrap_history {
            bytes.resize(self.layout.log.start, 0);
            return Ok(());
        }
        let past = shape.widths.taps - 1;
        for (span, history) in self
            .layout
            .histories
            .into_iter()
            .zip(state.resident_conv_histories())
        {
            if history.len() > past
                || history
                    .iter()
                    .any(|row| std::mem::size_of_val(row.as_slice()) * past != span.bytes)
            {
                return Err(bad("resident visit convolution history geometry changed"));
            }
            for row in history {
                append(bytes, row);
            }
            bytes.resize(span.end(), 0);
        }
        bytes.resize(self.layout.log.start, 0);
        Ok(())
    }

    fn readback(&self, pinned: &PinnedBuf, offset: usize) -> Result<Output, EngError> {
        let read = |span: Region| {
            Region {
                start: offset + span.start,
                ..span
            }
            .read(pinned)
        };
        let layout = &self.layout;
        let status = read(Region {
            start: layout.attention_status.start,
            bytes: layout.log.start - layout.attention_status.start,
        })?;
        if let Some(i) = status.iter().position(|word| word.to_bits() != 0) {
            return Err(bad(format!(
                "resident visit {} reported nonzero status word {i}: {:#x}",
                self.plan.visit.state_slot,
                status[i].to_bits()
            )));
        }
        let kv = self.plan.shape.widths.kv_channels;
        let h = self.plan.shape.widths.hidden;
        let log = |i: usize| {
            read(Region {
                start: layout.log.start + i * kv * WORD,
                bytes: kv * WORD,
            })
        };
        Ok(Output {
            current: read(layout.depth.row(h))?,
            history: [
                log(0)?,
                log(1)?,
                log(2)?,
                log(3)?,
                read(layout.histories[2].row(h))?,
                read(layout.mlp_raw)?,
            ],
        })
    }
}

struct Output {
    current: Vec<f32>,
    // Raw K, raw V, processed K, processed V, raw attention output, raw MLP output.
    history: [Vec<f32>; 6],
}

#[derive(Clone, Copy)]
struct Shared<'a> {
    depth: DeviceRange<'a>,
    bank: DeviceRange<'a>,
    pos: DeviceRange<'a>,
}

struct Execution<'a> {
    shared: Option<Shared<'a>>,
    plan: &'a Plan,
    layout: &'a Layout,
    weights: &'a Constants,
    arena: &'a DeviceBuffer,
    stream: &'a DeviceStream,
    context: &'a CudaContext,
}

impl Execution<'_> {
    fn row(&self, span: Region) -> Result<DeviceRange<'_>, EngError> {
        if let Some(s) = self.shared {
            for (slot, row) in [
                (self.layout.depth, s.depth),
                (self.layout.bank, s.bank),
                (self.layout.pos, s.pos),
            ] {
                if span.start == slot.start {
                    return row.slice(0, span.bytes);
                }
            }
        }
        span.device(self.arena)
    }

    fn enqueue(&self, cuda: &GgufCuda, position: usize) -> Result<(), EngError> {
        let l = self.layout;
        let shape = self.plan.shape;
        let width = shape.widths.hidden;
        // The bank queries and the attention query read different stream states (the bank and
        // the depth), so one launch runs them all, a block per mix. Under the decode glue the
        // attention query's job also norms its output (the attention input of a plain visit):
        // the glue folds norms into the mixes of plain visits only, since a banked visit's
        // attention feeds the raw attention input to its controller chain.
        let glue = Switch::DecodeGlue.on() && !shape.banked;
        let job = |(state, first_block): (Region, usize),
                   blocks: usize,
                   query,
                   out,
                   normed: Option<Region>| {
            Ok::<_, EngError>(MixJob {
                state: self.row(state.row((blocks + 1) * width))?,
                query: self.weights.row(query)?,
                out: self.row(out)?,
                first_block,
                blocks,
                addend: None,
                norm: normed
                    .map(|normed| self.norm(self.weights.attention[0], normed))
                    .transpose()?,
            })
        };
        let (depth, bank) = ((l.depth, 0), (l.bank, shape.bank_first));
        let mut jobs = Vec::with_capacity(MAX_MIX_JOBS);
        for (query, out) in self
            .weights
            .bank_queries
            .into_iter()
            .flatten()
            .zip(l.messages)
        {
            jobs.push(job(bank, shape.bank_blocks, query, out, None)?);
        }
        let (input, normed) = match self.weights.attention_query {
            Some(query) if glue => {
                let normed = l.attention_normed;
                jobs.push(job(
                    depth,
                    shape.blocks,
                    query,
                    l.attention_input,
                    Some(normed),
                )?);
                (normed, true)
            }
            Some(query) => {
                jobs.push(job(depth, shape.blocks, query, l.attention_input, None)?);
                (l.attention_input, false)
            }
            None => (l.depth.row(width), false),
        };
        if !jobs.is_empty() {
            self.context.inkling_stream_mix_jobs(
                self.stream,
                &jobs,
                &self.row(l.glue_health)?,
                (width, MIX_EPS),
            )?;
        }
        self.attention(cuda, position, (input, normed))?;
        // After the attention: the seal, or (under the glue) the attention output as the addend
        // of the depth mix that follows, or the add launch.
        let (blocks, addend) = self.advance_depth(glue)?;
        if !shape.banked && glue {
            // The depth mix adds the attention output itself and writes the weighted norm of its
            // output, so the plain MLP reads that row without a norm launch.
            let job = MixJob {
                state: self.row(l.depth.row((blocks + 1) * width))?,
                query: self.weights.row(self.weights.mlp_query)?,
                out: self.row(l.mlp_input)?,
                first_block: 0,
                blocks,
                addend: addend.map(|addend| self.row(addend)).transpose()?,
                norm: Some(self.norm(self.weights.mlp[0], l.mlp_normed)?),
            };
            self.context.inkling_stream_mix_jobs(
                self.stream,
                std::slice::from_ref(&job),
                &self.row(l.glue_health)?,
                (width, MIX_EPS),
            )?;
            return self.mlp(cuda, Some(l.mlp_normed));
        }
        self.mix(depth, blocks, self.weights.mlp_query, l.mlp_input, addend)?;
        self.mlp(cuda, None)
    }

    /// The norm epilogue of a mix job: the weighted norm of the job's output into `normed`.
    fn norm(&self, gamma: Region, normed: Region) -> Result<MixNorm<'_>, EngError> {
        Ok(MixNorm {
            gamma: self.weights.row(gamma)?,
            normed: self.row(normed)?,
            eps: self.plan.math.eps,
        })
    }

    fn mix(
        &self,
        (state, first_block): (Region, usize),
        blocks: usize,
        query: Region,
        out: Region,
        addend: Option<Region>,
    ) -> Result<(), EngError> {
        let width = self.plan.shape.widths.hidden;
        let state = self.row(state.row((blocks + 1) * width))?;
        let query = self.weights.row(query)?;
        let out = self.row(out)?;
        let health = self.row(self.layout.glue_health)?;
        if let Some(addend) = addend {
            // The add of the attention output is the job's addend: one launch for both.
            let job = MixJob {
                state,
                query,
                out,
                first_block,
                blocks,
                addend: Some(self.row(addend)?),
                norm: None,
            };
            return self.context.inkling_stream_mix_jobs(
                self.stream,
                std::slice::from_ref(&job),
                &health,
                (width, MIX_EPS),
            );
        }
        self.context.inkling_stream_mix(
            self.stream,
            MixCall {
                state,
                query,
                out,
                health,
            },
            MixShape {
                first_block,
                blocks,
                width,
                eps: MIX_EPS,
            },
        )
    }

    fn attention(
        &self,
        cuda: &GgufCuda,
        position: usize,
        (input, normed): (Region, bool),
    ) -> Result<(), EngError> {
        let l = self.layout;
        let [gamma, q_scale, rel_scale, k_conv, v_conv, conv] = self.weights.attention;
        let [q, k, v, _] = l.messages;
        // The plain attention's normed rows: the query's is the norm the attention-query mix
        // already wrote under the decode glue (the query projection reads it), else the message
        // row the plain attention norms `input` into; the key's and value's are message rows.
        let normed_rows = [
            if normed {
                self.row(l.attention_normed)?
            } else {
                self.row(q)?
            },
            self.row(k)?,
            self.row(v)?,
        ];
        let mut request = ResidentLocalDecode {
            visit: self.plan.visit,
            position,
            math: self.plan.math,
            input: self.row(input)?,
            messages: self
                .plan
                .shape
                .banked
                .then_some([self.row(q)?, self.row(k)?, self.row(v)?]),
            gammas: [self.weights.row(gamma)?; 3],
            weights: PrepWeights {
                query_scale: self.weights.row(q_scale)?,
                relative_scale: self.weights.row(rel_scale)?,
                key_kernel: self.weights.row(k_conv)?,
                value_kernel: self.weights.row(v_conv)?,
            },
            raw_history: [self.row(l.histories[0])?, self.row(l.histories[1])?],
            attn_conv: ConvRows {
                kernel: self.weights.row(conv)?,
                history: self.row(l.histories[2])?,
            },
            pos: self.row(l.pos)?,
            out: self.row(l.attention_out)?,
            log: self.row(l.log)?,
            status: self.row(l.attention_status)?,
        };
        if let Some((queries, gammas)) = self.weights.global_roles {
            let (inputs, coeff) = l
                .global_scratch
                .ok_or_else(|| bad("global decode lacks scratch"))?;
            let blocks = self.plan.shape.blocks;
            request.gammas = [
                self.weights.row(gamma)?,
                self.weights.row(gammas[0])?,
                self.weights.row(gammas[1])?,
            ];
            if !self.plan.shape.banked {
                let pre_normed = self.role_mixes((queries, gammas), inputs, blocks, normed)?;
                return cuda.enqueue_plain_decode(
                    self.stream,
                    request,
                    normed_rows,
                    pre_normed,
                    Some((
                        [self.row(inputs[0])?, self.row(inputs[1])?],
                        [self.row(coeff[0])?, self.row(coeff[1])?],
                    )),
                );
            }
            for (query, out) in queries.into_iter().zip(inputs) {
                self.mix((l.depth, 0), blocks, query, out, None)?;
            }
            return cuda.enqueue_banked_global(
                self.stream,
                request,
                [self.row(inputs[0])?, self.row(inputs[1])?],
                [self.row(coeff[0])?, self.row(coeff[1])?],
            );
        }
        if !self.plan.shape.banked {
            return cuda.enqueue_plain_decode(
                self.stream,
                request,
                normed_rows,
                [normed, false, false],
                None,
            );
        }
        cuda.enqueue_local_decode(self.stream, request)
    }

    /// The two role mixes of a plain global visit. Under the decode glue they run in one launch
    /// and each job also writes its normed role row; otherwise each mix has its own launch and
    /// the plain attention norms the role rows. Returns which of the plain attention's three
    /// normed rows (query, key, value) the mixes already wrote.
    fn role_mixes(
        &self,
        (queries, gammas): ([Region; 2], [Region; 2]),
        inputs: [Region; 2],
        blocks: usize,
        normed: bool,
    ) -> Result<[bool; 3], EngError> {
        let l = self.layout;
        let width = self.plan.shape.widths.hidden;
        if !Switch::DecodeGlue.on() {
            for (query, out) in queries.into_iter().zip(inputs) {
                self.mix((l.depth, 0), blocks, query, out, None)?;
            }
            return Ok([normed; 3]);
        }
        let state = self.row(l.depth.row((blocks + 1) * width))?;
        let jobs = [0, 1].map(|i| {
            Ok::<_, EngError>(MixJob {
                state,
                query: self.weights.row(queries[i])?,
                out: self.row(inputs[i])?,
                first_block: 0,
                blocks,
                addend: None,
                norm: Some(self.norm(gammas[i], l.messages[i + 1])?),
            })
        });
        let [first, second] = jobs;
        self.context.inkling_stream_mix_jobs(
            self.stream,
            &[first?, second?],
            &self.row(l.glue_health)?,
            (width, MIX_EPS),
        )?;
        Ok([normed, true, true])
    }

    /// Seal, or leave the depth stream alone after the attention: without a seal the attention
    /// output joins the current row, as the addend the following depth mix takes under the
    /// decode glue (`fold`), or through the add launch. Returns the blocks the MLP query mixes
    /// over and the addend.
    fn advance_depth(&self, fold: bool) -> Result<(usize, Option<Region>), EngError> {
        let shape = self.plan.shape;
        let l = self.layout;
        if !shape.seal {
            if fold {
                return Ok((shape.blocks, Some(l.attention_out)));
            }
            self.add(l.attention_out)?;
            return Ok((shape.blocks, None));
        }
        self.context.inkling_stream_seal(
            self.stream,
            SealCall {
                state: self.row(l.depth)?,
                next: self.row(l.attention_out)?,
                health: self.row(l.glue_health)?,
            },
            StreamShape {
                blocks: shape.blocks,
                width: shape.widths.hidden,
                eps: MIX_EPS,
            },
        )?;
        Ok((shape.blocks + 1, None))
    }

    fn add(&self, addend: Region) -> Result<(), EngError> {
        let width = self.plan.shape.widths.hidden;
        self.context.inkling_stream_add(
            self.stream,
            AddCall {
                state: self.row(self.layout.depth.row(width))?,
                row: self.row(addend)?,
                health: self.row(self.layout.glue_health)?,
            },
            width,
        )
    }

    /// The MLP half; `pre_normed` is the row the depth mix already normed under the decode
    /// glue, which the plain MLP reads instead of norming its input.
    fn mlp(&self, cuda: &GgufCuda, pre_normed: Option<Region>) -> Result<(), EngError> {
        let l = self.layout;
        let [gamma, bias, conv] = self.weights.mlp;
        let controls = &self.plan.controls;
        let request = ResidentMlpRequest {
            layer: self.plan.visit.weight_layer as u32,
            caps: controls.caps,
            scale: controls.scale,
            dtypes: controls.dtypes,
            controller: controls.names.each_ref().map(String::as_str),
            rank: controls.rank,
            eps: self.plan.math.eps,
            input: self.row(l.mlp_input)?,
            pre_normed: pre_normed.map(|row| self.row(row)).transpose()?,
            message: self.row(l.messages[3])?,
            gamma: self.weights.row(gamma)?,
            bias: self.weights.row(bias)?,
            out: self.row(l.mlp_raw)?,
            health: self.row(l.mlp_health)?,
            gate_health: self.row(l.glue_health)?,
            status: self.row(l.mlp_status)?,
        };
        if self.plan.shape.banked {
            cuda.enqueue_controlled_mlp(self.stream, request)?;
        } else {
            cuda.enqueue_plain_mlp(self.stream, request)?;
        }
        let shape = self.plan.shape.widths;
        self.context.inkling_conv(
            self.stream,
            ConvCall {
                input: self.row(l.mlp_raw)?,
                kernel: self.weights.row(conv)?,
                history: self.row(l.histories[3])?,
                out: self.row(l.mlp_conv)?,
                // The convolved row is added into the depth stream's current row in the same
                // launch, the add of the visit's MLP output.
                state: Some(self.row(l.depth.row(shape.hidden))?),
                health: self.row(l.glue_health)?,
            },
            (shape.hidden, shape.taps),
        )
    }
}

fn append(bytes: &mut Vec<u8>, row: &[f32]) {
    bytes.extend(row.iter().flat_map(|value| value.to_ne_bytes()));
}
