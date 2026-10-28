//! Bounded-row EXL3 decoder. Session state is separate from borrowed immutable weights.

use super::{Frame, RowAddCall, RowShape, bad, exl3};
use crate::EngError;
use crate::family::flint::{
    config::FULL_LAYERS,
    cuda_kv::{KvAppend, LayerKv, LayerKvAllocation},
    exl3_resident::{FlintExl3Resident, FlintLayerView},
    kv_plan::{KvPlan, KvStorage},
};
use crate::family::project::cuda::Exl3MatrixView;
use crate::kernels::cuda::{
    CudaContext, DeviceBuffer, DeviceRange, DeviceStream,
    prefill::{
        attn_prefill::{
            AttnKvLayout, AttnKvStorage, AttnKvView, AttnPrefillCall, AttnPrefillShape,
            AttnScalarLayout, AttnStrides, CPF_ATTN_SM_COUNT, CpfAttnSpan, CpfAttnSplit,
            cpf_attn_launch_plan, cpf_attn_part_bytes,
        },
        flint_qk::QkProducerCall,
    },
    resident::{
        add_norm_pack::AddNormCall,
        kv8_append::{Kv8CodeScale, Kv8Data, Kv8Shape},
    },
};
use crate::tensor::Dtype;

mod health;
mod scratch;
#[cfg(test)]
pub(crate) use health::test_observations as test_health_observations;
use health::{Group, Health, Slot};
use scratch::HotScratch;

type Result<T> = std::result::Result<T, EngError>;
pub(crate) type ProjectionCheck<'a> =
    dyn FnMut(&DeviceStream, &str, &DeviceBuffer, &DeviceBuffer) -> Result<()> + 'a;

const WIDTH: usize = 2048;
const ROUTED_INTERMEDIATE: usize = 512;
pub(crate) const VOCAB: usize = 150_144;

#[cfg(test)]
std::thread_local! {
    static TEST_COUNTS: std::cell::Cell<[usize; 4]> = const { std::cell::Cell::new([0; 4]) };
    static TEST_HEAD_READS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    static TEST_HEAD_RANGES: std::cell::RefCell<Vec<[usize; 3]>> = const { std::cell::RefCell::new(Vec::new()) };
    static TEST_Q8_CHUNK_LAUNCHES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    static TEST_COMMIT_ENDS: std::cell::RefCell<Vec<usize>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// Cumulative producers, physical F16 H2D KV writes, CPF queries and successful commits.
#[cfg(test)]
pub(crate) fn test_counts() -> [usize; 4] {
    TEST_COUNTS.with(std::cell::Cell::get)
}

#[cfg(test)]
pub(crate) fn test_head_reads() -> usize {
    TEST_HEAD_READS.with(std::cell::Cell::get)
}

/// Absolute buffer offset, view bytes, and owner bytes at head entry.
#[cfg(test)]
pub(crate) fn test_head_ranges() -> Vec<[usize; 3]> {
    TEST_HEAD_RANGES.with(|ranges| ranges.borrow().clone())
}

#[cfg(test)]
pub(crate) fn test_commit_ends() -> Vec<usize> {
    TEST_COMMIT_ENDS.with(|ends| ends.borrow().clone())
}

#[cfg(test)]
fn test_count(index: usize) {
    TEST_COUNTS.with(|counts| {
        let mut values = counts.get();
        values[index] += 1;
        counts.set(values);
    });
}

pub(crate) struct State {
    plan: KvPlan,
    kv: Vec<LayerKv>,
    hot: HotScratch,
    grouped: exl3::GroupedScratch,
    health: Option<Health>,
    poisoned: bool,
}

impl State {
    pub(crate) fn scratch_bytes(plan: &KvPlan) -> Result<u64> {
        let grouped = exl3::GroupedScratch::bytes(plan.context_rows(), WIDTH, ROUTED_INTERMEDIATE)?;
        let health = Health::bytes(plan)?;
        HotScratch::bytes(plan)?
            .checked_add(grouped)
            .and_then(|bytes| bytes.checked_add(health))
            .ok_or_else(|| EngError::MemoryPlan("Flint session scratch charge overflows".into()))
    }

    pub(crate) fn new(ctx: &CudaContext, plan: KvPlan) -> Result<Self> {
        let kv = (0..49)
            .map(|layer| LayerKvAllocation::new(&plan, layer)?.allocate(ctx))
            .collect::<Result<Vec<_>>>()?;
        let hot = HotScratch::new(ctx, &plan)?;
        let grouped =
            exl3::GroupedScratch::new(ctx, plan.context_rows(), WIDTH, ROUTED_INTERMEDIATE)?;
        Ok(Self {
            plan,
            kv,
            hot,
            grouped,
            health: Health::new(ctx)?,
            poisoned: false,
        })
    }

    pub(crate) fn fed(&self) -> usize {
        self.kv[0].cursor.committed_prefix()
    }

    #[cfg(test)]
    pub(crate) fn test_kv(&self) -> &[LayerKv] {
        &self.kv
    }

    #[cfg(test)]
    pub(crate) fn test_poisoned(&self) -> bool {
        self.poisoned
    }

    #[cfg(test)]
    pub(crate) fn test_late_health_enabled(&self) -> bool {
        self.health.is_some()
    }

    #[cfg(test)]
    pub(crate) fn test_dirty_hot_scratch(&mut self, stream: &DeviceStream) -> Result<()> {
        self.hot.dirty(stream)
    }

    pub(crate) fn reset(&mut self, stream: &DeviceStream) -> Result<()> {
        if self.health.is_some() {
            self.poisoned = true;
        }
        stream.synchronize()?;
        if let Some(health) = &self.health {
            health.clear(stream)?;
        }
        for cache in &mut self.kv {
            cache.cursor.reset_after_quiescence();
        }
        self.poisoned = false;
        Ok(())
    }
}

pub(crate) struct Decoder<'a> {
    ctx: &'a CudaContext,
    stream: &'a DeviceStream,
    model: &'a FlintExl3Resident,
}

impl<'a> Decoder<'a> {
    pub(crate) fn new(
        ctx: &'a CudaContext,
        stream: &'a DeviceStream,
        model: &'a FlintExl3Resident,
    ) -> Result<Self> {
        if model.layer_count() != 49
            || model.head()?.shape != [WIDTH, VOCAB]
            || model.embedding()?.bytes() != VOCAB * WIDTH * 4
            || model.final_norm()?.bytes() != WIDTH * 4
        {
            return Err(bad("EXL3 decoder requires full49 Flint dimensions"));
        }
        Ok(Self { ctx, stream, model })
    }

    pub(crate) fn step<T>(
        &self,
        state: &mut State,
        token: u32,
        accept: impl FnOnce(&DeviceBuffer, &[f32]) -> Result<T>,
    ) -> Result<T> {
        self.step_inner(state, token, None, accept)
    }

    #[cfg(test)]
    pub(crate) fn step_observed<T>(
        &self,
        state: &mut State,
        token: u32,
        checks: &mut ProjectionCheck<'_>,
        accept: impl FnOnce(&DeviceBuffer, &[f32]) -> Result<T>,
    ) -> Result<T> {
        self.step_inner(state, token, Some(checks), accept)
    }

    fn step_inner<T>(
        &self,
        state: &mut State,
        token: u32,
        checks: Option<&mut ProjectionCheck<'_>>,
        accept: impl FnOnce(&DeviceBuffer, &[f32]) -> Result<T>,
    ) -> Result<T> {
        self.rows_inner(state, &[token], checks, |last, hot, checks| {
            self.head(last, hot, checks, accept)
        })
    }

    pub(crate) fn feed_rows(&self, state: &mut State, tokens: &[u32]) -> Result<()> {
        if tokens.is_empty() {
            return Ok(());
        }
        self.rows_inner(state, tokens, None, |_, _, _| Ok(()))
    }

    #[cfg(test)]
    pub(crate) fn feed_rows_logits(&self, state: &mut State, tokens: &[u32]) -> Result<Vec<f32>> {
        self.rows_inner(state, tokens, None, |last, hot, checks| {
            self.head(last, hot, checks, |_, logits| Ok(logits.to_vec()))
        })
    }

    #[cfg(test)]
    pub(crate) fn selected_row_offsets(rows: usize, selected: &[usize]) -> Result<Vec<usize>> {
        use crate::family::flint::settings::PREFILL_MAX_ROWS;
        if !(1..=PREFILL_MAX_ROWS).contains(&rows) || selected.iter().any(|&row| row >= rows) {
            return Err(bad("EXL3 selected head rows are outside the chunk"));
        }
        selected
            .iter()
            .map(|&row| row.checked_mul(WIDTH * 4).ok_or_else(|| bad("head offset")))
            .collect()
    }

    #[cfg(test)]
    pub(crate) fn feed_rows_selected_logits(
        &self,
        state: &mut State,
        tokens: &[u32],
        selected: &[usize],
    ) -> Result<Vec<Vec<f32>>> {
        let offsets = Self::selected_row_offsets(tokens.len(), selected)?;
        self.rows_inner(state, tokens, None, |last, hot, _| {
            let bytes = WIDTH * 4;
            let span = (tokens.len() - 1)
                .checked_mul(bytes)
                .ok_or_else(|| bad("head offset"))?;
            let origin = last
                .offset()
                .checked_sub(span)
                .ok_or_else(|| bad("head offset"))?;
            offsets
                .into_iter()
                .map(|offset| {
                    let offset = origin
                        .checked_add(offset)
                        .ok_or_else(|| bad("head offset"))?;
                    let row = last.buffer().range(offset, bytes)?;
                    self.head(row, hot, None, |_, logits| Ok(logits.to_vec()))
                })
                .collect()
        })
    }

    fn rows_inner<'c, T>(
        &self,
        state: &mut State,
        tokens: &[u32],
        checks: Option<&mut ProjectionCheck<'c>>,
        finish: impl FnOnce(
            DeviceRange<'_>,
            &mut HotScratch,
            Option<&mut ProjectionCheck<'c>>,
        ) -> Result<T>,
    ) -> Result<T> {
        if !(1..=crate::family::flint::settings::PREFILL_MAX_ROWS).contains(&tokens.len()) {
            return Err(bad("EXL3 prefill requires 1..=8 tokens"));
        }
        if tokens.iter().any(|&token| token as usize >= VOCAB) {
            return Err(bad("EXL3 token is outside the vocabulary"));
        }
        if state.poisoned {
            return Err(bad("EXL3 decoder requires quiescent reset after failure"));
        }
        if state
            .fed()
            .checked_add(tokens.len())
            .is_none_or(|end| end > state.plan.context_rows())
        {
            return Err(EngError::ContextLimit("EXL3 context is full".into()));
        }
        state.poisoned = true;
        let result = self.compute(state, tokens, checks, finish);
        if let Err(original) = result {
            return Err(
                if let Some(health) = &state.health
                    && self.ctx.ensure_usable().is_ok()
                {
                    abort_error(original, health.pending_error(self.stream))
                } else {
                    original
                },
            );
        }
        state.poisoned = false;
        result
    }

    fn embedding(&self, tokens: &[u32]) -> Result<DeviceBuffer> {
        let embedding = self.model.embedding()?;
        let mut group = Group::new(self.ctx, self.stream, None, 0)?;
        let row = |token: u32| {
            embedding
                .buffer()
                .range(embedding.offset() + token as usize * WIDTH * 4, WIDTH * 4)
        };
        if let [token] = tokens {
            return publish(&mut group, row(*token)?, WIDTH, Slot::Norm);
        }
        let gathered = self.ctx.allocate(tokens.len() * WIDTH * 4)?;
        for (index, &token) in tokens.iter().enumerate() {
            self.ctx.copy_device(
                self.stream,
                row(token)?,
                gathered.range(index * WIDTH * 4, WIDTH * 4)?,
            )?;
        }
        publish(
            &mut group,
            gathered.range(0, gathered.bytes())?,
            WIDTH,
            Slot::Norm,
        )
    }

    fn compute<'c, T>(
        &self,
        session: &mut State,
        tokens: &[u32],
        mut checks: Option<&mut ProjectionCheck<'c>>,
        finish: impl FnOnce(
            DeviceRange<'_>,
            &mut HotScratch,
            Option<&mut ProjectionCheck<'c>>,
        ) -> Result<T>,
    ) -> Result<T> {
        let position = session.fed();
        let rows = tokens.len();
        #[cfg(test)]
        let _route_pass = super::route_trace::pass(position, rows)?;
        let mut state = self.embedding(tokens)?;
        let mut appends = Vec::with_capacity(49);
        for (index, cache) in session.kv.iter_mut().enumerate() {
            let layer = self.model.layer(index)?;
            let append = cache.cursor.begin_append(position, rows)?;
            let mut group = Group::new(self.ctx, self.stream, session.health.as_ref(), index)?;
            group.validate_layer(&layer).map_err(model_forward_error)?;
            let norm = normalize(
                &mut group,
                state.range(0, state.bytes())?,
                layer.input_norm,
                session.hot.norm(rows)?,
                Slot::Norm,
            )?;
            let prefix = format!("model.layers.{index}.self_attn");
            let q = projection(
                &mut group,
                layer.q,
                &norm,
                checks.as_deref_mut(),
                &format!("{prefix}.q_proj"),
                [Slot::RawQ, Slot::PublishQ],
            )?;
            let k = projection(
                &mut group,
                layer.k,
                &norm,
                checks.as_deref_mut(),
                &format!("{prefix}.k_proj"),
                [Slot::RawK, Slot::PublishK],
            )?;
            let v = projection(
                &mut group,
                layer.v,
                &norm,
                checks.as_deref_mut(),
                &format!("{prefix}.v_proj"),
                [Slot::RawV, Slot::PublishV],
            )?;
            let attended = attention(
                &mut group,
                [&q, &k, &v],
                &layer,
                ([&cache.keys, &cache.values], cache.scales.as_ref(), &append),
                !FULL_LAYERS.contains(&(index as u64)),
                session.hot.part()?,
            )?;
            let output = projection(
                &mut group,
                layer.o,
                &attended,
                checks.as_deref_mut(),
                &format!("{prefix}.o_proj"),
                [Slot::RawO, Slot::PublishO],
            )?;
            let residual = residual(&mut group, &state, &output)?;
            let norm = normalize(
                &mut group,
                residual.range(0, residual.bytes())?,
                layer.post_attention_norm,
                session.hot.norm(rows)?,
                Slot::PostNorm,
            )?;
            group.check()?;
            drop(group);
            #[cfg(test)]
            let route_layer = super::route_trace::layer(index)?;
            let ffn = exl3::forward_reusing(
                self.ctx,
                self.stream,
                &layer.ffn,
                norm.range(0, norm.bytes())?,
                exl3::ScratchLoans {
                    router: Some(session.hot.router()?),
                    grouped: Some(&session.grouped),
                },
            )
            .map_err(model_forward_error)?;
            #[cfg(test)]
            drop(route_layer);
            state = self::residual(
                &mut Group::new(self.ctx, self.stream, None, 0)?,
                &residual,
                &ffn,
            )?;
            appends.push(append);
        }
        let output = finish(
            state.range((rows - 1) * WIDTH * 4, WIDTH * 4)?,
            &mut session.hot,
            checks,
        )?;
        self.stream.synchronize()?;
        for append in appends {
            #[cfg(test)]
            TEST_COMMIT_ENDS.with(|ends| ends.borrow_mut().push(append.live_range().end));
            append.commit();
            #[cfg(test)]
            test_count(3);
        }
        Ok(output)
    }

    fn head<T>(
        &self,
        state: DeviceRange<'_>,
        hot: &mut HotScratch,
        checks: Option<&mut ProjectionCheck<'_>>,
        accept: impl FnOnce(&DeviceBuffer, &[f32]) -> Result<T>,
    ) -> Result<T> {
        #[cfg(test)]
        TEST_HEAD_RANGES.with(|ranges| {
            ranges
                .borrow_mut()
                .push([state.offset(), state.bytes(), state.buffer().bytes()]);
        });
        let norm = normalize(
            &mut Group::new(self.ctx, self.stream, None, 0)?,
            state,
            self.model.final_norm()?,
            hot.norm(1)?,
            Slot::Norm,
        )?;
        let head = exl3::project(
            self.ctx,
            self.stream,
            self.model.head()?,
            norm.range(0, norm.bytes())?,
        )
        .map_err(model_forward_error)?;
        if let Some(checks) = checks {
            checks(self.stream, "lm_head", &norm, &head)?;
        }
        #[cfg(test)]
        TEST_HEAD_READS.with(|count| count.set(count.get() + 1));
        let logits = values(self.stream, head.range(0, head.bytes())?)?;
        if logits.len() != VOCAB {
            return Err(bad("EXL3 head has invalid dimensions"));
        }
        if logits.iter().any(|v| !v.is_finite()) {
            return Err(EngError::ModelOutput(
                "Flint EXL3 head has nonfinite logits".into(),
            ));
        }
        // Sampling/status acceptance belongs inside the same all-layer transaction.
        accept(&head, &logits)
    }
}

// Only validated model-forward calls use this; request checks and acceptance stay outside it.
pub(super) fn model_forward_error(error: EngError) -> EngError {
    match error {
        EngError::BadRequest(message) => EngError::ModelOutput(message),
        other => other,
    }
}

fn abort_error(original: EngError, pending: Result<Option<EngError>>) -> EngError {
    match pending {
        Ok(Some(error)) => error,
        Ok(None) | Err(_) => original,
    }
}

fn publish(
    group: &mut Group<'_, '_>,
    raw: DeviceRange<'_>,
    columns: usize,
    slot: Slot,
) -> Result<DeviceBuffer> {
    let shape = exl3::row_shape(raw.bytes(), columns)?;
    let mut frame = group.frame(slot)?;
    let input = frame.input(raw, shape)?;
    let output = frame.publish(input, shape)?;
    group.finish(frame, output)
}

fn normalize(
    group: &mut Group<'_, '_>,
    input: DeviceRange<'_>,
    gamma: DeviceRange<'_>,
    packed: DeviceRange<'_>,
    slot: Slot,
) -> Result<DeviceBuffer> {
    let ctx = group.ctx;
    let stream = group.stream;
    let shape = exl3::row_shape(input.bytes(), WIDTH)?;
    let mut frame = group.frame(slot)?;
    let input = frame.input(input, shape)?;
    let output = frame.allocate(shape)?;
    ctx.add_norm_pack(
        stream,
        AddNormCall {
            m: shape.rows,
            n: WIDTH,
            eps: 1e-6,
            h: frame.range(input)?,
            delta: None,
            weight: gamma,
            xf: packed,
            norm_out: Some(frame.range(output)?),
        },
    )?;
    let output = frame.publish(output, shape)?;
    group.finish(frame, output)
}

fn projection(
    group: &mut Group<'_, '_>,
    weight: Exl3MatrixView<'_>,
    input: &DeviceBuffer,
    checks: Option<&mut ProjectionCheck<'_>>,
    prefix: &str,
    [raw_slot, publish_slot]: [Slot; 2],
) -> Result<DeviceBuffer> {
    let stream = group.stream;
    let raw = group
        .project(weight, input.range(0, input.bytes())?, raw_slot)
        .map_err(model_forward_error)?;
    if let Some(checks) = checks {
        group.check()?;
        checks(stream, prefix, input, &raw)?;
    }
    let output = publish(
        group,
        raw.range(0, raw.bytes())?,
        weight.shape[1],
        publish_slot,
    )?;
    if group.deferred() {
        group.keep(raw);
    }
    Ok(output)
}

fn residual(
    group: &mut Group<'_, '_>,
    input: &DeviceBuffer,
    delta: &DeviceBuffer,
) -> Result<DeviceBuffer> {
    let ctx = group.ctx;
    let stream = group.stream;
    if delta.bytes() != input.bytes() {
        return Err(bad("EXL3 residual dimensions disagree"));
    }
    let shape = exl3::row_shape(input.bytes(), WIDTH)?;
    let mut frame = group.frame(Slot::Residual)?;
    let output = frame.allocate(shape)?;
    ctx.row_add(
        stream,
        RowAddCall {
            x: input.range(0, input.bytes())?,
            y: delta.range(0, delta.bytes())?,
            out: frame.range(output)?,
            n: shape.rows * WIDTH,
        },
    )?;
    let output = frame.publish(output, shape)?;
    group.finish(frame, output)
}

fn producer_status(flag: u32) -> Result<()> {
    if flag != 0 {
        return Err(EngError::Backend {
            operation: "flint_qk_producer",
            message: format!("Q/K producer numeric flag {flag:#x}; output invalid"),
        });
    }
    Ok(())
}

fn prepare_qk(
    group: &mut Group<'_, '_>,
    qk: [&DeviceBuffer; 2],
    gamma: [DeviceRange<'_>; 2],
    position: Option<u32>,
) -> Result<[DeviceBuffer; 2]> {
    let ctx = group.ctx;
    let stream = group.stream;
    let rows = exl3::row_shape(qk[0].bytes(), 4096)?.rows;
    if qk[1].bytes() != rows * 512 * 4 {
        return Err(bad("EXL3 Q/K row counts disagree"));
    }
    let out = [
        ctx.allocate(rows * 4096 * 4)?,
        ctx.allocate(rows * 512 * 4)?,
    ];
    let scratch = [
        ctx.allocate(rows * 4096 * 4)?,
        ctx.allocate(rows * 512 * 4)?,
    ];
    let frame = group.frame(Slot::Producer)?;
    #[cfg(test)]
    test_count(0);
    ctx.flint_qk_producer(
        stream,
        QkProducerCall {
            q: qk[0].range(0, qk[0].bytes())?,
            k: qk[1].range(0, qk[1].bytes())?,
            gamma,
            out: [
                out[0].range(0, out[0].bytes())?,
                out[1].range(0, out[1].bytes())?,
            ],
            scratch: [
                scratch[0].range(0, scratch[0].bytes())?,
                scratch[1].range(0, scratch[1].bytes())?,
            ],
            flag: frame.flag()?,
            rows,
            position,
        },
    )?;
    if group.deferred() {
        for buffer in scratch {
            group.keep(buffer);
        }
    } else {
        let mut status = [0; 4];
        let flag = frame.flag()?;
        flag.buffer().read_at(stream, flag.offset(), &mut status)?;
        // StepStatus only recognizes 1/2/4, so validate this producer's mask8 separately.
        producer_status(u32::from_ne_bytes(status))?;
    }
    Ok(out)
}

fn f16_row(input: &[f32]) -> Result<Vec<u8>> {
    if input.len() != 512 {
        return Err(bad("Flint F16 KV write requires one 512-value row"));
    }
    let mut bytes = Vec::with_capacity(1024);
    for &value in input {
        let stored = half::f16::from_f32(value);
        if !value.is_finite() || value.to_bits() & 0xffff != 0 || !stored.is_finite() {
            return Err(EngError::ModelOutput(
                "Flint invalid BF16 input or overflow in physical KV cache".into(),
            ));
        }
        bytes.extend_from_slice(&stored.to_bits().to_ne_bytes());
    }
    Ok(bytes)
}

fn q8_view(planes: [&DeviceBuffer; 2], cells: usize) -> Result<AttnKvView<'_>> {
    Ok(AttnKvView {
        layout: AttnKvLayout {
            storage: AttnKvStorage::Q8Affine {
                group_size: 32,
                signed: true,
            },
            payload_strides: AttnStrides {
                time_stride_bytes: 128,
                head_stride_bytes: cells * 128,
            },
            scales: Some(AttnScalarLayout {
                dtype: Dtype::F16,
                strides: AttnStrides {
                    time_stride_bytes: 8,
                    head_stride_bytes: cells * 8,
                },
                group_stride_bytes: 2,
            }),
            bias: None,
        },
        payload: planes[0].range(0, planes[0].bytes())?,
        scales: Some(planes[1].range(0, planes[1].bytes())?),
        bias: None,
    })
}

fn write_cache<'b>(
    group: &Group<'_, '_>,
    kv: [&DeviceBuffer; 2],
    (cache, scales, append): (
        [&'b DeviceBuffer; 2],
        Option<&'b [DeviceBuffer; 2]>,
        &KvAppend<'_>,
    ),
    rows: usize,
) -> Result<Option<[AttnKvView<'b>; 2]>> {
    let scales = match (append.storage(), scales) {
        (KvStorage::F16, None) => None,
        (KvStorage::Q8G32, Some(scales)) => Some(scales),
        _ => return Err(bad("Flint cache storage and scale planes disagree")),
    };
    let runs = append.runs()?;
    if runs[0].1.start != 0 || runs[0].1.end != runs[1].1.start || runs[1].1.end != rows {
        return Err(bad("Flint cache append runs do not cover source rows"));
    }
    for (plane, row) in cache.iter().zip(kv) {
        if row.bytes() != rows * 512 * 4 {
            return Err(bad("EXL3 K/V row counts disagree"));
        }
        let values = values(group.stream, row.range(0, row.bytes())?)?;
        let mut bytes = Vec::with_capacity(rows * 1024);
        // Both profiles retain the published-BF16 and finite/F16-range acceptance guards.
        for row in values.as_chunks::<512>().0 {
            bytes.extend(f16_row(row)?);
        }
        if append.storage() == KvStorage::F16 {
            for (slot, source) in &runs {
                if source.is_empty() {
                    continue;
                }
                #[cfg(test)]
                test_count(1);
                plane.write_at(
                    group.stream,
                    slot * 1024,
                    &bytes[source.start * 1024..source.end * 1024],
                )?;
            }
        }
    }
    let Some(scales) = scales else {
        return Ok(None);
    };
    let cells = append.cache_rows();
    for (slot, source) in runs {
        if source.is_empty() {
            continue;
        }
        let offset = source.start * 512 * 4;
        let bytes = source.len() * 512 * 4;
        group.ctx.kv8_chunk(
            group.stream,
            Kv8Shape {
                rows: source.len(),
                n_kv_head: 4,
                head_dim: 128,
                cells,
                code_scale: Kv8CodeScale::StoredHalf,
            },
            Kv8Data {
                k: kv[0].range(offset, bytes)?,
                v: kv[1].range(offset, bytes)?,
                k_codes: cache[0].range(0, cache[0].bytes())?,
                k_scales: scales[0].range(0, scales[0].bytes())?,
                v_codes: cache[1].range(0, cache[1].bytes())?,
                v_scales: scales[1].range(0, scales[1].bytes())?,
            },
            slot,
        )?;
        #[cfg(test)]
        TEST_Q8_CHUNK_LAUNCHES.with(|count| count.set(count.get() + 1));
    }
    Ok(Some([
        q8_view([cache[0], &scales[0]], cells)?,
        q8_view([cache[1], &scales[1]], cells)?,
    ]))
}

fn attention(
    group: &mut Group<'_, '_>,
    qkv: [&DeviceBuffer; 3],
    layer: &FlintLayerView<'_>,
    (cache, scales, append): (
        [&DeviceBuffer; 2],
        Option<&[DeviceBuffer; 2]>,
        &KvAppend<'_>,
    ),
    sliding: bool,
    part: DeviceRange<'_>,
) -> Result<DeviceBuffer> {
    let ctx = group.ctx;
    let stream = group.stream;
    let rows = exl3::row_shape(qkv[0].bytes(), 4096)?.rows;
    let position = append.live_range().end - rows;
    let [q, k] = prepare_qk(
        group,
        [qkv[0], qkv[1]],
        [layer.q_norm, layer.k_norm],
        sliding.then_some(position as u32),
    )?;
    let kv_views = write_cache(group, [&k, qkv[2]], (cache, scales, append), rows)?;
    let storage = kv_views
        .as_ref()
        .map_or(AttnKvStorage::F16, |views| views[0].layout.storage);
    let geometry = HotScratch::geometry();
    let cache_rows = append.cache_rows();
    let span = CpfAttnSpan {
        window: if sliding { 513 } else { 0 },
        ring: if sliding && cache_rows >= 513 {
            cache_rows as i64
        } else {
            0
        },
    };
    let plans = (0..rows)
        .map(|row| {
            let plan = cpf_attn_launch_plan(
                geometry,
                position + row + 1,
                span.window as usize,
                CPF_ATTN_SM_COUNT,
            )?;
            Ok((plan, cpf_attn_part_bytes(geometry, plan)?))
        })
        .collect::<Result<Vec<_>>>()?;
    let needed = plans.iter().fold(0, |max, (_, bytes)| max.max(*bytes));
    if needed > part.bytes() {
        return Err(EngError::MemoryPlan(
            "EXL3 CPF exceeds persistent scratch".into(),
        ));
    }
    let shape = RowShape {
        rows,
        columns: 4096,
    };
    let mut frame = group.frame(Slot::Attention)?;
    let output = frame.allocate(shape)?;
    for (row, (plan, bytes)) in plans.into_iter().enumerate() {
        #[cfg(test)]
        test_count(2);
        ctx.attn_decode_sinks_via_prefill(
            stream,
            AttnPrefillCall {
                storage,
                kv_views,
                shape: AttnPrefillShape {
                    m: 1,
                    n_head: 32,
                    n_kv_head: 4,
                    head_dim: 128,
                    cache_rows,
                    ld: 0,
                    p0: (position + row) as i64,
                },
                span,
                split: CpfAttnSplit {
                    max_splits: plan.max_splits,
                    min_keys: plan.min_keys,
                },
                q_tile: plan.q_tile,
                scale: (128.0_f32).sqrt().recip(),
                q: q.range(row * 4096 * 4, 4096 * 4)?,
                proj: None,
                k_cache: cache[0].range(0, cache[0].bytes())?,
                v_cache: cache[1].range(0, cache[1].bytes())?,
                part: part.buffer().range(part.offset(), bytes)?,
                out: frame.buffers[output].0.range(row * 4096 * 4, 4096 * 4)?,
                status: Some(frame.flag()?),
                sinks: Some(layer.sinks),
            },
        )?;
    }
    let output = frame.publish(output, shape)?;
    if group.deferred() {
        group.keep(q);
        group.keep(k);
    }
    group.finish(frame, output)
}

pub(crate) fn values(stream: &DeviceStream, range: DeviceRange<'_>) -> Result<Vec<f32>> {
    if !range.bytes().is_multiple_of(4) {
        return Err(bad("EXL3 F32 backing length"));
    }
    let mut bytes = vec![0; range.bytes()];
    range.buffer().read_at(stream, range.offset(), &mut bytes)?;
    Ok(bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|word| f32::from_ne_bytes(*word))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    mod q8;
    mod real;

    #[test]
    fn late_health_abort_preserves_original_on_pending_read_failure() {
        let original = || EngError::ModelOutput("later physical F16 cache refusal".into());
        assert_eq!(abort_error(original(), Ok(None)), original());
        for error in [
            EngError::MemoryPlan("health staging failed".into()),
            EngError::Backend {
                operation: "health_read",
                message: "copy failed".into(),
            },
            EngError::ModelOutput("unexpected read failure".into()),
        ] {
            assert_eq!(abort_error(original(), Err(error)), original());
        }
    }

    #[test]
    fn late_health_abort_prefers_earlier_health_domain() {
        assert_eq!(
            abort_error(
                EngError::ModelOutput("later physical F16 cache refusal".into()),
                Ok(Frame::validate_health(2).err()),
            ),
            Frame::validate_health(2).unwrap_err(),
        );
    }

    #[test]
    fn producer_mask8_and_unknown_bits_fail_closed() {
        assert!(producer_status(0).is_ok());
        for flag in [1, 2, 4, 8, 9, 16, u32::MAX] {
            assert!(matches!(
                producer_status(flag),
                Err(EngError::Backend {
                    operation: "flint_qk_producer",
                    ..
                })
            ));
        }
    }

    #[test]
    fn model_forward_errors_preserve_non_request_classes() {
        for message in [
            "projection arithmetic flag",
            "router status",
            "late frame health",
        ] {
            assert_eq!(
                model_forward_error(EngError::BadRequest(message.into())),
                EngError::ModelOutput(message.into())
            );
        }
        for error in [
            EngError::Backend {
                operation: "flint_qk_producer",
                message: "status 8".into(),
            },
            EngError::Backend {
                operation: "argmax",
                message: "status 8".into(),
            },
            EngError::MemoryPlan("reserve".into()),
            EngError::ContextLimit("capacity".into()),
            EngError::ModelOutput("already classified".into()),
        ] {
            assert_eq!(model_forward_error(error.clone()), error);
        }
    }

    #[test]
    fn nonfinite_model_output_is_not_a_client_or_backend_failure() {
        assert!(matches!(
            f16_row(&[f32::NAN; 512]),
            Err(EngError::ModelOutput(_))
        ));
        assert!(matches!(f16_row(&[0.0; 511]), Err(EngError::BadRequest(_))));
    }

    #[test]
    fn physical_f16_bridge_keeps_bf16_values_and_refuses_overflow() {
        assert_eq!(
            f16_row(&[1.5; 512]).unwrap(),
            half::f16::from_f32(1.5).to_bits().to_ne_bytes().repeat(512)
        );
        for value in [f32::NAN, f32::INFINITY, 65536.0, 1.001] {
            assert!(f16_row(&[value; 512]).is_err());
        }
        assert!(f16_row(&[1.0; 511]).is_err());
    }

    #[test]
    fn device_qk_fixture_matches_test_only_cpu_oracle_bitwise() {
        if !crate::kernels::cuda::device_present("Flint serving Q/K fixture") {
            return;
        }
        let ctx = CudaContext::new().unwrap();
        let stream = ctx.stream().unwrap();
        let q: Vec<_> = (0..4096).map(|i| ((i % 29) - 14) as f32 / 8.0).collect();
        let k: Vec<_> = (0..512).map(|i| ((i % 23) - 11) as f32 / 16.0).collect();
        let gamma_q: Vec<_> = (0..128).map(|i| 1.0 + (i % 11) as f32 / 64.0).collect();
        let gamma_k: Vec<_> = (0..128).map(|i| 1.0 + (i % 7) as f32 / 32.0).collect();
        let upload = |row: &[f32]| {
            let bytes: Vec<_> = row.iter().flat_map(|value| value.to_ne_bytes()).collect();
            ctx.upload(&bytes).unwrap()
        };
        let inputs = [upload(&q), upload(&k)];
        let gamma = [upload(&gamma_q), upload(&gamma_k)];
        for position in [None, Some(0), Some(37), Some(513), Some(1026)] {
            let expected = crate::family::flint::cuda_ffn::qk_producer_oracle(
                &q,
                &k,
                [&gamma_q, &gamma_k],
                position,
            )
            .unwrap();
            let actual = prepare_qk(
                &mut Group::new(&ctx, &stream, None, 0).unwrap(),
                [&inputs[0], &inputs[1]],
                [
                    gamma[0].range(0, 512).unwrap(),
                    gamma[1].range(0, 512).unwrap(),
                ],
                position,
            )
            .unwrap();
            for (plane, expected) in actual.iter().zip(expected) {
                let actual = values(&stream, plane.range(0, plane.bytes()).unwrap()).unwrap();
                assert_eq!(
                    actual.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    "position {position:?}"
                );
            }
        }
        gamma[0]
            .write_at(&stream, 0, &f32::INFINITY.to_ne_bytes())
            .unwrap();
        assert!(matches!(
            prepare_qk(
                &mut Group::new(&ctx, &stream, None, 0).unwrap(),
                [&inputs[0], &inputs[1]],
                [
                    gamma[0].range(0, 512).unwrap(),
                    gamma[1].range(0, 512).unwrap()
                ],
                None
            ),
            Err(EngError::Backend {
                operation: "flint_qk_producer",
                ..
            })
        ));
    }
}
