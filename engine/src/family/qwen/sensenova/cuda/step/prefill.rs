//! The walk of the Loop trunk over a chunk of rows, and the mixers of a chunk of several
//! rows. Every launch of those is the rows twin of the one-row launch over the same weights and the
//! same shared state (the per-visit GDN states and conv rings, the attention planes, the
//! YaRN table): the dense projections are the vendored MMQ over the quantized f32 rows, the
//! GDN visit is `gdn_conv` then the sequential scan (the kernel's own test equates a chunk
//! of it to that many decode steps), the attention visit is `attn_prep_chunk` then the
//! production prefill attention, and the mHC kernels take the row count.
//!
//! The scratch holds [`PREFILL_ROWS`] rows, allocated with the step. A prompt walks every chunk through all the visits in order, so each chunk leaves
//! the shared state where the same tokens decoded one by one would; the head runs on the
//! last row.

use super::*;
use crate::family::step::recipe::{Layer, Model, Walk};
use crate::family::step::rows::{FEW_ROWS, Out as StepOut, Rows as StepRows, StepIo};
use crate::family::step::scratch::{Alloc, Core, OnDevice, Widths, Zeroed};
use crate::family::step::state::State;
use crate::family::step::walk::{self, Half};
use crate::family::step::{attend, delta, ffn, head, site};
use crate::kernels::cuda::prefill::attn_prefill::CPF_ATTN_PART_CAP;
use crate::kernels::cuda::prefill::mmq_size::{fixup_floats_max, q8_1_bytes as mmq_q8_1_bytes};

/// Rows of one chunk. The mmq chunk cap is 4096; at 1024 the attention partials fit three
/// splits under the 64 MiB part cap, and `row_add` takes a whole chunk's residual add in one
/// launch.
pub(crate) const PREFILL_ROWS: usize = 1024;

/// `row_add`'s float limit per launch.
const ROW_ADD_MAX: usize = 4_194_304;

/// `FIFI_SENSENOVA_PREFILL_GEMM=dq` puts every Q4_K/Q6_K projection on the dequantizing GEMM
/// (f32 activations, no q8_1 step): 4.8x slower than MMQ at 1024 rows (305 against 1490
/// tok/s on the 3090), and the accuracy reference until an f16 tensor-core GEMM serves these
/// blocks. Cached: every projection launch would otherwise take an env lock.
pub(super) fn exact_gemm() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("FIFI_SENSENOVA_PREFILL_GEMM").is_ok_and(|v| v == "dq"))
}

/// The prefill scratch: every decode row buffer at [`PREFILL_ROWS`] rows, the MMQ q8_1
/// columns of the four projection widths, and the attention partials at the part cap.
pub(super) struct Rows {
    pub(super) core: Core,
    /// The two mHC streams of every row, adjacent as `[a || b]`.
    pub(super) streams: DeviceBuffer,
    /// The routed input rows of the current mHC visit.
    pub(super) routed: DeviceBuffer,
    pub(super) router_normed: DeviceBuffer,
    pub(super) router_q8: DeviceBuffer,
    pub(super) router_logits: DeviceBuffer,
    pub(super) route_cell: DeviceBuffer,
    /// The attention context rows of the chunk, before the o projection.
    attn_out: DeviceBuffer,
    pub(super) gu: DeviceBuffer,
    /// The activation rows; the decode FFN borrows the first row (the column-split SwiGLU
    /// writes it, the vendored quantize reads it).
    pub(super) act: DeviceBuffer,
    pub(super) q_out: DeviceBuffer,
    pub(super) q8a: DeviceBuffer,
    pub(super) q8f: DeviceBuffer,
    pub(super) part: DeviceBuffer,
    /// The SM count the attention launch plan fills.
    nsm: usize,
}

/// The resident q8_1 bytes of `rows` rows of `k` values.
fn q8_rows(k: usize, rows: usize) -> Result<usize, EngError> {
    q8_1_bytes(k, rows).ok_or_else(|| EngError::BadRequest("a q8 rows size overflows".to_owned()))
}

/// Allocates the scratch for a step of `shape`; `nsm` is the device's SM count.
pub(super) fn open_rows(
    ctx: &CudaContext,
    shape: &LoopShape,
    nsm: usize,
) -> Result<Rows, EngError> {
    let (h, ff) = (shape.hidden, shape.feed_forward);
    let value_width = shape.delta_value_heads * shape.delta_value_dim;
    let q_width = shape.heads_q * shape.head_dim;
    let proj_width = (qkvz_row(shape) + 2 * shape.delta_value_heads).max(attn_row(shape));
    if PREFILL_ROWS.checked_mul(h).is_none_or(|n| n > ROW_ADD_MAX) {
        return Err(EngError::BadRequest(
            "the Loop shape's hidden breaks row_add's one-launch limit over a prefill chunk"
                .to_owned(),
        ));
    }
    let a = OnDevice(ctx);
    let floats = |width: usize| a.zeroed(PREFILL_ROWS * width * F32);
    let mmq =
        |k: usize| -> Result<DeviceBuffer, EngError> { a.zeroed(mmq_q8_1_bytes(k, PREFILL_ROWS)?) };
    let core = Core::new(
        &Zeroed(OnDevice(ctx)),
        Widths {
            rows: PREFILL_ROWS,
            hidden: h,
            streams: 1,
            proj: proj_width,
            conv: shape.conv_dim(),
            read: value_width,
            // The widest row a batched product reads.
            product: ff.max(h).max(value_width).max(q_width),
            vocab: shape.vocab,
            fixup: fixup_floats_max(nsm)? * F32,
        },
    )?;
    Ok(Rows {
        core,
        streams: floats(2 * h)?,
        routed: floats(h)?,
        router_normed: floats(2 * h)?,
        router_q8: a.zeroed(q8_rows(2 * h, PREFILL_ROWS)?)?,
        router_logits: floats(8)?,
        route_cell: floats(6)?,
        attn_out: floats(q_width)?,
        gu: floats(2 * ff)?,
        act: floats(ff)?,
        q_out: floats(q_width)?,
        q8a: mmq(q_width)?,
        q8f: mmq(ff)?,
        part: a.zeroed(CPF_ATTN_PART_CAP)?,
        nsm,
    })
}

impl LoopStep<'_> {
    /// The prompt: `embed_rows` holds `[m][hidden]`, token `t` at position `p0 + t`. Every
    /// chunk walks the 56 visits; the head runs on the last row, so `logits_row` is the next
    /// token's row, as after a decode step.
    pub(crate) fn prefill(&mut self, embed_rows: &[f32], p0: usize) -> Result<(), EngError> {
        let h = self.shape.hidden;
        if embed_rows.is_empty() || !embed_rows.len().is_multiple_of(h) {
            return Err(EngError::BadRequest(
                "the embed rows are empty or not whole hidden-wide rows".to_owned(),
            ));
        }
        // A chunk may regrow the workspace the recorded step holds: the next steps record again.
        self.replay = Replay::default();
        let mut m = 0;
        for (chunk, rows) in embed_rows.chunks(PREFILL_ROWS * h).enumerate() {
            m = rows.len() / h;
            ensure_rows(p0 + chunk * PREFILL_ROWS, m, self.cache_rows)?;
            self.walk(&le_f32(rows), m, p0 + chunk * PREFILL_ROWS, None)?;
        }
        self.head(m - 1, false)
    }

    /// The output norm of residual row `row` and the head, into the logits row.
    pub(super) fn head(&self, row: usize, exact: bool) -> Result<(), EngError> {
        let h = self.shape.hidden;
        let head = head::Head {
            norm: Some(StepIo::whole(self.output_norm)?),
            eps: self.shape.eps,
            weight: head::HeadWeight::Native(self.head),
            softcap: None,
            normed: self.rows.core.normed_rows(),
            logits: &self.rows.core.logits,
        };
        let last = floats(&self.rows.core.resid[self.rows.core.cur.get()], row * h, h)?;
        head.logits(&self.io(exact), last)
    }

    /// Row 0 of `buffer` joins the trace, once the stream drained.
    fn snap(
        &self,
        trace: &mut Option<&mut Vec<Vec<f32>>>,
        buffer: &DeviceBuffer,
    ) -> Result<(), EngError> {
        let Some(trace) = trace.as_deref_mut() else {
            return Ok(());
        };
        self.stream.synchronize()?;
        let mut bytes = vec![0u8; self.shape.hidden * F32];
        self.ctx.download(buffer, &mut bytes)?;
        let row = bytes.as_chunks::<4>().0.iter();
        trace.push(row.map(|c| f32::from_le_bytes(*c)).collect());
        Ok(())
    }

    /// One chunk: `m` rows whose first token sits at `p0`, through every visit in order. One
    /// row is a decode step: its mixers are the one-row kernels, which read the position from
    /// the step parameters. `trace` takes row 0 of every visit's mixer output and residual.
    pub(super) fn walk(
        &mut self,
        embed: &[u8],
        m: usize,
        p0: usize,
        mut trace: Option<&mut Vec<Vec<f32>>>,
    ) -> Result<(), EngError> {
        self.feed(embed, m, p0)?;
        self.visits(m, p0, trace.take())
    }

    /// The uploads of one chunk: its embedding rows into the residual and, for one row, the
    /// position into the step parameters.
    pub(super) fn feed(&self, embed: &[u8], m: usize, p0: usize) -> Result<(), EngError> {
        // Behind the last chunk's kernels: a blocking copy would land at once, and the
        // chunk before would write its final residual over it.
        self.rows.core.resid[0].write_at_enqueue(&self.stream, 0, embed)?;
        if m <= FEW_ROWS {
            let params = StepParams::new(0, p0 as i32)?;
            self.params
                .write_at_enqueue(&self.stream, 0, &params.to_bytes())?;
        }
        Ok(())
    }

    /// The visits of one fed chunk, in order. It queues launches only unless `trace` is set,
    /// and for one row the launches do not depend on `p0`. `trace` takes row 0 of every
    /// visit's mixer output and residual.
    pub(super) fn visits(
        &self,
        m: usize,
        p0: usize,
        mut trace: Option<&mut Vec<Vec<f32>>>,
    ) -> Result<(), EngError> {
        walk::visits(&Walk(self), m, p0, |_, half| {
            let rows = match half {
                Half::Mixer => &self.rows.core.mix_out,
                Half::Forward => &self.rows.core.resid[self.rows.core.cur.get()],
            };
            self.snap(&mut trace, rows)
        })
    }

    /// What the shared projections of this step need. `exact` puts them on the decoding
    /// routes.
    pub(super) fn io(&self, exact: bool) -> StepIo<'_> {
        Model::io(self, exact)
    }
}

impl Model for LoopStep<'_> {
    fn family(&self) -> &'static str {
        self.family
    }

    fn layers(&self) -> &[Layer] {
        &self.layers
    }

    fn io(&self, exact: bool) -> StepIo<'_> {
        StepIo {
            family: self.family,
            ctx: &self.ctx,
            stream: &self.stream,
            weights: self.weights,
            derived: None,
            mmq_q8: &self.rows.core.product_q8,
            fixup: &self.rows.core.fixup,
            exact,
        }
    }

    /// Each row count has its own switch for the decoding routes.
    fn exact(&self, n: usize) -> bool {
        if n == 1 { self.dq } else { self.exact_chunks }
    }

    fn core(&self) -> &Core {
        &self.rows.core
    }

    fn state(&self) -> &State {
        &self.state
    }

    fn hidden(&self) -> usize {
        self.shape.hidden
    }

    fn eps(&self) -> f32 {
        self.shape.eps
    }

    /// A chunk of a few rows reads its first position from the step parameters.
    fn begin(&self, n: usize) -> Result<(), EngError> {
        if n <= FEW_ROWS {
            self.ctx
                .step_begin(&self.stream, bytes_range(&self.params, STEP_PARAMS_BYTES)?)?;
        }
        Ok(())
    }

    fn delta(&self) -> (delta::Dims, Option<delta::ChannelRows<'_>>) {
        (self.delta_dims, None)
    }

    fn attend(
        &self,
        n: usize,
        p0: usize,
    ) -> Result<(attend::Dims, attend::AttendRows<'_>, attend::At<'_>), EngError> {
        let rows = &self.rows;
        let at = if n <= FEW_ROWS {
            attend::At::Row {
                pos: self.params_word(OFFSET_POS)?,
                kv_len: self.params_word(OFFSET_KV_LEN)?,
                status: self.params_word(OFFSET_STATUS)?,
            }
        } else {
            attend::At::Chunk {
                p0,
                sm_count: rows.nsm,
            }
        };
        let attend_rows = attend::AttendRows {
            input: rows.core.normed_rows(),
            proj: &rows.core.proj,
            query: &rows.q_out,
            attended: StepRows::both(&rows.q8a, &rows.attn_out),
            part: &rows.part,
            inv_freq: &self.inv_freq,
            out: &rows.core.mix_out,
        };
        Ok((self.attn_dims, attend_rows, at))
    }

    /// The gate and up rows share one strided row for the swiglu.
    fn dense(&self, n: usize) -> Result<(ffn::Join, ffn::Scratch<'_>), EngError> {
        let ff = self.shape.feed_forward;
        let gu = floats(&self.rows.gu, 0, n * 2 * ff)?;
        let half = |col_off| StepOut {
            dst: gu,
            ld: 2 * ff,
            col_off,
        };
        let scratch = ffn::Scratch {
            gate: half(0),
            up: half(ff),
            // The matvec route reads these q8_1 rows; the batched route reads the f32 rows.
            act: StepRows::both(&self.rows.q8f, &self.rows.act),
        };
        Ok((ffn::Join::Swiglu, scratch))
    }

    /// The shared expert works in the rows of the dense feed-forward.
    fn routed(
        &self,
        n: usize,
    ) -> Result<((moe::Dims, ffn::Join), &moe::Scratch, ffn::Scratch<'_>), EngError> {
        let Some((dims, scratch)) = &self.moe else {
            return Err(EngError::BadRequest(format!(
                "{}: this model has no routed feed-forward",
                self.family
            )));
        };
        let (join, shared) = self.dense(n)?;
        Ok(((*dims, join), scratch, shared))
    }

    fn routed_site(&self) -> Result<site::Routed<'_>, EngError> {
        let rows = &self.rows;
        Ok(site::Routed {
            hidden: self.shape.hidden,
            eps: self.shape.eps,
            pre_scale: self.routing.pre_scale,
            iterations: self.routing.iterations,
            fixed_identity: self.routing.fixed_identity,
            ones: &self.ones_router,
            streams: &rows.streams,
            normed: &rows.router_normed,
            normed_q8: &rows.router_q8,
            logits: &rows.router_logits,
            input: &rows.routed,
            cell: &rows.route_cell,
        })
    }
}
