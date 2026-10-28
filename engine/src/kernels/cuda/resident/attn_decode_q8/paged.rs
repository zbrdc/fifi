//! Operation-scoped page composition; one scratch owner, one stream and one immutable query.
use super::{
    AttnQ8CombineCall, AttnQ8PartialCall, AttnQ8Shape, PAGED_SPLITS, attn_q8_part_bytes,
    paged_ints, product, rule,
};
use crate::EngError;
use crate::kernels::cuda::context::{
    Claim, CudaContext, DeviceBuffer, DeviceRange, DeviceStream, require_apart, require_fit,
};

/// Only this owner exposes scratch to kernels; a pass borrows it exclusively.
/// Paged resource planning must replace the old arena part term, not retain both.
pub(crate) struct PagedQ8Scratch {
    buffer: DeviceBuffer,
    shape: AttnQ8Shape,
    page_cells: usize,
    stream_id: Option<u64>,
}

/// One query's immutable inputs. visible_keys is a positive count, not a position.
pub(crate) struct PagedQ8Input<'a> {
    pub(crate) q: DeviceRange<'a>,
    pub(crate) visible_keys: usize,
    pub(crate) scale: f32,
}

/// A ready page's planes. Its group identity must match the next required group.
/// Producers must finish on the pass stream or be explicitly joined before push.
pub(crate) struct PagedQ8Page<'a> {
    pub(crate) group: usize,
    pub(crate) k_codes: DeviceRange<'a>,
    pub(crate) k_scales: DeviceRange<'a>,
    pub(crate) v_codes: DeviceRange<'a>,
    pub(crate) v_scales: DeviceRange<'a>,
}

/// Each exclusive borrow starts a fresh generation, even if device bytes remain old.
/// The query must not be modified until this pass's queued work completes.
pub(crate) struct PagedQ8Pass<'a> {
    scratch: &'a mut PagedQ8Scratch,
    ctx: &'a CudaContext,
    stream: &'a DeviceStream,
    input: PagedQ8Input<'a>,
    next: usize,
}

impl CudaContext {
    /// Allocates exactly one full stride-64 scratch, before any page iteration.
    pub(crate) fn paged_q8_scratch(
        &self,
        shape: AttnQ8Shape,
        page_cells: usize,
    ) -> Result<PagedQ8Scratch, EngError> {
        self.ensure_usable()?;
        self.ensure_uncaptured("paged attention scratch")?;
        let bytes = PagedQ8Scratch::allocation_bytes(&shape, page_cells)?;
        Ok(PagedQ8Scratch {
            buffer: self.allocate(bytes)?,
            shape,
            page_cells,
            stream_id: None,
        })
    }
}

impl PagedQ8Scratch {
    /// The mutable borrow and one-stream binding retain the complete native stride.
    pub(crate) fn resident_workspace(
        &mut self,
        ctx: &CudaContext,
        stream: &DeviceStream,
    ) -> Result<(AttnQ8Shape, DeviceRange<'_>), EngError> {
        ctx.ensure_usable()?;
        ctx.ensure_uncaptured("resident attention scratch")?;
        ctx.check_launch_stream(stream)?;
        if self.stream_id.is_some_and(|id| id != stream.id()) {
            return Err(rule(
                "resident scratch",
                "scratch is bound to another stream",
            ));
        }
        let part = self.buffer.range(0, self.buffer.bytes())?;
        ctx.raw_range(&part)?;
        self.stream_id = Some(stream.id());
        Ok((self.shape, part))
    }

    /// The allocation and early admission use the same validated stride-64 size.
    pub(crate) fn allocation_bytes(
        shape: &AttnQ8Shape,
        page_cells: usize,
    ) -> Result<usize, EngError> {
        paged_ints(shape, page_cells, shape.cells, false)?;
        attn_q8_part_bytes(1, shape.n_kv_head, shape.head_dim, PAGED_SPLITS)
    }

    /// Byte-range fit cannot establish a host store's physical strides or logical capacity.
    pub(crate) fn check_geometry(
        &self,
        kv_heads: usize,
        head_dim: usize,
        capacity: usize,
        page_cells: usize,
    ) -> Result<(), EngError> {
        if (
            self.shape.n_kv_head,
            self.shape.head_dim,
            self.shape.cells,
            self.page_cells,
        ) != (kv_heads, head_dim, capacity, page_cells)
        {
            return Err(rule(
                "paged store",
                "scratch geometry differs from the KV store",
            ));
        }
        Ok(())
    }

    /// Binds one immutable query/scale/geometry and a single ordered stream.
    /// The private allocation plus mutable borrow excludes concurrent scratch generations.
    pub(crate) fn begin<'a>(
        &'a mut self,
        ctx: &'a CudaContext,
        stream: &'a DeviceStream,
        input: PagedQ8Input<'a>,
    ) -> Result<PagedQ8Pass<'a>, EngError> {
        ctx.ensure_usable()?;
        ctx.ensure_uncaptured("paged attention pass")?;
        ctx.check_launch_stream(stream)?;
        paged_ints(&self.shape, self.page_cells, input.visible_keys, false)?;
        if !(input.scale.is_finite() && input.scale > 0.0 && input.scale < 1e30) {
            return Err(rule("paged pass", "invalid score scale"));
        }
        if self.stream_id.is_some_and(|id| id != stream.id()) {
            return Err(rule("paged pass", "scratch is bound to another stream"));
        }
        let q_bytes = product(&[self.shape.n_head, self.shape.head_dim, 4])?;
        let part = self.buffer.range(0, self.buffer.bytes())?;
        require_fit("q", &input.q, q_bytes, 16)?;
        require_apart(
            &Claim::of("q", &input.q, q_bytes),
            &Claim::of("part", &part, part.bytes()),
        )?;
        ctx.raw_range(&input.q)?;
        ctx.raw_range(&part)?;
        self.stream_id = Some(stream.id());
        Ok(PagedQ8Pass {
            scratch: self,
            ctx,
            stream,
            input,
            next: 0,
        })
    }
}

impl PagedQ8Pass<'_> {
    fn groups(&self) -> usize {
        self.input.visible_keys.div_ceil(self.scratch.page_cells)
    }

    /// Queues exactly the next page. Shape, query, scale and local length cannot vary per push.
    pub(crate) fn push(&mut self, page: PagedQ8Page<'_>) -> Result<(), EngError> {
        if page.group != self.next || self.next >= self.groups() {
            return Err(rule(
                "paged pass",
                "duplicate, skipped or excess page group",
            ));
        }
        let cells = self.scratch.page_cells;
        let local = (self.input.visible_keys - self.next * cells).min(cells);
        let mut shape = self.scratch.shape;
        shape.cells = cells;
        self.ctx.attn_decode_q8_partial(
            self.stream,
            AttnQ8PartialCall {
                shape,
                visible_keys: local,
                group: self.next,
                scale: self.input.scale,
                q: self.input.q,
                k_codes: page.k_codes,
                k_scales: page.k_scales,
                v_codes: page.v_codes,
                v_scales: page.v_scales,
                part: self.scratch.buffer.range(0, self.scratch.buffer.bytes())?,
            },
        )?;
        self.next += 1;
        Ok(())
    }

    /// Consumes a complete generation; output is valid after the stream drains.
    /// No other writer may modify the query until the queued work completes.
    pub(crate) fn finish(
        self,
        proj: Option<DeviceRange<'_>>,
        out: DeviceRange<'_>,
    ) -> Result<(), EngError> {
        if self.next != self.groups() {
            return Err(rule("paged pass", "incomplete page generation"));
        }
        let mut shape = self.scratch.shape;
        if proj.is_some() {
            shape.ld = product(&[shape.n_head, shape.head_dim, 2])?;
        }
        self.ctx.attn_decode_q8_combine(
            self.stream,
            AttnQ8CombineCall {
                shape,
                page_cells: self.scratch.page_cells,
                visible_keys: self.input.visible_keys,
                part: self.scratch.buffer.range(0, self.scratch.buffer.bytes())?,
                proj,
                out,
            },
        )
    }
}
