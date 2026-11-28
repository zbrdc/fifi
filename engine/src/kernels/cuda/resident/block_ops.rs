//! The row operations of a hybrid decoder block: [`CudaContext::block_op`] queues one launch of
//! `native/resident/block_ops.cu` on a [`DeviceStream`] and returns when it is queued. It
//! allocates nothing, copies nothing and never synchronizes.
//!
//! Each operation is described by a [`BlockOp`] built by one constructor of this file; the slot
//! table, formulas and layouts of every operation are in `native/resident/block_ops.h`. All
//! operations take `n` token rows, so decode (`n = 1`) and prefill chunks use the same entry.
//!
//! Every refusal is a `BadRequest` and happens before any native call. In order: the backend is
//! usable; every integer fits a C int; each range holds the bytes the operation reads or writes
//! and is 4-byte aligned; each range the operation writes overlaps no other range of the call.
//! A refused call changes nothing.

use crate::EngError;
use crate::kernels::cuda::context::{
    Claim, CudaContext, DeviceRange, DeviceStream, require_apart, require_fit,
};
use crate::kernels::cuda::native::resident::block_ops::{
    BlockOpsRaw, FLOATS, INTS, PTRS, SlotRaw, block_ops_launch,
};
use crate::kernels::cuda::resident::quantize_q8_1::q8_1_bytes;

/// Slots of the generic argument block, named in refusal text.
const SLOT_NAMES: [&str; PTRS] = [
    "p0", "p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8", "p9", "p10", "p11",
];

const HC_HEAD: usize = 4;
const HC_MIXES: usize = 17;

/// One queued row operation: the code, its integers and floats, and the used slots.
pub(crate) struct BlockOp<'b> {
    op: i32,
    ints: [i32; INTS],
    floats: [f32; FLOATS],
    slots: Vec<(usize, DeviceRange<'b>, usize, bool)>,
}

fn int(value: usize) -> Result<i32, EngError> {
    i32::try_from(value)
        .map_err(|_| EngError::BadRequest("a block operation size does not fit a C int".into()))
}

fn bytes(factors: &[usize]) -> Result<usize, EngError> {
    factors
        .iter()
        .try_fold(1usize, |acc, factor| acc.checked_mul(*factor))
        .ok_or_else(|| EngError::BadRequest("a block operation size overflows".into()))
}

fn q8_rows(width: usize, rows: usize) -> Result<usize, EngError> {
    q8_1_bytes(width, rows)
        .ok_or_else(|| EngError::BadRequest("a block operation q8 size overflows".into()))
}

impl<'b> BlockOp<'b> {
    fn new(op: i32, ints: &[usize], floats: &[f32]) -> Result<Self, EngError> {
        let mut packed_ints = [0i32; INTS];
        for (slot, value) in ints.iter().enumerate() {
            packed_ints[slot] = int(*value)?;
        }
        let mut packed_floats = [0f32; FLOATS];
        packed_floats[..floats.len()].copy_from_slice(floats);
        Ok(Self {
            op,
            ints: packed_ints,
            floats: packed_floats,
            slots: Vec::new(),
        })
    }

    fn read(mut self, slot: usize, range: DeviceRange<'b>, need: usize) -> Self {
        self.slots.push((slot, range, need, false));
        self
    }

    fn write(mut self, slot: usize, range: DeviceRange<'b>, need: usize) -> Self {
        self.slots.push((slot, range, need, true));
        self
    }

    /// Mixed-residual read of one site: the stream mixes, the pre-weighted branch sum, and the
    /// post weights and stream-mixing matrix the matching apply consumes.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn hc_read(
        n: usize,
        h: usize,
        eps: f32,
        magnitude: f32,
        resid: DeviceRange<'b>,
        mix_w: DeviceRange<'b>,
        base: DeviceRange<'b>,
        scale: DeviceRange<'b>,
        branch: DeviceRange<'b>,
        post: DeviceRange<'b>,
        comb: DeviceRange<'b>,
    ) -> Result<Self, EngError> {
        let width = bytes(&[HC_HEAD, h])?;
        Ok(Self::new(1, &[n, h], &[eps, magnitude])?
            .read(0, resid, bytes(&[n, width, 4])?)
            .read(1, mix_w, bytes(&[HC_MIXES, width, 2])?)
            .read(2, base, HC_MIXES * 4)
            .read(3, scale, 16)
            .write(4, branch, bytes(&[n, h, 4])?)
            .write(5, post, bytes(&[n, HC_HEAD, 4])?)
            .write(6, comb, bytes(&[n, 16, 4])?))
    }

    /// The mean of the four streams of token row `row`.
    pub(crate) fn hc_mean(
        row: usize,
        h: usize,
        resid: DeviceRange<'b>,
        out: DeviceRange<'b>,
    ) -> Result<Self, EngError> {
        Ok(Self::new(4, &[row, h], &[])?
            .read(0, resid, bytes(&[row + 1, HC_HEAD, h, 4])?)
            .write(1, out, bytes(&[h, 4])?))
    }

    /// Copies each of `n` rows of `h` floats into all `streams` residual streams, bit for bit.
    pub(crate) fn hc_broadcast(
        n: usize,
        streams: usize,
        h: usize,
        src: DeviceRange<'b>,
        dst: DeviceRange<'b>,
    ) -> Result<Self, EngError> {
        if !(1..=65_536).contains(&n) || !(1..=16).contains(&streams) || !(1..=16_384).contains(&h)
        {
            return Err(EngError::BadRequest(format!(
                "hc_broadcast: invalid shape n={n}, streams={streams}, h={h}"
            )));
        }
        Ok(Self::new(17, &[n, streams, h], &[])?
            .read(0, src, bytes(&[n, h, 4])?)
            .write(1, dst, bytes(&[n, streams, h, 4])?))
    }

    /// Finishes the inputs of the shared delta-rule scan in the per-channel form: the `v`
    /// third of the `[q | k | v]` conv rows (`qkv`) scaled in place by the fold of its beta and
    /// delta logits, one decay value per key channel (`decay`) and the projection rows
    /// `[z | beta logits]` (`proj`, rows of `heads * head_dim + heads` floats).
    pub(crate) fn rda_pack(
        n: usize,
        heads: usize,
        head_dim: usize,
        lower_bound: f32,
        inputs: RdaInputs<'b>,
        outputs: RdaPacked<'b>,
    ) -> Result<Self, EngError> {
        let width = bytes(&[heads, head_dim])?;
        let row = bytes(&[n, width, 4])?;
        Ok(Self::new(6, &[n, heads, head_dim], &[lower_bound])?
            .read(0, inputs.g1, row)
            .read(1, inputs.dt_bias, bytes(&[width, 4])?)
            .read(2, inputs.a, bytes(&[heads, 4])?)
            .read(3, inputs.beta, bytes(&[n, heads, 4])?)
            .read(4, inputs.delta, bytes(&[n, heads, 4])?)
            .read(5, inputs.g2, row)
            .write(6, outputs.qkv, bytes(&[n, 3, width, 4])?)
            .write(7, outputs.decay, row)
            .write(8, outputs.proj, bytes(&[n, width + heads, 4])?))
    }

    /// The absorbed query of every head: `[wk_b[h] . q_nope | q_rope]`.
    pub(crate) fn mla_q_absorb(
        shape: MlaShape,
        qb: DeviceRange<'b>,
        wk_b: DeviceRange<'b>,
        qeff: DeviceRange<'b>,
    ) -> Result<Self, EngError> {
        let MlaShape {
            n,
            heads,
            nope,
            rope,
            rank,
        } = shape;
        Ok(Self::new(7, &[n, heads, nope, rope, rank], &[])?
            .read(0, qb, bytes(&[n, heads, nope + rope, 4])?)
            .read(1, wk_b, bytes(&[heads, rank, nope, 4])?)
            .write(2, qeff, bytes(&[n, heads, rank + rope, 4])?))
    }

    /// Normalizes the latent of `n` rows and appends latent plus rope part to the f16 cache at
    /// positions `pos0..pos0 + n`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn mla_kv_prep(
        n: usize,
        rank: usize,
        rope: usize,
        pos0: usize,
        cells: usize,
        eps: f32,
        kv: DeviceRange<'b>,
        norm_w: DeviceRange<'b>,
        cache: DeviceRange<'b>,
    ) -> Result<Self, EngError> {
        Ok(Self::new(8, &[n, rank, rope, pos0, cells], &[eps])?
            .read(0, kv, bytes(&[n, rank + rope, 4])?)
            .read(1, norm_w, bytes(&[rank, 4])?)
            .write(2, cache, bytes(&[cells, rank + rope, 2])?))
    }

    /// Normalizes the latent of one row and appends latent plus rope part to the f16 cache at the
    /// position stored in the device `pos` word (for graph-captured decode steps).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn mla_kv_prep_step(
        rank: usize,
        rope: usize,
        cells: usize,
        eps: f32,
        kv: DeviceRange<'b>,
        norm_w: DeviceRange<'b>,
        cache: DeviceRange<'b>,
        pos: DeviceRange<'b>,
    ) -> Result<Self, EngError> {
        if cells == 0 {
            return Err(EngError::BadRequest(
                "mla_kv_prep_step: cells must be positive".into(),
            ));
        }
        Ok(Self::new(19, &[rank, rope, cells], &[eps])?
            .read(0, kv, bytes(&[rank + rope, 4])?)
            .read(1, norm_w, bytes(&[rank, 4])?)
            .write(2, cache, bytes(&[cells, rank + rope, 2])?)
            .read(3, pos, 4))
    }

    /// Single-row shared-KV partial softmax attention across up to 64 heads per block on tensor
    /// cores (falling back to the scalar row kernel when the device lacks the shared-memory
    /// budget). Reads the current position from `pos_word` when given (for graph-captured decode
    /// steps), else from `pos`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn mla_attn_decode(
        shape: MlaShape,
        splits: usize,
        pos: usize,
        cells: usize,
        scale: f32,
        qeff: DeviceRange<'b>,
        cache: DeviceRange<'b>,
        part: DeviceRange<'b>,
        pos_word: Option<DeviceRange<'b>>,
    ) -> Result<Self, EngError> {
        let MlaShape {
            n,
            heads,
            rank,
            rope,
            ..
        } = shape;
        if n != 1
            || !(MLA_GROUP..=256).contains(&heads)
            || !heads.is_multiple_of(MLA_GROUP)
            || !(1..=64).contains(&splits)
            || cells == 0
            || (pos_word.is_none() && pos >= cells)
        {
            return Err(EngError::BadRequest(format!(
                "mla_attn_decode: invalid shape n={n}, heads={heads}, splits={splits}, pos={pos}, cells={cells}"
            )));
        }
        if !qeff.offset().is_multiple_of(8)
            || !cache.offset().is_multiple_of(16)
            || !part.offset().is_multiple_of(8)
        {
            return Err(EngError::BadRequest(
                "mla_attn_decode: qeff/part require 8-byte alignment and cache requires 16-byte"
                    .into(),
            ));
        }
        let part_bytes = bytes(&[heads / MLA_GROUP, splits, MLA_GROUP, rank + 4, 4])?;
        let mut op = Self::new(18, &[heads, rank, rope, splits, pos, cells], &[scale])?
            .read(0, qeff, bytes(&[heads, rank + rope, 4])?)
            .read(1, cache, bytes(&[cells, rank + rope, 2])?)
            .write(2, part, part_bytes);
        if let Some(pos_word) = pos_word {
            op = op.read(3, pos_word, 4);
        }
        Ok(op)
    }

    /// Partial softmax attention of `n` query rows over the cache, one partial per
    /// `(token, head group, split)`; `splits * keys_per_split` must cover `pos0 + n` keys.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn mla_attn(
        shape: MlaShape,
        pos0: usize,
        splits: usize,
        keys_per_split: usize,
        scale: f32,
        qeff: DeviceRange<'b>,
        cache: DeviceRange<'b>,
        part: DeviceRange<'b>,
    ) -> Result<Self, EngError> {
        Self::mla_score(
            9,
            shape,
            (pos0, splits, keys_per_split),
            scale,
            [qeff, cache, part],
        )
    }

    /// The same partials for a chunk of rows, scored on tensor cores in query tiles of 8 rows:
    /// the queries and the probabilities of a key tile are rounded to f16, so its bits differ
    /// from [`Self::mla_attn`]. A device without the kernel runs that one.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn mla_attn_chunk(
        shape: MlaShape,
        pos0: usize,
        splits: usize,
        keys_per_split: usize,
        scale: f32,
        qeff: DeviceRange<'b>,
        cache: DeviceRange<'b>,
        part: DeviceRange<'b>,
    ) -> Result<Self, EngError> {
        Self::mla_score(
            16,
            shape,
            (pos0, splits, keys_per_split),
            scale,
            [qeff, cache, part],
        )
    }

    /// The slots of both score operations: the queries, the cache and the partials.
    fn mla_score(
        op: i32,
        shape: MlaShape,
        (pos0, splits, keys_per_split): (usize, usize, usize),
        scale: f32,
        [qeff, cache, part]: [DeviceRange<'b>; 3],
    ) -> Result<Self, EngError> {
        let MlaShape {
            n,
            heads,
            rank,
            rope,
            ..
        } = shape;
        let part_bytes = bytes(&[n, heads / MLA_GROUP, splits, MLA_GROUP, rank + 4, 4])?;
        Ok(Self::new(
            op,
            &[n, heads, rank, rope, pos0, splits, keys_per_split],
            &[scale],
        )?
        .read(0, qeff, bytes(&[n, heads, rank + rope, 4])?)
        .read(1, cache, bytes(&[pos0 + n, rank + rope, 2])?)
        .write(2, part, part_bytes))
    }

    /// Combines the split partials into the attention mix of every head.
    pub(crate) fn mla_reduce(
        n: usize,
        heads: usize,
        rank: usize,
        splits: usize,
        part: DeviceRange<'b>,
        mix: DeviceRange<'b>,
    ) -> Result<Self, EngError> {
        let part_bytes = bytes(&[n, heads / MLA_GROUP, splits, MLA_GROUP, rank + 4, 4])?;
        Ok(Self::new(10, &[n, heads, rank, splits], &[])?
            .read(0, part, part_bytes)
            .write(1, mix, bytes(&[n, heads, rank, 4])?))
    }

    /// `out[t][h] = wv_b[h] . mix[t][h]`.
    pub(crate) fn mla_project(
        n: usize,
        heads: usize,
        rank: usize,
        v: usize,
        mix: DeviceRange<'b>,
        wv_b: DeviceRange<'b>,
        out: DeviceRange<'b>,
    ) -> Result<Self, EngError> {
        Ok(Self::new(11, &[n, heads, rank, v], &[])?
            .read(0, mix, bytes(&[n, heads, rank, 4])?)
            .read(1, wv_b, bytes(&[heads, v, rank, 4])?)
            .write(2, out, bytes(&[n, heads, v, 4])?))
    }

    /// The SiTU activation of gate and up rows, written as q8_1 rows, f32 rows, or both.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn situ_quant(
        rows: usize,
        w: usize,
        ld_gate: usize,
        ld_up: usize,
        (beta, linear_beta): (f32, f32),
        gate: DeviceRange<'b>,
        up: DeviceRange<'b>,
        outs: RowOutputs<'b>,
    ) -> Result<Self, EngError> {
        let span = |ld: usize| -> Result<usize, EngError> {
            Ok(bytes(&[rows.saturating_sub(1), ld, 4])?.saturating_add(w * 4))
        };
        let op = Self::new(12, &[rows, w, ld_gate, ld_up], &[beta, linear_beta])?
            .read(0, gate, span(ld_gate)?)
            .read(1, up, span(ld_up)?);
        outs.attach(op, rows, w)
    }

    /// `x * sigmoid(gate)` (or `x` alone), written as q8_1 rows, f32 rows, or both.
    pub(crate) fn gated_quant(
        rows: usize,
        w: usize,
        x: DeviceRange<'b>,
        gate: Option<DeviceRange<'b>>,
        outs: RowOutputs<'b>,
    ) -> Result<Self, EngError> {
        let mut op = Self::new(13, &[rows, w], &[])?.read(0, x, bytes(&[rows, w, 4])?);
        if let Some(gate) = gate {
            op = op.read(1, gate, bytes(&[rows, w, 4])?);
        }
        outs.attach(op, rows, w)
    }

    /// Fills `count` f32 values. No step uses it now; the device tests of the operations do.
    #[cfg(test)]
    pub(crate) fn fill(count: usize, value: f32, out: DeviceRange<'b>) -> Result<Self, EngError> {
        Ok(Self::new(15, &[count], &[value])?.write(0, out, bytes(&[count, 4])?))
    }
}

/// The outputs of the activation rows: q8_1 rows for the quantized matvec, f32 rows for the
/// batched GEMM, or both. At least one is present.
pub(crate) struct RowOutputs<'b> {
    pub(crate) q8: Option<DeviceRange<'b>>,
    pub(crate) f32: Option<DeviceRange<'b>>,
}

impl<'b> RowOutputs<'b> {
    fn attach(self, mut op: BlockOp<'b>, rows: usize, w: usize) -> Result<BlockOp<'b>, EngError> {
        if self.q8.is_none() && self.f32.is_none() {
            return Err(EngError::BadRequest(
                "a block operation needs a q8_1 or an f32 output".into(),
            ));
        }
        if let Some(q8) = self.q8 {
            op = op.write(2, q8, q8_rows(w, rows)?);
        }
        if let Some(y) = self.f32 {
            op = op.write(3, y, bytes(&[rows, w, 4])?);
        }
        Ok(op)
    }
}

/// Query heads per attention block of the latent attention.
pub(crate) const MLA_GROUP: usize = 8;

/// The geometry of the absorbed latent attention: `n` token rows, `heads` query heads,
/// per-head `nope` and `rope` widths of the query, and the latent `rank`.
#[derive(Clone, Copy, Debug)]
pub(crate) struct MlaShape {
    pub(crate) n: usize,
    pub(crate) heads: usize,
    pub(crate) nope: usize,
    pub(crate) rope: usize,
    pub(crate) rank: usize,
}

/// The ranges `rda_pack` writes for the shared delta scan; it reads `qkv` too.
pub(crate) struct RdaPacked<'b> {
    pub(crate) qkv: DeviceRange<'b>,
    pub(crate) decay: DeviceRange<'b>,
    pub(crate) proj: DeviceRange<'b>,
}

/// The ranges `rda_pack` only reads.
pub(crate) struct RdaInputs<'b> {
    pub(crate) g1: DeviceRange<'b>,
    pub(crate) dt_bias: DeviceRange<'b>,
    pub(crate) a: DeviceRange<'b>,
    pub(crate) beta: DeviceRange<'b>,
    pub(crate) delta: DeviceRange<'b>,
    pub(crate) g2: DeviceRange<'b>,
}

impl CudaContext {
    pub(crate) fn block_op(
        &self,
        stream: &DeviceStream,
        call: BlockOp<'_>,
    ) -> Result<(), EngError> {
        self.ensure_usable()?;
        let mut slots: [Option<SlotRaw<'_>>; PTRS] = std::array::from_fn(|_| None);
        for (index, range, need, written) in &call.slots {
            if slots[*index].is_some() {
                return Err(EngError::BadRequest(format!(
                    "the block operation fills slot {} twice",
                    SLOT_NAMES[*index]
                )));
            }
            require_fit(SLOT_NAMES[*index], range, *need, 4)?;
            slots[*index] = Some(SlotRaw {
                range: self.raw_range(range)?,
                need: *need,
                written: *written,
            });
        }
        for (index, range, need, written) in &call.slots {
            if !written {
                continue;
            }
            let mine = Claim::of(SLOT_NAMES[*index], range, *need);
            for (other_index, other, other_need, _) in &call.slots {
                if other_index != index {
                    require_apart(
                        &mine,
                        &Claim::of(SLOT_NAMES[*other_index], other, *other_need),
                    )?;
                }
            }
        }
        let raw = BlockOpsRaw {
            op: call.op,
            ints: call.ints,
            floats: call.floats,
            slots,
        };
        self.launch_on(stream, |stream_raw| block_ops_launch(stream_raw, &raw))
    }
}

#[cfg(all(test, feature = "native-cuda"))]
mod tests;
