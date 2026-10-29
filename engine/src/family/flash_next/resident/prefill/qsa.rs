//! Batched QSA attention over a chunk.
//!
//! `chunk` is the chunk form of `qsa::step`. The `T` rows of a chunk attend to the `P` cached
//! positions and to the chunk rows before them. The mask is causal and aligned to the last key, so
//! row `i` sees the positions `0..P + i`. The pinned MLX must align it that way when there are
//! more keys than queries; the test `grouped_attention_aligns_the_causal_mask_to_the_last_key` in
//! `tests/native_fast.rs` decides, and the chunk tests here repeat the check through the layer.
//!
//! The dense limit stays. A chunk whose last row passes it returns `ContextLimit` before any native
//! call. There is no sparse chunk yet.

use super::super::ops::{Ctx, dim};
use super::super::state::LayerState;
use super::super::weights::RQsa;
use crate::family::flash_next::qsa::QsaShape;
use crate::{
    EngError,
    kernels::{BinaryOp, RopeSpec, UnaryOp},
    tensor::ExecutionTensor,
};

/// One QSA chunk: the layer output rows and the replacement layer state.
pub(super) struct QsaChunk {
    /// Layer output `[T, H]`.
    pub(super) out: ExecutionTensor,
    /// `LayerState::Qsa` with `T` more positions. The caller replaces the old state with it.
    pub(super) state: LayerState,
}

fn overflow() -> EngError {
    EngError::BadRequest("resident qsa dimension overflows".into())
}

fn mismatch() -> EngError {
    EngError::BadRequest("resident qsa cache does not match its shape".into())
}

/// Checks, before any native call, that the cache holds `pos` positions and that the last row of a
/// chunk of `rows` rows still selects every indexer block, so dense attention equals the host
/// rule. The last row sees `pos + rows` positions. The rule of `qsa::check_dense` is monotone in
/// the position, so the last row decides for all rows.
pub(in super::super) fn check_chunk_dense(
    shape: &QsaShape,
    len: usize,
    pos: u32,
    rows: u32,
) -> Result<(), EngError> {
    let dims = shape.dims();
    if usize::try_from(pos).ok() != Some(len) {
        return Err(EngError::BadRequest(format!(
            "qsa cache length {len} does not match position {pos}"
        )));
    }
    let ratio = u64::from(dims.ratio);
    let blocks = u64::from(pos)
        .checked_add(u64::from(rows))
        .and_then(|visible| visible.checked_div(ratio));
    let limit = u64::from(dims.budget).checked_div(ratio);
    match (blocks, limit) {
        (Some(blocks), Some(limit)) if blocks <= limit => Ok(()),
        _ => Err(EngError::ContextLimit(
            "resident QSA chunk passes the dense limit; there is no sparse chunk".into(),
        )),
    }
}

/// The most rows that a chunk starting at `pos` can hold while `check_chunk_dense` passes: the
/// dense limit counts `(budget / ratio + 1) * ratio - 1` visible positions. It is zero at or past
/// the limit. The chunk step uses it to split a prompt that crosses the dense limit.
pub(in super::super) fn dense_rows(shape: &QsaShape, pos: u32) -> Result<u32, EngError> {
    let dims = shape.dims();
    let ratio = u64::from(dims.ratio);
    let blocks = u64::from(dims.budget)
        .checked_div(ratio)
        .and_then(|limit| limit.checked_add(1))
        .ok_or_else(overflow)?;
    let visible = blocks
        .checked_mul(ratio)
        .and_then(|n| n.checked_sub(1))
        .ok_or_else(overflow)?;
    let room = visible.saturating_sub(u64::from(pos));
    u32::try_from(room).map_err(|_| overflow())
}

/// Runs one dense QSA chunk of `x` `[T, H]` at position `pos`. It builds a lazy graph only: no
/// readback and no evaluate. `state` is never changed; on error nothing is returned.
pub(super) fn chunk(
    ctx: &mut Ctx<'_>,
    shape: &QsaShape,
    w: &RQsa,
    state: &LayerState,
    pos: u32,
    x: &ExecutionTensor,
) -> Result<QsaChunk, EngError> {
    let LayerState::Qsa { k, v, ik, len } = state else {
        return Err(EngError::BadRequest(
            "qsa chunk needs a QSA layer state".into(),
        ));
    };
    let &[rows, _] = x.shape() else {
        return Err(EngError::BadRequest(
            "qsa chunk input must be [T, H]".into(),
        ));
    };
    if rows == 0 {
        return Err(EngError::BadRequest("qsa chunk needs a row".into()));
    }
    check_chunk_dense(shape, *len, pos, rows)?;
    let d = shape.dims();
    let (heads, kv_heads, width) = (d.heads_q, d.heads_kv, d.head_dim);
    let cached = dim(*len)?;
    let cache = match (k, v, ik) {
        (None, None, None) if *len == 0 => None,
        (Some(k), Some(v), Some(ik)) if *len > 0 => {
            let kv_shape = [1, kv_heads, cached, width];
            let ik_shape = [cached, d.idx_dim];
            if k.shape() != kv_shape.as_slice()
                || v.shape() != kv_shape.as_slice()
                || ik.shape() != ik_shape.as_slice()
            {
                return Err(mismatch());
            }
            Some((k, v, ik))
        }
        _ => return Err(mismatch()),
    };
    let pair = width.checked_mul(2).ok_or_else(overflow)?;
    let idx_start = d.idx_heads.checked_mul(d.idx_dim).ok_or_else(overflow)?;
    let idx_stop = idx_start.checked_add(d.idx_dim).ok_or_else(overflow)?;
    let q_width = heads.checked_mul(width).ok_or_else(overflow)?;
    let next_len = usize::try_from(rows)
        .ok()
        .and_then(|rows| len.checked_add(rows))
        .ok_or_else(overflow)?;
    let rope = RopeSpec {
        dims: d.rotary_dim,
        base: d.theta,
        scale: 1.0,
        offset: pos,
    };

    // 1. Query and gate, interleaved per head: head h is row[h * 2D..(h + 1) * 2D] of each row.
    let qp = ctx.kernels.qmm(x, &w.q)?;
    let qp = ctx.kernels.reshape(&qp, &[rows, heads, pair])?;
    let q_raw = ctx.kernels.slice(&qp, &[0, 0, 0], &[rows, heads, width])?;
    let gate = ctx
        .kernels
        .slice(&qp, &[0, 0, width], &[rows, heads, pair])?;
    // 2. Query norm with `1 + w`, time on axis -2, then RoPE from position `pos`.
    let q = ctx.kernels.rms_norm(&q_raw, &w.q_norm1p, d.eps)?;
    let q = ctx.kernels.transpose(&q, &[1, 0, 2])?;
    let q = ctx.kernels.reshape(&q, &[1, heads, rows, width])?;
    let q = ctx.kernels.rope_scaled(&q, &rope)?;
    // 3. New keys: norm and RoPE per KV head.
    let kp = ctx.kernels.qmm(x, &w.k)?;
    let kp = ctx.kernels.reshape(&kp, &[rows, kv_heads, width])?;
    let k_new = ctx.kernels.rms_norm(&kp, &w.k_norm1p, d.eps)?;
    let k_new = ctx.kernels.transpose(&k_new, &[1, 0, 2])?;
    let k_new = ctx.kernels.reshape(&k_new, &[1, kv_heads, rows, width])?;
    let k_new = ctx.kernels.rope_scaled(&k_new, &rope)?;
    // 4. New values, and the raw indexer keys: the last `ID` projection values of each row.
    let vp = ctx.kernels.qmm(x, &w.v)?;
    let vp = ctx.kernels.reshape(&vp, &[rows, kv_heads, width])?;
    let v_new = ctx.kernels.transpose(&vp, &[1, 0, 2])?;
    let v_new = ctx.kernels.reshape(&v_new, &[1, kv_heads, rows, width])?;
    let ip = ctx.kernels.qmm(x, &w.idx)?;
    let ik_new = ctx.kernels.slice(&ip, &[0, idx_start], &[rows, idx_stop])?;
    // 5. Append on the time axis, or start the cache.
    let (k_all, v_all, ik_all) = match cache {
        None => (k_new, v_new, ik_new),
        Some((k, v, ik)) => (
            ctx.kernels.concatenate(&[k, &k_new], 2)?,
            ctx.kernels.concatenate(&[v, &v_new], 2)?,
            ctx.kernels.concatenate(&[ik, &ik_new], 0)?,
        ),
    };
    // 6. Causal attention over all positions; head h reads KV head h / (QH / KVH).
    let scale = 1.0 / (width as f32).sqrt();
    let att = ctx
        .kernels
        .grouped_attention(&q, &k_all, &v_all, scale, true)?;
    // 7. Back to rows, sigmoid output gate, and output projection.
    let att = ctx.kernels.transpose(&att, &[0, 2, 1, 3])?;
    let att = ctx.kernels.reshape(&att, &[rows, heads, width])?;
    let gate = ctx.kernels.unary(UnaryOp::Sigmoid, &gate)?;
    let o = ctx.kernels.binary(BinaryOp::Mul, &att, &gate)?;
    let o = ctx.kernels.reshape(&o, &[rows, q_width])?;
    let out = ctx.kernels.qmm(&o, &w.o)?;
    Ok(QsaChunk {
        out,
        state: LayerState::Qsa {
            k: Some(k_all),
            v: Some(v_all),
            ik: Some(ik_all),
            len: next_len,
        },
    })
}

#[cfg(test)]
mod tests {
    use super::super::super::fixtures;
    use super::super::super::ops::{f32_tensor, values};
    use super::super::super::qsa::{QsaStep, step as decode_step};
    use super::super::super::weights::{RAttn, ResidentWeights};
    use super::super::assert_parity;
    use super::*;
    use crate::family::flash_next::load::FlashWeights;
    use crate::kernels::MlxKernels;

    /// Input of position `pos`: `x[i] = ((i + 5 pos) % 11 - 5) / 16`, exact in f32 (the decode test).
    fn input(pos: u32) -> Vec<f32> {
        (0..64u32)
            .map(|i| (((i + 5 * pos) % 11) as f32 - 5.0) * 0.0625)
            .collect()
    }

    /// An adapter, the fixture weights, and the QSA shape (KVH 1, D 64, ID 32, ratio 4, budget 16).
    struct Rig {
        kernels: MlxKernels,
        weights: ResidentWeights,
        shape: QsaShape,
    }

    fn rig() -> Rig {
        let mut kernels = MlxKernels::try_new().unwrap();
        let ckpt = fixtures::checkpoint();
        let flash = FlashWeights::load(&ckpt, &fixtures::config()).unwrap();
        let weights = ResidentWeights::import(&mut kernels, &flash).unwrap();
        Rig {
            kernels,
            weights,
            shape: flash.shape().qsa,
        }
    }

    fn qsa_weights(weights: &ResidentWeights) -> &RQsa {
        let RAttn::Qsa(w) = &weights.layers[1].attn else {
            panic!("resident layer 1 is not QSA");
        };
        w
    }

    fn empty() -> LayerState {
        LayerState::Qsa {
            k: None,
            v: None,
            ik: None,
            len: 0,
        }
    }

    /// One decode step at `pos` on the fixture input.
    fn decode_one(rig: &mut Rig, state: &LayerState, pos: u32) -> QsaStep {
        let x = rig
            .kernels
            .import_tensor(&f32_tensor(vec![1, 64], &input(pos)).unwrap())
            .unwrap();
        let mut ctx = Ctx {
            kernels: &mut rig.kernels,
            consts: &rig.weights.consts,
        };
        decode_step(
            &mut ctx,
            &rig.shape,
            qsa_weights(&rig.weights),
            state,
            pos,
            &x,
        )
        .unwrap()
    }

    /// One chunk of `rows` rows from position `pos` on the fixture inputs.
    fn chunk_one(rig: &mut Rig, state: &LayerState, pos: u32, rows: u32) -> QsaChunk {
        let flat: Vec<f32> = (pos..pos + rows).flat_map(input).collect();
        let x = rig
            .kernels
            .import_tensor(&f32_tensor(vec![rows, 64], &flat).unwrap())
            .unwrap();
        let mut ctx = Ctx {
            kernels: &mut rig.kernels,
            consts: &rig.weights.consts,
        };
        chunk(
            &mut ctx,
            &rig.shape,
            qsa_weights(&rig.weights),
            state,
            pos,
            &x,
        )
        .unwrap()
    }

    /// Keys, values, and indexer keys of a QSA state as host values.
    fn cache_values(rig: &mut Rig, state: &LayerState) -> [Vec<f32>; 3] {
        let LayerState::Qsa {
            k: Some(k),
            v: Some(v),
            ik: Some(ik),
            ..
        } = state
        else {
            panic!("the state has no cache");
        };
        [k, v, ik].map(|t| values(&mut rig.kernels, t).unwrap())
    }

    /// A chunk of `rows` rows after `pos` cached positions equals `rows` decode steps.
    fn assert_chunk_matches_steps(pos: u32, rows: u32) {
        let mut rig = rig();
        let mut reference = empty();
        let mut base = empty();
        let mut want_rows: Vec<f32> = Vec::new();
        for at in 0..pos + rows {
            let next = decode_one(&mut rig, &reference, at);
            if at >= pos {
                want_rows.extend(values(&mut rig.kernels, &next.out).unwrap());
            }
            reference = next.state;
            if at < pos {
                base = decode_one(&mut rig, &base, at).state;
            }
        }
        let got = chunk_one(&mut rig, &base, pos, rows);
        let label = format!("P {pos} T {rows}");
        assert_parity(
            &values(&mut rig.kernels, &got.out).unwrap(),
            &want_rows,
            &format!("{label} output rows"),
        );
        let [gk, gv, gi] = cache_values(&mut rig, &got.state);
        let [wk, wv, wi] = cache_values(&mut rig, &reference);
        assert_parity(&gk, &wk, &format!("{label} keys"));
        assert_parity(&gv, &wv, &format!("{label} values"));
        assert_parity(&gi, &wi, &format!("{label} indexer keys"));
        let LayerState::Qsa { len, .. } = got.state else {
            panic!("not a QSA state");
        };
        assert_eq!(len, (pos + rows) as usize, "{label} length");
    }

    #[test]
    fn chunks_from_an_empty_cache_match_the_decode_steps() {
        let _serial = crate::native_test_lock();
        for rows in [1, 2, 3, 4, 7] {
            assert_chunk_matches_steps(0, rows);
        }
    }

    /// With 5 cached positions the rows of the chunk use rope positions 5 to 4 + T and see the
    /// cache. A top-left causal mask or a rope offset of 0 changes every row after the first.
    #[test]
    fn chunks_after_five_cached_positions_match_the_decode_steps() {
        let _serial = crate::native_test_lock();
        for rows in [1, 2, 3, 4, 7] {
            assert_chunk_matches_steps(5, rows);
        }
    }

    /// The room is `19 - pos` visible positions of the fixture, never negative, and it agrees with
    /// `check_chunk_dense` at the edge: the room passes and one more row is a `ContextLimit`.
    #[test]
    fn the_dense_room_ends_at_the_last_dense_position() {
        let ckpt = fixtures::checkpoint();
        let flash = FlashWeights::load(&ckpt, &fixtures::config()).unwrap();
        let shape = flash.shape().qsa;
        for (pos, room) in [(0u32, 19u32), (5, 14), (10, 9), (18, 1), (19, 0), (25, 0)] {
            assert_eq!(dense_rows(&shape, pos).unwrap(), room, "room at {pos}");
            if pos <= 19 {
                let len = pos as usize;
                assert!(
                    check_chunk_dense(&shape, len, pos, room).is_ok(),
                    "the room fits at {pos}"
                );
                assert!(
                    matches!(
                        check_chunk_dense(&shape, len, pos, room + 1),
                        Err(EngError::ContextLimit(_))
                    ),
                    "one more row passes the limit at {pos}"
                );
            }
        }
    }

    /// Ratio 4 and budget 16: the last row at visible position `v` is dense while `v / 4 <= 4`,
    /// that is `v <= 19`.
    #[test]
    fn the_chunk_dense_limit_counts_the_last_row() {
        let ckpt = fixtures::checkpoint();
        let flash = FlashWeights::load(&ckpt, &fixtures::config()).unwrap();
        let shape = flash.shape().qsa;
        assert!(check_chunk_dense(&shape, 10, 10, 9).is_ok(), "visible 19");
        assert!(
            matches!(
                check_chunk_dense(&shape, 10, 10, 10),
                Err(EngError::ContextLimit(_))
            ),
            "visible 20"
        );
        assert!(
            check_chunk_dense(&shape, 0, 0, 19).is_ok(),
            "one chunk to 19"
        );
        assert!(
            matches!(
                check_chunk_dense(&shape, 0, 0, 20),
                Err(EngError::ContextLimit(_))
            ),
            "one chunk to 20"
        );
        assert!(
            matches!(
                check_chunk_dense(&shape, 9, 10, 1),
                Err(EngError::BadRequest(_))
            ),
            "cache length differs from the position"
        );
        assert!(
            matches!(
                check_chunk_dense(&shape, u32::MAX as usize, u32::MAX, 1),
                Err(EngError::ContextLimit(_))
            ),
            "position overflow"
        );
    }
}
