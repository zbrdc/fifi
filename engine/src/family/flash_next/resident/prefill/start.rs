//! Chunk token-start import (plan S4 section 3.1).
//!
//! Hashing and row reads stay on the host. The rows of all tokens of a chunk import packed with one
//! synchronization and dequantize on the device. This is the chunk form of `import_token` in the
//! parent module.

use super::super::{
    ops::{Ctx, dim, short_import},
    ple::table_view,
    weights::ResidentWeights,
};
use crate::{
    EngError,
    family::flash_next::{ple, stored},
    kernels::{BinaryOp, PackedWeight},
    tensor::ExecutionTensor,
    weights::StreamedCheckpoint,
};

/// Device inputs of a chunk and the PLE history after it.
pub(super) struct ChunkStart {
    /// Hyper state `[T, C, H]`: each token's embedding row repeated over the branches.
    pub(super) hyper: ExecutionTensor,
    /// Scaled PLE embedding `[T, heads * head_dim]`.
    pub(super) emb: ExecutionTensor,
    /// The n-gram history after the last token; the step commits it.
    pub(super) history: Vec<u32>,
}

/// Plans and reads the embedding rows and the PLE rows of `tokens`, then imports them with one
/// synchronization. `history` is the n-gram history before the first token.
pub(super) fn import_tokens(
    ctx: &mut Ctx<'_>,
    ckpt: &StreamedCheckpoint,
    weights: &ResidentWeights,
    history: &[u32],
    tokens: &[u32],
) -> Result<ChunkStart, EngError> {
    let plan = ple::plan_chunk_rows(
        &weights.shape.ple,
        &table_view(&weights.ple)?,
        history,
        tokens,
    )?;
    let per_token = plan.rows.first().map_or(0, Vec::len) + 1;
    let mut host: Vec<PackedWeight> = Vec::new();
    for (&token, rows) in tokens.iter().zip(&plan.rows) {
        host.push(stored::read_slice(ckpt, &weights.embed, token)?);
        for &(table, local) in rows {
            let shard = weights
                .ple
                .shards
                .get(table)
                .ok_or_else(|| EngError::Checkpoint("ple row table is missing".into()))?;
            host.push(stored::read_slice(ckpt, shard, local)?);
        }
    }
    let refs: Vec<&PackedWeight> = host.iter().collect();
    let residents = ctx.kernels.import_packed_many(&refs)?;
    if residents.len() != host.len() {
        return Err(short_import());
    }
    let consts = ctx.consts;
    let k = &mut *ctx.kernels;
    let mut embed_rows = Vec::with_capacity(tokens.len());
    let mut ple_rows = Vec::with_capacity(tokens.len());
    for group in residents.chunks(per_token) {
        let (embed, rest) = group.split_first().ok_or_else(short_import)?;
        embed_rows.push(k.dequantize(embed)?);
        let parts = rest
            .iter()
            .map(|part| k.dequantize(part))
            .collect::<Result<Vec<_>, _>>()?;
        let part_refs: Vec<&ExecutionTensor> = parts.iter().collect();
        ple_rows.push(k.concatenate(&part_refs, 1)?);
    }
    let embed_refs: Vec<&ExecutionTensor> = embed_rows.iter().collect();
    let embed_all = k.concatenate(&embed_refs, 0)?;
    let branches = dim(weights.shape.hyper.branches())?;
    let spread = k.expand_dims(&embed_all, 1)?;
    let hyper = k.repeat(&spread, branches, 1)?;
    let ple_refs: Vec<&ExecutionTensor> = ple_rows.iter().collect();
    let joined = k.concatenate(&ple_refs, 0)?;
    let emb = k.binary(BinaryOp::Mul, &joined, &consts.ple_scale)?;
    Ok(ChunkStart {
        hyper,
        emb,
        history: plan.history,
    })
}

#[cfg(test)]
mod tests {
    use super::super::super::fixtures;
    use super::super::super::ops::{assert_close, values};
    use super::super::super::ple::import_token;
    use super::super::assert_parity;
    use super::*;
    use crate::family::flash_next::{load::FlashWeights, mlx::MlxProjector};
    use crate::family::project::Projector;
    use crate::kernels::MlxKernels;

    #[test]
    fn import_tokens_match_the_host_rows_and_history() {
        let _serial = crate::native_test_lock();
        let mut kernels = MlxKernels::try_new().unwrap();
        let ckpt = fixtures::checkpoint();
        let flash = FlashWeights::load(&ckpt, &fixtures::config()).unwrap();
        let weights = ResidentWeights::import(&mut kernels, &flash).unwrap();
        let view = flash.view();
        let shape = *flash.shape();
        let mut proj = MlxProjector::new(&ckpt).unwrap();
        // Token 7 is the end-of-sequence token of the fixture: it clears the history.
        let tokens = [3u32, 5, 7, 2];
        let mut want_hyper = Vec::new();
        let mut want_emb = Vec::new();
        let mut history: Vec<u32> = Vec::new();
        for &token in &tokens {
            let plan = ple::plan_rows(&shape.ple, &view.ple, &history, token).unwrap();
            let row = proj.rows(view.embed, &[token]).unwrap();
            want_hyper.extend(row.iter().chain(&row).copied());
            for &(table, local) in &plan.rows {
                let r = proj.rows(&view.ple.shards[table], &[local]).unwrap();
                want_emb.extend(r.iter().map(|v| v * view.ple.weight_scale));
            }
            history = plan.history;
        }
        assert_ne!(
            want_hyper[..128],
            want_hyper[128..256],
            "tokens 3 and 5 have different rows"
        );
        let mut ctx = Ctx {
            kernels: &mut kernels,
            consts: &weights.consts,
        };

        let start = import_tokens(&mut ctx, &ckpt, &weights, &[], &tokens).unwrap();

        assert_eq!(start.hyper.shape(), &[4, 2, 64][..], "hyper shape");
        assert_eq!(start.emb.shape(), &[4, 64][..], "emb shape");
        assert_eq!(start.history, history, "history equals the host chain");
        assert_eq!(
            start.history,
            [2],
            "history after the end token and token 2"
        );
        let hyper = values(&mut kernels, &start.hyper).unwrap();
        assert_parity(&hyper, &want_hyper, "hyper rows");
        let emb = values(&mut kernels, &start.emb).unwrap();
        assert_parity(&emb, &want_emb, "emb rows");
    }

    /// The fixture rows are known by hand. Embedding row `r` holds `c % 8 + BIASES[r % 4]` at
    /// element `c`, with `BIASES = [0, 0.25, 0.5, 0.75]`. The PLE table has multipliers 3, 5, 7,
    /// vocabulary 8 and offsets 0 and 8 for the two heads, scale 0.5, and end token 7. The hash of a
    /// head is `(3 * token ^ 5 * p1 [^ 7 * p2]) % 8` plus the offset, where the missing history is 7
    /// and only head 1 uses `p2`:
    /// token 3 after [] reads rows 2 and 11, token 5 after [3] reads 0 and 9, token 7 after [3, 5]
    /// reads 4 and 9, and token 2 after [] reads 5 and 12 (token 7 cleared the history).
    /// Row biases are `BIASES[row % 4]`. Output element `j` of a PLE row is
    /// `0.5 * (j % 8 + bias)` for head `j / 32`.
    #[test]
    fn import_tokens_match_the_hand_derived_rows() {
        let _serial = crate::native_test_lock();
        let mut kernels = MlxKernels::try_new().unwrap();
        let ckpt = fixtures::checkpoint();
        let flash = FlashWeights::load(&ckpt, &fixtures::config()).unwrap();
        let weights = ResidentWeights::import(&mut kernels, &flash).unwrap();
        let embed_bias = [0.75f32, 0.25, 0.75, 0.5];
        let ple_bias = [[0.5f32, 0.75], [0.0, 0.25], [0.0, 0.25], [0.25, 0.0]];
        let mut want_hyper = Vec::new();
        let mut want_emb = Vec::new();
        for (bias, heads) in embed_bias.iter().zip(&ple_bias) {
            for _branch in 0..2 {
                want_hyper.extend((0..64).map(|c| (c % 8) as f32 + bias));
            }
            want_emb.extend((0..64).map(|j| 0.5 * ((j % 8) as f32 + heads[j / 32])));
        }
        let mut ctx = Ctx {
            kernels: &mut kernels,
            consts: &weights.consts,
        };

        let start = import_tokens(&mut ctx, &ckpt, &weights, &[], &[3, 5, 7, 2]).unwrap();

        assert_eq!(start.hyper.shape(), &[4, 2, 64][..], "hyper shape");
        assert_eq!(start.emb.shape(), &[4, 64][..], "emb shape");
        assert_close(
            &values(&mut kernels, &start.hyper).unwrap(),
            &want_hyper,
            "hyper",
        );
        assert_close(&values(&mut kernels, &start.emb).unwrap(), &want_emb, "emb");
        assert_eq!(start.history, [2], "history");
    }

    #[test]
    fn a_chunk_of_one_token_equals_the_decode_import() {
        let _serial = crate::native_test_lock();
        let mut kernels = MlxKernels::try_new().unwrap();
        let ckpt = fixtures::checkpoint();
        let flash = FlashWeights::load(&ckpt, &fixtures::config()).unwrap();
        let weights = ResidentWeights::import(&mut kernels, &flash).unwrap();
        let mut ctx = Ctx {
            kernels: &mut kernels,
            consts: &weights.consts,
        };
        let one = import_token(&mut ctx, &ckpt, &weights, &[3, 5], 2).unwrap();
        let chunk = import_tokens(&mut ctx, &ckpt, &weights, &[3, 5], &[2]).unwrap();

        assert_eq!(chunk.hyper.shape(), &[1, 2, 64][..], "hyper shape");
        assert_eq!(chunk.emb.shape(), &[1, 64][..], "emb shape");
        assert_eq!(chunk.history, one.history, "history");
        let want_hyper = values(&mut kernels, &one.hyper).unwrap();
        let got_hyper = values(&mut kernels, &chunk.hyper).unwrap();
        assert_parity(&got_hyper, &want_hyper, "hyper");
        let want_emb = values(&mut kernels, &one.emb).unwrap();
        let got_emb = values(&mut kernels, &chunk.emb).unwrap();
        assert_parity(&got_emb, &want_emb, "emb");
    }

    #[test]
    fn a_chunk_with_an_out_of_range_token_is_rejected_before_any_native_call() {
        let _serial = crate::native_test_lock();
        let mut kernels = MlxKernels::try_new().unwrap();
        let ckpt = fixtures::checkpoint();
        let flash = FlashWeights::load(&ckpt, &fixtures::config()).unwrap();
        let weights = ResidentWeights::import(&mut kernels, &flash).unwrap();
        #[cfg(feature = "native-fault-qualification")]
        let before = kernels.qualification_sync_count();
        let mut ctx = Ctx {
            kernels: &mut kernels,
            consts: &weights.consts,
        };

        // The vocabulary has 8 tokens, so token 8 is the first id outside it.
        let err = import_tokens(&mut ctx, &ckpt, &weights, &[], &[3, 8]).err();

        assert!(
            matches!(&err, Some(EngError::BadRequest(message)) if message.contains("slice 8 is outside")),
            "out-of-range token error: {err:?}"
        );
        #[cfg(feature = "native-fault-qualification")]
        assert_eq!(
            kernels.qualification_sync_count(),
            before,
            "no synchronization before the rejection"
        );
        let mut ctx = Ctx {
            kernels: &mut kernels,
            consts: &weights.consts,
        };
        let retry = import_tokens(&mut ctx, &ckpt, &weights, &[], &[3, 5]).unwrap();
        assert_eq!(
            retry.history,
            [3, 5],
            "the adapter imports a valid chunk afterward"
        );
        assert_eq!(retry.hyper.shape(), &[2, 2, 64][..], "retry hyper shape");
    }

    #[test]
    fn an_empty_chunk_is_rejected_before_any_native_call() {
        let _serial = crate::native_test_lock();
        let mut kernels = MlxKernels::try_new().unwrap();
        let ckpt = fixtures::checkpoint();
        let flash = FlashWeights::load(&ckpt, &fixtures::config()).unwrap();
        let weights = ResidentWeights::import(&mut kernels, &flash).unwrap();
        #[cfg(feature = "native-fault-qualification")]
        let before = kernels.qualification_sync_count();
        let mut ctx = Ctx {
            kernels: &mut kernels,
            consts: &weights.consts,
        };

        let err = import_tokens(&mut ctx, &ckpt, &weights, &[3], &[]).err();

        assert!(
            matches!(&err, Some(EngError::BadRequest(message)) if message.contains("at least one token")),
            "empty chunk error: {err:?}"
        );
        #[cfg(feature = "native-fault-qualification")]
        assert_eq!(
            kernels.qualification_sync_count(),
            before,
            "no synchronization before the rejection"
        );
        let mut ctx = Ctx {
            kernels: &mut kernels,
            consts: &weights.consts,
        };
        let retry = import_tokens(&mut ctx, &ckpt, &weights, &[3], &[5]).unwrap();
        assert_eq!(
            retry.history,
            [3, 5],
            "the adapter imports a valid chunk afterward"
        );
    }

    #[cfg(feature = "native-fault-qualification")]
    #[test]
    fn a_chunk_takes_one_synchronization() {
        let _serial = crate::native_test_lock();
        let mut kernels = MlxKernels::try_new().unwrap();
        let ckpt = fixtures::checkpoint();
        let flash = FlashWeights::load(&ckpt, &fixtures::config()).unwrap();
        let weights = ResidentWeights::import(&mut kernels, &flash).unwrap();
        let before = kernels.qualification_sync_count();
        let mut ctx = Ctx {
            kernels: &mut kernels,
            consts: &weights.consts,
        };

        let start = import_tokens(&mut ctx, &ckpt, &weights, &[], &[3, 5, 7, 2]).unwrap();

        assert_eq!(
            kernels.qualification_sync_count(),
            before + 1,
            "chunk import synchronizations"
        );
        assert_eq!(start.emb.shape(), &[4, 64][..], "emb shape");
    }
}
