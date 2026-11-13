//! Header-only inspection of a Qwen3.8 Flash-Next EXL3 pack against the model's config.
//!
//! The stock pack keeps the HF names (`model.language_model.layers.N...`, `lm_head`) and stores
//! every routed expert as its own matrix. Quantized: `lm_head`, the DeltaNet `in_proj_qkv`,
//! `in_proj_z` and `out_proj`, the QSA `q_proj`, `k_proj`, `v_proj`, `o_proj` and indexer
//! `index_qk_proj`, and the routed and shared experts. Plain: the embedding, every
//! hyper-connection weight, `in_proj_a` and `in_proj_b`, the router and the shared-expert gate,
//! the PLE projections, and the norms and DeltaNet vectors. The pack has no final norm: the
//! mixer feeds `lm_head`. The n-gram table is in `ngram_embedding.safetensors`, outside the
//! index, and `weights::exl3::ngram` checks it; the MTP head is not walked.
//!
//! Device bytes count each tensor as the CUDA kernels read it: trellises as stored with `suh`
//! and `svh` in F32 (the EXL3 kernels), hyper-connection weights in their stored half dtype
//! (`kernels/cuda/resident/hyper.rs`), the router and the shared-expert gate in F32
//! (`kernels/cuda/resident/router.rs`), and the norms and DeltaNet vectors in F32. `in_proj_a` and
//! `in_proj_b` are F32 on the device as well: the gates are plain F32 projections, and the
//! Qwen3.6 step reads its own beta and alpha the same way (widened at upload, exact). The
//! embedding and the PLE projections count as stored; the PLE norms and convolution, which the host
//! PLE step reads, count as zero. The embedding's dtype is open (any half
//! dtype or F32) until the step's row gather fixes one; the real pack stores BF16.
//! [`inspect_flash_pack_listing`] also returns every visited tensor with the bytes counted for
//! it, which the CUDA trunk loader (`flash_next::cuda::load`) takes as its tensor list.

use super::Dims;
use crate::{
    EngError,
    family::exl3_pack::{EMBED_NAME, HEAD_PREFIX, Inspector, MoeBlock, PackReport, Root, Visit},
    tensor::Dtype,
    weights::StreamedCheckpoint,
};

/// The hyper-connection mixer that feeds `lm_head`: a read without an inject projection.
const MIXER: &str = "model.language_model.hyper_connection_mixer";

#[derive(Debug, Clone, PartialEq, Eq)]
pub(in crate::family) struct FlashPackReport {
    /// The shared walk report; its `expert_bytes` field is the part that expert tiers place.
    pub(in crate::family) pack: PackReport,
}

fn refuse(detail: String) -> EngError {
    EngError::Checkpoint(format!("EXL3 pack: {detail}"))
}

/// Walks the headers of the tensors the Flash-Next step reads from `source` against the model
/// in `config` (`config.json` bytes). Reads no payload.
pub(in crate::family) fn inspect_flash_pack(
    source: &StreamedCheckpoint,
    config: &[u8],
) -> Result<FlashPackReport, EngError> {
    walk_flash_pack(source, config, false).map(|(report, _)| report)
}

/// [`inspect_flash_pack`] and the list of every tensor it visited, in walk order, each with the
/// device bytes counted for it. The CUDA trunk loader reads its tensor list from here.
pub(in crate::family) fn inspect_flash_pack_listing(
    source: &StreamedCheckpoint,
    config: &[u8],
) -> Result<(FlashPackReport, Vec<Visit>), EngError> {
    walk_flash_pack(source, config, true)
}

fn walk_flash_pack(
    source: &StreamedCheckpoint,
    config: &[u8],
    record: bool,
) -> Result<(FlashPackReport, Vec<Visit>), EngError> {
    let dims = Dims::read(config)?;
    let mut walk = Inspector::new(source, Root::Layers);
    if record {
        walk = walk.recording();
    }
    let (h, vocab) = (dims.hidden as usize, dims.vocab as usize);
    walk.plain(EMBED_NAME, &[dims.vocab, dims.hidden], true)?;
    walk.matrix(HEAD_PREFIX, h, vocab)?;
    hyper(&mut walk, MIXER, &dims, false)?;
    let block = MoeBlock {
        hidden: h,
        experts: dims.experts as usize,
        inter: dims.moe_inter as usize,
        shared_inter: dims.shared_inter as usize,
    };
    for (layer, &qsa_layer) in dims.kinds.iter().enumerate() {
        let at = |suffix: &str| Root::Layers.name(layer, suffix);
        hyper(&mut walk, &at("attn_hyper_connection"), &dims, true)?;
        if qsa_layer {
            qsa(&mut walk, &at("self_attn"), &dims)?;
        } else {
            delta(&mut walk, &at("linear_attn"), &dims)?;
        }
        if layer == dims.ple_layer {
            ple(&mut walk, &at("ple"), &dims)?;
        }
        hyper(&mut walk, &at("mlp_hyper_connection"), &dims, true)?;
        walk.moe_block(layer, block)?;
    }
    let (pack, visits) = walk.report_with_visits()?;
    Ok((FlashPackReport { pack }, visits))
}

/// One hyper-connection site under `prefix`: every weight in one half dtype, which the kernel
/// takes as stored.
fn hyper(
    walk: &mut Inspector<'_>,
    prefix: &str,
    dims: &Dims,
    inject: bool,
) -> Result<(), EngError> {
    let mut tensors = vec![
        ("hc_norm.weight", vec![dims.width]),
        ("input_mix_weight_down.weight", vec![dims.rank, dims.width]),
        ("input_mix_weight_up.weight", vec![dims.width, dims.rank]),
    ];
    if inject {
        tensors.push((
            "block_inject_weight.weight",
            vec![dims.branches, dims.width],
        ));
    }
    let mut site = None;
    for (suffix, shape) in tensors {
        let name = format!("{prefix}.{suffix}");
        let dtype = walk.plain(&name, &shape, true)?;
        if !matches!(dtype, Dtype::F16 | Dtype::Bf16) {
            return Err(refuse(format!(
                "{name} is {dtype:?}, the kernel needs F16 or Bf16"
            )));
        }
        if *site.get_or_insert(dtype) != dtype {
            return Err(refuse(format!(
                "{name} is {dtype:?}, the rest of {prefix} is not"
            )));
        }
    }
    Ok(())
}

/// One Gated DeltaNet layer under `prefix` (`...linear_attn`).
fn delta(walk: &mut Inspector<'_>, prefix: &str, dims: &Dims) -> Result<(), EngError> {
    let h = dims.hidden as usize;
    let key_width = dims.key_heads as usize * dims.key_dim as usize;
    let value_width = dims.value_heads as usize * dims.value_dim as usize;
    let conv_dim = 2 * key_width + value_width;
    for (role, k, n) in [
        ("in_proj_qkv", h, conv_dim),
        ("in_proj_z", h, value_width),
        ("out_proj", value_width, h),
    ] {
        walk.matrix(&format!("{prefix}.{role}"), k, n)?;
    }
    for role in ["in_proj_a", "in_proj_b"] {
        let name = format!("{prefix}.{role}.weight");
        walk.plain(&name, &[dims.value_heads, dims.hidden], false)?;
    }
    let conv = u32::try_from(conv_dim).map_err(|_| refuse(format!("{prefix}: conv width")))?;
    for (role, shape) in [
        ("A_log", vec![dims.value_heads]),
        ("dt_bias", vec![dims.value_heads]),
        ("conv1d.weight", vec![conv, 1, dims.conv_kernel]),
        ("norm.weight", vec![dims.value_dim]),
    ] {
        walk.plain(&format!("{prefix}.{role}"), &shape, false)?;
    }
    Ok(())
}

/// One QSA attention layer under `prefix` (`...self_attn`): the gated query projection (two
/// rows per query value), the keys and values, the output, and the block indexer's projection
/// (one key head after the indexer's query heads).
fn qsa(walk: &mut Inspector<'_>, prefix: &str, dims: &Dims) -> Result<(), EngError> {
    let h = dims.hidden as usize;
    let q_width = dims.heads_q as usize * dims.head_dim as usize;
    let kv_width = dims.heads_kv as usize * dims.head_dim as usize;
    let idx_width = (dims.idx_heads as usize + 1) * dims.idx_dim as usize;
    for (role, k, n) in [
        ("q_proj", h, 2 * q_width),
        ("k_proj", h, kv_width),
        ("v_proj", h, kv_width),
        ("o_proj", q_width, h),
        ("indexer.index_qk_proj", h, idx_width),
    ] {
        walk.matrix(&format!("{prefix}.{role}"), k, n)?;
    }
    for (role, len) in [
        ("q_norm", dims.head_dim),
        ("k_norm", dims.head_dim),
        ("indexer.q_layernorm", dims.idx_dim),
        ("indexer.k_layernorm", dims.idx_dim),
    ] {
        walk.plain(&format!("{prefix}.{role}.weight"), &[len], false)?;
    }
    Ok(())
}

/// The PLE block under `prefix` (`...ple`) of the one layer that has it. Its n-gram rows are in
/// the separate n-gram file.
fn ple(walk: &mut Inspector<'_>, prefix: &str, dims: &Dims) -> Result<(), EngError> {
    walk.plain(
        &format!("{prefix}.key_proj.weight"),
        &[dims.width, dims.ple_embed],
        true,
    )?;
    walk.plain(
        &format!("{prefix}.value_proj.weight"),
        &[dims.hidden, dims.ple_embed],
        true,
    )?;
    // The norms and the convolution are read by the host PLE step (`ple::step` applies `1 + w`
    // itself); only the two projections are device weights.
    for norm in ["norm_conv", "norm_key", "norm_query"] {
        walk.host_plain(&format!("{prefix}.{norm}.weight"), &[dims.width])?;
    }
    walk.host_plain(
        &format!("{prefix}.conv1d.weight"),
        &[dims.width, 1, dims.ple_conv],
    )?;
    Ok(())
}

#[cfg(test)]
mod tests;
