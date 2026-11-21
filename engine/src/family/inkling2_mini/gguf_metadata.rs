//! Header-only geometry for the mainline fixed-loop Inkling 2 Mini GGUF.

use std::io::{Read, Seek, SeekFrom};

use serde::Serialize;

use super::{
    CORE_END, CORE_START, EXPERTS, HEAD_DIM, HIDDEN, INTERMEDIATE, KV_HEADS, LAYERS, MAX_CONTEXT,
    MODEL_TYPE, PASSES, QUERY_HEADS, TOP_K, WINDOW, checkpoint_error,
};
use crate::{
    EngError,
    weights::{GgufHeader, MetaValue},
};

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Geometry {
    pub physical_layers: usize,
    pub logical_layers: usize,
    pub core_start: usize,
    pub core_end: usize,
    pub passes: usize,
    pub global_layers: usize,
    pub local_layers: usize,
    pub hidden_size: usize,
    pub context_length: u32,
    pub sliding_window: u32,
    pub experts: usize,
    pub experts_per_token: usize,
}

pub(crate) fn inspect(
    header: &GgufHeader,
    source: &mut (impl Read + Seek),
) -> Result<Geometry, EngError> {
    if header.split().is_some_and(|split| split.count != 1) {
        return Err(checkpoint_error(
            "split GGUF checkpoint inspection is unsupported",
        ));
    }
    for (name, expected) in [
        ("block_count", LAYERS as u32),
        ("loop.start", CORE_START as u32),
        ("loop.end", CORE_END as u32),
        ("loop.count", PASSES as u32),
        ("loop.gate_rank", 64),
        ("embedding_length", HIDDEN as u32),
        ("context_length", MAX_CONTEXT),
        ("expert_feed_forward_length", INTERMEDIATE as u32),
        ("expert_count", EXPERTS as u32),
        ("expert_used_count", TOP_K as u32),
        ("attention.key_length", HEAD_DIM as u32),
        ("attention.value_length", HEAD_DIM as u32),
        ("attention.key_length_swa", HEAD_DIM as u32),
        ("attention.value_length_swa", HEAD_DIM as u32),
        ("attention.sliding_window", WINDOW),
    ] {
        let key = format!("{MODEL_TYPE}.{name}");
        if header.meta(&key) != Some(&MetaValue::U32(expected)) {
            return Err(checkpoint_error(format!("{key} must be {expected}")));
        }
    }
    if !matches!(
        header.meta("inkling2_loop.loop.varying"),
        None | Some(MetaValue::Bool(false))
    ) {
        return Err(checkpoint_error("inkling2_loop.loop.varying must be false"));
    }
    for (key, element_type, width, expected) in [
        (
            "inkling2_loop.attention.head_count",
            4,
            4,
            QUERY_HEADS as u32,
        ),
        (
            "inkling2_loop.attention.head_count_kv",
            4,
            4,
            KV_HEADS as u32,
        ),
        ("inkling2_loop.attention.sliding_window_pattern", 7, 1, 0),
    ] {
        let Some(MetaValue::Array(array)) = header.meta(key) else {
            return Err(checkpoint_error(format!("{key} must be an array")));
        };
        // The published conversion stores head counts as I32. Positive counts
        // have identical bytes in either integer encoding.
        let type_matches =
            array.elem_type == element_type || (element_type == 4 && array.elem_type == 5);
        if !type_matches || array.len != LAYERS as u64 {
            return Err(checkpoint_error(format!(
                "{key} has invalid element type or length"
            )));
        }
        source
            .seek(SeekFrom::Start(array.byte_range.start))
            .map_err(checkpoint_error)?;
        let mut bytes = [0; LAYERS * 4];
        source
            .read_exact(&mut bytes[..LAYERS * width])
            .map_err(checkpoint_error)?;
        for (layer, bytes) in bytes[..LAYERS * width].chunks_exact(width).enumerate() {
            let valid = match *bytes {
                [value] => value == u8::from(layer % 6 != 5),
                [a, b, c, d] => u32::from_le_bytes([a, b, c, d]) == expected,
                _ => unreachable!(),
            };
            if !valid {
                return Err(checkpoint_error(format!(
                    "{key}[{layer}] has an invalid value"
                )));
            }
        }
    }
    let logical_layers = LAYERS + (CORE_END - CORE_START) * (PASSES - 1);
    let global_layers = logical_layers / 6;
    Ok(Geometry {
        physical_layers: LAYERS,
        logical_layers,
        core_start: CORE_START,
        core_end: CORE_END,
        passes: PASSES,
        global_layers,
        local_layers: logical_layers - global_layers,
        hidden_size: HIDDEN,
        context_length: MAX_CONTEXT,
        sliding_window: WINDOW,
        experts: EXPERTS,
        experts_per_token: TOP_K,
    })
}
