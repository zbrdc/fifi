//! DFlash2 is a separate checkpoint, not the DSpark variant of `dflash`.
use crate::{
    EngError,
    weights::{GgufLimits, GgufType, GgufWeights, MetaValue},
};
use std::path::Path;
mod block;

pub(super) struct Dflash2(GgufWeights);

fn open(path: &Path) -> Result<GgufWeights, EngError> {
    let w = GgufWeights::open_any_type(
        path,
        GgufLimits {
            max_header_bytes: 32 << 20,
            max_tensor_bytes: 1 << 30,
        },
        None,
    )?;
    w.require_single_file("DFlash2")?;
    let h = w.header();
    if !matches!(h.meta("general.architecture"),Some(MetaValue::String(a)) if a == "dflash")
        || !matches!(
            h.meta("dflash.selector_softcap_after_correction"),
            Some(MetaValue::Bool(true))
        )
        || h.tensors().len() != 81
    {
        return Err(super::bad("not an Inkling DFlash2 checkpoint"));
    }
    for (key, expected) in [
        ("dflash.block_count", 5),
        ("dflash.embedding_length", 2304),
        ("dflash.feed_forward_length", 6912),
        ("dflash.attention.head_count", 32),
        ("dflash.attention.head_count_kv", 8),
        ("dflash.attention.key_length", 128),
        ("dflash.attention.value_length", 128),
        ("dflash.attention.sliding_window", 2048),
        ("tokenizer.ggml.mask_token_id", 200056),
        ("dflash.block_size", 8),
        ("dflash.selector_top_k", 16),
        ("dflash.selector_rank", 256),
        ("dflash.conv_kernel_size", 2),
        ("dflash.conv_group_size", 16),
    ] {
        if h.meta_u32(key) != Some(expected) {
            return Err(super::bad(format!("incorrect {key}")));
        }
    }
    for (key, expected) in [
        ("dflash.final_logit_softcapping", 30.),
        ("dflash.rope.freq_base", 10000.),
        ("dflash.attention.layer_norm_rms_epsilon", 1e-6),
    ] {
        if !matches!(h.meta(key), Some(MetaValue::F32(v)) if *v == expected) {
            return Err(super::bad(format!("incorrect {key}")));
        }
    }
    if !matches!(
        h.meta("dflash.attention.causal"),
        Some(MetaValue::Bool(false))
    ) {
        return Err(super::bad("expected noncausal attention"));
    }
    match h.meta("dflash.attention.sliding_window_pattern") {
        Some(MetaValue::Array(a)) if a.elem_type == 7 && a.len == 5 => {
            let mut flags = [0; 5];
            w.read_header_range(a.byte_range.start, &mut flags)?;
            if flags != [1; 5] {
                return Err(super::bad("expected five sliding layers"));
            }
        }
        _ => return Err(super::bad("missing sliding window pattern")),
    }
    match h.meta("dflash.target_visits") {
        Some(MetaValue::Array(a)) if a.elem_type == 5 && a.len == 4 => {
            let mut bytes = [0; 16];
            w.read_header_range(a.byte_range.start, &mut bytes)?;
            if bytes != [5i32, 23, 41, 48].map(i32::to_le_bytes).concat().as_slice() {
                return Err(super::bad("wrong target visits"));
            }
        }
        _ => return Err(super::bad("missing target visits")),
    }
    let tensor = |name: &str, dims: &[u64], matrix: bool| {
        if h.tensor(name).is_some_and(|t| {
            t.dims() == dims
                && if matrix {
                    matches!(t.ggml_type(), GgufType::Bf16 | GgufType::Q80)
                } else {
                    t.ggml_type() == GgufType::F32
                }
        }) {
            Ok(())
        } else {
            Err(super::bad(format!("invalid DFlash2 tensor {name}")))
        }
    };
    for (n, d) in [
        ("fc.weight", [9216, 2304]),
        ("selector_hidden.weight", [2304, 256]),
        ("selector_predecessor.weight", [256, 200058]),
        ("selector_successor.weight", [256, 200058]),
    ] {
        tensor(n, &d, true)?;
    }
    for n in ["enc.output_norm.weight", "output_norm.weight"] {
        tensor(n, &[2304], false)?;
    }
    for i in 0..5 {
        for (n, d) in [
            ("attn_q", [2304, 4096]),
            ("attn_k", [2304, 1024]),
            ("attn_v", [2304, 1024]),
            ("attn_output", [4096, 2304]),
            ("ffn_gate", [2304, 6912]),
            ("ffn_up", [2304, 6912]),
            ("ffn_down", [6912, 2304]),
            ("attn_conv_proj", [2304, 576]),
            ("ffn_conv_proj", [2304, 576]),
        ] {
            tensor(&format!("blk.{i}.{n}.weight"), &d, true)?;
        }
        for (n, d) in [
            ("attn_norm", 2304),
            ("ffn_norm", 2304),
            ("attn_q_norm", 128),
            ("attn_k_norm", 128),
        ] {
            tensor(&format!("blk.{i}.{n}.weight"), &[d], false)?;
        }
        for n in ["attn_conv_base", "ffn_conv_base"] {
            tensor(&format!("blk.{i}.{n}"), &[2304, 2, 2], false)?;
        }
    }
    Ok(w)
}

#[cfg(test)]
mod tests {
    #[test]
    #[ignore = "requires S0 conversion supplied via FIFI_INKLING_SPEC_DRAFT"]
    fn real_dflash2() {
        let path = std::env::var_os("FIFI_INKLING_SPEC_DRAFT").unwrap();
        let w = super::open(std::path::Path::new(&path)).unwrap();
        let (ty, data) = w.read_tensor("selector_hidden.weight").unwrap();
        let v = crate::weights::dequantize(ty, &data).unwrap();
        assert_eq!(v.len(), 2304 * 256);
        assert!(v.iter().all(|x| x.is_finite()) && v.iter().any(|x| *x != 0.0));
    }
}
