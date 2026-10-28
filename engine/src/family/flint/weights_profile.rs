//! The tensor inventory of RecursiveSI/flint-int4 revision 3b93f905.

use std::collections::BTreeMap;

use crate::tensor::Dtype;

#[derive(Default)]
pub(super) struct Profile {
    pub(super) tensors: BTreeMap<String, (Dtype, Vec<u32>)>,
    pub(super) packed: BTreeMap<String, (Vec<u32>, u8)>,
}

impl Profile {
    pub(super) fn pinned() -> Self {
        let mut profile = Self::default();
        profile.matrix("model.embed_tokens", &[150144, 2048], 8);
        profile.matrix("lm_head", &[150144, 2048], 4);
        profile.plain("model.norm.weight", &[2048]);
        for layer in 0..49 {
            let prefix = format!("model.layers.{layer}");
            for name in ["input_layernorm.weight", "post_attention_layernorm.weight"] {
                profile.plain(&format!("{prefix}.{name}"), &[2048]);
            }
            for name in ["q_norm.weight", "k_norm.weight"] {
                profile.plain(&format!("{prefix}.self_attn.{name}"), &[128]);
            }
            profile.plain(&format!("{prefix}.self_attn.softmax_offset"), &[32]);
            for (name, shape) in [
                ("q_proj", [4096, 2048]),
                ("k_proj", [512, 2048]),
                ("v_proj", [512, 2048]),
                ("o_proj", [2048, 4096]),
            ] {
                profile.matrix(&format!("{prefix}.self_attn.{name}"), &shape, 4);
            }
            if layer < 2 {
                profile.ffn(&format!("{prefix}.mlp"), 6144, false);
            } else {
                profile.plain(&format!("{prefix}.mlp.gate.weight"), &[256, 2048]);
                profile.ffn(&format!("{prefix}.mlp.shared_expert"), 1024, false);
                profile.ffn(&format!("{prefix}.mlp.switch_mlp"), 512, true);
            }
        }
        profile
    }

    fn plain(&mut self, name: &str, shape: &[u32]) {
        self.tensors
            .insert(name.into(), (Dtype::Bf16, shape.to_vec()));
    }

    fn matrix(&mut self, stem: &str, logical: &[u32], bits: u8) {
        self.packed.insert(stem.into(), (logical.to_vec(), bits));
        let last = logical.len() - 1;
        let mut payload = logical.to_vec();
        payload[last] = logical[last] / 8 * u32::from(bits);
        self.tensors
            .insert(format!("{stem}.weight"), (Dtype::U8, payload));
        let mut groups = logical.to_vec();
        groups[last] /= 64;
        self.plain(&format!("{stem}.scales"), &groups);
        self.plain(&format!("{stem}.biases"), &groups);
    }

    fn ffn(&mut self, stem: &str, intermediate: u32, stacked: bool) {
        for (name, rows, width) in [
            ("gate_proj", intermediate, 2048),
            ("up_proj", intermediate, 2048),
            ("down_proj", 2048, intermediate),
        ] {
            let shape = if stacked {
                vec![256, rows, width]
            } else {
                vec![rows, width]
            };
            self.matrix(&format!("{stem}.{name}"), &shape, 4);
        }
    }
}
