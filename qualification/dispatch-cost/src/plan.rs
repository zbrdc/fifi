//! The decode graph shapes and call counts. This module is portable.

use serde::Serialize;

pub const GROUP_SIZE: u32 = 64;
pub const BITS: u32 = 4;
pub const HEADS_Q: u32 = 8;
pub const HEADS_KV: u32 = 2;
pub const HEAD_DIM: u32 = 64;
pub const ROPE_THETA: f32 = 10_000.0;
pub const ROPE_PART: u32 = 64;
pub const ROPE_POS: u32 = 100;
pub const RMS_EPS: f32 = 1.0e-6;

/// Calls per layer: rms_norm x2, qmm x4, softmax, mm, add x2, silu, mul, rope, and two imports (q and k).
pub const CALLS_PER_LAYER: usize = 15;
/// Calls outside layers: one input import and one readback.
pub const CALLS_FIXED: usize = 2;

#[derive(Serialize, Clone, Debug)]
pub struct Config {
    pub hidden: u32,
    pub inter: u32,
    pub seq: u32,
    pub heads_q: u32,
    pub heads_kv: u32,
    pub head_dim: u32,
    pub bits: u32,
    pub group_size: u32,
}

impl Config {
    pub fn new(hidden: u32, inter: u32, seq: u32) -> Result<Self, String> {
        for (name, value) in [("hidden", hidden), ("inter", inter)] {
            if value == 0 || value % GROUP_SIZE != 0 {
                return Err(format!("{name} must be a nonzero multiple of {GROUP_SIZE}"));
            }
        }
        if seq == 0 {
            return Err("seq must be nonzero".into());
        }
        Ok(Self {
            hidden,
            inter,
            seq,
            heads_q: HEADS_Q,
            heads_kv: HEADS_KV,
            head_dim: HEAD_DIM,
            bits: BITS,
            group_size: GROUP_SIZE,
        })
    }

    pub fn calls_per_chain(&self, layers: usize) -> usize {
        CALLS_FIXED + CALLS_PER_LAYER * layers
    }

    /// Text description of one layer, for `--plan`.
    pub fn describe(&self) -> Vec<String> {
        let (h, i, s) = (self.hidden, self.inter, self.seq);
        vec![
            format!("import x [1,{h}] F32 (once per chain)"),
            format!(
                "import q [{},1,{}] and k [{},1,{}] (2 imports per layer)",
                self.heads_q, self.head_dim, self.heads_kv, self.head_dim
            ),
            format!("rms_norm x [1,{h}] w [{h}]"),
            format!("qmm n [1,{h}] with Wscore [{s},{h}] -> [1,{s}]"),
            format!("softmax [1,{s}]"),
            format!("mm p [1,{s}] V [{s},{h}] -> [1,{h}]"),
            format!("add x a -> [1,{h}]"),
            format!("rms_norm x1 [1,{h}] w [{h}]"),
            format!("qmm n2 with Wg [{i},{h}] -> [1,{i}]"),
            format!("qmm n2 with Wu [{i},{h}] -> [1,{i}]"),
            format!("silu g [1,{i}]"),
            format!("mul silu(g) u -> [1,{i}]"),
            format!("qmm act [1,{i}] with Wd [{h},{i}] -> [1,{h}]"),
            format!("add x1 d -> [1,{h}]"),
            "rope q, k (pos, theta 10000, part 64)".to_string(),
            format!("readback x [1,{h}] (once per chain)"),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_rejects_invalid_dimensions_with_documented_errors() {
        let cases = [
            ((0, 128, 1), "hidden must be a nonzero multiple of 64"),
            ((65, 128, 1), "hidden must be a nonzero multiple of 64"),
            ((128, 0, 1), "inter must be a nonzero multiple of 64"),
            ((128, 64, 0), "seq must be nonzero"),
        ];

        for ((hidden, inter, seq), expected_error) in cases {
            assert_eq!(Config::new(hidden, inter, seq).unwrap_err(), expected_error);
        }
    }

    #[test]
    fn valid_config_uses_the_declared_constants() {
        let config = Config::new(128, 256, 32).unwrap();

        assert_eq!(config.group_size, 64);
        assert_eq!(config.bits, 4);
        assert_eq!(config.heads_q, 8);
        assert_eq!(config.heads_kv, 2);
        assert_eq!(config.head_dim, 64);
        assert_eq!(GROUP_SIZE, 64);
        assert_eq!(BITS, 4);
        assert_eq!(HEADS_Q, 8);
        assert_eq!(HEADS_KV, 2);
        assert_eq!(HEAD_DIM, 64);
        assert_eq!(ROPE_THETA, 10_000.0);
        assert_eq!(ROPE_PART, 64);
        assert_eq!(ROPE_POS, 100);
        assert_eq!(RMS_EPS, 1.0e-6);
    }

    #[test]
    fn calls_per_chain_includes_fixed_and_per_layer_calls() {
        let config = Config::new(128, 256, 32).unwrap();

        assert_eq!(config.calls_per_chain(0), 2);
        assert_eq!(config.calls_per_chain(1), 17);
        assert_eq!(config.calls_per_chain(8), 122);
    }

    #[test]
    fn describe_names_outer_import_and_readback() {
        let description = Config::new(128, 256, 32).unwrap().describe();

        assert_eq!(description.len(), 16);
        assert_eq!(
            description.first().map(String::as_str),
            Some("import x [1,128] F32 (once per chain)")
        );
        assert_eq!(
            description.last().map(String::as_str),
            Some("readback x [1,128] (once per chain)")
        );
    }
}
