//! Resident decode of the stepping-stone family `Qwen3.6-35B-A3B` on MLX (decision 0046).
//!
//! All weights except the embedding table, including the 120 expert stacks, stay resident on the
//! engine thread. The embedding table stays in the checkpoint, and the token start reads one row
//! per token. The portable parts (`layout`, `tiny`) run on Linux. The graph builders need the Mac GPU.

mod chunk_plan;
mod layout;

#[cfg(all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64"))]
mod attn;
#[cfg(all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64"))]
mod decoder;
#[cfg(all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64"))]
mod delta;
#[cfg(all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64"))]
pub(super) mod executor;
#[cfg(all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64"))]
mod moe;
#[cfg(all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64"))]
mod ops;
#[cfg(all(
    test,
    feature = "native-mlx",
    target_os = "macos",
    target_arch = "aarch64"
))]
mod parity;
#[cfg(all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64"))]
mod prefill;
#[cfg(all(
    test,
    feature = "native-mlx",
    target_os = "macos",
    target_arch = "aarch64"
))]
mod real;
#[cfg(all(
    test,
    feature = "native-fault-qualification",
    target_os = "macos",
    target_arch = "aarch64"
))]
mod route_parity;
#[cfg(all(
    test,
    feature = "native-mlx",
    target_os = "macos",
    target_arch = "aarch64"
))]
mod scale_knob;
#[cfg(all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64"))]
mod state;
#[cfg(all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64"))]
mod step;
#[cfg(all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64"))]
mod weights;

/// The shared parity bounds and the per-dtype within-mode rule of the chunk gates (fix1 of packet
/// Q36-BF16-b). `parity` and `prefill/parity` are Mac-gated, but this module compiles on Linux in
/// test builds, so `within_mode_passes` — the rule gate 2 asks per row — has a portable unit
/// test. `passes` and `MIN_COSINE` are the decision-0046 item-4 gate that the decode parity
/// asserted before; `rel_passes` and `MAX_REL_ERROR` are the decision-0044 G2 bound. Fix4 of
/// packet Q36-BF16-b moved `env_number`, `min_with_index`, and `write_f32s` here from
/// `prefill/parity.rs` (they need no native code), so each keeps one definition with its own
/// unit test next to it.
#[cfg(test)]
mod within_mode {
    use super::super::settings::Activation;
    use crate::family::route::keeps_floored_anchor;

    /// The cosine gate of every parity run (decision 0046 item 4): 0.99999.
    pub(super) const MIN_COSINE: f64 = 0.999_99;
    /// The largest relative error of the F32 chunk gates (decision 0044 G2): 1e-4.
    pub(super) const MAX_REL_ERROR: f64 = 1e-4;
    /// The floor of the BF16 run's minimum per-layer cosine (fix5 of packet Q36-BF16-a; decision
    /// 0046 A6): 0.95, below the 0.9799 that the d1 BF16 run observed (evidence 9dec1b7c). The
    /// one definition that `parity.rs` imports too; fix2 of packet Q36-BF16-b moved it here so
    /// the gate-1 verdict below reaches it without the Mac-gated `parity` module.
    pub(super) const BF16_LAYER_COSINE_FLOOR: f64 = 0.95;
    /// The d1 BF16 minimum layer cosine (evidence 9dec1b7c): the no-ratchet anchor of the floor.
    pub(super) const D1_BF16_MIN_LAYER_COSINE: f64 = 0.9799;
    /// The largest regression below the d1 minimum that the floor allows without a recorded
    /// decision (fix5 of packet Q36-BF16-a).
    pub(super) const BF16_LAYER_COSINE_REGRESSION: f64 = 0.01;

    /// True for a finite cosine at or above the gate. A NaN fails.
    pub(super) fn passes(cosine: f64) -> bool {
        cosine.is_finite() && cosine >= MIN_COSINE
    }

    /// True for a relative error that is finite and at most `MAX_REL_ERROR`.
    pub(super) fn rel_passes(rel: f64) -> bool {
        rel.is_finite() && rel <= MAX_REL_ERROR
    }

    /// True for a within-mode LAYER row of the chunk gates under `activation` (packet
    /// Q36-BF16-b): the F32 control keeps both frozen bounds of decision 0044 G2 (cosine and
    /// relative error); a BF16 run needs a finite cosine only. The last-row cosine of one layer
    /// can fall far under any simple floor without moving the decisions: the slice b gate
    /// measured 0.716 at layer 30 (600 tokens, chunk 128) and 0.842 (600 tokens, chunk 256) with
    /// the logits cosine at 0.990 and 0.993, the top-5 sets equal and the 8 greedy tokens equal.
    /// The BF16 layer cosines are therefore REPORTED, with the layer of the minimum, and the
    /// floor sits on the logits row (`within_mode_logits_passes`).
    pub(super) fn within_mode_passes(activation: Activation, cos: f64, rel: f64) -> bool {
        if activation == Activation::F32 {
            passes(cos) && rel_passes(rel)
        } else {
            cos.is_finite()
        }
    }

    /// The floor of the BF16 logits cosine of gate 2 (chunk against one-token steps): 0.98. The
    /// lowest value over seven BF16 samples of the slice b gate (64 to 1000 tokens, chunks 32 to
    /// 256) is 0.9901, so the margin is about 0.01. PROVISIONAL (git-db, 2026-10-11): the sample
    /// is thin and the floor is re-derived after the broad format-token eval (Q36-BF16-EVAL).
    pub(super) const BF16_LOGITS_COSINE_FLOOR: f64 = 0.98;

    /// True for the LOGITS row of gate 2 under `activation`: as `within_mode_passes` for the F32
    /// control; a BF16 run needs a finite cosine of at least `BF16_LOGITS_COSINE_FLOOR`. The
    /// relative error stays reported.
    pub(super) fn within_mode_logits_passes(activation: Activation, cos: f64, rel: f64) -> bool {
        if activation == Activation::F32 {
            within_mode_passes(activation, cos, rel)
        } else {
            cos.is_finite() && cos >= BF16_LOGITS_COSINE_FLOOR
        }
    }

    /// The minimum of `values` and the index where it falls. A NaN counts as the minimum with
    /// its index — the FIRST NaN (fix4 of packet Q36-BF16-b), so the report of a broken run
    /// names the first layer that broke, not the last. An empty slice states its behavior: the
    /// loop seed `(INFINITY, 0)` comes back unchanged.
    pub(super) fn min_with_index(values: &[f64]) -> (f64, usize) {
        let mut best = (f64::INFINITY, 0usize);
        for (index, &value) in values.iter().enumerate() {
            if !best.0.is_nan() && (value.is_nan() || value < best.0) {
                best = (value, index);
            }
        }
        best
    }

    /// Parses `raw`, the text of the environment variable `name`, as a number from `low` to
    /// `high`. Panics naming `name` and the range when the text is not a number or falls outside
    /// it. Split out of `env_number` (fix4 of packet Q36-BF16-b) so the range rule has its own
    /// portable test: edition 2024 makes `std::env::set_var` unsafe and the crate denies
    /// `unsafe_code`, so a test cannot carry the variable into `env_number` itself.
    pub(super) fn number_in_range(name: &str, raw: &str, low: usize, high: usize) -> usize {
        let n: usize = raw
            .parse()
            .unwrap_or_else(|_| panic!("{name} must be a number from {low} to {high}"));
        if n < low || n > high {
            panic!("{name} must be a number from {low} to {high}");
        }
        n
    }

    /// Reads `name` from the environment and parses it as a number from `low` to `high`;
    /// unset gives `default`. Fix4 of packet Q36-BF16-b moved it here from `prefill/parity.rs`,
    /// which passes the chunk bound `MAX_CHUNK_ROWS` and the token bound derived from the
    /// context of its `decoder_for` call.
    pub(super) fn env_number(name: &str, default: usize, low: usize, high: usize) -> usize {
        std::env::var(name)
            .map(|value| number_in_range(name, &value, low, high))
            .unwrap_or(default)
    }

    /// Writes `values` to `path` as little-endian f32 bytes, the dump format that the `f32s`
    /// reader of `parity` reads back. Gated like its only caller (`prefill/parity.rs`) and its
    /// reader, so a Linux test build sees no unused definition.
    #[cfg(all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64"))]
    pub(super) fn write_f32s(path: &std::path::Path, values: &[f32]) {
        let mut bytes = Vec::with_capacity(values.len() * 4);
        for &value in values {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        std::fs::write(path, bytes).unwrap();
    }

    /// The verdict of gate 1 under `activation` (fix1 of packet Q36-BF16-b): the F32 control
    /// asserts every recorded failure — the frozen gate — while a BF16 run asserts the
    /// decision-0046 A6 fix5 floor instead and only reports its misses. Extracted from the gate
    /// so the branch rule has its own test. Fix2 of packet Q36-BF16-b moved it here from
    /// `prefill/parity.rs`, which is Mac-gated: the verdict needs no native code, so its test
    /// runs on Linux too.
    pub(super) fn chunk_gate1_verdict(
        activation: Activation,
        failures: &[String],
        min_cos: f64,
        nan_cosines: usize,
    ) -> Result<(), String> {
        if activation == Activation::F32 {
            if failures.is_empty() {
                Ok(())
            } else {
                Err(format!("chunk parity gate failed: {failures:#?}"))
            }
        } else {
            if nan_cosines != 0 {
                return Err(format!(
                    "a NaN cosine is a broken run, not a result: {nan_cosines} NaN layer rows"
                ));
            }
            if keeps_floored_anchor(
                min_cos,
                BF16_LAYER_COSINE_FLOOR,
                D1_BF16_MIN_LAYER_COSINE,
                BF16_LAYER_COSINE_REGRESSION,
            ) {
                Ok(())
            } else {
                Err(format!(
                    "BF16 chunk parity floor: min layer cosine {min_cos} is under 0.95 or regresses \
                     more than 0.01 below the d1 value 0.9799 without a recorded decision"
                ))
            }
        }
    }

    mod tests {
        use super::*;

        /// The F32 branch of the gate asserts both cosine and relative error; a BF16 LAYER row needs
        /// a finite cosine only (fix4 and fix5 of packet Q36-BF16-b): the slice b gate measured a
        /// layer cosine of 0.716 with the logits at 0.990, so a layer floor would fail legitimate
        /// samples. A row that keeps the cosine but misses the relative bound fails the F32 gate
        /// and passes the BF16 gate, a finite cosine under 0.99999 fails the F32 gate only, and a
        /// NaN cosine fails both.
        #[test]
        fn within_mode_passes_gates_cosine_and_relative_error_in_f32_and_finiteness_in_bf16() {
            assert!(
                within_mode_passes(Activation::F32, 1.0, 0.0),
                "both bounds met"
            );
            assert!(
                within_mode_passes(Activation::F32, 0.999_99, 1e-4),
                "the bounds are inclusive"
            );
            assert!(
                !within_mode_passes(Activation::F32, 1.0, 1e-3),
                "f32 gates the relative error"
            );
            assert!(
                !within_mode_passes(Activation::F32, 1.0, f64::NAN),
                "a NaN relative error fails f32"
            );
            assert!(
                !within_mode_passes(Activation::F32, 0.999_9, 0.0),
                "f32 gates the cosine"
            );
            assert!(
                !within_mode_passes(Activation::F32, f64::NAN, 0.0),
                "a NaN cosine fails"
            );
            assert!(
                within_mode_passes(Activation::Bf16, 1.0, 1.0),
                "bf16 reports the relative error without gating it"
            );
            assert!(
                within_mode_passes(Activation::Bf16, 0.716, 0.86),
                "a measured BF16 layer cosine (0.716 at layer 30, 600 tokens) is reported, not gated"
            );
            assert!(
                !within_mode_passes(Activation::Bf16, f64::NAN, 0.0),
                "a NaN cosine fails bf16"
            );
        }

        /// The LOGITS row of gate 2: the F32 control as the layer rows; a BF16 run needs a cosine
        /// of at least `BF16_LOGITS_COSINE_FLOOR` (0.98, inclusive), and the measured values pass
        /// (0.9853 and 0.9931 at 296 tokens, 0.9901 the lowest of seven samples).
        #[test]
        fn within_mode_logits_passes_gates_the_logits_floor_in_bf16_and_the_bounds_in_f32() {
            assert!(
                within_mode_logits_passes(Activation::F32, 1.0, 0.0),
                "f32 both bounds met"
            );
            assert!(
                !within_mode_logits_passes(Activation::F32, 1.0, 1e-3),
                "f32 gates the relative error of the logits too"
            );
            assert!(
                !within_mode_logits_passes(Activation::F32, 0.999_9, 0.0),
                "f32 gates the cosine of the logits"
            );
            assert!(
                within_mode_logits_passes(Activation::Bf16, 0.9931, 0.119),
                "a measured BF16 logits cosine passes"
            );
            assert!(
                within_mode_logits_passes(Activation::Bf16, 0.9901, 0.12),
                "the lowest measured BF16 logits cosine passes"
            );
            assert!(
                within_mode_logits_passes(Activation::Bf16, 0.98, 0.5),
                "the 0.98 floor is inclusive and the relative error is reported only"
            );
            assert!(
                !within_mode_logits_passes(Activation::Bf16, 0.979, 0.0),
                "a BF16 logits cosine under the floor fails"
            );
            assert!(
                !within_mode_logits_passes(Activation::Bf16, f64::NAN, 0.0),
                "a NaN logits cosine fails bf16"
            );
        }

        /// `min_with_index` reports the first NaN (the first layer that broke, fix4 of packet
        /// Q36-BF16-b), the plain minimum with its index, and — stated here because no caller
        /// feeds an empty slice — the unchanged loop seed for an empty slice.
        #[test]
        fn min_with_index_reports_the_first_nan_the_minimum_and_the_empty_seed() {
            let (value, index) = min_with_index(&[1.0, f64::NAN, 0.5, f64::NAN]);
            assert!(
                value.is_nan() && index == 1,
                "the first NaN wins, not the last: ({value}, {index})"
            );
            assert_eq!(
                min_with_index(&[3.0, 1.0, 2.0]),
                (1.0, 1),
                "the plain minimum with its index"
            );
            assert_eq!(
                min_with_index(&[f64::INFINITY, f64::INFINITY]),
                (f64::INFINITY, 0),
                "equal maxima keep the first index"
            );
            assert_eq!(
                min_with_index(&[]),
                (f64::INFINITY, 0),
                "an empty slice states the seed the loop starts from"
            );
        }

        /// `env_number` gives the default when the variable is unset, and `number_in_range` holds
        /// the range rule: both bounds are inclusive and the words of the panic name the variable
        /// and the range.
        #[test]
        fn env_number_defaults_when_unset_and_number_in_range_gates_the_range() {
            assert_eq!(
                env_number("Q36_WITHIN_MODE_ENV_NUMBER", 296, 16, 1000),
                296,
                "an unset variable gives the default"
            );
            assert_eq!(number_in_range("Q36_G2_TOKENS", "296", 16, 1000), 296);
            assert_eq!(
                number_in_range("Q36_G2_TOKENS", "16", 16, 1000),
                16,
                "the low bound is inclusive"
            );
            assert_eq!(
                number_in_range("Q36_G2_TOKENS", "1000", 16, 1000),
                1000,
                "the high bound is inclusive"
            );
        }

        #[test]
        #[should_panic(expected = "Q36_G2_TOKENS must be a number from 16 to 1000")]
        fn number_in_range_refuses_a_value_under_the_range() {
            number_in_range("Q36_G2_TOKENS", "15", 16, 1000);
        }

        #[test]
        #[should_panic(expected = "Q36_G2_TOKENS must be a number from 16 to 1000")]
        fn number_in_range_refuses_a_value_over_the_range() {
            number_in_range("Q36_G2_TOKENS", "1001", 16, 1000);
        }

        #[test]
        #[should_panic(expected = "Q36_G2_TOKENS must be a number from 16 to 1000")]
        fn number_in_range_refuses_text_that_is_not_a_number() {
            number_in_range("Q36_G2_TOKENS", "296x", 16, 1000);
        }

        /// `write_f32s` round-trips through `f32s`, the reader of the dump files: the bytes come
        /// back as the same floats in the same order. Gated with `write_f32s`, because its
        /// reader lives in the Mac-gated `parity` module.
        #[cfg(all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64"))]
        #[test]
        fn write_f32s_round_trips_through_the_f32s_reader() {
            use super::super::parity::f32s;

            let values = [0.0f32, -1.5, 3.25e10, f32::MIN_POSITIVE];
            let path = std::env::temp_dir().join("q36-within-mode-write-f32s-roundtrip.f32");
            write_f32s(&path, &values);
            assert_eq!(
                f32s(&std::fs::read(&path).unwrap()),
                values,
                "the dump bytes read back as the same floats"
            );
            std::fs::remove_file(&path).unwrap();
        }

        /// The F32 branch of the gate-1 verdict asserts every recorded failure and the BF16 branch
        /// asserts the floor instead (fix1 of packet Q36-BF16-b): a run that missed the frozen gate
        /// must fail in `f32` however good its floor numbers are, and a BF16 run with a broken
        /// floor or a NaN row must fail too. Fix2 of packet Q36-BF16-b moved this test next to the
        /// function so it runs on Linux as well.
        #[test]
        fn the_gate1_verdict_asserts_the_failures_in_f32_and_the_floor_in_bf16() {
            let misses = vec!["token 4 layer 3: cosine 0.5".to_string()];
            assert_eq!(
                chunk_gate1_verdict(Activation::F32, &[], 1.0, 0),
                Ok(()),
                "a clean F32 run passes"
            );
            let refused = chunk_gate1_verdict(Activation::F32, &misses, 1.0, 0).unwrap_err();
            assert!(refused.contains("chunk parity gate failed"), "{refused}");
            assert!(refused.contains("cosine 0.5"), "{refused}");
            assert_eq!(
                chunk_gate1_verdict(Activation::Bf16, &misses, 0.9799, 0),
                Ok(()),
                "a BF16 run keeps the d1 anchor and only reports its misses"
            );
            assert!(
                chunk_gate1_verdict(Activation::Bf16, &[], 0.5, 0).is_err(),
                "a BF16 run under the floor fails"
            );
            assert!(
                chunk_gate1_verdict(Activation::Bf16, &[], 0.965, 0).is_err(),
                "a BF16 run that regresses more than 0.01 below the anchor fails"
            );
            assert!(
                chunk_gate1_verdict(Activation::Bf16, &[], 1.0, 1).is_err(),
                "a NaN layer cosine is a broken run"
            );
        }
    }
}
