//! The decode-vs-prefill KL gate: the standard check every KV8/quant/fusion change runs before it
//! merges. Method per DeepInfer `tests/models/test_kl.py`: the last `window` prompt positions run
//! teacher-forced through the decode path, then the same tokens re-run through the prefill path,
//! and the per-position KL divergence of the two rows stays under a bound. DeepInfer asserts exact
//! equality behind `BATCH_INVARIANT=1`, an NVIDIA-only mode documented as slower; fifi has no
//! batch-invariant kernels, so this is the truth-tolerance form: control = the prefill row,
//! candidate = the decode row.
//!
//! What the two arms share and what they differ in: both prefill the head of the prompt as
//! chunks (three times, into three states, with a bitwise-equal output control), then the
//! decode arm feeds the window positions one `step` at a time (the served decode path) while
//! the prefill arm feeds the same ids as one-row `prefill` chunks (`prefill` runs a prompt of
//! one token as a chunk, not as a `step`, so the chunk graph — prefill attention, the chunked
//! KV append, the chunked Q8 quantize — is the prefill path even at one row). A third state
//! feeds the whole window as one multi-row chunk; its final row pairs with the decode arm's
//! last row, so drift that only a multi-row chunk shows is caught at that position. The
//! final position therefore contributes two rows (the one-row-chunk control and the
//! multi-row-chunk control both pair with the decode row) — intended: they are distinct
//! comparisons of the same position. The per-position controls are one-row chunks; extending
//! every position to a multi-row control (a snapshot rewind per position, about `window^2 / 2`
//! extra rows) is a follow-up slice.
//!
//! The gate runs the Q8 cache by default — the KV8 form the bound guards — through
//! `new_with_kv`; `new_with_activation` would pin the F32 control cache instead (its doc:
//! "the shipped default stays F32").
//!
//! The comparison core (this file's top level) is pure and carries Linux unit tests, like
//! `long_context`; the model-touching arms and gates live in the `mac` module behind the
//! native cfg, like that module's `mac` harness.
//!
//! The standard commands (the "script" a change runs):
//!
//! ```text
//! # the pure comparison core, part of the plain portable suite
//! cargo test --lib decode_prefill_kl -- --nocapture
//!
//! # the tiny-model gate (Q8 cache), part of the Mac plain suite
//! cargo test --release --features native-mlx --lib decode_vs_prefill_kl -- --nocapture
//!
//! # the real-weights gate (the checkpoint of the change under test)
//! Q36_CKPT_DIR=<CKPT_DIR> Q36_ACTIVATION=bf16 Q36_KV_DTYPE=q8 \
//! cargo test --release --features native-mlx --lib \
//!   real_weights_decode_vs_prefill_kl -- --ignored --nocapture
//! ```
//!
//! The real-weights bound: `Q36_KL_MEAN` (default 0.004), `Q36_KL_MAX` (default 0.05), and
//! `Q36_KL_TOP1` (default 0.99) — the tolerance family of the batch KL gate
//! (`batch_real_tests.rs`). The mean and worst-row bounds are judged per prompt, so one bad
//! prompt cannot be diluted by the others; the top-1 floor is judged over the pooled rows of
//! all prompts, where the real row counts give it its meaning. Every prompt runs and prints
//! its receipt, and the POOLED verdict line with the run's single `passed` field prints
//! whatever the earlier prompts did (its `passed` folds in the per-prompt bounds; a wrapper
//! finds it by the `DECODE_PREFILL_KL_POOLED` prefix, since the assert payload may print
//! after it). A breach fails the test, so the command exits nonzero; a structural panic can
//! stop the run before the POOLED line, so a wrapper treats a missing POOLED line as a
//! failure. The window is `Q36_KL_WINDOW` (default 32, at least 2, clamped to the prompt
//! length minus two); the cache form is `Q36_KV_DTYPE` (default `q8`, the house words
//! `f32 | bf16 | q8`) and the activation is `Q36_ACTIVATION` (default `bf16`, `f32 | bf16`);
//! the F32-activation/BF16-cache pair is not a valid cell and is refused up front. The tiny
//! bound is calibrated from the first Mac plain-suite receipt (see the tiny test).

use super::compile_logits::kl_divergence;

/// The pass bar over the compared positions: the mean KL, the worst single-position KL, and
/// the top-1 agreement rate (the share of positions where both rows pick the same greedy
/// token), boundaries included.
#[derive(Debug, PartialEq)]
struct KlBound {
    mean: f64,
    row: f64,
    top1_rate: f64,
}

/// One kept row of an arm: the logits row and the greedy token it picked.
#[derive(Debug, PartialEq)]
struct ArmRow {
    logits: Vec<f32>,
    token: Option<u32>,
}

/// What one compared position reports. `position` is the index in the prompt ids of the id the
/// two rows predict; `chunked` marks the multi-row-chunk row.
#[derive(Debug, PartialEq)]
struct RowKl {
    position: usize,
    kl_nats: f64,
    top1_equal: bool,
    chunked: bool,
}

/// The verdict over the compared rows. Pure: the arms produce the rows, this decides.
#[derive(Debug, PartialEq)]
struct KlReport {
    rows: Vec<RowKl>,
}

impl KlReport {
    fn mean_kl(&self) -> f64 {
        self.rows.iter().map(|row| row.kl_nats).sum::<f64>() / self.rows.len() as f64
    }

    fn worst(&self) -> &RowKl {
        self.rows
            .iter()
            .max_by(|a, b| a.kl_nats.total_cmp(&b.kl_nats))
            .expect("the report holds rows")
    }

    fn top1_rate(&self) -> f64 {
        self.rows.iter().filter(|row| row.top1_equal).count() as f64 / self.rows.len() as f64
    }

    /// A report with no rows, a non-finite KL, or any metric past its bound fails. The top-1
    /// rate bound is a floor; the other two are ceilings. A KL a hair below zero is the f64
    /// rounding of the `p * ln(p/q)` sum on near-equal rows, so the sibling gate's `-1e-12`
    /// slack applies here too (`batch_real_tests.rs`).
    fn passes(&self, bound: &KlBound) -> bool {
        self.passes_mean_and_row(bound) && self.top1_rate() >= bound.top1_rate
    }

    /// The per-prompt part of the verdict: the mean and worst-row bounds, without the pooled
    /// top-1 floor.
    fn passes_mean_and_row(&self, bound: &KlBound) -> bool {
        !self.rows.is_empty()
            && self
                .rows
                .iter()
                .all(|row| row.kl_nats.is_finite() && row.kl_nats >= -1e-12)
            && self.mean_kl() <= bound.mean
            && self.worst().kl_nats <= bound.row
    }
}

/// Compares the prefill rows (control) with the decode rows (candidate) position by position.
/// Row `i` of each arm predicts prompt id `head + i`; the arms must hold the same number of
/// rows, and every row must hold the full vocabulary.
fn compare_arms(control: &[ArmRow], candidate: &[ArmRow], head: usize) -> Vec<RowKl> {
    assert_eq!(
        control.len(),
        candidate.len(),
        "the arms kept different row counts"
    );
    control
        .iter()
        .zip(candidate)
        .enumerate()
        .map(|(index, (control, candidate))| {
            assert_eq!(
                control.logits.len(),
                candidate.logits.len(),
                "position {}: the rows differ in width",
                head + index
            );
            RowKl {
                position: head + index,
                kl_nats: kl_divergence(&control.logits, &candidate.logits)
                    .unwrap_or_else(|error| panic!("position {}: {error}", head + index)),
                top1_equal: control.token == candidate.token,
                chunked: false,
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The pure verdict: every boundary passes, every breach fails, and each bound fails
    /// alone.
    #[test]
    fn kl_bound_accepts_boundaries_and_rejects_bad_metrics() {
        let bound = KlBound {
            mean: 0.004,
            row: 0.05,
            top1_rate: 0.99,
        };
        let report = |kls: &[f64], flips: usize| {
            let rows = kls
                .iter()
                .enumerate()
                .map(|(index, &kl)| RowKl {
                    position: index,
                    kl_nats: kl,
                    top1_equal: index >= flips,
                    chunked: false,
                })
                .collect();
            KlReport { rows }
        };
        // The row boundary: worst row exactly 0.05 over 100 rows with mean 0.0005, and one
        // flip of 100 (top-1 exactly 0.99).
        let mut boundary = vec![0.0; 100];
        boundary[0] = 0.05;
        let boundary = report(&boundary, 1);
        assert_eq!(
            (boundary.worst().kl_nats, boundary.top1_rate()),
            (0.05, 0.99)
        );
        assert!(boundary.passes(&bound), "{boundary:?}");
        // The mean boundary: a single row of exactly the mean, under the row bound.
        let mean_boundary = report(&[0.004], 0);
        assert_eq!(mean_boundary.mean_kl(), 0.004);
        assert!(mean_boundary.passes(&bound), "{mean_boundary:?}");
        for rejected in [
            report(&[], 0),
            report(&[f64::NAN], 0),
            report(&[f64::INFINITY], 0),
            // The mean breach: every row at the row bound, so the mean 0.05 passes the row
            // bound and fails the mean bound.
            report(&vec![0.05; 100], 1),
            // The top-1 breach alone: every KL at zero, two flips of 100 (rate 0.98).
            report(&vec![0.0; 100], 2),
        ] {
            assert!(!rejected.passes(&bound), "{rejected:?}");
        }
        // The rounding slack: -1e-13 passes (an f64 rounding artifact), -1e-9 fails (real
        // disagreement).
        assert!(report(&[-1e-13], 0).passes(&bound));
        assert!(!report(&[-1e-9], 0).passes(&bound));
        // The split verdict: a report that fails only the top-1 floor still passes the
        // per-prompt mean-and-row part. The loop below rejects the validity and mean cases;
        // the isolated row case follows it under `spiked`.
        let floor_only = report(&vec![0.0; 100], 2);
        assert!(floor_only.passes_mean_and_row(&bound), "{floor_only:?}");
        for rejected in [
            report(&[], 0),
            report(&[f64::NAN], 0),
            report(&vec![0.05; 100], 0),
        ] {
            assert!(!rejected.passes_mean_and_row(&bound), "{rejected:?}");
        }
        // The worst-row breach: mean 0.0006 under the mean bound, one row at 0.06 past the
        // row bound - the isolated row case for both verdicts.
        let mut spiked = vec![0.0; 100];
        spiked[0] = 0.06;
        assert!(!report(&spiked, 0).passes(&bound));
        assert!(!report(&spiked, 0).passes_mean_and_row(&bound));
    }

    /// The comparison: equal rows give exactly zero KL and top-1 agreement, and a shifted row
    /// gives a positive KL that the report names at its position.
    #[test]
    fn compare_arms_marks_equal_rows_zero_and_shifted_rows_positive() {
        let row = |bump: f32| ArmRow {
            logits: [1.0, 0.5, -0.25, 0.0].iter().map(|&v| v + bump).collect(),
            token: Some(0),
        };
        let equal = compare_arms(&[row(0.0)], &[row(0.0)], 7);
        assert_eq!(
            equal,
            vec![RowKl {
                position: 7,
                kl_nats: 0.0,
                top1_equal: true,
                chunked: false
            }]
        );
        // A shifted candidate: the logits move, so the KL is positive and the greedy token
        // flips when the shift crosses the top-two gap (0.5 between the first two).
        let mut shifted = row(0.0);
        shifted.logits[1] += 1.0;
        shifted.token = Some(1);
        let compared = compare_arms(&[row(0.0)], &[shifted], 7);
        assert!(
            compared[0].kl_nats > 0.0 && !compared[0].top1_equal,
            "{compared:?}"
        );
        // A uniform bump leaves the softmax and the greedy token unchanged: KL exactly zero.
        assert_eq!(compare_arms(&[row(0.0)], &[row(3.0)], 7)[0].kl_nats, 0.0);
    }
}

/// The model-touching arms and gates, behind the native cfg like the `mac` harness of
/// `long_context`.
#[cfg(all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64"))]
mod mac {
    use super::super::decoder::{ResidentDecoder, StepMode};
    use super::super::moe::Routing;
    use super::super::state::ResidentState;
    use super::*;
    use crate::family::qwen36::host_load::HostWeights;
    use crate::family::qwen36::settings::{Activation, KvDtype, kv_dtype_name};
    use crate::family::qwen36::tiny;
    use crate::kernels::MlxKernels;

    /// The decode arm: the head is already prefilled, then every window id feeds one `step`
    /// and the row that predicts the next id is kept.
    fn decode_arm(
        decoder: &mut ResidentDecoder<'_>,
        state: &mut ResidentState,
        ids: &[u32],
        head: usize,
    ) -> Vec<ArmRow> {
        let mut rows = Vec::new();
        for &fed in &ids[head - 1..ids.len() - 1] {
            let result = decoder.step(state, fed, StepMode::Logits).unwrap();
            rows.push(ArmRow {
                logits: result.logits.unwrap(),
                token: result.token,
            });
        }
        rows
    }

    /// The prefill arm: the same head, then the same ids as one-row `prefill` chunks. Row `i`
    /// predicts `ids[head + i]`, like the decode arm.
    fn prefill_arm(
        decoder: &mut ResidentDecoder<'_>,
        state: &mut ResidentState,
        ids: &[u32],
        head: usize,
    ) -> Vec<ArmRow> {
        let mut rows = Vec::new();
        for &fed in &ids[head - 1..ids.len() - 1] {
            let result = decoder.prefill(state, &[fed], StepMode::Logits).unwrap();
            rows.push(ArmRow {
                logits: result.logits.unwrap(),
                token: result.token,
            });
        }
        rows
    }

    /// Runs the gate on one prompt. `window` is at least 2 and at most `ids.len() - 2`, so
    /// the head keeps a row beyond the window and the multi-row chunk holds more than one row.
    pub(super) fn run_gate(
        decoder: &mut ResidentDecoder<'_>,
        ids: &[u32],
        window: usize,
    ) -> KlReport {
        let len = ids.len();
        assert!(
            window >= 2 && len >= window + 2,
            "prompt of {len} ids, window {window}"
        );
        let head = len - window;
        // Three states with the same head prefill: the decode arm, the one-row-chunk prefill
        // arm, and the multi-row-chunk arm. The control is on the outputs the arms build on:
        // the prefill is deterministic, so the three head rows must agree bitwise (the token
        // alone could collide); disagreeing outputs mean the arms would not share a history
        // and the comparison below would not measure the path split.
        let mut states: Vec<_> = (0..3).map(|_| decoder.zero_state().unwrap()).collect();
        let mut controls = Vec::new();
        for state in &mut states {
            let control = decoder
                .prefill(state, &ids[..head - 1], StepMode::Logits)
                .unwrap();
            let logits = control.logits.unwrap();
            assert!(
                logits.iter().all(|value| value.is_finite()),
                "a head control row is not finite"
            );
            controls.push((control.token, logits));
        }
        let (first_token, first_logits) = &controls[0];
        for (token, logits) in &controls[1..] {
            assert_eq!(token, first_token, "the head prefill controls disagree");
            assert!(
                logits.len() == first_logits.len()
                    && logits
                        .iter()
                        .zip(first_logits)
                        .all(|(a, b)| a.to_bits() == b.to_bits()),
                "the head prefill control rows differ bitwise"
            );
        }
        let decode = decode_arm(decoder, &mut states[0], ids, head);
        let prefill = prefill_arm(decoder, &mut states[1], ids, head);
        let mut rows = compare_arms(&prefill, &decode, head);
        // The multi-row chunk: the whole window as one chunk, whose final row predicts the
        // last id. It pairs with the decode arm's last row.
        let chunked = decoder
            .prefill(&mut states[2], &ids[head - 1..len - 1], StepMode::Logits)
            .unwrap();
        let last = decode.last().unwrap();
        rows.push(RowKl {
            position: len - 1,
            kl_nats: kl_divergence(&chunked.logits.unwrap(), &last.logits)
                .unwrap_or_else(|error| panic!("the multi-row chunk row: {error}")),
            top1_equal: chunked.token == last.token,
            chunked: true,
        });
        assert_eq!(states[0].pos(), u32::try_from(len - 1).unwrap());
        assert_eq!(states[1].pos(), states[0].pos());
        assert_eq!(states[2].pos(), states[0].pos());
        KlReport { rows }
    }

    /// The receipt of one prompt: the metrics of its rows, with no verdict (the per-prompt
    /// row count is too small for the top-1 floor to mean anything; the verdict is pooled).
    fn receipt(
        prompt_tokens: usize,
        window: usize,
        kv: KvDtype,
        report: &KlReport,
    ) -> serde_json::Value {
        serde_json::json!({
            "prompt_tokens": prompt_tokens,
            "window": window,
            "compared_rows": report.rows.len(),
            "kv": kv_dtype_name(kv),
            "mean_kl_nats": report.mean_kl(),
            "max_kl_nats": report.worst().kl_nats,
            "max_kl_at": report.worst().position,
            "top1_rate": report.top1_rate(),
            "chunked_row_kl": report
                .rows
                .iter()
                .find(|row| row.chunked)
                .expect("the gate adds the chunked row")
                .kl_nats,
        })
    }

    /// The pooled verdict line the wrapper script reads: the one `passed` field of the run.
    /// Its own object, not a per-prompt receipt: the per-prompt fields (prompt tokens, the
    /// chunked row, a position inside one prompt) have no pooled meaning, and the worst
    /// position stays in the per-prompt receipts that name it.
    fn pooled_receipt(
        prompts: usize,
        kv: KvDtype,
        bound: &KlBound,
        per_prompt_ok: bool,
        passed: bool,
        report: &KlReport,
    ) -> serde_json::Value {
        serde_json::json!({
            "prompts": prompts,
            "compared_rows": report.rows.len(),
            "kv": kv_dtype_name(kv),
            "mean_kl_nats": report.mean_kl(),
            "max_kl_nats": report.worst().kl_nats,
            "top1_rate": report.top1_rate(),
            "per_prompt_bounds_passed": per_prompt_ok,
            "bound": {
                "mean": bound.mean,
                "row": bound.row,
                "top1_rate": bound.top1_rate
            },
            "passed": passed,
        })
    }

    /// Runs the gate over every prompt, prints each receipt and then the pooled verdict, and
    /// returns the pooled report with the run's single `passed` verdict (computed here once,
    /// printed by the POOLED line, asserted by the caller). Every prompt runs and every
    /// receipt prints whatever the earlier prompts did, so a wrapper reads the verdict from
    /// the `DECODE_PREFILL_KL_POOLED` line (the assert payload may print after it). A
    /// structural panic (a control mismatch, a row width) can still stop the run first, so a
    /// wrapper treats a missing POOLED line as a failure.
    fn run_all(
        decoder: &mut ResidentDecoder<'_>,
        prompts: &[Vec<u32>],
        window: usize,
        kv: KvDtype,
        bound: &KlBound,
    ) -> (KlReport, bool) {
        let (mut pooled, mut per_prompt_ok) = (Vec::new(), true);
        for prompt in prompts {
            let clamped = window.min(prompt.len() - 2);
            let report = run_gate(decoder, prompt, clamped);
            println!(
                "DECODE_PREFILL_KL {}",
                receipt(prompt.len(), clamped, kv, &report)
            );
            per_prompt_ok &= report.passes_mean_and_row(bound);
            pooled.extend(report.rows);
        }
        let pooled = KlReport { rows: pooled };
        let passed = per_prompt_ok && pooled.passes(bound);
        println!(
            "DECODE_PREFILL_KL_POOLED {}",
            pooled_receipt(prompts.len(), kv, bound, per_prompt_ok, passed, &pooled)
        );
        (pooled, passed)
    }

    /// The gate on the tiny KV8 model (the Q8 cache), the portable-form member of the Mac plain
    /// suite. The pinned bound holds the first Mac run's measured tiny numerics with headroom
    /// (pooled over 8 prompts / 72 rows: mean 5.5e-8, worst row 1.0e-6, top-1 rate 1.0): about an
    /// order above the measurement and over three orders below the real-weights bound family
    /// (Q36_KL_MEAN 0.004). A path split — a decode-side quantize change that misses the prefill
    /// side — lands orders past it.
    #[test]
    fn decode_vs_prefill_kl_passes_on_the_tiny_kv8_model() {
        let _serial = crate::native_test_lock();
        let ckpt = tiny::checkpoint(&tiny::TINY_KV8);
        let host =
            HostWeights::load(&ckpt, &tiny::TINY_KV8, &tiny::config(&tiny::TINY_KV8)).unwrap();
        let kv = KvDtype::Q8;
        let mut decoder = ResidentDecoder::new_with_kv(
            MlxKernels::try_new().unwrap(),
            &ckpt,
            host,
            128,
            Activation::Bf16,
            kv,
        )
        .unwrap();
        decoder.routing = Routing::Host;
        let bound = KlBound {
            mean: 1e-6,
            row: 1e-5,
            top1_rate: 0.99,
        };
        let prompts: Vec<Vec<u32>> = [10usize, 12, 14, 16, 20, 24, 28, 32]
            .into_iter()
            .enumerate()
            .map(|(slot, len)| (0..len).map(|pos| ((slot + pos) % 8) as u32).collect())
            .collect();
        let (pooled, passed) = run_all(&mut decoder, &prompts, 8, kv, &bound);
        assert!(passed, "{pooled:?}");
    }

    /// The real-weights gate: the standard command of the module doc. It loads the model fresh.
    #[test]
    #[ignore = "real weights: Q36_CKPT_DIR required; correctness only"]
    fn real_weights_decode_vs_prefill_kl() {
        let _serial = crate::native_test_lock();
        let dir = super::super::real::env_path("Q36_CKPT_DIR");
        let bytes = std::fs::read(dir.join("tokenizer.json")).unwrap();
        let tokenizer = crate::tokenizer::Tokenizer::from_bytes(&bytes).unwrap();
        let window = env_usize("Q36_KL_WINDOW", 32);
        let bound = KlBound {
            mean: env_f64("Q36_KL_MEAN", 0.004),
            row: env_f64("Q36_KL_MAX", 0.05),
            top1_rate: env_f64("Q36_KL_TOP1", 0.99),
        };
        let activation = env_setting(
            "Q36_ACTIVATION",
            Activation::Bf16,
            super::super::real::parse_activation,
        );
        let kv = env_setting(
            "Q36_KV_DTYPE",
            KvDtype::Q8,
            super::super::real::parse_kv_dtype,
        );
        // F32 decoders do not enable the BF16 casts, so that cache pair is not a valid cell
        // (the rule of `batch_tests.rs`); refuse it here instead of failing deep in a kernel.
        assert!(
            !(activation == Activation::F32 && kv == KvDtype::Bf16),
            "the F32-activation/BF16-cache pair is not a valid cell"
        );
        let prompts: Vec<_> = (0..8)
            .map(|slot| {
                let text = format!(
                    "Prompt {slot}. {}",
                    "Explain how ownership keeps Rust references valid. ".repeat(slot + 1)
                );
                tokenizer.encode(&text).unwrap()
            })
            .collect();
        // Every prompt must clear the minimum gate shape (two head rows plus a two-row
        // window); the per-prompt window clamps to the prompt length.
        assert!(
            prompts
                .iter()
                .all(|prompt| prompt.len() >= 4 && prompt.len() < 512),
            "prompt lengths: {:?}",
            prompts.iter().map(Vec::len).collect::<Vec<_>>()
        );
        let real = super::super::real::open(&dir);
        let host = HostWeights::load(
            &real.ckpt,
            &crate::family::spec::QWEN3_6_35B_A3B,
            &real.config,
        )
        .unwrap();
        let mut decoder = ResidentDecoder::new_with_kv(
            MlxKernels::try_new().unwrap(),
            &real.ckpt,
            host,
            512,
            activation,
            kv,
        )
        .unwrap();
        decoder.routing = Routing::Host;
        let (pooled, passed) = run_all(&mut decoder, &prompts, window, kv, &bound);
        assert!(passed, "{pooled:?}");
    }

    /// `name` as a usize, or `default`; a value that is not a number fails the run.
    fn env_usize(name: &str, default: usize) -> usize {
        std::env::var(name).map_or(default, |value| {
            value
                .parse()
                .unwrap_or_else(|_| panic!("{name} must be a usize: {value}"))
        })
    }

    /// `name` as an f64, or `default`; a value that is not a number fails the run.
    fn env_f64(name: &str, default: f64) -> f64 {
        std::env::var(name).map_or(default, |value| {
            value
                .parse()
                .unwrap_or_else(|_| panic!("{name} must be an f64: {value}"))
        })
    }

    /// `name` as a settings word through the house parser of `real.rs`, or `default` when
    /// unset; a bad or non-UTF-8 value fails the run. The house parser keeps the env knob and
    /// the config member from drifting apart.
    fn env_setting<T>(name: &str, default: T, parse: fn(Option<&str>) -> Result<T, String>) -> T {
        match std::env::var_os(name) {
            None => default,
            Some(value) => {
                let text = value
                    .to_str()
                    .unwrap_or_else(|| panic!("{name} must be UTF-8"));
                parse(Some(text)).unwrap_or_else(|error| panic!("{error}"))
            }
        }
    }
}
