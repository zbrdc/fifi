//! `compile_changes_nothing_in_the_logits`: the compiled decode blocks against the uncompiled
//! ones on the REAL checkpoint (a test-only measurement gate; it never runs in production).
//!
//! The tiny-model parity tests cannot see a numerics drift of the whole 40-layer stack, and the
//! real-checkpoint parity gates compare against the F32 host reference, not against the engine
//! itself. This test therefore runs the same 8 greedy decode steps twice on the real checkpoint —
//! once with `QWEN36_G2A_COMPILE` unset (the control) and once with `QWEN36_G2A_COMPILE=1` — and
//! writes the logits rows `[8, 248320]` as one f32 file per run into `Q36_OUT_DIR`, so the
//! coordinator compares the two files with `compare_logit_files` (two decoders in one process do
//! not fit in memory, so each run is its own process). The 922-token tool prompt (the `ids` field
//! of `IDS_FILE` plus the `suffix_ids` of `REF_JSON`, the same reader as `tool_forward`) is
//! prefilled through the serving chunk of the shipped config; the 8 steps then run in
//! `StepMode::Logits` with a host argmax between them, exactly the greedy chain of the
//! decode-timing harness.
//!
//! `compare_logit_files` is pure and carries Linux unit tests with hand-derived expectations
//! (bit equality, the maximum absolute difference, the top-5 overlap per row, and the
//! Kullback-Leibler divergence `KL(a||b)` per row over the softmax of each row).
//!
//! | Variable | Meaning | Default |
//! | --- | --- | --- |
//! | `Q36_CKPT_DIR` | the pinned checkpoint directory | required |
//! | `IDS_FILE` | the fifi render dump with the 919 prompt ids | required |
//! | `REF_JSON` | the reference with the 3 `suffix_ids` and the token counts | required |
//! | `Q36_OUT_DIR` | the output directory; must not exist, created at the start of the run | required |
//! | `Q36_ACTIVATION` | the decode activation dtype | `f32` |
//! | `QWEN36_G2A_COMPILE` | the compile switch of the run: `1` names the file `on` | unset (`off`) |
//!
//! The exact Mac commands (placeholders only), one process per run, a fresh `Q36_OUT_DIR` each:
//!
//! ```text
//! Q36_CKPT_DIR=<CKPT_DIR> IDS_FILE=<IDS_FILE> REF_JSON=<REF_JSON> Q36_OUT_DIR=<OUT_DIR_OFF> \
//! Q36_ACTIVATION=bf16 \
//! cargo test --release --features native-mlx --lib \
//!   family::qwen36::resident::compile_logits::compile_changes_nothing_in_the_logits -- --ignored --exact --nocapture
//! Q36_CKPT_DIR=<CKPT_DIR> IDS_FILE=<IDS_FILE> REF_JSON=<REF_JSON> Q36_OUT_DIR=<OUT_DIR_ON> \
//! Q36_ACTIVATION=bf16 QWEN36_G2A_COMPILE=1 \
//! cargo test --release --features native-mlx --lib \
//!   family::qwen36::resident::compile_logits::compile_changes_nothing_in_the_logits -- --ignored --exact --nocapture
//! ```
//!
//! The f32 pair is the same two commands with `Q36_ACTIVATION=f32`. The comparison of the two
//! files runs on the Mac of the run or on Linux:
//!
//! ```text
//! cargo test --release --features native-mlx --lib \
//!   family::qwen36::resident::compile_logits -- --nocapture --exact compare_logit_files_reads_two_files_and_compares_bytes
//! # or one line of the comparison on any files:
//! Q36_COMPARE_A=<OUT_DIR_OFF>/logits-compile-off.f32 Q36_COMPARE_B=<OUT_DIR_ON>/logits-compile-on.f32 \
//! cargo test --release --lib family::qwen36::resident::compile_logits::print_a_comparison -- --ignored --exact --nocapture
//! ```
//!
//! Mutation checks of the comparison (run on Linux, then revert exactly): make
//! `compare_logit_rows` compute the UNWEIGHTED sum of log ratios instead of `KL(a||b)` — the
//! hand-derived row must fail; make `top_k` take the `k - 1` largest values, or sort ascending —
//! the overlap expectations must fail (row 3 of the hand-derived set pins the descending rule:
//! an ascending top-5 gives it an overlap of 5 against the expected 4); pin `bits_equal` to
//! `true` — the file test must fail; disable the partial-row check — the width test must fail.
//! The Mac test's own mutation check is the one of `compiled.rs` (`in.values[25]` read for
//! `in.values[24]` in the delta closure body): with `QWEN36_G2A_COMPILE=1` the logits rows must
//! stop matching the control run.

use std::path::Path;

/// The vocabulary of the Q36 head, the row width of the logits files (the `vocab` of
/// `parity.rs`, 248,320 F32 values per row).
pub(super) const VOCAB: usize = 248_320;

/// The comparison of two logits files: the metrics the gate reads off the two runs.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct LogitComparison {
    /// Whether the two files hold the same bits (the strongest possible result: fusion changed
    /// nothing at all).
    pub(super) bits_equal: bool,
    /// The number of rows each file holds.
    pub(super) rows: usize,
    /// The row width (the vocabulary).
    pub(super) vocab: usize,
    /// The largest absolute difference over every row and column.
    pub(super) max_abs_diff: f32,
    /// For each row, how many of the 5 largest logits of `a` are among the 5 largest of `b`
    /// (0 to 5).
    pub(super) top5_overlap: Vec<usize>,
    /// For each row, `KL(a||b)` over the softmax of the two rows (0 when the rows are equal;
    /// f64 accumulation).
    pub(super) kl_divergence: Vec<f64>,
}

/// Reads one logits file as little-endian f32 values, rejecting a partial value.
fn read_f32s(path: &Path) -> Result<Vec<f32>, String> {
    let bytes = std::fs::read(path).map_err(|error| format!("read {}: {error}", path.display()))?;
    if bytes.len() % 4 != 0 {
        return Err(format!(
            "{} holds {} bytes: little-endian f32 values need a multiple of 4",
            path.display(),
            bytes.len()
        ));
    }
    Ok(bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|chunk| f32::from_le_bytes(*chunk))
        .collect())
}

/// The indices of the `k` largest values, the lowest index first on ties (the rule of
/// `parity::top5`).
fn top_k(logits: &[f32], k: usize) -> Vec<usize> {
    let mut ranked: Vec<(usize, f32)> = logits.iter().enumerate().map(|(i, v)| (i, *v)).collect();
    ranked.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    ranked.into_iter().take(k).map(|r| r.0).collect()
}

/// `KL(p||q)` over the softmax of two logits rows, accumulated in f64: with `p` and `q` the
/// softmax probabilities of `a` and `b`, the value is `sum_i p_i * ln(p_i / q_i)`; the softmax
/// is computed against the row maximum, so equal rows give exactly 0.
pub(super) fn kl_divergence(a: &[f32], b: &[f32]) -> Result<f64, String> {
    let (max_a, max_b) = (
        a.iter().copied().fold(f32::MIN, f32::max),
        b.iter().copied().fold(f32::MIN, f32::max),
    );
    let (sum_a, sum_b) = (
        a.iter().map(|v| f64::from(v - max_a).exp()).sum::<f64>(),
        b.iter().map(|v| f64::from(v - max_b).exp()).sum::<f64>(),
    );
    if !(sum_a > 0.0 && sum_b > 0.0) {
        return Err("a logits row is not finite, so its softmax has no mass".to_string());
    }
    let mut kl = 0.0f64;
    for (&x, &y) in a.iter().zip(b) {
        let (p, q) = (
            f64::from(x - max_a).exp() / sum_a,
            f64::from(y - max_b).exp() / sum_b,
        );
        kl += p * (p / q).ln();
    }
    Ok(kl)
}

/// Compares two logits buffers of `rows * vocab` values each. Both buffers must hold the same
/// number of whole rows, and `vocab` must be positive. Every metric is pure.
pub(super) fn compare_logit_rows(
    a: &[f32],
    b: &[f32],
    vocab: usize,
) -> Result<LogitComparison, String> {
    if vocab == 0 {
        return Err("the row width must be positive".to_string());
    }
    if a.len() != b.len() {
        return Err(format!(
            "the buffers hold {} and {} values, expected equal counts",
            a.len(),
            b.len()
        ));
    }
    if !a.len().is_multiple_of(vocab) {
        return Err(format!(
            "the buffers hold {} values, not a whole number of {}-wide rows",
            a.len(),
            vocab
        ));
    }
    let bits_equal = a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits());
    let max_abs_diff = a
        .iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0f32, f32::max);
    let mut top5_overlap = Vec::new();
    let mut row_kl = Vec::new();
    for (row_a, row_b) in a.chunks(vocab).zip(b.chunks(vocab)) {
        let (top_a, top_b) = (top_k(row_a, 5), top_k(row_b, 5));
        top5_overlap.push(top_a.iter().filter(|id| top_b.contains(id)).count());
        row_kl.push(kl_divergence(row_a, row_b)?);
    }
    Ok(LogitComparison {
        bits_equal,
        rows: a.len() / vocab,
        vocab,
        max_abs_diff,
        top5_overlap,
        kl_divergence: row_kl,
    })
}

/// Compares two logits files of `[rows, VOCAB]` little-endian f32 values (the output of the
/// ignored Mac test): bit equality, the maximum absolute difference, the top-5 overlap per row,
/// and `KL(a||b)` per row. The row width is the Q36 vocabulary, so the files must hold a whole
/// number of rows.
pub(super) fn compare_logit_files(a: &Path, b: &Path) -> Result<LogitComparison, String> {
    compare_logit_rows(&read_f32s(a)?, &read_f32s(b)?, VOCAB)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Writes one logits file of `rows` rows (little-endian f32).
    fn write(path: &Path, rows: &[Vec<f32>]) {
        let mut bytes = Vec::new();
        for row in rows {
            for value in row {
                bytes.extend_from_slice(&value.to_le_bytes());
            }
        }
        std::fs::write(path, bytes).unwrap();
    }

    /// A temp path with a unique name for this test.
    fn temp(name: &str) -> std::path::PathBuf {
        let mut path = std::env::temp_dir();
        path.push(format!("q36-compile-logits-{}-{name}", std::process::id()));
        path
    }

    /// The four hand-derived rows of the metric tests, at vocabulary 8:
    /// - row 0 is equal in both files: bit equality, no difference, full overlap, KL 0;
    /// - row 1 swaps the two leading logits of `[2, 1, 0, 0, 0, 0, 0, 0]`: the maximum absolute
    ///   difference is 1, the top 5 are the same five ids, and with `D = e^2 + e + 6` the softmax
    ///   gives `KL = (p0 - p1) * ln(p0/p1) = (e^2 - e)/D = 0.2899780392965615...`, because the
    ///   two rows share one denominator (equal logit multisets) and the log ratio of the leading
    ///   entries is `ln(e^(2-1)) = 1`;
    /// - row 2 puts `[4, 3, 2, 1]` at the front of `a` and at the back of `b`: only ids 0 and 4
    ///   stay in both top 5, and with `D4 = e^4 + e^3 + e^2 + e + 4` the divergence is
    ///   `(4e^4 + 3e^3 + 2e^2 + e - 4 - 3 - 2 - 1)/D4 = 3.222686136388098...`, again over one
    ///   shared denominator, where each column contributes `p_i * (a_i - b_i)`;
    /// - row 3 swaps the logits 5 and 0 between columns 4 and 5 of `[9, 8, 7, 6, 5, 0, 0, 0]`:
    ///   the descending top 5 of `a` keeps column 4 and that of `b` keeps column 5, so the
    ///   overlap is 4 (an ascending top-5 would give 5 — this row is what pins the rule), the
    ///   maximum absolute difference is 5, and with `D3 = 1 + e^-1 + e^-2 + e^-3 + e^-4 + 3e^-9`
    ///   the divergence is `5(e^-4 - e^-9)/D3 = 0.057874823144511645...` (one shared denominator,
    ///   columns 4 and 5 contribute `±5`).
    fn hand_rows() -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
        let row0 = vec![0.5, -0.25, 0.125, 0.0, 1.5, -2.0, 0.75, 0.375];
        let row1_a = vec![2.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let row1_b = vec![1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let row2_a = vec![4.0, 3.0, 2.0, 1.0, 0.0, 0.0, 0.0, 0.0];
        let row2_b = vec![0.0, 0.0, 0.0, 0.0, 4.0, 3.0, 2.0, 1.0];
        let row3_a = vec![9.0, 8.0, 7.0, 6.0, 5.0, 0.0, 0.0, 0.0];
        let row3_b = vec![9.0, 8.0, 7.0, 6.0, 0.0, 5.0, 0.0, 0.0];
        (
            vec![row0.clone(), row1_a, row2_a, row3_a],
            vec![row0, row1_b, row2_b, row3_b],
        )
    }

    #[test]
    fn the_metrics_match_the_hand_derived_values() {
        let (a, b) = hand_rows();
        let flat_a: Vec<f32> = a.iter().flatten().copied().collect();
        let flat_b: Vec<f32> = b.iter().flatten().copied().collect();
        let got = compare_logit_rows(&flat_a, &flat_b, 8).unwrap();
        assert_eq!(got.rows, 4, "four rows of vocabulary 8");
        assert_eq!(got.vocab, 8);
        assert!(!got.bits_equal, "rows 1 to 3 differ");
        assert_eq!(got.max_abs_diff, 5.0, "row 3 differs by 5 at column 4");
        assert_eq!(got.top5_overlap, vec![5, 5, 2, 4], "per-row top-5 overlap");
        // Row 0: identical rows, exactly 0. Row 1: (e^2 - e)/(e^2 + e + 6).
        assert_eq!(got.kl_divergence[0], 0.0, "equal rows give exactly 0");
        let e = std::f64::consts::E;
        let row1 = (e * e - e) / (e * e + e + 6.0);
        assert!(
            (got.kl_divergence[1] - row1).abs() <= 1e-12,
            "row 1 KL {} against the hand-derived {row1}",
            got.kl_divergence[1]
        );
        let row2 = (4.0 * e.powi(4) + 3.0 * e.powi(3) + 2.0 * e * e + e - 10.0)
            / (e.powi(4) + e.powi(3) + e * e + e + 4.0);
        assert!(
            (got.kl_divergence[2] - row2).abs() <= 1e-12,
            "row 2 KL {} against the hand-derived {row2}",
            got.kl_divergence[2]
        );
        let row3 = 5.0 * (e.powi(-4) - e.powi(-9))
            / (1.0 + e.powi(-1) + e.powi(-2) + e.powi(-3) + e.powi(-4) + 3.0 * e.powi(-9));
        assert!(
            (got.kl_divergence[3] - row3).abs() <= 1e-12,
            "row 3 KL {} against the hand-derived {row3}",
            got.kl_divergence[3]
        );
        assert!(
            (row1 - 0.2899780392965615).abs() <= 1e-15,
            "the decimal of the comment"
        );
        assert!(
            (row2 - 3.222686136388098).abs() <= 1e-15,
            "the decimal of the comment"
        );
        assert!(
            (row3 - 0.057874823144511645).abs() <= 1e-15,
            "the decimal of the comment"
        );
    }

    #[test]
    fn equal_buffers_are_bit_equal() {
        let row: Vec<f32> = (0..8).map(|i| i as f32 * 0.25 - 1.0).collect();
        let got = compare_logit_rows(&row, &row, 8).unwrap();
        assert_eq!(
            got,
            LogitComparison {
                bits_equal: true,
                rows: 1,
                vocab: 8,
                max_abs_diff: 0.0,
                top5_overlap: vec![5],
                kl_divergence: vec![0.0],
            },
            "one row against itself: every metric is the identity"
        );
    }

    #[test]
    fn wrong_widths_and_lengths_are_errors() {
        let row: Vec<f32> = vec![0.0; 8];
        assert!(
            compare_logit_rows(&row, &row, 0).is_err(),
            "a zero width is an error"
        );
        assert!(
            compare_logit_rows(&row, &row[..7], 8).is_err(),
            "different lengths are an error"
        );
        assert!(
            compare_logit_rows(&row[..6], &row[..6], 8).is_err(),
            "a partial row is an error"
        );
    }

    #[test]
    fn compare_logit_files_reads_two_files_and_compares_bytes() {
        // One full-width row against itself: the files read back bit-identical, every metric is
        // the identity, and the row width of the files is the real vocabulary.
        let mut row = vec![0.0f32; VOCAB];
        for (i, slot) in row.iter_mut().enumerate().take(6) {
            *slot = (6 - i) as f32 * 0.5;
        }
        let (path_a, path_b) = (temp("a.f32"), temp("b.f32"));
        write(&path_a, &[row.clone()]);
        write(&path_b, &[row.clone()]);
        let got = compare_logit_files(&path_a, &path_b).unwrap();
        assert_eq!(got.vocab, VOCAB, "the file comparison uses the real width");
        assert_eq!(got.rows, 1);
        assert!(got.bits_equal, "a file against itself is bit-identical");
        assert_eq!(got.max_abs_diff, 0.0);
        assert_eq!(got.top5_overlap, vec![5]);
        assert_eq!(got.kl_divergence, vec![0.0]);
        std::fs::remove_file(&path_a).unwrap();
        std::fs::remove_file(&path_b).unwrap();
        // One changed value at a known index: the maximum difference is that change, the bits
        // differ, and the same leading ids keep the top-5 overlap whole.
        row[VOCAB - 1] = 2.5;
        let mut changed = row.clone();
        changed[VOCAB - 1] = 1.5;
        write(&path_a, &[row]);
        write(&path_b, &[changed]);
        let got = compare_logit_files(&path_a, &path_b).unwrap();
        assert_eq!(got.rows, 1);
        assert!(!got.bits_equal);
        assert_eq!(got.max_abs_diff, 1.0, "2.5 against 1.5 at the last column");
        assert_eq!(got.top5_overlap, vec![5], "the same leading ids");
        std::fs::remove_file(&path_a).unwrap();
        std::fs::remove_file(&path_b).unwrap();
        // A partial f32 value is an error, and so is a file that is not a whole row.
        std::fs::write(&path_a, vec![0u8; 5]).unwrap();
        assert!(compare_logit_files(&path_a, &path_b).is_err());
        std::fs::write(&path_a, vec![0u8; VOCAB * 4 - 2]).unwrap();
        assert!(compare_logit_files(&path_a, &path_b).is_err());
        std::fs::remove_file(&path_a).unwrap();
    }
}

/// One comparison printed as JSON for the gate report (the pure metrics over two files).
#[cfg(all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64"))]
#[test]
#[ignore = "set Q36_COMPARE_A and Q36_COMPARE_B to the two logits files of the runs"]
fn print_a_comparison() {
    let a = std::env::var("Q36_COMPARE_A").unwrap_or_else(|_| panic!("set Q36_COMPARE_A"));
    let b = std::env::var("Q36_COMPARE_B").unwrap_or_else(|_| panic!("set Q36_COMPARE_B"));
    let got =
        compare_logit_files(Path::new(&a), Path::new(&b)).unwrap_or_else(|error| panic!("{error}"));
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "a": a, "b": b,
            "bits_equal": got.bits_equal, "rows": got.rows, "vocab": got.vocab,
            "max_abs_diff": got.max_abs_diff, "top5_overlap_per_row": got.top5_overlap,
            "kl_a_b_per_row": got.kl_divergence,
        }))
        .unwrap()
    );
}

/// The lowest index of the largest value of `row` (the greedy rule of the decode-timing harness).
#[cfg(all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64"))]
fn argmax_lowest(row: &[f32]) -> u32 {
    let mut best = 0usize;
    for (i, &value) in row.iter().enumerate() {
        if value > row[best] {
            best = i;
        }
    }
    best as u32
}

/// Writes the logits rows `[8, VOCAB]` of the run as one f32 file into `Q36_OUT_DIR`, with the
/// run's small report beside it: 8 greedy decode steps after the 922-token tool prompt, with the
/// compile switch of THIS process (unset names `off`, `1` names `on`). Two decoders in one
/// process do not fit in memory, so the off and the on run are two processes and the reader
/// of the two output files compares them. See the module doc for the commands.
#[cfg(all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64"))]
#[test]
#[ignore = "needs the real checkpoint on the Mac; run with --ignored, once per switch value"]
fn compile_changes_nothing_in_the_logits() {
    use super::decoder::{ResidentDecoder, StepMode};
    use super::parity::{CONFIG_SHA256, sha256};
    use super::real::{activation_from_env, env_path, open};
    use super::tool_forward::read_ids;
    use crate::{
        family::{qwen36::host_load::HostWeights, spec::QWEN3_6_35B_A3B},
        kernels::MlxKernels,
    };

    let _serial = crate::native_test_lock();
    let ckpt_dir = env_path("Q36_CKPT_DIR");
    let out = super::tool_forward::out_dir_must_be_new();
    let out_dir = super::tool_forward::create_out_dir(&out);
    let ids = {
        let (ids_path, ref_path) = (env_path("IDS_FILE"), env_path("REF_JSON"));
        let ids_value: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&ids_path).unwrap()).unwrap();
        let ref_value: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&ref_path).unwrap()).unwrap();
        read_ids(&ids_value, &ref_value)
    };
    assert_eq!(ids.len(), 922, "the tool prompt of the finding");
    let activation = activation_from_env();
    let compiled = super::compiled::requested();
    let mode = if compiled { "on" } else { "off" };

    // The load of `tool_forward`: the pinned config and the product constructor of the
    // `Q36_ACTIVATION` lever, with clean compile counters (with the switch on, the prefill and
    // the warm-up trace each block kind once and the 8 steps hit the MLX cache).
    let real = open(&ckpt_dir);
    assert_eq!(
        sha256(&real.config),
        CONFIG_SHA256,
        "config.json differs from the pinned checkpoint"
    );
    let host = HostWeights::load(&real.ckpt, &QWEN3_6_35B_A3B, &real.config).unwrap();
    let mut decoder = ResidentDecoder::new_with_activation(
        MlxKernels::try_new().unwrap(),
        &real.ckpt,
        host,
        super::tool_forward::CONTEXT_LIMIT,
        activation,
    )
    .unwrap();
    decoder.kernels.reset_compile_counters();
    decoder.kernels.clear_compile_cache().unwrap();
    let mut state = decoder.zero_state().unwrap();

    // Prefill the first 921 ids through the serving chunk of the shipped config, exactly the
    // piece loop of `tool_forward::serve_run`; the last id stays pending for the first logits
    // step, so the 8 rows are the distributions AFTER the whole 922-token prompt.
    let chunk = super::tool_forward::serve_chunk_setting(activation);
    decoder.set_chunk_rows(chunk).unwrap();
    let width = usize::try_from(chunk).unwrap_or(1);
    for rows in ids[..921].chunks(width.max(1)) {
        if let &[token] = rows {
            decoder.step(&mut state, token, StepMode::Prompt).unwrap();
        } else {
            decoder.prefill(&mut state, rows, StepMode::Prompt).unwrap();
        }
    }

    // 8 greedy decode steps in `StepMode::Logits`: the full head row of each step is the file
    // row, and the host argmax feeds the next step.
    let started = std::time::Instant::now();
    let mut token = ids[921];
    let mut rows: Vec<f32> = Vec::with_capacity(8 * VOCAB);
    let mut greedy = Vec::with_capacity(8);
    for _ in 0..8 {
        let got = decoder.step(&mut state, token, StepMode::Logits).unwrap();
        let row = got.logits.expect("the logits step returns the head row");
        assert_eq!(row.len(), VOCAB, "the head row is the vocabulary");
        greedy.push(token);
        token = argmax_lowest(&row);
        rows.extend(row);
    }
    let step_seconds = started.elapsed().as_secs_f64();
    let logits_path = out_dir.join(format!("logits-compile-{mode}.f32"));
    std::fs::write(
        &logits_path,
        rows.iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>(),
    )
    .unwrap();
    let report = serde_json::json!({
        "tool": "fifi compile_changes_nothing_in_the_logits",
        "compile": compiled,
        "activation_dtype": crate::family::qwen36::settings::activation_name(activation),
        "compile_counters": super::compiled::report(&decoder.kernels)["compile_counters"].clone(),
        "prompt_ids": ids.len(), "steps": 8, "greedy_ids": greedy,
        "step_seconds": step_seconds,
        "logits_file": logits_path.display().to_string(),
        "compare_with": "compare_logit_files of this module, or the print_a_comparison test",
    });
    std::fs::write(
        out_dir.join(format!("logits-compile-{mode}.json")),
        serde_json::to_vec_pretty(&report).unwrap(),
    )
    .unwrap();
    println!("{}", serde_json::to_string_pretty(&report).unwrap());
}
