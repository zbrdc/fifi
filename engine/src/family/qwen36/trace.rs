//! An optional per-layer, per-step dump of the decode hidden row, a diagnostic for locating the
//! first layer where two runs that differ only in chunk size diverge.
//!
//! `Q36_LAYER_TRACE_DIR` controls it. Unset or empty means off. A value that is not an existing
//! directory means off, with one warning on stderr that names the path, so a typo cannot break
//! decoding. Any other value means on: each non-prompt step then writes one little-endian F32
//! file per layer, `step{S:04}_layer{L:02}.f32` (the dump format that the `f32s` reader of
//! `parity` reads). The row dumped is the residual after the layer's mixer, before its MoE.
//! The check does not probe write access: an existing directory that is unwritable, full, or
//! removed later makes the step fail with `Backend` and poisons the session until reset.
//!
//! The step counter is process-wide: every decoder, session, request and warm-up step of the
//! process advances it. Files of two runs therefore line up only between fresh processes that
//! run the same request sequence. The variable is read once, on the first non-prompt step.
//!
//! The CUDA engine adds a second counter for prefill pieces (the chunk-size diagnostic needs
//! the piece rows too, and a piece is not a decode step): each prefill piece claims
//! `begin_prefill` once and writes `prefill{P:04}_layer{L:02}.f32`, holding the last
//! (up to) five rows of the piece's post-mixer residual, row-major. Keeping the two counters
//! separate means the decode files of a `[295,5]` run and a `[300]` run still align from
//! `step0000`, whatever the number of prefill pieces. Under the trace, the CUDA decoder runs
//! EAGER (no graph capture): a readback cannot live inside a captured graph, and a traced
//! diagnostic run does not need one.
//!
//! When off, a step costs one cached `OnceLock` load and each layer one `Option` check; the trace
//! adds no graph operation, readback, write or counter increment.
//!
//! The directory rule, file names and file format are portable and carry Linux tests; only the
//! readback of the row from the MLX graph is Mac-only.

use crate::EngError;
use std::ffi::OsString;
use std::path::PathBuf;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU32, Ordering};

/// The output directory of one trace and its step counter.
pub(super) struct LayerTrace {
    dir: Option<PathBuf>,
    steps: AtomicU32,
    /// The prefill-piece counter of the CUDA chunk-size diagnostic (the module doc).
    prefill_pieces: AtomicU32,
}

/// Resolves the text of `Q36_LAYER_TRACE_DIR`: `Ok(None)` for unset or empty (off), `Ok(Some)`
/// for an existing directory (on), and the warning that names the path when it is not one.
fn resolve(raw: Option<OsString>) -> Result<Option<PathBuf>, String> {
    let Some(path) = raw.filter(|text| !text.is_empty()).map(PathBuf::from) else {
        return Ok(None);
    };
    if path.is_dir() {
        Ok(Some(path))
    } else {
        Err(format!(
            "q36: Q36_LAYER_TRACE_DIR {} is not a directory; the layer trace is off",
            path.display()
        ))
    }
}

/// The dump format: the values as little-endian F32 bytes. The one encoder of the trace files
/// and of the parity dumps (`within_mode::write_f32s`), which the `f32s` reader of `parity`
/// reads back.
pub(super) fn f32_le_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

impl LayerTrace {
    /// Builds the trace from the text of `Q36_LAYER_TRACE_DIR` (see the module doc for the
    /// rule). A value that is not a directory prints its warning once and leaves the trace off.
    /// Taking the text instead of reading the environment keeps the rule testable, because
    /// edition 2024 makes `std::env::set_var` unsafe and the crate denies `unsafe_code`.
    pub(super) fn from_env_value(raw: Option<OsString>) -> Self {
        let dir = resolve(raw).unwrap_or_else(|warning| {
            eprintln!("{warning}");
            None
        });
        Self {
            dir,
            steps: AtomicU32::new(0),
            prefill_pieces: AtomicU32::new(0),
        }
    }

    /// Claims the index of the next traced step, or `None` when the trace is off. An off trace
    /// claims nothing and touches no file.
    pub(super) fn begin_step(&self) -> Option<u32> {
        self.dir
            .is_some()
            .then(|| self.steps.fetch_add(1, Ordering::Relaxed))
    }

    /// Writes one row as little-endian F32 to `step{step:04}_layer{layer:02}.f32` in the trace
    /// directory. An off trace writes nothing. A write error names the step and the layer.
    pub(super) fn write_row(&self, step: u32, layer: usize, row: &[f32]) -> Result<(), EngError> {
        let Some(dir) = &self.dir else {
            return Ok(());
        };
        std::fs::write(
            dir.join(format!("step{step:04}_layer{layer:02}.f32")),
            f32_le_bytes(row),
        )
        .map_err(|error| EngError::Backend {
            operation: "qwen36 layer trace",
            message: format!("write step {step} layer {layer}: {error}"),
        })
    }

    /// Claims the index of the next traced prefill piece, or `None` when the trace is off.
    pub(super) fn begin_prefill(&self) -> Option<u32> {
        self.dir
            .is_some()
            .then(|| self.prefill_pieces.fetch_add(1, Ordering::Relaxed))
    }

    /// Writes the rows of one prefill piece as little-endian F32 to
    /// `prefill{piece:04}_layer{layer:02}.f32`, row-major. An off trace writes nothing.
    pub(super) fn write_prefill(
        &self,
        piece: u32,
        layer: usize,
        rows: &[f32],
    ) -> Result<(), EngError> {
        let Some(dir) = &self.dir else {
            return Ok(());
        };
        std::fs::write(
            dir.join(format!("prefill{piece:04}_layer{layer:02}.f32")),
            f32_le_bytes(rows),
        )
        .map_err(|error| EngError::Backend {
            operation: "qwen36 layer trace",
            message: format!("write prefill piece {piece} layer {layer}: {error}"),
        })
    }

    /// Writes the logits row of one traced step as little-endian F32 to
    /// `step{step:04}_logits.f32`. An off trace writes nothing.
    pub(super) fn write_logits(&self, step: u32, row: &[f32]) -> Result<(), EngError> {
        let Some(dir) = &self.dir else {
            return Ok(());
        };
        std::fs::write(
            dir.join(format!("step{step:04}_logits.f32")),
            f32_le_bytes(row),
        )
        .map_err(|error| EngError::Backend {
            operation: "qwen36 layer trace",
            message: format!("write step {step} logits: {error}"),
        })
    }
}

/// The variable that turns the trace on.
const TRACE_DIR_VAR: &str = "Q36_LAYER_TRACE_DIR";

/// The process trace: `Q36_LAYER_TRACE_DIR` is read once, on the first use.
fn process_trace() -> &'static LayerTrace {
    static TRACE: OnceLock<LayerTrace> = OnceLock::new();
    TRACE.get_or_init(|| LayerTrace::from_env_value(std::env::var_os(TRACE_DIR_VAR)))
}

/// Whether the trace is on. The CUDA decoder also uses this to force EAGER steps (the module
/// doc: a readback cannot live inside a captured graph). Reads the variable once.
pub(in crate::family::qwen36) fn trace_enabled() -> bool {
    process_trace().dir.is_some()
}

/// Claims the next traced step index, or `None` when the trace is off.
pub(in crate::family::qwen36) fn begin_step() -> Option<u32> {
    process_trace().begin_step()
}

/// Claims the next traced prefill-piece index, or `None` when the trace is off.
pub(in crate::family::qwen36) fn begin_prefill() -> Option<u32> {
    process_trace().begin_prefill()
}

/// Writes one decode-step row (`step{S:04}_layer{L:02}.f32`); a no-op when off.
pub(in crate::family::qwen36) fn write_step_row(
    step: u32,
    layer: usize,
    row: &[f32],
) -> Result<(), EngError> {
    process_trace().write_row(step, layer, row)
}

/// Writes one prefill piece's last rows (`prefill{P:04}_layer{L:02}.f32`); a no-op when off.
pub(in crate::family::qwen36) fn write_prefill_rows(
    piece: u32,
    layer: usize,
    rows: &[f32],
) -> Result<(), EngError> {
    process_trace().write_prefill(piece, layer, rows)
}

/// Writes the logits row of a traced decode step (`step{S:04}_logits.f32`), the top-2 margin
/// evidence of the chunk-split diagnostic; a no-op when off.
pub(in crate::family::qwen36) fn write_step_logits(step: u32, row: &[f32]) -> Result<(), EngError> {
    process_trace().write_logits(step, row)
}

#[cfg(all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64"))]
pub(in crate::family::qwen36) use native::hidden;

#[cfg(all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64"))]
mod native {
    use super::super::resident::ops::{Ctx, cast_to, values};
    use super::process_trace;
    use crate::{
        EngError,
        tensor::{Dtype, ExecutionTensor},
    };

    /// The MLX readback of one hidden row: the hidden carries the activation dtype (BF16 in
    /// serving), so the dump casts first; the cast and the readback belong to the traced
    /// run only.
    pub(in crate::family::qwen36) fn hidden(
        ctx: &mut Ctx<'_>,
        layer: usize,
        step: u32,
        tensor: &ExecutionTensor,
    ) -> Result<(), EngError> {
        let wide = cast_to(ctx, tensor, Dtype::F32)?;
        let row = values(ctx.kernels, &wide)?;
        process_trace().write_row(step, layer, &row)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    /// Collects the file names of `dir`, sorted.
    fn file_names(dir: &Path) -> Vec<String> {
        let mut names: Vec<String> = std::fs::read_dir(dir)
            .expect("the trace directory lists")
            .map(|entry| {
                entry
                    .expect("a directory entry reads")
                    .file_name()
                    .to_string_lossy()
                    .into_owned()
            })
            .collect();
        names.sort();
        names
    }

    /// Decodes a little-endian F32 dump to bit patterns, so a NaN or a signed zero compares.
    fn bits(path: &Path) -> Vec<u32> {
        let bytes = std::fs::read(path).expect("the dump reads");
        assert_eq!(bytes.len() % 4, 0, "a dump holds whole f32 values");
        bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|chunk| u32::from_le_bytes(*chunk))
            .collect()
    }

    /// With the variable unset or empty the trace is off and says nothing: no directory
    /// resolves, no step is claimed, and a write succeeds without a file.
    #[test]
    fn an_unset_or_empty_variable_leaves_the_trace_off() {
        for raw in [None, Some(OsString::new())] {
            assert_eq!(resolve(raw.clone()), Ok(None));
            let trace = LayerTrace::from_env_value(raw);
            assert!(trace.dir.is_none());
            assert_eq!(trace.begin_step(), None);
            assert_eq!(trace.begin_step(), None, "an off trace claims nothing");
            trace
                .write_row(0, 0, &[1.0])
                .expect("an off trace writes nothing and succeeds");
        }
    }

    /// A value that is not an existing directory turns the trace off, with a warning that names
    /// the path, instead of failing every decode step later. A regular file and a missing path
    /// are the two cases.
    #[test]
    fn a_value_that_is_not_a_directory_leaves_the_trace_off_with_a_warning() {
        let dir = tempfile::tempdir().expect("a temporary directory");
        let file = dir.path().join("a-file");
        std::fs::write(&file, b"x").expect("a plain file");
        for path in [file, dir.path().join("missing")] {
            let warning = resolve(Some(path.clone().into_os_string())).expect_err("not a dir");
            assert!(warning.contains(&path.display().to_string()), "{warning}");
            assert!(warning.contains("Q36_LAYER_TRACE_DIR"), "{warning}");
            let trace = LayerTrace::from_env_value(Some(path.into_os_string()));
            assert!(trace.dir.is_none());
            assert_eq!(trace.begin_step(), None);
            trace
                .write_row(0, 0, &[1.0])
                .expect("an off trace writes nothing and succeeds");
        }
        assert_eq!(file_names(dir.path()), ["a-file"], "nothing was written");
    }

    /// With the variable set to a directory, each traced step writes one file per layer named by
    /// step and layer, and the files hold the rows bit for bit. Step indices count up from zero.
    #[test]
    fn a_named_directory_receives_one_file_per_layer_per_step() {
        let dir = tempfile::tempdir().expect("a temporary directory");
        let trace = LayerTrace::from_env_value(Some(dir.path().as_os_str().to_owned()));
        let rows: [&[f32]; 3] = [
            &[1.5, -0.0, f32::NAN, f32::INFINITY],
            &[2.0, 3.0, 4.0, 5.0],
            &[-1.0, 0.25, 1e-30, f32::MIN_POSITIVE],
        ];
        let step = trace.begin_step().expect("the first step is traced");
        assert_eq!(step, 0);
        for (layer, row) in rows.iter().enumerate() {
            trace.write_row(step, layer, row).expect("a row writes");
        }
        let step = trace.begin_step().expect("the second step is traced");
        assert_eq!(step, 1);
        trace.write_row(step, 0, rows[1]).expect("a row writes");

        assert_eq!(
            file_names(dir.path()),
            [
                "step0000_layer00.f32",
                "step0000_layer01.f32",
                "step0000_layer02.f32",
                "step0001_layer00.f32",
            ]
        );
        for (layer, row) in rows.iter().enumerate() {
            let expected: Vec<u32> = row.iter().map(|value| value.to_bits()).collect();
            let path = dir.path().join(format!("step0000_layer{layer:02}.f32"));
            assert_eq!(bits(&path), expected, "layer {layer} of step 0");
        }
        let second: Vec<u32> = rows[1].iter().map(|value| value.to_bits()).collect();
        assert_eq!(bits(&dir.path().join("step0001_layer00.f32")), second);
    }

    /// The CUDA chunk-size diagnostic's files: prefill pieces count on their OWN counter (so a
    /// `[295,5]` run's decode files still align with a `[300]` run's from `step0000`), and a
    /// piece's file holds all of its rows row-major; the logits row lands beside the layer rows.
    #[test]
    fn prefill_pieces_and_logits_have_their_own_names_and_counters() {
        let dir = tempfile::tempdir().expect("a temporary directory");
        let trace = LayerTrace::from_env_value(Some(dir.path().as_os_str().to_owned()));
        let piece = trace.begin_prefill().expect("the first piece is traced");
        assert_eq!(piece, 0);
        let rows = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        trace
            .write_prefill(piece, 7, &rows)
            .expect("the piece rows write");
        // A decode step between pieces advances the step counter only.
        let step = trace.begin_step().expect("a step is traced");
        assert_eq!(step, 0);
        trace.write_logits(step, &[9.0, 8.0]).expect("logits write");
        assert_eq!(
            trace.begin_prefill().expect("the second piece"),
            1,
            "pieces count independently of steps"
        );
        assert_eq!(
            file_names(dir.path()),
            ["prefill0000_layer07.f32", "step0000_logits.f32"]
        );
        let expected: Vec<u32> = rows.iter().map(|value| value.to_bits()).collect();
        assert_eq!(bits(&dir.path().join("prefill0000_layer07.f32")), expected);
    }

    /// A directory that disappears after the check still gives a `Backend` error that names the
    /// step and the layer, not a silent loss of the row.
    #[test]
    fn a_directory_removed_after_the_check_is_a_backend_error_naming_the_row() {
        let dir = tempfile::tempdir().expect("a temporary directory");
        let gone = dir.path().join("gone");
        std::fs::create_dir(&gone).expect("the trace directory");
        let trace = LayerTrace::from_env_value(Some(gone.clone().into_os_string()));
        assert_eq!(trace.begin_step(), Some(0));
        std::fs::remove_dir(&gone).expect("the directory goes away");
        let error = trace.write_row(7, 12, &[1.0]).expect_err("no directory");
        match error {
            EngError::Backend { operation, message } => {
                assert_eq!(operation, "qwen36 layer trace");
                assert!(message.contains("step 7 layer 12"), "{message}");
            }
            other => panic!("expected a Backend error, got {other:?}"),
        }
    }
}
