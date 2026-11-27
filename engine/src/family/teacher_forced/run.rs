//! The run of a quality leg: the id stream, the windows and the output directory read from the
//! environment, and the window loop over a [`TokenStepper`]. Only the CUDA legs run it.

use super::{HEADER_BYTES, RECORD_BYTES, Sidecar, bad};
use crate::{EngError, family::session::TokenStepper};
use sha2::{Digest, Sha256};
use std::{
    path::{Path, PathBuf},
    time::Instant,
};

fn env(name: &str) -> Result<String, EngError> {
    std::env::var(name).map_err(|_| bad(format!("{name} is not set")))
}

/// A window of the id stream: where its prefill starts, where scoring starts, and where it ends.
pub(crate) type Window = (usize, usize, usize);

fn layout_of(path: &Path) -> Result<(usize, Vec<Window>), EngError> {
    let value: serde_json::Value =
        serde_json::from_slice(&std::fs::read(path).map_err(|e| bad(format!("layout: {e}")))?)
            .map_err(|e| bad(format!("layout: {e}")))?;
    let context = value["context_tokens"]
        .as_u64()
        .ok_or_else(|| bad("layout: context_tokens"))? as usize;
    let windows = value["windows"]
        .as_array()
        .ok_or_else(|| bad("layout: windows"))?
        .iter()
        .map(|w| {
            let at = |i: usize| w[i].as_u64().map(|v| v as usize);
            match (at(0), at(1), at(2)) {
                (Some(lo), Some(score_lo), Some(hi)) if lo < score_lo && score_lo < hi => {
                    Ok((lo, score_lo, hi))
                }
                _ => Err(bad("layout: a window is not [lo, score_lo, hi]")),
            }
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok((context, windows))
}

fn ids_of(path: &Path, sha: &str) -> Result<Vec<u32>, EngError> {
    let bytes = std::fs::read(path).map_err(|e| bad(format!("ids: {e}")))?;
    let digest = Sha256::digest(&bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect::<String>();
    if digest != sha {
        return Err(bad(format!("the id stream is {digest}, not {sha}")));
    }
    Ok(bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| u32::from_le_bytes(*c))
        .collect())
}

/// The id stream, the windows and the output of one scoring run.
pub(crate) struct Run {
    pub(crate) ids: Vec<u32>,
    /// The layout's `context_tokens`: the most tokens a window holds.
    pub(crate) context: usize,
    windows: Vec<Window>,
    range: (usize, usize),
    out: PathBuf,
    reference: Option<PathBuf>,
    /// The prompt depths of a timing run (`KVQ_TIMING`, comma separated); empty for a scoring run.
    timing: Vec<usize>,
}

impl Run {
    pub(crate) fn from_env() -> Result<Self, EngError> {
        let ids = ids_of(Path::new(&env("KVQ_IDS")?), &env("KVQ_IDS_SHA")?)?;
        let (context, windows) = layout_of(Path::new(&env("KVQ_LAYOUT")?))?;
        let bounds = env("KVQ_WINDOWS")?
            .split(':')
            .map(|x| x.parse::<usize>())
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| bad(format!("KVQ_WINDOWS: {e}")))?;
        let &[lo, hi] = bounds.as_slice() else {
            return Err(bad("KVQ_WINDOWS is not lo:hi"));
        };
        if lo > hi || hi > windows.len() {
            return Err(bad(format!(
                "KVQ_WINDOWS {lo}:{hi} is outside the {} windows",
                windows.len()
            )));
        }
        if windows.iter().any(|&(_, _, end)| end > ids.len()) {
            return Err(bad("a window ends past the id stream"));
        }
        let timing = match std::env::var("KVQ_TIMING") {
            Ok(text) => text
                .split(',')
                .map(|x| x.parse::<usize>())
                .collect::<Result<Vec<_>, _>>()
                .map_err(|e| bad(format!("KVQ_TIMING: {e}")))?,
            Err(_) => Vec::new(),
        };
        let out = PathBuf::from(env("KVQ_OUT")?);
        std::fs::create_dir_all(&out).map_err(|e| bad(format!("KVQ_OUT: {e}")))?;
        Ok(Self {
            ids,
            context,
            windows,
            range: (lo, hi),
            out,
            reference: std::env::var("KVQ_REF").ok().map(PathBuf::from),
            timing,
        })
    }

    fn sidecar(&self, w: usize) -> PathBuf {
        self.out.join(format!("w{w:02}.kvq"))
    }

    /// The windows of the range whose sidecar is not complete yet.
    pub(crate) fn pending(&self) -> Vec<(usize, Window)> {
        (self.range.0..self.range.1)
            .map(|w| (w, self.windows[w]))
            .filter(|&(w, (_, score_lo, hi))| {
                let done = std::fs::metadata(self.sidecar(w)).is_ok_and(|m| {
                    m.len() as usize == HEADER_BYTES + (hi - score_lo) * RECORD_BYTES
                });
                if done {
                    println!("kvq window {w}: sidecar present, skipping");
                }
                !done
            })
            .collect()
    }

    /// The scoring of window `w`, against the reference run's sidecar of the window when the
    /// run names one.
    pub(crate) fn start(&self, w: usize) -> Result<Sidecar, EngError> {
        let (_, score_lo, hi) = self.windows[w];
        let reference = self
            .reference
            .as_ref()
            .map(|dir| {
                std::fs::read(dir.join(format!("w{w:02}.kvq")))
                    .map_err(|e| bad(format!("window {w}: reference sidecar: {e}")))
            })
            .transpose()?;
        Sidecar::new(w, hi - score_lo, reference)
    }

    pub(crate) fn finish(&self, sidecar: Sidecar) -> Result<(), EngError> {
        let w = sidecar.window;
        let bytes = sidecar.finish()?;
        std::fs::write(self.sidecar(w), bytes).map_err(|e| bad(format!("window {w}: {e}")))
    }
}

/// Scores the pending windows of `run` on `stepper`: a reset, the window's unscored tokens fed in
/// prompt chunks of `chunk` rows, then one forced token per scored position, each logits row
/// narrowed to the first `vocab` ids. A run with `KVQ_TIMING` set times the stepper instead.
pub(crate) fn score_windows(
    stepper: &mut impl TokenStepper,
    run: &Run,
    chunk: usize,
    vocab: usize,
) -> Result<(), EngError> {
    if !run.timing.is_empty() {
        return time_depths(stepper, run, chunk);
    }
    for (w, (lo, score_lo, hi)) in run.pending() {
        let started = Instant::now();
        println!("kvq window {w} start: prefill {} ids", score_lo - 1 - lo);
        stepper.reset()?;
        for piece in run.ids[lo..score_lo - 1].chunks(chunk.max(1)) {
            stepper.feed_rows(piece)?;
        }
        println!(
            "kvq window {w} prefill done [{:.0}s]",
            started.elapsed().as_secs_f64()
        );
        let mut sidecar = run.start(w)?;
        for p in score_lo..hi {
            let row = stepper.logits(run.ids[p - 1])?;
            let row = row
                .get(..vocab)
                .ok_or_else(|| bad(format!("window {w}: a logits row of {} ids", row.len())))?;
            sidecar.push(row, run.ids[p])?;
        }
        run.finish(sidecar)?;
    }
    Ok(())
}

/// Forced one-token steps timed after each prompt of a timing run.
const TIMED_STEPS: usize = 64;

/// Times `stepper` at each prompt depth of the run, twice (the first pass pays one-time setup):
/// the prompt fed in chunks of `chunk` rows up to the first logits row, which is the time to the
/// first token, then [`TIMED_STEPS`] forced one-token steps, each with its logits row read back.
fn time_depths(stepper: &mut impl TokenStepper, run: &Run, chunk: usize) -> Result<(), EngError> {
    for &depth in &run.timing {
        if depth < 2 || depth + TIMED_STEPS > run.ids.len() || depth + TIMED_STEPS > run.context {
            return Err(bad(format!(
                "a timing depth of {depth} does not fit the stream or the context"
            )));
        }
        for pass in 0..2 {
            stepper.reset()?;
            let started = Instant::now();
            for piece in run.ids[..depth - 1].chunks(chunk.max(1)) {
                stepper.feed_rows(piece)?;
            }
            stepper.logits(run.ids[depth - 1])?;
            let first = started.elapsed().as_secs_f64();
            let started = Instant::now();
            for p in depth..depth + TIMED_STEPS {
                stepper.logits(run.ids[p])?;
            }
            let steps = started.elapsed().as_secs_f64();
            println!(
                "kvq timing depth {depth} pass {pass}: first token {:.1} ms ({:.0} prompt tok/s), then {:.2} ms per step ({:.1} tok/s)",
                first * 1e3,
                depth as f64 / first,
                steps * 1e3 / TIMED_STEPS as f64,
                TIMED_STEPS as f64 / steps
            );
        }
    }
    Ok(())
}
