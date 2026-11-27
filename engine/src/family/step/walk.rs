//! The walk of a step.
//!
//! [`feed`] cuts tokens into pieces of the step's chunk size, checks the bounds and collects the
//! head rows. [`visits`] runs one chunk through a model's visits: every visit is a mixer half
//! and a feed-forward half, and every half reads its input rows from the residual, runs its
//! branch and returns the branch to the residual. A model supplies the three parts of a half
//! ([`Visits`]); the order, and the point after each half where a caller may read rows, are
//! shared.

use crate::EngError;

/// Which rows of a fed chunk run through the output head.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum HeadRows {
    None,
    Last,
    #[cfg(test)]
    All,
}

/// A model's forward pass over one chunk of token rows.
pub(crate) trait ChunkStep {
    /// The name of the model in refusals.
    fn family(&self) -> &'static str;
    /// The most rows of one chunk.
    fn chunk_rows(&self) -> usize;
    /// Positions the state holds.
    fn cells(&self) -> usize;
    fn vocab(&self) -> usize;
    /// Runs `tokens` (at most [`Self::chunk_rows`]) at positions `pos0..` through every layer.
    fn run_chunk(&mut self, tokens: &[u32], pos0: usize) -> Result<(), EngError>;
    /// The logits of row `row` of the chunk just run.
    fn head_row(&self, row: usize) -> Result<Vec<f32>, EngError>;
    /// Waits for the queued work of the step.
    fn drain(&self) -> Result<(), EngError>;
}

/// Feeds `tokens` at positions `pos0..`, in chunks. Returns the logits of the rows `head` names
/// (one row of `vocab` values each).
pub(crate) fn feed<S: ChunkStep>(
    step: &mut S,
    tokens: &[u32],
    pos0: usize,
    head: HeadRows,
) -> Result<Vec<Vec<f32>>, EngError> {
    let family = step.family();
    if tokens.is_empty() {
        return Err(EngError::BadRequest(format!(
            "{family}: a feed needs a token"
        )));
    }
    if pos0
        .checked_add(tokens.len())
        .is_none_or(|end| end > step.cells())
    {
        return Err(EngError::ContextLimit(format!(
            "{family}: {} tokens at position {pos0} exceed the {} cached positions",
            tokens.len(),
            step.cells()
        )));
    }
    if let Some(&token) = tokens.iter().find(|&&t| t as usize >= step.vocab()) {
        return Err(EngError::BadRequest(format!(
            "{family}: token {token} is outside the vocabulary of {}",
            step.vocab()
        )));
    }
    let chunk = step.chunk_rows();
    let mut logits = Vec::new();
    let mut done = 0;
    while done < tokens.len() {
        let piece = &tokens[done..(done + chunk).min(tokens.len())];
        let last_piece = done + piece.len() == tokens.len();
        step.run_chunk(piece, pos0 + done)?;
        match head {
            HeadRows::None => {}
            HeadRows::Last if !last_piece => {}
            HeadRows::Last => logits.push(step.head_row(piece.len() - 1)?),
            #[cfg(test)]
            HeadRows::All => {
                for row in 0..piece.len() {
                    logits.push(step.head_row(row)?);
                }
            }
        }
        done += piece.len();
    }
    if head == HeadRows::None {
        step.drain()?;
    }
    Ok(logits)
}

/// The two halves of a visit.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Half {
    Mixer,
    Forward,
}

/// The visits of a model over one chunk of rows.
pub(crate) trait Visits {
    /// Visits of one walk.
    fn count(&self) -> usize;
    /// Queued before the first visit of a chunk of `n` rows.
    fn begin(&self, n: usize) -> Result<(), EngError>;
    /// Queued before the first half of visit `visit`, for a model whose state changes between
    /// visits outside their halves (a repeated pass that restarts its streams). Device work only:
    /// a recorded step replays it as recorded.
    fn enter(&self, _visit: usize, _n: usize, _p0: usize) -> Result<(), EngError> {
        Ok(())
    }
    /// The residual read of a half: its input rows, normed.
    fn read(&self, visit: usize, half: Half, n: usize) -> Result<(), EngError>;
    /// The branch of a half over the rows the read left; `p0` is the position of row 0.
    fn branch(&self, visit: usize, half: Half, n: usize, p0: usize) -> Result<(), EngError>;
    /// The return of the branch rows to the residual.
    fn apply(&self, visit: usize, half: Half, n: usize) -> Result<(), EngError>;
}

/// Runs a chunk of `n` rows whose first row sits at `p0` through every visit of `model`. `tap`
/// runs after each half, behind its launches: a trace or a reader of the rows of that point.
pub(crate) fn visits<V: Visits + ?Sized>(
    model: &V,
    n: usize,
    p0: usize,
    mut tap: impl FnMut(usize, Half) -> Result<(), EngError>,
) -> Result<(), EngError> {
    model.begin(n)?;
    for visit in 0..model.count() {
        model.enter(visit, n, p0)?;
        for half in [Half::Mixer, Half::Forward] {
            model.read(visit, half, n)?;
            model.branch(visit, half, n, p0)?;
            model.apply(visit, half, n)?;
            tap(visit, half)?;
        }
    }
    Ok(())
}
