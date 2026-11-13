//! The chunk-family contract: the eight operations every chunk loop shares, defined from the
//! real chunk drivers as they stand. A method with no counterpart on every side is not here;
//! the pieces only one family runs (last-row logits, its projection staging, the resume
//! generation) join with that family's implementor, not with the trait.
//!
//! The context is opaque to the trait: each family's wrapper owns a concrete context struct of
//! the pieces its chunk body captures (stream, launch, arena views, model weights), and the
//! associated type keeps `dyn` usable — object safety is part of the contract the shared
//! driver relies on.

use crate::EngError;

/// The per-chunk surface of a chunked family: window open, status clear, input staging, the
/// layer run, the health read, the commit, and the sequence-level reset and miss report.
/// Methods map one-to-one onto operations the existing chunk loops perform.
pub(crate) trait ChunkFamily {
    /// The chunk context: the family's captured stream, handles and views, opaque here.
    type Ctx;

    /// Opens the prefill window; the family decides its period and what it gates.
    fn begin_window(&mut self, ctx: &mut Self::Ctx) -> Result<(), EngError>;

    /// Clears the per-chunk status word, ordered behind nothing (the stream is drained at the
    /// chunk boundary the previous chunk's health read left).
    fn clear_chunk_status(&mut self, ctx: &mut Self::Ctx) -> Result<(), EngError>;

    /// Stages the chunk's device inputs for `m` rows at `p0`, behind the open window. The
    /// family's payload rides the context, derived on the host before the first device write,
    /// so a refused input changes nothing on the device.
    fn write_chunk_inputs(
        &mut self,
        ctx: &mut Self::Ctx,
        p0: usize,
        m: usize,
    ) -> Result<(), EngError>;

    /// Queues the chunk's layers for `m` rows at `p0`, behind the staged inputs.
    fn run_layers(&mut self, ctx: &mut Self::Ctx, p0: usize, m: usize) -> Result<(), EngError>;

    /// Reads the status word at chunk close; `0` is clean. One ordered status read per
    /// chunk, at close.
    fn read_health(&mut self, ctx: &mut Self::Ctx) -> Result<i32, EngError>;

    /// Publishes the `m` committed rows of the chunk at `p0` and closes the window
    /// (`status` is the word `read_health` returned; non-zero poisons).
    fn commit_chunk(
        &mut self,
        ctx: &mut Self::Ctx,
        p0: usize,
        m: usize,
        status: i32,
    ) -> Result<(), EngError>;

    /// Zeroes the persistent state: a new sequence.
    #[allow(dead_code, reason = "first caller arrives with the shared driver")]
    fn reset(&mut self, ctx: &mut Self::Ctx) -> Result<(), EngError>;

    /// The bank's miss count for the per-job report, when a bank exists.
    #[allow(dead_code, reason = "first caller arrives with the shared driver")]
    fn bank_misses(&self) -> Option<u64> {
        None
    }
}
