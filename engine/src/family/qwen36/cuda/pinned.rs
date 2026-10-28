//! The pinned model's host-side facts shared by the stage 1 gate and the serve start
//! (CUDA-SERVE-1): the rotary dimension, the free-after-load floor of the memory admission,
//! and the head width of the pinned model. Pure move out of `gate/run.rs` (the bodies are
//! byte-identical): the gate's qualified loop and the server's CUDA start read the SAME
//! constants, so a divergence between the two admission checks is impossible. Pure host
//! arithmetic: no context, no device, so a missing file or a bad rule fails in seconds, not
//! after a 20 GB upload.

use super::super::host::HostShape;
use super::dims::StepDims;
use crate::{
    EngError,
    family::{QWEN3_6_35B_A3B, gguf_model::Inventory},
};

/// Rotary dimension of the pinned model (head_dim 256 times the partial-rotary factor 0.25);
/// the GGUF carries no config.json, and the RD-9b fixtures name the same 64.
pub(super) const ROTARY_DIM: u32 = 64;
/// Less than this many bytes free after the load is a typed refusal, not an error (0040 A2).
pub(super) const MIN_FREE_AFTER_LOAD: usize = 1 << 30;

/// The head width of the pinned model from the host-side dimensions: no context, no device, so
/// the rules can be checked before any device work (RD-10a-17b item 4).
pub(super) fn head_rows_of() -> Result<usize, EngError> {
    let inventory = Inventory::for_spec(&QWEN3_6_35B_A3B)?;
    let shape = HostShape::from_spec(&QWEN3_6_35B_A3B, ROTARY_DIM)?;
    Ok(StepDims::new(&shape, &inventory)?.shape.vocab)
}
