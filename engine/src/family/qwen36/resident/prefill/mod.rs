//! Batched prefill blocks of the resident Q36 decoder.
//!
//! Each function is the chunk form of a decode block in the parent module. It takes `T` token rows,
//! builds lazy graphs, and never reads back or evaluates, except the MoE router readback of the
//! host-routed path and the sub-chunk evaluations that the step counts (the device-routed MoE is
//! fully lazy). `step` wires the blocks into `ResidentDecoder::prefill`. The decode blocks in the
//! parent module stay unchanged and stay the reference. All experts are resident, so the chunk has
//! no page pool and no expert waves.

mod attn;
mod conv;
mod delta;
mod moe;
#[cfg(test)]
mod parity;
pub(super) mod step;
#[cfg(test)]
mod timing;
