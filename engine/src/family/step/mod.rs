//! The shared resident step on one CUDA device: what a model's forward pass needs that is not its
//! own math.
//!
//! A step runs a chunk of token rows layer by layer, and decoding is the chunk of one row, so
//! prefill and decode are one walk. [`walk`] holds that walk around a model's layers: the pieces
//! of a feed, the bounds and the head rows. [`rows`] holds the activation rows a projection reads
//! and the projections themselves, chosen by the type of each weight tensor.
//!
//! [`ffn`] holds the gated feed-forward block over those projections, [`moe`] the routed block
//! of a router, a shared expert and the kept experts, [`experts`] the routed expert projections
//! of a chunk, and [`site`] the residual sites: the plain add and the routed
//! two-stream site. [`depth`] is the depth-attention site, attention residuals over depth.
//!
//! [`delta`] and [`attend`] are the mixers of a visit: the gated delta rule and the gated full
//! attention, each over one row or a chunk.
//!
//! [`state`] is what a step carries from token to token: the rings, records and cache planes of
//! its visits. [`scratch`] is where a step's scratch comes from: one declaration both plans and allocates
//! it. [`head`] is the final norm and the projection onto the vocabulary. [`replay`] records a
//! one-row step once and replays it, so a decode step is one call. [`tail`] is the token tail of
//! a decode step: the embedding gather before it and the pick after it, on the device.
//!
//! [`recipe`] is a model's list of layers, each naming its tensors, and the one walk over them. A
//! model supplies that list, its widths, the rows it works in and the mixers nobody else has.

pub(crate) mod attend;
pub(crate) mod conv;
pub(crate) mod delta;
pub(crate) mod depth;
pub(crate) mod experts;
pub(crate) mod ffn;
pub(crate) mod head;
pub(crate) mod moe;
pub(crate) mod recipe;
pub(crate) mod replay;
pub(crate) mod rows;
pub(crate) mod scratch;
pub(crate) mod site;
pub(crate) mod state;
pub(crate) mod tail;
pub(crate) mod walk;
