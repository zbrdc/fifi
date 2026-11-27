//! The Qwen lineage: its children, each a spec, a recipe and a weight map over the shared engine.
//! What the children share and the engine does not (name conventions, chat template quirks and
//! layer kinds unique to the lineage) belongs here.

pub(crate) mod cuda;
#[allow(
    dead_code,
    reason = "the factory wires the MLX and the opt-in CUDA executors (`cuda_decode`); the allow covers the pieces still reached only by tests until the family is qualified"
)]
pub(crate) mod flash_next;
#[allow(
    dead_code,
    reason = "the Q36 host reference serves later work; until it lands, only tests and the Mac reference run call it"
)]
pub mod qwen36;
#[cfg(feature = "native-cuda")]
pub(crate) mod recipe;
pub(crate) mod sensenova;
