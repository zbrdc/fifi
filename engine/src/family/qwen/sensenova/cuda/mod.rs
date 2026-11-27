//! The CUDA side of SenseNova-Loop that is its own: host geometry over the shared
//! device-planning shape. The step and the serve wiring it runs are the lineage's
//! ([`crate::family::qwen::cuda`]).

pub(crate) mod geometry;
