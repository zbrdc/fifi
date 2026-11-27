//! Resident-decode operations: the kernels a decode step runs with its weights and state on
//! the device. Each op states its contract, refuses before any native call, and keeps its tests
//! beside it, the shape of the CUDA adapter's resident tree.
pub(crate) mod dense_gemv;
