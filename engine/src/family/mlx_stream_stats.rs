//! Snapshot the existing MLX and shard counters around one streamed forward.

use std::time::Instant;

use super::stored::{ExpertReadStats, qualification_expert_reads};
use crate::kernels::{MlxKernels, QualificationTiming};

pub(super) struct StreamSnapshot {
    started: Instant,
    reads: ExpertReadStats,
    native: QualificationTiming,
    page_copies: u64,
    syncs: u64,
}

impl StreamSnapshot {
    pub(super) fn begin(kernels: &MlxKernels) -> Self {
        Self {
            started: Instant::now(),
            reads: qualification_expert_reads(),
            native: kernels.qualification_timing(),
            page_copies: kernels.qualification_page_copies(),
            syncs: kernels.qualification_sync_count(),
        }
    }

    pub(super) fn finish(self, kernels: &MlxKernels) -> serde_json::Value {
        let forward_ns = u64::try_from(self.started.elapsed().as_nanos()).unwrap_or(u64::MAX);
        let reads = qualification_expert_reads();
        let native = kernels.qualification_timing();
        serde_json::json!({
            "forward_ns": forward_ns,
            "slice_reads": reads.slice_reads - self.reads.slice_reads,
            "read_bytes": reads.read_bytes - self.reads.read_bytes,
            "read_ns": reads.read_ns - self.reads.read_ns,
            "page_imports": reads.page_imports - self.reads.page_imports,
            "page_import_ns": reads.import_ns - self.reads.import_ns,
            "page_copies": kernels.qualification_page_copies() - self.page_copies,
            "metal_syncs": kernels.qualification_sync_count() - self.syncs,
            "native_import_calls": native.import_calls - self.native.import_calls,
            "native_import_ns": native.import_ns - self.native.import_ns,
            "native_readback_calls": native.readback_calls - self.native.readback_calls,
            "native_readback_ns": native.readback_ns - self.native.readback_ns,
            "native_evaluate_calls": native.evaluate_calls - self.native.evaluate_calls,
            "native_evaluate_ns": native.evaluate_ns - self.native.evaluate_ns,
        })
    }
}
