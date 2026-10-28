//! Bounded GGUF re-layout for the shared transport; no retained payload or CPU evaluator.

use std::{fs::File, os::unix::fs::FileExt};

use crate::{
    EngError,
    expert_tier::{ExpertId, ExpertPayload, ExpertPayloadSource},
    family::flint::gguf_weights,
    weights::{GgufHeader, repack_nvfp4_into},
};

pub(crate) struct GgufSource<'a> {
    file: &'a File,
    header: &'a GgufHeader,
}

impl<'a> GgufSource<'a> {
    pub(crate) fn new(file: &'a File, header: &'a GgufHeader) -> Result<Self, EngError> {
        gguf_weights::expert_sources(header, 2, 0)?;
        Ok(Self { file, header })
    }
}

fn zeros(bytes: usize) -> Result<Vec<u8>, EngError> {
    let mut data = Vec::new();
    data.try_reserve_exact(bytes)
        .map_err(|_| EngError::MemoryPlan("Flint expert payload allocation failed".into()))?;
    data.resize(bytes, 0);
    Ok(data)
}

impl ExpertPayloadSource for GgufSource<'_> {
    fn parts(&self) -> usize {
        3
    }

    fn payload(&mut self, id: ExpertId, part: usize) -> Result<ExpertPayload, EngError> {
        #[cfg(test)]
        let descriptor = super::super::tests::profile::wall("routed_descriptor_raw_alloc");
        let sources = gguf_weights::expert_sources(self.header, id.layer, id.expert)?;
        let source = [sources.gate, sources.up, sources.down]
            .into_iter()
            .nth(part)
            .ok_or_else(|| EngError::BadRequest("Flint expert part must be gate/up/down".into()))?;
        let bytes = usize::try_from(source.data.end - source.data.start)
            .map_err(|_| EngError::MemoryPlan("Flint expert range exceeds usize".into()))?;
        let mut raw = zeros(bytes)?;
        let mut scalar = [0; 4];
        #[cfg(test)]
        drop(descriptor);
        #[cfg(test)]
        let read = super::super::tests::profile::wall("routed_read_exact_at");
        self.file
            .read_exact_at(&mut raw, source.data.start)
            .and_then(|()| self.file.read_exact_at(&mut scalar, source.scale.start))
            .map_err(|e| EngError::Checkpoint(format!("Flint expert read: {e}")))?;
        #[cfg(test)]
        drop(read);
        #[cfg(test)]
        let _repack = super::super::tests::profile::wall("routed_bind_alloc_repack");
        let view = source.bind(&raw, &scalar)?;
        let elements = view.shape[0] * view.shape[1];
        let mut data = zeros(elements / 2)?;
        let mut scales = zeros(elements / 16)?;
        repack_nvfp4_into(view.data, &mut data, &mut scales)?;
        Ok(ExpertPayload {
            data,
            scales,
            tensor_scale: view.tensor_scale,
            shape: view.shape.map(|n| n as u32),
        })
    }

    fn release_layer(&mut self, _: u32) -> Result<(), EngError> {
        Err(EngError::BadRequest(
            "Flint mixed layers require expert-granular release".into(),
        ))
    }

    fn release_experts(&mut self, _: &[ExpertId]) -> Result<(), EngError> {
        // Payloads are transient; file-cache pages remain charged, not reclaimed or discounted.
        Ok(())
    }
}
