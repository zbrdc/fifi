//! Hybrid decode on the stack: the Nemotron glue of the shared expert tier (`crate::expert_tier`).
//! Placement at load, the CPU-routed MoE block, and the per-token split.

use std::time::Instant;

use super::StackProjector;
#[cfg(all(feature = "native-cuda", target_os = "linux"))]
use super::packed_cache_key;
use super::{
    ExpertIndex, ExpertProjection, GpuLayerPlan, HybridConfig, HybridReport, HybridRuntime,
    MlpWeights, NemotronStack, Residency, RoutedExpert, TokenSplit, WeightSource, block_error,
    invalid, moe_apply_selected_routed, moe_apply_selected_routed_batch,
};
use crate::{
    EngError,
    expert_tier::{
        ExpertId, ExpertPayload, ExpertPayloadSource, ExpertPick, GpuExpertStore, place_gpu_layers,
    },
    family::nemotron35::{NemotronBlockKind, cpu_experts::new_cpu_experts},
    tensor::Tensor,
};
#[cfg(all(feature = "native-cuda", target_os = "linux"))]
use crate::{expert_tier::VRAM_HEADROOM_BYTES, family::nemotron35::cuda_projection::CudaProjector};

/// The tier's view of the checkpoint: up is part 0, down is part 1.
struct NemotronPayloads<'a, S: WeightSource> {
    source: &'a mut S,
}

impl<S: WeightSource> ExpertPayloadSource for NemotronPayloads<'_, S> {
    fn parts(&self) -> usize {
        2
    }

    fn payload(&mut self, id: ExpertId, part: usize) -> Result<ExpertPayload, EngError> {
        let which = if part == 0 {
            ExpertProjection::Up
        } else {
            ExpertProjection::Down
        };
        let packed = self
            .source
            .expert_packed(id.layer, ExpertIndex::Routed(id.expert), which)?;
        Ok(ExpertPayload {
            data: packed.packed,
            scales: packed.scale,
            tensor_scale: packed.weight_scale_2,
            shape: packed.shape,
        })
    }

    fn release_layer(&mut self, layer: u32) -> Result<(), EngError> {
        self.source.release_expert_layer(layer)
    }
}

/// The host projector has no device: no memory is free, so nothing is resident and a fixed
/// count above zero is refused as not fitting.
struct NoDevice;

impl GpuExpertStore for NoDevice {
    fn upload(&mut self, _: ExpertId, _: usize, _: &ExpertPayload) -> Result<(), EngError> {
        Err(invalid("the host projector has no device to hold experts"))
    }

    fn free_bytes(&self) -> Result<u64, EngError> {
        Ok(0)
    }

    fn reserved_bytes(&self) -> u64 {
        0
    }

    fn finish_placement(&mut self, _: u64) -> Result<(), EngError> {
        Ok(())
    }
}

#[cfg(all(feature = "native-cuda", target_os = "linux"))]
impl GpuExpertStore for CudaProjector {
    fn upload(
        &mut self,
        id: ExpertId,
        part: usize,
        payload: &ExpertPayload,
    ) -> Result<(), EngError> {
        let which = if part == 0 {
            ExpertProjection::Up
        } else {
            ExpertProjection::Down
        };
        self.load_resident_expert(
            packed_cache_key(id.layer, ExpertIndex::Routed(id.expert), which),
            &payload.data,
            &payload.scales,
            payload.tensor_scale,
            payload.shape,
        )
    }

    fn free_bytes(&self) -> Result<u64, EngError> {
        self.free_device_bytes()
    }

    fn reserved_bytes(&self) -> u64 {
        self.dense_reserve_bytes()
    }

    fn finish_placement(&mut self, free_bytes: u64) -> Result<(), EngError> {
        self.close_packed_uploads();
        // The dense cache gets what is left, never into the headroom.
        self.clamp_weight_cache_budget(
            free_bytes
                .saturating_sub(VRAM_HEADROOM_BYTES)
                .saturating_sub(self.stage_bytes()),
        );
        Ok(())
    }
}

impl<S: WeightSource> NemotronStack<S> {
    /// Switches the stack to hybrid decode: each routed expert is device-resident or CPU-served.
    /// Only a fresh stack may switch (nothing may have been decoded yet).
    pub(crate) fn enable_hybrid(&mut self, config: HybridConfig) -> Result<(), EngError> {
        if self.hybrid.is_some() || self.working.is_some() {
            return Err(invalid(
                "hybrid decode must be enabled once, before any token",
            ));
        }
        let started = Instant::now();
        let moe_layers: Vec<u32> = self
            .spec
            .blocks
            .iter()
            .enumerate()
            .filter(|(_, kind)| **kind == NemotronBlockKind::Moe)
            .map(|(index, _)| index as u32)
            .collect();
        let mut runtime =
            HybridRuntime::new(new_cpu_experts(config.cpu_threads)?, config.gpu_batch);
        runtime.residency = self.place_gpu_layers(config.gpu_layers, &moe_layers)?;
        for &layer in &moe_layers {
            if !runtime.residency.has_cold(layer) {
                continue;
            }
            let views = self.source.cpu_expert_layer(layer)?;
            runtime.cpu.add_layer(layer, views);
        }
        runtime.gpu_resident_bytes = self.resident_bytes();
        runtime.load = started.elapsed();
        self.hybrid = Some(runtime);
        Ok(())
    }

    /// The hybrid's placement and per-token split; `None` on the sequential path.
    pub fn hybrid_report(&self) -> Option<HybridReport> {
        self.hybrid.as_ref().map(HybridRuntime::report)
    }

    /// Whole-layer placement through the shared tier, fitted to the free VRAM.
    fn place_gpu_layers(
        &mut self,
        plan: GpuLayerPlan,
        moe_layers: &[u32],
    ) -> Result<Residency, EngError> {
        let e = self.spec.moe;
        let nominal = nominal_layer_bytes(e.routed_experts, e.moe_intermediate_size, e.latent_size);
        let Self {
            source, projector, ..
        } = self;
        let mut payloads = NemotronPayloads { source };
        match projector {
            #[cfg(all(feature = "native-cuda", target_os = "linux"))]
            StackProjector::Cuda(cuda) => {
                if let Some(stream) = cuda.stream_plan() {
                    let mut pending = stream.banks.reserve(cuda, stream.banks.host_bytes())?;
                    place_gpu_layers(
                        plan,
                        moe_layers,
                        e.routed_experts,
                        nominal,
                        &mut pending,
                        &mut payloads,
                    )
                } else {
                    place_gpu_layers(
                        plan,
                        moe_layers,
                        e.routed_experts,
                        nominal,
                        cuda,
                        &mut payloads,
                    )
                }
            }
            StackProjector::Host(_) => place_gpu_layers(
                plan,
                moe_layers,
                e.routed_experts,
                nominal,
                &mut NoDevice,
                &mut payloads,
            ),
        }
    }

    fn resident_bytes(&self) -> u64 {
        #[cfg(all(feature = "native-cuda", target_os = "linux"))]
        if let StackProjector::Cuda(cuda) = &self.projector {
            return cuda.resident_bytes();
        }
        0
    }

    /// One decode row of a CPU-served MoE layer: the routed experts run on the pool while the
    /// device computes the shared expert, then the latent down projection and residual follow.
    pub(super) fn moe_cpu_routed(
        &mut self,
        layer: u32,
        prefix: &str,
        normalized: &Tensor,
        hidden: &Tensor,
        selected: &[Vec<RoutedExpert>],
    ) -> Result<Tensor, EngError> {
        let subop = std::env::var_os("NEM35_SUBOP_TIMING").is_some_and(|v| v == "1");
        let mut st = std::time::Instant::now();
        let g = self.spec.geometry;
        let e = self.spec.moe;
        // Fix 2 (reordered): compute the shared expert via the device path
        // FIRST. When it succeeds, skip the F32 shared expert load entirely
        // (eliminates ~6.7 GiB of heap from the expert_cache).
        #[cfg(all(feature = "native-cuda", target_os = "linux"))]
        let device_shared: Option<Vec<f32>> = {
            if let StackProjector::Cuda(cuda) = &mut self.projector {
                let up = self
                    .source
                    .expert_packed(layer, ExpertIndex::Shared, ExpertProjection::Up)
                    .ok();
                let down = self
                    .source
                    .expert_packed(layer, ExpertIndex::Shared, ExpertProjection::Down)
                    .ok();
                if let (Some(up), Some(down)) = (up, down) {
                    let _ = cuda.ensure_shared_resident(
                        layer,
                        &up.packed,
                        &up.scale,
                        up.weight_scale_2,
                        up.shape,
                        &down.packed,
                        &down.scale,
                        down.weight_scale_2,
                        down.shape,
                    );
                }
                super::super::read(normalized, &[1, g.hidden_size])
                    .ok()
                    .and_then(|values| cuda.shared_mlp(layer, &values).ok())
            } else {
                None
            }
        };
        #[cfg(not(all(feature = "native-cuda", target_os = "linux")))]
        let device_shared: Option<Vec<f32>> = None;
        // Only load F32 shared weights when the device path is unavailable.
        let skip_shared = device_shared.is_some();
        let shared_up = if skip_shared {
            super::tensor(&[1, 1], &[0.0])?
        } else {
            self.expert_weight(layer, ExpertIndex::Shared, ExpertProjection::Up)?
        };
        let shared_down = if skip_shared {
            super::tensor(&[1, 1], &[0.0])?
        } else {
            self.expert_weight(layer, ExpertIndex::Shared, ExpertProjection::Down)?
        };
        let latent_up = self.mixer_weight(
            prefix,
            "fc1_latent_proj.weight",
            &[e.latent_size, g.hidden_size],
        )?;
        let latent_down = self.mixer_weight(
            prefix,
            "fc2_latent_proj.weight",
            &[g.hidden_size, e.latent_size],
        )?;
        if subop {
            eprintln!("cpu_split weight_load {:?}", st.elapsed());
            st = std::time::Instant::now();
        }
        let shared = MlpWeights {
            up: &shared_up,
            down: &shared_down,
        };
        let latent = e.latent_size as usize;
        let Self {
            hybrid, projector, ..
        } = self;
        let hybrid = hybrid
            .as_mut()
            .ok_or_else(|| invalid("CPU-routed MoE needs hybrid decode"))?;
        // Borrowed once for the device-stream arm of the chunk closure below.
        #[cfg(all(feature = "native-cuda", target_os = "linux"))]
        let source = &mut self.source;
        let rows = hidden.shape().first().copied().unwrap_or(0);
        if rows > 1 {
            // The CPU pool preserves ascending-expert accumulation. The opted-in
            // device callback uses selection order and approximate F16 operands.
            // Latent/shared/residual work remains per row; shared runs on the host.
            let output = moe_apply_selected_routed_batch(
                hidden,
                normalized,
                selected,
                &latent_up,
                &latent_down,
                shared,
                projector,
                |projector, xs, rows| {
                    let picks: Vec<Vec<ExpertPick>> = rows
                        .iter()
                        .map(|row| {
                            row.iter()
                                .map(|pick| ExpertPick {
                                    expert: pick.index,
                                    weight: pick.weight,
                                })
                                .collect()
                        })
                        .collect();
                    let pick_refs: Vec<&[ExpertPick]> =
                        picks.iter().map(|row| row.as_slice()).collect();
                    let mut routed = vec![0.0; xs.len()];
                    // Large chunks may stream the layer's routed experts to the
                    // device instead (Basalt: the CPU pool serves few-row work).
                    #[cfg(all(feature = "native-cuda", target_os = "linux"))]
                    if let StackProjector::Cuda(cuda) = projector {
                        let env = std::env::var_os("NEM35_CHUNK_EXPERTS");
                        let env = env.as_deref().and_then(|value| value.to_str());
                        let pairs = picks
                            .iter()
                            .try_fold(0usize, |n, row| n.checked_add(row.len()));
                        if pairs.is_some_and(|count| {
                            cuda.stream_plan()
                                .is_some_and(|plan| count <= plan.max_pairs)
                                && crate::expert_tier::gpu_stream::wants_device_stream(
                                    rows.len(),
                                    count,
                                    env,
                                )
                        }) {
                            return super::streaming::gpu_stream_routed(
                                cuda,
                                hybrid,
                                source,
                                layer,
                                self.spec.moe.routed_experts,
                                xs,
                                &picks,
                            );
                        }
                    }
                    #[cfg(not(all(feature = "native-cuda", target_os = "linux")))]
                    let _ = projector;
                    hybrid
                        .cpu
                        .routed_batch(layer, xs, &pick_refs, &mut routed)?;
                    Ok(routed)
                },
            )?;
            return Ok(output);
        }
        let (output, timing) = moe_apply_selected_routed(
            hidden,
            normalized,
            selected,
            &latent_up,
            &latent_down,
            shared,
            projector,
            |x, picks| {
                let picks: Vec<ExpertPick> = picks
                    .iter()
                    .map(|pick| ExpertPick {
                        expert: pick.index,
                        weight: pick.weight,
                    })
                    .collect();
                let mut routed = vec![0.0; latent];
                let pool_started = std::time::Instant::now();
                // Gate /proc/self/stat behind the timing flag: the read
                // allocates a String and parses on EVERY cpu.routed call.
                let stat_before = if subop {
                    std::fs::read_to_string("/proc/self/stat")
                        .ok()
                        .and_then(|s| {
                            let fields: Vec<&str> = s.split_whitespace().collect();
                            (fields.len() > 11).then(|| {
                                (
                                    fields[9].parse::<u64>().unwrap_or(0),
                                    fields[11].parse::<u64>().unwrap_or(0),
                                )
                            })
                        })
                } else {
                    None
                };
                hybrid.cpu.routed(layer, x, &picks, &mut routed)?;
                if subop {
                    let elapsed = pool_started.elapsed();
                    let stat_after =
                        std::fs::read_to_string("/proc/self/stat")
                            .ok()
                            .and_then(|s| {
                                let fields: Vec<&str> = s.split_whitespace().collect();
                                (fields.len() > 11).then(|| {
                                    (
                                        fields[9].parse::<u64>().unwrap_or(0),
                                        fields[11].parse::<u64>().unwrap_or(0),
                                    )
                                })
                            });
                    let (minflt, majflt) = match (stat_before, stat_after) {
                        (Some((b_min, b_maj)), Some((a_min, a_maj))) => {
                            (a_min - b_min, a_maj - b_maj)
                        }
                        _ => (0, 0),
                    };
                    eprintln!(
                        "cpu_split pool_call {:?} minflt={minflt} majflt={majflt}",
                        elapsed
                    );
                }
                Ok(routed)
            },
            device_shared,
        )
        .map_err(block_error)?;
        if subop {
            eprintln!("cpu_split total_routed {:?}", st.elapsed());
        }
        hybrid.current.cpu_routed += timing.routed;
        hybrid.current.cpu_exposed += timing.exposed;
        hybrid.current.shared += timing.shared;
        Ok(output)
    }

    /// Closes the books of one token: its wall time and the device calls it made.
    pub(super) fn finish_token_split(
        &mut self,
        wall: std::time::Duration,
        device_before: super::DeviceSnapshot,
    ) {
        let after = self.projector.device_snapshot();
        let Some(hybrid) = self.hybrid.as_mut() else {
            return;
        };
        let mut split: TokenSplit = std::mem::take(&mut hybrid.current);
        split.wall = wall;
        split.device_calls = after
            .call
            .saturating_sub(device_before.call)
            .saturating_sub(after.upload.saturating_sub(device_before.upload));
        split.uploads = after.upload.saturating_sub(device_before.upload);
        split.upload_bytes = after
            .upload_bytes
            .saturating_sub(device_before.upload_bytes);
        // Under NEM35_TOKEN_TIMING=1 each step's split also goes to stderr as it closes, so a
        // run cut off at its time limit keeps the steps it finished. Milliseconds.
        if std::env::var_os("NEM35_TOKEN_TIMING").is_some_and(|v| v == "1") {
            let ms = |duration: std::time::Duration| duration.as_secs_f64() * 1e3;
            eprintln!(
                "nem35_split wall={:.1} cpu_routed={:.1} cpu_exposed={:.1} shared={:.1} \
                 gpu_routed={:.1} device_calls={:.1} uploads={:.1} upload_bytes={}",
                ms(split.wall),
                ms(split.cpu_routed),
                ms(split.cpu_exposed),
                ms(split.shared),
                ms(split.gpu_routed),
                ms(split.device_calls),
                ms(split.uploads),
                split.upload_bytes
            );
        }
        hybrid.tokens.push(split);
    }
}

/// One MoE layer's routed experts as stored: 512 experts, an up and a down projection each,
/// packed nibbles (`rows * columns / 2`) plus one scale byte per 16 columns. Allocation rounding
/// comes on top; the tier replaces this with the measured cost after the first layer.
fn nominal_layer_bytes(experts: u32, intermediate: u32, latent: u32) -> u64 {
    let matrix = u64::from(intermediate) * u64::from(latent);
    let per_projection = matrix / 2 + matrix / 16;
    u64::from(experts) * 2 * per_projection
}

#[cfg(test)]
mod tests {
    #[test]
    fn nominal_layer_bytes_matches_the_pinned_checkpoint() {
        // 512 experts x (1,376,256 packed + 172,032 scales) x 2 projections = 1512 MiB.
        assert_eq!(
            super::nominal_layer_bytes(512, 2688, 1024),
            1512 * 1024 * 1024
        );
    }
}
