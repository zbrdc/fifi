//! The Nemotron glue of the CPU expert tier: the NVFP4 kernel behind the shared
//! [`CpuExpertKernel`] interface and the expert views over a mapped shard.
//!
//! The experts of a CPU-served MoE layer are never copied: the layer's shard file is mapped
//! read-only by the shared tier (the scoped `MappedFile`; no mlock, no madvise) and the NVFP4
//! pool kernel reads the packed bytes and group scales straight from the page cache. The pool is
//! created once per engine and its workers park between tokens.

use std::time::Duration;

use super::loader::ExpertLayerMapping;
use crate::{
    EngError,
    expert_tier::{CpuExpertKernel, CpuTier, ExpertPick, map_read_only},
    kernels::nvfp4_cpu::{
        ExpertPick as KernelPick, ExpertPool, ExpertView, LayerExperts, Nvfp4CpuError, Nvfp4Matrix,
        PoolConfig, WaitPolicy,
    },
};

/// Physical cores of the 7900X; the kernel is bandwidth-bound and 24 SMT threads gain nothing.
pub(crate) const DEFAULT_THREADS: usize = 12;
/// Workers spin this long after a call so consecutive CPU layers find them hot, then park.
const SPIN_WINDOW: Duration = Duration::from_micros(500);

/// The NVFP4 routed-expert kernel and its worker pool.
pub(crate) struct Nvfp4Kernel {
    pool: ExpertPool,
}

impl Nvfp4Kernel {
    pub(crate) fn new(threads: usize) -> Result<Self, EngError> {
        let pool = ExpertPool::new(PoolConfig {
            threads,
            wait: WaitPolicy::SpinThenPark(SPIN_WINDOW),
            ..PoolConfig::default()
        })
        .map_err(cpu_error)?;
        Ok(Self { pool })
    }
}

impl CpuExpertKernel for Nvfp4Kernel {
    type Layer = LayerExperts<'static>;

    fn threads(&self) -> usize {
        self.pool.threads()
    }

    fn routed(
        &self,
        layer: &LayerExperts<'static>,
        x: &[f32],
        picks: &[ExpertPick],
        out: &mut [f32],
    ) -> Result<(), EngError> {
        let picks: Vec<KernelPick> = picks
            .iter()
            .map(|pick| KernelPick {
                expert: pick.expert,
                weight: pick.weight,
            })
            .collect();
        self.pool
            .routed_experts(layer, x, &picks, out)
            .map_err(cpu_error)
    }

    fn routed_batch(
        &self,
        layer: &LayerExperts<'static>,
        xs: &[f32],
        picks: &[&[ExpertPick]],
        out: &mut [f32],
    ) -> Result<(), EngError> {
        let width = xs.len() / picks.len();
        let grouped: Vec<Vec<KernelPick>> = picks
            .iter()
            .map(|row| {
                row.iter()
                    .map(|pick| KernelPick {
                        expert: pick.expert,
                        weight: pick.weight,
                    })
                    .collect()
            })
            .collect();
        // The pool takes at most MAX_BATCH_ROWS rows; a row's accumulation
        // order (ascending expert id) is independent of its group.
        for ((xs, picks), out) in xs
            .chunks(width * crate::kernels::nvfp4_cpu::MAX_BATCH_ROWS)
            .zip(grouped.chunks(crate::kernels::nvfp4_cpu::MAX_BATCH_ROWS))
            .zip(out.chunks_mut(width * crate::kernels::nvfp4_cpu::MAX_BATCH_ROWS))
        {
            let refs: Vec<&[KernelPick]> = picks.iter().map(|row| row.as_slice()).collect();
            self.pool
                .routed_experts_batch(layer, xs, &refs, out)
                .map_err(cpu_error)?;
        }
        Ok(())
    }
}

/// The pool plus the mapped expert views of every CPU-served layer.
pub(crate) type CpuExperts = CpuTier<Nvfp4Kernel>;

pub(crate) fn new_cpu_experts(threads: usize) -> Result<CpuExperts, EngError> {
    Ok(CpuTier::new(Nvfp4Kernel::new(threads)?))
}

/// Maps the shard of one layer read-only (through the shared tier, which leaks the mapping for the
/// life of the model) and builds the zero-copy NVFP4 expert views over it.
pub(crate) fn layer_experts_from_mapping(
    mapping: &ExpertLayerMapping,
) -> Result<LayerExperts<'static>, EngError> {
    let bytes: &'static [u8] = map_read_only(&mapping.path, mapping.file_len)?;
    let slice = |range: &std::ops::Range<usize>| {
        bytes
            .get(range.clone())
            .ok_or_else(|| bad("tensor range exceeds the mapped shard"))
    };
    let scale = |range: &std::ops::Range<usize>| -> Result<f32, EngError> {
        let raw: [u8; 4] = slice(range)?
            .try_into()
            .map_err(|_| bad("weight_scale_2 must be four bytes"))?;
        let value = f32::from_le_bytes(raw);
        if !value.is_finite() || value <= 0.0 {
            return Err(bad("weight_scale_2 must be finite and positive"));
        }
        Ok(value)
    };
    let matrix = |ranges: &super::loader::ProjectionRanges| {
        Nvfp4Matrix::new(
            slice(&ranges.packed)?,
            slice(&ranges.scales)?,
            scale(&ranges.weight_scale_2)?,
            ranges.rows,
            ranges.columns,
        )
        .map_err(cpu_error)
    };
    let views = mapping
        .experts
        .iter()
        .map(|expert| {
            Ok(ExpertView {
                up: matrix(&expert.up)?,
                down: matrix(&expert.down)?,
            })
        })
        .collect::<Result<Vec<_>, EngError>>()?;
    LayerExperts::new(views).map_err(cpu_error)
}

/// Timing of the CPU routed experts of one layer over repeated random calls.
#[derive(Clone, Debug, PartialEq)]
pub struct CpuLayerTiming {
    pub layer: u32,
    /// The first call, which pays the page faults of a cold layer.
    pub first: Duration,
    /// The remaining calls in milliseconds, sorted ascending.
    pub warm_ms: Vec<f64>,
}

impl CpuLayerTiming {
    /// Median of the warm calls in milliseconds.
    pub fn p50_ms(&self) -> f64 {
        self.warm_ms
            .get(self.warm_ms.len() / 2)
            .copied()
            .unwrap_or(f64::NAN)
    }

    /// 90th percentile of the warm calls in milliseconds.
    pub fn p90_ms(&self) -> f64 {
        let last = self.warm_ms.len().saturating_sub(1);
        self.warm_ms
            .get((self.warm_ms.len() * 9 / 10).min(last))
            .copied()
            .unwrap_or(f64::NAN)
    }
}

/// Times the CPU routed experts of `layers` of a real checkpoint, one row and the model's top-k
/// per call, with random latent rows and random distinct experts. It measures the kernel through
/// the same mapping and pool the decode uses, without the rest of the stack, so a slow dense path
/// cannot hide it. Page-cache state is the caller's to set up: the first call of a layer is
/// reported apart from the warm ones.
pub fn time_cpu_experts(
    model_dir: &std::path::Path,
    layers: &[u32],
    threads: usize,
    iterations: usize,
) -> Result<Vec<CpuLayerTiming>, EngError> {
    let mut loader = super::loader::NemotronLoader::open(model_dir, 1)?;
    let moe = loader.moe_geometry();
    let (latent, count, top_k) = (
        moe.latent_size as usize,
        moe.routed_experts,
        moe.experts_per_token as usize,
    );
    let mut cpu = new_cpu_experts(threads)?;
    let mut state = 0x9e37_79b9_7f4a_7c15_u64;
    let mut next = move || {
        // xorshift64*: enough randomness to pick experts and activations.
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        state.wrapping_mul(0x2545_f491_4f6c_dd1d) >> 11
    };
    let mut timings = Vec::new();
    for &layer in layers {
        let mapping = loader.expert_layer_mapping(layer as usize)?;
        cpu.add_layer(layer, layer_experts_from_mapping(&mapping)?);
        let mut first = Duration::ZERO;
        let mut warm_ms = Vec::with_capacity(iterations);
        for call in 0..iterations.max(1) + 1 {
            let x: Vec<f32> = (0..latent)
                .map(|_| (next() % 2001) as f32 / 1000.0 - 1.0)
                .collect();
            let mut picks: Vec<ExpertPick> = Vec::with_capacity(top_k);
            while picks.len() < top_k {
                let expert = (next() % u64::from(count)) as u32;
                if picks.iter().all(|pick| pick.expert != expert) {
                    picks.push(ExpertPick {
                        expert,
                        weight: 1.0 / top_k as f32,
                    });
                }
            }
            let mut out = vec![0.0; latent];
            let started = std::time::Instant::now();
            cpu.routed(layer, &x, &picks, &mut out)?;
            let elapsed = started.elapsed();
            if call == 0 {
                first = elapsed;
            } else {
                warm_ms.push(elapsed.as_secs_f64() * 1e3);
            }
        }
        warm_ms.sort_by(f64::total_cmp);
        timings.push(CpuLayerTiming {
            layer,
            first,
            warm_ms,
        });
    }
    Ok(timings)
}

fn cpu_error(error: Nvfp4CpuError) -> EngError {
    match error {
        Nvfp4CpuError::Weights(_) => EngError::Checkpoint(error.to_string()),
        Nvfp4CpuError::Shape(_) | Nvfp4CpuError::Input(_) => {
            EngError::BadRequest(error.to_string())
        }
        Nvfp4CpuError::Pool(_) => EngError::Backend {
            operation: "nemotron35_cpu_experts",
            message: error.to_string(),
        },
    }
}

fn bad(message: impl Into<String>) -> EngError {
    EngError::Checkpoint(format!("Nemotron CPU experts: {}", message.into()))
}

#[cfg(test)]
mod tests {
    use std::io::Write as _;

    use super::*;
    use crate::{
        family::nemotron35::loader::{ExpertRanges, ProjectionRanges},
        kernels::nvfp4::WeightBlock,
    };

    const LATENT: usize = 32;
    const INTER: usize = 64;
    const EXPERTS: usize = 6;
    const PROJECTION_BYTES: usize = INTER * LATENT / 2 + INTER * LATENT / 16 + 4;

    /// Deterministic bytes: a small LCG, so the fixture needs no random-number crate.
    struct Lcg(u64);

    impl Lcg {
        fn next(&mut self) -> u8 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (self.0 >> 33) as u8
        }
    }

    /// One shard-shaped file with a junk prefix, so every range starts at a nonzero, unaligned
    /// offset, holding `EXPERTS` experts of two projections each (packed, scales, scalar).
    fn fixture(weight_scale_2: f32) -> (tempfile::NamedTempFile, ExpertLayerMapping) {
        let mut rng = Lcg(7);
        let mut bytes = vec![0xaa_u8; 13];
        let mut experts = Vec::new();
        let mut projection = |bytes: &mut Vec<u8>, rows: usize, columns: usize| {
            let packed_start = bytes.len();
            bytes.extend((0..rows * columns / 2).map(|_| rng.next()));
            let scales_start = bytes.len();
            // E4M3 codes 0x28..=0x47 are finite positive scales of moderate size.
            bytes.extend((0..rows * columns / 16).map(|_| 0x28 + rng.next() % 32));
            let scalar_start = bytes.len();
            bytes.extend_from_slice(&weight_scale_2.to_le_bytes());
            ProjectionRanges {
                packed: packed_start..scales_start,
                scales: scales_start..scalar_start,
                weight_scale_2: scalar_start..scalar_start + 4,
                rows,
                columns,
            }
        };
        for _ in 0..EXPERTS {
            let up = projection(&mut bytes, INTER, LATENT);
            let down = projection(&mut bytes, LATENT, INTER);
            experts.push(ExpertRanges { up, down });
        }
        assert_eq!(bytes.len(), 13 + EXPERTS * 2 * PROJECTION_BYTES);
        let mut file = tempfile::NamedTempFile::new().expect("temporary shard");
        file.write_all(&bytes).expect("write shard");
        file.flush().expect("flush shard");
        let mapping = ExpertLayerMapping {
            path: file.path().to_path_buf(),
            file_len: bytes.len() as u64,
            experts,
        };
        (file, mapping)
    }

    /// The engine's own sequential NVFP4 reference for one expert: up, relu2, down.
    fn reference(
        bytes: &[u8],
        mapping: &ExpertLayerMapping,
        pick: &ExpertPick,
        x: &[f32],
    ) -> Vec<f32> {
        let ranges = &mapping.experts[pick.expert as usize];
        let project = |ranges: &ProjectionRanges, input: &[f32]| {
            let scale =
                f32::from_le_bytes(bytes[ranges.weight_scale_2.clone()].try_into().unwrap());
            let mut output = vec![0.0; ranges.rows];
            WeightBlock {
                packed: &bytes[ranges.packed.clone()],
                weight_scale: &bytes[ranges.scales.clone()],
                weight_scale_2: scale,
                input_scale: None,
            }
            .matvec(ranges.rows, ranges.columns, input, &mut output)
            .expect("reference matvec");
            output
        };
        let mut hidden = project(&ranges.up, x);
        for value in &mut hidden {
            *value = value.max(0.0).powi(2);
        }
        project(&ranges.down, &hidden)
            .into_iter()
            .map(|value| value * pick.weight)
            .collect()
    }

    #[test]
    fn mapped_views_match_the_sequential_reference_within_rounding() {
        let (file, mapping) = fixture(0.0625);
        let bytes = std::fs::read(file.path()).unwrap();
        let mut cpu = new_cpu_experts(3).expect("pool");
        assert!(!cpu.serves(7));
        cpu.add_layer(7, layer_experts_from_mapping(&mapping).expect("views"));
        assert!(cpu.serves(7));
        assert_eq!(cpu.layer_count(), 1);
        assert_eq!(cpu.threads(), 3);

        let x: Vec<f32> = (0..LATENT)
            .map(|i| ((i * 7 % 11) as f32 - 5.0) * 0.25)
            .collect();
        let picks = [
            ExpertPick {
                expert: 4,
                weight: 0.75,
            },
            ExpertPick {
                expert: 1,
                weight: 1.5,
            },
            ExpertPick {
                expert: 5,
                weight: 0.125,
            },
        ];
        let mut out = vec![f32::NAN; LATENT];
        cpu.routed(7, &x, &picks, &mut out).expect("routed experts");

        // The reference accumulates in ascending expert id, like the kernel.
        let mut sorted = picks;
        sorted.sort_by_key(|pick| pick.expert);
        let mut expected = vec![0.0_f32; LATENT];
        for pick in &sorted {
            for (sum, value) in expected
                .iter_mut()
                .zip(reference(&bytes, &mapping, pick, &x))
            {
                *sum += value;
            }
        }
        let peak = expected.iter().fold(0.0_f32, |peak, v| peak.max(v.abs()));
        assert!(peak > 0.0, "fixture must produce a nonzero output");
        for (got, want) in out.iter().zip(&expected) {
            assert!(
                (got - want).abs() <= 1e-5 * peak,
                "kernel {got} differs from the reference {want} (peak {peak})"
            );
        }
    }

    #[test]
    fn layer_timing_reports_median_and_tail_of_the_warm_calls() {
        let timing = CpuLayerTiming {
            layer: 3,
            first: Duration::from_millis(100),
            warm_ms: (1..=10).map(f64::from).collect(),
        };
        assert_eq!(timing.p50_ms(), 6.0);
        assert_eq!(timing.p90_ms(), 10.0);
        let empty = CpuLayerTiming {
            layer: 0,
            first: Duration::ZERO,
            warm_ms: Vec::new(),
        };
        assert!(empty.p50_ms().is_nan() && empty.p90_ms().is_nan());
        let missing = std::path::Path::new("/nonexistent-nem35-model");
        assert!(time_cpu_experts(missing, &[1], 2, 3).is_err());
    }

    #[test]
    fn refusals_are_typed_and_leave_no_partial_state() {
        let (file, mapping) = fixture(0.0625);
        let mut cpu = new_cpu_experts(2).expect("pool");
        let mut out = vec![0.0; LATENT];
        let pick = [ExpertPick {
            expert: 0,
            weight: 1.0,
        }];
        // A layer that was never mapped is a caller error, not a panic.
        assert!(matches!(
            cpu.routed(3, &[0.0; LATENT], &pick, &mut out),
            Err(EngError::BadRequest(_))
        ));
        cpu.add_layer(3, layer_experts_from_mapping(&mapping).expect("views"));
        // An expert id past the layer, and a non-finite activation, are input errors.
        let missing = [ExpertPick {
            expert: EXPERTS as u32,
            weight: 1.0,
        }];
        assert!(matches!(
            cpu.routed(3, &[0.0; LATENT], &missing, &mut out),
            Err(EngError::BadRequest(_))
        ));
        let mut bad_x = [0.0; LATENT];
        bad_x[5] = f32::NAN;
        assert!(matches!(
            cpu.routed(3, &bad_x, &pick, &mut out),
            Err(EngError::BadRequest(_))
        ));
        drop(file);
    }

    #[test]
    fn a_wrong_file_length_range_or_scale_is_refused_when_mapping() {
        let (_file, mut mapping) = fixture(0.0625);
        mapping.file_len += 1;
        assert!(matches!(
            layer_experts_from_mapping(&mapping),
            Err(EngError::Checkpoint(_))
        ));
        mapping.file_len -= 1;
        let end = mapping.file_len as usize;
        mapping.experts[2].down.weight_scale_2 = end..end + 4;
        assert!(matches!(
            layer_experts_from_mapping(&mapping),
            Err(EngError::Checkpoint(_))
        ));
        for scale in [0.0, -1.0, f32::NAN, f32::INFINITY] {
            let (_file, mapping) = fixture(scale);
            assert!(
                matches!(
                    layer_experts_from_mapping(&mapping),
                    Err(EngError::Checkpoint(_))
                ),
                "scale {scale} must be refused"
            );
        }
    }
}
