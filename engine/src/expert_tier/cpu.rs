//! The host side of the tier: a format-specific CPU kernel, its worker pool, and the read-only
//! mapped views of the experts that stay on the host.

use std::{collections::BTreeMap, path::Path};

use crate::{EngError, family::qwen36::tiered::mapped_file::MappedFile};

/// One selected expert and the router weight its output is scaled by.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ExpertPick {
    pub expert: u32,
    pub weight: f32,
}

/// The format-specific CPU kernel of the tier, with the pool it runs on.
///
/// `Layer` is the kernel's own view of one layer's experts (for Nemotron, 512 pairs of NVFP4
/// matrix views into the mapped shard). The kernel owns its threads: it is created once per
/// engine, and its call is synchronous, with the calling thread computing too.
pub trait CpuExpertKernel: Sync {
    type Layer: Send + Sync;

    fn threads(&self) -> usize;

    /// One activation row through the picked experts of one layer:
    /// `out = sum_e weight_e * expert_e(x)`, accumulated in ascending expert id.
    fn routed(
        &self,
        layer: &Self::Layer,
        x: &[f32],
        picks: &[ExpertPick],
        out: &mut [f32],
    ) -> Result<(), EngError>;

    /// Several rows at once: row `t` of `xs`/`out` uses `picks[t]`, and each
    /// row still accumulates its experts in ascending expert id — so a row's
    /// result is bit-identical to routing it alone. The default loops
    /// [`Self::routed`]; kernels with a real batch entry override it.
    fn routed_batch(
        &self,
        layer: &Self::Layer,
        xs: &[f32],
        picks: &[&[ExpertPick]],
        out: &mut [f32],
    ) -> Result<(), EngError> {
        if picks.is_empty() || picks.iter().any(|row| row.is_empty()) {
            return Err(EngError::BadRequest(
                "CPU experts: routed_batch needs one non-empty pick set per row".into(),
            ));
        }
        let width = xs.len() / picks.len();
        if width == 0 || !xs.len().is_multiple_of(picks.len()) {
            return Err(EngError::BadRequest(
                "CPU experts: routed_batch activation rows do not match the pick sets".into(),
            ));
        }
        for ((x, picks), out) in xs
            .chunks_exact(width)
            .zip(picks)
            .zip(out.chunks_exact_mut(width))
        {
            self.routed(layer, x, picks, out)?;
        }
        Ok(())
    }
}

/// The kernel plus the mapped layers it serves.
pub struct CpuTier<K: CpuExpertKernel> {
    kernel: K,
    layers: BTreeMap<u32, K::Layer>,
}

impl<K: CpuExpertKernel> CpuTier<K> {
    pub fn new(kernel: K) -> Self {
        Self {
            kernel,
            layers: BTreeMap::new(),
        }
    }

    pub fn threads(&self) -> usize {
        self.kernel.threads()
    }

    pub fn add_layer(&mut self, layer: u32, experts: K::Layer) {
        self.layers.insert(layer, experts);
    }

    #[cfg(test)]
    pub fn serves(&self, layer: u32) -> bool {
        self.layers.contains_key(&layer)
    }

    #[cfg(test)]
    pub fn layer_count(&self) -> usize {
        self.layers.len()
    }

    pub fn layer_ids(&self) -> Vec<u32> {
        self.layers.keys().copied().collect()
    }

    /// One row through the picked experts of `layer`. A layer that was never mapped is a caller
    /// error, not a panic.
    pub fn routed(
        &self,
        layer: u32,
        x: &[f32],
        picks: &[ExpertPick],
        out: &mut [f32],
    ) -> Result<(), EngError> {
        let experts = self.layers.get(&layer).ok_or_else(|| {
            EngError::BadRequest(format!("CPU experts: layer {layer} is not CPU-served"))
        })?;
        self.kernel.routed(experts, x, picks, out)
    }

    /// Rows through their picked experts of `layer`, one pick-set per row.
    pub fn routed_batch(
        &self,
        layer: u32,
        xs: &[f32],
        picks: &[&[ExpertPick]],
        out: &mut [f32],
    ) -> Result<(), EngError> {
        let experts = self.layers.get(&layer).ok_or_else(|| {
            EngError::BadRequest(format!("CPU experts: layer {layer} is not CPU-served"))
        })?;
        self.kernel.routed_batch(experts, xs, picks, out)
    }
}

/// Maps a whole file read-only and returns its bytes for the life of the process.
///
/// The mapping is leaked on purpose: expert views borrow it for as long as the model lives, and a
/// model lives as long as the process (the checkpoint is immutable by policy). The scoped
/// `MappedFile` (decision 0017, Addendum 1) does the mapping: no mlock, no madvise, and a refusal
/// when the length is not the one the caller expects.
pub fn map_read_only(path: &Path, expected_len: u64) -> Result<&'static [u8], EngError> {
    let file = MappedFile::open(path, expected_len)
        .map_err(|error| EngError::Checkpoint(format!("map {}: {error}", path.display())))?;
    let file: &'static MappedFile = Box::leak(Box::new(file));
    Ok(file.as_ref())
}

#[cfg(test)]
mod tests {
    use std::io::Write as _;

    use super::*;

    /// A kernel that sums `weight * x[expert]` so the tier's plumbing is checkable on its own.
    struct Summing;

    impl CpuExpertKernel for Summing {
        type Layer = &'static [u8];

        fn threads(&self) -> usize {
            3
        }

        fn routed(
            &self,
            layer: &Self::Layer,
            x: &[f32],
            picks: &[ExpertPick],
            out: &mut [f32],
        ) -> Result<(), EngError> {
            for pick in picks {
                let scale = f32::from(
                    *layer
                        .get(pick.expert as usize)
                        .ok_or_else(|| EngError::BadRequest("expert past the layer".to_owned()))?,
                );
                for (slot, value) in out.iter_mut().zip(x) {
                    *slot += pick.weight * scale * value;
                }
            }
            Ok(())
        }
    }

    #[test]
    fn the_tier_routes_to_the_kernel_with_the_mapped_layer() {
        let mut file = tempfile::NamedTempFile::new().expect("temporary file");
        file.write_all(&[1, 2, 3, 4]).unwrap();
        file.flush().unwrap();
        let bytes = map_read_only(file.path(), 4).expect("map");
        assert_eq!(bytes, [1, 2, 3, 4]);

        let mut tier = CpuTier::new(Summing);
        assert!(!tier.serves(5));
        tier.add_layer(5, bytes);
        assert!(tier.serves(5));
        assert_eq!(tier.layer_ids(), vec![5]);
        assert_eq!(tier.layer_count(), 1);
        assert_eq!(tier.threads(), 3);
        let mut out = [0.0; 2];
        let picks = [
            ExpertPick {
                expert: 1,
                weight: 0.5,
            },
            ExpertPick {
                expert: 3,
                weight: 2.0,
            },
        ];
        tier.routed(5, &[1.0, 10.0], &picks, &mut out).unwrap();
        // 0.5 * 2 + 2 * 4 = 9 per unit of x.
        assert_eq!(out, [9.0, 90.0]);
        assert!(matches!(
            tier.routed(6, &[1.0, 1.0], &picks, &mut out),
            Err(EngError::BadRequest(_))
        ));
        assert!(matches!(
            tier.routed(
                5,
                &[1.0, 1.0],
                &[ExpertPick {
                    expert: 9,
                    weight: 1.0
                }],
                &mut out
            ),
            Err(EngError::BadRequest(_))
        ));
    }

    #[test]
    fn a_wrong_length_or_a_missing_file_is_a_checkpoint_error() {
        let mut file = tempfile::NamedTempFile::new().expect("temporary file");
        file.write_all(&[9; 8]).unwrap();
        file.flush().unwrap();
        assert!(matches!(
            map_read_only(file.path(), 9),
            Err(EngError::Checkpoint(_))
        ));
        assert!(matches!(
            map_read_only(Path::new("/nonexistent-expert-shard"), 1),
            Err(EngError::Checkpoint(_))
        ));
    }
}
