//! The tensor names of a stock EXL3 pack and the header-only walk that checks them, shared by the
//! families whose packs use the HF layout (Qwen3.6, Qwen3.8 Flash-Next).
//!
//! An [`Inspector`] reads no payload: each call checks one tensor's stored dtype and shape against
//! the model, and the walk sums the device bytes the load will hold. A quantized matrix must have
//! the current component set (trellis, `suh`, `svh`, one codebook marker) and every matrix of a
//! pack must name the same codebook. A family's own walk decides which tensors to visit.

use std::collections::BTreeMap;

use crate::{EngError, tensor::Dtype, weights::StreamedCheckpoint};

const LAYER_ROOT: &str = "model.language_model.layers";
pub(in crate::family) const EMBED_NAME: &str = "model.language_model.embed_tokens.weight";
pub(in crate::family) const NORM_NAME: &str = "model.language_model.norm.weight";
pub(in crate::family) const HEAD_PREFIX: &str = "lm_head";

/// Root of every tensor name of the MTP head's layers in a stock pack.
const MTP_LAYER_ROOT: &str = "mtp.layers";

/// Where the layers of a walk live in the pack: the model's own layers or the MTP head's.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::family) enum Root {
    Layers,
    Mtp,
}

impl Root {
    /// The full pack name of `suffix` in `layer`.
    pub(in crate::family) fn name(self, layer: usize, suffix: &str) -> String {
        let root = match self {
            Self::Layers => LAYER_ROOT,
            Self::Mtp => MTP_LAYER_ROOT,
        };
        format!("{root}.{layer}.{suffix}")
    }
}

/// The full pack name of `suffix` in `layer`.
pub(in crate::family) fn layer_name(layer: usize, suffix: &str) -> String {
    Root::Layers.name(layer, suffix)
}

/// The router weight of a layer under `root`: a plain (unquantized) tensor of expert scores.
pub(in crate::family) fn router_name_in(root: Root, layer: usize) -> String {
    root.name(layer, "mlp.gate.weight")
}

/// The sigmoid gate of the shared expert under `root`: a plain tensor of one row.
pub(in crate::family) fn shared_gate_name_in(root: Root, layer: usize) -> String {
    root.name(layer, "mlp.shared_expert_gate.weight")
}

/// One projection of one routed expert under `root`.
pub(in crate::family) fn expert_name_in(
    root: Root,
    layer: usize,
    expert: usize,
    part: ExpertPart,
) -> String {
    root.name(layer, &format!("mlp.experts.{expert}.{}", part.suffix()))
}

/// One projection of the shared expert under `root`.
pub(in crate::family) fn shared_name_in(root: Root, layer: usize, part: ExpertPart) -> String {
    root.name(layer, &format!("mlp.shared_expert.{}", part.suffix()))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::family) enum ExpertPart {
    Gate,
    Up,
    Down,
}

impl ExpertPart {
    pub(in crate::family) const ALL: [Self; 3] = [Self::Gate, Self::Up, Self::Down];

    fn suffix(self) -> &'static str {
        match self {
            Self::Gate => "gate_proj",
            Self::Up => "up_proj",
            Self::Down => "down_proj",
        }
    }
}

/// The widths of one routed-expert block: `hidden` in and out, `experts` routed experts of
/// `inter` columns each, and one shared expert of `shared_inter`.
#[derive(Debug, Clone, Copy)]
pub(in crate::family) struct MoeBlock {
    pub(in crate::family) hidden: usize,
    pub(in crate::family) experts: usize,
    pub(in crate::family) inter: usize,
    pub(in crate::family) shared_inter: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(in crate::family) struct PackReport {
    /// Device bytes the load holds: trellises as stored, scales and plain vectors widened to
    /// F32, the embedding as stored.
    pub(in crate::family) device_bytes: u64,
    /// Device bytes of the routed experts alone, the part that expert tiers place.
    pub(in crate::family) expert_bytes: u64,
    /// Quantized matrices (a routed expert counts one per projection).
    pub(in crate::family) matrices: usize,
    /// Matrices per rate, keyed by twice the bits per weight.
    pub(in crate::family) rates: BTreeMap<u8, usize>,
    pub(in crate::family) codebook: Marker,
}

/// The family-neutral planning summary of one pack directory: exactly the
/// loader's own header walk, with its totals copied out for callers outside `family`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct PackPlan {
    /// Device bytes the load will hold ([`PackReport::device_bytes`]).
    pub(crate) device_bytes: u64,
    /// Device bytes of the routed experts alone ([`PackReport::expert_bytes`]).
    pub(crate) expert_bytes: u64,
    /// Quantized matrices (a routed expert counts one per projection).
    pub(crate) matrices: usize,
    /// Matrices per rate, keyed by twice the bits per weight.
    pub(crate) rates: BTreeMap<u8, usize>,
    /// The codebook marker tensor every matrix of the pack carries, by name.
    pub(crate) codebook: &'static str,
}

/// The name of the codebook marker a matrix carries, for rows that retell the walk.
pub(in crate::family) fn codebook_name(marker: Marker) -> &'static str {
    match marker {
        Marker::Mul1 => "Mul1",
        Marker::Mcg => "Mcg",
    }
}

/// Copies a walk report into the family-neutral plan.
pub(in crate::family) fn plan_from_report(report: PackReport) -> PackPlan {
    PackPlan {
        device_bytes: report.device_bytes,
        expert_bytes: report.expert_bytes,
        matrices: report.matrices,
        rates: report.rates,
        codebook: codebook_name(report.codebook),
    }
}

/// What kind of tensor a walk visited.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(in crate::family) enum VisitKind {
    /// A quantized matrix under `name` (its prefix), `k` inputs by `n` outputs, at `rate` (twice
    /// the bits per weight).
    Matrix { k: usize, n: usize, rate: u8 },
    /// A plain tensor with its stored dtype and shape; `keep` is the walk's rule for its device
    /// form: as stored (`true`) or widened to F32 (`false`).
    Plain {
        dtype: Dtype,
        shape: Vec<u32>,
        keep: bool,
    },
    /// A plain tensor the device does not hold: a host feed reads it from the pack. Checked like
    /// any plain tensor, counted as zero device bytes.
    Host { dtype: Dtype, shape: Vec<u32> },
}

/// One tensor a recording walk visited, with the device bytes the walk counted for it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(in crate::family) struct Visit {
    pub(in crate::family) name: String,
    pub(in crate::family) kind: VisitKind,
    pub(in crate::family) bytes: u64,
}

/// The codebook marker tensor a matrix carries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::family) enum Marker {
    Mul1,
    Mcg,
}

fn refuse(detail: String) -> EngError {
    EngError::Checkpoint(format!("EXL3 pack: {detail}"))
}

/// The walk: the source and the running totals.
pub(in crate::family) struct Inspector<'a> {
    source: &'a StreamedCheckpoint,
    root: Root,
    bytes: u64,
    expert_bytes: u64,
    matrices: usize,
    rates: BTreeMap<u8, usize>,
    marker: Option<Marker>,
    visits: Option<Vec<Visit>>,
}

impl<'a> Inspector<'a> {
    /// A walk over the layers under `root` of `source`.
    pub(in crate::family) fn new(source: &'a StreamedCheckpoint, root: Root) -> Self {
        Self {
            source,
            root,
            bytes: 0,
            expert_bytes: 0,
            matrices: 0,
            rates: BTreeMap::new(),
            marker: None,
            visits: None,
        }
    }

    /// Records every tensor the walk visits, in order, for [`Self::report_with_visits`]. A loader
    /// that consumes the list cannot drift from the walk that checked the pack.
    pub(in crate::family) fn recording(mut self) -> Self {
        self.visits = Some(Vec::new());
        self
    }

    fn record(&mut self, name: &str, kind: VisitKind, bytes: u64) {
        if let Some(visits) = &mut self.visits {
            visits.push(Visit {
                name: name.to_owned(),
                kind,
                bytes,
            });
        }
    }

    /// Where the layers of this walk live.
    pub(in crate::family) fn root(&self) -> Root {
        self.root
    }

    /// What the walk found so far.
    pub(in crate::family) fn report(self) -> Result<PackReport, EngError> {
        Ok(PackReport {
            device_bytes: self.bytes,
            expert_bytes: self.expert_bytes,
            matrices: self.matrices,
            rates: self.rates,
            codebook: self
                .marker
                .ok_or_else(|| refuse("the pack has no matrix".into()))?,
        })
    }

    /// [`Self::report`] and the visits of a [`Self::recording`] walk (empty for a plain one).
    pub(in crate::family) fn report_with_visits(
        mut self,
    ) -> Result<(PackReport, Vec<Visit>), EngError> {
        let visits = self.visits.take().unwrap_or_default();
        Ok((self.report()?, visits))
    }

    fn stored(&self, name: &str, want: &[u32]) -> Result<Dtype, EngError> {
        let (dtype, shape) = self
            .source
            .stored(name)
            .ok_or_else(|| refuse(format!("missing tensor {name}")))?;
        if shape != want {
            return Err(refuse(format!(
                "{name} is {shape:?}, the model needs {want:?}"
            )));
        }
        Ok(dtype)
    }

    fn add(&mut self, bytes: u64) {
        self.bytes += bytes;
    }

    /// A plain tensor of `shape`, half width as stored and F32 on the device, or kept as stored.
    /// Returns the stored dtype.
    pub(in crate::family) fn plain(
        &mut self,
        name: &str,
        shape: &[u32],
        keep: bool,
    ) -> Result<Dtype, EngError> {
        let dtype = self.stored(name, shape)?;
        if !matches!(dtype, Dtype::F16 | Dtype::Bf16 | Dtype::F32) {
            return Err(refuse(format!("{name} has dtype {dtype:?}")));
        }
        let values = shape.iter().map(|&d| u64::from(d)).product::<u64>();
        let bytes = values * if keep { dtype.byte_width() } else { 4 };
        self.add(bytes);
        self.record(
            name,
            VisitKind::Plain {
                dtype,
                shape: shape.to_vec(),
                keep,
            },
            bytes,
        );
        Ok(dtype)
    }

    /// A plain tensor that stays off the device because a host feed reads it (the PLE norms and
    /// convolution): checked against `shape` like [`Self::plain`], counted as zero bytes.
    pub(in crate::family) fn host_plain(
        &mut self,
        name: &str,
        shape: &[u32],
    ) -> Result<Dtype, EngError> {
        let dtype = self.stored(name, shape)?;
        if !matches!(dtype, Dtype::F16 | Dtype::Bf16 | Dtype::F32) {
            return Err(refuse(format!("{name} has dtype {dtype:?}")));
        }
        self.record(
            name,
            VisitKind::Host {
                dtype,
                shape: shape.to_vec(),
            },
            0,
        );
        Ok(dtype)
    }

    /// The quantized matrix under `prefix`, `k` inputs by `n` outputs, at some rate; returns the
    /// rate (twice the bits per weight).
    pub(in crate::family) fn matrix(
        &mut self,
        prefix: &str,
        k: usize,
        n: usize,
    ) -> Result<u8, EngError> {
        let dim = |value: usize| {
            u32::try_from(value)
                .map_err(|_| refuse(format!("{prefix}: dimension {value} exceeds u32")))
        };
        if !k.is_multiple_of(128) || !n.is_multiple_of(128) {
            return Err(refuse(format!(
                "{prefix}: {k} by {n} is not a multiple of 128 on both axes"
            )));
        }
        let (tiles_k, tiles_n) = (dim(k / 16)?, dim(n / 16)?);
        let name = format!("{prefix}.trellis");
        let (dtype, shape) = self
            .source
            .stored(&name)
            .ok_or_else(|| refuse(format!("missing tensor {name}")))?;
        let words = match shape.as_slice() {
            [a, b, words] if (*a, *b) == (tiles_k, tiles_n) && dtype == Dtype::I16 => *words,
            _ => {
                return Err(refuse(format!(
                    "{name} is {dtype:?} {shape:?}, the model needs I16 [{tiles_k}, {tiles_n}, w]"
                )));
            }
        };
        if words % 8 != 0 || !(16..=128).contains(&words) {
            return Err(refuse(format!("{name} has {words} words per tile")));
        }
        for (suffix, len) in [("suh", k), ("svh", n)] {
            let name = format!("{prefix}.{suffix}");
            if self.stored(&name, &[dim(len)?])? != Dtype::F16 {
                return Err(refuse(format!("{name} is not F16")));
            }
        }
        for legacy in ["su", "sv", "scale"] {
            if self.source.contains(&format!("{prefix}.{legacy}")) {
                return Err(refuse(format!(
                    "{prefix} stores the old component {legacy}"
                )));
            }
        }
        self.mark(prefix)?;
        let trellis = u64::from(tiles_k) * u64::from(tiles_n) * u64::from(words) * 2;
        let bytes = trellis + (k as u64 + n as u64) * 4;
        self.add(bytes);
        self.matrices += 1;
        let rate = (words / 8) as u8;
        self.record(prefix, VisitKind::Matrix { k, n, rate }, bytes);
        *self.rates.entry(rate).or_default() += 1;
        Ok(rate)
    }

    fn mark(&mut self, prefix: &str) -> Result<(), EngError> {
        let found = [("mul1", Marker::Mul1), ("mcg", Marker::Mcg)]
            .into_iter()
            .filter(|(suffix, _)| self.source.contains(&format!("{prefix}.{suffix}")))
            .map(|(_, marker)| marker)
            .collect::<Vec<_>>();
        let [marker] = found.as_slice() else {
            return Err(refuse(format!(
                "{prefix} carries {} codebook markers",
                found.len()
            )));
        };
        if *self.marker.get_or_insert(*marker) != *marker {
            return Err(refuse(format!(
                "{prefix} has another codebook than the pack"
            )));
        }
        Ok(())
    }

    /// The routed-expert block of `layer`: the router and the shared-expert gate as plain tensors
    /// (F32 on the device), the routed experts of each projection at one rate (a bank has one
    /// stride), and the shared expert. Returns the device bytes of the routed experts alone.
    pub(in crate::family) fn moe_block(
        &mut self,
        layer: usize,
        block: MoeBlock,
    ) -> Result<u64, EngError> {
        let MoeBlock {
            hidden: h,
            experts,
            inter,
            shared_inter,
        } = block;
        let dim = |value: usize| u32::try_from(value).unwrap_or(u32::MAX);
        self.plain(
            &router_name_in(self.root, layer),
            &[dim(experts), dim(h)],
            false,
        )?;
        self.plain(&shared_gate_name_in(self.root, layer), &[1, dim(h)], false)?;
        let mut routed = 0;
        for part in ExpertPart::ALL {
            let (k, n) = match part {
                ExpertPart::Down => (inter, h),
                _ => (h, inter),
            };
            let before = self.bytes;
            let mut bank_rate = None;
            for expert in 0..experts {
                let rate = self.matrix(&expert_name_in(self.root, layer, expert, part), k, n)?;
                if *bank_rate.get_or_insert(rate) != rate {
                    return Err(refuse(format!(
                        "layer {layer} {part:?} experts differ in rate: a bank has one stride"
                    )));
                }
            }
            routed += self.bytes - before;
            self.expert_bytes += self.bytes - before;
            let (k, n) = match part {
                ExpertPart::Down => (shared_inter, h),
                _ => (h, shared_inter),
            };
            self.matrix(&shared_name_in(self.root, layer, part), k, n)?;
        }
        Ok(routed)
    }
}
