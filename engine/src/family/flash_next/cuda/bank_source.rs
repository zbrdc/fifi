//! The expert records of the real Flash-Next pack: the [`RecordSource`] of the expert bank.
//!
//! WHAT: [`PackRecordSource`] reads the nine components of each of an expert's three matrices
//! (`trellis`, `suh`, `svh` and the codebook words) from an opened pack, loads them as `Matrix`es
//! and packs them with `SlotBankLayout::pack_matrices` into the one record the bank stores
//! (`trellis | suh | svh` per part, scales widened to F32). The caller opens the pack once, through
//! `FlashPackFiles::open` (one `LocalGuard` over the directory, every shard and the side files),
//! and lends its `Checkpoint`; this module opens nothing. WHY a borrow: the trunk loader and the
//! n-gram feed read the same pack, so one open serves all of them. The bank reads the pack at build
//! only (the host store and the first fill of the slots); a miss copies from the pinned store.
//!
//! The layout (widths and rates) is derived from the pack itself: the gate matrix of expert 0 of
//! layer 0 gives `[hidden, inter]` and the rate, `pack_matrices` refuses any other expert whose
//! shape or rate differs (the refusal names the layer and expert), a matrix with a bias is refused
//! (the record carries none), and `new` checks once that `record * layers * experts` is the
//! pack's routed-expert total from the header walk. Nothing here allocates device memory.
//!
//! Staged: the CUDA decode step is the first caller.

#![cfg_attr(not(test), allow(dead_code))]

use super::bank::RecordSource;
use crate::{
    EngError,
    family::{
        exl3_pack::{ExpertPart, Root, expert_name_in},
        flash_next::{
            load::{model_shape, pack_files::FlashPackFiles},
            moe::slot_bank::SlotBankLayout,
        },
    },
    weights::exl3::{COMPONENT_SUFFIXES, Checkpoint, Matrix},
};

/// Bytes one matrix read may take (the components of one matrix are well under 1 MiB at 3 bpw).
const MATRIX_READ_BUDGET: u64 = 1 << 30;

fn pack_error(detail: String) -> EngError {
    EngError::Checkpoint(format!("Flash-Next expert records: {detail}"))
}

/// The records of the routed experts of an opened pack.
pub(super) struct PackRecordSource<'a> {
    checkpoint: &'a Checkpoint,
    layout: SlotBankLayout,
    layers: usize,
    experts: usize,
}

impl<'a> PackRecordSource<'a> {
    /// Takes the layer and expert counts from `config.json` and the record layout from the first
    /// expert of the first layer, then checks the layout against the header walk's routed-expert
    /// total (`files.report.expert_bytes`), so every expert has the layout's size or the pack was
    /// refused at open.
    pub(super) fn from_files(files: &'a FlashPackFiles) -> Result<Self, EngError> {
        let source = Self::new(&files.checkpoint)?;
        source.check_expert_bytes(files.report.expert_bytes)?;
        Ok(source)
    }

    /// As [`Self::from_files`] without the check against the header walk's total, for a caller
    /// that holds only the opened `Checkpoint`.
    pub(super) fn new(checkpoint: &'a Checkpoint) -> Result<Self, EngError> {
        let (shape, kinds) = model_shape(&checkpoint.model_config)?;
        let (layers, experts) = (kinds.len(), shape.moe.experts());
        let gate = load_matrix(
            checkpoint,
            &expert_name_in(Root::Layers, 0, 0, ExpertPart::Gate),
        )?;
        refuse_bias(&gate, 0, 0, ExpertPart::Gate)?;
        let [hidden, inter] = gate.shape();
        let words = usize::from(gate.bitrate_twice()) * 8;
        let layout = SlotBankLayout::new(hidden, inter, [words; 3])?;
        Ok(Self::from_parts(checkpoint, layout, layers, experts))
    }

    /// A source over `checkpoint` with a known layout and counts (tests build small packs).
    pub(super) fn from_parts(
        checkpoint: &'a Checkpoint,
        layout: SlotBankLayout,
        layers: usize,
        experts: usize,
    ) -> Self {
        Self {
            checkpoint,
            layout,
            layers,
            experts,
        }
    }

    /// Refuses a pack whose routed-expert total is not `record * layers * experts`.
    pub(super) fn check_expert_bytes(&self, total: u64) -> Result<(), EngError> {
        let records = (self.layers as u64).checked_mul(self.experts as u64);
        let expected =
            records.and_then(|count| count.checked_mul(self.layout.record_bytes() as u64));
        if expected != Some(total) {
            return Err(pack_error(format!(
                "{} layers of {} experts at {} bytes a record are {expected:?} bytes, the header \
                 walk counts {total}",
                self.layers,
                self.experts,
                self.layout.record_bytes()
            )));
        }
        Ok(())
    }

    pub(super) fn layout(&self) -> &SlotBankLayout {
        &self.layout
    }

    pub(super) fn layers(&self) -> usize {
        self.layers
    }

    pub(super) fn experts(&self) -> usize {
        self.experts
    }

    /// The record of `expert` in `layer`. Refuses an index outside the pack, a matrix with a bias,
    /// and an expert whose matrices differ in shape or rate from the layout (the refusal names the
    /// layer and the expert). A missing tensor or a failed read is refused by the loader with the
    /// tensor's name.
    pub(super) fn read_record(&self, layer: usize, expert: usize) -> Result<Vec<u8>, EngError> {
        if layer >= self.layers || expert >= self.experts {
            return Err(pack_error(format!(
                "layer {layer} expert {expert} is outside {} layers of {} experts",
                self.layers, self.experts
            )));
        }
        let mut matrices = Vec::with_capacity(3);
        for part in ExpertPart::ALL {
            let matrix = load_matrix(
                self.checkpoint,
                &expert_name_in(Root::Layers, layer, expert, part),
            )?;
            refuse_bias(&matrix, layer, expert, part)?;
            matrices.push(matrix);
        }
        self.layout
            .pack_matrices([&matrices[0], &matrices[1], &matrices[2]])
            .map_err(|error| {
                pack_error(format!(
                    "layer {layer} expert {expert}: {}",
                    error.describe()
                ))
            })
    }
}

impl RecordSource for PackRecordSource<'_> {
    fn record(&self, layer: usize, expert: u16, out: &mut [u8]) -> Result<(), EngError> {
        let record = self.read_record(layer, usize::from(expert))?;
        if out.len() != record.len() {
            return Err(pack_error(format!(
                "a record is {} bytes, the buffer has {}",
                record.len(),
                out.len()
            )));
        }
        out.copy_from_slice(&record);
        Ok(())
    }
}

/// The record carries no bias (`Exl3Bank` is bias-free too), and `pack_matrices` would drop it.
fn refuse_bias(
    matrix: &Matrix,
    layer: usize,
    expert: usize,
    part: ExpertPart,
) -> Result<(), EngError> {
    if matrix.bias().is_some() {
        return Err(pack_error(format!(
            "layer {layer} expert {expert} {part:?} has a bias, which a bank record does not carry"
        )));
    }
    Ok(())
}

/// One matrix of the pack by its tensor-name prefix: its nine components, read whole.
fn load_matrix(checkpoint: &Checkpoint, prefix: &str) -> Result<Matrix, EngError> {
    let names = COMPONENT_SUFFIXES.map(|suffix| format!("{prefix}.{suffix}"));
    let keys = names.each_ref().map(String::as_str);
    let selected = checkpoint.source.read_selected(&keys, MATRIX_READ_BUDGET)?;
    Matrix::load(&selected, prefix, &checkpoint.config)
}
