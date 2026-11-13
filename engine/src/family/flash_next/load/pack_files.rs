//! Every file of a Flash-Next EXL3 pack, opened through one local-storage guard.
//!
//! The pack directory holds the weight shards (named by the index), `config.json`, the index, the
//! n-gram table `ngram_embedding.safetensors` (outside the index) and, in the stock pack, the MTP
//! hyper-connection patch. [`FlashPackFiles::open`] reads the mount table once and judges all of
//! them with it before it opens anything else: the directory and each shard through
//! [`Checkpoint::open_with`], the two small metadata files and the optional MTP patch by name
//! (a file can be a symlink into a network mount although its directory is local), and the n-gram
//! table through [`NgramFile::open`]. Then it walks the headers against `config.json` and checks
//! the n-gram header against the model's layout.

// Staged: the Flash-Next CUDA loader is the first caller.
#![cfg_attr(not(test), allow(dead_code))]

use std::path::Path;

use super::{Dims, exl3_pack::inspect_flash_pack};
use crate::{
    EngError,
    family::exl3_pack::PackReport,
    family::exl3_pack::Root,
    storage_guard::LocalGuard,
    weights::exl3::{Checkpoint, ngram::NgramLayout, ngram_file::NgramFile},
};

/// The n-gram table, 30.40 GiB at 3.05 bpw.
pub(in crate::family) const NGRAM_FILE: &str = "ngram_embedding.safetensors";
/// The MTP hyper-connection mixer of the stock pack. The step does not read it yet.
pub(in crate::family) const MTP_PATCH_FILE: &str = "mtp_hyper_connection_mixer_patch.safetensors";
/// Files outside the weight index that the guard names besides the n-gram table.
pub(in crate::family) const SIDE_FILES: [&str; 3] = [
    "config.json",
    "model.safetensors.index.json",
    MTP_PATCH_FILE,
];

/// A pack opened through one guard.
pub(in crate::family) struct FlashPackFiles {
    pub(in crate::family) checkpoint: Checkpoint,
    pub(in crate::family) ngram: NgramFile,
    pub(in crate::family) report: PackReport,
}

/// The tensor-name prefix of the n-gram table: the layer with the PLE block.
pub(in crate::family) fn ngram_key(ple_layer: usize) -> String {
    Root::Layers.name(ple_layer, "ple.ple_embedding.ngram_embedding")
}

/// What the model requires of the n-gram table.
fn ngram_layout(dims: &Dims) -> NgramLayout {
    NgramLayout {
        heads: dims.ple_heads as usize,
        row_dim: dims.ple_head_dim as usize,
        ngram: dims.ngram as usize,
    }
}

/// Judges the named files that are not shards, and the n-gram table, with `guard`. A file that
/// does not exist is judged by its nearest existing ancestor, so a missing optional file is
/// not an error here.
pub(in crate::family) fn guard_side_files(dir: &Path, guard: &LocalGuard) -> Result<(), EngError> {
    for name in SIDE_FILES.into_iter().chain([NGRAM_FILE]) {
        guard.check(&dir.join(name), "model weights")?;
    }
    Ok(())
}

impl FlashPackFiles {
    pub(in crate::family) fn open(dir: &Path) -> Result<Self, EngError> {
        let guard = LocalGuard::new()?;
        guard.check(dir, "model weights")?;
        guard_side_files(dir, &guard)?;
        let checkpoint = Checkpoint::open_with(dir, &guard)?;
        let dims = Dims::read(&checkpoint.model_config)?;
        let ngram = NgramFile::open(
            &dir.join(NGRAM_FILE),
            &guard,
            &ngram_key(dims.ple_layer),
            ngram_layout(&dims),
        )?;
        let report = inspect_flash_pack(&checkpoint.source, &checkpoint.model_config)?;
        Ok(Self {
            checkpoint,
            ngram,
            report,
        })
    }
}

#[cfg(test)]
mod tests;
