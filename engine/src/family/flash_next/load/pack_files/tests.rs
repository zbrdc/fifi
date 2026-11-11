//! The guard names every file of a pack, and the n-gram table's layout comes from the config.

use super::super::tests::CONFIG;
#[cfg(target_os = "linux")]
use super::FlashPackFiles;
use super::{Dims, NGRAM_FILE, SIDE_FILES, guard_side_files, ngram_key, ngram_layout};
use crate::weights::exl3::ngram::NgramLayout;

#[test]
fn the_ngram_key_and_layout_follow_the_real_config() {
    let dims = Dims::read(CONFIG).unwrap();
    assert_eq!(
        ngram_key(dims.ple_layer),
        "model.language_model.layers.1.ple.ple_embedding.ngram_embedding"
    );
    assert_eq!(
        ngram_layout(&dims),
        NgramLayout {
            heads: 16,
            row_dim: 160,
            ngram: 3
        }
    );
}

#[test]
fn a_missing_optional_file_is_not_a_guard_error() {
    let dir = tempfile::tempdir().unwrap();
    let guard = crate::storage_guard::LocalGuard::new().unwrap();
    guard_side_files(dir.path(), &guard).unwrap();
}

// Each case links exactly one named file of a local directory to a file on a network mount.
// Without the guard on that name the call would succeed.
#[cfg(target_os = "linux")]
#[test]
fn a_link_to_a_network_mount_is_refused_for_every_file_outside_the_shards() {
    for name in SIDE_FILES.into_iter().chain([NGRAM_FILE]) {
        let share = tempfile::tempdir().unwrap();
        let local = tempfile::tempdir().unwrap();
        let target = share.path().join(name);
        std::fs::write(&target, b"x").unwrap();
        std::os::unix::fs::symlink(&target, local.path().join(name)).unwrap();
        let refused = crate::storage_guard::seam::with_nfs_at(share.path(), || {
            let guard = crate::storage_guard::LocalGuard::new().unwrap();
            guard_side_files(local.path(), &guard)
                .err()
                .unwrap_or_else(|| panic!("{name} was not judged"))
        });
        assert!(
            refused.describe().contains("must be on local storage"),
            "{name}: {}",
            refused.describe()
        );
    }
}

// Fails if `FlashPackFiles::open` stops calling `guard_side_files`: a local directory holding only a
// link to the named file on a network mount. Without the guard the error is the missing or unparsable
// config, not the refusal.
#[cfg(target_os = "linux")]
#[test]
fn open_refuses_a_linked_side_file_before_it_reads_the_pack() {
    for name in SIDE_FILES.into_iter().chain([NGRAM_FILE]) {
        let share = tempfile::tempdir().unwrap();
        let local = tempfile::tempdir().unwrap();
        let target = share.path().join(name);
        std::fs::write(&target, b"x").unwrap();
        std::os::unix::fs::symlink(&target, local.path().join(name)).unwrap();
        let refused = crate::storage_guard::seam::with_nfs_at(share.path(), || {
            FlashPackFiles::open(local.path()).err().expect("refused")
        });
        assert!(
            refused.describe().contains("must be on local storage"),
            "{name}: {}",
            refused.describe()
        );
    }
}

// Fails if an entry leaves the list (the per-file test above iterates the list itself).
#[test]
fn the_guard_list_is_exactly_the_files_a_pack_holds_outside_its_shards() {
    assert_eq!(
        SIDE_FILES,
        [
            "config.json",
            "model.safetensors.index.json",
            "mtp_hyper_connection_mixer_patch.safetensors"
        ]
    );
    assert_eq!(NGRAM_FILE, "ngram_embedding.safetensors");
}
