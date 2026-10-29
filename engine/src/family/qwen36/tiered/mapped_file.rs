#![allow(unsafe_code)]
//! The one scoped module that maps a model file read-only.
//!
//! WHAT: [`MappedFile`] opens a file read-only, refuses a length the caller did
//! not expect (the GGUF header's end of data), and maps the whole file
//! PROT_READ, so the CPU expert tier reads experts straight from the OS page
//! cache with plain faults and DEFAULT advice (a measurement showed that any
//! madvise hint is harmful; this module never calls madvise). WHY: `memmap2`
//! 0.9.11 marks the file-backed map constructor `unsafe` because another writer
//! can invalidate the mapped references, and the crate's unsafe-code policy
//! forbids a safe constructor that silently passes that obligation on; this
//! file is the ONE place carrying this scoped `allow(unsafe_code)`, it opens
//! and maps read-only only, recording length and mtime at open and re-checking
//! them around the map and on demand, and it holds exactly ONE `unsafe` block -
//! the `Mmap` map call - whose only view out is `AsRef<[u8]>` (a safe
//! [`MappedRegion`](super::cpu_pool::MappedRegion)), so no pointer leaves.
//! Residual risk: truncation under a live mapping still faults (SIGBUS) when a
//! page past the new end is touched, and an in-place overwrite by another
//! process changes the bytes behind the `&[u8]` without any fault; no
//! Rust-level check closes that window, only narrows it, and model files are
//! immutable by policy. Opened by the tiered load.

use std::fmt;
use std::fs::File;
use std::path::Path;
use std::time::SystemTime;

use memmap2::{Mmap, MmapOptions};

/// Why a mapping could not be made or trusted. Every variant leaves no mapping.
#[derive(Debug, PartialEq)]
pub enum MapError {
    /// An open, stat or map failure; the message carries the local cause.
    Io(String),
    /// The file has zero bytes: an empty mapping would hide a wrong-path bug.
    Empty,
    /// The file length is not the `expected_len` the caller passed (the GGUF
    /// header's end of data); slicing expert ranges into it runs out of bounds.
    LengthMismatch {
        /// Length the caller derived from the GGUF header, in bytes.
        expected: u64,
        /// Length the file actually has, in bytes.
        actual: u64,
    },
    /// The length or mtime of the open handle differs from what was recorded
    /// at open, so the bytes cannot be trusted. `what` is `"length"` or `"mtime"`.
    Changed {
        /// What moved: `"length"` or `"mtime"`.
        what: &'static str,
    },
    /// The length does not fit a `usize` on this host, so no slice can address it.
    TooLarge,
}

impl fmt::Display for MapError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MapError::Io(message) => write!(f, "mapping I/O failure: {message}"),
            MapError::Empty => write!(f, "refusing to map an empty file"),
            MapError::LengthMismatch { expected, actual } => {
                write!(f, "file is {actual} bytes, expected {expected}")
            }
            MapError::Changed { what } => write!(f, "the open file's {what} changed"),
            MapError::TooLarge => write!(f, "file length does not fit a usize here"),
        }
    }
}

impl std::error::Error for MapError {}

/// A read-only mapping of a whole file, kept with the open handle it was made
/// from so the recorded length and mtime can be re-checked against the same
/// inode at any time (a rename-over of the path cannot matter: the handle and
/// the mapping belong to the inode; the handle is kept exactly so
/// `check_unchanged` fstats the file that was mapped). `AsRef<[u8]>` is the
/// only view out, which makes this a region for the CPU pool.
pub struct MappedFile {
    file: File,
    map: Mmap,
    len: u64,
    mtime: SystemTime,
}

impl MappedFile {
    /// Opens `path` read-only, stats the open handle, refuses an empty file,
    /// a length != `expected_len` and a length that cannot fit a `usize`
    /// (`TooLarge`), maps the whole file read-only, then stats again: a
    /// length or mtime that moved across the map is `Changed`.
    pub fn open(path: &Path, expected_len: u64) -> Result<MappedFile, MapError> {
        let file = File::open(path).map_err(io_err)?;
        let (len, mtime) = stat(&file)?;
        if len == 0 {
            return Err(MapError::Empty);
        }
        if len != expected_len {
            return Err(MapError::LengthMismatch {
                expected: expected_len,
                actual: len,
            });
        }
        if usize::try_from(len).is_err() {
            return Err(MapError::TooLarge);
        }
        // SAFETY: the file was opened read-only and this maps it PROT_READ (no
        // writable, no MAP_SHARED-writable mapping); the expected length was
        // just confirmed on this handle and the second stat below confirms
        // length and mtime did not move across the map. Residual risk: a
        // truncation under the live map still SIGBUSes on touch and nothing in
        // production narrows that window (`check_unchanged` is a test-side
        // check today); an in-place overwrite by another process raises no
        // fault at all and changes the bytes the CPU tier reads. Model files
        // are immutable by policy, the operator's contract (stated on
        // `TieredOptions`), and that is the argument this block rests on.
        let map = unsafe { MmapOptions::new().map(&file) }.map_err(io_err)?;
        let (after_len, after_mtime) = stat(&file)?;
        if after_len != len {
            return Err(MapError::Changed { what: "length" });
        }
        if after_mtime != mtime {
            return Err(MapError::Changed { what: "mtime" });
        }
        Ok(MappedFile {
            file,
            map,
            len,
            mtime,
        })
    }

    /// File length in bytes recorded at open; also the mapped length.
    pub fn len(&self) -> u64 {
        self.len
    }

    /// Always false (`open` refuses an empty file); here for container
    /// conventions.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Re-stats the OPEN handle (fstat; a rename-over of the path cannot
    /// matter because the mapping belongs to the inode): `Changed { what:
    /// "length" | "mtime" }` when the file differs from the recorded values,
    /// else `Ok(())` - the narrowest cheap evidence the bytes did not move. No production path calls
    /// it yet (the tests do); it narrows a window, it does not close it.
    pub fn check_unchanged(&self) -> Result<(), MapError> {
        let (len, mtime) = stat(&self.file)?;
        if len != self.len {
            return Err(MapError::Changed { what: "length" });
        }
        if mtime != self.mtime {
            return Err(MapError::Changed { what: "mtime" });
        }
        Ok(())
    }
}

/// Wraps an I/O failure's message; every failure here leaves no mapping.
fn io_err(error: std::io::Error) -> MapError {
    MapError::Io(error.to_string())
}

/// fstats the open handle: its length and mtime, the identity pair.
fn stat(file: &File) -> Result<(u64, SystemTime), MapError> {
    let metadata = file.metadata().map_err(io_err)?;
    let mtime = metadata.modified().map_err(io_err)?;
    Ok((metadata.len(), mtime))
}

impl AsRef<[u8]> for MappedFile {
    /// The whole mapping: the only view out of the module, read-only bytes.
    fn as_ref(&self) -> &[u8] {
        self.map.as_ref()
    }
}

impl fmt::Debug for MappedFile {
    /// Prints the length and recorded mtime only, never bytes (a model file
    /// is gigabytes).
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MappedFile")
            .field("len", &self.len)
            .field("mtime", &self.mtime)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::{MapError, MappedFile};
    use crate::family::qwen36::tiered::cpu_pool::MappedRegion;
    use std::fs;
    use std::io::Write as _;
    use std::sync::Arc;
    use tempfile::NamedTempFile;

    /// The tests' deterministic generator: an in-file LCG with a fixed seed.
    struct Lcg(u64);

    impl Lcg {
        fn next_u64(&mut self) -> u64 {
            let mixed = self.0.wrapping_mul(6364136223846793005);
            self.0 = mixed.wrapping_add(1442695040888963407);
            self.0 ^ (self.0 >> 21)
        }

        fn fill(&mut self, bytes: &mut [u8]) {
            for chunk in bytes.chunks_mut(8) {
                let value = self.next_u64().to_le_bytes();
                chunk.copy_from_slice(&value[..chunk.len()]);
            }
        }
    }

    /// Writes a fresh `NamedTempFile` of `len` bytes shaped like a GGUF: the
    /// magic then a fixed-seed LCG pattern; the handle is returned so the
    /// file is removed on every path.
    fn gguf_shaped(len: u64) -> NamedTempFile {
        let mut file = NamedTempFile::new().expect("temp file in TMPDIR");
        let mut bytes = vec![0u8; usize::try_from(len).expect("test length fits usize")];
        bytes[..4].copy_from_slice(b"GGUF");
        Lcg(0x5eed_1234_5678_9abc).fill(&mut bytes[4..]);
        file.write_all(&bytes).expect("fixture write");
        file.flush().expect("fixture flush");
        file
    }

    #[test]
    fn maps_a_gguf_shaped_file_and_matches_a_plain_read() {
        let len = (1 << 20) + 13;
        let file = gguf_shaped(len);
        let mapped = MappedFile::open(file.path(), len).expect("matching length maps");
        assert_eq!(mapped.len(), len);
        assert!(!mapped.is_empty());
        assert_eq!(&mapped.as_ref()[..4], b"GGUF");
        let plain = fs::read(file.path()).expect("plain read");
        assert_eq!(mapped.as_ref(), plain.as_slice());
        // The pool story: the value serves as a shared region through the
        // blanket `MappedRegion` impl for every `AsRef<[u8]>`.
        let region: Arc<dyn MappedRegion> = Arc::new(mapped);
        assert_eq!(region.bytes(), plain.as_slice());
    }

    #[test]
    fn open_refuses_a_length_mismatch_an_empty_file_and_a_missing_file() {
        let file = gguf_shaped(4096);
        assert!(matches!(
            MappedFile::open(file.path(), 4097),
            Err(MapError::LengthMismatch {
                expected: 4097,
                actual: 4096
            })
        ));
        let empty = NamedTempFile::new().expect("temp file in TMPDIR");
        match MappedFile::open(empty.path(), 0) {
            Err(MapError::Empty) => {}
            other => panic!("expected an empty refusal, got {other:?}"),
        }
        let missing = NamedTempFile::new().expect("temp file in TMPDIR");
        let gone = missing.path().to_path_buf();
        drop(missing); // NamedTempFile removes it, so the path is gone.
        match MappedFile::open(&gone, 1) {
            Err(MapError::Io(_)) => {}
            other => panic!("expected an I/O failure, got {other:?}"),
        }
    }

    #[test]
    fn check_unchanged_passes_on_an_untouched_file_and_names_a_truncation() {
        let len: u64 = 8192 + 96;
        let file = gguf_shaped(len);
        let mapped = MappedFile::open(file.path(), len).expect("matching length maps");
        assert_eq!(mapped.check_unchanged(), Ok(()));
        // Truncate through a SECOND handle; this test never reads the mapped
        // bytes again (a page past the new end is the documented SIGBUS).
        let second = fs::OpenOptions::new()
            .write(true)
            .open(file.path())
            .expect("second writable handle");
        second.set_len(len - 4096).expect("truncate");
        match mapped.check_unchanged() {
            Err(MapError::Changed { what: "length" }) => {}
            other => panic!("expected a length change, got {other:?}"),
        }
        drop(mapped);
        drop(second);
    }

    #[test]
    fn a_renamed_over_path_does_not_disturb_an_open_mapping() {
        let len: u64 = 8192 + 7;
        let file = gguf_shaped(len);
        let expected = fs::read(file.path()).expect("plain read of the original");
        let mapped = MappedFile::open(file.path(), len).expect("matching length maps");
        let mut other = NamedTempFile::new().expect("temp file in TMPDIR");
        other.write_all(&[7u8; 128]).expect("fixture write");
        other.flush().expect("fixture flush");
        fs::rename(other.path(), file.path()).expect("rename over the mapped path");
        // The open handle pins the ORIGINAL inode, so nothing checked can move.
        assert_eq!(mapped.check_unchanged(), Ok(()));
        assert_eq!(mapped.as_ref(), expected.as_slice());
    }

    #[test]
    fn the_module_has_exactly_one_unsafe_block_and_the_other_tiered_modules_have_none() {
        let this_file = include_str!("mapped_file.rs");
        let production = this_file.split("mod tests").next().unwrap_or_default();
        // Assembled at run time so this source omits the literal it counts.
        let needle: String = ["uns", "afe", " {"].concat();
        assert_eq!(production.matches(&needle).count(), 1);
        for source in [
            include_str!("mod.rs"),
            include_str!("policy.rs"),
            include_str!("residency.rs"),
            include_str!("report.rs"),
            include_str!("cpu_pool.rs"),
            include_str!("budget.rs"),
            include_str!("guess.rs"),
            include_str!("locator.rs"),
        ] {
            assert!(!source.contains(&needle));
        }
    }
}
