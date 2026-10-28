#![allow(unsafe_code)]
//! Drops file ranges from the OS page cache once their bytes live somewhere else.
//!
//! WHAT: [`drop_file_range`] asks the kernel to evict the clean, unmapped cache pages of one byte
//! range of an open file (`posix_fadvise(POSIX_FADV_DONTNEED)`). WHY: the Nemotron hybrid keeps
//! every routed expert in exactly one place. An expert uploaded to the GPU must not also sit in the
//! host page cache, which the CPU-served experts need for themselves (the "exclusive of the page
//! cache" rule of the Qwen3.6 tier). A read of the file leaves its pages cached; this evicts them.
//!
//! Scope: the one `unsafe` block is the foreign call; the declaration below is the whole FFI
//! surface and std already links libc, so no crate is added. The call moves no data and cannot
//! invalidate a reference: it is advice about cache residency, and a page that is mapped or
//! dirty stays in place. The source-scan test pins one `unsafe` block in this file.

use std::{fs::File, io};

#[cfg(target_os = "linux")]
mod linux {
    use std::ffi::c_int;

    /// `POSIX_FADV_DONTNEED` on Linux.
    pub(super) const FADV_DONTNEED: c_int = 4;

    unsafe extern "C" {
        pub(super) fn posix_fadvise(fd: c_int, offset: i64, len: i64, advice: c_int) -> c_int;
    }
}

/// Evicts the unmapped, clean cache pages of `offset..offset + length` of `file`.
///
/// A zero `length` is a no-op (POSIX would read it as "to the end of the file"). Returns the OS
/// error when the kernel refuses the advice. A no-op on targets without `posix_fadvise`.
pub fn drop_file_range(file: &File, offset: u64, length: u64) -> io::Result<()> {
    if length == 0 {
        return Ok(());
    }
    let (Ok(offset), Ok(length)) = (i64::try_from(offset), i64::try_from(length)) else {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "file range does not fit off_t",
        ));
    };
    drop_range(file, offset, length)
}

#[cfg(target_os = "linux")]
fn drop_range(file: &File, offset: i64, length: i64) -> io::Result<()> {
    use std::os::fd::AsRawFd;
    let fd = file.as_raw_fd();
    // SAFETY: valid fd (`file` is borrowed), range checked into off_t above, no pointers, advice only.
    let status = unsafe { linux::posix_fadvise(fd, offset, length, linux::FADV_DONTNEED) };
    // posix_fadvise returns the error number itself, not -1 with errno set.
    if status == 0 {
        Ok(())
    } else {
        Err(io::Error::from_raw_os_error(status))
    }
}

#[cfg(not(target_os = "linux"))]
fn drop_range(_file: &File, _offset: i64, _length: i64) -> io::Result<()> {
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::io::{Read, Seek, SeekFrom, Write};

    use super::drop_file_range;

    #[test]
    fn dropping_a_range_leaves_the_file_readable_and_identical() {
        let mut file = tempfile::tempfile().expect("temporary file");
        let bytes: Vec<u8> = (0..(1 << 20)).map(|i| (i * 31 % 251) as u8).collect();
        file.write_all(&bytes).expect("write");
        file.sync_all().expect("sync, so the pages are clean");
        drop_file_range(&file, 4096, 512 * 1024).expect("advice on a clean range");
        drop_file_range(&file, 0, 0).expect("empty range is a no-op");
        drop_file_range(&file, 900_000, 10 << 20).expect("a range past the end is accepted");
        let mut again = Vec::new();
        file.seek(SeekFrom::Start(0)).expect("seek");
        file.read_to_end(&mut again).expect("read back");
        assert_eq!(again, bytes);
    }

    #[test]
    fn a_range_that_does_not_fit_off_t_is_refused() {
        let file = tempfile::tempfile().expect("temporary file");
        let error = drop_file_range(&file, u64::MAX, 1).expect_err("offset past off_t");
        assert_eq!(error.kind(), std::io::ErrorKind::InvalidInput);
        let error = drop_file_range(&file, 0, u64::MAX).expect_err("length past off_t");
        assert_eq!(error.kind(), std::io::ErrorKind::InvalidInput);
    }

    #[test]
    fn the_module_has_exactly_one_unsafe_block() {
        let production = include_str!("page_cache.rs")
            .split("mod tests")
            .next()
            .unwrap_or_default();
        // Assembled at run time so this source omits the literal it counts.
        let needle: String = ["uns", "afe", " {"].concat();
        assert_eq!(production.matches(&needle).count(), 1);
        let previous = production
            .lines()
            .take_while(|line| !line.contains(&needle))
            .filter(|line| !line.trim().is_empty())
            .last()
            .unwrap_or_default();
        assert!(previous.contains("SAFETY"), "{previous}");
    }
}
