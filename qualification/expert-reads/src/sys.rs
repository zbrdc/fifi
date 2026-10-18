//! The only unsafe code of the command: a read-only mapping that is never dereferenced, `mincore`
//! on it, `fcntl(F_NOCACHE)`, and the page size (S3 plan section 3.6: "its `unsafe` is limited to
//! `libc::mincore` on a read-only mapping").

use memmap2::Mmap;
use std::fs::File;

/// The virtual-memory page size.
pub fn page_size() -> Result<u64, String> {
    // SAFETY: `sysconf` reads a system constant and has no memory effects.
    let size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    u64::try_from(size)
        .ok()
        .filter(|&s| s > 0)
        .ok_or_else(|| "the page size is unavailable".to_string())
}

/// A read-only shared mapping of a whole shard. The command never reads through it: `mincore`
/// only reports which of its pages are resident, which for a shared file mapping includes pages
/// in the file cache that this process never touched.
pub struct Residency {
    map: Mmap,
    page: u64,
}

impl Residency {
    pub fn new(file: &File, page: u64) -> Result<Self, String> {
        // SAFETY: the mapping is read-only and never dereferenced, so a concurrent change of the
        // file cannot change memory that this process reads.
        let map = unsafe { Mmap::map(file) }.map_err(|e| format!("mmap: {e}"))?;
        Ok(Self { map, page })
    }

    /// Resident bytes of `offset..offset + len` of the shard.
    pub fn resident_bytes(&self, offset: u64, len: u64) -> Result<u64, String> {
        let end = offset
            .checked_add(len)
            .filter(|&end| end <= self.map.len() as u64)
            .ok_or("range passes the end of the shard")?;
        let first = offset / self.page * self.page;
        let pages = (end - first).div_ceil(self.page);
        let mut vec = vec![0u8; usize::try_from(pages).map_err(|_| "too many pages")?];
        let map_len = usize::try_from(pages * self.page)
            .map_err(|_| "range too long")?
            .min(self.map.len() - usize::try_from(first).map_err(|_| "offset too large")?);
        // SAFETY: `first` is a page multiple inside the mapping, `map_len` stays inside it, and
        // `vec` has one byte per page of that range.
        let status = unsafe {
            let base = self
                .map
                .as_ptr()
                .add(usize::try_from(first).map_err(|_| "offset")?);
            libc::mincore(base as *mut libc::c_void, map_len, vec.as_mut_ptr().cast())
        };
        if status != 0 {
            return Err(format!("mincore: {}", std::io::Error::last_os_error()));
        }
        Ok(crate::resident_bytes(&vec, self.page, first, offset, len))
    }
}

/// Opens `path` with the file cache turned off for its reads (`F_NOCACHE`, macOS only).
#[cfg(target_os = "macos")]
pub fn open_nocache(path: &std::path::Path) -> Result<Option<File>, String> {
    use std::os::fd::AsRawFd;
    let file = File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    // SAFETY: `fcntl(F_NOCACHE, 1)` sets a flag on a descriptor that `file` owns.
    let status = unsafe { libc::fcntl(file.as_raw_fd(), libc::F_NOCACHE, 1) };
    if status == -1 {
        return Err(format!("F_NOCACHE: {}", std::io::Error::last_os_error()));
    }
    Ok(Some(file))
}

/// `F_NOCACHE` does not exist on this system.
#[cfg(not(target_os = "macos"))]
pub fn open_nocache(_path: &std::path::Path) -> Result<Option<File>, String> {
    Ok(None)
}
