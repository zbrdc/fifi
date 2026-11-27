//! The tensors of a GGUF file on the device, over the file's pages with no copy.

use std::path::Path;

use super::{DeviceBuffer, DeviceRange, MetalContext};
use crate::EngError;
use crate::family::mapped_file::MappedFile;
use crate::weights::GgufHeader;

/// The tensors of one mapped GGUF file. The file's whole pages are one device buffer, and a
/// tensor's bytes are a range of it. A tensor that the last partial page of the file cuts is a
/// copy. Nothing here keeps the device from writing into the pages: a kernel binds them as
/// input only.
// shortcut: one buffer for the whole file, so a file longer than the device's longest buffer
// is refused; cut the file into page-aligned pieces when a model file exceeds the limit.
pub(crate) struct WeightMap {
    /// The file's whole pages. Declared before the file: the fields drop in order, and the
    /// buffer must be gone before its pages are unmapped.
    pages: DeviceBuffer,
    /// The tensors the last partial page cuts: the offset of each in the file and its copy.
    cut: Vec<(u64, DeviceBuffer)>,
    header: GgufHeader,
    file: MappedFile,
}

impl WeightMap {
    /// Maps `path` and gives its pages to the device. `max_header_bytes` bounds the header.
    pub(crate) fn open(
        context: &MetalContext,
        path: &Path,
        max_header_bytes: u64,
    ) -> Result<Self, EngError> {
        crate::storage_guard::require_local(path, "model weights")?;
        let len = std::fs::metadata(path)
            .map_err(|error| EngError::Checkpoint(format!("stat {}: {error}", path.display())))?
            .len();
        let file = MappedFile::open(path, len)
            .map_err(|error| EngError::Checkpoint(format!("map {}: {error}", path.display())))?;
        let bytes: &[u8] = file.as_ref();
        let header = GgufHeader::from_bytes(bytes, len, max_header_bytes)?;
        let longest = context.max_buffer_length()?;
        if len > longest as u64 {
            return Err(EngError::BadRequest(format!(
                "the file is {len} bytes and the device's longest buffer is {longest}"
            )));
        }
        let pages = context.wrap_mapped(bytes)?;
        let mut cut = Vec::new();
        for tensor in header.tensors() {
            let range = tensor.abs_range();
            if range.end > pages.bytes() as u64 {
                let copy = context.upload(&bytes[range.start as usize..range.end as usize])?;
                cut.push((range.start, copy));
            }
        }
        Ok(Self {
            pages,
            cut,
            header,
            file,
        })
    }

    /// The parsed header: the tensors with their types and dimensions.
    pub(crate) fn header(&self) -> &GgufHeader {
        &self.header
    }

    /// The storage of tensor `name` in the file.
    pub(crate) fn range(&self, name: &str) -> Result<DeviceRange<'_>, EngError> {
        let tensor = self
            .header
            .tensor(name)
            .ok_or_else(|| EngError::BadRequest(format!("no tensor {name} in the mapped file")))?;
        let start = tensor.abs_offset();
        // The ranges lie inside the file, whose length fits a usize.
        let bytes = tensor.byte_len() as usize;
        if let Some((_, copy)) = self.cut.iter().find(|(offset, _)| *offset == start) {
            return copy.range(0, bytes);
        }
        self.pages.range(start as usize, bytes)
    }
}

#[cfg(test)]
mod tests {
    use std::io::Write as _;

    use super::WeightMap;
    use crate::EngError;
    use crate::kernels::metal::testkit::Rig;
    use crate::weights::GgufType;

    /// A GGUF file of F32 tensors and no key: 123 bytes of header, data from 128. Every size is
    /// a multiple of the 32-byte alignment, as the parser requires.
    fn gguf_file(tensors: &[(&str, usize)]) -> Vec<u8> {
        let mut out = vec![0x47, 0x47, 0x55, 0x46];
        out.extend_from_slice(&3u32.to_le_bytes());
        out.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
        out.extend_from_slice(&0u64.to_le_bytes());
        let mut offset = 0u64;
        for (name, floats) in tensors {
            out.extend_from_slice(&(name.len() as u64).to_le_bytes());
            out.extend_from_slice(name.as_bytes());
            out.extend_from_slice(&1u32.to_le_bytes());
            out.extend_from_slice(&(*floats as u64).to_le_bytes());
            out.extend_from_slice(&GgufType::F32.id().to_le_bytes());
            out.extend_from_slice(&offset.to_le_bytes());
            offset += 4 * *floats as u64;
        }
        out.resize(128, 0);
        for (_, floats) in tensors {
            for i in 0..*floats {
                out.extend_from_slice(&(i as f32).to_le_bytes());
            }
        }
        out
    }

    fn written(bytes: &[u8]) -> tempfile::NamedTempFile {
        let mut file = tempfile::Builder::new()
            .prefix("metal-map-")
            .tempfile()
            .expect("a temp file");
        file.write_all(bytes).expect("the write");
        file.flush().expect("the flush");
        file
    }

    #[test]
    fn a_tensor_is_a_range_of_the_file_pages_until_the_last_page_cuts_it() {
        // 128 + 8,192 + 8,224 + 96 bytes: the last page is partial at 4 KiB and at 16 KiB.
        let bytes = gguf_file(&[("a", 2048), ("b", 2056), ("c", 24)]);
        assert_eq!(bytes.len(), 16640);
        let file = written(&bytes);
        let rig = Rig::new();
        let map = WeightMap::open(&rig.ctx, file.path(), 1 << 20).expect("the map");
        assert_eq!(map.header().tensors().len(), 3);
        assert_eq!(map.pages.bytes(), 16384);
        let base = map.file.as_ref().as_ptr() as usize;

        // `a` ends in the whole pages: its storage is the file's own, and no host write.
        let a = map.range("a").expect("a");
        assert_eq!(a.address(), base + 128);
        assert_eq!(a.read().expect("the read"), bytes[128..8320]);
        assert!(matches!(
            a.write(&bytes[128..8320]),
            Err(EngError::BadRequest(_))
        ));

        // `b` and `c` end in the last partial page: copies that hold the file's bytes.
        for (name, range) in [("b", 8320..16544usize), ("c", 16544..16640)] {
            let found = map.range(name).expect("a cut tensor");
            assert_ne!(found.address(), base + range.start);
            assert_eq!(found.read().expect("the read"), bytes[range]);
        }
        assert_eq!(map.cut.len(), 2);
        assert!(matches!(map.range("d"), Err(EngError::BadRequest(_))));
        assert_eq!(rig.ctx.live_buffers(), 3);
    }

    #[test]
    fn a_file_without_a_whole_page_or_without_a_header_is_refused_and_the_backend_lives() {
        let rig = Rig::new();
        // 128 + 1,024 bytes.
        let short = written(&gguf_file(&[("a", 256)]));
        assert!(matches!(
            WeightMap::open(&rig.ctx, short.path(), 1 << 20),
            Err(EngError::BadRequest(_))
        ));
        let plain = written(&vec![0u8; 16640]);
        assert!(WeightMap::open(&rig.ctx, plain.path(), 1 << 20).is_err());

        // Each refusal made no buffer and did not fail the backend: an upload still works.
        assert_eq!(rig.ctx.live_buffers(), 0);
        let kept = rig
            .ctx
            .upload(&[7u8; 16])
            .expect("an upload after the refusals");
        assert_eq!(rig.ctx.live_buffers(), 1);
        drop(kept);
        assert_eq!(rig.ctx.live_buffers(), 0);
    }
}
