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
    use std::time::Instant;

    use half::bf16;

    use super::WeightMap;
    use crate::EngError;
    use crate::kernels::metal::testkit::{Rig, f32_bytes, floats};
    use crate::kernels::metal::{
        DenseDtype, DenseGemvCall, DenseGemvShape, DeviceBuffer, DeviceRange, dense_gemv_bytes,
    };
    use crate::weights::GgufType;

    fn elements(dims: &[u64]) -> usize {
        dims.iter().product::<u64>() as usize
    }

    fn payload_bytes(ty: GgufType, dims: &[u64]) -> u64 {
        match ty {
            GgufType::F32 => 4 * elements(dims) as u64,
            GgufType::Bf16 => 2 * elements(dims) as u64,
            _ => unreachable!("the types the tests write"),
        }
    }

    /// A finite BF16 weight from its index: -64 to 64 in steps of 1/32.
    fn weight(i: usize) -> bf16 {
        bf16::from_f32(((i % 4096) as f32 - 2048.0) * 0.03125)
    }

    /// A GGUF file of tensors and no key: 24 bytes, then `8 + name + 4 + 8 * dims + 4 + 8`
    /// bytes a tensor. The data start at the next multiple of 32 and every payload is padded
    /// to one, as the parser requires. An F32 payload counts up from 0.
    fn gguf_file(tensors: &[(&str, GgufType, &[u64])]) -> Vec<u8> {
        let mut out = vec![0x47, 0x47, 0x55, 0x46];
        out.extend_from_slice(&3u32.to_le_bytes());
        out.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
        out.extend_from_slice(&0u64.to_le_bytes());
        let mut offset = 0u64;
        for (name, ty, dims) in tensors {
            out.extend_from_slice(&(name.len() as u64).to_le_bytes());
            out.extend_from_slice(name.as_bytes());
            out.extend_from_slice(&(dims.len() as u32).to_le_bytes());
            for dim in *dims {
                out.extend_from_slice(&dim.to_le_bytes());
            }
            out.extend_from_slice(&ty.id().to_le_bytes());
            out.extend_from_slice(&offset.to_le_bytes());
            offset += payload_bytes(*ty, dims).div_ceil(32) * 32;
        }
        out.resize(out.len().div_ceil(32) * 32, 0);
        out.reserve(offset as usize);
        for (_, ty, dims) in tensors {
            for i in 0..elements(dims) {
                match ty {
                    GgufType::F32 => out.extend_from_slice(&(i as f32).to_le_bytes()),
                    GgufType::Bf16 => {
                        out.extend_from_slice(&weight(i).to_bits().to_le_bytes());
                    }
                    _ => unreachable!("the types the tests write"),
                }
            }
            out.resize(out.len().div_ceil(32) * 32, 0);
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

    /// A dense matvec of the weight `w` in BF16: the activation and the result give the call
    /// its bytes.
    fn gemv<'b>(
        shape: DenseGemvShape,
        w: DeviceRange<'b>,
        x: &'b DeviceBuffer,
        y: &'b DeviceBuffer,
    ) -> DenseGemvCall<'b> {
        let need = dense_gemv_bytes(shape, DenseDtype::Bf16).expect("the call's bytes");
        DenseGemvCall {
            shape,
            dtype: DenseDtype::Bf16,
            w,
            x: x.range(0, need.x).expect("the activation's range"),
            y: y.range(0, need.y).expect("the result's range"),
        }
    }

    #[test]
    fn a_tensor_is_a_range_of_the_file_pages_until_the_last_page_cuts_it() {
        // 128 + 8,192 + 8,224 + 96 bytes: the last page is partial at 4 KiB and at 16 KiB.
        let bytes = gguf_file(&[
            ("a", GgufType::F32, &[2048]),
            ("b", GgufType::F32, &[2056]),
            ("c", GgufType::F32, &[24]),
        ]);
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
    fn the_dense_matvec_from_the_file_pages_gives_the_bits_an_upload_gives() {
        // `w` is BF16 [1024, 512], 1 MiB, and ends inside the whole pages; `tail` ends in the
        // last partial page. At 4 KiB and at 16 KiB pages the covered pages end at 1,064,960.
        let bytes = gguf_file(&[
            ("w", GgufType::Bf16, &[512, 1024]),
            ("tail", GgufType::F32, &[4096]),
        ]);
        assert_eq!(bytes.len(), 128 + 1_048_576 + 16_384);
        let file = written(&bytes);
        let rig = Rig::new();
        let map = WeightMap::open(&rig.ctx, file.path(), 1 << 20).expect("the map");
        assert_eq!(map.pages.bytes(), 1_064_960);
        assert_eq!(map.cut.len(), 1, "only the tail is a copy");
        let drawn: Vec<f32> = (0..2 * 512)
            .map(|i| (i % 997) as f32 * 0.001 - 0.5)
            .collect();
        let x = rig.ctx.upload(&f32_bytes(&drawn)).expect("the activation");
        let result = || {
            rig.ctx
                .upload(&vec![0u8; 2 * 1024 * 4])
                .expect("a result buffer")
        };
        let (y_upload, y_pages) = (result(), result());
        let uploaded = rig
            .ctx
            .upload(&bytes[128..128 + 1_048_576])
            .expect("the uploaded weight");
        let w_pages = map.range("w").expect("the weight from the pages");
        let w_upload = uploaded
            .range(0, 1_048_576)
            .expect("the uploaded weight's range");
        assert_eq!(w_pages.address(), map.file.as_ref().as_ptr() as usize + 128);
        let shape = DenseGemvShape::packed(2, 1024, 512);

        rig.ctx
            .dense_gemv(gemv(shape, w_upload, &x, &y_upload))
            .expect("the upload's matvec");
        rig.ctx
            .dense_gemv(gemv(shape, w_pages, &x, &y_pages))
            .expect("the pages' matvec");
        let waited = y_upload.read(0, 2 * 1024 * 4).expect("the read");
        assert!(
            floats(&waited).iter().any(|value| *value != 0.0),
            "the results were written"
        );
        assert_eq!(
            y_pages.read(0, 2 * 1024 * 4).expect("the read"),
            waited,
            "one waited launch"
        );

        let mut frame = rig.ctx.frame_of(3);
        for _ in 0..5 {
            frame
                .dense_gemv(gemv(shape, w_pages, &x, &y_pages))
                .expect("a dispatch");
        }
        frame.drain().expect("the drain");
        assert_eq!(
            y_pages.read(0, 2 * 1024 * 4).expect("the read"),
            waited,
            "a frame of five dispatches"
        );
        assert_eq!(
            w_pages.read().expect("the read"),
            bytes[128..128 + 1_048_576],
            "the calls changed the file's pages"
        );
    }

    #[test]
    #[ignore = "prints the dense matvec's time from the file's pages against an upload"]
    fn the_dense_matvec_from_the_file_pages_is_timed() {
        // The vocabulary's projection, BF16 [200,058, 2304]: 921.9 MB on the pages, the tail
        // in the last partial page.
        const ROWS: usize = 200_058;
        const COLS: usize = 2304;
        const BYTES: usize = 2 * ROWS * COLS;
        let bytes = gguf_file(&[
            ("w", GgufType::Bf16, &[COLS as u64, ROWS as u64]),
            ("tail", GgufType::F32, &[4096]),
        ]);
        assert_eq!(bytes.len(), 128 + BYTES + 16_384);
        let file = written(&bytes);
        let rig = Rig::new();
        let map = WeightMap::open(&rig.ctx, file.path(), 1 << 20).expect("the map");
        assert_eq!(map.cut.len(), 1, "only the tail is a copy");
        let drawn: Vec<f32> = (0..COLS).map(|i| (i % 997) as f32 * 0.001 - 0.5).collect();
        let x = rig.ctx.upload(&f32_bytes(&drawn)).expect("the activation");
        let y = rig
            .ctx
            .upload(&vec![0u8; ROWS * 4])
            .expect("a result buffer");
        let uploaded = rig
            .ctx
            .upload(&bytes[128..128 + BYTES])
            .expect("the uploaded weight");
        let pages = map.range("w").expect("the weight from the pages");
        let upload = uploaded
            .range(0, BYTES)
            .expect("the uploaded weight's range");
        let shape = DenseGemvShape::packed(1, ROWS, COLS);
        let rate = |us: f64| BYTES as f64 / us / 1e3;

        // The first launch from the pages pins them: wall time the GPU clock does not show.
        let start = Instant::now();
        rig.ctx
            .dense_gemv(gemv(shape, pages, &x, &y))
            .expect("the first launch from the pages");
        println!(
            "mapped GEMV [{ROWS}, {COLS}] {:.1} MB: the first launch from the pages, wall {:.3} ms",
            BYTES as f64 / 1e6,
            start.elapsed().as_secs_f64() * 1e3
        );
        for round in 1..=2 {
            for (label, w) in [("an upload", upload), ("the file's pages", pages)] {
                let gpu_ns = rig
                    .ctx
                    .dense_gemv_timed(gemv(shape, w, &x, &y), 16)
                    .expect("sixteen launches");
                let each = gpu_ns as f64 / 1e3 / 16.0;
                println!(
                    "mapped GEMV round {round}, {label}: sixteen waited launches, {each:.3} us a launch, {:.1} GB/s",
                    rate(each)
                );
                let mut frame = rig.ctx.frame_of(64);
                for _ in 0..128 {
                    frame
                        .dense_gemv(gemv(shape, w, &x, &y))
                        .expect("a dispatch");
                }
                let each = frame.drain().expect("the drain") as f64 / 1e3 / 128.0;
                println!(
                    "mapped GEMV round {round}, {label}: 128 dispatches in a frame of 64, {each:.3} us a dispatch, {:.1} GB/s",
                    rate(each)
                );
            }
        }

        // The bits are the same from both storages.
        rig.ctx
            .dense_gemv(gemv(shape, upload, &x, &y))
            .expect("the upload's launch");
        let from_upload = y.read(0, ROWS * 4).expect("the read");
        rig.ctx
            .dense_gemv(gemv(shape, pages, &x, &y))
            .expect("the pages' launch");
        assert_eq!(
            y.read(0, ROWS * 4).expect("the read"),
            from_upload,
            "the pages give the upload's bits"
        );
    }

    #[test]
    fn a_file_without_a_whole_page_or_without_a_header_is_refused_and_the_backend_lives() {
        let rig = Rig::new();
        // 128 + 1,024 bytes.
        let short = written(&gguf_file(&[("a", GgufType::F32, &[256])]));
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
