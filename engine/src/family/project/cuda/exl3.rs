//! Typed EXL3 upload seam; family routing remains off until model qualification.

use half::f16;

use super::{CudaProjector, Reserve};
#[cfg(test)]
use crate::weights::exl3::Checkpoint;
use crate::{
    EngError,
    kernels::cuda::{
        CudaContext, DeviceBuffer, DeviceRange, DeviceStream,
        resident::exl3::{Exl3IndexedCall, Exl3JobsCall, Exl3LinearCall},
    },
    tensor::{Dtype, Tensor},
    weights::exl3::{Codebook, Matrix},
};

pub(crate) struct Exl3Weight {
    // Reverse upload order also applies when the complete weight is dropped.
    pub(crate) bias: Option<DeviceBuffer>,
    pub(crate) svh: DeviceBuffer,
    pub(crate) suh: DeviceBuffer,
    pub(crate) trellis: DeviceBuffer,
    pub(crate) shape: [usize; 2],
    pub(crate) bits_twice: u8,
    pub(crate) book: Codebook,
}

/// Same-shaped, same-rate matrices without bias in one device buffer: the routed experts of one
/// projection of one layer. One allocation replaces one per matrix and expert; a member's
/// trellis and scales live in one record of `trellis + scale` bytes at a fixed stride, so a
/// slot fill or miss copy is a single contiguous write.
pub(crate) struct Exl3Bank {
    records: DeviceBuffer,
    count: usize,
    shape: [usize; 2],
    bits_twice: u8,
    book: Codebook,
}

/// The whole-bank ranges of an id-indexed launch: member 0's spans extended over every member.
#[derive(Clone, Copy)]
pub(crate) struct Exl3BankSpan<'a> {
    trellis: DeviceRange<'a>,
    suh: DeviceRange<'a>,
    svh: DeviceRange<'a>,
    members: u32,
    trellis_stride: usize,
    scale_stride: usize,
    shape: [usize; 2],
    bits_twice: u8,
    book: Codebook,
}

impl<'a> Exl3BankSpan<'a> {
    /// A span over caller-held, record-strided weights this module did not load: the bank
    /// staging of a slot map or a ring copy. The arguments are the fields of the struct, in
    /// order; the kernels validate every extent per call.
    #[allow(clippy::too_many_arguments)] // a struct-literal of a private-fields type
    pub(in crate::family) fn from_parts(
        trellis: DeviceRange<'a>,
        suh: DeviceRange<'a>,
        svh: DeviceRange<'a>,
        members: u32,
        trellis_stride: usize,
        scale_stride: usize,
        shape: [usize; 2],
        bits_twice: u8,
        book: Codebook,
    ) -> Self {
        Self {
            trellis,
            suh,
            svh,
            members,
            trellis_stride,
            scale_stride,
            shape,
            bits_twice,
            book,
        }
    }

    /// The `[k, n]` every member projects with; the pairs of one launch share it.
    pub(in crate::family) fn shape(&self) -> [usize; 2] {
        self.shape
    }

    /// The call that projects `io`'s row with the member `ids[slot]`; the ids are chosen on the
    /// device when the launch runs.
    pub(in crate::family) fn indexed_call(
        &self,
        io: &Exl3Io<'a>,
        ids: DeviceRange<'a>,
        slot: u32,
    ) -> Exl3IndexedCall<'a> {
        Exl3IndexedCall {
            call: Exl3LinearCall {
                trellis: self.trellis,
                suh: self.suh,
                svh: self.svh,
                bias: None,
                x: io.x,
                xh: io.xh,
                z: io.z,
                out: io.out,
                flag: io.flag,
                rows: io.rows,
                split_k: 2,
                k: self.shape[0],
                n: self.shape[1],
                bits_twice: self.bits_twice,
                book: self.book,
                fused: io.fused,
            },
            ids,
            slot,
            members: self.members,
            trellis_stride: self.trellis_stride,
            scale_stride: self.scale_stride,
            offsets: None,
        }
    }
}

/// Borrowed packed components; the owner outlives every in-flight projection.
#[derive(Clone, Copy)]
pub(crate) struct Exl3MatrixView<'a> {
    pub(crate) trellis: DeviceRange<'a>,
    pub(crate) suh: DeviceRange<'a>,
    pub(crate) svh: DeviceRange<'a>,
    pub(crate) bias: Option<DeviceRange<'a>>,
    pub(crate) shape: [usize; 2],
    pub(crate) bits_twice: u8,
    pub(crate) book: Codebook,
}

/// Contiguous F32 activation/output rows and disjoint caller-owned scratch.
/// The owner initializes the sticky flag once and retains it across dependent work.
/// Standalone projection checks health before and after enqueue. The private Flint Frame
/// may defer scratch-only value consumers until its router/final status observations,
/// before any unchecked address/index/geometry decision or external output/state publication.
pub(crate) struct Exl3Io<'a> {
    pub(crate) x: DeviceRange<'a>,
    pub(crate) xh: DeviceRange<'a>,
    pub(crate) z: DeviceRange<'a>,
    pub(crate) out: DeviceRange<'a>,
    pub(crate) flag: DeviceRange<'a>,
    pub(crate) rows: usize,
    /// Producer-fusion arm: out receives the bf16 roundtrip the separate
    /// publish kernel would write, so the caller skips that publish.
    pub(crate) fused: bool,
}

/// The members of one bank (or one matrix as a bank of one) as the job-table projection addresses
/// them: member `i` starts `i * trellis_stride` u16 elements into `trellis` and `i * scale_stride`
/// floats into `suh` and `svh`, the addressing of [`Exl3Bank::view`].
#[derive(Clone, Copy)]
pub(crate) struct Exl3Members<'a> {
    pub(crate) trellis: DeviceRange<'a>,
    pub(crate) suh: DeviceRange<'a>,
    pub(crate) svh: DeviceRange<'a>,
    /// u16 elements between the trellises of two members.
    pub(crate) trellis_stride: usize,
    /// F32 values between the `suh` (and the `svh`) of two members.
    pub(crate) scale_stride: usize,
    pub(crate) members: usize,
    pub(crate) shape: [usize; 2],
    pub(crate) bits_twice: u8,
    pub(crate) book: Codebook,
}

/// The rows of a job-table projection: the F32 input rows, the scratch of the rotation, the
/// partials and the output (row `r` of a job at `out[(row0 + r) * out_ld + out_col ..]`), the
/// sticky flag and the device table.
pub(crate) struct Exl3JobsIo<'a> {
    pub(crate) x: DeviceRange<'a>,
    pub(crate) xh: DeviceRange<'a>,
    pub(crate) z: DeviceRange<'a>,
    pub(crate) out: DeviceRange<'a>,
    pub(crate) flag: DeviceRange<'a>,
    pub(crate) jobs: DeviceRange<'a>,
    pub(crate) jobs_count: usize,
    pub(crate) max_rows: usize,
    pub(crate) total_rows: usize,
    pub(crate) out_ld: usize,
    pub(crate) out_col: usize,
}

impl<'a> Exl3Members<'a> {
    /// Maps borrowed spans only; enqueueing and observing health remain caller responsibilities.
    pub(in crate::family) fn jobs_call(&self, io: &Exl3JobsIo<'a>) -> Exl3JobsCall<'a> {
        Exl3JobsCall {
            trellis: self.trellis,
            suh: self.suh,
            svh: self.svh,
            trellis_stride: self.trellis_stride,
            scale_stride: self.scale_stride,
            members: self.members,
            x: io.x,
            xh: io.xh,
            z: io.z,
            out: io.out,
            flag: io.flag,
            jobs: io.jobs,
            jobs_count: io.jobs_count,
            max_rows: io.max_rows,
            total_rows: io.total_rows,
            out_ld: io.out_ld,
            out_col: io.out_col,
            k: self.shape[0],
            n: self.shape[1],
            bits_twice: self.bits_twice,
            book: self.book,
            in_rows: None,
            out_rows: None,
            x_rows: io.total_rows,
            out_total_rows: io.total_rows,
        }
    }
}

impl<'a> Exl3MatrixView<'a> {
    /// The id-indexed call that projects `io`'s row with this one matrix: a bank of ONE member, so
    /// `ids[slot]` must read 0 when the launch runs (the kernels add `id * stride` to the
    /// matrix's own addresses, and 0 adds nothing). The strides are the matrix's own sizes, which
    /// the launch only needs non-zero; they are never multiplied by a nonzero id. A matrix with a
    /// bias keeps it: the kernels add `bias[col]` of the call, not of a member.
    pub(in crate::family) fn indexed_call(
        self,
        io: &Exl3Io<'a>,
        ids: DeviceRange<'a>,
        slot: u32,
    ) -> Exl3IndexedCall<'a> {
        Exl3IndexedCall {
            call: Exl3LinearCall {
                trellis: self.trellis,
                suh: self.suh,
                svh: self.svh,
                bias: self.bias,
                x: io.x,
                xh: io.xh,
                z: io.z,
                out: io.out,
                flag: io.flag,
                rows: io.rows,
                split_k: 2,
                k: self.shape[0],
                n: self.shape[1],
                bits_twice: self.bits_twice,
                book: self.book,
                fused: io.fused,
            },
            ids,
            slot,
            members: 1,
            trellis_stride: self.trellis.bytes(),
            scale_stride: self.suh.bytes() / size_of::<f32>(),
            offsets: None,
        }
    }
}

impl Exl3MatrixView<'_> {
    /// Reads F32 directly, before either the q8_1 or tiled fp16 activation pack.
    pub(crate) fn project(
        &self,
        context: &CudaContext,
        stream: &DeviceStream,
        io: Exl3Io<'_>,
    ) -> Result<(), EngError> {
        context.ensure_usable()?;
        if io.flag.bytes() < 4 {
            return Err(EngError::BadRequest(
                "EXL3 arithmetic flag needs four bytes".into(),
            ));
        }
        // shortcut: host status checks exclude graph capture; use resident status before timing.
        let check_flag = || {
            let mut status = [0u8; 4];
            io.flag
                .buffer()
                .read_at(stream, io.flag.offset(), &mut status)?;
            if u32::from_le_bytes(status) & 1 != 0 {
                return Err(EngError::BadRequest(
                    "EXL3 projection flagged invalid arithmetic".into(),
                ));
            }
            Ok(())
        };
        check_flag()?;
        context.exl3_linear(stream, self.linear_call(&io))?;
        check_flag()
    }

    /// Maps borrowed spans only; enqueueing and observing health remain caller responsibilities.
    pub(in crate::family) fn linear_call<'a>(&'a self, io: &Exl3Io<'a>) -> Exl3LinearCall<'a> {
        Exl3LinearCall {
            trellis: self.trellis,
            suh: self.suh,
            svh: self.svh,
            bias: self.bias,
            x: io.x,
            xh: io.xh,
            z: io.z,
            out: io.out,
            flag: io.flag,
            rows: io.rows,
            split_k: 2,
            k: self.shape[0],
            n: self.shape[1],
            bits_twice: self.bits_twice,
            book: self.book,
            fused: io.fused,
        }
    }
}

impl Exl3Weight {
    /// The matrix as a bank of one member for a job-table projection (a weight with a bias is
    /// refused: the job-table output adds none).
    pub(crate) fn members(&self) -> Result<Exl3Members<'_>, EngError> {
        if self.bias.is_some() {
            return Err(EngError::BadRequest(
                "EXL3 job-table projection: the matrix has a bias".into(),
            ));
        }
        Ok(Exl3Members {
            trellis: self.trellis.range(0, self.trellis.bytes())?,
            suh: self.suh.range(0, self.suh.bytes())?,
            svh: self.svh.range(0, self.svh.bytes())?,
            trellis_stride: 0,
            scale_stride: 0,
            members: 1,
            shape: self.shape,
            bits_twice: self.bits_twice,
            book: self.book,
        })
    }

    pub(crate) fn view(&self) -> Result<super::Exl3MatrixView<'_>, EngError> {
        Ok(Exl3MatrixView {
            trellis: self.trellis.range(0, self.trellis.bytes())?,
            suh: self.suh.range(0, self.suh.bytes())?,
            svh: self.svh.range(0, self.svh.bytes())?,
            bias: self
                .bias
                .as_ref()
                .map(|b| b.range(0, b.bytes()))
                .transpose()?,
            shape: self.shape,
            bits_twice: self.bits_twice,
            book: self.book,
        })
    }

    pub(crate) fn project(
        &self,
        context: &CudaContext,
        stream: &DeviceStream,
        io: Exl3Io<'_>,
    ) -> Result<(), EngError> {
        context.ensure_usable()?;
        self.view()?.project(context, stream, io)
    }

    pub(crate) fn device_buffers(&self) -> impl Iterator<Item = &DeviceBuffer> {
        [
            Some(&self.trellis),
            Some(&self.suh),
            Some(&self.svh),
            self.bias.as_ref(),
        ]
        .into_iter()
        .flatten()
    }

    pub(crate) fn device_bytes(&self) -> usize {
        self.device_buffers().map(DeviceBuffer::bytes).sum()
    }
}

impl Exl3Bank {
    fn trellis_stride(&self) -> usize {
        self.shape[0] * self.shape[1] / 256 * usize::from(self.bits_twice) * 16
    }

    /// F32 scale bytes of one member: `suh` then `svh`.
    fn scale_stride(&self) -> usize {
        (self.shape[0] + self.shape[1]) * size_of::<f32>()
    }

    /// Bytes of one member's record: its trellis then its scales, one contiguous unit.
    fn record_stride(&self) -> usize {
        self.trellis_stride() + self.scale_stride()
    }

    pub(crate) fn count(&self) -> usize {
        self.count
    }

    pub(crate) fn shape(&self) -> [usize; 2] {
        self.shape
    }

    pub(crate) fn bits_twice(&self) -> u8 {
        self.bits_twice
    }

    /// The record buffer in member order, for a banked slot fill: member `i`'s trellis and
    /// scales are the `record` bytes at `i * record`
    /// ([`exl3_member_layout`] per component).
    pub(crate) fn bank_buffer(&self) -> &DeviceBuffer {
        &self.records
    }

    /// The bytes between the records of two members (see [`Self::bank_buffer`]).
    pub(crate) fn record_bytes(&self) -> usize {
        self.record_stride()
    }

    pub(crate) fn view(&self, index: usize) -> Result<Exl3MatrixView<'_>, EngError> {
        if index >= self.count {
            return Err(EngError::BadRequest(format!(
                "EXL3 bank member {index} is outside {} members",
                self.count
            )));
        }
        let (trellis, scales) = (self.trellis_stride(), self.scale_stride());
        let suh_bytes = self.shape[0] * size_of::<f32>();
        let at = index * self.record_stride();
        Ok(Exl3MatrixView {
            trellis: self.records.range(at, trellis)?,
            suh: self.records.range(at + trellis, suh_bytes)?,
            svh: self
                .records
                .range(at + trellis + suh_bytes, scales - suh_bytes)?,
            bias: None,
            shape: self.shape,
            bits_twice: self.bits_twice,
            book: self.book,
        })
    }

    pub(crate) fn members(&self) -> Result<Exl3Members<'_>, EngError> {
        let [k, n] = self.shape;
        let trellis = self.trellis_stride();
        let record = self.record_stride();
        let last = (self.count - 1) * record;
        Ok(Exl3Members {
            // The kernel indexes each component by the record stride from the component's
            // own base offset inside the shared buffer.
            trellis: self.records.range(0, last + trellis)?,
            suh: self.records.range(trellis, last + k * size_of::<f32>())?,
            svh: self
                .records
                .range(trellis + k * size_of::<f32>(), last + n * size_of::<f32>())?,
            trellis_stride: record / size_of::<u16>(),
            scale_stride: record / size_of::<f32>(),
            members: self.count,
            shape: self.shape,
            bits_twice: self.bits_twice,
            book: self.book,
        })
    }

    /// The bank as one span for an id-indexed launch.
    pub(crate) fn span(&self) -> Result<Exl3BankSpan<'_>, EngError> {
        let suh_bytes = self.shape[0] * size_of::<f32>();
        let (trellis, _) = (self.trellis_stride(), self.scale_stride());
        let members = u32::try_from(self.count)
            .map_err(|_| EngError::BadRequest("EXL3 bank has too many members".into()))?;
        let record = self.record_stride();
        Ok(Exl3BankSpan {
            trellis: self.records.range(0, self.records.bytes())?,
            suh: self
                .records
                .range(trellis, self.records.bytes() - trellis)?,
            svh: self.records.range(
                trellis + suh_bytes,
                self.records.bytes() - trellis - suh_bytes,
            )?,
            members,
            trellis_stride: record,
            scale_stride: record / size_of::<f32>(),
            shape: self.shape,
            bits_twice: self.bits_twice,
            book: self.book,
        })
    }

    pub(crate) fn device_buffers(&self) -> impl Iterator<Item = &DeviceBuffer> {
        [&self.records].into_iter()
    }

    pub(crate) fn device_bytes(&self) -> usize {
        self.device_buffers().map(DeviceBuffer::bytes).sum()
    }
}

impl CudaProjector {
    /// Reads one matrix's stored components; the checkpoint retains metadata, not shard payloads.
    #[cfg(test)]
    pub(crate) fn load_exl3(
        &self,
        checkpoint: &Checkpoint,
        prefix: &str,
        reserve: Reserve,
    ) -> Result<Exl3Weight, EngError> {
        self.context.ensure_usable()?;
        let names: Vec<_> = crate::weights::exl3::COMPONENT_SUFFIXES
            .into_iter()
            .map(|suffix| format!("{prefix}.{suffix}"))
            .collect();
        let keys: Vec<_> = names.iter().map(String::as_str).collect();
        let selected = checkpoint.source.read_selected(&keys, 1 << 30)?;
        let matrix = Matrix::load(&selected, prefix, &checkpoint.config)?;
        self.upload_exl3(&matrix, reserve)
    }

    /// Uploads an already admitted matrix without expanding its packed trellis.
    pub(crate) fn upload_exl3(
        &self,
        matrix: &Matrix,
        reserve: Reserve,
    ) -> Result<Exl3Weight, EngError> {
        self.context.ensure_usable()?;
        let trellis = matrix
            .trellis()
            .host_bytes()
            .ok_or_else(|| EngError::Checkpoint("EXL3 trellis requires host storage".into()))?;
        let suh = float_bytes(matrix.suh())?;
        let svh = float_bytes(matrix.svh())?;
        let bias = matrix.bias().map(float_bytes).transpose()?;
        let bytes = trellis
            .len()
            .checked_add(suh.len())
            .and_then(|bytes| bytes.checked_add(svh.len()))
            .and_then(|bytes| bytes.checked_add(bias.as_ref().map_or(0, Vec::len)))
            .and_then(|bytes| u64::try_from(bytes).ok())
            .ok_or_else(|| EngError::MemoryPlan("EXL3 upload byte count overflow".into()))?;
        self.admit(bytes, reserve)?;
        let trellis = self.context.upload(trellis)?;
        let suh = self.context.upload(&suh)?;
        let svh = self.context.upload(&svh)?;
        let bias = bias
            .as_deref()
            .map(|bytes| self.context.upload(bytes))
            .transpose()?;
        Ok(Exl3Weight {
            bias,
            svh,
            suh,
            trellis,
            shape: matrix.shape(),
            bits_twice: matrix.bitrate_twice(),
            book: matrix.codebook(),
        })
    }
}

impl CudaProjector {
    /// Uploads `count` members into one bank. `member(i)` yields the stored matrix of member `i`,
    /// one at a time, so the host holds one matrix beside the bank's scale vector. Every member
    /// must match the first in shape, rate and codebook and carry no bias; the whole bank is
    /// admitted before the first allocation.
    pub(crate) fn upload_exl3_bank(
        &self,
        count: usize,
        reserve: Reserve,
        mut member: impl FnMut(usize) -> Result<Matrix, EngError>,
    ) -> Result<Exl3Bank, EngError> {
        self.context.ensure_usable()?;
        if count == 0 {
            return Err(EngError::BadRequest("EXL3 bank without members".into()));
        }
        let first = member(0)?;
        let (shape, bits_twice, book) = (first.shape(), first.bitrate_twice(), first.codebook());
        let (stride, scale_stride) = exl3_member_layout(shape, bits_twice);
        let overflow = || EngError::MemoryPlan("EXL3 bank byte count overflow".into());
        let record = stride.checked_add(scale_stride).ok_or_else(overflow)?;
        let total = record
            .checked_mul(count)
            .and_then(|bytes| u64::try_from(bytes).ok())
            .ok_or_else(overflow)?;
        self.admit(total, reserve)?;
        let mut records = Vec::new();
        records
            .try_reserve_exact(total as usize)
            .map_err(|error| EngError::MemoryPlan(format!("EXL3 bank record staging: {error}")))?;
        let mut next = Some(first);
        for index in 0..count {
            let matrix = match next.take() {
                Some(matrix) => matrix,
                None => member(index)?,
            };
            if matrix.shape() != shape
                || matrix.bitrate_twice() != bits_twice
                || matrix.codebook() != book
                || matrix.bias().is_some()
            {
                return Err(EngError::Checkpoint(format!(
                    "EXL3 bank member {index} differs from member 0 in shape, rate, codebook or bias"
                )));
            }
            let staged = exl3_member_staging(&matrix)?;
            records.extend_from_slice(&staged.trellis);
            records.extend_from_slice(&staged.scales);
        }
        let records = self.context.upload(&records)?;
        // The member views divide the record by u16/f32 element sizes and the register
        // kernel fast path wants 16B alignment; the layout guarantees both.
        debug_assert_eq!(exl3_member_layout(shape, bits_twice).0 % 16, 0);
        debug_assert_eq!(
            (exl3_member_layout(shape, bits_twice).0 + exl3_member_layout(shape, bits_twice).1) % 4,
            0
        );
        Ok(Exl3Bank {
            records,
            count,
            shape,
            bits_twice,
            book,
        })
    }
}

/// The byte layout of one member of an [`Exl3Bank`]: the trellis stride (u16 elements) and the
/// scale stride (bytes, `suh` then `svh` of F32) between two members. Shared by the device bank
/// upload and the banked host store, which must paste the same layout into slots.
pub(crate) fn exl3_member_layout(shape: [usize; 2], bits_twice: u8) -> (usize, usize) {
    let stride = shape[0] * shape[1] / 256 * usize::from(bits_twice) * 16;
    let scale_stride = (shape[0] + shape[1]) * size_of::<f32>();
    (stride, scale_stride)
}

/// One member's bytes in the bank layout: the packed trellis as stored and the `suh`/`svh`
/// scales widened to F32, concatenated as the bank's scale buffer orders them. The banked host
/// store holds exactly these bytes for every expert, so a miss copy moves them unchanged.
pub(crate) struct Exl3MemberStaging {
    pub(crate) trellis: Vec<u8>,
    pub(crate) scales: Vec<u8>,
}

pub(crate) fn exl3_member_staging(matrix: &Matrix) -> Result<Exl3MemberStaging, EngError> {
    let trellis = matrix
        .trellis()
        .host_bytes()
        .ok_or_else(|| EngError::Checkpoint("EXL3 trellis requires host storage".into()))?;
    let mut scales = float_bytes(matrix.suh())?;
    scales.extend_from_slice(&float_bytes(matrix.svh())?);
    Ok(Exl3MemberStaging {
        trellis: trellis.to_vec(),
        scales,
    })
}

fn float_bytes(tensor: &Tensor) -> Result<Vec<u8>, EngError> {
    let bytes = tensor
        .host_bytes()
        .ok_or_else(|| EngError::Checkpoint("EXL3 scale or bias requires host storage".into()))?;
    let multiplier = match tensor.dtype() {
        Dtype::F16 => 2,
        Dtype::F32 => 1,
        _ => {
            return Err(EngError::Checkpoint(
                "EXL3 scale or bias requires F16 or F32".into(),
            ));
        }
    };
    let output_len = bytes
        .len()
        .checked_mul(multiplier)
        .ok_or_else(|| EngError::MemoryPlan("EXL3 staging byte count overflow".into()))?;
    let mut output = Vec::new();
    output
        .try_reserve_exact(output_len)
        .map_err(|error| EngError::MemoryPlan(format!("EXL3 staging allocation: {error}")))?;
    if tensor.dtype() == Dtype::F32 {
        output.extend_from_slice(bytes);
    } else {
        for word in bytes.as_chunks::<2>().0 {
            output.extend_from_slice(
                &f16::from_bits(u16::from_le_bytes(*word))
                    .to_f32()
                    .to_le_bytes(),
            );
        }
    }
    Ok(output)
}

/// One row through `view` on a fresh stream, read back with the arithmetic flag: the F32 output
/// and the flag word. For device tests of every caller of the seam.
#[cfg(test)]
pub(crate) fn project_one_row(
    projector: &CudaProjector,
    view: &Exl3MatrixView<'_>,
    input: &[f32],
) -> Result<(Vec<f32>, u32), EngError> {
    let [k, n] = view.shape;
    let ctx = projector.context();
    let stream = ctx.stream()?;
    let bytes = |values: &[f32]| {
        values
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<u8>>()
    };
    let x = ctx.upload(&bytes(input))?;
    let xh = ctx.upload(&vec![0; k * 4])?;
    let z = ctx.upload(&vec![0; n * 4])?;
    let out = ctx.upload(&vec![0; n * 4])?;
    let flag = ctx.upload(&[0; 4])?;
    view.project(
        ctx,
        &stream,
        Exl3Io {
            x: x.range(0, k * 4)?,
            xh: xh.range(0, k * 4)?,
            z: z.range(0, n * 4)?,
            out: out.range(0, n * 4)?,
            flag: flag.range(0, 4)?,
            rows: 1,
            fused: false,
        },
    )?;
    stream.synchronize()?;
    let mut raw = vec![0; n * 4];
    out.read_at(&stream, 0, &mut raw)?;
    let mut status = [0; 4];
    flag.read_at(&stream, 0, &mut status)?;
    let values = raw
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect();
    Ok((values, u32::from_le_bytes(status)))
}

#[cfg(test)]
mod bank_tests;
#[cfg(test)]
mod real_projection_tests;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::{Dtype, Tensor};

    #[test]
    fn scale_staging_preserves_half_values_and_float_bias_bits() {
        let bits = [0x0000u16, 0x8000, 0x3c00, 0xc000, 0x0001];
        let scale = Tensor::try_from_host(
            Dtype::F16,
            vec![5],
            bits.into_iter().flat_map(u16::to_le_bytes).collect(),
        )
        .unwrap();
        let expected = [0.0f32, -0.0, 1.0, -2.0, f32::from_bits(0x33800000)];
        assert_eq!(
            float_bytes(&scale).unwrap(),
            expected
                .into_iter()
                .flat_map(f32::to_le_bytes)
                .collect::<Vec<_>>()
        );
        let raw = [1.25f32, -0.0]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect::<Vec<_>>();
        let bias = Tensor::try_from_host(Dtype::F32, vec![2], raw.clone()).unwrap();
        assert_eq!(float_bytes(&bias).unwrap(), raw);
        let invalid = Tensor::try_from_host(Dtype::I16, vec![1], vec![0; 2]).unwrap();
        assert!(matches!(
            float_bytes(&invalid),
            Err(crate::EngError::Checkpoint(_))
        ));
    }
}
