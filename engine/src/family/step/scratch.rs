//! Where the scratch of a step comes from.
//!
//! A model declares its scratch once, as a function of an [`Alloc`]. Run with [`OnDevice`] the
//! declaration allocates the buffers; run with [`Count`] it only adds up their bytes, so the
//! memory plan of a load cannot drift from what the load then allocates.

use std::cell::Cell;

use super::delta::DeltaRows;
use super::rows::{Rows, q8};
use super::stepper::ShortlistIo;
use crate::EngError;
use crate::kernels::cuda::prefill::mmq_size::q8_1_bytes as product_q8_bytes;
use crate::kernels::cuda::resident::shortlist;
use crate::kernels::cuda::{CudaContext, DeviceBuffer, DeviceStream};

pub(crate) trait Alloc {
    type Buf;

    /// `bytes` bytes whose content the step writes before it reads.
    fn bytes(&self, bytes: usize) -> Result<Self::Buf, EngError>;

    /// `bytes` zero bytes.
    fn zeroed(&self, bytes: usize) -> Result<Self::Buf, EngError>;

    /// `count` f32 values.
    fn floats(&self, count: usize) -> Result<Self::Buf, EngError> {
        self.bytes(count * size_of::<f32>())
    }

    /// The q8_1 rows of `rows` rows of `width` values.
    fn q8(&self, width: usize, rows: usize) -> Result<Self::Buf, EngError> {
        self.bytes(q8(width, rows))
    }
}

/// Allocates on the device of a context.
pub(crate) struct OnDevice<'a>(pub(crate) &'a CudaContext);

impl Alloc for OnDevice<'_> {
    type Buf = DeviceBuffer;

    fn bytes(&self, bytes: usize) -> Result<DeviceBuffer, EngError> {
        self.0.allocate(bytes)
    }

    fn zeroed(&self, bytes: usize) -> Result<DeviceBuffer, EngError> {
        self.0.upload(&vec![0u8; bytes])
    }
}

/// Hands out zero bytes wherever the inner allocator would hand out unwritten ones.
pub(crate) struct Zeroed<A>(pub(crate) A);

impl<A: Alloc> Alloc for Zeroed<A> {
    type Buf = A::Buf;

    fn bytes(&self, bytes: usize) -> Result<A::Buf, EngError> {
        self.0.zeroed(bytes)
    }

    fn zeroed(&self, bytes: usize) -> Result<A::Buf, EngError> {
        self.0.zeroed(bytes)
    }
}

/// Adds up the bytes a declaration asks for and allocates nothing.
#[derive(Default)]
pub(crate) struct Count(Cell<u64>);

impl Count {
    pub(crate) fn total(&self) -> u64 {
        self.0.get()
    }

    fn add(&self, bytes: usize) {
        self.0.set(self.0.get().saturating_add(bytes as u64));
    }
}

impl Alloc for Count {
    type Buf = ();

    fn bytes(&self, bytes: usize) -> Result<(), EngError> {
        self.add(bytes);
        Ok(())
    }

    fn zeroed(&self, bytes: usize) -> Result<(), EngError> {
        self.add(bytes);
        Ok(())
    }
}

/// The widths of the rows of a step, in values per token row.
#[derive(Clone, Copy)]
pub(crate) struct Widths {
    /// Token rows of one chunk.
    pub(crate) rows: usize,
    pub(crate) hidden: usize,
    /// Residual streams a token row carries.
    pub(crate) streams: usize,
    /// The widest strided row a mixer's projections write.
    pub(crate) proj: usize,
    /// The widest convolved row.
    pub(crate) conv: usize,
    /// The widest row a mixer reads back before its output projection.
    pub(crate) read: usize,
    /// The widest input of a batched product.
    pub(crate) product: usize,
    pub(crate) vocab: usize,
    /// Bytes of the batched product's fix-up words on the device.
    pub(crate) fixup: usize,
}

/// The rows every step works in: the residual pair, the normed input of a branch, the rows of a
/// mixer, the scratch of the batched product and the logits row.
pub(crate) struct Core<B = DeviceBuffer> {
    pub(crate) resid: [B; 2],
    /// Which plane of the pair holds the residual.
    pub(crate) cur: Cell<usize>,
    pub(crate) normed: B,
    pub(crate) normed_q8: B,
    /// The mixer output rows and the feed-forward output rows, hidden wide.
    pub(crate) mix_out: B,
    pub(crate) forward_out: B,
    pub(crate) proj: B,
    pub(crate) conv_out: B,
    pub(crate) read: B,
    pub(crate) read_q8: B,
    pub(crate) product_q8: B,
    pub(crate) fixup: B,
    pub(crate) logits: B,
    pub(crate) shortlist: B,
}

impl<B> Core<B> {
    pub(crate) fn new<A: Alloc<Buf = B>>(a: &A, w: Widths) -> Result<Self, EngError> {
        let n = w.rows;
        let stream_rows = n * w.streams * w.hidden;
        // The rows a mixer reads back serve the quantized matvec and the batched product.
        let read_q8 = q8(w.read, n).max(product_q8_bytes(w.read, n)?);
        Ok(Self {
            resid: [a.floats(stream_rows)?, a.floats(stream_rows)?],
            cur: Cell::new(0),
            normed: a.floats(n * w.hidden)?,
            normed_q8: a.q8(w.hidden, n)?,
            mix_out: a.floats(n * w.hidden)?,
            forward_out: a.floats(n * w.hidden)?,
            proj: a.floats(n * w.proj)?,
            conv_out: a.floats(n * w.conv)?,
            read: a.floats(n * w.read)?,
            read_q8: a.bytes(read_q8)?,
            product_q8: a.bytes(product_q8_bytes(w.product, n)?)?,
            fixup: a.bytes(w.fixup)?,
            logits: a.floats(w.vocab)?,
            shortlist: a.bytes(shortlist::MASS_SCRATCH_BYTES)?,
        })
    }
}

impl Core {
    /// The normed input rows of a branch.
    pub(crate) fn normed_rows(&self) -> Rows<'_> {
        Rows::both(&self.normed_q8, &self.normed)
    }

    /// The rows of a delta-rule visit.
    pub(crate) fn delta_rows(&self) -> DeltaRows<'_> {
        DeltaRows {
            input: self.normed_rows(),
            proj: &self.proj,
            conv_out: &self.conv_out,
            read: Rows::both(&self.read_q8, &self.read),
            out: &self.mix_out,
            channel: None,
        }
    }

    /// Device buffers for logits readback and host-equal shortlist sampling.
    pub(crate) fn shortlist_io<'a>(
        &'a self,
        ctx: &'a CudaContext,
        stream: &'a DeviceStream,
        vocab: usize,
    ) -> ShortlistIo<'a> {
        ShortlistIo {
            ctx,
            stream,
            logits: &self.logits,
            scratch: &self.shortlist,
            vocab,
        }
    }
}
