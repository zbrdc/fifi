//! The short convolution of a step: each channel of a row convolved causally with its last
//! `taps - 1` inputs, which a ring keeps on the device between steps.
//!
//! The ring and the row entry are shared; what a convolution computes is its [`Variant`]. A
//! ring states the order its rows are stored in ([`Order`]), so a variant that reads its past
//! oldest first can share the ring's shape and health word with one that reads newest first.

use crate::EngError;
use crate::kernels::cuda::resident::short_conv::ConvCall;
use crate::kernels::cuda::{CudaContext, DeviceRange, DeviceStream};

/// The order a ring's rows are stored in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Order {
    /// Row 0 is the newest past input; a step shifts the rows down and writes its input at
    /// row 0.
    NewestFirst,
}

/// The arithmetic of a convolution.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Variant {
    /// No activation and a residual: `(acc + x) * (1 / sqrt(taps + 1))`, the reciprocal of a
    /// rounded square root, where `acc` starts at 0 and adds each lag's product from the
    /// current input back to the oldest, every operation rounded to nearest even. Reads a
    /// [`Order::NewestFirst`] ring.
    Residual,
}

impl Variant {
    fn order(self) -> Order {
        match self {
            Self::Residual => Order::NewestFirst,
        }
    }
}

/// The past inputs of one convolution: `taps - 1` rows of `width` floats in `order`.
#[derive(Clone, Copy)]
pub(crate) struct Ring<'a> {
    pub(crate) history: DeviceRange<'a>,
    pub(crate) width: usize,
    pub(crate) taps: usize,
    pub(crate) order: Order,
}

impl<'a> Ring<'a> {
    /// A ring of `taps - 1` rows of `width` floats, the newest first.
    pub(crate) fn newest_first(history: DeviceRange<'a>, width: usize, taps: usize) -> Self {
        Self {
            history,
            width,
            taps,
            order: Order::NewestFirst,
        }
    }
}

/// The rows of one convolved row: the input, the taps (`width * taps` floats, tap `t` of
/// channel `c` at `c * taps + t`), the output (exactly the input, or apart from it), a row
/// the output is added into in the same launch, and the health word a value that is not
/// finite marks.
pub(crate) struct Row<'a> {
    pub(crate) input: DeviceRange<'a>,
    pub(crate) taps: DeviceRange<'a>,
    pub(crate) out: DeviceRange<'a>,
    pub(crate) add: Option<DeviceRange<'a>>,
    pub(crate) health: DeviceRange<'a>,
}

/// Convolves one row with `ring`'s past under `variant`, and moves the row into the ring.
pub(crate) fn row(
    ctx: &CudaContext,
    stream: &DeviceStream,
    variant: Variant,
    ring: Ring<'_>,
    row: Row<'_>,
) -> Result<(), EngError> {
    if ring.order != variant.order() {
        return Err(EngError::BadRequest(format!(
            "a {variant:?} convolution reads a {:?} ring, not a {:?} one",
            variant.order(),
            ring.order
        )));
    }
    match variant {
        Variant::Residual => ctx.short_conv(
            stream,
            ConvCall {
                input: row.input,
                kernel: row.taps,
                history: ring.history,
                out: row.out,
                state: row.add,
                health: row.health,
            },
            (ring.width, ring.taps),
        ),
    }
}
