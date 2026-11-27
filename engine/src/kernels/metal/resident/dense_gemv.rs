//! Shared matrix-vector projection for F16 and BF16 weights, the contract of the CUDA adapter's
//! dense matvec. Activations and results stay in device buffers between launches.
//!
//! [`MetalContext::dense_gemv`] encodes one kernel, commits its command buffer and waits. It
//! allocates nothing and copies nothing. `ld_y` lets the result land inside a wider row.
//!
//! Every refusal is a `BadRequest` and happens before any native call, because a status from the
//! native layer other than a missing device latches the backend terminal. A refused call changes
//! nothing.

use std::rc::Rc;

use crate::EngError;

use super::super::context::{Grid, adapter_bug};
use super::super::native::{self, Binding, RawKernel};
use super::super::{DeviceRange, MetalContext};

#[cfg(test)]
mod tests;

const SOURCE: &str = include_str!("../native/dense_gemv.metal");
const F16_KERNEL: &str = "dense_gemv_f16";
const BF16_KERNEL: &str = "dense_gemv_bf16";

pub(crate) const GEMV_MAX_M: usize = 32768;
pub(crate) const GEMV_MAX_ROWS: usize = 1 << 20;
pub(crate) const GEMV_MAX_COLS: usize = 12288;
pub(crate) const GEMV_MAX_LD: usize = 1 << 20;

/// Weight rows a threadgroup covers: eight SIMD groups of a row each.
const ROWS_PER_THREADGROUP: usize = 8;
const THREADGROUP: usize = 256;
/// The alignment Metal requires of a bound buffer range.
const RANGE_ALIGN: usize = 4;
/// The alignment of the vector path's 16-byte loads.
const VECTOR_ALIGN: usize = 16;
/// The weight element: binary16 or bfloat16.
const CODE_BYTES: usize = 2;
/// The activation and result element.
const VALUE_BYTES: usize = 4;
const FLAG_ZERO: [u8; 4] = [0; 4];

/// The stored dtype of a dense weight.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum DenseDtype {
    /// IEEE binary16.
    F16,
    /// bfloat16.
    Bf16,
}

impl DenseDtype {
    fn kernel(self) -> &'static str {
        match self {
            Self::F16 => F16_KERNEL,
            Self::Bf16 => BF16_KERNEL,
        }
    }
}

/// One dense matvec, `y[t][r] = sum over c of w[r][c] * x[t][c]`, for `t` below `m`, `r` below
/// `rows` and `c` below `cols`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct DenseGemvShape {
    /// Activation rows, the tokens of the batch.
    pub(crate) m: usize,
    /// Weight rows, the output features.
    pub(crate) rows: usize,
    /// Weight columns, the input features.
    pub(crate) cols: usize,
    /// Elements between two activation rows; at least `cols`.
    pub(crate) ld_x: usize,
    /// Elements between two result rows; at least `rows`.
    pub(crate) ld_y: usize,
}

impl DenseGemvShape {
    pub(crate) const MAX_ACTIVATION_ROWS: usize = GEMV_MAX_M;

    /// Compact activation rows and compact results: each stride equals the width.
    pub(crate) fn packed(m: usize, rows: usize, cols: usize) -> Self {
        Self {
            m,
            rows,
            cols,
            ld_x: cols,
            ld_y: rows,
        }
    }
}

/// The bytes the three ranges of a call need at least.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct DenseGemvBytes {
    pub(crate) w: usize,
    pub(crate) x: usize,
    pub(crate) y: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct DenseGemvCall<'b> {
    pub(crate) shape: DenseGemvShape,
    pub(crate) dtype: DenseDtype,
    pub(crate) w: DeviceRange<'b>,
    pub(crate) x: DeviceRange<'b>,
    pub(crate) y: DeviceRange<'b>,
}

pub(crate) fn dense_gemv_bytes(
    shape: DenseGemvShape,
    dtype: DenseDtype,
) -> Result<DenseGemvBytes, EngError> {
    gemv_bytes(shape, dtype).ok_or_else(|| invalid_shape(shape))
}

fn gemv_bytes(shape: DenseGemvShape, dtype: DenseDtype) -> Option<DenseGemvBytes> {
    let DenseGemvShape {
        m,
        rows,
        cols,
        ld_x,
        ld_y,
    } = shape;
    let code_bytes = match dtype {
        DenseDtype::F16 | DenseDtype::Bf16 => CODE_BYTES,
    };
    let stride =
        |stride: usize, width: usize| (stride >= width && stride <= GEMV_MAX_LD).then_some(stride);
    let ld_x = stride(ld_x, cols)?;
    let ld_y = stride(ld_y, rows)?;
    if m == 0
        || m > GEMV_MAX_M
        || rows == 0
        || rows > GEMV_MAX_ROWS
        || cols == 0
        || cols > GEMV_MAX_COLS
    {
        return None;
    }
    Some(DenseGemvBytes {
        w: rows.checked_mul(cols)? * code_bytes,
        x: (m - 1).checked_mul(ld_x)?.checked_add(cols)? * VALUE_BYTES,
        y: (m - 1).checked_mul(ld_y)?.checked_add(rows)? * VALUE_BYTES,
    })
}

fn invalid_shape(shape: DenseGemvShape) -> EngError {
    EngError::BadRequest(format!(
        "the Metal dense matvec takes 1 to {GEMV_MAX_M} rows of activations, 1 to {GEMV_MAX_ROWS} weight rows, 1 to {GEMV_MAX_COLS} columns and strides from the row width to {GEMV_MAX_LD}, got {shape:?}"
    ))
}

/// The kernel's parameters, the layout of `GemvParams` in the shader.
#[repr(C)]
#[derive(Clone, Copy)]
struct GemvParams {
    m: u32,
    rows: u32,
    cols: u32,
    ld_x: u32,
    ld_y: u32,
    vector: u32,
}

const _: () = assert!(std::mem::size_of::<GemvParams>() == 24);

impl GemvParams {
    fn bytes(self) -> [u8; 24] {
        let mut out = [0u8; 24];
        let fields = [
            self.m,
            self.rows,
            self.cols,
            self.ld_x,
            self.ld_y,
            self.vector,
        ];
        for (slot, value) in fields.into_iter().enumerate() {
            out[slot * 4..slot * 4 + 4].copy_from_slice(&value.to_le_bytes());
        }
        out
    }
}

/// A call after its checks: the three bindings, the shader and its parameters.
#[derive(Clone, Copy)]
struct DenseGemvPlan {
    w: Binding,
    x: Binding,
    y: Binding,
    kernel: &'static str,
    params: GemvParams,
    rows: usize,
    m: usize,
}

impl MetalContext {
    /// One dense matvec, `y[t][r] = sum over c of w[r][c] * x[t][c]`, every product one rounded
    /// multiply and add.
    pub(crate) fn dense_gemv(&self, call: DenseGemvCall<'_>) -> Result<(), EngError> {
        self.dense_gemv_repeated(call, 1).map(|_| ())
    }

    /// The same dispatch `repeats` times in one command buffer, for a timing test: the GPU time
    /// of the command buffer in nanoseconds.
    #[cfg(test)]
    pub(crate) fn dense_gemv_timed(
        &self,
        call: DenseGemvCall<'_>,
        repeats: usize,
    ) -> Result<u64, EngError> {
        self.dense_gemv_repeated(call, repeats)
    }

    fn dense_gemv_repeated(
        &self,
        call: DenseGemvCall<'_>,
        repeats: usize,
    ) -> Result<u64, EngError> {
        let plan = self.plan_dense_gemv(&call)?;
        let kernel: Rc<RawKernel> = self.kernel(plan.kernel, SOURCE)?;
        // Eight weight rows a threadgroup of 256 threads, one activation row a grid row.
        let grid: Grid = [
            plan.rows.div_ceil(ROWS_PER_THREADGROUP) * THREADGROUP,
            plan.m,
            1,
        ];
        self.shared.with_flag(|flag| {
            native::buffer_write(flag, 0, &FLAG_ZERO)
                .map_err(|_| adapter_bug("the write that clears the non-finite weight flag"))?;
            let bindings = [
                plan.w,
                plan.x,
                plan.y,
                Binding {
                    pointer: flag.pointer(),
                    offset: 0,
                },
            ];
            let gpu_ns = self.launch_repeated(
                &kernel,
                &bindings,
                &plan.params.bytes(),
                grid,
                [THREADGROUP, 1, 1],
                repeats,
            )?;
            let set = native::buffer_read(flag, 0, FLAG_ZERO.len())
                .map_err(|_| adapter_bug("the read of the non-finite weight flag"))?;
            if set != FLAG_ZERO {
                return Err(EngError::BadRequest(
                    "the Metal dense matvec read a weight that is not finite".to_owned(),
                ));
            }
            Ok(gpu_ns)
        })
    }

    fn plan_dense_gemv(&self, call: &DenseGemvCall<'_>) -> Result<DenseGemvPlan, EngError> {
        let shape = call.shape;
        let need = gemv_bytes(shape, call.dtype).ok_or_else(|| invalid_shape(shape))?;
        let ranges = [
            ("weight", &call.w, need.w),
            ("activation", &call.x, need.x),
            ("result", &call.y, need.y),
        ];
        for (what, range, need) in ranges {
            let same = range.owner() == self.id();
            let fits = range.bytes() >= need && range.offset() % RANGE_ALIGN == 0;
            if !same || !fits {
                let whose = if same { "" } else { " of another adapter" };
                return Err(EngError::BadRequest(format!(
                    "the call needs {need} bytes of {what} at a multiple of {RANGE_ALIGN} in a buffer of its own Metal context; the range holds {} bytes at offset {}{whose}",
                    range.bytes(),
                    range.offset(),
                )));
            }
        }
        if overlaps(&call.y, &call.w) || overlaps(&call.y, &call.x) {
            return Err(EngError::BadRequest(
                "the result range overlaps the weight or the activation range".to_owned(),
            ));
        }
        let DenseGemvShape {
            m,
            rows,
            cols,
            ld_x,
            ld_y,
        } = shape;
        let vector = u32::from(
            call.w.offset().is_multiple_of(VECTOR_ALIGN)
                && call.x.offset().is_multiple_of(VECTOR_ALIGN)
                && cols.is_multiple_of(8)
                && ld_x.is_multiple_of(4),
        );
        let params = GemvParams {
            m: m as u32,
            rows: rows as u32,
            cols: cols as u32,
            ld_x: ld_x as u32,
            ld_y: ld_y as u32,
            vector,
        };
        Ok(DenseGemvPlan {
            w: call.w.binding(),
            x: call.x.binding(),
            y: call.y.binding(),
            kernel: call.dtype.kernel(),
            params,
            rows,
            m,
        })
    }
}

/// Whether two ranges share an address.
fn overlaps(a: &DeviceRange<'_>, b: &DeviceRange<'_>) -> bool {
    a.address() < b.address() + b.bytes() && b.address() < a.address() + a.bytes()
}
