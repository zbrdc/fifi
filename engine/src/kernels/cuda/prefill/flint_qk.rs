//! D128 Q/K preparation in F32 storage. Caller supplies BF16-published projections;
//! full layers use None, sliding layers use Some(pos0). No V or F16 KV conversion.
//! Launch-only: retain owners and consume outputs only after completion and flag==0.

use crate::EngError;
use crate::kernels::cuda::context::{
    Claim, CudaContext, DeviceRange, DeviceStream, require_apart, require_fit,
};
use crate::kernels::cuda::native::prefill::flint_qk::{self as ffi, QkRaw};
use crate::kernels::cuda::native::resident::row_scale::{Bf16PublishRaw, bf16_publish_launch};

pub(crate) struct QkProducerCall<'a> {
    pub(crate) q: DeviceRange<'a>,
    pub(crate) k: DeviceRange<'a>,
    pub(crate) gamma: [DeviceRange<'a>; 2],
    pub(crate) out: [DeviceRange<'a>; 2],
    pub(crate) scratch: [DeviceRange<'a>; 2],
    pub(crate) flag: DeviceRange<'a>,
    pub(crate) rows: usize,
    pub(crate) position: Option<u32>,
}

fn raw<'a>(
    context: &CudaContext,
    call: &QkProducerCall<'a>,
    input: [DeviceRange<'a>; 2],
) -> Result<QkRaw<'a>, EngError> {
    Ok(QkRaw {
        q: context.raw_range(&input[0])?,
        k: context.raw_range(&input[1])?,
        gamma: [
            context.raw_range(&call.gamma[0])?,
            context.raw_range(&call.gamma[1])?,
        ],
        out: [
            context.raw_range(&call.scratch[0])?,
            context.raw_range(&call.scratch[1])?,
        ],
        flag: context.raw_range(&call.flag)?,
        rows: call.rows,
        position: call.position.unwrap_or(0),
    })
}

impl CudaContext {
    /// F64 ordered head RMSNorm -> shared BF16 publication -> optional split-half
    /// F64 RoPE(theta10000, pos0+row) -> second shared BF16 publication.
    /// Invalid geometry/ownership/aliases refuse before any launch. Numeric errors
    /// latch shared mask8 without clearing existing status; finite scratch is published.
    pub(crate) fn flint_qk_producer(
        &self,
        stream: &DeviceStream,
        call: QkProducerCall<'_>,
    ) -> Result<(), EngError> {
        self.ensure_usable()?;
        let [q_bytes, k_bytes] = ffi::qk_geometry(call.rows, call.position)
            .ok_or_else(|| EngError::BadRequest("D128 Q/K geometry or position overflow".into()))?;
        let regions = [
            ("q", &call.q, q_bytes, false),
            ("k", &call.k, k_bytes, false),
            ("gamma q", &call.gamma[0], 512, false),
            ("gamma k", &call.gamma[1], 512, false),
            ("scratch q", &call.scratch[0], q_bytes, true),
            ("scratch k", &call.scratch[1], k_bytes, true),
            ("out q", &call.out[0], q_bytes, true),
            ("out k", &call.out[1], k_bytes, true),
            ("Q/K flag", &call.flag, 4, true),
        ];
        for (index, &(name, range, bytes, written)) in regions.iter().enumerate() {
            self.raw_range(range)?;
            require_fit(name, range, bytes, 4)?;
            for &(other_name, other, needed, other_written) in &regions[..index] {
                if written || other_written {
                    require_apart(
                        &Claim::of(name, range, bytes),
                        &Claim::of(other_name, other, needed),
                    )?;
                }
            }
        }
        let norm = raw(self, &call, [call.q, call.k])?;
        let rope = call
            .position
            .map(|_| raw(self, &call, call.out))
            .transpose()?;
        let publish = [
            Bf16PublishRaw {
                x: self.raw_range(&call.scratch[0])?,
                y: self.raw_range(&call.out[0])?,
                flag: self.raw_range(&call.flag)?,
                rows: call.rows,
                columns: 4096,
            },
            Bf16PublishRaw {
                x: self.raw_range(&call.scratch[1])?,
                y: self.raw_range(&call.out[1])?,
                flag: self.raw_range(&call.flag)?,
                rows: call.rows,
                columns: 512,
            },
        ];
        self.launch_on(stream, |stream| {
            ffi::qk_launch(stream, &norm, false)?;
            for call in &publish {
                bf16_publish_launch(stream, call)?;
            }
            if let Some(rope) = &rope {
                ffi::qk_launch(stream, rope, true)?;
                for call in &publish {
                    bf16_publish_launch(stream, call)?;
                }
            }
            Ok(())
        })
    }
}

#[cfg(test)]
mod tests;
