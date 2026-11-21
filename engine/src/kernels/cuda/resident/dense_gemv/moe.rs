//! One stream sequence for an ordered set of resident GGUF experts.

use super::DenseDtype;
use crate::{
    EngError,
    kernels::cuda::{
        CudaContext, DeviceRange, DeviceStream,
        context::{Claim, require_apart, require_fit},
        native::resident::dense_gemv::moe::{GgufMoeRaw, launch_gguf_moe},
    },
};

pub(crate) use crate::kernels::cuda::native::resident::dense_gemv::moe::{
    GgufMoePick, GgufMoeShape,
};

pub(crate) struct GgufMoeCall<'a> {
    pub(crate) shape: GgufMoeShape,
    pub(crate) dtypes: [DenseDtype; 2],
    pub(crate) caps: [f32; 2],
    pub(crate) picks: &'a [GgufMoePick],
    pub(crate) stacks: [DeviceRange<'a>; 2],
    pub(crate) x: DeviceRange<'a>,
    pub(crate) scratch: DeviceRange<'a>,
    pub(crate) y: DeviceRange<'a>,
}

impl CudaContext {
    pub(crate) fn gguf_moe(
        &self,
        stream: &DeviceStream,
        call: GgufMoeCall<'_>,
    ) -> Result<(), EngError> {
        self.ensure_usable()?;
        let raw = self.plan_gguf_moe(&call)?;
        self.launch_on(stream, |stream_raw| launch_gguf_moe(stream_raw, raw))
    }

    fn plan_gguf_moe<'a>(&self, c: &GgufMoeCall<'a>) -> Result<GgufMoeRaw<'a>, EngError> {
        let bad = || EngError::BadRequest("invalid CUDA GGUF expert group".into());
        let scratch = c.shape.scratch_bytes(c.picks.len()).ok_or_else(bad)?;
        let stacks = c.shape.stack_bytes(c.dtypes).ok_or_else(bad)?;
        let positive = |v: f32| v.is_finite() && v > 0.0;
        if !c.caps.into_iter().all(positive)
            || !c.picks.iter().all(|p| {
                (p.expert as usize) < c.shape.experts
                    && positive(p.gate_scale)
                    && positive(p.down_scale)
                    && p.weight.is_finite()
            })
        {
            return Err(bad());
        }
        let row = c.shape.hidden * size_of::<f32>();
        require_fit("gate stack", &c.stacks[0], stacks[0], 2)?;
        require_fit("down stack", &c.stacks[1], stacks[1], 2)?;
        require_fit("x", &c.x, row, 4)?;
        require_fit("scratch", &c.scratch, scratch, 4)?;
        require_fit("y", &c.y, row, 4)?;

        let temporary = Claim::of("scratch", &c.scratch, scratch);
        let output = Claim::of("y", &c.y, row);
        require_apart(&temporary, &output)?;
        for source in [
            Claim::of("gate stack", &c.stacks[0], stacks[0]),
            Claim::of("down stack", &c.stacks[1], stacks[1]),
            Claim::of("x", &c.x, row),
        ] {
            require_apart(&temporary, &source)?;
            require_apart(&output, &source)?;
        }
        Ok(GgufMoeRaw {
            shape: c.shape,
            dtypes: c.dtypes,
            caps: c.caps,
            picks: c.picks,
            stacks: [self.raw_range(&c.stacks[0])?, self.raw_range(&c.stacks[1])?],
            x: self.raw_range(&c.x)?,
            scratch: self.raw_range(&c.scratch)?,
            y: self.raw_range(&c.y)?,
        })
    }
}
