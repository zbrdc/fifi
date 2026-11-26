//! One stream sequence for an ordered set of GGUF experts.

use super::DenseDtype;
use crate::{
    EngError,
    kernels::cuda::{
        CudaContext, DeviceRange, DeviceStream,
        context::{Claim, require_apart, require_fit},
        native::resident::dense_gemv::moe::{
            GgufMoePhase, GgufMoeRaw, GgufMoeRawPicks, GgufMoeRawWeights, GgufMoeSchedule,
            launch_gguf_moe, launch_gguf_moe_phase,
        },
        resident::{
            matvec_expert::ExpertMatvecCall,
            matvec_q::QuantType,
            quantize_q8_1::{QuantizeCall, q8_1_bytes},
        },
    },
};

pub(crate) use crate::kernels::cuda::native::resident::dense_gemv::moe::{
    GgufMoePick, GgufMoeShape,
};

pub(crate) enum GgufMoeWeights<'a> {
    Stacks([DeviceRange<'a>; 2]),
    /// Matrix ranges in the same order as the picks.
    Separate(&'a [[DeviceRange<'a>; 2]]),
}

pub(crate) enum GgufMoePicks<'a> {
    Host(&'a [GgufMoePick]),
    /// Rank-ordered i32 IDs and F32 weights, with F32 gate and down scales indexed by expert.
    Device {
        count: usize,
        ids: DeviceRange<'a>,
        weights: DeviceRange<'a>,
        scales: [DeviceRange<'a>; 2],
    },
}

impl GgufMoePicks<'_> {
    fn len(&self) -> usize {
        match self {
            Self::Host(picks) => picks.len(),
            Self::Device { count, .. } => *count,
        }
    }
}

pub(crate) struct GgufMoeCall<'a> {
    pub(crate) shape: GgufMoeShape,
    pub(crate) dtypes: [DenseDtype; 2],
    pub(crate) caps: [f32; 2],
    pub(crate) picks: GgufMoePicks<'a>,
    pub(crate) weights: GgufMoeWeights<'a>,
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
        self.gguf_moe_scheduled(stream, call, GgufMoeSchedule::default())
            .map(|_| ())
    }

    /// Requests optional resident expert schedules and reports what was queued.
    pub(crate) fn gguf_moe_scheduled(
        &self,
        stream: &DeviceStream,
        call: GgufMoeCall<'_>,
        schedule: GgufMoeSchedule,
    ) -> Result<GgufMoeSchedule, EngError> {
        self.ensure_usable()?;
        let raw = self.plan_gguf_moe(&call)?;
        if schedule.q8_window && self.gguf_moe_q8(stream, &call)? {
            return Ok(GgufMoeSchedule {
                q8_window: true,
                ..GgufMoeSchedule::default()
            });
        }
        let split = self.dense_group_policy().decode_split;
        self.launch_on(stream, |stream_raw| {
            launch_gguf_moe(stream_raw, raw, schedule, split)
        })
    }

    /// The q8_1 schedule of a checked `call`: the expert input quantized into the down rows,
    /// the gate and up products of every pick into the start of the scratch, the activation,
    /// the middle rows quantized into the dead products, the down products into the down rows,
    /// and the weighted sum. Returns false, with nothing queued, unless the picks are on the
    /// device over stacks and both products have a q8_1 case for their types and row lengths.
    fn gguf_moe_q8(&self, stream: &DeviceStream, c: &GgufMoeCall<'_>) -> Result<bool, EngError> {
        let (
            GgufMoePicks::Device { count, ids, .. },
            GgufMoeWeights::Stacks([gate, down]),
            [
                DenseDtype::PackedGguf(gate_type),
                DenseDtype::PackedGguf(down_type),
            ],
        ) = (&c.picks, &c.weights, c.dtypes)
        else {
            return Ok(false);
        };
        let (Some(gate_type), Some(down_type)) = (
            QuantType::from_gguf(gate_type),
            QuantType::from_gguf(down_type),
        ) else {
            return Ok(false);
        };
        let (
            count,
            GgufMoeShape {
                hidden,
                middle,
                experts,
            },
        ) = (*count, c.shape);
        let word = size_of::<f32>();
        let products = c.scratch.slice(0, count * 2 * middle * word)?;
        let middles = c.scratch.slice(products.bytes(), count * middle * word)?;
        let downs = c
            .scratch
            .slice(products.bytes() + middles.bytes(), count * hidden * word)?;
        let (Some(input_bytes), Some(middle_bytes)) =
            (q8_1_bytes(hidden, 1), q8_1_bytes(middle, count))
        else {
            return Ok(false);
        };
        if input_bytes > downs.bytes() || middle_bytes > products.bytes() {
            return Ok(false);
        }
        let input_q8 = downs.slice(0, input_bytes)?;
        let middle_q8 = products.slice(0, middle_bytes)?;
        let product = |ty, w_stack, x_q8, x_per_slot, y, [k, rows]: [usize; 2]| ExpertMatvecCall {
            ty,
            w_stack,
            ids: *ids,
            x_q8,
            x_per_slot,
            y,
            y_slot_stride: rows,
            k,
            rows,
            n_expert: experts,
            n_used: count,
            glu_gate: None,
        };
        let gate_up = product(
            gate_type,
            *gate,
            input_q8,
            false,
            products,
            [hidden, 2 * middle],
        );
        let down = product(down_type, *down, middle_q8, true, downs, [middle, hidden]);
        if !self.matvec_expert_takes(&gate_up) || !self.matvec_expert_takes(&down) {
            return Ok(false);
        }
        let phase = |phase| {
            let raw = self.plan_gguf_moe(c)?;
            self.launch_on(stream, |stream_raw| {
                launch_gguf_moe_phase(stream_raw, raw, phase)
            })
        };
        self.quantize_q8_1(
            stream,
            QuantizeCall {
                x: c.x,
                y: input_q8,
                k: hidden,
                ncols: 1,
            },
        )?;
        self.matvec_expert(stream, gate_up)?;
        phase(GgufMoePhase::Activate)?;
        self.quantize_q8_1(
            stream,
            QuantizeCall {
                x: middles,
                y: middle_q8,
                k: middle,
                ncols: count,
            },
        )?;
        self.matvec_expert(stream, down)?;
        phase(GgufMoePhase::Sum)?;
        Ok(true)
    }

    fn plan_gguf_moe<'a>(&self, c: &GgufMoeCall<'a>) -> Result<GgufMoeRaw<'a>, EngError> {
        let scratch = c.shape.scratch_bytes(c.picks.len()).ok_or_else(bad_group)?;
        let matrix_bytes = c.shape.matrix_bytes(c.dtypes).ok_or_else(bad_group)?;
        if !c.caps.into_iter().all(positive) {
            return Err(bad_group());
        }
        let row = c.shape.hidden * size_of::<f32>();
        require_fit("x", &c.x, row, 4)?;
        require_fit("scratch", &c.scratch, scratch, 4)?;
        require_fit("y", &c.y, row, 4)?;

        let temporary = Claim::of("scratch", &c.scratch, scratch);
        let output = Claim::of("y", &c.y, row);
        require_apart(&temporary, &output)?;
        let input = Claim::of("x", &c.x, row);
        require_apart(&temporary, &input)?;
        require_apart(&output, &input)?;
        let writes = [&temporary, &output];
        let picks = self.moe_picks(c, writes)?;
        let weights = self.moe_weights(c, matrix_bytes, writes)?;
        Ok(GgufMoeRaw {
            shape: c.shape,
            dtypes: c.dtypes,
            caps: c.caps,
            picks,
            weights,
            x: self.raw_range(&c.x)?,
            scratch: self.raw_range(&c.scratch)?,
            y: self.raw_range(&c.y)?,
        })
    }

    fn moe_picks<'a>(
        &self,
        call: &GgufMoeCall<'a>,
        writes: [&Claim<'_>; 2],
    ) -> Result<GgufMoeRawPicks<'a>, EngError> {
        match &call.picks {
            GgufMoePicks::Host(picks) => {
                if !picks.iter().all(|p| {
                    (p.expert as usize) < call.shape.experts
                        && positive(p.gate_scale)
                        && positive(p.down_scale)
                        && p.weight.is_finite()
                }) {
                    return Err(bad_group());
                }
                Ok(GgufMoeRawPicks::Host(picks))
            }
            GgufMoePicks::Device {
                count,
                ids,
                weights,
                scales,
            } => {
                if *count > call.shape.experts || !matches!(call.weights, GgufMoeWeights::Stacks(_))
                {
                    return Err(bad_group());
                }
                let picks_bytes = *count * size_of::<f32>();
                let scales_bytes = call.shape.experts * size_of::<f32>();
                for (name, range, bytes) in [
                    ("expert IDs", ids, picks_bytes),
                    ("route weights", weights, picks_bytes),
                    ("gate scales", &scales[0], scales_bytes),
                    ("down scales", &scales[1], scales_bytes),
                ] {
                    require_fit(name, range, bytes, 4)?;
                    for write in writes {
                        require_apart(write, &Claim::of(name, range, bytes))?;
                    }
                }
                Ok(GgufMoeRawPicks::Device {
                    count: *count,
                    ids: self.raw_range(ids)?,
                    weights: self.raw_range(weights)?,
                    scales: [self.raw_range(&scales[0])?, self.raw_range(&scales[1])?],
                })
            }
        }
    }

    fn moe_weights<'a>(
        &self,
        call: &GgufMoeCall<'a>,
        sizes: [usize; 2],
        writes: [&Claim<'_>; 2],
    ) -> Result<GgufMoeRawWeights<'a>, EngError> {
        match call.weights {
            GgufMoeWeights::Stacks(stacks) => {
                for (stack, size) in stacks.iter().zip(sizes) {
                    let bytes = size.checked_mul(call.shape.experts).ok_or_else(bad_group)?;
                    validate_weight(stack, bytes, "expert stack", writes)?;
                }
                Ok(GgufMoeRawWeights::Stacks([
                    self.raw_range(&stacks[0])?,
                    self.raw_range(&stacks[1])?,
                ]))
            }
            GgufMoeWeights::Separate(parts) => {
                if parts.len() != call.picks.len() {
                    return Err(bad_group());
                }
                let mut matrices = Vec::with_capacity(parts.len());
                for ranges in parts {
                    for (range, size) in ranges.iter().zip(sizes) {
                        validate_weight(range, size, "expert matrix", writes)?;
                    }
                    matrices.push([self.raw_range(&ranges[0])?, self.raw_range(&ranges[1])?]);
                }
                Ok(GgufMoeRawWeights::Separate(matrices))
            }
        }
    }
}

fn positive(value: f32) -> bool {
    value.is_finite() && value > 0.0
}

fn bad_group() -> EngError {
    EngError::BadRequest("invalid CUDA GGUF expert group".into())
}

fn validate_weight(
    range: &DeviceRange<'_>,
    bytes: usize,
    name: &'static str,
    writes: [&Claim<'_>; 2],
) -> Result<(), EngError> {
    require_fit(name, range, bytes, 2)?;
    for write in writes {
        require_apart(write, &Claim::of(name, range, bytes))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests;
