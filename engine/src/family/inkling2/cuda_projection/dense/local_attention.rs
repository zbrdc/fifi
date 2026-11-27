//! Queue the same one-row GEMVs as the host path, targeting resident attention.

use super::{
    ControlledAttentionInput, ControllerWeights, DenseStep, ResidentProjection, Slot, bad,
};
use crate::{
    EngError,
    family::inkling2::cpu_forward::MIX_EPS,
    kernels::cuda::{
        CudaContext, DeviceRange, DeviceStream, PinnedBuf,
        resident::{
            inkling_decode_controller::{ApplyRole, GateRole, PAIR_RANK},
            inkling_stream::NormCall,
        },
    },
};

#[allow(clippy::large_enum_variant)]
pub(in crate::family::inkling2::cuda_projection) enum LocalInputs<'a> {
    Plain([&'a [f32]; 3]),
    Controlled(
        [ControlledAttentionInput<'a>; 3],
        [ControllerWeights<'a>; 3],
        f32,
    ),
}

pub(in crate::family::inkling2::cuda_projection) struct LocalProjection<'a> {
    pub dense: &'a DenseStep,
    pub inputs: LocalInputs<'a>,
    pub weights: [ResidentProjection<'a>; 6],
}

pub(in crate::family::inkling2::cuda_projection) struct LocalTargets<'a> {
    pub projections: [DeviceRange<'a>; 6],
    pub controller_health: Option<DeviceRange<'a>>,
    pub stream: &'a DeviceStream,
    pub transfer: &'a mut PinnedBuf,
    pub upload_at: usize,
}

impl DenseStep {
    pub(in crate::family::inkling2::cuda_projection) fn local_inputs_fit(
        &self,
        hidden: usize,
        controlled: bool,
    ) -> bool {
        (1..=4096).contains(&hidden)
            && self
                .range(Slot::Output, 0, if controlled { 24 } else { 3 } * hidden)
                .is_ok()
    }

    pub(in crate::family::inkling2::cuda_projection) fn enqueue_local(
        &self,
        context: &CudaContext,
        inputs: LocalInputs<'_>,
        weights: [ResidentProjection<'_>; 6],
        targets: LocalTargets<'_>,
    ) -> Result<(), EngError> {
        let (rows, hidden, control) = match inputs {
            LocalInputs::Plain(rows) => (rows.to_vec(), rows[0].len(), None),
            LocalInputs::Controlled(inputs, weights, eps) => {
                let hidden = inputs[0].input.len();
                if inputs.iter().any(|row| {
                    row.gate.len() != 2 * hidden
                        || [row.input, row.message, row.gamma].map(<[f32]>::len) != [hidden; 3]
                }) {
                    return Err(bad("invalid CUDA local controller input shape"));
                }
                let staged = inputs
                    .iter()
                    .map(|row| row.gate)
                    .chain(
                        inputs
                            .iter()
                            .flat_map(|row| [row.input, row.message, row.gamma]),
                    )
                    .collect();
                (
                    staged,
                    hidden,
                    Some((self.controlled_attention_calls(hidden, weights)?, eps)),
                )
            }
        };
        let (start, upload) = if control.is_some() { (21, 15) } else { (0, 3) };
        if !rows
            .iter()
            .flat_map(|row| row.iter())
            .all(|value| value.is_finite())
        {
            return Err(bad("CUDA local attention requires finite input"));
        }
        let packed: Vec<_> = rows
            .into_iter()
            .flatten()
            .map(|v| v.to_ne_bytes())
            .collect();
        targets
            .transfer
            .write(targets.upload_at, packed.as_flattened())?;
        context.copy_from_pinned(
            targets.stream,
            targets.transfer,
            targets.upload_at,
            self.range(Slot::Output, 0, upload * hidden)?,
        )?;
        if let Some((calls, eps)) = control {
            context.dense_gemv_group(targets.stream, &calls)?;
            context.inkling_controller_norm(
                targets.stream,
                self.range(Slot::Output, 6 * hidden, 18 * hidden)?,
                targets
                    .controller_health
                    .ok_or_else(|| bad("missing local controller health"))?,
                (hidden, eps),
            )?;
        }
        let calls = weights
            .into_iter()
            .zip(targets.projections)
            .map(|(p, y)| {
                let x = self.range(Slot::Output, (start + p.input) * hidden, hidden)?;
                self.call((&p.weight.codes, p.weight.shape, p.dtype), x, y)
            })
            .collect::<Result<Vec<_>, _>>()?;
        context.dense_gemv_group(targets.stream, &calls)
    }
}

/// Device rows of the cross-pass gates of one visit. Without a bank the host applies no
/// controller and the roles share the normed input (`attention_inputs_scalar`); the chain
/// norms `input` with `gamma` through an identity control whose zero gain and forget rows
/// come from `zeros` (`2 * hidden` words).
pub(in crate::family::inkling2::cuda_projection) enum ResidentLocalInputs<'a> {
    Plain {
        input: DeviceRange<'a>,
        gamma: DeviceRange<'a>,
        zeros: DeviceRange<'a>,
    },
    Controlled {
        input: DeviceRange<'a>,
        messages: [DeviceRange<'a>; 3],
        gammas: [DeviceRange<'a>; 3],
    },
}

/// One resident visit: its rows, the three controllers (a bank only), the norm eps and
/// the six attention projections.
pub(in crate::family::inkling2::cuda_projection) struct ResidentLocal<'a, const N: usize = 6> {
    pub inputs: ResidentLocalInputs<'a>,
    pub controllers: Option<[ControllerWeights<'a>; 3]>,
    pub eps: f32,
    pub weights: [ResidentProjection<'a>; N],
}

pub(in crate::family::inkling2::cuda_projection) struct ResidentLocalTargets<'a, const N: usize = 6>
{
    /// Q, raw K/V, relative, sinks, gates; global also projects dynamic K/V taps.
    pub projections: [DeviceRange<'a>; N],
    /// `roles` health words of the controller norm.
    pub controller_health: DeviceRange<'a>,
    /// The six gate half health words; `None` without a bank.
    pub gate_health: Option<DeviceRange<'a>>,
}

// Row offsets in the idle logits slot, in hidden-word rows.
const RAW_AT: usize = 6;
/// Where the controller GEMVs write the `[gain, forget]` pair of each of the three roles.
const PAIRS_AT: usize = RAW_AT + 3 * 3;
const NORMED_AT: usize = 21;
const PLAIN_NORMED_AT: usize = 5;
const GATE_HEALTH_WORDS: usize = 6;

/// The slot holds the gate rows, then the controller's `[gain, forget]` pairs and the normed
/// role rows.
struct ResidentLayout {
    roles: usize,
    normed_at: usize,
}

impl ResidentLayout {
    fn new(bank: bool) -> Self {
        let (roles, normed_at) = if bank {
            (3, NORMED_AT)
        } else {
            (1, PLAIN_NORMED_AT)
        };
        Self { roles, normed_at }
    }
}

fn row_words(range: &DeviceRange<'_>) -> Result<usize, EngError> {
    if !range.bytes().is_multiple_of(size_of::<f32>()) {
        return Err(bad("CUDA resident local attention rows hold whole floats"));
    }
    Ok(range.bytes() / size_of::<f32>())
}

fn require_rows(name: &str, rows: &[&DeviceRange<'_>], words: usize) -> Result<(), EngError> {
    if rows
        .iter()
        .map(|row| row_words(row))
        .collect::<Result<Vec<_>, _>>()?
        != vec![words; rows.len()]
    {
        return Err(bad(format!(
            "CUDA resident local attention {name} rows hold {words} words each"
        )));
    }
    Ok(())
}

impl ResidentLocalInputs<'_> {
    fn input(&self) -> &DeviceRange<'_> {
        match self {
            Self::Plain { input, .. } | Self::Controlled { input, .. } => input,
        }
    }
}

impl DenseStep {
    /// The device-resident form of [`Self::enqueue_local`]: the same controller and
    /// projection GEMVs reading rows already on the device, health into the visit's status
    /// words. Every extent is checked before the first launch.
    pub(in crate::family::inkling2::cuda_projection) fn enqueue_local_resident<'a>(
        &'a self,
        context: &CudaContext,
        stream: &DeviceStream,
        local: ResidentLocal<'a>,
        targets: ResidentLocalTargets<'a>,
    ) -> Result<(), EngError> {
        self.enqueue_roles_resident(context, stream, local, targets, None)
    }

    pub(in crate::family::inkling2::cuda_projection) fn enqueue_roles_resident<
        'a,
        const N: usize,
    >(
        &'a self,
        context: &CudaContext,
        stream: &DeviceStream,
        local: ResidentLocal<'a, N>,
        targets: ResidentLocalTargets<'a, N>,
        role_inputs: Option<[DeviceRange<'a>; 3]>,
    ) -> Result<(), EngError> {
        let ResidentLocal {
            inputs,
            controllers,
            eps,
            weights,
        } = local;
        let bank = matches!(inputs, ResidentLocalInputs::Controlled { .. });
        if controllers.is_some() != bank {
            return Err(bad("CUDA resident local attention controllers need a bank"));
        }
        let hidden = row_words(inputs.input())?;
        let layout = ResidentLayout::new(bank);
        let health = &targets.controller_health;
        if !(1..=4096).contains(&hidden)
            || row_words(health)? != layout.roles
            || !eps.is_finite()
            || eps <= 0.0
        {
            return Err(bad("invalid CUDA resident local attention input or health"));
        }
        for (projection, y) in weights.iter().zip(&targets.projections) {
            if projection.input >= 3 || row_words(y)? != projection.weight.shape[0] {
                return Err(bad("CUDA resident local attention projection rows differ"));
            }
        }
        // The rows of every controller role that the separate launches still norm: the
        // caller's triples, the gain and forget rows the GEMVs wrote (or the caller's zero
        // rows), and the normed output rows. The fused chain of the decode glue leaves none.
        let controller_rows: Vec<[DeviceRange<'_>; 6]> = match &inputs {
            ResidentLocalInputs::Plain {
                input,
                gamma,
                zeros,
            } => {
                require_rows("gamma", &[gamma], hidden)?;
                require_rows("zero", &[zeros], 2 * hidden)?;
                // One role with identity control: the caller's zero rows are the gain and
                // forget rows, so nothing is copied into the workspace.
                let out = self.range(Slot::Output, layout.normed_at * hidden, hidden)?;
                vec![[
                    *input,
                    *input,
                    *gamma,
                    zeros.slice(0, hidden * size_of::<f32>())?,
                    zeros.slice(hidden * size_of::<f32>(), hidden * size_of::<f32>())?,
                    out,
                ]]
            }
            ResidentLocalInputs::Controlled {
                input,
                messages,
                gammas,
            } => {
                let gate_health = targets
                    .gate_health
                    .ok_or_else(|| bad("missing CUDA resident local gate health"))?;
                require_rows("gate health", &[&gate_health], GATE_HEALTH_WORDS)?;
                for role in 0..3 {
                    require_rows("role", &[&messages[role], &gammas[role]], hidden)?;
                }
                let roles = role_inputs.unwrap_or([*input; 3]);
                require_rows("role input", &roles.each_ref(), hidden)?;
                let controllers = controllers.expect("bank checked");
                if self.fused_controllers(
                    context,
                    stream,
                    (&roles, *messages, *gammas, *health, gate_health),
                    controllers,
                    (&layout, hidden, eps),
                )? {
                    Vec::new()
                } else {
                    self.norm_gates(context, stream, (&roles, messages, gate_health), hidden)?;
                    self.separate_controllers(
                        context,
                        stream,
                        (roles, *messages, *gammas),
                        controllers,
                        (hidden, &layout),
                    )?
                }
            }
        };
        if !controller_rows.is_empty() {
            context.inkling_controller_norm_rows(
                stream,
                &controller_rows,
                *health,
                (hidden, eps),
            )?;
        }
        let calls = weights
            .into_iter()
            .zip(targets.projections)
            .map(|(p, y)| {
                let role = if bank { p.input } else { 0 };
                let x = self.range(Slot::Output, (layout.normed_at + role) * hidden, hidden)?;
                self.call((&p.weight.codes, p.weight.shape, p.dtype), x, y)
            })
            .collect::<Result<Vec<_>, _>>()?;
        context.dense_gemv_group(stream, &calls)
    }

    /// The controller chain of the three roles as its separate launches: the controller
    /// GEMVs write the gain and forget rows into the work slot; the norm reads them next to
    /// the caller's input, message and gamma rows.
    fn separate_controllers<'a>(
        &'a self,
        context: &CudaContext,
        stream: &DeviceStream,
        (roles, messages, gammas): (
            [DeviceRange<'a>; 3],
            [DeviceRange<'a>; 3],
            [DeviceRange<'a>; 3],
        ),
        controllers: [ControllerWeights<'_>; 3],
        (hidden, layout): (usize, &ResidentLayout),
    ) -> Result<Vec<[DeviceRange<'a>; 6]>, EngError> {
        let calls = self.controlled_attention_calls(hidden, controllers)?;
        context.dense_gemv_group(stream, &calls)?;
        let mut rows = Vec::with_capacity(3);
        for role in 0..3 {
            rows.push([
                roles[role],
                messages[role],
                gammas[role],
                self.range(Slot::Output, (PAIRS_AT + 2 * role) * hidden, hidden)?,
                self.range(Slot::Output, (PAIRS_AT + 1 + 2 * role) * hidden, hidden)?,
                self.range(Slot::Output, (layout.normed_at + role) * hidden, hidden)?,
            ]);
        }
        Ok(rows)
    }

    /// The controller chain of the three roles as the two launches of the decode glue
    /// (`FIFI_INKLING_CUDA_DECODE_GLUE`), the bits of `norm_gates`, `dense_gemv_group` and
    /// `inkling_controller_norm_rows` above: one `decode_controller_gate_down` over the three
    /// gate rows and one `decode_controller_pair_norm` writing the three normed rows. Each
    /// role's reduced row and the three tickets take the raw triple block of the slot, which
    /// the resident chain leaves unused. `false` keeps the separate launches, before any
    /// launch: a weight that is not F32 or not of rank [`PAIR_RANK`], or a slot too small for
    /// the rows.
    #[allow(clippy::type_complexity)]
    fn fused_controllers(
        &self,
        context: &CudaContext,
        stream: &DeviceStream,
        (roles, messages, gammas, health, gate_health): (
            &[DeviceRange<'_>; 3],
            [DeviceRange<'_>; 3],
            [DeviceRange<'_>; 3],
            DeviceRange<'_>,
            DeviceRange<'_>,
        ),
        controllers: [ControllerWeights<'_>; 3],
        (layout, hidden, eps): (&ResidentLayout, usize, f32),
    ) -> Result<bool, EngError> {
        // Rows of the slot are counted in floats.
        let reduced_at = RAW_AT * hidden;
        let tickets_at = (reduced_at + 3 * PAIR_RANK).next_multiple_of(4);
        let Ok(tickets) = self.range(Slot::Output, tickets_at, 3) else {
            return Ok(false);
        };
        let mut gates = Vec::with_capacity(3);
        let mut applies = Vec::with_capacity(3);
        for role in 0..3 {
            let reduced = match self.range(Slot::Output, reduced_at + role * PAIR_RANK, PAIR_RANK) {
                Ok(reduced) => reduced,
                Err(_) => return Ok(false),
            };
            // The `[input | message]` row `norm_gates` wrote, then the pair and output rows.
            let gate = self.range(Slot::Output, 2 * role * hidden, 2 * hidden)?;
            let gain = self.range(Slot::Output, (PAIRS_AT + 2 * role) * hidden, hidden)?;
            let forget = self.range(Slot::Output, (PAIRS_AT + 1 + 2 * role) * hidden, hidden)?;
            let out = self.range(Slot::Output, (layout.normed_at + role) * hidden, hidden)?;
            let calls = self.controller_calls(controllers[role], [gate, reduced, gain, forget])?;
            let Some(fused) = self.fused_controller(&calls, hidden) else {
                return Ok(false);
            };
            let word = size_of::<u32>();
            gates.push(GateRole {
                input: roles[role],
                message: messages[role],
                gate: fused.gate,
                down: fused.down,
                reduced: fused.reduced,
                health: [
                    gate_health.slice(2 * role * word, word)?,
                    gate_health.slice((2 * role + 1) * word, word)?,
                ],
            });
            applies.push(ApplyRole {
                input: roles[role],
                message: messages[role],
                gamma: gammas[role],
                reduced: fused.reduced,
                gain_w: fused.gain_w,
                forget_w: fused.forget_w,
                gain: fused.gain,
                forget: fused.forget,
                out,
                health: health.slice(role * word, word)?,
            });
        }
        context.decode_controller_gate_down(
            stream,
            &gates,
            tickets,
            [hidden, PAIR_RANK],
            MIX_EPS,
        )?;
        context.decode_controller_pair_norm(stream, &applies, tickets, hidden, eps)?;
        Ok(true)
    }

    /// The six gate halves: one `inkling_stream_norm_rows` launch norms the shared input and
    /// each message into `[input | message]` rows with eps `MIX_EPS`; health into `gate_health`.
    fn norm_gates(
        &self,
        context: &CudaContext,
        stream: &DeviceStream,
        (inputs, messages, health): (
            &[DeviceRange<'_>; 3],
            &[DeviceRange<'_>; 3],
            DeviceRange<'_>,
        ),
        hidden: usize,
    ) -> Result<(), EngError> {
        let mut rows = Vec::with_capacity(6);
        for role in 0..3 {
            for (half, src) in [&inputs[role], &messages[role]].into_iter().enumerate() {
                let at = (2 * role + half) * hidden;
                rows.push(NormCall {
                    src: *src,
                    dst: self.range(Slot::Output, at, hidden)?,
                    health: health.slice((2 * role + half) * size_of::<f32>(), size_of::<f32>())?,
                });
            }
        }
        context.inkling_stream_norm_rows(stream, &rows, (hidden, MIX_EPS))
    }
}
