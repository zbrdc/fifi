// The original resident path; the decode-glue switch leaves this path unchanged.
use super::*;

impl DenseStep {
    pub(in crate::family::inkling2::cuda_projection) fn enqueue_roles_baseline<
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
        let layout = ResidentLayout::new(hidden, bank);
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
        match &inputs {
            ResidentLocalInputs::Plain {
                input,
                gamma,
                zeros,
            } => {
                require_rows("gamma", &[gamma], hidden)?;
                require_rows("zero", &[zeros], 2 * hidden)?;
                self.copy_rows(
                    context,
                    stream,
                    &[(input, 0), (input, 1), (gamma, 2)],
                    hidden,
                )?;
                context.copy_device(
                    stream,
                    *zeros,
                    self.range(Slot::Output, 3 * hidden, 2 * hidden)?,
                )?;
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
                self.norm_gates_baseline(context, stream, (&roles, messages, gate_health), hidden)?;
                for role in 0..3 {
                    let at = RAW_AT + 3 * role;
                    self.copy_rows(
                        context,
                        stream,
                        &[
                            (&roles[role], at),
                            (&messages[role], at + 1),
                            (&gammas[role], at + 2),
                        ],
                        hidden,
                    )?;
                }
                let calls =
                    self.controlled_attention_calls(hidden, controllers.expect("bank checked"))?;
                context.dense_gemv_group(stream, &calls)?;
            }
        }
        let workspace = self.range(
            Slot::Output,
            layout.workspace_at,
            CONTROLLER_ROWS * layout.roles * hidden,
        )?;
        context.inkling_controller_norm_roles(
            stream,
            workspace,
            *health,
            ([hidden, layout.roles], eps),
        )?;
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

    pub(super) fn copy_rows(
        &self,
        context: &CudaContext,
        stream: &DeviceStream,
        rows: &[(&DeviceRange<'_>, usize)],
        hidden: usize,
    ) -> Result<(), EngError> {
        for (source, row) in rows {
            context.copy_device(
                stream,
                **source,
                self.range(Slot::Output, row * hidden, hidden)?,
            )?;
        }
        Ok(())
    }

    pub(super) fn norm_gates_baseline(
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
        for role in 0..3 {
            for (half, src) in [&inputs[role], &messages[role]].into_iter().enumerate() {
                let at = (2 * role + half) * hidden;
                let dst = self.range(Slot::Output, at, hidden)?;
                let health =
                    health.slice((2 * role + half) * size_of::<f32>(), size_of::<f32>())?;
                context.inkling_stream_norm(
                    stream,
                    NormCall {
                        src: *src,
                        dst,
                        health,
                    },
                    (hidden, MIX_EPS),
                )?;
            }
        }
        Ok(())
    }
}
