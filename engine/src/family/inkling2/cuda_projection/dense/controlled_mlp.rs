//! Keep a one-row MLP controller on the router/expert stream.

use std::mem::{align_of, size_of};

use super::{ControlledAttentionInput, ControllerWeights, DenseStep, FusedController, Slot, bad};
use crate::{
    EngError,
    family::inkling2::cpu_forward::MIX_EPS,
    kernels::cuda::{
        Claim, CudaContext, DeviceRange, DeviceStream, require_apart, require_fit,
        resident::{
            dense_gemv::DenseGemvCall,
            inkling_decode_controller::{ApplyRole, GateRole, PAIR_RANK},
            inkling_stream::NormCall,
        },
    },
};

impl DenseStep {
    pub(in crate::family::inkling2::cuda_projection) fn run_controlled_routed_layer(
        &self,
        context: &CudaContext,
        input: ControlledAttentionInput<'_>,
        bias: &[f32],
        (weights, eps): (ControllerWeights<'_>, f32),
        enqueue: impl FnOnce(&DeviceStream, [DeviceRange<'_>; 4]) -> Result<(), EngError>,
    ) -> Result<Option<Vec<f32>>, EngError> {
        let width = input.input.len();
        if !(1..=4096).contains(&width) || !width.is_multiple_of(4) {
            return Ok(None);
        }
        let status_at = bias.len();
        // The router loads its input as float4. Keep every activation row aligned.
        let gate_at = (status_at + 1).next_multiple_of(4);
        let work_at = gate_at + 2 * width;
        let health_at = work_at + 6 * width;
        let Ok(readback) = self.range(Slot::Output, health_at, width + 1) else {
            return Ok(None);
        };
        if input.gate.len() != 2 * width
            || input.message.len() != width
            || input.gamma.len() != width
        {
            return Err(bad("invalid CUDA controlled MLP input shape"));
        }
        let upload = self.range(Slot::Output, 0, work_at + 3 * width)?;
        let norm = self.range(Slot::Output, work_at, 6 * width)?;
        let health = self.range(Slot::Output, health_at, 1)?;
        let bias_range = self.range(Slot::Output, 0, bias.len())?;
        let status = self.range(Slot::Output, status_at, 1)?;
        let x = self.range(Slot::Output, work_at + 5 * width, width)?;
        let y = self.range(Slot::Output, health_at + 1, width)?;
        let reduced = self.range(Slot::Reduced, 0, weights[0].0.shape[0])?;
        let calls = self.controller_calls(
            weights,
            [
                self.range(Slot::Output, gate_at, 2 * width)?,
                reduced,
                self.range(Slot::Output, work_at + 3 * width, width)?,
                self.range(Slot::Output, work_at + 4 * width, width)?,
            ],
        )?;
        let zeros = [0.0; 4];
        let rows = [
            bias,
            &zeros[..gate_at - status_at],
            input.gate,
            input.input,
            input.message,
            input.gamma,
        ];
        let mut output = self.run_rows(context, &self.stream, &rows, [upload, readback], || {
            context.dense_gemv_group(&self.stream, &calls)?;
            context.inkling_controller_norm_roles(&self.stream, norm, health, ([width, 1], eps))?;
            enqueue(&self.stream, [x, bias_range, status, y])
        })?;
        // run_rows validated both the controller's sentinel and the expert result.
        output.copy_within(1.., 0);
        output.truncate(width);
        Ok(Some(output))
    }
}

/// The device rows of one enqueue-only controlled routed layer. Every row lives in an
/// allocation other than this step's arena (the copies refuse two ranges of one allocation)
/// and holds exactly the floats of its role: `input`, `message` and `gamma` are the
/// controller's raw rows (`gamma` is the MLP norm weight), `bias` the router bias, `out`
/// receives the expert sum, `health` is one controller float, `gate_health` is one sticky
/// float for both gate halves and `status` is one router word.
pub(in crate::family::inkling2::cuda_projection) struct RoutedRows<'a> {
    pub input: DeviceRange<'a>,
    pub message: DeviceRange<'a>,
    pub gamma: DeviceRange<'a>,
    pub bias: DeviceRange<'a>,
    pub out: DeviceRange<'a>,
    pub health: DeviceRange<'a>,
    pub gate_health: DeviceRange<'a>,
    pub status: DeviceRange<'a>,
}

/// The `Slot::Output` layout of both entries: the bias, the status word and its float4
/// padding, the two gate halves, then the controller workspace
/// (`[input, message, gamma | gain, forget | normalized row]`, `width` floats each).
#[derive(Clone, Copy)]
struct Layout {
    experts: usize,
    width: usize,
    gate_at: usize,
    work_at: usize,
}

impl Layout {
    fn new(experts: usize, width: usize) -> Result<Self, EngError> {
        if !(1..=4096).contains(&width) || !width.is_multiple_of(4) || experts == 0 {
            return Err(bad(format!(
                "invalid CUDA controlled MLP shape: {experts} experts, width {width}"
            )));
        }
        let gate_at = (experts + 1).next_multiple_of(4);
        Ok(Self {
            experts,
            width,
            gate_at,
            work_at: gate_at + 2 * width,
        })
    }
}

/// The arena ranges of the layout and the controller GEMVs over them.
struct Workspace<'a> {
    width: usize,
    gate: DeviceRange<'a>,
    norm: DeviceRange<'a>,
    bias: DeviceRange<'a>,
    x: DeviceRange<'a>,
    calls: [DenseGemvCall<'a>; 3],
    /// One u32 the fused chain of the decode glue counts its blocks with; the work block's
    /// first word, which the chain does not otherwise use.
    tickets: DeviceRange<'a>,
}

/// Every caller row holds exactly the floats of its role, 4-aligned: the copies take equal
/// byte ranges. The written rows share no byte with each other or with a read row.
fn check_rows(rows: &RoutedRows<'_>, layout: &Layout) -> Result<(), EngError> {
    let Layout { experts, width, .. } = *layout;
    let [row, bias, word] = [
        width * size_of::<f32>(),
        experts * size_of::<f32>(),
        size_of::<f32>(),
    ];
    for (name, range, need) in [
        ("controlled MLP input", &rows.input, row),
        ("controlled MLP message", &rows.message, row),
        ("controlled MLP gamma", &rows.gamma, row),
        ("controlled MLP router bias", &rows.bias, bias),
        ("controlled MLP output", &rows.out, row),
        ("controlled MLP health", &rows.health, word),
        ("controlled MLP gate health", &rows.gate_health, word),
        ("controlled MLP status", &rows.status, word),
    ] {
        require_fit(name, range, need, align_of::<f32>())?;
        if range.bytes() != need {
            return Err(EngError::BadRequest(format!(
                "the CUDA {name} range holds {} bytes but the call needs exactly {need}",
                range.bytes()
            )));
        }
    }
    let writes = [
        Claim::of("controlled MLP output", &rows.out, row),
        Claim::of("controlled MLP health", &rows.health, word),
        Claim::of("controlled MLP gate health", &rows.gate_health, word),
        Claim::of("controlled MLP status", &rows.status, word),
    ];
    let reads = [
        Claim::of("controlled MLP input", &rows.input, row),
        Claim::of("controlled MLP message", &rows.message, row),
        Claim::of("controlled MLP gamma", &rows.gamma, row),
        Claim::of("controlled MLP router bias", &rows.bias, bias),
    ];
    for (index, write) in writes.iter().enumerate() {
        for other in writes.iter().skip(index + 1) {
            require_apart(write, other)?;
        }
        for read in &reads {
            require_apart(write, read)?;
        }
    }
    Ok(())
}

impl DenseStep {
    fn routed_workspace<'a>(
        &'a self,
        layout: &Layout,
        weights: ControllerWeights<'a>,
    ) -> Result<Workspace<'a>, EngError> {
        let Layout {
            experts,
            width,
            gate_at,
            work_at,
        } = *layout;
        let gate = self.range(Slot::Output, gate_at, 2 * width)?;
        let norm = self.range(Slot::Output, work_at, 6 * width)?;
        let calls = self.controller_calls(
            weights,
            [
                gate,
                self.range(Slot::Reduced, 0, weights[0].0.shape[0])?,
                self.range(Slot::Output, work_at + 3 * width, width)?,
                self.range(Slot::Output, work_at + 4 * width, width)?,
            ],
        )?;
        Ok(Workspace {
            width,
            gate,
            norm,
            bias: self.range(Slot::Output, 0, experts)?,
            x: self.range(Slot::Output, work_at + 5 * width, width)?,
            calls,
            tickets: self.range(Slot::Output, work_at, 1)?,
        })
    }

    /// Whether [`Self::enqueue_controlled_routed_layer`] has room for `experts` router bias
    /// floats, `width` hidden floats and these controller weights in this step's arena.
    pub(in crate::family::inkling2::cuda_projection) fn controlled_routed_fits(
        &self,
        (experts, width): (usize, usize),
        weights: ControllerWeights<'_>,
    ) -> bool {
        Layout::new(experts, width)
            .is_ok_and(|layout| self.routed_workspace(&layout, weights).is_ok())
    }

    /// Enqueue-only form of [`Self::run_controlled_routed_layer`] for the resident decode
    /// token: no pinned buffer, no synchronize, no host finite check, and every launch on
    /// `stream`. The caller's raw rows and bias are copied into the arena, the two gate halves
    /// are normalized on the device with `MIX_EPS` (the host's eps for the gate halves) and
    /// the controller runs as in the host entry; the expert sum lands in `rows.out`.
    ///
    /// Gate arithmetic sets sticky NaN in `rows.gate_health`; the controller writes its
    /// own result to `rows.health` and the router writes `rows.status`.
    pub(in crate::family::inkling2::cuda_projection) fn enqueue_controlled_routed_layer(
        &self,
        context: &CudaContext,
        stream: &DeviceStream,
        (rows, (weights, eps), [experts, width]): (
            RoutedRows<'_>,
            (ControllerWeights<'_>, f32),
            [usize; 2],
        ),
        enqueue: impl FnOnce(&DeviceStream, [DeviceRange<'_>; 4]) -> Result<(), EngError>,
    ) -> Result<(), EngError> {
        if !eps.is_finite() || eps <= 0.0 {
            return Err(EngError::BadRequest(
                "invalid CUDA controlled MLP epsilon".into(),
            ));
        }
        // Use the loaded router geometry, not lengths from unvalidated caller rows.
        let layout = Layout::new(experts, width)?;
        check_rows(&rows, &layout)?;
        let work = self.routed_workspace(&layout, weights)?;
        self.launch_controlled(context, stream, (&rows, &work, eps), enqueue)
    }

    fn launch_controlled(
        &self,
        context: &CudaContext,
        stream: &DeviceStream,
        (rows, work, eps): (&RoutedRows<'_>, &Workspace<'_>, f32),
        enqueue: impl FnOnce(&DeviceStream, [DeviceRange<'_>; 4]) -> Result<(), EngError>,
    ) -> Result<(), EngError> {
        let row = rows.input.bytes();
        if let Some(fused) = self.fused_controller(&work.calls, work.width) {
            return self.launch_fused(context, stream, (rows, work, eps), fused, enqueue);
        }
        // The gate halves in one launch: the normalized input, then the normalized message.
        let gates = [
            (rows.input, work.gate.slice(0, row)?),
            (rows.message, work.gate.slice(row, row)?),
        ]
        .map(|(src, dst)| NormCall {
            src,
            dst,
            health: rows.gate_health,
        });
        context.inkling_stream_norm_rows(stream, &gates, (work.width, MIX_EPS))?;
        // The controller GEMVs write the gain and forget rows into the work block; the norm
        // reads them next to the caller's input, message and gamma rows where they live.
        let [gain, forget] = [3, 4].map(|i| work.norm.slice(i * row, row));
        let [gain, forget] = [gain?, forget?];
        context.copy_device(stream, rows.bias, work.bias)?;
        context.dense_gemv_group(stream, &work.calls)?;
        context.inkling_controller_norm_rows(
            stream,
            &[[rows.input, rows.message, rows.gamma, gain, forget, work.x]],
            rows.health,
            (work.width, eps),
        )?;
        enqueue(stream, [work.x, work.bias, rows.status, rows.out])
    }

    /// The chain of the decode glue (`FIFI_INKLING_CUDA_DECODE_GLUE`): the gate norms and the
    /// down reduction in one launch, both projections and the controller norm in another, the
    /// bits of the four launches above. The router reads the caller's bias row where it lives;
    /// nothing is copied into the arena.
    fn launch_fused(
        &self,
        context: &CudaContext,
        stream: &DeviceStream,
        (rows, work, eps): (&RoutedRows<'_>, &Workspace<'_>, f32),
        fused: FusedController<'_>,
        enqueue: impl FnOnce(&DeviceStream, [DeviceRange<'_>; 4]) -> Result<(), EngError>,
    ) -> Result<(), EngError> {
        let FusedController {
            gate,
            down,
            reduced,
            gain_w,
            gain,
            forget_w,
            forget,
        } = fused;
        let health = [rows.gate_health; 2];
        let gates = GateRole {
            input: rows.input,
            message: rows.message,
            gate,
            down,
            reduced,
            health,
        };
        context.decode_controller_gate_down(
            stream,
            std::slice::from_ref(&gates),
            work.tickets,
            [work.width, PAIR_RANK],
            MIX_EPS,
        )?;
        let apply = ApplyRole {
            input: rows.input,
            message: rows.message,
            gamma: rows.gamma,
            reduced,
            gain_w,
            forget_w,
            gain,
            forget,
            out: work.x,
            health: rows.health,
        };
        context.decode_controller_pair_norm(
            stream,
            std::slice::from_ref(&apply),
            work.tickets,
            work.width,
            eps,
        )?;
        enqueue(stream, [work.x, rows.bias, rows.status, rows.out])
    }
}

#[cfg(test)]
mod tests {
    use super::super::{DensePlan, Resident};
    use super::*;
    use crate::{
        family::inkling2::cuda_projection::upload_tensor,
        kernels::cuda::{
            device_present,
            resident::{
                dense_gemv::DenseDtype,
                router::{RouterLogitsCall, RouterShape, router_packed_logits_bytes},
            },
        },
    };

    #[test]
    fn mlp_chain_keeps_normalized_row_on_device_and_reports_invalid_values() {
        if !device_present("Inkling CUDA controlled MLP") {
            return;
        }
        let context = CudaContext::new().unwrap();
        let width = 32;
        let step = DenseStep::new(&context, &DensePlan::new([512, 64], width, 1).unwrap()).unwrap();
        let weight = |shape: [usize; 2]| Resident {
            shape,
            codes: upload_tensor(&context, &vec![0; shape.iter().product::<usize>() * 4]).unwrap(),
        };
        let [down, gain, forget] = [weight([1, 64]), weight([32, 1]), weight([32, 1])];
        let weights = [&down, &gain, &forget].map(|w| (w, DenseDtype::F32));
        let shape = RouterShape {
            m: 1,
            hidden: width,
            n_expert: 8,
            top_k: 2,
        };
        let router = weight([shape.n_expert, width]);
        let bytes = router_packed_logits_bytes(shape).unwrap();
        let logits = context.allocate(bytes).unwrap();
        let echo = context.allocate(width * size_of::<f32>()).unwrap();
        let run = |input: &[f32]| {
            step.run_controlled_routed_layer(
                &context,
                ControlledAttentionInput {
                    gate: &[1.0; 64],
                    input,
                    message: &[2.0; 32],
                    gamma: &[0.5; 32],
                },
                &[0.0; 8],
                (weights, 1e-5),
                |stream, [x, _, _, y]| {
                    context.router_logits_rows(
                        stream,
                        RouterLogitsCall {
                            shape,
                            x,
                            w: router.codes.range(0, router.codes.bytes())?,
                            logits: logits.range(0, bytes)?,
                        },
                    )?;
                    let echo = echo.range(0, echo.bytes())?;
                    context.copy_device(stream, x, echo)?;
                    context.copy_device(stream, echo, y)
                },
            )
        };
        let input = [2.0; 32];
        let output = run(&input).unwrap().unwrap();
        let expected = 1.0 / (4.0_f32 + 1e-5).sqrt();
        assert!(output.iter().all(|&v| (v - expected).abs() <= f32::EPSILON));
        assert!(run(&[f32::MAX; 32]).is_err());
        assert_eq!(run(&input).unwrap().unwrap(), output);
    }
}

#[cfg(test)]
mod enqueue_tests {
    use std::mem::size_of;

    use super::super::{DensePlan, Resident};
    use super::*;
    use crate::{
        family::inkling2::{cpu_forward::MIX_EPS, cuda_projection::upload_tensor},
        kernels::cuda::{
            DeviceBuffer, device_present,
            resident::{
                dense_gemv::DenseDtype,
                router::{RouterLogitsCall, RouterShape, router_packed_logits_bytes},
            },
        },
    };

    const EPS: f32 = 1e-5;
    const EXPERTS: usize = 8;
    const RANK: usize = 16;

    /// One enqueue-only controlled routed layer over synthetic F32 weights: the controller
    /// (`down` `[rank, 2w]`, `gain` and `forget` `[w, rank]`), an F32 router, a caller
    /// allocation `[input | message | gamma | out | health]` of `width` floats per row and a
    /// caller word allocation `[bias: experts floats | status]`.
    struct Fixture {
        context: CudaContext,
        step: DenseStep,
        weights: [Resident; 3],
        router: Resident,
        logits: DeviceBuffer,
        echo: DeviceBuffer,
        rows: DeviceBuffer,
        words: DeviceBuffer,
        width: usize,
    }

    fn bytes(values: &[f32]) -> Vec<u8> {
        values.iter().flat_map(|v| v.to_ne_bytes()).collect()
    }

    fn floats(bytes: &[u8]) -> Vec<f32> {
        bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_ne_bytes(*b))
            .collect()
    }

    /// The host's `rms_norm_with_scale`: the normalized copy of `row`.
    fn normalized(row: &[f32], eps: f32) -> Vec<f32> {
        let scale = (row.iter().map(|v| v * v).sum::<f32>() / row.len() as f32 + eps)
            .sqrt()
            .recip();
        row.iter().map(|v| v * scale).collect()
    }

    /// Router logits, then the normalized row echoed into the expert output.
    fn experts(
        context: &CudaContext,
        (router, logits, echo, shape): (&Resident, &DeviceBuffer, &DeviceBuffer, RouterShape),
        stream: &DeviceStream,
        [x, _, _, y]: [DeviceRange<'_>; 4],
    ) -> Result<(), EngError> {
        context.router_logits_rows(
            stream,
            RouterLogitsCall {
                shape,
                x,
                w: router.codes.range(0, router.codes.bytes())?,
                logits: logits.range(0, router_packed_logits_bytes(shape)?)?,
            },
        )?;
        let echo = echo.range(0, echo.bytes())?;
        context.copy_device(stream, x, echo)?;
        context.copy_device(stream, echo, y)
    }

    impl Fixture {
        fn new(width: usize) -> Self {
            let context = CudaContext::new().unwrap();
            let gate_at = (EXPERTS + 1).next_multiple_of(4);
            // Room for the host entry too, which also keeps health and output in the arena.
            let plan = DensePlan::new([gate_at + 9 * width + 8, 2 * width], width, RANK).unwrap();
            let step = DenseStep::new(&context, &plan).unwrap();
            let weight = |shape: [usize; 2], period: usize| Resident {
                shape,
                codes: upload_tensor(
                    &context,
                    &(0..shape.iter().product::<usize>())
                        .flat_map(|i| {
                            let value =
                                (((i * 7 + 1) % period * 2) as i32 - period as i32) as f32 / 256.0;
                            value.to_ne_bytes().to_vec()
                        })
                        .collect::<Vec<u8>>(),
                )
                .unwrap(),
            };
            let [down, gain, forget] = [
                weight([RANK, 2 * width], 9),
                weight([width, RANK], 7),
                weight([width, RANK], 5),
            ];
            let router = weight([EXPERTS, width], 11);
            let logits = context
                .allocate(router_packed_logits_bytes(Self::shape(width)).unwrap())
                .unwrap();
            let echo = context.allocate(width * size_of::<f32>()).unwrap();
            let rows = context.allocate(5 * width * size_of::<f32>()).unwrap();
            let words = context.allocate((EXPERTS + 1) * size_of::<f32>()).unwrap();
            Self {
                context,
                step,
                weights: [down, gain, forget],
                router,
                logits,
                echo,
                rows,
                words,
                width,
            }
        }

        fn shape(width: usize) -> RouterShape {
            RouterShape {
                m: 1,
                hidden: width,
                n_expert: EXPERTS,
                top_k: 2,
            }
        }

        fn controller(&self) -> ControllerWeights<'_> {
            self.weights.each_ref().map(|w| (w, DenseDtype::F32))
        }

        fn input(&self) -> Vec<f32> {
            (0..self.width)
                .map(|i| ((i * 3) % 17) as f32 * 0.1 - 0.8)
                .collect()
        }

        fn message(&self) -> Vec<f32> {
            (0..self.width)
                .map(|i| ((i * 5) % 23) as f32 * 0.07 - 0.5)
                .collect()
        }

        fn gamma(&self) -> Vec<f32> {
            (0..self.width)
                .map(|i| 0.5 + (i % 7) as f32 * 0.1)
                .collect()
        }

        fn bias(&self) -> Vec<f32> {
            (0..EXPERTS).map(|i| i as f32 * 0.25 - 1.0).collect()
        }

        /// The caller's rows: `[input | message | gamma | out: 7 | health: 7 | gate health: 7]` and
        /// `[bias | status: 0]`.
        fn seed(&self, input: &[f32], message: &[f32]) {
            let stream = self.context.stream().unwrap();
            let mut values = [input, message, &self.gamma()].concat();
            values.extend(std::iter::repeat_n(7.0, 2 * self.width));
            self.rows.write_at(&stream, 0, &bytes(&values)).unwrap();
            self.words
                .write_at(&stream, 0, &bytes(&self.bias()))
                .unwrap();
            self.words
                .write_at(&stream, EXPERTS * size_of::<f32>(), &[0; 4])
                .unwrap();
        }

        fn range(&self, row: usize, floats: usize) -> DeviceRange<'_> {
            self.rows
                .range(
                    row * self.width * size_of::<f32>(),
                    floats * size_of::<f32>(),
                )
                .unwrap()
        }

        fn routed_rows(&self) -> RoutedRows<'_> {
            RoutedRows {
                input: self.range(0, self.width),
                message: self.range(1, self.width),
                gamma: self.range(2, self.width),
                bias: self.words.range(0, EXPERTS * size_of::<f32>()).unwrap(),
                out: self.range(3, self.width),
                health: self.range(4, 1),
                gate_health: self.range(4, 2).slice(4, 4).unwrap(),
                status: self.words.range(EXPERTS * size_of::<f32>(), 4).unwrap(),
            }
        }

        fn chain(
            &self,
            context: &CudaContext,
            stream: &DeviceStream,
            ranges: [DeviceRange<'_>; 4],
        ) -> Result<(), EngError> {
            experts(
                context,
                (
                    &self.router,
                    &self.logits,
                    &self.echo,
                    Self::shape(self.width),
                ),
                stream,
                ranges,
            )
        }

        /// The oracle: the host entry on the same rows, with the host's gate halves.
        fn host_entry(&self, input: &[f32], message: &[f32]) -> Vec<f32> {
            let gate = [normalized(input, MIX_EPS), normalized(message, MIX_EPS)].concat();
            self.step
                .run_controlled_routed_layer(
                    &self.context,
                    ControlledAttentionInput {
                        gate: &gate,
                        input,
                        message,
                        gamma: &self.gamma(),
                    },
                    &self.bias(),
                    (self.controller(), EPS),
                    |stream, ranges| self.chain(&self.context, stream, ranges),
                )
                .unwrap()
                .unwrap()
        }

        fn enqueue(&self, rows: RoutedRows<'_>) -> Result<(), EngError> {
            let stream = self.context.stream().unwrap();
            self.step.enqueue_controlled_routed_layer(
                &self.context,
                &stream,
                (rows, (self.controller(), EPS), [EXPERTS, self.width]),
                |stream, ranges| self.chain(&self.context, stream, ranges),
            )
        }

        /// The caller's output row and both health words, read back behind the stream.
        fn read_out(&self) -> (Vec<f32>, [f32; 2]) {
            let stream = self.context.stream().unwrap();
            let mut out = vec![0u8; (self.width + 2) * size_of::<f32>()];
            self.rows
                .read_at(&stream, 3 * self.width * size_of::<f32>(), &mut out)
                .unwrap();
            let health = out.split_off(self.width * size_of::<f32>());
            (floats(&out), floats(&health).try_into().unwrap())
        }
    }

    #[test]
    fn enqueue_only_chain_matches_the_host_entry_and_lands_in_the_callers_row() {
        if !device_present("Inkling CUDA enqueue-only controlled MLP") {
            return;
        }
        for width in [2304, 32] {
            let fixture = Fixture::new(width);
            let (input, message) = (fixture.input(), fixture.message());
            fixture.seed(&input, &message);
            let expected = fixture.host_entry(&input, &message);
            fixture
                .enqueue(fixture.routed_rows())
                .unwrap_or_else(|e| panic!("width {width}: {e:?}"));
            let (out, health) = fixture.read_out();
            assert_eq!(health, [0.0, 7.0], "width {width}: health");
            assert_eq!(out.len(), width);
            for (i, (a, b)) in out.iter().zip(&expected).enumerate() {
                assert!(
                    (a - b).abs() <= 1e-4,
                    "width {width}: out[{i}] = {a}, host entry {b}"
                );
            }
            assert_ne!(out.first(), Some(&7.0), "width {width}: the sentinel row");
            assert!(
                fixture
                    .step
                    .controlled_routed_fits((EXPERTS, width), fixture.controller()),
                "width {width}: the chain fits"
            );
        }
    }

    #[test]
    fn enqueue_only_chain_reports_a_nan_input_through_health_and_output() {
        if !device_present("Inkling CUDA enqueue-only controlled MLP NaN") {
            return;
        }
        let fixture = Fixture::new(32);
        let input = vec![f32::NAN; 32];
        fixture.seed(&input, &fixture.message());
        fixture.enqueue(fixture.routed_rows()).unwrap();
        let (out, health) = fixture.read_out();
        assert!(out.iter().all(|v| v.is_nan()), "out {out:?}");
        assert!(health.iter().all(|v| v.is_nan()), "health {health:?}");
    }

    #[test]
    fn enqueue_only_gate_overflow_survives_a_successful_controller() {
        if !device_present("Inkling CUDA enqueue-only controlled MLP gate overflow") {
            return;
        }
        let fixture = Fixture::new(32);
        // The message RMS overflows and its scale is zero. Both normalized gate halves
        // are then zero, so the controller coefficients and its output all stay finite.
        fixture.seed(&[0.0; 32], &[1e20; 32]);
        fixture.enqueue(fixture.routed_rows()).unwrap();
        let (out, [controller, gate]) = fixture.read_out();
        assert_eq!(controller, 0.0);
        assert!(out.iter().all(|v| v.is_finite()));
        assert!(gate.is_nan());
    }

    #[test]
    fn enqueue_only_chain_rejects_before_any_launch() {
        if !device_present("Inkling CUDA enqueue-only controlled MLP rejections") {
            return;
        }
        let fixture = Fixture::new(32);
        fixture.seed(&fixture.input(), &fixture.message());
        let refused = |fixture: &Fixture, rows: RoutedRows<'_>| {
            let error = fixture.enqueue(rows).unwrap_err();
            assert!(matches!(error, EngError::BadRequest(_)), "{error:?}");
            let (out, health) = fixture.read_out();
            assert!(
                out.iter().all(|&v| v == 7.0) && health == [7.0; 2],
                "a rejected call wrote {out:?} / {health:?}"
            );
        };
        let mut rows = fixture.routed_rows();
        rows.input = fixture.range(0, 31);
        refused(&fixture, rows);
        let mut rows = fixture.routed_rows();
        rows.bias = fixture
            .words
            .range(0, (EXPERTS - 1) * size_of::<f32>())
            .unwrap();
        refused(&fixture, rows);
        let mut rows = fixture.routed_rows();
        rows.health = fixture.range(4, 2);
        refused(&fixture, rows);
        let mut rows = fixture.routed_rows();
        rows.gate_health = rows.health;
        refused(&fixture, rows);
        let mut rows = fixture.routed_rows();
        rows.out = fixture.range(3, 31);
        refused(&fixture, rows);
        // `out` over `health`: the expert kernel would overwrite the health word.
        let mut rows = fixture.routed_rows();
        rows.health = fixture.range(3, 1);
        refused(&fixture, rows);
        // The arena of a small step has no room for the gate halves and the workspace.
        let small = DenseStep::new(
            &fixture.context,
            &DensePlan::new([64, 64], 32, RANK).unwrap(),
        )
        .unwrap();
        assert!(!small.controlled_routed_fits((EXPERTS, 32), fixture.controller()));
        assert!(
            !fixture
                .step
                .controlled_routed_fits((EXPERTS, 2305), fixture.controller()),
            "a width that is not a multiple of 4 never fits"
        );
        let stream = fixture.context.stream().unwrap();
        let error = small
            .enqueue_controlled_routed_layer(
                &fixture.context,
                &stream,
                (
                    fixture.routed_rows(),
                    (fixture.controller(), EPS),
                    [EXPERTS, 32],
                ),
                |_, _| panic!("an undersized arena must not invoke the router"),
            )
            .unwrap_err();
        assert!(
            matches!(&error, EngError::Checkpoint(message)
            if message.contains("projection exceeds its arena slot")),
            "{error:?}"
        );
        assert_eq!(fixture.read_out(), (vec![7.0; 32], [7.0; 2]));
    }
}
