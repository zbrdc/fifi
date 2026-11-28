//! The resident chain against the host-entry chain on synthetic weights.

use std::{collections::VecDeque, mem::size_of};

use super::*;
use crate::{
    family::inkling2::{
        Attention, LayerVisit,
        cpu_forward::MIX_EPS,
        cuda_projection::{
            Resident,
            dense::{
                DensePlan, DenseStep, ResidentLocal, ResidentLocalInputs, ResidentLocalTargets,
                ResidentProjection,
            },
            gguf::{
                ControlledAttentionInput, ConvRows, PrepWeights, resident_local::request_rows,
                resident_local::validate_rows,
            },
            upload_tensor,
        },
    },
    kernels::cuda::{
        DeviceBuffer, device_present, resident::relative_attention::RelativeAttentionShape,
    },
};

const WORD: usize = size_of::<f32>();

#[derive(Clone, Copy)]
struct Geometry {
    hidden: usize,
    query_heads: usize,
    kv_heads: usize,
    dims: usize,
    rank: usize,
}

impl Geometry {
    fn shape(self) -> RelativeAttentionShape {
        RelativeAttentionShape {
            query_heads: self.query_heads,
            kv_heads: self.kv_heads,
            relative_dims: self.dims,
            relative_extent: 2,
            capacity: 4,
        }
    }

    fn local(self) -> LocalShape {
        LocalShape {
            query_heads: self.query_heads,
            kv_heads: self.kv_heads,
            relative_dims: self.dims,
        }
    }

    fn q(self) -> usize {
        self.query_heads * HEAD_DIM
    }

    fn kv(self) -> usize {
        self.kv_heads * HEAD_DIM
    }
}

/// Synthetic FP32 weights of every matrix the chain reads, filled by a small LCG.
struct Weights {
    wq: Resident,
    wk: Resident,
    wv: Resident,
    wr: Resident,
    sinks: Resident,
    gates: Resident,
    wo: Resident,
    controllers: [[Resident; 3]; 3],
}

fn matrix(context: &CudaContext, [rows, cols]: [usize; 2], seed: u32) -> Resident {
    let mut state = seed | 1;
    let values: Vec<f32> = (0..rows * cols)
        .map(|_| {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            ((state >> 8) as f32 / 8388608.0 - 1.0) * 0.05
        })
        .collect();
    let bytes: Vec<_> = values.iter().flat_map(|v| v.to_ne_bytes()).collect();
    Resident {
        codes: upload_tensor(context, &bytes).unwrap(),
        shape: [rows, cols],
    }
}

fn weights(context: &CudaContext, g: Geometry) -> Weights {
    let [h, q, kv] = [g.hidden, g.q(), g.kv()];
    let m = |shape, seed| matrix(context, shape, seed);
    Weights {
        wq: m([q, h], 3),
        wk: m([kv, h], 5),
        wv: m([kv, h], 7),
        wr: m([g.query_heads * g.dims, h], 11),
        sinks: m([g.query_heads, h], 13),
        gates: m([q, h], 17),
        wo: m([h, q], 19),
        controllers: [0, 1, 2].map(|role| {
            [
                m([g.rank, 2 * h], 23 + 3 * role),
                m([h, g.rank], 24 + 3 * role),
                m([h, g.rank], 25 + 3 * role),
            ]
        }),
    }
}

fn resident_projection(weight: &Resident, input: usize) -> ResidentProjection<'_> {
    ResidentProjection {
        weight,
        dtype: DenseDtype::F32,
        input,
    }
}

/// One device row of `words` floats, uploadable and readable back.
fn device_row(context: &CudaContext, values: &[f32]) -> DeviceBuffer {
    let bytes: Vec<_> = values.iter().flat_map(|v| v.to_ne_bytes()).collect();
    upload_tensor(context, &bytes).unwrap()
}

fn read(buffer: &DeviceBuffer, stream: &DeviceStream, words: usize) -> Vec<f32> {
    let mut bytes = vec![0; words * WORD];
    buffer.read_at(stream, 0, &mut bytes).unwrap();
    bytes
        .as_chunks::<WORD>()
        .0
        .iter()
        .map(|chunk| f32::from_ne_bytes(*chunk))
        .collect()
}

fn lcg(seed: u32) -> impl FnMut(usize) -> Vec<f32> {
    let mut state = seed | 1;
    move |words| {
        (0..words)
            .map(|_| {
                state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                (state >> 8) as f32 / 8388608.0 - 1.0
            })
            .collect()
    }
}

fn rms_norm(row: &mut [f32], eps: f32) {
    let scale = (row.iter().map(|x| x * x).sum::<f32>() / row.len() as f32 + eps)
        .sqrt()
        .recip();
    for value in row {
        *value *= scale;
    }
}

/// `CausalConv::apply` with a static kernel: tap `taps - 1 - lag` per channel, the
/// residual, the `1 / sqrt(taps + 1)` norm, then `push_front` and `truncate(taps - 1)`.
fn causal_conv(
    history: &mut VecDeque<Vec<f32>>,
    input: Vec<f32>,
    kernel: &[f32],
    taps: usize,
) -> Vec<f32> {
    let width = input.len();
    let mut output = vec![0.0; width];
    for (lag, source) in std::iter::once(&input).chain(history.iter()).enumerate() {
        for (channel, (result, &x)) in output.iter_mut().zip(source).enumerate() {
            *result += kernel[channel * taps + taps - 1 - lag] * x;
        }
    }
    let norm = ((taps + 1) as f32).sqrt().recip();
    for (result, x) in output.iter_mut().zip(&input) {
        *result = (*result + x) * norm;
    }
    history.push_front(input);
    history.truncate(taps - 1);
    output
}

struct Harness {
    stream: DeviceStream,
    dense: DenseStep,
    weights: Weights,
    oracle: ResidentAttention,
    resident: ResidentAttention,
    math: PrepMath,
    // Constant rows, host copies first, then their device rows.
    gamma: Vec<f32>,
    scales: [Vec<f32>; 2],
    conv_kernels: [Vec<f32>; 2],
    attn_kernel: Vec<f32>,
    d_gamma: DeviceBuffer,
    d_scales: [DeviceBuffer; 2],
    d_conv_kernels: [DeviceBuffer; 2],
    d_attn_kernel: DeviceBuffer,
}

fn attention(context: &CudaContext, g: Geometry) -> ResidentAttention {
    let shape = g.shape();
    let [
        out,
        cache_bytes,
        _,
        status,
        q,
        k,
        v,
        rel,
        table_bytes,
        sinks,
        gates,
        pos,
    ] = shape.region_bytes().unwrap();
    let scratch_bytes = out + status + q + k + v + rel + sinks + gates + pos;
    Plan {
        visits: vec![Visit {
            shape,
            kv_format: crate::kernels::cuda::resident::relative_attention::RelativeKvFormat::F32,
            table_at: 0,
            table_bytes,
            cache_at: table_bytes,
            cache_bytes,
        }],
        tables: vec![0; table_bytes],
        scratch_bytes,
        local_workspace: Some(workspace(shape, Some(g.hidden), g.hidden)),
        projected_rows: Some(g.hidden),
    }
    .load(context, false)
    .unwrap()
    .unwrap()
}

/// Both chains start from the same host step at position 0, so both cursors are at 1.
fn seed(attention: &ResidentAttention, context: &CudaContext, g: Geometry) {
    let zero = vec![0.0; g.kv()];
    attention
        .step(
            context,
            0,
            Row {
                query: &vec![0.0; g.q()],
                key: &zero,
                value: &zero,
                relative: &vec![0.0; g.query_heads * g.dims],
                sinks: &vec![0.0; g.query_heads],
                gates: &vec![0.0; g.q()],
            },
            History {
                position: 0,
                keys: &VecDeque::new(),
                values: &VecDeque::new(),
            },
            None,
        )
        .unwrap();
}

impl Harness {
    fn new(context: &CudaContext, g: Geometry, seed_value: u32) -> Self {
        let stream = context.stream().unwrap();
        let plan = DensePlan::new([25 * g.hidden, 2 * g.hidden], g.hidden, g.rank).unwrap();
        let dense = DenseStep::new(context, &plan).unwrap();
        let weights = weights(context, g);
        let oracle = attention(context, g);
        let resident = attention(context, g);
        seed(&oracle, context, g);
        seed(&resident, context, g);
        let mut generate = lcg(seed_value);
        let gamma: Vec<f32> = (0..g.hidden).map(|i| 0.5 + 0.1 * (i % 7) as f32).collect();
        let scales = [generate(g.q()), generate(g.query_heads * g.dims)];
        let conv_kernels = [
            generate(g.kv() * PrepShape::TAPS),
            generate(g.kv() * PrepShape::TAPS),
        ];
        let attn_kernel = generate(g.hidden * PrepShape::TAPS);
        let d_scales = scales.each_ref().map(|values| device_row(context, values));
        let d_conv_kernels = conv_kernels
            .each_ref()
            .map(|values| device_row(context, values));
        let d_attn_kernel = device_row(context, &attn_kernel);
        let d_gamma = device_row(context, &gamma);
        let math = PrepMath {
            qk_limit: 0.5,
            rel_limit: 1.0,
            eps: 1e-5,
            log: None,
        };
        Self {
            stream,
            dense,
            weights,
            oracle,
            resident,
            math,
            gamma,
            scales,
            conv_kernels,
            attn_kernel,
            d_gamma,
            d_scales,
            d_conv_kernels,
            d_attn_kernel,
        }
    }
}

/// The host state one visit advances: raw K/V histories, the attention history, and the
/// caller's device buffers of the same rows plus the request's other rows.
struct HostVisit {
    position: usize,
    input: Vec<f32>,
    messages: Option<[Vec<f32>; 3]>,
    raw: [VecDeque<Vec<f32>>; 2],
    attn: VecDeque<Vec<f32>>,
}

/// Caller-owned device rows of one visit slot.
struct Rows {
    input: DeviceBuffer,
    messages: Option<[DeviceBuffer; 3]>,
    raw_history: [DeviceBuffer; 2],
    attn_history: DeviceBuffer,
    pos: DeviceBuffer,
    out: DeviceBuffer,
    log: DeviceBuffer,
    status: DeviceBuffer,
}

impl Harness {
    fn rows(&self, context: &CudaContext, g: Geometry, bank: bool) -> Rows {
        let past = PrepShape::TAPS - 1;
        let zeros = |words| device_row(context, &vec![0.0; words]);
        Rows {
            input: zeros(g.hidden),
            messages: bank.then(|| [0; 3].map(|_| zeros(g.hidden))),
            raw_history: [zeros(past * g.kv()), zeros(past * g.kv())],
            attn_history: zeros(past * g.hidden),
            pos: device_row(context, &[0.0]),
            out: zeros(g.hidden),
            log: zeros(4 * g.kv()),
            status: zeros(local_status(g.local(), bank).words()),
        }
    }

    /// Upload the visit's input rows and the zero-padded histories.
    fn stage(&self, context: &CudaContext, g: Geometry, visit: &HostVisit, rows: &Rows) {
        let write = |buffer: &DeviceBuffer, words: &[f32]| {
            let bytes: Vec<_> = words.iter().flat_map(|v| v.to_ne_bytes()).collect();
            context.upload_at(buffer, 0, &bytes).unwrap();
        };
        write(&rows.input, &visit.input);
        for (buffer, message) in rows
            .messages
            .iter()
            .flatten()
            .zip(visit.messages.iter().flatten())
        {
            write(buffer, message);
        }
        let past = PrepShape::TAPS - 1;
        for (plane, history) in visit.raw.iter().enumerate() {
            let mut words = Vec::new();
            for lag in 0..past {
                words.extend(
                    history
                        .get(lag)
                        .cloned()
                        .unwrap_or_else(|| vec![0.0; g.kv()]),
                );
            }
            write(&rows.raw_history[plane], &words);
        }
        let mut words = Vec::new();
        for lag in 0..past {
            words.extend(
                visit
                    .attn
                    .get(lag)
                    .cloned()
                    .unwrap_or_else(|| vec![0.0; g.hidden]),
            );
        }
        write(&rows.attn_history, &words);
        context
            .upload_at(
                &rows.pos,
                0,
                &i32::try_from(visit.position).unwrap().to_ne_bytes(),
            )
            .unwrap();
    }

    fn request<'a>(
        &'a self,
        g: Geometry,
        rows: &'a Rows,
        position: usize,
        visit: &LayerVisit,
    ) -> ResidentLocalDecode<'a> {
        let r = |buffer: &'a DeviceBuffer, words: usize| buffer.range(0, words * WORD).unwrap();
        let hidden = g.hidden;
        let past = PrepShape::TAPS - 1;
        ResidentLocalDecode {
            visit: *visit,
            position,
            math: self.math,
            input: r(&rows.input, hidden),
            messages: rows
                .messages
                .as_ref()
                .map(|messages| messages.each_ref().map(|buffer| r(buffer, hidden))),
            gammas: [r(&self.d_gamma, hidden); 3],
            weights: PrepWeights {
                query_scale: r(&self.d_scales[0], g.q()),
                relative_scale: r(&self.d_scales[1], g.query_heads * g.dims),
                key_kernel: r(&self.d_conv_kernels[0], g.kv() * PrepShape::TAPS),
                value_kernel: r(&self.d_conv_kernels[1], g.kv() * PrepShape::TAPS),
            },
            raw_history: rows
                .raw_history
                .each_ref()
                .map(|buffer| r(buffer, past * g.kv())),
            attn_conv: ConvRows {
                kernel: r(&self.d_attn_kernel, hidden * PrepShape::TAPS),
                history: r(&rows.attn_history, past * hidden),
            },
            pos: r(&rows.pos, 1),
            out: r(&rows.out, hidden),
            log: r(&rows.log, 4 * g.kv()),
            status: r(&rows.status, rows.status.bytes() / WORD),
        }
    }

    /// The dense half as `GgufCuda::enqueue_local_decode` queues it.
    fn enqueue_dense<'a>(
        &'a self,
        context: &'a CudaContext,
        request: &'a ResidentLocalDecode<'a>,
        g: Geometry,
    ) -> impl FnOnce([DeviceRange<'_>; 4], ResidentStatus<'_>) -> Result<(), EngError> + 'a {
        move |[q, relative, sinks, gates], status| {
            let w = &self.weights;
            let projections = [
                resident_projection(&w.wq, 0),
                resident_projection(&w.wk, 1),
                resident_projection(&w.wv, 2),
                resident_projection(&w.wr, 0),
                resident_projection(&w.sinks, 0),
                resident_projection(&w.gates, 0),
            ];
            let inputs = match request.messages {
                Some(messages) => ResidentLocalInputs::Controlled {
                    input: request.input,
                    messages,
                    gammas: request.gammas,
                },
                None => ResidentLocalInputs::Plain {
                    input: request.input,
                    gamma: request.gammas[0],
                    zeros: status.zeros.unwrap(),
                },
            };
            let kv = g.kv();
            let log = |index: usize| request.log.slice(index * kv * WORD, kv * WORD);
            self.dense.enqueue_local_resident(
                context,
                &self.stream,
                ResidentLocal {
                    inputs,
                    controllers: request.messages.is_some().then(|| {
                        w.controllers
                            .each_ref()
                            .map(|role| role.each_ref().map(|m| (m, DenseDtype::F32)))
                    }),
                    eps: request.math.eps,
                    weights: projections,
                },
                ResidentLocalTargets {
                    projections: [q, log(0)?, log(1)?, relative, sinks, gates],
                    controller_health: status.controller_health,
                    gate_health: status.gate_health,
                },
            )
        }
    }

    fn enqueue<'a>(
        &'a self,
        context: &'a CudaContext,
        request: &'a ResidentLocalDecode<'a>,
        g: Geometry,
    ) -> Result<(), EngError> {
        let projection = OutputProjection {
            weight: self
                .weights
                .wo
                .codes
                .range(0, self.weights.wo.codes.bytes())
                .unwrap(),
            dtype: DenseDtype::F32,
            rows: g.hidden,
        };
        let dense = self.enqueue_dense(context, request, g);
        self.resident.enqueue_resident_local(
            context,
            &self.stream,
            request,
            ResidentParts {
                shape: g.local(),
                projection,
                hidden: g.hidden,
            },
            dense,
        )
    }

    /// The host-entry chain: `step_local` with host rows, then the attention convolution.
    fn oracle(&self, context: &CudaContext, visit: &mut HostVisit, g: Geometry) -> Vec<f32> {
        let count = (visit.position).min(PrepShape::TAPS - 1);
        let read_weights = |[q, relative, k, v]: [usize; 4]| {
            let rows = [
                &self.scales[0],
                &self.scales[1],
                &self.conv_kernels[0],
                &self.conv_kernels[1],
            ];
            debug_assert_eq!(rows.each_ref().map(|row| row.len()), [q, relative, k, v]);
            Ok(rows.each_ref().map(|row| Cow::Borrowed(row.as_slice())))
        };
        debug_assert_eq!(visit.raw.each_ref().map(|h| h.len()), [count; 2]);
        let w = &self.weights;
        let projections = [
            resident_projection(&w.wq, 0),
            resident_projection(&w.wk, 1),
            resident_projection(&w.wv, 2),
            resident_projection(&w.wr, 0),
            resident_projection(&w.sinks, 0),
            resident_projection(&w.gates, 0),
        ];
        let gates: [Vec<f32>; 3];
        let mut plain;
        let inputs = match &visit.messages {
            Some(messages) => {
                gates = std::array::from_fn(|role| {
                    let mut gate = visit.input.clone();
                    gate.extend_from_slice(&messages[role]);
                    let (left, right) = gate.split_at_mut(g.hidden);
                    rms_norm(left, MIX_EPS);
                    rms_norm(right, MIX_EPS);
                    gate
                });
                LocalInputs::Controlled(
                    std::array::from_fn(|role| ControlledAttentionInput {
                        gate: &gates[role],
                        input: &visit.input,
                        message: &messages[role],
                        gamma: &self.gamma,
                    }),
                    w.controllers
                        .each_ref()
                        .map(|role| role.each_ref().map(|m| (m, DenseDtype::F32))),
                    self.math.eps,
                )
            }
            None => {
                plain = visit.input.clone();
                rms_norm(&mut plain, self.math.eps);
                for (value, gamma) in plain.iter_mut().zip(&self.gamma) {
                    *value *= gamma;
                }
                LocalInputs::Plain([&plain; 3])
            }
        };
        let result = self
            .oracle
            .step_local(
                context,
                (0, visit.position),
                LocalPrep {
                    math: self.math,
                    read_weights: &read_weights,
                    raw: [&visit.raw[0], &visit.raw[1]],
                },
                LocalProjection {
                    dense: &self.dense,
                    inputs,
                    weights: projections,
                },
                Some(OutputProjection {
                    weight: w.wo.codes.range(0, w.wo.codes.bytes()).unwrap(),
                    dtype: DenseDtype::F32,
                    rows: g.hidden,
                }),
            )
            .unwrap()
            .expect("oracle chains");
        for (history, raw) in visit.raw.iter_mut().zip(result.rows.iter().take(2)) {
            history.push_front(raw.clone());
            history.truncate(PrepShape::TAPS - 1);
        }
        causal_conv(
            &mut visit.attn,
            result.output,
            &self.attn_kernel,
            PrepShape::TAPS,
        )
    }
}

const VISIT: LayerVisit = LayerVisit {
    weight_layer: 3,
    state_slot: 0,
    attention: Attention::Local,
};

fn close(actual: &[f32], expected: &[f32], what: &str) {
    assert_eq!(actual.len(), expected.len(), "{what} length");
    for (index, (a, b)) in actual.iter().zip(expected).enumerate() {
        assert!((a - b).abs() <= 1e-4, "{what}[{index}]: {a} != {b}");
    }
}

/// Advance both chains through `positions` visits and compare every row they produce.
fn compare(context: &CudaContext, g: Geometry, bank: bool, positions: usize) {
    let harness = Harness::new(context, g, if bank { 101 } else { 211 });
    harness
        .resident
        .prepare_resident_workspace(context, (g.local(), g.hidden), bank)
        .unwrap();
    let rows = harness.rows(context, g, bank);
    let mut generate = lcg(307);
    let mut visit = HostVisit {
        position: 1,
        input: generate(g.hidden),
        messages: bank.then(|| std::array::from_fn(|_| generate(g.hidden))),
        // The cursor was seeded by a zero KV row at position 0.
        raw: std::array::from_fn(|_| VecDeque::from([vec![0.0; g.kv()]])),
        attn: VecDeque::from([vec![0.0; g.hidden]]),
    };
    let past = PrepShape::TAPS - 1;
    let padded = |history: &VecDeque<Vec<f32>>, width| {
        let mut words = Vec::new();
        for lag in 0..past {
            words.extend(
                history
                    .get(lag)
                    .cloned()
                    .unwrap_or_else(|| vec![0.0; width]),
            );
        }
        words
    };
    for position in 1..=positions {
        visit.position = position;
        harness.stage(context, g, &visit, &rows);
        let expected = harness.oracle(context, &mut visit, g);
        let request = harness.request(g, &rows, position, &VISIT);
        let before = harness.stream.enqueued();
        harness.enqueue(context, &request, g).unwrap();
        assert!(harness.stream.enqueued() > before);
        assert!(
            !harness
                .resident
                .can_chain_local(VISIT.state_slot, position + 1)
        );
        harness.stream.synchronize().unwrap();
        let status = read(&rows.status, &harness.stream, request.status.bytes() / WORD);
        assert!(
            status.iter().all(|v| *v == 0.0),
            "position {position}: status {status:?}"
        );
        harness
            .resident
            .publish_resident_cursor(VISIT.state_slot, position);
        let out = read(&rows.out, &harness.stream, g.hidden);
        close(&out, &expected, "output");
        let log = read(&rows.log, &harness.stream, 4 * g.kv());
        let raw = [padded(&visit.raw[0], g.kv()), padded(&visit.raw[1], g.kv())];
        // The log's raw rows are the newest history rows; the processed K and V follow.
        close(&log[..g.kv()], &raw[0][..g.kv()], "log raw K");
        close(&log[g.kv()..2 * g.kv()], &raw[1][..g.kv()], "log raw V");
        for (plane, history) in rows.raw_history.iter().enumerate() {
            let shifted = read(history, &harness.stream, past * g.kv());
            close(&shifted, &raw[plane], "shifted raw history");
        }
        let shifted = read(&rows.attn_history, &harness.stream, past * g.hidden);
        close(
            &shifted,
            &padded(&visit.attn, g.hidden),
            "shifted attention history",
        );
        visit.input = generate(g.hidden);
        if let Some(messages) = &mut visit.messages {
            for message in messages {
                *message = generate(g.hidden);
            }
        }
    }
}

#[test]
fn resident_local_decode_matches_the_host_entry_chain() {
    if !device_present("Inkling CUDA resident local decode") {
        return;
    }
    let context = CudaContext::new().unwrap();
    let small = Geometry {
        hidden: 32,
        query_heads: 1,
        kv_heads: 1,
        dims: 4,
        rank: 16,
    };
    for bank in [false, true] {
        compare(&context, small, bank, 6);
    }
    let mini = Geometry {
        hidden: 2304,
        query_heads: 24,
        kv_heads: 8,
        dims: 16,
        rank: 64,
    };
    compare(&context, mini, true, 5);
}

#[test]
fn resident_local_decode_reports_non_finite_input_in_health_without_an_error() {
    if !device_present("Inkling CUDA resident local decode health") {
        return;
    }
    let context = CudaContext::new().unwrap();
    let g = Geometry {
        hidden: 32,
        query_heads: 1,
        kv_heads: 1,
        dims: 4,
        rank: 16,
    };
    let harness = Harness::new(&context, g, 401);
    harness
        .resident
        .prepare_resident_workspace(&context, (g.local(), g.hidden), true)
        .unwrap();
    let rows = harness.rows(&context, g, true);
    let mut generate = lcg(503);
    let mut input = generate(g.hidden);
    input[7] = f32::NAN;
    let visit = HostVisit {
        position: 1,
        input,
        messages: Some(std::array::from_fn(|_| generate(g.hidden))),
        raw: [VecDeque::new(), VecDeque::new()],
        attn: VecDeque::new(),
    };
    harness.stage(&context, g, &visit, &rows);
    let request = harness.request(g, &rows, 1, &VISIT);
    harness.enqueue(&context, &request, g).unwrap();
    harness.stream.synchronize().unwrap();
    let status = read(&rows.status, &harness.stream, request.status.bytes() / WORD);
    assert!(!harness.resident.can_chain_local(VISIT.state_slot, 1));
    assert!(!harness.resident.can_chain_local(VISIT.state_slot, 2));
    let layout = local_status(g.local(), true);
    let gates = &status[layout.prep_words + layout.roles..][..layout.gates];
    assert!(
        gates.iter().any(|v| v.is_nan()),
        "gate health {gates:?} of {status:?}"
    );
}

#[test]
fn resident_local_decode_rejections_fire_before_any_launch() {
    if !device_present("Inkling CUDA resident local decode rejections") {
        return;
    }
    let context = CudaContext::new().unwrap();
    let g = Geometry {
        hidden: 32,
        query_heads: 1,
        kv_heads: 1,
        dims: 4,
        rank: 16,
    };
    let harness = Harness::new(&context, g, 601);
    let rows = harness.rows(&context, g, true);
    let visit = HostVisit {
        position: 1,
        input: vec![0.5; g.hidden],
        messages: Some(std::array::from_fn(|_| vec![0.25; g.hidden])),
        raw: [VecDeque::new(), VecDeque::new()],
        attn: VecDeque::new(),
    };
    harness.stage(&context, g, &visit, &rows);
    let before = harness.stream.enqueued();
    // A position the cursor does not know is refused before the workspace exists.
    let stale = harness.request(g, &rows, 2, &VISIT);
    assert!(harness.enqueue(&context, &stale, g).is_err());
    // Every caller range is checked against its exact words before the first launch.
    let request = harness.request(g, &rows, 1, &VISIT);
    assert!(harness.enqueue(&context, &request, g).is_err());
    assert!(harness.resident.can_chain_local(VISIT.state_slot, 1));
    let shape = (g.local(), g.hidden, PrepShape::TAPS);
    let short = ResidentLocalDecode {
        out: rows.out.range(0, (g.hidden - 1) * WORD).unwrap(),
        ..request
    };
    assert!(validate_rows(&request_rows(&short, shape)).is_err());
    let aliasing = ResidentLocalDecode {
        out: rows.log.range(0, g.hidden * WORD).unwrap(),
        ..request
    };
    assert!(validate_rows(&request_rows(&aliasing, shape)).is_err());
    assert_eq!(harness.stream.enqueued(), before);
    // A gamma of the wrong width through the dense entry alone.
    let projection = resident_projection(&harness.weights.wq, 0);
    let targets = ResidentLocalTargets {
        projections: [
            rows.log.range(0, g.q() * WORD).unwrap(),
            rows.log.range(g.q() * WORD, g.kv() * WORD).unwrap(),
            rows.log
                .range((g.q() + g.kv()) * WORD, g.kv() * WORD)
                .unwrap(),
            rows.log
                .range((g.q() + 2 * g.kv()) * WORD, g.query_heads * g.dims * WORD)
                .unwrap(),
            rows.log
                .range(
                    (g.q() + 2 * g.kv() + g.query_heads * g.dims) * WORD,
                    g.query_heads * WORD,
                )
                .unwrap(),
            rows.raw_history[0].range(0, g.q() * WORD).unwrap(),
        ],
        controller_health: rows.status.range(0, 3 * WORD).unwrap(),
        gate_health: Some(rows.status.range(3 * WORD, 6 * WORD).unwrap()),
    };
    let w = &harness.weights;
    let wrong = ResidentLocal {
        inputs: ResidentLocalInputs::Controlled {
            input: request.input,
            messages: request.messages.unwrap(),
            gammas: [rows.out.range(0, (g.hidden - 1) * WORD).unwrap(); 3],
        },
        controllers: Some(
            w.controllers
                .each_ref()
                .map(|role| role.each_ref().map(|m| (m, DenseDtype::F32))),
        ),
        eps: 1e-5,
        weights: [
            projection,
            resident_projection(&w.wk, 1),
            resident_projection(&w.wv, 2),
            resident_projection(&w.wr, 0),
            resident_projection(&w.sinks, 0),
            resident_projection(&w.gates, 0),
        ],
    };
    assert!(
        harness
            .dense
            .enqueue_local_resident(&context, &harness.stream, wrong, targets)
            .is_err()
    );
    assert_eq!(harness.stream.enqueued(), before);
}
