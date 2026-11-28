use super::*;
use crate::{
    family::inkling2::{
        cpu_forward::{route, sigmoid},
        cuda_projection::gguf::ExpertStack,
    },
    kernels::cuda::{
        CapturePlan, device_present,
        resident::dense_gemv::{
            GgufMoeCall, GgufMoePick, GgufMoePicks, GgufMoeShape, GgufMoeWeights,
        },
    },
    weights::GgufType,
};

const ROUTER: RouterShape = RouterShape {
    m: 1,
    hidden: 32,
    n_expert: 33,
    top_k: 8,
};
const MOE: GgufMoeShape = GgufMoeShape {
    hidden: ROUTER.hidden,
    middle: 32,
    experts: ROUTER.n_expert,
};
const SCALE: f32 = 2.5;

fn floats(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_ne_bytes())
        .collect()
}

fn download<const N: usize>(stream: &DeviceStream, range: DeviceRange<'_>) -> [[u8; 4]; N] {
    let mut bytes = [[0; 4]; N];
    range
        .buffer()
        .read_at(stream, range.offset(), bytes.as_flattened_mut())
        .unwrap();
    bytes
}

fn stack(rows: usize, phase: usize) -> Vec<u8> {
    (0..rows)
        .flat_map(|row| {
            // Q8_0: a FP16 scale followed by 32 signed Q8 samples.
            0x2000u16.to_le_bytes().into_iter().chain(
                (0..32).map(move |col| (((row * 19 + col * 13 + phase) % 31) as i8 - 15) as u8),
            )
        })
        .collect()
}

fn scales(expert: usize) -> [f32; 2] {
    [
        0.5 + (expert % 5) as f32 / 8.0,
        1.5 - (expert % 7) as f32 / 8.0,
    ]
}

fn fixture_work(ctx: &CudaContext) -> RouteScratch {
    let rows = [0, 1].map(|part| {
        (0..MOE.experts)
            .flat_map(|expert| scales(expert)[part].to_le_bytes())
            .collect::<Vec<_>>()
    });
    let parts = rows.each_ref().map(|row| ExpertStack {
        bytes: &[],
        file_range: 0..0,
        tensor_scales: Some(row),
        shape: [0, 0],
    });
    RoutePlan::new(ROUTER, &[ExpertLayer { layer: 0, parts }])
        .unwrap()
        .upload(ctx)
        .unwrap()
}

#[test]
fn inkling_device_route_scale_plan_keeps_zero_and_invalid_values_on_host() {
    let valid = [0.5f32, 2.0].map(f32::to_le_bytes);
    let expected = [1.0f32, 1.0, 0.5, 2.0].map(f32::to_ne_bytes);
    assert_eq!(
        scale_rows([None, Some(valid.as_flattened())], 2).unwrap(),
        expected.as_flattened()
    );
    for bad_value in [0.0f32, f32::NAN, -1.0] {
        let invalid = [0.5f32, bad_value].map(f32::to_le_bytes);
        assert!(scale_rows([None, Some(invalid.as_flattened())], 2).is_none());
    }
    assert!(scale_rows([None, Some(&valid[0])], 2).is_none());
}

struct Rig {
    ctx: CudaContext,
    stream: DeviceStream,
    work: RouteScratch,
    x: DeviceBuffer,
    router: DeviceBuffer,
    bias: DeviceBuffer,
    status: DeviceBuffer,
    gate: DeviceBuffer,
    down: DeviceBuffer,
    scratch: DeviceBuffer,
    y: DeviceBuffer,
}

fn all(buffer: &DeviceBuffer) -> DeviceRange<'_> {
    buffer.range(0, buffer.bytes()).unwrap()
}

impl Rig {
    fn new(bias: &[f32]) -> Self {
        let ctx = CudaContext::new().unwrap();
        let mut input = [0.0; ROUTER.hidden];
        input[0] = 1.0;
        input[1] = 0.25;
        Self {
            stream: ctx.stream().unwrap(),
            work: fixture_work(&ctx),
            x: ctx.upload(&floats(&input)).unwrap(),
            router: ctx.allocate(ROUTER.n_expert * ROUTER.hidden * 4).unwrap(),
            bias: ctx.upload(&floats(bias)).unwrap(),
            status: ctx.upload(&0i32.to_ne_bytes()).unwrap(),
            gate: ctx.upload(&stack(MOE.experts * 2 * MOE.middle, 5)).unwrap(),
            down: ctx.upload(&stack(MOE.experts * MOE.hidden, 9)).unwrap(),
            scratch: ctx
                .allocate(MOE.scratch_bytes(ROUTER.top_k).unwrap())
                .unwrap(),
            y: ctx.allocate(MOE.hidden * 4).unwrap(),
            ctx,
        }
    }

    fn set_logits(&self, logits: &[f32; ROUTER.n_expert]) {
        // x[0] is 1 and the other weighted columns are zero, so the projection is exact.
        let mut router = [0.0; ROUTER.n_expert * ROUTER.hidden];
        for (row, &logit) in router
            .as_chunks_mut::<{ ROUTER.hidden }>()
            .0
            .iter_mut()
            .zip(logits)
        {
            row[0] = logit;
        }
        self.router
            .write_at(&self.stream, 0, &floats(&router))
            .unwrap();
    }

    fn route_and_experts(&self) -> Result<(), EngError> {
        let [ids, weights] = self.work.enqueue(
            &self.ctx,
            &self.stream,
            RouterInputs {
                x: all(&self.x),
                w: all(&self.router),
                bias: all(&self.bias),
                status: all(&self.status),
                scale: SCALE,
            },
        )?;
        self.experts(GgufMoePicks::Device {
            count: ROUTER.top_k,
            ids,
            weights,
            scales: self.work.scale_ranges(0)?,
        })
    }

    fn experts(&self, picks: GgufMoePicks<'_>) -> Result<(), EngError> {
        self.ctx.gguf_moe(
            &self.stream,
            GgufMoeCall {
                shape: MOE,
                dtypes: [DenseDtype::from_gguf(GgufType::Q80).unwrap(); 2],
                caps: [1.0, 2.0],
                picks,
                weights: GgufMoeWeights::Stacks([all(&self.gate), all(&self.down)]),
                x: all(&self.x),
                scratch: all(&self.scratch),
                y: all(&self.y),
            },
        )
    }

    fn output(&self) -> [f32; MOE.hidden] {
        self.stream.synchronize().unwrap();
        download(&self.stream, all(&self.y)).map(f32::from_ne_bytes)
    }
}

fn cpu_picks(logits: &[f32; ROUTER.n_expert], bias: &[f32; ROUTER.n_expert]) -> Vec<GgufMoePick> {
    let values = logits.map(sigmoid);
    let ids = route(&values, bias, ROUTER.top_k).unwrap();
    let sum = ids
        .iter()
        .map(|&id| values[id])
        .sum::<f32>()
        .max(ROUTE_NORMALIZATION_FLOOR);
    ids.into_iter()
        .map(|id| {
            let [gate_scale, down_scale] = scales(id);
            GgufMoePick {
                expert: id as u32,
                gate_scale,
                down_scale,
                weight: SCALE * (values[id] / sum),
            }
        })
        .collect()
}

#[test]
fn inkling_device_route_graph_matches_cpu_picks_and_packed_experts() {
    if !device_present("inkling_device_route_graph_matches_cpu_picks_and_packed_experts") {
        return;
    }
    // Bias promotes low scores, with a tie across two groups of 32 experts.
    let mut bias = [-3.0; ROUTER.n_expert];
    bias[0] = -1.0;
    bias[32] = -1.0;
    bias[13] = 1.0;
    let rig = Rig::new(&bias);
    let mut ordinary = [0.0; ROUTER.n_expert];
    ordinary[13] = -5.0; // Wins despite an unbiased sigmoid below the rest.
    ordinary[2] = 1.0;
    let mut close_scores = [0.375; ROUTER.n_expert];
    close_scores[13] = -2.0;
    // Adjacent logits round to the same F32 sigmoid and bias rank.
    close_scores[32] = 0.375f32.next_up();
    let mut tiny = [-20.0; ROUTER.n_expert];
    tiny[13] = -14.0; // Nonzero score below the normalization floor even after summing picks.
    rig.set_logits(&ordinary);
    rig.route_and_experts().unwrap();
    let eager = rig.output();
    let (graph, ()) = rig
        .ctx
        .capture(
            &rig.stream,
            CapturePlan {
                pins: &[],
                holds: Vec::new(),
            },
            || rig.route_and_experts(),
        )
        .unwrap();
    assert_eq!(graph.nodes(), 6);
    let [_, ids, weights] = rig.work.ranges().unwrap();

    for (case, logits) in [ordinary, close_scores, tiny, [-120.0; ROUTER.n_expert]]
        .into_iter()
        .enumerate()
    {
        rig.set_logits(&logits);
        graph.launch(&rig.stream).unwrap();
        let got = rig.output();
        if case == 0 {
            assert_eq!(got.map(f32::to_bits), eager.map(f32::to_bits));
        }
        let picks = cpu_picks(&logits, &bias);
        let ids = download::<{ ROUTER.top_k }>(&rig.stream, ids).map(i32::from_ne_bytes);
        let weights = download::<{ ROUTER.top_k }>(&rig.stream, weights).map(f32::from_ne_bytes);
        for (rank, pick) in picks.iter().enumerate() {
            assert_eq!(ids[rank], pick.expert as i32);
            assert_eq!(
                weights[rank].to_bits(),
                pick.weight.to_bits(),
                "case {case}, rank {rank}"
            );
        }
        rig.experts(GgufMoePicks::Host(&picks)).unwrap();
        assert_eq!(
            got.map(f32::to_bits),
            rig.output().map(f32::to_bits),
            "case {case}"
        );
    }

    let mut fault = ordinary;
    fault[0] = f32::NAN;
    rig.set_logits(&fault);
    graph.launch(&rig.stream).unwrap();
    assert!(rig.output().iter().all(|value| value.is_nan()));
    assert_eq!(
        download::<{ ROUTER.top_k }>(&rig.stream, ids).map(i32::from_ne_bytes),
        [-1; ROUTER.top_k]
    );
    assert_eq!(
        download::<1>(&rig.stream, all(&rig.status)).map(i32::from_ne_bytes),
        [1]
    );

    // Independently check the positive out-of-range device ID with a nonzero route weight.
    let invalid_ids = [(MOE.experts as i32).to_ne_bytes(); ROUTER.top_k];
    ids.buffer()
        .write_at(&rig.stream, ids.offset(), invalid_ids.as_flattened())
        .unwrap();
    weights
        .buffer()
        .write_at(&rig.stream, weights.offset(), &floats(&[1.0; ROUTER.top_k]))
        .unwrap();
    rig.experts(GgufMoePicks::Device {
        count: ROUTER.top_k,
        ids,
        weights,
        scales: rig.work.scale_ranges(0).unwrap(),
    })
    .unwrap();
    assert!(rig.output().iter().all(|value| value.is_nan()));
}
