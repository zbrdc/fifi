//! Device tests of the q8_1 expert schedule (3090): it stays close to the f32 products on a type
//! with a window case, and a call it does not take (host picks) runs the f32 products with the
//! same bits.

use super::*;
use crate::kernels::cuda::{
    DeviceBuffer, DeviceRange, native::resident::dense_gemv::moe::GgufGate, testkit,
};
use crate::weights::GgufType;

const SHAPE: GgufMoeShape = GgufMoeShape {
    hidden: 256,
    middle: 128,
    experts: 16,
};
const PICKS: [i32; 8] = [0, 3, 5, 6, 9, 11, 12, 15];
const Q8: GgufMoeSchedule = GgufMoeSchedule {
    gate: GgufGate::Packed,
    exact_q2_down: false,
    q8_window: true,
};

fn floats(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_ne_bytes()).collect()
}

/// `rows` rows of `k` IQ4_NL values: blocks of an FP16 scale near 0.002 and random nibbles.
fn stack(rng: &mut testkit::Rng, rows: usize, k: usize) -> Vec<u8> {
    let bytes = QuantType::Iq4Nl.weight_bytes(rows, k).expect("stack size");
    let mut out = vec![0; bytes];
    for block in out.as_chunks_mut::<18>().0 {
        block[..2].copy_from_slice(&0x1819u16.to_le_bytes());
        for byte in &mut block[2..] {
            *byte = rng.below(256) as u8;
        }
    }
    out
}

struct Rig {
    ctx: CudaContext,
    stream: DeviceStream,
    gate: DeviceBuffer,
    down: DeviceBuffer,
    ids: DeviceBuffer,
    weights: DeviceBuffer,
    scales: [DeviceBuffer; 2],
    x: DeviceBuffer,
    scratch: DeviceBuffer,
    y: DeviceBuffer,
    host: Vec<GgufMoePick>,
}

impl Rig {
    fn new() -> Self {
        let ctx = CudaContext::new().expect("context");
        let mut rng = testkit::Rng::new(0x05ee_d0e8);
        let gate = stack(&mut rng, 2 * SHAPE.middle * SHAPE.experts, SHAPE.hidden);
        let down = stack(&mut rng, SHAPE.hidden * SHAPE.experts, SHAPE.middle);
        let x: Vec<f32> = (0..SHAPE.hidden).map(|_| rng.normal() as f32).collect();
        let weights: Vec<f32> = (0..PICKS.len()).map(|i| 0.05 + i as f32 / 40.0).collect();
        let scale = |base: f32| -> Vec<f32> {
            (0..SHAPE.experts).map(|e| base + e as f32 / 64.0).collect()
        };
        let (gates, downs) = (scale(0.75), scale(1.25));
        let host = PICKS
            .iter()
            .zip(&weights)
            .map(|(&id, &weight)| GgufMoePick {
                expert: id as u32,
                gate_scale: gates[id as usize],
                down_scale: downs[id as usize],
                weight,
            })
            .collect();
        let ids: Vec<u8> = PICKS.iter().flat_map(|id| id.to_ne_bytes()).collect();
        Self {
            stream: ctx.stream().expect("stream"),
            gate: ctx.upload(&gate).expect("gate"),
            down: ctx.upload(&down).expect("down"),
            ids: ctx.upload(&ids).expect("ids"),
            weights: ctx.upload(&floats(&weights)).expect("weights"),
            scales: [gates, downs].map(|row| ctx.upload(&floats(&row)).expect("scales")),
            x: ctx.upload(&floats(&x)).expect("x"),
            scratch: ctx
                .allocate(SHAPE.scratch_bytes(PICKS.len()).expect("scratch size"))
                .expect("scratch"),
            y: ctx.allocate(SHAPE.hidden * 4).expect("y"),
            host,
            ctx,
        }
    }

    /// The expert sequence under `schedule`, with the picks on the device or, with `host`, in
    /// the call.
    fn run(&self, schedule: GgufMoeSchedule, host: bool) -> (GgufMoeSchedule, Vec<f32>) {
        fn all(buffer: &DeviceBuffer) -> DeviceRange<'_> {
            buffer.range(0, buffer.bytes()).expect("range")
        }
        let picks = if host {
            GgufMoePicks::Host(&self.host)
        } else {
            GgufMoePicks::Device {
                count: PICKS.len(),
                ids: all(&self.ids),
                weights: all(&self.weights),
                scales: [all(&self.scales[0]), all(&self.scales[1])],
            }
        };
        let taken = self
            .ctx
            .gguf_moe_scheduled(
                &self.stream,
                GgufMoeCall {
                    shape: SHAPE,
                    dtypes: [DenseDtype::from_gguf(GgufType::Iq4Nl).expect("packed type"); 2],
                    caps: [8.0, 8.0],
                    picks,
                    weights: GgufMoeWeights::Stacks([all(&self.gate), all(&self.down)]),
                    x: all(&self.x),
                    scratch: all(&self.scratch),
                    y: all(&self.y),
                },
                schedule,
            )
            .expect("experts");
        let mut bytes = vec![0; SHAPE.hidden * 4];
        self.y.read_at(&self.stream, 0, &mut bytes).expect("read");
        let y = bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|w| f32::from_ne_bytes(*w))
            .collect();
        (taken, y)
    }
}

#[test]
fn q8_1_expert_products_stay_close_to_the_f32_products() {
    if !testkit::device_present("q8_1_expert_products_stay_close_to_the_f32_products") {
        return;
    }
    let rig = Rig::new();
    let (exact, f32_y) = rig.run(GgufMoeSchedule::default(), false);
    assert!(!exact.q8_window);
    let (taken, q8_y) = rig.run(Q8, false);
    assert!(
        taken.q8_window,
        "IQ4_NL takes the q8_1 window on both products"
    );
    let norm = f32_y.iter().map(|v| v * v).sum::<f32>().sqrt();
    let error = f32_y
        .iter()
        .zip(&q8_y)
        .map(|(a, b)| (a - b) * (a - b))
        .sum::<f32>()
        .sqrt();
    assert!(norm > 0.0 && q8_y.iter().all(|v| v.is_finite()));
    assert!(error / norm < 0.03, "relative L2 {}", error / norm);
    assert_ne!(f32_y, q8_y, "the q8_1 products are not the f32 products");
}

#[test]
fn a_call_the_q8_1_window_does_not_take_runs_the_f32_products() {
    if !testkit::device_present("a_call_the_q8_1_window_does_not_take_runs_the_f32_products") {
        return;
    }
    let rig = Rig::new();
    let (_, baseline) = rig.run(GgufMoeSchedule::default(), true);
    let (taken, y) = rig.run(Q8, true);
    assert!(!taken.q8_window, "host picks keep the f32 products");
    assert_eq!(
        baseline.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        y.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
    );
}
