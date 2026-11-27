//! Device tests (Mac) of the frame against the waiting launch. The shader is
//! `x[i] = x[i] * 2 + by`, whose result tells two orders of two dispatches apart. Each test
//! holds an adapter lease, so the suite runs with `--test-threads=1`. The timing test prints
//! and is ignored.

use std::rc::Rc;
use std::time::{Duration, Instant};

use super::Frame;
use crate::kernels::metal::context::Grid;
use crate::kernels::metal::native::{Binding, RawKernel};
use crate::kernels::metal::{DeviceBuffer, MetalContext};

const SOURCE: &str = "#include <metal_stdlib>\nusing namespace metal;\nkernel void affine_step(device float *x [[buffer(0)]], constant float &by [[buffer(1)]],\n                                uint i [[thread_position_in_grid]]) {\n  x[i] = x[i] * 2.0f + by;\n}\n";
const NAME: &str = "affine_step";
/// The values of `x`. The grid has one thread for each.
const VALUES: usize = 64;
const GRID: Grid = [VALUES, 1, 1];
const GROUP: Grid = [32, 1, 1];

struct Rig {
    ctx: MetalContext,
    x: DeviceBuffer,
}

impl Rig {
    fn new() -> Self {
        let ctx = MetalContext::new().expect("a Metal context");
        let x = ctx.upload(&vec![0u8; 4 * VALUES]).expect("an upload");
        Self { ctx, x }
    }

    fn kernel(&self) -> Rc<RawKernel> {
        self.ctx.kernel(NAME, SOURCE).expect("the shader")
    }

    fn binding(&self) -> Binding {
        self.x.range(0, 4 * VALUES).expect("a range").binding()
    }

    /// One dispatch of the shader into `frame`.
    fn step(&self, frame: &mut Frame, by: f32) {
        frame
            .encode(
                &self.kernel(),
                &[self.binding()],
                &by.to_le_bytes(),
                GRID,
                GROUP,
                1,
            )
            .expect("a dispatch");
    }

    /// One waited launch of the same shader.
    fn launch(&self, by: f32) {
        self.ctx
            .launch_repeated(
                &self.kernel(),
                &[self.binding()],
                &by.to_le_bytes(),
                GRID,
                GROUP,
                1,
            )
            .expect("a waited launch");
    }

    fn zero(&self) {
        self.x.write(0, &vec![0u8; 4 * VALUES]).expect("the fill");
    }

    fn values(&self) -> Vec<f32> {
        let bytes = self.x.read(0, 4 * VALUES).expect("a read");
        bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect()
    }
}

#[test]
fn a_frame_gives_the_bits_of_the_same_waited_launches() {
    let rig = Rig::new();
    let mut frame = rig.ctx.frame_of(2);
    for _ in 0..3 {
        rig.step(&mut frame, 0.75);
    }
    assert!(
        frame.drain().expect("the drain") > 0,
        "the device reports the GPU time of the token"
    );
    // 0.75, then 2.25, then 5.25: each is exact in f32. Two command buffers, the second of
    // one dispatch.
    assert_eq!(rig.values(), vec![5.25f32; VALUES]);

    rig.zero();
    for _ in 0..3 {
        rig.launch(0.75);
    }
    assert_eq!(rig.values(), vec![5.25f32; VALUES]);
}

#[test]
fn dispatch_calls_of_a_frame_keep_their_order_across_command_buffers() {
    let rig = Rig::new();
    let mut frame = rig.ctx.frame_of(1);
    rig.step(&mut frame, 1.0);
    rig.step(&mut frame, 0.5);
    frame.drain().expect("the drain");
    // 0 * 2 + 1, then 1 * 2 + 0.5. The other order gives 2.
    assert_eq!(rig.values(), vec![2.5f32; VALUES]);
}

#[test]
fn a_host_touch_waits_for_the_command_buffer_that_binds_its_buffer() {
    let rig = Rig::new();
    let other = rig.ctx.upload(&1.0f32.to_le_bytes()).expect("an upload");
    let mut frame = rig.ctx.frame_of(1);
    assert_eq!(
        frame.drain(),
        Ok(0),
        "a frame that encoded nothing waits for nothing"
    );

    rig.step(&mut frame, 0.75);
    assert_eq!(
        frame.in_flight(),
        1,
        "the command buffer is committed, not waited for"
    );
    other
        .write(0, &2.0f32.to_le_bytes())
        .expect("a write of a buffer no command buffer binds");
    assert_eq!(
        frame.in_flight(),
        1,
        "a buffer that is not bound waits for nothing"
    );
    assert_eq!(
        rig.values(),
        vec![0.75f32; VALUES],
        "the read waited for the command buffer that binds the buffer"
    );
    assert_eq!(frame.in_flight(), 0);

    rig.step(&mut frame, 0.75);
    assert!(frame.drain().expect("the drain") > 0);
    assert_eq!(frame.in_flight(), 0);
    assert_eq!(rig.values(), vec![2.25f32; VALUES]);
}

#[test]
fn one_command_buffer_stays_ahead_of_the_gpu() {
    let rig = Rig::new();
    let mut frame = rig.ctx.frame_of(1);
    for _ in 0..3 {
        rig.step(&mut frame, 0.75);
        assert_eq!(frame.in_flight(), 1, "one command buffer ahead, never two");
    }
    frame.drain().expect("the drain");
    assert_eq!(frame.in_flight(), 0);
    assert_eq!(rig.values(), vec![5.25f32; VALUES]);
}

#[test]
fn a_dropped_frame_keeps_the_work_it_encoded() {
    let rig = Rig::new();
    let mut frame = rig.ctx.frame_of(8);
    rig.step(&mut frame, 1.0);
    rig.step(&mut frame, 0.5);
    assert_eq!(
        frame.in_flight(),
        0,
        "the command buffer is open until the frame is full"
    );
    drop(frame);
    assert_eq!(rig.values(), vec![2.5f32; VALUES]);
}

/// Prints, for 128 dispatches of a 256 MiB read, the wall time and the GPU time of a frame of
/// 64 dispatches a command buffer against one-call launches. A gate reads the lines. Each form
/// must write its sums over a sentinel, so a dispatch that misses its buffer fails here.
#[test]
#[ignore = "prints the timing of a frame against one-call launches"]
fn a_frame_keeps_the_gpu_busy_between_256_mb_reads() {
    const READ_SOURCE: &str = "#include <metal_stdlib>\nusing namespace metal;\nkernel void read_sum(device const float *x [[buffer(0)]], device float *out [[buffer(1)]],\n                              constant uint2 &shape [[buffer(2)]], uint i [[thread_position_in_grid]]) {\n  float acc = 0.0f;\n  for (uint k = i; k < shape.x; k += shape.y) {\n    acc += x[k];\n  }\n  out[i] = acc;\n}\n";
    const READ_THREADS: usize = 1 << 16;
    const READ_FLOATS: usize = 1 << 26;
    const DISPATCHES: usize = 128;

    let ctx = MetalContext::new().expect("a Metal context");
    let data = ctx
        .upload(&vec![0u8; 4 * READ_FLOATS])
        .expect("the 256 MiB buffer");
    let out = ctx.upload(&vec![0u8; 4 * READ_THREADS]).expect("the sums");
    let kernel = ctx.kernel("read_sum", READ_SOURCE).expect("the shader");
    let bindings = [
        data.range(0, 4 * READ_FLOATS).expect("a range").binding(),
        out.range(0, 4 * READ_THREADS).expect("a range").binding(),
    ];
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&(READ_FLOATS as u32).to_le_bytes());
    bytes.extend_from_slice(&(READ_THREADS as u32).to_le_bytes());
    let grid = [READ_THREADS, 1, 1];
    let group = [256, 1, 1];
    // The pattern every sum must replace: a NaN the sum of zeros never produces.
    let sentinel: Vec<u8> = 0x7fc0_dead_u32
        .to_le_bytes()
        .iter()
        .copied()
        .cycle()
        .take(4 * READ_THREADS)
        .collect();
    let fill = || out.write(0, &sentinel).expect("the fill of the sums");
    let written = || {
        let bytes = out.read(0, 4 * READ_THREADS).expect("a read of the sums");
        assert!(
            bytes.iter().all(|&byte| byte == 0),
            "every thread wrote its sum of zeros over the sentinel"
        );
    };

    let report = |form: &str, wall: Duration, gpu_ns: u64| {
        let wall_ms = wall.as_secs_f64() * 1e3;
        let gpu_ms = gpu_ns as f64 / 1e6;
        let idle = 100.0 * (wall_ms - gpu_ms) / wall_ms;
        eprintln!(
            "{form}: {DISPATCHES} dispatches of a 256 MiB read, wall {wall_ms:.3} ms, GPU {gpu_ms:.3} ms, the GPU idle {idle:.1} percent of the wall time"
        );
    };
    for round in 1..=2 {
        fill();
        let start = Instant::now();
        let mut gpu_ns = 0;
        for _ in 0..DISPATCHES {
            gpu_ns += ctx
                .launch_repeated(&kernel, &bindings, &bytes, grid, group, 1)
                .expect("a waited launch");
        }
        written();
        report(
            &format!("round {round}, one-call launch"),
            start.elapsed(),
            gpu_ns,
        );

        fill();
        let mut frame = ctx.frame();
        let start = Instant::now();
        for _ in 0..DISPATCHES {
            frame
                .encode(&kernel, &bindings, &bytes, grid, group, 1)
                .expect("a dispatch");
        }
        let gpu_ns = frame.drain().expect("the drain");
        written();
        report(
            &format!("round {round}, frame of 64 dispatches a command buffer"),
            start.elapsed(),
            gpu_ns,
        );
    }
}
