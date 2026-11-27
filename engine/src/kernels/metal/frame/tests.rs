//! Device tests (Mac) of the frame against the waiting launch. The shader is
//! `x[i] = x[i] * 2 + by`, whose result tells two orders of two dispatches apart. Each test
//! holds an adapter lease, so the suite runs with `--test-threads=1`. The timing test prints
//! and is ignored.

use std::rc::Rc;
use std::time::{Duration, Instant};

use super::Frame;
use crate::EngError;
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
    rig.step(&mut frame, 0.75);
    assert_eq!(
        frame.in_flight(),
        1,
        "the first command buffer is committed, not waited for"
    );
    for _ in 0..2 {
        rig.step(&mut frame, 0.75);
        assert_eq!(
            frame.in_flight(),
            2,
            "the one the GPU runs and one ahead, never three"
        );
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
fn a_fill_of_a_frame_lands_between_the_dispatches_around_it() {
    let rig = Rig::new();
    // One command buffer an operation, then all three in one: the order holds in both.
    for per_buffer in [1, 8] {
        rig.zero();
        let mut frame = rig.ctx.frame_of(per_buffer);
        rig.step(&mut frame, 1.0);
        rig.x.zero_at(&mut frame, 16 * 4, 32 * 4).expect("the fill");
        rig.step(&mut frame, 3.0);
        frame.drain().expect("the drain");
        // 1 * 2 + 3 outside the fill, 0 * 2 + 3 inside it. A fill that ran last leaves 0
        // inside it; one that ran first leaves 5 everywhere.
        let want: Vec<f32> = (0..VALUES)
            .map(|i| if (16..48).contains(&i) { 3.0 } else { 5.0 })
            .collect();
        assert_eq!(rig.values(), want, "per_buffer {per_buffer}");
    }
}

#[test]
fn a_fill_takes_any_bytes_up_to_the_whole_buffer() {
    let ctx = MetalContext::new().expect("a Metal context");
    let data = ctx.upload(&[0xa5u8; 33]).expect("an upload");
    let mut frame = ctx.frame_of(8);
    data.zero_at(&mut frame, 5, 7)
        .expect("a fill off the grid of four");
    data.zero_at(&mut frame, 26, 7)
        .expect("a fill to the last byte");
    frame.drain().expect("the drain");
    let want: Vec<u8> = (0..33)
        .map(|i| {
            if (5..12).contains(&i) || (26..33).contains(&i) {
                0
            } else {
                0xa5
            }
        })
        .collect();
    assert_eq!(data.read(0, 33).expect("a read"), want);
    data.zero_at(&mut frame, 0, 33).expect("the whole buffer");
    frame.drain().expect("the drain");
    assert_eq!(data.read(0, 33).expect("a read"), vec![0u8; 33]);
}

#[test]
fn a_refused_fill_changes_nothing() {
    let rig = Rig::new();
    let alien = MetalContext::new().expect("a second adapter lease");
    let theirs = alien.upload(&1.0f32.to_le_bytes()).expect("an upload");
    let mut frame = rig.ctx.frame_of(8);
    rig.step(&mut frame, 2.0);
    let refusal = |result: Result<(), EngError>, what: &str| match result {
        Err(EngError::BadRequest(message)) => assert!(message.contains(what), "{message}"),
        other => panic!("the fill was not refused: {other:?}"),
    };
    let bytes = 4 * VALUES;
    refusal(
        rig.x.zero_at(&mut frame, 0, 0),
        "of 0 bytes at offset 0 does not fit a buffer of 256 bytes",
    );
    refusal(
        rig.x.zero_at(&mut frame, bytes, 4),
        "of 4 bytes at offset 256 does not fit",
    );
    refusal(
        rig.x.zero_at(&mut frame, bytes - 3, 4),
        "of 4 bytes at offset 253 does not fit",
    );
    refusal(
        frame.zero(&theirs.range(0, 4).expect("a range")),
        "of another adapter",
    );
    assert_eq!(frame.in_flight(), 0, "the refusals committed nothing");
    assert!(frame.drain().expect("the drain") > 0);
    assert_eq!(rig.values(), vec![2.0f32; VALUES]);
    assert_eq!(theirs.read(0, 4).expect("a read"), 1.0f32.to_le_bytes());
}

#[test]
#[ignore = "prints the fill's time on this device"]
fn a_fill_is_timed() {
    const FILLS: usize = 128;
    let ctx = MetalContext::new().expect("a Metal context");
    println!("device: {} live bytes before the timing", ctx.live_bytes());
    // From a small buffer to 16 MiB.
    for bytes in [64usize << 10, 1 << 20, 16 << 20] {
        let data = ctx.upload(&vec![0xa5u8; bytes]).expect("an upload");
        let range = data.range(0, bytes).expect("a range");
        let one = || {
            let mut frame = ctx.frame_of(1);
            frame.zero(&range).expect("the fill");
            frame.drain().expect("the drain")
        };
        for _ in 0..5 {
            one();
        }
        let mut singles: Vec<u64> = (0..21).map(|_| one()).collect();
        singles.sort_unstable();
        let mut frame = ctx.frame_of(FILLS);
        for _ in 0..FILLS {
            frame.zero(&range).expect("the fill");
        }
        let framed = frame.drain().expect("the drain") / FILLS as u64;
        println!(
            "ZERO bytes {bytes}: one launch {} us, {FILLS} in a frame {} us a dispatch",
            singles[singles.len() / 2] as f64 / 1e3,
            framed as f64 / 1e3,
        );
    }
}

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
