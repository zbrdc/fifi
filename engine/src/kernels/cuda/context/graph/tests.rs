//! Tests of the capture window and executable graphs (seam 1: the Rust wrappers against the C
//! entries against the vendored kernels, inside a capture).
//!
//! "Bytes equal" means `to_bits()` equality of every f32; every comparison against the blocking
//! path is exact. A kernel's first launch loads its module, which a capture forbids, so every
//! device test runs its body once eagerly before it captures (`warm`). The owner counts, the
//! terminal flag and the queued work are process-wide, so these tests assume `--test-threads=1`,
//! as the device suite is always run (decision 0040).
//!
//! The assertions on a refusal run after `capture` returns: a panic inside a body would unwind
//! through the window and latch the backend for every later test.

use std::sync::atomic::{AtomicBool, Ordering};

use super::super::LIVE_GRAPHS;
use super::{deferred_len, is_capture_refusal, window_open};
use crate::EngError;
use crate::kernels::cuda::native::{self, resident};
use crate::kernels::cuda::resident::matmul_expert::{ExpertCall, ExpertSelect, ExpertStack};
use crate::kernels::cuda::testkit;
use crate::kernels::cuda::{
    CapturePlan, CudaContext, DeviceBuffer, DeviceEvent, DeviceStream, GraphExec, MatmulIo,
    PackCall, Projection, TERMINAL, TiledWeight, native_error, stage_operation,
};
use crate::weights::GgufType;

/// The text of every refusal inside a window.
const WINDOW_TEXT: &str = "not allowed inside a capture window";
/// Rows of one expert, and experts of the stack of the expert test.
const EXPERT_ROWS: usize = 64;
const EXPERTS: usize = 16;
/// Bytes of the ids buffer of the expert test.
const IDS_BYTES: usize = 256;

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|value| value.to_bits()).collect()
}

fn graphs() -> usize {
    LIVE_GRAPHS.load(Ordering::Acquire)
}

fn no_pins() -> CapturePlan<'static> {
    CapturePlan {
        pins: &[],
        holds: Vec::new(),
    }
}

/// The error of a result that must be one.
fn refusal<T>(result: Result<T, EngError>, case: &str) -> EngError {
    match result {
        Ok(_) => panic!("{case}: expected an error, got Ok"),
        Err(error) => error,
    }
}

/// Asserts a `BadRequest` whose message contains `needle`.
fn assert_bad_request(error: &EngError, case: &str, needle: &str) {
    assert!(
        matches!(error, EngError::BadRequest(message) if message.contains(needle)),
        "{case}: expected BadRequest containing {needle:?}, got {error:?}"
    );
}

/// Asserts the non-terminal capture refusal with `status`.
fn assert_capture_refusal(error: &EngError, case: &str, status: i32) {
    let EngError::Backend { operation, message } = error else {
        panic!("{case}: expected a Backend error, got {error:?}");
    };
    assert_eq!(*operation, "cuda capture", "{case}: {message}");
    assert!(
        message.ends_with(&format!("(status {status})")),
        "{case}: {message}"
    );
    assert!(
        !TERMINAL.load(Ordering::Acquire),
        "{case}: a listed status must not latch the backend"
    );
}

/// What a refused call answered inside a window: the call, the wording the refusal must carry, and
/// the answer.
type Answers = Vec<(&'static str, &'static str, String)>;

/// Records what a call inside a window answered: the `BadRequest` text, or a marker.
fn note<T>(log: &mut Answers, call: &'static str, what: &'static str, result: Result<T, EngError>) {
    let answer = match result {
        Ok(_) => "ACCEPTED".to_owned(),
        Err(EngError::BadRequest(message)) => message,
        Err(other) => format!("{other:?}"),
    };
    log.push((call, what, answer));
}

/// One pack and matmul of `m` tokens against `rows` rows of a weight, on one stream.
struct Rig {
    ctx: CudaContext,
    stream: DeviceStream,
    weight: TiledWeight,
    x_buffer: DeviceBuffer,
    xf: DeviceBuffer,
    y: DeviceBuffer,
    x: Vec<f32>,
    need: native::Activations,
    rows: usize,
    k: usize,
    m: usize,
}

impl Rig {
    fn new(ctx: CudaContext, ty: GgufType, rows: usize, k: usize, m: usize) -> Self {
        let stream = ctx.stream().expect("stream");
        let weight = testkit::upload_weight(&ctx, ty, rows, k);
        let x = testkit::input(m, k);
        let need = native::activation_bytes(rows, k, m).expect("sizes of the call");
        let x_buffer = testkit::upload_f32(&ctx, &x);
        let xf = testkit::upload_filled(&ctx, 0xAA, need.xf);
        let y = testkit::upload_filled(&ctx, 0xFF, need.y);
        Self {
            ctx,
            stream,
            weight,
            x_buffer,
            xf,
            y,
            x,
            need,
            rows,
            k,
            m,
        }
    }

    fn standard(ty: GgufType, rows: usize, k: usize, m: usize) -> Self {
        Self::new(CudaContext::new().expect("context"), ty, rows, k, m)
    }

    /// The pack and the matmul of all rows, queued on the rig's stream.
    fn pack_matmul(&self) -> Result<(), EngError> {
        self.ctx.pack(
            &self.stream,
            PackCall {
                x: self.x_buffer.range(0, self.need.x)?,
                xf: self.xf.range(0, self.need.xf)?,
                m: self.m,
                k: self.k,
            },
        )?;
        self.ctx.matmul(
            &self.stream,
            Projection {
                weight: &self.weight,
                first_row: 0,
                rows: self.rows,
            },
            MatmulIo {
                xf: self.xf.range(0, self.need.xf)?,
                y: self.y.range(0, self.need.y)?,
                m: self.m,
            },
        )
    }

    /// Runs the body once eagerly (it loads every kernel module), then restores the sentinel.
    fn warm(&self) {
        self.pack_matmul().expect("warm-up launches");
        self.stream.synchronize().expect("drain the warm-up");
        self.fill_y();
    }

    /// Fills `y` with the 0xFF sentinel (NaN floats), so a value that was written is visible.
    fn fill_y(&self) {
        self.y
            .write_at(&self.stream, 0, &vec![0xFF; self.need.y])
            .expect("restore the sentinel");
    }

    /// Warms, then captures the pack and the matmul.
    fn capture(&self) -> GraphExec<'static> {
        self.warm();
        let (graph, ()) = self
            .ctx
            .capture(&self.stream, no_pins(), || self.pack_matmul())
            .expect("capture of the pack and the matmul");
        graph
    }

    fn y_bytes(&self) -> Vec<u8> {
        self.stream.synchronize().expect("drain the stream");
        testkit::read_bytes(&self.ctx, &self.y, self.need.y)
    }

    fn y_bits(&self) -> Vec<u32> {
        bits(&testkit::floats(&self.y_bytes()))
    }

    fn blocking(&self, x: &[f32]) -> Vec<u32> {
        bits(&testkit::blocking(
            &self.ctx,
            &self.weight,
            (0, self.rows),
            x,
        ))
    }

    fn set_x(&self, x: &[f32]) {
        let bytes: Vec<u8> = x.iter().flat_map(|value| value.to_le_bytes()).collect();
        self.x_buffer
            .write_at(&self.stream, 0, &bytes)
            .expect("rewrite x");
    }
}

#[test]
fn capture_statuses_form_the_closed_list() {
    for status in 0..=1100 {
        let flag = AtomicBool::new(false);
        let _ = native_error(&flag, "cuda capture", status);
        let listed = (900..=906).contains(&status) || status == 2;
        assert_eq!(
            flag.load(Ordering::Acquire),
            !listed,
            "status {status} at the capture stage"
        );
    }
    for status in [-1, -2] {
        let flag = AtomicBool::new(false);
        let _ = native_error(&flag, "cuda capture", status);
        assert!(flag.load(Ordering::Acquire), "adapter status {status}");
    }
    let flag = AtomicBool::new(false);
    let _ = native_error(&flag, "cuda launch", 901);
    assert!(
        flag.load(Ordering::Acquire),
        "901 outside the capture operation"
    );
    for status in [907, 908] {
        let flag = AtomicBool::new(false);
        let _ = native_error(&flag, "cuda capture", status);
        assert!(
            flag.load(Ordering::Acquire),
            "status {status} stays terminal"
        );
    }
    assert_eq!(
        stage_operation(resident::graph::STAGE_CAPTURE),
        "cuda capture"
    );
}

/// Nodes of a captured pack and matmul: the pack zeroes xf first for `m <= 16` (a memset and a
/// kernel), and the matmul of a shape that splits along k zeroes y first (a 2D memset and a
/// kernel).
fn expected_nodes(m: usize, splits: bool) -> usize {
    let pack = if m <= 16 { 2 } else { 1 };
    pack + if splits { 2 } else { 1 }
}

#[test]
fn captured_pack_and_matmul_replay_to_the_same_bytes_as_the_blocking_path() {
    if !testkit::device_present(
        "captured_pack_and_matmul_replay_to_the_same_bytes_as_the_blocking_path",
    ) {
        return;
    }
    let shapes = [
        (GgufType::Q4K, 512),
        (GgufType::Q6K, 512),
        (GgufType::Q80, 256),
        (GgufType::Q4K, 4096),
    ];
    for (ty, k) in shapes {
        let splits = k == 4096;
        for m in [1, 8, 16, 17, 33, 65] {
            if splits && m > 16 {
                continue;
            }
            let case = format!("{ty:?} k={k} m={m}");
            let rig = Rig::standard(ty, 64, k, m);
            let graph = rig.capture();
            assert_eq!(rig.y_bytes(), vec![0xFF; rig.need.y], "{case}: capture ran");
            assert_eq!(graph.nodes(), expected_nodes(m, splits), "{case}: nodes");
            graph.launch(&rig.stream).expect("first replay");
            assert_eq!(rig.y_bits(), rig.blocking(&rig.x), "{case}: first replay");
            let first = rig.y_bits();
            let changed: Vec<f32> = rig.x.iter().map(|value| -value).collect();
            rig.set_x(&changed);
            graph.launch(&rig.stream).expect("second replay");
            assert_eq!(
                rig.y_bits(),
                rig.blocking(&changed),
                "{case}: second replay"
            );
            assert_ne!(rig.y_bits(), first, "{case}: the replay ignored the new x");
        }
    }
}

/// A stack of experts, the ids that pick one, and one batch of `m` tokens.
struct ExpertRig {
    rig: Rig,
    ids: DeviceBuffer,
}

impl ExpertRig {
    fn new(m: usize, ids: &[i32]) -> Self {
        let ctx = CudaContext::new().expect("context");
        let rig = Rig::new(ctx, GgufType::Q4K, EXPERT_ROWS * EXPERTS, 512, m);
        let mut bytes: Vec<u8> = ids.iter().flat_map(|id| id.to_le_bytes()).collect();
        bytes.resize(IDS_BYTES, 0);
        let ids = testkit::upload_bytes(&rig.ctx, &bytes);
        Self { rig, ids }
    }

    fn need(&self) -> native::Activations {
        native::activation_bytes(EXPERT_ROWS, self.rig.k, self.rig.m).expect("sizes of the call")
    }

    /// The pack, then the matmul of expert `ids[slot]` into `y`.
    fn expert_body(&self, slot: usize, y: &DeviceBuffer) -> Result<(), EngError> {
        let rig = &self.rig;
        let need = self.need();
        rig.ctx.pack(
            &rig.stream,
            PackCall {
                x: rig.x_buffer.range(0, need.x)?,
                xf: rig.xf.range(0, need.xf)?,
                m: rig.m,
                k: rig.k,
            },
        )?;
        rig.ctx.matmul_expert(
            &rig.stream,
            ExpertCall {
                stack: ExpertStack {
                    weight: &rig.weight,
                    experts: EXPERTS,
                },
                select: ExpertSelect {
                    ids: self.ids.range(0, 16)?,
                    slot,
                    count: EXPERTS,
                },
                io: MatmulIo {
                    xf: rig.xf.range(0, need.xf)?,
                    y: y.range(0, need.y)?,
                    m: rig.m,
                },
            },
        )
    }

    /// The direct matmul of expert `id`'s rows, drained and read back.
    fn direct(&self, id: usize) -> Vec<u32> {
        let rig = &self.rig;
        let need = self.need();
        let y = testkit::upload_filled(&rig.ctx, 0xFF, need.y);
        rig.ctx
            .matmul(
                &rig.stream,
                Projection {
                    weight: &rig.weight,
                    first_row: id * EXPERT_ROWS,
                    rows: EXPERT_ROWS,
                },
                MatmulIo {
                    xf: rig.xf.range(0, need.xf).expect("xf range"),
                    y: y.range(0, need.y).expect("y range"),
                    m: rig.m,
                },
            )
            .expect("direct matmul");
        rig.stream.synchronize().expect("drain");
        bits(&testkit::floats(&testkit::read_bytes(&rig.ctx, &y, need.y)))
    }
}

#[test]
fn captured_matmul_expert_follows_the_ids_between_replays() {
    if !testkit::device_present("captured_matmul_expert_follows_the_ids_between_replays") {
        return;
    }
    for m in [1, 8] {
        let case = format!("m={m}");
        let rig = ExpertRig::new(m, &[2, 5]);
        let y = testkit::upload_filled(&rig.rig.ctx, 0xFF, rig.need().y);
        rig.expert_body(1, &y).expect("warm-up");
        rig.rig.stream.synchronize().expect("drain the warm-up");
        y.write_at(&rig.rig.stream, 0, &vec![0xFF; rig.need().y])
            .expect("restore the sentinel");
        let (graph, ()) = rig
            .rig
            .ctx
            .capture(&rig.rig.stream, no_pins(), || rig.expert_body(1, &y))
            .expect("capture of the expert matmul");
        assert_eq!(graph.nodes(), 3, "{case}: pack (2) + expert (1)");
        let read = |rig: &ExpertRig| {
            rig.rig.stream.synchronize().expect("drain");
            bits(&testkit::floats(&testkit::read_bytes(
                &rig.rig.ctx,
                &y,
                rig.need().y,
            )))
        };
        assert_eq!(
            read(&rig),
            vec![u32::MAX; rig.need().y / 4],
            "{case}: capture ran"
        );
        graph.launch(&rig.rig.stream).expect("first replay");
        let first = read(&rig);
        assert_eq!(first, rig.direct(5), "{case}: expert 5");
        let ids: Vec<u8> = [7i32, 9].iter().flat_map(|id| id.to_le_bytes()).collect();
        rig.ids
            .write_at(&rig.rig.stream, 0, &ids)
            .expect("rewrite the ids");
        graph.launch(&rig.rig.stream).expect("second replay");
        let second = read(&rig);
        assert_eq!(second, rig.direct(9), "{case}: expert 9");
        assert_ne!(second, first, "{case}: the replay kept the old id");
    }
}

/// Two streams, two events and the buffers of a pack with two half-matmuls.
struct ForkRig {
    rig: Rig,
    branch: DeviceStream,
    fork: DeviceEvent,
    join: DeviceEvent,
    y_a: DeviceBuffer,
    y_b: DeviceBuffer,
}

impl ForkRig {
    fn new() -> Self {
        let rig = Rig::standard(GgufType::Q4K, 16, 512, 1);
        let branch = rig.ctx.stream().expect("branch stream");
        let fork = rig.ctx.event().expect("fork event");
        let join = rig.ctx.event().expect("join event");
        let half = native::activation_bytes(8, 512, 1).expect("sizes").y;
        let y_a = testkit::upload_filled(&rig.ctx, 0xFF, half);
        let y_b = testkit::upload_filled(&rig.ctx, 0xFF, half);
        Self {
            rig,
            branch,
            fork,
            join,
            y_a,
            y_b,
        }
    }

    fn half(
        &self,
        first_row: usize,
        y: &DeviceBuffer,
        stream: &DeviceStream,
    ) -> Result<(), EngError> {
        let rig = &self.rig;
        rig.ctx.matmul(
            stream,
            Projection {
                weight: &rig.weight,
                first_row,
                rows: 8,
            },
            MatmulIo {
                xf: rig.xf.range(0, rig.need.xf)?,
                y: y.range(0, y.bytes())?,
                m: rig.m,
            },
        )
    }

    /// Pack on A, fork to B for the upper half, join back, lower half on A. `joined` false leaves
    /// the branch unjoined.
    fn body(&self, joined: bool) -> Result<(), EngError> {
        let rig = &self.rig;
        rig.ctx.pack(
            &rig.stream,
            PackCall {
                x: rig.x_buffer.range(0, rig.need.x)?,
                xf: rig.xf.range(0, rig.need.xf)?,
                m: rig.m,
                k: rig.k,
            },
        )?;
        rig.stream.record(&self.fork)?;
        self.branch.wait(&self.fork)?;
        self.half(8, &self.y_b, &self.branch)?;
        self.branch.record(&self.join)?;
        if joined {
            rig.stream.wait(&self.join)?;
        }
        self.half(0, &self.y_a, &rig.stream)
    }

    /// Runs the body eagerly (loads the modules) and drains both streams.
    fn warm(&self) {
        self.body(true).expect("warm-up");
        self.rig.stream.synchronize().expect("drain A");
        self.branch.synchronize().expect("drain B");
    }

    /// Fills both outputs with the 0xFF sentinel, so only a launch that ran can change them.
    fn fill_outputs(&self) {
        for buffer in [&self.y_a, &self.y_b] {
            buffer
                .write_at(&self.rig.stream, 0, &vec![0xFF; buffer.bytes()])
                .expect("restore the sentinel");
        }
    }

    /// The bits of the sentinel in both outputs.
    fn sentinel(&self) -> (Vec<u32>, Vec<u32>) {
        let bits_of = |buffer: &DeviceBuffer| vec![u32::MAX; buffer.bytes() / 4];
        (bits_of(&self.y_a), bits_of(&self.y_b))
    }

    fn halves(&self) -> (Vec<u32>, Vec<u32>) {
        self.rig.stream.synchronize().expect("drain A");
        self.branch.synchronize().expect("drain B");
        let read = |buffer: &DeviceBuffer| {
            bits(&testkit::floats(&testkit::read_bytes(
                &self.rig.ctx,
                buffer,
                buffer.bytes(),
            )))
        };
        (read(&self.y_a), read(&self.y_b))
    }
}

#[test]
fn fork_and_join_branches_run_in_one_graph() {
    if !testkit::device_present("fork_and_join_branches_run_in_one_graph") {
        return;
    }
    let fork = ForkRig::new();
    fork.warm();
    fork.fill_outputs();
    let (graph, ()) = fork
        .rig
        .ctx
        .capture(&fork.rig.stream, no_pins(), || fork.body(true))
        .expect("capture of the fork and join");
    assert_eq!(graph.nodes(), 4, "pack (2) + two matmuls");
    assert_eq!(fork.halves(), fork.sentinel(), "the capture ran nothing");
    graph.launch(&fork.rig.stream).expect("replay");
    let (y_a, y_b) = fork.halves();
    let ctx = &fork.rig.ctx;
    let weight = &fork.rig.weight;
    assert_eq!(
        y_a,
        bits(&testkit::blocking(ctx, weight, (0, 8), &fork.rig.x))
    );
    assert_eq!(
        y_b,
        bits(&testkit::blocking(ctx, weight, (8, 8), &fork.rig.x))
    );
}

#[test]
fn an_unjoined_branch_is_a_non_terminal_capture_refusal_and_the_stream_recovers() {
    if !testkit::device_present(
        "an_unjoined_branch_is_a_non_terminal_capture_refusal_and_the_stream_recovers",
    ) {
        return;
    }
    let fork = ForkRig::new();
    fork.warm();
    let live = graphs();
    let error = refusal(
        fork.rig
            .ctx
            .capture(&fork.rig.stream, no_pins(), || fork.body(false)),
        "capture with an unjoined branch",
    );
    assert_capture_refusal(&error, "unjoined branch", 904);
    assert!(!window_open(), "the window is closed");
    assert_eq!(deferred_len(), 0);
    assert_eq!(graphs(), live, "no graph was made");
    assert!(
        !fork.rig.stream.is_capturing().expect("query A"),
        "A left capture"
    );
    assert!(
        !fork.branch.is_capturing().expect("query B"),
        "B left capture"
    );
    fork.fill_outputs();
    fork.body(true).expect("eager launches after the refusal");
    let (y_a, y_b) = fork.halves();
    let (ctx, weight) = (&fork.rig.ctx, &fork.rig.weight);
    assert_eq!(
        y_a,
        bits(&testkit::blocking(ctx, weight, (0, 8), &fork.rig.x))
    );
    assert_eq!(
        y_b,
        bits(&testkit::blocking(ctx, weight, (8, 8), &fork.rig.x))
    );
    let (graph, ()) = fork
        .rig
        .ctx
        .capture(&fork.rig.stream, no_pins(), || fork.body(true))
        .expect("a correct capture on the same streams");
    assert_eq!(graph.nodes(), 4);
}

#[test]
fn a_refused_capture_is_observable_and_the_eager_rerun_succeeds() {
    if !testkit::device_present("a_refused_capture_is_observable_and_the_eager_rerun_succeeds") {
        return;
    }
    let fork = ForkRig::new();
    fork.warm();
    let error = refusal(
        fork.rig
            .ctx
            .capture(&fork.rig.stream, no_pins(), || fork.body(false)),
        "capture with an unjoined branch",
    );
    assert!(
        is_capture_refusal(&error),
        "a listed status is a refusal: {error:?}"
    );
    let other = [
        EngError::BadRequest("boom".to_owned()),
        EngError::Backend {
            operation: "cuda synchronize",
            message: "x (status 900)".to_owned(),
        },
    ];
    for error in &other {
        assert!(
            !is_capture_refusal(error),
            "not a capture refusal: {error:?}"
        );
    }
    // The documented side effect: the join event was recorded inside the failed window.
    assert_bad_request(
        &refusal(fork.join.synchronize(), "host wait before the rerun"),
        "marked event",
        "recorded inside a capture",
    );
    fork.fill_outputs();
    fork.body(true).expect("the eager rerun of the same body");
    fork.join
        .synchronize()
        .expect("the rerun re-recorded the join event");
    fork.fork.synchronize().expect("and the fork event");
    let (y_a, y_b) = fork.halves();
    let (ctx, weight) = (&fork.rig.ctx, &fork.rig.weight);
    assert_eq!(
        y_a,
        bits(&testkit::blocking(ctx, weight, (0, 8), &fork.rig.x))
    );
    assert_eq!(
        y_b,
        bits(&testkit::blocking(ctx, weight, (8, 8), &fork.rig.x))
    );
}

/// Everything the refusal test needs from a window: made before it, used inside it.
struct Window {
    rig: Rig,
    buffer: DeviceBuffer,
    event: DeviceEvent,
    word: crate::kernels::cuda::PinnedWord,
    earlier: GraphExec<'static>,
}

impl Window {
    fn new() -> Self {
        let rig = Rig::standard(GgufType::Q4K, 64, 512, 1);
        let buffer = testkit::upload_filled(&rig.ctx, 0, 64);
        let event = rig.ctx.event().expect("event");
        let word = rig.ctx.pinned_word().expect("pinned word");
        let earlier = rig.capture();
        Self {
            rig,
            buffer,
            event,
            word,
            earlier,
        }
    }
}

#[test]
fn non_launch_calls_are_refused_inside_the_window() {
    if !testkit::device_present("non_launch_calls_are_refused_inside_the_window") {
        return;
    }
    let w = Window::new();
    let ctx = &w.rig.ctx;
    let stream = &w.rig.stream;
    let one_row = testkit::input(1, 512);
    let (destroys, bytes, owners) = (ctx.destroy_calls(), ctx.live_bytes(), ctx.live_owners());
    let log_cell = std::cell::RefCell::new(Vec::new());
    let result = ctx.capture(stream, no_pins(), || {
        let mut log = log_cell.borrow_mut();
        let log = &mut *log;
        {
            w.rig.pack_matmul()?;
            note(log, "upload", "an upload", ctx.upload(&[1u8; 16]));
            note(
                log,
                "stream synchronize",
                "a stream synchronize",
                stream.synchronize(),
            );
            note(
                log,
                "event synchronize",
                "an event synchronize",
                w.event.synchronize(),
            );
            note(
                log,
                "linear_rows",
                "a blocking linear call",
                ctx.linear_rows(&w.rig.weight, 0, 8, &one_row),
            );
            note(log, "stream", "a stream creation", ctx.stream());
            note(log, "event", "an event creation", ctx.event());
            note(
                log,
                "pinned_word",
                "a pinned word allocation",
                ctx.pinned_word(),
            );
            note(log, "allocate", "an allocation", ctx.allocate(64));
            note(log, "memory_info", "a memory query", ctx.memory_info());
            note(
                log,
                "read_at",
                "a ranged copy",
                w.buffer.read_at(stream, 0, &mut [0u8; 4]),
            );
            note(
                log,
                "write_at",
                "a ranged copy",
                w.buffer.write_at(stream, 0, &[0u8; 4]),
            );
            note(
                log,
                "download",
                "a download",
                ctx.download(&w.buffer, &mut [0u8; 4]),
            );
            note(
                log,
                "copy_word_to_host",
                "a word copy",
                ctx.copy_word_to_host(stream, w.buffer.range(0, 8)?, &w.word),
            );
            note(
                log,
                "graph launch",
                "a graph launch",
                w.earlier.launch(stream),
            );
            note(
                log,
                "nested capture",
                "a nested capture",
                ctx.capture(stream, no_pins(), || Ok(())).map(|_| ()),
            );
        }
        Ok(())
    });
    let log = log_cell.into_inner();
    assert_eq!(log.len(), 15, "every call was made");
    for (call, what, answer) in &log {
        let wanted = format!("cuda capture: {what} is {WINDOW_TEXT}");
        assert!(answer.contains(&wanted), "{call}: {answer}");
    }
    let (graph, ()) = result.expect("the capture itself succeeds");
    assert_eq!(ctx.destroy_calls(), destroys, "no native destroy");
    assert_eq!(ctx.live_bytes(), bytes, "no allocation");
    let now = ctx.live_owners();
    assert_eq!(
        (now.streams, now.events, now.pinned, now.graphs),
        (
            owners.streams,
            owners.events,
            owners.pinned,
            owners.graphs + 1
        ),
        "no owner made but the graph itself"
    );
    assert!(!window_open());
    assert_eq!(graph.nodes(), 3, "pack (2) + matmul (1) were recorded");
    drop(graph);
}

#[test]
fn a_workspace_grow_is_refused_inside_the_window_and_the_workspace_is_unchanged() {
    if !testkit::device_present(
        "a_workspace_grow_is_refused_inside_the_window_and_the_workspace_is_unchanged",
    ) {
        return;
    }
    let small = native::activation_bytes(64, 512, 1).expect("small workspace");
    let larger = native::activation_bytes(64, 512, 33).expect("larger call");
    let ctx = CudaContext::with_workspace(small).expect("context with a small workspace");
    let rig = Rig::new(ctx, GgufType::Q4K, 64, 512, 1);
    rig.warm();
    let (allocations, bytes) = (rig.ctx.workspace_allocations(), rig.ctx.workspace_bytes());
    let (graph, answer) = rig
        .ctx
        .capture(&rig.stream, no_pins(), || {
            rig.pack_matmul()?;
            let mut workspace = rig.ctx.workspace.borrow_mut();
            Ok(rig.ctx.reserve_workspace(&mut workspace, larger))
        })
        .expect("the capture itself succeeds");
    drop(graph);
    assert_bad_request(
        &refusal(answer, "grow"),
        "grow",
        "a workspace grow is not allowed inside a capture window",
    );
    assert_eq!(rig.ctx.workspace_allocations(), allocations);
    assert_eq!(rig.ctx.workspace_bytes(), bytes);
    let mut workspace = rig.ctx.workspace.borrow_mut();
    rig.ctx
        .reserve_workspace(&mut workspace, larger)
        .expect("the same grow succeeds after the window");
    drop(workspace);
    assert_eq!(rig.ctx.workspace_allocations(), allocations + 1);
}

#[test]
fn drops_inside_the_window_wait_for_its_end() {
    if !testkit::device_present("drops_inside_the_window_wait_for_its_end") {
        return;
    }
    let w = Window::new();
    let ctx = &w.rig.ctx;
    let doomed_buffer = testkit::upload_filled(ctx, 0, 128);
    let doomed_stream = ctx.stream().expect("stream");
    let doomed_event = ctx.event().expect("event");
    let doomed_word = ctx.pinned_word().expect("word");
    let doomed_graph = w.rig.capture();
    let (graphs_before, bytes, owners, destroys) = (
        graphs(),
        ctx.live_bytes(),
        ctx.live_owners(),
        ctx.destroy_calls(),
    );
    let inside = std::cell::Cell::new(None);
    let result = ctx.capture(&w.rig.stream, no_pins(), || {
        w.rig.pack_matmul()?;
        drop(doomed_buffer);
        drop(doomed_stream);
        drop(doomed_event);
        drop(doomed_word);
        drop(doomed_graph);
        inside.set(Some((
            ctx.live_bytes(),
            ctx.live_owners(),
            graphs(),
            deferred_len(),
        )));
        Ok(())
    });
    assert_eq!(
        inside.get(),
        Some((bytes, owners, graphs_before, 5)),
        "nothing freed inside"
    );
    let (graph, ()) = result.expect("the capture itself succeeds");
    assert_eq!(ctx.live_bytes(), bytes - 128, "the buffer was freed after");
    let after = ctx.live_owners();
    assert_eq!(after.streams, owners.streams - 1);
    assert_eq!(after.events, owners.events - 1);
    assert_eq!(after.pinned, owners.pinned - 8);
    assert_eq!(
        graphs(),
        graphs_before,
        "one graph dropped, the new one made"
    );
    assert_eq!(
        ctx.destroy_calls(),
        destroys + 4,
        "stream, event, word, graph"
    );
    drop(graph);
    assert_eq!(deferred_len(), 0);
}

#[test]
fn an_event_recorded_inside_a_window_is_refused_for_host_waits_afterwards() {
    if !testkit::device_present(
        "an_event_recorded_inside_a_window_is_refused_for_host_waits_afterwards",
    ) {
        return;
    }
    let w = Window::new();
    let ctx = &w.rig.ctx;
    let stream = &w.rig.stream;
    let other = ctx.stream().expect("other stream");
    let eager = ctx.event().expect("eager event");
    stream.record(&eager).expect("eager record");
    ctx.copy_word_to_host(stream, w.buffer.range(0, 8).expect("range"), &w.word)
        .expect("eager word copy");
    let (graph, ()) = ctx
        .capture(stream, no_pins(), || {
            w.rig.pack_matmul()?;
            stream.record(&w.event)
        })
        .expect("capture with a record");
    drop(graph);
    let sync = refusal(w.event.synchronize(), "event synchronize");
    let wait = refusal(other.wait(&w.event), "wait outside a window");
    let read = refusal(w.word.read_after(&w.event), "read_after");
    for (case, error) in [
        ("synchronize", &sync),
        ("wait", &wait),
        ("read_after", &read),
    ] {
        assert_bad_request(error, case, "recorded inside a capture");
    }
    assert!(!TERMINAL.load(Ordering::Acquire), "no native call was made");
    eager
        .synchronize()
        .expect("an eager event still synchronizes");
    stream
        .record(&w.event)
        .expect("eager record clears the flag");
    w.event.synchronize().expect("the flag is clear");
    w.word
        .read_after(&w.event)
        .expect("the word reads after the copy");
}

#[test]
fn a_failing_body_ends_the_capture_cleanly() {
    if !testkit::device_present("a_failing_body_ends_the_capture_cleanly") {
        return;
    }
    let rig = Rig::standard(GgufType::Q4K, 64, 512, 1);
    rig.warm();
    let live = graphs();
    let error = refusal(
        rig.ctx.capture(&rig.stream, no_pins(), || {
            rig.pack_matmul()?;
            Err::<(), _>(EngError::BadRequest("boom".to_owned()))
        }),
        "failing body",
    );
    assert_eq!(error, EngError::BadRequest("boom".to_owned()));
    assert_eq!(graphs(), live, "no graph");
    assert!(!rig.stream.is_capturing().expect("query"), "left capture");
    assert_eq!(deferred_len(), 0);
    rig.pack_matmul().expect("an eager launch afterwards");
    assert_eq!(rig.y_bits(), rig.blocking(&rig.x), "and it ran");
}

#[test]
fn a_graph_keeps_its_arena_holds_until_it_is_dropped() {
    if !testkit::device_present("a_graph_keeps_its_arena_holds_until_it_is_dropped") {
        return;
    }
    let small = native::activation_bytes(64, 512, 1).expect("small workspace");
    let ctx = CudaContext::with_workspace(small).expect("context with a small workspace");
    let rig = Rig::new(ctx, GgufType::Q4K, 64, 512, 1);
    let arena = rig
        .ctx
        .arena(&[crate::kernels::cuda::ArenaTensor {
            bytes: 256,
            first: 0,
            last: 0,
        }])
        .expect("arena");
    let wide = testkit::input(33, 512);
    rig.warm();
    let plan = CapturePlan {
        pins: &[&rig.x_buffer, rig.weight.buffer()],
        holds: vec![rig.ctx.workspace_hold().expect("hold"), arena.hold()],
    };
    let (graph, ()) = rig
        .ctx
        .capture(&rig.stream, plan, || rig.pack_matmul())
        .expect("capture with holds");
    assert_eq!(arena.holds(), 1);
    let allocations = rig.ctx.workspace_allocations();
    let error = refusal(rig.ctx.linear(&rig.weight, &wide), "grow under a graph");
    assert_bad_request(&error, "grow under a graph", "retire, never free");
    assert_eq!(rig.ctx.workspace_allocations(), allocations);
    drop(graph);
    assert_eq!(arena.holds(), 0);
    rig.ctx
        .linear(&rig.weight, &wide)
        .expect("the same call grows the workspace");
    assert_eq!(rig.ctx.workspace_allocations(), allocations + 1);
}

#[test]
fn an_illegal_native_call_in_the_window_is_a_non_terminal_capture_refusal() {
    if !testkit::device_present(
        "an_illegal_native_call_in_the_window_is_a_non_terminal_capture_refusal",
    ) {
        return;
    }
    let rig = Rig::standard(GgufType::Q4K, 64, 512, 1);
    rig.warm();
    let error = refusal(
        rig.ctx.capture(&rig.stream, no_pins(), || {
            rig.pack_matmul()?;
            // The one deliberate illegal call: a synchronize of the capturing stream.
            let raw = rig.stream.raw()?;
            resident::stream_synchronize(raw).map_err(super::super::device_error)
        }),
        "illegal call",
    );
    assert_capture_refusal(&error, "synchronize of a capturing stream", 900);
    assert!(!rig.stream.is_capturing().expect("query"), "left capture");
    rig.pack_matmul().expect("an eager launch afterwards");
    assert_eq!(rig.y_bits(), rig.blocking(&rig.x), "and it ran");
}
