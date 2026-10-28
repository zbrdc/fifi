//! Raw-FFI refusal rows of the grouped launch (lane C's ruling on the part 1 question: every
//! C-side launch check of `mmq_mmid.cu` gets a row that reaches it with the Rust wrapper AND
//! the FFI mirror's own `check_mmid` out of the way, each row killing a single-check mutant of
//! the C source). The Rust side refuses first in production, so without this table a C check
//! could be deleted and nothing would notice.
//!
//! A row is the legal base call (4 experts, 4 tokens of 2 ids, Q4_K, K 256, N 128, `ld` 128, every
//! range in its own device allocation) with ONE edit: a geometry field, a capacity, or a
//! pointer. The capacities are first fitted to the edited geometry by the Rust mirror of the
//! sizing rules (`launch_needs`, built on G1's functions, independent of the C expressions), so a
//! row violates only its own check and satisfies every check after it: a C check deleted from
//! the source then lets the call through to the launch, and the row fails on `rc != -1`.
//! The rows follow the C order: `valid_grouped`, `valid_limits`, `valid_ranges`. They are
//! issued with no wrapper, through `mmq_mmid_launch_raw`.
//!
//! Not reachable by a row, and argued: `e >= 1` (subsumed by `ne / m <= e` with `ne / m >= 1`),
//! `nrows_x <= INT_MAX` (subsumed by `nrows_dst >= nrows_x` with `nrows_dst <= INT_MAX`), and
//! `nsm > 0` (the entry queries `nsm` itself).
//!
//! Device-test structure: a `Result`-only device phase issues every row without synchronizing
//! (a refused call queues nothing), the rows are asserted in order afterwards, and only when
//! all were refused does a second phase run the base call and read its output back: the HARD
//! GATE then asserts the base call's `dst` is finite (it is zero: the q8_1 buffer is zeros) and
//! that the refused rows left the NaN prefill untouched.

use std::ffi::{c_int, c_void};
use std::ptr;

use super::{MmidArgsRaw, MmidGeometry, launch_needs, mmq_mmid_launch_raw};
use crate::EngError;
use crate::kernels::cuda::native::STATUS_INVALID_ARGUMENT;
use crate::kernels::cuda::prefill::mmq_mmid::tests::device_facts;
use crate::kernels::cuda::testkit;
use crate::kernels::cuda::{CudaContext, DeviceBuffer, DeviceStream};
use crate::weights::GgufType;

/// Bytes of each device allocation behind a range: far above the base call's needs, so a row
/// whose check a mutant deletes launches inside its allocations as far as the geometry allows.
const ALLOC: usize = 16 << 20;
/// The NaN prefill of `dst`, as bytes.
const NAN: [u8; 4] = 0x7fc0_dead_u32.to_le_bytes();

/// The legal base geometry: 4 experts of 128 rows by K 256, 4 tokens, 2 ids, `ld` 128.
const BASE: MmidGeometry = MmidGeometry {
    type_x: 12,
    ncols_x: 256,
    nrows_x: 128,
    ncols_dst: 8,
    stride_row_x: 1,
    ncols_y: 8,
    nrows_dst: 128,
    nchannels_x: 4,
    nchannels_y: 4,
    stride_channel_x: 128,
    stride_channel_y: 0,
    stride_channel_dst: 0,
    nsamples_x: 1,
    nsamples_y: 1,
    stride_sample_x: 0,
    stride_sample_y: 0,
    stride_sample_dst: 0,
    ncols_max: 4,
    ncols_opt: 4,
};

/// Indexes of the six ranges, in the order x, y, ids_dst, expert_bounds, dst, fixup.
const X: usize = 0;
const Y: usize = 1;
const IDS: usize = 2;
const BOUNDS: usize = 3;
const DST: usize = 4;
const FIXUP: usize = 5;

/// The device allocations behind the six ranges, the base address of each, and the SM count the
/// entry will see.
struct Fx {
    bufs: [DeviceBuffer; 6],
    addr: [*mut c_void; 6],
    nsm: c_int,
}

/// The device address of the start of `buf`.
fn address(ctx: &CudaContext, buf: &DeviceBuffer) -> Result<*mut c_void, EngError> {
    let range = buf.range(0, buf.bytes())?;
    Ok(ctx.raw_range(&range)?.address())
}

impl Fx {
    fn new(ctx: &CudaContext) -> Result<Self, EngError> {
        // The one facts call of the tests of this series, shared with `prefill::mmq_mmid::tests`.
        let sms = device_facts(ctx)?.nsm;
        let nsm = c_int::try_from(sms)
            .map_err(|_| EngError::BadRequest(format!("the device reports {sms} SMs")))?;
        // Weights: random Q4_K blocks (4 experts x 128 rows, K 256) padded to the allocation.
        let mut weights = testkit::weight_source(GgufType::Q4K, 4 * 128, 256);
        weights.resize(ALLOC, 0);
        // Ids: compact column c -> token slot c, two columns per expert.
        let mut ids: Vec<u8> = (0..8i32).flat_map(i32::to_le_bytes).collect();
        ids.resize(1 << 20, 0);
        let mut bounds: Vec<u8> = [0i32, 2, 4, 6, 8]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        bounds.resize(1 << 16, 0);
        let nan: Vec<u8> = NAN.iter().copied().cycle().take(ALLOC / 2).collect();
        let bufs = [
            ctx.upload(&weights)?,
            ctx.upload(&vec![0u8; ALLOC / 2])?,
            ctx.upload(&ids)?,
            ctx.upload(&bounds)?,
            ctx.upload(&nan)?,
            ctx.upload(&vec![0u8; 6 << 20])?,
        ];
        let addr = [
            address(ctx, &bufs[X])?,
            address(ctx, &bufs[Y])?,
            address(ctx, &bufs[IDS])?,
            address(ctx, &bufs[BOUNDS])?,
            address(ctx, &bufs[DST])?,
            address(ctx, &bufs[FIXUP])?,
        ];
        Ok(Self { bufs, addr, nsm })
    }

    /// The call of the base geometry after `edit`, its capacities fitted to the edited geometry
    /// (the allocation sizes when the geometry has no sizing, e.g. a negative field).
    fn args(&self, edit: impl FnOnce(&mut MmidGeometry)) -> MmidArgsRaw {
        let mut g = BASE;
        edit(&mut g);
        let caps = launch_needs(&g, self.nsm).unwrap_or([ALLOC; 6]);
        MmidArgsRaw {
            x: self.addr[X].cast_const(),
            y: self.addr[Y].cast_const(),
            ids_dst: self.addr[IDS].cast_const(),
            expert_bounds: self.addr[BOUNDS].cast_const(),
            dst: self.addr[DST],
            fixup: self.addr[FIXUP],
            x_bytes: caps[0],
            y_bytes: caps[1],
            ids_dst_bytes: caps[2],
            expert_bounds_bytes: caps[3],
            dst_bytes: caps[4],
            fixup_bytes: caps[5],
            geometry: g,
        }
    }
}

/// One row: its name (the guard it reaches, with the C expression) and the call it issues; `None`
/// is the NULL `args` pointer.
struct Row {
    name: &'static str,
    call: Option<MmidArgsRaw>,
}

fn geo(fx: &Fx, name: &'static str, edit: impl FnOnce(&mut MmidGeometry)) -> Row {
    Row {
        name,
        call: Some(fx.args(edit)),
    }
}

fn then(fx: &Fx, name: &'static str, edit: impl FnOnce(&mut MmidArgsRaw)) -> Row {
    let mut args = fx.args(|_| {});
    edit(&mut args);
    Row {
        name,
        call: Some(args),
    }
}

/// Every row, in the order of `mmq_mmid.cu`.
fn rows(fx: &Fx) -> Vec<Row> {
    let a = &fx.addr;
    let at = |i: usize, bytes: usize| a[i].wrapping_byte_add(bytes);
    let mut rows = vec![Row {
        name: "entry: args == NULL",
        call: None,
    }];
    // valid_grouped
    rows.extend([
        geo(fx, "g1 ncols_opt == m", |g| g.ncols_opt = 5),
        geo(fx, "g2 m >= 1", |g| (g.ncols_max, g.ncols_opt) = (-1, -1)),
        geo(fx, "g3 m <= MAX_TOKENS", |g| {
            (g.ncols_max, g.ncols_opt, g.ncols_dst, g.ncols_y) = (2049, 2049, 4098, 4098);
        }),
        geo(fx, "g4 ncols_y == ncols_dst", |g| g.ncols_y = 9),
        geo(fx, "g5 ne >= m", |g| (g.ncols_dst, g.ncols_y) = (0, 0)),
        geo(fx, "g6 ne % m == 0", |g| (g.ncols_dst, g.ncols_y) = (9, 9)),
        geo(fx, "g7 nchannels_y == nchannels_x", |g| g.nchannels_y = 5),
        geo(fx, "g9 e <= MAX_EXPERTS", |g| {
            (g.nchannels_x, g.nchannels_y) = (65_536, 65_536)
        }),
        geo(fx, "g10 ne / m <= MAX_TOP_K", |g| {
            (g.ncols_dst, g.ncols_y, g.nchannels_x, g.nchannels_y) = (132, 132, 33, 33);
        }),
        geo(fx, "g11 ne / m <= e", |g| {
            (g.nchannels_x, g.nchannels_y) = (1, 1)
        }),
        geo(fx, "g12 type has an arm", |g| g.type_x = 8),
        geo(fx, "g13 k % 256 == 0", |g| g.ncols_x = 384),
        geo(fx, "g14 k >= 256", |g| (g.ncols_x, g.stride_row_x) = (0, 0)),
        geo(fx, "g15 k <= kMaxK", |g| {
            (g.ncols_x, g.stride_row_x, g.stride_channel_x) = (33_024, 129, 128 * 129);
        }),
        geo(fx, "g16 stride_row_x == k / 256", |g| g.stride_row_x = 2),
        geo(fx, "g17 nrows_x >= 1", |g| {
            (g.nrows_x, g.nrows_dst) = (0, 0)
        }),
        geo(fx, "g19 nsamples_x == 1", |g| g.nsamples_x = 2),
        geo(fx, "g20 nsamples_y == 1", |g| g.nsamples_y = 2),
        geo(fx, "g21 stride_sample_x == 0", |g| g.stride_sample_x = 1),
        geo(fx, "g22 stride_sample_y == 0", |g| g.stride_sample_y = 1),
        geo(fx, "g23 stride_sample_dst == 0", |g| {
            g.stride_sample_dst = 1
        }),
        geo(fx, "g24 stride_channel_y == 0", |g| g.stride_channel_y = 1),
        geo(fx, "g25 stride_channel_dst == 0", |g| {
            g.stride_channel_dst = 1
        }),
        geo(fx, "g26 nrows_dst >= nrows_x", |g| g.nrows_dst = 127),
        geo(fx, "g27 nrows_dst <= INT_MAX", |g| {
            (
                g.ncols_max,
                g.ncols_opt,
                g.ncols_dst,
                g.ncols_y,
                g.nrows_dst,
            ) = (1, 1, 1, 1, 1 << 31);
        }),
    ]);
    // valid_limits
    rows.extend([
        geo(fx, "l1 tiles <= i32::MAX / 100", |g| {
            (g.ncols_max, g.ncols_opt, g.ncols_dst, g.ncols_y) = (2048, 2048, 2048, 2048);
            (g.nchannels_x, g.nchannels_y, g.nrows_x, g.nrows_dst) = (256, 256, 700_032, 700_032);
            g.stride_channel_x = 700_032;
        }),
        geo(fx, "l2 tiles * k blocks < 2^30", |g| {
            (g.ncols_max, g.ncols_opt, g.ncols_dst, g.ncols_y) = (2048, 2048, 2048, 2048);
            (g.nchannels_x, g.nchannels_y, g.nrows_x, g.nrows_dst) = (256, 256, 32_768, 32_768);
            (g.ncols_x, g.stride_row_x, g.stride_channel_x) = (32_768, 128, 32_768 * 128);
        }),
        geo(fx, "l3 stride >= n * kb", |g| g.stride_channel_x = 127),
        geo(fx, "l4 stride <= INT_MAX", |g| {
            (g.nchannels_x, g.nchannels_y, g.ncols_dst, g.ncols_y) = (1, 1, 4, 4);
            g.stride_channel_x = 1 << 31;
        }),
        geo(fx, "l5 weight index <= INT_MAX", |g| {
            g.stride_channel_x = 1 << 30
        }),
        geo(fx, "l6 dst index <= INT_MAX", |g| g.nrows_dst = 1 << 30),
    ]);
    // valid_ranges: per range a NULL pointer, a capacity one byte short, and a misaligned pointer.
    rows.extend([
        then(fx, "x null", |c| c.x = ptr::null()),
        then(fx, "x capacity (blocks * block bytes)", |c| c.x_bytes -= 1),
        then(fx, "x alignment 16", |c| c.x = at(X, 4).cast_const()),
        then(fx, "y null", |c| c.y = ptr::null()),
        then(fx, "y capacity (rule M1 pad)", |c| c.y_bytes -= 1),
        then(fx, "y alignment 16", |c| c.y = at(Y, 4).cast_const()),
        then(fx, "ids_dst null", |c| c.ids_dst = ptr::null()),
        then(fx, "ids_dst capacity (rule M2, B2)", |c| {
            c.ids_dst_bytes -= 1
        }),
        then(fx, "ids_dst alignment 4", |c| {
            c.ids_dst = at(IDS, 2).cast_const()
        }),
        then(fx, "expert_bounds null", |c| c.expert_bounds = ptr::null()),
        then(fx, "expert_bounds capacity (n_experts + 1)", |c| {
            c.expert_bounds_bytes -= 1
        }),
        then(fx, "expert_bounds alignment 4", |c| {
            c.expert_bounds = at(BOUNDS, 2).cast_const()
        }),
        then(fx, "dst null", |c| c.dst = ptr::null_mut()),
        then(fx, "dst capacity", |c| c.dst_bytes -= 1),
        then(fx, "dst alignment 4", |c| c.dst = at(DST, 2)),
        then(fx, "fixup null", |c| c.fixup = ptr::null_mut()),
        then(fx, "fixup capacity (nsm * 128 * 128 floats)", |c| {
            c.fixup_bytes -= 1
        }),
        then(fx, "fixup alignment 16", |c| c.fixup = at(FIXUP, 4)),
        then(fx, "dst over x", |c| c.dst = a[X]),
        then(fx, "dst over y", |c| c.dst = a[Y]),
        then(fx, "dst over ids_dst", |c| c.dst = a[IDS]),
        then(fx, "dst over expert_bounds", |c| c.dst = a[BOUNDS]),
        then(fx, "fixup over x", |c| c.fixup = a[X]),
        then(fx, "fixup over y", |c| c.fixup = a[Y]),
        then(fx, "fixup over ids_dst", |c| c.fixup = a[IDS]),
        then(fx, "fixup over expert_bounds", |c| c.fixup = a[BOUNDS]),
        then(fx, "fixup over dst", |c| c.fixup = a[DST]),
    ]);
    rows
}

/// Issues one raw call and returns the entry's status.
fn issue(
    ctx: &CudaContext,
    stream: &DeviceStream,
    call: Option<&MmidArgsRaw>,
) -> Result<c_int, EngError> {
    let args = call.map_or(ptr::null(), ptr::from_ref);
    ctx.launch_on(stream, |raw| {
        let mut stage = 0;
        // SAFETY: the pointers of `call` address live device allocations of `Fx` (or are NULL, or
        // deliberately misaligned, which the entry refuses before any CUDA call); `args` is a
        // live struct or NULL, which the entry checks. A refused call queues nothing.
        Ok(unsafe { mmq_mmid_launch_raw(args, raw.as_ptr(), &mut stage) })
    })
}

/// What the refusal phase gave: each row's name and status, and `dst` after the rows when all
/// were refused.
type Refusals = (Vec<(&'static str, c_int)>, Option<Vec<u8>>);

fn refusal_phase() -> Result<Refusals, EngError> {
    let ctx = CudaContext::new()?;
    let stream = ctx.stream()?;
    let fx = Fx::new(&ctx)?;
    let mut statuses = Vec::new();
    for row in rows(&fx) {
        println!("raw row {}", row.name);
        statuses.push((row.name, issue(&ctx, &stream, row.call.as_ref())?));
    }
    let refused = statuses
        .iter()
        .all(|&(_, rc)| rc == STATUS_INVALID_ARGUMENT);
    let dst = if refused {
        let mut bytes = vec![0u8; ALLOC / 2];
        ctx.download(&fx.bufs[DST], &mut bytes)?;
        Some(bytes)
    } else {
        None
    };
    Ok((statuses, dst))
}

/// The base call, run and read back: its status and `dst`.
fn accept_phase() -> Result<(c_int, Vec<u8>), EngError> {
    let ctx = CudaContext::new()?;
    let stream = ctx.stream()?;
    let fx = Fx::new(&ctx)?;
    let base = fx.args(|_| {});
    let rc = issue(&ctx, &stream, Some(&base))?;
    stream.synchronize()?;
    let mut bytes = vec![0u8; ALLOC / 2];
    ctx.download(&fx.bufs[DST], &mut bytes)?;
    Ok((rc, bytes))
}

#[test]
fn mmq_mmid_raw_ffi_refuses_every_c_side_launch_check() {
    if !testkit::device_present("mmq_mmid_raw_ffi_refuses_every_c_side_launch_check") {
        return;
    }
    let (statuses, dst) = refusal_phase().expect("the refusal phase");
    for (name, rc) in &statuses {
        assert_eq!(
            *rc, STATUS_INVALID_ARGUMENT,
            "raw row {name}: expected INVALID_ARGUMENT, got {rc}"
        );
    }
    println!("{} raw rows, every one refused", statuses.len());
    // A refused call queued nothing: dst is still the NaN prefill.
    let dst = dst.expect("all rows refused");
    assert!(
        dst.as_chunks::<4>().0.iter().all(|c| *c == NAN),
        "raw rows: a refused call wrote into dst"
    );
    // The base call is accepted and runs: the zero q8_1 buffer gives zero outputs (the HARD GATE
    // reads the 8 rows of 128 floats, all written, all finite).
    let (rc, out) = accept_phase().expect("the accept phase");
    assert_eq!(rc, 0, "raw base call: expected success, got {rc}");
    let floats: Vec<f32> = out
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect();
    assert!(
        floats[..8 * 128].iter().all(|v| v.is_finite()),
        "HARD GATE (raw base call): dst holds a non-finite value (an unwritten output)"
    );
    assert!(
        floats[..8 * 128].iter().all(|&v| v == 0.0),
        "raw base call: dst is not zero"
    );
    assert!(
        floats[8 * 128..].iter().all(|v| v.to_bits() == 0x7fc0_dead),
        "raw base call wrote past dst"
    );
}
