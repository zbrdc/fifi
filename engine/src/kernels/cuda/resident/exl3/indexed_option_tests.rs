//! The opt-in options of the id-indexed launch ([`IndexedOptions`]) against the default launch.
//!
//! - The switches parse strictly; an option the single-row kernel cannot run with is refused at
//!   the session start; the load line names exactly the options that are on.
//! - The DP4A decode of book 2 must give the bits `decode_state` gives for every one of the 65536
//!   states, and both must equal the host reference (`Codebook::Mul1`).
//! - Its byte sum must be exact for every byte value in every lane, with the 0x6400 offset and
//!   nothing above bit 15.
//! - Every option must write the bytes of the default launch of the same 20 calls: rotated
//!   inputs, partials, outputs and flag, for every codebook, several rates and an unaligned bank.
//! - An id outside the bank in the tail pairs (16 to 19, the 20-pair launch's own) skips exactly
//!   those pairs and raises the flag.
//!
//! The device tests set the options on their context, which wins over the environment. A host
//! without a device skips them.

use std::ffi::OsStr;

use super::{
    DP4A_VAR, Exl3IndexedCall, Exl3LinearCall, INDEXED_CAPACITY, INDEXED_PAIRS_VAR, IndexedOptions,
    PIPE_VAR, SINGLE_GATHER_VAR, SingleRow, TABLE_VAR, check_indexed_with, dp4a_from_env,
    indexed_from_env, indexed_pairs_from_env, merge_indexed, pipe_from_env, table_from_env,
};
use crate::kernels::cuda::native::resident::exl3::TABLE_PAIRS_MAX;
use crate::kernels::cuda::{CudaContext, DeviceBuffer, DeviceRange, DeviceStream, testkit};
use crate::weights::exl3::Codebook;

/// The multiplier of book 2's hash (`decode_state`).
const MUL1: u32 = 0x83dc_d12d;
const PAIRS: usize = 20;
const OFF: IndexedOptions = IndexedOptions::OFF;
const DP4A: IndexedOptions = IndexedOptions::DP4A;
const PAIRS20: IndexedOptions = IndexedOptions::PAIRS20;
const BOTH: IndexedOptions = IndexedOptions::BOTH;
const PIPE: IndexedOptions = IndexedOptions::PIPE;
const PAIRS20_PIPE: IndexedOptions = IndexedOptions::PAIRS20_PIPE;
const BOTH_PIPE: IndexedOptions = IndexedOptions::BOTH_PIPE;
const TABLE: IndexedOptions = IndexedOptions::TABLE;
const TABLE_ALL: IndexedOptions = IndexedOptions::TABLE_ALL;

#[test]
fn the_switches_parse_strictly() {
    let parse =
        |f: fn(Option<&OsStr>) -> Result<Option<bool>, String>, v: &str| f(Some(OsStr::new(v)));
    assert_eq!(dp4a_from_env(None), Ok(None));
    assert_eq!(indexed_pairs_from_env(None), Ok(None));
    assert_eq!(pipe_from_env(None), Ok(None));
    assert_eq!(table_from_env(None), Ok(None));
    for (value, on) in [("0", false), ("1", true), (" 1 ", true)] {
        assert_eq!(parse(table_from_env, value), Ok(Some(on)), "{value:?}");
    }
    for bad in ["", "on", "2", "01", "table"] {
        let error = parse(table_from_env, bad).expect_err(bad);
        assert!(error.contains(TABLE_VAR), "{bad:?}: {error}");
    }
    for (value, on) in [("0", false), ("1", true), (" 1 ", true)] {
        assert_eq!(parse(pipe_from_env, value), Ok(Some(on)), "{value:?}");
    }
    for bad in ["", "on", "2", "01", "pipe"] {
        let error = parse(pipe_from_env, bad).expect_err(bad);
        assert!(error.contains(PIPE_VAR), "{bad:?}: {error}");
    }
    for (value, on) in [("0", false), ("1", true), (" 1 ", true)] {
        assert_eq!(parse(dp4a_from_env, value), Ok(Some(on)), "{value:?}");
    }
    for (value, on) in [("16", false), ("20", true), (" 20 ", true)] {
        assert_eq!(
            parse(indexed_pairs_from_env, value),
            Ok(Some(on)),
            "{value:?}"
        );
    }
    for bad in ["", "on", "true", "2", "01", "dp4a"] {
        let error = parse(dp4a_from_env, bad).expect_err(bad);
        assert!(error.contains(DP4A_VAR), "{bad:?}: {error}");
    }
    for bad in ["", "1", "8", "32", "on", "pairs20"] {
        let error = parse(indexed_pairs_from_env, bad).expect_err(bad);
        assert!(error.contains(INDEXED_PAIRS_VAR), "{bad:?}: {error}");
    }
}

/// A context default fills the options whose variable is unset, and a set variable wins for its
/// own option alone: `0` and `16` switch a registered default off, `1` and `20` switch one on.
#[test]
fn a_set_variable_wins_over_the_context_default_for_its_own_option() {
    assert_eq!(merge_indexed(OFF, (None, None, None, None)), OFF);
    assert_eq!(merge_indexed(BOTH, (None, None, None, None)), BOTH);
    assert_eq!(
        merge_indexed(BOTH, (Some(false), None, None, None)),
        PAIRS20
    );
    assert_eq!(merge_indexed(BOTH, (None, Some(false), None, None)), DP4A);
    assert_eq!(
        merge_indexed(BOTH, (Some(false), Some(false), None, None)),
        OFF
    );
    assert_eq!(merge_indexed(OFF, (Some(true), None, None, None)), DP4A);
    assert_eq!(merge_indexed(DP4A, (None, Some(true), None, None)), BOTH);
    assert_eq!(merge_indexed(OFF, (None, None, Some(true), None)), PIPE);
    assert_eq!(merge_indexed(OFF, (None, None, None, Some(true))), TABLE);
    assert_eq!(
        merge_indexed(TABLE_ALL, (Some(false), None, Some(false), Some(false))),
        OFF
    );
    assert_eq!(
        merge_indexed(BOTH_PIPE, (Some(false), Some(false), None, None)),
        PIPE
    );
    assert_eq!(merge_indexed(BOTH, (None, None, Some(false), None)), BOTH);
}

/// The pair of switches the environment read goes through (the function the production read
/// calls): unset is off, each switch selects its option alone, and a bad value of either refuses
/// the pair and names that switch.
#[test]
fn the_two_switches_select_their_options_independently() {
    let os = |v: &'static str| Some(OsStr::new(v));
    let read = |dp4a, pairs, pipe, table| {
        indexed_from_env(dp4a, pairs, pipe, table).map(|explicit| merge_indexed(OFF, explicit))
    };
    assert_eq!(read(None, None, None, None), Ok(OFF));
    assert_eq!(read(os("1"), None, None, None), Ok(DP4A));
    assert_eq!(read(None, os("20"), None, None), Ok(PAIRS20));
    assert_eq!(read(os("1"), os("20"), None, None), Ok(BOTH));
    assert_eq!(read(os("0"), os("16"), None, None), Ok(OFF));
    assert_eq!(read(None, None, os("1"), None), Ok(PIPE));
    assert_eq!(read(os("1"), os("20"), os("1"), None), Ok(BOTH_PIPE));
    assert_eq!(read(None, None, None, os("1")), Ok(TABLE));
    // The pair count env still fills its own option under the table form (which ignores it).
    assert_eq!(
        read(os("1"), os("20"), os("1"), os("1")),
        Ok(IndexedOptions {
            dp4a: true,
            pairs20: true,
            pipeline: true,
            table: true
        })
    );
    for (dp4a, pairs, pipe, table, var) in [
        (os("on"), None, None, None, DP4A_VAR),
        (os("1"), os("32"), None, None, INDEXED_PAIRS_VAR),
        (None, os(""), None, None, INDEXED_PAIRS_VAR),
        (os("2"), os("20"), None, None, DP4A_VAR),
        (None, None, os("yes"), None, PIPE_VAR),
        (os("1"), None, os("2"), None, PIPE_VAR),
        (None, None, None, os("table"), TABLE_VAR),
    ] {
        let error = read(dp4a, pairs, pipe, table).expect_err("a bad value");
        assert!(
            error.contains(var),
            "{dp4a:?} {pairs:?} {pipe:?} {table:?}: {error}"
        );
    }
}

#[test]
fn an_option_changes_only_the_launches_it_names() {
    for calls in 1..=PAIRS {
        let over = calls > INDEXED_CAPACITY;
        assert_eq!(OFF.native(calls), None);
        assert_eq!(DP4A.native(calls), Some(1));
        assert_eq!(PAIRS20.native(calls), over.then_some(2));
        assert_eq!(BOTH.native(calls), Some(if over { 3 } else { 1 }));
        assert_eq!(PIPE.native(calls), Some(4));
        assert_eq!(PAIRS20_PIPE.native(calls), Some(if over { 6 } else { 4 }));
        assert_eq!(BOTH_PIPE.native(calls), Some(if over { 7 } else { 5 }));
    }
    for (options, capacity, name) in [
        (OFF, 16, "off"),
        (DP4A, 16, "dp4a"),
        (PAIRS20, 20, "pairs20"),
        (BOTH, 20, "both"),
        (PIPE, 16, "pipe"),
        (PAIRS20_PIPE, 20, "pairs20+pipe"),
        (BOTH_PIPE, 20, "all"),
        (TABLE, 65535, "table"),
        (TABLE_ALL, 65535, "dp4a+table+pipe"),
    ] {
        assert_eq!(options.capacity(), capacity, "{options:?}");
        assert_eq!(options.name(), name);
    }
    assert_eq!(IndexedOptions::default(), OFF);
}

#[test]
fn an_option_refuses_a_wide_kernel_of_more_than_one_tile() {
    for kernel in [SingleRow::Legacy, SingleRow::Gather, SingleRow::Wide(1)] {
        for options in IndexedOptions::ALL {
            assert!(
                check_indexed_with(kernel, options).is_ok(),
                "{kernel:?} {options:?}"
            );
        }
    }
    for kernel in [SingleRow::Wide(2), SingleRow::Wide(4)] {
        assert!(check_indexed_with(kernel, OFF).is_ok(), "{kernel:?}");
        // The refusal names the switches that are on, and the kernel switch to change.
        for options in [DP4A, PAIRS20, BOTH, PIPE, BOTH_PIPE, TABLE, TABLE_ALL] {
            let error = check_indexed_with(kernel, options).expect_err("a wide kernel");
            let whole = format!("{error:?}");
            // The first part names the options that are on; the tail names both off values.
            let (text, tail) = whole.split_once(", or turn").expect("the off values");
            assert!(
                text.contains(SINGLE_GATHER_VAR),
                "{kernel:?} {options:?}: {whole}"
            );
            assert_eq!(
                text.contains(DP4A_VAR),
                options.dp4a,
                "{options:?}: {whole}"
            );
            assert_eq!(
                text.contains(INDEXED_PAIRS_VAR),
                options.pairs20,
                "{options:?}: {whole}"
            );
            assert_eq!(
                text.contains(PIPE_VAR),
                options.pipeline,
                "{options:?}: {whole}"
            );
            assert_eq!(
                text.contains(TABLE_VAR),
                options.table,
                "{options:?}: {whole}"
            );
            assert!(
                tail.contains(&format!("{DP4A_VAR}=0"))
                    && tail.contains(&format!("{INDEXED_PAIRS_VAR}=16"))
                    && tail.contains(&format!("{PIPE_VAR}=0"))
                    && tail.contains(&format!("{TABLE_VAR}=0")),
                "{options:?}: {whole}"
            );
        }
    }
}

#[test]
fn the_load_line_names_exactly_the_options_that_are_on() {
    assert_eq!(OFF.suffix(), "");
    assert_eq!(DP4A.suffix(), " exl3_dp4a=on");
    assert_eq!(PAIRS20.suffix(), " exl3_indexed_pairs=20");
    assert_eq!(BOTH.suffix(), " exl3_dp4a=on exl3_indexed_pairs=20");
    assert_eq!(PIPE.suffix(), " exl3_pipeline=on");
    assert_eq!(
        BOTH_PIPE.suffix(),
        " exl3_dp4a=on exl3_indexed_pairs=20 exl3_pipeline=on"
    );
    assert_eq!(TABLE.suffix(), " exl3_table=on");
    assert_eq!(
        TABLE_ALL.suffix(),
        " exl3_dp4a=on exl3_pipeline=on exl3_table=on"
    );
}

/// The session check and the load line read the context's options: a two-tile single-row kernel
/// with an option on stops the start, and the line names the options after the kernel. Needs
/// `FIFI_EXL3_SINGLE_GATHER` unset (an explicit value wins over the context's kernel).
#[test]
fn the_session_check_and_the_load_line_see_the_context_options() {
    if !testkit::device_present("EXL3 indexed option session check") {
        return;
    }
    let ctx = CudaContext::new().expect("context");
    ctx.set_single_row_default(SingleRow::Wide(2));
    ctx.set_indexed_options(Some(OFF));
    assert!(ctx.check_single_gather().is_ok());
    ctx.set_indexed_options(Some(DP4A));
    assert!(ctx.check_single_gather().is_err(), "dp4a with two tiles");
    ctx.set_single_row_default(SingleRow::Wide(1));
    assert!(ctx.check_single_gather().is_ok());
    ctx.set_indexed_options(Some(BOTH));
    assert_eq!(
        ctx.single_gather_suffix(),
        " single_gather=on single_kernel=wide1 exl3_dp4a=on exl3_indexed_pairs=20"
    );
    ctx.set_indexed_options(Some(OFF));
    assert_eq!(
        ctx.single_gather_suffix(),
        " single_gather=on single_kernel=wide1"
    );
}

/// A default a session registers on its context reaches the session check and the load line, and
/// another context in the process keeps the default launch: the real path of the Flash-Next start.
/// Skipped, not passed, when a launch switch is set in the environment.
#[test]
fn a_registered_default_reaches_the_session_check_and_stays_on_its_context() {
    if !testkit::device_present("EXL3 indexed default session check") {
        return;
    }
    let set: Vec<&str> = [SINGLE_GATHER_VAR, DP4A_VAR, INDEXED_PAIRS_VAR, PIPE_VAR]
        .into_iter()
        .filter(|var| std::env::var_os(var).is_some())
        .collect();
    if !set.is_empty() {
        eprintln!("skipped: {set:?} set in the environment");
        return;
    }
    let ctx = CudaContext::new().expect("context");
    ctx.set_indexed_default(BOTH);
    ctx.set_single_row_default(SingleRow::Wide(2));
    assert!(
        ctx.check_single_gather().is_err(),
        "the default options with two tiles"
    );
    ctx.set_single_row_default(SingleRow::Wide(1));
    assert!(ctx.check_single_gather().is_ok());
    assert_eq!(
        ctx.single_gather_suffix(),
        " single_gather=on single_kernel=wide1 exl3_dp4a=on exl3_indexed_pairs=20"
    );
    assert_eq!(ctx.exl3_indexed_capacity(), 20);
    let other = CudaContext::new().expect("second context");
    assert_eq!(other.exl3_indexed_capacity(), 16);
    assert_eq!(other.single_gather_suffix(), "");
}

/// 1024 + s is an F16 integer for every byte sum s (0 to 1020), and its bits are 0x6400 + s:
/// the claim the DP4A decode rests on.
#[test]
fn the_offset_byte_sum_is_an_exact_f16_bit_pattern() {
    for sum in 0u16..=1020 {
        let value = half::f16::from_f64(1024.0 + f64::from(sum));
        assert_eq!(value.to_bits(), 0x6400 + sum, "sum {sum}");
        assert_eq!(value.to_f64(), 1024.0 + f64::from(sum), "sum {sum}");
    }
}

/// Words with every byte value in every lane, every byte repeated in all four lanes, the hash
/// boundaries, the hashed word of every state and random words.
fn probe_words() -> Vec<u32> {
    let mut words = Vec::new();
    for lane in 0..4 {
        words.extend((0u32..256).map(|v| v << (8 * lane)));
    }
    words.extend((0u32..256).map(|v| v * 0x0101_0101));
    words.extend([
        0,
        u32::MAX,
        0x8080_8080,
        0x7f7f_7f7f,
        0x7fff_ffff,
        0x8000_0000,
        0x0000_00ff,
        0xff00_0000,
        0x00ff_00ff,
        0xff00_ff00,
    ]);
    words.extend((0u32..=0xffff).map(|state| state.wrapping_mul(MUL1)));
    let mut rng = testkit::Rng::new(0x5347_00d4);
    words.extend((0..4096).map(|_| rng.next_u64() as u32));
    words
}

#[test]
fn the_dp4a_decode_equals_the_book2_decode_for_every_state_and_byte() {
    if !testkit::device_present("EXL3 DP4A decode probe") {
        return;
    }
    let ctx = CudaContext::new().expect("context");
    let stream = ctx.stream().expect("stream");
    let words = probe_words();
    let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    let input = testkit::upload_bytes(&ctx, &bytes);
    let sums = testkit::upload_filled(&ctx, 0xff, words.len() * 8);
    let decoded = testkit::upload_filled(&ctx, 0xff, 65536 * 4);
    ctx.exl3_book2_probe(
        &stream,
        &input.range(0, input.bytes()).unwrap(),
        &sums.range(0, sums.bytes()).unwrap(),
        &decoded.range(0, decoded.bytes()).unwrap(),
    )
    .expect("probe");
    stream.synchronize().unwrap();
    let sums: Vec<u32> = testkit::read_bytes(&ctx, &sums, words.len() * 8)
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| u32::from_le_bytes(*b))
        .collect();
    for (i, &word) in words.iter().enumerate() {
        let want: u32 = word.to_le_bytes().iter().map(|&b| u32::from(b)).sum();
        assert_eq!(sums[2 * i], want, "masked byte sum of {word:#010x}");
        // Unsigned lanes: a byte of 0x80 or more adds its value, not a negative one, and the
        // offset never carries past bit 15.
        assert_eq!(
            sums[2 * i + 1],
            0x6400 + want,
            "DP4A byte sum of {word:#010x}"
        );
    }
    let decoded: Vec<u16> = testkit::read_bytes(&ctx, &decoded, 65536 * 4)
        .as_chunks::<2>()
        .0
        .iter()
        .map(|b| u16::from_le_bytes(*b))
        .collect();
    let mut differ = Vec::new();
    for state in 0..=u16::MAX {
        let host = Codebook::Mul1.decode_state(state);
        let s = usize::from(state);
        if decoded[2 * s] != host || decoded[2 * s + 1] != host {
            differ.push((state, host, decoded[2 * s], decoded[2 * s + 1]));
        }
    }
    assert!(
        differ.is_empty(),
        "{} states differ (state, host, decode_state, DP4A), first: {:x?}",
        differ.len(),
        &differ[..differ.len().min(8)]
    );
}

/// One bank of `members` random matrices of a rate and codebook, 20 calls on it and the buffers
/// they write.
struct Bank {
    book: Codebook,
    bits_twice: u8,
    k: usize,
    n: usize,
    members: usize,
    offset: usize,
    trellis: DeviceBuffer,
    scales: DeviceBuffer,
    x: DeviceBuffer,
}

/// What a launch wrote: rotated inputs, partials, outputs and the flag.
type Written = (Vec<u8>, Vec<u8>, Vec<u8>, Vec<u8>);

impl Bank {
    fn new(ctx: &CudaContext, rng: &mut testkit::Rng, kind: (Codebook, u8, usize)) -> Self {
        Self::with_shape(ctx, rng, kind, (512, 384), PAIRS)
    }

    /// A bank of an explicit `(k, n)` (both multiples of 128, so whole blocks retire at the
    /// per-record guards of the table launch) holding `rows` input rows: mixed-shape batches
    /// need two shapes of one codebook, and a batch past the parameter block's pair cap needs
    /// more rows than it.
    fn with_shape(
        ctx: &CudaContext,
        rng: &mut testkit::Rng,
        kind: (Codebook, u8, usize),
        shape: (usize, usize),
        rows: usize,
    ) -> Self {
        let (book, bits_twice, offset) = kind;
        let (k, n, members) = (shape.0, shape.1, 12usize);
        let stride = Self::stride(k, n, bits_twice);
        let mut bytes = vec![0xa5; offset];
        bytes.extend((0..members * stride).map(|_| rng.next_u64() as u8));
        let scale_values: Vec<f32> = (0..members * (k + n))
            .map(|i| 0.4 + (i % 11) as f32 / 16.0)
            .collect();
        let x_values: Vec<f32> = (0..rows * k)
            .map(|_| (rng.unit() as f32 - 0.5) * 8.0)
            .collect();
        Self {
            book,
            bits_twice,
            k,
            n,
            members,
            offset,
            trellis: testkit::upload_bytes(ctx, &bytes),
            scales: testkit::upload_f32(ctx, &scale_values),
            x: testkit::upload_f32(ctx, &x_values),
        }
    }

    fn stride(k: usize, n: usize, bits_twice: u8) -> usize {
        k * n / 256 * usize::from(bits_twice) * 16
    }

    /// The call of pair `p` (member `ids[p]`, slot `p`) with caller-chosen rotated-input and
    /// output ranges, so one launch can mix banks and shapes across records. The default
    /// launch demands each pair's `xh` and `z` follow the previous pair's in one buffer; the
    /// table launch takes them scattered, so the ranges are the caller's choice.
    fn one_call<'a>(
        &'a self,
        p: usize,
        xh: DeviceRange<'a>,
        z: DeviceRange<'a>,
        out: &'a DeviceBuffer,
        flag: &'a DeviceBuffer,
        ids: &'a DeviceBuffer,
    ) -> Exl3IndexedCall<'a> {
        let (k, n) = (self.k, self.n);
        let stride = Self::stride(k, n, self.bits_twice);
        Exl3IndexedCall {
            call: Exl3LinearCall {
                trellis: self
                    .trellis
                    .range(self.offset, self.members * stride)
                    .unwrap(),
                suh: self.scales.range(0, self.scales.bytes()).unwrap(),
                svh: self
                    .scales
                    .range(k * 4, self.scales.bytes() - k * 4)
                    .unwrap(),
                bias: None,
                x: self.x.range(p * k * 4, k * 4).unwrap(),
                xh,
                z,
                out: out.range(0, n * 4).unwrap(),
                flag: flag.range(0, 4).unwrap(),
                rows: 1,
                split_k: 2,
                k,
                n,
                bits_twice: self.bits_twice,
                book: self.book,
                fused: false,
            },
            ids: ids.range(0, ids.bytes()).unwrap(),
            slot: p as u32,
            members: self.members as u32,
            trellis_stride: stride,
            scale_stride: k + n,
            offsets: None,
        }
    }

    /// The 20 calls of `ids` in launches of the context's capacity; what they wrote.
    fn run(&self, ctx: &CudaContext, stream: &DeviceStream, ids: &[i32; PAIRS]) -> Written {
        self.run_in(ctx, stream, ids, (ctx.exl3_indexed_capacity(), 1))
            .expect("the launches")
    }

    /// The 20 calls of `ids` (call p reads member `ids[p]`) in launches of `chunk` calls with
    /// `cols` tiles per block, each pair writing its own rows of buffers filled with 0xff; what
    /// they wrote, or the refusal of a launch.
    fn run_in(
        &self,
        ctx: &CudaContext,
        stream: &DeviceStream,
        ids: &[i32; PAIRS],
        (chunk, cols): (usize, u32),
    ) -> Result<Written, crate::EngError> {
        let (k, n) = (self.k, self.n);
        let xh = testkit::upload_filled(ctx, 0xff, PAIRS * k * 4);
        let z = testkit::upload_filled(ctx, 0xff, PAIRS * n * 4);
        let out = testkit::upload_filled(ctx, 0xff, PAIRS * n * 4);
        let flag = testkit::upload_bytes(ctx, &0u32.to_le_bytes());
        let id_bytes: Vec<u8> = ids.iter().flat_map(|id| id.to_le_bytes()).collect();
        let id_buffer = testkit::upload_bytes(ctx, &id_bytes);
        let stride = Self::stride(k, n, self.bits_twice);
        let calls: Vec<Exl3IndexedCall<'_>> = (0..PAIRS)
            .map(|p| Exl3IndexedCall {
                call: Exl3LinearCall {
                    trellis: self
                        .trellis
                        .range(self.offset, self.members * stride)
                        .unwrap(),
                    suh: self.scales.range(0, self.scales.bytes()).unwrap(),
                    svh: self
                        .scales
                        .range(k * 4, self.scales.bytes() - k * 4)
                        .unwrap(),
                    bias: None,
                    x: self.x.range(p * k * 4, k * 4).unwrap(),
                    xh: xh.range(p * k * 4, k * 4).unwrap(),
                    z: z.range(p * n * 4, n * 4).unwrap(),
                    out: out.range(p * n * 4, n * 4).unwrap(),
                    flag: flag.range(0, 4).unwrap(),
                    rows: 1,
                    split_k: 2,
                    k,
                    n,
                    bits_twice: self.bits_twice,
                    book: self.book,
                    fused: false,
                },
                ids: id_buffer.range(0, id_buffer.bytes()).unwrap(),
                slot: p as u32,
                members: self.members as u32,
                trellis_stride: stride,
                scale_stride: k + n,
                offsets: None,
            })
            .collect();
        for launch in calls.chunks(chunk) {
            ctx.exl3_grouped_indexed(stream, launch, cols)?;
        }
        stream.synchronize()?;
        Ok((
            testkit::read_bytes(ctx, &xh, PAIRS * k * 4),
            testkit::read_bytes(ctx, &z, PAIRS * n * 4),
            testkit::read_bytes(ctx, &out, PAIRS * n * 4),
            testkit::read_bytes(ctx, &flag, 4),
        ))
    }
}

/// The bytes pair `p` owns in each of the rotated inputs, partials and outputs.
fn pair_spans(written: &Written, p: usize, k: usize, n: usize) -> [&[u8]; 3] {
    [
        &written.0[p * k * 4..(p + 1) * k * 4],
        &written.1[p * n * 4..(p + 1) * n * 4],
        &written.2[p * n * 4..(p + 1) * n * 4],
    ]
}

#[test]
fn indexed_options_write_the_bytes_of_the_default_launch() {
    if !testkit::device_present("EXL3 indexed options") {
        return;
    }
    let ctx = CudaContext::new().expect("context");
    let stream = ctx.stream().expect("stream");
    let mut rng = testkit::Rng::new(0x5347_0020);
    // Ten picks laid out as the block's gate and up pairs: pick s owns calls 2 s and 2 s + 1.
    let picks = [3i32, 0, 11, 7, 2, 9, 5, 10, 1, 6];
    let ids: [i32; PAIRS] = std::array::from_fn(|p| picks[p / 2]);
    // Book 2 at Flash-Next's rate (6), at the rates with their own bodies (8, 12), at an odd rate,
    // and with a bank 2 bytes into its buffer; the other books must be unchanged by the DP4A
    // option too.
    for kind in [
        (Codebook::Mul1, 6u8, 0usize),
        (Codebook::Mul1, 8, 0),
        (Codebook::Mul1, 12, 0),
        (Codebook::Mul1, 5, 0),
        (Codebook::Mul1, 6, 2),
        (Codebook::Mcg, 8, 0),
        (Codebook::ThreeInst, 12, 2),
    ] {
        let bank = Bank::new(&ctx, &mut rng, kind);
        let (k, n) = (bank.k, bank.n);
        ctx.set_indexed_options(Some(OFF));
        let want = bank.run(&ctx, &stream, &ids);
        assert_eq!(want.3, [0; 4], "{kind:?}: default flag");
        // The default launches wrote every pair, the second launch's included, so equality below
        // shows the options wrote them too.
        for p in 0..PAIRS {
            for span in pair_spans(&want, p, k, n) {
                assert!(
                    span.iter().any(|&b| b != 0xff),
                    "{kind:?}: pair {p} unwritten"
                );
            }
        }
        let mut bad = ids;
        bad[17] = bank.members as i32;
        bad[19] = -1;
        for options in [
            DP4A,
            PAIRS20,
            BOTH,
            PIPE,
            PAIRS20_PIPE,
            BOTH_PIPE,
            TABLE,
            TABLE_ALL,
        ] {
            ctx.set_indexed_options(Some(options));
            let got = bank.run(&ctx, &stream, &ids);
            assert_eq!(got, want, "{kind:?} {options:?}");
            // Ids outside the bank in the tail: exactly those pairs are skipped, the flag is
            // raised, every other pair matches.
            let gapped = bank.run(&ctx, &stream, &bad);
            assert_eq!(gapped.3, 1u32.to_le_bytes(), "{kind:?} {options:?}: flag");
            for p in 0..PAIRS {
                let spans = pair_spans(&gapped, p, k, n);
                if p == 17 || p == 19 {
                    for span in spans {
                        assert!(
                            span.iter().all(|&b| b == 0xff),
                            "{kind:?} {options:?}: skipped pair {p} was written"
                        );
                    }
                } else {
                    assert_eq!(
                        spans,
                        pair_spans(&want, p, k, n),
                        "{kind:?} {options:?}: pair {p}"
                    );
                }
            }
        }
    }
    ctx.set_indexed_options(None);
}

/// One launch of 20 calls: refused by the default launch and the DP4A option alone, taken by the
/// 20-pair option with or without DP4A with the bytes of 16 + 4 default launches. A launch an
/// option runs refuses more than one tile per block.
#[test]
fn only_the_twenty_pair_option_takes_twenty_calls_in_one_launch() {
    if !testkit::device_present("EXL3 indexed option capacity") {
        return;
    }
    let ctx = CudaContext::new().expect("context");
    let stream = ctx.stream().expect("stream");
    let mut rng = testkit::Rng::new(0x5347_0021);
    let bank = Bank::new(&ctx, &mut rng, (Codebook::Mul1, 6, 0));
    let ids: [i32; PAIRS] = std::array::from_fn(|p| (p * 5 % bank.members) as i32);
    ctx.set_indexed_options(Some(OFF));
    let want = bank.run(&ctx, &stream, &ids);
    for options in [OFF, DP4A, PIPE] {
        ctx.set_indexed_options(Some(options));
        assert_eq!(ctx.exl3_indexed_capacity(), INDEXED_CAPACITY);
        assert!(
            bank.run_in(&ctx, &stream, &ids, (PAIRS, 1)).is_err(),
            "{options:?} took 20 calls"
        );
    }
    for options in [PAIRS20, BOTH, PAIRS20_PIPE, BOTH_PIPE, TABLE, TABLE_ALL] {
        ctx.set_indexed_options(Some(options));
        assert_eq!(
            ctx.exl3_indexed_capacity(),
            if options.table {
                TABLE_PAIRS_MAX
            } else {
                PAIRS
            }
        );
        let got = bank.run_in(&ctx, &stream, &ids, (PAIRS, 1)).unwrap();
        assert_eq!(got, want, "{options:?}");
    }
    // DP4A runs one tile per block in every launch; the 20-pair option only in a launch of more
    // than 16 calls, so its 16-call launches keep the default kernel at two tiles (same bytes).
    for options in [DP4A, BOTH] {
        ctx.set_indexed_options(Some(options));
        assert!(
            bank.run_in(&ctx, &stream, &ids, (16, 2)).is_err(),
            "{options:?} took two tiles per block"
        );
    }
    ctx.set_indexed_options(Some(PAIRS20));
    assert!(bank.run_in(&ctx, &stream, &ids, (PAIRS, 2)).is_err());
    assert_eq!(bank.run_in(&ctx, &stream, &ids, (16, 2)).unwrap(), want);
    ctx.set_indexed_options(Some(TABLE));
    assert!(bank.run_in(&ctx, &stream, &ids, (PAIRS, 2)).is_err());
    ctx.set_indexed_options(None);
}

#[test]
fn the_launch_refuses_overlapping_records_before_any_native_call() {
    if !testkit::device_present("EXL3 cross-record isolation") {
        return;
    }
    let ctx = CudaContext::new().expect("context");
    let stream = ctx.stream().expect("stream");
    let mut rng = testkit::Rng::new(0x5347_0022);
    let bank = Bank::new(&ctx, &mut rng, (Codebook::Mul1, 6, 0));
    let flag = testkit::upload_bytes(&ctx, &0u32.to_le_bytes());
    let ids = testkit::upload_bytes(&ctx, &[3i32, 7].map(|id| id.to_le_bytes()).concat());
    // Two calls whose outputs alias: the same span of one buffer, so the two records would race
    // on it. The shared flag is the only alias a batch may write.
    let xh0 = testkit::upload_filled(&ctx, 0xff, bank.k * 4);
    let xh1 = testkit::upload_filled(&ctx, 0xff, bank.k * 4);
    let z0 = testkit::upload_filled(&ctx, 0xff, bank.n * 4);
    let z1 = testkit::upload_filled(&ctx, 0xff, bank.n * 4);
    let shared_out = testkit::upload_filled(&ctx, 0xff, bank.n * 4);
    let mut aliased_out = bank.one_call(
        0,
        xh0.range(0, bank.k * 4).unwrap(),
        z0.range(0, bank.n * 4).unwrap(),
        &shared_out,
        &flag,
        &ids,
    );
    let second = bank.one_call(
        1,
        xh1.range(0, bank.k * 4).unwrap(),
        z1.range(0, bank.n * 4).unwrap(),
        &shared_out,
        &flag,
        &ids,
    );
    aliased_out.slot = 0;
    let aliased_outputs = [aliased_out, second];
    // Overlapping rotated-input spans: two ranges of one buffer that share bytes, a
    // write-write alias the records would race on.
    let shared_xh = testkit::upload_filled(&ctx, 0xff, bank.k * 4 * 3 / 2);
    let overlapping_xh = [
        bank.one_call(
            0,
            shared_xh.range(0, bank.k * 4).unwrap(),
            z0.range(0, bank.n * 4).unwrap(),
            &xh0,
            &flag,
            &ids,
        ),
        bank.one_call(
            1,
            shared_xh.range(bank.k * 4 / 2, bank.k * 4).unwrap(),
            z1.range(0, bank.n * 4).unwrap(),
            &xh1,
            &flag,
            &ids,
        ),
    ];
    // Both classes are refused by the shared host prologue (a native refusal would latch
    // the backend terminal for the rest of the process), and the context still serves a
    // good launch afterwards — the proof the refusal never reached the device layer.
    for options in [OFF, TABLE] {
        ctx.set_indexed_options(Some(options));
        for (what, calls) in [
            ("aliased outputs", aliased_outputs.as_slice()),
            ("overlapping xh", overlapping_xh.as_slice()),
        ] {
            let refused = ctx.exl3_grouped_indexed(&stream, calls, 1);
            let error = refused.expect_err("the overlapping records were taken");
            assert!(
                matches!(error, crate::EngError::BadRequest(_)),
                "{what} under {options:?}: {error:?}"
            );
        }
        a_good_launch_follows(&ctx, &stream, &bank, &flag, &ids);
    }
    ctx.set_indexed_options(None);
}

/// The proof a refusal stayed host-side: the same context still runs a launch. A native
/// refusal would have latched the backend terminal, and this would fail instead.
fn a_good_launch_follows(
    ctx: &CudaContext,
    stream: &DeviceStream,
    bank: &Bank,
    flag: &DeviceBuffer,
    ids: &DeviceBuffer,
) {
    let xh = testkit::upload_filled(ctx, 0xff, 2 * bank.k * 4);
    let z = testkit::upload_filled(ctx, 0xff, 2 * bank.n * 4);
    let outs: Vec<_> = (0..2)
        .map(|_| testkit::upload_filled(ctx, 0xff, bank.n * 4))
        .collect();
    let calls: Vec<_> = (0..2)
        .map(|p| {
            bank.one_call(
                p,
                xh.range(p * bank.k * 4, bank.k * 4).unwrap(),
                z.range(p * bank.n * 4, bank.n * 4).unwrap(),
                &outs[p],
                flag,
                ids,
            )
        })
        .collect();
    ctx.exl3_grouped_indexed(stream, &calls, 1)
        .expect("the context still serves a good launch");
}

#[test]
fn the_table_launch_mixes_shapes_in_one_kernel() {
    if !testkit::device_present("EXL3 table mixed shapes") {
        return;
    }
    let ctx = CudaContext::new().expect("context");
    let stream = ctx.stream().expect("stream");
    let mut rng = testkit::Rng::new(0x5347_0023);
    let kind = (Codebook::Mul1, 6u8, 0usize);
    // The layer's two projections in one batch: the gate/up shape and the down shape swap k and
    // n (both multiples of 128, the launch geometry's unit; whole blocks retire at the
    // per-record guards).
    let banks = [
        Bank::with_shape(&ctx, &mut rng, kind, (768, 512), 2),
        Bank::with_shape(&ctx, &mut rng, kind, (512, 768), 2),
    ];
    let flag = testkit::upload_bytes(&ctx, &0u32.to_le_bytes());
    // Reference round: the default launch walks pair `p`'s `xh` and `z` at byte `p * k * 4`
    // (and `p * n * 4`) of one contiguous buffer per bank, so each bank packs them; outputs
    // stay separate allocations. Mixed round: per-pair buffers, which the table launch takes
    // scattered — the equality across the two shapes proves the mixed records all ran.
    struct Packed {
        xh: DeviceBuffer,
        z: DeviceBuffer,
        outs: Vec<DeviceBuffer>,
    }
    let reference: Vec<Packed> = banks
        .iter()
        .map(|bank| Packed {
            xh: testkit::upload_filled(&ctx, 0xff, 2 * bank.k * 4),
            z: testkit::upload_filled(&ctx, 0xff, 2 * bank.n * 4),
            outs: (0..2)
                .map(|_| testkit::upload_filled(&ctx, 0xff, bank.n * 4))
                .collect(),
        })
        .collect();
    let mixed: Vec<Vec<[DeviceBuffer; 3]>> = banks
        .iter()
        .map(|bank| {
            (0..2)
                .map(|_| {
                    [
                        testkit::upload_filled(&ctx, 0xff, bank.k * 4),
                        testkit::upload_filled(&ctx, 0xff, bank.n * 4),
                        testkit::upload_filled(&ctx, 0xff, bank.n * 4),
                    ]
                })
                .collect()
        })
        .collect();
    // Reference: each bank's two calls launched alone under the default launch.
    ctx.set_indexed_options(Some(OFF));
    let mut want = Vec::new();
    for (bank, packed) in banks.iter().zip(&reference) {
        let ids = testkit::upload_bytes(&ctx, &[3i32, 7].map(|id| id.to_le_bytes()).concat());
        let calls: Vec<Exl3IndexedCall<'_>> = (0..2)
            .map(|p| {
                bank.one_call(
                    p,
                    packed.xh.range(p * bank.k * 4, bank.k * 4).unwrap(),
                    packed.z.range(p * bank.n * 4, bank.n * 4).unwrap(),
                    &packed.outs[p],
                    &flag,
                    &ids,
                )
            })
            .collect();
        ctx.exl3_grouped_indexed(&stream, &calls, 1)
            .expect("default");
    }
    stream.synchronize().expect("sync");
    for packed in &reference {
        for out in &packed.outs {
            want.push(testkit::read_bytes(&ctx, out, out.bytes()));
        }
    }
    // Mixed: all four calls of both shapes in one table launch; each record's bytes equal the
    // default launch of its own calls.
    ctx.set_indexed_options(Some(TABLE));
    let ids = testkit::upload_bytes(&ctx, &[3i32, 7].map(|id| id.to_le_bytes()).concat());
    let mut calls = Vec::new();
    for (bank, pairs) in banks.iter().zip(&mixed) {
        for (p, [xh, z, out]) in pairs.iter().enumerate() {
            calls.push(bank.one_call(
                p,
                xh.range(0, bank.k * 4).unwrap(),
                z.range(0, bank.n * 4).unwrap(),
                out,
                &flag,
                &ids,
            ));
        }
    }
    ctx.exl3_grouped_indexed(&stream, &calls, 1)
        .expect("the mixed-shape table launch");
    stream.synchronize().expect("sync");
    let flat = mixed.iter().flatten();
    for (row, want) in flat.zip(&want) {
        assert_eq!(&testkit::read_bytes(&ctx, &row[2], row[2].bytes()), want);
    }
    assert_eq!(testkit::read_bytes(&ctx, &flag, 4), [0; 4]);
    ctx.set_indexed_options(None);
}

/// One table launch per batch of `per_bank` records from each bank (two shapes), checked
/// against the same records run as chunked 20-pair option launches. Returns each record's
/// output bytes in bank-then-pair order — the order the reference chunks wrote them.
fn batch_matches_wide_chunks(
    ctx: &CudaContext,
    stream: &DeviceStream,
    banks: &[Bank],
    flag: &DeviceBuffer,
    per_bank: usize,
) -> Vec<Vec<u8>> {
    let ids: Vec<u8> = (0..per_bank)
        .flat_map(|i| ((3 + i % 5) as i32).to_le_bytes())
        .collect();
    let ids = testkit::upload_bytes(ctx, &ids);
    // Reference: each bank's records in chunks of 20 (the widest parameter block), their
    // `xh` and `z` packed per chunk as the option launch requires.
    ctx.set_indexed_options(Some(OFF));
    let mut want = Vec::new();
    for bank in banks {
        for chunk in (0..per_bank).collect::<Vec<_>>().chunks(20) {
            let rows = chunk.len();
            let xh = testkit::upload_filled(ctx, 0xff, rows * bank.k * 4);
            let z = testkit::upload_filled(ctx, 0xff, rows * bank.n * 4);
            let outs: Vec<_> = (0..rows)
                .map(|_| testkit::upload_filled(ctx, 0xff, bank.n * 4))
                .collect();
            let calls: Vec<_> = chunk
                .iter()
                .enumerate()
                .map(|(at, &p)| {
                    bank.one_call(
                        p,
                        xh.range(at * bank.k * 4, bank.k * 4).unwrap(),
                        z.range(at * bank.n * 4, bank.n * 4).unwrap(),
                        &outs[at],
                        flag,
                        &ids,
                    )
                })
                .collect();
            ctx.exl3_grouped_indexed_wide(stream, &calls, 1)
                .expect("the reference chunk");
            for out in &outs {
                want.push(testkit::read_bytes(ctx, out, out.bytes()));
            }
        }
    }
    stream.synchronize().expect("sync");
    // The batch: every record of every bank in one table launch, their buffers scattered.
    ctx.set_indexed_options(Some(TABLE));
    // The buffers must be owned outside the loop that borrows them into the calls (the
    // calls outlive each iteration).
    let mut staged = Vec::with_capacity(banks.len() * per_bank);
    for (bank_idx, bank) in banks.iter().enumerate() {
        for p in 0..per_bank {
            staged.push((
                bank_idx,
                p,
                testkit::upload_filled(ctx, 0xff, bank.k * 4),
                testkit::upload_filled(ctx, 0xff, bank.n * 4),
                testkit::upload_filled(ctx, 0xff, bank.n * 4),
            ));
        }
    }
    let mut calls = Vec::with_capacity(staged.len());
    let mut outs = Vec::with_capacity(staged.len());
    for (bank_idx, p, xh, z, out) in &staged {
        let bank = &banks[*bank_idx];
        calls.push(bank.one_call(
            *p,
            xh.range(0, bank.k * 4).unwrap(),
            z.range(0, bank.n * 4).unwrap(),
            out,
            flag,
            &ids,
        ));
        outs.push(out);
    }
    ctx.exl3_grouped_indexed(stream, &calls, 1)
        .expect("the batched table launch");
    stream.synchronize().expect("sync");
    for (out, want) in outs.iter().zip(&want) {
        assert_eq!(&testkit::read_bytes(ctx, out, out.bytes()), want);
    }
    ctx.set_indexed_options(None);
    want
}

#[test]
fn the_table_launch_batches_past_the_pair_cap_and_grows_its_region() {
    if !testkit::device_present("EXL3 table past the pair cap") {
        return;
    }
    let ctx = CudaContext::new().expect("context");
    let stream = ctx.stream().expect("stream");
    let mut rng = testkit::Rng::new(0x5347_0024);
    let kind = (Codebook::Mul1, 6u8, 0usize);
    // Two shapes, thirty rows each: the first batch (60 records, 7680 bytes of records)
    // crosses the staging region's 4096-byte start and grows it, the eight-record batch
    // reuses the grown region, and the ninety-six-record batch grows it again — every batch
    // byte-equal to its chunked 20-pair reference.
    let banks = [
        Bank::with_shape(&ctx, &mut rng, kind, (768, 512), 48),
        Bank::with_shape(&ctx, &mut rng, kind, (512, 768), 48),
    ];
    let flag = testkit::upload_bytes(&ctx, &0u32.to_le_bytes());
    batch_matches_wide_chunks(&ctx, &stream, &banks, &flag, 30);
    batch_matches_wide_chunks(&ctx, &stream, &banks, &flag, 4);
    batch_matches_wide_chunks(&ctx, &stream, &banks, &flag, 48);
    assert_eq!(testkit::read_bytes(&ctx, &flag, 4), [0; 4]);
}

#[test]
fn the_table_launch_keeps_one_region_per_stream() {
    if !testkit::device_present("EXL3 table region per stream") {
        return;
    }
    let ctx = CudaContext::new().expect("context");
    let first = ctx.stream().expect("stream");
    let second = ctx.stream().expect("stream");
    let mut rng = testkit::Rng::new(0x5347_0025);
    let kind = (Codebook::Mul1, 6u8, 0usize);
    let bank = Bank::with_shape(&ctx, &mut rng, kind, (512, 384), 6);
    let flag = testkit::upload_bytes(&ctx, &0u32.to_le_bytes());
    // The reference bytes of the six records, once per stream (the same records, so the
    // same bytes).
    let want = batch_matches_wide_chunks(&ctx, &first, std::slice::from_ref(&bank), &flag, 6);
    batch_matches_wide_chunks(&ctx, &second, std::slice::from_ref(&bank), &flag, 6);
    // Two interleaved batches on two streams: each stages its records into its own
    // stream's region, so the second launch's copy must not overwrite the first's before
    // its kernel read them. Both synchronize only after both launches are queued.
    ctx.set_indexed_options(Some(TABLE));
    let ids: Vec<u8> = (0..6i32).flat_map(|i| (3 + i % 5).to_le_bytes()).collect();
    let ids = testkit::upload_bytes(&ctx, &ids);
    let mut launches = Vec::new();
    for stream in [&first, &second] {
        // Buffers owned per stream, then borrowed into the calls (the staged tuple moves
        // with its borrows into `launches`).
        let mut staged = Vec::with_capacity(6);
        for _ in 0..6 {
            staged.push((
                testkit::upload_filled(&ctx, 0xff, bank.k * 4),
                testkit::upload_filled(&ctx, 0xff, bank.n * 4),
                testkit::upload_filled(&ctx, 0xff, bank.n * 4),
            ));
        }
        let mut calls = Vec::with_capacity(6);
        for (p, (xh, z, out)) in staged.iter().enumerate() {
            calls.push(bank.one_call(
                p,
                xh.range(0, bank.k * 4).unwrap(),
                z.range(0, bank.n * 4).unwrap(),
                out,
                &flag,
                &ids,
            ));
        }
        ctx.exl3_grouped_indexed(stream, &calls, 1)
            .expect("the batched table launch");
        launches.push(staged);
    }
    first.synchronize().expect("sync");
    second.synchronize().expect("sync");
    ctx.set_indexed_options(None);
    // Each stream's bytes equal the reference of the same records.
    for staged in &launches {
        // Each staged row is (xh, z, out) in record order: the out is the third element.
        for ((_, _, out), want) in staged.iter().zip(&want) {
            assert_eq!(&testkit::read_bytes(&ctx, out, out.bytes()), want);
        }
    }
    assert_eq!(testkit::read_bytes(&ctx, &flag, 4), [0; 4]);
}
