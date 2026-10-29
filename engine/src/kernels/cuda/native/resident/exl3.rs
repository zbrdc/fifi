//! EXL3 stream-only C ABI. RawRange proves live allocation bounds; this layer
//! repeats geometry, extent, alignment and overlap checks before C repeats them.
//! Dispatch/stream ownership is enforced by CudaContext; frees drain device use.
//! No raw pointer or unsafe function leaves this module. Faults surface at sync.
#![cfg_attr(
    not(test),
    expect(dead_code, reason = "EXL3 M1, family routing stays off")
)]
use super::{DeviceFailure, RawRange, RawStream, STAGE_NONE, Span, checked, invalid_argument};
use crate::weights::exl3::Codebook;
use std::ffi::{c_int, c_void};
use std::mem::offset_of;

#[repr(C)]
#[derive(Clone, Copy)]
struct PrepArgs {
    trellis: *const u16,
    trellis_bytes: usize,
    out: *mut u16,
    out_bytes: usize,
    k: usize,
    n: usize,
    bits_twice: c_int,
    book: c_int,
}
const _: () = assert!(
    size_of::<PrepArgs>() == 56
        && offset_of!(PrepArgs, trellis) == 0
        && offset_of!(PrepArgs, trellis_bytes) == 8
        && offset_of!(PrepArgs, out) == 16
        && offset_of!(PrepArgs, out_bytes) == 24
        && offset_of!(PrepArgs, k) == 32
        && offset_of!(PrepArgs, n) == 40
        && offset_of!(PrepArgs, bits_twice) == 48
        && offset_of!(PrepArgs, book) == 52
);
unsafe extern "C" {
    fn inference_cuda_exl3_prep(
        args: *const PrepArgs,
        stream: *mut c_void,
        stage: *mut c_int,
    ) -> c_int;
}

pub(in crate::kernels::cuda) fn geometry(
    k: usize,
    n: usize,
    bits: u8,
    book: Codebook,
) -> Option<(usize, usize)> {
    if !(128..=(1 << 22)).contains(&k)
        || !(128..=(1 << 22)).contains(&n)
        || !k.is_multiple_of(128)
        || !n.is_multiple_of(128)
    {
        return None;
    }
    let supported = (2..=16).contains(&bits) && bits.is_multiple_of(2)
        || book == Codebook::Mul1 && matches!(bits, 3 | 5 | 7);
    let count = k.checked_mul(n)?;
    if !supported || count > (1usize << 32) {
        return None;
    }
    Some((count / 256 * usize::from(bits) * 16, count * 2))
}
fn book_tag(book: Codebook) -> c_int {
    match book {
        Codebook::ThreeInst => 0,
        Codebook::Mcg => 1,
        Codebook::Mul1 => 2,
    }
}

pub(in crate::kernels::cuda) struct PrepRaw<'a> {
    pub(in crate::kernels::cuda) trellis: RawRange<'a>,
    pub(in crate::kernels::cuda) out: RawRange<'a>,
    pub(in crate::kernels::cuda) k: usize,
    pub(in crate::kernels::cuda) n: usize,
    pub(in crate::kernels::cuda) bits_twice: u8,
    pub(in crate::kernels::cuda) book: Codebook,
}
pub(in crate::kernels::cuda) fn prep(
    stream: &RawStream,
    call: &PrepRaw<'_>,
) -> Result<(), DeviceFailure> {
    let (packed, dense) =
        geometry(call.k, call.n, call.bits_twice, call.book).ok_or_else(invalid_argument)?;
    if call.trellis.bytes() < packed
        || call.out.bytes() < dense
        || !call.trellis.offset().is_multiple_of(2)
        || !call.out.offset().is_multiple_of(2)
        || !Span::of(&call.trellis, packed).apart(&Span::of(&call.out, dense))
    {
        return Err(invalid_argument());
    }
    let args = PrepArgs {
        trellis: call.trellis.address().cast(),
        trellis_bytes: call.trellis.bytes(),
        out: call.out.address().cast(),
        out_bytes: call.out.bytes(),
        k: call.k,
        n: call.n,
        bits_twice: c_int::from(call.bits_twice),
        book: book_tag(call.book),
    };
    let mut stage = STAGE_NONE;
    // SAFETY: live bounded ranges, complete extents, alignment and disjointness proved above;
    // C repeats these checks. Explicit live stream is under the dispatch guard.
    let status = unsafe { inference_cuda_exl3_prep(&args, stream.as_ptr(), &mut stage) };
    checked(status, stage)
}

/// Working byte extents in ABI order: trellis, suh, svh, bias, x, xh, z, out, flag.
pub(in crate::kernels::cuda) fn linear_extents(
    rows: usize,
    k: usize,
    n: usize,
    bits: u8,
    book: Codebook,
    split_k: u32,
) -> Option<[usize; 9]> {
    let (packed, _) = geometry(k, n, bits, book)?;
    if !(1..=65536).contains(&rows) || !matches!(split_k, 2 | 4 | 8) {
        return None;
    }
    let input = rows.checked_mul(k)?;
    let output = rows.checked_mul(n)?;
    if input > (1usize << 32) || output > (1usize << 32) {
        return None;
    }
    let banks = if rows <= 8 { split_k as usize - 1 } else { 1 };
    let z = output.checked_mul(4)?.checked_mul(banks)?;
    Some([
        packed,
        k * 4,
        n * 4,
        n * 4,
        input * 4,
        input * 4,
        z,
        output * 4,
        4,
    ])
}

pub(in crate::kernels::cuda) fn midrows_extents(
    rows: usize,
    k: usize,
    n: usize,
    bits: u8,
    book: Codebook,
) -> Option<[usize; 9]> {
    if !(9..=144).contains(&rows) {
        return None;
    }
    linear_extents(rows, k, n, bits, book, 2)
}

#[repr(C)]
#[derive(Clone, Copy)]
struct LinearArgs {
    trellis: *const u16,
    trellis_bytes: usize,
    suh: *const f32,
    suh_bytes: usize,
    svh: *const f32,
    svh_bytes: usize,
    bias: *const f32,
    bias_bytes: usize,
    x: *const f32,
    x_bytes: usize,
    xh: *mut f32,
    xh_bytes: usize,
    z: *mut f32,
    z_bytes: usize,
    out: *mut f32,
    out_bytes: usize,
    flag: *mut u32,
    flag_bytes: usize,
    rows: usize,
    k: usize,
    n: usize,
    bits_twice: c_int,
    book: c_int,
}
const _: () = assert!(
    size_of::<LinearArgs>() == 176
        && offset_of!(LinearArgs, trellis) == 0
        && offset_of!(LinearArgs, trellis_bytes) == 8
        && offset_of!(LinearArgs, suh) == 16
        && offset_of!(LinearArgs, suh_bytes) == 24
        && offset_of!(LinearArgs, svh) == 32
        && offset_of!(LinearArgs, svh_bytes) == 40
        && offset_of!(LinearArgs, bias) == 48
        && offset_of!(LinearArgs, bias_bytes) == 56
        && offset_of!(LinearArgs, x) == 64
        && offset_of!(LinearArgs, x_bytes) == 72
        && offset_of!(LinearArgs, xh) == 80
        && offset_of!(LinearArgs, xh_bytes) == 88
        && offset_of!(LinearArgs, z) == 96
        && offset_of!(LinearArgs, z_bytes) == 104
        && offset_of!(LinearArgs, out) == 112
        && offset_of!(LinearArgs, out_bytes) == 120
        && offset_of!(LinearArgs, flag) == 128
        && offset_of!(LinearArgs, flag_bytes) == 136
        && offset_of!(LinearArgs, rows) == 144
        && offset_of!(LinearArgs, k) == 152
        && offset_of!(LinearArgs, n) == 160
        && offset_of!(LinearArgs, bits_twice) == 168
        && offset_of!(LinearArgs, book) == 172
);
unsafe extern "C" {
    #[cfg(test)]
    fn inference_cuda_exl3_linear(
        args: *const LinearArgs,
        stream: *mut c_void,
        stage: *mut c_int,
    ) -> c_int;
    fn inference_cuda_exl3_linear_midrows(
        args: *const LinearArgs,
        stream: *mut c_void,
        stage: *mut c_int,
    ) -> c_int;
    fn inference_cuda_exl3_linear_split(
        args: *const LinearArgs,
        split_k: u32,
        stream: *mut c_void,
        stage: *mut c_int,
    ) -> c_int;
}

pub(in crate::kernels::cuda) struct LinearRaw<'a> {
    pub(in crate::kernels::cuda) trellis: RawRange<'a>,
    pub(in crate::kernels::cuda) suh: RawRange<'a>,
    pub(in crate::kernels::cuda) svh: RawRange<'a>,
    pub(in crate::kernels::cuda) bias: Option<RawRange<'a>>,
    pub(in crate::kernels::cuda) x: RawRange<'a>,
    pub(in crate::kernels::cuda) xh: RawRange<'a>,
    pub(in crate::kernels::cuda) z: RawRange<'a>,
    pub(in crate::kernels::cuda) out: RawRange<'a>,
    pub(in crate::kernels::cuda) flag: RawRange<'a>,
    pub(in crate::kernels::cuda) rows: usize,
    pub(in crate::kernels::cuda) split_k: u32,
    pub(in crate::kernels::cuda) k: usize,
    pub(in crate::kernels::cuda) n: usize,
    pub(in crate::kernels::cuda) bits_twice: u8,
    pub(in crate::kernels::cuda) book: Codebook,
}
pub(in crate::kernels::cuda) fn linear(
    stream: &RawStream,
    call: &LinearRaw<'_>,
) -> Result<(), DeviceFailure> {
    linear_mode(stream, call, false)
}

pub(in crate::kernels::cuda) fn linear_midrows(
    stream: &RawStream,
    call: &LinearRaw<'_>,
) -> Result<(), DeviceFailure> {
    linear_mode(stream, call, true)
}

fn linear_args(call: &LinearRaw<'_>, midrows: bool) -> Result<LinearArgs, DeviceFailure> {
    if !matches!(call.split_k, 2 | 4 | 8) {
        return Err(invalid_argument());
    }
    let needs = if midrows {
        midrows_extents(call.rows, call.k, call.n, call.bits_twice, call.book)
    } else {
        linear_extents(
            call.rows,
            call.k,
            call.n,
            call.bits_twice,
            call.book,
            call.split_k,
        )
    }
    .ok_or_else(invalid_argument)?;
    let ranges = [
        Some(&call.trellis),
        Some(&call.suh),
        Some(&call.svh),
        call.bias.as_ref(),
        Some(&call.x),
        Some(&call.xh),
        Some(&call.z),
        Some(&call.out),
        Some(&call.flag),
    ];
    for (i, range) in ranges.iter().enumerate() {
        let Some(range) = range else { continue };
        let alignment = if i == 0 { 2 } else { 4 };
        if range.bytes() < needs[i] || !range.offset().is_multiple_of(alignment) {
            return Err(invalid_argument());
        }
        for (j, other) in ranges[..i].iter().enumerate() {
            if let Some(other) = other
                && (i >= 5 || j >= 5)
                && !Span::of(range, needs[i]).apart(&Span::of(other, needs[j]))
            {
                return Err(invalid_argument());
            }
        }
    }
    let args = LinearArgs {
        trellis: call.trellis.address().cast(),
        trellis_bytes: call.trellis.bytes(),
        suh: call.suh.address().cast(),
        suh_bytes: call.suh.bytes(),
        svh: call.svh.address().cast(),
        svh_bytes: call.svh.bytes(),
        bias: call
            .bias
            .as_ref()
            .map_or(std::ptr::null(), |r| r.address().cast()),
        bias_bytes: call.bias.as_ref().map_or(0, RawRange::bytes),
        x: call.x.address().cast(),
        x_bytes: call.x.bytes(),
        xh: call.xh.address().cast(),
        xh_bytes: call.xh.bytes(),
        z: call.z.address().cast(),
        z_bytes: call.z.bytes(),
        out: call.out.address().cast(),
        out_bytes: call.out.bytes(),
        flag: call.flag.address().cast(),
        flag_bytes: call.flag.bytes(),
        rows: call.rows,
        k: call.k,
        n: call.n,
        bits_twice: c_int::from(call.bits_twice),
        book: book_tag(call.book),
    };
    Ok(args)
}

fn linear_mode(
    stream: &RawStream,
    call: &LinearRaw<'_>,
    midrows: bool,
) -> Result<(), DeviceFailure> {
    let args = linear_args(call, midrows)?;
    let mut stage = STAGE_NONE;
    // SAFETY: all live owners, extents, alignment and disjoint writes checked above,
    // repeated in C; the explicit live stream is protected by the dispatch guard.
    #[cfg(all(test, feature = "native-cuda", target_os = "linux"))]
    if launch_observer::recording() {
        return checked(
            launch_observer::linear(&args, call.split_k, midrows, stream, &mut stage),
            stage,
        );
    }
    let status = unsafe {
        if midrows {
            inference_cuda_exl3_linear_midrows(&args, stream.as_ptr(), &mut stage)
        } else {
            inference_cuda_exl3_linear_split(&args, call.split_k, stream.as_ptr(), &mut stage)
        }
    };
    checked(status, stage)
}

unsafe extern "C" {
    fn inference_cuda_exl3_grouped(
        args: *const LinearArgs,
        count: usize,
        stream: *mut c_void,
        stage: *mut c_int,
    ) -> c_int;
    fn inference_cuda_exl3_grouped_gathered(
        args: *const LinearArgs,
        count: usize,
        stream: *mut c_void,
        stage: *mut c_int,
    ) -> c_int;
}

pub(in crate::kernels::cuda) fn grouped(
    stream: &RawStream,
    calls: &[LinearRaw<'_>],
    gather: bool,
) -> Result<(), DeviceFailure> {
    if calls.is_empty() || calls.len() > 48 {
        return Err(invalid_argument());
    }
    let mut pairs = 0usize;
    let mut args = Vec::with_capacity(calls.len());
    for call in calls {
        pairs = pairs.checked_add(call.rows).ok_or_else(invalid_argument)?;
        if !(1..=8).contains(&call.rows) || call.split_k != 2 || pairs > 48 {
            return Err(invalid_argument());
        }
        args.push(linear_args(call, false)?);
    }
    let mut stage = STAGE_NONE;
    #[cfg(all(test, feature = "native-cuda", target_os = "linux"))]
    if launch_observer::recording() {
        // Identical launch plan either way; the observed entry records which
        // variant actually ran so tests cannot claim gather without running it.
        return checked(
            launch_observer::grouped(&args, stream, &mut stage, gather),
            stage,
        );
    }
    // SAFETY: every record has live bounded owners and validated local extents.
    // C validates all group shapes, contiguous writes and cross-group aliases before ANY enqueue.
    let status = unsafe {
        if gather {
            inference_cuda_exl3_grouped_gathered(
                args.as_ptr(),
                args.len(),
                stream.as_ptr(),
                &mut stage,
            )
        } else {
            inference_cuda_exl3_grouped(args.as_ptr(), args.len(), stream.as_ptr(), &mut stage)
        }
    };
    checked(status, stage)
}

#[cfg(all(test, feature = "native-cuda", target_os = "linux"))]
pub(in crate::kernels::cuda) mod launch_observer {
    use super::*;
    use std::cell::Cell;

    thread_local! {
        static COUNT: Cell<Option<u64>> = const { Cell::new(None) };
        static ROUTED: Cell<bool> = const { Cell::new(false) };
        /// 0 none, 1 shuffle, 2 gather, 3 mixed across a leg's dispatches.
        static GATHERED: Cell<u8> = const { Cell::new(0) };
    }

    #[repr(C)]
    struct ObserveArgs {
        rows: usize,
        k: usize,
        n: usize,
        split_k: u32,
        midrows: u32,
        fail_at: u32,
    }

    #[repr(C)]
    #[derive(Default)]
    struct Observation {
        emissions: [[u32; 7]; 3],
        count: u32,
        complete: u32,
    }

    const _: () = assert!(
        size_of::<ObserveArgs>() == 40
            && offset_of!(ObserveArgs, split_k) == 24
            && offset_of!(ObserveArgs, fail_at) == 32
            && size_of::<Observation>() == 92
            && offset_of!(Observation, count) == 84
            && offset_of!(Observation, complete) == 88
    );

    unsafe extern "C" {
        fn inference_cuda_exl3_observe_linear(
            args: *const ObserveArgs,
            observation: *mut Observation,
        ) -> c_int;
        fn inference_cuda_exl3_observe_grouped(
            args: *const ObserveArgs,
            observation: *mut Observation,
        ) -> c_int;
        fn inference_cuda_exl3_grouped_observed(
            args: *const LinearArgs,
            count: usize,
            stream: *mut c_void,
            stage: *mut c_int,
            launches: *mut u64,
            variant: *mut u32,
        ) -> c_int;
        fn inference_cuda_exl3_grouped_observed_gathered(
            args: *const LinearArgs,
            count: usize,
            stream: *mut c_void,
            stage: *mut c_int,
            launches: *mut u64,
            variant: *mut u32,
        ) -> c_int;
        fn inference_cuda_exl3_linear_observed(
            args: *const LinearArgs,
            split_k: u32,
            midrows: u32,
            stream: *mut c_void,
            stage: *mut c_int,
            launches: *mut u64,
        ) -> c_int;
    }

    pub(in crate::kernels::cuda) fn grouped_emissions(
        [rows, k, n]: [usize; 3],
        fail_at: u32,
    ) -> Result<(Vec<[u32; 7]>, bool), DeviceFailure> {
        let args = ObserveArgs {
            rows,
            k,
            n,
            split_k: 2,
            midrows: 0,
            fail_at,
        };
        let mut out = Observation::default();
        // SAFETY: live C-layout host records; this calls only the shared host emitter.
        let status = unsafe { inference_cuda_exl3_observe_grouped(&args, &mut out) };
        checked(status, STAGE_NONE)?;
        if out.count > 3 || out.complete > 1 {
            return Err(invalid_argument());
        }
        Ok((
            out.emissions[..out.count as usize].to_vec(),
            out.complete != 0,
        ))
    }

    pub(super) fn grouped(
        args: &[LinearArgs],
        stream: &RawStream,
        stage: &mut c_int,
        gather: bool,
    ) -> c_int {
        let mut launches = 0;
        let mut variant = 0u32;
        // SAFETY: live validated records and owners; C checks cross-group aliases
        // before launch. All three writable host outputs remain live through this
        // call; C writes the variant only after a successful projection enqueue.
        let status = unsafe {
            if gather {
                inference_cuda_exl3_grouped_observed_gathered(
                    args.as_ptr(),
                    args.len(),
                    stream.as_ptr(),
                    stage,
                    &mut launches,
                    &mut variant,
                )
            } else {
                inference_cuda_exl3_grouped_observed(
                    args.as_ptr(),
                    args.len(),
                    stream.as_ptr(),
                    stage,
                    &mut launches,
                    &mut variant,
                )
            }
        };
        COUNT.with(|count| count.set(Some(count.get().unwrap().checked_add(launches).unwrap())));
        // The variant comes from the device launch, never the request; a refusal
        // or zero launches records nothing and fails the leg's assert.
        if status == 0 && variant != 0 {
            GATHERED.with(|used| {
                let code = variant as u8;
                let seen = used.get();
                if seen == 0 {
                    used.set(code);
                } else if seen != code {
                    used.set(3);
                }
            });
        }
        status
    }

    /// Host metadata only: the C binding uses the production emitter, not CUDA.
    pub(in crate::kernels::cuda) fn emissions(
        [rows, k, n]: [usize; 3],
        split_k: u32,
        midrows: bool,
        fail_at: u32,
    ) -> Result<(Vec<[u32; 7]>, bool), DeviceFailure> {
        let args = ObserveArgs {
            rows,
            k,
            n,
            split_k,
            midrows: u32::from(midrows),
            fail_at,
        };
        let mut out = Observation::default();
        // SAFETY: both C-layout host records are live and writable for this call;
        // the entry validates geometry and never interprets a device pointer.
        let status = unsafe { inference_cuda_exl3_observe_linear(&args, &mut out) };
        checked(status, STAGE_NONE)?;
        if out.count > 3 || out.complete > 1 {
            return Err(invalid_argument());
        }
        Ok((
            out.emissions[..out.count as usize].to_vec(),
            out.complete != 0,
        ))
    }

    struct Capture;
    impl Drop for Capture {
        fn drop(&mut self) {
            COUNT.with(|count| count.set(None));
        }
    }

    pub(in crate::kernels::cuda) fn capture<R>(run: impl FnOnce() -> R) -> (R, u64) {
        COUNT.with(|count| {
            assert!(count.get().is_none(), "nested EXL3 launch capture");
            count.set(Some(0));
        });
        // Reset here, not in Drop: callers read the variant after capture returns.
        GATHERED.with(|used| used.set(0));
        let guard = Capture;
        let result = run();
        let count = COUNT.with(|count| count.get().unwrap());
        drop(guard);
        (result, count)
    }

    pub(in crate::kernels::cuda) struct RoutedScope(bool);
    impl Drop for RoutedScope {
        fn drop(&mut self) {
            ROUTED.with(|routed| routed.set(self.0));
        }
    }

    pub(in crate::kernels::cuda) fn routed_scope() -> RoutedScope {
        RoutedScope(ROUTED.with(|routed| routed.replace(true)))
    }

    pub(super) fn recording() -> bool {
        ROUTED.with(Cell::get) && COUNT.with(|count| count.get().is_some())
    }

    /// Which grouped variant a captured leg actually ran: none, all shuffle,
    /// all gather, or mixed. Written by C at the successful projection enqueue.
    pub(in crate::kernels::cuda) fn grouped_variant() -> Option<&'static str> {
        match GATHERED.with(Cell::get) {
            0 => None,
            1 => Some("shuffle"),
            2 => Some("gather"),
            _ => Some("mixed"),
        }
    }

    pub(super) fn linear(
        args: &LinearArgs,
        split_k: u32,
        midrows: bool,
        stream: &RawStream,
        stage: &mut c_int,
    ) -> c_int {
        let mut launches = 0;
        // SAFETY: the caller validated all live device spans and owns the dispatch
        // guard; the two additional writable host outputs outlive this call.
        let status = unsafe {
            inference_cuda_exl3_linear_observed(
                args,
                split_k,
                u32::from(midrows),
                stream.as_ptr(),
                stage,
                &mut launches,
            )
        };
        COUNT.with(|count| count.set(Some(count.get().unwrap().checked_add(launches).unwrap())));
        status
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn midrows_geometry_requires_9_to_144_rows_and_f32_working_spans() {
        for book in [Codebook::ThreeInst, Codebook::Mcg, Codebook::Mul1] {
            for rows in [
                0usize,
                1,
                8,
                9,
                15,
                16,
                31,
                32,
                63,
                64,
                127,
                128,
                144,
                145,
                65536,
                usize::MAX,
            ] {
                let expected = (9..=144).contains(&rows).then(|| {
                    [
                        8192,
                        1024,
                        512,
                        512,
                        rows * 1024,
                        rows * 1024,
                        rows * 512,
                        rows * 512,
                        4,
                    ]
                });
                assert_eq!(
                    midrows_extents(rows, 256, 128, 4, book),
                    expected,
                    "{book:?}/rows{rows}"
                );
            }
        }
        for (k, n, bits) in [
            (0, 128, 4),
            (127, 128, 4),
            (128, 129, 4),
            (256, 128, 9),
            (usize::MAX, 128, 4),
        ] {
            assert!(midrows_extents(9, k, n, bits, Codebook::Mul1).is_none());
        }
    }

    #[test]
    fn batched_geometry_checks_both_input_and_output_products() {
        let book = Codebook::Mul1;
        assert_eq!(
            linear_extents(3, 256, 128, 3, book, 2),
            Some([6144, 1024, 512, 512, 3072, 3072, 1536, 1536, 4])
        );
        for rows in [0, 65537, usize::MAX] {
            assert!(linear_extents(rows, 256, 128, 3, book, 2).is_none());
        }
        assert!(linear_extents(65536, 1 << 22, 128, 2, book, 2).is_none());
        assert!(linear_extents(65536, 128, 1 << 22, 2, book, 2).is_none());
        assert!(linear_extents(65536, 65536, 65536, 16, book, 2).is_some());
    }

    #[test]
    fn split_extents_reserve_all_partial_banks_and_reject_other_counts() {
        for splits in [2u32, 4, 8] {
            for rows in [1, 7, 8, 9, 2048] {
                let needs = linear_extents(rows, 384, 128, 7, Codebook::Mul1, splits).unwrap();
                let banks = if rows <= 8 { splits as usize - 1 } else { 1 };
                assert_eq!(needs[6], rows * 128 * 4 * banks);
                assert_eq!(needs[7], rows * 128 * 4);
            }
        }
        for splits in [0, 1, 3, 16, u32::MAX] {
            assert!(linear_extents(1, 128, 128, 2, Codebook::Mul1, splits).is_none());
            assert!(linear_extents(9, 128, 128, 2, Codebook::Mul1, splits).is_none());
        }
    }

    fn grouped_c_fixture(rows: usize, first: usize) -> LinearArgs {
        use std::ptr::{null, without_provenance, without_provenance_mut};
        LinearArgs {
            trellis: without_provenance(0x10000),
            trellis_bytes: 6144,
            suh: without_provenance(0x20000),
            suh_bytes: 1024,
            svh: without_provenance(0x30000),
            svh_bytes: 512,
            bias: null(),
            bias_bytes: 0,
            x: without_provenance(0x50000),
            x_bytes: rows * 1024,
            xh: without_provenance_mut(0x60000 + first * 1024),
            xh_bytes: rows * 1024,
            z: without_provenance_mut(0x70000 + first * 512),
            z_bytes: rows * 512,
            out: without_provenance_mut(0x80000 + first * 512),
            out_bytes: rows * 512,
            flag: without_provenance_mut(0x90000),
            flag_bytes: 4,
            rows,
            k: 256,
            n: 128,
            bits_twice: 3,
            book: 2,
        }
    }

    #[test]
    fn malformed_grouped_c_arguments_refuse_before_touching_cuda() {
        use std::ptr::{without_provenance, without_provenance_mut};
        let base = [grouped_c_fixture(1, 0), grouped_c_fixture(1, 1)];
        let mut cases = vec![
            ("no groups", vec![]),
            ("49 groups", vec![base[0]; 49]),
            ("zero rows", vec![grouped_c_fixture(0, 0)]),
            ("nine rows", vec![grouped_c_fixture(9, 0)]),
            ("reordered groups", vec![base[1], base[0]]),
        ];
        // Group 8 keeps the overwritten flag aligned if the pair cap regresses.
        let mut pairs: Vec<_> = (0..8).map(|i| grouped_c_fixture(6, i * 6)).collect();
        pairs.push(grouped_c_fixture(1, 48));
        cases.push(("49 pairs", pairs));
        for (name, field) in [("xh gap", 0), ("z gap", 1), ("out gap", 2)] {
            let mut calls = base;
            match field {
                0 => calls[1].xh = without_provenance_mut(0x60404),
                1 => calls[1].z = without_provenance_mut(0x70204),
                _ => calls[1].out = without_provenance_mut(0x80204),
            }
            cases.push((name, calls.to_vec()));
        }
        let mut calls = base;
        calls[1].k = 512;
        calls[1].trellis_bytes = 12288;
        calls[1].suh_bytes = 2048;
        calls[1].x_bytes = 2048;
        calls[1].xh_bytes = 2048;
        cases.push(("different k with valid local extents", calls.to_vec()));
        let mut calls = base;
        calls[1].n = 256;
        calls[1].trellis_bytes = 12288;
        calls[1].svh_bytes = 1024;
        calls[1].z_bytes = 1024;
        calls[1].out_bytes = 1024;
        cases.push(("different n with valid local extents", calls.to_vec()));
        let mut calls = base;
        calls[1].flag = without_provenance_mut(0x91000);
        cases.push(("different flag", calls.to_vec()));
        let mut calls = base;
        // Each call is disjoint locally; group-0 Z overlaps only group-1 XH.
        calls[0].z = without_provenance_mut(0x60600);
        calls[1].z = without_provenance_mut(0x60800);
        cases.push(("Z union overlaps following XH", calls.to_vec()));
        let mut calls = base;
        // This read ends at group-1 out, so only the cross-group union overlaps.
        calls[1].x = without_provenance(0x7fe00);
        cases.push(("input overlaps preceding output", calls.to_vec()));
        let mut calls = base;
        // The 6144-byte read ends at group-1 Z; only group-0 Z overlaps it.
        calls[1].trellis = without_provenance(0x6ea00);
        cases.push(("trellis overlaps preceding Z", calls.to_vec()));
        let mut calls = base;
        calls[1].z_bytes -= 1;
        cases.push(("short split-2 Z bank", calls.to_vec()));
        for (name, calls) in cases {
            let mut stage = -1;
            // SAFETY: malformed metadata must be refused before dereferencing the
            // fabricated spans or using the sentinel stream, as for the batched fixture.
            let status = unsafe {
                inference_cuda_exl3_grouped(
                    calls.as_ptr(),
                    calls.len(),
                    without_provenance_mut(16),
                    &mut stage,
                )
            };
            assert_eq!(
                status,
                super::super::super::STATUS_INVALID_ARGUMENT,
                "{name}"
            );
            assert_eq!(stage, STAGE_NONE, "{name}");
        }
    }

    #[test]
    fn malformed_grouped_rust_arguments_refuse_before_touching_cuda() {
        use super::super::RawDevice;
        use crate::kernels::cuda as adapter;
        use std::ptr::NonNull;
        use std::sync::atomic::{AtomicBool, Ordering};
        let mut memory = vec![0u32; 9000];
        let device = RawDevice {
            ptr: NonNull::new(memory.as_mut_ptr().cast()).unwrap(),
            bytes: memory.len() * 4,
        };
        let range = |offset, bytes| RawRange::new(&device, offset, bytes).unwrap();
        let call = |rows, split_k| LinearRaw {
            trellis: range(0, 6144),
            suh: range(8192, 1024),
            svh: range(9216, 512),
            bias: None,
            x: range(10240, 8192),
            xh: range(18432, 8192),
            z: range(26624, 4096),
            out: range(30720, 4096),
            flag: range(34816, 4),
            rows,
            split_k,
            k: 256,
            n: 128,
            bits_twice: 3,
            book: Codebook::Mul1,
        };
        // Existing raw-wrapper seam: invalid calls must never consume this stream.
        let stream = RawStream(NonNull::dangling());
        for (rows, split, count) in [(1, 2, 0), (1, 2, 49), (0, 2, 1), (9, 2, 1), (1, 4, 1)] {
            let calls: Vec<_> = (0..count).map(|_| call(rows, split)).collect();
            assert_eq!(grouped(&stream, &calls, false), Err(invalid_argument()));
        }
        let mut calls: Vec<_> = (0..6).map(|_| call(8, 2)).collect();
        calls.push(call(1, 2));
        assert_eq!(grouped(&stream, &calls, false), Err(invalid_argument()));
        let mut short = call(1, 2);
        short.z = range(26624, 511);
        assert_eq!(grouped(&stream, &[short], true), Err(invalid_argument()));

        let mut calls = [call(1, 2), call(1, 2)];
        calls[0].z = range(19968, 512);
        calls[1].xh = range(19456, 1024);
        calls[1].z = range(20480, 512);
        calls[1].out = range(31232, 512);
        for call in &calls {
            assert!(linear_args(call, false).is_ok());
        }
        let failure = grouped(&stream, &calls, false).expect_err("cross-group XH/Z alias");
        assert_eq!(failure, invalid_argument());

        // launch_on maps this native refusal to a terminal error, not BadRequest.
        // A local latch exercises that mapper without poisoning the shared backend.
        let terminal = AtomicBool::new(false);
        assert!(adapter::ensure_usable(&terminal).is_ok());
        let error = adapter::native_error(
            &terminal,
            adapter::stage_operation(failure.stage),
            failure.status,
        );
        assert!(matches!(
            error,
            crate::EngError::Backend {
                operation: adapter::ADAPTER_OPERATION,
                ..
            }
        ));
        assert!(terminal.load(Ordering::Acquire));
        assert!(matches!(
            adapter::ensure_usable(&terminal),
            Err(crate::EngError::Backend {
                operation: adapter::EXECUTE_OPERATION,
                ..
            })
        ));
    }

    #[test]
    fn malformed_batched_c_arguments_refuse_before_touching_cuda() {
        let base = LinearArgs {
            trellis: std::ptr::without_provenance(0x10000),
            trellis_bytes: 6144,
            suh: std::ptr::without_provenance(0x20000),
            suh_bytes: 1024,
            svh: std::ptr::without_provenance(0x30000),
            svh_bytes: 512,
            bias: std::ptr::null(),
            bias_bytes: 0,
            x: std::ptr::without_provenance(0x50000),
            x_bytes: 1024,
            xh: std::ptr::without_provenance_mut(0x60000),
            xh_bytes: 1024,
            z: std::ptr::without_provenance_mut(0x70000),
            z_bytes: 512,
            out: std::ptr::without_provenance_mut(0x80000),
            out_bytes: 512,
            flag: std::ptr::without_provenance_mut(0x90000),
            flag_bytes: 4,
            rows: 1,
            k: 256,
            n: 128,
            bits_twice: 3,
            book: 2,
        };
        let mut cases = [base; 16];
        cases[0].rows = 0;
        cases[1].rows = 65537;
        cases[2].k = usize::MAX;
        cases[3].book = 3;
        cases[4].bits_twice = 9;
        cases[5].trellis_bytes -= 1;
        cases[6].suh_bytes -= 1;
        cases[7].out_bytes -= 1;
        cases[8].flag_bytes = 2;
        cases[9].out = base.x.cast_mut();
        cases[10].xh = base.z;
        cases[11].flag = std::ptr::without_provenance_mut(0x90001);
        cases[12].bias_bytes = 4;
        cases[13].out = std::ptr::without_provenance_mut(usize::MAX & !3);
        cases[14].svh = std::ptr::null();
        cases[15].rows = usize::MAX;
        for (index, args) in cases.iter().enumerate() {
            let mut stage = -99;
            // SAFETY: malformed scalar/region arguments are host-rejected before CUDA;
            // the opaque stream and fabricated device addresses must remain unused.
            let status = unsafe {
                inference_cuda_exl3_linear(args, std::ptr::without_provenance_mut(16), &mut stage)
            };
            assert_eq!(status, invalid_argument().status, "case {index}");
            assert_eq!(stage, STAGE_NONE, "case {index}");
        }
        for split_k in [2u32, 4, 8] {
            let need = 512 * (split_k as usize - 1);
            let mut bad = [base; 7];
            for args in &mut bad {
                args.z_bytes = need;
            }
            bad[0].z_bytes -= 1;
            bad[1].z = std::ptr::without_provenance_mut(0x70001);
            let last_bank = 0x70000 + need - 512;
            bad[2].flag = std::ptr::without_provenance_mut(last_bank);
            bad[3].out = std::ptr::without_provenance_mut(last_bank);
            bad[4].xh = std::ptr::without_provenance_mut(last_bank);
            bad[5].suh = std::ptr::without_provenance(last_bank);
            bad[6].z = std::ptr::without_provenance_mut(usize::MAX & !3);
            for (index, args) in bad.iter().enumerate() {
                let mut stage = -99;
                // SAFETY: every full-bank span defect must refuse before CUDA.
                let status = unsafe {
                    inference_cuda_exl3_linear_split(
                        args,
                        split_k,
                        std::ptr::without_provenance_mut(16),
                        &mut stage,
                    )
                };
                assert_eq!(
                    status,
                    invalid_argument().status,
                    "split{split_k}/case{index}"
                );
                assert_eq!(stage, STAGE_NONE, "split{split_k}/case{index}");
            }
        }
        for rows in [0usize, 1, 8, 145, 65536] {
            let args = LinearArgs {
                x: std::ptr::without_provenance(0x10000000),
                x_bytes: rows * 1024,
                xh: std::ptr::without_provenance_mut(0x20000000),
                xh_bytes: rows * 1024,
                z: std::ptr::without_provenance_mut(0x30000000),
                z_bytes: rows * 512,
                out: std::ptr::without_provenance_mut(0x40000000),
                out_bytes: rows * 512,
                rows,
                ..base
            };
            let mut stage = -99;
            // SAFETY: bad midrows counts must refuse before CUDA. For nonzero rows,
            // the regions fit legacy geometry, so no other span defect masks this guard.
            let status = unsafe {
                inference_cuda_exl3_linear_midrows(
                    &args,
                    std::ptr::without_provenance_mut(16),
                    &mut stage,
                )
            };
            assert_eq!(status, invalid_argument().status, "midrows/rows{rows}");
            assert_eq!(stage, STAGE_NONE, "midrows/rows{rows}");
        }
        for split_k in [0, 1, 3, 16, u32::MAX] {
            let mut stage = -99;
            // SAFETY: invalid split counts refuse before the sentinel stream is used.
            let status = unsafe {
                inference_cuda_exl3_linear_split(
                    &base,
                    split_k,
                    std::ptr::without_provenance_mut(16),
                    &mut stage,
                )
            };
            assert_eq!(status, invalid_argument().status, "split{split_k}");
            assert_eq!(stage, STAGE_NONE, "split{split_k}");
        }
    }

    #[test]
    fn malformed_c_arguments_refuse_before_touching_cuda() {
        let base = PrepArgs {
            trellis: std::ptr::without_provenance(0x10000),
            trellis_bytes: 4096,
            out: std::ptr::without_provenance_mut(0x20000),
            out_bytes: 32768,
            k: 128,
            n: 128,
            bits_twice: 4,
            book: 0,
        };
        let mut cases = [base; 8];
        cases[0].k = 0;
        cases[1].n = usize::MAX;
        cases[2].book = 3;
        cases[3].bits_twice = 3;
        cases[4].trellis_bytes -= 1;
        cases[5].out_bytes -= 1;
        cases[6].out = base.trellis.cast_mut();
        cases[7].out = std::ptr::without_provenance_mut(0x20001);
        for (index, args) in cases.iter().enumerate() {
            let mut stage = -99;
            // SAFETY: every case is rejected in host validation before any pointer is
            // dereferenced or CUDA called. The opaque sentinel stream must stay unused.
            let status = unsafe {
                inference_cuda_exl3_prep(args, std::ptr::without_provenance_mut(16), &mut stage)
            };
            assert_eq!(
                status,
                super::super::invalid_argument().status,
                "case {index}"
            );
            assert_eq!(stage, STAGE_NONE, "case {index}");
        }
    }

    #[test]
    fn geometry_admits_only_the_format_rates_and_checked_tile_extents() {
        for book in [Codebook::ThreeInst, Codebook::Mcg, Codebook::Mul1] {
            for bits in 0..=255 {
                let valid = matches!(bits, 2 | 4 | 6 | 8 | 10 | 12 | 14 | 16)
                    || book == Codebook::Mul1 && matches!(bits, 3 | 5 | 7);
                let expected =
                    valid.then_some((128 * 256 / 256 * usize::from(bits) * 16, 128 * 256 * 2));
                assert_eq!(geometry(128, 256, bits, book), expected, "{book:?}/{bits}");
            }
        }
        for (k, n) in [
            (0, 128),
            (128, 0),
            (127, 128),
            (128, 129),
            (usize::MAX, 128),
            (128, usize::MAX),
            (1 << 22, 1 << 22),
        ] {
            assert!(geometry(k, n, 6, Codebook::Mul1).is_none(), "{k}/{n}");
        }
        assert!(geometry(1 << 22, 1024, 16, Codebook::Mul1).is_some());
    }
}
