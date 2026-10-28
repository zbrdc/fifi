use super::*;
use crate::kernels::cuda::native::{RawDevice, STATUS_INVALID_ARGUMENT};
use std::ptr;

unsafe extern "C" {
    fn inference_cuda_mmq_nvfp4_dense_fill(args: *const Nvfp4Args, out: *mut [u64; 27]) -> c_int;
}

fn base() -> Nvfp4Args {
    let at = |address| ptr::without_provenance_mut::<c_void>(address);
    Nvfp4Args {
        matrix: MmqDenseArgsRaw {
            w: at(0x1000_0000),
            y: at(0x2000_0000),
            dst: at(0x3000_0000),
            fixup: at(0x4000_0000),
            fixup_bytes: 1 << 20,
            ty: 40,
            k: 2048,
            n: 6144,
            m: 5,
            ld: 6144,
            col_off: 0,
        },
        y_scale: at(0x5000_0000).cast(),
    }
}

#[test]
fn nvfp4_native_fill_pins_real_dense_shape_and_legacy_refusals() {
    let mut view = [u64::MAX; 27];
    let args = base();
    // SAFETY: fill compares fake addresses without dereferencing; 27 u64s are
    // the existing 216-byte, 8-aligned view. No CUDA operation occurs.
    assert_eq!(
        unsafe { inference_cuda_mmq_nvfp4_dense_fill(&args, &mut view) },
        0
    );
    assert_eq!(view[5], args.y_scale as u64);
    assert_eq!(&view[8..14], &[2048, 6144, 5, 32, 5, 6144]);
    assert_eq!(view[26], 40);
    assert_eq!(size_of::<MmqDenseArgsRaw>(), 64);
    assert_eq!(
        super::super::tests::fill(&args.matrix).err(),
        Some(STATUS_INVALID_ARGUMENT)
    );
    let cases: [fn(&mut Nvfp4Args); 8] = [
        |a| a.matrix.ty = 8,
        |a| a.matrix.k = 384,
        |a| a.matrix.m = 0,
        |a| a.matrix.ld += 1,
        |a| a.y_scale = ptr::null(),
        |a| a.y_scale = ptr::without_provenance(0x5000_0001),
        |a| a.y_scale = a.matrix.dst.cast(),
        |a| a.y_scale = a.matrix.fixup.cast(),
    ];
    for mutate in cases {
        let mut args = base();
        mutate(&mut args);
        // SAFETY: pure fill contract above, including invalid-address comparisons.
        assert_eq!(
            unsafe { inference_cuda_mmq_nvfp4_dense_fill(&args, &mut view) },
            STATUS_INVALID_ARGUMENT
        );
    }
}

#[test]
fn review_dense_native_tail_factors_cannot_overlap_writes() {
    for before_fixup in [false, true] {
        let mut args = base();
        let written = if before_fixup {
            args.matrix.fixup
        } else {
            args.matrix.dst
        };
        args.y_scale = ptr::without_provenance(written.addr() - 20);
        let mut view = [u64::MAX; 27];
        // SAFETY: pure fill compares the synthetic ranges, without dereferencing.
        assert_eq!(
            unsafe { inference_cuda_mmq_nvfp4_dense_fill(&args, &mut view) },
            STATUS_INVALID_ARGUMENT
        );
        assert_eq!(view, [u64::MAX; 27]);
    }
}

#[test]
fn review_dense_native_weight_indices_cannot_wrap() {
    let at = |address| ptr::without_provenance_mut::<c_void>(address);
    let mut args = base();
    args.matrix.w = at(1 << 40);
    args.matrix.y = at(2 << 40);
    args.matrix.dst = at(3 << 40);
    args.matrix.fixup = at(4 << 40);
    args.y_scale = at(5 << 40).cast();
    args.matrix.k = 32768;
    args.matrix.m = 1;
    let limit = i32::MAX / (32768 / 64);
    for (n, expected) in [
        (limit, 0),
        (limit + 1, STATUS_INVALID_ARGUMENT),
        (4_194_432, STATUS_INVALID_ARGUMENT),
    ] {
        args.matrix.n = n;
        args.matrix.ld = n;
        let mut view = [u64::MAX; 27];
        // SAFETY: pure fill; far-apart synthetic addresses also isolate the index guard.
        assert_eq!(
            unsafe { inference_cuda_mmq_nvfp4_dense_fill(&args, &mut view) },
            expected
        );
        if expected != 0 {
            assert_eq!(view, [u64::MAX; 27]);
        }
    }
}

#[test]
fn nvfp4_raw_ranges_refuse_short_factors_and_written_aliases() {
    let shape = Nvfp4DenseShape::new(2048, 6144, 5).unwrap();
    let w = RawDevice::unbacked(shape.needs.w);
    let y = RawDevice::unbacked(shape.needs.y);
    let dst = RawDevice::unbacked(shape.needs.dst);
    let fixup = RawDevice::unbacked(1 << 20);
    let factors = RawDevice::unbacked(shape.factor_bytes + 4);
    let io = MmqDenseRaw {
        w: RawRange::new(&w, 0, shape.needs.w).unwrap(),
        y: RawRange::new(&y, 0, shape.needs.y).unwrap(),
        dst: RawRange::new(&dst, 0, shape.needs.dst).unwrap(),
        fixup: RawRange::new(&fixup, 0, 1 << 20).unwrap(),
        ints: shape.ints,
    };
    check(
        &io,
        &RawRange::new(&factors, 0, shape.factor_bytes).unwrap(),
    )
    .unwrap();
    for range in [
        RawRange::new(&factors, 0, shape.factor_bytes - 1).unwrap(),
        RawRange::new(&factors, 1, shape.factor_bytes).unwrap(),
        RawRange::new(&dst, 0, shape.factor_bytes).unwrap(),
        RawRange::new(&fixup, 0, shape.factor_bytes).unwrap(),
    ] {
        assert!(check(&io, &range).is_err());
    }
}
