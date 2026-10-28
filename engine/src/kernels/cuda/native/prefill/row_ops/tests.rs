use super::*;

fn refused<T>(entry: unsafe extern "C" fn(*const T, *mut c_void, *mut c_int) -> c_int, args: &T) {
    let mut stage = -99;
    // SAFETY: every case has an invalid shape/span. The entry must reject it before
    // using the deliberately non-live stream or any device address; args/stage are live.
    let status = unsafe { entry(args, std::ptr::dangling_mut::<c_void>(), &mut stage) };
    assert_eq!(status, -1, "must reject before a CUDA launch");
    assert_eq!(stage, STAGE_NONE);
}

fn unary() -> UnaryArgs {
    UnaryArgs {
        x: 0x1000usize as *const f32,
        x_bytes: 16,
        out: 0x3000usize as *mut f32,
        out_bytes: 16,
        flag: 0x4000usize as *mut u32,
        flag_bytes: 4,
        rows: 1,
        columns: 4,
    }
}

#[test]
fn row_ops_unary_c_abi_rejects_geometry_alias_short_alignment_and_address_wrap() {
    for case in 0..6 {
        let mut a = unary();
        match case {
            0 => a.rows = 0,
            1 => a.x_bytes = 3,
            2 => a.out = a.x.cast_mut(),
            3 => a.flag = a.out.cast(),
            4 => a.x = (usize::MAX - 3) as *const f32,
            5 => a.x = 0x1001usize as *const f32,
            _ => unreachable!(),
        }
        refused(inference_cuda_row_sigmoid_launch, &a);
    }
}

#[test]
fn row_ops_multiply_c_abi_rejects_bad_broadcast_and_written_aliases() {
    for case in 0..4 {
        let b = unary();
        let mut a = MulArgs {
            x: b.x,
            x_bytes: b.x_bytes,
            rhs: 0x2000usize as *const f32,
            rhs_bytes: 16,
            out: b.out,
            out_bytes: b.out_bytes,
            flag: b.flag,
            flag_bytes: b.flag_bytes,
            rows: 1,
            columns: 4,
            rhs_columns: 4,
        };
        match case {
            0 => a.rhs_columns = 2,
            1 => a.rhs_bytes = 3,
            2 => a.out = a.rhs.cast_mut(),
            3 => a.flag_bytes = 3,
            _ => unreachable!(),
        }
        refused(inference_cuda_row_mul_launch, &a);
    }
}

#[test]
fn row_ops_gather_c_abi_rejects_bad_source_and_index_spans() {
    for case in 0..4 {
        let b = unary();
        let mut a = GatherArgs {
            x: b.x,
            x_bytes: b.x_bytes,
            indices: 0x2000usize as *const i32,
            indices_bytes: 4,
            out: b.out,
            out_bytes: b.out_bytes,
            flag: b.flag,
            flag_bytes: b.flag_bytes,
            rows: 1,
            columns: 4,
            source_rows: 1,
        };
        match case {
            0 => a.source_rows = 0,
            1 => a.indices_bytes = 3,
            2 => a.indices = 0x2001usize as *const i32,
            3 => a.out = a.indices.cast_mut().cast(),
            _ => unreachable!(),
        }
        refused(inference_cuda_row_gather_launch, &a);
    }
}

#[test]
fn row_ops_rank_sum_c_abi_rejects_rank_source_extent_and_alias() {
    for case in 0..4 {
        let b = unary();
        let mut a = SumArgs {
            x: b.x,
            x_bytes: 32,
            out: b.out,
            out_bytes: b.out_bytes,
            flag: b.flag,
            flag_bytes: b.flag_bytes,
            rows: 1,
            columns: 4,
            rank: 2,
        };
        match case {
            0 => a.rank = 0,
            1 => a.rank = 33,
            2 => a.x_bytes = 16,
            3 => a.out = a.x.cast_mut(),
            _ => unreachable!(),
        }
        refused(inference_cuda_row_rank_sum_launch, &a);
    }
}
