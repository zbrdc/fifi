use super::*;

fn refused<T>(entry: unsafe extern "C" fn(*const T, *mut c_void, *mut c_int) -> c_int, args: &T) {
    let mut stage = -99;
    // SAFETY: each case violates a checked precondition. The C entry must return
    // before touching the fake stream/device addresses; args and stage are live.
    let code = unsafe { entry(args, std::ptr::dangling_mut::<c_void>(), &mut stage) };
    assert_eq!(code, -1, "invalid input must not reach a CUDA launch");
    assert_eq!(stage, STAGE_NONE);
}

fn logits() -> RouterLogitsArgsRaw {
    RouterLogitsArgsRaw {
        x: 0x1000usize as *const f32,
        x_bytes: 16,
        w: 0x2000usize as *const f32,
        w_bytes: 128,
        logits: 0x3000usize as *mut f32,
        logits_bytes: 32,
        m: 1,
        hidden: 4,
        n_expert: 8,
    }
}

fn select() -> RouterSigmoidArgsRaw {
    RouterSigmoidArgsRaw {
        logits: 0x1000usize as *const f32,
        logits_bytes: 32,
        ids: 0x2000usize as *mut c_int,
        ids_bytes: 24,
        weights: 0x3000usize as *mut f32,
        weights_bytes: 24,
        status: 0x4000usize as *mut c_int,
        status_bytes: 4,
        m: 1,
        n_expert: 8,
        top_k: 6,
    }
}

#[test]
fn split_router_logits_c_abi_rejects_geometry_sizes_alignment_wrap_and_alias() {
    for case in 0..10 {
        let mut a = logits();
        match case {
            0 => a.m = 0,
            1 => a.hidden = 5,
            2 => a.n_expert = 257,
            3 => a.x_bytes = 15,
            4 => a.w_bytes = 127,
            5 => a.logits_bytes = 31,
            6 => a.x = 0x1004usize as *const f32,
            7 => a.logits = (usize::MAX - 3) as *mut f32,
            8 => a.logits = a.x.cast_mut(),
            9 => a.logits = a.w.cast_mut(),
            _ => unreachable!(),
        }
        refused(router_logits_launch_raw, &a);
    }
}

#[test]
fn split_router_select_c_abi_rejects_geometry_short_align_wrap_and_alias() {
    for case in 0..12 {
        let mut a = select();
        match case {
            0 => a.m = 4097,
            1 => a.top_k = 9,
            2 => a.logits_bytes = 31,
            3 => a.ids_bytes = 23,
            4 => a.weights_bytes = 23,
            5 => a.status_bytes = 3,
            6 => a.ids = 0x2001usize as *mut c_int,
            7 => a.logits = (usize::MAX - 3) as *const f32,
            8 => a.ids = a.logits.cast_mut().cast(),
            9 => a.weights = a.ids.cast(),
            10 => a.status = a.weights.cast(),
            11 => a.status = a.logits.cast_mut().cast(),
            _ => unreachable!(),
        }
        refused(router_sigmoid_launch_raw, &a);
    }
}

#[test]
fn split_router_requires_explicit_stream_before_any_cuda_work() {
    let mut stage = -99;
    // SAFETY: NULL stream is rejected before CUDA runtime use on otherwise valid metadata.
    assert_eq!(
        unsafe { router_logits_launch_raw(&logits(), std::ptr::null_mut(), &mut stage) },
        -1
    );
    assert_eq!(stage, STAGE_NONE);
    // SAFETY: same explicit-stream precondition for the selection entry.
    assert_eq!(
        unsafe { router_sigmoid_launch_raw(&select(), std::ptr::null_mut(), &mut stage) },
        -1
    );
    assert_eq!(stage, STAGE_NONE);
}
