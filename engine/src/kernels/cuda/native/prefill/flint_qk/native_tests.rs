use super::*;

fn descriptor() -> QkArgs {
    QkArgs {
        q: 0x10000 as *const f32,
        q_bytes: 65536,
        k: 0x50000 as *const f32,
        k_bytes: 8192,
        gamma_q: 0x90000 as *const f32,
        gamma_q_bytes: 512,
        gamma_k: 0xd0000 as *const f32,
        gamma_k_bytes: 512,
        out_q: 0x110000 as *mut f32,
        out_q_bytes: 65536,
        out_k: 0x150000 as *mut f32,
        out_k_bytes: 8192,
        flag: 0x190000 as *mut u32,
        flag_bytes: 4,
        rows: 4,
        position: 0,
        reserved: 0,
    }
}

#[test]
fn native_refusals_are_cpu_only_and_precede_any_cuda_call() {
    type Corrupt = fn(&mut QkArgs);
    let cases: &[(&str, Corrupt)] = &[
        ("zero rows", |a| a.rows = 0),
        ("large rows", |a| a.rows = 4097),
        ("overflow rows", |a| a.rows = usize::MAX),
        ("position overflow", |a| a.position = u32::MAX),
        ("reserved", |a| a.reserved = 1),
        ("short q", |a| a.q_bytes -= 1),
        ("short k", |a| a.k_bytes -= 1),
        ("short gamma q", |a| a.gamma_q_bytes = 511),
        ("short gamma k", |a| a.gamma_k_bytes = 511),
        ("short out q", |a| a.out_q_bytes -= 1),
        ("short out k", |a| a.out_k_bytes -= 1),
        ("short flag", |a| a.flag_bytes = 3),
        ("unaligned q", |a| a.q = a.q.wrapping_byte_add(1)),
        ("unaligned gamma", |a| {
            a.gamma_k = a.gamma_k.wrapping_byte_add(1)
        }),
        ("unaligned out", |a| a.out_k = a.out_k.wrapping_byte_add(1)),
        ("unaligned flag", |a| a.flag = a.flag.wrapping_byte_add(1)),
        ("null q", |a| a.q = std::ptr::null()),
        ("null k", |a| a.k = std::ptr::null()),
        ("null gamma", |a| a.gamma_k = std::ptr::null()),
        ("null out", |a| a.out_q = std::ptr::null_mut()),
        ("null flag", |a| a.flag = std::ptr::null_mut()),
        ("out input alias", |a| a.out_q = a.q.cast_mut()),
        ("out output alias", |a| a.out_k = a.out_q),
        ("gamma output alias", |a| a.gamma_q = a.out_q.cast_const()),
        ("flag input alias", |a| a.flag = a.k.cast_mut().cast()),
        ("address overflow", |a| {
            a.out_q = (usize::MAX - 3) as *mut f32
        }),
    ];
    for &(name, corrupt) in cases {
        let mut args = descriptor();
        corrupt(&mut args);
        for rotary in [false, true] {
            let mut stage = -77;
            // SAFETY: host descriptor is live; each case is invalid and must
            // refuse before using these deliberately non-device addresses.
            let status = unsafe {
                if rotary {
                    inference_cuda_flint_qk_rope(&args, std::ptr::dangling_mut(), &mut stage)
                } else {
                    inference_cuda_flint_qk_norm(&args, std::ptr::dangling_mut(), &mut stage)
                }
            };
            assert_eq!(status, invalid_argument().status, "{name}, rotary={rotary}");
            assert_eq!(stage, STAGE_NONE, "{name}");
        }
    }
    let args = descriptor();
    let mut stage = -77;
    // SAFETY: null stream/descriptor each force refusal before launch.
    let status = unsafe { inference_cuda_flint_qk_norm(&args, std::ptr::null_mut(), &mut stage) };
    assert_eq!(status, invalid_argument().status);
    assert_eq!(stage, STAGE_NONE);
    // SAFETY: null descriptor is checked before any dereference.
    let status = unsafe {
        inference_cuda_flint_qk_rope(std::ptr::null(), std::ptr::dangling_mut(), &mut stage)
    };
    assert_eq!(status, invalid_argument().status);
    assert_eq!(stage, STAGE_NONE);
}
