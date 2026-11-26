use super::*;
use crate::kernels::cuda::context::DeviceBuffer;
use crate::weights::{GgufType, dequantize};

/// The kernel type and the host type of each block layout the native dispatcher serves: the
/// seven layouts of the pinned Flash-Next checkpoint's matrices (Q6_K, Q8_0, IQ2_S, IQ3_XXS,
/// IQ4_NL, IQ4_XS and Q2_0) and every other type a checkpoint may carry into these kernels.
pub(in crate::kernels::cuda) const CASES: [(QuantType, GgufType); 18] = [
    (QuantType::Q6K, GgufType::Q6K),
    (QuantType::Q80, GgufType::Q80),
    (QuantType::Iq2S, GgufType::Iq2S),
    (QuantType::Iq3Xxs, GgufType::Iq3Xxs),
    (QuantType::Iq4Nl, GgufType::Iq4Nl),
    (QuantType::Iq4Xs, GgufType::Iq4Xs),
    (QuantType::Q20, GgufType::Q20),
    (QuantType::Iq2Xxs, GgufType::Iq2Xxs),
    (QuantType::Q40, GgufType::Q40),
    (QuantType::Iq2Xs, GgufType::Iq2Xs),
    (QuantType::Iq3S, GgufType::Iq3S),
    (QuantType::Iq1S, GgufType::Iq1S),
    (QuantType::Iq1M, GgufType::Iq1M),
    (QuantType::Q50, GgufType::Q50),
    (QuantType::Q2K, GgufType::Q2K),
    (QuantType::Q3K, GgufType::Q3K),
    (QuantType::Q4K, GgufType::Q4K),
    (QuantType::Q5K, GgufType::Q5K),
];

/// Every refusal of this module is a `BadRequest`.
fn refused<T>(result: Result<T, EngError>) -> EngError {
    match result {
        Err(error @ EngError::BadRequest(_)) => error,
        Err(other) => panic!("refused with {other:?}, not BadRequest"),
        Ok(_) => panic!("accepted"),
    }
}

#[test]
fn dense_geometry_and_sizes_are_checked_before_any_device() {
    let need = dense_needs(QuantType::Q6K, 512, 3, 1).unwrap();
    assert_eq!((need.w, need.x, need.y), (3 * 2 * 210, 512 * 4, 3 * 4));
    let need = dense_needs(QuantType::Iq3Xxs, 256, 5, 3).unwrap();
    assert_eq!((need.w, need.x, need.y), (5 * 98, 256 * 4 * 3, 5 * 3 * 4));
    for (k, rows, ncols) in [
        (0usize, 1usize, 1usize),
        (255, 1, 1),
        (257, 1, 1),
        (256, 0, 1),
        (256, 1, 0),
        (256, 1, 9),
        (1 << 31, 1, 1),
        (256, 1 << 31, 1),
        (usize::MAX, 1, 1),
    ] {
        let error = refused(dense_needs(QuantType::Q6K, k, rows, ncols));
        let EngError::BadRequest(text) = error else {
            panic!()
        };
        eprintln!("dense k={k} rows={rows} ncols={ncols}: {text}");
    }
}

#[test]
fn expert_geometry_and_sizes_are_checked_before_any_device() {
    let mut shape = ExpertShape {
        ty: QuantType::Iq2S,
        k: 512,
        rows: 256,
        n_expert: 5,
        n_used: 3,
        x_per_slot: false,
        y_slot_stride: 260,
        gate: false,
    };
    let need = expert_needs(&shape).unwrap();
    assert_eq!(
        (need.w_stack, need.ids, need.x, need.y),
        (5 * 256 * 2 * 82, 12, 512 * 4, (2 * 260 + 256) * 4)
    );
    shape.gate = true;
    assert_eq!(expert_needs(&shape).unwrap().y, (2 * 260 + 256) * 4);
    // A gate stack goes only with the shared activation row, so clear it before per-slot x.
    shape.gate = false;
    shape.x_per_slot = true;
    assert_eq!(expert_needs(&shape).unwrap().x, 3 * 512 * 4);
    for name in [
        "k",
        "rows",
        "n_expert",
        "n_used low",
        "n_used high",
        "n_used over the stack",
        "stride",
        "gate with per-slot x",
        "expert overflow",
    ] {
        let mut bad = shape;
        match name {
            "k" => bad.k = 255,
            "rows" => bad.rows = 0,
            "n_expert" => bad.n_expert = 0,
            "n_used low" => bad.n_used = 0,
            "n_used high" => bad.n_used = MAX_USED + 1,
            "n_used over the stack" => bad.n_used = 6,
            "stride" => bad.y_slot_stride = 255,
            "gate with per-slot x" => bad.gate = true,
            "expert overflow" => bad.n_expert = usize::MAX,
            _ => unreachable!(),
        }
        let error = refused(expert_needs(&bad));
        let EngError::BadRequest(text) = error else {
            panic!()
        };
        eprintln!("expert {name}: {text}");
    }
}

/// Finite f16 scales, all normal: the smallest normal, +-0.25, 1, -2 and 32.
const SCALES: [u16; 6] = [0x0400, 0x2c00, 0xac00, 0x3c00, 0xbc00, 0x4c00];

/// `rows` rows of `k` values of `ty` as random, valid blocks with finite scales, and
/// `ncols` deterministic activation rows of `k` values.
pub(in crate::kernels::cuda) fn fixture(
    ty: QuantType,
    rows: usize,
    k: usize,
    ncols: usize,
    seed: u32,
) -> (Vec<u8>, Vec<f32>) {
    let block = ty.block_size();
    let bytes = ty.block_bytes();
    let mut state = seed;
    let mut blocks = vec![0u8; rows * (k / block) * bytes];
    for (i, b) in blocks.chunks_exact_mut(bytes).enumerate() {
        for value in b.iter_mut() {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            *value = (state >> 24) as u8;
        }
        if ty == QuantType::Iq1M {
            // Its f16 scale is the top nibble of each of four words at bytes 48..56; bits
            // 4w..4w+3 of the scale are the top nibble of word w, its high byte's.
            let scale = SCALES[i % SCALES.len()];
            for word in 0..4 {
                let high = 48 + 2 * word + 1;
                b[high] = (b[high] & 0x0f) | ((((scale >> (4 * word)) & 0xf) as u8) << 4);
            }
            continue;
        }
        // Every other layout keeps f16 super-block scales at fixed offsets: Q6_K and Q3_K
        // close the block with `d`; Q4_K and Q5_K open with `d` then `dmin`; Q2_K closes
        // with `d` then `dmin`; the rest open with `d` alone.
        let offsets: &[usize] = match ty {
            QuantType::Q6K | QuantType::Q3K => &[bytes - 2],
            QuantType::Q4K | QuantType::Q5K => &[0, 2],
            QuantType::Q2K => &[bytes - 4, bytes - 2],
            _ => &[0],
        };
        for (j, &at) in offsets.iter().enumerate() {
            b[at..at + 2].copy_from_slice(&SCALES[(i + j) % SCALES.len()].to_le_bytes());
        }
    }
    let x = (0..k * ncols)
        .map(|i| ((i * 13 % 37) as f32 - 18.0) / 9.0)
        .collect();
    (blocks, x)
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_ne_bytes()).collect()
}

fn f32_words(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_ne_bytes(*b))
        .collect()
}

/// Reads a whole device buffer back.
fn read(output: &DeviceBuffer, stream: &DeviceStream) -> Vec<u8> {
    let mut actual = vec![0u8; output.bytes()];
    output.read_at(stream, 0, &mut actual).unwrap();
    actual
}

/// Runs the exact dense projection into a guarded output range and returns `y`, after
/// checking that the bytes on both sides of the range are untouched.
#[allow(clippy::too_many_arguments)] // test helper: the kernel's inputs go in one by one
fn run_dense(
    ctx: &CudaContext,
    stream: &DeviceStream,
    ty: QuantType,
    blocks: &[u8],
    x: &[f32],
    rows: usize,
    k: usize,
    ncols: usize,
) -> Vec<f32> {
    let w = ctx.upload(blocks).unwrap();
    let xb = f32_bytes(x);
    assert_eq!(xb.len(), k * ncols * 4);
    let input = ctx.upload(&xb).unwrap();
    let y_bytes = rows * ncols * 4;
    let output = ctx.upload(&vec![0xa5; y_bytes + 8]).unwrap();
    ctx.matvec_gguf_f32(
        stream,
        DenseF32Call {
            ty,
            w: w.range(0, blocks.len()).unwrap(),
            x: input.range(0, xb.len()).unwrap(),
            y: output.range(4, y_bytes).unwrap(),
            k,
            rows,
            ncols,
        },
    )
    .unwrap();
    let actual = read(&output, stream);
    assert_eq!(&actual[..4], &[0xa5; 4]);
    assert_eq!(&actual[y_bytes + 4..], &[0xa5; 4]);
    f32_words(&actual[4..y_bytes + 4])
}

/// The f64 dot of a host-dequantized row with the activation, and the sum of the products'
/// absolute values (the scale of the error bound).
fn reference(weight_row: &[f32], x: &[f32]) -> (f64, f64) {
    weight_row
        .iter()
        .zip(x)
        .fold((0.0f64, 0.0f64), |(sum, abs), (&w, &a)| {
            let p = f64::from(w) * f64::from(a);
            (sum + p, abs + p.abs())
        })
}

/// The error bound of one output: a lane's chain of `k / 32` fused multiply-adds and the
/// five-step warp reduction round a running sum of that size by half an ulp at a time.
fn bound(k: usize, absolute: f64) -> f64 {
    2e-7 * (k / 32 + 8) as f64 * absolute + 1e-30
}

#[test]
#[ignore = "requires an admitted CUDA device"]
fn dense_matches_the_host_dequantizer_in_f64_for_every_type() {
    let ctx = CudaContext::new().unwrap();
    let stream = ctx.stream().unwrap();
    let mut worst = 0.0f64;
    for (ty, host) in CASES {
        let block = ty.block_size();
        for (rows, k) in [(1, block), (5, 4 * block), (9, 3 * block), (17, 2 * block)] {
            let (blocks, x) = fixture(ty, rows, k, 1, 0x6f32_1234);
            let got = run_dense(&ctx, &stream, ty, &blocks, &x, rows, k, 1);
            let weights = dequantize(host, &blocks).unwrap();
            assert_eq!(weights.len(), rows * k);
            for (row, (weight_row, &value)) in weights.chunks_exact(k).zip(&got).enumerate() {
                let (expected, absolute) = reference(weight_row, &x);
                let limit = bound(k, absolute);
                let ratio = (f64::from(value) - expected).abs() / limit;
                worst = worst.max(ratio);
                assert!(
                    value.is_finite() && ratio <= 1.0,
                    "type={ty:?} rows={rows} k={k} row={row} got={value} expected={expected} bound={limit}"
                );
            }
        }
    }
    eprintln!("exact dense projection: largest error as a fraction of its bound {worst:.4}");
}

#[test]
#[ignore = "requires an admitted CUDA device"]
fn a_column_of_a_wide_launch_is_bit_identical_to_a_one_column_launch() {
    let ctx = CudaContext::new().unwrap();
    let stream = ctx.stream().unwrap();
    let (rows, ncols) = (5, 3);
    for (ty, _) in CASES {
        let k = 4 * ty.block_size();
        let (blocks, x) = fixture(ty, rows, k, ncols, 0x51ed_270b);
        let wide = run_dense(&ctx, &stream, ty, &blocks, &x, rows, k, ncols);
        for c in 0..ncols {
            let single = run_dense(
                &ctx,
                &stream,
                ty,
                &blocks,
                &x[c * k..(c + 1) * k],
                rows,
                k,
                1,
            );
            assert_eq!(
                wide[c * rows..(c + 1) * rows],
                single,
                "type={ty:?} column={c}"
            );
        }
    }
}

/// Runs one exact expert launch into a guarded output range; returns the output bytes whole
/// (guard bytes included), for the caller to check the slot padding.
#[allow(clippy::too_many_arguments)] // test helper: the kernel's inputs go in one by one
fn run_expert(
    ctx: &CudaContext,
    stream: &DeviceStream,
    ty: QuantType,
    up: &[u8],
    gate: Option<&[u8]>,
    ids: &[i32],
    x: &[f32],
    x_per_slot: bool,
    rows: usize,
    k: usize,
    n_expert: usize,
    y_slot_stride: usize,
) -> Vec<u8> {
    let w = ctx.upload(up).unwrap();
    let xb = f32_bytes(x);
    let input = ctx.upload(&xb).unwrap();
    let idb: Vec<u8> = ids.iter().flat_map(|i| i.to_le_bytes()).collect();
    let idbuf = ctx.upload(&idb).unwrap();
    let gatebuf = gate.map(|g| ctx.upload(g).unwrap());
    let y_bytes = y_slot_stride * ids.len() * 4;
    let output = ctx.upload(&vec![0xa5; y_bytes + 8]).unwrap();
    ctx.matvec_gguf_f32_expert(
        stream,
        ExpertF32Call {
            ty,
            w_stack: w.range(0, up.len()).unwrap(),
            ids: idbuf.range(0, idb.len()).unwrap(),
            x: input.range(0, xb.len()).unwrap(),
            x_per_slot,
            y: output.range(4, y_bytes).unwrap(),
            y_slot_stride,
            k,
            rows,
            n_expert,
            n_used: ids.len(),
            glu_gate: gatebuf.as_ref().map(|g| g.range(0, g.bytes()).unwrap()),
        },
    )
    .unwrap();
    let actual = read(&output, stream);
    assert_eq!(&actual[..4], &[0xa5; 4]);
    assert_eq!(&actual[y_bytes + 4..], &[0xa5; 4]);
    actual
}

#[test]
#[ignore = "requires an admitted CUDA device"]
fn expert_slots_follow_the_ids_fuse_the_gate_and_keep_the_padding() {
    const K: usize = 512;
    const ROWS: usize = 256;
    const N_EXPERT: usize = 5;
    const STRIDE: usize = ROWS + 4;
    const USED: usize = 5;
    // Slot 3 names an expert outside the stack and must come back as NaN rows.
    const IDS: [i32; USED] = [3, 0, 4, 7, 3];
    let ctx = CudaContext::new().unwrap();
    let stream = ctx.stream().unwrap();
    let mut worst = 0.0f64;
    for (ty, host) in CASES {
        let (up, x) = fixture(ty, N_EXPERT * ROWS, K, USED, 0x6f32_1234);
        let (gate, _) = fixture(ty, N_EXPERT * ROWS, K, 1, 0x0bad_f00d);
        let up_f32 = dequantize(host, &up).unwrap();

        let plain = run_expert(
            &ctx,
            &stream,
            ty,
            &up,
            None,
            &IDS,
            &x[..K],
            false,
            ROWS,
            K,
            N_EXPERT,
            STRIDE,
        );
        let fused = run_expert(
            &ctx,
            &stream,
            ty,
            &up,
            Some(&gate),
            &IDS,
            &x[..K],
            false,
            ROWS,
            K,
            N_EXPERT,
            STRIDE,
        );
        let gate_only = run_expert(
            &ctx,
            &stream,
            ty,
            &gate,
            None,
            &IDS,
            &x[..K],
            false,
            ROWS,
            K,
            N_EXPERT,
            STRIDE,
        );
        let per_slot = run_expert(
            &ctx, &stream, ty, &up, None, &IDS, &x, true, ROWS, K, N_EXPERT, STRIDE,
        );
        for (slot, &id) in IDS.iter().enumerate() {
            let rows = &plain[4 + slot * STRIDE * 4..4 + (slot * STRIDE + ROWS) * 4];
            let padding = &plain[4 + (slot * STRIDE + ROWS) * 4..4 + (slot + 1) * STRIDE * 4];
            assert_eq!(padding, &[0xa5; 4 * 4], "type={ty:?} slot={slot} padding");
            assert_eq!(
                &fused[4 + (slot * STRIDE + ROWS) * 4..4 + (slot + 1) * STRIDE * 4],
                &[0xa5; 16]
            );
            let got = f32_words(rows);
            if !(0..N_EXPERT as i32).contains(&id) {
                assert!(got.iter().all(|v| v.is_nan()), "type={ty:?} slot={slot}");
                assert!(
                    f32_words(&fused[4 + slot * STRIDE * 4..4 + (slot * STRIDE + ROWS) * 4])
                        .iter()
                        .all(|v| v.is_nan()),
                    "type={ty:?} slot={slot} fused"
                );
                continue;
            }
            let weights = &up_f32[id as usize * ROWS * K..(id as usize + 1) * ROWS * K];
            let per_slot_got =
                f32_words(&per_slot[4 + slot * STRIDE * 4..4 + (slot * STRIDE + ROWS) * 4]);
            let fused_got =
                f32_words(&fused[4 + slot * STRIDE * 4..4 + (slot * STRIDE + ROWS) * 4]);
            let gate_got =
                f32_words(&gate_only[4 + slot * STRIDE * 4..4 + (slot * STRIDE + ROWS) * 4]);
            for row in 0..ROWS {
                let (expected, absolute) = reference(&weights[row * K..(row + 1) * K], &x[..K]);
                let ratio = (f64::from(got[row]) - expected).abs() / bound(K, absolute);
                worst = worst.max(ratio);
                assert!(ratio <= 1.0, "type={ty:?} slot={slot} row={row} plain");
                // The per-slot rows ride their own activation row.
                let (expected, absolute) = reference(
                    &weights[row * K..(row + 1) * K],
                    &x[slot * K..(slot + 1) * K],
                );
                let ratio = (f64::from(per_slot_got[row]) - expected).abs() / bound(K, absolute);
                worst = worst.max(ratio);
                assert!(ratio <= 1.0, "type={ty:?} slot={slot} row={row} per-slot");
                // The fused value is the up dot times silu of the gate dot, each as the
                // kernel rounded them; the host silu differs from expf by a few ulp of its
                // argument, which silu turns into a relative error of at most that times |g|.
                let want = got[row] * (gate_got[row] / (1.0 + (-gate_got[row]).exp()));
                let tolerance =
                    4e-7 * (1.0 + f64::from(gate_got[row].abs())) * f64::from(want.abs()) + 1e-30;
                let fused = fused_got[row];
                let error = (f64::from(fused) - f64::from(want)).abs();
                assert!(
                    error <= tolerance,
                    "type={ty:?} slot={slot} row={row} fused: {fused} vs {want} ({error} > {tolerance})"
                );
            }
        }
    }
    eprintln!("exact expert projection: largest error as a fraction of its bound {worst:.4}");
}

#[test]
#[ignore = "requires an admitted CUDA device"]
fn refusals_happen_before_the_native_call_and_leave_the_context_usable() {
    let ctx = CudaContext::new().unwrap();
    let stream = ctx.stream().unwrap();
    let (ty, k, rows) = (QuantType::Q6K, 256usize, 256usize);
    let w_bytes = rows * 210;
    let x_bytes = k * 4;
    let y_bytes = rows * 4;
    let w = ctx.upload(&vec![0; w_bytes + 4]).unwrap();
    let x = ctx.upload(&vec![0; x_bytes + 20]).unwrap();
    let y = ctx.upload(&vec![0xa5; y_bytes + 8]).unwrap();
    let dense = |w, x, y, k, rows, ncols| {
        ctx.matvec_gguf_f32(
            &stream,
            DenseF32Call {
                ty,
                w,
                x,
                y,
                k,
                rows,
                ncols,
            },
        )
    };
    let (w_all, x_all, y_all) = (
        w.range(0, w_bytes).unwrap(),
        x.range(0, x_bytes).unwrap(),
        y.range(4, y_bytes).unwrap(),
    );
    for (k, rows, ncols) in [
        (255usize, 256usize, 1usize),
        (0, 256, 1),
        (256, 0, 1),
        (256, 256, 0),
        (256, 256, 9),
    ] {
        assert!(matches!(
            dense(w_all, x_all, y_all, k, rows, ncols),
            Err(EngError::BadRequest(_))
        ));
    }
    for (weight, input, output) in [
        (w.range(0, w_bytes - 1).unwrap(), x_all, y_all),
        (w_all, x.range(0, x_bytes - 4).unwrap(), y_all),
        (w_all, x_all, y.range(4, y_bytes - 4).unwrap()),
        (w.range(2, w_bytes).unwrap(), x_all, y_all),
        (w_all, x.range(4, x_bytes).unwrap(), y_all),
        (w_all, x_all, y.range(6, y_bytes).unwrap()),
        (w_all, x_all, x.range(0, y_bytes).unwrap()),
        (w_all, x_all, w.range(0, y_bytes).unwrap()),
    ] {
        assert!(matches!(
            dense(weight, input, output, k, rows, 1),
            Err(EngError::BadRequest(_))
        ));
    }
    let ids: Vec<u8> = [0i32, 1].iter().flat_map(|i| i.to_le_bytes()).collect();
    let ids = ctx.upload(&ids).unwrap();
    let ids = ids.range(0, 8).unwrap();
    // Two experts of the Q6_K rows: the size a gate stack or an expert stack must have. The
    // buffer also holds an output apart from both stacks, sized for both used slots.
    let stack_bytes = 2 * w_bytes;
    let apart_bytes = 2 * y_bytes;
    let big_bytes = 2 * stack_bytes + apart_bytes + 8;
    let big = ctx.upload(&vec![0; big_bytes]).unwrap();
    let expert = |call: ExpertF32Call<'_>| ctx.matvec_gguf_f32_expert(&stream, call);
    let call = |x, y, gate, n_used, stride, per_slot| ExpertF32Call {
        ty,
        w_stack: big.range(0, stack_bytes).unwrap(),
        ids,
        x,
        x_per_slot: per_slot,
        y,
        y_slot_stride: stride,
        k,
        rows,
        n_expert: 2,
        n_used,
        glu_gate: gate,
    };
    for (n_used, stride) in [(0usize, 256usize), (33, 256), (3, 256), (2, 255)] {
        assert!(matches!(
            expert(call(x_all, y_all, None, n_used, stride, false)),
            Err(EngError::BadRequest(_))
        ));
    }
    let gate = big.range(stack_bytes, stack_bytes).unwrap();
    let apart_y = big.range(2 * stack_bytes, apart_bytes).unwrap();
    assert!(matches!(
        expert(call(x_all, apart_y, Some(gate), 2, 256, true)),
        Err(EngError::BadRequest(_))
    ));
    assert!(matches!(
        expert(call(
            x_all,
            apart_y,
            Some(big.range(stack_bytes, stack_bytes - 1).unwrap()),
            2,
            256,
            false
        )),
        Err(EngError::BadRequest(_))
    ));
    // The output inside the gate stack's claim, and inside the expert stack's.
    for start in [stack_bytes - 512, 512] {
        assert!(matches!(
            expert(call(
                x_all,
                big.range(start, y_bytes).unwrap(),
                Some(gate),
                2,
                256,
                false
            )),
            Err(EngError::BadRequest(_))
        ));
    }
    // The same stacks with the output apart from both are accepted.
    expert(call(x_all, apart_y, Some(gate), 2, 256, false)).unwrap();
    let foreign = CudaContext::new().unwrap();
    let other_stream = foreign.stream().unwrap();
    let other = foreign.upload(&vec![0; x_bytes + 16]).unwrap();
    assert!(matches!(
        expert(call(
            other.range(0, x_bytes).unwrap(),
            apart_y,
            None,
            2,
            256,
            false
        )),
        Err(EngError::BadRequest(_))
    ));
    assert!(matches!(
        ctx.matvec_gguf_f32_expert(&other_stream, call(x_all, apart_y, None, 2, 256, false)),
        Err(EngError::BadRequest(_))
    ));
    assert!(matches!(
        dense(w_all, other.range(0, x_bytes).unwrap(), y_all, k, rows, 1),
        Err(EngError::BadRequest(_))
    ));
    assert!(matches!(
        ctx.matvec_gguf_f32(
            &other_stream,
            DenseF32Call {
                ty,
                w: w_all,
                x: x_all,
                y: y_all,
                k,
                rows,
                ncols: 1,
            },
        ),
        Err(EngError::BadRequest(_))
    ));
    // No refusal reached the native side: nothing was written and the context still runs.
    assert_eq!(read(&y, &stream), vec![0xa5; y_bytes + 8]);
    assert_eq!(read(&big, &stream), vec![0; big_bytes]);
    dense(w_all, x_all, y_all, k, rows, 1).unwrap();
    expert(call(x_all, apart_y, Some(gate), 2, 256, false)).unwrap();
    assert_eq!(read(&y, &stream)[4..y_bytes + 4], vec![0; y_bytes]);
    let written = read(&big, &stream);
    assert!(written[..2 * stack_bytes].iter().all(|&b| b == 0));
    assert_eq!(written[2 * stack_bytes..], vec![0; apart_bytes + 8]);
}
