use super::*;
use crate::kernels::mlx::{KvBlock, SharedAttention};
use sha2::{Digest, Sha256};

/// A single BF16 query row over 32 heads: the production T=1 decode shape.
fn t1_query(k: &mut MlxKernels) -> ExecutionTensor {
    let mut data = Vec::new();
    for head in 0..32u32 {
        for column in 0..128u32 {
            let v = row_value(head % 4, head + column, column);
            data.extend(v.to_le_bytes());
        }
    }
    let host = Tensor::try_from_host(Dtype::F32, vec![1, 32, 1, 128], data).unwrap();
    let raw = k.import_tensor(&host).unwrap();
    k.cast(&raw, Dtype::Bf16).unwrap()
}

/// Raw F32 sink logits per query head, the shape `shared_attention` demands.
fn t1_sinks(k: &mut MlxKernels) -> ExecutionTensor {
    let data: Vec<_> = (0..32u32)
        .flat_map(|head| (row_value(head % 4, head, 0) / 8.0).to_le_bytes())
        .collect();
    let host = Tensor::try_from_host(Dtype::F32, vec![1, 32], data).unwrap();
    k.import_tensor(&host).unwrap()
}

/// The routing contract of the decode-attention slice, driven by the flag the
/// way FIFI_FLINT_MLX_COALESCE_RESIDUAL tests are: run once with
/// FIFI_FLINT_MLX_DECODE_ATTN=1 or the env unset (the flagged default
/// since the 2026-10-27 flip; a rows == 1 BF16 call with raw F32 sinks
/// must route to the VENDOR fused SDPA and NOT the prefill-shaped float
/// flash) and once with =0 (pins the flag-off path on the prefill-shaped
/// flash). Each run is a SEPARATE process invocation: the
/// flag is process-cached (`decode_attn_flagged`'s OnceLock), so in-process
/// toggling cannot be observed by design.
#[test]
#[ignore = "explicit tiny Mac GPU allocation; no model; run twice via the flag"]
fn t1_decode_routing_follows_the_decode_attn_flag() {
    let flagged = crate::kernels::mlx::decode_attn_flagged();
    let _serial = crate::native_test_lock();
    let mut k = MlxKernels::try_new().unwrap();
    k.allow_bf16(true);
    let raw_keys = input(&mut k, 0, 64, false);
    let raw_values = input(&mut k, 0, 64, true);
    let keys = k.reshape(&raw_keys, &[1, 4, 64, 128]).unwrap();
    let values = k.reshape(&raw_values, &[1, 4, 64, 128]).unwrap();
    let q = t1_query(&mut k);
    let sinks = t1_sinks(&mut k);
    crate::kernels::mlx::native::attn_dispatch::reset();
    let out = k
        .shared_attention(SharedAttention {
            q: &q,
            kv: KvBlock::Float {
                keys: &keys,
                values: &values,
            },
            query_start: 63,
            key_origin: 0,
            window: None,
            sinks: Some(&sinks),
            scale: 1.0 / 128.0_f32.sqrt(),
        })
        .unwrap();
    let (prefill_t1, decode) = crate::kernels::mlx::native::attn_dispatch::counts();
    assert_eq!(out.shape(), [1, 32, 1, 128]);
    assert_eq!(
        (prefill_t1, decode),
        if flagged { (0, 1) } else { (1, 0) },
        "FIFI_FLINT_MLX_DECODE_ATTN={} routing at rows == 1",
        flagged
    );
    // v2 note: with =1 the route is the VENDOR fused SDPA, so the logits hash
    // is no longer bitwise-equal to the flag-off flash hash; the oracle test
    // below owns the tolerance contract (old v1 comment: the decode entry launches the
    // same kernel, so the outputs are bit-identical).
    let f32_out = k.cast(&out, Dtype::F32).unwrap();
    let host = k.readback(&f32_out).unwrap();
    let bytes = host.host_bytes().unwrap();
    let hash = Sha256::digest(bytes);
    println!("T1_DECODE_ATTN flagged={} logits_sha256={hash:x}", flagged);
}

/// The routing contract of the T1-flash drift slice (S3, 2026-10-25): the
/// F32 one-row sink-bearing shape - exactly the Flint per-token arm - stays
/// on vendor SDPA by default and moves to the prefill float-flash kernel
/// when FIFI_FLINT_MLX_T1_FLASH is set, so the per-token arm runs the SAME
/// kernel as the B512 prefill arm. Separate processes per flag state (the
/// flag is process-cached by design). The shared `input()` helper casts to
/// BF16; this shape needs F32 planes to match the production F32 arm, so it
/// builds them locally.
#[test]
#[ignore = "explicit tiny Mac GPU allocation; no model; run twice via the flag"]
fn t1_f32_routing_follows_the_t1_flash_flag() {
    let flagged = match std::env::var("FIFI_FLINT_MLX_T1_FLASH") {
        Err(_) => false,
        Ok(value) => crate::tokenizer::env_allows_value(Some(&value)),
    };
    let _serial = crate::native_test_lock();
    let mut k = MlxKernels::try_new().unwrap();
    let f32_planes = |k: &mut MlxKernels, value: bool| {
        let mut data = Vec::new();
        for head in 0..4u32 {
            for position in 0..64u32 {
                for column in 0..128u32 {
                    let v = row_value(head, position, column);
                    data.extend((if value { -v - 0.25 } else { v }).to_le_bytes());
                }
            }
        }
        let host = Tensor::try_from_host(Dtype::F32, vec![4, 64, 128], data).unwrap();
        k.import_tensor(&host).unwrap()
    };
    let key_planes = f32_planes(&mut k, false);
    let value_planes = f32_planes(&mut k, true);
    let keys = k.reshape(&key_planes, &[1, 4, 64, 128]).unwrap();
    let values = k.reshape(&value_planes, &[1, 4, 64, 128]).unwrap();
    // The F32 query row: the production per-token arm is F32, not BF16.
    let mut data = Vec::new();
    for head in 0..32u32 {
        for column in 0..128u32 {
            let v = row_value(head % 4, head + column, column);
            data.extend(v.to_le_bytes());
        }
    }
    let host = Tensor::try_from_host(Dtype::F32, vec![1, 32, 1, 128], data).unwrap();
    let q = k.import_tensor(&host).unwrap();
    let sinks = t1_sinks(&mut k);
    crate::kernels::mlx::native::attn_dispatch::reset();
    let out = k
        .shared_attention(SharedAttention {
            q: &q,
            kv: KvBlock::Float {
                keys: &keys,
                values: &values,
            },
            query_start: 63,
            key_origin: 0,
            window: None,
            sinks: Some(&sinks),
            scale: 1.0 / 128.0_f32.sqrt(),
        })
        .unwrap();
    let (prefill_t1, decode) = crate::kernels::mlx::native::attn_dispatch::counts();
    assert_eq!(out.shape(), [1, 32, 1, 128]);
    assert_eq!(
        (prefill_t1, decode),
        if flagged { (1, 0) } else { (0, 0) },
        "FIFI_FLINT_MLX_T1_FLASH={} routing at the F32 rows == 1 shape",
        flagged
    );
    // Hash harness for the window: the flagged run's logits must match the
    // B512 flash arm's numerics, not the vendor arm's.
    let host_out = k.readback(&out).unwrap();
    let bytes = host_out.host_bytes().unwrap();
    let hash = Sha256::digest(bytes);
    println!("T1_FLASH flagged={flagged} logits_sha256={hash:x}");
}

/// Graph-cache probe (2026-10-25): HOST wall time of one fixed-shape rows=1
/// layer-subgraph stand-in (rms_norm + 4 projections + residual + rms_norm +
/// silu-FFN + residual; 13 adapter ops), run op-by-op (eager) vs through the
/// compiled closure with a fixed cache key. Prints per-call microseconds for
/// both arms plus the compile counters; asserts the two outputs are bitwise
/// identical (compile must not change numerics) and that the compiled arm
/// replayed from cache (body entries == 1). Weights are closure INPUTS, so
/// nothing bakes and no invalidation exists. Mac GPU allocation, no model.
#[cfg(feature = "native-fault-qualification")]
#[test]
#[ignore = "explicit tiny Mac GPU allocation; no model; graph-cache timing probe"]
fn flint_layer_compile_probe_host_time() {
    use crate::kernels;
    use std::time::Instant;
    let _serial = crate::native_test_lock();
    let mut k = MlxKernels::try_new().unwrap();
    let mut f32_input = |shape: Vec<u32>, seed: f32| {
        let len: usize = shape.iter().map(|&d| d as usize).product();
        let data: Vec<_> = (0..len)
            .map(|i| ((i as f32 * seed) % 0.25) - 0.125)
            .flat_map(f32::to_le_bytes)
            .collect();
        let host = Tensor::try_from_host(Dtype::F32, shape, data).unwrap();
        k.import_tensor(&host).unwrap()
    };
    let x = f32_input(vec![1, 2048], 0.031);
    let wq = f32_input(vec![2048, 2048], 0.017);
    let wk = f32_input(vec![2048, 2048], 0.019);
    let wv = f32_input(vec![2048, 2048], 0.023);
    let wo = f32_input(vec![2048, 2048], 0.029);
    let wg = f32_input(vec![2048, 8192], 0.037);
    let wu = f32_input(vec![2048, 8192], 0.041);
    let wd = f32_input(vec![8192, 2048], 0.043);
    let g1 = f32_input(vec![2048], 0.047);
    let g2 = f32_input(vec![2048], 0.053);
    let eps = 1e-6_f32;

    // Eager arm: the exact adapter-op chain the compiled body replays.
    let eager_once = |k: &mut MlxKernels| {
        let n = k.rms_norm(&x, &g1, eps).unwrap();
        let _q = k.mm(&n, &wq).unwrap();
        let _kk = k.mm(&n, &wk).unwrap();
        let v = k.mm(&n, &wv).unwrap();
        let o = k.mm(&v, &wo).unwrap();
        let r1 = k.add(&o, &x).unwrap();
        let n2 = k.rms_norm(&r1, &g2, eps).unwrap();
        let gg = k.mm(&n2, &wg).unwrap();
        let u = k.mm(&n2, &wu).unwrap();
        let sg = k.silu(&gg).unwrap();
        let hu = k.mul(&sg, &u).unwrap();
        let d = k.mm(&hu, &wd).unwrap();
        k.add(&d, &r1).unwrap()
    };
    let compiled_once = |k: &mut MlxKernels| {
        k.compile_probe_flint_layer(&x, &wq, &wk, &wv, &wo, &wg, &wu, &wd, &g1, &g2, eps)
            .unwrap()
    };

    // Warm both arms (first compile = the trace; first eager = allocator warm).
    drop(eager_once(&mut k));
    let warm_compiled = compiled_once(&mut k);

    // Bitwise identity of the two arms before any timing.
    let warm_eager = eager_once(&mut k);
    let read = |k: &mut MlxKernels, t: &ExecutionTensor| {
        let host = k.readback(t).unwrap();
        host.host_bytes().unwrap().to_vec()
    };
    let eager_bytes = read(&mut k, &warm_eager);
    let compiled_bytes = read(&mut k, &warm_compiled);
    assert_eq!(
        Sha256::digest(&eager_bytes),
        Sha256::digest(&compiled_bytes),
        "compiled replay changed the layer-subgraph numerics"
    );

    // The body runs only while TRACING: a steady replay leaves the body-entry
    // count at its warm value. If it grew per iteration, MLX re-traced every
    // call (the Qwen-B1 failure mode) and the timing below would be invalid.
    let bodies_before = kernels::mlx::native::flint_layer_probe_body_entries();
    let n_iters = 1000_u32;
    let start = Instant::now();
    for _ in 0..n_iters {
        drop(compiled_once(&mut k));
    }
    let compiled_us = start.elapsed().as_secs_f64() / f64::from(n_iters) * 1e6;
    let bodies_after = kernels::mlx::native::flint_layer_probe_body_entries();
    assert_eq!(
        bodies_after, bodies_before,
        "the compiled arm re-traced per call (Qwen-B1 failure mode) - timing invalid"
    );

    let start = Instant::now();
    for _ in 0..n_iters {
        drop(eager_once(&mut k));
    }
    let eager_us = start.elapsed().as_secs_f64() / f64::from(n_iters) * 1e6;

    let cut = (eager_us - compiled_us) / eager_us * 100.0;
    println!(
        "FLINT_LAYER_PROBE eager_us={eager_us:.2} compiled_us={compiled_us:.2} cut={cut:.1}% over {n_iters} iters"
    );
}

/// Per-kernel-floor calibration (2026-10-25, quiet follow-up to the graph-cache
/// probe): HOST build + evaluate wall of M back-to-back tiny adds ([1,2048]
/// F32), op-by-op eager vs ONE compiled closure replayed from cache, R
/// repetitions each. Prints us per kernel for both arms; calibration input for
/// the op-count lever estimates. Mac GPU allocation, no model.
#[cfg(feature = "native-fault-qualification")]
#[test]
#[ignore = "explicit tiny Mac GPU allocation; no model; per-kernel floor calibration"]
fn tiny_adds_eager_vs_compiled_floor() {
    use std::time::Instant;
    let _serial = crate::native_test_lock();
    let mut k = MlxKernels::try_new().unwrap();
    let mut bytes = Vec::with_capacity(2048 * 4);
    for _ in 0..2048 {
        bytes.extend_from_slice(&0.25_f32.to_le_bytes());
    }
    let host = Tensor::try_from_host(Dtype::F32, vec![1, 2048], bytes).unwrap();
    let x = k.import_tensor(&host).unwrap();

    let m_override = std::env::var("FIFI_ADD_CHAIN_M")
        .ok()
        .and_then(|v| v.parse::<u32>().ok());
    const R: u32 = 50;
    let m: u32 = m_override.unwrap_or(256);
    let eager_once = |k: &mut MlxKernels| {
        let mut t = k.add(&x, &x).unwrap();
        for _ in 1..m {
            t = k.add(&t, &x).unwrap();
        }
        t
    };

    eprintln!("PHASE eager-warm");
    drop(eager_once(&mut k));
    eprintln!("PHASE eager-timed");
    let start = Instant::now();
    for _ in 0..R {
        let t = eager_once(&mut k);
        k.readback(&t).unwrap();
    }
    let eager_total = start.elapsed().as_secs_f64();

    let compiled_once = |k: &mut MlxKernels| k.compile_probe_add_chain(&x, m).unwrap();
    eprintln!("PHASE compiled-trace");
    drop(compiled_once(&mut k));
    eprintln!("PHASE compiled-timed");
    let start = Instant::now();
    for _ in 0..R {
        let t = compiled_once(&mut k);
        k.readback(&t).unwrap();
    }
    let compiled_total = start.elapsed().as_secs_f64();

    let floor = (eager_total - compiled_total) / (eager_total) * 100.0;
    println!(
        "TINY_ADD_FLOOR M={m} R={R} eager_ms_per_rep={:.2} compiled_ms_per_rep={:.2}          eager_us_per_kernel={:.3} compiled_us_per_kernel={:.3} wall_cut={floor:.1}%",
        eager_total * 1e3 / f64::from(R),
        compiled_total * 1e3 / f64::from(R),
        eager_total / f64::from(R * m) * 1e6,
        compiled_total / f64::from(R * m) * 1e6,
    );
}

/// Ring-view flag polarity and routing (host unit test on the Mac build,
/// T1_FLASH convention; the dispatch proof is the ignored gate in the
/// qualification profile module). The env mapping and BOTH route polarities
/// go through the pure helpers so the process-wide OnceLock is never
/// poisoned and the test holds in any flag state.
#[test]
fn ring_view_flag_polarity_and_routes() {
    use crate::family::flint::mlx_forward::{
        RingViewGeometry, ring_view_env_flag, ring_view_routes, ring_view_routes_with,
    };
    // Default-on polarity (env_allows_default_on): absent is the new
    // default ON; a set value survives only as the explicit enable words;
    // non-UTF-8 and everything else roll back OFF.
    assert!(ring_view_env_flag(Ok("1".into())));
    assert!(ring_view_env_flag(Ok("true".into())));
    assert!(ring_view_env_flag(Ok("on".into())));
    assert!(!ring_view_env_flag(Ok("0".into())));
    assert!(!ring_view_env_flag(Ok("yes".into())));
    assert!(!ring_view_env_flag(Ok(String::new())));
    assert!(ring_view_env_flag(Err(std::env::VarError::NotPresent)));
    assert!(!ring_view_env_flag(Err(std::env::VarError::NotUnicode(
        std::ffi::OsString::new()
    ))));
    let geo = |full, wrapped, one_row, chained| RingViewGeometry {
        full,
        wrapped,
        one_row,
        chained,
    };
    // Both polarities of the full route candidate, through the pure form.
    assert!(ring_view_routes_with(geo(true, false, true, true), true));
    assert!(!ring_view_routes_with(geo(true, false, true, true), false));
    assert!(
        !ring_view_routes_with(geo(false, false, true, true), true),
        "SWA keeps the union"
    );
    assert!(
        !ring_view_routes_with(geo(true, true, true, true), true),
        "wrapped needs the concat"
    );
    assert!(
        !ring_view_routes_with(geo(true, false, false, true), true),
        "multi-row chunk"
    );
    assert!(
        !ring_view_routes_with(geo(true, false, true, false), true),
        "fenced commit"
    );
    // The wrapper is exactly the process flag, in whatever state this
    // process runs (armed or not), so the test holds either way.
    assert_eq!(
        ring_view_routes(geo(true, false, true, true)),
        crate::family::flint::mlx_forward::ring_view_flagged()
    );
}

/// Weights-free whole-plane parity gate (ring view, 2026-10-25 review): at
/// every probe position the whole-plane call - rows at or beyond len
/// POISONED with NaN - must be bitwise equal to the exactly-sized union
/// call, proving the kernel consumes len = p0+1 and never reads retired
/// ring rows. BF16 planes with raw F32 sinks is the production decode
/// shape. Non-periodic generator (row_value); ~34 MB GPU, no model weights:
/// safe to run any time, daily engine up or down.
#[test]
#[ignore = "explicit tiny Mac GPU allocation; no model; any time"]
fn whole_plane_attention_matches_union_bitwise() {
    let _serial = crate::native_test_lock();
    let mut k = MlxKernels::try_new().unwrap();
    k.allow_bf16(true);
    let cap = 8192u32;
    let q = t1_query(&mut k);
    let sinks = t1_sinks(&mut k);
    // The flash kernel guards loads by key < length, so a NaN-poisoned
    // capacity tail is invisible to it. For the vendor arm the poison stays
    // NaN too (MLX 64ea011c reads K/V only inside `if (use_key)`, and the
    // W24 armed run produced finite, near-equal outputs) - BUT the two arms
    // then take the vendor 1-pass and 2-pass kernels at different S, which
    // round the output independently: the armed comparison is therefore
    // ulp-tolerance (observed <= 1 bf16 output ulp at p0=511 in W24), while
    // the unflagged flash arms stay BITWISE (same kernel, poison contract).
    // The capacity-plane guard (s == q_off + rows) puts the ARMED whole-plane
    // arm on the FLASH kernel - the ring-view shape whose garbage tail must
    // not reach the vendor kernel - so armed the comparison is CROSS-ROUTE
    // (vendor 1-pass union vs flash whole): route tolerance, not one ulp.
    // The p0 == cap-1 plane has no garbage tail (len == cap is an exact
    // plane), so both arms stay on the vendor kernel there and stay bitwise.
    let armed = crate::kernels::mlx::decode_attn_flagged();
    let max_diff = if armed { 0.05f32 } else { f32::NAN };
    // (p0, key_origin): origin 0 sweeps the vendor kernel widths and the
    // SWA edge; the nonzero-origin entries are the RING capacity shape -
    // a wrapped plane whose oldest rows retired - where s > q_off + rows
    // must keep the poisoned capacity arm on FLASH (a guard bug like
    // s <= query_start + rows passes the origin-0 fixtures only).
    let cases: [(u32, u32); 11] = [
        (0, 0),
        (1, 0),
        (511, 0),
        (512, 0),
        (513, 0),
        (4095, 0),
        (8190, 0),
        (8191, 0),
        (163, 100),
        (1023, 511),
        (4223, 4096),
    ];
    for (p0, origin) in cases {
        let len = p0 + 1;
        let resident = len - origin;
        let (mut union_k, mut union_v, mut whole_k, mut whole_v) =
            (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for head in 0..4u32 {
            for pos in 0..cap {
                for column in 0..128u32 {
                    let abs = origin + pos;
                    // Union: exactly the resident rows (abs in origin..=p0).
                    if abs < len {
                        let v = row_value(head, abs, column);
                        union_k.extend_from_slice(&v.to_le_bytes());
                        union_v.extend_from_slice(&(-v - 0.25).to_le_bytes());
                    }
                    // Whole plane: every row the plane stores; rows past the
                    // valid history (abs > p0) carry the NaN poison.
                    if abs <= p0 {
                        let v = row_value(head, abs, column);
                        whole_k.extend_from_slice(&v.to_le_bytes());
                        whole_v.extend_from_slice(&(-v - 0.25).to_le_bytes());
                    } else {
                        whole_k.extend_from_slice(&f32::NAN.to_le_bytes());
                        whole_v.extend_from_slice(&f32::NAN.to_le_bytes());
                    }
                }
            }
        }
        let import = |k: &mut MlxKernels, data: Vec<u8>, rows: u32| {
            let host = Tensor::try_from_host(Dtype::F32, vec![4, rows, 128], data).unwrap();
            let raw = k.import_tensor(&host).unwrap();
            k.cast(&raw, Dtype::Bf16).unwrap()
        };
        let uk = import(&mut k, union_k, resident);
        let uv = import(&mut k, union_v, resident);
        let wk = import(&mut k, whole_k, cap);
        let wv = import(&mut k, whole_v, cap);
        let call = |k: &mut MlxKernels, keys: &ExecutionTensor, values: &ExecutionTensor| {
            let keys = k.reshape(keys, &[1, 4, keys.shape()[1], 128]).unwrap();
            let values = k.reshape(values, &[1, 4, values.shape()[1], 128]).unwrap();
            k.shared_attention(SharedAttention {
                q: &q,
                kv: KvBlock::Float {
                    keys: &keys,
                    values: &values,
                },
                query_start: p0,
                key_origin: origin,
                window: None,
                sinks: Some(&sinks),
                scale: 1.0 / 128.0_f32.sqrt(),
            })
            .unwrap()
        };
        crate::kernels::mlx::native::attn_dispatch::reset();
        let out_union = call(&mut k, &uk, &uv);
        let (prefill, decode) = crate::kernels::mlx::native::attn_dispatch::counts();
        // Exactly one kernel launch per call, whichever entry dispatch takes
        // it. Armed, the capacity-plane guard SPLITS the arms by design: the
        // exact-length union runs the vendor fused SDPA (decode entry) while
        // the capacity whole-plane runs flash (prefill entry) - that split is
        // itself the proof the guard is active. Unflagged both arms run the
        // float flash.
        assert_eq!(
            prefill + decode,
            1,
            "union arm must launch exactly one kernel at p0={p0}"
        );
        assert_eq!(
            (prefill, decode),
            if armed { (0, 1) } else { (1, 0) },
            "union arm must take the exact-length route at p0={p0}"
        );
        let out_whole = call(&mut k, &wk, &wv);
        let (p2, d2) = crate::kernels::mlx::native::attn_dispatch::counts();
        assert_eq!(
            (p2 - prefill) + (d2 - decode),
            1,
            "whole-plane arm must launch exactly one kernel at p0={p0}"
        );
        let whole_takes_flash = !armed || len < cap;
        assert_eq!(
            (p2 - prefill, d2 - decode),
            if whole_takes_flash { (1, 0) } else { (0, 1) },
            "whole-plane arm must take the capacity-guarded route at p0={p0}"
        );
        let read = |k: &mut MlxKernels, t: &ExecutionTensor| {
            let f = k.cast(t, Dtype::F32).unwrap();
            k.readback(&f).unwrap().host_bytes().unwrap().to_vec()
        };
        assert_eq!(out_union.shape(), [1, 32, 1, 128]);
        let union_bytes = read(&mut k, &out_union);
        let whole_bytes = read(&mut k, &out_whole);
        // A finite reference: identical NaNs on both arms would also compare
        // equal as bytes, so pin that BOTH arms are real output first.
        for (label, bytes) in [("union", &union_bytes), ("whole", &whole_bytes)] {
            for chunk in bytes.as_chunks::<4>().0 {
                let v = f32::from_le_bytes(*chunk);
                assert!(
                    v.is_finite(),
                    "{label} arm output must be finite at p0={p0}"
                );
            }
        }
        if armed && whole_takes_flash {
            // Cross-route now (vendor 1-pass union at S=len vs flash whole at
            // S=cap): the two kernels round independently, so the comparison
            // carries the route tolerance while still proving neither arm
            // consumed a NaN row beyond len.
            let mut worst = 0.0f32;
            for (a, b) in union_bytes
                .as_chunks::<4>()
                .0
                .iter()
                .zip(whole_bytes.as_chunks::<4>().0.iter())
            {
                let (x, y) = (f32::from_le_bytes(*a), f32::from_le_bytes(*b));
                worst = worst.max((x - y).abs());
            }
            assert!(
                worst <= max_diff,
                "whole-plane vs union beyond route tolerance at p0={p0}: {worst}"
            );
        } else {
            assert_eq!(
                union_bytes, whole_bytes,
                "whole-plane with NaN beyond len must match the union bitwise at p0={p0}"
            );
        }
        // Exact-length oracle: a regression that truncates len identically in
        // both arms would pass every comparison above, so prove the reader
        // consumes row p0 and row 0. AMPLIFY the probed row (K x64, V pinned
        // to -16) instead of zeroing it: an amplified row dominates the
        // softmax, so the output must move at EVERY p0 — a zeroed row can
        // drown in softmax dilution at large p0 (found at p0=511).
        let amplify_row = |k: &mut MlxKernels, plane: &ExecutionTensor, row: u32, value: bool| {
            let mut data: Vec<u8> = Vec::new();
            let f32plane = k.cast(plane, Dtype::F32).unwrap();
            let bytes = k
                .readback(&f32plane)
                .unwrap()
                .host_bytes()
                .unwrap()
                .to_vec();
            for head in 0..4u32 {
                for pos in 0..cap {
                    for column in 0..128u32 {
                        let base = (((head * cap + pos) * 128 + column) * 4) as usize;
                        let raw_v = f32::from_le_bytes([
                            bytes[base],
                            bytes[base + 1],
                            bytes[base + 2],
                            bytes[base + 3],
                        ]);
                        let v = if pos == row {
                            if value { -16.0f32 } else { raw_v * 64.0 }
                        } else {
                            raw_v
                        };
                        data.extend_from_slice(&v.to_le_bytes());
                    }
                }
            }
            let host = Tensor::try_from_host(Dtype::F32, vec![4, cap, 128], data).unwrap();
            let raw = k.import_tensor(&host).unwrap();
            k.cast(&raw, Dtype::Bf16).unwrap()
        };
        for probe_row in [resident - 1, 0] {
            let ak = amplify_row(&mut k, &wk, probe_row, false);
            let av = amplify_row(&mut k, &wv, probe_row, true);
            let out_amplified = call(&mut k, &ak, &av);
            assert_ne!(
                read(&mut k, &out_amplified),
                whole_bytes,
                "amplifying visible row {probe_row} must move the output at p0={p0}"
            );
        }
    }
}

fn import_f32(k: &mut MlxKernels, shape: Vec<u32>, data: &[f32]) -> ExecutionTensor {
    let bytes = data
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect::<Vec<_>>();
    let host = Tensor::try_from_host(Dtype::F32, shape, bytes).unwrap();
    k.import_tensor(&host).unwrap()
}

/// Real checkpoint sink magnitudes (model.layers.*.self_attn.softmax_offset,
/// 2026-10-25 table: range -9.0..+12.375): the model-wide max plus layer-0
/// heads 1..7. All BF16-exact - the checkpoint stores the sinks AS BF16.
const REAL_SINKS: [f32; 8] = [
    12.375,
    7.906_25,
    5.656_25,
    4.906_25,
    4.281_25,
    3.765_625,
    3.062_5,
    1.554_687_5,
];

fn real_sink(head: u32) -> f32 {
    REAL_SINKS[(head % 8) as usize]
}

/// Host F32 oracle for one T=1 case: `rows` keys at positions [0, rows), a
/// query row at absolute position `p0`, optional sliding window, per-head raw
/// sinks. Returns per-case stats plus the oracle's own sink-sensitivity
/// numbers so the test can prove the sink carries real probability mass
/// (the first oracle underflowed: sinks /8 against scores O(100) gave the
/// sink <= 1.8e-61 mass and could not see a dropped sink at all).
#[allow(clippy::too_many_arguments)]
fn t1_oracle_case(
    k: &mut MlxKernels,
    out: &ExecutionTensor,
    rows: u32,
    p0: u32,
    key_origin: u32,
    window: Option<u32>,
    scale: f32,
    sink_for_head: &dyn Fn(u32) -> f32,
) -> (f32, f32, f32, f32) {
    // The row_value generator aligns query and key positions, so raw dots
    // run O(4000); the callers pass scale 1/4096, which lands the scores at
    // O(1) - BELOW the sink band, so the learned offsets carry real mass at
    // every S in the sweep (verified numerically: min strong-head sink mass
    // 0.0017 at S=8192, 0.57 at the SWA edge).
    let raw_k = input(k, 0, rows, false);
    let f = k.cast(&raw_k, Dtype::F32).unwrap();
    let kb = k.readback(&f).unwrap().host_bytes().unwrap().to_vec();
    let raw_v = input(k, 0, rows, true);
    let f = k.cast(&raw_v, Dtype::F32).unwrap();
    let vb = k.readback(&f).unwrap().host_bytes().unwrap().to_vec();
    let f = k.cast(out, Dtype::F32).unwrap();
    let ob = k.readback(&f).unwrap().host_bytes().unwrap().to_vec();
    let at = |b: &[u8], head: u32, pos: u32, col: u32| -> f32 {
        let base = (((head * rows + pos) * 128 + col) * 4) as usize;
        f32::from_le_bytes([b[base], b[base + 1], b[base + 2], b[base + 3]])
    };
    let mut worst_abs = 0.0f32;
    let mut rel2 = 0.0f32;
    let mut min_p_sink_strong = 1.0f32;
    let mut max_drop_effect = 0.0f32;
    for head in 0..32u32 {
        let kv = head / 8;
        let sink = sink_for_head(head);
        let mut scores = Vec::with_capacity(rows as usize);
        for pos in 0..rows {
            let logical = key_origin + pos;
            let keep = logical <= p0 && window.is_none_or(|w| p0 - logical < w);
            if !keep {
                scores.push(f32::NEG_INFINITY);
                continue;
            }
            let mut dot = 0.0f32;
            for col in 0..128u32 {
                dot += row_value(head % 4, head + col, col) * at(&kb, kv, pos, col);
            }
            scores.push(dot * scale);
        }
        let m = scores
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max)
            .max(sink);
        let p_sink = (sink - m).exp();
        let mut l = p_sink;
        let mut acc = [0.0f32; 128];
        let mut l_no_sink = 0.0f32;
        let mut acc_no_sink = [0.0f32; 128];
        for pos in 0..rows {
            let w = (scores[pos as usize] - m).exp();
            l_no_sink += w;
            for col in 0..128u32 {
                let v = at(&vb, kv, pos, col);
                acc[col as usize] += w * v;
                acc_no_sink[col as usize] += w * v;
            }
        }
        l += l_no_sink;
        if sink >= 3.0 {
            min_p_sink_strong = min_p_sink_strong.min(p_sink / l);
        }
        for col in 0..128u32 {
            let want = acc[col as usize] / l;
            let want_no_sink = acc_no_sink[col as usize] / l_no_sink;
            max_drop_effect = max_drop_effect.max((want - want_no_sink).abs());
            let base = ((head * 128 + col) * 4) as usize;
            let got = f32::from_le_bytes([ob[base], ob[base + 1], ob[base + 2], ob[base + 3]]);
            assert!(got.is_finite(), "head {head} col {col} not finite");
            worst_abs = worst_abs.max((got - want).abs());
            rel2 += (got - want) * (got - want);
        }
    }
    // Sum of squared errors (NOT its root): the caller derives the true RMS
    // as (sse / elements).sqrt(). Returning the root made the gate compute
    // SSE^(1/4)/64 - a fourth-root passes far more than RMS allows (found in
    // the final blind review).
    (worst_abs, rel2, min_p_sink_strong, max_drop_effect)
}

/// The T=1 arm against the host F32 oracle under EITHER route: unflagged the
/// flash kernel serves it, flagged the vendor fused SDPA does. The sweep
/// covers the single-pass and two-pass vendor kernel widths (64 / 1024 /
/// 4096 / 8192 keys) and SWA edge shapes (a window keeping only the last 16
/// keys). Sinks use REAL checkpoint magnitudes so the sink carries real
/// probability mass - the fixture self-checks assert it, and a dropped-sink
/// mutant must move the oracle far beyond the route tolerance or the
/// comparison proves nothing about the sink. Armed, the device casts sinks
/// to BF16 (bit-exact on these values), so the armed oracle uses the same
/// rounded sinks.
#[test]
#[ignore = "explicit tiny Mac GPU allocation; no model; run twice via the flag"]
fn t1_attention_matches_the_host_oracle_under_either_route() {
    let flagged = crate::kernels::mlx::decode_attn_flagged();
    let _serial = crate::native_test_lock();
    let mut k = MlxKernels::try_new().unwrap();
    k.allow_bf16(true);
    let q = t1_query(&mut k);
    // 1/4096 lands the generator's O(4000) aligned dots at O(1) scores,
    // below the sink band (see t1_oracle_case).
    let scale = 1.0f32 / 4096.0;
    let cases: [(u32, Option<u32>, u32); 10] = [
        (64, None, 0),
        (1024, None, 0),
        (4096, None, 0),
        (8192, None, 0),
        (64, Some(16), 0),
        (1024, Some(16), 0),
        // Ring geometries (E9): a wrapped plane whose oldest rows retired.
        // key_origin > 0, decode at p0 = origin + rows - 1; the oracle's
        // visibility math must follow the LOGICAL positions. These cover:
        // the window fully inside the resident plane, the same shape at a
        // larger origin (geometry must not depend on where the wrap was),
        // a window older than the plane head (every resident row visible:
        // the s == q_off + rows capacity plane), and a retired tail with no
        // window at all.
        (64, Some(16), 32),
        (64, Some(16), 80),
        (64, Some(80), 32),
        (1024, None, 512),
    ];
    for (rows, window, origin) in cases {
        let p0 = origin + rows - 1;
        let raw_keys = input(&mut k, 0, rows, false);
        let keys = k.reshape(&raw_keys, &[1, 4, rows, 128]).unwrap();
        let raw_values = input(&mut k, 0, rows, true);
        let values = k.reshape(&raw_values, &[1, 4, rows, 128]).unwrap();
        let sink_data: Vec<f32> = (0..32u32).map(real_sink).collect();
        let sinks = import_f32(&mut k, vec![1, 32], &sink_data);
        let out = k
            .shared_attention(SharedAttention {
                q: &q,
                kv: KvBlock::Float {
                    keys: &keys,
                    values: &values,
                },
                query_start: p0,
                key_origin: origin,
                window,
                sinks: Some(&sinks),
                scale,
            })
            .unwrap();
        assert_eq!(out.shape(), [1, 32, 1, 128]);
        let oracle_sink: Box<dyn Fn(u32) -> f32> = if flagged {
            Box::new(move |h: u32| crate::kernels::mlx::bf16_round_trip(real_sink(h)))
        } else {
            Box::new(real_sink)
        };
        let (worst_abs, sse, min_p_sink, drop_effect) =
            t1_oracle_case(&mut k, &out, rows, p0, origin, window, scale, &oracle_sink);
        println!(
            "T1_ORACLE flagged={flagged} rows={rows} window={window:?} origin={origin} \
             worst_abs={worst_abs:.5} rms={:.6} \
             min_p_sink(strong heads)={min_p_sink:.4} drop_effect={drop_effect:.4}",
            (sse / 4096.0).sqrt()
        );
        assert!(
            min_p_sink >= 1e-3,
            "fixture underflow: the sink must carry mass (rows={rows}, window={window:?})"
        );
        assert!(
            drop_effect >= 0.2,
            "fixture desensitized: dropping the sink must move the oracle far \
             beyond the route tolerance (rows={rows}, window={window:?})"
        );
        assert!(
            worst_abs <= 0.05 && (sse / 4096.0).sqrt() <= 0.02,
            "T=1 route (flagged={flagged}) drifted from the oracle at \
             rows={rows} window={window:?}: abs {worst_abs:.5} rms {}",
            (sse / 4096.0).sqrt()
        );
    }
}
