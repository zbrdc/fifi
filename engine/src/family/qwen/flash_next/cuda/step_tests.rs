//! The device layers of `FlashStep` against the host reference (`hyper.rs`, `qsa.rs`,
//! `deltanet.rs`, `moe.rs`, in the order of `step.rs`) on the real Flash-Next 3.05 bpw pack.
//!
//! `FIFI_FLASH_NEXT_EXL3_PACK` names a local pack directory: a copy with the headers of every shard
//! and the payload of the chosen layers is enough (the sparse layer-3 copy holds layer 3), and
//! `FIFI_FLASH_NEXT_STEP_LAYERS` lists the layers of the window (`3` by default). The full local
//! pack serves any layers.
//!
//! The host reference projects with the AVX-512 CPU kernel on the stored matrices (the exact
//! codebook value, f32 accumulation), which is one binary16 rounding away from the device kernels
//! (about 2e-4 relative per projection, `moe/real_attention_tests.rs`); the routed experts are read
//! and projected on demand. The device keeps the routed experts as an all-resident bank of the
//! window's layers. Each layer is run in steps (hyper read, mixer, apply, hyper read, MoE, apply)
//! with a drain and a read-back after the mixer and after the MoE block, so a divergence names its
//! block. The stream, the block outputs and the status words are compared at every token of a short
//! sequence; the QSA cache and the DeltaNet state therefore have to agree across tokens too.
//!
//! The head check (`FIFI_FLASH_NEXT_STEP_HEAD`) runs the final mixer and `lm_head` on the stream the
//! window leaves, not the full host step: neither side applies the PLE update, so a window that
//! crosses the PLE layer is a truncated model on both sides and its logits only compare the head.
//!
//! Needs a GPU, `--test-threads=1` and (for the full window) several GiB; ignored by default.

#[path = "step_tests/qsa_input_projection_probe.rs"]
mod qsa_input_projection_probe;
#[path = "step_tests/shared_projection_probe.rs"]
mod shared_projection_probe;

use crate::family::exl3::launch::Launch;
use std::{collections::HashMap, path::Path, time::Duration};

use super::{
    bank::{HostMemory, HostPinPlan, HostTerms, RecordSource, device::DeviceBank},
    bank_source::PackRecordSource,
    load::{
        TrunkList,
        upload::{layer_weights, root_weights},
    },
    moe_block::SplitMoe,
    state::{QsaCacheStorage, StateSlot},
    step::{AllResidentMoe, FlashStep, MIXER, MoeBlock, SplitMoeAt, bank_key, split_trunk},
    step_plan::{PREFILL_TILE_ROWS, Site, Slot},
};
use crate::family::expert_bank::{BankGeometry, ExpertBankState, HostMode, ReplaceOnMissLru};
use crate::{
    EngError,
    family::{
        exl3::source::PackReader,
        exl3_pack::{ExpertPart, Root, expert_name_in, layer_name},
        project::{
            DenseProjector, DenseWeight, Projector,
            cuda::{CudaProjector, CudaWeight, CudaWeights, Reserve},
        },
        qwen::flash_next::{
            deltanet::{self, DeltaState, DeltaWeights},
            hyper::{self, HyperWeights},
            load::model_shape,
            moe::slot_bank::OwnedPart,
            moe::{self, MoeWeights},
            qsa::{self, QsaCache, QsaWeights},
            step::ModelShape,
        },
        stored::host_f32,
    },
    kernels::{
        cuda::{DeviceStream, device_present},
        exl3_cpu::{Exl3Isa, Exl3Projection, ExpertCpu},
        nvfp4_cpu::WaitPolicy,
    },
    weights::exl3::{COMPONENT_SUFFIXES, Checkpoint, Matrix},
};

#[test]
fn native_decode_test_arena_payload_is_below_one_gib() {
    use super::{
        state::StatePlan,
        state_tests::real,
        step_plan::{StepOptions, StepPlan},
    };
    use crate::family::qwen::flash_next::moe::context_budget::KvFormat;
    let (_, g) = real();
    let state = StatePlan::new(&g, 1, KvFormat::F16).unwrap();
    let base = StepPlan::new(&g, 1, 1, &StepOptions::for_sm_count(82)).unwrap();
    let native = base.clone().with_gguf_decode_q8(256).unwrap();
    let payload = state.arena_bytes() + base.arena_bytes() + native.arena_bytes() + g.hidden * 4;
    eprintln!("native decode test overlapping arena payload {payload} bytes");
    assert!(payload < (1usize << 30));
}

#[test]
#[ignore = "synthetic CUDA microbenchmark; requires an admitted device"]
fn gguf_qsa_group_quantize_microbenchmark() {
    use crate::kernels::cuda::{
        CudaContext,
        resident::quantize_q8_1::{QuantizeCall, q8_1_bytes},
    };
    let ctx = CudaContext::new().unwrap();
    let stream = ctx.stream().unwrap();
    let uploaded =
        crate::family::qwen::flash_next::load::gguf::tests::synthetic_qsa_trunk(&ctx, &stream);
    let weights = CudaWeights::from_native(uploaded).unwrap();
    let config = include_bytes!("../../../../../tests/fixtures/qwen3_8_flash_next_config.json");
    let (model, kinds) = model_shape(config).unwrap();
    let step =
        FlashStep::new_gguf_decode(&ctx, &stream, &weights, model, &kinds, 1, 6144, 3..4).unwrap();
    let x = ctx.allocate(2560 * 4).unwrap();
    x.write_at(
        &stream,
        0,
        &(0..2560)
            .flat_map(|_| 1f32.to_ne_bytes())
            .collect::<Vec<_>>(),
    )
    .unwrap();
    let q8 = ctx.allocate(q8_1_bytes(2560, 1).unwrap()).unwrap();
    let row = ctx.allocate(1024 * 4).unwrap();
    let kname = layer_name(3, "self_attn.k_proj");
    let vname = layer_name(3, "self_attn.v_proj");
    let quantize = || {
        ctx.quantize_q8_1(
            &stream,
            QuantizeCall {
                x: x.range(0, 2560 * 4)?,
                y: q8.range(0, q8.bytes())?,
                k: 2560,
                ncols: 1,
            },
        )
    };
    let group = || {
        step.project_group(
            &[(&kname, 0, 512), (&vname, 512, 512)],
            x.range(0, 2560 * 4)?,
            &row.range(0, 1024 * 4)?,
            1,
        )
    };
    for _ in 0..20 {
        quantize().unwrap();
        group().unwrap();
    }
    stream.synchronize().unwrap();
    let start = ctx.event_timing().unwrap();
    let end = ctx.event_timing().unwrap();
    const N: usize = 400;
    stream.record_timing(&start).unwrap();
    for _ in 0..N {
        quantize().unwrap();
    }
    stream.record_timing(&end).unwrap();
    stream.synchronize().unwrap();
    let quant_us = start.elapsed_ms_since(&end).unwrap() * 1000.0 / N as f32;
    stream.record_timing(&start).unwrap();
    for _ in 0..N {
        group().unwrap();
    }
    stream.record_timing(&end).unwrap();
    stream.synchronize().unwrap();
    let group_us = start.elapsed_ms_since(&end).unwrap() * 1000.0 / N as f32;
    let mut values = vec![0u8; 1024 * 4];
    row.read_at(&stream, 0, &mut values).unwrap();
    for (i, b) in values.as_chunks::<4>().0.iter().enumerate() {
        let expected = if i < 512 { 5120.0 } else { 7680.0 };
        let actual = f32::from_ne_bytes(*b);
        assert!((actual - expected).abs() < 2.0, "{i}: {actual}");
    }
    assert!(quant_us.is_finite() && quant_us > 0.0 && group_us.is_finite() && group_us > 0.0);
    eprintln!(
        "synthetic Q8_0 k=2560 n=512+512 rows=1 iterations={N}: quantize_us={quant_us:.3} grouped_us={group_us:.3} (equal-width group shares quantization); not full-model timing"
    );
}

#[test]
#[ignore = "requires an admitted CUDA device"]
fn gguf_qsa_mixer_tracks_synthetic_token_history() {
    use crate::kernels::cuda::{CudaContext, resident::step_params::StepParams};
    let ctx = CudaContext::new().unwrap();
    let stream = ctx.stream().unwrap();
    let uploaded =
        crate::family::qwen::flash_next::load::gguf::tests::synthetic_qsa_trunk(&ctx, &stream);
    let weights = CudaWeights::from_native(uploaded).unwrap();
    let config = include_bytes!("../../../../../tests/fixtures/qwen3_8_flash_next_config.json");
    let (model, kinds) = model_shape(config).unwrap();
    let step =
        FlashStep::new_gguf_decode(&ctx, &stream, &weights, model, &kinds, 5, 6144, 3..4).unwrap();
    let mut sum = 0.0f32;
    for (pos, scale) in [1.0f32, 0.5, 0.25, 0.125, 0.0625].into_iter().enumerate() {
        step.begin_token().unwrap();
        let (words, at) = step.words_buffer().unwrap();
        words
            .write_at(
                &stream,
                at,
                &StepParams::new(1, pos as i32).unwrap().to_bytes(),
            )
            .unwrap();
        let mixed = step.slot(Slot::Mixed(Site::Attn)).unwrap();
        let input: Vec<u8> = (0..2560).flat_map(|_| scale.to_ne_bytes()).collect();
        mixed
            .buffer()
            .write_at(&stream, mixed.offset(), &input)
            .unwrap();
        step.mixer(3, pos, 1).unwrap();
        let output = step
            .read_f32(step.slot(Slot::BlockOut(Site::Attn)).unwrap(), 2560)
            .unwrap();
        let (status, flag) = step.health().unwrap();
        assert_eq!((status, flag), (0, 0));
        sum += scale;
        // All query/key norm gammas are zero, so all attended values have equal scores.
        let expected = 4.0 * 6144.0 * 7680.0 * sum / (pos + 1) as f32;
        for value in output {
            assert!(
                (value - expected).abs() / expected < 0.01,
                "pos={pos}: {value} != {expected}"
            );
        }
    }
}

#[test]
#[ignore = "requires an admitted CUDA device and CPU host reference"]
fn gguf_qsa_mixer_matches_synthetic_host_history() {
    run_qsa_mixer(QsaCacheStorage::F16);
}

/// The same mixer with f32 QSA caches: every stored key is the f32 value the fp16 step rounds
/// (bit for bit once rounded), and nearly none is a number fp16 holds.
#[test]
#[ignore = "requires an admitted CUDA device and CPU host reference"]
fn gguf_qsa_mixer_with_f32_caches_stores_the_unrounded_keys() {
    let exact = run_qsa_mixer(QsaCacheStorage::F32);
    let rounded = run_qsa_mixer(QsaCacheStorage::F16);
    // The plane is `[positions][n_kv][256]`: four bytes a key against two.
    assert_eq!(exact.len(), 2 * rounded.len());
    let keys = exact
        .as_chunks::<4>()
        .0
        .iter()
        .map(|w| f32::from_le_bytes(*w))
        .collect::<Vec<_>>();
    let halves = rounded
        .as_chunks::<2>()
        .0
        .iter()
        .map(|w| u16::from_le_bytes(*w));
    let mut unrounded = 0;
    for (key, stored) in keys.iter().zip(halves) {
        let half = half::f16::from_f32(*key);
        assert_eq!(
            half.to_bits(),
            stored,
            "the f32 key {key} is not the value the fp16 step stored"
        );
        unrounded += usize::from(half.to_f32() != *key);
    }
    eprintln!(
        "synthetic GGUF QSA: {unrounded} of {} stored keys are not fp16 numbers",
        keys.len()
    );
    assert!(
        unrounded * 10 > keys.len() * 9,
        "only {unrounded} of {} keys are not fp16 numbers",
        keys.len()
    );
}

/// Runs the five synthetic tokens of layer 3 through a step whose QSA caches are `storage` and
/// returns the layer's final key plane, raw: f32 values, or fp16 bits.
fn run_qsa_mixer(storage: QsaCacheStorage) -> Vec<u8> {
    use crate::kernels::cuda::{CudaContext, resident::step_params::StepParams};
    let ctx = CudaContext::new().unwrap();
    let stream = ctx.stream().unwrap();
    let inputs: Vec<Vec<f32>> = [1.0f32, 0.5, 0.25, 0.125, 0.0625]
        .into_iter()
        .map(|scale| {
            (0..2560)
                .map(|i| scale * (1 + i % 7) as f32 / 7.0)
                .collect()
        })
        .collect();
    let (uploaded, reference) =
        crate::family::qwen::flash_next::load::gguf::tests::synthetic_qsa_host_reference(
            &ctx, &stream, &inputs,
        );
    let weights = CudaWeights::from_native(uploaded).unwrap();
    let config = include_bytes!("../../../../../tests/fixtures/qwen3_8_flash_next_config.json");
    let (model, kinds) = model_shape(config).unwrap();
    let step = FlashStep::new_gguf_decode_in(
        &ctx,
        &stream,
        &weights,
        model,
        &kinds,
        5,
        6144,
        storage,
        3..4,
    )
    .unwrap();
    for (pos, (input, want)) in inputs.iter().zip(&reference).enumerate() {
        step.begin_token().unwrap();
        let (words, at) = step.words_buffer().unwrap();
        words
            .write_at(
                &stream,
                at,
                &StepParams::new(1, pos as i32).unwrap().to_bytes(),
            )
            .unwrap();
        let mixed = step.slot(Slot::Mixed(Site::Attn)).unwrap();
        mixed
            .buffer()
            .write_at(
                &stream,
                mixed.offset(),
                &input
                    .iter()
                    .flat_map(|v| v.to_ne_bytes())
                    .collect::<Vec<_>>(),
            )
            .unwrap();
        step.mixer(3, pos, 1).unwrap();
        let got = step
            .read_f32(step.slot(Slot::BlockOut(Site::Attn)).unwrap(), 2560)
            .unwrap();
        assert_eq!(step.health().unwrap(), (0, 0));
        let error = relative_l2(&got, want);
        eprintln!("synthetic GGUF QSA position {pos}: relative L2 {error:.6}");
        assert!(error < 0.03, "position {pos}: relative L2 {error}");
    }
    step.read_state_bytes(StateSlot::QsaK(3)).unwrap()
}

/// The setting the GGUF decode owner's constructor reads: fp16 unless it names `f32`; anything
/// else is refused, so a misspelling never silently keeps fp16.
#[test]
fn the_qsa_cache_setting_names_f16_or_f32_and_refuses_the_rest() {
    use super::step::parse_qsa_cache_storage;
    assert_eq!(parse_qsa_cache_storage(None), Ok(QsaCacheStorage::F16));
    assert_eq!(
        parse_qsa_cache_storage(Some("f16")),
        Ok(QsaCacheStorage::F16)
    );
    assert_eq!(
        parse_qsa_cache_storage(Some("f32")),
        Ok(QsaCacheStorage::F32)
    );
    for other in ["", "F32", "f32 ", "int8", "bf16", "fp16"] {
        assert!(
            parse_qsa_cache_storage(Some(other)).is_err(),
            "{other:?} must be refused"
        );
    }
}

#[test]
#[ignore = "requires the verified pinned GGUF, first-token host capture and an admitted CUDA device"]
fn gguf_shared_expert_matches_pinned_host() {
    use crate::kernels::cuda::CudaContext;
    let ctx = CudaContext::new().unwrap();
    let stream = ctx.stream().unwrap();
    let (uploaded, input, want) =
        crate::family::qwen::flash_next::load::gguf::tests::pinned_shared_host_reference(&ctx);
    let weights = CudaWeights::from_native(uploaded).unwrap();
    let config = include_bytes!("../../../../../tests/fixtures/qwen3_8_flash_next_config.json");
    let (model, kinds) = model_shape(config).unwrap();
    let step =
        FlashStep::new_gguf_decode(&ctx, &stream, &weights, model, &kinds, 1, 6144, 0..1).unwrap();
    assert!(step.gguf_shared_expert(1).is_err());
    let mixed = step.slot(Slot::Mixed(Site::Mlp)).unwrap();
    mixed
        .buffer()
        .write_at(
            &stream,
            mixed.offset(),
            &input
                .iter()
                .flat_map(|v| v.to_ne_bytes())
                .collect::<Vec<_>>(),
        )
        .unwrap();
    step.gguf_shared_expert(0).unwrap();
    let shared = want[0].len();
    let gu = step
        .read_f32(step.slot(Slot::SharedGu).unwrap(), 2 * shared)
        .unwrap();
    let act = step
        .read_f32(step.slot(Slot::SharedAct).unwrap(), shared)
        .unwrap();
    let out = step
        .read_f32(step.slot(Slot::SharedOut).unwrap(), want[3].len())
        .unwrap();
    let errors = [
        relative_l2(&gu[..shared], &want[0]),
        relative_l2(&gu[shared..], &want[1]),
        relative_l2(&act, &want[2]),
        relative_l2(&out, &want[3]),
    ];
    eprintln!(
        "pinned GGUF shared expert relative L2 gate={:.6} up={:.6} act={:.6} down={:.6}",
        errors[0], errors[1], errors[2], errors[3]
    );
    assert!(gu.iter().chain(&act).chain(&out).all(|v| v.is_finite()));
    assert!(
        errors.iter().all(|e| *e < 0.05),
        "shared expert relative L2 {errors:?}"
    );
    assert_eq!(step.health().unwrap(), (0, 0));
}

#[test]
#[ignore = "requires the verified pinned GGUF, first-token host capture and an admitted CUDA device"]
fn gguf_hyper_sites_match_pinned_host() {
    use crate::{family::recipe::Site as GgufSite, kernels::cuda::CudaContext};
    let ctx = CudaContext::new().unwrap();
    let stream = ctx.stream().unwrap();
    let config = include_bytes!("../../../../../tests/fixtures/qwen3_8_flash_next_config.json");
    for (gguf_site, step_site, prefix) in [
        (
            GgufSite::Mixer,
            Site::Attn,
            layer_name(0, "attn_hyper_connection"),
        ),
        (
            GgufSite::Ffn,
            Site::Mlp,
            layer_name(0, "mlp_hyper_connection"),
        ),
        (GgufSite::Head, Site::Final, MIXER.to_owned()),
    ] {
        let (uploaded, input, host) =
            crate::family::qwen::flash_next::load::gguf::tests::pinned_hyper_host_reference(
                &ctx, &stream, gguf_site,
            );
        let weights = CudaWeights::from_weights(uploaded).unwrap();
        let (model, kinds) = model_shape(config).unwrap();
        let step =
            FlashStep::new_gguf_decode(&ctx, &stream, &weights, model, &kinds, 1, 6144, 0..1)
                .unwrap();
        let state = step.slot(Slot::Stream).unwrap();
        state
            .buffer()
            .write_at(
                &stream,
                state.offset(),
                &input
                    .iter()
                    .flat_map(|x| x.to_ne_bytes())
                    .collect::<Vec<_>>(),
            )
            .unwrap();
        step.hyper_read(step_site, &prefix, 1).unwrap();
        let mixed = step
            .read_f32(step.slot(Slot::Mixed(step_site)).unwrap(), host.mixed.len())
            .unwrap();
        let mixed_error = relative_l2(&mixed, &host.mixed);
        assert!(mixed.iter().all(|v| v.is_finite()));
        assert!(
            mixed_error < 0.003,
            "{gguf_site:?} mixed relative L2 {mixed_error}"
        );
        if let Some(want) = host.gate {
            let gate = step
                .read_f32(step.slot(Slot::Gate(step_site)).unwrap(), want.len())
                .unwrap();
            let gate_error = relative_l2(&gate, &want);
            assert!(gate.iter().all(|v| v.is_finite()));
            assert!(
                gate_error < 0.003,
                "{gguf_site:?} gate relative L2 {gate_error}"
            );
            eprintln!(
                "pinned GGUF hyper {gguf_site:?}: mixed relative L2 {mixed_error:.8}, gate relative L2 {gate_error:.8}"
            );
        } else {
            assert_eq!(gguf_site, GgufSite::Head);
            eprintln!(
                "pinned GGUF hyper {gguf_site:?}: mixed relative L2 {mixed_error:.8}, no inject gate"
            );
        }
    }
}

#[test]
#[ignore = "requires the verified pinned GGUF, first-token host capture and an admitted CUDA device"]
fn gguf_hyper_delta_matches_pinned_first_token() {
    use crate::{
        family::recipe::Site as GgufSite,
        kernels::cuda::{CudaContext, resident::step_params::StepParams},
    };
    let ctx = CudaContext::new().unwrap();
    let stream = ctx.stream().unwrap();
    let (mut uploaded, mut host_stream, hyper_read) =
        crate::family::qwen::flash_next::load::gguf::tests::pinned_hyper_host_reference(
            &ctx,
            &stream,
            GgufSite::Mixer,
        );
    let initial = host_stream.clone();
    let (delta_weights, host_block, _, _) =
        crate::family::qwen::flash_next::load::gguf::tests::pinned_delta_host_reference(
            &ctx,
            &stream,
            std::slice::from_ref(&hyper_read.mixed),
        );
    uploaded.extend(delta_weights);
    let weights = CudaWeights::from_native(uploaded).unwrap();
    let config = include_bytes!("../../../../../tests/fixtures/qwen3_8_flash_next_config.json");
    let (model, kinds) = model_shape(config).unwrap();
    hyper::apply(
        &model.hyper,
        &mut host_stream,
        hyper_read.gate.as_ref().unwrap(),
        &host_block[0],
    )
    .unwrap();
    let step =
        FlashStep::new_gguf_decode(&ctx, &stream, &weights, model, &kinds, 1, 6144, 0..1).unwrap();
    step.begin_token().unwrap();
    let (words, at) = step.words_buffer().unwrap();
    words
        .write_at(&stream, at, &StepParams::new(1, 0).unwrap().to_bytes())
        .unwrap();
    let state = step.slot(Slot::Stream).unwrap();
    state
        .buffer()
        .write_at(
            &stream,
            state.offset(),
            &initial
                .iter()
                .flat_map(|x| x.to_ne_bytes())
                .collect::<Vec<_>>(),
        )
        .unwrap();
    step.hyper_read(Site::Attn, &layer_name(0, "attn_hyper_connection"), 1)
        .unwrap();
    let mixed = step
        .read_f32(
            step.slot(Slot::Mixed(Site::Attn)).unwrap(),
            hyper_read.mixed.len(),
        )
        .unwrap();
    step.mixer(0, 0, 1).unwrap();
    let block = step
        .read_f32(
            step.slot(Slot::BlockOut(Site::Attn)).unwrap(),
            host_block[0].len(),
        )
        .unwrap();
    step.hyper_apply(Site::Attn, 1).unwrap();
    let residual = step
        .read_f32(step.slot(Slot::Stream).unwrap(), host_stream.len())
        .unwrap();
    assert_eq!(step.health().unwrap(), (0, 0));
    let (mixed_error, block_error, residual_error) = (
        relative_l2(&mixed, &hyper_read.mixed),
        relative_l2(&block, &host_block[0]),
        relative_l2(&residual, &host_stream),
    );
    eprintln!(
        "pinned token 9707 layer-0 hyper+DeltaNet: mixed relative L2 {mixed_error:.8}, block relative L2 {block_error:.8}, residual relative L2 {residual_error:.8}"
    );
    assert!(mixed_error < 0.003);
    assert!(block_error < 0.03);
    assert!(residual_error < 0.03);
}

#[test]
#[ignore = "requires the verified pinned GGUF, first-token host capture and an admitted CUDA device"]
fn gguf_first_layer_moe_matches_pinned_host() {
    use crate::family::qwen::flash_next::load::gguf::tests::{
        pinned_delta_host_reference, pinned_hyper_host_reference, pinned_router_host_reference,
        pinned_shared_host_reference,
    };
    use crate::{
        family::recipe::Site as GgufSite,
        kernels::cuda::{CudaContext, resident::step_params::StepParams},
    };
    let ctx = CudaContext::new().unwrap();
    let stream = ctx.stream().unwrap();
    let (mut uploaded, initial, attn_host) =
        pinned_hyper_host_reference(&ctx, &stream, GgufSite::Mixer);
    let (delta, _, _, _) =
        pinned_delta_host_reference(&ctx, &stream, std::slice::from_ref(&attn_host.mixed));
    let (ffn_hyper, _, _) = pinned_hyper_host_reference(&ctx, &stream, GgufSite::Ffn);
    let (shared, host_input, host_shared) = pinned_shared_host_reference(&ctx);
    let mut router = pinned_router_host_reference(&ctx, &stream, &host_input);
    uploaded.extend(delta);
    uploaded.extend(ffn_hyper);
    uploaded.extend(shared);
    uploaded.append(&mut router.weights);
    let weights = CudaWeights::from_native(uploaded).unwrap();
    let config = include_bytes!("../../../../../tests/fixtures/qwen3_8_flash_next_config.json");
    let (model, kinds) = model_shape(config).unwrap();
    let step =
        FlashStep::new_gguf_decode(&ctx, &stream, &weights, model, &kinds, 1, 6144, 0..1).unwrap();
    step.begin_token().unwrap();
    let (words, at) = step.words_buffer().unwrap();
    words
        .write_at(&stream, at, &StepParams::new(1, 0).unwrap().to_bytes())
        .unwrap();
    let state = step.slot(Slot::Stream).unwrap();
    state
        .buffer()
        .write_at(
            &stream,
            state.offset(),
            &initial
                .iter()
                .flat_map(|x| x.to_ne_bytes())
                .collect::<Vec<_>>(),
        )
        .unwrap();
    step.hyper_read(Site::Attn, &layer_name(0, "attn_hyper_connection"), 1)
        .unwrap();
    step.mixer(0, 0, 1).unwrap();
    step.hyper_apply(Site::Attn, 1).unwrap();
    step.hyper_read(Site::Mlp, &layer_name(0, "mlp_hyper_connection"), 1)
        .unwrap();
    let mixed = step
        .read_f32(step.slot(Slot::Mixed(Site::Mlp)).unwrap(), host_input.len())
        .unwrap();
    let input_error = relative_l2(&mixed, &host_input);
    let gpu_gate = check_composed_router(&step, 0, &router, true);
    step.gguf_shared_expert(0).unwrap();
    let out = step
        .read_f32(step.slot(Slot::SharedOut).unwrap(), host_shared[3].len())
        .unwrap();
    let output_error = relative_l2(&out, &host_shared[3]);
    let gated: Vec<_> = out.iter().map(|x| x * gpu_gate).collect();
    let host_gated: Vec<_> = host_shared[3]
        .iter()
        .map(|x| x * router.shared_gate)
        .collect();
    let gated_error = relative_l2(&gated, &host_gated);
    eprintln!(
        "pinned token 9707 layer-0 composed FFN input relative L2 {input_error:.6}, shared output {output_error:.6}, gated output {gated_error:.6}"
    );
    assert!(input_error < 0.03 && output_error < 0.05 && gated_error < 0.05);
    // This test drives disk refills from the GPU classifier's log; it is not the production executor.
    use crate::family::qwen::flash_next::load::gguf::routed_probe;
    let capture = std::env::var_os("FIFI_FLASH_GGUF_HOST_CAPTURE").expect("set host capture");
    for initial_misses in [0, 2, 10] {
        let (routed_error, total_error) = routed_probe::run(
            &ctx,
            &stream,
            routed_probe::Inputs {
                layer: 0,
                precision: step.gguf_decode_precision(),
                host_capture: std::path::Path::new(&capture),
                x: step.slot(Slot::Mixed(Site::Mlp)).unwrap(),
                route_ids: step.slot(Slot::TopkId).unwrap(),
                route_weights: step.slot(Slot::TopkWeight).unwrap(),
                initial_misses,
                diagnose_route_mismatch: false,
                shared_gate: step.slot(Slot::SharedGate).unwrap(),
                shared_out: step.slot(Slot::SharedOut).unwrap(),
                block_out: step.slot(Slot::BlockOut(Site::Mlp)).unwrap(),
            },
            &router,
        );
        assert!(routed_error < 0.05 && total_error < 0.05);
    }
    step.hyper_apply(Site::Mlp, 1).unwrap();
    let path = std::env::var_os("FIFI_FLASH_GGUF_HOST_LAYERS")
        .expect("set the verified host layer capture");
    let host_layers: Vec<Vec<Vec<f32>>> =
        serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
    let host_layer = &host_layers[0][0];
    let layer = step
        .read_f32(step.slot(Slot::Stream).unwrap(), host_layer.len())
        .unwrap();
    let layer_error = relative_l2(&layer, host_layer);
    eprintln!("pinned token 9707 full layer-0 residual relative L2 {layer_error:.6}");
    assert!(layer_error < 0.03);
    assert_eq!(step.health().unwrap(), (0, 0));
}

mod gguf_full_probe;
#[path = "step_tests/gguf_head_probe.rs"]
mod gguf_head_probe;
#[path = "step_tests/gguf_prefix_probe.rs"]
mod gguf_prefix_probe;

fn assert_pinned_file(path: &std::ffi::OsStr, expected: &str) {
    use sha2::{Digest, Sha256};
    let got = format!("{:x}", Sha256::digest(std::fs::read(path).unwrap()));
    assert_eq!(got, expected, "unexpected pinned artifact {:?}", path);
}

/// The actual layer-0 device residual crosses the existing host PLE boundary, then returns
/// to the device for layer 1. This is a staged test, not a device-resident PLE or executor.
#[test]
#[ignore = "requires pinned checkpoint, layer-0/1 host captures and an admitted CUDA device"]
fn gguf_two_layers_with_ple_match_pinned_host() {
    use crate::family::qwen::flash_next::{
        cuda::driver::TokenFeeds,
        load::gguf::{
            GgufModel, routed_probe,
            tests::{
                pinned_delta_host_reference, pinned_hyper_host_reference,
                pinned_router_host_reference, pinned_router_host_reference_at,
                pinned_shared_host_reference, pinned_shared_host_reference_at,
            },
            token_feed::GgufTokenFeeds,
        },
    };
    use crate::{
        family::recipe::Site as GgufSite,
        kernels::cuda::{CudaContext, resident::step_params::StepParams},
    };
    let path = std::env::var_os("FIFI_FLASH_GGUF").expect("set verified checkpoint");
    let capture0 = std::env::var_os("FIFI_FLASH_GGUF_HOST_CAPTURE").expect("set layer-0 capture");
    let capture1 =
        std::env::var_os("FIFI_FLASH_GGUF_HOST_CAPTURE_LAYER1").expect("set layer-1 capture");
    let layers_path =
        std::env::var_os("FIFI_FLASH_GGUF_HOST_LAYERS").expect("set verified host layers");
    // These receipts identify the pinned token-9707 oracle and the small first shard; the
    // other full checkpoint shards must still be independently verified before this test.
    for (path, digest) in [
        (
            &path,
            "9d5644b371ffac921427c8021cc7242dab8e3ba37d7b48d1e79fe3770fbf9915",
        ),
        (
            &capture0,
            "b0873cedfb9733049335904d64635d20b155e68614de0166a287333f5b5640e6",
        ),
        (
            &capture1,
            "32f7f3ac5de6366f4f5b14b825963eb5ca1332790d1727d8bdd5c13b63264e35",
        ),
        (
            &layers_path,
            "62fdd25ecb4f3e14f2f4417de026421e6e6557c972eadee3e779e9cc6656bace",
        ),
    ] {
        assert_pinned_file(path, digest);
    }
    let model = GgufModel::open(std::path::Path::new(&path))
        .unwrap()
        .unwrap();
    let ngram = model.ngram().unwrap();
    let mut feed = GgufTokenFeeds::new(&model, &ngram, 8).unwrap();
    let start = feed.begin(9707).unwrap();
    let ctx = CudaContext::new().unwrap();
    let stream = ctx.stream().unwrap();
    let (mut uploaded, initial, host_attn) =
        pinned_hyper_host_reference(&ctx, &stream, GgufSite::Mixer);
    assert_eq!(start.stream, initial);
    uploaded.extend(pinned_delta_host_reference(&ctx, &stream, &[host_attn.mixed]).0);
    uploaded.extend(pinned_hyper_host_reference(&ctx, &stream, GgufSite::Ffn).0);
    let (shared0, input0, shared_host0) = pinned_shared_host_reference(&ctx);
    let mut router0 = pinned_router_host_reference(&ctx, &stream, &input0);
    uploaded.extend(shared0);
    uploaded.append(&mut router0.weights);
    uploaded.extend(
        model
            .upload_hyper(GgufSite::Mixer, Some(1), &ctx, &stream)
            .unwrap(),
    );
    uploaded.extend(model.upload_delta_trunk(1, &ctx, &stream).unwrap());
    uploaded.extend(
        model
            .upload_hyper(GgufSite::Ffn, Some(1), &ctx, &stream)
            .unwrap(),
    );
    let (shared1, input1, shared_host1) =
        pinned_shared_host_reference_at(&ctx, 1, std::path::Path::new(&capture1));
    let mut router1 = pinned_router_host_reference_at(&ctx, &stream, &input1, 1);
    uploaded.extend(shared1);
    uploaded.append(&mut router1.weights);
    let weights = CudaWeights::from_native(uploaded).unwrap();
    let shape = model.config().model_shape().unwrap();
    assert_eq!(shape.ple_layer, 1);
    assert!(model.config().kinds()[..2].iter().all(|&qsa| !qsa));
    let max_k = model.map().max_quantized_input_width().unwrap();
    eprintln!("pinned GGUF metadata-derived maximum native projection width={max_k}");
    let step = FlashStep::new_gguf_decode(
        &ctx,
        &stream,
        &weights,
        shape,
        model.config().kinds(),
        1,
        max_k,
        0..2,
    )
    .unwrap();
    step.begin_token().unwrap();
    let (words, at) = step.words_buffer().unwrap();
    words
        .write_at(&stream, at, &StepParams::new(1, 0).unwrap().to_bytes())
        .unwrap();
    let state = step.slot(Slot::Stream).unwrap();
    state
        .buffer()
        .write_at(
            &stream,
            state.offset(),
            &initial
                .iter()
                .flat_map(|x| x.to_ne_bytes())
                .collect::<Vec<_>>(),
        )
        .unwrap();
    let host_layers: Vec<Vec<Vec<f32>>> =
        serde_json::from_slice(&std::fs::read(layers_path).unwrap()).unwrap();
    for (layer, host_input, host_shared, router, capture) in [
        (0, &input0, &shared_host0, &router0, &capture0),
        (1, &input1, &shared_host1, &router1, &capture1),
    ] {
        if layer == shape.ple_layer {
            let mut residual = step.read_f32(state, initial.len()).unwrap();
            let update = feed.ple_update(9707, &residual).unwrap();
            assert_eq!(update.len(), residual.len());
            for (x, delta) in residual.iter_mut().zip(update) {
                *x += delta;
            }
            state
                .buffer()
                .write_at(
                    &stream,
                    state.offset(),
                    &residual
                        .iter()
                        .flat_map(|x| x.to_ne_bytes())
                        .collect::<Vec<_>>(),
                )
                .unwrap();
        }
        step.hyper_read(Site::Attn, &layer_name(layer, "attn_hyper_connection"), 1)
            .unwrap();
        step.mixer(layer, 0, 1).unwrap();
        step.hyper_apply(Site::Attn, 1).unwrap();
        step.hyper_read(Site::Mlp, &layer_name(layer, "mlp_hyper_connection"), 1)
            .unwrap();
        let mixed = step
            .read_f32(step.slot(Slot::Mixed(Site::Mlp)).unwrap(), host_input.len())
            .unwrap();
        let input_error = relative_l2(&mixed, host_input);
        check_composed_router(&step, layer, router, layer == 0);
        step.gguf_shared_expert(layer).unwrap();
        let shared = step
            .read_f32(step.slot(Slot::SharedOut).unwrap(), host_shared[3].len())
            .unwrap();
        let shared_error = relative_l2(&shared, &host_shared[3]);
        let misses: &[usize] = if layer == 0 { &[0] } else { &[0, 2, 10] };
        let mut errors = (0.0, 0.0);
        for &initial_misses in misses {
            errors = routed_probe::run(
                &ctx,
                &stream,
                routed_probe::Inputs {
                    layer,
                    precision: step.gguf_decode_precision(),
                    host_capture: std::path::Path::new(capture),
                    x: step.slot(Slot::Mixed(Site::Mlp)).unwrap(),
                    route_ids: step.slot(Slot::TopkId).unwrap(),
                    route_weights: step.slot(Slot::TopkWeight).unwrap(),
                    initial_misses,
                    diagnose_route_mismatch: false,
                    shared_gate: step.slot(Slot::SharedGate).unwrap(),
                    shared_out: step.slot(Slot::SharedOut).unwrap(),
                    block_out: step.slot(Slot::BlockOut(Site::Mlp)).unwrap(),
                },
                router,
            );
            assert!(errors.0 < 0.08 && errors.1 < 0.08);
        }
        let (routed_error, total_error) = errors;
        step.hyper_apply(Site::Mlp, 1).unwrap();
        let residual = step.read_f32(state, host_layers[0][layer].len()).unwrap();
        let residual_error = relative_l2(&residual, &host_layers[0][layer]);
        eprintln!(
            "pinned token 9707 two-layer probe: layer={layer} ffn_input={input_error:.6} shared={shared_error:.6} routed={routed_error:.6} combined={total_error:.6} residual={residual_error:.6}"
        );
        assert!(
            input_error < 0.05
                && shared_error < 0.08
                && routed_error < 0.08
                && total_error < 0.08
                && residual_error < 0.05
        );
    }
    assert_eq!(step.health().unwrap(), (0, 0));
}

fn check_composed_router(
    step: &FlashStep<'_>,
    layer: usize,
    want: &crate::family::qwen::flash_next::load::gguf::tests::PinnedRouterReference,
    exact_order: bool,
) -> f32 {
    let (gate, matched) = measure_composed_router(step, layer, want, exact_order);
    assert!(matched, "GPU and host expert routes differ");
    gate
}

fn measure_composed_router(
    step: &FlashStep<'_>,
    layer: usize,
    want: &crate::family::qwen::flash_next::load::gguf::tests::PinnedRouterReference,
    exact_order: bool,
) -> (f32, bool) {
    use crate::kernels::cuda::resident::router::{
        RouterCall, RouterOut, RouterShape, RouterWeights,
    };
    let g = &step.geometry;
    step.ctx
        .router(
            &step.stream,
            RouterCall {
                shape: RouterShape {
                    m: 1,
                    hidden: g.hidden,
                    n_expert: g.experts,
                    top_k: g.top_k,
                },
                x: step.slot(Slot::Mixed(Site::Mlp)).unwrap(),
                weights: RouterWeights {
                    w: step.float(&layer_name(layer, "mlp.gate.weight")).unwrap(),
                    w_shared: Some(
                        step.float(&layer_name(layer, "mlp.shared_expert_gate.weight"))
                            .unwrap(),
                    ),
                },
                logits: step.slot(Slot::RouterLogits).unwrap(),
                out: RouterOut {
                    ids: step.slot(Slot::TopkId).unwrap(),
                    weights: step.slot(Slot::TopkWeight).unwrap(),
                    shared_gate: Some(step.slot(Slot::SharedGate).unwrap()),
                    status: step.status_word().unwrap(),
                },
            },
        )
        .unwrap();
    let logits = step
        .read_f32(step.slot(Slot::RouterLogits).unwrap(), g.experts)
        .unwrap();
    let route_weights = step
        .read_f32(step.slot(Slot::TopkWeight).unwrap(), g.top_k)
        .unwrap();
    let gate = step
        .read_f32(step.slot(Slot::SharedGate).unwrap(), 1)
        .unwrap()[0];
    let ids = step.slot(Slot::TopkId).unwrap();
    let mut bytes = vec![0; g.top_k * 4];
    ids.buffer()
        .read_at(&step.stream, ids.offset(), &mut bytes)
        .unwrap();
    let ids: Vec<i32> = bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|x| i32::from_ne_bytes(*x))
        .collect();
    let expected_ids: Vec<i32> = want.route.iter().map(|(id, _)| *id as i32).collect();
    let logit_error = relative_l2(&logits, &want.logits);
    eprintln!(
        "composed router layer={layer}: logits relative L2 {logit_error:.6}, ids GPU={ids:?} host={expected_ids:?}, gate GPU={gate:.6} host={:.6}",
        want.shared_gate
    );
    assert!(ids.iter().all(|id| *id >= 0 && (*id as usize) < g.experts));
    let mut actual = ids.clone();
    let mut expected = expected_ids.clone();
    actual.sort_unstable();
    expected.sort_unstable();
    let sets_match = actual == expected;
    if !sets_match {
        for id in actual
            .iter()
            .chain(&expected)
            .filter(|id| !actual.contains(id) || !expected.contains(id))
        {
            let index = *id as usize;
            eprintln!(
                "route boundary expert={id}: gpu_logit={:.9} host_logit={:.9} delta={:.9}",
                logits[index],
                want.logits[index],
                logits[index] - want.logits[index]
            );
        }
    }
    let common_weights_ok = route_weights.iter().zip(&ids).all(|(got, id)| {
        want.route
            .iter()
            .find(|(expert, _)| *expert as i32 == *id)
            .is_none_or(|(_, expected)| (got - expected).abs() < 0.02)
    });
    let numerical_ok =
        logit_error < 0.03 && common_weights_ok && (gate - want.shared_gate).abs() < 0.02;
    if sets_match {
        assert!(numerical_ok, "router numerical tolerance failed");
    } else {
        eprintln!("route diagnostic numerical tolerances satisfied={numerical_ok}");
    }
    (gate, sets_match && (!exact_order || ids == expected_ids))
}

#[test]
#[ignore = "requires an admitted CUDA device and CPU host reference"]
fn gguf_delta_mixer_matches_synthetic_host_history() {
    check_delta_mixer(crate::weights::GgufType::Q80, false);
}

#[test]
#[ignore = "requires an admitted CUDA device and CPU host reference"]
fn gguf_delta_q6_mixer_matches_synthetic_host_history() {
    check_delta_mixer(crate::weights::GgufType::Q6K, false);
}

#[test]
#[ignore = "requires the verified pinned GGUF, layer-0 host capture and an admitted CUDA device"]
fn gguf_delta_mixer_matches_pinned_host_history() {
    check_delta_mixer(crate::weights::GgufType::Q6K, true);
}

fn check_delta_mixer(out_ty: crate::weights::GgufType, pinned: bool) {
    use crate::kernels::cuda::{CudaContext, resident::step_params::StepParams};
    let ctx = CudaContext::new().unwrap();
    let stream = ctx.stream().unwrap();
    let inputs: Vec<Vec<f32>> = if pinned {
        let path = std::env::var_os("FIFI_FLASH_GGUF_HOST_CAPTURE").expect("set the host capture");
        let capture = std::fs::read_to_string(path).unwrap();
        let input: Vec<f32> = capture
            .lines()
            .map(|line| serde_json::from_str::<serde_json::Value>(line).unwrap())
            .find(|row| row["name"] == "blk.0.attn_qkv.weight")
            .map(|row| serde_json::from_value(row["input"].clone()).unwrap())
            .expect("layer-0 DeltaNet input in host capture");
        assert_eq!(input.len(), 2560);
        (0..3)
            .map(|token| {
                input
                    .iter()
                    .enumerate()
                    .map(|(i, x)| {
                        x * (1.0 + token as f32 * 0.01) + token as f32 * (i % 7) as f32 * 0.0001
                    })
                    .collect()
            })
            .collect()
    } else {
        (0..3)
            .map(|token| {
                (0..2560)
                    .map(|i| 0.03 + ((i * (token + 1) + token * 7) % 19) as f32 * 0.002)
                    .collect()
            })
            .collect()
    };
    let (uploaded, reference, host_rec, host_ring) = if pinned {
        crate::family::qwen::flash_next::load::gguf::tests::pinned_delta_host_reference(
            &ctx, &stream, &inputs,
        )
    } else {
        crate::family::qwen::flash_next::load::gguf::tests::synthetic_delta_host_reference(
            &ctx, &stream, &inputs, out_ty,
        )
    };
    assert!(reference.iter().flatten().all(|v| v.is_finite()));
    assert!(reference.iter().flatten().any(|v| v.abs() > 1e-6));
    assert!(host_rec.iter().any(|v| v.abs() > 1e-6));
    let weights = CudaWeights::from_native(uploaded).unwrap();
    let config = include_bytes!("../../../../../tests/fixtures/qwen3_8_flash_next_config.json");
    let (model, kinds) = model_shape(config).unwrap();
    let step = FlashStep::new_gguf_decode(
        &ctx,
        &stream,
        &weights,
        model,
        &kinds,
        inputs.len(),
        6144,
        0..1,
    )
    .unwrap();
    for (pos, (input, want)) in inputs.iter().zip(&reference).enumerate() {
        step.begin_token().unwrap();
        let (words, at) = step.words_buffer().unwrap();
        words
            .write_at(
                &stream,
                at,
                &StepParams::new(1, pos as i32).unwrap().to_bytes(),
            )
            .unwrap();
        let mixed = step.slot(Slot::Mixed(Site::Attn)).unwrap();
        mixed
            .buffer()
            .write_at(
                &stream,
                mixed.offset(),
                &input
                    .iter()
                    .flat_map(|v| v.to_ne_bytes())
                    .collect::<Vec<_>>(),
            )
            .unwrap();
        step.mixer(0, pos, 1).unwrap();
        let got = step
            .read_f32(step.slot(Slot::BlockOut(Site::Attn)).unwrap(), want.len())
            .unwrap();
        assert_eq!(step.health().unwrap(), (0, 0));
        let error = relative_l2(&got, want);
        eprintln!(
            "GGUF DeltaNet pinned={pinned} {out_ty:?} position {pos}: output relative L2 {error:.6}"
        );
        assert!(error < 0.03, "position {pos}: relative L2 {error}");
        if out_ty == crate::weights::GgufType::Q6K && pos + 1 == inputs.len() {
            // Negative control: the raw Q6_K matrix expects tiled, not grouped, heads.
            let wrong_out = ctx.allocate(want.len() * 4).unwrap();
            step.project(
                &layer_name(0, "linear_attn.out_proj"),
                step.slot(Slot::DeltaCore).unwrap(),
                wrong_out.range(0, want.len() * 4).unwrap(),
                1,
            )
            .unwrap();
            let wrong = step
                .read_f32(wrong_out.range(0, want.len() * 4).unwrap(), want.len())
                .unwrap();
            let wrong_error = relative_l2(&wrong, want);
            eprintln!(
                "GGUF DeltaNet pinned={pinned} Q6_K identity-layout negative-control relative L2 {wrong_error:.6}"
            );
            assert!(
                wrong_error > error * 2.0 + 0.001,
                "fixture did not distinguish missing head tile: {wrong_error} vs {error}"
            );
        }
    }
    let rec = step
        .read_f32(
            step.state_slot(StateSlot::DeltaRec(0)).unwrap(),
            host_rec.len(),
        )
        .unwrap();
    let ring = step
        .read_f32(
            step.state_slot(StateSlot::DeltaConv(0)).unwrap(),
            host_ring.len(),
        )
        .unwrap();
    let channels = step.geometry.delta.conv_dim;
    let taps = host_ring.len() / channels;
    let mut planar = vec![0.0; host_ring.len()];
    for channel in 0..channels {
        for tap in 0..taps {
            planar[tap * channels + channel] = host_ring[channel * taps + tap];
        }
    }
    let (rec_error, ring_error) = (relative_l2(&rec, &host_rec), relative_l2(&ring, &planar));
    eprintln!(
        "GGUF DeltaNet pinned={pinned} final state relative L2 rec={rec_error:.6} ring={ring_error:.6}"
    );
    assert!(rec_error < 0.03, "recurrent state relative L2 {rec_error}");
    assert!(
        ring_error < 0.03,
        "convolution ring relative L2 {ring_error}"
    );
}

#[test]
#[ignore = "requires an admitted CUDA device"]
fn gguf_qsa_trunk_upload_projects_synthetic_weights() {
    use crate::kernels::cuda::CudaContext;
    let ctx = CudaContext::new().unwrap();
    let stream = ctx.stream().unwrap();
    let uploaded =
        crate::family::qwen::flash_next::load::gguf::tests::synthetic_qsa_trunk(&ctx, &stream);
    assert_eq!(uploaded.len(), 9);
    let weights = CudaWeights::from_native(uploaded).unwrap();
    let config = include_bytes!("../../../../../tests/fixtures/qwen3_8_flash_next_config.json");
    let (model, kinds) = model_shape(config).unwrap();
    let step =
        FlashStep::new_gguf_decode(&ctx, &stream, &weights, model, &kinds, 1, 6144, 0..1).unwrap();
    let x = ctx.allocate(2560 * 4).unwrap();
    x.write_at(
        &stream,
        0,
        &(0..2560)
            .flat_map(|_| 1f32.to_ne_bytes())
            .collect::<Vec<_>>(),
    )
    .unwrap();
    let name = |suffix| layer_name(3, suffix);
    let qname = name("self_attn.q_proj");
    let kname = name("self_attn.k_proj");
    let vname = name("self_attn.v_proj");
    let width = |name: &str| match weights.get(name).unwrap() {
        CudaWeight::Native { weight, .. } => weight.n_rows(),
        _ => panic!("native weight expected"),
    };
    let (q, k, v) = (width(&qname), width(&kname), width(&vname));
    assert_eq!(k, v);
    let out = ctx.allocate((q.max(k + v).max(640)) * 4).unwrap();
    for (name, cols, indexer) in [
        (&qname, q, false),
        (&name("self_attn.indexer.index_qk_proj"), 640, true),
    ] {
        step.project(
            name,
            x.range(0, 2560 * 4).unwrap(),
            out.range(0, cols * 4).unwrap(),
            1,
        )
        .unwrap();
        let mut values = vec![1u8; cols * 4];
        out.read_at(&stream, 0, &mut values).unwrap();
        for (i, bytes) in values.as_chunks::<4>().0.iter().enumerate() {
            let actual = f32::from_ne_bytes(*bytes);
            let expected = if !indexer || i < 512 { 2560.0 } else { 5120.0 };
            assert!((actual - expected).abs() < 1.0, "row {i}: {actual}");
        }
    }
    step.project_group(
        &[(&kname, 0, k), (&vname, k, v)],
        x.range(0, 2560 * 4).unwrap(),
        &out.range(0, (k + v) * 4).unwrap(),
        1,
    )
    .unwrap();
    let mut values = vec![1u8; (k + v) * 4];
    out.read_at(&stream, 0, &mut values).unwrap();
    for (i, bytes) in values.as_chunks::<4>().0.iter().enumerate() {
        let actual = f32::from_ne_bytes(*bytes);
        let expected = if i < k { 5120.0 } else { 7680.0 };
        assert!((actual - expected).abs() < 2.0, "row {i}: {actual}");
    }
    let oname = name("self_attn.o_proj");
    let CudaWeight::Native { weight: o, .. } = weights.get(&oname).unwrap() else {
        panic!("native output projection expected")
    };
    let ox = ctx.allocate(o.k() * 4).unwrap();
    ox.write_at(
        &stream,
        0,
        &(0..o.k())
            .flat_map(|_| 1f32.to_ne_bytes())
            .collect::<Vec<_>>(),
    )
    .unwrap();
    step.project(
        &oname,
        ox.range(0, o.k() * 4).unwrap(),
        out.range(0, o.n_rows() * 4).unwrap(),
        1,
    )
    .unwrap();
    let mut values = vec![0u8; o.n_rows() * 4];
    out.read_at(&stream, 0, &mut values).unwrap();
    for bytes in values.as_chunks::<4>().0 {
        let actual = f32::from_ne_bytes(*bytes);
        assert!((actual - (4 * o.k()) as f32).abs() < 4.0, "{actual}");
    }
}

#[test]
#[ignore = "requires an admitted CUDA device"]
fn gguf_head_loader_projects_logits() {
    use crate::kernels::cuda::CudaContext;
    let ctx = CudaContext::new().unwrap();
    let stream = ctx.stream().unwrap();
    let head = || crate::family::qwen::flash_next::load::gguf::tests::synthetic_native_head(&ctx);
    assert!(CudaWeights::from_native(vec![]).is_err());
    assert!(CudaWeights::from_native(vec![head(), head()]).is_err());
    let raw = || CudaWeight::Raw {
        name: "aux".into(),
        ty: crate::weights::GgufType::F32,
        buffer: ctx.allocate(4).unwrap(),
    };
    assert!(CudaWeights::from_native(vec![raw()]).is_err());
    let indexer =
        crate::family::qwen::flash_next::load::gguf::tests::synthetic_indexer(&ctx, &stream);
    let weights = CudaWeights::from_native(vec![head(), raw(), indexer]).unwrap();
    let config = include_bytes!("../../../../../tests/fixtures/qwen3_8_flash_next_config.json");
    let (model, kinds) = model_shape(config).unwrap();
    let step =
        FlashStep::new_gguf_decode(&ctx, &stream, &weights, model, &kinds, 1, 256, 0..1).unwrap();
    let input = ctx.allocate(1024).unwrap();
    let x: Vec<u8> = (0..256).flat_map(|_| 1f32.to_ne_bytes()).collect();
    input.write_at(&stream, 0, &x).unwrap();
    let output = ctx.allocate(16).unwrap();
    step.project(
        "lm_head",
        input.range(0, 1024).unwrap(),
        output.range(0, 16).unwrap(),
        1,
    )
    .unwrap();
    let mut got = [0u8; 16];
    output.read_at(&stream, 0, &mut got).unwrap();
    for bytes in got.as_chunks::<4>().0 {
        let value = f32::from_ne_bytes(*bytes);
        assert!((value - 256.0).abs() < 0.5, "{value}");
    }
    let index_in = ctx.allocate(2560 * 4).unwrap();
    index_in
        .write_at(
            &stream,
            0,
            &(0..2560)
                .flat_map(|_| 1f32.to_ne_bytes())
                .collect::<Vec<_>>(),
        )
        .unwrap();
    let index_out = ctx.allocate(640 * 4).unwrap();
    step.project(
        &layer_name(3, "self_attn.indexer.index_qk_proj"),
        index_in.range(0, 2560 * 4).unwrap(),
        index_out.range(0, 640 * 4).unwrap(),
        1,
    )
    .unwrap();
    let mut result = vec![0u8; 640 * 4];
    index_out.read_at(&stream, 0, &mut result).unwrap();
    for (i, bytes) in result.as_chunks::<4>().0.iter().enumerate() {
        let value = f32::from_ne_bytes(*bytes);
        let expected = if i < 512 { 2560.0 } else { 5120.0 };
        assert!((value - expected).abs() < 1.0, "{i}: {value}");
    }
    let group = ctx.allocate(32).unwrap();
    group.write_at(&stream, 0, &[0x55; 32]).unwrap();
    let row = group.range(0, 32).unwrap();
    let x = input.range(0, 1024).unwrap();
    assert!(
        step.project_group(&[("lm_head", 0, 4), ("lm_head", 4, 3)], x, &row, 1)
            .is_err()
    );
    assert!(
        step.project_group(&[("lm_head", 0, 4), ("lm_head", 2, 4)], x, &row, 1)
            .is_err()
    );
    assert!(
        step.project_group(
            &[("lm_head", 0, 4), ("lm_head", 4, 4)],
            x,
            &group.range(0, 16).unwrap(),
            1
        )
        .is_err()
    );
    assert!(
        step.project_group(
            &[("lm_head", 0, 4), ("lm_head", 4, 4)],
            x,
            &input.range(0, 32).unwrap(),
            1
        )
        .is_err()
    );
    let mut input_after = vec![0u8; 1024];
    input.read_at(&stream, 0, &mut input_after).unwrap();
    assert_eq!(
        input_after,
        (0..256)
            .flat_map(|_| 1f32.to_ne_bytes())
            .collect::<Vec<_>>()
    );
    let mut grouped = [0u8; 32];
    group.read_at(&stream, 0, &mut grouped).unwrap();
    assert_eq!(grouped, [0x55; 32]);
    step.project_group(&[("lm_head", 0, 4), ("lm_head", 4, 4)], x, &row, 1)
        .unwrap();
    group.read_at(&stream, 0, &mut grouped).unwrap();
    for bytes in grouped.as_chunks::<4>().0 {
        let value = f32::from_ne_bytes(*bytes);
        assert!((value - 256.0).abs() < 0.5, "{value}");
    }
}

#[test]
#[ignore = "requires an admitted CUDA device"]
fn native_decode_projection_uses_supplied_scratch() {
    use crate::{
        family::step::rows::NativeRow,
        kernels::cuda::{
            CudaContext, NativeWeight,
            resident::{matvec_q::QuantType, quantize_q8_1::q8_1_bytes},
        },
    };
    let ctx = CudaContext::new().unwrap();
    let stream = ctx.stream().unwrap();
    let mut blocks = Vec::new();
    for _ in 0..32 {
        blocks.extend_from_slice(&0x3c00u16.to_le_bytes());
        blocks.extend_from_slice(&[1u8; 32]);
    }
    let w = ctx.allocate(blocks.len()).unwrap();
    w.write_at(&stream, 0, &blocks).unwrap();
    let weight = NativeWeight::from_buffer(w, QuantType::Q80, 256, 4, None).unwrap();
    let input = ctx.allocate(1024).unwrap();
    let x: Vec<u8> = (0..256).flat_map(|_| 1f32.to_ne_bytes()).collect();
    input.write_at(&stream, 0, &x).unwrap();
    let n = q8_1_bytes(256, 1).unwrap();
    let scratch = ctx.allocate(n).unwrap();
    let output = ctx.allocate(16).unwrap();
    scratch.write_at(&stream, 0, &vec![0x55; n]).unwrap();
    let call = NativeRow {
        weight: &weight,
        input: input.range(0, 1024).unwrap(),
        output: output.range(0, 16).unwrap(),
    };
    let q8 = scratch.range(0, n).unwrap();
    assert!(
        call.quantized(&ctx, &stream, weight.buffer().range(0, n).unwrap())
            .is_err()
    );
    for bad_output in [
        output.range(0, 4).unwrap(),
        weight.buffer().range(blocks.len() - 16, 16).unwrap(),
    ] {
        assert!(
            NativeRow {
                output: bad_output,
                ..call
            }
            .quantized(&ctx, &stream, q8)
            .is_err()
        );
    }
    let mut unchanged = vec![0; n];
    scratch.read_at(&stream, 0, &mut unchanged).unwrap();
    assert_eq!(unchanged, vec![0x55; n]);
    let mut weight_after = vec![0; blocks.len()];
    weight
        .buffer()
        .read_at(&stream, 0, &mut weight_after)
        .unwrap();
    assert_eq!(weight_after, blocks);
    // The opt-out uses the supplied scratch; the exact reference doesn't read or write it.
    call.exact(&ctx, &stream).unwrap();
    let mut expected = [0u8; 16];
    output.read_at(&stream, 0, &mut expected).unwrap();
    scratch.read_at(&stream, 0, &mut unchanged).unwrap();
    assert_eq!(unchanged, vec![0x55; n]);
    call.quantized(&ctx, &stream, q8).unwrap();
    let mut got = [0u8; 16];
    output.read_at(&stream, 0, &mut got).unwrap();
    scratch.read_at(&stream, 0, &mut unchanged).unwrap();
    assert_ne!(unchanged, vec![0x55; n]);
    for (got, expected) in got
        .as_chunks::<4>()
        .0
        .iter()
        .zip(expected.as_chunks::<4>().0)
    {
        assert!((f32::from_ne_bytes(*got) - f32::from_ne_bytes(*expected)).abs() < 0.5);
    }
    let weights = CudaWeights::from_weights(vec![CudaWeight::Native {
        name: "native".into(),
        weight,
    }])
    .unwrap();
    let config = include_bytes!("../../../../../tests/fixtures/qwen3_8_flash_next_config.json");
    let (model, kinds) = model_shape(config).unwrap();
    let admitted = super::gguf_decode_arena_bytes(&model, &kinds, 1, ctx.sm_count(), 256).unwrap();
    let step =
        FlashStep::new_gguf_decode(&ctx, &stream, &weights, model, &kinds, 1, 256, 0..1).unwrap();
    assert_eq!(step.scratch_bytes_for_test() as u64, admitted);
    output.write_at(&stream, 0, &[0; 16]).unwrap();
    step.project(
        "native",
        input.range(0, 1024).unwrap(),
        output.range(0, 16).unwrap(),
        1,
    )
    .unwrap();
    output.read_at(&stream, 0, &mut got).unwrap();
    assert_eq!(got, expected);
    assert_eq!(got.as_chunks::<4>().0, &[256.0f32.to_ne_bytes(); 4]);
}

#[test]
#[ignore = "needs CUDA and FIFI_FLASH_NEXT_EXL3_PACK layer-3 trunk"]
fn shared_arena_ownership_does_not_enable_shared_decode() {
    assert!(
        device_present("Flash-Next prefill phase"),
        "a GPU is required"
    );
    let pack = checkpoint();
    let projector = CudaProjector::new().expect("projector");
    let ctx = projector.context();
    let stream = ctx.stream().expect("stream");
    let weights = device_weights(&projector, &pack, &[3], (false, false));
    for shared in [false, true] {
        let make = if shared {
            FlashStep::new_prefill
        } else {
            FlashStep::new
        };
        let step = make(
            ctx,
            &stream,
            &weights,
            &pack.model_config,
            POSITIONS,
            3,
            3..4,
        )
        .expect("metadata-only step");
        assert_eq!(step.step_plan.index(Slot::PrefillPerm).is_some(), shared);
        assert!(
            !step.shared_prefill(),
            "decode starts outside the prefill window"
        );
        step.prefill_window.set(true);
        assert_eq!(step.shared_prefill(), shared);
        if shared {
            let launch = Launch {
                ctx,
                stream: &stream,
            };
            let actual = step
                .prefill_projector(&launch, 3)
                .expect("actual projector");
            assert_eq!(actual.tile_rows, PREFILL_TILE_ROWS);
            let jobs = crate::family::exl3::prefill::dense_jobs(3, actual.tile_rows);
            assert_eq!(
                actual.scratch.jobs.bytes(),
                crate::kernels::cuda::resident::exl3::Exl3Job::table(&jobs).len()
            );
        }
        step.prefill_window.set(false);
        assert!(!step.shared_prefill(), "closing the window restores decode");
        step.prefill_window.set(true);
        step.reset().expect("reset inside an open window");
        assert!(!step.shared_prefill(), "reset closes the window");
    }
}

/// Tokens of the sequence a layer runs.
const TOKENS: usize = 10;
/// Cached positions of the run (a multiple of the indexer ratio).
const POSITIONS: usize = 64;

fn checkpoint() -> Checkpoint {
    let dir = std::env::var_os("FIFI_FLASH_NEXT_EXL3_PACK").expect("FIFI_FLASH_NEXT_EXL3_PACK");
    Checkpoint::open(Path::new(&dir)).expect("open the pack")
}

fn window() -> Vec<usize> {
    std::env::var("FIFI_FLASH_NEXT_STEP_LAYERS")
        .unwrap_or_else(|_| "3".into())
        .split(',')
        .map(|layer| layer.trim().parse().expect("a layer number"))
        .collect()
}

fn matrix(checkpoint: &Checkpoint, prefix: &str) -> Matrix {
    let names = COMPONENT_SUFFIXES.map(|suffix| format!("{prefix}.{suffix}"));
    let keys = names.each_ref().map(String::as_str);
    let selected = checkpoint
        .source
        .read_selected(&keys, 1 << 30)
        .unwrap_or_else(|e| panic!("{prefix}: {e:?}"));
    Matrix::load(&selected, prefix, &checkpoint.config).expect("a matrix")
}

// ---------------------------------------------------------------------------------------------
// The host reference.

/// A weight of the host reference.
enum W {
    Dense(DenseWeight),
    /// A stored EXL3 matrix, projected by the CPU kernel.
    Exl3(OwnedPart),
    /// The routed experts of `layer`, one stack per part, read on demand.
    Stack(usize, ExpertPart),
}

struct RealProjector<'c> {
    cpu: ExpertCpu,
    checkpoint: &'c Checkpoint,
    experts: HashMap<(usize, usize, usize), OwnedPart>,
}

impl<'c> RealProjector<'c> {
    fn new(checkpoint: &'c Checkpoint) -> Self {
        assert_eq!(
            Exl3Isa::detect(),
            Exl3Isa::Avx512,
            "the reference needs AVX-512"
        );
        Self {
            cpu: ExpertCpu::new(
                4,
                WaitPolicy::SpinThenPark(Duration::from_micros(500)),
                Some(Exl3Isa::Avx512),
            )
            .expect("the CPU kernel"),
            checkpoint,
            experts: HashMap::new(),
        }
    }

    fn project(&self, part: &OwnedPart, x: &[f32]) -> Result<Vec<f32>, EngError> {
        let projection =
            Exl3Projection::new(part.shape, part.words, &part.trellis, &part.suh, &part.svh)?;
        let mut out = vec![0.0f32; part.shape[1]];
        self.cpu.project(&projection, x, &mut out)?;
        Ok(out)
    }
}

impl Projector for RealProjector<'_> {
    type Weight = W;

    fn linear(&mut self, weight: &W, x: &[f32]) -> Result<Vec<f32>, EngError> {
        match weight {
            W::Dense(dense) => DenseProjector::default().linear(dense, x),
            W::Exl3(part) => self.project(part, x),
            W::Stack(..) => Err(EngError::BadRequest("linear on an expert stack".into())),
        }
    }

    fn expert_linear(&mut self, stack: &W, expert: u32, x: &[f32]) -> Result<Vec<f32>, EngError> {
        let W::Stack(layer, part) = stack else {
            return Err(EngError::BadRequest(
                "expert_linear on a plain matrix".into(),
            ));
        };
        let key = (*layer, expert as usize, *part as usize);
        if !self.experts.contains_key(&key) {
            let name = expert_name_in(Root::Layers, *layer, expert as usize, *part);
            let owned = OwnedPart::from_matrix(&matrix(self.checkpoint, &name))?;
            self.experts.insert(key, owned);
        }
        let part = &self.experts[&key];
        self.project(part, x)
    }

    fn rows(&mut self, _: &W, _: &[u32]) -> Result<Vec<f32>, EngError> {
        Err(EngError::BadRequest("rows are not used by a layer".into()))
    }
}

fn dense(checkpoint: &Checkpoint, name: &str, rows: usize, cols: usize) -> W {
    W::Dense(DenseWeight::matrix(
        rows,
        cols,
        host_f32(&checkpoint.source, name, rows * cols).unwrap_or_else(|e| panic!("{name}: {e:?}")),
    ))
}

fn exl3(checkpoint: &Checkpoint, prefix: &str) -> W {
    W::Exl3(OwnedPart::from_matrix(&matrix(checkpoint, prefix)).expect("a part"))
}

fn vector(checkpoint: &Checkpoint, name: &str, len: usize) -> Vec<f32> {
    host_f32(&checkpoint.source, name, len).unwrap_or_else(|e| panic!("{name}: {e:?}"))
}

struct HostSite {
    norm: Vec<f32>,
    down: W,
    up: W,
    inject: W,
}

fn host_site(checkpoint: &Checkpoint, model: &ModelShape, prefix: &str) -> HostSite {
    let (width, rank, branches) = (
        model.hyper.width(),
        model.hyper.rank(),
        model.hyper.branches(),
    );
    HostSite {
        norm: vector(checkpoint, &format!("{prefix}.hc_norm.weight"), width),
        down: dense(
            checkpoint,
            &format!("{prefix}.input_mix_weight_down.weight"),
            rank,
            width,
        ),
        up: dense(
            checkpoint,
            &format!("{prefix}.input_mix_weight_up.weight"),
            width,
            rank,
        ),
        inject: dense(
            checkpoint,
            &format!("{prefix}.block_inject_weight.weight"),
            branches,
            width,
        ),
    }
}

impl HostSite {
    fn weights(&self) -> HyperWeights<'_, W> {
        HyperWeights {
            norm: &self.norm,
            down: &self.down,
            up: &self.up,
            inject: Some(&self.inject),
        }
    }
}

enum HostMixer {
    Delta {
        qkv: W,
        z: W,
        b: W,
        a: W,
        out: W,
        conv: Vec<f32>,
        a_log: Vec<f32>,
        dt_bias: Vec<f32>,
        norm: Vec<f32>,
        state: DeltaState,
    },
    Qsa {
        q: W,
        k: W,
        v: W,
        o: W,
        idx: W,
        norms: [Vec<f32>; 4],
        cache: QsaCache,
    },
}

struct HostLayer {
    layer: usize,
    attn: HostSite,
    mlp: HostSite,
    mixer: HostMixer,
    router: Vec<f32>,
    stacks: [W; 3],
    shared: [W; 3],
    shared_scalar: W,
}

fn host_layer(
    checkpoint: &Checkpoint,
    model: &ModelShape,
    kinds: &[bool],
    layer: usize,
) -> HostLayer {
    let h = model.hyper.hidden();
    let at = |suffix: &str| layer_name(layer, suffix);
    let mixer = if kinds[layer] {
        let dims = model.qsa.dims();
        let norm = |suffix: &str, len: u32| vector(checkpoint, &at(suffix), len as usize);
        HostMixer::Qsa {
            q: exl3(checkpoint, &at("self_attn.q_proj")),
            k: exl3(checkpoint, &at("self_attn.k_proj")),
            v: exl3(checkpoint, &at("self_attn.v_proj")),
            o: exl3(checkpoint, &at("self_attn.o_proj")),
            idx: exl3(checkpoint, &at("self_attn.indexer.index_qk_proj")),
            norms: [
                norm("self_attn.q_norm.weight", dims.head_dim),
                norm("self_attn.k_norm.weight", dims.head_dim),
                norm("self_attn.indexer.q_layernorm.weight", dims.idx_dim),
                norm("self_attn.indexer.k_layernorm.weight", dims.idx_dim),
            ],
            cache: QsaCache::new(),
        }
    } else {
        let d = &model.delta;
        let (heads, conv_dim) = (
            d.value_heads(),
            2 * d.key_heads() * d.key_dim() + d.value_heads() * d.value_dim(),
        );
        HostMixer::Delta {
            qkv: exl3(checkpoint, &at("linear_attn.in_proj_qkv")),
            z: exl3(checkpoint, &at("linear_attn.in_proj_z")),
            b: dense(checkpoint, &at("linear_attn.in_proj_b.weight"), heads, h),
            a: dense(checkpoint, &at("linear_attn.in_proj_a.weight"), heads, h),
            out: exl3(checkpoint, &at("linear_attn.out_proj")),
            conv: vector(checkpoint, &at("linear_attn.conv1d.weight"), conv_dim * 4),
            a_log: vector(checkpoint, &at("linear_attn.A_log"), heads),
            dt_bias: vector(checkpoint, &at("linear_attn.dt_bias"), heads),
            norm: vector(checkpoint, &at("linear_attn.norm.weight"), d.value_dim()),
            state: DeltaState::new(&model.delta),
        }
    };
    let part = |p: ExpertPart| W::Stack(layer, p);
    let shared = |suffix: &str| exl3(checkpoint, &at(&format!("mlp.shared_expert.{suffix}")));
    HostLayer {
        layer,
        attn: host_site(checkpoint, model, &at("attn_hyper_connection")),
        mlp: host_site(checkpoint, model, &at("mlp_hyper_connection")),
        mixer,
        router: vector(checkpoint, &at("mlp.gate.weight"), model.moe.experts() * h),
        stacks: [
            part(ExpertPart::Gate),
            part(ExpertPart::Up),
            part(ExpertPart::Down),
        ],
        shared: [shared("gate_proj"), shared("up_proj"), shared("down_proj")],
        shared_scalar: dense(checkpoint, &at("mlp.shared_expert_gate.weight"), 1, h),
    }
}

/// The host blocks of a layer at one token, in the order of `step.rs`.
struct HostRun {
    mixed_attn: Vec<f32>,
    attn_out: Vec<f32>,
    mixed_mlp: Vec<f32>,
    moe_out: Vec<f32>,
}

fn host_token(
    projector: &mut RealProjector<'_>,
    model: &ModelShape,
    layer: &mut HostLayer,
    pos: u32,
    stream: &mut [f32],
) -> HostRun {
    let read = hyper::read(projector, &model.hyper, &layer.attn.weights(), stream).expect("read");
    let gate = read.gate.clone().expect("gate");
    let attn_out = match &mut layer.mixer {
        HostMixer::Qsa {
            q,
            k,
            v,
            o,
            idx,
            norms,
            cache,
        } => qsa::step(
            projector,
            &model.qsa,
            &QsaWeights {
                q,
                k,
                v,
                o,
                idx,
                q_norm: &norms[0],
                k_norm: &norms[1],
                idx_q_norm: &norms[2],
                idx_k_norm: &norms[3],
            },
            cache,
            pos,
            &read.mixed,
        )
        .expect("qsa"),
        HostMixer::Delta {
            qkv,
            z,
            b,
            a,
            out,
            conv,
            a_log,
            dt_bias,
            norm,
            state,
        } => deltanet::step(
            projector,
            &model.delta,
            &DeltaWeights {
                qkv,
                z,
                b,
                a,
                out,
                conv,
                a_log,
                dt_bias,
                norm,
            },
            state,
            &read.mixed,
        )
        .expect("delta"),
    };
    hyper::apply(&model.hyper, stream, &gate, &attn_out).expect("apply");
    let read_m = hyper::read(projector, &model.hyper, &layer.mlp.weights(), stream).expect("read");
    let gate_m = read_m.gate.clone().expect("gate");
    let moe_out = moe::step(
        projector,
        &model.moe,
        &MoeWeights {
            router: &layer.router,
            gate: &layer.stacks[0],
            up: &layer.stacks[1],
            down: &layer.stacks[2],
            shared_gate: &layer.shared[0],
            shared_up: &layer.shared[1],
            shared_down: &layer.shared[2],
            shared_scalar: &layer.shared_scalar,
        },
        &read_m.mixed,
    )
    .expect("moe");
    hyper::apply(&model.hyper, stream, &gate_m, &moe_out).expect("apply");
    HostRun {
        mixed_attn: read.mixed,
        attn_out,
        mixed_mlp: read_m.mixed,
        moe_out,
    }
}

// ---------------------------------------------------------------------------------------------
// The device side.

/// The window's weights on the device: each layer's trunk tensors and its routed expert banks.
fn device_weights(
    projector: &CudaProjector,
    checkpoint: &Checkpoint,
    layers: &[usize],
    (with_head, with_banks): (bool, bool),
) -> CudaWeights {
    let list = TrunkList::new(&checkpoint.source, &checkpoint.model_config).expect("trunk list");
    let reader = PackReader::new(projector, checkpoint, Reserve::default());
    let (model, _) = model_shape(&checkpoint.model_config).expect("shape");
    let experts = model.moe.experts();
    let mut all = Vec::new();
    if with_head {
        all.extend(root_weights(&reader, &list).expect("the mixer and lm_head"));
    }
    for &layer in layers {
        all.extend(layer_weights(&reader, &list, layer).expect("the layer's trunk"));
        for part in ExpertPart::ALL.into_iter().filter(|_| with_banks) {
            let bank = projector
                .upload_exl3_bank(experts, Reserve::default(), |expert| {
                    let name = expert_name_in(Root::Layers, layer, expert, part);
                    Ok(matrix(checkpoint, &name))
                })
                .expect("a routed bank");
            let suffix = match part {
                ExpertPart::Gate => "gate_proj",
                ExpertPart::Up => "up_proj",
                ExpertPart::Down => "down_proj",
            };
            let key = bank_key(layer, suffix);
            all.push(CudaWeight::Exl3Bank { name: key, bank });
        }
    }
    CudaWeights::from_exl3(all).expect("the window's weights")
}

fn relative_l2(got: &[f32], want: &[f32]) -> f64 {
    assert_eq!(got.len(), want.len());
    let diff: f64 = got
        .iter()
        .zip(want)
        .map(|(g, w)| (f64::from(*g) - f64::from(*w)).powi(2))
        .sum();
    let norm: f64 = want.iter().map(|w| f64::from(*w).powi(2)).sum();
    (diff / norm.max(1e-30)).sqrt()
}

/// The largest absolute difference.
fn max_abs(got: &[f32], want: &[f32]) -> f64 {
    got.iter()
        .zip(want)
        .map(|(g, w)| (f64::from(*g) - f64::from(*w)).abs())
        .fold(0.0, f64::max)
}

/// A stream whose four branches differ, a different one per token.
fn input(width: usize, hidden: usize, token: usize) -> Vec<f32> {
    (0..width)
        .map(|i| {
            let branch = i / hidden;
            let raw = (((i + 613 * (token + 1)) * 7919 % 2003) as f32 - 1001.0) / 700.0;
            raw * (1.0 + 0.5 * branch as f32)
        })
        .collect()
}

/// Bounds on the relative L2 of each quantity against the host, each about three to four times the
/// worst value measured over ten tokens of the real pack (layer 3, layers 0 to 2, the head, the
/// split block). The device rounds every EXL3 weight (about 2e-4 per projection) and every
/// activation entering an EXL3 projection to binary16, and the host does not; the quantities that
/// never pass through an EXL3 projection (the hyper read of the attention site, all f32) are
/// therefore held far tighter than the block outputs, and a stream that is mostly the residual
/// tighter than a block. A missing term, a wrong head order or a wrong cache row moves a block by
/// 1e-2 or more.
struct Bounds {
    /// Mixed row of the attention site: measured 2.2e-6.
    mixed_attn: f64,
    /// Mixed row of the MLP site, after the attention block's EXL3 projections: measured 9.1e-5.
    mixed_mlp: f64,
    /// The residual stream after a layer: measured 5.5e-5.
    stream: f64,
    /// A block output, the DeltaNet state and ring, the logits: measured 4.6e-4 at most.
    block: f64,
}

const BOUNDS: Bounds = Bounds {
    mixed_attn: 1e-5,
    mixed_mlp: 3e-4,
    stream: 2e-4,
    block: 1.5e-3,
};

/// The MoE block a run uses.
#[derive(Clone, Copy, Debug)]
enum MoeKind {
    /// Qwen3.6's block over routed experts that are all resident (three banks per layer).
    Resident,
    /// The split block over a bank of `slots` slots per layer, held experts `0..slots` at the start.
    Split { slots: usize },
}

/// Bank layer `i` holds pack layer `layers[i]`.
struct WindowRecords<'a, 'b> {
    inner: &'b PackRecordSource<'a>,
    layers: Vec<usize>,
}

impl RecordSource for WindowRecords<'_, '_> {
    fn record(&self, layer: usize, expert: u16, out: &mut [u8]) -> Result<(), EngError> {
        self.inner.record(self.layers[layer], expert, out)
    }
}

enum Block<'w> {
    Resident(AllResidentMoe<'w>),
    Split(Box<SplitMoeAt<'w>>),
}

impl Block<'_> {
    fn as_block(&mut self) -> &mut dyn MoeBlock {
        match self {
            Self::Resident(block) => block,
            Self::Split(block) => block.as_mut(),
        }
    }

    /// Misses the bank has served so far (0 for the resident block).
    fn misses(&self) -> u64 {
        match self {
            Self::Resident(_) => 0,
            Self::Split(block) => block.inner.bank().state().stats().misses,
        }
    }
}

/// The split block over a bank of `slots` slots for each of `layers`.
fn split_block<'w>(
    projector: &CudaProjector,
    checkpoint: &Checkpoint,
    weights: &'w CudaWeights,
    layers: &[usize],
    slots: usize,
    dims: &crate::family::exl3::launch::MoeDims,
) -> SplitMoeAt<'w> {
    let source = PackRecordSource::new(checkpoint).expect("the record source");
    let record = source.layout().record_bytes();
    let geometry = BankGeometry::new(layers.len(), dims.n_expert, dims.top_k, slots, record)
        .expect("geometry");
    let held: Vec<i32> = (0..dims.n_expert as i32)
        .map(|expert| {
            if (expert as usize) < slots {
                expert
            } else {
                -1
            }
        })
        .collect();
    let rows: Vec<Vec<i32>> = layers.iter().map(|_| held.clone()).collect();
    let row_refs: Vec<&[i32]> = rows.iter().map(Vec::as_slice).collect();
    let counts: Vec<Vec<u64>> = layers
        .iter()
        .map(|_| (0..dims.n_expert as u64).rev().collect())
        .collect();
    let policy = ReplaceOnMissLru::from_fill(geometry, &row_refs, &counts, None).expect("the fill");
    let state = ExpertBankState::with_policy(geometry, HostMode::Exclusive, policy).expect("state");
    let plan = HostPinPlan::new(
        &geometry,
        HostMode::Exclusive,
        &HostMemory::from_pair(62 << 30, 55 << 30),
        &HostTerms::DEFAULT,
    )
    .expect("the host plan");
    let bank = DeviceBank::build(
        projector.context(),
        state,
        &WindowRecords {
            inner: &source,
            layers: layers.to_vec(),
        },
        &plan,
        None,
    )
    .expect("build the bank");
    let trunks = layers
        .iter()
        .map(|&layer| split_trunk(weights, layer).expect("the trunk of the split block"))
        .collect();
    SplitMoeAt::new(
        SplitMoe::new(projector.context(), trunks, bank, source.layout(), dims)
            .expect("the split block"),
        layers,
    )
    .expect("consecutive layers")
}

/// What a run leaves for comparison with another run: per token, the MoE output and the stream of
/// the last layer, and the bank's misses over the token (all layers).
struct RunOut {
    moe_out: Vec<Vec<f32>>,
    streams: Vec<Vec<f32>>,
    misses: Vec<u64>,
    /// Per DeltaNet layer of the window, the device recurrent state followed by the conv ring after
    /// the sequence.
    states: Vec<Vec<f32>>,
}

/// How the layers of a token are driven.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Drive {
    /// Block by block with a drain and a read-back after each, compared with the host.
    Stepped,
    /// All layers queued by one `run_layers_without_ple` call and one drain, as the decode loop
    /// does; only the stream after the last layer is read.
    Composed,
}

/// One run of a window of layers.
#[derive(Clone, Copy, Debug)]
struct Spec<'l> {
    layers: &'l [usize],
    kind: MoeKind,
    tokens: usize,
    /// Replaces `indexer_budget` of the pack's config (the pack's 2048 selects every block of a
    /// short sequence), host reference and device alike.
    budget: Option<u32>,
    drive: Drive,
}

impl<'l> Spec<'l> {
    fn new(layers: &'l [usize], kind: MoeKind) -> Self {
        Self {
            layers,
            kind,
            tokens: TOKENS,
            budget: None,
            drive: Drive::Stepped,
        }
    }
}

/// The pack's config with `indexer_budget` replaced.
fn config_with_budget(config: &[u8], budget: u32) -> Vec<u8> {
    let text = String::from_utf8(config.to_vec()).expect("a utf-8 config");
    let key = "\"indexer_budget\"";
    let at = text.find(key).expect("indexer_budget in the config") + key.len();
    let rest = &text[at..];
    let digits_from = rest.find(|c: char| c.is_ascii_digit()).expect("a number");
    let digits = rest[digits_from..]
        .find(|c: char| !c.is_ascii_digit())
        .expect("the end of the number");
    let (start, end) = (at + digits_from, at + digits_from + digits);
    format!("{}{budget}{}", &text[..start], &text[end..]).into_bytes()
}

fn run_window(layers: &[usize], kind: MoeKind) -> RunOut {
    run_spec(Spec::new(layers, kind))
}

fn run_spec(spec: Spec<'_>) -> RunOut {
    let Spec {
        layers,
        kind,
        tokens,
        budget,
        drive,
    } = spec;
    assert!(
        device_present("Flash-Next layer parity"),
        "a GPU is required"
    );
    let checkpoint = checkpoint();
    let config = match budget {
        Some(budget) => config_with_budget(&checkpoint.model_config, budget),
        None => checkpoint.model_config.clone(),
    };
    let (model, kinds) = model_shape(&config).expect("shape");
    let projector = CudaProjector::new().expect("projector");
    let ctx = projector.context();
    let stream: DeviceStream = ctx.stream().expect("stream");
    let with_head = std::env::var_os("FIFI_FLASH_NEXT_STEP_HEAD").is_some();
    let weights = device_weights(
        &projector,
        &checkpoint,
        layers,
        (with_head, matches!(kind, MoeKind::Resident)),
    );
    let first = *layers.first().expect("a layer");
    let end = layers.last().expect("a layer") + 1;
    let step = FlashStep::new(ctx, &stream, &weights, &config, POSITIONS, 1, first..end)
        .expect("the step");
    let hidden = model.hyper.hidden();
    let width = model.hyper.width();
    let mut host_projector = RealProjector::new(&checkpoint);
    let mut host: Vec<HostLayer> = layers
        .iter()
        .map(|&layer| host_layer(&checkpoint, &model, &kinds, layer))
        .collect();
    let host_head = with_head.then(|| {
        let width = model.hyper.width();
        let rank = model.hyper.rank();
        (
            vector(&checkpoint, &format!("{MIXER}.hc_norm.weight"), width),
            dense(
                &checkpoint,
                &format!("{MIXER}.input_mix_weight_down.weight"),
                rank,
                width,
            ),
            dense(
                &checkpoint,
                &format!("{MIXER}.input_mix_weight_up.weight"),
                width,
                rank,
            ),
            exl3(&checkpoint, "lm_head"),
        )
    });
    let (words, words_at) = step.words_buffer().expect("words");
    let mut block = match kind {
        MoeKind::Resident => Block::Resident(AllResidentMoe { weights: &weights }),
        MoeKind::Split { slots } => Block::Split(Box::new(split_block(
            &projector,
            &checkpoint,
            &weights,
            layers,
            slots,
            &step.moe_dims(1),
        ))),
    };
    let mut out = RunOut {
        moe_out: Vec::new(),
        streams: Vec::new(),
        misses: Vec::new(),
        states: Vec::new(),
    };

    for token in 0..tokens {
        let mut host_stream = input(width, hidden, token);
        let misses_start = block.misses();
        let mut last = (Vec::new(), Vec::new());
        let stream_range = step.slot(Slot::Stream).expect("stream");
        step.begin_token().expect("begin");
        let params = crate::kernels::cuda::resident::step_params::StepParams::new(1, token as i32)
            .expect("params");
        words
            .write_at(&stream, words_at, &params.to_bytes())
            .expect("params");
        stream_range
            .buffer()
            .write_at(
                &stream,
                stream_range.offset(),
                &host_stream
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect::<Vec<u8>>(),
            )
            .expect("stream");
        if drive == Drive::Composed {
            step.run_layers_without_ple(token, 1, block.as_block())
                .expect("the layers");
            step.finish_token().expect("a clean token");
            let device_stream = step
                .read_f32(step.slot(Slot::Stream).unwrap(), width)
                .expect("stream");
            // Only the stream survives the whole window: the block outputs are lifetime-aliased.
            out.moe_out.push(Vec::new());
            out.streams.push(device_stream);
            out.misses.push(block.misses() - misses_start);
            continue;
        }
        for (index, &layer) in layers.iter().enumerate() {
            let reference = host_token(
                &mut host_projector,
                &model,
                &mut host[index],
                token as u32,
                &mut host_stream,
            );
            // Device, in steps with a read-back after each block.
            let attn_prefix = layer_name(layer, "attn_hyper_connection");
            step.hyper_read(Site::Attn, &attn_prefix, 1)
                .expect("hyper read");
            let mixed = step
                .read_f32(step.slot(Slot::Mixed(Site::Attn)).unwrap(), hidden)
                .expect("mixed");
            step.mixer(layer, token, 1).expect("mixer");
            let attn_out = step
                .read_f32(step.slot(Slot::BlockOut(Site::Attn)).unwrap(), hidden)
                .expect("block out");
            step.hyper_apply(Site::Attn, 1).expect("apply");
            step.hyper_read(Site::Mlp, &layer_name(layer, "mlp_hyper_connection"), 1)
                .expect("hyper read");
            let mixed_mlp = step
                .read_f32(step.slot(Slot::Mixed(Site::Mlp)).unwrap(), hidden)
                .expect("mixed");
            step.moe_block(layer, 0, block.as_block(), 1).expect("moe");
            let moe_out = step
                .read_f32(step.slot(Slot::BlockOut(Site::Mlp)).unwrap(), hidden)
                .expect("moe out");
            step.hyper_apply(Site::Mlp, 1).expect("apply");
            let device_stream = step
                .read_f32(step.slot(Slot::Stream).unwrap(), width)
                .expect("stream");
            let (status, flag) = step.health().expect("health");
            let pairs: [(&str, &[f32], &[f32], f64); 5] = [
                (
                    "mixed(attn)",
                    &mixed,
                    &reference.mixed_attn,
                    BOUNDS.mixed_attn,
                ),
                ("attn out", &attn_out, &reference.attn_out, BOUNDS.block),
                (
                    "mixed(mlp)",
                    &mixed_mlp,
                    &reference.mixed_mlp,
                    BOUNDS.mixed_mlp,
                ),
                ("moe out", &moe_out, &reference.moe_out, BOUNDS.block),
                ("stream", &device_stream, &host_stream, BOUNDS.stream),
            ];
            let errors: Vec<(&str, f64, f64)> = pairs
                .iter()
                .map(|(name, got, want, bound)| (*name, relative_l2(got, want), *bound))
                .collect();
            println!(
                "layer {layer} token {token}: status {status} flag {flag}; {}",
                pairs
                    .iter()
                    .map(|(name, got, want, _)| format!(
                        "{name} rel {:.2e} max {:.2e} (ref max {:.2e})",
                        relative_l2(got, want),
                        max_abs(got, want),
                        want.iter().fold(0.0f32, |m, v| m.max(v.abs()))
                    ))
                    .collect::<Vec<_>>()
                    .join("; ")
            );
            assert_eq!(
                (status, flag),
                (0, 0),
                "layer {layer} token {token}: health"
            );
            for (name, error, bound) in errors {
                assert!(
                    error <= bound,
                    "layer {layer} token {token}: {name} differs from the host by {error:.3e} \
                     (bound {bound:.1e})"
                );
            }
            // The next layer reads the host stream the device holds: keep them in step, so one
            // layer's error does not compound into the next layer's comparison.
            last = (moe_out.clone(), device_stream.clone());
            host_stream = device_stream;
        }
        out.moe_out.push(last.0);
        out.streams.push(last.1);
        out.misses.push(block.misses() - misses_start);
        if host_head.is_none() {
            step.finish_token().expect("a clean token");
        }
        if let Some((norm, down, up, lm_head)) = &host_head {
            step.head(1).expect("the head");
            step.finish_token().expect("a clean token with the head");
            let vocab = step.geometry.vocab;
            let device_logits = step
                .read_f32(step.slot(Slot::Logits).unwrap(), vocab)
                .expect("logits");
            let read = hyper::read(
                &mut host_projector,
                &model.hyper,
                &HyperWeights {
                    norm,
                    down,
                    up,
                    inject: None,
                },
                &host_stream,
            )
            .expect("final mixer");
            let host_logits = host_projector
                .linear(lm_head, &read.mixed)
                .expect("lm_head");
            let top = |logits: &[f32], count: usize| -> Vec<usize> {
                let mut order: Vec<usize> = (0..logits.len()).collect();
                order.sort_by(|a, b| logits[*b].total_cmp(&logits[*a]));
                order.truncate(count);
                order
            };
            let (device_top, host_top) = (top(&device_logits, 20), top(&host_logits, 20));
            let overlap = device_top.iter().filter(|id| host_top.contains(id)).count();
            let error = relative_l2(&device_logits, &host_logits);
            println!(
                "token {token}: logits rel {error:.2e} max {:.2e} (ref max {:.2e}); top-1 device {} \
                 host {}; top-20 overlap {overlap}",
                max_abs(&device_logits, &host_logits),
                host_logits.iter().fold(0.0f32, |m, v| m.max(v.abs())),
                device_top[0],
                host_top[0],
            );
            // A truncated model has no preferred token; the top-1 agreement is reported, the
            // logits error is the gate.
            assert!(
                error <= BOUNDS.block,
                "token {token}: logits differ by {error:.3e}"
            );
        }
    }
    // The recurrent state and the conv ring of every DeltaNet layer after the sequence: drift here
    // compounds over a long decode, where a block's error does not.
    for (index, &layer) in layers.iter().enumerate() {
        let HostMixer::Delta { state, .. } = &host[index].mixer else {
            continue;
        };
        let delta = step.geometry.delta;
        step.stream.synchronize().expect("drain");
        let rec = step
            .read_f32(
                step.state_slot(StateSlot::DeltaRec(layer)).unwrap(),
                state.rec().len(),
            )
            .expect("recurrent state");
        let ring_device = step
            .read_f32(
                step.state_slot(StateSlot::DeltaConv(layer)).unwrap(),
                state.ring().len(),
            )
            .expect("conv ring");
        // The device ring is planar `[tap][channel]` (oldest tap first), the host's `[channel][tap]`.
        let taps = state.ring().len() / delta.conv_dim;
        let mut ring_host_planar = vec![0.0f32; state.ring().len()];
        for channel in 0..delta.conv_dim {
            for tap in 0..taps {
                ring_host_planar[tap * delta.conv_dim + channel] =
                    state.ring()[channel * taps + tap];
            }
        }
        out.states
            .push(rec.iter().chain(&ring_device).copied().collect());
        if drive == Drive::Composed {
            // No host reference ran: the state is compared with the stepped run's, bit for bit.
            continue;
        }
        let rec_error = relative_l2(&rec, state.rec());
        let ring_error = relative_l2(&ring_device, &ring_host_planar);
        println!(
            "layer {layer} after {tokens} tokens: recurrent state rel {rec_error:.2e} max {:.2e} \
             (ref max {:.2e}); conv ring rel {ring_error:.2e} max {:.2e} (ref max {:.2e})",
            max_abs(&rec, state.rec()),
            state.rec().iter().fold(0.0f32, |m, v| m.max(v.abs())),
            max_abs(&ring_device, &ring_host_planar),
            ring_host_planar.iter().fold(0.0f32, |m, v| m.max(v.abs())),
        );
        assert!(
            rec_error <= BOUNDS.block,
            "layer {layer}: recurrent state {rec_error:.3e}"
        );
        assert!(
            ring_error <= BOUNDS.block,
            "layer {layer}: conv ring {ring_error:.3e}"
        );
    }
    out
}

#[test]
#[ignore = "needs a GPU, --test-threads=1 and the pack in FIFI_FLASH_NEXT_EXL3_PACK (layers in FIFI_FLASH_NEXT_STEP_LAYERS)"]
fn the_device_layers_match_the_host_reference_on_the_real_pack() {
    run_window(&window(), MoeKind::Resident);
}

/// The indexer's selection has to pick: with the pack's budget of 2048 positions every block of a
/// short sequence is selected and a wrong ranking cannot show. A budget of 8 positions is two
/// blocks of the pack's four; from the 13th token on there are more complete blocks than the
/// selection takes, so the cache, the indexer scores and the selection all decide the output.
/// Run on the window's QSA layers (the DeltaNet layers do not read the budget). fp16 score
/// rounding can flip two near-equal blocks; a flip shows as an error far above the bounds, and is
/// then reported rather than bounded away.
#[test]
#[ignore = "needs a GPU, --test-threads=1 and the pack in FIFI_FLASH_NEXT_EXL3_PACK (layers in FIFI_FLASH_NEXT_STEP_LAYERS)"]
fn the_selection_of_blocks_matches_the_host_when_it_has_to_choose() {
    let layers = window();
    let (model, kinds) = model_shape(&checkpoint().model_config).expect("shape");
    assert!(
        layers.iter().any(|&layer| kinds[layer]),
        "the window {layers:?} has no QSA layer"
    );
    let tokens = 16;
    let budget = 8;
    let ratio = model.qsa.dims().ratio as usize;
    assert!(
        tokens / ratio > budget / ratio,
        "{tokens} tokens give {} complete blocks, a budget of {budget} takes {}: the selection would not choose",
        tokens / ratio,
        budget / ratio
    );
    run_spec(Spec {
        tokens,
        budget: Some(budget as u32),
        ..Spec::new(&layers, MoeKind::Resident)
    });
}

/// The decode loop queues every layer of a token and drains once. That path has to give the same
/// stream, bit for bit, as the block-by-block run the host comparison uses (same kernels in the
/// same order on the same stream; the drains between blocks must change nothing).
#[test]
#[ignore = "needs a GPU, --test-threads=1 and the pack in FIFI_FLASH_NEXT_EXL3_PACK (layers in FIFI_FLASH_NEXT_STEP_LAYERS)"]
fn the_composed_layers_equal_the_stepped_layers_bit_for_bit() {
    let layers = window();
    let stepped = run_window(&layers, MoeKind::Resident);
    let composed = run_spec(Spec {
        drive: Drive::Composed,
        ..Spec::new(&layers, MoeKind::Resident)
    });
    assert_eq!(stepped.streams.len(), composed.streams.len());
    assert!(
        !stepped.states.is_empty(),
        "the window {layers:?} has no DeltaNet layer: the state comparison would pass vacuously \
         (use FIFI_FLASH_NEXT_STEP_LAYERS=0,1,2,3)"
    );
    assert_eq!(stepped.states.len(), composed.states.len());
    for (index, (a, b)) in stepped.states.iter().zip(&composed.states).enumerate() {
        let equal = a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits());
        println!("DeltaNet layer {index} of the window: state and ring bit for bit: {equal}");
        assert!(
            equal,
            "DeltaNet layer {index} of the window: the composed state differs"
        );
    }
    for token in 0..stepped.streams.len() {
        let equal = stepped.streams[token]
            .iter()
            .zip(&composed.streams[token])
            .all(|(a, b)| a.to_bits() == b.to_bits());
        println!(
            "token {token}: composed stream against stepped, rel {:.2e}, bit for bit: {equal}",
            relative_l2(&composed.streams[token], &stepped.streams[token])
        );
        assert!(
            equal,
            "token {token}: the composed run differs from the stepped run"
        );
    }
}

/// A failed call poisons the step: every later call refuses until `reset`, a nonzero status word
/// poisons at `finish_token`, and a `run_layers` that would cross the PLE layer refuses and names
/// the numbers.
#[test]
#[ignore = "needs a GPU, --test-threads=1 and the pack in FIFI_FLASH_NEXT_EXL3_PACK (layer 3)"]
fn a_failed_call_poisons_the_step_until_reset() {
    assert!(device_present("Flash-Next poison"), "a GPU is required");
    let checkpoint = checkpoint();
    let projector = CudaProjector::new().expect("projector");
    let ctx = projector.context();
    let stream: DeviceStream = ctx.stream().expect("stream");
    let weights = device_weights(&projector, &checkpoint, &[3], (false, true));
    let step = FlashStep::new(
        ctx,
        &stream,
        &weights,
        &checkpoint.model_config,
        POSITIONS,
        1,
        3..4,
    )
    .expect("the step");
    let mut moe = AllResidentMoe { weights: &weights };

    // A layer whose weights were never uploaded fails, and poisons.
    assert!(!step.is_poisoned());
    let error = step
        .run_layer(5, 0, 1, &mut moe)
        .expect_err("layer 5 has no weights");
    println!("layer 5: {error:?}");
    assert!(step.is_poisoned());
    for refused in [
        step.begin_token(),
        step.run_layer(3, 0, 1, &mut moe),
        step.head(1),
    ] {
        let text = format!("{:?}", refused.expect_err("a poisoned step refuses"));
        assert!(text.contains("reset"), "{text}");
    }
    step.reset().expect("reset");
    assert!(!step.is_poisoned());

    // A clean token finishes clean; a nonzero status word fails the finish and poisons.
    step.begin_token().expect("begin");
    step.finish_token().expect("nothing ran, nothing wrong");
    let status = step.status_word().expect("status word");
    status
        .buffer()
        .write_at(&stream, status.offset(), &1u32.to_le_bytes())
        .expect("write the status word");
    let error = step.finish_token().expect_err("a nonzero status word");
    println!("status word 1: {error:?}");
    assert!(step.is_poisoned());
    step.reset().expect("reset");

    // The PLE refusal names the layer and the range, and does not poison (nothing ran).
    let ple = step.ple_layer();
    let crossing = FlashStep::new(
        ctx,
        &stream,
        &weights,
        &checkpoint.model_config,
        POSITIONS,
        1,
        (ple - 1)..(ple + 1),
    )
    .expect("a window across the PLE layer");
    let error = crossing
        .run_layers((ple - 1)..(ple + 1), 0, 1, &mut moe)
        .expect_err("a range across the PLE layer");
    let text = format!("{error:?}");
    println!("PLE: {text}");
    assert!(text.contains(&format!("PLE layer {ple}")), "{text}");
    assert!(!crossing.is_poisoned());
    // A range outside the window refuses too, and does not poison.
    let outside = crossing
        .run_layers(ple..(ple + 5), 0, 1, &mut moe)
        .expect_err("a range outside the window");
    let text = format!("{outside:?}");
    println!("outside: {text}");
    assert!(text.contains("not inside"), "{text}");
    assert!(!crossing.is_poisoned());
}

#[test]
#[ignore = "needs a GPU, --test-threads=1 and the full pack in FIFI_FLASH_NEXT_EXL3_PACK (6 GiB)"]
fn the_split_moe_block_matches_the_resident_block_and_the_host_across_tokens() {
    let layers = window();
    let resident = run_window(&layers, MoeKind::Resident);
    // (a) A bank that holds every expert: no miss, one pass.
    let full = run_window(&layers, MoeKind::Split { slots: 512 });
    assert!(
        full.misses.iter().all(|&m| m == 0),
        "misses {:?}",
        full.misses
    );
    let mut worst = 0.0f64;
    for token in 0..TOKENS {
        let moe = relative_l2(&full.moe_out[token], &resident.moe_out[token]);
        let stream = relative_l2(&full.streams[token], &resident.streams[token]);
        let bits_equal = full.moe_out[token]
            .iter()
            .zip(&resident.moe_out[token])
            .all(|(a, b)| a.to_bits() == b.to_bits());
        println!(
            "512 slots, token {token}: moe out against the resident block rel {moe:.2e} max {:.2e}, \
             stream rel {stream:.2e}, bit for bit: {bits_equal}",
            max_abs(&full.moe_out[token], &resident.moe_out[token])
        );
        worst = worst.max(moe).max(stream);
    }
    assert!(
        worst <= 1e-5,
        "the 512-slot split block differs from the resident block by {worst:.3e}"
    );
    // (b) A small bank: the picks miss at every token, and the bank state carries across tokens.
    let small = run_window(&layers, MoeKind::Split { slots: 32 });
    println!("32 slots: misses per token {:?}", small.misses);
    assert!(
        small.misses.iter().all(|&m| m > 0),
        "a 32-slot bank must miss at every token: {:?}",
        small.misses
    );
    // Top-10 of 512 experts per layer and token with 32 held: about 9 of the 10 picks miss, so a
    // floor of 3 misses per layer and token leaves room for a lucky token and still fails a bank
    // that does not replace.
    let floor = 3 * layers.len() as u64;
    assert!(
        small.misses.iter().all(|&m| m >= floor),
        "a 32-slot bank must miss at least {floor} times per token: {:?}",
        small.misses
    );
}

#[test]
#[ignore = "needs a GPU, --test-threads=1 and the pack in FIFI_FLASH_NEXT_EXL3_PACK (layers in FIFI_FLASH_NEXT_STEP_LAYERS)"]
fn a_chunk_of_rows_matches_the_same_rows_one_at_a_time() {
    // The chunked prefill contract at the step level: M rows through one in-window
    // shared-pass chunk of a rows=M step equal M launches of a rows=1 decode step of
    // the same split block (equal-start banks), stream rows and DeltaNet states alike.
    // The tolerance covers the two MoE launch families of the split block (the
    // decode arm's indexed launches against the shared pass's job-table tiles; the
    // scan itself is bit-exact by contract).
    assert!(
        device_present("Flash-Next chunk parity"),
        "a GPU is required"
    );
    let layers = window();
    let m = 4usize;
    let checkpoint = checkpoint();
    let (model, _kinds) = model_shape(&checkpoint.model_config).expect("shape");
    let hidden = model.hyper.hidden();
    let width = model.hyper.width();
    let projector = CudaProjector::new().expect("projector");
    let ctx = projector.context();
    let stream: DeviceStream = ctx.stream().expect("stream");
    let weights = device_weights(&projector, &checkpoint, &layers, (false, true));
    let first = *layers.first().expect("a layer");
    let end = *layers.last().expect("a layer") + 1;
    let single = FlashStep::new(
        ctx,
        &stream,
        &weights,
        &checkpoint.model_config,
        POSITIONS,
        1,
        first..end,
    )
    .expect("the single-row step");
    let chunk = FlashStep::new_prefill(
        ctx,
        &stream,
        &weights,
        &checkpoint.model_config,
        POSITIONS,
        m,
        first..end,
    )
    .expect("the chunk step");
    // Each block takes its own step's dims: the shared pass checks the chunk's rows
    // against the dims the block was planned with (rows=1 would refuse rows=4).
    let slots = 20;
    let dims = single.moe_dims(1);
    let mut singles_block = split_block(&projector, &checkpoint, &weights, &layers, slots, &dims);
    let chunk_dims = chunk.moe_dims(m);
    let mut chunk_block = split_block(
        &projector,
        &checkpoint,
        &weights,
        &layers,
        slots,
        &chunk_dims,
    );

    // One row at a time, outside any prefill window (the decode arm).
    let mut singles = Vec::new();
    for token in 0..m {
        let row = input(width, hidden, token);
        single.begin_token().expect("begin");
        write_params(&single, 1, token as i32);
        write_stream_row(&single, &row);
        single
            .run_layers_without_ple(token, 1, &mut singles_block)
            .expect("the layers");
        single.finish_token().expect("a clean token");
        singles.push(
            single
                .read_f32(single.slot(Slot::Stream).unwrap(), width)
                .expect("stream"),
        );
    }

    // The same rows as one chunk, inside the prefill window (the shared pass).
    let rows: Vec<f32> = (0..m)
        .flat_map(|token| input(width, hidden, token))
        .collect();
    chunk_block.begin_prefill().expect("open the window");
    chunk.prefill_window.set(true);
    chunk.begin_chunk(m).expect("begin");
    write_params(&chunk, 1, 0);
    write_stream_row(&chunk, &rows);
    chunk
        .run_layers_without_ple(0, m, &mut chunk_block)
        .expect("the layers");
    chunk.finish_token().expect("a clean chunk");
    chunk.prefill_window.set(false);
    let chunked = chunk
        .read_f32(chunk.slot(Slot::Stream).unwrap(), m * width)
        .expect("stream");

    for token in 0..m {
        let got = &chunked[token * width..(token + 1) * width];
        let error = relative_l2(got, &singles[token]);
        println!("row {token}: relative L2 {error:.3e}");
        assert!(
            error < 4e-4,
            "row {token} of the chunk drifts from the single-row run: relative L2 {error:.3e}"
        );
    }

    // The DeltaNet recurrent state and conv ring after the sequence.
    for &layer in &layers {
        if chunk.geometry.layers.get(layer) != Some(&crate::family::AttentionKind::DeltaNet) {
            continue;
        }
        let ring_len = single
            .state_slot(StateSlot::DeltaConv(layer))
            .unwrap()
            .bytes()
            / 4;
        let rec_len = single
            .state_slot(StateSlot::DeltaRec(layer))
            .unwrap()
            .bytes()
            / 4;
        single.stream.synchronize().expect("drain");
        let a = single
            .read_f32(
                single.state_slot(StateSlot::DeltaRec(layer)).unwrap(),
                rec_len,
            )
            .expect("recurrent state");
        let b = chunk
            .read_f32(
                chunk.state_slot(StateSlot::DeltaRec(layer)).unwrap(),
                rec_len,
            )
            .expect("recurrent state");
        let error = relative_l2(&b, &a);
        println!(
            "layer {layer} recurrent state: relative L2 {error:.3e} ({} values)",
            rec_len
        );
        // The chunk's projections run the job-table kernel (FP16-rounded operands, F32
        // accumulation) where the single rows run legacy F32, and that rounding difference
        // compounds through the recurrent state; 1e-3 covers it with room.
        assert!(
            error < 1e-3,
            "layer {layer}: the chunk's recurrent state drifts beyond the jobs-kernel rounding"
        );
        let ra = single
            .read_f32(
                single.state_slot(StateSlot::DeltaConv(layer)).unwrap(),
                ring_len,
            )
            .expect("conv ring");
        let rb = chunk
            .read_f32(
                chunk.state_slot(StateSlot::DeltaConv(layer)).unwrap(),
                ring_len,
            )
            .expect("conv ring");
        let error = relative_l2(&rb, &ra);
        println!("layer {layer} conv ring: relative L2 {error:.3e}");
        assert!(
            error < 1e-3,
            "layer {layer}: the chunk's conv ring drifts beyond the jobs-kernel rounding"
        );
    }
}

#[test]
#[ignore = "needs a GPU, an 8 GiB lease and the pack in FIFI_FLASH_NEXT_EXL3_PACK (root weights only)"]
fn shared_prefill_owner_allocates_the_extended_list_and_zeroes_it() {
    assert!(device_present(
        "shared_prefill_owner_allocates_the_extended_list_and_zeroes_it"
    ));
    let checkpoint = checkpoint();
    let projector = CudaProjector::new().unwrap();
    let ctx = projector.context();
    let stream = ctx.stream().unwrap();
    let weights = device_weights(&projector, &checkpoint, &[], (true, false));
    let step = FlashStep::new_prefill(
        ctx,
        &stream,
        &weights,
        &checkpoint.model_config,
        64,
        64,
        0..1,
    )
    .unwrap();
    let view = step.slot(Slot::PrefillY).unwrap();
    let scratch = view.buffer();
    assert_eq!(scratch.bytes(), step.step_plan.arena_bytes());
    assert!(step.step_plan.index(Slot::PrefillPerm).is_some());
    assert!(step.step_plan.index(Slot::PrefillY).is_some());
    for entry in step.step_plan.slots() {
        let range = step.slot(entry.slot).unwrap();
        assert_eq!(range.bytes(), entry.bytes);
        assert_eq!(
            step.step_plan.placement(entry.slot),
            Some((range.offset(), range.bytes()))
        );
    }
    let mut bytes = vec![0u8; scratch.bytes()];
    scratch.read_at(&stream, 0, &mut bytes).unwrap();
    assert!(bytes.iter().all(|&byte| byte == 0));
}

fn write_params(step: &FlashStep<'_>, token: i32, pos: i32) {
    let (words, words_at) = step.words_buffer().expect("words");
    let params =
        crate::kernels::cuda::resident::step_params::StepParams::new(token, pos).expect("params");
    words
        .write_at(&step.stream, words_at, &params.to_bytes())
        .expect("params");
}

fn write_stream_row(step: &FlashStep<'_>, values: &[f32]) {
    let range = step.slot(Slot::Stream).expect("stream");
    range
        .buffer()
        .write_at(
            &step.stream,
            range.offset(),
            &values
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<u8>>(),
        )
        .expect("stream");
}
