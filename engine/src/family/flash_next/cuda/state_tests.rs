//! Host tests of the state plan: sizes against the host reference and the kernel size functions,
//! no overlap, the bytes at 32K, 128K and 256K against the memory table of
//! `docs/plans/flash-next-cuda.md`, and the bookkeeping book at the plan's capacity.

use super::state::{
    FlashGeometry, StatePlan, StateSlot, WORD_PREFILL_STATUS, WORD_RESULT_PAIR, WORDS_BYTES,
};
use crate::{
    EngError,
    family::{
        AttentionKind,
        flash_next::{
            deltanet::DeltaState,
            load::model_shape,
            moe::context_budget::{KvFormat, KvGeometry},
            ple::PleState,
            step::ModelShape,
        },
        graph_state::book::Health,
        qwen36::cuda::state_layout::Word,
    },
    kernels::cuda::{
        prefill::gdn_scan_seq::{ScanShape, scan_bytes},
        resident::attn_prep::cache_bytes,
    },
};

const CONFIG: &[u8] = include_bytes!("../../../../tests/fixtures/qwen3_8_flash_next_config.json");
const KIB: usize = 1 << 10;
const MIB: usize = 1 << 20;
const GIB: usize = 1 << 30;

pub(super) fn real() -> (ModelShape, FlashGeometry) {
    let (shape, kinds) = model_shape(CONFIG).expect("the model shape");
    let geometry = FlashGeometry::new(&shape, &kinds).expect("the geometry");
    (shape, geometry)
}

#[test]
fn the_geometry_reads_the_flash_next_config() {
    let (_, g) = real();
    assert_eq!(g.layers.len(), 48);
    assert_eq!(g.delta_layers(), 36);
    assert_eq!(g.qsa.cached_layers, 12);
    assert_eq!(g.layers[3], AttentionKind::Qsa);
    assert_eq!(g.layers[4], AttentionKind::DeltaNet);
    assert_eq!(
        (g.hidden, g.branches, g.width(), g.rank),
        (2560, 4, 10240, 320)
    );
    assert_eq!(g.delta.conv_dim, 10240);
    assert_eq!(g.delta.proj_ld(), 10240 + 6144 + 96);
    assert_eq!(
        (g.qsa.heads_q, g.qsa.heads_kv, g.qsa.head_dim),
        (24, 2, 256)
    );
    assert_eq!((g.qsa.idx_heads, g.qsa.idx_dim, g.qsa.ratio), (4, 128, 4));
    assert_eq!(g.qsa.budget_blocks(), 512);
    assert_eq!(g.qsa.proj_ld(), 12288 + 1024 + 640);
    assert_eq!(
        (g.experts, g.top_k, g.inter, g.shared_inter),
        (512, 10, 640, 640)
    );
    assert_eq!(g.vocab, 248_320);
    assert_eq!(g.ple_embed, 2560);
    assert_eq!(g.ple_layer, 1);
}

#[test]
fn the_device_state_has_the_bytes_of_the_host_state() {
    let (shape, g) = real();
    let plan = StatePlan::new(&g, 32_768, KvFormat::F16).unwrap();
    let host_delta = DeltaState::new(&shape.delta).byte_len();
    assert_eq!(host_delta, 3_145_728 + 122_880);
    for (layer, kind) in g.layers.iter().enumerate() {
        if *kind == AttentionKind::DeltaNet {
            let rec = plan.placement(StateSlot::DeltaRec(layer)).unwrap().1;
            let ring = plan.placement(StateSlot::DeltaConv(layer)).unwrap().1;
            assert_eq!(rec + ring, host_delta, "layer {layer}");
        }
    }
    let bytes = plan.bytes().unwrap();
    assert_eq!(bytes.delta_state_bytes, 36 * host_delta);
    // The figure of the context budget's load plan: 117.7 MB, 112.2 MiB.
    assert_eq!(bytes.delta_state_bytes, 117_669_888);
    let ple = plan.placement(StateSlot::PleRing).unwrap().1;
    assert_eq!(ple, PleState::new(&shape.ple).byte_len());
    assert_eq!(ple, 368_640);
}

#[test]
fn the_state_sizes_match_the_kernel_size_functions() {
    let (_, g) = real();
    let d = &g.delta;
    let scan = scan_bytes(ScanShape {
        m: 1,
        p0: 0,
        ctx_tokens: 1,
        n_k_heads: d.key_heads,
        n_v_heads: d.value_heads,
        head_dim: 128,
        conv_ld: d.proj_ld(),
        ld: d.proj_ld(),
        z_off: d.conv_dim,
        beta_off: d.conv_dim + d.value_width(),
        alpha_off: d.conv_dim + d.value_width() + d.value_heads,
    })
    .expect("scan sizes");
    let plan = StatePlan::new(&g, 4096, KvFormat::F16).unwrap();
    assert_eq!(
        plan.placement(StateSlot::DeltaRec(0)).unwrap().1,
        scan.state
    );
}

#[test]
fn the_plan_lists_every_state_tensor_once() {
    let (_, g) = real();
    let plan = StatePlan::new(&g, 32_768, KvFormat::F16).unwrap();
    // 36 x (rec, conv) + 12 x (K, V, keys, ring) + PLE ring + words.
    assert_eq!(plan.slots().len(), 36 * 2 + 12 * 4 + 2);
    for (index, &(slot, _)) in plan.slots().iter().enumerate() {
        assert_eq!(plan.index(slot), Some(index), "{slot:?} appears once");
    }
    assert_eq!(plan.arena_tensors().len(), plan.slots().len());
}

#[test]
fn state_tensors_never_overlap_and_are_aligned() {
    let (_, g) = real();
    let plan = StatePlan::new(&g, 32_768, KvFormat::F16).unwrap();
    let placed: Vec<(usize, usize)> = plan
        .slots()
        .iter()
        .map(|&(slot, _)| plan.placement(slot).unwrap())
        .collect();
    for (i, &(offset, bytes)) in placed.iter().enumerate() {
        assert_eq!(offset % 256, 0, "tensor {i}");
        assert!(offset + bytes <= plan.arena_bytes(), "tensor {i}");
        for &(other, other_bytes) in &placed[i + 1..] {
            assert!(
                offset + bytes <= other || other + other_bytes <= offset,
                "tensors at {offset} and {other} overlap"
            );
        }
    }
    // Persistent tensors alias nothing: the arena is exactly their sum.
    let sum: usize = plan.slots().iter().map(|&(_, bytes)| bytes).sum();
    assert_eq!(plan.arena_bytes(), sum);
}

#[test]
fn the_bytes_at_32k_128k_and_256k_match_the_memory_table() {
    let (_, g) = real();
    // (positions, K/V bytes, indexer-key bytes)
    let table = [
        (32_768, 768 * MIB, 48 * MIB),
        (131_072, 3 * GIB, 192 * MIB),
        (262_144, 6 * GIB, 384 * MIB),
    ];
    for (positions, kv, keys) in table {
        let bytes = StatePlan::new(&g, positions, KvFormat::F16)
            .unwrap()
            .bytes()
            .unwrap();
        // The plan: 24 KiB of K/V per token (6.0 GiB at 262,144), 0.38 GiB of indexer keys.
        assert_eq!(bytes.kv_bytes, positions * 24 * KIB, "K/V at {positions}");
        assert_eq!(bytes.kv_bytes, kv);
        assert_eq!(bytes.index_key_bytes, keys, "keys at {positions}");
        assert_eq!(bytes.cap_blocks, positions / 4);
        // The context budget's K/V + indexer bytes per position (F16): 26,112.
        assert_eq!(bytes.bytes_per_position, 26_112);
        assert_eq!(bytes.context_bytes(), positions * bytes.bytes_per_position);
        assert_eq!(
            bytes.arena_bytes,
            bytes.fixed_bytes() + bytes.context_bytes(),
            "the arena is the fixed part plus the context part at {positions}"
        );
    }
    let at_256k = StatePlan::new(&g, 262_144, KvFormat::F16)
        .unwrap()
        .bytes()
        .unwrap();
    assert!((at_256k.index_key_bytes as f64 / GIB as f64 - 0.38).abs() < 0.01);
}

#[test]
fn the_fixed_state_is_the_same_at_every_context() {
    let (_, g) = real();
    let small = StatePlan::new(&g, 4096, KvFormat::F16)
        .unwrap()
        .bytes()
        .unwrap();
    let large = StatePlan::new(&g, 262_144, KvFormat::F16)
        .unwrap()
        .bytes()
        .unwrap();
    assert_eq!(small.fixed_bytes(), large.fixed_bytes());
    // 12 raw-key rings of 4 x 128 f32, the PLE ring, the words block.
    assert_eq!(small.other_fixed_bytes, 12 * 2048 + 368_640 + WORDS_BYTES);
}

#[test]
fn a_context_that_is_not_a_multiple_of_the_ratio_rounds_the_key_blocks_up() {
    let (_, g) = real();
    let bytes = StatePlan::new(&g, 4097, KvFormat::F16)
        .unwrap()
        .bytes()
        .unwrap();
    assert_eq!(bytes.cap_blocks, 1025);
    assert_eq!(bytes.index_key_bytes, 12 * 1025 * 128 * 4);
}

#[test]
fn the_words_block_holds_the_step_words_apart() {
    assert_eq!(WORD_RESULT_PAIR % 8, 0);
    const { assert!(WORD_PREFILL_STATUS + 4 <= WORDS_BYTES) };
    assert_eq!(
        KvFormat::F16.element_bytes(),
        2,
        "the QSA attention reads fp16 caches"
    );
}

#[test]
fn a_shape_the_kernels_do_not_take_is_refused() {
    let (shape, mut kinds) = model_shape(CONFIG).unwrap();
    // The PLE layer must be a layer of the model.
    kinds.truncate(1);
    let error = FlashGeometry::new(&shape, &kinds).expect_err("PLE layer outside");
    assert!(matches!(error, EngError::BadRequest(_)), "{error:?}");
    let (_, g) = real();
    for positions in [0usize, usize::MAX] {
        let error = StatePlan::new(&g, positions, KvFormat::F16).expect_err("context outside");
        assert!(matches!(error, EngError::BadRequest(_)), "{error:?}");
    }
}

#[test]
fn the_book_has_the_capacity_of_the_context() {
    let (_, g) = real();
    let plan = StatePlan::new(&g, 32_768, KvFormat::F16).unwrap();
    let mut book = plan.new_book(7);
    assert_eq!(book.health(), Health::Poisoned);
    book.finish_overwrite(0);
    book.params_written();
    assert_eq!(book.health(), Health::Valid);
    // Fill the context one token at a time in two big jumps, then one more is refused.
    book.finish_overwrite(32_767);
    book.params_written();
    let admitted = book.enqueue(1).expect("the last position");
    assert_eq!(admitted.admit_pos, 32_767);
    book.commit(1, 0).expect("commit");
    assert_eq!(book.tokens(), 32_768);
    let error = book.enqueue(1).expect_err("the context is full");
    assert!(matches!(error, EngError::BadRequest(_)), "{error:?}");
}

#[test]
fn the_kv_bytes_come_from_the_context_budgets_geometry() {
    let (_, g) = real();
    let geometry = g.qsa.kv_geometry().unwrap();
    // The geometry the context budget is built with for this model (its own test pins 26,112).
    assert_eq!(
        geometry,
        KvGeometry {
            cached_layers: 12,
            heads_kv: 2,
            head_dim: 256,
            idx_dim: 128,
            ratio: 4,
        }
    );
    assert_eq!(geometry.position_bytes(KvFormat::F16).unwrap(), 26_112);
    for positions in [32_768usize, 262_144] {
        let bytes = StatePlan::new(&g, positions, KvFormat::F16)
            .unwrap()
            .bytes()
            .unwrap();
        assert_eq!(bytes.bytes_per_position, 26_112);
        assert_eq!(bytes.context_bytes(), positions * 26_112);
    }
}

#[test]
fn a_format_without_a_kernel_is_refused() {
    let (_, g) = real();
    assert!(!KvFormat::Int8.has_kernels());
    let error = StatePlan::new(&g, 32_768, KvFormat::Int8).expect_err("Int8 has no QSA kernel");
    assert!(matches!(error, EngError::BadRequest(_)), "{error:?}");
}

/// The shape the config gives after `edit` of its `text_config`.
fn shape_after(edit: impl Fn(&mut serde_json::Value)) -> Result<FlashGeometry, EngError> {
    let mut config: serde_json::Value = serde_json::from_slice(CONFIG).unwrap();
    edit(&mut config["text_config"]);
    let (shape, kinds) = model_shape(&serde_json::to_vec(&config).unwrap())
        .expect("the edited config is still a valid model shape");
    FlashGeometry::new(&shape, &kinds)
}

#[test]
fn shapes_the_kernels_refuse_fail_at_plan_time() {
    use serde_json::json;
    let cases: [(&str, serde_json::Value, &str); 8] = [
        (
            "linear_key_head_dim",
            json!(64),
            "DeltaNet scan takes heads",
        ),
        (
            "linear_value_head_dim",
            json!(64),
            "DeltaNet scan takes heads",
        ),
        ("linear_conv_kernel_dim", json!(3), "conv takes a kernel"),
        ("head_dim", json!(128), "QSA attention takes heads"),
        ("indexer_compress_ratio", json!(3), "does not divide"),
        ("indexer_compress_ratio", json!(128), "ratio of 1 to 64"),
        ("indexer_head_dim", json!(100), "multiple of 32"),
        ("indexer_n_heads", json!(128), "query values"),
    ];
    for (field, value, needle) in cases {
        let error = shape_after(|text| text[field] = value.clone())
            .expect_err(&format!("{field} = {value} must be refused"));
        let EngError::BadRequest(message) = &error else {
            panic!("{field}: {error:?}");
        };
        assert!(message.contains(needle), "{field}: {message}");
    }
    // The unedited config is accepted.
    assert!(shape_after(|_| {}).is_ok());
}

#[test]
fn the_qsa_state_matches_the_kernel_sizes_and_covers_every_position() {
    let (_, g) = real();
    let layer = 3;
    let plan = StatePlan::new(&g, 4096, KvFormat::F16).unwrap();
    let bytes = |slot| plan.placement(slot).unwrap().1;
    // The fp16 cache of `attn_prep` / `qsa_attn`: rows x kv heads x 256 x 2.
    assert_eq!(bytes(StateSlot::QsaK(layer)), cache_bytes(4096, 2));
    assert_eq!(bytes(StateSlot::QsaV(layer)), cache_bytes(4096, 2));
    assert_eq!(bytes(StateSlot::QsaK(layer)), 4_194_304);
    // Keys f32 [cap_blocks][idx_dim] and the ring f32 [ratio][idx_dim] (`qsa_index`).
    assert_eq!(bytes(StateSlot::QsaIndexKeys(layer)), 1024 * 128 * 4);
    assert_eq!(bytes(StateSlot::QsaRawRing(layer)), 4 * 128 * 4);
    // The index kernel admits `position + m <= cap_blocks * ratio`: every context is covered, with
    // fewer than one spare block.
    for positions in [1usize, 3, 4, 5, 4097, 4100, 32_768, 32_769] {
        let blocks = StatePlan::new(&g, positions, KvFormat::F16)
            .unwrap()
            .bytes()
            .unwrap()
            .cap_blocks;
        assert!(blocks * 4 >= positions, "{positions}");
        assert!((blocks - 1) * 4 < positions, "{positions}");
    }
}

#[test]
fn the_words_block_is_the_qwen36_contract() {
    assert_eq!(WORD_RESULT_PAIR, Word::ResultPair.offset());
    assert_eq!(WORD_PREFILL_STATUS, Word::PrefillStatus.offset());
    assert_eq!(super::state::WORD_STEP_PARAMS, Word::StepParams.offset());
    assert_eq!(WORDS_BYTES, 768);
    assert!(Word::StepParams.bytes() <= 256 && Word::ResultPair.bytes() <= 256);
}
