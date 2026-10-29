//! Unit tests of the chunk gate and the `feed_rows` routing of the CUDA executor. The step itself
//! needs the CUDA device and the pinned 20 GB model, so the gate is pinned on its pure pieces, the
//! routing on its one switch, the per-token identity on the shared `DecodeSession` over host
//! steppers, and the device-only wiring on the merged source (the `start_pins` style).

use super::common_tests;
use super::dims::KvMode;
use super::executor::{ChunkPath, chunk_path_of, feed_per_token, piece_of};
use super::layer_table::LayerTable;
use super::mixer_geometry::MixerGeometry;
use super::start_pins::{pinned, section_of};
use super::step::{CHUNK_MOE_ROLES, ChunkFacts, ChunkVerdict, any_chunk_weight_q6k, chunk_verdict};
use crate::EngError;
use crate::family::gguf_model::Role;
use crate::family::project::cuda::WeightLayout;
use crate::family::session::{DecodeSession, TokenStepper};
use crate::kernels::cuda::resident::matvec_q::QuantType;

type Result<T> = std::result::Result<T, EngError>;

/// The context limit of the session tests.
const LIMIT: usize = 32_768;

/// The facts of a session that passes every chunk-gate condition.
fn ready() -> ChunkFacts {
    ChunkFacts {
        layout: WeightLayout::Native,
        kv_mode: KvMode::Fp16,
        kv8_chunked_prefill: false,
        tiered: false,
        mmq_q6k: false,
        m_sess_cap: 2048,
        arena: Some(2048),
    }
}

#[test]
fn the_gate_accepts_a_native_fp16_plain_session_and_carries_the_admitted_m() {
    assert_eq!(
        chunk_verdict(&ready()),
        ChunkVerdict::Ready { m_sess: 2048 },
        "the accept case"
    );
    // The ADMITTED size rides the accept verdict, never the cap: a session whose arena admitted 512
    // of a 2,048 cap reports 512.
    let admitted = ChunkFacts {
        arena: Some(512),
        ..ready()
    };
    assert_eq!(
        chunk_verdict(&admitted),
        ChunkVerdict::Ready { m_sess: 512 },
        "the admitted M rides the accept verdict, not the cap"
    );
}

#[test]
fn the_gate_refuses_each_condition_with_its_typed_reason() {
    // One row per condition: the Tiled layout, the KV8 cache (v1 has no KV8 prefill), the tiered
    // session, and the Q6_K MMQ weight of the chunk.
    let tiled = ChunkFacts {
        layout: WeightLayout::Tiled,
        ..ready()
    };
    assert_eq!(chunk_verdict(&tiled), ChunkVerdict::TiledLayout);
    let kv8 = ChunkFacts {
        kv_mode: KvMode::Int8G32,
        ..ready()
    };
    assert_eq!(chunk_verdict(&kv8), ChunkVerdict::Kv8);
    let tiered = ChunkFacts {
        tiered: true,
        ..ready()
    };
    assert_eq!(chunk_verdict(&tiered), ChunkVerdict::Tiered);
    let q6k = ChunkFacts {
        mmq_q6k: true,
        ..ready()
    };
    assert_eq!(chunk_verdict(&q6k), ChunkVerdict::Q6KWeight);
}

#[test]
fn the_gate_reports_the_earliest_refusal() {
    // The refusal order: Native, then Fp16, then tiered, then the Q6_K scan.
    let tiled_and_kv8 = ChunkFacts {
        layout: WeightLayout::Tiled,
        kv_mode: KvMode::Int8G32,
        ..ready()
    };
    assert_eq!(chunk_verdict(&tiled_and_kv8), ChunkVerdict::TiledLayout);
    let kv8_and_tiered = ChunkFacts {
        kv_mode: KvMode::Int8G32,
        tiered: true,
        ..ready()
    };
    assert_eq!(chunk_verdict(&kv8_and_tiered), ChunkVerdict::Kv8);
    let tiered_and_q6k = ChunkFacts {
        tiered: true,
        mmq_q6k: true,
        ..ready()
    };
    assert_eq!(chunk_verdict(&tiered_and_q6k), ChunkVerdict::Tiered);
}

#[test]
fn kv8_chunking_is_opt_in_and_keeps_the_other_refusals() {
    let kv8 = ChunkFacts {
        kv_mode: KvMode::Int8G32,
        kv8_chunked_prefill: true,
        m_sess_cap: 512,
        arena: Some(512),
        ..ready()
    };
    assert_eq!(chunk_verdict(&kv8), ChunkVerdict::Ready { m_sess: 512 });
    assert_eq!(
        chunk_path_of(&chunk_verdict(&kv8)),
        ChunkPath::Chunked { m_sess: 512 }
    );
    assert_eq!(piece_of(&chunk_verdict(&kv8), 0), Some(512));
    let small = ChunkFacts {
        m_sess_cap: 16,
        arena: Some(128),
        ..kv8
    };
    assert_eq!(chunk_verdict(&small), ChunkVerdict::Ready { m_sess: 16 });
    assert_eq!(piece_of(&chunk_verdict(&small), 0), Some(16));
    assert_eq!(
        chunk_verdict(&ChunkFacts { arena: None, ..kv8 }),
        ChunkVerdict::NoArena
    );
    assert_eq!(
        chunk_verdict(&ChunkFacts {
            tiered: true,
            ..kv8
        }),
        ChunkVerdict::Tiered
    );
    assert_eq!(
        chunk_verdict(&ChunkFacts {
            mmq_q6k: true,
            ..kv8
        }),
        ChunkVerdict::Q6KWeight
    );
}

#[test]
fn the_chunk_scan_covers_the_routed_stacks_and_the_shared_expert() {
    // The MMQ weights of the chunk beside the dense projections: the three routed expert stacks and
    // the three shared-expert projections. The router (raw f32), the embedding gather and the
    // decode-only head are not scanned.
    assert_eq!(
        CHUNK_MOE_ROLES,
        [
            Role::FfnGateExps,
            Role::FfnUpExps,
            Role::FfnDownExps,
            Role::FfnGateShexp,
            Role::FfnUpShexp,
            Role::FfnDownShexp,
        ]
    );
}

/// The layer table and mixer geometry of the pinned shape (the host fixtures of
/// `common_tests`): the scan's own inputs, so its rows run without a device. The
/// `quant_of` lookup is the only thing a real load provides.
fn scan_fixtures() -> (LayerTable, MixerGeometry) {
    let dims = common_tests::dims();
    let table = LayerTable::new(&dims, &common_tests::inventory()).expect("layer table");
    let geometry = MixerGeometry::new(&dims).expect("geometry");
    (table, geometry)
}

/// The name of `role` in `layer`, looked up once for the closures below.
fn role_name(table: &LayerTable, layer: usize, role: Role) -> String {
    table.name(layer, role).expect("role name").to_string()
}

#[test]
fn the_chunk_scan_finds_a_q6_k_dense_projection() {
    let (table, geometry) = scan_fixtures();
    let kind = table.slot(0).expect("layer 0").kind;
    let role = geometry
        .dense_specs(WeightLayout::Native, kind)
        .expect("specs")[0]
        .role;
    let want = role_name(&table, 0, role);
    let quant_of = move |name: &str| {
        if name == want {
            Some(QuantType::Q6K)
        } else {
            Some(QuantType::Q80)
        }
    };
    assert_eq!(
        any_chunk_weight_q6k(&table, &geometry, quant_of),
        Ok(true),
        "a Q6_K dense projection of layer 0 refuses the chunk path"
    );
}

#[test]
fn the_chunk_scan_finds_a_q6_k_expert_stack() {
    let (table, geometry) = scan_fixtures();
    let want = role_name(&table, 5, Role::FfnDownExps);
    let quant_of = move |name: &str| {
        if name == want {
            Some(QuantType::Q6K)
        } else {
            Some(QuantType::Q4K)
        }
    };
    assert_eq!(
        any_chunk_weight_q6k(&table, &geometry, quant_of),
        Ok(true),
        "a Q6_K routed stack of layer 5 refuses the chunk path"
    );
}

#[test]
fn the_chunk_scan_passes_a_load_whose_only_q6_k_is_the_head() {
    // The pinned file's only Q6_K tensor is `output.weight` (the head), which never meets the
    // chunk's MMQs: the scan must not ask for it, so a load whose only Q6_K is the head accepts.
    let (table, geometry) = scan_fixtures();
    let quant_of = |name: &str| {
        if name == "output.weight" {
            Some(QuantType::Q6K)
        } else {
            Some(QuantType::Q80)
        }
    };
    assert_eq!(any_chunk_weight_q6k(&table, &geometry, quant_of), Ok(false));
}

#[test]
fn absent_moe_stacks_read_as_not_q6_k_under_a_tiered_load() {
    // A tiered load leaves the routed stacks in the file: `None` for the six MoE roles
    // reads as not Q6_K (the `Tiered` verdict refuses such a session anyway), while the
    // dense projections and the shared expert stay loaded.
    let (table, geometry) = scan_fixtures();
    let mut moe_names = Vec::new();
    for layer in 0..table.len() {
        for role in CHUNK_MOE_ROLES {
            moe_names.push(role_name(&table, layer, role));
        }
    }
    let quant_of =
        move |name: &str| (!moe_names.iter().any(|moe| moe == name)).then_some(QuantType::Q4K);
    assert_eq!(any_chunk_weight_q6k(&table, &geometry, quant_of), Ok(false));
}

#[test]
fn an_absent_dense_projection_is_refused_by_name() {
    // Dense projections load in every mode, so a missing one is a `Checkpoint` refusal naming the
    // tensor, never a silent skip that would under-report the gate.
    let (table, geometry) = scan_fixtures();
    let kind = table.slot(0).expect("layer 0").kind;
    let role = geometry
        .dense_specs(WeightLayout::Native, kind)
        .expect("specs")[0]
        .role;
    let absent = role_name(&table, 0, role);
    let missing = absent.clone();
    let quant_of = move |name: &str| (name != missing).then_some(QuantType::Q80);
    match any_chunk_weight_q6k(&table, &geometry, quant_of) {
        Err(EngError::Checkpoint(message)) => {
            assert!(message.contains(&absent), "names the tensor: {message}");
            assert!(
                message.contains("dense projections load in every mode"),
                "states the rule: {message}"
            );
        }
        other => panic!("expected a Checkpoint refusal, got {other:?}"),
    }
}

#[test]
fn the_chunk_path_takes_the_admitted_m_sess() {
    // THE switch: every refusal keeps the per-token arm, and only the accept takes the chunk arm at
    // its admitted `m_sess`.
    for verdict in [
        ChunkVerdict::TiledLayout,
        ChunkVerdict::Kv8,
        ChunkVerdict::Tiered,
        ChunkVerdict::Q6KWeight,
        ChunkVerdict::NoArena,
        ChunkVerdict::Ready { m_sess: 128 },
        ChunkVerdict::Ready { m_sess: 2048 },
    ] {
        let want = match verdict {
            ChunkVerdict::Ready { m_sess } => ChunkPath::Chunked { m_sess },
            _ => ChunkPath::PerToken,
        };
        assert_eq!(chunk_path_of(&verdict), want, "{verdict:?}");
    }
}

/// A host stepper with the CUDA override's own `feed_rows` wiring: a settable verdict,
/// routed through the real `chunk_path_of`, its arm through the real `feed_per_token`.
/// Everything but the two match lines is the trait's own surface.
struct Routed {
    fed: Vec<u32>,
    chunked: Vec<(usize, usize)>,
    verdict: ChunkVerdict,
}

impl TokenStepper for Routed {
    const OPERATION: &'static str = "routed";

    fn feed(&mut self, token: u32) -> Result<()> {
        self.fed.push(token);
        Ok(())
    }

    fn feed_rows(&mut self, tokens: &[u32]) -> Result<()> {
        match chunk_path_of(&self.verdict) {
            ChunkPath::PerToken => feed_per_token(self, tokens),
            ChunkPath::Chunked { m_sess } => {
                self.chunked.push((m_sess, tokens.len()));
                Ok(())
            }
        }
    }

    fn logits(&mut self, _token: u32) -> Result<Vec<f32>> {
        Ok(Vec::new())
    }

    fn fed(&self) -> usize {
        self.fed.len()
    }

    fn reset(&mut self) -> Result<()> {
        self.fed.clear();
        Ok(())
    }
}

/// A stepper that keeps the trait's default `feed_rows`: the loop every session runs
/// today.
struct DefaultLoop(Vec<u32>);

impl TokenStepper for DefaultLoop {
    const OPERATION: &'static str = "default";

    fn feed(&mut self, token: u32) -> Result<()> {
        self.0.push(token);
        Ok(())
    }

    fn logits(&mut self, _token: u32) -> Result<Vec<f32>> {
        Ok(Vec::new())
    }

    fn fed(&self) -> usize {
        self.0.len()
    }

    fn reset(&mut self) -> Result<()> {
        self.0.clear();
        Ok(())
    }
}

#[test]
fn feed_rows_feeds_token_for_token_as_the_default_loop_for_every_verdict() {
    // `QwenResidentStep` itself needs the CUDA device and the pinned 20 GB model, and the
    // tree's host-side fake steppers (the `Fake` and `Plain` of the `family::session`
    // tests) are private to that module, so the override is pinned at the lowest level
    // available: the shared `DecodeSession` driving two host steppers — one with the
    // override's exact routing, one with the trait default — over the same prompt call
    // sequences, for every verdict the gate can return.
    // The refusals only: an admitted verdict now takes the chunk arm, whose rows are the
    // chunk driver's own rows and not one token per step.
    let verdicts = [
        ChunkVerdict::TiledLayout,
        ChunkVerdict::Kv8,
        ChunkVerdict::Tiered,
        ChunkVerdict::Q6KWeight,
        ChunkVerdict::NoArena,
    ];
    let sequences: Vec<Vec<Vec<u32>>> = vec![
        vec![vec![1, 2, 3]],
        vec![vec![7]],
        vec![vec![1, 2], vec![3, 4, 5]],
        // A one-token prompt leaves nothing for `feed_rows`; the next chunk's rows carry
        // its pending token first (the session's own rule).
        vec![vec![7], vec![1, 2], vec![3, 4, 5]],
        vec![(0..100).collect()],
    ];
    for verdict in verdicts {
        for calls in &sequences {
            let mut routed = Routed {
                fed: Vec::new(),
                chunked: Vec::new(),
                verdict,
            };
            let mut routed_session = DecodeSession::default();
            let mut plain = DefaultLoop(Vec::new());
            let mut plain_session = DecodeSession::default();
            for toks in calls {
                routed_session.prefill(&mut routed, toks, LIMIT).unwrap();
                plain_session.prefill(&mut plain, toks, LIMIT).unwrap();
            }
            assert_eq!(routed.fed, plain.0, "{verdict:?}: fed tokens");
            assert_eq!(
                routed_session.pending(),
                plain_session.pending(),
                "{verdict:?}: pending token"
            );
            assert_eq!(
                routed_session.history(),
                plain_session.history(),
                "{verdict:?}: history"
            );
        }
    }
}

#[test]
fn c1_cuda_executor_emits_the_job_local_prefix_report() {
    let body = pinned(section_of(
        include_str!("executor.rs"),
        "executor.rs",
        "fn prefix_report(",
    ));
    assert!(
        body.contains("self.last_prefill_report().prefix_lookup"),
        "{body}"
    );
    assert!(body.contains("self.prefix.entries_len()"), "{body}");
    assert!(body.contains("self.prefix.roots_len()"), "{body}");
    assert!(body.contains("Some(PrefixReport{"), "{body}");
}

#[test]
fn c1_lookup_wiring_observes_success_after_import_and_slot_return() {
    let body = pinned(section_of(
        include_str!("executor.rs"),
        "executor.rs",
        "fn restore_prefix(",
    ));
    let flag = body
        .find("self.prefix.set_shadow_enabled(shadow_block_lcp_enabled());")
        .unwrap();
    let restore = body.find("letcovered=st.restore_prefix(").unwrap();
    let returned = body.find("self.prefix=slot;").unwrap();
    let observe = body.find("self.record_prefix_lookup(prompt,n);").unwrap();
    assert!(
        flag < restore && restore < returned && returned < observe,
        "{body}"
    );
    assert!(
        body.contains("ifletOk(n)=covered{self.record_prefix_lookup(prompt,n);"),
        "{body}"
    );
    assert!(
        body.contains("}covered}"),
        "original restore error must propagate: {body}"
    );
}

#[test]
fn c2_staging_preserves_store_and_reporting_is_read_only() {
    let stage = pinned(section_of(
        include_str!("prefix_runtime.rs"),
        "prefix_runtime.rs",
        "fn stage_pinned_prefix(",
    ));
    assert!(!stage.contains("self.prefix="), "{stage}");
    assert!(
        !stage.contains("self.prefix.set_shadow_enabled("),
        "{stage}"
    );
    assert!(
        stage.contains("ifkind==SaveKind::Tip{self.prefix.take_tip_if(key,&path,"),
        "{stage}"
    );
    assert!(
        stage.contains("pool.can_recycle(&self.stream,&self.state,saved)"),
        "{stage}"
    );
    assert!(
        stage.contains("pool.begin(self.context,&self.stream,&self.state,recycled)"),
        "{stage}"
    );
    let report = pinned(section_of(
        include_str!("executor.rs"),
        "executor.rs",
        "fn prefix_report(",
    ));
    assert!(
        report.contains("entries_len().saturating_add(self.prefix.roots_len())"),
        "{report}"
    );
    assert!(report.contains(".unwrap_or(u32::MAX)"), "{report}");
    assert!(report.contains("bytes:0,"), "{report}");
    assert!(
        !report.contains("complete_prefix_save("),
        "report must not publish pending saves"
    );
}

#[test]
fn feed_rows_routes_through_the_chunk_gate() {
    // The step cannot run on the CPU, so the wiring is pinned on the merged source through the
    // `start_pins` matcher (comments stripped, whitespace collapsed, so a comment or a dead
    // spelling cannot satisfy a pin): the override routes every batch through `chunk_ready` and
    // takes its arm through the per-token loop.
    let body = pinned(section_of(
        include_str!("executor.rs"),
        "executor.rs",
        "fn feed_rows(",
    ));
    assert!(body.contains("chunk_path_of(&verdict)"), "{body}");
    assert!(body.contains("self.chunk_ready()"), "{body}");
    // The per-token arm is written inline and records one Feed step per token:
    // the pin names the recording, not a helper name.
    assert!(body.contains("for&tokenintokens"), "{body}");
    assert!(body.contains("self.record_feed("), "{body}");
    assert!(
        body.contains("self.feed_rows_chunked(tokens,m_sess)"),
        "{body}"
    );
    assert!(
        body.contains("note_feed_rows(&verdict,tokens.len())"),
        "{body}"
    );
}

#[test]
fn the_step_builds_its_chunk_facts_and_the_start_hands_the_cap_over() {
    // The build-time facts and the serve wiring, pinned the same way: `new` runs the Q6_K
    // scan, `chunk_ready` reads the stored facts, and `make` opens KV8 only by opt-in
    // before handing over the 512-row cap. The old default remains unchanged.
    let step = include_str!("step.rs");
    let new = pinned(section_of(step, "step.rs", "pub(crate) fn new("));
    assert!(new.contains("any_chunk_weight_q6k("), "{new}");
    let ready = pinned(section_of(step, "step.rs", "fn chunk_ready("));
    assert!(
        ready.contains("chunk_verdict(&ChunkFacts{") && ready.contains("mmq_q6k"),
        "{ready}"
    );
    let make = pinned(section_of(include_str!("start.rs"), "start.rs", "fn make("));
    let built = make
        .find("QwenResidentStep::new(")
        .expect("make builds the step");
    let opt_in = make
        .find("set_kv8_chunked_prefill(kv8_chunked)?;")
        .expect("make applies the explicit KV8 opt-in");
    let cap = make
        .find("set_m_sess_cap(ifkv8_chunked{modes.chunk_rows}else{DEFAULT_CUDA_PREFILL_CHUNK});")
        .expect("make hands the opted-in cap over");
    assert!(
        built < opt_in && opt_in < cap,
        "step {built}, opt-in {opt_in}, cap {cap}"
    );
}

#[test]
fn a_commented_out_cap_call_does_not_satisfy_the_pin() {
    // The pins strip comments through the `start_pins` matcher, so commenting out the cap handover
    // must FAIL the pin. The row runs the matcher on a doctored `make`: the commented call is gone
    // after stripping, the live call stays.
    let doctored = "fn make(rules: QwenRules) -> Result<()> {
    let mut step = QwenResidentStep::new(context, res, weights, options, rules)?;
    // step.set_m_sess_cap(DEFAULT_CUDA_PREFILL_CHUNK);
    Ok(step)
}
fn tail() {}";
    assert!(
        !pinned(doctored).contains("set_m_sess_cap(DEFAULT_CUDA_PREFILL_CHUNK);"),
        "the commented-out call must not survive the matcher"
    );
    let live = "fn make(rules: QwenRules) -> Result<()> {
    let mut step = QwenResidentStep::new(context, res, weights, options, rules)?;
    step.set_m_sess_cap(DEFAULT_CUDA_PREFILL_CHUNK);
    Ok(step)
}
fn tail() {}";
    let stripped = pinned(live);
    let cap = stripped
        .find("set_m_sess_cap(DEFAULT_CUDA_PREFILL_CHUNK);")
        .expect("the live call survives the matcher");
    let built = stripped
        .find("QwenResidentStep::new(")
        .expect("the step is built");
    assert!(built < cap, "step {built}, cap {cap}");
}

#[test]
fn default_worker_piece_caps_measured_depth_without_shrinking_the_arena() {
    let arena = ChunkVerdict::Ready { m_sess: 1024 };
    for (p0, want) in [
        (0, 1024),
        (72_704, 1024),
        (72_705, 512),
        (73_728, 512),
        (127_000, 512),
    ] {
        assert_eq!(piece_of(&arena, p0), Some(want), "depth {p0}");
    }
    assert_eq!(arena, ChunkVerdict::Ready { m_sess: 1024 });
}

#[test]
fn default_worker_regrouping_preserves_the_cuda_chunk_plan() {
    use super::prefill_plan::chunk_list;

    let arena = ChunkVerdict::Ready { m_sess: 1024 };
    for p0 in [65_536, 72_704, 72_705, 73_727, 73_728, 81_920, 119_321] {
        let end = p0 + 4097;
        let mut at = p0;
        let mut grouped = Vec::new();
        while at < end {
            let rows = piece_of(&arena, at).unwrap().min(end - at);
            grouped.extend(chunk_list(at, rows, 1024).unwrap());
            at += rows;
        }
        assert_eq!(grouped, chunk_list(p0, 4097, 1024).unwrap(), "depth {p0}");
    }
}

/// The initial worker piece is the admitted M_sess, and every refusal answers None so the caller's
/// fallback stands.
#[test]
fn the_worker_piece_follows_the_verdict() {
    for m_sess in [2048usize, 1024, 512, 256, 128] {
        assert_eq!(
            piece_of(&ChunkVerdict::Ready { m_sess }, 0),
            Some(m_sess),
            "the admitted size is the piece"
        );
    }
    for refusal in [
        ChunkVerdict::TiledLayout,
        ChunkVerdict::Kv8,
        ChunkVerdict::Tiered,
        ChunkVerdict::Q6KWeight,
        ChunkVerdict::NoArena,
    ] {
        assert_eq!(
            piece_of(&refusal, 245_760),
            None,
            "{refusal:?} keeps the fallback"
        );
    }
    let arena = ChunkVerdict::Ready { m_sess: 512 };
    assert_eq!(piece_of(&arena, 155_136), Some(512));
    assert_eq!(piece_of(&arena, 155_137), Some(256));
    assert_eq!(piece_of(&arena, 196_096), Some(256));
    assert_eq!(piece_of(&arena, 196_097), Some(256));
    assert_eq!(piece_of(&arena, 245_760), Some(256));
    for m_sess in [16, 128, 256] {
        assert_eq!(
            piece_of(&ChunkVerdict::Ready { m_sess }, 127_000),
            Some(m_sess)
        );
    }
}
