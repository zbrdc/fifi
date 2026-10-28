//! RED pin for the warm-turn incremental encoder (approved 01a17235):
//! a per-session cache that reuses the previous turn's head render + ids
//! and tokenizes only the appended delta, with byte-exact ids on every
//! turn and a full-encode fallback on ANY doubt.
//!
//! Conditions pinned here (B's approval):
//! (1) EXACT — every warm turn's ids are byte-identical to a full encode;
//!     the corpus test walks each real multi-turn session turn by turn;
//!     the self-check variant cross-checks every call in test builds.
//! (2) bounded memory — `accounted_bytes()` reports the live cache bytes.
//! (3) keyed — a different system opener, tool list, or thinking mode
//!     never reuses the cache.
//! fallbacks share the boundary derivation's eligibility gate and refuse
//! on prefix mismatch, non-added-token seams, and partial added-token
//! heads.

use sha2::{Digest, Sha256};

use super::boundary_derive_tests::{
    TEMPLATE_PLAIN, TOKENIZER_JSON, TOKENIZER_JSON_LSTRIP, fixture_bundle, fixture_bundle_with,
    msgs,
};
use super::bundle::resend_verify_enabled;
use crate::tokenizer::{
    PromptMessage, Role, Thinking, TokenizerBundle, TokenizerPin, ToolCallSpec, ToolSpec,
    WarmTurnCache,
};

/// The real pinned-template bundle, same loader as the boundary corpus
/// test. None (skip) when the model dir is absent — portable hosts run
/// everything else; the gate host sets FIFI_QWEN36_MODEL_DIR or uses the
/// pinu default.
fn qwen_bundle() -> Option<TokenizerBundle> {
    let dir = super::boundary_derive_tests::qwen_model_dir()?;
    let read = |n: &str| std::fs::read(dir.join(n)).unwrap();
    let digest = |b: &[u8]| -> [u8; 32] {
        let mut hasher = Sha256::new();
        hasher.update(b);
        hasher.finalize().into()
    };
    let pin = TokenizerPin {
        tokenizer_json: digest(&read("tokenizer.json")),
        tokenizer_config_json: digest(&read("tokenizer_config.json")),
        chat_template_jinja: digest(&read("chat_template.jinja")),
        generation_config_json: digest(&read("generation_config.json")),
        vocab_size: 248077,
    };
    TokenizerBundle::load(&dir, &pin)
        .expect("real bundle")
        .into()
}

/// One corpus conversation through the production converter.
fn corpus_turn(
    conv: &serde_json::Value,
    upto: usize,
) -> (
    Vec<crate::tokenizer::PromptMessage>,
    Vec<crate::tokenizer::ToolSpec>,
) {
    let body = serde_json::json!({
        "model": "qwen3.6-35b-a3b",
        "max_tokens": 16,
        "system": conv["system"],
        "messages": conv["messages"].as_array().unwrap()[..upto],
        "tools": conv["tools"],
    });
    let request = crate::server::anthropic::parse_anthropic(&serde_json::to_vec(&body).unwrap())
        .expect("production anthropic parse");
    (request.input.messages, request.input.tools)
}

fn corpus() -> serde_json::Value {
    serde_json::from_str(include_str!("testdata/boundary-corpus.json")).expect("corpus")
}

/// Walk every corpus conversation turn by turn on ONE cache: turn k sends
/// messages[0..k] (the pure-append CC pattern); the warm ids must equal a
/// full encode of the SAME messages byte for byte, and the boundaries must
/// equal the three-pass result, on every single turn (B condition 1).
#[test]
fn warm_corpus_turns_are_byte_identical_to_full_encodes() {
    let Some(bundle) = qwen_bundle() else {
        return;
    };
    for conv in corpus()["conversations"].as_array().unwrap() {
        let total = conv["messages"].as_array().unwrap().len();
        let mut cache = WarmTurnCache::new();
        for turn in 1..=total {
            let (messages, tools) = corpus_turn(conv, turn);
            let (warm_ids, warm_bounds) = bundle
                .encode_prompt_warm(&mut cache, &messages, &tools, Thinking::Default)
                .expect("warm encode");
            let reference = bundle
                .encode_prompt_with_boundary(&messages, &tools, Thinking::Default)
                .expect("three-pass reference");
            assert_eq!(
                warm_ids, reference.0,
                "corpus {} turn {turn}/{total}: warm ids must be byte-identical to a full encode",
                conv["name"]
            );
            assert_eq!(
                warm_bounds.tip, reference.1.tip,
                "corpus {} turn {turn} tip",
                conv["name"]
            );
            assert_eq!(
                warm_bounds.root, reference.1.root,
                "corpus {} turn {turn} root",
                conv["name"]
            );
        }
        assert!(
            cache.reuses() >= 1,
            "corpus {} must exercise at least one actual reuse (SF1)",
            conv["name"]
        );
    }
}

/// One tool spec by name (the key binds the whole serialized list).
fn tool_named(name: &str) -> ToolSpec {
    ToolSpec(serde_json::json!({
        "type": "function",
        "function": {
            "name": name,
            "description": "Read a source file.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"]
            }
        }
    }))
}

/// The cache key binds the system opener, the tool list, and the thinking
/// mode — ONE render-neutral variable at a time (Sol round 3 SF1). Each
/// case runs TWO identically seeded caches through the SAME append on
/// TEMPLATE_TURN_BODY_ONLY, where system, tools, and thinking never
/// render: the control (same key) MUST reuse, the probe (one key input
/// different) MUST fall back with byte-identical ids and boundaries to
/// the control — so the refusal can only come from the key check, never
/// from a prefix or empty-delta refusal.
#[test]
fn warm_key_blocks_reuse_one_variable_at_a_time() {
    let (_d, bundle) = fixture_bundle(TEMPLATE_TURN_BODY_ONLY);
    let tools = vec![tool_named("read_file")];
    let turn_one = msgs(&[("system", "You are careful."), ("user", "hello")]);
    let turn_two = msgs(&[
        ("system", "You are careful."),
        ("user", "hello"),
        ("assistant", "hi"),
        ("user", "and again"),
    ]);

    // The control: an identically seeded cache, the SAME append, the same
    // key — it MUST reuse, which proves the append itself is reusable.
    let mut control = WarmTurnCache::new();
    let _ = bundle
        .encode_prompt_warm(&mut control, &turn_one, &tools, Thinking::Default)
        .expect("seed");
    let (control_ids, control_bounds) = bundle
        .encode_prompt_warm(&mut control, &turn_two, &tools, Thinking::Default)
        .expect("control: same key");
    assert_eq!(control.reuses(), 1, "control: the same key must reuse");
    assert_eq!(control.fallbacks(), 1, "control: only the seed fell back");
    let reference = bundle
        .encode_prompt(&turn_two, &tools, Thinking::Default)
        .expect("full");
    assert_eq!(control_ids, reference);
    assert!(
        control_bounds.tip.is_some(),
        "the reused append must carry a proven tip"
    );

    // Each probe: a FRESH, identically seeded cache, the SAME append with
    // exactly one render-neutral key input changed. Ids and boundaries
    // must equal the control's byte for byte (the render is identical),
    // and the probe must NOT reuse — the key check refused.
    let hasty = msgs(&[
        ("system", "You are hasty."),
        ("user", "hello"),
        ("assistant", "hi"),
        ("user", "and again"),
    ]);
    let probes: Vec<(&str, Vec<ToolSpec>, Vec<PromptMessage>, Thinking)> = vec![
        (
            "tools",
            vec![tool_named("search")],
            turn_two.clone(),
            Thinking::Default,
        ),
        ("system", tools.clone(), hasty, Thinking::Default),
        ("thinking", tools.clone(), turn_two.clone(), Thinking::On),
    ];
    for (what, probe_tools, probe_msgs, probe_thinking) in probes {
        let mut probe = WarmTurnCache::new();
        let _ = bundle
            .encode_prompt_warm(&mut probe, &turn_one, &tools, Thinking::Default)
            .expect("seed");
        assert_eq!(probe.reuses(), 0, "{what}: fresh probe starts cold");
        let (ids, bounds) = bundle
            .encode_prompt_warm(&mut probe, &probe_msgs, &probe_tools, probe_thinking)
            .expect("{what} probe");
        assert_eq!(
            ids, control_ids,
            "{what}: render-neutral change must keep the control's ids"
        );
        assert_eq!(bounds.tip, control_bounds.tip, "{what}: same tip");
        assert_eq!(bounds.root, control_bounds.root, "{what}: same root");
        assert_eq!(probe.reuses(), 0, "{what}: a key change must not reuse");
        assert_eq!(probe.fallbacks(), 2, "{what}: the refusal is counted");
    }
}

/// An edited history (an earlier message replaced) breaks the render
/// prefix; the warm path must fall back to a full encode and stay exact.
#[test]
fn warm_falls_back_when_history_is_edited() {
    let Some(bundle) = qwen_bundle() else {
        return;
    };
    let mut cache = WarmTurnCache::new();
    let tools = Vec::new();
    let base = msgs(&[
        ("system", "You are careful."),
        ("user", "hello"),
        ("assistant", "hi"),
        ("user", "keep this part exactly"),
    ]);
    bundle
        .encode_prompt_warm(&mut cache, &base, &tools, Thinking::Default)
        .expect("seed");
    // The edit sits in a LATER message, past the boundary render (the
    // leading system message plus the first user turn): this conversation's
    // own history changed, which is a seam-class miss.
    let edited = msgs(&[
        ("system", "You are careful."),
        ("user", "hello"),
        ("assistant", "hi"),
        ("user", "this part was rewritten"),
    ]);
    let (warm_ids, _) = bundle
        .encode_prompt_warm(&mut cache, &edited, &tools, Thinking::Default)
        .expect("warm edited");
    let full = bundle
        .encode_prompt(&edited, &tools, Thinking::Default)
        .expect("full edited");
    assert_eq!(warm_ids, full, "an edited history must take the full path");
    assert_eq!(
        cache.last_fallback_reason(),
        Some("seam"),
        "an edit past the boundary render is a seam-class miss of this conversation"
    );
}

/// The cheap length-compare's known limit, pinned: an edit INSIDE the first
/// user turn is byte-indistinguishable from a foreign conversation's first
/// turn (same key, divergence before the boundary ends), so it reports
/// `other-conversation` too. Ids stay exact either way.
#[test]
fn warm_cannot_tell_a_first_turn_edit_from_a_foreign_conversation() {
    let Some(bundle) = qwen_bundle() else {
        return;
    };
    let mut cache = WarmTurnCache::new();
    let tools = Vec::new();
    let base = msgs(&[("system", "You are careful."), ("user", "hello")]);
    bundle
        .encode_prompt_warm(&mut cache, &base, &tools, Thinking::Default)
        .expect("seed");
    let edited_first_turn = msgs(&[("system", "You are careful."), ("user", "goodbye")]);
    let (warm_ids, _) = bundle
        .encode_prompt_warm(&mut cache, &edited_first_turn, &tools, Thinking::Default)
        .expect("warm edited first turn");
    let full = bundle
        .encode_prompt(&edited_first_turn, &tools, Thinking::Default)
        .expect("full edited first turn");
    assert_eq!(warm_ids, full);
    assert_eq!(
        cache.last_fallback_reason(),
        Some("other-conversation"),
        "a divergence inside the boundary render reports the foreign-conversation cause"
    );
}

/// Sol MF (root boundary, revival round): after a system-only seed the
/// cached root is None (the seed's root equals its tip, so the derivation
/// filter rejects it), but the first warm append's prompt DOES derive a
/// root. The warm path must re-derive the root for the appended prompt
/// instead of returning the stale cached None — the derived path is the
/// oracle, and ids stay exact either way.
#[test]
fn warm_re_derives_the_root_after_a_system_only_seed() {
    let (_d, bundle) = fixture_bundle(TEMPLATE_TURN);
    let mut cache = WarmTurnCache::new();
    let tools = Vec::new();
    let seed = msgs(&[("system", "You are careful.")]);
    let _ = bundle
        .encode_prompt_warm(&mut cache, &seed, &tools, Thinking::Default)
        .expect("seed");
    let with_turn = msgs(&[("system", "You are careful."), ("user", "hello")]);
    let (_, warm_bounds) = bundle
        .encode_prompt_warm(&mut cache, &with_turn, &tools, Thinking::Default)
        .expect("warm hit");
    let (_, derived_bounds) = bundle
        .encode_prompt_with_boundary(&with_turn, &tools, Thinking::Default)
        .expect("derived");
    assert!(
        derived_bounds.root.is_some(),
        "the fixture must derive a root here, else this test pins nothing"
    );
    assert_eq!(
        warm_bounds.root, derived_bounds.root,
        "the warm root is re-derived for the appended prompt, not the seed's None"
    );
}

/// Sol-6 MF: a SYSTEM-ONLY seed (allowed by parse_messages and the plain
/// templates, though the pinned Qwen template rejects it). The seed's
/// boundary covers its whole head with no turn; the first warm hit that
/// appends the turn must re-derive the boundary from the renders it already
/// has (no hit-path render), so the THIRD request — a foreign first turn —
/// still labels other-conversation, not seam. Ids exact throughout.
#[test]
fn warm_updates_the_boundary_after_a_system_only_seed_appends_a_turn() {
    let (_d, bundle) = fixture_bundle(TEMPLATE_TURN);
    let mut cache = WarmTurnCache::new();
    let tools = Vec::new();
    // 1. A system-only seed: fallback, and the boundary covers the whole
    // (system-only) head.
    let seed = msgs(&[("system", "You are careful.")]);
    let (seed_ids, _) = bundle
        .encode_prompt_warm(&mut cache, &seed, &tools, Thinking::Default)
        .expect("seed");
    assert_eq!(
        seed_ids,
        bundle
            .encode_prompt(&seed, &tools, Thinking::Default)
            .expect("full seed")
    );
    // 2. The first warm hit appends the turn: reuse, and the boundary is
    // re-derived (the new head IS system + first turn).
    let with_turn = msgs(&[("system", "You are careful."), ("user", "hello")]);
    let (hit_ids, _) = bundle
        .encode_prompt_warm(&mut cache, &with_turn, &tools, Thinking::Default)
        .expect("warm hit");
    assert_eq!(
        hit_ids,
        bundle
            .encode_prompt(&with_turn, &tools, Thinking::Default)
            .expect("full hit")
    );
    assert_eq!(cache.reuses(), 1, "the appended turn reuses");
    // 3. A foreign first turn (same system): diverges inside the boundary
    // that step 2 re-derived, so it is another conversation, not a seam.
    let foreign = msgs(&[("system", "You are careful."), ("user", "unrelated")]);
    let (warm_ids, _) = bundle
        .encode_prompt_warm(&mut cache, &foreign, &tools, Thinking::Default)
        .expect("warm foreign");
    assert_eq!(
        warm_ids,
        bundle
            .encode_prompt(&foreign, &tools, Thinking::Default)
            .expect("full foreign")
    );
    assert_eq!(
        cache.last_fallback_reason(),
        Some("other-conversation"),
        "the re-derived boundary classifies the foreign turn"
    );
}

/// Opus MF2a: after a system-only seed, the boundary's derivability still
/// depends on history. A head that grew PAST the first turn in one warm
/// hit leaves the boundary render underivable from that hit's renders —
/// the next divergence must report "unresolved", never a guessed
/// "seam". The direct-seed control (same middle request, seeded fresh)
/// DOES derive the boundary and reports other-conversation: the pair
/// documents the history-dependence honestly instead of hiding it.
#[test]
fn warm_names_an_unresolved_boundary_after_a_head_grew_past_the_first_turn() {
    let (_d, bundle) = fixture_bundle(TEMPLATE_TURN);
    let tools = Vec::new();
    let middle = msgs(&[
        ("system", "You are careful."),
        ("user", "hello"),
        ("assistant", "hi"),
        ("user", "more"),
    ]);
    let foreign = msgs(&[("system", "You are careful."), ("user", "unrelated")]);
    // History (a): system-only seed, then one warm append that grows the
    // head past the first turn.
    let mut cache = WarmTurnCache::new();
    bundle
        .encode_prompt_warm(
            &mut cache,
            &msgs(&[("system", "You are careful.")]),
            &tools,
            Thinking::Default,
        )
        .expect("system-only seed");
    let (hit_ids, _) = bundle
        .encode_prompt_warm(&mut cache, &middle, &tools, Thinking::Default)
        .expect("warm append past the first turn");
    assert_eq!(
        hit_ids,
        bundle
            .encode_prompt(&middle, &tools, Thinking::Default)
            .expect("full middle")
    );
    assert_eq!(cache.reuses(), 1);
    let (warm_ids, _) = bundle
        .encode_prompt_warm(&mut cache, &foreign, &tools, Thinking::Default)
        .expect("warm foreign");
    assert_eq!(
        warm_ids,
        bundle
            .encode_prompt(&foreign, &tools, Thinking::Default)
            .expect("full foreign")
    );
    assert_eq!(
        cache.last_fallback_reason(),
        Some("unresolved"),
        "an underivable boundary reports unresolved, never a guessed seam"
    );
    // Direct-seed control: the same middle request seeded fresh derives
    // the boundary, so the same foreign turn is other-conversation.
    let mut direct = WarmTurnCache::new();
    bundle
        .encode_prompt_warm(&mut direct, &middle, &tools, Thinking::Default)
        .expect("direct seed");
    let (direct_ids, _) = bundle
        .encode_prompt_warm(&mut direct, &foreign, &tools, Thinking::Default)
        .expect("direct foreign");
    assert_eq!(
        direct_ids,
        bundle
            .encode_prompt(&foreign, &tools, Thinking::Default)
            .expect("full foreign again")
    );
    assert_eq!(
        direct.last_fallback_reason(),
        Some("other-conversation"),
        "the control pins what the unresolved label hides: a derivable boundary classifies the same divergence"
    );
}

/// Opus MF2b: the pinned Qwen template demotes non-initial system
/// messages to user turns, so an ALL-SYSTEM head's whole render is
/// boundary region. The endpoint must ADVANCE with every all-system hit
/// — a stale endpoint mislabeled a divergence in the demoted region as
/// seam, while a direct seed of the same history says other-conversation.
#[test]
fn warm_advances_the_boundary_for_all_system_heads() {
    let Some(bundle) = qwen_bundle() else {
        return;
    };
    let tools = Vec::new();
    let two = msgs(&[("system", "You are careful."), ("system", "house rules")]);
    let three = msgs(&[
        ("system", "You are careful."),
        ("system", "house rules"),
        ("system", "corpus note"),
    ]);
    let foreign = msgs(&[
        ("system", "You are careful."),
        ("system", "house rules"),
        ("system", "a different corpus note"),
    ]);
    let mut cache = WarmTurnCache::new();
    bundle
        .encode_prompt_warm(&mut cache, &two, &tools, Thinking::Default)
        .expect("all-system seed");
    let (hit_ids, _) = bundle
        .encode_prompt_warm(&mut cache, &three, &tools, Thinking::Default)
        .expect("all-system append hit");
    assert_eq!(
        hit_ids,
        bundle
            .encode_prompt(&three, &tools, Thinking::Default)
            .expect("full three")
    );
    assert_eq!(cache.reuses(), 1, "the all-system append reuses");
    let (warm_ids, _) = bundle
        .encode_prompt_warm(&mut cache, &foreign, &tools, Thinking::Default)
        .expect("warm foreign third system");
    assert_eq!(
        warm_ids,
        bundle
            .encode_prompt(&foreign, &tools, Thinking::Default)
            .expect("full foreign")
    );
    assert_eq!(
        cache.last_fallback_reason(),
        Some("other-conversation"),
        "the advanced endpoint classifies a divergence inside the demoted region"
    );
    // Direct-seed control: the same middle request seeded fresh reaches
    // the same label — history no longer decides.
    let mut direct = WarmTurnCache::new();
    bundle
        .encode_prompt_warm(&mut direct, &three, &tools, Thinking::Default)
        .expect("direct seed");
    let (direct_ids, _) = bundle
        .encode_prompt_warm(&mut direct, &foreign, &tools, Thinking::Default)
        .expect("direct foreign");
    assert_eq!(
        direct_ids,
        bundle
            .encode_prompt(&foreign, &tools, Thinking::Default)
            .expect("full foreign again")
    );
    assert_eq!(
        direct.last_fallback_reason(),
        Some("other-conversation"),
        "direct and warm-seeded histories agree"
    );
}

/// A resend AFTER A HIT reuses the stored result outright: no render, no
/// encode, no fallback (the replay report's repeat-turn shape). The RED
/// pin: today a resend is refused (`reason=resend`) and pays the derived
/// fallback — ~4 full-conversation renders + a full encode at 128K.
#[test]
fn warm_serves_a_resend_after_a_hit_without_rework() {
    let (_d, bundle) = fixture_bundle(TEMPLATE_TURN);
    let tools = Vec::new();
    let first = msgs(&[("system", "You are careful."), ("user", "hello")]);
    let mut cache = WarmTurnCache::new();
    bundle
        .encode_prompt_warm(&mut cache, &first, &tools, Thinking::Default)
        .expect("seed");
    let appended = msgs(&[
        ("system", "You are careful."),
        ("user", "hello"),
        ("assistant", "hi"),
        ("user", "more"),
    ]);
    let (hit_ids, hit_bounds) = bundle
        .encode_prompt_warm(&mut cache, &appended, &tools, Thinking::Default)
        .expect("appended hit");
    let (fallbacks, reuses) = (cache.fallbacks(), cache.reuses());
    bundle.reset_counts();
    let (resend_ids, resend_bounds) = bundle
        .encode_prompt_warm(&mut cache, &appended, &tools, Thinking::Default)
        .expect("resend");
    assert_eq!(resend_ids, hit_ids, "a resend returns the hit's ids");
    assert_eq!(resend_bounds.tip, hit_bounds.tip);
    assert_eq!(resend_bounds.root, hit_bounds.root);
    assert_eq!(
        (cache.fallbacks(), cache.reuses()),
        (fallbacks, reuses + 1),
        "a resend is a counted reuse, never a fallback"
    );
    // In test builds the resend shortcut's diagnostic cross-check runs
    // ONE full encode (Opus N4); production (self_check off) does none.
    assert_eq!(
        bundle.counts(),
        (1, 1),
        "a resend pays only the test-build cross-check's full encode"
    );
    let full = bundle
        .encode_prompt(&appended, &tools, Thinking::Default)
        .expect("full oracle");
    assert_eq!(resend_ids, full, "byte-for-byte the full encode");
    // A THIRD identical resend still hits: the cache keeps the result.
    bundle.reset_counts();
    let (again_ids, _) = bundle
        .encode_prompt_warm(&mut cache, &appended, &tools, Thinking::Default)
        .expect("second resend");
    assert_eq!(again_ids, hit_ids);
    assert_eq!(bundle.counts(), (1, 1));
}

/// Sol r2 MF (absent vs empty): the pinned Qwen template treats an
/// ABSENT reasoning_content differently from a present empty one, but
/// the old digest hashed both as length-0 + "" — the resend returned
/// another request's ids. Both directions must MISS here, with ids from
/// the fallback matching the full oracle.
#[test]
fn warm_resend_misses_when_only_reasoning_presence_changes() {
    for (first_reasoning, second_reasoning) in [(None, Some("")), (Some(""), None)] {
        let (_d, mut bundle) = fixture_bundle(TEMPLATE_FIELDS);
        // reasoning_content only reaches the template when replay is on
        // (the renderer strips it otherwise), and the absent-vs-empty
        // distinction the digest must see lives in that branch.
        bundle.force_flags_on();
        let tools = Vec::new();
        let mut cache = WarmTurnCache::new();
        bundle
            .encode_prompt_warm(&mut cache, &seed_conv(), &tools, Thinking::Default)
            .expect("seed");
        let hit = appended_with_reasoning(first_reasoning);
        bundle
            .encode_prompt_warm(&mut cache, &hit, &tools, Thinking::Default)
            .expect("append hit");
        assert_eq!(cache.reuses(), 1);
        let miss = appended_with_reasoning(second_reasoning);
        let (miss_ids, _) = bundle
            .encode_prompt_warm(&mut cache, &miss, &tools, Thinking::Default)
            .expect("reasoning-presence change");
        let full_ids = bundle
            .encode_prompt(&miss, &tools, Thinking::Default)
            .expect("oracle");
        let hit_ids = bundle
            .encode_prompt(&hit, &tools, Thinking::Default)
            .expect("hit oracle");
        assert_ne!(
            miss_ids, hit_ids,
            "presence change must not reuse the hit ids"
        );
        assert_eq!(miss_ids, full_ids);
    }
}

/// Sol r2 MF (field boundaries): the old digest concatenated tool-call
/// id + name + arguments with no framing, so shifting text across field
/// boundaries ("call_1"+"Read" == "call_1R"+"ead") hashed identically
/// while the template renders different calls. Both shifts must MISS.
#[test]
fn warm_resend_misses_when_tool_call_field_boundaries_shift() {
    let shifts = [
        (["he", "llo", "{\"a\":1}"], ["hel", "lo", "{\"a\":1}"]),
        (["he", "llo", "{\"a\":1}"], ["hell", "o", "{\"a\":1}"]),
    ];
    for (hit_call, miss_call) in shifts {
        let (_d, bundle) = fixture_bundle(TEMPLATE_FIELDS);
        let tools = Vec::new();
        let mut cache = WarmTurnCache::new();
        bundle
            .encode_prompt_warm(&mut cache, &seed_conv(), &tools, Thinking::Default)
            .expect("seed");
        let hit = appended_with_call(hit_call);
        bundle
            .encode_prompt_warm(&mut cache, &hit, &tools, Thinking::Default)
            .expect("append hit");
        assert_eq!(cache.reuses(), 1);
        let miss = appended_with_call(miss_call);
        let fallbacks = cache.fallbacks();
        let (miss_ids, _) = bundle
            .encode_prompt_warm(&mut cache, &miss, &tools, Thinking::Default)
            .expect("tool-call boundary shift");
        let full_ids = bundle
            .encode_prompt(&miss, &tools, Thinking::Default)
            .expect("oracle");
        let hit_ids = bundle
            .encode_prompt(&hit, &tools, Thinking::Default)
            .expect("hit oracle");
        assert_ne!(
            miss_ids, hit_ids,
            "a field-boundary shift must not reuse the hit ids"
        );
        assert_eq!(miss_ids, full_ids, "shift must fall back byte-exact");
        assert_eq!(cache.fallbacks(), fallbacks + 1, "shift counted once");
    }
}

/// Sol r2 MF3 / Sol-final F1 (effective flags, each pinned alone): two
/// bundles from the SAME pins with different effective rendering share
/// file identity, so only the flag fold in the key keeps their caches
/// apart. The pairs cover replay-only and wrap-only differences, so
/// deleting EITHER flag-hash line in `identity_fold` breaks its case.
#[test]
fn warm_resend_misses_when_effective_flags_differ_on_same_pins() {
    let (_d, mut plain) = fixture_bundle(TEMPLATE_FIELDS);
    // Identical fixture content -> identical pins, so the bundles share
    // file identity by construction; only the effective flags differ.
    plain.force_flags_on(); // (replay on, wrap on)
    let (_d2, mut wrap_off) = fixture_bundle(TEMPLATE_FIELDS);
    wrap_off.force_flags_on();
    wrap_off.disable_notice_wrap(); // (replay on, wrap off)
    let (_d3, mut replay_off) = fixture_bundle(TEMPLATE_FIELDS);
    replay_off.disable_replay(); // (replay off, wrap off)
    let tools = Vec::new();
    // `renders_differ`: the wrap flag only changes rendering when a
    // notice is present, which this conversation has none of — for the
    // wrap-only pair the two bundles' ids are genuinely EQUAL, so the
    // fallback count is the discriminator (a dropped flag-hash would
    // wrongly reuse and add zero fallbacks).
    let pair = |a: &TokenizerBundle, b: &TokenizerBundle, renders_differ: bool| {
        let mut cache = WarmTurnCache::new();
        a.encode_prompt_warm(&mut cache, &seed_conv(), &tools, Thinking::Default)
            .expect("seed");
        let (hit_ids, _) = a
            .encode_prompt_warm(&mut cache, &flag_conv(), &tools, Thinking::Default)
            .expect("append hit");
        assert_eq!(cache.reuses(), 1);
        let fallbacks = cache.fallbacks();
        let (ids, _) = b
            .encode_prompt_warm(&mut cache, &flag_conv(), &tools, Thinking::Default)
            .expect("other configuration");
        let oracle = b
            .encode_prompt(&flag_conv(), &tools, Thinking::Default)
            .expect("other oracle");
        assert_eq!(
            cache.fallbacks(),
            fallbacks + 1,
            "a differently configured bundle is a fallback, never a reuse"
        );
        assert_eq!(ids, oracle);
        if renders_differ {
            assert_ne!(ids, hit_ids);
        }
    };
    // Pair 1 differs in BOTH flags (on,on vs off,off): it proves the
    // general cross-configuration miss but catches no single line.
    // Pair 2 ((on,on) vs (on,off)) differs ONLY in wrap: it catches
    // deleting the notice_toolwrap hash line. Pair 3 ((on,off) vs
    // (off,off)) differs ONLY in replay: it catches deleting the
    // think_replay hash line.
    pair(&plain, &replay_off, true); // both flags differ — general miss
    pair(&plain, &wrap_off, false); // catches: deleting the notice_toolwrap hash line
    pair(&wrap_off, &replay_off, true); // catches: deleting the think_replay hash line
}

/// Every input the render reads must reach the resend digest or the key:
/// each case starts from a FRESH cache, proves a real append HIT
/// (reuses == 1), then changes exactly one input and must MISS with
/// byte-exact fallback ids. TEMPLATE_FIELDS renders reasoning, tool
/// calls, tools, and thinking, so tools/thinking changes are visible to
/// BOTH the key and the render, not just the key.
#[test]
fn warm_resend_misses_when_any_render_input_changes() {
    let case = |name: &str,
                hit_msgs: Vec<PromptMessage>,
                miss_msgs: Vec<PromptMessage>,
                hit_tools: Vec<ToolSpec>,
                miss_tools: Vec<ToolSpec>,
                miss_thinking: Thinking| {
        let (_d, bundle) = fixture_bundle(TEMPLATE_FIELDS);
        let mut cache = WarmTurnCache::new();
        bundle
            .encode_prompt_warm(&mut cache, &seed_conv(), &hit_tools, Thinking::Default)
            .expect("seed");
        bundle
            .encode_prompt_warm(&mut cache, &hit_msgs, &hit_tools, Thinking::Default)
            .expect("append hit");
        assert_eq!(
            cache.reuses(),
            1,
            "{name}: hit state proven before the miss"
        );
        let fallbacks = cache.fallbacks();
        let (miss_ids, _) = bundle
            .encode_prompt_warm(&mut cache, &miss_msgs, &miss_tools, miss_thinking)
            .expect(name);
        assert_eq!(cache.fallbacks(), fallbacks + 1, "{name}: one fallback");
        let full_ids = bundle
            .encode_prompt(&miss_msgs, &miss_tools, miss_thinking)
            .expect("oracle");
        assert_eq!(miss_ids, full_ids, "{name}: fallback byte-exact");
    };
    let empty: Vec<ToolSpec> = Vec::new();
    let base = appended_with_reasoning(None);
    let mut edited = base.clone();
    edited.last_mut().unwrap().content = Some("more!".into());
    case(
        "one-token-different last message",
        base.clone(),
        edited,
        empty.clone(),
        empty.clone(),
        Thinking::Default,
    );
    let mut reordered = base.clone();
    reordered.swap(1, 3);
    case(
        "reordered messages",
        base.clone(),
        reordered,
        empty.clone(),
        empty.clone(),
        Thinking::Default,
    );
    let mut earlier = base.clone();
    earlier[1].content = Some("hello there".into());
    case(
        "edited earlier message",
        base.clone(),
        earlier,
        empty.clone(),
        empty.clone(),
        Thinking::Default,
    );
    let mut system = base.clone();
    system[0].content = Some("You are terse.".into());
    case(
        "changed system",
        base.clone(),
        system,
        empty.clone(),
        empty.clone(),
        Thinking::Default,
    );
    case(
        "changed tools (tool_choice folds here)",
        base.clone(),
        base.clone(),
        empty.clone(),
        vec![tool_named("read_file")],
        Thinking::Default,
    );
    case(
        "flipped thinking",
        base.clone(),
        base.clone(),
        empty.clone(),
        empty.clone(),
        Thinking::On,
    );
    // Render-neutral fields below: ids stay EQUAL, so the fallback count
    // is the discriminator — a digest that dropped the field would wrongly
    // reuse (0 fallbacks) and fail the +1 assert.
    let mut named = base.clone();
    named[2].name = Some("reader".into());
    case(
        "assistant name added",
        base.clone(),
        named,
        empty.clone(),
        empty.clone(),
        Thinking::Default,
    );
    let mut tool_id = base.clone();
    tool_id[2].tool_call_id = Some("call_9".into());
    case(
        "tool_call_id added",
        base.clone(),
        tool_id,
        empty.clone(),
        empty.clone(),
        Thinking::Default,
    );
    case(
        "reasoning text changed",
        appended_with_reasoning(Some("hello")),
        appended_with_reasoning(Some("world")),
        empty.clone(),
        empty.clone(),
        Thinking::Default,
    );
}

/// Sol-6 SF: a work-counter oracle for the warm paths. Render counts
/// (slot 0 of `counts()`): the mid-history re-seed pays the derivation's
/// full + head renders, the re-seed's head render, and the first-turn
/// boundary render (4); an appended HIT pays only its own full + head
/// renders plus the test-build cross-check's two (4) — NO boundary
/// render, pinning the fold's no-extra-render fix; a head that ENDS at
/// the first turn re-seeds without the boundary render (3) — the head
/// render IS the boundary render, pinning that R1 fix.
#[test]
fn warm_counts_renders_on_the_hit_and_reseed_paths() {
    let (_d, bundle) = fixture_bundle(TEMPLATE_TURN);
    let tools = Vec::new();
    let middle = msgs(&[
        ("system", "You are careful."),
        ("user", "hello"),
        ("assistant", "hi"),
        ("user", "more"),
    ]);
    let mut cache = WarmTurnCache::new();
    bundle.reset_counts();
    bundle
        .encode_prompt_warm(&mut cache, &middle, &tools, Thinking::Default)
        .expect("mid-history seed");
    assert_eq!(
        bundle.counts().0,
        4,
        "re-seed past the first turn: derivation full + head, re-seed head, first-turn boundary renders"
    );
    bundle.reset_counts();
    let appended = msgs(&[
        ("system", "You are careful."),
        ("user", "hello"),
        ("assistant", "hi"),
        ("user", "more"),
        ("assistant", "done"),
        ("user", "again"),
    ]);
    let (hit_ids, _) = bundle
        .encode_prompt_warm(&mut cache, &appended, &tools, Thinking::Default)
        .expect("appended hit");
    assert_eq!(
        hit_ids,
        bundle
            .encode_prompt(&appended, &tools, Thinking::Default)
            .expect("full appended")
    );
    assert_eq!(
        bundle.counts().0,
        2 + 2,
        "hit path: its own full + head renders, plus the test-build cross-check's two — no boundary render"
    );
    let mut fresh = WarmTurnCache::new();
    bundle.reset_counts();
    let first_turn = msgs(&[("system", "You are careful."), ("user", "hello")]);
    bundle
        .encode_prompt_warm(&mut fresh, &first_turn, &tools, Thinking::Default)
        .expect("whole-head seed");
    assert_eq!(
        bundle.counts().0,
        3,
        "a head ending at the first turn: the head render IS the boundary render, so no boundary render"
    );
}

/// MF2 (Opus round 1): the agents' traffic is SYSTEM-FIRST, and the warm key
/// already pins the system text — so the boundary must sit past the first
/// NON-system message, or a same-system foreign conversation (exactly the
/// replay report's side-request shape) would classify as a seam miss.
#[test]
fn warm_names_a_system_first_foreign_conversation_not_a_seam_miss() {
    let Some(bundle) = qwen_bundle() else {
        return;
    };
    let mut cache = WarmTurnCache::new();
    let tools = Vec::new();
    let mine = msgs(&[
        ("system", "You are careful."),
        ("user", "hello"),
        ("assistant", "hi"),
    ]);
    bundle
        .encode_prompt_warm(&mut cache, &mine, &tools, Thinking::Default)
        .expect("seed");
    let foreign = msgs(&[
        ("system", "You are careful."),
        ("user", "a completely different turn"),
    ]);
    let (warm_ids, _) = bundle
        .encode_prompt_warm(&mut cache, &foreign, &tools, Thinking::Default)
        .expect("warm foreign");
    let full = bundle
        .encode_prompt(&foreign, &tools, Thinking::Default)
        .expect("full foreign");
    assert_eq!(warm_ids, full, "a foreign conversation takes the full path");
    assert_eq!(
        cache.last_fallback_reason(),
        Some("other-conversation"),
        "a same-system foreign first turn diverges inside the boundary, not at a seam"
    );
}

/// An identical resend (no appended delta) and a truncated history (the new
/// head is a strict prefix of the cached one) are prefix shapes a divergence
/// POSITION cannot name; each gets its own cause. Ids stay exact.
#[test]
fn warm_names_a_resend_and_a_truncated_history() {
    let Some(bundle) = qwen_bundle() else {
        return;
    };
    let mut cache = WarmTurnCache::new();
    let tools = Vec::new();
    let turn = msgs(&[
        ("system", "You are careful."),
        ("user", "hello"),
        ("assistant", "hi"),
        ("user", "more"),
    ]);
    bundle
        .encode_prompt_warm(&mut cache, &turn, &tools, Thinking::Default)
        .expect("seed");
    let (resend_ids, _) = bundle
        .encode_prompt_warm(&mut cache, &turn, &tools, Thinking::Default)
        .expect("resend");
    let full = bundle
        .encode_prompt(&turn, &tools, Thinking::Default)
        .expect("full");
    assert_eq!(resend_ids, full);
    assert_eq!(
        cache.last_fallback_reason(),
        Some("resend"),
        "an identical re-post names its shape"
    );
    let truncated = msgs(&[("system", "You are careful."), ("user", "hello")]);
    let (truncated_ids, _) = bundle
        .encode_prompt_warm(&mut cache, &truncated, &tools, Thinking::Default)
        .expect("truncated");
    let full_truncated = bundle
        .encode_prompt(&truncated, &tools, Thinking::Default)
        .expect("full truncated");
    assert_eq!(truncated_ids, full_truncated);
    assert_eq!(
        cache.last_fallback_reason(),
        Some("truncated"),
        "a head that shrank to a strict prefix names its shape"
    );
}

/// A plain-text template (no added token at the delta start) must refuse
/// the seam and fall back, exactly like the boundary derivation's
/// plain-suffix rule.
#[test]
fn warm_falls_back_when_the_delta_starts_without_an_added_token() {
    let (_d, bundle) = fixture_bundle(TEMPLATE_PLAIN);
    let mut cache = WarmTurnCache::new();
    let tools = Vec::new();
    let turn_one = msgs(&[("system", "You are careful."), ("user", "hello")]);
    bundle
        .encode_prompt_warm(&mut cache, &turn_one, &tools, Thinking::Default)
        .expect("seed");
    let turn_two = msgs(&[
        ("system", "You are careful."),
        ("user", "hello"),
        ("assistant", "hi"),
        ("user", "more"),
    ]);
    let (warm_ids, _) = bundle
        .encode_prompt_warm(&mut cache, &turn_two, &tools, Thinking::Default)
        .expect("warm plain");
    let full = bundle
        .encode_prompt(&turn_two, &tools, Thinking::Default)
        .expect("full plain");
    assert_eq!(warm_ids, full, "a plain-text delta must not be reused");
    assert_eq!(
        cache.last_fallback_reason(),
        Some("seam"),
        "a delta that starts without an added token is a junction refusal"
    );
}

/// An ineligible tokenizer (lstrip added token — the boundary gate's own
/// counterexample) must never take the warm path.
#[test]
fn warm_uses_the_boundary_eligibility_gate() {
    // The turn-separated template: an append on the ELIGIBLE twin of this
    // tokenizer provably reuses (warm_reuses_on_a_pure_appended_turn), so
    // a refusal here is attributable to the eligibility gate alone.
    let (_d, bundle) = fixture_bundle_with(TOKENIZER_JSON_LSTRIP, TEMPLATE_TURN);
    let mut cache = WarmTurnCache::new();
    let tools = Vec::new();
    let turn_one = msgs(&[("system", "You are careful."), ("user", "hello")]);
    let (warm_ids, _) = bundle
        .encode_prompt_warm(&mut cache, &turn_one, &tools, Thinking::Default)
        .expect("warm one");
    let full = bundle
        .encode_prompt(&turn_one, &tools, Thinking::Default)
        .expect("full one");
    assert_eq!(
        warm_ids, full,
        "ineligible tokenizer must take the full path"
    );
    assert_eq!(
        cache.last_fallback_reason(),
        Some("seed"),
        "an empty slot names its seed first, whatever the tokenizer"
    );
    // Warm case (Sol round 3 SF1): the fallback still SEEDS the cache, so
    // an APPENDED turn on the seeded cache must refuse again — the
    // eligibility gate holds warm, not just cold, and the append rules
    // out an empty-delta/prefix explanation.
    let turn_two = msgs(&[
        ("system", "You are careful."),
        ("user", "hello"),
        ("assistant", "hi"),
        ("user", "and again"),
    ]);
    let (warm_ids, _) = bundle
        .encode_prompt_warm(&mut cache, &turn_two, &tools, Thinking::Default)
        .expect("warm two: appended turn, seeded cache");
    let full_two = bundle
        .encode_prompt(&turn_two, &tools, Thinking::Default)
        .expect("full two");
    assert_eq!(warm_ids, full_two, "still exact on the seeded cache");
    assert_eq!(cache.reuses(), 0, "an ineligible tokenizer never reuses");
    assert_eq!(cache.fallbacks(), 2, "each refusal is counted");
    assert_eq!(
        cache.last_fallback_reason(),
        Some("ineligible"),
        "on a seeded slot the refusing gate names itself"
    );
    assert_eq!(
        cache.digest_calls(),
        0,
        "an ineligible tokenizer is never hashed, key match or not (Opus delta5 SF2)"
    );
    // Opus delta SF: "key" beats the gates — on this ineligible fixture,
    // a request whose KEY differs (foreign system opener) must name the
    // key change, not the eligibility gate (or the head-partial scan it
    // never reaches).
    let foreign_key = msgs(&[("system", "You are terse."), ("user", "hello")]);
    let (warm_ids, _) = bundle
        .encode_prompt_warm(&mut cache, &foreign_key, &tools, Thinking::Default)
        .expect("warm foreign key");
    let full_foreign = bundle
        .encode_prompt(&foreign_key, &tools, Thinking::Default)
        .expect("full foreign key");
    assert_eq!(warm_ids, full_foreign);
    assert_eq!(
        cache.last_fallback_reason(),
        Some("key"),
        "a key change outranks the eligibility and head-partial gates"
    );
}

/// Opus delta SF3: when a KEY change and a refusing gate occur together
/// (a foreign conversation whose tool list ALSO differs), the classifier's
/// order decides the name — key before the warm attempt's gates. The pins
/// fail if the order ever flips (a foreign tool-less request would then be
/// miscounted as other-conversation instead of key).
#[test]
fn warm_names_a_key_change_even_when_the_history_is_foreign() {
    let Some(bundle) = qwen_bundle() else {
        return;
    };
    let mut cache = WarmTurnCache::new();
    let tools = vec![tool_named("read_file")];
    let mine = msgs(&[("system", "You are careful."), ("user", "hello")]);
    bundle
        .encode_prompt_warm(&mut cache, &mine, &tools, Thinking::Default)
        .expect("seed");
    // The tools change (a key change) AND the first turn differs (what
    // alone would be other-conversation): key wins.
    let foreign_key = msgs(&[("system", "You are careful."), ("user", "unrelated")]);
    let (warm_ids, _) = bundle
        .encode_prompt_warm(&mut cache, &foreign_key, &Vec::new(), Thinking::Default)
        .expect("warm foreign key");
    let full = bundle
        .encode_prompt(&foreign_key, &Vec::new(), Thinking::Default)
        .expect("full foreign key");
    assert_eq!(warm_ids, full);
    assert_eq!(
        cache.last_fallback_reason(),
        Some("key"),
        "a key change outranks the boundary divergence"
    );
}

/// A divergence that lands INSIDE a multi-byte UTF-8 character: the byte
/// walk stops at the first differing byte, mid-character, and the position
/// classifies exactly like a single-byte one (no panic, no misclassification).
#[test]
fn warm_classifies_a_divergence_inside_a_multibyte_character() {
    let Some(bundle) = qwen_bundle() else {
        return;
    };
    let mut cache = WarmTurnCache::new();
    let tools = Vec::new();
    let mine = msgs(&[("user", "hi \u{1F600} tail")]);
    bundle
        .encode_prompt_warm(&mut cache, &mine, &tools, Thinking::Default)
        .expect("seed");
    let foreign = msgs(&[("user", "hi \u{1F641} other")]);
    let (warm_ids, _) = bundle
        .encode_prompt_warm(&mut cache, &foreign, &tools, Thinking::Default)
        .expect("warm foreign multibyte");
    let full = bundle
        .encode_prompt(&foreign, &tools, Thinking::Default)
        .expect("full foreign multibyte");
    assert_eq!(warm_ids, full);
    assert_eq!(
        cache.last_fallback_reason(),
        Some("other-conversation"),
        "a mid-character divergence still classifies by position"
    );
}

/// A turn-separated template: every message is prefixed with the added
/// token `<|end|>`, so an appended turn's delta and the generation prompt
/// both start with a complete added token — the shape under which the
/// warm path can actually reuse (the real Qwen template has it too).
const TEMPLATE_TURN: &str = concat!(
    r#"{% for m in messages %}<|end|>{{ m.role }}:{{ m.content }}{% endfor %}"#,
    r#"{% if add_generation_prompt %}<|end|>hello{% endif %}"#,
);

/// A turn-separated fixture that renders only non-system message bodies —
/// never the system opener, the tool list, or the thinking mode. Every
/// warm-key input is render-neutral here, so a reuse refusal can only
/// come from the KEY itself, never from a render-prefix mismatch.
const TEMPLATE_TURN_BODY_ONLY: &str = concat!(
    r#"{% for m in messages %}{% if m.role != 'system' %}<|end|>{{ m.role }}:{{ m.content }}{% endif %}{% endfor %}"#,
    r#"{% if add_generation_prompt %}<|end|>hello{% endif %}"#,
);

/// The fixture tokenizer plus the eligible added token `o<|end|>` and the
/// letters the role names need — Opus's MF1 counterexample vocabulary:
/// leftmost-longest merges the "o" at the end of a head ending in "hello"
/// with the `<|end|>` that opens the generation prompt into ONE token.
const TOKENIZER_JSON_SPAN: &str = r#"{"version":"1.0","truncation":null,"padding":null,"added_tokens":[{"id":100,"content":" world","special":false,"single_word":false,"lstrip":false,"rstrip":false,"normalized":false},{"id":101,"content":"<|end|>","special":true,"single_word":false,"lstrip":false,"rstrip":false,"normalized":false},{"id":102,"content":"o<|end|>","special":false,"single_word":false,"lstrip":false,"rstrip":false,"normalized":false}],"normalizer":null,"pre_tokenizer":{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":true},"post_processor":{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":false},"decoder":{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":false},"model":{"type":"BPE","dropout":null,"unk_token":null,"continuing_subword_prefix":null,"end_of_word_suffix":null,"fuse_unk":false,"byte_fallback":false,"ignore_merges":false,"vocab":{"h":0,"e":1,"l":2,"o":3,"Ġ":4,"w":5,"r":6,"d":7,"hello":8,"a":9,"b":10,"he":11,"hel":12,"hell":13,":":14,"u":15,"s":16,"t":17,"m":18,"i":19,"n":20,"y":21,"c":22},"merges":[["h","e"],["he","l"],["hel","l"],["hell","o"]]}}"#;

/// MF1 (Opus): the new head ends inside a proper prefix of the added
/// token `o<|end|>` (it ends in "hello"), the generation prompt starts
/// with the complete `<|end|>`, and every other precondition passes. The
/// suffix proof cannot see the token that spans the join, so the head
/// guard must refuse the seam and the warm path must fall back — with ids
/// exactly equal to the full encode (which merges `o<|end|>`).
#[test]
fn warm_falls_back_when_a_token_spans_the_head_generation_join() {
    let (_d, bundle) = fixture_bundle_with(TOKENIZER_JSON_SPAN, TEMPLATE_TURN);
    let mut cache = WarmTurnCache::new();
    let tools = Vec::new();
    let turn_one = msgs(&[("user", "hi")]);
    bundle
        .encode_prompt_warm(&mut cache, &turn_one, &tools, Thinking::Default)
        .expect("seed");
    let turn_two = msgs(&[("user", "hi"), ("assistant", "ab"), ("user", "hello")]);
    let (warm_ids, _) = bundle
        .encode_prompt_warm(&mut cache, &turn_two, &tools, Thinking::Default)
        .expect("warm with a spanning added token must fall back, not diverge");
    let full = bundle
        .encode_prompt(&turn_two, &tools, Thinking::Default)
        .expect("full");
    assert_eq!(
        warm_ids, full,
        "a token spanning the head/generation join must not be split"
    );
    assert_eq!(
        cache.reuses(),
        0,
        "the spanning-token scenario must be refused, not reused"
    );
    assert_eq!(
        cache.last_fallback_reason(),
        Some("head-partial-token"),
        "the new head ending inside a partial added token names itself"
    );
}

/// SF2: the module doc promises refusal on partial-added-token heads. The
/// head ends in " ", a proper prefix of the added token ` world` — the
/// old-head guard must refuse reuse (counted) and stay exact.
#[test]
fn warm_refuses_a_head_ending_inside_a_partial_added_token() {
    let (_d, bundle) = fixture_bundle(TEMPLATE_TURN);
    let mut cache = WarmTurnCache::new();
    let tools = Vec::new();
    let turn_one = msgs(&[("user", "hello ")]);
    bundle
        .encode_prompt_warm(&mut cache, &turn_one, &tools, Thinking::Default)
        .expect("seed");
    let turn_two = msgs(&[("user", "hello "), ("assistant", "ab"), ("user", "more")]);
    let (warm_ids, _) = bundle
        .encode_prompt_warm(&mut cache, &turn_two, &tools, Thinking::Default)
        .expect("warm");
    let full = bundle
        .encode_prompt(&turn_two, &tools, Thinking::Default)
        .expect("full");
    assert_eq!(warm_ids, full);
    assert_eq!(
        cache.reuses(),
        0,
        "a head ending inside a partial added token must never reuse"
    );
    assert!(cache.fallbacks() >= 2, "each refusal is counted");
}

/// SF1: a two-turn pure append on the turn-separated template — the
/// second turn MUST take the warm path (exactly one reuse), so a
/// regression that makes warm_turn always return false fails here.
#[test]
fn warm_reuses_on_a_pure_appended_turn() {
    let (_d, bundle) = fixture_bundle(TEMPLATE_TURN);
    let mut cache = WarmTurnCache::new();
    let tools = Vec::new();
    let turn_one = msgs(&[("user", "hi")]);
    bundle
        .encode_prompt_warm(&mut cache, &turn_one, &tools, Thinking::Default)
        .expect("seed");
    assert_eq!(cache.fallbacks(), 1, "the seed is a fallback");
    assert_eq!(
        cache.last_fallback_reason(),
        Some("seed"),
        "an empty slot names its cause"
    );
    assert_eq!(cache.reuses(), 0);
    let turn_two = msgs(&[("user", "hi"), ("assistant", "ab"), ("user", "cd")]);
    let (warm_ids, warm_bounds) = bundle
        .encode_prompt_warm(&mut cache, &turn_two, &tools, Thinking::Default)
        .expect("warm append");
    let full = bundle
        .encode_prompt(&turn_two, &tools, Thinking::Default)
        .expect("full");
    assert_eq!(warm_ids, full);
    // The three-pass oracle: the reused turn's ids AND boundaries must
    // equal the original (pre-derivation) boundary path's answer too.
    let oracle = bundle
        .encode_prompt_with_boundary(&turn_two, &tools, Thinking::Default)
        .expect("three-pass oracle");
    assert_eq!(warm_ids, oracle.0, "warm ids equal the three-pass oracle");
    assert_eq!(
        warm_bounds.tip, oracle.1.tip,
        "warm tip equals the three-pass oracle"
    );
    assert_eq!(
        warm_bounds.root, oracle.1.root,
        "warm root equals the three-pass oracle"
    );
    assert_eq!(
        cache.reuses(),
        1,
        "the appended turn must be REUSED, not fallen back"
    );
    assert_eq!(cache.fallbacks(), 1, "no second fallback");
}

/// SF1: the self-check must run on a WARM call, not on the seed fallback.
#[test]
fn warm_self_check_runs_on_a_reused_turn() {
    let Some(bundle) = qwen_bundle() else {
        return;
    };
    let mut cache = WarmTurnCache::with_self_check();
    let tools = Vec::new();
    let turn_one = msgs(&[("system", "You are careful."), ("user", "hello")]);
    bundle
        .encode_prompt_warm(&mut cache, &turn_one, &tools, Thinking::Default)
        .expect("seed");
    assert_eq!(cache.reuses(), 0, "the seed is a fallback");
    let turn_two = msgs(&[
        ("system", "You are careful."),
        ("user", "hello"),
        ("assistant", "hi"),
        ("user", "and again"),
    ]);
    let (ids, bounds) = bundle
        .encode_prompt_warm(&mut cache, &turn_two, &tools, Thinking::Default)
        .expect("warm with self-check");
    let full = bundle
        .encode_prompt(&turn_two, &tools, Thinking::Default)
        .expect("full");
    assert_eq!(ids, full);
    assert!(bounds.tip.is_some());
    assert_eq!(cache.reuses(), 1, "the self-checked call must be a reuse");
}

/// Bounded memory (B condition 2): the cache reports its live bytes —
/// the head render, four bytes per cached id, and the key's owned
/// strings (system opener + serialized tool list).
#[test]
fn warm_cache_reports_accounted_bytes() {
    let Some(bundle) = qwen_bundle() else {
        return;
    };
    let mut cache = WarmTurnCache::new();
    let tools = Vec::new();
    let turn = msgs(&[("system", "You are careful."), ("user", "hello")]);
    let _ = bundle
        .encode_prompt_warm(&mut cache, &turn, &tools, Thinking::Default)
        .expect("warm");
    let bytes = cache.accounted_bytes();
    assert!(bytes > 0, "a seeded cache must account its bytes");
    assert!(
        bytes < turn.len() * 4096,
        "the cache must be bounded by the rendered turn, got {bytes}"
    );
}

/// Sol r1 MF (render-preserving role edits): a marked-user seed and its
/// demoted-system twin render byte-identically under the Qwen template, so
/// the twin warm-hits. But the raw first-turn selector MOVED (2 -> 3
/// rows); the hit must not keep the seed's endpoint — the foreign twin
/// (same key, unrelated final turn) must classify other-conversation,
/// exactly what directly seeding the same history reports.
#[test]
fn warm_hit_refreshes_the_endpoint_when_roles_alias_the_render() {
    let Some(bundle) = qwen_bundle() else {
        return;
    };
    let mut cache = WarmTurnCache::new();
    let tools = Vec::new();
    let seed = msgs(&[
        ("system", "You are careful."),
        ("user", "[system] house rules"),
    ]);
    let _ = bundle
        .encode_prompt_warm(&mut cache, &seed, &tools, Thinking::Default)
        .expect("seed");
    let hit = msgs(&[
        ("system", "You are careful."),
        ("system", "house rules"),
        ("user", "hello"),
    ]);
    let _ = bundle
        .encode_prompt_warm(&mut cache, &hit, &tools, Thinking::Default)
        .expect("warm hit");
    assert_eq!(
        cache.reuses(),
        1,
        "the demoted-system twin renders identically and reuses"
    );
    let foreign = msgs(&[
        ("system", "You are careful."),
        ("system", "house rules"),
        ("user", "unrelated"),
    ]);
    let _ = bundle
        .encode_prompt_warm(&mut cache, &foreign, &tools, Thinking::Default)
        .expect("foreign");
    assert_eq!(
        cache.last_fallback_reason(),
        Some("other-conversation"),
        "an unrelated final turn diverges inside the refreshed endpoint"
    );
    // The direct-seed oracle: the same history seeded fresh must classify
    // the same foreign turn identically.
    let mut fresh = WarmTurnCache::new();
    let _ = bundle
        .encode_prompt_warm(&mut fresh, &hit, &tools, Thinking::Default)
        .expect("direct seed");
    let _ = bundle
        .encode_prompt_warm(&mut fresh, &foreign, &tools, Thinking::Default)
        .expect("foreign after direct seed");
    assert_eq!(
        fresh.last_fallback_reason(),
        Some("other-conversation"),
        "directly seeded history agrees"
    );
}

/// Sol r2 MF (unchanged selector, content growth): a whole-head endpoint
/// must track the head it covers. Seed [system "hello", user "hello"];
/// the middle request grows the user content to "hello world" — same raw
/// rows, byte-extend head, eligible one-token delta, so it warm-hits while
/// first_message_end stays 2. The endpoint must then cover the NEW whole
/// head (what directly seeding the middle history stores), or the third
/// request ("hello war", diverging inside it) mislabels seam.
#[test]
fn whole_head_endpoint_tracks_the_head_across_content_growth() {
    let (_d, bundle) = fixture_bundle(TEMPLATE_TURN);
    let mut cache = WarmTurnCache::new();
    let tools = Vec::new();
    let seed = msgs(&[("system", "hello"), ("user", "hello")]);
    let _ = bundle
        .encode_prompt_warm(&mut cache, &seed, &tools, Thinking::Default)
        .expect("seed");
    let middle = msgs(&[("system", "hello"), ("user", "hello world")]);
    let _ = bundle
        .encode_prompt_warm(&mut cache, &middle, &tools, Thinking::Default)
        .expect("content growth warm-hits");
    assert_eq!(
        cache.reuses(),
        1,
        "the grown user content byte-extends the head and reuses"
    );
    let foreign = msgs(&[("system", "hello"), ("user", "hello war")]);
    let _ = bundle
        .encode_prompt_warm(&mut cache, &foreign, &tools, Thinking::Default)
        .expect("foreign");
    assert_eq!(
        cache.last_fallback_reason(),
        Some("other-conversation"),
        "a divergence inside the grown head is a foreign conversation"
    );
    // The direct-seed oracle: seeding the middle history fresh must
    // classify the same foreign turn identically.
    let mut fresh = WarmTurnCache::new();
    let _ = bundle
        .encode_prompt_warm(&mut fresh, &middle, &tools, Thinking::Default)
        .expect("direct seed");
    let _ = bundle
        .encode_prompt_warm(&mut fresh, &foreign, &tools, Thinking::Default)
        .expect("foreign after direct seed");
    assert_eq!(
        fresh.last_fallback_reason(),
        Some("other-conversation"),
        "directly seeded history agrees"
    );
}

/// A turn template that renders EVERY per-message field the resend
/// digest must see: content, reasoning_content (presence and text —
/// `{% if m.reasoning_content is string %}` matches the pinned Qwen
/// template's absent-vs-empty distinction), tool-call id/name/arguments
/// (boundaries visible), plus the tool list and thinking mode.
const TEMPLATE_FIELDS: &str = concat!(
    r#"{% if tools %}<|end|>TOOLS{% for t in tools %}{{ t }}{% endfor %}{% endif %}"#,
    r#"{% if enable_thinking %}<|end|>THINK{% endif %}"#,
    r#"{% for m in messages %}<|end|>{{ m.role }}:{{ m.content }}"#,
    r#"{% if m.reasoning_content is string %}<|end|>hello{% endif %}"#,
    r#"{% for c in m.tool_calls %}{{ c.id }}<|end|>{{ c.name }}<|end|>{{ c.arguments }}{% endfor %}"#,
    r#"{% endfor %}"#,
    r#"{% if add_generation_prompt %}<|end|>hello{% endif %}"#,
);

fn message(role: Role, content: &str) -> PromptMessage {
    PromptMessage {
        role,
        content: Some(content.into()),
        reasoning_content: None,
        name: None,
        tool_call_id: None,
        tool_calls: Vec::new(),
    }
}

/// [system, user "hello"]: the seed conversation the field tests append to.
fn seed_conv() -> Vec<PromptMessage> {
    vec![
        message(Role::System, "You are careful."),
        message(Role::User, "hello"),
    ]
}

fn appended_with_reasoning(reasoning: Option<&str>) -> Vec<PromptMessage> {
    let mut msgs = seed_conv();
    let mut assistant = message(Role::Assistant, "hi");
    assistant.reasoning_content = reasoning.map(str::to_string);
    msgs.push(assistant);
    msgs.push(message(Role::User, "more"));
    msgs
}

fn appended_with_call(call: [&str; 3]) -> Vec<PromptMessage> {
    let mut msgs = seed_conv();
    let mut assistant = message(Role::Assistant, "");
    assistant.tool_calls = vec![ToolCallSpec {
        id: call[0].into(),
        name: call[1].into(),
        arguments: call[2].into(),
    }];
    msgs.push(assistant);
    msgs.push(message(Role::User, "more"));
    msgs
}

/// An appended conversation whose assistant carries reasoning_content:
/// replay-flagged bundles render it differently, so the same-pins
/// effective-flags test needs a conversation the flags actually touch.
fn flag_conv() -> Vec<PromptMessage> {
    appended_with_reasoning(Some("secret"))
}

/// SF-C: different TEMPLATE BYTES with identical flags must miss — the
/// pin digests in the identity fold are the only guard left once the
/// flags agree, and dropping `self.identity.hash` from the fold breaks
/// exactly this test (Opus delta SF3).
#[test]
fn warm_resend_misses_when_template_bytes_differ_flags_equal() {
    let (_d, plain) = fixture_bundle(TEMPLATE_FIELDS);
    // One byte changed in the template source: the pinned digests (and
    // the render) differ, the effective flags stay identical.
    let (_d2, other) = fixture_bundle(&format!("{TEMPLATE_FIELDS} "));
    let tools = Vec::new();
    let mut cache = WarmTurnCache::new();
    plain
        .encode_prompt_warm(&mut cache, &seed_conv(), &tools, Thinking::Default)
        .expect("seed");
    let (hit_ids, _) = plain
        .encode_prompt_warm(&mut cache, &flag_conv(), &tools, Thinking::Default)
        .expect("append hit");
    assert_eq!(cache.reuses(), 1);
    let fallbacks = cache.fallbacks();
    let (other_ids, _) = other
        .encode_prompt_warm(&mut cache, &flag_conv(), &tools, Thinking::Default)
        .expect("other bundle");
    let oracle = other
        .encode_prompt(&flag_conv(), &tools, Thinking::Default)
        .expect("other oracle");
    assert_eq!(
        cache.fallbacks(),
        fallbacks + 1,
        "different template bytes are a fallback, never a reuse"
    );
    assert_eq!(other_ids, oracle, "the other bundle's ids are its own");
    assert_ne!(
        other_ids, hit_ids,
        "a different template renders different ids"
    );
}

/// Sol r2 lifecycle: a reseed (a truncated conversation here) must
/// DISCARD the stored last turn. Returning to the old, longer
/// conversation afterwards is a legitimate APPEND against the reseeded
/// head — and that is exactly the observable proof the discard
/// happened: had the stored result survived, the resend gate (same key,
/// same digest) would fire the shortcut, which renders only the
/// test-build cross-check (1 render); the append pays the full append
/// shape (4). Either way the ids must be byte-exact.
#[test]
fn warm_reseed_discards_the_stored_resend_result() {
    let (_d, bundle) = fixture_bundle(TEMPLATE_FIELDS);
    let tools = Vec::new();
    let mut cache = WarmTurnCache::new();
    bundle
        .encode_prompt_warm(&mut cache, &seed_conv(), &tools, Thinking::Default)
        .expect("seed");
    let long = appended_with_reasoning(None);
    bundle
        .encode_prompt_warm(&mut cache, &long, &tools, Thinking::Default)
        .expect("append hit");
    assert_eq!(cache.reuses(), 1);
    // Truncation forces a fallback that RESEEDS the slot.
    bundle
        .encode_prompt_warm(&mut cache, &seed_conv(), &tools, Thinking::Default)
        .expect("truncated");
    let fallbacks = cache.fallbacks();
    let reuses = cache.reuses();
    bundle.reset_counts();
    // The old conversation again: the same key and the same digest the
    // discarded turn stored — only the discard keeps this on the append
    // path instead of the shortcut.
    let (ids, _) = bundle
        .encode_prompt_warm(&mut cache, &long, &tools, Thinking::Default)
        .expect("old conversation after reseed");
    let oracle = bundle
        .encode_prompt(&long, &tools, Thinking::Default)
        .expect("oracle");
    assert_eq!(ids, oracle);
    assert_eq!(cache.fallbacks(), fallbacks, "an append, not a fallback");
    assert_eq!(
        cache.reuses(),
        reuses + 1,
        "an append hit, not skipped work"
    );
    assert_eq!(
        bundle.counts().0,
        4,
        "the full append render shape — a surviving stored result would fire the shortcut (1)"
    );
}

/// Sol-final F2: the cross-check predicate's PRODUCTION arm (both
/// switches off) must stay off — replacing the condition with `true`
/// would make every production resend pay a full encode.
/// KNOWN GAP (recorded, accepted by review): the wiring
/// `resend_verify_enabled(cache.self_check, cfg!(test))` cannot be
/// pinned in-process — `cfg!(test)` is true in every test build, so
/// replacing the second argument with `true` changes no test-build
/// observable. Only the predicate function above is pinned.
#[test]
fn resend_verify_predicate_off_in_production() {
    assert!(!resend_verify_enabled(false, false));
    assert!(resend_verify_enabled(true, false));
    assert!(resend_verify_enabled(false, true));
}

/// Sol-final F2 / Opus delta4 SF / Sol7 MF3: the accounting must count
/// a POPULATED LastTurn (ids + 32-byte digest) EXACTLY. The
/// head_and_key_bytes seam recomputes the non-last-turn part
/// independently, so the split equality catches an undercount in either
/// half — head growth cannot mask a dropped digest term.
#[test]
fn warm_cache_accounts_the_stored_resend_result() {
    let (_d, bundle) = fixture_bundle(TEMPLATE_FIELDS);
    let tools = Vec::new();
    let mut hit = WarmTurnCache::new();
    let mut seeded_only = WarmTurnCache::new();
    for cache in [&mut hit, &mut seeded_only] {
        bundle
            .encode_prompt_warm(cache, &seed_conv(), &tools, Thinking::Default)
            .expect("seed");
    }
    bundle
        .encode_prompt_warm(&mut hit, &flag_conv(), &tools, Thinking::Default)
        .expect("append hit");
    assert_eq!(hit.reuses(), 1);
    let full = bundle
        .encode_prompt(&flag_conv(), &tools, Thinking::Default)
        .expect("oracle");
    let lt = hit.last_turn_bytes();
    assert!(
        lt >= 4 * full.len() + 32,
        "the inspector must report the ids capacity plus the 32-byte digest"
    );
    assert_eq!(
        hit.accounted_bytes(),
        hit.head_and_key_bytes() + lt,
        "the footprint is exactly head+key plus the stored last turn"
    );
    assert!(
        hit.accounted_bytes() > seeded_only.accounted_bytes(),
        "the populated cache outweighs the seed-only cache"
    );
}

/// Sol5 MF (stochastic tokenizers): a BPE with dropout != 0 encodes the
/// same bytes differently per call. It must be ineligible — no warm
/// reuse, no resend shortcut, no boundary derivation — even with every
/// OTHER eligibility flag clean.
///
/// Determinism note (COO requirement): NO assertion here depends on
/// dropout luck. The eligibility gate, the refusal mapping, the digest
/// counter, and the reuse counter are all deterministic functions of
/// the eligibility flag; the stochastic ids are computed but never
/// compared; the commit message records the repeated-run verification.
#[test]
fn warm_rejects_stochastic_dropout_tokenizer() {
    let json = TOKENIZER_JSON.replace("\"dropout\":null", "\"dropout\":0.5");
    let (_d, bundle) = fixture_bundle_with(&json, TEMPLATE_TURN);
    assert!(
        !bundle.tokenizer().boundary_eligible(),
        "the gate itself is the deterministic pin: dropout makes the tokenizer ineligible"
    );
    let mut cache = WarmTurnCache::new();
    let tools = Vec::new();
    let turn = msgs(&[("system", "You are careful."), ("user", "hello")]);
    // No id equality here: with dropout 0.5 even two FULL encodes can
    // disagree — that non-determinism is exactly why the warm path must
    // refuse. The gate is proven by the fallback accounting alone.
    let (_warm_ids, _) = bundle
        .encode_prompt_warm(&mut cache, &turn, &tools, Thinking::Default)
        .expect("warm");
    assert_eq!(
        cache.last_fallback_reason(),
        Some("seed"),
        "a dropout tokenizer must never enter the warm path"
    );
    // The seeding fallback sets the key, so the SAME request is exactly
    // the key-matched digest-gate case: still a full path, never a reuse.
    let (_again, _) = bundle
        .encode_prompt_warm(&mut cache, &turn, &tools, Thinking::Default)
        .expect("warm again");
    // This same-key request must refuse — never a reuse — and the
    // digest counter proves the gate never even hashed the conversation.
    // NOTE: no reason-string assert here: whether the first fallback
    // seeded the key depends on the DERIVED boundary shape, which a
    // dropout tokenizer makes luck-dependent (the reason flips between
    // "seed" and "ineligible" across runs); the reason mapping is
    // deterministic only for deterministic tokenizers.
    assert_eq!(
        cache.digest_calls(),
        0,
        "ineligible requests are never hashed"
    );
    assert_eq!(cache.reuses(), 0, "no shortcut on a stochastic tokenizer");
}

/// COO SF: the digest call site is pinned by the invocation counter —
/// a cold request, a key-changed request, and an ineligible same-key
/// request each compute ZERO digests; only the eligible key-matched
/// candidate computes exactly one.
#[test]
fn warm_digest_is_computed_only_for_eligible_key_matched_requests() {
    let (_d, bundle) = fixture_bundle(TEMPLATE_FIELDS);
    let tools = Vec::new();
    let mut cache = WarmTurnCache::new();
    assert_eq!(cache.digest_calls(), 0, "counter starts at zero");

    bundle
        .encode_prompt_warm(&mut cache, &seed_conv(), &tools, Thinking::Default)
        .expect("seed (cold)");
    assert_eq!(
        cache.digest_calls(),
        0,
        "a cold request has no key to match and must not hash the conversation"
    );

    bundle
        .encode_prompt_warm(&mut cache, &flag_conv(), &tools, Thinking::Default)
        .expect("append hit");
    assert_eq!(
        cache.digest_calls(),
        1,
        "the append's key still matches, so it computes ONE digest — then misses it and appends"
    );

    bundle
        .encode_prompt_warm(&mut cache, &flag_conv(), &tools, Thinking::Default)
        .expect("resend (the shortcut)");
    assert_eq!(
        cache.digest_calls(),
        2,
        "exactly one digest per request: the resend's matches the stored inputs"
    );

    // A REAL key change: the key folds system + tools + thinking +
    // identity, not the message list — flip thinking.
    bundle
        .encode_prompt_warm(&mut cache, &flag_conv(), &tools, Thinking::On)
        .expect("key-changed");
    assert_eq!(
        cache.digest_calls(),
        2,
        "a key-changed request must not hash the conversation (count unchanged)"
    );
}

/// COO SF (identity): SAME template bytes and SAME flags, DIFFERENT
/// tokenizer bytes — only the tokenizer pins in the identity fold keep
/// the two bundles apart, so replacing `identity.hash` with a
/// template-only hash breaks exactly this test.
#[test]
fn warm_resend_misses_when_only_tokenizer_bytes_differ() {
    // Same vocab-plus-an-unused-token: eligible, renders identically,
    // so ONLY the tokenizer pin (not the render) can separate them.
    let json = TOKENIZER_JSON.replace(r#""hello":8,"a":9"#, r#""hello":8,"zz":14,"a":9"#);
    let (_d, a) = fixture_bundle(TEMPLATE_FIELDS);
    let (_d2, b) = fixture_bundle_with(&json, TEMPLATE_FIELDS);
    let tools = Vec::new();
    let mut cache = WarmTurnCache::new();
    a.encode_prompt_warm(&mut cache, &seed_conv(), &tools, Thinking::Default)
        .expect("seed");
    let (hit_ids, _) = a
        .encode_prompt_warm(&mut cache, &flag_conv(), &tools, Thinking::Default)
        .expect("append hit");
    assert_eq!(cache.reuses(), 1);
    let fallbacks = cache.fallbacks();
    let (ids, _) = b
        .encode_prompt_warm(&mut cache, &flag_conv(), &tools, Thinking::Default)
        .expect("other tokenizer");
    let oracle = b
        .encode_prompt(&flag_conv(), &tools, Thinking::Default)
        .expect("other oracle");
    assert_eq!(
        cache.fallbacks(),
        fallbacks + 1,
        "a same-template, same-flags, DIFFERENT-tokenizer bundle must be a fallback, never a reuse"
    );
    assert_eq!(ids, oracle);
    // The fallback count is the discriminator (the vocab growth shifts
    // the fixture's marker ids, so id equality with the hit is neither
    // expected nor required).
    let _ = hit_ids;
}

/// Sol6 test-adequacy: a ROLE-ONLY edit of the FIRST message (content
/// identical, non-system key unchanged) must miss — pinning that the
/// digest covers msgs[0] (a `&msgs[1..]` hash passes every other test).
#[test]
fn warm_resend_misses_on_first_message_role_only_change() {
    let (_d, bundle) = fixture_bundle(TEMPLATE_FIELDS);
    let tools = Vec::new();
    let mut cache = WarmTurnCache::new();
    bundle
        .encode_prompt_warm(&mut cache, &seed_conv(), &tools, Thinking::Default)
        .expect("seed");
    bundle
        .encode_prompt_warm(&mut cache, &flag_conv(), &tools, Thinking::Default)
        .expect("append hit");
    assert_eq!(cache.reuses(), 1);
    // System -> Developer keeps the key UNCHANGED (warm_key maps both
    // roles to the system field — the Qwen grammar itself rewrites
    // developer messages to system, so only the digest can tell the
    // roles apart) while TEMPLATE_FIELDS renders the role name.
    let mut flipped = flag_conv();
    flipped[0] = message(Role::Developer, "You are careful.");
    let fallbacks = cache.fallbacks();
    let digests = cache.digest_calls();
    let (ids, _) = bundle
        .encode_prompt_warm(&mut cache, &flipped, &tools, Thinking::Default)
        .expect("role-flipped resend");
    let oracle = bundle
        .encode_prompt(&flipped, &tools, Thinking::Default)
        .expect("oracle");
    assert_eq!(
        cache.fallbacks(),
        fallbacks + 1,
        "a role-only first-message edit is a digest miss, never a reuse"
    );
    assert_eq!(
        cache.digest_calls(),
        digests + 1,
        "the key matched, so the digest gate ran — msgs[0] must be inside it"
    );
    assert_eq!(ids, oracle);
}

/// KNOWN GAP (recorded, accepted by review): a POPULATED cache hit by a
/// FAILING fallback request (fallback error, entry retained) has no
/// test — forcing a fallback error needs a fault-injection seam that
/// does not exist; adding one is deferred as follow-up surface.
///
/// COO SF (lifecycle): with a POPULATED last turn (a hit happened), an
/// unrelated head request must not SERVE the stored result — its changed
/// first message means a KEY miss, so it refuses before any digest work
/// and reseeds (documented: the stored last turn is discarded); the
/// original conversation's next request then re-seeds byte-exact rather
/// than consuming a stale shortcut.
#[test]
fn warm_unrelated_head_reseeds_and_discards_the_stored_result() {
    let (_d, bundle) = fixture_bundle(TEMPLATE_FIELDS);
    let tools = Vec::new();
    let mut cache = WarmTurnCache::new();
    bundle
        .encode_prompt_warm(&mut cache, &seed_conv(), &tools, Thinking::Default)
        .expect("seed");
    bundle
        .encode_prompt_warm(&mut cache, &flag_conv(), &tools, Thinking::Default)
        .expect("append hit");
    assert_eq!(cache.reuses(), 1);

    // An unrelated conversation: different FIRST message, so the stored
    // head is not a prefix of anything in this request.
    let mut foreign = seed_conv();
    foreign[0].content = Some("You are brief.".to_string());
    foreign.push(message(Role::User, "bye"));
    let (ids, _) = bundle
        .encode_prompt_warm(&mut cache, &foreign, &tools, Thinking::Default)
        .expect("foreign conversation");
    let oracle = bundle
        .encode_prompt(&foreign, &tools, Thinking::Default)
        .expect("foreign oracle");
    assert_eq!(ids, oracle);
    assert_eq!(
        cache.reuses(),
        1,
        "an unrelated conversation must not consume the stored result"
    );

    // The foreign request RESEEDED the cache (documented: a different
    // head discards the last turn), so the original conversation's next
    // request re-seeds too — byte-exact, never a stale shortcut.
    let (again, _) = bundle
        .encode_prompt_warm(&mut cache, &flag_conv(), &tools, Thinking::Default)
        .expect("original resend");
    assert_eq!(again, oracle_of(&bundle, &flag_conv(), &tools));
    assert_eq!(cache.reuses(), 1, "the reseed discarded the stored result");
}

fn oracle_of(bundle: &TokenizerBundle, turn: &[PromptMessage], tools: &[ToolSpec]) -> Vec<u32> {
    bundle
        .encode_prompt(turn, tools, Thinking::Default)
        .expect("oracle")
}

/// Sol8 HIGH / Opus delta6 SF1 (id-rewriting post-processors): a
/// TemplateProcessing that repeats its input sequence DUPLICATES every
/// encode — loadable in tokenizers 0.21.4 once "special_tokens":{} is
/// present — and a MISTAGGED JSON (type "ByteLevel", missing
/// trim_offsets, carrying the duplicating template body) deserializes
/// as that same TemplateProcessing because PostProcessorWrapper is
/// untagged and ignores the tag. Both must be INELIGIBLE: the load-time
/// behavioral probe sees parts double ([8,8] + [14,14] against a whole
/// of [8,14,8,14]) and refuses. The ids assertions pin the duplication
/// itself, so the probe's refusal is attributable.
#[test]
fn warm_rejects_duplicating_post_processors_behaviorally() {
    let base_pp = r#""post_processor":{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":false}"#;
    let dup_body = |tag: &str| {
        format!(
            r#""post_processor":{{"type":"{tag}","add_prefix_space":false,"single":[{{"Sequence":{{"id":"A","type_id":0}}}},{{"Sequence":{{"id":"A","type_id":0}}}}],"pair":[{{"Sequence":{{"id":"A","type_id":0}}}},{{"Sequence":{{"id":"B","type_id":1}}}}],"special_tokens":{{}}}}"#
        )
    };
    for (label, tag) in [
        ("plain tag", "TemplateProcessing"),
        ("mistagged ByteLevel", "ByteLevel"),
    ] {
        let json = TOKENIZER_JSON.replace(base_pp, &dup_body(tag));
        let (_d, bundle) = fixture_bundle_with(&json, TEMPLATE_TURN);
        let whole = bundle.tokenizer().encode("hello world").expect("whole");
        let head = bundle.tokenizer().encode("hello").expect("head");
        let tail = bundle.tokenizer().encode(" world").expect("tail");
        // The duplication must be OBSERVABLE (parts double independently
        // of the whole), so the probe's refusal is attributable to real
        // id rewriting, not to an unrelated quirk.
        assert_ne!(whole, [head.clone(), tail.clone()].concat(), "{label}");
        // Exact doubling, not merely some unrelated difference.
        assert_eq!(head.len(), 2, "{label}: head doubled");
        assert_eq!(head[0], head[1], "{label}: head ids identical");
        assert_eq!(tail.len(), 2, "{label}: tail doubled");
        assert_eq!(tail[0], tail[1], "{label}: tail ids identical");
        assert_eq!(whole, [head[0], tail[0], head[0], tail[0]], "{label}");
        assert!(
            !bundle.tokenizer().boundary_eligible(),
            "{label}: the behavioral probe must refuse a duplicating post-processor"
        );
    }
}

/// Opus delta6 SF1 negative: a template whose single sequence references
/// the PAIR input (`B`) is not a sane single-encoding template; it
/// PANICS inside tokenizers encode, so the load itself is rejected
/// (see the assertion inside).
#[test]
fn warm_rejects_pair_sequence_in_single_template() {
    let base_pp = r#""post_processor":{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":false}"#;
    let json = TOKENIZER_JSON.replace(
        base_pp,
        r#""post_processor":{"type":"TemplateProcessing","single":[{"Sequence":{"id":"B","type_id":0}}],"pair":[{"Sequence":{"id":"A","type_id":0}},{"Sequence":{"id":"B","type_id":1}}],"special_tokens":{}}"#,
    );
    // Sequence B in a single template panics inside tokenizers encode,
    // and such a tokenizer would panic on every request too — the load
    // itself must fail, not merely mark the tokenizer ineligible.
    match crate::tokenizer::Tokenizer::from_bytes(json.as_bytes()) {
        Err(err) => {
            let msg = format!("{err:?}");
            assert!(msg.contains("panics on encode"), "unexpected error: {msg}");
        }
        Ok(_) => panic!("encode-panicking tokenizer must be rejected at load"),
    }
}

/// Opus delta7 SF1 / delta8 SF1 (SentencePiece-style): a normalizer
/// that PREPENDS a marker to any standalone encode (Prepend "▁" +
/// Replace " "→"▁", as in Llama-2/Mistral). The fixture vocab carries
/// the byte-level spelling of ▁ ("â", "ĸ", "ģ"), so the marker really
/// reaches the ids — the plain-text seam genuinely MISMATCHES (a
/// standalone " beta" gains a second marker), while the warm seam —
/// the right chunk STARTING with the plain added token — concatenates
/// exactly. The probe tests only warm-shaped seams, so this tokenizer
/// stays ELIGIBLE; a delta7-style plain-seam probe would have wrongly
/// excluded it.
#[test]
fn warm_sentencepiece_style_normalizer_stays_eligible() {
    // Byte-level chars of U+2581, added with fresh ids so the Prepend
    // marker is observable (Opus delta8 SF1: without them the BPE
    // silently drops the marker and the test passes for the wrong
    // reason).
    let mut doc: serde_json::Value = serde_json::from_str(TOKENIZER_JSON).expect("fixture json");
    *doc.get_mut("post_processor").expect("pp") = serde_json::json!({
        "type": "TemplateProcessing",
        "single": [
            {"SpecialToken": {"id": "<|end|>", "type_id": 0}},
            {"Sequence": {"id": "A", "type_id": 0}}
        ],
        "pair": [
            {"Sequence": {"id": "A", "type_id": 0}},
            {"Sequence": {"id": "B", "type_id": 1}}
        ],
        "special_tokens": {
            "<|end|>": {"id": "<|end|>", "ids": [101], "tokens": ["<|end|>"]}
        }
    });
    *doc.get_mut("normalizer").expect("normalizer") = serde_json::json!({
        "type": "Sequence",
        "normalizers": [
            {"type": "Prepend", "prepend": "▁"},
            {"type": "Replace", "pattern": {"String": " "}, "content": "▁"}
        ]
    });
    let vocab = doc
        .get_mut("model")
        .and_then(|m| m.get_mut("vocab"))
        .and_then(|v| v.as_object_mut())
        .expect("vocab");
    let next = vocab
        .values()
        .filter_map(|v| v.as_u64())
        .max()
        .expect("ids")
        + 1;
    for (offset, ch) in ["â", "ĸ", "ģ"].iter().enumerate() {
        vocab.insert((*ch).to_string(), serde_json::json!(next + offset as u64));
    }
    let json = serde_json::to_string(&doc).expect("serialize");
    let (_d, bundle) = fixture_bundle_with(&json, TEMPLATE_TURN);
    let tok = bundle.tokenizer();
    assert!(
        tok.boundary_eligible(),
        "SentencePiece-style prepend must stay warm-eligible: the warm seam starts the right chunk with an added token"
    );
    // The plain-text seam REALLY mismatches here — proving the marker
    // is observable and the eligibility does not come from a vacuous
    // fixture.
    let whole = tok.encode("alpha beta").expect("whole");
    let head = tok.encode("alpha").expect("head");
    let tail = tok.encode(" beta").expect("tail");
    assert_ne!(
        [head.clone(), tail.clone()].concat(),
        whole,
        "the Prepend marker must make the plain-text seam mismatch"
    );
    // The warm-shaped seam is byte-exact.
    let whole = tok.encode("hello<|end|> world").expect("whole");
    let head = tok.encode("hello").expect("head");
    let tail = tok.encode("<|end|> world").expect("tail");
    assert_eq!([head, tail].concat(), whole);
}

/// Opus delta8 SF2: the synthetic post-processor check must catch a
/// duplicating template on a tokenizer with NO plain special added
/// tokens (the seam loop is empty and would catch nothing). Chat
/// markers registered with special:false are exactly such a case, and
/// the warm path still cuts on non-special added tokens, so this
/// fixture pins the check that carries the proof.
#[test]
fn warm_synthetic_check_catches_duplication_without_plain_specials() {
    let base_pp = r#""post_processor":{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":false}"#;
    let mut json = TOKENIZER_JSON.replace(
        base_pp,
        r#""post_processor":{"type":"TemplateProcessing","single":[{"Sequence":{"id":"A","type_id":0}},{"Sequence":{"id":"A","type_id":0}}],"pair":[{"Sequence":{"id":"A","type_id":0}},{"Sequence":{"id":"B","type_id":1}}],"special_tokens":{}}"#,
    );
    // Every added token non-special: the seam loop has nothing to test.
    assert_eq!(json.matches(r#""special":true"#).count(), 1);
    json = json.replace(r#""special":true"#, r#""special":false"#);
    let inner = tokenizers::Tokenizer::from_bytes(json.as_bytes()).expect("loads");
    assert!(
        inner
            .get_added_vocabulary()
            .get_added_tokens_decoder()
            .values()
            .all(|t| !t.special),
        "fixture must have no plain specials"
    );
    let (_d, bundle) = fixture_bundle_with(&json, TEMPLATE_TURN);
    assert!(
        !bundle.tokenizer().boundary_eligible(),
        "the synthetic post-processor check must refuse duplication even with an empty seam loop"
    );
}

/// Opus delta6 SF2: the ByteLevel post-processor never inserts or
/// rewrites ids (offset trimming only), so NVFP4/Nemotron-style
/// add_prefix_space=true tokenizers stay warm-ELIGIBLE — the behavioral
/// probe decides on behavior, not on the JSON field.
#[test]
fn warm_byte_level_prefix_space_post_processor_stays_eligible() {
    let json = TOKENIZER_JSON.replace(
        r#""post_processor":{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":false}"#,
        r#""post_processor":{"type":"ByteLevel","add_prefix_space":true,"trim_offsets":false,"use_regex":true}"#,
    );
    let (_d, bundle) = fixture_bundle_with(&json, TEMPLATE_TURN);
    assert!(
        bundle.tokenizer().boundary_eligible(),
        "ByteLevel post-processing trims offsets only — eligible (Opus SF2)"
    );
    let whole = bundle.tokenizer().encode("hello world").expect("whole");
    let head = bundle.tokenizer().encode("hello").expect("head");
    let tail = bundle.tokenizer().encode(" world").expect("tail");
    assert_eq!([head, tail].concat(), whole);
}

/// The Gemma4-style post-processor (specials + one input sequence in the
/// structured template) is loadable AND id-preserving at
/// add_special_tokens=false (specials are skipped), so the behavioral
/// probe keeps it warm-ELIGIBLE and chunked encodes stay byte-exact
/// (COO: the [BOS] $A positive fixture).
#[test]
fn warm_bos_template_post_processor_stays_eligible_and_chunk_exact() {
    let json = TOKENIZER_JSON.replace(
        r#""post_processor":{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":false}"#,
        r#""post_processor":{"type":"TemplateProcessing","single":[{"SpecialToken":{"id":"<|end|>","type_id":0}},{"Sequence":{"id":"A","type_id":0}}],"pair":[{"SpecialToken":{"id":"<|end|>","type_id":0}},{"Sequence":{"id":"A","type_id":0}},{"SpecialToken":{"id":"<|end|>","type_id":1}},{"Sequence":{"id":"B","type_id":1}}],"special_tokens":{"<|end|>":{"id":"<|end|>","ids":[101],"tokens":["<|end|>"]}}}"#,
    );
    let (_d, bundle) = fixture_bundle_with(&json, TEMPLATE_TURN);
    assert!(
        bundle.tokenizer().boundary_eligible(),
        "specials + one input sequence is id-preserving at add_special_tokens=false"
    );
    // Chunk exactness through the engine's own encode: concatenating
    // independently encoded pieces reproduces the whole-string encode.
    let whole = bundle.tokenizer().encode("hello world").expect("whole");
    let head = bundle.tokenizer().encode("hello").expect("head");
    let tail = bundle.tokenizer().encode(" world").expect("tail");
    assert_eq!(
        [head, tail].concat(),
        whole,
        "no special token leaks into chunked encodes at add_special_tokens=false"
    );
    // And the warm path reuses normally on this shape.
    let mut cache = WarmTurnCache::new();
    let tools = Vec::new();
    bundle
        .encode_prompt_warm(&mut cache, &seed_conv(), &tools, Thinking::Default)
        .expect("seed");
    let (ids, _) = bundle
        .encode_prompt_warm(&mut cache, &flag_conv(), &tools, Thinking::Default)
        .expect("append hit");
    assert_eq!(cache.reuses(), 1);
    assert_eq!(
        ids,
        bundle
            .encode_prompt(&flag_conv(), &tools, Thinking::Default)
            .expect("oracle")
    );
}

/// COO (a): the PINNED Qwen3.6 tokenizer stays ELIGIBLE under the
/// full gate. Checks the real file when it is on disk (env override or
/// the known model dir) and skips otherwise; prints the loaded value so
/// the receipt shows it either way.
#[test]
fn pinned_qwen_tokenizer_stays_eligible() {
    let Some(dir) = super::boundary_derive_tests::qwen_model_dir() else {
        return;
    };
    let path = dir.join("tokenizer.json");
    let bytes = std::fs::read(&path).expect("read pinned tokenizer");
    let tok = crate::tokenizer::Tokenizer::from_bytes(&bytes).expect("load pinned tokenizer");
    eprintln!(
        "pinned qwen tokenizer ({}): boundary_eligible={}",
        path.display(),
        tok.boundary_eligible()
    );
    assert!(
        tok.boundary_eligible(),
        "the pinned daily-model tokenizer must stay warm-eligible under the gate"
    );
}
