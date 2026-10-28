//! The text pins of the CUDA serve start (CUDA-SERVE-1 test 6): the capture setting must
//! reach the step after its construction and before the return, and the rules must be built
//! from the loaded bundle BEFORE every device call. The step cannot run on a CPU, so the
//! wiring is pinned on the source text of `start.rs`, and for the tiered mode (OFF-2-F) also
//! of `tiered_load.rs` and of the capture backstop in `step.rs`, in the gate's c17 style:
//! comments stripped, whitespace collapsed, so only the real calls match.

#[test]
fn serve_hands_the_capture_setting_to_the_step() {
    // The step cannot run on the CPU, so the wiring is pinned on start.rs's own text (the c17
    // style of the gate's build_step pin): comments stripped, whitespace collapsed, so only
    // the real call with its `?` matches and a comment or a dead spelling cannot.
    let body = pinned(fn_section("fn make("));
    let length = body.len();
    let call = body.find("set_capture_config(CaptureConfig{capture:settings.capture");
    let built = body.find("QwenResidentStep::new(");
    let returned = body.rfind("Ok((");
    assert!(
        call.is_some() && built.is_some() && returned.is_some(),
        "{length} bytes"
    );
    let (call, built, returned) = (call.unwrap(), built.unwrap(), returned.unwrap());
    assert!(
        built < call && call < returned,
        "construction {built}, capture {call}, return {returned}"
    );
    assert!(!body.contains("capture:false") && !body.contains("CaptureConfig::default"));
}

#[test]
fn load_options_writes_the_native_layout_outright() {
    // The S2 seam admits two layouts; the serve load carries the native one IN ITS OWN TEXT
    // (MSG-70 condition 8), not through a derived default: the S3c-qualified flip is this one
    // field of `load_options`, and a mutant back to Tiled (or to a default) fails here.
    let body = pinned(fn_section("fn load_options("));
    assert!(
        body.contains("LoadOptions{"),
        "load_options builds the options itself"
    );
    assert!(body.contains("layout:WeightLayout::Native,"));
    assert!(!body.contains("WeightLayout::Tiled"));
    assert!(!body.contains("..Default::default()"));
}

#[test]
fn the_guard_is_committed_between_the_leaks_and_the_step() {
    // KUBDAN-MSG-72 item 3: the commit pins the one-engine flag right after the LAST leak and
    // BEFORE the step and the capture call, so a failure from either keeps the flag up (the
    // process is restarted, not rebuilt) and a success does not release it. Mutants "commit
    // removed" and "commit moved after the step or capture" break this order pin.
    let make = pinned(fn_section("fn make("));
    let leaks = make.rfind("Box::leak(").expect("make leaks its owners");
    let commit = make
        .find("guard.commit();")
        .expect("make commits the guard");
    let step = make
        .find("QwenResidentStep::new(")
        .expect("make builds the step");
    let capture = make
        .find("set_capture_config(CaptureConfig{capture:settings.capture")
        .expect("make hands the capture setting over");
    assert!(
        leaks < commit && commit < step && step < capture,
        "last leak {leaks}, commit {commit}, step {step}, capture {capture}"
    );
}

#[test]
fn make_takes_the_step_layout_from_the_loaded_weights() {
    // KUBDAN-MSG-72 item 1 (the S3b merge): make builds the step options AFTER the load, from
    // the loaded weights' own layout, and BEFORE the step resources; never from a layout
    // constant or from the load options. Mutant m20 "layout read from the load options" breaks
    // the call this pin finds.
    let make = pinned(fn_section("fn make("));
    let load = make.find("load_model(").expect("make loads the model");
    let layout = make
        .find("device.layout()?")
        .expect("make reads the loaded weights' own layout");
    let kv = make.find("kv_mode_of(settings.kv,layout)?").expect(
        "make derives the KV mode from the parsed form and that layout (B4-F: not a constant)",
    );
    let options = make
        .find("step_options(settings.context_limit,layout,kv_mode)?")
        .expect("make builds the options from that layout and the resolved KV mode");
    let resources = make
        .find("StepResources::new(")
        .expect("make builds the step resources");
    assert!(
        load < layout && layout < kv && kv < options && options < resources,
        "load {load}, layout {layout}, kv {kv}, options {options}, resources {resources}"
    );
    assert!(
        !make.contains("WeightLayout::"),
        "make names no layout constant"
    );
    // Admission now retains a CPU plan before startup effects. Resident mode still charges
    // its resolved KV form; paged mode charges the checked physical resource plan.
    let start = pinned(fn_section("fn start("));
    let planned = start
        .find("StartupPlan::new(settings,verify,modes)?")
        .unwrap();
    let tokenizer = start.find("TokenizerBundle::load").unwrap();
    assert!(planned < tokenizer);
    let plan = pinned(fn_section("fn new("));
    assert!(plan.contains("serve_kv_mode(settings)?"));
    assert!(plan.contains("context_tokens(settings.context_limit,kv_mode)?"));
    assert!(plan.contains("None=>kv_cache_bytes(capacity,kv_mode)?"));
    assert!(plan.contains("u64::try_from(plan.device_bytes()?)"));
    assert!(make.contains("StepResources::allocate_paged(projector.context(),plan)?"));
}

#[test]
fn the_cuda_step_reports_the_admitted_pinned_store_limits() {
    let limits = pinned(section_of(
        include_str!("executor.rs"),
        "executor.rs",
        "fn prefix_limits(",
    ));
    assert_eq!(
        limits,
        concat!(
            "fnprefix_limits(&self)->StoreLimits{",
            "ifself.is_tiered(){StoreLimits{roots:0,entries:0,}}",
            "else{self.prefix.limits()}}",
        )
    );
    let admission = pinned(section_of(
        include_str!("prefix_runtime.rs"),
        "prefix_runtime.rs",
        "fn enable_pinned_prefix(",
    ));
    assert!(
        admission.contains(concat!(
            "letretained=pool.retained_entries();",
            "self.prefix=PrefixSlot::with_limits(StoreLimits{roots:0,entries:retained,})",
        )),
        "{admission}"
    );
}

#[test]
fn the_non_tiered_flag_off_path_keeps_the_one_entry_default_store() {
    let constructor = pinned(section_of(
        include_str!("step.rs"),
        "step.rs",
        "pub(crate) fn new(",
    ));
    assert!(constructor.contains("prefix:PrefixSlot::default(),"));
    let make = pinned(fn_section("fn make("));
    assert!(make.contains("ifmodes.prefix!=PrefixMemory::Disabled{step.enable_pinned_prefix()?;"));
    assert_eq!(
        crate::family::session::PrefixSlot::<(), ()>::default().limits(),
        crate::family::session::StoreLimits {
            roots: 0,
            entries: 1,
        }
    );
}

#[test]
fn the_step_backstop_refuses_the_tiered_capture_pair_before_the_session_runs() {
    // OFF-2-F4: the settings, the start, the gate and the block each refuse tiered together with a
    // capture; this is the step's own backstop, the one a caller that builds the step by hand meets.
    // It runs after the sync-each check and before the "capture mode is fixed" check, so deleting it or
    // moving it behind the started test fails here.
    let body = pinned(section_of(
        include_str!("step.rs"),
        "step.rs",
        "pub(crate) fn set_capture_config(",
    ));
    let mode = body.find("check_mode(self.sync_each,config.capture)?;");
    let tiered = body.find("tiered_capture_refusal(self.tiered_mode,config.capture)?;");
    let started = body.find("ifself.started{");
    let (mode, tiered, started) = (
        mode.expect("the sync-each check"),
        tiered.expect("the tiered capture refusal"),
        started.expect("the started check"),
    );
    assert!(
        mode < tiered && tiered < started,
        "check_mode {mode}, tiered {tiered}, started {started}"
    );
}

#[test]
fn the_tiered_load_builds_its_pool_through_tier_pool_and_names_no_kernel() {
    // OFF-2-F3: the kernel of the tier is chosen in ONE place (tier_kernel, behind tier_pool). A
    // build_tiered that named a kernel or built the pool itself would bypass the choice the gate (b)
    // runs and the device rule-C tests pin.
    let body = pinned(section_of(
        include_str!("tiered_load.rs"),
        "tiered_load.rs",
        "pub(super) fn build_tiered(",
    ));
    assert!(body.contains("tier_pool(plan.threads,shape)"), "{body}");
    assert!(!body.contains("ExpertKernel") && !body.contains("CpuPool::"));
}

#[test]
fn a_failed_token_aborts_the_tiered_host_before_the_error_returns() {
    // A failure between begin_layer and finish_layer leaves a layer in flight, and every later
    // begin_layer would refuse until the engine was rebuilt. run_token hands the failure of its
    // inner token to the abort and returns the same result; the abort reaches the host through the
    // runtime. Deleting the call, or narrowing it to a branch of the error, fails here.
    let wrapper = pinned(section_of(
        include_str!("step.rs"),
        "step.rs",
        "pub(crate) fn run_token(",
    ));
    assert!(wrapper.contains("self.run_token_output(token,want_logits,false)"));
    let token = pinned(section_of(
        include_str!("step.rs"),
        "step.rs",
        "fn run_token_output(",
    ));
    assert!(
        token.contains("letran=self.run_token_inner(token,want_logits,want_shortlist);"),
        "{token}"
    );
    assert!(
        token.contains("ifran.is_err(){self.abort_tiered_token();}ran"),
        "{token}"
    );
    let abort = pinned(section_of(
        include_str!("step.rs"),
        "step.rs",
        "fn abort_tiered_token(",
    ));
    assert!(
        abort.contains("runtime.borrow_mut().abort_token();"),
        "{abort}"
    );
    let runtime = pinned(section_of(
        include_str!("tiered_step.rs"),
        "tiered_step.rs",
        "pub(super) fn abort_token(",
    ));
    assert!(runtime.contains("self.host.abort_token();"), "{runtime}");
}

#[test]
fn the_tiered_token_report_carries_the_layer_waits_and_the_bank_fills() {
    // The plain report says one wait per token, which undercounts the tiered step by two orders of
    // magnitude. The report of a token folds in what the runtime counted: every layer's block, the
    // fills of the boundary, and nothing left over from a failed token.
    let token = pinned(section_of(
        include_str!("step.rs"),
        "step.rs",
        "fn run_token_inner(",
    ));
    assert!(
        token.contains("self.report=with_host_syncs(report,self.take_tiered_syncs());"),
        "{token}"
    );
    // The fills of the boundary belong to the token that made them: taking the count before
    // `end_tiered_token` would carry them into the next token's report.
    let ended = token.find("self.end_tiered_token()?;");
    let taken = token.find("self.take_tiered_syncs()");
    assert!(
        ended.zip(taken).is_some_and(|(ended, taken)| ended < taken),
        "end_tiered_token at {ended:?}, take_tiered_syncs at {taken:?}"
    );
    let take = pinned(section_of(
        include_str!("step.rs"),
        "step.rs",
        "fn take_tiered_syncs(",
    ));
    assert!(take.contains("runtime.borrow_mut().take_syncs()"), "{take}");
    let layer = pinned(section_of(
        include_str!("tiered_step.rs"),
        "tiered_step.rs",
        "pub(super) fn run_layer(",
    ));
    assert!(layer.contains("self.syncs.add(made);"), "{layer}");
    let boundary = pinned(section_of(
        include_str!("tiered_step.rs"),
        "tiered_step.rs",
        "pub(super) fn end_token(",
    ));
    assert!(
        boundary.contains("self.syncs.writes+=report.swaps*ExpertBank::FILL_WRITES;"),
        "{boundary}"
    );
    let abort = pinned(section_of(
        include_str!("tiered_step.rs"),
        "tiered_step.rs",
        "pub(super) fn abort_token(",
    ));
    assert!(
        abort.contains("self.syncs=HostSyncs::default();"),
        "{abort}"
    );
}

#[test]
fn the_rules_are_built_before_the_first_device_call() {
    // start(): the bundle and the rules precede the guard and the spawn, and the eos ids come
    // from the SAME loaded bundle — never hard-coded ids, never another file.
    let start = pinned(fn_section("fn start("));
    let load = start
        .find("TokenizerBundle::load")
        .expect("start loads the bundle");
    let eos = start
        .find("bundle.eos_token_ids().to_vec()")
        .expect("start reads the bundle's eos");
    let rules = start
        .find("QwenRules::new(")
        .expect("start builds the rules");
    let guard = start
        .find("BuildGuard::acquire()")
        .expect("start takes the guard");
    let spawn = start
        .find("spawn_engine_noted(")
        .expect("start spawns the engine thread");
    assert!(load < eos && eos < rules && rules < guard && guard < spawn);
    // make(): builds no rules of its own (they arrive built from the caller thread, before
    // every device call) and opens the device before it opens or loads the model.
    let make = pinned(fn_section("fn make("));
    assert!(
        !make.contains("QwenRules::new("),
        "make builds rules: they belong before the device"
    );
    let device = make
        .find("CudaProjector::new()")
        .expect("make opens the device");
    assert!(make.find("open_qwen3_6_35b_a3b(").unwrap() > device);
    assert!(make.find("load_model(").unwrap() > device);
    // The rules reach make: the parameter list names them (a mutant that drops them fails).
    assert!(make.contains("rules:QwenRules") && make.contains("QwenResidentStep::new("));
}

#[test]
fn the_ignored_key_lines_print_through_the_helper_before_the_bundle() {
    // KUBDAN-MSG-71 item 5: start() prints one line per ignored MLX key THROUGH the pure
    // helper, before it reads the tokenizer bundle. Deleting the loop, or printing a constant
    // instead of calling the helper, breaks the pin.
    let start = pinned(fn_section("fn start("));
    let lines = start
        .find("ignored_mlx_line(key)")
        .expect("start prints the ignored-key lines");
    let load = start
        .find("TokenizerBundle::load")
        .expect("start loads the bundle");
    assert!(lines < load, "lines {lines}, bundle {load}");
    assert!(
        start.contains("forkeyin&settings.ignored_mlx_keys{"),
        "the loop walks the settings' own list"
    );
}

#[test]
fn the_sha256_toggle_is_resolved_in_start_not_in_make() {
    // SERVE-1-F F4: a bad Q36_VERIFY_GGUF_SHA256 spelling must fail FAST, at startup —
    // start() resolves the toggle before the bundle, the guard and the spawn — while make()
    // receives only the resolved bool (a mutant that moves the read or the validation onto
    // the engine thread, after the projector exists, breaks both halves of the pin).
    let start = pinned(fn_section("fn start("));
    let resolve = start
        .find("sha256_wanted(std::env::var(VERIFY_SHA256_VAR).ok().as_deref())?")
        .expect("start resolves the toggle");
    let load = start
        .find("TokenizerBundle::load")
        .expect("start loads the bundle");
    assert!(resolve < load, "resolve {resolve}, bundle {load}");
    let spawn = start
        .find("spawn_engine_noted(")
        .expect("start spawns the engine thread");
    assert!(resolve < spawn, "resolve {resolve}, spawn {spawn}");
    let make = pinned(fn_section("fn make("));
    assert!(
        !make.contains("sha256_wanted(") && !make.contains("std::env::var("),
        "make must not read or validate the toggle on the engine thread"
    );
    // The identity check gates the 20 GB hash on the RESOLVED bool alone.
    let identity = pinned(fn_section("fn check_gguf_identity("));
    let gate = identity.find("ifverify{").expect("the verify gate");
    let hash = identity
        .find("check_sha256(path,GGUF_SHA256)?;")
        .expect("the full-file hash");
    assert!(gate < hash, "gate {gate}, hash {hash}");
}

#[test]
fn code_only_strips_comments_and_literals_and_never_loops() {
    // Short inputs, microseconds: the test proves every helper ADVANCES (the 12:16 lesson —
    // a helper that returns at once on its first byte grows `out` without bound).
    let stripped = code_only("let a = 1; // c\nlet b = 2;");
    assert!(stripped.contains("let a = 1;") && stripped.contains("let b = 2;"));
    assert!(!stripped.contains("// c"));
    // Nested block comments, a `/*` inside a string, an unterminated comment, an unterminated
    // string: every case ends in bounded time.
    let nested = code_only("/* outer /* inner */ still a comment */ let c = 3;");
    assert!(nested.contains("let c = 3;") && !nested.contains("still a comment"));
    let quoted = code_only(r#"let s = "/*"; let d = 4;"#);
    assert!(quoted.contains("let d = 4;") && !quoted.contains("/*"));
    assert!(code_only("let e = 5; /* never closed").contains("let e = 5;"));
    assert!(code_only(r#"let f = "unterminated"#).contains("let f ="));
    // A char literal goes; a lifetime stays (the gate's own pin).
    let chars = code_only("let g = 'x'; let h = 6;");
    assert!(chars.contains("let h = 6;") && !chars.contains("'x'"));
    assert!(code_only("fn f(step: &mut QwenResidentStep<'_>) {}").contains("QwenResidentStep<'_>"));
}

/// The text of one `fn` item of start.rs's production code, from the line that holds `start`
/// to the next line that opens a `fn` item (the hardened c17 `gate/run.rs` style).
fn fn_section(start: &str) -> &'static str {
    section_of(include_str!("start.rs"), "start.rs", start)
}

/// The same for any source file: `file` names it in the panic message. Shared with the
/// other test modules of the family (the chunk-gate pins of `executor_tests`, review B-4).
pub(super) fn section_of(source: &'static str, file: &str, start: &str) -> &'static str {
    let at = source
        .find(start)
        .unwrap_or_else(|| panic!("{file} has no {start:?}"));
    let line_start = source[..at].rfind('\n').map_or(0, |nl| nl + 1);
    let mut end = source.len();
    let mut line = source[at..]
        .find('\n')
        .map_or(source.len(), |nl| at + nl + 1);
    while line < source.len() {
        let next = source[line..]
            .find('\n')
            .map_or(source.len(), |nl| line + nl + 1);
        let text = source[line..next].trim_start();
        if starts_fn_item(text) {
            end = line;
            break;
        }
        line = next;
    }
    &source[line_start..end]
}

/// True when `line` opens a `fn` item behind any `pub` prefix.
fn starts_fn_item(line: &str) -> bool {
    let rest = ["pub(crate) ", "pub(super) ", "pub "]
        .iter()
        .find_map(|prefix| line.strip_prefix(prefix))
        .unwrap_or(line);
    rest.starts_with("fn ")
}

/// The section text without comments and without string or char literals (the hardened
/// `gate/run.rs` helper): every stripped piece becomes one space, so a comment cannot satisfy
/// a pin (lifetimes like `'static` are kept).
fn code_only(text: &str) -> String {
    let bytes = text.as_bytes();
    let mut out = String::with_capacity(text.len());
    let (mut plain, mut at) = (0, 0);
    while at < bytes.len() {
        let skipped = match bytes[at] {
            b'/' if bytes.get(at + 1) == Some(&b'/') => Some(skip_line_comment(bytes, at)),
            b'/' if bytes.get(at + 1) == Some(&b'*') => Some(skip_block(bytes, at + 2, 1)),
            b'"' => Some(skip_quoted(bytes, at + 1, b'"')),
            b'\'' if opens_char_literal(bytes, at) => Some(skip_quoted(bytes, at + 1, b'\'')),
            _ => None,
        };
        if let Some(end) = skipped {
            out.push_str(&text[plain..at]);
            out.push(' ');
            at = end;
            plain = at;
        } else {
            at += 1;
        }
    }
    out.push_str(&text[plain..]);
    out
}

/// The index after the `//` line comment starting at `at` (the newline itself stays): the
/// gate's own `skip_line_comment` form, copied verbatim — no predicate to invert, the loop
/// advances on every byte it reads (the 12:16 lesson: a `skip_while(bytes, at, |b| b !=
/// b'\n')` helper returned at once on the `/` and grew the output without bound).
fn skip_line_comment(bytes: &[u8], mut at: usize) -> usize {
    while at < bytes.len() && bytes[at] != b'\n' {
        at += 1;
    }
    at
}

/// The index after the nested block comment whose body starts at `at` (depth 1 on entry).
fn skip_block(bytes: &[u8], mut at: usize, mut depth: usize) -> usize {
    while depth > 0 && at < bytes.len() {
        match (bytes.get(at), bytes.get(at + 1)) {
            (Some(b'/'), Some(b'*')) => {
                depth += 1;
                at += 2;
            }
            (Some(b'*'), Some(b'/')) => {
                depth -= 1;
                at += 2;
            }
            _ => at += 1,
        }
    }
    at
}

/// The index after the quoted literal (string or char) whose body starts at `at`.
fn skip_quoted(bytes: &[u8], mut at: usize, quote: u8) -> usize {
    while at < bytes.len() {
        match bytes[at] {
            b'\\' => at = (at + 2).min(bytes.len()),
            b if b == quote => return at + 1,
            _ => at += 1,
        }
    }
    at
}

/// True when the `'` at `at` opens a char literal rather than a lifetime.
fn opens_char_literal(bytes: &[u8], at: usize) -> bool {
    matches!(
        (bytes.get(at + 1), bytes.get(at + 2)),
        (Some(b'\\'), _) | (Some(_), Some(b'\''))
    )
}

/// Code only, whitespace REMOVED (the gate's own form): a rustfmt re-wrap cannot break a pin.
/// Shared with the other test modules of the family (the chunk-gate pins of
/// `executor_tests`, review B-4).
pub(super) fn pinned(section: &str) -> String {
    code_only(section).split_whitespace().collect::<String>()
}
