//! The text pins of the CUDA serve start (CUDA-SERVE-1 test 6): the capture setting must
//! reach the step after its construction and before the return, and the rules must be built
//! from the loaded bundle BEFORE every device call. The step cannot run on a CPU, so the
//! wiring is pinned on `start.rs`'s own text in the gate's c17 style: comments stripped,
//! whitespace collapsed, so only the real calls match.

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
    let options = make
        .find("step_options(settings.context_limit,device.layout()?)?")
        .expect("make derives the step layout from the loaded weights");
    let resources = make
        .find("StepResources::new(")
        .expect("make builds the step resources");
    assert!(
        load < options && options < resources,
        "load {load}, options {options}, resources {resources}"
    );
    assert!(
        !make.contains("WeightLayout::"),
        "make names no layout constant"
    );
    assert!(
        !make.contains(".layout,"),
        "make reads no other layout field"
    );
    // start() validates the limit only: no step options and no layout before the device.
    let start = pinned(fn_section("fn start("));
    assert!(start.contains("context_tokens(settings.context_limit)?"));
    assert!(!start.contains("step_options(") && !start.contains("WeightLayout::"));
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
        .find("spawn_engine(")
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
        .find("spawn_engine(")
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
    let source = include_str!("start.rs");
    let at = source
        .find(start)
        .unwrap_or_else(|| panic!("start.rs has no {start:?}"));
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
fn pinned(section: &str) -> String {
    code_only(section).split_whitespace().collect::<String>()
}
