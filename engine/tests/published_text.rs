//! Published text describes behavior in plain words.
//!
//! The source tree is published as open source. Readers outside the project cannot resolve
//! internal planning labels, such as numbered decision records, work-item codes, team names or
//! message references, so this test rejects them in the files that ship. Vendored sources are
//! skipped because their bytes are pinned. State the behavior, the reason and the constraint
//! instead, and keep the provenance in the commit message.
//!
//! The examples below assemble their labels from pieces, so this file itself carries none.

use std::fs;
use std::path::{Path, PathBuf};

/// Files and directories that ship, relative to the repository root.
/// Development tooling directories are not part of the shipped source;
/// their prose is not what this gate checks.
const ROOTS: &[&str] = &[
    "README.md",
    "THIRD_PARTY_NOTICES.md",
    "doc",
    "api",
    "web/index.html",
    "web/src",
    "engine",
    "docs",
    "qualification",
    ".github",
];

/// Directory names that hold build output, installed packages or pinned third-party bytes.
const SKIPPED_DIRS: &[&str] = &["target", "vendor", "node_modules", "dist", ".test-dist"];

/// Text formats whose prose a reader sees.
const EXTENSIONS: &[&str] = &[
    "rs", "c", "cc", "cpp", "h", "hpp", "cu", "cuh", "m", "mm", "metal", "md", "ts", "js", "html",
    "css", "py", "sh", "toml", "yaml", "yml", "json",
];

/// Hyphenated uppercase words that are ordinary public vocabulary. Loaded from
/// the shared policy file so the sanitizer and this gate read one authority
/// and cannot drift.
fn allowed_tags() -> &'static [String] {
    static ALLOWED: std::sync::OnceLock<Vec<String>> = std::sync::OnceLock::new();
    ALLOWED.get_or_init(|| read_policy_tags(&shared_policy_path()))
}

fn shared_policy_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../scripts/sanitize_policy.json")
}

/// The public checkout can carry benchmark scripts without publication tooling.
fn publication_policy_available(root: &Path) -> bool {
    let missing = matches!(
        shared_policy_path().symlink_metadata(),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound
    );
    if missing {
        assert!(
            std::env::var("FIFI_TEST_REQUIRE_DATA").as_deref() != Ok("1"),
            "FIFI_TEST_REQUIRE_DATA=1 but the shared publication policy is absent"
        );
        let publisher_hook = root.join(".githooks/publisher-integration");
        if !publisher_hook
            .try_exists()
            .expect("checking for the publisher integration hook")
        {
            eprintln!("skipping: no shared publication policy in the public checkout");
            return false;
        }
    }
    // Validate even if no scanned word needs the allowlist.
    allowed_tags();
    true
}

/// Fails closed: a missing or malformed policy panics this gate, never skips.
fn read_policy_tags(path: &Path) -> Vec<String> {
    let text = fs::read_to_string(path)
        .unwrap_or_else(|error| panic!("shared policy unreadable at {}: {error}", path.display()));
    let policy: serde_json::Value = serde_json::from_str(&text)
        .unwrap_or_else(|error| panic!("shared policy must be valid JSON: {error}"));
    policy["allowed_tags"]
        .as_array()
        .expect("policy allowed_tags must be an array")
        .iter()
        .map(|entry| {
            entry["tag"]
                .as_str()
                .expect("every policy entry names a tag")
                .to_string()
        })
        .collect()
}

/// Tells whether the text after a phrase starts the number or code that makes it a reference.
type Follows = fn(&str) -> bool;

/// Phrases that introduce an internal reference when a number or a code follows them.
const NUMBERED_PHRASES: &[(&str, Follows)] = &[
    ("decision ", starts_with_four_digits),
    ("packet ", starts_with_code),
    ("packets ", starts_with_code),
    ("lane ", is_single_capital),
    ("addendum ", starts_with_digit),
    ("fix ", is_capital_then_digit),
];

/// Phrases that are internal on their own: hosts, host paths, tools, rulings, the unpublished
/// planning directory, and disclaimers about copied code.
const PLAIN_PHRASES: &[&str] = &[
    concat!("kub", "dan"),
    "user decision",
    "user's decision",
    "the baton",
    concat!("ledger", "/"),
    concat!("/media/", "dan"),
    concat!("/home/", "dan"),
    concat!("/users/", "dan"),
    concat!("fifi/", "scrch"),
    concat!("fifi/", "ledger"),
    concat!("fifi/", "knowledge"),
    concat!("fifi/", "wrktr"),
    concat!("dan", "lap"),
    "no code copied",
    "no code taken",
    "no other code is copied",
];

/// Host names that are too short to match inside other words.
const HOST_WORDS: &[&str] = &[concat!("pi", "nu")];

#[test]
fn published_files_carry_no_internal_plan_labels() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
    if !publication_policy_available(&root) {
        return;
    }
    let files = published_files(&root);
    let mut findings = Vec::new();
    for file in &files {
        let Ok(text) = fs::read_to_string(file) else {
            continue;
        };
        let name = file
            .strip_prefix(&root)
            .unwrap_or(file)
            .display()
            .to_string();
        let scan = scan_of(&name);
        for (index, line) in text.lines().enumerate() {
            for label in labels_for(scan, line) {
                findings.push(format!("{name}:{}: {label}", index + 1));
            }
        }
    }
    assert!(
        findings.is_empty(),
        "internal plan labels in published files; describe the behavior in plain words \
         and keep the provenance in the commit message:\n{}",
        findings.join("\n")
    );
}

/// The shipped text files under the roots: the files Git tracks, plus new files it does not
/// ignore. Ignored local files, such as the unpublished planning directories, are not shipped.
/// Reading the list needs `git` on `PATH`, and the test fails clearly without it.
fn published_files(root: &Path) -> Vec<PathBuf> {
    let output = std::process::Command::new("git")
        .current_dir(root)
        .args([
            "ls-files",
            "-z",
            "--cached",
            "--others",
            "--exclude-standard",
            "--",
        ])
        .args(ROOTS)
        .output()
        .expect("published_text needs git on PATH to list the shipped files");
    assert!(
        output.status.success(),
        "git ls-files failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let mut files = Vec::new();
    for entry in output
        .stdout
        .split(|byte| *byte == 0)
        .filter(|entry| !entry.is_empty())
    {
        let Ok(relative) = std::str::from_utf8(entry) else {
            continue;
        };
        let path = Path::new(relative);
        let skipped = path.components().any(|part| {
            part.as_os_str()
                .to_str()
                .is_some_and(|name| SKIPPED_DIRS.contains(&name))
        });
        // This file lists the phrases it looks for.
        // The pinned/ publisher snapshots carry the live hook bytes verbatim:
        // frozen, like vendored sources, so their regex and fixtures are not prose.
        let pinned_archive = relative.starts_with("qualification/publisher/pinned/");
        if !skipped
            && !pinned_archive
            && has_text_extension(path)
            && !path.ends_with("tests/published_text.rs")
        {
            files.push(root.join(path));
        }
    }
    files
}

fn has_text_extension(path: &Path) -> bool {
    path.extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| EXTENSIONS.contains(&extension))
}

/// What a file is checked for.
#[derive(Clone, Copy)]
enum Scan {
    /// Everything: plan labels, work-item codes, hosts and tools.
    Prose,
    /// Decision records, whose numbered decision and addendum references are the content.
    Records,
    /// Recorded data, which carries its own codes; only host details are rejected.
    Data,
}

fn scan_of(name: &str) -> Scan {
    if name.starts_with("docs/decisions/") {
        Scan::Records
    } else if name.ends_with(".json") {
        Scan::Data
    } else {
        Scan::Prose
    }
}

fn labels_for(scan: Scan, line: &str) -> Vec<String> {
    match scan {
        Scan::Prose => labels(line),
        Scan::Data => host_labels(line),
        Scan::Records => labels(line)
            .into_iter()
            .filter(|label| !label.starts_with("decision ") && !label.starts_with("addendum "))
            .collect(),
    }
}

/// The host paths, host names and internal tools named on one line.
fn host_labels(line: &str) -> Vec<String> {
    // JSON escapes a slash as `\/`, and Windows paths use `\`; read both as `/`.
    let lower = line
        .to_ascii_lowercase()
        .replace("\\/", "/")
        .replace('\\', "/");
    let mut found = Vec::new();
    for phrase in PLAIN_PHRASES {
        if lower.contains(phrase) {
            found.push((*phrase).to_string());
        }
    }
    for host in HOST_WORDS {
        for (start, _) in lower.match_indices(host) {
            if word_starts_at(&lower, start) && word_ends_at(&lower, start + host.len()) {
                found.push((*host).to_string());
            }
        }
    }
    found
}

fn labels(line: &str) -> Vec<String> {
    let lower = line.to_ascii_lowercase();
    let mut found = Vec::new();
    for (phrase, follows) in NUMBERED_PHRASES {
        for (start, _) in lower.match_indices(phrase) {
            let rest = &line[start + phrase.len()..];
            if word_starts_at(&lower, start) && follows(rest) {
                found.push(format!("{phrase}{}", first_word(rest)));
            }
        }
    }
    found.extend(host_labels(line));
    let words = line.split(|c: char| !(c.is_ascii_alphanumeric() || c == '-'));
    for word in words.map(|word| word.trim_matches('-')) {
        if is_tag(word) && !allowed_tags().iter().any(|allowed| allowed == word) {
            found.push(word.to_string());
        }
    }
    found
}

/// A work-item code: an uppercase head, then hyphenated parts that are numbers with an
/// optional trailing letter, uppercase words, single lowercase letters, or a lowercase letter
/// followed by such a number. A code carries a digit or a one-letter part, which keeps
/// capitalized emphasis out. A single capital minus a number is arithmetic.
fn is_tag(word: &str) -> bool {
    let mut parts = word.split('-');
    let Some(head) = parts.next() else {
        return false;
    };
    let head_ok = head.starts_with(|c: char| c.is_ascii_uppercase())
        && head
            .chars()
            .all(|c| c.is_ascii_uppercase() || c.is_ascii_digit());
    let tail: Vec<&str> = parts.collect();
    let shaped = head_ok && !tail.is_empty() && tail.iter().all(|part| is_tag_part(part));
    let coded = word.chars().any(|c| c.is_ascii_digit()) || tail.iter().any(|part| part.len() == 1);
    let arithmetic = head.len() == 1
        && tail
            .iter()
            .all(|part| part.chars().all(|c| c.is_ascii_digit()));
    shaped && coded && !arithmetic
}

fn is_tag_part(part: &str) -> bool {
    let bytes = part.as_bytes();
    let upper = !part.is_empty()
        && part
            .chars()
            .all(|c| c.is_ascii_uppercase() || c.is_ascii_digit());
    let letter = bytes.len() == 1 && bytes[0].is_ascii_lowercase();
    let lettered = bytes.len() >= 2 && bytes[0].is_ascii_lowercase() && is_numbered(&part[1..]);
    upper || letter || is_numbered(part) || lettered
}

/// Digits, then at most one lowercase letter.
fn is_numbered(part: &str) -> bool {
    let digits = part.trim_end_matches(|c: char| c.is_ascii_lowercase());
    !digits.is_empty()
        && digits.chars().all(|c| c.is_ascii_digit())
        && part.len() - digits.len() <= 1
}

fn word_starts_at(text: &str, index: usize) -> bool {
    text[..index]
        .chars()
        .next_back()
        .is_none_or(|c| !c.is_ascii_alphanumeric())
}

fn word_ends_at(text: &str, index: usize) -> bool {
    text[index..]
        .chars()
        .next()
        .is_none_or(|c| !c.is_ascii_alphanumeric())
}

fn first_word(text: &str) -> &str {
    let end = text
        .find(|c: char| !(c.is_ascii_alphanumeric() || c == '-'))
        .unwrap_or(text.len());
    &text[..end]
}

fn starts_with_four_digits(text: &str) -> bool {
    let digits = text.chars().take_while(char::is_ascii_digit).count();
    digits == 4
}

fn starts_with_code(text: &str) -> bool {
    text.starts_with(|c: char| c.is_ascii_uppercase() || c.is_ascii_digit())
}

fn is_single_capital(text: &str) -> bool {
    let mut chars = text.chars();
    let first = chars.next().is_some_and(|c| c.is_ascii_uppercase());
    first && chars.next().is_none_or(|c| !c.is_ascii_alphanumeric())
}

fn is_capital_then_digit(text: &str) -> bool {
    let mut chars = text.chars();
    chars.next().is_some_and(|c| c.is_ascii_uppercase())
        && chars.next().is_some_and(|c| c.is_ascii_digit())
}

fn starts_with_digit(text: &str) -> bool {
    text.starts_with(|c: char| c.is_ascii_digit())
}

#[test]
fn the_label_rules_match_codes_and_skip_ordinary_words() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
    if !publication_policy_available(&root) {
        return;
    }
    assert_eq!(
        // KV8-B is shared-policy ordinary vocabulary: detected shape, skipped.
        labels(concat!("see AB", "-7c and KV8", "-B")),
        [concat!("AB", "-7c")]
    );
    assert_eq!(labels(concat!("KV", "-v1 rows")), [concat!("KV", "-v1")]);
    assert_eq!(
        labels(concat!("the decision ", "1234 path")),
        [concat!("decision ", "1234")]
    );
    assert_eq!(
        labels(concat!("run by lane", " Q today")),
        [concat!("lane", " Q")]
    );
    assert_eq!(
        labels(concat!("the fix", " L3 store")),
        [concat!("fix", " L3")]
    );
    assert_eq!(
        labels(concat!("see ledger", "/plans")),
        [concat!("ledger", "/")]
    );
    assert_eq!(
        labels(concat!("weights in /home/", "dan/models")),
        [concat!("/home/", "dan")]
    );
    assert_eq!(
        labels(concat!("built on pi", "nu today")),
        [concat!("pi", "nu")]
    );
    assert!(labels(concat!("a s", "pin", "up of the pool")).is_empty());
    let record = concat!("see decision ", "0035 and the pi", "nu host");
    assert_eq!(labels_for(Scan::Records, record), [concat!("pi", "nu")]);
    assert!(labels_for(Scan::Data, concat!("task P0A", "-01")).is_empty());
    let escaped = concat!(r#"{"path": "\/home\/"#, r#"dan\/x"}"#);
    assert_eq!(labels_for(Scan::Data, escaped), [concat!("/home/", "dan")]);
    assert_eq!(
        labels_for(Scan::Data, concat!(r"C:\Users\", r"dan\x")),
        [concat!("/users/", "dan")]
    );
    assert_eq!(labels("ideas only, no code copied"), ["no code copied"]);
    assert!(labels("UTF-8 text, SHA-256 pins, Qwen3.6-35B-A3B, x-api-key").is_empty());
    assert!(
        labels("a decision tree, the lane width, Content-Type, sm_86, a fix for it").is_empty()
    );
    assert!(labels(concat!("PER-LAYER growth over C", "-1 rows")).is_empty());
}

#[test]
fn a_missing_or_malformed_policy_fails_closed() {
    let dir = tempfile::tempdir().expect("temporary directory");
    let missing = dir.path().join("no-such-policy.json");
    let result = std::panic::catch_unwind(|| read_policy_tags(&missing));
    assert!(result.is_err(), "a missing policy must panic, not skip");

    let malformed = dir.path().join("bad-policy.json");
    fs::write(&malformed, "{ not json").unwrap();
    let result = std::panic::catch_unwind(|| read_policy_tags(&malformed));
    assert!(result.is_err(), "a malformed policy must panic, not skip");
}

#[test]
fn ignored_local_files_are_not_published_files() {
    // The check shells out to git: the public CI image is the one host
    // without the binary, so skip there - unless FIFI_TEST_REQUIRE_DATA=1
    // demands the real run.
    if std::process::Command::new("git")
        .arg("--version")
        .output()
        .is_err()
    {
        assert!(
            std::env::var("FIFI_TEST_REQUIRE_DATA").as_deref() != Ok("1"),
            "FIFI_TEST_REQUIRE_DATA=1 but no git binary is in PATH: the ignored-files check cannot run"
        );
        eprintln!("skipping: no git binary in PATH");
        return;
    }
    let repo = tempfile::tempdir().expect("temporary repository");
    let root = repo.path();
    let git = |args: &[&str]| {
        let status = std::process::Command::new("git")
            .current_dir(root)
            .args(args)
            .status()
            .expect("git");
        assert!(status.success(), "git {args:?}");
    };
    git(&["init", "-q"]);
    fs::write(root.join(".gitignore"), "/docs/\n").unwrap();
    fs::create_dir(root.join("docs")).unwrap();
    fs::write(root.join("docs/local.md"), "ignored").unwrap();
    fs::write(root.join("README.md"), "shipped").unwrap();
    let names: Vec<String> = published_files(root)
        .iter()
        .map(|file| file.strip_prefix(root).unwrap().display().to_string())
        .collect();
    assert_eq!(names, ["README.md"]);
}
