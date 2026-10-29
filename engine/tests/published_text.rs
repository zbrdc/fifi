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
const ROOTS: &[&str] = &[
    "README.md",
    "THIRD_PARTY_NOTICES.md",
    "doc",
    "api",
    "web/index.html",
    "web/src",
    "engine",
];

/// Directory names that hold build output, installed packages or pinned third-party bytes.
const SKIPPED_DIRS: &[&str] = &["target", "vendor", "node_modules", "dist", ".test-dist"];

/// Text formats whose prose a reader sees.
const EXTENSIONS: &[&str] = &[
    "rs", "c", "cc", "cpp", "h", "hpp", "cu", "cuh", "m", "mm", "metal", "md", "ts", "js", "html",
    "css", "py", "sh", "toml", "yaml", "yml",
];

/// Hyphenated uppercase words that are ordinary public vocabulary.
const ALLOWED_TAGS: &[&str] = &[
    "UTF-8",
    "UTF-16",
    "SHA-1",
    "SHA-256",
    "SHA-512",
    "IEEE-754",
    "FNV-1a",
    "AVX-512",
    "MLX-LM",
    "GPT-2",
    "NON-INFRINGEMENT",
    "LICENSE-2",
    "LICENSE-APACHE",
    "LICENSE-MIT",
    "BF16-ROUNDED",
    "BF16-REPRESENTABLE",
    "CPU-FP64",
];

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

/// Phrases that are internal on their own: hosts, tools, rulings, the unpublished planning
/// directory, and disclaimers about copied code.
const PLAIN_PHRASES: &[&str] = &[
    concat!("kub", "dan"),
    "user decision",
    "user's decision",
    "the baton",
    concat!("ledger", "/"),
    "no code copied",
    "no code taken",
    "no other code is copied",
];

#[test]
fn published_files_carry_no_internal_plan_labels() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
    let mut files = Vec::new();
    for entry in ROOTS {
        collect(&root.join(entry), &mut files);
    }
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
        for (index, line) in text.lines().enumerate() {
            for label in labels(line) {
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

/// Adds every shipped text file under `path` to `files`.
fn collect(path: &Path, files: &mut Vec<PathBuf>) {
    if path.is_file() {
        // This file lists the phrases it looks for.
        if has_text_extension(path) && !path.ends_with("tests/published_text.rs") {
            files.push(path.to_path_buf());
        }
        return;
    }
    let Ok(entries) = fs::read_dir(path) else {
        return;
    };
    for entry in entries.flatten() {
        let child = entry.path();
        let skipped = child.is_dir()
            && child
                .file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| SKIPPED_DIRS.contains(&name));
        if !skipped {
            collect(&child, files);
        }
    }
}

fn has_text_extension(path: &Path) -> bool {
    path.extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| EXTENSIONS.contains(&extension))
}

/// The internal labels of one line.
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
    for phrase in PLAIN_PHRASES {
        if lower.contains(phrase) {
            found.push((*phrase).to_string());
        }
    }
    let words = line.split(|c: char| !(c.is_ascii_alphanumeric() || c == '-'));
    for word in words.map(|word| word.trim_matches('-')) {
        if is_tag(word) && !ALLOWED_TAGS.contains(&word) {
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
    assert_eq!(
        labels(concat!("see AB", "-7c and KV8", "-B")),
        [concat!("AB", "-7c"), concat!("KV8", "-B")]
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
    assert_eq!(labels("ideas only, no code copied"), ["no code copied"]);
    assert!(labels("UTF-8 text, SHA-256 pins, Qwen3.6-35B-A3B, x-api-key").is_empty());
    assert!(
        labels("a decision tree, the lane width, Content-Type, sm_86, a fix for it").is_empty()
    );
    assert!(labels(concat!("PER-LAYER growth over C", "-1 rows")).is_empty());
}
