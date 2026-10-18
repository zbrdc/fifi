#!/usr/bin/env python3
"""RD-10a-2: deterministic SOURCE check of the public surface of family::qwen36 (no rustdoc, no nightly, no build).

Usage: public_items.py ENGINE_SRC [--base-rev REV --repo REPO] [--write LISTING.txt]
The files are TOKENIZED, not grepped: comments (line, nested block) and string, raw-string and char literals are removed first, so a
`pub` split across lines, after attributes, or inside a string or comment is judged correctly. Every `.rs` file under
family/qwen36/ is found by walking the directory tree (a new file is covered at once). The rules:
  * every `pub` NOT followed by `(` is a public item: module, fn, struct, enum, trait, const, static, type, union, a field of a struct,
    a method of an impl or trait. The MULTISET must be EXACTLY {mod qwen36, mod cuda, mod gate, fn run_step_gate, struct GateSettings,
    struct GateReport}, every name exactly once (a second `pub fn run_step_gate` with the pinned signature or a second
    `pub struct GateReport` fails); `pub(crate)`, `pub(super)`, `pub(in ...)` are not public.
  * any `pub use` (so also any glob) and any `macro_rules!` under family/qwen36/ fails; outside family/qwen36/ a `pub use` or a
    `pub(crate) use` whose path names qwen36, cuda, gate, GateSettings, GateReport or run_step_gate fails (a re-export of the pinned
    surface); `#[doc(hidden)]`, `#[macro_export]`, `#[no_mangle]`, `#[export_name]` fail.
  * cfg: `pub` items count under ANY cfg or feature gate, except inside a `#[cfg(test)] mod ... { }` block.
  * trait impls whose header mentions a gate type (also `Box<GateReport>`, `From<GateReport> for String`) carry no `pub`: they are listed
    and must be exactly {FromStr for GateSettings, Display for GateReport}; every .rs file of the whole engine src OUTSIDE the chain is
    searched for impl headers, `use` declarations and `type` aliases that name a gate type (any one fails, so an `as` alias cannot hide an impl). An inherent `impl GateReport { pub fn ... }` is a public method.
  * a field counts only when its struct is plain `pub` (the fields of a `pub(super)` struct are not public API).
  * any item-level macro invocation (`make_pub!(X);`, `thread_local! { ... }`) fails, because its expansion cannot be seen; so does an
    `extern` block; the header of `pub fn run_step_gate` is pinned to the allowed text.
  * the `qwen36` declaration in family/mod.rs must be `pub mod qwen36;`; with --base-rev the `pub` items of family/mod.rs may differ
    from the base only by `mod qwen36`.
Exit 0 = exactly the allowed surface, 1 = a difference (each line names file:line), 2 = typed STOP (path missing or CLI misuse).
"""
from collections import Counter
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rstokens import skip_group, tokenize  # noqa: E402

ALLOWED_PUB = {"mod qwen36", "mod cuda", "mod gate", "fn run_step_gate", "struct GateSettings", "struct GateReport"}
ALLOWED_IMPLS = {"impl FromStr for GateSettings", "impl Display for GateReport"}
GATE_TYPES = ("GateSettings", "GateReport")
REEXPORT_NAMES = ("qwen36", "cuda", "gate", "GateSettings", "GateReport", "run_step_gate")
KINDS = {"mod", "fn", "struct", "enum", "trait", "type", "const", "static", "use", "union", "macro"}
QUALIFIERS = {"const", "async", "unsafe", "extern"}
FORBIDDEN_ATTRS = ("doc(hidden)", "macro_export", "no_mangle", "export_name")
SIGNATURE = "pub fn run_step_gate ( settings : & GateSettings ) - > Result < GateReport , EngError >"
ITEM_CTX = ("mod", "struct_pub", "impl", "trait", "extern")






def block_kind(header):
    """Kind of the block a `{` opens: decided by the FIRST keyword in token order, so `fn f(x: impl Fn())` and `fn g() -> impl Iterator`
    are functions (lane L's N1), `pub extern "C" fn` is a function, and `extern "C" {` is an extern block."""
    words = [t for _, t in header]
    for k, w in enumerate(words):
        if w == "extern":
            return "fn" if "fn" in words[k:] else "extern"
        if w in ("impl", "struct", "union", "enum", "trait", "mod", "fn"):
            if w in ("struct", "union"):
                return "struct_pub" if "pub" in words[:k] and "pub(r)" not in words else "struct"
            return w
    return "other"


def normalized(header):
    """`impl fmt::Display for GateReport` -> `impl Display for GateReport` (path prefixes dropped)."""
    return re.sub(r"\b\w+ : : ", "", " ".join(t for _, t in header))


def scan_file(path):
    """(pub items, trait impls on the gate types, forbidden constructs) as lists of (line, text)."""
    toks = tokenize(Path(path).read_text())
    pubs, impls, bad, sigs, testmods = [], [], [], [], []
    stack = []            # block kinds
    header = []           # tokens since the last boundary at this level
    attrs = []            # attribute texts seen since the last item
    i = 0
    while i < len(toks):
        line, t = toks[i]
        if t == "#" and i + 1 < len(toks) and toks[i + 1][1] in ("[", "!"):
            j = i + 1 + (toks[i + 1][1] == "!")
            end = skip_group(toks, j, "[", "]")
            text = "".join(x for _, x in toks[j + 1:end - 1])
            attrs.append(text)
            if any(f in text for f in FORBIDDEN_ATTRS):
                bad.append((line, f"#[{text}]"))
            i = end
            continue
        ctx = stack[-1] if stack else "mod"
        if t == "macro_rules" and i + 1 < len(toks) and toks[i + 1][1] == "!":
            bad.append((line, "macro_rules!"))
        elif ctx in ITEM_CTX and "=" not in [w for _, w in header] and re.fullmatch(r"[A-Za-z_]\w*", t) \
                and i + 2 < len(toks) and toks[i + 1][1] == "!" and toks[i + 2][1] in "([{":
            bad.append((line, f"item-level macro invocation {t}!"))
        if t == "pub" and ctx in ITEM_CTX:
            nxt = toks[i + 1][1] if i + 1 < len(toks) else ""
            if nxt == "(":
                i = skip_group(toks, i + 1, "(", ")")
                header += [(line, "pub(r)")]
                continue
            j = i + 1
            while j < len(toks) and (toks[j][1] in QUALIFIERS or toks[j][1] == "STR"):
                j += 1
            kw = toks[j][1] if j < len(toks) else ""
            name = toks[j + 1][1] if j + 1 < len(toks) else ""
            if ctx in ("impl", "trait") and kw == "fn":
                pubs.append((line, f"method {name}"))
            elif ctx == "struct_pub" and toks[j + 1][1:] == (":",):
                pubs.append((line, f"field {kw}"))
            elif kw in KINDS:
                pubs.append((line, f"{kw} {name if kw != 'use' else '*'}".strip()))
            else:
                pubs.append((line, f"field {kw}" if ctx == "struct_pub" else f"unknown {kw}"))
            if kw == "fn" and name == "run_step_gate" and ctx == "mod":
                end = j
                while end < len(toks) and toks[end][1] not in ("{", ";"):
                    end += 1
                sigs.append((line, " ".join(x for _, x in toks[i:end])))
        if t == "{":
            kind = block_kind(header)
            test_mod = kind == "mod" and any(re.fullmatch(r"cfg\(test\)", a) for a in attrs)
            if kind == "impl" and any(w in GATE_TYPES for _, w in header):
                impls.append((header[0][0], normalized(header)))
            if kind == "extern":
                bad.append((header[0][0] if header else line, "extern block"))
            if test_mod:
                i = skip_group(toks, i, "{", "}")
                header, attrs = [], []
                continue
            stack.append(kind)
            header, attrs = [], []
        elif t == "}":
            if stack:
                stack.pop()
            header, attrs = [], []
        elif t == ";":
            words = [w for _, w in header]
            if "mod" in words and any(re.fullmatch(r"cfg\(test\)", a) for a in attrs):
                testmods.append(words[words.index("mod") + 1])
            header, attrs = [], []
        else:
            header.append((line, t))
        i += 1
    return pubs, impls, bad, sigs, testmods


def gate_mentions(path):
    """(line, text) of every impl header, `use` declaration and `type` alias that names a gate type, in a file outside the chain."""
    toks = tokenize(Path(path).read_text())
    out = []
    for k, (line, t) in enumerate(toks):
        if t in ("impl", "use", "type"):
            end = k
            stops = (";",) if t == "use" else ("{", ";")      # a `use a::{B, C};` list is part of the declaration
            while end < len(toks) and toks[end][1] not in stops:
                end += 1
            header = toks[k:end]
            if any(w in GATE_TYPES for _, w in header):
                out.append((line, normalized(header)))
    return out


def reexport_mentions(path):
    """(line, text) of every `pub use` or `pub(crate) use` declaration whose path names the pinned chain, in a file outside the chain."""
    toks = tokenize(Path(path).read_text())
    out = []
    for k, (line, t) in enumerate(toks):
        if t != "use":
            continue
        if k >= 1 and toks[k - 1][1] == "pub":
            vis = "pub"
        elif k >= 4 and toks[k - 1][1] == ")" and toks[k - 2][1] == "crate" and toks[k - 3][1] == "(" and toks[k - 4][1] == "pub":
            vis = "pub(crate)"
        else:
            continue
        end = k
        while end < len(toks) and toks[end][1] != ";":
            end += 1
        header = toks[k:end]
        if any(w in REEXPORT_NAMES for _, w in header):
            out.append((line, f"{vis} {normalized(header)}"))
    return out


def family_pubs(text):
    toks = tokenize(text)
    out = set()
    for k, (_, t) in enumerate(toks):
        if t == "pub" and k + 2 < len(toks) and toks[k + 1][1] != "(" and toks[k + 1][1] in KINDS:
            out.add(f"{toks[k + 1][1]} {toks[k + 2][1]}")
    return out


def check(src, base_family=None):
    src = Path(src)
    fam, tree = src / "family" / "mod.rs", src / "family" / "qwen36"
    if not fam.is_file() or not tree.is_dir():
        raise SystemExit(f"STOP: {src} has no family/mod.rs or family/qwen36/")
    got, problems, listing, all_impls = Counter(), [], [], Counter()
    ftoks = tokenize(fam.read_text())
    quens = [k for k, (_, t) in enumerate(ftoks) if t == "mod" and k + 1 < len(ftoks) and ftoks[k + 1][1] == "qwen36"]
    if len(quens) != 1 or not (quens[0] >= 1 and ftoks[quens[0] - 1][1] == "pub" and (quens[0] < 2 or ftoks[quens[0] - 2][1] != ")")):
        problems.append(f"family/mod.rs: the qwen36 declaration must be exactly one `pub mod qwen36;` (found {len(quens)} declarations)")
    else:
        got["mod qwen36"] += 1
        listing.append(f"family/mod.rs:{ftoks[quens[0]][0]} mod qwen36")
    skipped = []                    # `#[cfg(test)] mod x;` file modules and everything below them are test-only
    for f in sorted(tree.rglob("*.rs"), key=lambda q: (len(q.parts), q.name != "mod.rs", str(q))):
        if any(f == s or s.parent in f.parents and s.name == "mod.rs" for s in skipped):
            continue
        pubs, impls, bad, sigs, testmods = scan_file(f)
        home = f.parent if f.name in ("mod.rs", "lib.rs") else f.parent / f.stem
        skipped += [q for m in testmods for q in (home / f"{m}.rs", home / m / "mod.rs")]
        rel = f.relative_to(src)
        for n, item in pubs:
            listing.append(f"{rel}:{n} {item}")
            got[item] += 1
            if item not in ALLOWED_PUB:
                problems.append(f"{rel}:{n} EXTRA pub item: {item}")
        for n, item in impls:
            listing.append(f"{rel}:{n} {item}")
            all_impls[item] += 1
            if item not in ALLOWED_IMPLS:
                problems.append(f"{rel}:{n} EXTRA trait impl on a gate type: {item}")
        problems += [f"{rel}:{n} FORBIDDEN: {s}" for n, s in bad]
        problems += [f"{rel}:{n} SIGNATURE of run_step_gate changed: {s}" for n, s in sigs if s != SIGNATURE]
    for f in sorted(src.rglob("*.rs")):
        if tree in f.parents:
            continue
        for n, header in gate_mentions(f):
            problems.append(f"{f.relative_to(src)}:{n} a gate type is named OUTSIDE family/qwen36/ (impl, use or alias; an `as` alias would hide an impl): {header}")
        for n, header in reexport_mentions(f):
            problems.append(f"{f.relative_to(src)}:{n} RE-EXPORT of the pinned surface outside family/qwen36/ (`pub use` or `pub(crate) use` naming the chain): {header}")
    for m in sorted(ALLOWED_PUB):                # the allow-list is a MULTISET: every name exactly once
        if got[m] == 0:
            problems.append(f"MISSING pub item: {m}")
        elif got[m] > 1:
            problems.append(f"DUPLICATE pub item: {m} ({got[m]} copies, allowed exactly once)")
    for m in sorted(ALLOWED_IMPLS):
        if all_impls[m] == 0:
            problems.append(f"MISSING trait impl: {m}")
        elif all_impls[m] > 1:
            problems.append(f"DUPLICATE trait impl: {m} ({all_impls[m]} copies, allowed exactly once)")
    if base_family is not None:
        now, then = family_pubs(fam.read_text()), family_pubs(base_family)
        problems += [f"family/mod.rs: new pub item besides qwen36: {x}" for x in sorted(now - then - {"mod qwen36"})]
        problems += [f"family/mod.rs: pub item removed: {x}" for x in sorted(then - now)]
    return problems, listing


def base_text(repo, rev):
    try:
        p = subprocess.run(["git", "-C", repo, "show", f"{rev}:engine/src/family/mod.rs"], capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.TimeoutExpired) as e:
        raise SystemExit(f"STOP: cannot read family/mod.rs at {rev}: {e}")
    if p.returncode != 0:
        raise SystemExit(f"STOP: cannot read family/mod.rs at {rev}: {p.stderr.strip()}")
    return p.stdout


OPTIONS = ("--base-rev", "--repo", "--write")


def opt_value(argv, flag):
    """Value of `flag`; a typed STOP (rc 2, one line, no traceback) when the option comes without a value."""
    if flag not in argv:
        return None
    i = argv.index(flag) + 1
    if i >= len(argv) or argv[i] in OPTIONS:
        raise SystemExit(f"STOP: {flag} needs a value")
    return argv[i]


def main(argv):
    if len(argv) < 2:
        print("STOP: usage: public_items.py ENGINE_SRC [--base-rev REV --repo REPO] [--write FILE]")
        return 2
    base_rev, repo, write = opt_value(argv, "--base-rev"), opt_value(argv, "--repo"), opt_value(argv, "--write")
    if (base_rev is None) != (repo is None):
        raise SystemExit("STOP: --base-rev and --repo must be given together")
    base = base_text(repo, base_rev) if base_rev is not None else None
    problems, listing = check(argv[1], base)
    text = "\n".join(listing) + "\n"
    if write is not None:
        Path(write).write_text(text)
    print(text, end="")
    for p in problems:
        print(p)
    print("OK: exactly the allowed public surface" if not problems else "FAIL: public surface differs")
    return 1 if problems else 0


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv))
    except SystemExit as e:
        if isinstance(e.code, str):
            print(e.code)
            sys.exit(2)
        raise
