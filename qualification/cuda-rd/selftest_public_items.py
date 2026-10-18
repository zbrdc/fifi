#!/usr/bin/env python3
"""CPU self-test of public_items.py on synthetic source trees (no build). Usage: python3 -B -I selftest_public_items.py SCRATCH_DIR
The good tree is the real gate.rs of this packet plus stub modules; every planted deviation must be flagged (exit 1)."""
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import public_items as pi  # noqa: E402

if len(sys.argv) < 2:                          # CLI misuse: the documented typed STOP, not a traceback
    print("STOP: usage: selftest_public_items.py SCRATCH_DIR")
    sys.exit(2)
root = Path(sys.argv[1]) / "pubsrc"
GATE = (HERE.parent.parent / "engine/src/family/qwen36/cuda/gate.rs")
if not GATE.is_file():
    GATE = HERE / "gate.rs"
good_gate = GATE.read_text()
REAL = {f"family/qwen36/cuda/gate/{f.name}": f.read_text() for f in sorted((GATE.parent / "gate").glob("*.rs"))}  # the REAL submodules
BASE = {
    **REAL,
    "family/other.rs": "",
    "family/mod.rs": "mod other;\n#[cfg(all(feature = \"native-cuda\"))]\npub mod qwen36;\n",
    "family/qwen36/mod.rs": "mod host;\n#[cfg(feature = \"native-cuda\")]\npub mod cuda;\npub(crate) struct HostShape;\n",
    "family/qwen36/host.rs": "pub(crate) fn f() {}\npub(super) fn g() {}\n// pub fn in_a_comment() {}\n/* pub fn in_a_block */\n",
    "family/qwen36/cuda/mod.rs": "pub mod gate;\n",
    "family/qwen36/cuda/gate.rs": good_gate,
}
HOST = BASE["family/qwen36/host.rs"]
CASES = {  # name -> (files to change, should fail)
    "good": ({}, False),
    # lane L's final check: shapes of the REAL files in family/qwen36 that must stay clean (N1)
    "fn with an `impl Fn` parameter and a macro in its body (fixtures.rs shape)": ({"family/qwen36/host.rs": "fn f(bias: impl Fn(usize) -> f32) -> f32 {\n    let s = format!(\"{}\", 1);\n    bias(s.len())\n}\n"}, False),
    "fn returning impl Iterator with vec! (cause 1, second form)": ({"family/qwen36/host.rs": "fn g() -> impl Iterator<Item = u32> {\n    vec![1, 2].into_iter()\n}\n"}, False),
    "method with an `impl Fn` parameter and println!": ({"family/qwen36/host.rs": "struct A;\nimpl A {\n    fn m(&self, f: impl Fn()) {\n        println!(\"x\");\n        f();\n    }\n}\n"}, False),
    "const and static initializers with include_str!/concat!/vec! (host_real.rs shape)": ({"family/qwen36/host.rs": 'const E: &str = include_str!("x.txt");\nstatic F: &str = concat!("a", "b");\n'}, False),
    "type alias with a macro on the right-hand side": ({"family/qwen36/host.rs": "type V = Vec<u32>;\nconst W: usize = line!() as usize;\n"}, False),
    "#[cfg(test)] mod x; file module with pub items and macros": ({"family/qwen36/mod.rs": BASE["family/qwen36/mod.rs"] + "#[cfg(test)]\nmod fixtures;\n", "family/qwen36/fixtures.rs": "pub fn helper() { println!(\"x\"); }\npub struct T { pub a: u32 }\n"}, False),
    "#[cfg(test)] mod x; with a nested file below it": ({"family/qwen36/mod.rs": BASE["family/qwen36/mod.rs"] + "#[cfg(test)]\nmod tfix;\n", "family/qwen36/tfix/mod.rs": "pub mod inner;\n", "family/qwen36/tfix/inner.rs": "pub fn leak() {}\n"}, False),
    "a plain `mod x;` file module with a pub item still counts": ({"family/qwen36/mod.rs": BASE["family/qwen36/mod.rs"] + "mod plainfile;\n", "family/qwen36/plainfile.rs": "pub fn leak() {}\n"}, True),
    "an item-level macro in a fn header WITHOUT `=` still fails": ({"family/qwen36/host.rs": "make_it!(X);\nfn f(x: impl Fn()) {}\n"}, True),
    "alias import of a gate type outside the chain (use ... as R)": ({"family/other.rs": "use crate::family::qwen36::cuda::gate::GateReport as R;\nimpl R {\n    pub fn leak(&self) {}\n}\n"}, True),
    "use list naming a gate type outside the chain": ({"family/other.rs": "use crate::family::qwen36::cuda::gate::{GateSettings, run_step_gate};\n"}, True),
    "type alias of a gate type outside the chain": ({"family/other.rs": "type R = crate::family::qwen36::cuda::gate::GateReport;\n"}, True),
    # M1 and lane L's later plants
    "pub(super) and private structs with pub fields are not public (as in the real phases.rs)": ({"family/qwen36/host.rs": "pub(super) struct T { pub a: u32 }\nstruct P { pub b: u32 }\n"}, False),
    "plain pub struct with pub fields (not a gate type)": ({"family/qwen36/host.rs": "pub struct Q { pub a: u32 }\n"}, True),
    "macro invoked at item level, defined elsewhere": ({"family/qwen36/host.rs": "make_pub!(Leak);\n"}, True),
    "pub inside thread_local!": ({"family/qwen36/host.rs": "thread_local! { pub static LEAK: u32 = 0; }\n"}, True),
    "pub fn inside an extern block": ({"family/qwen36/host.rs": 'extern "C" {\n    pub fn c();\n}\n'}, True),
    "inherent impl of a gate type outside family/qwen36": ({"family/other.rs": "impl crate::family::qwen36::cuda::gate::GateReport {\n    pub fn x(&self) {}\n}\n"}, True),
    "trait impl for Box<GateReport>": ({"family/qwen36/cuda/gate.rs": good_gate + "\nimpl Clone for Box<GateReport> {\n    fn clone(&self) -> Self { todo!() }\n}\n"}, True),
    "impl From<GateReport> for String": ({"family/qwen36/cuda/gate.rs": good_gate + "\nimpl From<GateReport> for String {\n    fn from(_: GateReport) -> String { todo!() }\n}\n"}, True),
    "run_step_gate with an extra parameter": ({"family/qwen36/cuda/gate.rs": good_gate.replace("settings: &GateSettings)", "settings: &GateSettings, x: u32)")}, True),
    "unsafe run_step_gate": ({"family/qwen36/cuda/gate.rs": good_gate.replace("pub fn run_step_gate", "pub unsafe fn run_step_gate")}, True),
    # (c) things that must NOT count
    "pub(in path) and pub(super) and pub(self) are not public": ({"family/qwen36/host.rs": "pub(in crate::family) fn a() {}\npub(self) fn b() {}\n"}, False),
    "pub inside a string, a raw string, a char context and comments": ({"family/qwen36/host.rs": 'const S: &str = "pub fn leak() {}";\nconst R: &str = r#"pub struct X; \"pub fn y\""#;\n/* pub fn z() {}\n/* nested pub fn w() {} */ */\n// pub fn v() {}\nfn c() { let _ = (\'p\', "pub"); }\n'}, False),
    "pub items inside #[cfg(test)] mod tests are not counted": ({"family/qwen36/host.rs": "#[cfg(test)]\nmod tests {\n    pub fn helper() {}\n    pub struct T;\n}\n"}, False),
    "pub items inside a function body are local": ({"family/qwen36/host.rs": "fn f() {\n    pub fn inner() {}\n}\n"}, False),
    # (a) hidden from a line scan
    "pub split across lines": ({"family/qwen36/host.rs": "pub\nfn leak() {}\n"}, True),
    "pub after an attribute": ({"family/qwen36/host.rs": "#[doc(hidden)]\npub fn leak() {}\n"}, True),
    "pub after a multi-line attribute": ({"family/qwen36/host.rs": "#[cfg_attr(\n    feature = \"x\",\n    allow(dead_code)\n)]\npub fn leak() {}\n"}, True),
    "pub async unsafe fn": ({"family/qwen36/host.rs": "pub async unsafe fn z() {}\n"}, True),
    "pub extern \"C\" fn": ({"family/qwen36/host.rs": 'pub extern "C" fn z() {}\n'}, True),
    # (b) macros and uses
    "pub use": ({"family/qwen36/mod.rs": BASE["family/qwen36/mod.rs"] + "pub use host::g;\n"}, True),
    "pub use glob": ({"family/qwen36/host.rs": "pub use super::cuda::*;\n"}, True),
    "macro_rules": ({"family/qwen36/host.rs": "macro_rules! m { () => { pub fn made() {} } }\n"}, True),
    "macro_export": ({"family/qwen36/host.rs": "#[macro_export]\nmacro_rules! m { () => {} }\n"}, True),
    "doc(hidden) on a private item": ({"family/qwen36/host.rs": "#[doc(hidden)]\nfn h() {}\n"}, True),
    "no_mangle": ({"family/qwen36/host.rs": "#[no_mangle]\nfn h() {}\n"}, True),
    # (d) cfg and feature gates count
    "pub item under a feature gate": ({"family/qwen36/host.rs": '#[cfg(feature = "native-cuda")]\npub fn gated() {}\n'}, True),
    "pub item in a cfg(not(test)) module": ({"family/qwen36/host.rs": "#[cfg(not(test))]\nmod m {\n    pub fn leak() {}\n}\n"}, True),
    # (e) inherent and trait impls
    "pub method on a gate type": ({"family/qwen36/cuda/gate.rs": good_gate + "\nimpl GateReport {\n    pub fn extra(&self) -> u32 { 1 }\n}\n"}, True),
    "pub method on another type": ({"family/qwen36/host.rs": "struct A;\nimpl A {\n    pub fn m(&self) {}\n}\n"}, True),
    "pub method in a trait impl block of a gate type": ({"family/qwen36/cuda/gate.rs": good_gate.replace("fn fmt(", "pub fn fmt(")}, True),
    "extra trait impl on a gate type": ({"family/qwen36/cuda/gate.rs": good_gate + "\nimpl Default for GateSettings {\n    fn default() -> Self { todo!() }\n}\n"}, True),
    "generic extra trait impl on a gate type": ({"family/qwen36/cuda/gate.rs": good_gate + "\nimpl<T> From<T> for GateReport {\n    fn from(_: T) -> Self { todo!() }\n}\n"}, True),
    "pub field on a gate type": ({"family/qwen36/cuda/gate.rs": good_gate.replace("    raw: String,", "    pub raw: String,")}, True),
    "pub tuple-less field after attributes": ({"family/qwen36/cuda/gate.rs": good_gate.replace("    gguf: PathBuf,", "    #[allow(dead_code)]\n    pub gguf: PathBuf,")}, True),
    "pub const and static, enum, trait, type, union": ({"family/qwen36/host.rs": "pub const N: u32 = 1;\npub static M: u32 = 2;\npub enum E { A }\npub trait T {}\npub type Y = u32;\npub union U { a: u32 }\n"}, True),
    # structure
    "qwen36 still pub(crate)": ({"family/mod.rs": "mod other;\npub(crate) mod qwen36;\n"}, True),
    "qwen36 declared twice": ({"family/mod.rs": "pub mod qwen36;\nmod qwen36;\n"}, True),
    "cuda module missing pub": ({"family/qwen36/mod.rs": "mod host;\nmod cuda;\n"}, True),
    "gate function removed": ({"family/qwen36/cuda/gate.rs": good_gate.replace("pub fn run_step_gate", "fn run_step_gate")}, True),
    "run_step_gate moved into an impl (a method, not the function)": ({"family/qwen36/cuda/gate.rs": good_gate.replace("pub fn run_step_gate(settings: &GateSettings)", "pub fn run_step_gate_x(settings: &GateSettings)")}, True),
    "FromStr impl removed": ({"family/qwen36/cuda/gate.rs": good_gate.replace("impl FromStr for GateSettings", "impl Foo for Bar")}, True),
    # (f) the directory tree is walked
    "pub item in a new nested file": ({"family/qwen36/cuda/extra.rs": "    pub fn nested() {}\n"}, True),
    "pub item in a new deep directory": ({"family/qwen36/deep/er/mod.rs": "pub struct Z;\n"}, True),
    # a duplicate of an ALLOWED name must fail (the allow-list is a multiset)
    "duplicate fn run_step_gate with the pinned signature": ({"family/qwen36/cuda/gate.rs": good_gate + "\npub fn run_step_gate(settings: &GateSettings) -> Result<GateReport, EngError> {\n    todo!()\n}\n"}, True),
    "duplicate struct GateReport": ({"family/qwen36/cuda/gate.rs": good_gate + "\npub struct GateReport;\n"}, True),
    # re-exports of the pinned surface outside the chain
    "pub use of qwen36 in family/other.rs": ({"family/other.rs": "pub use crate::family::qwen36;\n"}, True),
    "pub mod other in family/mod.rs plus a pub use of qwen36 in family/other.rs": ({"family/mod.rs": "pub mod other;\n#[cfg(all(feature = \"native-cuda\"))]\npub mod qwen36;\n", "family/other.rs": "pub use crate::family::qwen36;\n"}, True),
    "pub(crate) use of run_step_gate in family/other.rs": ({"family/other.rs": "pub(crate) use crate::family::qwen36::cuda::gate::run_step_gate;\n"}, True),
}

EXPECT = {  # every should-fail case -> the rule its output must name (the rule names of public_items.py)
    "a plain `mod x;` file module with a pub item still counts": "EXTRA pub item",
    "an item-level macro in a fn header WITHOUT `=` still fails": "FORBIDDEN",
    "alias import of a gate type outside the chain (use ... as R)": "a gate type is named OUTSIDE family/qwen36/",
    "use list naming a gate type outside the chain": "a gate type is named OUTSIDE family/qwen36/",
    "type alias of a gate type outside the chain": "a gate type is named OUTSIDE family/qwen36/",
    "plain pub struct with pub fields (not a gate type)": "EXTRA pub item",
    "macro invoked at item level, defined elsewhere": "FORBIDDEN",
    "pub inside thread_local!": "FORBIDDEN",
    "pub fn inside an extern block": "FORBIDDEN",
    "inherent impl of a gate type outside family/qwen36": "a gate type is named OUTSIDE family/qwen36/",
    "trait impl for Box<GateReport>": "EXTRA trait impl",
    "impl From<GateReport> for String": "EXTRA trait impl",
    "run_step_gate with an extra parameter": "SIGNATURE of run_step_gate changed",
    "unsafe run_step_gate": "SIGNATURE of run_step_gate changed",
    "pub split across lines": "EXTRA pub item",
    "pub after an attribute": "EXTRA pub item",
    "pub after a multi-line attribute": "EXTRA pub item",
    "pub async unsafe fn": "EXTRA pub item",
    "pub extern \"C\" fn": "EXTRA pub item",
    "pub use": "EXTRA pub item",
    "pub use glob": "EXTRA pub item",
    "macro_rules": "FORBIDDEN",
    "macro_export": "FORBIDDEN",
    "doc(hidden) on a private item": "FORBIDDEN",
    "no_mangle": "FORBIDDEN",
    "pub item under a feature gate": "EXTRA pub item",
    "pub item in a cfg(not(test)) module": "EXTRA pub item",
    "pub method on a gate type": "EXTRA pub item",
    "pub method on another type": "EXTRA pub item",
    "pub method in a trait impl block of a gate type": "EXTRA pub item",
    "extra trait impl on a gate type": "EXTRA trait impl",
    "generic extra trait impl on a gate type": "EXTRA trait impl",
    "pub field on a gate type": "EXTRA pub item",
    "pub tuple-less field after attributes": "EXTRA pub item",
    "pub const and static, enum, trait, type, union": "EXTRA pub item",
    "qwen36 still pub(crate)": "the qwen36 declaration must be exactly one",
    "qwen36 declared twice": "the qwen36 declaration must be exactly one",
    "cuda module missing pub": "MISSING pub item",
    "gate function removed": "MISSING pub item",
    "run_step_gate moved into an impl (a method, not the function)": "MISSING pub item",
    "FromStr impl removed": "MISSING trait impl",
    "pub item in a new nested file": "EXTRA pub item",
    "pub item in a new deep directory": "EXTRA pub item",
    "duplicate fn run_step_gate with the pinned signature": "DUPLICATE pub item",
    "duplicate struct GateReport": "DUPLICATE pub item",
    "pub use of qwen36 in family/other.rs": "RE-EXPORT",
    "pub mod other in family/mod.rs plus a pub use of qwen36 in family/other.rs": "RE-EXPORT",
    "pub(crate) use of run_step_gate in family/other.rs": "RE-EXPORT",
}


def build(name, files):
    d = root / name.replace(" ", "_")
    for k, v in {**BASE, **files}.items():
        p = d / k
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(v)
    return d


def stop_ok(r):
    """The documented typed STOP: rc 2, exactly one message line, no traceback."""
    return r.returncode == 2 and "Traceback" not in r.stderr and len(r.stdout.splitlines()) == 1 and r.stdout.startswith("STOP")


def main():
    ok = True
    for name, (files, should_fail) in CASES.items():
        d = build(name, files)
        r = subprocess.run([sys.executable, "-B", "-I", str(HERE / "public_items.py"), str(d)], capture_output=True, text=True)
        expect = EXPECT.get(name)
        if should_fail:                        # rc 1 exactly AND the output names the intended rule
            good = r.returncode == 1 and bool(expect) and expect in r.stdout and "Traceback" not in r.stderr
        else:                                  # rc 0 exactly
            good = r.returncode == 0 and "Traceback" not in r.stderr
        ok &= good
        print(f"{'ok  ' if good else 'FAIL'} {name}: rc={r.returncode} expect={expect!r} {[l for l in r.stdout.splitlines() if 'EXTRA' in l or 'MISSING' in l or 'DUPLICATE' in l or 'FORBIDDEN' in l or 'RE-EXPORT' in l or 'SIGNATURE' in l or 'declarations' in l][:1]}")
    # base-rev check on a throw-away git repository: family/mod.rs may differ from the base only by qwen36
    repo = root / "gitrepo"
    subprocess.run(["rm", "-rf", str(repo)], check=True)
    (repo / "engine/src/family").mkdir(parents=True)
    g = lambda *a: subprocess.run(["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t", *a], capture_output=True, text=True, check=True)  # noqa: E731
    g("init", "-q")
    (repo / "engine/src/family/mod.rs").write_text("pub mod keep;\nmod qwen36;\n")
    g("add", "-A"); g("commit", "-q", "-m", "base")
    rev = g("rev-parse", "HEAD").stdout.strip()
    for label, fam, want, expect in (("only qwen36 added", "pub mod keep;\npub mod qwen36;\n", 0, None),
                                     ("another pub mod added", "pub mod keep;\npub mod qwen36;\npub mod sneaky;\n", 1, "new pub item besides qwen36"),
                                     ("an existing pub mod removed", "pub mod qwen36;\n", 1, "pub item removed")):
        d = build("base " + label, {"family/mod.rs": fam})
        r = subprocess.run([sys.executable, "-B", "-I", str(HERE / "public_items.py"), str(d), "--base-rev", rev, "--repo", str(repo)], capture_output=True, text=True)
        good = (r.returncode == 0 and "Traceback" not in r.stderr) if want == 0 else (r.returncode == 1 and expect in r.stdout and "Traceback" not in r.stderr)
        ok &= good
        print(f"{'ok  ' if good else 'FAIL'} base-rev: {label}: rc={r.returncode} (want {want}) expect={expect!r}")
    # 16c F10: a hung `git show` (the base-rev read) is a typed STOP, never an uncaught TimeoutExpired
    hung = root / "hung-bin"; hung.mkdir(exist_ok=True)
    (hung / "git").write_text("#!/usr/bin/env bash\nsleep 30\n"); (hung / "git").chmod(0o755)
    real_run = pi.subprocess.run
    pi.subprocess.run = lambda cmd, **k: real_run(cmd, **{**k, "timeout": 0.2})   # shrink the 120 s; same raise site
    saved = os.environ["PATH"]; os.environ["PATH"] = f"{hung}:{saved}"
    why = None
    try:
        pi.base_text(str(repo), rev)
    except SystemExit as e:
        why = str(e)
    except subprocess.TimeoutExpired:
        why = "UNCAUGHT TimeoutExpired"
    finally:
        pi.subprocess.run = real_run; os.environ["PATH"] = saved
    good = why is not None and why.startswith("STOP")
    ok &= good
    print(f"{'ok  ' if good else 'FAIL'} base-rev: a hung git show -> typed STOP (got {why!r})")
    r = subprocess.run([sys.executable, "-B", "-I", str(HERE / "public_items.py"), str(root / "nope")], capture_output=True, text=True)
    good = stop_ok(r) and "has no family/mod.rs" in r.stdout
    ok &= good
    print(f"{'ok  ' if good else 'FAIL'} missing source dir -> typed STOP: rc={r.returncode}")
    d = build("cli misuse", {})                # CLI misuse -> the documented typed STOP (rc 2, one line, no traceback)
    for label, args, expect in (("--repo with no value", [str(d), "--repo"], "STOP: --repo needs a value"),
                                ("--write with no value", [str(d), "--write"], "STOP: --write needs a value")):
        r = subprocess.run([sys.executable, "-B", "-I", str(HERE / "public_items.py"), *args], capture_output=True, text=True)
        good = stop_ok(r) and expect in r.stdout
        ok &= good
        print(f"{'ok  ' if good else 'FAIL'} {label} -> typed STOP: rc={r.returncode} expect={expect!r}")
    r = subprocess.run([sys.executable, "-B", "-I", str(HERE / "selftest_public_items.py")], capture_output=True, text=True)  # no argv: the child exits at the guard, no recursion
    expect = "STOP: usage: selftest_public_items.py SCRATCH_DIR"
    good = stop_ok(r) and expect in r.stdout
    ok &= good
    print(f"{'ok  ' if good else 'FAIL'} selftest with no argv -> typed STOP: rc={r.returncode} expect={expect!r}")
    print("PUBLIC-ITEMS SELFTEST", "OK" if ok else "FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
