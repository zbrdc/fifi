#!/usr/bin/env python3
"""Measure engine line coverage by manual LLVM instrumentation.

cargo-llvm-cov does not run on the Mac. This tool builds the engine test
binaries with `-C instrument-coverage`, runs every test binary, merges the raw
profiles, and reports line coverage with llvm-profdata and llvm-cov. It is a
measurement tool. The engine never calls it.

Phases:
  run     build the test binaries and run them. Needs no LLVM tool.
  report  merge the raw profiles and write the report and summary.json.
          Run it on any host that has LLVM tools matched to the rustc that
          built the binaries.
  all     run, then report (default).

Rules the tool enforces:
  * It writes nothing into the worktree. The scratch directory, the target
    directory, the cargo home, and the output directory must lie outside the
    worktree. Every build and test process gets LLVM_PROFILE_FILE in the
    output directory. The tool checks the worktree before and after the run.
  * The major LLVM version of llvm-profdata and llvm-cov must equal the LLVM
    version of the rustc that built the binaries. A mismatch is a recorded
    failure. The report then does not run, unless --allow-llvm-mismatch asks
    for a diagnostic report. A diagnostic report is still a recorded failure.
    With --phase all the run still happens, so a matched host can report later.
  * The coverage targets (60 percent per file, 80 percent aggregate) are
    constants. They are observations in summary.json, not gate results.

Python standard library only. The tool supports Python 3.8 and later.
"""

import argparse
import hashlib
import json
import os
import platform
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = "mac-native-coverage/1"

# Targets from AGENTS.md. They are constants on purpose. Do not add options
# that lower them.
TARGET_FILE_PERCENT = 60.0
TARGET_AGGREGATE_PERCENT = 80.0

# llvm-cov reports "N functions have mismatched data" when it combines binaries
# that hold the same function with different data. It then drops the counters
# of those functions, so the result understates coverage. A profile from another
# build mismatches most functions. A result fails when N exceeds this fraction
# of all mapped functions. The value is a tool heuristic. Smaller counts are
# recorded as warnings.
MISMATCH_FAILURE_FRACTION = 0.05

# The combined expression is byte-identical to the S8 script expression
# "(/\.cargo|cargo-home|/registry/|/rustc/|/vendor/|/tests/|cargo/registry)".
DEFAULT_EXCLUDES = [
    ("cargo_home", r"/\.cargo"),
    ("cargo_home_directory", r"cargo-home"),
    ("registry_sources", r"/registry/"),
    ("rust_std_sources", r"/rustc/"),
    ("vendored_crates", r"/vendor/"),
    ("integration_tests", r"/tests/"),
    ("cargo_registry", r"cargo/registry"),
]

SOURCE_SUFFIXES = (".rs", ".c", ".cc", ".cpp", ".cxx", ".h", ".hpp", ".m", ".mm")
RUST_SUFFIX = ".rs"
REMOVED_ENV = ("CARGO_ENCODED_RUSTFLAGS", "LLVM_PROFILE_FILE", "RUSTC_WRAPPER", "RUSTC_WORKSPACE_WRAPPER")


class Refuse(Exception):
    """The tool refuses to start. It has run nothing and written nothing."""


# ----------------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------------


def utc_now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(path):
    digest = hashlib.sha256()
    with open(str(path), "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


def resolve(path):
    return Path(path).expanduser().resolve()


def is_within(child, parent):
    try:
        Path(child).relative_to(parent)
        return True
    except ValueError:
        return False


def quote(argv):
    return " ".join(shlex.quote(str(part)) for part in argv)


def capture(argv, env=None, cwd=None, timeout=300):
    """Run a short read-only command and keep its text output."""
    try:
        done = subprocess.run(
            [str(a) for a in argv], env=env, cwd=str(cwd) if cwd else None,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            universal_newlines=True, timeout=timeout)
        return {"argv": [str(a) for a in argv], "exit": done.returncode,
                "stdout": done.stdout, "stderr": done.stderr}
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"argv": [str(a) for a in argv], "exit": None, "stdout": "",
                "stderr": "%s: %s" % (type(error).__name__, error)}


def run_to_file(argv, env, cwd, log_path, timeout, stderr_path=None):
    """Run a long command. Send output to files. Kill only this command's group."""
    started = time.time()
    timed_out = False
    with open(str(log_path), "w") as out:
        err = open(str(stderr_path), "w") if stderr_path else subprocess.STDOUT
        try:
            child = subprocess.Popen(
                [str(a) for a in argv], env=env, cwd=str(cwd), stdout=out,
                stderr=err, start_new_session=True)
            try:
                code = child.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
                os.killpg(child.pid, signal.SIGKILL)
                code = child.wait()
        except OSError as error:
            out.write("%s: %s\n" % (type(error).__name__, error))
            code = None
        finally:
            if stderr_path:
                err.close()
    return {"argv": [str(a) for a in argv], "exit": code,
            "seconds": round(time.time() - started, 3), "timed_out": timed_out}


def llvm_version(text):
    """Return (major, minor, patch) from tool output, or None."""
    match = re.search(r"LLVM version:? (\d+)(?:\.(\d+))?(?:\.(\d+))?", text or "")
    if not match:
        return None
    return tuple(int(part) if part else 0 for part in match.groups())


def percent(covered, count):
    return round(100.0 * covered / count, 2) if count else None


def combined_exclude(extra):
    names = [regex for _, regex in DEFAULT_EXCLUDES]
    names.extend(regex for _, regex in extra)
    return "(" + "|".join(names) + ")"


# ----------------------------------------------------------------------------
# Command line
# ----------------------------------------------------------------------------

EPILOG = """
examples:
  Mac, one host, S8 settings (fails on this Mac while only Apple LLVM 21 exists):
    mac_native_coverage.py --worktree /Users/dan/fifi/wrktr/T/W --scratch /Users/dan/fifi/scrch/T/W \\
      --name native-cov-HEAD --expect-head FULLSHA --cargo-home /Users/dan/fifi/scrch/T/W/cargo-home \\
      --path-prepend /Users/dan/fifi/scrch/9071b86/p0a04-native-gate/environment/build-tools/bin \\
      --features native-fault-qualification --fixture ELEMENTWISE_FIXTURES=/path/elementwise-fixtures.json \\
      --fixture RMS_NORM_FIXTURES=/path/rms-fixtures-final.json --fixture LAYER_PARITY_FIXTURES=/path/frozen-layer-parity-fixtures.json

  Matched tools: run on the Mac (--phase run), copy the output directory and
  the test binaries to a host with matched LLVM tools, then report there:
    mac_native_coverage.py --phase report --scratch DIR --name native-cov-HEAD \\
      --binaries-dir DIR/binaries --llvm-tools-dir .../lib/rustlib/HOST/bin --report-tag matched

  Show the commands without running anything: add --dry-run.
"""


def parse_pair(text, flag):
    if "=" not in text:
        raise argparse.ArgumentTypeError("%s needs NAME=VALUE" % flag)
    name, value = text.split("=", 1)
    if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", name):
        raise argparse.ArgumentTypeError("%s has an invalid name: %s" % (flag, name))
    return name, value


def build_parser():
    p = argparse.ArgumentParser(
        description="Measure engine line coverage by manual LLVM instrumentation.",
        epilog=EPILOG, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--phase", choices=("all", "run", "report"), default="all",
                   help="run builds and executes; report merges and reports; all does both (default)")
    p.add_argument("--dry-run", action="store_true",
                   help="print the commands and checks; execute nothing and write nothing")
    p.add_argument("--worktree", help="Git worktree of the source under test (required for run and all)")
    p.add_argument("--scratch", required=True, help="task scratch directory outside the worktree")
    p.add_argument("--name", required=True,
                   help="output directory name under --scratch (new for run and all, existing for report)")
    p.add_argument("--expect-head", help="full or abbreviated HEAD the worktree must have")
    p.add_argument("--allow-dirty", action="store_true",
                   help="continue on a dirty worktree; the result is a recorded failure")
    p.add_argument("--manifest-dir", help="crate directory relative to the worktree (default: engine)", default="engine")
    p.add_argument("--target-dir", help="CARGO_TARGET_DIR (default: SCRATCH/native-cov-target)")
    p.add_argument("--seed-target-dir", help="copy this target directory to --target-dir when --target-dir is absent")
    p.add_argument("--cargo-home", help="CARGO_HOME (default: inherit)")
    p.add_argument("--path-prepend", action="append", default=[], metavar="DIR",
                   help="directory added at the front of PATH (repeatable)")
    p.add_argument("--features", default="", help="comma-separated cargo features")
    p.add_argument("--online", action="store_true", help="omit --offline from cargo (default: offline and locked)")
    p.add_argument("--fixture", action="append", default=[], metavar="NAME=PATH", type=lambda t: parse_pair(t, "--fixture"),
                   help="set env NAME to an existing file and record its SHA-256 (repeatable)")
    p.add_argument("--env", action="append", default=[], metavar="NAME=VALUE", type=lambda t: parse_pair(t, "--env"),
                   help="set another env variable for build and test runs (repeatable)")
    p.add_argument("--no-cpp-instrumentation", action="store_true",
                   help="do not set INFERENCE_NATIVE_COVERAGE=1 (build.rs then leaves C++ uninstrumented)")
    p.add_argument("--test-threads", type=int, default=1, help="value of --test-threads for each test binary (default 1)")
    p.add_argument("--test-arg", action="append", default=[], metavar="ARG", help="extra test binary argument (repeatable)")
    p.add_argument("--test-timeout", type=int, default=3600, help="seconds allowed for each test binary (default 3600)")
    p.add_argument("--exclude", action="append", default=[], metavar="LABEL=REGEX", type=lambda t: parse_pair(t, "--exclude"),
                   help="add an --ignore-filename-regex alternative (repeatable)")
    p.add_argument("--include-prefix", action="append", default=[], metavar="PATH",
                   help="source prefix that counts as in scope (default: WORKTREE/engine/src/); repeatable")
    p.add_argument("--report-tag", default="default", help="report directory suffix: OUT/report-TAG (default: default)")
    p.add_argument("--llvm-profdata", help="explicit llvm-profdata path")
    p.add_argument("--llvm-cov", help="explicit llvm-cov path")
    p.add_argument("--llvm-tools-dir", help="directory that holds llvm-profdata and llvm-cov")
    p.add_argument("--allow-llvm-mismatch", action="store_true",
                   help="write a diagnostic report despite an LLVM major mismatch (still a recorded failure)")
    p.add_argument("--binaries-dir", help="report phase: directory with copies of the test binaries (matched by name and SHA-256)")
    p.add_argument("--existing-profdata", help="report phase: use this merged profile instead of merging raw profiles")
    return p


# ----------------------------------------------------------------------------
# Plan: paths, environment, commands. Pure computation, no side effects.
# ----------------------------------------------------------------------------


class Plan(object):
    pass


def make_plan(args):
    plan = Plan()
    plan.args = args
    plan.phase = args.phase
    plan.worktree = resolve(args.worktree) if args.worktree else None
    if Path(args.name).name != args.name or args.name in ("", ".", ".."):
        raise Refuse("--name must be a plain directory name: %s" % args.name)
    plan.scratch = resolve(args.scratch)
    plan.out = plan.scratch / args.name
    plan.report_dir = plan.out / ("report-" + args.report_tag)
    plan.crate = (plan.worktree / args.manifest_dir) if plan.worktree else None
    plan.target = resolve(args.target_dir) if args.target_dir else plan.scratch / "native-cov-target"
    plan.cargo_home = resolve(args.cargo_home) if args.cargo_home else None
    plan.tmp = plan.scratch / "tmp"
    plan.features = [f for f in re.split(r"[,\s]+", args.features) if f]
    plan.fixtures = [(n, str(resolve(v))) for n, v in args.fixture]
    plan.excludes = list(args.exclude)
    plan.exclude_regex = combined_exclude(plan.excludes)
    plan.prof_dir = plan.out / "prof"
    plan.prof_build_dir = plan.out / "prof-build"
    plan.include_prefixes = [str(p) for p in args.include_prefix]
    return plan


def child_env(plan, profile_file):
    env = dict(os.environ)
    for name in REMOVED_ENV:
        env.pop(name, None)
    if plan.args.path_prepend:
        env["PATH"] = os.pathsep.join([str(resolve(p)) for p in plan.args.path_prepend] + [env.get("PATH", "")])
    if plan.cargo_home:
        env["CARGO_HOME"] = str(plan.cargo_home)
    env["CARGO_TARGET_DIR"] = str(plan.target)
    for name in ("TMPDIR", "TMP", "TEMP"):
        env[name] = str(plan.tmp)
    env["RUSTFLAGS"] = "-C instrument-coverage"
    if not plan.args.no_cpp_instrumentation:
        env["INFERENCE_NATIVE_COVERAGE"] = "1"
    for name, value in plan.fixtures:
        env[name] = value
    for name, value in plan.args.env:
        env[name] = value
    env["LLVM_PROFILE_FILE"] = str(profile_file)
    return env


def env_delta(plan):
    """The variables this tool sets, for the dry run and the record."""
    env = child_env(plan, plan.prof_dir / "<test-binary>" / "%p-%m.profraw")
    names = ["PATH", "CARGO_HOME", "CARGO_TARGET_DIR", "TMPDIR", "TMP", "TEMP", "RUSTFLAGS",
             "INFERENCE_NATIVE_COVERAGE", "LLVM_PROFILE_FILE"]
    names += [n for n, _ in plan.fixtures] + [n for n, _ in plan.args.env]
    delta = {}
    for name in names:
        if name in env and (name not in os.environ or os.environ.get(name) != env[name] or name == "LLVM_PROFILE_FILE"):
            delta[name] = env[name]
    return delta


def cargo_build_argv(plan):
    argv = ["cargo", "test", "--no-run", "--message-format=json", "--locked"]
    if not plan.args.online:
        argv.append("--offline")
    if plan.features:
        argv += ["--features", ",".join(plan.features)]
    return argv


def test_argv(plan, exe):
    return [str(exe), "--test-threads=%d" % plan.args.test_threads] + list(plan.args.test_arg)


def path_checks(plan):
    """Return (name, ok, detail) tuples for the no-write-to-worktree rule."""
    checks = []
    if plan.worktree is None:
        return checks
    for label, path in (("scratch", plan.scratch), ("output", plan.out), ("target-dir", plan.target),
                        ("cargo-home", plan.cargo_home), ("tmp", plan.tmp)):
        if path is None:
            continue
        inside = is_within(path, plan.worktree)
        checks.append(("%s outside worktree" % label, not inside, str(path)))
    return checks


# ----------------------------------------------------------------------------
# Dry run
# ----------------------------------------------------------------------------


def dry_run(plan):
    a = plan.args
    lines = ["DRY RUN. The tool executes nothing and writes nothing.", ""]
    lines.append("phase: %s" % plan.phase)
    lines.append("worktree: %s" % (plan.worktree or "(not given)"))
    lines.append("scratch: %s" % plan.scratch)
    lines.append("output directory: %s" % plan.out)
    lines.append("report directory: %s" % plan.report_dir)
    lines.append("")
    refused = False
    if plan.phase in ("run", "all"):
        if plan.worktree is None:
            lines.append("REFUSE: --worktree is required for phase %s" % plan.phase)
            refused = True
        lines.append("checks that the real run makes:")
        for name, ok, detail in path_checks(plan):
            lines.append("  %s: %s (%s)" % ("PASS" if ok else "REFUSE", name, detail))
            refused = refused or not ok
        lines.append("  the output directory must not exist (%s)" % ("exists now" if plan.out.exists() else "absent now"))
        lines.append("  the worktree must be clean before and after the run; no *.profraw file may appear in it")
        if a.expect_head:
            lines.append("  HEAD must start with %s" % a.expect_head)
        for name, path in plan.fixtures:
            lines.append("  fixture %s must exist: %s" % (name, path))
        lines.append("")
        lines.append("environment set for build and test runs (other variables are inherited):")
        for name, value in sorted(env_delta(plan).items()):
            lines.append("  %s=%s" % (name, shlex.quote(value)))
        lines.append("  removed if present: %s" % ", ".join(REMOVED_ENV))
        lines.append("")
        crate = plan.crate or Path("<worktree>") / a.manifest_dir
        lines.append("step 1. record source and toolchain identity (read-only)")
        lines.append("  git -C %s rev-parse HEAD" % (plan.worktree or "<worktree>"))
        lines.append("  git -C %s rev-parse HEAD:%s" % (plan.worktree or "<worktree>", a.manifest_dir))
        lines.append("  git -C %s status --porcelain --untracked-files=all" % (plan.worktree or "<worktree>"))
        lines.append("  (cd %s && rustc -Vv && rustc --print sysroot && cargo -Vv)" % crate)
        if not a.no_cpp_instrumentation:
            lines.append("  /usr/bin/clang++ --version   (build.rs compiles the C++ guard with this compiler on macOS)")
        if a.seed_target_dir and not plan.target.exists():
            copy = ["cp", "-Rc", a.seed_target_dir, str(plan.target)] if sys.platform == "darwin" else \
                ["cp", "-R", "--reflink=auto", a.seed_target_dir, str(plan.target)]
            lines.append("step 2. seed the target directory")
            lines.append("  " + quote(copy))
        lines.append("step 3. build the instrumented test binaries (profile file: %s)" % (plan.prof_build_dir / "%p-%m.profraw"))
        lines.append("  (cd %s && %s > %s 2> %s)" % (crate, quote(cargo_build_argv(plan)),
                                                      plan.out / "build.json", plan.out / "build.stderr"))
        lines.append("step 4. run each test binary that cargo reports (profile file: %s)" %
                     (plan.prof_dir / "<test-binary>" / "%p-%m.profraw"))
        lines.append("  (cd %s && %s > %s 2>&1)" % (crate, quote(test_argv(plan, "<test-binary>")),
                                                    plan.out / "logs" / "<test-binary>.log"))
        lines.append("step 5. write %s" % (plan.out / "run-manifest.json"))
        lines.append("")
    if plan.phase in ("report", "all"):
        lines.append("tool selection (report phase): first match wins; an explicit path always wins")
        lines.append("  order: --llvm-profdata/--llvm-cov, --llvm-tools-dir, rustc sysroot lib/rustlib/HOST/bin, xcrun --find (macOS), PATH")
        lines.append("  both tools must have the LLVM major version of the rustc that built the binaries")
        lines.append("  a mismatch records a failure and skips the report unless --allow-llvm-mismatch is set")
        profdata = a.existing_profdata or plan.report_dir / "merged.profdata"
        lines.append("step 6. merge raw profiles" if not a.existing_profdata else "step 6. use the given merged profile")
        if not a.existing_profdata:
            lines.append("  %s" % quote(["<llvm-profdata>", "merge", "-sparse", "--input-files=" + str(plan.report_dir / "profraw-files.txt"),
                                          "-o", profdata]))
        objects = ["<test-binary-1>", "-object", "<test-binary-2>", "..."]
        ignore = "--ignore-filename-regex=" + plan.exclude_regex
        lines.append("step 7. text report")
        lines.append("  %s > %s" % (quote(["<llvm-cov>", "report", "-instr-profile=" + str(profdata)] + objects + [ignore]),
                                     plan.report_dir / "report.txt"))
        lines.append("step 8. JSON summary from the same inputs")
        lines.append("  %s > %s" % (quote(["<llvm-cov>", "export", "-summary-only", "-instr-profile=" + str(profdata)] + objects + [ignore]),
                                     plan.report_dir / "export-summary.json"))
        lines.append("step 9. write %s" % (plan.report_dir / "summary.json"))
        lines.append("")
        lines.append("excluded path patterns (one ignore-filename-regex):")
        lines.append("  " + plan.exclude_regex)
        lines.append("in-scope prefixes: " + (", ".join(plan.include_prefixes) or "WORKTREE/%s/src/ (default)" % a.manifest_dir))
        lines.append("")
    lines.append("targets in summary.json: %.0f percent per file, %.0f percent aggregate (fixed constants; observation only)"
                 % (TARGET_FILE_PERCENT, TARGET_AGGREGATE_PERCENT))
    lines.append("")
    lines.append("DRY RUN RESULT: " + ("the real run would refuse" if refused else "the plan passes its path checks"))
    print("\n".join(lines))
    return 2 if refused else 0


# ----------------------------------------------------------------------------
# Identity records
# ----------------------------------------------------------------------------


def git(plan, *argv):
    return capture(["git", "-C", str(plan.worktree)] + list(argv))


def tool_identity(path):
    path = Path(path)
    info = {"path": str(path), "exists": path.exists()}
    if path.exists():
        real = path.resolve()
        info["realpath"] = str(real)
        try:
            info["sha256"] = sha256_file(real)
            info["bytes"] = real.stat().st_size
        except OSError as error:
            info["sha256_error"] = str(error)
    return info


def gather_identity(plan, env):
    """Read-only facts about the source, the toolchain, and the fixtures."""
    a = plan.args
    head = git(plan, "rev-parse", "HEAD")
    tree = git(plan, "rev-parse", "HEAD:" + a.manifest_dir)
    status = git(plan, "status", "--porcelain", "--untracked-files=all")
    tracked = git(plan, "ls-files", a.manifest_dir + "/src")
    if head["exit"] != 0 or tree["exit"] != 0 or status["exit"] != 0:
        raise Refuse("git identity query failed in %s: %s" % (plan.worktree, head["stderr"].strip() or tree["stderr"].strip()))
    head_sha = head["stdout"].strip()
    if a.expect_head and not head_sha.startswith(a.expect_head.lower()):
        raise Refuse("HEAD is %s, not %s" % (head_sha, a.expect_head))
    dirty = status["stdout"].strip()
    if dirty and not a.allow_dirty:
        raise Refuse("the worktree is not clean:\n%s" % dirty)
    lock = plan.crate / "Cargo.lock"
    ident = {
        "worktree": str(plan.worktree),
        "worktree_as_given": str(Path(a.worktree).expanduser().absolute()),
        "head": head_sha,
        "engine_tree": tree["stdout"].strip(),
        "engine_tree_path": a.manifest_dir,
        "worktree_dirty_before": bool(dirty),
        "worktree_status_before": dirty,
        "cargo_lock_sha256": sha256_file(lock) if lock.is_file() else None,
        "tracked_source_files": [p for p in tracked["stdout"].splitlines() if p.endswith(SOURCE_SUFFIXES)],
    }
    rustc_vv = capture(["rustc", "-Vv"], env=env, cwd=plan.crate)
    sysroot = capture(["rustc", "--print", "sysroot"], env=env, cwd=plan.crate)
    cargo_vv = capture(["cargo", "-Vv"], env=env, cwd=plan.crate)
    if rustc_vv["exit"] != 0 or sysroot["exit"] != 0:
        raise Refuse("rustc is not runnable: %s" % rustc_vv["stderr"].strip())
    host = re.search(r"^host: (\S+)", rustc_vv["stdout"], re.M)
    rustc_path = shutil.which("rustc", path=env.get("PATH"))
    rustc_real = Path(sysroot["stdout"].strip()) / "bin" / "rustc"
    ident["rustc"] = {
        "version_verbose": rustc_vv["stdout"],
        "llvm_version": (re.search(r"^LLVM version: (\S+)", rustc_vv["stdout"], re.M) or [None, None])[1],
        "host": host.group(1) if host else None,
        "sysroot": sysroot["stdout"].strip(),
        "proxy": tool_identity(rustc_path) if rustc_path else None,
        "binary": tool_identity(rustc_real),
    }
    cargo_path = shutil.which("cargo", path=env.get("PATH"))
    ident["cargo"] = {"version_verbose": cargo_vv["stdout"], "proxy": tool_identity(cargo_path) if cargo_path else None,
                      "binary": tool_identity(Path(sysroot["stdout"].strip()) / "bin" / "cargo")}
    if not a.no_cpp_instrumentation and sys.platform == "darwin":
        clang = capture(["/usr/bin/clang++", "--version"], env=env)
        found = capture(["xcrun", "--find", "clang++"], env=env)
        ident["cpp_compiler"] = {
            "note": "engine/build.rs compiles guard.cpp with /usr/bin/clang++ and -fprofile-instr-generate -fcoverage-mapping",
            "version": clang["stdout"], "shim": tool_identity("/usr/bin/clang++"),
            "xcrun_find": found["stdout"].strip(),
            "xcrun_binary": tool_identity(found["stdout"].strip()) if found["stdout"].strip() else None,
        }
    fixtures = []
    for name, path in plan.fixtures:
        p = Path(path)
        if not p.is_file():
            raise Refuse("fixture %s is not a file: %s" % (name, path))
        fixtures.append({"env": name, "path": path, "sha256": sha256_file(p), "bytes": p.stat().st_size})
    ident["fixtures"] = fixtures
    host_info = {"platform": platform.platform(), "machine": platform.machine(), "python": platform.python_version()}
    if sys.platform == "darwin":
        host_info["sw_vers"] = capture(["sw_vers"])["stdout"]
        host_info["xcode"] = capture(["xcode-select", "-p"])["stdout"].strip()
    ident["host"] = host_info
    return ident


# ----------------------------------------------------------------------------
# LLVM tool selection
# ----------------------------------------------------------------------------


def evaluate_tool(name, source, path, rustc_major):
    info = tool_identity(path)
    info.update({"source": source, "tool": name})
    if info["exists"]:
        ran = capture([info["realpath"] if "realpath" in info else path, "--version"])
        info["version_text"] = ran["stdout"] + ran["stderr"]
        ver = llvm_version(info["version_text"])
        info["llvm_version"] = ".".join(str(n) for n in ver) if ver else None
        info["llvm_major"] = ver[0] if ver else None
        info["matches_rustc_major"] = bool(ver) and ver[0] == rustc_major
    else:
        info["matches_rustc_major"] = False
    return info


def candidate_paths(plan, name, sysroot_dirs):
    a = plan.args
    found = []
    explicit = {"llvm-profdata": a.llvm_profdata, "llvm-cov": a.llvm_cov}[name]
    if explicit:
        found.append(("argument", explicit, True))
    if a.llvm_tools_dir:
        found.append(("llvm-tools-dir", str(Path(a.llvm_tools_dir) / name), False))
    for directory in sysroot_dirs:
        found.append(("rustc-sysroot", str(Path(directory) / name), False))
    if sys.platform == "darwin":
        shown = capture(["xcrun", "--find", name])
        if shown["exit"] == 0 and shown["stdout"].strip():
            found.append(("xcrun", shown["stdout"].strip(), False))
    on_path = shutil.which(name)
    if on_path:
        found.append(("PATH", on_path, False))
    return found


def select_tools(plan, rustc_llvm, rustc_host, sysroots):
    """Pick llvm-profdata and llvm-cov. Return the record and the failure reasons.

    sysroots lists rustc sysroots in order. The first is the sysroot of the rustc that built the binaries."""
    ver = llvm_version("LLVM version: " + (rustc_llvm or ""))
    rustc_major = ver[0] if ver else None
    sysroot_dirs = []
    for sysroot in sysroots:
        if sysroot and Path(sysroot, "lib", "rustlib").is_dir():
            preferred = Path(sysroot, "lib", "rustlib", rustc_host or "", "bin")
            others = sorted(Path(sysroot, "lib", "rustlib").glob("*/bin"))
            sysroot_dirs += [d for d in [preferred] + others if d not in sysroot_dirs]
    record = {"rustc_llvm_version": rustc_llvm, "rustc_llvm_major": rustc_major, "candidates": {}, "selected": {}}
    reasons = []
    for name in ("llvm-profdata", "llvm-cov"):
        cands = []
        for source, path, forced in candidate_paths(plan, name, sysroot_dirs):
            item = evaluate_tool(name, source, path, rustc_major)
            item["forced"] = forced
            cands.append(item)
        record["candidates"][name] = cands
        existing = [c for c in cands if c["exists"]]
        forced = [c for c in cands if c["forced"]]
        if forced and not forced[0]["exists"]:
            reasons.append("llvm_tool_missing: the explicit %s path does not exist: %s" % (name, forced[0]["path"]))
            record["selected"][name] = None
            continue
        chosen = forced[0] if forced else None
        if chosen is None:
            chosen = next((c for c in existing if c["matches_rustc_major"]), None)
        if chosen is None and existing:
            chosen = existing[0]
        record["selected"][name] = chosen
        if chosen is None:
            reasons.append("llvm_tool_missing: no %s found" % name)
    sel = record["selected"]
    have_both = all(sel[n] for n in ("llvm-profdata", "llvm-cov"))
    majors = {n: (sel[n] or {}).get("llvm_major") for n in sel}
    ok = have_both and rustc_major is not None and all(m == rustc_major for m in majors.values())
    record["llvm_match"] = {
        "ok": ok,
        "rustc_llvm_major": rustc_major,
        "llvm_profdata_major": majors["llvm-profdata"],
        "llvm_cov_major": majors["llvm-cov"],
        "full_version_equal": have_both and all((sel[n] or {}).get("llvm_version") == rustc_llvm for n in sel),
        "rule": "the major LLVM version of both tools must equal the LLVM version of the rustc that built the binaries",
    }
    if have_both and not ok:
        reasons.append("llvm_major_mismatch: rustc LLVM %s, llvm-profdata %s, llvm-cov %s" % (
            rustc_llvm, (sel["llvm-profdata"] or {}).get("llvm_version"), (sel["llvm-cov"] or {}).get("llvm_version")))
    return record, reasons


# ----------------------------------------------------------------------------
# Phase: run
# ----------------------------------------------------------------------------


def find_profraw(root):
    found = []
    for base, dirs, files in os.walk(str(root)):
        dirs[:] = [d for d in dirs if d != ".git"]
        found.extend(os.path.join(base, f) for f in files if f.endswith(".profraw"))
    return sorted(found)


def write_sums(directory, skip_dirs=(), name="SHA256SUMS"):
    directory = Path(directory)
    rows = []
    for base, dirs, files in os.walk(str(directory)):
        dirs[:] = sorted(d for d in dirs if not (Path(base) == directory and d in skip_dirs))
        for f in sorted(files):
            full = Path(base) / f
            if f == name and Path(base) == directory:
                continue
            rows.append("%s  ./%s" % (sha256_file(full), full.relative_to(directory)))
    (directory / name).write_text("\n".join(rows) + "\n")
    return name


def do_run(plan, ident):
    a = plan.args
    if plan.out.exists():
        raise Refuse("the output directory exists: %s" % plan.out)
    for directory in (plan.out, plan.out / "logs", plan.prof_dir, plan.prof_build_dir, plan.tmp):
        directory.mkdir(parents=True, exist_ok=True)
    if a.seed_target_dir and not plan.target.exists():
        copy = ["cp", "-Rc", a.seed_target_dir, str(plan.target)] if sys.platform == "darwin" else \
            ["cp", "-R", "--reflink=auto", a.seed_target_dir, str(plan.target)]
        seeded = capture(copy, timeout=3600)
        if seeded["exit"] != 0:
            raise Refuse("the target directory seed failed: %s" % seeded["stderr"].strip())
    failures = []
    manifest = {"schema": SCHEMA + "/run", "recorded_at_utc": utc_now(), "tool_sha256": sha256_file(__file__),
                "out": str(plan.out), "identity": ident, "features": plan.features,
                "rustflags": "-C instrument-coverage",
                "cpp_instrumentation": not a.no_cpp_instrumentation,
                "env_set": env_delta(plan), "env_removed": list(REMOVED_ENV),
                "target_dir": str(plan.target), "cargo_home": str(plan.cargo_home) if plan.cargo_home else None}

    build_env = child_env(plan, plan.prof_build_dir / "%p-%m.profraw")
    build_cmd = cargo_build_argv(plan)
    build = run_to_file(build_cmd, build_env, plan.crate, plan.out / "build.json", 7200, plan.out / "build.stderr")
    manifest["build"] = build
    print("build exit %s in %ss" % (build["exit"], build["seconds"]), flush=True)
    binaries = []
    skipped = []
    if build["exit"] == 0:
        crate_manifest = str(plan.crate / "Cargo.toml")
        seen = set()
        for line in (plan.out / "build.json").read_text().splitlines():
            try:
                msg = json.loads(line)
            except ValueError:
                continue
            exe = msg.get("executable") if msg.get("reason") == "compiler-artifact" else None
            if not exe or exe in seen:
                continue
            seen.add(exe)
            same_crate = str(Path(msg.get("manifest_path", "")).resolve()) == str(Path(crate_manifest).resolve())
            if same_crate and msg.get("profile", {}).get("test"):
                binaries.append({"name": Path(exe).name, "path": exe, "target": msg.get("target", {}).get("name"),
                                 "kind": msg.get("target", {}).get("kind"), "sha256": sha256_file(exe),
                                 "bytes": os.path.getsize(exe)})
            else:
                skipped.append({"path": exe, "reason": "not a test-profile executable of the crate under test"})
        if not binaries:
            failures.append("no_test_binaries: cargo reported no test executable")
    else:
        failures.append("build_failed: cargo exit %s" % build["exit"])
    manifest["test_binaries"] = binaries
    manifest["executables_not_attributed"] = skipped

    results = []
    for item in binaries:
        profile_dir = plan.prof_dir / item["name"]
        profile_dir.mkdir(parents=True, exist_ok=True)
        env = child_env(plan, profile_dir / "%p-%m.profraw")
        log = plan.out / "logs" / (item["name"] + ".log")
        outcome = run_to_file(test_argv(plan, item["path"]), env, plan.crate, log, a.test_timeout)
        text = log.read_text(errors="replace")
        counts = {"passed": 0, "failed": 0, "ignored": 0}
        for m in re.finditer(r"test result: \w+\. (\d+) passed; (\d+) failed; (\d+) ignored", text):
            counts["passed"] += int(m.group(1))
            counts["failed"] += int(m.group(2))
            counts["ignored"] += int(m.group(3))
        outcome.update({"name": item["name"], "log": str(log.relative_to(plan.out)), "counts": counts,
                        "skipped_markers": len(re.findall(r"SKIPPED", text)),
                        "profraw_files": len(list(profile_dir.glob("*.profraw")))})
        results.append(outcome)
        print("%s exit %s" % (item["name"], outcome["exit"]), flush=True)
        if outcome["exit"] != 0:
            failures.append("test_failed: %s exit %s%s" % (item["name"], outcome["exit"], " (timeout)" if outcome["timed_out"] else ""))
    manifest["test_runs"] = results
    manifest["test_totals"] = {k: sum(r["counts"][k] for r in results) for k in ("passed", "failed", "ignored")}

    stray = find_profraw(plan.worktree)
    status_after = git(plan, "status", "--porcelain", "--untracked-files=all")
    dirty_after = status_after["stdout"].strip()
    manifest["worktree_after"] = {"dirty": bool(dirty_after), "status": dirty_after, "profraw_files_in_worktree": stray}
    if stray:
        failures.append("profraw_in_worktree: %d file(s)" % len(stray))
    if dirty_after and not ident["worktree_dirty_before"]:
        failures.append("worktree_modified_by_run")
    if ident["worktree_dirty_before"]:
        failures.append("worktree_dirty: the result cannot be reproduced from the commit")
    head_after = git(plan, "rev-parse", "HEAD")["stdout"].strip()
    if head_after != ident["head"]:
        failures.append("head_changed_during_run")
    manifest["failures"] = failures
    manifest["profraw_files"] = len(find_profraw(plan.prof_dir))
    write_json(plan.out / "run-manifest.json", manifest)
    write_sums(plan.out, skip_dirs=[d.name for d in plan.out.glob("report-*")], name="run.SHA256SUMS")
    return manifest


# ----------------------------------------------------------------------------
# Phase: report
# ----------------------------------------------------------------------------


def parse_report_total(text):
    """Return (line_count, line_missed) from the TOTAL row of llvm-cov report."""
    header = None
    for line in text.splitlines():
        if line.startswith("Filename"):
            header = [c.strip() for c in re.split(r"\s{2,}", line.strip())]
        elif line.startswith("TOTAL") and header and "Lines" in header:
            tokens = line.split()
            index = header.index("Lines")
            try:
                return int(tokens[index]), int(tokens[index + 1])
            except (IndexError, ValueError):
                return None
    return None


def verify_sums(directory, name):
    """Recompute the hashes that the run phase listed. Return (checked, problems)."""
    listing = Path(directory) / name
    if not listing.is_file():
        return 0, ["run_sums_missing: %s" % name]
    checked, problems = 0, []
    for line in listing.read_text().splitlines():
        digest, _, rel = line.partition("  ")
        target = Path(directory) / rel
        checked += 1
        if not target.is_file() or sha256_file(target) != digest:
            problems.append("run_file_changed: %s" % rel)
    return checked, problems


def locate_binaries(manifest, binaries_dir):
    located = []
    problems = []
    for item in manifest["test_binaries"]:
        candidates = [Path(binaries_dir) / item["name"]] if binaries_dir else [Path(item["path"])]
        path = next((c for c in candidates if c.is_file()), None)
        if path is None:
            problems.append("binary_missing: %s" % candidates[0])
            continue
        digest = sha256_file(path)
        if digest != item["sha256"]:
            problems.append("binary_changed: %s" % path)
            continue
        located.append(str(path))
    return located, problems


def relative_name(filename, roots):
    for root in roots:
        if filename.startswith(root + "/"):
            return filename[len(root) + 1:]
    return filename


def do_report(plan, manifest, tools, tool_reasons):
    a = plan.args
    report = plan.report_dir
    if report.exists():
        raise Refuse("the report directory exists: %s" % report)
    report.mkdir(parents=True)
    ident = manifest["identity"]
    roots = sorted({ident["worktree"], ident.get("worktree_as_given", ident["worktree"])})
    prefixes = list(plan.include_prefixes) or [r + "/" + ident["engine_tree_path"] + "/src/" for r in roots]
    failures = list(manifest.get("failures", [])) + list(tool_reasons)
    warnings = []
    summary = {"schema": SCHEMA, "recorded_at_utc": utc_now(), "tool_sha256": sha256_file(__file__),
               "scope": "File-level line-coverage observations only. This report does not compute production or "
                        "qualification module coverage under Decision 0034. It is not a gate result. See "
                        "docs/decisions/0034-coverage-accounting.md.",
               "module_accounting": "not_computed",
               "module_accounting_decision": "docs/decisions/0034-coverage-accounting.md",
               "source": {k: ident[k] for k in ("worktree", "head", "engine_tree", "engine_tree_path", "cargo_lock_sha256")},
               "build": {"features": manifest["features"], "rustflags": manifest["rustflags"],
                         "cpp_instrumentation": manifest["cpp_instrumentation"], "env_set": manifest["env_set"],
                         "env_removed": manifest["env_removed"], "target_dir": manifest["target_dir"],
                         "cargo_home": manifest["cargo_home"], "command": manifest["build"]["argv"]},
               "toolchain": {"rustc": ident["rustc"], "cargo": ident["cargo"], "cpp_compiler": ident.get("cpp_compiler"),
                             "host": ident["host"], "llvm": tools},
               "fixtures": ident["fixtures"],
               "tests": {"binaries": manifest["test_binaries"], "runs": manifest["test_runs"], "totals": manifest["test_totals"],
                         "profraw_files": manifest["profraw_files"],
                         "executables_not_attributed": manifest["executables_not_attributed"]},
               "exclusions": {"ignore_filename_regex": plan.exclude_regex,
                              "labels": [{"label": l, "regex": r} for l, r in DEFAULT_EXCLUDES + plan.excludes],
                              "also_not_measured": [
                                  "doctests (cargo test --no-run builds none)",
                                  "executables that are not test-profile artifacts of the crate, including spawned child processes",
                                  "ignored tests, which the test binaries skip",
                                  "code behind features that the feature list leaves off, for example native-ane and its bridge"]},
               "include_prefixes": prefixes}
    cpp = ident.get("cpp_compiler")
    if cpp and llvm_version(cpp["version"]) and tools["llvm_match"]["rustc_llvm_major"] not in (None, llvm_version(cpp["version"])[0]):
        warnings.append("cpp_compiler_llvm_version_differs: C++ coverage data comes from clang++ %s; rustc LLVM is %s. "
                        "No host tool can match both." % (".".join(map(str, llvm_version(cpp["version"]))), ident["rustc"]["llvm_version"]))
    for run in manifest["test_runs"]:
        if run["skipped_markers"]:
            warnings.append("skipped_marker: %s printed SKIPPED %d time(s); a fixture variable may be unset" % (run["name"], run["skipped_markers"]))

    missing = any(tools["selected"][n] is None for n in ("llvm-profdata", "llvm-cov"))
    if missing or (tool_reasons and not a.allow_llvm_mismatch):
        summary.update({"status": "failed", "failure_reasons": failures, "warnings": warnings,
                        "note": "The report did not run because the LLVM tools are missing or do not match the rustc."})
        write_json(report / "summary.json", summary)
        write_sums(report)
        return summary

    checked, sum_problems = verify_sums(plan.out, "run.SHA256SUMS")
    summary["run_directory_check"] = {"listing": "run.SHA256SUMS", "files_checked": checked, "problems": sum_problems[:20]}
    failures.extend(sum_problems[:20])
    binaries, problems = locate_binaries(manifest, a.binaries_dir)
    failures.extend(problems)
    if problems or not binaries:
        summary.update({"status": "failed", "failure_reasons": failures + ([] if binaries else ["no_binaries_to_report"]),
                        "warnings": warnings})
        write_json(report / "summary.json", summary)
        write_sums(report)
        return summary

    profdata_tool = tools["selected"]["llvm-profdata"]["realpath"]
    cov_tool = tools["selected"]["llvm-cov"]["realpath"]
    profraw = find_profraw(plan.out / "prof")
    empties = [p for p in profraw if os.path.getsize(p) == 0]
    if empties:
        failures.append("empty_profraw: %d file(s)" % len(empties))
    commands = {}
    if a.existing_profdata:
        merged = Path(a.existing_profdata)
        commands["merge"] = None
        summary["profile"] = {"source": "existing-profdata", "path": str(merged),
                              "note": "merge skipped; the tool that wrote this profile is not recorded"}
    else:
        listing = report / "profraw-files.txt"
        listing.write_text("\n".join(profraw) + "\n")
        merged = report / "merged.profdata"
        merge_cmd = [profdata_tool, "merge", "-sparse", "--input-files=" + str(listing), "-o", str(merged)]
        done = capture(merge_cmd, timeout=7200)
        (report / "merge.log").write_text(done["stdout"] + done["stderr"])
        commands["merge"] = {"argv": merge_cmd, "exit": done["exit"]}
        summary["profile"] = {"source": "merged from raw profiles", "raw_profile_count": len(profraw),
                              "profraw_list_sha256": sha256_file(listing)}
        if done["exit"] != 0:
            failures.append("merge_failed: exit %s" % done["exit"])
    if not merged.is_file():
        failures.append("merged_profile_missing")
        summary.update({"status": "failed", "failure_reasons": failures, "warnings": warnings, "commands": commands})
        write_json(report / "summary.json", summary)
        write_sums(report)
        return summary
    summary["profile"].update({"merged_profdata": str(merged), "sha256": sha256_file(merged), "bytes": merged.stat().st_size})

    tail = ["-instr-profile=" + str(merged), binaries[0]]
    for extra in binaries[1:]:
        tail += ["-object", extra]
    tail.append("--ignore-filename-regex=" + plan.exclude_regex)
    report_cmd = [cov_tool, "report"] + tail
    export_cmd = [cov_tool, "export", "-summary-only"] + tail
    text = capture(report_cmd, timeout=7200)
    (report / "report.txt").write_text(text["stdout"])
    (report / "report.stderr").write_text(text["stderr"])
    data = capture(export_cmd, timeout=7200)
    (report / "export-summary.json").write_text(data["stdout"])
    (report / "export.stderr").write_text(data["stderr"])
    commands["report"] = {"argv": report_cmd, "exit": text["exit"]}
    commands["export"] = {"argv": export_cmd, "exit": data["exit"]}
    summary["commands"] = commands
    mismatched = {}
    for label, result in (("report", text), ("export", data)):
        if result["exit"] != 0:
            failures.append("%s_failed: exit %s" % (label, result["exit"]))
        stderr_lines = [l for l in result["stderr"].splitlines() if l.strip()]
        found = re.search(r"(\d+) functions? have mismatched data", result["stderr"])
        mismatched[label] = int(found.group(1)) if found else 0
        if stderr_lines:
            warnings.append("%s stderr: %d line(s), first: %s" % (label, len(stderr_lines), stderr_lines[0][:200]))

    files = []
    totals = None
    if data["exit"] == 0:
        try:
            doc = json.loads(data["stdout"])
            entry = doc["data"][0]
            totals = entry["totals"]
            for item in entry["files"]:
                lines = item["summary"]["lines"]
                files.append({"filename": item["filename"], "lines": lines,
                              "functions": item["summary"].get("functions"), "regions": item["summary"].get("regions")})
        except (ValueError, KeyError, IndexError) as error:
            failures.append("export_unreadable: %s" % error)
    total_functions = totals["functions"]["count"] if totals and totals.get("functions") else None
    summary["profile_function_mismatch"] = {
        "report": mismatched["report"], "export": mismatched["export"], "total_functions": total_functions,
        "failure_fraction": MISMATCH_FAILURE_FRACTION,
        "note": "llvm-cov drops the counters of functions whose data differs between combined binaries, so the "
                "result understates coverage. The tool fails the result when the dropped functions exceed the "
                "fraction of all functions and otherwise records a warning."}
    for label, count in mismatched.items():
        if count and (total_functions is None or count > MISMATCH_FAILURE_FRACTION * total_functions):
            failures.append("profile_binary_mismatch: %s stderr reports %d functions with mismatched data" % (label, count))
        elif count:
            warnings.append("profile_function_mismatch: %s dropped the counters of %d of %s functions" % (label, count, total_functions))
    rows, out_of_scope = [], []
    for item in sorted(files, key=lambda f: f["filename"]):
        name = item["filename"]
        rel = relative_name(name, roots)
        row = {"file": rel, "language": "rust" if name.endswith(RUST_SUFFIX) else "other",
               "lines_count": item["lines"]["count"], "lines_covered": item["lines"]["covered"],
               "lines_percent": percent(item["lines"]["covered"], item["lines"]["count"]),
               "lines_percent_llvm": item["lines"].get("percent"),
               "functions_count": (item["functions"] or {}).get("count"), "functions_covered": (item["functions"] or {}).get("covered"),
               "regions_count": (item["regions"] or {}).get("count"), "regions_covered": (item["regions"] or {}).get("covered")}
        rows.append(row)
        if not any(name.startswith(p) for p in prefixes):
            out_of_scope.append(rel)

    def total(selected):
        count = sum(r["lines_count"] for r in selected)
        covered = sum(r["lines_covered"] for r in selected)
        return {"files": len(selected), "lines_count": count, "lines_covered": covered, "lines_percent": percent(covered, count)}

    all_total = total(rows)
    summary["files"] = rows
    summary["totals"] = {"all_files": all_total, "rust_only": total([r for r in rows if r["language"] == "rust"]),
                         "non_rust": total([r for r in rows if r["language"] != "rust"])}
    if totals is not None:
        summary["totals"]["llvm_export_totals_lines"] = totals["lines"]
        if totals["lines"]["count"] != all_total["lines_count"] or totals["lines"]["covered"] != all_total["lines_covered"]:
            failures.append("export_totals_disagree: file rows do not sum to the export totals")
    parsed = parse_report_total(text["stdout"]) if text["exit"] == 0 else None
    if parsed is None:
        warnings.append("report_total_not_parsed: the text report has no readable TOTAL row")
    else:
        summary["totals"]["text_report_total_lines"] = {"count": parsed[0], "missed": parsed[1]}
        if parsed[0] != all_total["lines_count"] or parsed[0] - parsed[1] != all_total["lines_covered"]:
            failures.append("report_export_disagree: text report TOTAL differs from the export")
    if out_of_scope:
        failures.append("files_out_of_scope: %d mapped file(s) lie outside the include prefixes" % len(out_of_scope))
    summary["out_of_scope_files"] = out_of_scope

    mapped = {r["file"] for r in rows}
    summary["tracked_source_files_without_mapping"] = [p for p in ident["tracked_source_files"] if p not in mapped]
    summary["mapped_files_not_tracked"] = sorted(m for m in mapped if m.startswith(ident["engine_tree_path"] + "/src/")
                                                 and m not in set(ident["tracked_source_files"]))

    below = [{"file": r["file"], "lines_percent": r["lines_percent"]} for r in rows
             if r["lines_percent"] is None or r["lines_percent"] < TARGET_FILE_PERCENT]
    rust_total = summary["totals"]["rust_only"]
    summary["targets"] = {
        "per_file_percent": TARGET_FILE_PERCENT, "aggregate_percent": TARGET_AGGREGATE_PERCENT,
        "files_below_per_file_target": below,
        "aggregate_percent_all_files": all_total["lines_percent"],
        "aggregate_meets_target_all_files": bool(all_total["lines_percent"] is not None and all_total["lines_percent"] >= TARGET_AGGREGATE_PERCENT),
        "aggregate_percent_rust_only": rust_total["lines_percent"],
        "aggregate_meets_target_rust_only": bool(rust_total["lines_percent"] is not None and rust_total["lines_percent"] >= TARGET_AGGREGATE_PERCENT),
        "note": "Observation only. The 60 percent module and 80 percent aggregate targets are fixed. "
                "This file-level report does not compute Decision 0034 module accounting or establish a gate result."}
    summary["artifacts"] = {n: {"sha256": sha256_file(report / n)} for n in ("report.txt", "export-summary.json") if (report / n).is_file()}
    summary["status"] = "complete" if not failures else "failed"
    summary["failure_reasons"] = failures
    summary["warnings"] = warnings
    write_json(report / "summary.json", summary)
    write_sums(report)
    return summary


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        plan = make_plan(args)
        if args.dry_run:
            return dry_run(plan)
        if args.phase in ("run", "all"):
            if plan.worktree is None or not (plan.crate / "Cargo.toml").is_file():
                raise Refuse("--worktree must be a checkout with %s/Cargo.toml" % args.manifest_dir)
            for name, ok, detail in path_checks(plan):
                if not ok:
                    raise Refuse("%s: %s" % (name, detail))
            if plan.out.exists():
                raise Refuse("the output directory exists: %s" % plan.out)
            plan.tmp.mkdir(parents=True, exist_ok=True)
            ident = gather_identity(plan, child_env(plan, plan.scratch / "unused.profraw"))
            tools, tool_reasons = None, []
            if args.phase == "all":
                tools, tool_reasons = select_tools(plan, ident["rustc"]["llvm_version"], ident["rustc"]["host"],
                                                   [ident["rustc"]["sysroot"]])
                if tool_reasons:
                    print("LLVM tool check failed: " + "; ".join(tool_reasons), file=sys.stderr, flush=True)
            manifest = do_run(plan, ident)
            print("run phase: %d failure(s)" % len(manifest["failures"]), flush=True)
        else:
            manifest_path = plan.out / "run-manifest.json"
            if not manifest_path.is_file():
                raise Refuse("no run manifest at %s" % manifest_path)
            manifest = json.loads(manifest_path.read_text())
            if plan.report_dir.exists():
                raise Refuse("the report directory exists: %s" % plan.report_dir)
            local = capture(["rustc", "--print", "sysroot"])["stdout"].strip() or None
            sysroots = [manifest["identity"]["rustc"].get("sysroot"), local]
            version = manifest["identity"]["rustc"]["version_verbose"]
            host = re.search(r"^host: (\S+)", version, re.M)
            tools, tool_reasons = select_tools(plan, manifest["identity"]["rustc"]["llvm_version"],
                                               host.group(1) if host else None, sysroots)
        if args.phase == "run":
            return 0 if not manifest["failures"] else 1
        summary = do_report(plan, manifest, tools, tool_reasons)
        print("report phase: status %s" % summary["status"], flush=True)
        for reason in summary.get("failure_reasons", []):
            print("  failure: " + reason, flush=True)
        print("summary: %s" % (plan.report_dir / "summary.json"), flush=True)
        return 0 if summary["status"] == "complete" else 1
    except Refuse as error:
        print("REFUSED: %s" % error, file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
