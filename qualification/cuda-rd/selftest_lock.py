#!/usr/bin/env python3
"""CPU self-test of lockcheck.py (the qualification Cargo.lock agrees with the engine's). Usage: python3 -I selftest_lock.py SCRATCH_DIR"""
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
if len(sys.argv) < 2:                          # CLI misuse: the typed STOP, never a traceback
    print("STOP: usage: python3 -I selftest_lock.py SCRATCH_DIR")
    sys.exit(2)
S = Path(sys.argv[1]) / "lock"
S.mkdir(parents=True, exist_ok=True)
results = []


def check(name, cond):
    results.append(cond)
    print(("ok   " if cond else "FAIL ") + name)


def run(cmd):
    return subprocess.run(cmd, capture_output=True, text=True)


def main():
    e, q = S / "engine.lock", S / "qual.lock"
    e.write_text('[[package]]\nname = "a"\nversion = "1.0.0"\nsource = "registry+x"\n\n[[package]]\nname = "b"\nversion = "2.0.0"\nsource = "registry+x"\n')
    q.write_text('[[package]]\nname = "a"\nversion = "1.0.0"\nsource = "registry+x"\n\n[[package]]\nname = "cuda-rd"\nversion = "0.1.0"\n')
    run_lock = lambda: run([sys.executable, "-I", str(HERE / "lockcheck.py"), str(q), str(e)])  # noqa: E731
    check("lockcheck: shared packages equal and only cuda-rd extra -> AGREE (exit 0)", run_lock().returncode == 0)
    q.write_text(q.read_text().replace('version = "1.0.0"', 'version = "1.0.1"'))
    check("lockcheck: a shared package at another version -> exit 1", run_lock().returncode == 1)
    q.write_text('[[package]]\nname = "a"\nversion = "1.0.0"\nsource = "registry+x"\n\n[[package]]\nname = "zz"\nversion = "0.1.0"\n')
    check("lockcheck: a package only in the qualification lock -> exit 1", run_lock().returncode == 1)
    check("lockcheck: missing file -> typed STOP", run([sys.executable, "-I", str(HERE / "lockcheck.py"), str(S / "none"), str(e)]).returncode == 2)
    real = HERE / "Cargo.lock"
    check("the committed Cargo.lock parses and has the cuda-rd package", real.is_file() and any(k[0] == "cuda-rd" for k in __import__("importlib").import_module("lockcheck").packages(real)))
    # sp7b: the entries are keyed by (name, version, source) and the checksums of shared entries are compared
    a1 = '[[package]]\nname = "a"\nversion = "1.0.0"\nsource = "registry+x"\nchecksum = "aaa"\n'
    a2 = '[[package]]\nname = "a"\nversion = "2.0.0"\nsource = "registry+x"\nchecksum = "bbb"\n'
    cuda = '[[package]]\nname = "cuda-rd"\nversion = "0.1.0"\n'
    e.write_text(a1)
    q.write_text('[[package]]\nname = "a"\nversion = "1.0.0"\nsource = "registry+y"\nchecksum = "aaa"\n' + cuda)
    check("lockcheck: a shared package with another source -> exit 1", run_lock().returncode == 1)
    e.write_text(a1)
    q.write_text('[[package]]\nname = "a"\nversion = "1.0.0"\nsource = "registry+x"\nchecksum = "zzz"\n' + cuda)
    check("lockcheck: a shared package with another checksum -> exit 1", run_lock().returncode == 1)
    e.write_text(a1 + a2)
    q.write_text(a1 + a2 + cuda)
    r = run_lock()
    check("lockcheck: one package at two versions is two entries -> AGREE with the counts", r.returncode == 0 and "3 entries in the qualification lock, 2 shared" in r.stdout)
    e.write_text(a2)
    q.write_text(a1 + a2 + cuda)
    check("lockcheck: a shared package dropped from the engine lock -> exit 1", run_lock().returncode == 1)
    # 16c R1: a lock with no entries is refused, never "0 entries ... AGREE"
    q.write_text("")
    r = run_lock()
    check("lockcheck: an empty qualification lock -> typed STOP (exit 2), never AGREE", r.returncode == 2 and "no package entries" in r.stdout)
    q.write_text(a1 + cuda)
    e.write_text("this file has no package block at all\n")
    r = run_lock()
    check("lockcheck: a block-less lock (the engine one too) -> typed STOP (exit 2)", r.returncode == 2 and "engine" in r.stdout)
    r = run([sys.executable, "-I", str(HERE / "selftest_lock.py")])
    check("selftest_lock: run without SCRATCH_DIR -> typed STOP (exit 2), never a traceback", r.returncode == 2 and "STOP" in r.stdout and "Traceback" not in r.stderr)
    # 16c F3: a [[patch.unused]] block is not a package (the real lock's mlx-sys entry): never counted, never compared
    e.write_text(a1 + a2 + '[[package]]\nname = "mlx-sys"\nversion = "0.6.0"\nchecksum = "engine-side"\n')
    q.write_text(a1 + a2 + cuda + '[[patch.unused]]\nname = "mlx-sys"\nversion = "0.6.0"\nchecksum = "differs"\n')
    r = run_lock()
    check("lockcheck: a [[patch.unused]] block is ignored even when its name and version match an engine package",
          r.returncode == 0 and "3 entries in the qualification lock, 2 shared" in r.stdout)
    real_pkgs = __import__("importlib").import_module("lockcheck").packages(real)
    check("the committed Cargo.lock is 136 package entries (the [[patch.unused]] mlx-sys block is not one)", len(real_pkgs) == 136)

    print("LOCK SELFTEST", "OK" if all(results) else "FAILED", f"({len(results)} checks)")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
