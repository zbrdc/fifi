#!/usr/bin/env python3
"""RD-10: refuse a qualification binary built in the debug profile or carrying device-debug kernels (nvcc -G).

Usage: check_release_build.py <cuda-rd-run binary> [--out release.json] [--cuobjdump CMD...]
Exit 0 = release, 2 = STOP (a miss, or a typed infrastructure problem). Never a traceback.
Checks:
 1. `<binary> build-info` prints JSON: profile=release, opt_level not 0, debug=false, debug_assertions=false.
 2. `cuobjdump -elf <binary>` (default: the CUDA 12.2 toolkit inside the build image, the toolkit that built it) shows a
    cubin ("Fatbin elf code") and NEITHER marker of device debug: `.nv_debug_info_reg_` or "has debug info".
    NOT keyed on .nv_debug_line_sass / .nv_debug_ptx_txt: -O3 -lineinfo objects carry those too (lane L, CUDA 12.2; lead
    probe, CUDA 13.0; proved again on 12.2 in selftest_release_check.sh).
The JSON written with --out holds the SHA-256 of the examined executable; hold.sh compares it with the one it runs.
"""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

BAD = (".nv_debug_info_reg_", "has debug info")
IMAGE = "inference-cuda-build:cuda12.2-ubuntu22.04"
DEFAULT_TOOL = ["docker", "run", "--rm", "-v", "/media/dan/nvme:/media/dan/nvme", IMAGE, "cuobjdump"]


class Stop(Exception):
    pass


def run(cmd):
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    except (OSError, subprocess.TimeoutExpired) as e:
        raise Stop(f"cannot run {cmd[0]}: {e}") from e
    return p.returncode, p.stdout, p.stderr


def check_info(info):
    miss = []
    if info.get("profile") != "release":
        miss.append(f"profile={info.get('profile')}")
    if str(info.get("opt_level")) in ("0", "None", "unset"):
        miss.append(f"opt_level={info.get('opt_level')}")
    if info.get("debug") not in (False, "false"):
        miss.append(f"debug={info.get('debug')}")
    if info.get("debug_assertions"):
        miss.append("debug_assertions=true")
    return miss


def check_elf(text):
    miss = [b for b in BAD if b in text]
    if "Fatbin elf code" not in text:
        miss.append("no cubin found")
    elif "arch = sm_86" not in text:
        miss.append("no sm_86 cubin")
    return miss


def examine(binary, tool):
    if not binary.is_file():
        raise Stop(f"binary not found: {binary}")
    rc, out, err = run([str(binary), "build-info"])
    if rc != 0:
        raise Stop(f"build-info failed: {err.strip()}")
    try:
        info = json.loads(out)
    except ValueError as e:
        raise Stop(f"build-info is not JSON: {e}") from e
    rc, elf, err = run([*tool, "-elf", str(binary)])
    if rc != 0:
        raise Stop(f"cuobjdump failed: {err.strip()[-300:]}")
    _, ver, _ = run([*tool, "--version"])
    version = next((ln.strip() for ln in ver.splitlines() if ln.startswith("Cuda compilation tools")), "")
    miss = check_info(info) + check_elf(elf) + ([] if version else ["cuobjdump version not recorded"])
    return {"build_info": info, "cuobjdump_version": version, "misses": miss,
            "exe_sha256": hashlib.sha256(binary.read_bytes()).hexdigest()}


def main(argv):
    if len(argv) < 2:
        print("STOP: usage: check_release_build.py BINARY [--out FILE] [--cuobjdump CMD...]")
        return 2
    tool = argv[argv.index("--cuobjdump") + 1:] if "--cuobjdump" in argv else DEFAULT_TOOL
    try:
        res = examine(Path(argv[1]), tool)
    except Stop as e:
        print(f"STOP: {e}")
        return 2
    text = json.dumps(res, sort_keys=True)
    if "--out" in argv:
        Path(argv[argv.index("--out") + 1]).write_text(text + "\n")
    print(text)
    if res["misses"]:
        print("STOP: not a release build: " + ", ".join(res["misses"]))
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
