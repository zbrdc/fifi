#!/usr/bin/env python3
"""A fake `docker` for the self-tests. It runs no container and executes no program.

`image inspect <image>` succeeds for images that start with `fake@sha256:`. `run` refuses
(exit 125) when a required isolation flag is missing, then answers like a Python run of a script
that holds only `print(<integer arithmetic>)` lines, evaluated with `ast` and never with exec.
FAKE_DOCKER_EXIT makes `run` end with that status. FAKE_DOCKER_FLOOD makes it print without end.
The fake refuses (exit 125) when its own environment holds a variable that looks like a key, a
token, a secret, or a password: the harness must clean the environment of the docker client.
A leading `-H <host>` is stripped and appended to the file in FAKE_DOCKER_HOSTS; a call without it
is refused. `info` reports the security option `rootless` unless FAKE_DOCKER_ROOTLESS is 0, and
fails when FAKE_DOCKER_NODAEMON is set. `version` answers with runc and containerd components.
Each call appends its argument list as one JSON line to the file in FAKE_DOCKER_LOG.
"""

import ast
import json
import operator
import os
import re
import sys

OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul}
REQUIRED_PAIRS = (("--pull", "never"), ("--log-driver", "none"), ("--network", "none"), ("--user", "65534:65534"), ("--memory", "256m"),
                  ("--cpus", "1"), ("--pids-limit", "64"))


def evaluate(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in OPS:
        return OPS[type(node.op)](evaluate(node.left), evaluate(node.right))
    raise ValueError("unsupported expression")


def main(argv):
    if any(re.search(r"KEY|TOKEN|SECRET|PASSWORD", name, re.IGNORECASE)
           for name in os.environ if not name.startswith("FAKE_DOCKER_")):
        print("fake docker: the environment of the docker client holds a secret", file=sys.stderr)
        return 125
    if argv[:1] != ["-H"] or len(argv) < 3:
        print("fake docker: the daemon must be pinned with -H <host>", file=sys.stderr)
        return 125
    if os.environ.get("FAKE_DOCKER_HOSTS"):
        with open(os.environ["FAKE_DOCKER_HOSTS"], "a", encoding="utf-8") as handle:
            handle.write(argv[1] + "\n")
    if os.environ.get("DOCKER_HOST") != argv[1]:
        print("fake docker: DOCKER_HOST must name the pinned host", file=sys.stderr)
        return 125
    argv = argv[2:]
    log = os.environ.get("FAKE_DOCKER_LOG")
    if log:
        with open(log, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(argv) + "\n")
    if argv[:1] == ["info"]:
        if os.environ.get("FAKE_DOCKER_NODAEMON"):
            print("Cannot connect to the Docker daemon", file=sys.stderr)
            return 1
        rootless = os.environ.get("FAKE_DOCKER_ROOTLESS") != "0"
        print(json.dumps(["name=seccomp,profile=builtin"] + (["name=rootless"] if rootless else [])))
        return 0
    if argv[:1] == ["version"]:
        print(json.dumps([{"Name": "Engine", "Version": "fake-29"},
                          {"Name": "containerd", "Version": "fake-2"},
                          {"Name": "runc", "Version": "fake-1"}]))
        return 0
    if argv[:2] == ["image", "inspect"]:
        if argv[-1].startswith("fake@sha256:"):
            print("sha256:" + "0" * 64)
            return 0
        print("Error: No such image: " + argv[-1], file=sys.stderr)
        return 1
    if argv[:1] == ["rm"]:
        return 0
    if argv[:1] != ["run"]:
        return 125
    for flag, value in REQUIRED_PAIRS:
        if flag not in argv or argv[argv.index(flag) + 1] != value:
            print(f"fake docker: missing {flag} {value}", file=sys.stderr)
            return 125
    if "--read-only" not in argv or "--tmpfs" not in argv:
        print("fake docker: missing --read-only or --tmpfs", file=sys.stderr)
        return 125
    mounts = [argv[i + 1] for i, a in enumerate(argv) if a == "-v"]
    if len(mounts) != 1 or not mounts[0].endswith(":/work:ro"):
        print("fake docker: the script directory must be mounted read-only", file=sys.stderr)
        return 125
    if os.environ.get("FAKE_DOCKER_EXIT"):
        return int(os.environ["FAKE_DOCKER_EXIT"])
    if os.environ.get("FAKE_DOCKER_FLOOD"):
        try:
            while True:
                sys.stdout.write("x" * 1000 + "\n")
                sys.stdout.flush()
        except BrokenPipeError:
            return 1
    host_dir = mounts[0][: -len(":/work:ro")]
    script = os.path.join(host_dir, os.path.basename(argv[-1]))
    try:
        tree = ast.parse(open(script, encoding="utf-8").read())
        out = []
        for statement in tree.body:
            call = statement.value
            assert isinstance(statement, ast.Expr) and call.func.id == "print"
            out.append(" ".join(str(evaluate(a)) for a in call.args))
    except (OSError, SyntaxError, ValueError, AssertionError, AttributeError):
        print("Traceback: fake docker runs only print(<arithmetic>) programs", file=sys.stderr)
        return 1
    print("\n".join(out))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
