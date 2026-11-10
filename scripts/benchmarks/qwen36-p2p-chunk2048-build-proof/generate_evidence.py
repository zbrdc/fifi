#!/usr/bin/env python3
"""Verify and summarize the host-only native-CUDA build receipt."""

import hashlib
import json
from pathlib import Path
import sys

TASK = Path("/home/ben/fifi-scratch/48125d3d5344/p2p-chunk2048")
STAMP = "1794345297907869478"
COMMIT = "f6d1c506f4e7ebe0ad3daa69290417c8e92639f2"
HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "qwen36-p2p-chunk2048-build.json"


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def events(path):
    result = []
    for line in Path(path).read_text().splitlines():
        try:
            result.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return result


def main():
    receipt_path = TASK / "logs" / f"verified-build-{STAMP}.json"
    receipt = json.loads(receipt_path.read_text())
    if (
        receipt["commit"] != COMMIT
        or receipt["target"] != "native-cuda linux sm86 release"
        or receipt["device_tests_run"] is not False
        or len(receipt["physical_peer_tests_enumerated"]) != 5
    ):
        raise ValueError("build receipt differs from the pinned host-only run")
    for kind, target, is_test in (
        ("test", "inference_engine", True),
        ("server", "enginesrv", False),
    ):
        binary = Path(receipt[f"{kind}_binary"])
        log = Path(receipt[f"{kind}_build_log"])
        if (
            sha(binary) != receipt[f"{kind}_binary_sha256"]
            or sha(log) != receipt[f"{kind}_build_log_sha256"]
        ):
            raise ValueError("binary or cargo build log changed")
        rows = events(log)
        artifacts = [
            x
            for x in rows
            if x.get("reason") == "compiler-artifact"
            and x.get("target", {}).get("name") == target
            and x.get("executable") == str(binary)
            and x.get("manifest_path") == str(TASK / "wt/engine/Cargo.toml")
            and x.get("profile", {}).get("test") is is_test
            and x.get("profile", {}).get("opt_level") == "3"
            and x.get("fresh") is False
            and "native-cuda" in x.get("features", [])
        ]
        if len(artifacts) != 1 or [
            x.get("success") for x in rows if x.get("reason") == "build-finished"
        ] != [True]:
            raise ValueError("cargo log does not identify a successful fresh build")
    host = []
    for item in receipt["host_results"]:
        path = Path(item["log"])
        output = path.read_text()
        if (
            path.parent != TASK / "logs"
            or sha(path) != item["sha256"]
            or item["exit"] != 0
            or item["pass"] is not True
            or f"test {item['name']} ... ok" not in output
            or "test result: ok. 1 passed; 0 failed; 0 ignored;" not in output
        ):
            raise ValueError("a host check is missing, skipped, failed or changed")
        host.append({"name": item["name"], "log_sha256": sha(path), "output": output})
    if len(host) != 10 or len({x["name"] for x in host}) != 10:
        raise ValueError("host checks must contain ten distinct passes")
    preflight_path = TASK / "device-check/device-preflight.json"
    preflight = json.loads(preflight_path.read_text())
    if (
        preflight["commit"] != COMMIT
        or preflight["binary_sha256"] != receipt["test_binary_sha256"]
        or [x[0] for x in preflight["tests"]]
        != receipt["physical_peer_tests_enumerated"]
        or any(x[1] is not True for x in preflight["tests"])
    ):
        raise ValueError("device-test enumeration or preflight differs")
    output = {
        "scope": "Native-CUDA build and host checks only; no device or performance test.",
        "source_commit": COMMIT,
        "target": receipt["target"],
        "build_device_visibility": "CUDA_VISIBLE_DEVICES empty",
        "receipt_sha256": sha(receipt_path),
        "test_binary_sha256": receipt["test_binary_sha256"],
        "server_binary_sha256": receipt["server_binary_sha256"],
        "test_build_log_sha256": receipt["test_build_log_sha256"],
        "server_build_log_sha256": receipt["server_build_log_sha256"],
        "host_checks": host,
        "physical_peer_tests_enumerated": receipt["physical_peer_tests_enumerated"],
        "device_preflight_sha256": sha(preflight_path),
        "device_tests_run": False,
        "limitations": [
            "The 2,048-row candidate has not been tested on physical GPUs or measured for throughput.",
            "The host checks and test enumeration do not qualify device behavior, memory use, or long-context quality.",
        ],
    }
    text = json.dumps(output, indent=2, allow_nan=False) + "\n"
    if "--check" in sys.argv:
        if OUT.read_text() != text:
            raise ValueError(
                "recorded build evidence differs from the verified artifacts"
            )
        print("Build evidence matches the receipt, logs, host checks and preflight.")
    else:
        OUT.write_text(text)
        print("Wrote", OUT)


if __name__ == "__main__":
    main()
