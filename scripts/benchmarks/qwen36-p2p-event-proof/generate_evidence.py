#!/usr/bin/env python3
"""Rebuild the recorded evidence from this host's guarded scratch artifacts.

This is a provenance snapshot, not a portable benchmark launcher. It expects
all original logs and the exact pinned binaries to remain in the scratch path.
"""

import hashlib
import importlib.util
import json
from pathlib import Path
import sys

TASK = Path("/home/ben/fifi-scratch/0cae2ae995db/p2p-async-chunk")
HERE = Path(__file__).resolve().parent
OUTPUT = HERE.parent / "qwen36-p2p-event-results.json"
COMMIT = "cc635bbdf4f9adb7f261d0721b63200ea7293341"
STAMP = "1794327065986698758"
PEER = TASK / "device-check/pair-reserved-n17921o8"
PEER_SMOKE = TASK / "device-check/pair-reserved-4p4wjiin"
SMOKE = TASK / "benchmark-check/pair-reserved-p4vsw0x0"
RUNS = (
    TASK / "benchmark-check/pair-reserved-5wdndbev",
    TASK / "benchmark-check/pair-reserved-fgnv4j5e",
)


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load(path):
    return json.loads(path.read_text())


def import_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def checked_guard(path, require_free):
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if not rows:
        raise ValueError("guard has no samples")
    for row in rows:
        if (
            row["other_jobs"]
            or row["builders"]
            or row["unexpected"]
            or not row["states_ok"]
            or row["clients"]
            or row["listeners"]
            or any(row["non_target_utilization"].values())
        ):
            raise ValueError("guard recorded competing work or an unsafe service state")
        if require_free and (
            len(row["pair_free_mib"]) != 2 or min(row["pair_free_mib"].values()) < 1024
        ):
            raise ValueError("GPU free memory was missing or below the floor")
    return {
        "samples": len(rows),
        "sha256": sha(path),
        "minimum_free_mib": min(
            (v for r in rows for v in r["pair_free_mib"].values()), default=None
        )
        if require_free
        else None,
    }


def restored(campaign, kind):
    complete, restore = (
        load(campaign / "complete.json"),
        load(campaign / "restore.json"),
    )
    if (
        complete.get("restored") is not True
        or restore.get("restored") is not True
        or complete.get("result_type") != kind
    ):
        raise ValueError("service restoration or campaign completion missing")
    return {
        "reason": restore["reason"],
        "complete_sha256": sha(campaign / "complete.json"),
        "restore_sha256": sha(campaign / "restore.json"),
    }


def main():
    bench = import_path("event_benchmark", TASK / "benchmark-check/attention_device.py")
    peer = bench.peer
    receipt_path = TASK / "logs" / f"verified-build-{STAMP}.json"
    receipt = load(receipt_path)
    if (
        receipt["commit"] != COMMIT
        or sha(Path(receipt["test_binary"])) != receipt["test_binary_sha256"]
        or sha(Path(receipt["server_binary"])) != receipt["server_binary_sha256"]
    ):
        raise ValueError("build source or binary changed")
    host = []
    for item in receipt["host_results"]:
        p = Path(item["log"])
        text = p.read_text()
        if (
            item["pass"] is not True
            or item["exit"] != 0
            or sha(p) != item["sha256"]
            or "test result: ok. 1 passed; 0 failed; 0 ignored;" not in text
        ):
            raise ValueError("host test proof changed")
        host.append(
            {"name": item["name"], "log_sha256": item["sha256"], "output": text}
        )
    if len(host) != 9:
        raise ValueError("expected nine selected host tests")

    peer_summary = load(PEER / "attention/summary.json")
    if (
        peer_summary["inputs"]["commit"] != COMMIT
        or len(peer_summary["outcomes"]) != 4
        or peer_summary["tests_failed"]
    ):
        raise ValueError("peer test summary differs")
    peer_cases = []
    for result in peer_summary["outcomes"]:
        log = PEER / "attention" / result["log"]
        if not result["passed"] or result["exit"] != 0 or sha(log) != result["sha256"]:
            raise ValueError("peer test log changed")
        peer.validate(result["test"], log.read_text())
        peer_cases.append(
            {
                "case": result["case"],
                "test": result["test"],
                "log_sha256": sha(log),
                "output": log.read_text(),
            }
        )
    smoke_receipt = peer.validate_smoke_receipt(str(PEER_SMOKE), peer_summary["inputs"])
    peer_proof = {
        "cases": peer_cases,
        "guard": checked_guard(PEER / "guard.jsonl", False),
        "restoration": restored(PEER, "p2p_diagnostic_matrix"),
        "summary_sha256": sha(PEER / "attention/summary.json"),
        "preceding_smoke": {
            "receipt_sha256": smoke_receipt,
            "guard": checked_guard(PEER_SMOKE / "guard.jsonl", False),
            "restoration": restored(PEER_SMOKE, "p2p_correctness_tests"),
        },
    }

    smoke_summary = load(SMOKE / "attention/summary.json")
    if smoke_summary["inputs"]["peer"]["commit"] != COMMIT:
        raise ValueError("smoke source differs")
    smoke = {
        "guard": checked_guard(SMOKE / "guard.jsonl", True),
        "restoration": restored(SMOKE, "p2p_correctness_tests"),
        "summary_sha256": sha(SMOKE / "attention/summary.json"),
    }
    runs = []
    for campaign in RUNS:
        p = campaign / "attention"
        summary = load(p / "summary.json")
        info = summary["inputs"]
        if (
            info["peer"]["commit"] != COMMIT
            or info["server"]["server_binary_sha256"] != receipt["server_binary_sha256"]
            or info["chunk_rows_by_arm"] != {"control": 512, "candidate": 512}
        ):
            raise ValueError("served comparison inputs differ")
        bench.validate_peer_proof(str(PEER), info["peer"])
        bench.validate_smoke_receipt(str(SMOKE), info)
        comparisons = {}
        artifact_hashes = {}
        for label in ("short", "long"):
            control, candidate = (
                load(p / f"{arm}-{label}.json") for arm in ("control", "candidate")
            )
            recomputed = bench.compare.compare(
                control,
                (p / "control-engine.log").read_text(),
                candidate,
                (p / "candidate-engine.log").read_text(),
            )
            if recomputed != summary["comparisons"][label]:
                raise ValueError("prefill comparison changed")
            comparisons[label] = recomputed
            for arm in ("control", "candidate"):
                artifact_hashes[f"{arm}-{label}.json"] = sha(p / f"{arm}-{label}.json")
        control, candidate = (
            load(p / f"{arm}-decode.json") for arm in ("control", "candidate")
        )
        recomputed = bench.decode_comparison(
            control,
            (p / "control-engine.log").read_text(),
            candidate,
            (p / "candidate-engine.log").read_text(),
        )
        if recomputed != summary["comparisons"]["decode"]:
            raise ValueError("decode comparison changed")
        samples = {}
        for arm, data in (("control", control), ("candidate", candidate)):
            samples[arm] = [
                {k: v for k, v in sample.items() if k != "answer_prefix"}
                for sample in data["samples"]
            ]
            artifact_hashes[f"{arm}-decode.json"] = sha(p / f"{arm}-decode.json")
            artifact_hashes[f"{arm}-engine.log"] = sha(p / f"{arm}-engine.log")
        comparisons["decode"] = {
            **recomputed,
            "payload_sha256": control["payload_sha256"],
            "samples": samples,
        }
        runs.append(
            {
                "order": info["order"],
                "comparisons": comparisons,
                "artifact_sha256": artifact_hashes,
                "free_mib_after_warmup": {
                    name: data["free_mib_after_warmup"]
                    for name, data in summary["arms"].items()
                },
                "guard": checked_guard(campaign / "guard.jsonl", True),
                "restoration": restored(campaign, "p2p_diagnostic_matrix"),
                "summary_sha256": sha(p / "summary.json"),
            }
        )
    if [r["order"] for r in runs] != ["control-first", "candidate-first"]:
        raise ValueError("both comparison orders required")
    for label in ("short", "long", "decode"):
        if (
            runs[0]["comparisons"][label]["payload_sha256"]
            != runs[1]["comparisons"][label]["payload_sha256"]
        ):
            raise ValueError("payload changed between orders")
    fingerprints = [
        load(TASK / "logs" / name)
        for name in (
            f"secondary-before-{STAMP}.json",
            f"secondary-after-smoke-{STAMP}.json",
            f"secondary-after-full-{STAMP}.json",
            f"secondary-after-served-smoke-{STAMP}.json",
            f"secondary-after-ab-control-first-{STAMP}.json",
            f"secondary-after-ab-candidate-first-{STAMP}.json",
        )
    ]
    if any(item != fingerprints[0] for item in fingerprints):
        raise ValueError("2080 process or configuration changed")
    snapshots = {
        name: sha(HERE / name)
        for name in (
            "benchmark_harness.py.txt",
            "compare_prefill.py",
            "bench_p2p_decode.py",
        )
    }
    drivers = load(RUNS[0] / "attention/summary.json")["inputs"]["drivers"]
    if (
        snapshots["compare_prefill.py"] != drivers["compare_prefill.py"]
        or snapshots["bench_p2p_decode.py"] != drivers["scripts/bench_p2p_decode.py"]
    ):
        raise ValueError("measurement driver snapshot differs")
    result = {
        "scope": "Guarded synthetic Qwen3.6 P2P event-barrier comparison; no production qualification.",
        "source_commit": COMMIT,
        "model": "Qwen3.6 35B A3B Q4_K_M",
        "model_sha256": load(RUNS[0] / "attention/summary.json")["inputs"]["model"][
            "sha256"
        ],
        "devices": ["RTX 3060 12 GB", "RTX 3080 10 GB"],
        "kv_cache": "q8",
        "configured_context_tokens": 262144,
        "prefix_reuse": False,
        "chunk_rows": 512,
        "samples_per_arm_per_order": 3,
        "arm_difference": {
            "control": {"Q36_P2P_ASYNC_BROADCAST": "0"},
            "candidate": {"Q36_P2P_ASYNC_BROADCAST": "1"},
        },
        "common_sync_broadcast": "1",
        "host_expert_bytes_requested": [1395864370, 3489660928],
        "host_expert_bytes_selected": [1434451968, 3548381184],
        "build": {
            "target": receipt["target"],
            "test_binary_sha256": receipt["test_binary_sha256"],
            "server_binary_sha256": receipt["server_binary_sha256"],
            "receipt_sha256": sha(receipt_path),
            "test_build_log_sha256": receipt["test_build_log_sha256"],
            "server_build_log_sha256": receipt["server_build_log_sha256"],
            "selected_host_tests": host,
        },
        "peer_correctness": peer_proof,
        "served_smoke": smoke,
        "runs": runs,
        "secondary_service_unchanged": {
            "ready": True,
            "fingerprint_sha256": fingerprints[0]["sha256"],
        },
        "measurement_driver_sha256": drivers,
        "proof_snapshot_sha256": snapshots,
        "limitations": [
            "Synthetic single prompts; three samples per arm and order do not characterize general workloads.",
            "Server prefill timing runs from job start through the first emitted batch, including first-token generation.",
            "The decode sample is a deterministic 31-token prompt with 128 completion tokens; matching hashes cover only this output.",
            "Configured 262144 context was not filled. No 128k/256k recall, coding-session quality, or production qualification is established.",
            "The first-order harness was not separately hashed at runtime. The checked-in snapshot is the later reverse-order harness; its only intervening change allowed reuse of the same candidate-only smoke across orders.",
            "The snapshot is host-specific and depends on local guarded-runner infrastructure and scratch artifacts; it is not a portable standalone benchmark.",
            "The tiered-upload main base has separate unresolved broad-test history; these P2P checks do not qualify that base.",
        ],
    }
    serialized = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if "--check" in sys.argv:
        if OUTPUT.read_text() != serialized:
            raise ValueError("recorded evidence differs from current artifacts")
        print(
            "Recorded evidence matches guarded artifacts and comparison recalculation."
        )
    else:
        OUTPUT.write_text(serialized)
        print("Wrote", OUTPUT)


if __name__ == "__main__":
    main()
