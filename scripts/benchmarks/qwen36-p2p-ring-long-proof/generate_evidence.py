#!/usr/bin/env python3
"""Rebuild a host-specific cold-recall receipt from guarded scratch artifacts."""

import hashlib
import json
from pathlib import Path
import sys

TASK = Path("/home/ben/fifi-scratch/0cae2ae995db/p2p-async-chunk")
ROOT = TASK / "ring-long-check"
SMOKE = ROOT / "pair-reserved-5pymv4pe"
FULL = ROOT / "pair-reserved-83klc0_u"
HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "qwen36-p2p-ring-long-recall.json"


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def guard(path):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines()]
    if not rows:
        raise ValueError("guard contains no samples")
    for row in rows:
        if (
            row["builders"]
            or row["other_jobs"]
            or row["unexpected"]
            or not row["states_ok"]
            or row["clients"]
            or row["listeners"]
            or any(row["non_target_utilization"].values())
            or len(row["pair_free_mib"]) != 2
            or min(row["pair_free_mib"].values()) < 1024
        ):
            raise ValueError("guard was not clear or the VRAM floor was violated")
    return {
        "samples": len(rows),
        "minimum_free_mib": min(v for r in rows for v in r["pair_free_mib"].values()),
        "sha256": sha(path),
    }


def restored(campaign, kind):
    complete, restore = (
        load(campaign / "complete.json"),
        load(campaign / "restore.json"),
    )
    if (
        complete.get("restored") is not True
        or complete.get("result_type") != kind
        or restore.get("restored") is not True
    ):
        raise ValueError("campaign did not restore the original service")
    return {
        "reason": restore["reason"],
        "complete_sha256": sha(campaign / "complete.json"),
        "restore_sha256": sha(campaign / "restore.json"),
    }


def main():
    sys.path.insert(0, str(ROOT))
    import attention_device as long  # noqa: PLC0415

    smoke, full = (
        load(SMOKE / "attention/summary.json"),
        load(FULL / "attention/summary.json"),
    )
    info = full["inputs"]
    if (
        full["mode"] != "full"
        or full["all_targets_completed"] is not True
        or full["all_recall_passed"] is not True
        or info != smoke["inputs"]
        or info["peer"]["commit"] != "cc635bbdf4f9adb7f261d0721b63200ea7293341"
        or info["server"]["server_binary_sha256"]
        != "0f276c5f071a0249b7b82e714de09e94722768e3fee22a1fd529b68784640f5e"
        or info["long_source"]["commit"] != "90f1de648b4b18cb76e1e2f43218be9b6890a8d8"
        or info["recall_targets"]["full"] != [8192, 32768, 131072]
    ):
        raise ValueError("long-run inputs or outcome changed")
    smoke_receipt = long.validate_smoke_receipt(str(SMOKE), info)
    log = FULL / "attention/candidate-engine.log"
    if sha(log) != full["server_log_sha256"]:
        raise ValueError("server log changed")
    results = []
    for target, recorded in zip(
        info["recall_targets"]["full"], full["results"], strict=True
    ):
        raw = FULL / f"attention/recall-{target}.json"
        if sha(raw) != full["recall_sha256"][str(target)]:
            raise ValueError("recall result changed")
        checked = long.validate_result(load(raw), log.read_text(), target)
        if checked != recorded or not checked["recall_passed"]:
            raise ValueError("recall output does not match server telemetry")
        results.append({**checked, "raw_result_sha256": sha(raw)})
    fingerprints = [
        load(ROOT / name)
        for name in (
            "2080-before-smoke.json",
            "2080-after-smoke.json",
            "2080-after-full.json",
        )
    ]
    if any(item != fingerprints[0] for item in fingerprints):
        raise ValueError("2080 service changed during the campaign")
    snapshots = {
        "long_recall_harness.py.txt": "attention_device.py",
        "reservation_harness.py.txt": "reserve_pair_attention.py",
    }
    for snapshot, source in snapshots.items():
        if (
            sha(HERE / snapshot) != info["runner_sources"][source]
            or sha(ROOT / source) != info["runner_sources"][source]
        ):
            raise ValueError("runner snapshot does not match the measured source")
    output = {
        "scope": "One guarded synthetic cold recall request per length; no coding or warm-cache qualification.",
        "model": "Qwen3.6 35B A3B Q4_K_M",
        "model_sha256": info["model"]["sha256"],
        "devices": ["RTX 3060 12 GB", "RTX 3080 10 GB"],
        "context_limit": 262144,
        "kv_cache": "q8",
        "prefix_reuse": False,
        "chunk_rows": 512,
        "settings": {
            "INFERENCE_CUDA_P2P_RING_MULTIBLOCK": "1",
            "Q36_P2P_ASYNC_BROADCAST": "0",
        },
        "tested_source_commit": info["peer"]["commit"],
        "server_binary_sha256": info["server"]["server_binary_sha256"],
        "driver_source": info["long_source"],
        "runner_sources": info["runner_sources"],
        "peer_proof": info["peer_proof"],
        "results": results,
        "smoke": {
            "receipt_sha256": smoke_receipt,
            "guard": guard(SMOKE / "guard.jsonl"),
            "restoration": restored(SMOKE, "p2p_correctness_tests"),
        },
        "full": {
            "guard": guard(FULL / "guard.jsonl"),
            "restoration": restored(FULL, "p2p_diagnostic_matrix"),
            "summary_sha256": sha(FULL / "attention/summary.json"),
            "server_log_sha256": sha(log),
        },
        "secondary_service_unchanged": {
            "ready": True,
            "fingerprint_sha256": fingerprints[0]["sha256"],
        },
        "limitations": [
            "Each length is one cold synthetic prompt with three markers, not a coding-quality benchmark.",
            "No prefix reuse or multi-turn session behavior was tested.",
            "The configured 262144-token context was not filled in this campaign.",
            "Server prefill timing includes first emitted-batch work.",
            "This result does not qualify the experimental ring or the main base for default use.",
        ],
    }
    text = json.dumps(output, indent=2, allow_nan=False) + "\n"
    if "--check" in sys.argv:
        if OUT.read_text() != text:
            raise ValueError("recorded evidence differs from the checked artifacts")
        print("Long recall evidence matches guarded artifacts and server telemetry.")
    else:
        OUT.write_text(text)
        print("Wrote", OUT)


if __name__ == "__main__":
    main()
