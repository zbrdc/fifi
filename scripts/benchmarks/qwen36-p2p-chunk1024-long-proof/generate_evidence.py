#!/usr/bin/env python3
"""Rebuild the completed 128k result from a later-interrupted guarded run."""

import hashlib
import json
from pathlib import Path
import sys

ROOT = Path("/home/ben/fifi-scratch/725f186d7122/p2p-chunk1024/long1024-check")
SMOKE = ROOT / "pair-reserved-qoj75kwl"
FULL = ROOT / "pair-reserved-5r1dvd84"
HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "qwen36-p2p-chunk1024-long-recall.json"


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def clear(row):
    return (
        not row["builders"]
        and not row["other_jobs"]
        and not row["unexpected"]
        and row["states_ok"]
        and not row["clients"]
        and not row["listeners"]
        and not any(row["non_target_utilization"].values())
        and len(row["pair_free_mib"]) == 2
        and min(row["pair_free_mib"].values()) >= 1024
    )


def main():
    sys.path.insert(0, str(ROOT))
    import attention_device as long  # noqa: PLC0415

    smoke = load(SMOKE / "attention/summary.json")
    info = load(FULL / "inputs.json")["probe"]
    smoke_sha = info.pop("smoke_receipt_sha256")
    if (
        info != smoke["inputs"]
        or info["peer"]["commit"] != "3a0e82c26a7a2259ac11ff6cd915018946598264"
        or info["server"]["server_binary_sha256"]
        != "ce78f8e8c7b098312793fab4a8150d6e7a998b4c34b45041865795bf5950fa88"
        or info["long_source"]["commit"] != "90f1de648b4b18cb76e1e2f43218be9b6890a8d8"
        or info["recall_targets"]["full"] != [131072, 261888]
        or info["chunk_rows_by_arm"]["candidate"] != 1024
        or long.validate_smoke_receipt(str(SMOKE), info) != smoke_sha
    ):
        raise ValueError("source, candidate or same-input smoke differs")
    if load(FULL / "restore.json").get("restored") is not True:
        raise ValueError("interrupted campaign did not restore the pair")
    if (FULL / "complete.json").exists() or (
        FULL / "attention/recall-261888.json"
    ).exists():
        raise ValueError("the interrupted campaign must not be reported complete")

    rows = [
        json.loads(line) for line in (FULL / "guard.jsonl").read_text().splitlines()
    ]
    smoke_rows = [
        json.loads(line) for line in (SMOKE / "guard.jsonl").read_text().splitlines()
    ]
    if not smoke_rows or not all(map(clear, smoke_rows)) or len(rows) < 2:
        raise ValueError("smoke or interrupted guard records missing")
    if not all(map(clear, rows[:-1])) or clear(rows[-1]):
        raise ValueError("guard did not remain clear until its final interruption")
    if rows[-1]["states_ok"] or not any(rows[-1]["non_target_utilization"].values()):
        raise ValueError("final guard observation differs from recorded interruption")
    if any(min(row["pair_free_mib"].values()) < 1024 for row in rows):
        raise ValueError("pair free-VRAM floor violated")

    progress = load(FULL / "attention/progress.json")
    raw_path = FULL / "attention/recall-131072.json"
    raw = load(raw_path)
    log_path = FULL / "attention/candidate-engine.log"
    log_text = log_path.read_text()
    long.bench.check_path(log_text, "candidate")
    result = long.validate_result(raw, log_text, 131072)
    if (
        progress.get("state") != "running"
        or progress.get("results") != [result]
        or not result["recall_passed"]
        or raw["context_limit"] != 262144
        or raw["max_completion_tokens"] != 64
        or raw["finish_reason"] != "stop"
        or raw["prompt_tokens"] + raw["max_completion_tokens"] > raw["context_limit"]
    ):
        raise ValueError("completed 128k result or completion budget differs")
    partial = load(FULL / "attention/recall-261888.json.partial")
    if partial.get("state") != "counted" or partial.get("prompt_tokens") != 262010:
        raise ValueError("interrupted near-256k request differs")
    parsed = []
    for line in log_text.splitlines():
        try:
            parsed.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    next_counts = [
        r
        for r in parsed
        if isinstance(r, dict)
        and r.get("endpoint") == "/v1/token/encode"
        and r.get("prompt_tokens") == partial["prompt_tokens"]
    ]
    if len(next_counts) != 1 or not (
        raw_path.stat().st_mtime
        <= next_counts[0]["started_at_ms"] / 1000
        < rows[-1]["time"]
    ):
        raise ValueError(
            "128k completion was not recorded before the next request and guard interruption"
        )

    before, after_smoke, after_interrupt = (
        load(ROOT / name)
        for name in (
            "2080-before-smoke.json",
            "2080-after-smoke.json",
            "2080-after-interruption.json",
        )
    )
    if before != after_smoke or before["sha256"] != after_interrupt["sha256"]:
        raise ValueError("secondary binary or configuration fingerprints differ")
    if not after_interrupt["ready"] or before["pid"] == after_interrupt["pid"]:
        raise ValueError("the recorded 2080 process restart is missing")

    snapshots = {
        "long_recall_harness.py.txt": "attention_device.py",
        "reservation_harness.py.txt": "reserve_pair_attention.py",
        "benchmark_harness.py.txt": "benchmark_device.py",
        "peer_harness.py.txt": "peer_evidence.py",
        "scope_guard.py.txt": "scope_guard.py",
    }
    for snapshot, source in snapshots.items():
        if (
            sha(HERE / snapshot) != info["runner_sources"][source]
            or sha(ROOT / source) != info["runner_sources"][source]
        ):
            raise ValueError("runner snapshot differs from the measured source")
    for name in ("bench_long_context.py", "bench_engines.py"):
        if sha(HERE.parent.parent / name) != info["long_source"]["files"][name]:
            raise ValueError("recall driver snapshot differs")
    if (
        sha(HERE / "shared_guard.py.txt")
        != info["runner_sources"]["shared_guard_sha256"]
    ):
        raise ValueError("shared guard snapshot differs")

    output = {
        "scope": "One completed synthetic cold 128k recall request before a later guarded interruption; no coding or warm-cache qualification.",
        "model": "Qwen3.6 35B A3B Q4_K_M",
        "model_sha256": info["model"]["sha256"],
        "devices": ["RTX 3060 12 GB", "RTX 3080 10 GB"],
        "context_limit": 262144,
        "kv_cache": "q8",
        "prefix_reuse": False,
        "chunk_rows": 1024,
        "settings": {
            "INFERENCE_CUDA_P2P_RING_MULTIBLOCK": "1",
            "Q36_P2P_SYNC_BROADCAST": "1",
        },
        "tested_source_commit": info["peer"]["commit"],
        "source_head_during_test": "48125d3d5344d44bfb1bbc29777809cfb3a72393",
        "source_head_delta": "Documentation and benchmark evidence only; pinned in the peer harness.",
        "server_binary_sha256": info["server"]["server_binary_sha256"],
        "driver_source": info["long_source"],
        "runner_sources": info["runner_sources"],
        "peer_proof": info["peer_proof"],
        "result": {
            **result,
            "finish_reason": raw["finish_reason"],
            "max_completion_tokens": raw["max_completion_tokens"],
            "raw_result_sha256": sha(raw_path),
        },
        "smoke": {
            "receipt_sha256": smoke_sha,
            "guard_samples": len(smoke_rows),
            "guard_sha256": sha(SMOKE / "guard.jsonl"),
            "restored": True,
        },
        "interrupted_campaign": {
            "clear_samples_before_interruption": len(rows) - 1,
            "final_guard_sample": rows[-1],
            "guard_sha256": sha(FULL / "guard.jsonl"),
            "minimum_free_mib": min(
                v for row in rows for v in row["pair_free_mib"].values()
            ),
            "completed_request_before_interruption": True,
            "near_256k": "Interrupted; prompt counted but no completion or recall result.",
            "near_256k_prompt_tokens": partial["prompt_tokens"],
            "partial_result_sha256": sha(FULL / "attention/recall-261888.json.partial"),
            "server_log_sha256": sha(log_path),
            "runner_log_sha256": sha(FULL / "runner.log"),
            "restore_sha256": sha(FULL / "restore.json"),
            "restored": True,
        },
        "secondary_service": {
            "ready_after": True,
            "process_restarted": True,
            "binary_and_config_sha256_unchanged": before["sha256"],
        },
        "limitations": [
            "One cold synthetic prompt with three markers is not a coding-quality benchmark.",
            "No prefix reuse or multi-turn behavior was tested.",
            "The near-256k run was interrupted after the 2080 service restarted; it has no completion result.",
            "Server prefill timing includes first emitted-batch work.",
            "This result does not qualify the experimental chunk or the main base for default use.",
        ],
    }
    text = json.dumps(output, indent=2, allow_nan=False) + "\n"
    if "--check" in sys.argv:
        if OUT.read_text() != text:
            raise ValueError("recorded evidence differs from guarded artifacts")
        print("128k evidence matches the completed request and interrupted campaign.")
    else:
        OUT.write_text(text)
        print("Wrote", OUT)


if __name__ == "__main__":
    main()
