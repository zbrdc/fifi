#!/usr/bin/env python3
"""Rebuild observer evidence from this host's guarded scratch artifacts.

The original server, logs, and measurement drivers must remain at the pinned
scratch paths. This provenance snapshot is not a portable workload launcher.
"""

import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import sys

TASK = Path("/home/ben/fifi-scratch/0cae2ae995db/p2p-async-chunk")
HERE = Path(__file__).resolve().parent
OUTPUT = HERE.parent / "qwen36-p2p-resource-results.json"
SMOKE = TASK / "observer-check/pair-reserved-x17oc4pv"
RUNS = (
    TASK / "observer-check/pair-reserved-3baxmmlf",
    TASK / "observer-check/pair-reserved-829az9ou",
)
PEER = TASK / "device-check/pair-reserved-n17921o8"
STAMP = "1794327065986698758"
SERVER_COMMIT = "cc635bbdf4f9adb7f261d0721b63200ea7293341"
TREE_HEAD = "c273f2a7eac3236577edbfe79b6a34dc09b48028"
OBSERVER_COMMIT = "dc5a2e670da765467c905b384e4cdcb2e01f755b"
OBSERVER_SHA = "7b62cd2da0b7077cb7924f7362f01a52a976f67a314c466f57f071471690e87e"


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
        raise ValueError("campaign completion or original-service restoration missing")
    return {
        "reason": restore["reason"],
        "complete_sha256": sha(campaign / "complete.json"),
        "restore_sha256": sha(campaign / "restore.json"),
    }


def guard(path):
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
            or len(row["pair_free_mib"]) != 2
            or min(row["pair_free_mib"].values()) < 1024
        ):
            raise ValueError(
                "guard recorded competing work, bad service state or insufficient VRAM"
            )
    return {
        "samples": len(rows),
        "minimum_free_mib": min(v for r in rows for v in r["pair_free_mib"].values()),
        "sha256": sha(path),
    }


def stats(values):
    if not values:
        raise ValueError("no matching samples")
    return {
        "count": len(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def cpu_test_log(name, count):
    path = TASK / "logs" / name
    output = path.read_text()
    if f"Ran {count} tests" not in output or "\nOK\n" not in output:
        raise ValueError("CPU driver test evidence is incomplete")
    return {"tests_passed": count, "log_sha256": sha(path)}


def long_batch_samples(rows, phase):
    samples = [row for row in rows if row.get("type") == "sample"]
    inside = [
        cur
        for prev, cur in zip(samples, samples[1:])
        if prev["started_monotonic_ns"] >= phase["start_monotonic_ns"]
        and cur["finished_monotonic_ns"] <= phase["end_monotonic_ns"]
    ]
    if len(inside) < 3:
        raise ValueError("too few fully contained long-batch intervals")
    cpu = {}
    for key in (
        "process_total_cores",
        "process_user_cores",
        "process_system_cores",
        "busiest_thread_cores",
        "system_busy_fraction",
    ):
        values = [
            r["cpu"][key] for r in inside if r["cpu"] and r["cpu"].get(key) is not None
        ]
        cpu[key] = stats(values)
    cpu["busiest_thread_names"] = sorted(
        {
            r["cpu"]["busiest_thread_name"]
            for r in inside
            if r["cpu"] and r["cpu"].get("busiest_thread_name")
        }
    )
    gpu = []
    for index in range(2):
        values = [row["gpus"][index] for row in inside]
        detail = {
            "uuid": values[0]["uuid"],
            "gpu_busy_percent_samples": [v["gpu_busy_percent"] for v in values],
            "pcie_generation_samples": sorted(
                {
                    v["pcie_generation"]
                    for v in values
                    if v["pcie_generation"] is not None
                }
            ),
            "pcie_width_samples": sorted(
                {v["pcie_width"] for v in values if v["pcie_width"] is not None}
            ),
        }
        for key in ("pcie_rx_kb_per_s", "pcie_tx_kb_per_s"):
            detail[key + "_samples"] = [
                v[key]
                for v in values
                if v[key] is not None
                and v[key + "_started_monotonic_ns"] >= phase["start_monotonic_ns"]
                and v[key + "_finished_monotonic_ns"] <= phase["end_monotonic_ns"]
            ]
        gpu.append(detail)
    return {
        "batch_start_monotonic_ns": phase["start_monotonic_ns"],
        "batch_end_monotonic_ns": phase["end_monotonic_ns"],
        "fully_contained_cpu_intervals": len(inside),
        "cpu": cpu,
        "gpus": gpu,
        "observer_wall_ms": stats([r["observer_wall_ms"] for r in inside]),
        "observer_cpu_ms": stats([r["observer_cpu_ms"] for r in inside]),
    }


def main():
    bench = import_path(
        "resource_benchmark", TASK / "observer-check/attention_device.py"
    )
    receipt_path = TASK / "logs" / f"verified-build-{STAMP}.json"
    receipt = load(receipt_path)
    if (
        receipt["commit"] != SERVER_COMMIT
        or sha(Path(receipt["server_binary"])) != receipt["server_binary_sha256"]
    ):
        raise ValueError("server source or binary changed")
    observer = HERE / "observer_measured.py.txt"
    if sha(observer) != OBSERVER_SHA:
        raise ValueError("reviewed observer differs from measured observer")
    smoke_summary = load(SMOKE / "attention/summary.json")
    if smoke_summary["inputs"]["peer"]["commit"] != SERVER_COMMIT:
        raise ValueError("smoke server source differs")
    smoke = {
        "guard": guard(SMOKE / "guard.jsonl"),
        "restoration": restored(SMOKE, "p2p_correctness_tests"),
        "summary_sha256": sha(SMOKE / "attention/summary.json"),
    }
    runs = []
    for campaign in RUNS:
        p = campaign / "attention"
        summary = load(p / "summary.json")
        info = summary["inputs"]
        if (
            info["peer"]["commit"] != SERVER_COMMIT
            or info["source_tree_head"] != TREE_HEAD
            or info["server"]["server_binary_sha256"] != receipt["server_binary_sha256"]
            or info["drivers"]["scripts/bench_host_gpu_resources.py"] != OBSERVER_SHA
            or info["chunk_rows_by_arm"] != {"control": 512, "candidate": 512}
        ):
            raise ValueError("comparison inputs differ")
        bench.validate_peer_proof(str(PEER), info["peer"])
        bench.validate_smoke_receipt(str(SMOKE), info)
        if summary["arms"]["control"]["observer"] is not None:
            raise ValueError("control was instrumented")
        observed = summary["arms"]["candidate"]
        source = p / "candidate-resource.jsonl"
        probe = bench.validate_probe(
            source, observed["server_pid"], observed["server_start_ticks"], 3
        )
        if probe != observed["observer"]:
            raise ValueError("observer summary differs from raw samples")
        name = (
            "control-first-resource.jsonl"
            if info["order"] == "control-first"
            else "candidate-first-resource.jsonl"
        )
        if sha(source) != sha(HERE / name):
            raise ValueError("checked-in observer sample differs")
        rows = [json.loads(line) for line in source.read_text().splitlines()]
        comparisons = {}
        hashes = {"resource_jsonl": sha(source)}
        for label in ("short", "long"):
            control, candidate = (
                load(p / f"{arm}-{label}.json") for arm in ("control", "candidate")
            )
            value = bench.compare.compare(
                control,
                (p / "control-engine.log").read_text(),
                candidate,
                (p / "candidate-engine.log").read_text(),
            )
            if value != summary["comparisons"][label]:
                raise ValueError("prefill comparison changed")
            comparisons[label] = value
            for arm in ("control", "candidate"):
                hashes[f"{arm}-{label}.json"] = sha(p / f"{arm}-{label}.json")
        control, candidate = (
            load(p / f"{arm}-decode.json") for arm in ("control", "candidate")
        )
        value = bench.decode_comparison(
            control,
            (p / "control-engine.log").read_text(),
            candidate,
            (p / "candidate-engine.log").read_text(),
        )
        if value != summary["comparisons"]["decode"]:
            raise ValueError("decode comparison changed")
        decode_samples = {}
        for arm, data in (("control", control), ("candidate", candidate)):
            decode_samples[arm] = [
                {k: v for k, v in item.items() if k != "answer_prefix"}
                for item in data["samples"]
            ]
            hashes[f"{arm}-decode.json"] = sha(p / f"{arm}-decode.json")
            hashes[f"{arm}-engine.log"] = sha(p / f"{arm}-engine.log")
        comparisons["decode"] = {
            **value,
            "payload_sha256": control["payload_sha256"],
            "samples": decode_samples,
        }
        runs.append(
            {
                "order": info["order"],
                "comparisons": comparisons,
                "long_batch": long_batch_samples(rows, observed["phases"]["long"]),
                "observer": probe,
                "artifact_sha256": hashes,
                "guard": guard(campaign / "guard.jsonl"),
                "restoration": restored(campaign, "p2p_diagnostic_matrix"),
                "summary_sha256": sha(p / "summary.json"),
            }
        )
    if [r["order"] for r in runs] != ["control-first", "candidate-first"]:
        raise ValueError("both orders are required")
    for label in ("short", "long", "decode"):
        if (
            runs[0]["comparisons"][label]["payload_sha256"]
            != runs[1]["comparisons"][label]["payload_sha256"]
        ):
            raise ValueError("workload differs between orders")
    fingerprints = [
        load(TASK / "logs" / name)
        for name in (
            f"secondary-before-observer-{STAMP}.json",
            f"secondary-after-observer-smoke-{STAMP}.json",
            f"secondary-after-observer-control-first-{STAMP}.json",
            f"secondary-after-observer-candidate-first-{STAMP}.json",
        )
    ]
    if any(x != fingerprints[0] for x in fingerprints):
        raise ValueError("secondary service identity or configuration changed")
    snapshots = {
        name: sha(HERE / name)
        for name in (
            "control-first-resource.jsonl",
            "candidate-first-resource.jsonl",
            "observer_harness.py.txt",
            "compare_prefill.py",
            "bench_p2p_decode.py",
            "topology-observation.json",
        )
    }
    inputs = load(RUNS[0] / "attention/summary.json")["inputs"]
    if (
        snapshots["compare_prefill.py"] != inputs["drivers"]["compare_prefill.py"]
        or snapshots["bench_p2p_decode.py"]
        != inputs["drivers"]["scripts/bench_p2p_decode.py"]
    ):
        raise ValueError("measurement driver snapshots differ")
    result = {
        "scope": "Guarded observer overhead and sampled resource use; no causal bottleneck or production qualification claim.",
        "observer_source_commit": OBSERVER_COMMIT,
        "observer_sha256": OBSERVER_SHA,
        "server_tested_commit": SERVER_COMMIT,
        "server_source_tree_head": TREE_HEAD,
        "server_binary_sha256": receipt["server_binary_sha256"],
        "build_receipt_sha256": sha(receipt_path),
        "model": "Qwen3.6 35B A3B Q4_K_M",
        "model_sha256": inputs["model"]["sha256"],
        "devices": ["RTX 3060 12 GB", "RTX 3080 10 GB"],
        "configured_context_tokens": 262144,
        "kv_cache": "q8",
        "prefix_reuse": False,
        "chunk_rows": 512,
        "common_flags": {"Q36_P2P_ASYNC_BROADCAST": "0", "Q36_P2P_SYNC_BROADCAST": "1"},
        "arm_difference": {
            "control": "observer off",
            "candidate": "observer on, 2-second interval",
        },
        "samples_per_arm_per_order": 3,
        "served_smoke": smoke,
        "runs": runs,
        "secondary_service_unchanged": {
            "ready": True,
            "fingerprint_sha256": fingerprints[0]["sha256"],
        },
        "measurement_driver_sha256": inputs["drivers"],
        "post_measurement_topology": load(HERE / "topology-observation.json"),
        "cpu_driver_tests": {
            "observer_harness": cpu_test_log("observer-harness-cpu-tests.log", 39),
            "resource_observer": cpu_test_log("resource-observer-cpu-tests.log", 8),
            "prefill_comparator": cpu_test_log("resource-comparator-cpu-tests.log", 5),
        },
        "proof_snapshot_sha256": snapshots,
        "limitations": [
            "One synthetic workload and three samples per arm/order do not characterize general workloads.",
            "Long-batch windows cover three 4,274-token requests and their transitions, not only individual prefill kernels.",
            "Engine CPU rates are average cores over fully contained sample intervals; they do not distinguish useful work from driver polling or prove a CPU bottleneck.",
            "GPU busy percentages do not show compute efficiency and can include waiting kernels.",
            "PCIe RX/TX values are independent 20 ms NVML sampling windows in KB/s, not a continuous traffic total; they cannot distinguish host from peer traffic.",
            "The observer does not measure host DRAM bandwidth and excludes child processes from the sampled engine CPU total.",
            "The observer controller snapshot was not independently hashed at run time; measurement drivers were hash-checked by the harness.",
            "Configured 262144 context was not filled; no long-coding-session or 128k/256k qualification is established.",
            "The local main base has separate unresolved broad-test history; these measurements do not qualify that base.",
        ],
    }
    serialized = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if "--check" in sys.argv:
        if OUTPUT.read_text() != serialized:
            raise ValueError("recorded evidence differs from guarded artifacts")
        print(
            "Recorded resource evidence matches guarded artifacts and recalculated comparisons."
        )
    else:
        OUTPUT.write_text(serialized)
        print("Wrote", OUTPUT)


if __name__ == "__main__":
    main()
