#!/usr/bin/env python3
"""Recompute host/GPU interval overlap from a recorded Nsight SQLite trace."""

import argparse
import hashlib
import json
from pathlib import Path
import sqlite3

TRACE_SHA256 = "8acde7211ac96e73f6024e200c517f6888a5e1760fe0612da51de3eed4ff5744"
UUIDS = {
    0: "704fef60-5280-9b88-fa3d-d11135caf576",
    1: "282e266a-24ac-6fa8-a16d-4281e52b4c0a",
}


def union(intervals):
    """Return sorted, disjoint half-open intervals without double counting."""
    result = []
    for start, end in sorted(intervals):
        if end < start:
            raise ValueError("interval ends before it starts")
        if end == start:
            continue
        if result and start <= result[-1][1]:
            result[-1] = (result[-1][0], max(result[-1][1], end))
        else:
            result.append((start, end))
    return result


def intersection(left, right):
    """Intersect two sorted, disjoint interval lists."""
    out = []
    i = j = 0
    while i < len(left) and j < len(right):
        start, end = max(left[i][0], right[j][0]), min(left[i][1], right[j][1])
        if start < end:
            out.append((start, end))
        if left[i][1] < right[j][1]:
            i += 1
        else:
            j += 1
    return out


def seconds(intervals):
    return round(sum(end - start for start, end in intervals) / 1_000_000_000, 6)


def summarize(sync, q4, waits, activity):
    """Describe overlap; categories may overlap and must not be added."""
    sync = union(sync)
    q4 = {key: union(value) for key, value in q4.items()}
    waits = {key: union(value) for key, value in waits.items()}
    activity = {key: union(value) for key, value in activity.items()}
    if not sync or set(q4) != {0, 1} or set(waits) != {0, 1} or set(activity) != {0, 1}:
        raise ValueError("expected synchronization and two GPU event sets")
    if any(not q4[i] or not waits[i] or not activity[i] for i in (0, 1)):
        raise ValueError("missing required GPU events")
    either_q4 = union(q4[0] + q4[1])
    either_activity = union(activity[0] + activity[1])
    active_overlap = seconds(intersection(sync, either_activity))
    return {
        "sync_union_seconds": seconds(sync),
        "sync_with_q4_either_gpu_seconds": seconds(intersection(sync, either_q4)),
        "sync_with_recorded_activity_either_gpu_seconds": active_overlap,
        "sync_without_recorded_activity_either_gpu_seconds": round(
            seconds(sync) - active_overlap, 6
        ),
        "sync_with_3060_wait_and_3080_q4_seconds": seconds(
            intersection(sync, intersection(waits[0], q4[1]))
        ),
        "sync_with_3080_wait_and_3060_q4_seconds": seconds(
            intersection(sync, intersection(waits[1], q4[0]))
        ),
        "per_gpu": {
            str(i): {
                "sync_with_q4_seconds": seconds(intersection(sync, q4[i])),
                "sync_with_wait_sum_seconds": seconds(intersection(sync, waits[i])),
                "sync_with_recorded_activity_seconds": seconds(
                    intersection(sync, activity[i])
                ),
            }
            for i in (0, 1)
        },
    }


def analyze(trace):
    with trace.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != TRACE_SHA256:
        raise ValueError("trace hash differs from the recorded profile")
    with sqlite3.connect(f"{trace.resolve().as_uri()}?mode=ro", uri=True) as db:
        devices = dict(db.execute("SELECT cudaId, uuid FROM TARGET_INFO_CUDA_DEVICE"))
        if devices != UUIDS:
            raise ValueError("trace GPU identities do not match the recorded pair")
        sync_rows = list(
            db.execute(
                "SELECT r.start,r.end,r.globalTid FROM CUPTI_ACTIVITY_KIND_RUNTIME r JOIN StringIds s ON s.id=r.nameId WHERE s.value='cudaStreamSynchronize_v3020'"
            )
        )
        if len(sync_rows) != 9892 or len({row[2] for row in sync_rows}) != 1:
            raise ValueError("synchronization count or host thread differs")
        q4, waits, activity = {}, {}, {}
        for gpu in (0, 1):

            def named(name):
                return list(
                    db.execute(
                        "SELECT k.start,k.end FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id=k.shortName WHERE k.deviceId=? AND s.value=?",
                        (gpu, name),
                    )
                )

            q4[gpu] = named("mul_mat_q")
            waits[gpu] = named("wait_sum_kernel")
            activity[gpu] = list(
                db.execute(
                    "SELECT start,end FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE deviceId=? UNION ALL SELECT start,end FROM CUPTI_ACTIVITY_KIND_MEMCPY WHERE deviceId=?",
                    (gpu, gpu),
                )
            )
    return {
        "scope": "Offline overlap analysis of one previously captured CUDA software trace, not a new performance test.",
        "trace_sha256": digest,
        "tested_source_commit": "76dec8a61d040efa97beefe273bed8dc9e68ff57",
        "model": "Qwen3.6 35B A3B Q4_K_M; q8 KV",
        "prompt_tokens": 4274,
        "chunk_rows": 256,
        "synchronization_calls": len(sync_rows),
        "distinct_host_threads": 1,
        "devices": {"0": "RTX 3060 12 GB", "1": "RTX 3080 10 GB"},
        "overlap": summarize([(s, e) for s, e, _ in sync_rows], q4, waits, activity),
        "limitations": [
            "CUDA tracing perturbs timing; this is one old 256-row synthetic prompt, not the current 1024-row server.",
            "Host synchronization duration is elapsed time, not CPU computation or CPU utilization.",
            "GPU activity includes wait kernels; an active kernel does not establish useful compute saturation.",
            "Overlapping categories are not additive and do not establish causality or possible speedup.",
            "Mapped-host expert reads are not necessarily CUDA memcpy events; this trace does not identify weight placement or memory stalls.",
            "No CPU samples or call stacks were collected; unrecorded activity may exist.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    result = json.dumps(analyze(args.trace), indent=2, allow_nan=False) + "\n"
    if args.check:
        if args.output.read_text() != result:
            raise ValueError("recorded result differs from the trace")
        print("Recorded overlap matches the trace.")
    else:
        args.output.write_text(result)
        print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
