"""RD-10 timing rules of lane M REPLY-72 (Q8, Q9) and lane C, judged on the data: card samples DURING each timed region
(never the idle pre-check: an idle 3090 downclocks), and the warm-up conditions. CPU only, stdlib only, `python3 -I`.
"""
import json
import statistics
from pathlib import Path

import a5_eval as ev

CLOCK_FLOOR_MHZ = 1395          # RTX 3090 base clock; ANY sample in the timed region below it voids the run (REPLY-72 Q9)
BUSY_UTIL = 50                  # a sample is busy at GPU utilization >= 50 percent
BUSY_MEDIAN_MHZ = 1600          # the median of the busy samples must reach this, else the run is voided (lane C)
MIN_BUSY_FOR_MEDIAN = 10        # lane M Q12: the busy-median rule needs at least 10 busy samples; with fewer it is "not evaluated"
MIN_SAMPLES = 3                 # fewer samples inside the timed region cannot show a clock: void
MAX_VOIDS = 3                   # three consecutive voided runs: BLOCKED


def judge_window(samples, t0_ns, t1_ns, pid):
    """Card samples taken DURING one timed region (never the idle pre-check: an idle 3090 downclocks).
    Voids the run if: fewer than MIN_SAMPLES samples fall inside; ANY sample inside is below the base clock (every sample, busy or
    not); at least 10 busy samples (utilization >= 50) exist and their median is below 1600 MHz; another compute process appeared.
    Every sample is required to carry `apps`.
    With fewer than 10 busy samples the median rule is NOT EVALUATED (reported, not a void; lane M Q12).
    Returns the total and busy sample counts, the minimum and the busy median for the report."""
    inside = [s for s in samples if t0_ns <= s["t"] * 1e9 <= t1_ns]
    clocks = [float(s["clocks.sm"]) for s in inside]
    busy = [float(s["clocks.sm"]) for s in inside if float(s["utilization.gpu"]) >= BUSY_UTIL]
    why = []
    if len(inside) < MIN_SAMPLES:
        why.append(f"only {len(inside)} card samples inside the timed region")
    if clocks and min(clocks) < CLOCK_FLOOR_MHZ:
        why.append(f"a sample at {min(clocks):.0f} MHz is below the {CLOCK_FLOOR_MHZ} MHz floor")
    med = statistics.median(busy) if busy else None
    evaluated = len(busy) >= MIN_BUSY_FOR_MEDIAN
    if evaluated and med < BUSY_MEDIAN_MHZ:
        why.append(f"busy median {med:.0f} MHz is below {BUSY_MEDIAN_MHZ} MHz")
    missing = [s for s in inside if "apps" not in s]
    if missing:
        raise ev.Stop(f"{len(missing)} card sample(s) inside the timed region have no apps list: the foreign-process rule cannot be judged")
    if any(set(s["apps"]) - {pid} for s in inside):
        why.append("another compute process appeared during the timed region")
    return {"n": len(inside), "n_busy": len(busy), "min_mhz": min(clocks) if clocks else None, "median_busy_mhz": med,
            "median_rule": "evaluated" if evaluated else f"not evaluated ({len(busy)} busy samples, fewer than {MIN_BUSY_FOR_MEDIAN})", "voided": bool(why), "why": why}


def read_watch(path):
    try:
        return [json.loads(ln) for ln in Path(path).read_text().splitlines() if ln.strip()]
    except (OSError, ValueError) as e:
        raise ev.Stop(f"cannot read the card samples {path}: {e}") from e


def clock_stats(name, doc, watch_path):
    """Per run of one raw file: the judged window. Raises ev.Stop if the watch file is missing or unusable."""
    samples = read_watch(watch_path)
    pid = doc["identities"]["pid"]
    return {(name, p["id"]): judge_window(samples, p["runs"][0]["timed_start_unix_ns"], p["runs"][0]["timed_end_unix_ns"], pid)
            for p in doc.get("prompts", [])}


def warmup_problems(docs):
    """REPLY-72 Q8 conditions, checked on the data: graph instantiated before the timed region (graph mode); the timed
    region starts after the prompt pass synchronized. A miss is a FAIL of the gate, listed by call and prompt."""
    why = []
    for n, d in sorted(docs.items()):
        if d.get("incomplete"):
            continue
        graph = d["identities"]["capture_requested"] == "graph"
        for p in d["prompts"]:
            r = p["runs"][0]
            if graph and not r["graph_ready_before_timed"]:
                why.append(f"{n} {p['id']}: the graph was not instantiated before the timed region")
            if r["prompt_synced_unix_ns"] > r["timed_start_unix_ns"]:
                why.append(f"{n} {p['id']}: the timed region started before the prompt pass synchronized")
    return why
