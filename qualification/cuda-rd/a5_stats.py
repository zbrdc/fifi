#!/usr/bin/env python3
"""A5 exactness-rule statistics (lane M) from the Qwen3.6 oracle. CPU only, read-only on the oracle."""
import json
import sys
from pathlib import Path

import numpy as np

ORACLE = Path("/media/dan/nvme/fifi/scrch/c62fd02/oracle/qwen36")
PROMPTS = ["P1", "P2", "P3", "P4", "P5"]
FLOOR = 0.05
FACTOR = 2.0
CAP_PCT = 5.0
RUNS = {"cpu12": "cpu/clean", "cpu6": "cpu/threads6", "cuda": "cuda/follow"}


def load(run, p):
    return json.loads((ORACLE / RUNS[run] / p / "run.json").read_text())["steps"]


def logit_of(step, tok):
    for t, v in step["top"]:
        if t == tok:
            return v
    return None


def per_step(p):
    runs = {r: load(r, p) for r in RUNS}
    ref = runs["cpu12"]
    rows = []
    for i, rs in enumerate(ref):
        tok = rs["top"][0][0]
        margin = rs["top"][0][1] - rs["top"][1][1]
        argmaxes = [runs[r][i]["top"][0][0] for r in RUNS]
        lg = [logit_of(runs[r][i], tok) for r in RUNS]
        missing = any(v is None for v in lg)
        agree = len(set(argmaxes)) == 1 and not missing
        spread = None if missing else max(lg) - min(lg)
        rows.append({"i": i, "margin": margin, "agree": agree, "missing": missing, "spread": spread,
                     "argmax_cpu6": argmaxes[1], "argmax_cuda": argmaxes[2], "ref": tok})
    return rows


def gate(rows, s_of):
    """s_of(row) -> S for the row. Returns gated count."""
    return sum(1 for r in rows if r["agree"] and r["margin"] > FACTOR * s_of(r))


def pct(n, tot):
    return 100.0 * n / tot


def dist(vals):
    a = np.array(vals, dtype=float)
    return [a.min(), *np.percentile(a, [50, 90, 95, 99]), a.max()]


def fmt(x, n=4):
    return f"{x:.{n}f}"


def main():
    data = {p: per_step(p) for p in PROMPTS}
    out = []
    w = out.append
    # spread values: floor applied; primary pool = every step where the three runs all have the ref token in top-5
    def floored(r):
        return max(r["spread"], FLOOR) if r["spread"] is not None else None

    pool_all = [floored(r) for p in PROMPTS for r in data[p] if r["spread"] is not None]
    pool_agree = [floored(r) for p in PROMPTS for r in data[p] if r["agree"]]
    s_p95_all, s_p95_agree = float(np.percentile(pool_all, 95)), float(np.percentile(pool_agree, 95))
    s_max_all, s_max_agree = max(pool_all), max(pool_agree)

    w("## Disagreement and distributions\n")
    w("| prompt | steps | ref top-1 disagreement (argmax differs in some run) | of those: ref token missing from a top-5 | S_t undefined |")
    w("|---|---|---|---|---|")
    for p in PROMPTS:
        rows = data[p]
        dis = sum(1 for r in rows if not r["agree"])
        miss = sum(1 for r in rows if r["missing"])
        w(f"| {p} | {len(rows)} | {dis} | {miss} | {miss} |")
    tot_dis = sum(1 for p in PROMPTS for r in data[p] if not r["agree"])
    w(f"| all | {128*5} | {tot_dis} | {sum(1 for p in PROMPTS for r in data[p] if r['missing'])} | |\n")
    w("Disagreement steps (any of CPU-t6 / CUDA argmax differs from CPU-t12), with the run that differs:\n")
    for p in PROMPTS:
        items = [f"{r['i']}({'t6' if r['argmax_cpu6'] != r['ref'] else ''}{'+' if r['argmax_cpu6'] != r['ref'] and r['argmax_cuda'] != r['ref'] else ''}{'cuda' if r['argmax_cuda'] != r['ref'] else ''})" for r in data[p] if not r["agree"]]
        w(f"- {p}: " + (", ".join(items) if items else "none"))
    w("")

    def dist_table(title, getter, rows_filter):
        w(f"### {title}\n")
        w("| prompt | n | min | p50 | p90 | p95 | p99 | max |")
        w("|---|---|---|---|---|---|---|---|")
        allv = []
        for p in PROMPTS:
            v = [getter(r) for r in data[p] if rows_filter(r)]
            allv += v
            w(f"| {p} | {len(v)} | " + " | ".join(fmt(x) for x in dist(v)) + " |")
        w(f"| overall | {len(allv)} | " + " | ".join(fmt(x) for x in dist(allv)) + " |\n")

    dist_table("S_t (spread floored at 0.05), all steps where defined (primary pool)", floored, lambda r: r["spread"] is not None)
    dist_table("S_t (floored), only steps where all three argmax agree", floored, lambda r: r["agree"])
    dist_table("Raw spread (no floor), all steps where defined", lambda r: r["spread"], lambda r: r["spread"] is not None)
    dist_table("margin_t (reference CPU -t12 top-1 minus top-2), all steps", lambda r: r["margin"], lambda r: True)

    def rule_table(title, s_of, note):
        w(f"### {title}\n")
        w(note + "\n")
        w("| prompt | S | gated | excluded | excluded % | <=5% cap |")
        w("|---|---|---|---|---|---|")
        ok_all = True
        for p in PROMPTS:
            rows = data[p]
            s_desc = s_of(p, None)
            g = sum(1 for r in rows if r["agree"] and r["margin"] > FACTOR * s_of(p, r))
            e = len(rows) - g
            ok = pct(e, len(rows)) <= CAP_PCT
            ok_all &= ok
            w(f"| {p} | {s_desc} | {g} | {e} | {pct(e, len(rows)):.2f} | {'yes' if ok else 'NO'} |")
        w(f"\nAll five prompts within cap: {'yes' if ok_all else 'no'}\n")
        return ok_all

    res = {}
    res["R1"] = rule_table("R1 per-step S_t (git-db proposal)",
                           lambda p, r: "per-step" if r is None else max(r["spread"], FLOOR) if r["spread"] is not None else 1e9,
                           "S_t = max(spread_t, 0.05); steps with ref token missing from a run's top-5 are excluded.")
    for tag, s, label in (("all", s_p95_all, "primary pool"), ("agree", s_p95_agree, "agreeing-steps pool")):
        res[f"R2-{tag}"] = rule_table(f"R2 constant S = p95 ({label})", lambda p, r, s=s: fmt(s) if r is None else s,
                                       f"S = {s:.4f}; pool = {len(pool_all) if tag == 'all' else len(pool_agree)} step spreads (numpy linear-interpolated p95).")
    for tag, s, label in (("all", s_max_all, "primary pool"), ("agree", s_max_agree, "agreeing-steps pool")):
        res[f"R3-{tag}"] = rule_table(f"R3 constant S = global max ({label})", lambda p, r, s=s: fmt(s) if r is None else s,
                                       f"S = {s:.4f}.")
    pmax = {p: max(floored(r) for r in data[p] if r["spread"] is not None) for p in PROMPTS}
    pmax_a = {p: max(floored(r) for r in data[p] if r["agree"]) for p in PROMPTS}
    res["R4-all"] = rule_table("R4 per-prompt S = max over steps (primary pool)", lambda p, r: fmt(pmax[p]) if r is None else pmax[p], "S per prompt as shown.")
    res["R4-agree"] = rule_table("R4 per-prompt S = max over steps (agreeing-steps pool)", lambda p, r: fmt(pmax_a[p]) if r is None else pmax_a[p], "S per prompt as shown.")

    # sensitivity: constant S sweep
    w("### Sensitivity: constant S sweep (gated if all agree and margin > 2*S)\n")
    sweep = [0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2]
    w("Excluded steps (of 128) per prompt; the percentage cap is 5% = at most 6 steps (7 steps = 5.47%).\n")
    w("| S | 2S | " + " | ".join(PROMPTS) + " | all five within cap |")
    w("|---|---|" + "---|" * len(PROMPTS) + "---|")
    for s in sweep:
        ex = [128 - gate(data[p], lambda r, s=s: s) for p in PROMPTS]
        w(f"| {s} | {2*s} | " + " | ".join(f"{e} ({pct(e,128):.1f}%)" for e in ex) + f" | {'yes' if all(pct(e,128)<=CAP_PCT for e in ex) else 'no'} |")
    w("")
    # exact breakpoints: for each prompt the largest constant S within cap, and over all
    w("### Largest constant S that keeps each prompt within the 5% cap (fact, not a recommendation)\n")
    w("Excluded count at constant S = disagreement steps + agreeing steps with margin <= 2S. A prompt stays within cap while at most 6 steps are excluded, so k = 7 - (disagreement steps) is the rank of the smallest agreeing margin that must stay above 2S.\n")
    w("| prompt | disagreement steps | binding margin | largest S within cap (exclusive) |")
    w("|---|---|---|---|")
    caps = []
    for p in PROMPTS:
        rows = data[p]
        dis = sum(1 for r in rows if not r["agree"])
        allowed = 6 - dis
        ms = sorted(r["margin"] for r in rows if r["agree"])
        if allowed < 0:
            w(f"| {p} | {dis} | none (cap already exceeded) | - |")
            caps.append(-1)
            continue
        # excluded agreeing steps = count(margin <= 2S) <= allowed  => 2S < ms[allowed]
        lim = ms[allowed] / 2
        caps.append(lim)
        w(f"| {p} | {dis} | rank {allowed + 1} among agreeing margins = {ms[allowed]:.4f} | S < {lim:.4f} |")
    w(f"\nAcross all five prompts: S < {min(caps):.4f}\n" if min(caps) >= 0 else "\nCap unattainable for at least one prompt.\n")

    w("### Per-step S_t rule: marginal effect of scaling S_t by k (gated if agree and margin > 2*k*S_t)\n")
    w("| k | " + " | ".join(PROMPTS) + " |")
    w("|---|" + "---|" * len(PROMPTS))
    for k in (0.5, 1, 2, 4, 8):
        ex = [128 - sum(1 for r in data[p] if r["agree"] and r["margin"] > FACTOR * k * floored(r)) for p in PROMPTS]
        w(f"| {k} | " + " | ".join(f"{e} ({pct(e,128):.1f}%)" for e in ex) + " |")
    w("")

    # self-check
    w("## Self-check against STRATA-MSG-10 / compare files\n")
    cmp_ = json.loads((ORACLE / "compare/cpu_vs_cuda.json").read_text())
    ref = {p: load("cpu12", p) for p in PROMPTS}
    cu = {p: load("cuda", p) for p in PROMPTS}
    c6 = {p: load("cpu6", p) for p in PROMPTS}
    p3 = max(abs(a["top"][0][1] - b["top"][0][1]) for a, b in zip(ref["P3"], cu["P3"]) if a["argmax"] == b["argmax"])
    p1m = min(s["top"][0][1] - s["top"][1][1] for s in ref["P1"])
    streams = json.loads((ORACLE / "ref/streams.json").read_text())
    p1m_streams = min(s["margin"] for s in streams["P1"]["steps"])
    stream_match = all(streams[p]["steps"][i]["top"] == ref[p][i]["top"] for p in PROMPTS for i in range(128))
    p1_t6 = max(abs(a["top"][0][1] - b["top"][0][1]) for a, b in zip(ref["P1"], c6["P1"]) if a["argmax"] == b["argmax"])
    w("| quantity | recomputed here | published | match |")
    w("|---|---|---|---|")
    pub3 = cmp_["cpu_vs_cuda_follow"]["P3"]["max_top1_logit_abs_diff_same_argmax"]
    w(f"| P3 CPU-t12 vs CUDA max top-1 logit diff (same argmax steps) | {p3:.6f} | 1.778 (JSON {pub3:.6f}) | {'yes' if round(p3,3)==1.778 else 'NO'} |")
    pubm = cmp_["cpu_stream_self_check"]["P1"]["min_margin"]
    w(f"| P1 min reference margin | {p1m:.6f} (ref/streams.json: {p1m_streams:.6f}) | 0.0198 (JSON {pubm:.6f}) | {'yes' if round(p1m,4)==0.0198 else 'NO'} |")
    w(f"| P1 CPU-t12 vs CPU-t6 max top-1 diff (same argmax) | {p1_t6:.6f} | 1.315 | {'yes' if round(p1_t6,3)==1.315 else 'NO'} |")
    w(f"| ref/streams.json top-5 equals cpu/clean for all 640 steps | {stream_match} | - | {'yes' if stream_match else 'NO'} |")
    d = {p: [i for i, (a, b) in enumerate(zip(ref[p], cu[p])) if a['argmax'] != b['argmax']] for p in PROMPTS}
    pubd = {p: cmp_["cpu_vs_cuda_follow"][p]["argmax_disagree_steps"] for p in PROMPTS}
    w(f"| CPU-t12 vs CUDA argmax-disagreement steps, all prompts | {d} | {pubd} | {'yes' if d == pubd else 'NO'} |")
    print("\n".join(out))


if __name__ == "__main__":
    sys.exit(main())
