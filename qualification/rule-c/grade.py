#!/usr/bin/env python3
"""Rule C grade of forced cuda-rd-raw/2 files against the frozen CPU oracle (rule C of CUDA-SERVE-PLAN,
"SERVE-1 G1 on Xavier"; the S3c retol bounds). a5_eval is imported from ../cuda-rd, resolved from this file.

usage: grade.py ORACLE_DIR RAW.json [RAW.json ...] [--gated-steps JSON]

Inputs: the ORACLE_DIR tree (cpu/clean run.json and prompt id files, checked by a5_eval against the hashes frozen in
qualification/cuda-rd/a5-gated-steps.json) and cuda-rd-raw/2 forced RAW files, which all pass the shared raw gate
(rawcheck_rulec.check_raw: schema, flags, duplicate ids, the oracle's expected prompt set for the run name, oracle
step count, finite argmax_logit/lse/top32) before any number is printed. For every step whose argmax equals the
oracle token: dev = |argmax_logit - oracle top-1 logit| (the "absolute top-1 logit deviation vs cpu12" of the S3c retol,
no step selection). Reports n / median / p95 / p99 / max over all files together, the token mismatches, the S3c bounds
verdict (median <= 0.165, p95 <= 0.651, p99 <= 1.131; 3090 native control 0.107 / 0.532 / 1.030) and a5_eval's A5
verdict for information, plus the tiered counters and the decode rate of each file.
The VERDICT line is the bounds verdict of rule C; the token mismatches in the TOTAL line are printed for information
only -- rule C's token gate (margins against 1.25x the largest independent-source flip margin, 0.511 on the
acceptance data) is computed where the flip margins live, a5_retol.py section 5. The union of prompt ids over ALL
RAW files must be the oracle's gated set exactly (P1-P5), each exactly once; --allow-partial skips that coverage
requirement and prints VERDICT PARTIAL instead of a verdict.
Exit 0 VERDICT PASS, 1 VERDICT FAIL or PARTIAL, 2 refused input (bad raw, missing or tampered oracle, unreadable
file - message on stderr), 3 crash or internal error.  stdlib only.
"""
import argparse
import json
import statistics
import sys
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
QUAL = HERE.parent / "cuda-rd"
for _p in (str(HERE), str(QUAL)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import a5_eval as ev  # noqa: E402
import rawcheck_rulec as rc  # noqa: E402

BOUNDS = (0.165, 0.651, 1.131)
CONTROL = (0.107, 0.532, 1.030)


def percentile(sorted_values, q):
    """Nearest-rank percentile of a sorted list: rank = max(1, round(q*n + 0.4999999)); the ruling's rule-C numbers
    (n=632: 0.1069 / 0.5317 / 1.0295) come from this definition, on the TOTAL line's p95/p99. a5_retol.py keeps its
    own q() (index int(x*n)); the two agree at the recorded sizes (n=622, n=632) but differ in general, e.g. at
    n=620, q=0.95, where 0.95*620 == 589.0 exactly and they pick adjacent values."""
    if not sorted_values:
        return float("nan")
    rank = max(1, round(q * len(sorted_values) + 0.4999999))
    return sorted_values[min(rank, len(sorted_values)) - 1]


def bounds_verdict(med, p95, p99):
    """S3c bounds verdict: PASS only when all three statistics sit at or below their frozen bound."""
    return med <= BOUNDS[0] and p95 <= BOUNDS[1] and p99 <= BOUNDS[2]


def grade_file(path, doc, ref):
    devs, mismatches, a5 = [], [], []
    rates, replays, counters = [], [], []
    for prompt in doc["prompts"]:
        pid = prompt["id"]
        for run in prompt["runs"]:
            stream = ref.streams[pid]
            for j, step in enumerate(run["steps"]):
                want = stream[j]
                if step["argmax"] != want["token"]:
                    mismatches.append((pid, j))
                    continue
                devs.append(abs(step["argmax_logit"] - want["top"][0][1]))
            try:
                a5.append((pid, ev.eval_a5(run, prompt["ids_sha256"], ref, pid)))
            except ev.Stop as stop:
                a5.append((pid, {"ok": None, "fails": [str(stop)]}))
            steps = len(run["steps"])
            rates.append(steps / (run["decode_ns"] / 1e9))
            replays.append(run["phase_ns"]["replay"] / steps / 1e6)
            if "tiered" in run:
                counters.append(run["tiered"])
    return doc, devs, mismatches, a5, rates, replays, counters


def main(oracle, paths, gated=None, allow_partial=False):
    if gated is None:
        ref = ev.Reference(oracle=Path(oracle))
    else:
        ref = ev.Reference(oracle=Path(oracle), gated_path=Path(gated), expect=ev.sha_file(gated))
    n_steps = {pid: len(stream) for pid, stream in ref.streams.items()}
    docs = []
    for path in paths:  # the shared raw gate, for every file, before any number is printed
        doc = json.load(open(path))
        rc.check_raw(doc, Path(path).name, fields=("argmax_logit", "lse", "top32"), n_steps=n_steps, mode="forced")
        docs.append((path, doc))
    partial = rc.check_coverage([(Path(p).name, [pr["id"] for pr in d["prompts"]]) for p, d in docs],
                                sorted(ref.streams), allow_partial=allow_partial)
    all_devs, all_miss = [], []
    for path, doc in docs:
        doc, devs, miss, a5, rates, replays, counters = grade_file(path, doc, ref)
        all_devs += devs
        all_miss += miss
        ident = doc.get("identities", {})
        print(f"== {Path(path).name}: prompts {[p['id'] for p in doc['prompts']]}, tiered {ident.get('tiered')}, "
              f"steps graded {len(devs)}, token mismatches {len(miss)}")
        print(f"   decode {', '.join(f'{r:.2f}' for r in rates)} tok/s; replay {', '.join(f'{r:.1f}' for r in replays)} ms/step")
        for pid, verdict in a5:
            print(f"   A5 {pid}: ok={verdict.get('ok')} fails={len(verdict.get('fails', []))} max_dev={verdict.get('max_logit_dev')}")
        for c in counters:
            hits, served = c.get("bank_hits", 0), c.get("cpu_served", 0)
            routed, cpu_w = c.get("routed_weight_ppm", 0), c.get("cpu_weight_ppm", 0)
            picks = hits + served
            print(f"   counters: bank_hits {hits}, cpu_served {served} -> CPU pick share "
                  f"{100 * served / picks if picks else 0:.1f}%, CPU weight share {100 * cpu_w / routed if routed else 0:.1f}%, "
                  f"swaps {c.get('swaps')}, ssd_misses {c.get('ssd_misses')}, ssd_bytes {c.get('ssd_bytes')}")
    ordered = sorted(all_devs)
    med = statistics.median(ordered) if ordered else float("nan")
    p95, p99 = percentile(ordered, 0.95), percentile(ordered, 0.99)
    top = max(ordered) if ordered else float("nan")
    print(f"TOTAL n={len(ordered)} (token mismatches {len(all_miss)}): median {med:.3f} / p95 {p95:.3f} / p99 {p99:.3f} / max {top:.2f}")
    print(f"bounds (S3c): <= {BOUNDS[0]} / {BOUNDS[1]} / {BOUNDS[2]}; 3090 native control {CONTROL[0]} / {CONTROL[1]} / {CONTROL[2]}")
    verdict = bounds_verdict(med, p95, p99)
    print("VERDICT", "PARTIAL" if partial else ("PASS" if verdict else "FAIL"))
    return verdict and not partial


def cli(argv=None):
    ap = argparse.ArgumentParser(description='rule C grade of forced raws vs the frozen oracle (S3c bounds verdict)')
    ap.add_argument('oracle', help='oracle dir (cpu/clean run.json trees and prompt id files, hashed in a5-gated-steps.json)')
    ap.add_argument('raws', nargs='+', metavar='RAW.json', help='cuda-rd-raw/2 forced run files to grade')
    ap.add_argument('--gated-steps', default=None, metavar='JSON',
                    help='a5-gated-steps.json for the reference hash table '
                         '(default: the frozen ../cuda-rd/a5-gated-steps.json, resolved from a5_eval)')
    ap.add_argument('--allow-partial', action='store_true',
                    help='grade a subset of the oracle prompts: print VERDICT PARTIAL instead of refusing '
                         '(exit 1 either way, so a partial run can never be read as a PASS)')
    args = ap.parse_args(argv)
    try:
        verdict = main(args.oracle, args.raws, args.gated_steps, args.allow_partial)
    except (rc.Refused, ev.Stop, OSError, json.JSONDecodeError) as refused:
        print(f"refused: {refused}", file=sys.stderr)
        raise SystemExit(2)
    except Exception:
        traceback.print_exc(file=sys.stderr)
        raise SystemExit(3)
    sys.exit(0 if verdict else 1)


if __name__ == "__main__":
    cli()
