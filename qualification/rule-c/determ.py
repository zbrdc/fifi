#!/usr/bin/env python3
"""Self-determinism of two cuda-rd-raw/2 files (rule C of CUDA-SERVE-PLAN, "SERVE-1 G1 on Xavier";
the Xavier SERVE-1 acceptance pair, lane C ruling 2026-10-14).

usage: determ.py RAW_A.json RAW_B.json

Both files pass the shared raw gate (rawcheck_rulec.check_raw: schema, incomplete/refused flags, duplicate prompt
ids) and a structural check (no empty prompts, runs or step lists; the two paths must differ) before anything is
printed. Per prompt and run: the number of steps whose logits-row hash, argmax and in_token are equal in both files;
the prompts' ids_sha256 and the mode must match too.
Exit 0 DETERMINISTIC, 1 NOT IDENTICAL, 2 refused input or unreadable file (message on stderr), 3 crash or internal
error.  stdlib only.
"""
import argparse
import json
import sys
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import rawcheck_rulec as rc  # noqa: E402


def load(path):
    raw = json.loads(Path(path).read_text())
    try:
        rc.check_raw(raw, Path(path).name, fields=())
    except rc.Refused as refused:
        raise rc.Refused(str(refused)) from None
    if not raw["prompts"]:
        raise rc.Refused(f"{path}: no prompts")
    for pa in raw["prompts"]:
        for ra in pa["runs"]:  # empty runs are refused by the shared gate
            if not ra["steps"]:
                raise rc.Refused(f"{path}: {pa['id']} has an empty step list")
    return raw


def main(path_a, path_b):
    if Path(path_a).resolve() == Path(path_b).resolve():
        raise rc.Refused(f"{path_a}: the same file given twice; self-determinism compares two runs")
    a, b = load(path_a), load(path_b)
    ok = a["mode"] == b["mode"] and [p["id"] for p in a["prompts"]] == [p["id"] for p in b["prompts"]]
    ok = ok and all(pa.get("ids_sha256") == pb.get("ids_sha256") for pa, pb in zip(a["prompts"], b["prompts"]))
    for pa, pb in zip(a["prompts"], b["prompts"]):
        if len(pa["runs"]) != len(pb["runs"]):
            raise rc.Refused(f"{path_b}: {pa['id']} has {len(pb['runs'])} runs, {path_a} has {len(pa['runs'])} runs")
    print(f"mode {a['mode']} vs {b['mode']}; prompts {[p['id'] for p in a['prompts']]}")
    for pa, pb in zip(a["prompts"], b["prompts"]):
        for ra, rb in zip(pa["runs"], pb["runs"]):
            sa, sb = ra["steps"], rb["steps"]
            equal = sum(x["hash"] == y["hash"] and x["argmax"] == y["argmax"] and x["in_token"] == y["in_token"]
                        for x, y in zip(sa, sb))
            print(f"{pa['id']}: step hash, argmax and in_token equal {equal}/{len(sa)} (other file {len(sb)} steps)")
            ok = ok and equal == len(sa) == len(sb)
    print("DETERMINISTIC" if ok else "NOT IDENTICAL")
    return 0 if ok else 1


def cli(argv=None):
    ap = argparse.ArgumentParser(description='self-determinism check of two cuda-rd-raw/2 files (rule C)')
    ap.add_argument('raw_a', help='first raw file')
    ap.add_argument('raw_b', help='second raw file')
    args = ap.parse_args(argv)
    try:
        code = main(args.raw_a, args.raw_b)
    except rc.Refused as refused:
        print(f"refused: {refused}", file=sys.stderr)
        code = 2
    except (OSError, json.JSONDecodeError) as stopped:
        print(f"stopped: {stopped}", file=sys.stderr)
        code = 2
    except Exception:
        traceback.print_exc(file=sys.stderr)
        code = 3
    sys.exit(code)


if __name__ == "__main__":
    cli()
