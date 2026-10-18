#!/usr/bin/env python3
"""Write a5-gated-steps.json: the exact A5 gated and excluded step sets (REPLY-40/41).

Companion to a5_stats.py (imported unchanged, same per-step data and gating rule). Reads the oracle only.
Output is deterministic: no timestamps, sorted keys, 1-space indent, floats rounded to 9 significant digits.
Usage: a5_gated_steps.py OUTPUT.json
"""
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("a5_stats", HERE / "a5_stats.py")
a5 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(a5)

ORACLE = a5.ORACLE
SCHEMA = "a5-gated-steps/1"
RULE_TEXT = (
    "REPLY-40 items 1-3 as frozen by REPLY-41. S_t is the largest pairwise difference of the top-1 logit at step t "
    "among the three frozen references (llama.cpp CPU -t 12, CPU -t 6, and the CUDA build), all teacher-forced on "
    "the CPU -t 12 greedy stream, with a floor of 0.05; the top-1 logit of each reference is its logit for the CPU "
    "-t 12 top-1 token. Gated step: all three references agree on the top-1 token at step t, and the CPU -t 12 "
    "top-1 margin at t is greater than 2 S_t. Excluded steps are not graded."
)
PASS_RULE = (
    "On every gated step of all five prompts, fifi's teacher-forced argmax must equal reference_token, and "
    "|fifi logit for reference_token - ref_top1_logit| must be at most tolerance (2 * S_t). Excluded steps are "
    "not graded. Top-5 set agreement and the log-sum-exp difference are reported per step, not gated."
)


def sha_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def f9(x):
    """Round to 9 significant digits; json then writes the shortest repr of that float."""
    return float(f"{x:.9g}")


def manifest_files():
    return json.loads((ORACLE / "MANIFEST.json").read_text())["files"]


def input_hashes():
    files = manifest_files()
    out = {"MANIFEST.json": sha_file(ORACLE / "MANIFEST.json")}
    for p in a5.PROMPTS:
        rels = [f"{d}/{p}/run.json" for d in a5.RUNS.values()] + [f"prompts/{p}.ids.json", f"prompts/{p}.ids"]
        for rel in rels:
            h = sha_file(ORACLE / rel)
            if h != files[rel]["sha256"]:
                raise SystemExit(f"hash mismatch against MANIFEST.json: {rel}")
            out[rel] = h
    return dict(sorted(out.items()))


def prompt_ids_sha(p):
    ids = json.loads((ORACLE / f"prompts/{p}.ids.json").read_text())
    joined = ",".join(str(i) for i in ids["ids"])
    sha = hashlib.sha256(joined.encode()).hexdigest()
    if sha != ids["ids_sha256"]:
        raise SystemExit(f"{p}: recomputed ids sha differs from prompts file")
    return sha, ids["n_tokens"]


def prompt_block(p):
    rows = a5.per_step(p)
    ref = a5.load("cpu12", p)
    sha, n_prompt = prompt_ids_sha(p)
    gated, excluded = [], []
    for r in rows:
        s_t = max(r["spread"], a5.FLOOR)
        if not r["agree"]:
            excluded.append({"step": r["i"], "reason": "reference_disagreement"})
        elif r["margin"] > a5.FACTOR * s_t:
            gated.append({"step": r["i"], "reference_token": r["ref"], "ref_top1_logit": f9(ref[r["i"]]["top"][0][1]),
                          "S_t": f9(s_t), "tolerance": f9(a5.FACTOR * s_t), "margin_t": f9(r["margin"])})
        else:
            excluded.append({"step": r["i"], "reason": "margin_le_2S"})
    n = len(rows)
    return {"prompt_ids_sha256": sha, "n_prompt_tokens": n_prompt, "n_steps": n,
            "n_gated": len(gated), "n_excluded": len(excluded),
            "n_excluded_reference_disagreement": sum(e["reason"] == "reference_disagreement" for e in excluded),
            "n_excluded_margin_le_2S": sum(e["reason"] == "margin_le_2S" for e in excluded),
            "excluded_share_percent": f9(100.0 * len(excluded) / n),
            "gated": gated, "excluded": excluded}


def main():
    manifest = json.loads((ORACLE / "MANIFEST.json").read_text())
    ident = manifest["identities"]
    doc = {
        "schema": SCHEMA,
        "rule": RULE_TEXT,
        "pass_rule": PASS_RULE,
        "floor": a5.FLOOR,
        "tolerance_factor": a5.FACTOR,
        "number_format": "floats rounded to 9 significant digits (printf %.9g) then written as the shortest "
                         "round-trip decimal; S_t, tolerance, margin_t and ref_top1_logit are each rounded "
                         "independently from the full-precision value; the checker uses the tolerance field",
        "json_format": "sorted keys, 1-space indent, ensure_ascii, trailing newline",
        "inputs_sha256": input_hashes(),
        "oracle_dir": str(ORACLE),
        "references": {"cpu_t12": "cpu/clean", "cpu_t6": "cpu/threads6", "cuda": "cuda/follow"},
        "gguf": {"sha256": ident["gguf"]["sha256"], "size": ident["gguf"]["size"], "origin": ident["gguf"]["origin"]},
        "llama_cpp_commit": ident["llama_cpp_commit"],
        "scripts": {"a5_gated_steps.py": sha_file(HERE / "a5_gated_steps.py"), "a5_stats.py": sha_file(HERE / "a5_stats.py")},
        "prompts": {p: prompt_block(p) for p in a5.PROMPTS},
    }
    Path(sys.argv[1]).write_text(json.dumps(doc, sort_keys=True, indent=1) + "\n")


if __name__ == "__main__":
    main()
