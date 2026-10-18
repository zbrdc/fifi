#!/usr/bin/env python3
"""MLX pin purity check (user directive 2026-10-14: the engine follows the latest MLX release).

Two commands, standard library only; run them with `python3 -I`.

  purity.py freeze <evidence_dir> <out.json> [--mlx-version V] [--mlx-commit C] [--engine-head H]
      Reads the results of one Mac gate run (`mxc6-bank-off|on/bank-fifi.json`,
      `mxc4-route-{f32,bf16}-{off,on}/route-parity.json`) and writes the baseline.
  purity.py compare <baseline.json> <evidence_dir>
      Compares a new gate run with the baseline.
      Exit 0: the uncompiled path is bit-identical and the compiled path passes its gates.
      Exit 1: the UNCOMPILED path differs (a pin bump must not do that: stop).
      Exit 2: the uncompiled path is identical but the compiled path fails a gate.
      Exit 3: an input is missing or malformed.

The rule: the uncompiled path must stay bit-identical across an MLX pin bump (bank quality fields and the
route-parity logits hashes). The compiled path may move with MLX's own compile code generation; it must pass
the gates (top-1 on every bank prompt, minimum P(format) at least MIN_P, route-parity cosine at least its floor)
and its shift is reported.
"""
import hashlib
import json
import sys
from pathlib import Path

MIN_P = 0.99
BANK_FIELDS = ("p_format", "top1_id", "top1_is_format", "logit_format")
ROUTE_FIELDS = ("min_logits_cosine", "greedy_ids_equal_in_all_steps", "device_logits_sha256",
                "host_logits_sha256", "cells_differing", "cell_gap_max", "cell_gap_median")
ROUTES = ("f32-off", "f32-on", "bf16-off", "bf16-on")
MODES = ("off", "on")


class Invalid(Exception):
    pass


def load(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError) as err:
        raise Invalid(f"cannot read {path}: {err}")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bank_rows(evidence, mode):
    j = load(Path(evidence) / f"mxc6-bank-{mode}" / "bank-fifi.json")
    try:
        configs = list(j["configs"])
        rows = {c: [{f: p["results"][c][f] for f in BANK_FIELDS} for p in j["prompts"]] for c in configs}
    except (KeyError, TypeError) as err:
        raise Invalid(f"bank-fifi.json of mode {mode} is malformed: {err!r}")
    if not rows or any(not r for r in rows.values()):
        raise Invalid(f"bank-fifi.json of mode {mode} holds no prompts")
    return rows


def route_row(evidence, name):
    j = load(Path(evidence) / f"mxc4-route-{name}" / "route-parity.json")
    try:
        row = {f: j[f] for f in ROUTE_FIELDS}
        row["min_logits_cosine_floor"] = j["min_logits_cosine_floor"]
    except KeyError as err:
        raise Invalid(f"route-parity.json of {name} lacks {err}")
    return row


def freeze(evidence, out, meta):
    sources = {}
    for mode in MODES:
        p = Path(evidence) / f"mxc6-bank-{mode}" / "bank-fifi.json"
        sources[f"mxc6-bank-{mode}/bank-fifi.json"] = sha256(p)
    for name in ROUTES:
        p = Path(evidence) / f"mxc4-route-{name}" / "route-parity.json"
        sources[f"mxc4-route-{name}/route-parity.json"] = sha256(p)
    base = {
        "schema": 1,
        "meta": meta,
        "sources_sha256": sources,
        "bank": {mode: bank_rows(evidence, mode) for mode in MODES},
        "route": {name: route_row(evidence, name) for name in ROUTES},
    }
    Path(out).write_text(json.dumps(base, indent=1, sort_keys=True) + "\n")


def compare(baseline, evidence):
    base = load(baseline)
    purity, gates, shifts = [], [], []
    for mode in MODES:
        new = bank_rows(evidence, mode)
        old = base["bank"][mode]
        if set(new) != set(old):
            raise Invalid(f"bank configs differ: {sorted(new)} against {sorted(old)}")
        for cfg in sorted(new):
            if len(new[cfg]) != len(old[cfg]):
                raise Invalid(f"bank {mode}/{cfg}: {len(new[cfg])} prompts against {len(old[cfg])}")
            for i, (a, b) in enumerate(zip(old[cfg], new[cfg])):
                diff = [f for f in BANK_FIELDS if a[f] != b[f]]
                if mode == "off" and diff:
                    purity.append(f"bank off/{cfg} prompt {i}: {', '.join(diff)} differ")
                if mode == "on":
                    for f in diff:
                        if isinstance(a[f], (int, float)) and not isinstance(a[f], bool):
                            shifts.append((f"bank on/{cfg}/{f}", abs(a[f] - b[f])))
            if mode == "on":
                if not all(r["top1_is_format"] for r in new[cfg]):
                    gates.append(f"bank on/{cfg}: the format token is not top-1 on every prompt")
                low = min(r["p_format"] for r in new[cfg])
                if low < MIN_P:
                    gates.append(f"bank on/{cfg}: minimum P(format) {low:.4f} is below {MIN_P}")
    for name in ROUTES:
        new = route_row(evidence, name)
        old = base["route"][name]
        if name.endswith("-off"):
            for f in ("device_logits_sha256", "host_logits_sha256", "min_logits_cosine",
                      "greedy_ids_equal_in_all_steps"):
                if old[f] != new[f]:
                    purity.append(f"route {name}: {f} differs")
        else:
            if new["min_logits_cosine"] < new["min_logits_cosine_floor"]:
                gates.append(f"route {name}: cosine {new['min_logits_cosine']} is below the floor "
                             f"{new['min_logits_cosine_floor']}")
            if not new["greedy_ids_equal_in_all_steps"]:
                gates.append(f"route {name}: greedy ids differ in some step")
            shifts.append((f"route {name}/min_logits_cosine", abs(old["min_logits_cosine"] - new["min_logits_cosine"])))
            shifts.append((f"route {name}/cell_gap_max", abs(old["cell_gap_max"] - new["cell_gap_max"])))
    return purity, gates, shifts


def main(argv):
    try:
        if len(argv) >= 4 and argv[1] == "freeze":
            meta = {"mlx_version": "", "mlx_commit": "", "engine_head": ""}
            rest = argv[4:]
            for i in range(0, len(rest) - 1, 2):
                key = rest[i].lstrip("-").replace("-", "_")
                if key not in meta:
                    raise Invalid(f"unknown option {rest[i]}")
                meta[key] = rest[i + 1]
            freeze(argv[2], argv[3], meta)
            return 0
        if len(argv) == 4 and argv[1] == "compare":
            purity, gates, shifts = compare(argv[2], argv[3])
            for line in purity:
                print("PURITY", line)
            for line in gates:
                print("GATE", line)
            worst = {}
            for name, value in shifts:
                key = name.split("/", 1)[0] + "/" + name.split("/")[-1]
                worst[key] = max(worst.get(key, 0.0), value)
            for key in sorted(worst):
                print(f"shift {key}: max {worst[key]:.6g}")
            if purity:
                return 1
            return 2 if gates else 0
    except Invalid as err:
        print("INVALID", err)
        return 3
    print(__doc__)
    return 3


if __name__ == "__main__":
    sys.exit(main(sys.argv))
