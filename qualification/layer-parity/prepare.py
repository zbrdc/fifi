#!/usr/bin/env python3
"""Freeze the layer-parity-1 fixture from the pinned Python MLX."""
import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import mlx.core as mx

SCHEMA = "layer-parity-1"
BITS = 4
GROUP = 64
HIDDEN = 2048
MOE_INTER = 512
EXPERTS = 256
GRID = 16.0
SCALE_GRID = 8.0
UNIFORM_WORD = 0x44444444
UNIFORM_EXPECTED = 144.0
TOLERANCES = {"qmm": {"abs": 0.0, "rel": 0.0}, "mm": {"abs": 0.0, "rel": 0.0}}
REFERENCE_OP = (
    "mx.quantized_matmul(transpose=True, group_size=64, bits=4, mode=affine)"
)
REFERENCE_MM_OP = "mx.matmul"
QUANT = {"mode": "affine", "bits": BITS, "group_size": GROUP}


def draw(seed, count, lo, hi):
    state = seed & 0xFFFFFFFF
    out = []
    for _ in range(count):
        state = (state * 1664525 + 1013904223) & 0xFFFFFFFF
        out.append(lo + ((state >> 8) % (hi - lo + 1)))
    return out


def exact_values(seed, count):
    return [value / GRID for value in draw(seed, count, -63, 63)]


def scale_values(seed, count):
    return [value / SCALE_GRID for value in draw(seed, count, 4, 44)]


def bias_values(seed, count):
    return [value / SCALE_GRID for value in draw(seed, count, -32, 32)]


def packed_weight(seed, logical):
    rows, cols = logical
    packed_cols = cols * BITS // 32
    scale_cols = cols // GROUP
    return {
        "logical": [rows, cols],
        "payload": {
            "shape": [rows, packed_cols],
            "words": draw(seed, rows * packed_cols, 0, 2**32 - 1),
        },
        "scales": {
            "shape": [rows, scale_cols],
            "dtype": "bfloat16",
            "values": scale_values(seed + 1, rows * scale_cols),
        },
        "biases": {
            "shape": [rows, scale_cols],
            "dtype": "bfloat16",
            "values": bias_values(seed + 2, rows * scale_cols),
        },
    }


def freeze(out):
    mx.eval(out)
    values = out.astype(mx.float32).reshape(-1).tolist()
    if not all(math.isfinite(value) for value in values):
        raise SystemExit("reference produced a nonfinite value")
    return values


def reference_qmm(x_values, weight):
    cols = len(x_values)
    x = mx.array(x_values, mx.float32).reshape([1, cols])
    payload = mx.array(weight["payload"]["words"], mx.uint32)
    payload = payload.reshape(weight["payload"]["shape"])
    scales = mx.array(weight["scales"]["values"], mx.bfloat16)
    scales = scales.reshape(weight["scales"]["shape"])
    biases = mx.array(weight["biases"]["values"], mx.bfloat16)
    biases = biases.reshape(weight["biases"]["shape"])
    out = mx.quantized_matmul(
        x, payload, scales, biases,
        transpose=True, group_size=GROUP, bits=BITS, mode="affine",
    )
    return freeze(out)


def qmm_cell(cell_id, seed_a, seed_w, logical):
    rows, cols = logical
    weight = packed_weight(seed_w, logical)
    activation = {
        "dtype": "float32",
        "shape": [1, cols],
        "values": exact_values(seed_a, cols),
    }
    expected = reference_qmm(activation["values"], weight)
    return {
        "id": cell_id,
        "kind": "qmm",
        "family": "qmm",
        "reference_op": REFERENCE_OP,
        "quant": dict(QUANT),
        "activation": activation,
        "weight": weight,
        "expected": {"shape": [1, rows], "values": expected},
    }


def uniform_probe():
    rows, cols = 1, 64
    packed_cols = cols * BITS // 32
    weight = {
        "logical": [rows, cols],
        "payload": {
            "shape": [rows, packed_cols],
            "words": [UNIFORM_WORD] * (rows * packed_cols),
        },
        "scales": {"shape": [rows, 1], "dtype": "bfloat16", "values": [0.5]},
        "biases": {"shape": [rows, 1], "dtype": "bfloat16", "values": [0.25]},
    }
    activation = {"dtype": "float32", "shape": [1, cols], "values": [1.0] * cols}
    expected = reference_qmm(activation["values"], weight)
    if expected != [UNIFORM_EXPECTED]:
        raise SystemExit(f"uniform probe mismatch: {expected}")
    return {
        "id": "qmm-uniform-hand",
        "kind": "qmm",
        "family": "qmm",
        "reference_op": REFERENCE_OP,
        "quant": dict(QUANT),
        "activation": activation,
        "weight": weight,
        "expected": {"shape": [1, rows], "values": [UNIFORM_EXPECTED]},
    }


def reference_mm(x_values, weight_values, dims):
    m, k, n = dims
    x = mx.array(x_values, mx.float32).reshape([m, k])
    w = mx.array(weight_values, mx.float32).reshape([k, n])
    return freeze(mx.matmul(x, w))


def mm_cell(cell_id, seed_a, seed_w, dims):
    m, k, n = dims
    activation = {
        "dtype": "float32",
        "shape": [m, k],
        "values": exact_values(seed_a, m * k),
    }
    weight = {
        "dtype": "float32",
        "shape": [k, n],
        "values": exact_values(seed_w, k * n),
    }
    expected = reference_mm(activation["values"], weight["values"], dims)
    return {
        "id": cell_id,
        "kind": "mm",
        "family": "mm",
        "reference_op": REFERENCE_MM_OP,
        "activation": activation,
        "weight": weight,
        "expected": {"shape": [m, n], "values": expected},
    }


def binary_hashes():
    mlx_dir = Path(mx.__file__).parent
    names = sorted(str(path.relative_to(mlx_dir))
                   for path in mlx_dir.glob("core.cpython-*.so"))
    names.append("lib/libmlx.dylib")
    out = {}
    for name in names:
        data = (mlx_dir / name).read_bytes()
        out[name] = hashlib.sha256(data).hexdigest()
    return out


def provenance():
    return {
        "generator": "qualification/layer-parity/prepare.py",
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "mlx_python_version": mx.__version__,
        "python_version": sys.version.split()[0],
        "binary_hashes": binary_hashes(),
    }


def build_fixture():
    return {
        "schema": SCHEMA,
        "tolerances": TOLERANCES,
        "cells": [
            qmm_cell("qmm-gate-up", 101, 201, [MOE_INTER, HIDDEN]),
            qmm_cell("qmm-down", 102, 202, [HIDDEN, MOE_INTER]),
            mm_cell("mm-router", 301, 401, (1, HIDDEN, EXPERTS)),
            mm_cell("mm-mix", 302, 402, (1, MOE_INTER, 64)),
            mm_cell("mm-tiny", 303, 403, (2, 4, 3)),
        ],
        "probes": [uniform_probe()],
        "provenance": provenance(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    out_path = Path(args.output)
    if out_path.exists():
        raise SystemExit(f"refusing to overwrite {out_path}")
    fixture = build_fixture()
    out_path.write_text(json.dumps(fixture, sort_keys=True, indent=2) + "\n")
    digest = hashlib.sha256(out_path.read_bytes()).hexdigest()
    print(f"fixtures_sha256 {digest}")
    print(f"cells {len(fixture['cells'])}")


if __name__ == "__main__":
    main()
