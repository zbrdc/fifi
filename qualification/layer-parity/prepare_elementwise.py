#!/usr/bin/env python3
"""Freeze resident elementwise outputs from the pinned Python MLX reference."""
import argparse
import hashlib
import json
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

SCHEMA = "resident-elementwise-1"


def f32(values):
    return np.asarray(values, dtype=np.float32)


def bits(array):
    return np.asarray(array, dtype="<f4").view("<u4").reshape(-1).tolist()


def tensor(values, shape):
    array = mx.array(f32(values).reshape(shape))
    return array, {"dtype": "float32", "shape": list(shape), "bits": bits(array)}


def silu(x):
    # The Rust shim multiplies the input by its sigmoid in F32, so the reference does the same.
    return x * mx.sigmoid(x)


def unary(cell_id, op, reference, values, shape):
    x, frozen = tensor(values, shape)
    output = reference(x)
    mx.eval(output)
    return {
        "id": cell_id,
        "op": op,
        "inputs": [frozen],
        "output": {"dtype": "float32", "shape": list(output.shape), "bits": bits(output)},
    }


def binary(cell_id, op, reference, left, right, shape):
    a, frozen_a = tensor(left, shape)
    b, frozen_b = tensor(right, shape)
    output = reference(a, b)
    mx.eval(output)
    return {
        "id": cell_id,
        "op": op,
        "inputs": [frozen_a, frozen_b],
        "output": {"dtype": "float32", "shape": list(output.shape), "bits": bits(output)},
    }


def softmax(x):
    return mx.softmax(x, axis=-1, precise=True)


def lattice():
    return [step / 4 for step in range(-8, 9)]


def reciprocal_operands(numerators, denominator):
    return f32(numerators) / np.float32(denominator)


def cells():
    grid = lattice()
    mul_left = reciprocal_operands([3, -2, 5, 1, 7, -4], 3)
    mul_right = reciprocal_operands([-5, 4, 2, 9, -1, 6], 7)
    add_left = reciprocal_operands([1, 10, -3, 8, 100, -7], 10)
    add_right = reciprocal_operands([2, -10, 6, 1, 9, 7], 10)
    softmax_rows = reciprocal_operands([0, 4, 8, -12, -4, 2], 4)
    return [
        unary("silu-hand", "silu", silu, [0.0, 1.0], (2,)),
        unary("silu-lattice", "silu", silu, grid, (len(grid),)),
        binary("mul-ones", "mul", mx.multiply, [1.0] * 6, [1.0] * 6, (2, 3)),
        binary("mul-lattice", "mul", mx.multiply, mul_left, mul_right, (2, 3)),
        binary("add-ones", "add", mx.add, [1.0] * 6, [1.0] * 6, (2, 3)),
        binary("add-lattice", "add", mx.add, add_left, add_right, (2, 3)),
        unary("softmax-hand", "softmax", softmax, [0.0, 1.0], (2,)),
        unary("softmax-lattice", "softmax", softmax, grid, (len(grid),)),
        unary("softmax-rows", "softmax", softmax, softmax_rows, (2, 3)),
    ]


def decomposition_matches_nn_silu():
    x = mx.array(f32(lattice()))
    mx.eval(x)
    left, right = silu(x), nn.silu(x)
    mx.eval(left, right)
    return bits(left) == bits(right)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--mlx-source", required=True, type=Path)
    args = parser.parse_args()
    pin = subprocess.check_output(
        ["git", "-C", str(args.mlx_source), "rev-parse", "HEAD"], text=True
    ).strip()
    ops_hash = hashlib.sha256((args.mlx_source / "mlx/ops.cpp").read_bytes()).hexdigest()
    data = {
        "schema": SCHEMA,
        "reference": {
            "silu": "x * mx.sigmoid(x)",
            "mul": "mx.multiply",
            "add": "mx.add",
            "softmax": "mx.softmax(x, axis=-1, precise=True)",
        },
        "mlx_python_version": version("mlx"),
        "python_version": sys.version.split()[0],
        "numpy_version": np.__version__,
        "mlx_source_commit": pin,
        "mlx_ops_cpp_sha256": ops_hash,
        "nn_silu_bitwise_matches_decomposition": decomposition_matches_nn_silu(),
        "cells": cells(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as output:
        json.dump(data, output, sort_keys=True, separators=(",", ":"))
        output.write("\n")
    print(hashlib.sha256(args.output.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
