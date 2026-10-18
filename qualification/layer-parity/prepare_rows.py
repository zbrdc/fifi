#!/usr/bin/env python3
"""Freeze resident row-operation outputs from the pinned Python MLX GPU reference."""
import argparse
import hashlib
import json
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path

import mlx.core as mx
import numpy as np

SCHEMA = "resident-rows-1"
PYTHON_VERSION = "3.12.14"
MLX_VERSION = "0.32.2.dev20260927+1f8e74e3"
MLX_COMMIT = "1f8e74e3f12f31365464a6867c6579f0e9b29d85"


def bits(array):
    return np.asarray(array, dtype="<f4").view("<u4").reshape(-1).tolist()


def tensor(values, shape):
    array = mx.array(np.asarray(values, dtype=np.float32).reshape(shape))
    mx.eval(array)
    return array, {"dtype": "float32", "shape": list(shape), "bits": bits(array)}


def broadcast_cell(cell_id, values, width, rows):
    source, frozen_source = tensor(values, (1, width))
    output = mx.broadcast_to(source, (rows, width))
    mx.eval(output)
    return {
        "id": cell_id,
        "op": "broadcast_rows",
        "rows": rows,
        "inputs": [frozen_source],
        "output": {"dtype": "float32", "shape": list(output.shape), "bits": bits(output)},
    }


def weighted_row_sum_cell(cell_id, row_values, weights, count, width):
    source_rows, frozen_rows = tensor(row_values, (count, width))
    source_weights, frozen_weights = tensor(weights, (count, 1, 1))
    flat_weights = mx.reshape(source_weights, (count, 1))
    output = mx.sum(mx.multiply(source_rows, flat_weights), axis=0, keepdims=True)
    mx.eval(output)
    return {
        "id": cell_id,
        "op": "weighted_row_sum",
        "inputs": [frozen_rows, frozen_weights],
        "output": {"dtype": "float32", "shape": list(output.shape), "bits": bits(output)},
    }


def check_environment(mlx_source):
    if sys.version.split()[0] != PYTHON_VERSION:
        raise SystemExit(f"Python {sys.version.split()[0]} differs from pin {PYTHON_VERSION}")
    if mx.__version__ != MLX_VERSION or version("mlx") != MLX_VERSION:
        raise SystemExit(f"MLX {mx.__version__} differs from pin {MLX_VERSION}")
    commit = subprocess.check_output(
        ["git", "-C", str(mlx_source), "rev-parse", "HEAD"], text=True
    ).strip()
    if commit != MLX_COMMIT:
        raise SystemExit(f"MLX source commit {commit} differs from pin {MLX_COMMIT}")
    status = subprocess.check_output(
        ["git", "-C", str(mlx_source), "status", "--porcelain"], text=True
    )
    if status.strip():
        raise SystemExit("the MLX source checkout is not clean")
    if not mx.is_available(mx.gpu):
        raise SystemExit("the MLX GPU is unavailable")
    mx.set_default_device(mx.gpu)
    if mx.default_device().type != mx.gpu:
        raise SystemExit("the MLX default device is not the GPU")
    ops_hash = hashlib.sha256((mlx_source / "mlx/ops.cpp").read_bytes()).hexdigest()
    return commit, ops_hash


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--mlx-source", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    commit, ops_hash = check_environment(args.mlx_source)
    cells = [
        broadcast_cell("broadcast-hand", [1.0, 2.0], 2, 3),
        broadcast_cell("broadcast-wide", [0.125, -0.75, 1.5, 2.25], 4, 5),
        weighted_row_sum_cell(
            "weighted-hand", [1.0, 2.0, 3.0, 4.0], [0.25, 0.75], 2, 2
        ),
        weighted_row_sum_cell(
            "weighted-irregular",
            [0.5, -1.25, 2.0, 3.5, 4.25, -0.75, -2.0, 0.125, 1.0, 5.0, 2.25, -3.0],
            [0.125, 0.5, 0.375],
            3,
            4,
        ),
    ]
    data = {
        "schema": SCHEMA,
        "reference": {
            "broadcast_rows": "mx.broadcast_to(x, (rows, width))",
            "weighted_row_sum": "mx.sum(mx.multiply(rows, mx.reshape(weights, (K, 1))), axis=0, keepdims=True)",
        },
        "python_version": sys.version.split()[0],
        "mlx_python_version": version("mlx"),
        "numpy_version": np.__version__,
        "mlx_source_commit": commit,
        "mlx_ops_cpp_sha256": ops_hash,
        "device": str(mx.default_device()),
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "cells": cells,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as output:
        json.dump(data, output, sort_keys=True, separators=(",", ":"))
        output.write("\n")
    print(hashlib.sha256(args.output.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
