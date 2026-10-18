#!/usr/bin/env python3
"""Freeze resident RMS outputs from the pinned Python MLX reference."""
import argparse
import hashlib
import json
import subprocess
from importlib.metadata import version
from pathlib import Path

import mlx.core as mx
import numpy as np


def array_bytes(array, dtype):
    if dtype == "bfloat16":
        values = np.asarray(array.astype(mx.float32), dtype="<f4").view("<u4")
        return (values >> 16).astype("<u2").tobytes().hex()
    return np.asarray(array).tobytes().hex()


def reference_input(source):
    array = mx.array(source["values"], dtype=getattr(mx, source["dtype"])).reshape(source["shape"])
    return {**source, "bytes": array_bytes(array, source["dtype"])}


def freeze(cell_id, x, weight, eps):
    x = reference_input(x)
    weight = reference_input(weight)
    x_array = mx.array(x["values"], dtype=getattr(mx, x["dtype"])).reshape(x["shape"])
    weight_array = mx.array(weight["values"], dtype=getattr(mx, weight["dtype"]))
    output = mx.fast.rms_norm(x_array, weight_array, eps)
    mx.eval(output)
    return {
        "id": cell_id,
        "x": x,
        "weight": weight,
        "eps": eps,
        "output": {
            "dtype": str(output.dtype).rsplit(".", 1)[-1],
            "shape": list(output.shape),
            "bytes": array_bytes(output, str(output.dtype).rsplit(".", 1)[-1]),
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--mlx-source", required=True, type=Path)
    args = parser.parse_args()
    pin = subprocess.check_output(
        ["git", "-C", str(args.mlx_source), "rev-parse", "HEAD"], text=True
    ).strip()
    fast_cpp = args.mlx_source / "mlx/fast.cpp"
    fast_hash = hashlib.sha256(fast_cpp.read_bytes()).hexdigest()
    cells = [
        freeze(
            "f32-epsilon-pass-through",
            {"dtype": "float32", "shape": [2, 2], "values": [3.0, 4.0, 1.0, 2.0]},
            {"dtype": "float32", "shape": [2], "values": [2.0, 2.0]},
            3.5,
        ),
        freeze(
            "f16-bf16-promotion",
            {"dtype": "float16", "shape": [2, 2], "values": [1.0, 2.0, 3.0, 4.0]},
            {"dtype": "bfloat16", "shape": [2], "values": [2.0, 1.0]},
            1e-6,
        ),
    ]
    data = {
        "schema": "resident-rms-norm-1",
        "reference": "mx.fast.rms_norm",
        "mlx_python_version": version("mlx"),
        "mlx_source_commit": pin,
        "mlx_fast_cpp_sha256": fast_hash,
        "cells": cells,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as output:
        json.dump(data, output, sort_keys=True, separators=(",", ":"))
        output.write("\n")
    print(hashlib.sha256(args.output.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
