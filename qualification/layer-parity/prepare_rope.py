#!/usr/bin/env python3
"""Freeze resident RoPE outputs from the pinned Python MLX reference on the GPU."""
import argparse
import hashlib
import json
import math
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

SCHEMA = "resident-rope-1"
MLX_VERSION = "0.32.2.dev20260927+1f8e74e3"
MLX_COMMIT = "1f8e74e3f12f31365464a6867c6579f0e9b29d85"
CORE_EXTENSION_SHA256 = "935cc4e023f4a3b548981bac866e7edb708256c42b04508f5f633f1c2f7839b3"
METALLIB_SHA256 = "85d38aba445a48f72900dd75544d7ba63176908a6dd8a768772ea46c56bb08ce"
METALLIB_BYTES = 182351120
SOURCE_SHA256 = {
    "mlx/fast.cpp": "29b31f96bdb02afdadbde3f2bcf42cf458fc69385a908bbde9fc3a14d23de077",
    "mlx/backend/metal/kernels/rope.metal": "2b35221ebea033ff43100b85eaaddec5c0e34e3cfd52e7c692573764b2b7d0ab",
    "mlx/backend/metal/rope.cpp": "9ec2d2f298381af6445b1e2281dd90521988861ed170a602944ec0f1fca17ecc",
    "python/mlx/nn/layers/positional_encoding.py": "613835daf6977ec6e0c34159349d68ddcc234958e82d18adb159d7a8bf9d0c77",
}
HEAD_DIM = 256
Q_HEADS = 16
KV_HEADS = 2
THETA = 10_000_000.0
# Bound on the formula comparison per output lane, in units of |x1| + |x2| of its input pair.
# 2^-13 is the Metal Shading Language bound for fast::sin and fast::cos on [-pi, pi]. 2^-16 covers
# the frequency, phase, and arithmetic rounding terms. The math record derives both terms.
FORMULA_BOUND = 2.0**-13 + 2.0**-16

# id, rank, batch, tokens, pos, part, theta, probes
CELLS = [
    ("identity-pos0", 3, None, 1, 0, 64, THETA, False),
    ("pos1", 3, None, 1, 1, 64, THETA, True),
    ("pos63", 3, None, 1, 63, 64, THETA, True),
    ("pos64", 3, None, 1, 64, 64, THETA, True),
    ("pos65", 3, None, 1, 65, 64, THETA, True),
    ("pos262143", 3, None, 1, 262143, 64, THETA, True),
    ("theta10000-pos1", 3, None, 1, 1, 64, 10_000.0, True),
    ("part2-pos3", 3, None, 1, 3, 2, THETA, True),
    ("part256-pos1", 3, None, 1, 1, 256, THETA, True),
    ("layout-rank3-t3-pos63", 3, None, 3, 63, 64, THETA, True),
    ("layout-rank4-t3-pos0", 4, 1, 3, 0, 64, THETA, True),
    ("layout-rank4-t3-pos63", 4, 1, 3, 63, 64, THETA, True),
    ("layout-rank4-b2-t1-pos1", 4, 2, 1, 1, 64, THETA, True),
    ("layout-rank4-b2-t2-pos1", 4, 2, 2, 1, 64, THETA, True),
    ("part256-rank4-t3-pos63", 4, 1, 3, 63, 256, THETA, True),
]


def f32(values):
    return np.asarray(values, dtype=np.float32)


def bits(array):
    return np.asarray(array, dtype="<f4").view("<u4").reshape(-1).tolist()


def grid_step(salt, head, token, lane, rotary):
    # Integer steps of 1/16. Rotary lanes use -31..-1 and 1..31. Tail lanes use distinct
    # nonzero steps, so a tail lane that moves or changes shows in the output.
    if rotary:
        n = (head * 37 + token * 23 + lane * 11 + salt) % 62
        return n - 31 if n < 31 else n - 30
    n = (lane * 7 + head * 5 + token * 3 + salt) % 401
    return n - 200 if n != 200 else 201


def build_input(shape, salt, part, probes):
    heads, tokens = shape[-3], shape[-2]
    batches = shape[0] if len(shape) == 4 else 1
    half = part // 2
    values = np.zeros((batches, heads, tokens, HEAD_DIM), dtype=np.float32)
    for batch in range(batches):
        for head in range(heads):
            for token in range(tokens):
                for lane in range(HEAD_DIM):
                    step = grid_step(salt + 101 * batch, head, token, lane, lane < part)
                    values[batch, head, token, lane] = step / 16
    if probes:
        # Head 0, token 0 of the first batch: pair 0 is (0, 1) and the last pair is (1, 0).
        # Their outputs are (-sin a, cos a) and (cos a, sin a), which expose sign and pairing.
        if half > 1:
            values[0, 0, 0, 0], values[0, 0, 0, half] = 0.0, 1.0
        values[0, 0, 0, half - 1], values[0, 0, 0, part - 1] = 1.0, 0.0
    return values.reshape(shape)


def formula(x, pos, theta, part):
    # Float64 rotation of the pair (j, j + part / 2) by pos_t * theta ** (-j / (part / 2)).
    x = np.asarray(x, dtype=np.float64)
    half = part // 2
    out = x.copy()
    inverse = np.power(float(theta), -np.arange(half, dtype=np.float64) / half)
    angle = (pos + np.arange(x.shape[-2], dtype=np.float64))[:, None] * inverse[None, :]
    cosine, sine = np.cos(angle), np.sin(angle)
    x1, x2 = x[..., :half], x[..., half:part]
    out[..., :half] = x1 * cosine - x2 * sine
    out[..., half:part] = x1 * sine + x2 * cosine
    return out


def rope(x, part, theta, pos, stream):
    out = mx.fast.rope(
        x, part, traditional=False, base=theta, scale=1.0, offset=pos, stream=stream
    )
    mx.eval(out)
    return out


def tensor_record(array):
    return {"dtype": "float32", "shape": list(array.shape), "bits": bits(array)}


def shapes(rank, batch, tokens):
    if rank == 3:
        return [Q_HEADS, tokens, HEAD_DIM], [KV_HEADS, tokens, HEAD_DIM]
    return [batch, Q_HEADS, tokens, HEAD_DIM], [batch, KV_HEADS, tokens, HEAD_DIM]


def cell(spec):
    cell_id, rank, batch, tokens, pos, part, theta, probes = spec
    q_shape, k_shape = shapes(rank, batch, tokens)
    q_values = build_input(q_shape, 3, part, probes)
    k_values = build_input(k_shape, 5, part, probes)
    outputs, inputs, checks = [], [], []
    formula_gate = pos + tokens - 1 <= math.pi
    reference = {}
    cpu_difference = 0.0
    for name, values in (("q", q_values), ("k", k_values)):
        x = mx.array(values)
        mx.eval(x)
        out = rope(x, part, theta, pos, mx.gpu)
        again = rope(x, part, theta, pos, mx.gpu)
        module = nn.RoPE(part, traditional=False, base=theta, scale=1.0)(x, offset=pos)
        mx.eval(module)
        cpu = rope(x, part, theta, pos, mx.cpu)
        if bits(out) != bits(again):
            raise SystemExit(f"{cell_id}: the GPU operation is not repeatable")
        checks.append(bits(out) == bits(module))
        cpu_difference = max(
            cpu_difference,
            float(np.max(np.abs(np.asarray(cpu, dtype=np.float32) - np.asarray(out, dtype=np.float32)))),
        )
        inputs.append(tensor_record(x))
        outputs.append(tensor_record(out))
        if formula_gate:
            reference[name] = formula(values, pos, theta, part).reshape(-1).tolist()
    record = {
        "id": cell_id,
        "step": {"pos": pos, "theta": theta, "part": part},
        "inputs": inputs,
        "outputs": outputs,
        "probes": probes,
        "max_angle_rad": float(pos + tokens - 1),
        "formula_gate": formula_gate,
        "nn_rope_bitwise_matches_fast_rope": all(checks),
        "cpu_fallback_max_abs_diff": cpu_difference,
        "identity_bitwise": inputs[0]["bits"] == outputs[0]["bits"]
        and inputs[1]["bits"] == outputs[1]["bits"],
    }
    if formula_gate:
        record["formula"] = reference
    return record


def binary_hashes():
    mlx_dir = Path(mx.__file__).parent
    names = sorted(str(path.relative_to(mlx_dir)) for path in mlx_dir.glob("core.cpython-*.so"))
    names.extend(["lib/libmlx.dylib", "lib/mlx.metallib"])
    return {name: hashlib.sha256((mlx_dir / name).read_bytes()).hexdigest() for name in names}


def check_environment(source):
    if mx.__version__ != MLX_VERSION:
        raise SystemExit(f"mlx version {mx.__version__} differs from the pinned build {MLX_VERSION}")
    hashes = binary_hashes()
    core = [digest for name, digest in hashes.items() if name.startswith("core.")]
    if core != [CORE_EXTENSION_SHA256]:
        raise SystemExit("mlx core extension differs from the pinned parity build")
    metallib = Path(mx.__file__).parent / "lib/mlx.metallib"
    if hashes["lib/mlx.metallib"] != METALLIB_SHA256 or metallib.stat().st_size != METALLIB_BYTES:
        raise SystemExit("mlx.metallib differs from the pinned parity build")
    if not mx.is_available(mx.gpu):
        raise SystemExit("the pinned Metal device is unavailable")
    mx.set_default_device(mx.gpu)
    if mx.default_device().type != mx.gpu:
        raise SystemExit("the default device is not the GPU")
    commit = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if commit != MLX_COMMIT:
        raise SystemExit(f"mlx source commit {commit} differs from the pin {MLX_COMMIT}")
    status = subprocess.check_output(["git", "-C", str(source), "status", "--porcelain"], text=True)
    if status.strip():
        raise SystemExit("the mlx source checkout is not clean")
    source_hashes = {}
    for name, expected in SOURCE_SHA256.items():
        digest = hashlib.sha256((source / name).read_bytes()).hexdigest()
        if digest != expected:
            raise SystemExit(f"{name} differs from the pinned source")
        source_hashes[name] = digest
    return hashes, commit, source_hashes


def device_record():
    info = mx.device_info()
    return {
        "default": str(mx.default_device()),
        "info": {key: value for key, value in info.items() if isinstance(value, (str, int))},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--mlx-source", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    hashes, commit, source_hashes = check_environment(args.mlx_source)
    cells = [cell(spec) for spec in CELLS]
    data = {
        "schema": SCHEMA,
        "reference": {
            "rope": "mx.fast.rope(x, part, traditional=False, base=theta, scale=1.0, offset=pos, stream=mx.gpu)",
            "formula": "float64 rotation of the pair (j, j + part / 2) by (pos + t) * theta ** (-j / (part / 2))",
            "tail": "lanes from part onward stay unchanged",
        },
        "formula_bound": FORMULA_BOUND,
        "head_dim": HEAD_DIM,
        "mlx_python_version": version("mlx"),
        "mlx_version_string": mx.__version__,
        "python_version": sys.version.split()[0],
        "numpy_version": np.__version__,
        "device": device_record(),
        "binary_hashes": hashes,
        "mlx_source_commit": commit,
        "mlx_source_sha256": source_hashes,
        "generator": "qualification/layer-parity/prepare_rope.py",
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "cells": cells,
    }
    if not cells[0]["identity_bitwise"]:
        print("warning: the position zero cell is not bitwise identity", file=sys.stderr)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as output:
        json.dump(data, output, sort_keys=True, separators=(",", ":"))
        output.write("\n")
    print(hashlib.sha256(args.output.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
