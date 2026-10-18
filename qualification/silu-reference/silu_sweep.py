#!/usr/bin/env python3
"""Freeze SiLU variants on one fixed sweep (P1 probe, decision 0028).

Run twice with the pinned parity-env interpreter, once with MLX_DISABLE_COMPILE=1 and once
without. MLX reads only the presence of that variable, so MLX_DISABLE_COMPILE=0 also
disables compile. The script refuses to overwrite an output file and refuses any other
MLX build. It records the device, the MLX version, and the core extension digest.
"""
import argparse
import hashlib
import importlib.util
import os
import sys
from functools import partial
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

MODELS = Path("/Users/dan/edge0/.venv/lib/python3.12/site-packages/mlx_lm/models")
ACTIVATIONS = MODELS / "activations.py"
ACTIVATIONS_SHA256 = "dbca5bae41ba0a8380a53903c3e98da37c1e15b46383bc2edb5806ba94fafe72"
QWEN3_NEXT = MODELS / "qwen3_next.py"
QWEN3_NEXT_SHA256 = "997c3aee5cfad64c58a628bf93f5560ec549d2666070439e6db8e06fcb84b838"
MLX_VERSION = "0.32.2.dev20260927+1f8e74e3"
CORE_SHA256 = "935cc4e023f4a3b548981bac866e7edb708256c42b04508f5f633f1c2f7839b3"


def sha256_of(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def core_extension():
    location = Path(mx.__file__)
    if location.suffix == ".so":
        return location
    matches = sorted(location.parent.glob("core.cpython-*.so")) or sorted(
        location.parent.parent.glob("core.cpython-*.so")
    )
    if not matches:
        raise SystemExit("core extension not found")
    return matches[0]


def check_environment():
    """Refuse to run on any interpreter other than the pinned parity-env rebuild."""
    if mx.__version__ != MLX_VERSION:
        raise SystemExit(f"unexpected mlx version {mx.__version__}")
    digest = sha256_of(core_extension())
    if digest != CORE_SHA256:
        raise SystemExit(f"core extension digest mismatch: {digest}")
    return digest


def load_swiglu():
    digest = sha256_of(ACTIVATIONS)
    if digest != ACTIVATIONS_SHA256:
        raise SystemExit(f"activations.py digest mismatch: {digest}")
    spec = importlib.util.spec_from_file_location("pinned_activations", ACTIVATIONS)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.swiglu


def load_precise_swiglu():
    """Execute lines 58 to 62 of the pinned qwen3_next.py, not a copy."""
    digest = sha256_of(QWEN3_NEXT)
    if digest != QWEN3_NEXT_SHA256:
        raise SystemExit(f"qwen3_next.py digest mismatch: {digest}")
    lines = QWEN3_NEXT.read_text().splitlines()[57:62]
    if not lines[0].startswith("@partial(mx.compile") or "_precise_swiglu" not in lines[1]:
        raise SystemExit("qwen3_next.py lines 58 to 62 changed")
    namespace = {"mx": mx, "nn": nn, "partial": partial}
    exec(compile("\n".join(lines) + "\n", str(QWEN3_NEXT), "exec"), namespace)
    return namespace["_precise_swiglu"]


def sweep():
    rng = np.random.default_rng(28)
    edge = [0.0, -0.0, 1e-30, -1e-30, 1e-38, -1e-38, 88.0, -88.0, 100.0, -100.0,
            1e4, -1e4, np.inf, -np.inf, np.nan]
    parts = [
        np.linspace(-8, 8, 4097, dtype=np.float32),
        rng.uniform(-30, 30, 1 << 20).astype(np.float32),
        (rng.standard_normal(1 << 20) * 4).astype(np.float32),
        np.array(edge, dtype=np.float32),
    ]
    gate = np.concatenate(parts)
    up = np.random.default_rng(29).standard_normal(gate.size).astype(np.float32)
    return gate, up


def f32_bits(array):
    return np.asarray(array.astype(mx.float32), dtype="<f4").view("<u4")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("refusing to overwrite " + str(args.output))
    core_digest = check_environment()
    swiglu = load_swiglu()
    precise_swiglu = load_precise_swiglu()
    gate_np, up_np = sweep()
    out = {"compile_disabled": np.array(["MLX_DISABLE_COMPILE" in os.environ]),
           "compile_flag_value": np.array([os.environ.get("MLX_DISABLE_COMPILE", "<unset>")]),
           "device": np.array([str(mx.default_device())]),
           "mlx_version": np.array([mx.__version__]),
           "core_sha256": np.array([core_digest]),
           "gate_bits": gate_np.view("<u4")}
    print("mlx", mx.__version__, "device", mx.default_device(), "compile_disabled",
          out["compile_disabled"][0], "interpreter", sys.executable, file=sys.stderr)
    for name in ("float32", "bfloat16"):
        dtype = getattr(mx, name)
        x = mx.array(gate_np).astype(dtype)
        u = mx.array(up_np).astype(dtype)
        up32, x32 = u.astype(mx.float32), x.astype(mx.float32)
        variants = {
            "decomp": x * mx.sigmoid(x),
            "nn_silu": nn.silu(x),
            "swiglu": swiglu(x, u),
            "swiglu_decomp": (x * mx.sigmoid(x)) * u,
            "precise": precise_swiglu(x, x, u),
            "precise_decomp": ((x32 * mx.sigmoid(x32)) * up32).astype(dtype),
        }
        mx.eval(list(variants.values()))
        for key, value in variants.items():
            out[f"{name}.{key}"] = f32_bits(value)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as handle:
        np.savez_compressed(handle, **out)
    print(hashlib.sha256(args.output.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
