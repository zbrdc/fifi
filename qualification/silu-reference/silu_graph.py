#!/usr/bin/env python3
"""Print the graph node names of the compiled SiLU functions (P3 probe, decision 0028).

Run once with and once without MLX_DISABLE_COMPILE=1. The script writes DOT files
into the directory given by --out and prints the node labels that contain the
fused or unfused primitive names.
"""
import argparse
import re
import sys
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from silu_sweep import check_environment, load_precise_swiglu, load_swiglu  # noqa: E402


def labels(path):
    text = Path(path).read_text()
    return sorted(set(re.findall(r'label\s*=\s*"([^"]*)"', text)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    check_environment()
    swiglu = load_swiglu()
    precise_swiglu = load_precise_swiglu()
    x = mx.array(np.linspace(-2, 2, 17, dtype=np.float32))
    u = mx.array(np.linspace(1, 3, 17, dtype=np.float32))
    cases = {
        "nn_silu": nn.silu(x),
        "swiglu": swiglu(x, u),
        "precise_swiglu": precise_swiglu(x, x, u),
        "eager_decomp": x * mx.sigmoid(x),
    }
    for name, value in cases.items():
        path = args.out / f"{name}.dot"
        mx.export_to_dot(str(path), value)
        print(name, labels(path))


if __name__ == "__main__":
    main()
