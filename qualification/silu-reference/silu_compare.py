#!/usr/bin/env python3
"""Compare the frozen SiLU sweeps of the two modes (P1 probe, decision 0028). NumPy only."""
import argparse
import json
from pathlib import Path

import numpy as np

PAIRS_IN_ONE_MODE = [
    ("decomp", "nn_silu"),
    ("swiglu_decomp", "swiglu"),
    ("precise_decomp", "precise"),
]
VARIANTS = ["decomp", "nn_silu", "swiglu", "swiglu_decomp", "precise", "precise_decomp"]


def ordered(bits):
    bits = bits.astype(np.int64)
    return np.where(bits >> 31 == 1, 0x80000000 - bits, bits)


def is_nan(bits):
    """NaN means exponent all ones and mantissa nonzero. Infinity is not NaN."""
    return ((bits & 0x7F800000) == 0x7F800000) & ((bits & 0x007FFFFF) != 0)


def compare(left, right, gate_bits, scale):
    """Compare two arrays of F32 bit patterns.

    bit_differences counts elements whose bits differ, except pairs where both are NaN.
    nan_payload_differences counts pairs where both are NaN with different bits.
    nan_mismatches counts pairs where exactly one element is NaN.
    A zero-difference result needs bit_differences == 0 and nan_payload_differences == 0.
    """
    left, right = left.astype(np.uint32), right.astype(np.uint32)
    left_nan, right_nan = is_nan(left), is_nan(right)
    both_nan = left_nan & right_nan
    differ = (left != right) & ~both_nan
    nan_payload = both_nan & (left != right)
    nan_mismatch = left_nan ^ right_nan
    numeric = differ & ~nan_mismatch
    ulp = np.abs(ordered(left) - ordered(right)) // scale
    gate = gate_bits.view("<f4")
    head = slice(0, 4097)
    return {
        "count": int(left.size),
        "bit_differences": int(differ.sum()),
        "nan_payload_differences": int(nan_payload.sum()),
        "nan_mismatches": int(nan_mismatch.sum()),
        "zero_difference": bool(differ.sum() == 0 and nan_payload.sum() == 0),
        "max_ulp": int(ulp[numeric].max()) if numeric.any() else 0,
        "differences_gate_negative": int((numeric & (gate < 0)).sum()),
        "differences_gate_nonnegative": int((numeric & (gate >= 0)).sum()),
        "differences_abs_gate_ge_8": int((numeric & (np.abs(gate) >= 8)).sum()),
        "first4097_differences": int(numeric[head].sum()),
        "first4097_max_ulp": int(ulp[head][numeric[head]].max()) if numeric[head].any() else 0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eager", required=True, type=Path)
    parser.add_argument("--compiled", required=True, type=Path)
    args = parser.parse_args()
    eager, compiled = np.load(args.eager), np.load(args.compiled)
    assert bool(eager["compile_disabled"][0]) and not bool(compiled["compile_disabled"][0])
    for key in ("device", "mlx_version", "core_sha256"):
        assert eager[key][0] == compiled[key][0], key
    gate = eager["gate_bits"]
    assert (gate == compiled["gate_bits"]).all()
    report = {"device": str(eager["device"][0]), "mlx_version": str(eager["mlx_version"][0]),
              "core_sha256": str(eager["core_sha256"][0])}
    for dtype, scale in (("float32", 1), ("bfloat16", 1 << 16)):
        for mode, data in (("eager", eager), ("compiled", compiled)):
            for base, other in PAIRS_IN_ONE_MODE:
                key = f"{dtype}/{mode}/{base}_vs_{other}"
                report[key] = compare(data[f"{dtype}.{base}"], data[f"{dtype}.{other}"], gate, scale)
        for name in VARIANTS:
            key = f"{dtype}/eager_vs_compiled/{name}"
            report[key] = compare(eager[f"{dtype}.{name}"], compiled[f"{dtype}.{name}"], gate, scale)
    print(json.dumps(report, indent=1, sort_keys=True))


if __name__ == "__main__":
    main()
