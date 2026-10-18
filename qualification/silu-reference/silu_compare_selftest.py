#!/usr/bin/env python3
"""NumPy-only self-test of silu_compare.compare for NaN, infinity, and zero edge pairs."""
import importlib.util
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location("silu_compare", Path(__file__).parent / "silu_compare.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

POS_INF, NEG_INF = 0x7F800000, 0xFF800000
QNAN, NEG_QNAN, OTHER_NAN = 0x7FC00000, 0xFFC00000, 0x7FC00001
POS_ZERO, NEG_ZERO, ONE = 0x00000000, 0x80000000, 0x3F800000


def run(left_bits, right_bits):
    left = np.array([left_bits], dtype=np.uint32)
    right = np.array([right_bits], dtype=np.uint32)
    return module.compare(left, right, np.array([0.0], dtype="<f4").view("<u4"), 1)


def expect(name, result, **fields):
    for key, value in fields.items():
        assert result[key] == value, (name, key, result[key], value)


def main():
    expect("equal values", run(ONE, ONE), bit_differences=0, nan_payload_differences=0, zero_difference=True)
    expect("equal inf", run(POS_INF, POS_INF), bit_differences=0, zero_difference=True)
    expect("equal NaN", run(QNAN, QNAN), bit_differences=0, nan_payload_differences=0, zero_difference=True)
    expect("inf vs NaN", run(POS_INF, QNAN), bit_differences=1, nan_mismatches=1, zero_difference=False)
    expect("NaN vs inf", run(QNAN, NEG_INF), bit_differences=1, nan_mismatches=1, zero_difference=False)
    expect("+inf vs -inf", run(POS_INF, NEG_INF), bit_differences=1, nan_mismatches=0, zero_difference=False)
    expect("NaN vs -NaN", run(QNAN, NEG_QNAN), bit_differences=0, nan_payload_differences=1, zero_difference=False)
    expect("NaN payload", run(QNAN, OTHER_NAN), bit_differences=0, nan_payload_differences=1, zero_difference=False)
    expect("+0 vs -0", run(POS_ZERO, NEG_ZERO), bit_differences=1, max_ulp=0, zero_difference=False)
    expect("one ulp", run(ONE, ONE + 1), bit_differences=1, max_ulp=1, zero_difference=False)
    mixed = module.compare(
        np.array([ONE, POS_INF, QNAN], dtype=np.uint32),
        np.array([ONE, POS_INF, QNAN], dtype=np.uint32),
        np.zeros(3, dtype="<f4").view("<u4"),
        1,
    )
    expect("all equal", mixed, bit_differences=0, nan_payload_differences=0, nan_mismatches=0, zero_difference=True)
    print("silu_compare self-test passed")


if __name__ == "__main__":
    main()
