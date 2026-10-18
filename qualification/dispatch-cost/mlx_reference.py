#!/usr/bin/env python3
"""Python MLX reference for the dispatch cost graph.

The graph matches the Rust command. Two modes run: "sync" calls mx.eval after
each op, and "lazy" builds the whole chain and calls mx.eval once at the end.
Run with MLX_DISABLE_COMPILE=1. This script is a measurement command only.
"""

import argparse
import hashlib
import json
import math
import os
import sys
import time
from array import array

GROUP = 64
BITS = 4
HEADS_Q, HEADS_KV, HEAD_DIM = 8, 2, 64
ROPE_THETA, ROPE_PART, ROPE_POS = 10000.0, 64, 100
RMS_EPS = 1.0e-6
CALLS_PER_LAYER, CALLS_FIXED = 15, 2
M32 = 0xFFFFFFFF


class Stream:
    """Same xorshift32 stream as the Rust synth module."""

    def __init__(self, seed):
        state = (seed * 2654435761 + 1) & M32
        self.state = state if state else 1

    def next_u32(self):
        x = self.state
        x ^= (x << 13) & M32
        x ^= x >> 17
        x ^= (x << 5) & M32
        self.state = x
        return x

    def unit(self):
        return (self.next_u32() >> 8) / 16777216.0 - 0.5


def words(seed, count):
    s = Stream(seed)
    return array("I", (s.next_u32() for _ in range(count)))


def floats(seed, count, fn):
    s = Stream(seed)
    return array("f", (fn(s.unit()) for _ in range(count)))


def activation(seed, n):
    return floats(seed, n, lambda u: u * 2.0)


def dense(seed, n):
    return floats(seed, n, lambda u: u)


def norm(seed, n):
    return floats(seed, n, lambda u: 1.0 + 0.25 * u)


def scales(seed, n):
    return floats(seed, n, lambda u: 0.015625 * (1.0 + u))


def biases(seed, n):
    return floats(seed, n, lambda u: -7.5 * (0.015625 * (1.0 + u)))


def percentile95(sorted_values):
    n = len(sorted_values)
    rank = min(max(math.ceil(n * 0.95), 1), n)
    return sorted_values[rank - 1]


def summarize(samples):
    s = sorted(samples)
    n = len(s)
    if n == 0:
        return {"samples": 0}
    median = s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2.0
    return {
        "samples": n,
        "median_us": median,
        "p95_us": percentile95(s),
        "min_us": s[0],
        "max_us": s[-1],
        "mean_us": sum(s) / n,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=["sync", "lazy", "both"], default="both")
    ap.add_argument("--layers", default="1,8")
    ap.add_argument("--hidden", type=int, default=1024)
    ap.add_argument("--inter", type=int, default=2048)
    ap.add_argument("--seq", type=int, default=256)
    ap.add_argument("--reps", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--output")
    args = ap.parse_args()
    if os.environ.get("MLX_DISABLE_COMPILE") != "1":
        sys.exit("set MLX_DISABLE_COMPILE=1")
    if args.hidden % GROUP or args.inter % GROUP or args.seq < 1 or args.reps < 1:
        sys.exit("bad shape or repetition arguments")
    if args.warmup < 0:
        sys.exit("--warmup must not be negative")
    try:
        layer_counts = [int(v) for v in args.layers.split(",")]
    except ValueError:
        sys.exit("--layers needs comma-separated integers")
    if not layer_counts or any(v < 1 for v in layer_counts):
        sys.exit("--layers needs nonzero counts")

    import mlx.core as mx

    try:
        import numpy as np
    except ImportError:
        np = None

    def host(buf, shape, dtype):
        """Build an mx.array from a host buffer; the copy happens here."""
        if np is not None:
            base = np.frombuffer(buf, dtype=np.uint32 if dtype == mx.uint32 else np.float32)
            return mx.array(base.reshape(shape))
        return mx.array(list(buf), dtype=dtype).reshape(shape)

    h, i, s = args.hidden, args.inter, args.seq

    def packed(seed, rows, cols):
        w = cols * BITS // 32
        g = cols // GROUP
        return (
            host(words(seed, rows * w), (rows, w), mx.uint32),
            host(scales(seed + 1, rows * g), (rows, g), mx.float32),
            host(biases(seed + 1, rows * g), (rows, g), mx.float32),
        )

    def qmm(x, wt):
        return mx.quantized_matmul(x, wt[0], wt[1], wt[2], transpose=True,
                                   group_size=GROUP, bits=BITS)

    max_layers = max(layer_counts)
    t0 = time.perf_counter()
    layers = []
    for idx in range(max_layers):
        b = 100 * (idx + 1)
        layer = {
            "w1": host(norm(b + 1, h), (h,), mx.float32),
            "w2": host(norm(b + 5, h), (h,), mx.float32),
            "v": host(dense(b + 4, s * h), (s, h), mx.float32),
            "score": packed(b + 2, s, h),
            "gate": packed(b + 6, i, h),
            "up": packed(b + 8, i, h),
            "down": packed(b + 10, h, i),
            "q": activation(b + 12, HEADS_Q * HEAD_DIM),
            "k": activation(b + 13, HEADS_KV * HEAD_DIM),
        }
        mx.eval(layer["w1"], layer["w2"], layer["v"],
                *[t for n in ("score", "gate", "up", "down") for t in layer[n]])
        layers.append(layer)
    setup_us = (time.perf_counter() - t0) * 1e6
    x_buf = activation(7, h)

    def to_bytes(arr):
        flat = arr.reshape(-1).tolist()
        return array("f", flat).tobytes()

    def chain(slice_, mode, rec):
        """One decode chain. Returns (wall_us, output bytes)."""
        sync = mode == "sync"
        t_start = time.perf_counter()

        def call(name, fn):
            t = time.perf_counter()
            out = fn()
            if sync:
                mx.eval(out)
            rec.setdefault(name, []).append((time.perf_counter() - t) * 1e6)
            return out

        x = call("import", lambda: host(x_buf, (1, h), mx.float32))
        pending = []
        for L in slice_:
            q = call("import", lambda: host(L["q"], (HEADS_Q, 1, HEAD_DIM), mx.float32))
            k = call("import", lambda: host(L["k"], (HEADS_KV, 1, HEAD_DIM), mx.float32))
            n = call("rms_norm", lambda: mx.fast.rms_norm(x, L["w1"], RMS_EPS))
            sc = call("qmm", lambda: qmm(n, L["score"]))
            p = call("softmax", lambda: mx.softmax(sc, axis=-1))
            a = call("mm", lambda: mx.matmul(p, L["v"]))
            x1 = call("add", lambda: x + a)
            n2 = call("rms_norm", lambda: mx.fast.rms_norm(x1, L["w2"], RMS_EPS))
            g = call("qmm", lambda: qmm(n2, L["gate"]))
            u = call("qmm", lambda: qmm(n2, L["up"]))
            sg = call("silu", lambda: g * mx.sigmoid(g))
            act = call("mul", lambda: sg * u)
            d = call("qmm", lambda: qmm(act, L["down"]))
            x = call("add", lambda: x1 + d)
            qr = call("rope", lambda: mx.fast.rope(q, ROPE_PART, traditional=False,
                                                    base=ROPE_THETA, scale=1.0, offset=ROPE_POS))
            kr = call("rope_k", lambda: mx.fast.rope(k, ROPE_PART, traditional=False,
                                                      base=ROPE_THETA, scale=1.0, offset=ROPE_POS))
            pending += [qr, kr]
        t = time.perf_counter()
        if not sync:
            mx.eval(x, *pending)
        eval_us = (time.perf_counter() - t) * 1e6
        t = time.perf_counter()
        out = to_bytes(x)
        rec.setdefault("readback", []).append((time.perf_counter() - t) * 1e6)
        wall = (time.perf_counter() - t_start) * 1e6
        if not sync:
            rec.setdefault("final_eval", []).append(eval_us)
        return wall, out

    modes = ["sync", "lazy"] if args.mode == "both" else [args.mode]
    report = {
        "schema": "inference-engine.dispatch-cost-reference.v1",
        "mlx_version": mx.__version__,
        "mlx_disable_compile": os.environ.get("MLX_DISABLE_COMPILE"),
        "python": sys.version.split()[0],
        "numpy_host_buffers": np is not None,
        "config": {"hidden": h, "inter": i, "seq": s, "heads_q": HEADS_Q,
                   "heads_kv": HEADS_KV, "head_dim": HEAD_DIM, "bits": BITS,
                   "group_size": GROUP},
        "warmup": args.warmup,
        "repetitions": args.reps,
        "setup_us": setup_us,
        "timing": "sync: mx.eval after every op, rope is two ops (q and k); lazy: one mx.eval at chain end; chain wall includes imports and readback",
        "results": [],
    }
    for mode in modes:
        for count in layer_counts:
            sl = layers[:count]
            for _ in range(args.warmup):
                chain(sl, mode, {})
            rec, walls, hashes = {}, [], []
            for _ in range(args.reps):
                wall, out = chain(sl, mode, rec)
                walls.append(wall)
                hashes.append(hashlib.sha256(out).hexdigest())
            vals = array("f")
            vals.frombytes(out)
            report["results"].append({
                "mode": mode,
                "layers": count,
                "calls_per_chain": CALLS_FIXED + CALLS_PER_LAYER * count,
                "repetitions": args.reps,
                "chain": summarize(walls),
                "per_op": {k: summarize(v) for k, v in sorted(rec.items())},
                "output_sha256": hashes[0],
                "output_stable": len(set(hashes)) == 1,
                "output_finite": all(math.isfinite(v) for v in vals),
                "output_first4": list(vals[:4]),
            })
    text = json.dumps(report, indent=2)
    if args.output:
        with open(args.output, "w") as f:
            f.write(text)
    print(text)


if __name__ == "__main__":
    main()
