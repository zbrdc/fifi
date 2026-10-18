#!/usr/bin/env python3
"""Train the ANE decision head from one engine residual capture. NumPy only, offline."""
import json
import os
import sys

import numpy as np

SEED = 1234
HIDDEN_UNITS = 256
BATCH = 256
EPOCHS = 60
LEARNING_RATE = 1.5e-3
LR_DECAY = 0.97
ADAM_B1 = 0.9
ADAM_B2 = 0.999
ADAM_EPS = 1e-8
TOP_K = 8
ARRAY_NAMES = ("w1", "b1", "w2", "b2", "mean", "scale")


def fail(message):
    print(f"ane_head_train: error: {message}", file=sys.stderr)
    sys.exit(1)


def sigmoid(values):
    """The tanh form keeps the logistic function stable for large inputs."""
    return 0.5 * (1.0 + np.tanh(0.5 * values))


def silu(values):
    return values * sigmoid(values)


def silu_grad(values):
    sig = sigmoid(values)
    return sig * (1.0 + values * (1.0 - sig))


def meta_int(meta, key):
    """Returns the integer value, or None when the key is absent. Fails on a bad value."""
    if key not in meta:
        return None
    try:
        return int(meta[key])
    except (TypeError, ValueError):
        fail(f"layer_states.json key {key} is not an integer: {meta[key]!r}")


def load_capture(states_dir):
    """Reads the capture. Returns states [tokens, layers, hidden], ids [tokens, layers, 8], experts."""
    json_path = os.path.join(states_dir, "layer_states.json")
    bin_path = os.path.join(states_dir, "layer_states.bin")
    for path in (json_path, bin_path):
        if not os.path.isfile(path):
            fail(f"missing input file {path}")
    try:
        with open(json_path, encoding="utf-8") as stream:
            meta = json.load(stream)
        ids = np.asarray(meta["expert_ids"], dtype=np.int64)
    except (OSError, KeyError, TypeError, ValueError) as error:
        fail(f"cannot read {json_path}: {error}")
    if ids.ndim != 3 or ids.shape[2] != TOP_K:
        fail(f"expert_ids must have shape [token][layer][{TOP_K}], got {list(ids.shape)}")
    tokens, layers, _ = ids.shape
    if tokens < 2:
        fail(f"need at least 2 tokens for the split, got {tokens}")
    if layers < 2:
        fail(f"need at least 2 layers to form pairs, got {layers}")
    for key, value in (("tokens", tokens), ("layers", layers), ("top_k", TOP_K)):
        recorded = meta_int(meta, key)
        if recorded is not None and recorded != value:
            fail(f"layer_states.json {key}={recorded} disagrees with expert_ids value {value}")
    try:
        states = np.fromfile(bin_path, dtype="<f4")
    except OSError as error:
        fail(f"cannot read {bin_path}: {error}")
    cells = tokens * layers
    hidden = meta_int(meta, "hidden")
    if hidden is not None:
        if states.size != cells * hidden:
            fail(f"{bin_path} holds {states.size} values, the shape needs {cells * hidden}")
    else:
        if states.size == 0 or states.size % cells != 0:
            fail(f"{bin_path} holds {states.size} values, not a multiple of {cells}")
        hidden = states.size // cells
    experts = meta_int(meta, "experts")
    if experts is None:
        experts = int(ids.max()) + 1
    if int(ids.min()) < 0 or int(ids.max()) >= experts:
        fail(f"expert ids fall outside [0, {experts})")
    return states.reshape(tokens, layers, hidden), ids, experts


def split_pairs(states, ids):
    """Layer i output predicts layer i+1 experts. First token half trains, second half is held out."""
    tokens, layers, hidden = states.shape
    half = tokens // 2
    inputs = states[:, : layers - 1, :]
    targets = ids[:, 1:, :]
    return (
        inputs[:half].reshape(-1, hidden),
        targets[:half].reshape(-1, TOP_K),
        inputs[half:].reshape(-1, hidden),
        targets[half:].reshape(-1, TOP_K),
    )


def mean_overlap(predicted, truth):
    """Mean of the intersection size of each row pair, divided by 8."""
    hits = 0
    for pred_row, true_row in zip(predicted, truth):
        hits += len(set(pred_row.tolist()) & set(true_row.tolist()))
    return hits / (TOP_K * len(predicted))


def static_baseline(y_train, y_held, experts):
    """Predicts the 8 most frequent experts of each target layer. Ties use the lower id."""
    layers = y_train.shape[1]
    table = np.empty((layers, TOP_K), dtype=np.int64)
    for layer in range(layers):
        counts = np.bincount(y_train[:, layer, :].reshape(-1), minlength=experts)
        table[layer] = np.argsort(-counts, kind="stable")[:TOP_K]
    rows = y_held.reshape(-1, TOP_K)
    hits = 0
    for row, target in enumerate(rows):
        hits += len(set(table[row % layers].tolist()) & set(target.tolist()))
    return hits / (TOP_K * len(rows))


def train(x_train, targets, experts, rng):
    """Trains the MLP with sigmoid cross-entropy and Adam. Returns float64 weights."""
    rows, width = x_train.shape
    w1 = rng.normal(0.0, np.sqrt(2.0 / width), size=(HIDDEN_UNITS, width))
    b1 = np.zeros(HIDDEN_UNITS)
    w2 = rng.normal(0.0, np.sqrt(2.0 / HIDDEN_UNITS), size=(experts, HIDDEN_UNITS))
    b2 = np.zeros(experts)
    weights = {"w1": w1, "b1": b1, "w2": w2, "b2": b2}
    first = {key: np.zeros_like(value) for key, value in weights.items()}
    second = {key: np.zeros_like(value) for key, value in weights.items()}
    step = 0
    for epoch in range(EPOCHS):
        rate = LEARNING_RATE * LR_DECAY**epoch
        order = rng.permutation(rows)
        for start in range(0, rows, BATCH):
            batch = order[start : start + BATCH]
            x = x_train[batch]
            label = np.zeros((x.shape[0], experts))
            label[np.arange(x.shape[0])[:, None], targets[batch]] = 1.0
            pre = x @ w1.T + b1
            hidden = silu(pre)
            logits = hidden @ w2.T + b2
            d_logits = (sigmoid(logits) - label) / (x.shape[0] * experts)
            d_hidden = d_logits @ w2
            d_pre = d_hidden * silu_grad(pre)
            grads = {
                "w1": d_pre.T @ x,
                "b1": d_pre.sum(axis=0),
                "w2": d_logits.T @ hidden,
                "b2": d_logits.sum(axis=0),
            }
            step += 1
            for key, value in weights.items():
                first[key] = ADAM_B1 * first[key] + (1.0 - ADAM_B1) * grads[key]
                second[key] = ADAM_B2 * second[key] + (1.0 - ADAM_B2) * grads[key] ** 2
                m_hat = first[key] / (1.0 - ADAM_B1**step)
                v_hat = second[key] / (1.0 - ADAM_B2**step)
                value -= rate * m_hat / (np.sqrt(v_hat) + ADAM_EPS)
    return weights


def head_logits(weights, x):
    hidden = silu(x @ weights["w1"].T + weights["b1"])
    return hidden @ weights["w2"].T + weights["b2"]


def write_fixture(out_dir, weights, mean, scale, spec, metrics):
    os.makedirs(out_dir, exist_ok=True)
    arrays = {
        "w1": weights["w1"].astype("<f4"),
        "b1": weights["b1"].astype("<f4"),
        "w2": weights["w2"].astype("<f4"),
        "b2": weights["b2"].astype("<f4"),
        "mean": mean.astype("<f4"),
        "scale": scale.astype("<f4"),
    }
    entries = {}
    for name in ARRAY_NAMES:
        values = arrays[name]
        with open(os.path.join(out_dir, f"{name}.bin"), "wb") as stream:
            stream.write(values.tobytes())
        entries[name] = {
            "file": f"{name}.bin",
            "shape": [int(dim) for dim in values.shape],
            "bytes": int(values.nbytes),
        }
    manifest = {
        "arrays": entries,
        "input": int(spec["input"]),
        "hidden": HIDDEN_UNITS,
        "experts": int(spec["experts"]),
        "classes": 0,
        "dtype": "f32",
        "endianness": "little",
        "tokens": int(spec["tokens"]),
        "layers": int(spec["layers"]),
        "top_k": TOP_K,
        "seed": SEED,
        "heldout_exact8": round(metrics["heldout"], 6),
        "baseline_static": round(metrics["static"], 6),
        "baseline_prev_layer": round(metrics["prev"], 6),
    }
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2)
        stream.write("\n")


def main():
    states_dir = os.environ.get("HEAD_STATES_DIR")
    out_dir = os.environ.get("HEAD_FIXTURE_OUT")
    if not states_dir:
        fail("set HEAD_STATES_DIR to the capture directory")
    if not out_dir:
        fail("set HEAD_FIXTURE_OUT to the fixture output directory")
    states, ids, experts = load_capture(states_dir)
    tokens, layers, hidden = states.shape
    half = tokens // 2
    x_train, y_train, x_held, y_held = split_pairs(states, ids)
    prev_held = ids[half:, : layers - 1, :].reshape(-1, TOP_K)
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    scale = 1.0 / np.where(std == 0.0, 1.0, std)
    z_train = (x_train - mean) * scale
    z_held = (x_held - mean) * scale
    rng = np.random.default_rng(SEED)
    weights = train(z_train, y_train, experts, rng)
    predicted = np.argsort(-head_logits(weights, z_held), axis=1, kind="stable")[:, :TOP_K]
    metrics = {
        "heldout": mean_overlap(predicted, y_held),
        "static": static_baseline(ids[:half, 1:, :], ids[half:, 1:, :], experts),
        "prev": mean_overlap(prev_held, y_held),
    }
    spec = {"input": hidden, "experts": experts, "tokens": tokens, "layers": layers}
    write_fixture(out_dir, weights, mean, scale, spec, metrics)
    report = {
        "tokens": tokens,
        "layers": layers,
        "hidden": HIDDEN_UNITS,
        "heldout_exact8": round(metrics["heldout"], 4),
        "baseline_static": round(metrics["static"], 4),
        "baseline_prev_layer": round(metrics["prev"], 4),
    }
    print("ane_head_train=" + json.dumps(report, separators=(",", ":")))


if __name__ == "__main__":
    main()
