# ANE decision-head training fixture

This directory holds offline tooling that makes a trained decision head reproducible.
A head trained outside the repository reached a 43.7 percent held-out exact-8 hit rate.
`train.py` reproduces that head from a residual capture. The scripts use Python with
NumPy only. They do not use the network and they do not depend on the engine build.

## Inputs

`train.py` reads the capture directory from `HEAD_STATES_DIR`. The engine residual
capture produces these files:

- `layer_states.bin`: raw little-endian f32 values, row-major `[token][layer][hidden]`.
- `layer_states.json`: metadata plus `expert_ids`, an integer array with shape
  `[token][layer][8]`. Each row lists the routed experts of one layer in router rank
  order.

The optional keys `tokens`, `layers`, `top_k`, `hidden`, and `experts` are checked
against the arrays when they are present. `hidden` and `experts` are derived from the
data when they are absent.

## Pairs and split

The script forms one pair per token for each layer index `i` from 0 to `layers - 2`.
The residual output of layer `i` is the input. The routed experts of layer `i + 1`
are the target. The first half of the tokens trains the model. The second half is
held out.

## Model and metric

- Inputs are standardized with the training-split mean and standard deviation. A
  channel with zero standard deviation uses scale 1.
- The model computes `logits = W2 * silu(W1 * x + b1) + b2` with 256 hidden units and
  one output per expert.
- The loss is sigmoid cross-entropy over all experts. The optimizer is Adam with
  batch 256, 60 epochs, and a learning rate of 1.5e-3 that decays by 0.97 each epoch.
- The seed is fixed. Equal input gives equal numbers on one NumPy build.
- The metric is the exact-8 hit rate: the mean overlap of the predicted top-8 experts
  with the true top-8 experts, divided by 8, on the held-out half.
- The static baseline predicts the 8 most frequent experts of each target layer from
  the training split. The previous-layer baseline predicts the experts of layer `i`.
  Both baselines use the same pairs as the model.

## Outputs

`train.py` writes the fixture directory from `HEAD_FIXTURE_OUT`:

- `w1.bin`, shape `[hidden, input]`; `b1.bin`, shape `[hidden]`.
- `w2.bin`, shape `[experts, hidden]`; `b2.bin`, shape `[experts]`.
- `mean.bin` and `scale.bin`, shape `[input]`. `scale.bin` holds the reciprocal of the
  standard deviation, so the engine applies `(x - mean) * scale`.
- `manifest.json` with one `file`, `shape`, and `bytes` entry per array, the top-level
  `input`, `hidden`, `experts`, `classes`, `dtype`, and `endianness` fields, and the
  measured accuracies. `classes` is 0 because the trainer adds no class outputs.

All binary files hold raw little-endian f32 values.

## Command

Run from the repository root:

```sh
HEAD_STATES_DIR=/path/to/states HEAD_FIXTURE_OUT=/path/to/fixture \
  python3 qualification/ane-head/train.py
python3 qualification/ane-head/verify_fixture.py /path/to/fixture
```

`train.py` prints one line. The first three fields follow the capture:

```text
ane_head_train={"tokens":128,"layers":48,"hidden":256,"heldout_exact8":0.437,"baseline_static":0.237,"baseline_prev_layer":0.029}
```

`verify_fixture.py` checks every file size against the manifest and prints:

```text
ane_head_fixture={"files":6,"bytes":N,"ok":true}
```

## Expected numbers

The measured run reached a held-out exact-8 hit rate of 43.7 percent. The static
baseline reached 23.7 percent. The previous-layer baseline reached 2.9 percent. These
numbers come from one prompt of 128 tokens. A different capture gives different
numbers.

## Generated data

The fixture directory is generated data. It must not be committed to the repository.
