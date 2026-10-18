# SiLU reference probes

These scripts support decision 0028 (docs/decisions/0028-silu-reference.md, in the ledger).
They compare the compiled and the eager (compile off) SiLU arithmetic of the pinned Python MLX.
They do not run the Rust engine. They are not engine tests.
No probe has run yet. A passing run qualifies nothing until the Mac qualification owner records it.

## Files

- `silu_sweep.py` freezes six SiLU variants on one fixed sweep of 2,101,264 values, in F32 and bf16.
- `silu_compare.py` compares two sweep files and prints a JSON report.
- `silu_compare_selftest.py` tests the comparison code with NumPy only. It does not import MLX.
- `silu_graph.py` prints the graph node names of the compiled and eager functions.

## Interpreter

Use the `parity-env` interpreter for every script, including the self-test:

```sh
PY=/Users/dan/fifi/scrch/85dc6d0/p0a-oq04-mac/parity-env/venv/bin/python
```

It holds `mlx 0.32.2.dev20260927+1f8e74e3`. `silu_sweep.py` and `silu_graph.py` stop unless both checks pass:

- The MLX version string equals `0.32.2.dev20260927+1f8e74e3`.
- The SHA-256 of `core.cpython-*.so` equals `935cc4e023f4a3b548981bac866e7edb708256c42b04508f5f633f1c2f7839b3`.

The sweep script also checks the Edge0 copies of `activations.py` and `qwen3_next.py` against pinned hashes.
It runs lines 58 to 62 of the pinned `qwen3_next.py`, not a copy.
MLX tests only whether `MLX_DISABLE_COMPILE` exists. So `MLX_DISABLE_COMPILE=0` also turns compile off.

## Commands

Run the commands from the `qualification/silu-reference/` directory on the Mac.
Set `OUT` to a new directory in the Mac task scratch: `/Users/dan/fifi/scrch/<base-commit>/<task>/silu-reference`.
The sweep script refuses to overwrite an output file.

```sh
PY=/Users/dan/fifi/scrch/85dc6d0/p0a-oq04-mac/parity-env/venv/bin/python
OUT=/Users/dan/fifi/scrch/<base-commit>/<task>/silu-reference
mkdir -p $OUT

# Comparison code test (NumPy only, run with the same interpreter).
$PY silu_compare_selftest.py

# E1 to E3 and E5: freeze both modes, then compare.
MLX_DISABLE_COMPILE=1 $PY silu_sweep.py --output $OUT/eager.npz
$PY silu_sweep.py --output $OUT/compiled.npz
$PY silu_compare.py --eager $OUT/eager.npz --compiled $OUT/compiled.npz > $OUT/compare.json

# E4: graph node names.
MLX_DISABLE_COMPILE=1 $PY silu_graph.py --out $OUT/dot-eager
$PY silu_graph.py --out $OUT/dot-compiled
```

Expected observations (full table in the decision):

- E1: in `eager.npz`, each decomposition variant equals its function. `zero_difference` is true in F32 and bf16.
- E2: the `decomp` variant gives `zero_difference` true between `eager.npz` and `compiled.npz`.
- E3: in `compiled.npz`, `nn_silu` against `decomp` gives 1,151 differences and 8 ulp on the first 4,097 points.
- E4: compiled labels include `CompiledSigmoidMultiply` and `CompiledSigmoidMultiplyMultiply`. Eager labels show `Sigmoid` and `Multiply`.
- E5: report only. Read the counts, the ulp figures, and the sign and magnitude splits in `compare.json`.

## E7: activation dtype

E7 uses the stock Edge0 environment (MLX 0.30.6, mlx-lm 0.31.0) in its default compiled mode.
The coordinator picks the model and the memory limits.
The command wraps `swiglu` in `qwen3_next` and `switch_layers`, so it prints the dtype at each call:

```sh
export MODEL=<model directory chosen by the coordinator>
/Users/dan/edge0/.venv/bin/python - <<'PY'
import os
import mlx_lm
from mlx_lm.models import qwen3_next, switch_layers

for module in (qwen3_next, switch_layers):
    original = module.swiglu
    def spy(gate, x, original=original, name=module.__name__):
        print("swiglu", name, gate.dtype, x.dtype)
        return original(gate, x)
    module.swiglu = spy

model, tokenizer = mlx_lm.load(os.environ["MODEL"])
mlx_lm.generate(model, tokenizer, prompt="Hello", max_tokens=1)
PY
```

Expected observation: the printed dtypes are `mlx.core.bfloat16` at the routed expert and shared expert sites.

## E6: Rust comparison

E6 needs a Rust test that does not exist yet.
The test loads a new separate fixture with its own schema and digest.
The fixture comes from `eager.npz` and stays outside Git. Its hash pins it.
The S8 fixture stays unchanged.
The coordinator assigns the work, and the user agrees the test cases before anyone writes it.
