# mlx-lm gated delta kernel

This directory holds the Metal kernel of the gated delta rule from mlx-lm 0.31.0.
It is the only custom Metal kernel the engine takes from mlx-lm.

| Path | Content |
| --- | --- |
| `src/gated_delta.py` | `mlx_lm/models/gated_delta.py` of mlx-lm 0.31.0, byte-identical. The engine does not run it. |
| `LICENSE` | The MIT license of mlx-lm, byte-identical. |
| `gated_delta_step.metal` | The Metal source of the scalar-gate, unmasked variant. The engine embeds it with `include_str!`. |
| `extract_kernel.py` | Derives `gated_delta_step.metal` from `src/gated_delta.py`. |
| `check_provenance.py` | Checks every hash and the derived file. |
| `provenance.json` | Upstream package, version, path, hashes, and the (empty) patch list. |

The Metal source is an f-string inside `_make_gated_delta_kernel` in the Python file.
`extract_kernel.py` parses the Python file with `ast`, keeps the statements up to the `source` assignment, and lets Python evaluate the f-string.
It does not reimplement the f-string rules, so the `.metal` file is byte-traceable to the Python file.

```sh
python3 -I engine/vendor/mlx-lm-gated-delta/extract_kernel.py > /dev/null   # scalar, unmasked variant
python3 -I engine/vendor/mlx-lm-gated-delta/check_provenance.py            # expect: ok hashes, ok extract
```

The kernel runs one sequential loop over the tokens in each thread.
It reads the state as `[B, Hv, Dv, Dk]` and groups value heads over key heads with `hk = hv / (Hv / Hk)`.
The engine passes F32 tensors only. `InT` is `float`, so the output and the new state are F32.
