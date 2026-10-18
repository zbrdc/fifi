# MLX pin purity check

User directive of 2026-10-14: the engine follows the latest MLX release. A pin bump must not change the engine's own behavior.
This directory holds the check and the frozen baseline.

Rule. After a bump of the MLX pin (`engine/vendor/mlx-sys`), run the Mac gate steps `mxc4` (route parity, f32 and bf16, compiled off and on) and `mxc6` (the 24-prompt bank in the `steps` and `chunk256` configs, compiled off and on) on the new pin, then compare:

```sh
python3 -I qualification/mlx-purity/purity.py compare qualification/mlx-purity/baseline.json <evidence dir of the gate run>
```

The exit code is 0 when the uncompiled path is bit-identical and the compiled path passes its gates, 1 when the uncompiled path differs (stop: the bump changed something that is not MLX's compile code generation), 2 when only a compiled-path gate fails, and 3 for a missing or malformed input.
The uncompiled path (bank quality fields and the route-parity logits hashes) is the purity gate. The compiled path may move with MLX's own compile code generation. Its gates: the format token is top-1 on every bank prompt, the minimum P(format) is at least 0.99, and the route-parity cosine is at least its floor.

`baseline.json` is the frozen reference of the pin MLX v0.32.3 (mlx-c v0.7.0 with the one-line `gather_qmm` change recorded in `engine/vendor/README.md`). Re-freeze it after the next accepted bump:

```sh
python3 -I qualification/mlx-purity/purity.py freeze <evidence dir> qualification/mlx-purity/baseline.json --mlx-version <v> --mlx-commit <c> --engine-head <h>
```

Self-tests (standard library only): `python3 -I -m unittest discover -s qualification/mlx-purity -t qualification/mlx-purity`.

Numbers of the bump from MLX 0.32.2 to 0.32.3 (Mac gate runs on the same checkpoint and prompts): the uncompiled bank and route parity were bit-identical. The compiled path moved: the bank minimum P(format) rose from 0.9966 to 0.9989 (`steps`) and from 0.9898 to 0.9954 (`chunk256`), top-1 stayed 24 of 24, the largest P(format) change was 0.0056, and the bf16 route-parity minimum cosine rose from 0.99697 to 0.99817 (floor 0.995).
