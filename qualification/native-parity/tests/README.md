# Synthetic parity fixtures

The files under `fixtures/` contain synthetic data only.
They contain no private model weights.

`test_prepare.py` builds two affine-int4 modules and one excluded int8 module through the safetensors library.
The dense module has payload shape `[2, 16]`. The expert module has payload shape `[2, 2, 16]`.
All packed words and scales are zero. Each bias is BF16 negative one.
The first four selected identities are:

| Module | Coordinates |
| --- | --- |
| `dense` | `[1, 1]` |
| `experts` | `[0, 0, 1]` |
| `dense` | `[0, 0]` |
| `experts` | `[1, 0, 0]` |

The basis output contains 64 values of negative one.
The dense output is negative five because the specified activation coefficients sum to five.
The contract assessment (native-parity-contract.md, in the ledger) records independent Wolfram checks for these probe properties.
The preparation command captured the committed outputs on the Mac through the pinned Python MLX build.
The fixtures retain the corresponding package versions, binary-pin digest, and synthetic source hashes.
The frozen manifest SHA-256 is `e8646da64dae608e688049744ea05da8df804f76a8edab20f5400814e4dddd2d`.

Command tests modify copies in task scratch.
Modified fixture hashes authorize deliberate negative cases only. They cannot qualify the real checkpoint.
The numerical reference shares upstream MLX with the Rust backend.
The fixture therefore checks Rust loading, conversion, and invocation rather than independently proving MLX kernel correctness.
