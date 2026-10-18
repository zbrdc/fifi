# Int8 reference fixture and gate

This directory holds the S5 gate of decision 0031 (docs/decisions/0031-int8-reference-fixture.md, in the ledger).
It checks the int8 router and shared-expert gate path against frozen MLX outputs.
Comparison is exact bit equality. There is no tolerance.

## Files

- `prepare.py` builds the fixture on the Mac from the pinned checkpoint.
- `reference-pin.json` pins the MLX build. `prepare.py` refuses any other build.
- `src/` holds the Rust command. `tests/command.rs` holds the portable tests.

## Prepare the fixture

Use the pinned parity environment. The output directory must not exist.
Set `TMPDIR` on the same file system as the output.

```sh
TMPDIR=/Users/dan/fifi/scrch/<base>/p1-s5-int8-fixture/tmp \
  /Users/dan/fifi/scrch/85dc6d0/p0a-oq04-mac/parity-env/venv/bin/python -B prepare.py \
  --checkpoint /Users/dan/edge0/models/edge0-35b --output <NEW_DIR>
```

The script prints `manifest_sha256`. Run it twice into two new directories. Both digests must match.

## Run the gate

```sh
MLX_DISABLE_COMPILE=1 cargo run --release --locked -- \
  --cases <DIR> --manifest-sha256 <DIGEST> --output report.json
```

Add `--negative-control` to change one scale in a memory copy. That run must detect the change.
Add `--integrity-only` on any host to check digests, inventory, case list, and the portable load path.

Exit codes: 0 passed (or negative control detected), 1 mismatch (or negative control missed), 2 invalid input or error.
A synthetic fixture never sets `qualified`.

## Cases

1. Case 1: `TieredWeights::packed` with `dequant_matmul` for all 90 cases.
2. Case 2: resident `qmm` for the M = 1 cases.
3. Case 3: portable integrity self-test. The tests in `tests/command.rs` run it on Linux.
4. Case 5: negative control.
