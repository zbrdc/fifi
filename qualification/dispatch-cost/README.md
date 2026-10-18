# Dispatch cost measurement

This directory holds a measurement command for per-call dispatch cost of `MlxKernels`.
It is a measurement command, not a model-serving seam. Its host parser has a portable Cargo self-test.
It makes no claim about real-model throughput, Flash-Next capacity, or numerical parity.
The method and result record is the review (dispatch-cost.md, in the ledger).

## Contents

| File | Role |
| --- | --- |
| `src/` | The Rust command. It depends on the `inference-engine` crate with `native-mlx` and `native-fault-qualification`, and uses only public `MlxKernels` calls. |
| `src/host.rs` | The portable `vm.loadavg` parser. It keeps each position and rejects a non-finite average. |
| `mlx_reference.py` | The Python MLX reference of the same graph in two modes. |
| `summarize.py` | An optional script that joins both reports and prints the ratios. |
| `test_summarize.py` | The self-test for `summarize.py`. Run `python3 -m unittest test_summarize` from this directory. |

## Graph and shapes

One decode token runs through one layer or through N layers.
Defaults: hidden `H = 1024`, intermediate `I = 2048`, sequence `S = 256`, affine int4 with group size 64.
Weights and activations come from a fixed xorshift32 generator. The Rust and Python generators match.
Each layer has its own seed.

| Step | Call | Shapes |
| --- | --- | --- |
| input | `import` | x `[1,H]` F32, once per chain |
| rotary inputs | `import` | q `[8,1,64]`, k `[2,1,64]`, per layer |
| 1 | `rms_norm` | x `[1,H]`, weight `[H]` |
| 2 | `qmm` | n `[1,H]` with packed `[S,H]`, result `[1,S]` |
| 3 | `softmax` | `[1,S]` |
| 4 | `mm` | p `[1,S]` with V `[S,H]`, result `[1,H]` |
| 5 | `add` | x and a, `[1,H]` |
| 6 | `rms_norm` | `[1,H]` |
| 7 | `qmm` x2 | gate and up, packed `[I,H]`, result `[1,I]` |
| 8 | `silu`, `mul` | `[1,I]` |
| 9 | `qmm` | packed `[H,I]`, result `[1,H]` |
| 10 | `add` | `[1,H]` |
| 11 | `rope` | q and k, position 100, theta 10000, part 64 |
| end | `readback` | x `[1,H]`, once per chain |

A chain has `2 + 15 * N` calls.
The public interface has no reshape, so `rope` runs on separately imported q and k.
Its outputs are not part of the residual path.
One-time weight imports are timed as `setup_us` and stay outside every chain.
Rust `silu` is two MLX operations. Before slice L2 they completed eagerly; after L2 the call only builds them. The Python graph uses `x * mx.sigmoid(x)` under `MLX_DISABLE_COMPILE=1`, as decision 0028 (docs/decisions/0028-silu-reference.md, in the ledger) records.
Python `rope` is two operations, one for q and one for k, so its per-op counts differ from Rust.

## Commands

Run from the repository root. Use a new output directory on every run.

Run the portable host parser self-tests:

```sh
cargo test --manifest-path qualification/dispatch-cost/Cargo.toml --locked --offline host::tests
```

```sh
cargo run --release --locked --offline \
  --manifest-path qualification/dispatch-cost/Cargo.toml -- \
  --layers 1,8 --reps 100 --warmup 10 --output OUT/rust.json
```

```sh
MLX_DISABLE_COMPILE=1 \
/Users/dan/fifi/scrch/85dc6d0/p0a-oq04-mac/parity-env/venv/bin/python -B \
  qualification/dispatch-cost/mlx_reference.py \
  --mode both --layers 1,8 --reps 100 --warmup 10 --output OUT/python.json
```

```sh
python3 qualification/dispatch-cost/summarize.py OUT/rust.json OUT/python.json --output OUT/summary.json
```

Other flags: `--hidden`, `--inter`, `--seq`. Both commands must receive equal values.
The Rust command also accepts `--plan`. It prints the graph and call counts and runs on any host.
On a host without Apple Silicon the measurement exits with status 3.

## Output

The Rust report has schema `inference-engine.dispatch-cost.v1`.
The Python report has schema `inference-engine.dispatch-cost-reference.v1`.
The summary has schema `inference-engine.dispatch-cost-summary.v1`.
Each result gives the chain wall time and each operation's per-call time in microseconds.
Each time has a median and a nearest-rank 95th percentile over the repetitions.
Before slice L2, every Rust call completed native evaluation before it returned.
After L2, the elementwise, `mm`, and `qmm` calls only build the graph, so their per-call times measure graph construction. `rms_norm`, `rope`, and attention still complete inside the call until slice L3.
The Python `sync` mode matches the Rust behavior before L2 with `mx.eval` after each operation.
The Python `lazy` mode shows the cost without per-call synchronization.
The summary reports the lazy over sync ratio and the Rust over Python ratios.
Each result carries the SHA-256 of the final output and a finite check.
The hashes check that repetitions agree. They do not establish parity between Rust and Python.

## Release measurement instrumentation

Slice L5 measures resident release in separate time series and counts the synchronization calls inside each interval.
`release` times the 14 resident drops per layer. `release_unread` times the 13 lazy operation outputs in the same interval.
`release_evaluated_input` times the replaced x resident in the first layer, where its producer completed. `release_unread_input` times that drop in later layers, where the resident stays unevaluated.
`rope` releases the two q and k inputs it replaces. Its two synchronization calls are counted. RoPE release duration remains inside the complete `rope` call interval, so no release-only RoPE timing series exists.
`release_counts` reports the handle count and the synchronization count for one chain. Every repetition in a series must report equal counts.
The report field `synchronizations_per_chain` is now `l0_baseline_synchronizations_per_chain`. It holds the pre-L2 L0 estimate. It does not count synchronizations in the current lazy chain.

The counts come from `MlxKernels::qualification_sync_count`. That getter exists only under `native-fault-qualification`, which selects `native-mlx`. The production feature set does not expose it.
The counter counts process-wide attempts that reach its increment. It does not count successful completions or individual owners. The existing null-stream fault branch returns before that increment.
Each interval reads the counter before and after its own work, so each interval reports its own counter delta. The counter is process-wide, so caller isolation is a prerequisite, not a consequence of that subtraction. Measurement requires an isolated caller process that runs no competing native call during a measurement window. Reading the counter does not establish GPU completion.
The measured release intervals include counter reads, nested timer reads, and assertion bookkeeping. The outer 14-drop interval also contains the nested interval timers. These costs remain in the reported release durations. They are not calibrated out, and they are not uninstrumented production timings.
The report field `engine_crate` names the declared dependency features: `inference-engine native-mlx native-fault-qualification MlxKernels`. The label records declared manifest features. It does not record hardware availability or qualification success. Each qualification evidence packet must still record the exact resolved Cargo features and the compiler commands for that run.
These measurements instrument the release rule in decision 0032 (docs/decisions/0032-lazy-evaluation-dispatch-guard.md, in the ledger). They do not qualify production timing.
The Mac gate in `run_l0.sh` keeps its unchanged samples, repetitions, warmups, shapes, 90-second settle, load and GPU limits, and 1.25 ratio limit. Neither script changed.

## Conditions

Run on an idle host. `l0_gate.py` refuses PASS when the 1 minute load average is above 2.0 or the GPU utilization is above 5% or unknown. Record other processes, load average, and thermal state at start and end.
`host.rs` parses the three `vm.loadavg` fields by position. A malformed, absent, `nan`, or `inf` field is unknown in its own position only, and it never shifts a later average.
`l0_gate.py` derives `load_unknown` from its required `load_1m` sample alone. An unknown 5 minute or 15 minute average does not by itself prevent PASS. An unknown 1 minute average prevents PASS.
`l0_gate.py` and `run_l0.sh` exit with 0 for PASS, 1 for OPEN, 2 for invalid input, and 3 for NONSTANDARD.
`run_l0.sh` exits with 2 for a usage error or a non-empty output directory. It exits with 4 for a setup or bookkeeping failure outside the gate, such as a build, harness, host snapshot, or manifest value failure.
After the setup checks, only the gate gives the codes 0 to 3. It exits with 5 when the gate passes but a hash fails.
`sha256.txt` holds the paths with the output directory prefix. Run `shasum -c` from the directory that ran the script.
A hash failure after OPEN or NONSTANDARD keeps 1 or 3 and prints `hash failed`. A gate process that cannot start passes its own status, for example 127.
`run_l0.sh` waits `SETTLE_SECONDS` (default 90) after the build so the load average settles.
Another user service on the Mac is contention. Record it, and do not stop it.
Keep build output and reports in task scratch, and record the SHA-256 of every report.
