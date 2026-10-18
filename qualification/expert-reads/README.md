# Cold expert reads (S3 packet T0b)

This command measures the cold read rate of single experts of the Flash-Next checkpoint.
It answers Phase 0B deliverable 4 for experts and gate T0-P4 of `s3-resident-expert-cache.md` (in the ledger, section 3.6).

## Method

- The command needs no privileges. It never purges a cache and never runs `purge`.
- It reads the index and the shard headers, and finds the nine stacked tensors of each MoE layer (`gate_proj`, `up_proj`, and `down_proj`, each with `weight`, `scales`, and `biases`). One expert is one slice of each. The sizes come from the headers (decision 0038).
- It maps each shard read-only and never reads through the mapping. `mincore` on the mapping reports which pages are in memory, including file-cache pages that the process never touched.
- It draws experts in a seeded order over all layers and experts. It reads an expert only if `mincore` reports at least `--min-cold` (default 0.9) of its bytes not resident, checked right before the read. An expert is never read cold twice.
- A case reads `--experts` experts (default 64) in groups of `depth` threads, one expert per thread. Each thread reads its expert as nine positioned reads (`pread`). The group time runs from a common barrier to the end of the last thread. The queue depth is the number of threads, each with one read in flight.
- On macOS, the cases with `F_NOCACHE` run first, for each depth (default 1, 2, 4, 8, 16). They all read one fixed set of experts, so that the depths compare on the same byte ranges. A `F_NOCACHE` read leaves the file cache unchanged, so `mincore` before each case shows whether the set is still cold. Other systems record `nocache=unsupported`.
- Then, for each depth: cold buffered reads, then a warm re-read of the same experts. A buffered read fills the file cache and no cache purge is allowed, so each buffered depth reads a set of its own, drawn in the seeded order. The sets are disjoint and of equal size, and they are not the fixed set.
- After each buffered read, `mincore` must report the expert resident. Otherwise `mincore` does not observe the file cache, and the result is invalid (`mincore_sees_file_cache=false`).

The unsafe code is in `src/sys.rs` only: the read-only mapping, `mincore`, `fcntl(F_NOCACHE)`, and `sysconf`.

## Commands

Run it before the T0 trace run on the same day, so that the trace run does not warm the chosen experts.
Stop other workloads that read the checkpoint first.

```sh
export CARGO_TARGET_DIR=<task scratch>/target
cargo test --offline --manifest-path qualification/expert-reads/Cargo.toml
cargo build --offline --release --manifest-path qualification/expert-reads/Cargo.toml
"$CARGO_TARGET_DIR/release/expert-reads" --artifact <checkpoint dir> --out <dir>
```

Options: `--experts N`, `--depths 1,2,4,8,16`, `--seed N` (default 20261009), `--min-cold F`, `--prefix language_model.model.`.
The default run reads 64 experts in each of 15 cases (5 depths for `F_NOCACHE`, cold buffered, and warm buffered), about 2.65 GB, of which about 1.77 GB come from the SSD.

## Outputs

| File | Content |
| --- | --- |
| `reads.tsv` | One row per expert read: case, depth, group, layer, expert, bytes, nanoseconds, cold share before, resident share after. |
| `summary.tsv` | One row per case: experts, bytes, summed group wall time, bytes per second, p50 and p95 latency per expert, byte-weighted cold share before, resident share after, skipped candidates, GB per second. |
| `meta.txt` | Settings, page size, expert bytes, `F_NOCACHE` support, swap and `vm_stat` before and after, and the verdict. |

The verdict lines give, for `buffered` and `nocache`: the rate at depth 1, the best depth and its rate, the rate of each depth in GB/s (`<mode>_gb_per_second_depth<N>`, 10^9 bytes per second), and the smallest cold share of a case. The line `fixed_set_layer_expert` lists the fixed set as `layer:expert`, or `none`. `summary.tsv` ends each row with `gb_per_second`.
`t0_p4=pass` needs a depth-1 case, a cold share of at least `--min-cold` in every cold case, and `mincore_sees_file_cache=true`.
The simulator takes the depth-1 and best rates as `--bw t0b-qd1=<rate>` and `--bw t0b-best=<rate>`.
