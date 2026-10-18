# Zero-copy import evidence kit

Draft for goal task-2 item 2. Goal contract: "Zero-copy import check records a measured result."
The kit runs the existing `qualification/native-import` harness on the Mac Studio and produces the evidence the record template needs.

## Files

| File | Purpose |
| --- | --- |
| `run-import-evidence.sh` | POSIX shell Mac runner. Builds and runs the harness, captures `/usr/bin/time -l`, writes evidence hashes. |
| `import-evidence-record.md` | Record template the coordinating agent fills after the run. All numeric fields are `MEASURED_VALUE` placeholders. |

## What the kit measures

One preflight case of 67108864 bytes and three main cases of 1073741824 bytes through `mlx_array_new_data`.
Each case reports stage-sampled MLX active, cache, and peak bytes, process RSS, import and GPU timings, the copy check, and post-release retention.
The expected measured result is a copy: the pinned mlx-c header declares the call a copying import (`p0a-native-import-baseline.md`, in the ledger, "Source and ownership").
The record's section 8 defines exactly which measurements would justify or falsify a zero-copy claim.

## Preconditions

1. The Mac worktree is at the base commit with a clean tracked tree. The runner exits on a dirty tree.
2. The pinned activation exists at `/Users/dan/fifi/scrch/9071b86/p0a04-native-gate/environment/activate.sh`.
3. The pinned MLX checkout is at commit `64ea011cb65f14d9ce2737e60db9a4ae91ed7441` with a clean tree. The runner and `qualification/native-import/build.rs:10-22` both check this.
4. No other heavy GPU workload is running. The record states any contention that occurs.

## Run steps on the Mac

```sh
sh run-import-evidence.sh /Users/dan/fifi/wrktr/52640f0/p0a05-reserve-baseline \
    /Users/dan/fifi/scrch/52640f0/p0a05-reserve-baseline/import-evidence
```

The runner fails loudly. Each step redirects its own log under `set -e`, following the fixed `qualification/humaneval/run-baseline.sh` pattern from commit `1ab6a58`.
No step pipes through `tee`, so a failed case stops the run instead of masking its exit status.

The runner does not gate the 1 GiB cases on a parsed free-pressure number.
It records `sysctl vm.swapusage`, `vm_stat`, and `memory_pressure` before and after each case.
The coordinating agent checks the 50 percent free-pressure rule in `mem-before-1073741824-*.log` before accepting the main cases.

## Expected runtime

Not measured for this kit. The coordinator records the wall time of the run.
Prior runs of the same harness completed each 1 GiB case well inside a 240-second bound. Import took between 71622 and 130345 microseconds, and full host copy took between 111560 and 485745 microseconds under trainer contention (`p0a-native-import-baseline.md`).
Each case sleeps two seconds at the end. Release build time is in `build.log` on the first line block of `/usr/bin/time -l` output.

## Evidence files

All files land in the OUTDIR given on the command line.

| File | Content |
| --- | --- |
| `env.log` | Worktree, source commit, MLX commit, runner, UTC start time. |
| `source-commit.txt` | Measured source commit. |
| `build.log` | Release build output with `/usr/bin/time -l` statistics. |
| `binary.sha256` | Release binary hash. |
| `import-67108864-preflight.log` | Preflight case JSONL plus `/usr/bin/time -l`. |
| `import-1073741824-main-1.log` through `-main-3.log` | Main case JSONL plus `/usr/bin/time -l`. |
| `mem-before-*.log`, `mem-after-*.log` | Per-case host memory snapshots. |
| `mem-host-before.log`, `mem-host-after.log` | Whole-run host memory snapshots. |
| `import-evidence.sha256` | Manifest over every evidence file except itself. |

After the run, the coordinating agent fills `import-evidence-record.md`, checks the manifest, and copies the completed kit to persistent Linux scratch under `/media/dan/nvme/fifi/scrch/52640f0/`.

## Not covered

Direct file-backed mapping, MLX thread safety, the adapter harness mode, and Phase 0B feasibility.
See record sections 8 and 11 for the exact claim boundaries.
