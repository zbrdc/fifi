# P2P prefill synchronization trace

This note analyzes a previously recorded CUDA software trace of one cold
4,274-token Qwen3.6 35B A3B Q4_K_M prompt on an RTX 3060 + RTX 3080.
The traced server was commit `76dec8a61d040efa97beefe273bed8dc9e68ff57`,
using q8 KV, 256-row P2P chunks and disabled prefix reuse. It was not the
current serving process or the later 1,024-row experiment. The
[earlier profile](https://github.com/BColsey/fifi/blob/8b2034aea9bd409e16c3c562fcffa1cc20dccce0/scripts/benchmarks/qwen36-p2p-staging-profile.json)
records the capture, model and binary identities.

The trace contains 9,892 `cudaStreamSynchronize` calls on one host thread.
Their union is 12.650 seconds. During 12.642 seconds of that interval, at
least one card had a recorded CUDA kernel or memcpy event. The union with a
Q4 matrix-multiplication kernel on either card was 10.450 seconds. During
4.159 seconds, the 3060 was in a peer sum-wait kernel while the 3080 was
running Q4 work; the reverse overlap was 2.555 seconds. The
[overlap record](../../scripts/benchmarks/qwen36-p2p-sync-overlap.json)
contains the exact values, source hash and limits.

These intervals overlap and must not be added. Host synchronization elapsed
time is not CPU compute time, and a running GPU kernel can itself be waiting.
The trace does not record CPU call stacks, memory stalls or expert-weight
placement. GPU-accessible host-mapped expert reads need not appear as explicit
CUDA memcpy events, so the trace cannot establish whether host memory or PCIe
caused a Q4 delay. CUDA tracing also changes timing. These observations identify
hypotheses; they do not establish a bottleneck or predict a speedup.

With the original hash-matched Nsight SQLite export available, reproduce the
record from the repository root:

```sh
python3 scripts/benchmarks/qwen36-p2p-sync-overlap.py \
  --trace /path/to/trace.sqlite \
  --output scripts/benchmarks/qwen36-p2p-sync-overlap.json --check
```

A next experiment should identify resident versus host-mapped expert inputs
in the current server's trace, measure both cards and host activity, and compare
placements using the same binary, workload and memory-admission limits in both
orders. Host CPU profiling support must be checked before reserving GPU time;
this host currently denies a basic unprivileged `perf` counter probe.
