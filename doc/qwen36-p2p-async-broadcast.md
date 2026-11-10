# Experimental P2P broadcast event barriers

The Qwen3.6 multi-device path uses a double-buffered peer mailbox. Consecutive
broadcasts need a barrier so a publisher cannot reuse a slot before every
consumer has copied its contents. `Q36_P2P_SYNC_BROADCAST=1` provides that
barrier by synchronizing each GPU stream on the host.

Set `Q36_P2P_ASYNC_BROADCAST=1` before starting the engine to use CUDA events
for this barrier instead. Each consumer records an event after its copy, and
the publisher queues a wait for every peer before submitting later work. This
option takes precedence over `Q36_P2P_SYNC_BROADCAST` and is off by default.
It is intended for the non-captured multi-device path; cross-device event
waits are refused during graph capture.
[NVIDIA's multi-GPU documentation](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/multi-gpu-systems.html)
describes cross-device stream waits.

This port also applies the event barrier to the consecutive router broadcasts
in chunked prefill. The default remains the existing host barrier. The peer
tests exercise repeated messages, both host-sync settings, and compare the
same chunk partition with and without event barriers. These checks and a
served-model benchmark must pass on the exact build before this port is
qualified or its performance can be characterized.

To roll back, remove `Q36_P2P_ASYNC_BROADCAST`, retain
`Q36_P2P_SYNC_BROADCAST=1`, and restart the engine. Measure on the target
system before relying on the experimental path.

The [prefill driver](../scripts/bench_p2p_prefill.py) supports `--self-test`;
[bench_engines.py](../scripts/bench_engines.py) also reports server decode
telemetry. Compare both modes using the same model, context and placement,
and verify cache reuse before interpreting a prefill rate.

An [earlier experiment](../scripts/benchmarks/qwen36-p2p-async-broadcast-results.json)
recorded three short-prompt samples per arm on the 3060 + 3080 pair. It used
an earlier engine tree: its results do not qualify or measure this port, and
the configured 128k context was not filled. Rerun the device parity and
served A/B checks on the exact new build before drawing performance conclusions.

A separate [host synchronization trace](../scripts/benchmarks/qwen36-p2p-host-sync-profile.json)
motivates testing the chunked path: on an earlier build and a short synthetic
request, one host thread spent substantial wall time inside CUDA stream
synchronization calls. The trace did not sample CPU utilization or collect
call stacks, so it neither proves CPU saturation nor identifies which waits
are removable. Profiling perturbs the workload.
