# Running a second engine on an idle GPU

Run a second, independent `enginesrv` on a spare GPU next to an existing
deployment (for example a two-card P2P pair), and serve both models through one
LAN web GUI. The example host has an RTX 3080
and RTX 3060 P2P pair plus an RTX 2080 Ti 11 GB, serving Qwen3.6-35B-A3B
Q4_K_M GGUF. The observations below are limited to that host.

## Why a second instance, not a third ring member

An idle GPU of a *different* compute capability cannot join a multi-device
device set at all: `check_peer_facts` refuses mixed compute capabilities
(`engine/src/kernels/cuda/peer.rs`). Independent of that, the Qwen3.6 shard
geometry refuses 3 cards at Q4_K_M: the 16 attention query heads form exactly
two 8-per-KV-head decode groups, and the expert down projection's K width of
512 is exactly two Q4_K superblocks, so any 3-way cut splits a quant block
(`engine/src/family/qwen36/cuda/shard.rs`). A second process can use the spare
card independently, with a separate KV cache, sessions and failure domain.

## Build

Use the standard scratch worktree and a task-local `build-env.sh`. The one
new requirement is the arch list: a Turing card (sm_75) needs
`INFERENCE_CUDA_ARCHS=75`, because the build emits PTX for the newest listed
arch only and PTX does not JIT downward:

```bash
export INFERENCE_CUDA_ARCHS=75        # or "75,86" to cover both generations
export CUDA_VISIBLE_DEVICES=GPU-<uuid-of-the-spare-card>
cargo build --release --locked --features native-cuda --bin enginesrv
```

The kernels on the single-device serving path are kept sm_72-clean by
construction (the aarch64 default), so sm_75 compiles; the pre-Ampere arms
had never been board-run before 2026-11-08, when a full smoke (auth, models,
chat, streaming) passed on a 2080 Ti.

## Config

Single-device serving is first-class: omit `cuda_devices` (it defaults to
`[0]`, which is the one visible device). Tiered experts are what fit the
20 GB checkpoint on an 11 GB card: dense weights are resident, experts
stream from the model file's page cache. Reuse the primary deployment's
`model_dir` (read-only mmap, shared page cache — the second instance then
becomes ready in seconds).

```json
{
  "listen": "127.0.0.1:18793",
  "api_key_env": "ENGINESRV_API_KEY",
  "engines": [{
    "id": "q36-spare",
    "family": "qwen3_6_35b_a3b",
    "model": "fifi-q36-spare",
    "model_dir": "/path/to/shared/model",
    "tokenizer_dir": "/path/to/shared/model",
    "qwen36": {
      "context_limit": 131072,
      "kv_dtype": "q8",
      "capture": false,
      "tiered": { "slots": 85, "threads": 8, "preload": true }
    }
  }]
}
```

Give the engine a distinct `model` id (not the primary's `fifi-q36`) so a
merged endpoint can route by model name. The 2080 Ti was also observed to
load a 262144-position q8 configuration with 64 tiered slots per layer. These
settings are host-specific and do not prove that every prompt of that length
will complete.

Two engine-level facts to know before tuning further:

- **`kv_dtype: "q8"` uses less KV memory than `"f32"`**. Its exact footprint
  and performance depend on the layout, context and device.
- **`kv_offload_host_bytes` (host-pinned KV spill) is refused with tiered
  experts** — the two are mutually exclusive, and tiered is what fits the
  model, so the spill knob is unavailable in this shape.

## Prefill is per-token in tiered mode

The chunked prefill path requires non-tiered q8 with fixed admission and
context staging; tiered mode forces per-token feeding (the P2P pair is
per-token for its own reason — multi-device also refuses chunked prefill).
First-touch weight loading, host contention and prefix reuse can affect
timings. A continued chat does not automatically avoid reprocessing earlier
tokens: prefix reuse must be enabled and validated for the selected engine.

## One LAN GUI, both models

The web GUI already lists models from `/v1/models` and sends the selected
`model` on every request, so no GUI change is needed. Put a loopback-only
model router between the vite proxy and the engines (`scripts/web-model-router.mjs`):
it merges `GET /v1/models` across backends and routes every other request by
its `model` field, injecting the matching backend's API key. Point the proxy
at it with the existing env hook (`export FIFI_BACKEND=http://127.0.0.1:18795`
in `start-web.sh`) and add `Wants=`/`After=` on the router unit. Requests
without a model field (monitor panels, `/v1/system`) report the primary
engine by design. List the desired primary model first for new browser
sessions; an existing browser keeps its saved model selection. A request with
**Trace sampling alternatives** enabled also requires top-level
`"sample_trace": true` on the selected engine, followed by an engine restart.
With the model-aware router and GUI, a traced reply keeps its model association
even if the picker changes. Older clients that do not identify the model send
trace lookups to the primary; refresh the GUI after upgrading. Retained traces
include generated token pieces and are accessible through a keyless LAN GUI.

## GPU test coordination

A permanently resident process on the spare GPU breaks any machine-wide
"no compute processes" gate. Two policy fixes, both deployed here:

- **Scope the lease runner by UUID**: in `gpu-run-local`, check
  `nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-compute-apps` when
  `CUDA_VISIBLE_DEVICES` is in UUID form (indices enumerate differently for
  nvidia-smi vs CUDA children — do not scope on bare indices), falling back
  to the machine-wide check otherwise. This is never looser than the
  original guard.
- **Coordinate service handoffs under the shared lease**: acquire the
  existing shared `gpu.lock` before stopping or restarting either instance.
  Use the runner's `--reserve` mode with a validated handoff command that
  checks the inherited lease descriptor against the shared lock's device
  and inode. Keep the lease held while that command verifies service and
  device owners, stops the selected service, runs the test, and restores and
  confirms the original service. A busy lease means no service action. Do not
  release it before restoration is confirmed; if restoration fails, keep the holder
  alive and escalate. Never unlink or replace the lock file: that can create
  a different inode and allow simultaneous holders.
- **Check the other lanes' gates**: scope idle checks to the physical GPUs
  used without dropping any declared quiet conditions for other GPUs, CPU
  or PCIe traffic. Verify the live service unit, process and GPU ownership
  before treating another instance as expected. An executable path alone
  does not establish ownership or request idleness. Record changes and
  blocked work in the lane's coordination ledger.

The lease is advisory. A direct `systemctl` action outside the reserved
handoff can still interrupt a holder; all cooperating lanes must use the
same lease and handoff procedure.

Concurrent use can change the primary pair's performance. Measure any
interference with matched workloads and retained telemetry rather than
assuming a fixed penalty.

## Example server observations

The following are single observations on the 2080 Ti, not an A/B test. The
first rate divides prompt tokens by the server's time to first token for a
one-token request; it includes work beyond prefill. Decode is the server's
token-based rate for a separate 36-token streamed response. All six requests
reported zero reused tokens.

| Tiered slots / threads | Configured context | Prompt tokens | Prompt / first-token time | Server decode |
|---|---:|---:|---:|---:|
| 64 / 6 | 262,144 | 22,447 | 64.16 tokens/s | 44.94 tokens/s |
| 64 / 8 | 262,144 | 15,762 | 62.38 tokens/s | 48.28 tokens/s |
| 85 / 8 | 131,072 | 22,447 | 75.50 tokens/s | 50.17 tokens/s |

The [server observation record](../../scripts/benchmarks/second-gpu-server-observations.json)
contains the selected journal fields and startup lines. The journal does not
establish binary hashes, exclusive host resources or complete answer
correctness. The older benchmark driver counted nonempty stream chunks rather
than tokens; discard its decode figures. The corrected driver in
`scripts/bench_engines.py` uses matched server telemetry when available.
