# CUDA conventions the types do not enforce

Four rules govern sharing device memory across streams and lifetimes. Types enforce parts of these rules. The rest hold only by convention, and no type, assertion, or test rejects a change that breaks them. This page states each rule so a future change knows it breaks a contract rather than reshaping an accident. Every rule names its code site and what enforces it today.

Debug tools that help while working against these rules:
`FIFI_CUDA_POISON_ALLOC` and `FIFI_CUDA_LINEINFO` (`environment.md`), and
NVIDIA compute-sanitizer (memcheck, racecheck, synccheck) over a build made
with `FIFI_CUDA_LINEINFO`.

## Hidden synchronization sites

Every place the engine blocks on the device without a caller-visible
`synchronize` call, with why it exists and where the details live:

- `complete_pageable_copy` (`kernels/cuda/native/resident.rs`) blocks the
  host on `stream_synchronize`. It returns an owned pageable buffer (readback)
  or releases it (upload) only after its copy completes. It charges the wait
  to the page-io counters when armed.
- `download` (`kernels/cuda/context.rs`) calls the synchronous
  `cudaMemcpy`. It provides the stream ordering rule 4 describes. It also
  blocks the host until the copy completes, so every readback pays this
  hidden sync.
- `DeviceBuffer` upload/download (`kernels/cuda/context/buffer_io.rs`) — a
  pageable `cudaMemcpyAsync` costs a full staged copy the caller cannot see.
  The impl block's doc states the consumption-at-enqueue rule.
- The allocation cache's return path (`kernels/cuda/context/alloc_cache.rs`)
  synchronizes the device before a slot re-enters the pool. Lazy return defers
  that sync until reuse or drain, still doing a full device sync when it fires.
- The old projector's default-stream sync no longer exists. `engine/src/project/`
  contains no synchronize call. The current projector build path has no
  hidden sync.

## 1. Lazy return runs on one stream (alloc_cache.rs)

`AllocCache` may defer the device sync when a block goes idle. This happens if lazy return
arms (`set_lazy_alloc_return`, which no start arms now) and the lease has at most
one stream (`on_return`, `kernels/cuda/context/alloc_cache.rs`). Reissuing that
block is safe only because every other enqueued user of its bytes is on the
same stream, where ordering is program order.

What holds it today: `Lease::streams` counts `CudaContext::stream()` calls and
never decrements (`guard.rs`, `stream.rs`), and creating a second stream first
calls `cache.settle()`. Every host-synchronous write into a cached block
settles first (`host_write`, `AllocCache::copy`).

What would break it:
- A stream created outside `CudaContext::stream()`, which the counter misses (today `cudaStreamCreate*` appears once, in `stream.cu`).
- A consumer on the legacy NULL stream, which the counter ignores.
- A raw `native::upload_at` call that bypasses `host_write`.

Possible enforcement: assertions on the stream counter at return and reissue.
Alternatively, record an event at return and query it at reissue (`cudaEventQuery`), treating not-ready as a cache miss.

## 2. A capture pins the buffers its body uses (context/graph.rs)

A CUDA graph bakes in device addresses. Suppose a captured body uses a cached
block, and that block returns and reissues to another user. The graph's next
run then touches the new tenant's bytes. `CapturePlan::pins` lists the buffers a
capture claims to use. The start window does not compare it with the buffers
the body actually touches (`kernels/cuda/context/graph.rs`).

What holds it today: the capture window refuses allocation, free, sync and
graph run. It defers drops to `free_after_capture`. The capture author fills
the pin list by convention.

What would break it: a capture body that touches a cached buffer without
listing it in `pins`.

Possible enforcement: in debug builds, record the buffers `launch_on` sees
during capture and compare with `pins` at `capture_end`.

## 3. The refill source outlives the refill (family/exl3/bank.rs)

The expert-bank refill copies from the page-locked store into device memory
(`family/exl3/bank.rs`). The copy is asynchronous, so its source must stay
valid and unchanged while it runs. Most of this rule is now enforced in the
types:

- `pin()` seals the store, and `put()` on a sealed store refuses. The store's
  own finalize calls `pin()` on every part (bank.rs). No write path exists
  after load that could overwrite an in-flight copy's source.
- `Drop` calls `device_synchronize()` before `host_unregister`, so the
  unregister cannot race a refill DMA still reading the page-locked range,
  even on a drop path the streams have not drained.

What is still by convention: nothing material. Every call site keeps the store process-long-lived. A shorter-lived store stays
correct through `Drop`'s sync, at the cost of one device sync on its drop path.

## 4. The split refill's readback ordering rests on quiescence (family/exl3/bank.rs)

The split refill copies misses on a second stream, involving two orderings:

- Slot reads after the copy are event-ordered: the copy records an event and
  the calls that read the slot wait on it. The code enforces this part.
- The reverse direction — the refill's writes must not pass the readback whose
  picks produced them — rests on "both streams were quiescent at the
  readback". This is a host-side argument, not a device dependency.

What holds it today: the readback path waits on the start stream before it
plans the refill.

What would break it: removing that sync, or enqueueing the refill before the
readback completes.

Possible enforcement: an event on the readback stream that the copy stream
waits on, removing the quiescence argument. The path is opt-in
(`Q36_CUDA_BANK_OVERLAP=1`), so no served default depends on it.
