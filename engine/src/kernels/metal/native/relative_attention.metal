// One F32 relative-attention decode step, in three dispatches of one command buffer.
//
// The contract is the CUDA adapter's relative-attention step: twelve ranges in the ABI's order,
// the first four written. The commit fills the current row's slot of both rings. The partials
// score the ring's rows a K/V head at a time, the keys and values of a tile staged in threadgroup
// memory for every query head of the group, and keep each split's peak, sum and weighted value.
// The finish folds the partials against a peak that includes the sink and applies the gate. A
// position below zero or at 2^30 marks the status bit and nothing else runs.

#include <metal_stdlib>
using namespace metal;

constant constexpr int kHeadDim = 128;
constant constexpr int kLanes = 32;
constant constexpr int kColumns = kHeadDim / kLanes;
constant constexpr int kTile = 16;
// The query heads one threadgroup serves at most; a wider group takes more threadgroups.
constant constexpr int kGroups = 8;
constant constexpr int kWordMax = (1 << 30) - 1;
constant constexpr int kStatusInvalidStepState = 4;
// The floats of one partial: the peak, the sum, then the weighted value of each column.
constant constexpr int kPartial = 2 + kHeadDim;
// 1 / sqrt(128): the scale of every score, as the CUDA step's shader applies it.
constant constexpr float kScale = 0.08838834764831844f;

// The step's parameters, bound after the twelve ranges and the context's partials.
struct Params {
    uint query_heads;
    uint kv_heads;
    uint relative_dims;
    uint relative_extent;
    uint capacity;
    uint max_splits;
};

// The rows of the ring a step at `position` reads: the newest `capacity` positions at most.
inline int ring_rows(int position, uint capacity) {
    return min(position + 1, int(capacity));
}

// The rows of one split: whole tiles, the count spread over at most `max_splits` splits. Every
// dispatch derives the plan the same way from the device position, so no host word orders them.
inline int split_chunk(int count, uint max_splits) {
    const int spreads = (count + int(max_splits) - 1) / int(max_splits);
    return (spreads + kTile - 1) / kTile * kTile;
}

// The current row goes into its slot of both rings: one threadgroup a K/V head, 256 threads.
kernel void relative_attention_commit(
    constant Params &p [[buffer(13)]],
    device float *keys [[buffer(1)]],
    device float *values [[buffer(2)]],
    device atomic_int *status [[buffer(3)]],
    const device float *k [[buffer(5)]],
    const device float *v [[buffer(6)]],
    const device int *pos [[buffer(11)]],
    uint gid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]])
{
    const int position = *pos;
    if (position < 0 || position >= kWordMax) {
        if (gid == 0 && tid == 0) {
            atomic_fetch_or_explicit(status, kStatusInvalidStepState, memory_order_relaxed);
        }
        return;
    }
    const size_t slot = size_t(uint(position) % p.capacity);
    const bool value_plane = tid >= kHeadDim;
    const uint d = value_plane ? tid - kHeadDim : tid;
    const size_t cell = (slot * p.kv_heads + gid) * kHeadDim + d;
    const size_t src = gid * kHeadDim + d;
    if (value_plane) {
        values[cell] = v[src];
    } else {
        keys[cell] = k[src];
    }
}

// One threadgroup serves one split of one K/V head for up to `kGroups` of its query heads, one
// SIMD group a head; a wider group takes the grid's third axis. Every thread stages the split's
// tiles; a head scores them from threadgroup memory, so the rows cross the memory once a group.
kernel void relative_attention_partials(
    constant Params &p [[buffer(13)]],
    const device float *keys [[buffer(1)]],
    const device float *values [[buffer(2)]],
    const device float *q [[buffer(4)]],
    const device float *relative [[buffer(7)]],
    const device float *table [[buffer(8)]],
    const device int *pos [[buffer(11)]],
    device float *part [[buffer(12)]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]])
{
    const int position = *pos;
    if (position < 0 || position >= kWordMax) return;
    const int count = ring_rows(position, p.capacity);
    const int chunk = split_chunk(count, p.max_splits);
    if (int(gid.x) >= (count + chunk - 1) / chunk) return;

    const uint group = p.query_heads / p.kv_heads;
    // The threadgroup's size, as the dispatch sets it: a SIMD group a head, at most kGroups.
    const int threads = kLanes * min(int(group), kGroups);
    const int kv_head = int(gid.y);
    const int head = kv_head * int(group) + int(gid.z) * kGroups + int(simd);
    const bool live = int(gid.z) * kGroups + int(simd) < int(group);
    const int first = int(gid.x) * chunk;
    const int last = min(count, first + chunk);
    const int slot = position % int(p.capacity);

    threadgroup float key_tile[kTile][kHeadDim];
    threadgroup float value_tile[kTile][kHeadDim];

    float query[kColumns] = {0.f, 0.f, 0.f, 0.f};
    float acc[kColumns] = {0.f, 0.f, 0.f, 0.f};
    float rel = 0.f;
    float peak = -INFINITY;
    float sum = 0.f;
    if (live) {
        for (int i = 0; i < kColumns; ++i) {
            query[i] = q[size_t(head) * kHeadDim + lane + i * kLanes];
        }
        if (lane < p.relative_dims) {
            rel = relative[size_t(head) * p.relative_dims + lane];
        }
    }

    for (int base = first; base < last; base += kTile) {
        const int rows = min(kTile, last - base);
        // The last tile's reads are over before this tile's rows take their place.
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int i = int(tid); i < rows * kHeadDim; i += threads) {
            const int distance = base + (i >> 7);
            // A modulo, then one addition: a distance of several capacities is the ring's.
            int cell = (slot - distance) % int(p.capacity);
            if (cell < 0) cell += int(p.capacity);
            const size_t src = (size_t(cell) * p.kv_heads + uint(kv_head)) * kHeadDim + uint(i & (kHeadDim - 1));
            key_tile[i >> 7][i & (kHeadDim - 1)] = keys[src];
            value_tile[i >> 7][i & (kHeadDim - 1)] = values[src];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (!live) continue;

        // The tile's scores, the same in every lane of the head's group.
        float scores[kTile];
        float tile_peak = peak;
 #pragma unroll
        for (int j = 0; j < kTile; ++j) {
            if (j >= rows) break;
            float dot = 0.f;
            for (int i = 0; i < kColumns; ++i) {
                dot = fma(query[i], key_tile[j][lane + i * kLanes], dot);
            }
            float bias = 0.f;
            const int distance = base + j;
            if (distance < int(p.relative_extent)) {
                const float term = lane < p.relative_dims
                    ? rel * table[size_t(lane) * p.relative_extent + uint(distance)]
                    : 0.f;
                bias = simd_sum(term);
            }
            scores[j] = kScale * simd_sum(dot) + bias;
            tile_peak = fmax(tile_peak, scores[j]);
        }
        // One rescaling a tile: the partial moves from its running peak to the tile's.
        const float correction = exp(peak - tile_peak);
        sum *= correction;
        for (int i = 0; i < kColumns; ++i) acc[i] *= correction;
 #pragma unroll
        for (int j = 0; j < kTile; ++j) {
            if (j >= rows) break;
            const float mass = exp(scores[j] - tile_peak);
            sum += mass;
            for (int i = 0; i < kColumns; ++i) {
                acc[i] = fma(mass, value_tile[j][lane + i * kLanes], acc[i]);
            }
        }
        peak = tile_peak;
    }
    if (!live) return;

    const size_t cell = (size_t(head) * p.max_splits + gid.x) * kPartial;
    if (lane == 0) {
        part[cell] = peak;
        part[cell + 1] = sum;
    }
    for (int i = 0; i < kColumns; ++i) {
        part[cell + 2 + lane + i * kLanes] = acc[i];
    }
}

// The head's result from its splits: each partial weighed by the exponential of its peak
// against a top that includes the sink, whose mass joins the denominator. One thread a column.
kernel void relative_attention_finish(
    constant Params &p [[buffer(13)]],
    device float *out [[buffer(0)]],
    device atomic_int *status [[buffer(3)]],
    const device float *sinks [[buffer(9)]],
    const device float *gates [[buffer(10)]],
    const device int *pos [[buffer(11)]],
    device float *part [[buffer(12)]],
    uint gid [[threadgroup_position_in_grid]],
    uint d [[thread_index_in_threadgroup]])
{
    const int position = *pos;
    if (position < 0 || position >= kWordMax) return;
    const int count = ring_rows(position, p.capacity);
    const int chunk = split_chunk(count, p.max_splits);
    const int active = (count + chunk - 1) / chunk;
    const size_t head = gid;
    const float sink = sinks[head];
    float top = sink;
    for (int s = 0; s < active; ++s) {
        top = fmax(top, part[(head * p.max_splits + uint(s)) * kPartial]);
    }
    float sum = exp(sink - top);
    float value = 0.f;
    for (int s = 0; s < active; ++s) {
        const size_t cell = (head * p.max_splits + uint(s)) * kPartial;
        const float mass = exp(part[cell] - top);
        sum = fma(mass, part[cell + 1], sum);
        value = fma(mass, part[cell + 2 + d], value);
    }
    const float gate_logit = gates[head * kHeadDim + d];
    const float gate = 1.f / (1.f + exp(-gate_logit));
    const float result = (value / sum) * gate;
    // A resident gate can bypass host readback; the sigmoid alone masks infinite logits.
    if (!isfinite(result) || !isfinite(gate_logit) || (d == 0 && !isfinite(sink))) {
        atomic_fetch_or_explicit(status, kStatusInvalidStepState, memory_order_relaxed);
    }
    out[head * kHeadDim + d] = result;
}
