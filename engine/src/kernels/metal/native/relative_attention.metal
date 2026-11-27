// One F32 relative-attention decode step, in three dispatches of one command buffer.
//
// The contract is the CUDA adapter's relative-attention step: twelve ranges in the ABI's order,
// the first four written. The commit fills the current row's slot of both rings. The partials
// score the ring's rows a strip at a time, one SIMD group a strip of a K/V head for up to three
// of its query heads, each row's keys and values loaded once, and keep each strip's peak, sum
// and weighted value.
// The finish folds the partials against a peak that includes the sink and applies the gate. A
// position below zero or at 2^30 marks the status bit and nothing else runs.

#include <metal_stdlib>
using namespace metal;

constant constexpr int kHeadDim = 128;
constant constexpr int kTile = 16;
// The query heads one SIMD group scores for its K/V head; a wider group takes more of the grid.
constant constexpr int kSlice = 3;
// The SIMD groups of a threadgroup, one a K/V head and a strip; eight strips of a head in a row.
constant constexpr int kSimds = 8;
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

// Four columns at a lane's place, in scalars: a range may start at any multiple of four, and four
// scalar loads measure the same as a packed one.
inline float4 load4(const device float *row, uint lane) {
    const size_t i = size_t(lane) * 4;
    return float4(row[i], row[i + 1], row[i + 2], row[i + 3]);
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

// One SIMD group serves one strip of one K/V head for up to `kSlice` of its query heads; a wider
// group takes more of the grid. It loads each row's keys and values once and scores its heads
// from the same load. The grid counts threadgroups of `kSimds` SIMD groups: a K/V head's
// strips in a row, then its next slice, then the next head.
kernel void relative_attention_partials(
    constant Params &p [[buffer(13)]],
    const device float *keys [[buffer(1)]],
    const device float *values [[buffer(2)]],
    const device float *q [[buffer(4)]],
    const device float *relative [[buffer(7)]],
    const device float *table [[buffer(8)]],
    const device int *pos [[buffer(11)]],
    device float *part [[buffer(12)]],
    uint gid [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]])
{
    const int position = *pos;
    if (position < 0 || position >= kWordMax) return;
    const int count = ring_rows(position, p.capacity);
    const int chunk = split_chunk(count, p.max_splits);
    const int strips = (count + chunk - 1) / chunk;

    const uint group = p.query_heads / p.kv_heads;
    const int slices = (int(group) + kSlice - 1) / kSlice;
    const int index = int(gid) * kSimds + int(simd);
    const int per_head = strips * slices;
    if (index >= int(p.kv_heads) * per_head) return;
    const int kv_head = index / per_head;
    const int slice = (index % per_head) / strips;
    const int strip = index % strips;
    const int first = strip * chunk;
    const int last = min(count, first + chunk);
    const int heads = min(int(group) - slice * kSlice, kSlice);
    const int head0 = kv_head * int(group) + slice * kSlice;
    const int slot = position % int(p.capacity);

    float4 query[kSlice], acc[kSlice];
    float rel[kSlice], peak[kSlice], sum[kSlice];
 #pragma unroll
    for (int h = 0; h < kSlice; ++h) {
        query[h] = 0.f;
        acc[h] = 0.f;
        rel[h] = 0.f;
        peak[h] = -INFINITY;
        sum[h] = 0.f;
        if (h < heads) {
            query[h] = load4(q + size_t(head0 + h) * kHeadDim, lane);
            if (lane < p.relative_dims) {
                rel[h] = relative[size_t(head0 + h) * p.relative_dims + lane];
            }
        }
    }

    for (int distance = first; distance < last; ++distance) {
        // A modulo, then one addition: a distance of several capacities is the ring's.
        int cell = (slot - distance) % int(p.capacity);
        if (cell < 0) cell += int(p.capacity);
        const size_t src = (size_t(cell) * p.kv_heads + uint(kv_head)) * kHeadDim;
        const float4 key = load4(keys + src, lane);
        const float4 value = load4(values + src, lane);
        // The bias's column at this distance: zero beyond the extent, zero in a lane past the
        // dimensions; the same in every head.
        const bool biased = distance < int(p.relative_extent);
        const float column = biased && lane < p.relative_dims
            ? table[size_t(lane) * p.relative_extent + uint(distance)]
            : 0.f;
 #pragma unroll
        for (int h = 0; h < kSlice; ++h) {
            if (h >= heads) continue;
            const float dot = simd_sum(query[h].x * key.x + query[h].y * key.y
                                       + query[h].z * key.z + query[h].w * key.w);
            const float bias = biased ? simd_sum(rel[h] * column) : 0.f;
            const float score = kScale * dot + bias;
            // One rescaling a row at the most, when a score raises the partial's peak.
            if (score > peak[h]) {
                const float correction = exp(peak[h] - score);
                sum[h] *= correction;
                acc[h] *= correction;
                peak[h] = score;
            }
            const float mass = exp(score - peak[h]);
            sum[h] += mass;
            acc[h] = fma(mass, value, acc[h]);
        }
    }

 #pragma unroll
    for (int h = 0; h < kSlice; ++h) {
        if (h >= heads) continue;
        const size_t cell = (size_t(head0 + h) * p.max_splits + uint(strip)) * kPartial;
        if (lane == 0) {
            part[cell] = peak[h];
            part[cell + 1] = sum[h];
        }
        // A lane's four components are the four columns after its own.
        part[cell + 2 + lane * 4] = acc[h].x;
        part[cell + 2 + lane * 4 + 1] = acc[h].y;
        part[cell + 2 + lane * 4 + 2] = acc[h].z;
        part[cell + 2 + lane * 4 + 3] = acc[h].w;
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
