// One relative-attention decode step with F32 planes, in three dispatches of one command buffer.
//
// The contract is the CUDA adapter's relative-attention step: twelve ranges in the ABI's order,
// the first four written. The commit fills the current row's slot of both rings. The partials
// score the ring's rows a strip at a time, one SIMD group a strip of a K/V head for up to three
// of its query heads, each row's keys and values loaded once, and keep each strip's peak, sum
// and weighted value.
// The finish folds the partials against a peak that includes the sink and applies the gate. A
// position below zero or at 2^30 marks the status bit and nothing else runs.
//
// The rings hold F32, F16 or Int8 rows. The F32 entry points carry no suffix, the F16 and Int8
// ones `_f16` and `_int8`, and share their bodies. An Int8 row is 128 codes, then four half
// scales of 32 values each.

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
// The splits the finish folds an iteration, their loads in flight together.
constant constexpr int kFold = 8;
// 1 / sqrt(128): the scale of every score, as the CUDA step's shader applies it.
constant constexpr float kScale = 0.08838834764831844f;
// An Int8 ring row: groups of 32 values, a code a value over a half scale a group.
constant constexpr int kInt8Group = 32;
constant constexpr int kInt8HeadBytes = kHeadDim + (kHeadDim / kInt8Group) * 2;

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

// The same four columns of an F16 row, widened exactly. Four scalar loads: a row starts at any
// multiple of four bytes, which is not one of eight.
inline float4 load4(const device half *row, uint lane) {
    const size_t i = size_t(lane) * 4;
    return float4(float(row[i]), float(row[i + 1]), float(row[i + 2]), float(row[i + 3]));
}

// The same four columns of an Int8 row: the lane's four codes over its group's scale, a product
// that is exact. A row starts at a multiple of four bytes, so the four codes are one load.
inline float4 load4(const device int8_t *row, uint lane) {
    const device char4 *packed = reinterpret_cast<const device char4 *>(row);
    const device half *scales = reinterpret_cast<const device half *>(row + kHeadDim);
    return float4(packed[lane]) * float(scales[lane / (kInt8Group / 4)]);
}

// The elements a row of a ring takes: an F32 or F16 row its 128 values, an Int8 row its 136 bytes.
inline size_t ring_stride(const device float *) { return kHeadDim; }
inline size_t ring_stride(const device half *) { return kHeadDim; }
inline size_t ring_stride(const device int8_t *) { return kInt8HeadBytes; }

// The current row goes into its slot of both rings: one threadgroup a K/V head, 256 threads,
// one a value. An F32 ring takes the value as it is; an F16 ring rounds it once, as the CUDA
// step's `__float2half_rn` does.
template <typename Ring>
inline void commit_body(
    constant Params &p,
    device Ring *keys,
    device Ring *values,
    device atomic_int *status,
    const device float *k,
    const device float *v,
    const device int *pos,
    uint gid,
    uint tid) {
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
        values[cell] = Ring(v[src]);
    } else {
        keys[cell] = Ring(k[src]);
    }
}

kernel void relative_attention_commit(
    constant Params &p [[buffer(13)]],
    device float *keys [[buffer(1)]],
    device float *values [[buffer(2)]],
    device atomic_int *status [[buffer(3)]],
    const device float *k [[buffer(5)]],
    const device float *v [[buffer(6)]],
    const device int *pos [[buffer(11)]],
    uint gid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    commit_body(p, keys, values, status, k, v, pos, gid, tid);
}

kernel void relative_attention_commit_f16(
    constant Params &p [[buffer(13)]],
    device half *keys [[buffer(1)]],
    device half *values [[buffer(2)]],
    device atomic_int *status [[buffer(3)]],
    const device float *k [[buffer(5)]],
    const device float *v [[buffer(6)]],
    const device int *pos [[buffer(11)]],
    uint gid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    commit_body(p, keys, values, status, k, v, pos, gid, tid);
}

// The Int8 commit: the same grid, a column from the SIMD coordinates, so that the 32 lanes of a
// SIMD group hold the 32 values of one group of one plane. A group's scale is its greatest
// magnitude over 127, NaN propagated, rounded to half and stored; a code is the value over the
// stored scale's value, rounded to the nearest even and clamped, 0 when the scale's bits are 0
// or the quotient is not a number. The library is built without fast math, so each division
// rounds once.
kernel void relative_attention_commit_int8(
    constant Params &p [[buffer(13)]],
    device int8_t *keys [[buffer(1)]],
    device int8_t *values [[buffer(2)]],
    device atomic_int *status [[buffer(3)]],
    const device float *k [[buffer(5)]],
    const device float *v [[buffer(6)]],
    const device int *pos [[buffer(11)]],
    uint gid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]]) {
    const int position = *pos;
    if (position < 0 || position >= kWordMax) {
        if (gid == 0 && tid == 0) {
            atomic_fetch_or_explicit(status, kStatusInvalidStepState, memory_order_relaxed);
        }
        return;
    }
    const size_t slot = size_t(uint(position) % p.capacity);
    const uint d = simd * uint(kInt8Group) + lane;
    const bool value_plane = d >= uint(kHeadDim);
    device int8_t *rings = value_plane ? values : keys;
    device int8_t *base = rings + (slot * p.kv_heads + gid) * kInt8HeadBytes;
    const float x = (value_plane ? v : k)[gid * kHeadDim + d];
    // The greatest magnitude of the group, a NaN over every value, as the CUDA step's warp max.
    float magnitude = fabs(x);
    for (int offset = kInt8Group / 2; offset > 0; offset >>= 1) {
        const float other = simd_shuffle_xor(magnitude, ushort(offset));
        magnitude = magnitude != magnitude || other != other ? NAN : max(magnitude, other);
    }
    const half stored = half(magnitude / 127.f);
    const float quotient = x / float(stored);
    int code = 0;
    if (as_type<ushort>(stored) != 0 && !isnan(quotient)) {
        code = int(clamp(rint(quotient), -127.f, 127.f));
    }
    base[d] = int8_t(code);
    if (d % uint(kInt8Group) == 0) {
        reinterpret_cast<device half *>(base + kHeadDim)[d / uint(kInt8Group)] = stored;
    }
}

// One SIMD group serves one strip of one K/V head for up to `kSlice` of its query heads; a wider
// group takes more of the grid. It loads each row's keys and values once and scores its heads
// from the same load. The grid counts threadgroups of `kSimds` SIMD groups: a K/V head's
// strips in a row, then its next slice, then the next head. The F32, F16 and Int8 entry points
// share the body.
template <typename Ring>
inline void partials_body(
    constant Params &p,
    const device Ring *keys,
    const device Ring *values,
    const device float *q,
    const device float *relative,
    const device float *table,
    const device int *pos,
    device float *part,
    uint gid,
    uint lane,
    uint simd) {
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
        const size_t src = (size_t(cell) * p.kv_heads + uint(kv_head)) * ring_stride(keys);
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
    uint simd [[simdgroup_index_in_threadgroup]]) {
    partials_body(p, keys, values, q, relative, table, pos, part, gid, lane, simd);
}

kernel void relative_attention_partials_f16(
    constant Params &p [[buffer(13)]],
    const device half *keys [[buffer(1)]],
    const device half *values [[buffer(2)]],
    const device float *q [[buffer(4)]],
    const device float *relative [[buffer(7)]],
    const device float *table [[buffer(8)]],
    const device int *pos [[buffer(11)]],
    device float *part [[buffer(12)]],
    uint gid [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]]) {
    partials_body(p, keys, values, q, relative, table, pos, part, gid, lane, simd);
}

kernel void relative_attention_partials_int8(
    constant Params &p [[buffer(13)]],
    const device int8_t *keys [[buffer(1)]],
    const device int8_t *values [[buffer(2)]],
    const device float *q [[buffer(4)]],
    const device float *relative [[buffer(7)]],
    const device float *table [[buffer(8)]],
    const device int *pos [[buffer(11)]],
    device float *part [[buffer(12)]],
    uint gid [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]]) {
    partials_body(p, keys, values, q, relative, table, pos, part, gid, lane, simd);
}

// The head's result from its splits: each partial weighed by the exponential of its peak
// against a top that includes the sink, whose mass joins the denominator. One thread a column.
// Eight splits' words are in flight an iteration: a rolled loop waits on every load, and the
// chain of 512 waits costs more than the partials that made them.
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
    const size_t base = head * p.max_splits;
    const float sink = sinks[head];
    float top = sink;
    for (int s = 0; s < active; s += kFold) {
        float peak[kFold];
#pragma unroll
        for (int u = 0; u < kFold; ++u) {
            peak[u] = s + u < active ? part[(base + uint(s + u)) * kPartial] : -INFINITY;
        }
#pragma unroll
        for (int u = 0; u < kFold; ++u) top = fmax(top, peak[u]);
    }
    float sum = exp(sink - top);
    float value = 0.f;
    for (int s = 0; s < active; s += kFold) {
        float peak[kFold], sums[kFold], acc[kFold];
#pragma unroll
        for (int u = 0; u < kFold; ++u) {
            if (s + u < active) {
                const size_t cell = (base + uint(s + u)) * kPartial;
                peak[u] = part[cell];
                sums[u] = part[cell + 1];
                acc[u] = part[cell + 2 + d];
            } else {
                peak[u] = -INFINITY;
                sums[u] = 0.f;
                acc[u] = 0.f;
            }
        }
#pragma unroll
        for (int u = 0; u < kFold; ++u) {
            const float mass = exp(peak[u] - top);
            sum = fma(mass, sums[u], sum);
            value = fma(mass, acc[u], value);
        }
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
