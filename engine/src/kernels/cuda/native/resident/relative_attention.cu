#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>

#include "abi.h"
#include "device.h"
#include "mma_f16.cuh"
#include "resident/attn_split_reduce.cuh"
#include "resident/log_query_scale.cuh"
#include "resident/relative_attention.h"
#include "resident/relative_attention_tile.h"
#include "resident/step_params.h"

namespace {

constexpr int kHeadDim = 128;
constexpr int kMaxHeads = 64;
constexpr int kWordMax = (1 << 30) - 1;
constexpr int kWarp = 32;
constexpr int kWarps = 8;
constexpr int kThreads = kWarps * kWarp;
constexpr int kColumns = kHeadDim / kWarp;
constexpr float kScale = 0.08838834764831844f;  // 1 / sqrt(128)

using Args = inference_cuda_relative_attention_args;

struct Rows {
    int head, kv_head, slot, count;
};

struct Partials {
    float stats[kWarps * 2];
    float values[kWarps * kHeadDim];
};

__device__ float warp_sum(float value) {
#pragma unroll
    for (int offset = kWarp / 2; offset > 0; offset /= 2) {
        value += __shfl_xor_sync(0xffffffffu, value, offset);
    }
    return value;
}

struct KvPair {
    const float *key, *value;
};

struct SerialKv {
    __device__ KvPair operator()(const Args &a, const Rows &r, int distance) const {
        const int row = r.slot - distance;
        const size_t slot = row < 0 ? row + a.capacity : row;
        const size_t src = (slot * a.kv_heads + r.kv_head) * kHeadDim;
        return {distance == 0 ? a.k + r.kv_head * kHeadDim : a.keys + src,
                distance == 0 ? a.v + r.kv_head * kHeadDim : a.values + src};
    }
};

// The score of the row at `distance`: the scaled QK dot of the lane's key columns, summed across
// the warp, plus the learned bias. `biases` is the block's table of the bias by distance, or null
// when the warp sums each bias itself.
__device__ __forceinline__ float row_score(const Args &a, const float (&query)[kColumns], float rel,
                                           int distance, const float (&key)[kColumns],
                                           const float *biases) {
    const int lane = threadIdx.x % kWarp;
    float dot = 0.f;
#pragma unroll
    for (int i = 0; i < kColumns; ++i) {
        dot = fmaf(query[i], key[i], dot);
    }
    float bias = 0.f;
    if (distance < a.relative_extent) {
        if (biases != nullptr) {
            bias = biases[distance];
        } else {
            const float term = lane < a.relative_dims
                                   ? rel * a.table[size_t(lane) * a.relative_extent + distance]
                                   : 0.f;
            bias = warp_sum(term);
        }
    }
    return kScale * warp_sum(dot) + bias;
}

// The most query heads of one K/V group a block serves. Each K/V row a warp loads is scored and
// folded for every head of the block, so the rows cross the cache once per group instead of once
// per head; four heads bound the block's partials and bias tables in shared memory.
constexpr int kMaxBlockHeads = 4;

// The heads per block for a group of `group` query heads: its largest divisor up to the bound.
int block_heads(int group) {
    for (int heads = kMaxBlockHeads; heads > 1; --heads) {
        if (group % heads == 0) return heads;
    }
    return 1;
}

// The bias table covers extents up to kBiasRows. Below kBiasMinRows rows the warps sum each bias
// themselves, which costs less than a fill and a barrier.
constexpr int kBiasRows = 1024;
constexpr int kBiasMinRows = 64;

__device__ bool bias_table(const Args &a, int count) {
    return a.relative_extent <= kBiasRows && count >= kBiasMinRows;
}

// The relative bias of each distance below min(extent, count), one thread per distance. The adds
// follow warp_sum's butterfly (lanes at or past relative_dims add zero), so every score keeps the
// bits of the warp-summed bias; the explicit roundings keep the products and sums from contracting
// into FMAs the shuffled form cannot form. The caller synchronizes the block before reading it.
__device__ void fill_biases(const Args &a, int head, int count, float *biases) {
    const int rows = min(a.relative_extent, count);
    for (int distance = threadIdx.x; distance < rows; distance += kThreads) {
        float v[kWarp];
#pragma unroll
        for (int l = 0; l < kWarp; ++l) {
            v[l] = l < a.relative_dims
                       ? __fmul_rn(a.relative[head * a.relative_dims + l],
                                   a.table[size_t(l) * a.relative_extent + distance])
                       : 0.f;
        }
#pragma unroll
        for (int offset = kWarp / 2; offset > 0; offset /= 2) {
#pragma unroll
            for (int l = 0; l < offset; ++l) v[l] = __fadd_rn(v[l], v[l + offset]);
        }
        biases[distance] = v[0];
    }
}

// The lane's columns of one K or V row.
__device__ __forceinline__ void load_columns(const float *row, float (&out)[kColumns]) {
    const int lane = threadIdx.x % kWarp;
#pragma unroll
    for (int i = 0; i < kColumns; ++i) {
        out[i] = row[lane + i * kWarp];
    }
}

// Rows a warp loads and scores before it folds them. The loads and the scores of a batch do not
// depend on the running partial, so they are issued together; the folds then run in distance
// order, as one row at a time would, so the results keep their bits.
constexpr int kBatch = 4;

// One warp computes the weighted sum over its interleaved subset of newest-first rows: warp w
// adds the distances w, w + kWarps, ... for each of the H heads from r.head, which share the
// K/V head. Each head folds the rows in the order a block of its own would, from its own state,
// so its result keeps its bits. Only the source addresses vary by reader; the FP32 recurrence and
// order do not. `biases` holds each head's table, or is null when the warps sum each bias.
template <int H, class ReadKv>
__device__ void partial(const Args &a, const Rows &r, Partials (&p)[H], ReadKv read_kv,
                        const float (*biases)[kBiasRows]) {
    const int lane = threadIdx.x % kWarp, warp = threadIdx.x / kWarp;
    float query[H][kColumns];
    float acc[H][kColumns] = {};
    float rel[H], peak[H], sum[H];
#pragma unroll
    for (int h = 0; h < H; ++h) {
        const int head = r.head + h;
#pragma unroll
        for (int i = 0; i < kColumns; ++i) {
            query[h][i] = a.q[head * kHeadDim + lane + i * kWarp];
        }
        rel[h] = lane < a.relative_dims ? a.relative[head * a.relative_dims + lane] : 0.f;
        peak[h] = -INFINITY;
        sum[h] = 0.f;
    }
    for (int first = warp; first < r.count; first += kWarps * kBatch) {
        float keys[kBatch][kColumns], values[kBatch][kColumns];
#pragma unroll
        for (int j = 0; j < kBatch; ++j) {
            const int distance = first + j * kWarps;
            if (distance < r.count) {
                const auto [key, value] = read_kv(a, r, distance);
                load_columns(key, keys[j]);
                load_columns(value, values[j]);
            }
        }
#pragma unroll
        for (int h = 0; h < H; ++h) {
            const float *table = biases == nullptr ? nullptr : biases[h];
            float scores[kBatch];
#pragma unroll
            for (int j = 0; j < kBatch; ++j) {
                const int distance = first + j * kWarps;
                scores[j] = distance < r.count
                                ? row_score(a, query[h], rel[h], distance, keys[j], table)
                                : -INFINITY;
            }
#pragma unroll
            for (int j = 0; j < kBatch; ++j) {
                const float score = scores[j];
                if (score == -INFINITY) continue;
                const float next = fmaxf(peak[h], score);
                const float correction = expf(peak[h] - next);
                const float mass = expf(score - next);
                sum[h] = fmaf(sum[h], correction, mass);
#pragma unroll
                for (int i = 0; i < kColumns; ++i) {
                    acc[h][i] = fmaf(mass, values[j][i], acc[h][i] * correction);
                }
                peak[h] = next;
            }
        }
    }
#pragma unroll
    for (int h = 0; h < H; ++h) {
        if (lane == 0) {
            p[h].stats[warp * 2] = peak[h];
            p[h].stats[warp * 2 + 1] = sum[h];
        }
#pragma unroll
        for (int i = 0; i < kColumns; ++i) {
            p[h].values[warp * kHeadDim + lane + i * kWarp] = acc[h][i];
        }
    }
}

// The caller synchronizes the block between two calls: they share the scratch below.
__device__ void finish(const Args &a, const Rows &r, const Partials &p) {
    __shared__ float sums[kWarps], masses[kWarps], maxima[kWarps];
    const int d = threadIdx.x;
    const int active = min(kWarps, r.count);
    const float sink = a.sinks[r.head];
    const float peak = fmaxf(sink, inference::attn_split_max<kThreads>(
                                      d, active, p.stats, 2, sums, maxima));
    if (d < active) masses[d] = expf(p.stats[d * 2] - peak);
    __syncthreads();
    if (d < kHeadDim) {
        float sum, value;
        inference::attn_split_sum(active, expf(sink - peak), masses, sums, p.values + d,
                                  kHeadDim, sum, value);
        const int out = r.head * kHeadDim + d;
        const float gate_logit = a.gates[out];
        const float gate = 1.f / (1.f + expf(-gate_logit));
        const float result = (value / sum) * gate;
        // A resident gate can bypass host readback; sigmoid alone masks infinite logits.
        if (!isfinite(result) || !isfinite(gate_logit) || (d == 0 && !isfinite(sink))) {
            atomicOr(a.status, INFERENCE_CUDA_STATUS_INVALID_STEP_STATE);
        }
        a.out[out] = result;
    }
}

// Finishes the H heads from r.head, one after another.
template <int H>
__device__ void finish_heads(const Args &a, const Rows &r, const Partials (&p)[H]) {
#pragma unroll
    for (int h = 0; h < H; ++h) {
        finish(a, Rows{r.head + h, r.kv_head, r.slot, r.count}, p[h]);
        if (h + 1 < H) __syncthreads();
    }
}

// Block b serves query head b. A decode step scores one row per head, so a block per head keeps
// the grid as wide as the heads; grouping them (as the prompt tiles do) would leave most of the
// device idle for the sake of K/V rows that the cache serves to the group's other blocks anyway.
__global__ void relative_attention(Args a) {
    __shared__ Partials p[1];
    __shared__ float biases[1][kBiasRows];
    const int pos = *a.pos, head = blockIdx.x, d = threadIdx.x;
    if (pos < 0 || pos >= kWordMax) {
        if (head == 0 && d == 0) atomicOr(a.status, INFERENCE_CUDA_STATUS_INVALID_STEP_STATE);
        return;
    }
    const int group = a.query_heads / a.kv_heads;
    const Rows r{head, head / group, pos % a.capacity, min(pos + 1, a.capacity)};
    // Only the first query head in each group writes the current cache row. All
    // blocks read the current K/V directly and read the cache only for older rows.
    if (head % group == 0 && d < kHeadDim) {
        const int src = r.kv_head * kHeadDim + d;
        const size_t dst = size_t(r.slot) * a.kv_heads * kHeadDim + src;
        a.keys[dst] = a.k[src];
        a.values[dst] = a.v[src];
    }
    const bool table = bias_table(a, r.count);
    if (table) {
        fill_biases(a, head, r.count, biases[0]);
        __syncthreads();
    }
    partial<1>(a, r, p, SerialKv{}, table ? biases : nullptr);
    __syncthreads();
    finish(a, r, p[0]);
}

__device__ int row_stride(int stride, int width) { return stride == 0 ? width : stride; }

struct SnapshotKv {
    int query, stride;
    __device__ KvPair operator()(const Args &a, const Rows &r, int distance) const {
        const auto src = snapshot::source(query, r.slot, distance, a.capacity);
        const int channels = a.kv_heads * kHeadDim;
        const size_t offset = size_t(src.row) * (src.staged ? stride : channels)
                            + r.kv_head * kHeadDim;
        return {(src.staged ? a.k : a.keys) + offset,
                (src.staged ? a.v : a.values) + offset};
    }
};

// Block (b, query) serves the H query heads from b * H.
template <int H>
__global__ void snapshot_queries(Args a, int first, int input_stride, int query_stride,
                                 int side_stride) {
    __shared__ Partials p[H];
    __shared__ float biases[H][kBiasRows];
    const int query = blockIdx.y, head = blockIdx.x * H, pos = first + query;
    a.status += query;
    const int channels = a.query_heads * kHeadDim;
    a.out += query * channels;
    if (a.pos[query * row_stride(input_stride, 1)] != pos) {
        if (head == 0 && threadIdx.x == 0) atomicOr(a.status, INFERENCE_CUDA_STATUS_INVALID_STEP_STATE);
        // Output projection can already be queued. Give it initialized inputs even
        // though the host will discard the whole tile based on status.
        if (threadIdx.x < kHeadDim) {
#pragma unroll
            for (int h = 0; h < H; ++h) a.out[(head + h) * kHeadDim + threadIdx.x] = 0.f;
        }
        return;
    }
    const int group = a.query_heads / a.kv_heads;
    const Rows r{head, head / group, pos % a.capacity, min(pos + 1, a.capacity)};
    // K/V stay at the tile base; the reader chooses a staged row or the pre-tile ring.
    a.q += query * row_stride(query_stride, channels);
    a.relative += query * row_stride(query_stride, a.query_heads * a.relative_dims);
    a.sinks += query * row_stride(side_stride, a.query_heads);
    a.gates += query * row_stride(side_stride, channels);
    const bool table = bias_table(a, r.count);
    if (table) {
#pragma unroll
        for (int h = 0; h < H; ++h) fill_biases(a, head + h, r.count, biases[h]);
        __syncthreads();
    }
    partial<H>(a, r, p, SnapshotKv{query, row_stride(input_stride, a.kv_heads * kHeadDim)},
               table ? biases : nullptr);
    __syncthreads();
    finish_heads<H>(a, r, p);
}

// The tensor-core form of a snapshot tile: block (head, b) serves query rows 64 b .. 64 b + 63
// of one query head, 16 rows per warp, over every key any of them sees, 32 keys a step. Q, K,
// V, the probabilities, the relative coefficients and the bias table enter the MMAs rounded to
// f16; scores, the online softmax, the sink and the outputs stay f32. The relative bias of a
// row and a key depends on their distance, so each step multiplies the warp's 16 coefficient
// rows by the table rows of the 47 distances its keys can have, and every score reads its
// bias from that product by distance.
constexpr int kTileQueries = 64;
constexpr int kTileKeys = 32;
constexpr int kTileWarps = kTileQueries / 16;
constexpr int kTileThreads = kTileWarps * kWarp;
constexpr int kKeyPitch = kHeadDim + 8;  // halves; 272-byte rows keep ldmatrix rows 16-byte aligned
constexpr int kRelSteps = 2;             // relative dims up to 32, in MMA k-steps of 16
constexpr int kRelPitch = 16 * kRelSteps + 8;
constexpr int kBiasCols = 16 + kTileKeys;          // the 47 distances of one warp, rounded to n8
constexpr int kSliceRows = 16 * (kTileWarps - 1) + kBiasCols;  // the distances of the block

struct TileSmem {
    __half k[kTileKeys][kKeyPitch];
    __half v[kTileKeys][kKeyPitch];
    __half table[kSliceRows][kRelPitch];  // distance-major, relative dim contiguous
    float bias[kTileWarps][16][kBiasCols];
};

#if __CUDA_ARCH__ >= 750
using inference::mma::ldsm_x4_trans;
using inference::mma::mma_16816;
using inference::mma::smem_addr;

__device__ __forceinline__ unsigned pack_half2(float lo, float hi) {
    const __half2 h = __floats2half2_rn(lo, hi);
    return *reinterpret_cast<const unsigned *>(&h);
}

// The A fragments of 16 rows of a row-major f32 plane (rows past `live` and columns past
// `width` read as zero): k-step s covers columns 16 s .. 16 s + 15.
template <int Steps>
__device__ __forceinline__ void load_rows(const float *const (&row)[2], const bool (&live)[2],
                                          int width, unsigned (&frag)[Steps][4]) {
    const int t = threadIdx.x % 4;
#pragma unroll
    for (int s = 0; s < Steps; ++s) {
#pragma unroll
        for (int part = 0; part < 4; ++part) {
            const int i = part % 2, c = 16 * s + 2 * t + 8 * (part / 2);
            const float lo = live[i] && c < width ? row[i][c] : 0.f;
            const float hi = live[i] && c + 1 < width ? row[i][c + 1] : 0.f;
            frag[s][part] = pack_half2(lo, hi);
        }
    }
}
#endif

__global__ void __launch_bounds__(kTileThreads) snapshot_tile(Args a, int first, int count,
                                                              int input_stride, int query_stride,
                                                              int side_stride) {
#if __CUDA_ARCH__ >= 750
    __shared__ __align__(16) TileSmem sm;
    const int lane = threadIdx.x % kWarp, warp = threadIdx.x / kWarp, g = lane / 4, t = lane % 4;
    const int head = blockIdx.x, kv_head = head / (a.query_heads / a.kv_heads);
    const int row0 = blockIdx.y * kTileQueries, rows = min(kTileQueries, count - row0);
    const int channels = a.query_heads * kHeadDim, kv_channels = a.kv_heads * kHeadDim;
    const int kv_stride = row_stride(input_stride, kv_channels);
    // The thread's two rows of the warp's 16: g and g + 8.
    int row[2];
    bool live[2], placed[2];
    const float *q_row[2], *rel_row[2];
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        row[i] = row0 + 16 * warp + g + 8 * i;
        live[i] = row[i] < count;
        placed[i] = live[i] && a.pos[size_t(row[i]) * row_stride(input_stride, 1)] == first + row[i];
        if (live[i] && !placed[i] && head == 0 && t == 0) {
            atomicOr(a.status + row[i], INFERENCE_CUDA_STATUS_INVALID_STEP_STATE);
        }
        q_row[i] = a.q + size_t(row[i]) * row_stride(query_stride, channels) + head * kHeadDim;
        rel_row[i] = a.relative +
                     size_t(row[i]) * row_stride(query_stride, a.query_heads * a.relative_dims) +
                     head * a.relative_dims;
    }
    unsigned qf[kHeadDim / 16][4], rf[kRelSteps][4];
    load_rows<kHeadDim / 16>(q_row, live, kHeadDim, qf);
    load_rows<kRelSteps>(rel_row, live, a.relative_dims, rf);

    float peak[2] = {-INFINITY, -INFINITY}, mass[2] = {0.f, 0.f};
    float acc[kHeadDim / 8][4] = {};
    const int last = first + row0 + rows - 1;  // the newest key of the block
    const int oldest = max(0, first + row0 - a.capacity + 1);
    for (int k0 = oldest; k0 <= last; k0 += kTileKeys) {
        __syncthreads();
        // K and V of keys k0 .. k0 + 31 as f16: a staged row of the tile, or the pre-tile ring.
        for (int i = threadIdx.x; i < kTileKeys * (kHeadDim / 4); i += kTileThreads) {
            const int key = i / (kHeadDim / 4), c = 4 * (i % (kHeadDim / 4)), kp = k0 + key;
            float kx[4] = {}, vx[4] = {};
            if (kp <= last) {
                const size_t at = kp >= first
                                      ? size_t(kp - first) * kv_stride + kv_head * kHeadDim + c
                                      : (size_t(kp % a.capacity) * a.kv_heads + kv_head) * kHeadDim + c;
                const float *ks = kp >= first ? a.k : a.keys, *vs = kp >= first ? a.v : a.values;
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    kx[e] = ks[at + e];
                    vx[e] = vs[at + e];
                }
            }
            *reinterpret_cast<uint2 *>(&sm.k[key][c]) =
                make_uint2(pack_half2(kx[0], kx[1]), pack_half2(kx[2], kx[3]));
            *reinterpret_cast<uint2 *>(&sm.v[key][c]) =
                make_uint2(pack_half2(vx[0], vx[1]), pack_half2(vx[2], vx[3]));
        }
        // The table rows of the block's distances: slice row j is distance d_lo + j, zero where
        // no bias applies (a negative distance is masked below).
        const int d_lo = first + row0 - k0 - (kTileKeys - 1);
        for (int i = threadIdx.x; i < kSliceRows * 16 * kRelSteps; i += kTileThreads) {
            const int j = i % kSliceRows, l = i / kSliceRows, d = d_lo + j;
            const bool on = l < a.relative_dims && d >= 0 && d < a.relative_extent;
            sm.table[j][l] = __float2half_rn(on ? a.table[size_t(l) * a.relative_extent + d] : 0.f);
        }
        __syncthreads();

        // bias[i][c]: row i of the warp against distance d_lo + 16 warp + c.
        {
            float b[kBiasCols / 8][4] = {};
#pragma unroll
            for (int n = 0; n < kBiasCols / 8; ++n) {
                const __half *tr = sm.table[16 * warp + 8 * n + g];
#pragma unroll
                for (int s = 0; s < kRelSteps; ++s) {
                    mma_16816(b[n], rf[s], *reinterpret_cast<const unsigned *>(tr + 16 * s + 2 * t),
                              *reinterpret_cast<const unsigned *>(tr + 16 * s + 2 * t + 8));
                }
            }
#pragma unroll
            for (int n = 0; n < kBiasCols / 8; ++n) {
                *reinterpret_cast<float2 *>(&sm.bias[warp][g][8 * n + 2 * t]) = make_float2(b[n][0], b[n][1]);
                *reinterpret_cast<float2 *>(&sm.bias[warp][g + 8][8 * n + 2 * t]) = make_float2(b[n][2], b[n][3]);
            }
            __syncwarp();
        }

        // Scores of the warp's 16 rows against the 32 keys.
        float s[kTileKeys / 8][4] = {};
#pragma unroll
        for (int n = 0; n < kTileKeys / 8; ++n) {
            const __half *kr = sm.k[8 * n + g];
#pragma unroll
            for (int ks = 0; ks < kHeadDim / 16; ++ks) {
                mma_16816(s[n], qf[ks], *reinterpret_cast<const unsigned *>(kr + 16 * ks + 2 * t),
                          *reinterpret_cast<const unsigned *>(kr + 16 * ks + 2 * t + 8));
            }
        }
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const int r = g + 8 * i, pos = first + row0 + 16 * warp + r;
            float top = -INFINITY;
#pragma unroll
            for (int n = 0; n < kTileKeys / 8; ++n) {
#pragma unroll
                for (int e = 0; e < 2; ++e) {
                    const int j = 8 * n + 2 * t + e, distance = pos - (k0 + j);
                    float &x = s[n][2 * i + e];
                    x = distance >= 0 && distance < a.capacity && k0 + j <= last
                            ? fmaf(kScale, x, sm.bias[warp][r][kTileKeys - 1 + r - j])
                            : -INFINITY;
                    top = fmaxf(top, x);
                }
            }
            top = fmaxf(top, __shfl_xor_sync(0xffffffffu, top, 1));
            top = fmaxf(top, __shfl_xor_sync(0xffffffffu, top, 2));
            const float next = fmaxf(peak[i], top);
            const float correction = next == -INFINITY ? 1.f : expf(peak[i] - next);
            peak[i] = next;
            mass[i] *= correction;
#pragma unroll
            for (int n = 0; n < kHeadDim / 8; ++n) {
                acc[n][2 * i] *= correction;
                acc[n][2 * i + 1] *= correction;
            }
#pragma unroll
            for (int n = 0; n < kTileKeys / 8; ++n) {
#pragma unroll
                for (int e = 0; e < 2; ++e) {
                    float &x = s[n][2 * i + e];
                    x = next == -INFINITY ? 0.f : expf(x - next);
                    mass[i] += x;
                }
            }
        }
        // acc += P V, P from the score fragments, V by transposed ldmatrix (matrix m of a load
        // is keys 16 kk + 8 (m % 2) .. + 7 at dims 16 np + 8 (m / 2) .. + 7).
#pragma unroll
        for (int kk = 0; kk < kTileKeys / 16; ++kk) {
            const unsigned p[4] = {pack_half2(s[2 * kk][0], s[2 * kk][1]),
                                   pack_half2(s[2 * kk][2], s[2 * kk][3]),
                                   pack_half2(s[2 * kk + 1][0], s[2 * kk + 1][1]),
                                   pack_half2(s[2 * kk + 1][2], s[2 * kk + 1][3])};
            const int m = lane / 8, key = 16 * kk + 8 * (m % 2) + lane % 8;
#pragma unroll
            for (int np = 0; np < kHeadDim / 16; ++np) {
                unsigned v[4];
                ldsm_x4_trans(smem_addr(&sm.v[key][16 * np + 8 * (m / 2)]), v);
                mma_16816(acc[2 * np], p, v[0], v[1]);
                mma_16816(acc[2 * np + 1], p, v[2], v[3]);
            }
        }
    }

    // The sink joins the denominator once; the gate scales each output.
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        float total = mass[i];
        total += __shfl_xor_sync(0xffffffffu, total, 1);
        total += __shfl_xor_sync(0xffffffffu, total, 2);
        if (!live[i]) continue;
        float *out = a.out + size_t(row[i]) * channels + head * kHeadDim;
        if (!placed[i]) {
#pragma unroll
            for (int n = 0; n < kHeadDim / 8; ++n) out[8 * n + 2 * t] = out[8 * n + 2 * t + 1] = 0.f;
            continue;
        }
        const float sink = a.sinks[size_t(row[i]) * row_stride(side_stride, a.query_heads) + head];
        const float top = fmaxf(peak[i], sink);
        const float keep = expf(peak[i] - top);
        const float denominator = fmaf(total, keep, expf(sink - top));
        const float *gates = a.gates + size_t(row[i]) * row_stride(side_stride, channels) + head * kHeadDim;
        bool bad = !isfinite(sink);
#pragma unroll
        for (int n = 0; n < kHeadDim / 8; ++n) {
#pragma unroll
            for (int e = 0; e < 2; ++e) {
                const int d = 8 * n + 2 * t + e;
                const float gate_logit = gates[d];
                const float gate = 1.f / (1.f + expf(-gate_logit));
                const float result = (acc[n][2 * i + e] * keep / denominator) * gate;
                bad |= !isfinite(result) || !isfinite(gate_logit);
                out[d] = result;
            }
        }
        if (bad) atomicOr(a.status + row[i], INFERENCE_CUDA_STATUS_INVALID_STEP_STATE);
    }
#else
    // Never launched below sm_75 (the entry checks the loaded code); refuse the tile if it is.
    if (threadIdx.x == 0) atomicOr(a.status + blockIdx.y * kTileQueries, INFERENCE_CUDA_STATUS_INVALID_STEP_STATE);
#endif
}

__global__ void snapshot_commit(Args a, int first, int count, int input_stride) {
    __shared__ int valid;
    if (threadIdx.x == 0) {
        valid = 1;
        for (int i = 0; i < count; ++i) valid &= (a.status[i] == 0);
    }
    __syncthreads();
    if (!valid) return;
    const int row = snapshot::first_commit(count, a.capacity) + blockIdx.x;
    const int slot = (first + row) % a.capacity;
    const int channels = a.kv_heads * kHeadDim;
    for (int d = threadIdx.x; d < channels; d += blockDim.x) {
        const size_t src = size_t(row) * row_stride(input_stride, channels) + d;
        const size_t dst = size_t(slot) * channels + d;
        a.keys[dst] = a.k[src];
        a.values[dst] = a.v[src];
    }
}

// The host uses an ordered FP32 sum, not a parallel reduction. Parallelize
// across heads and channels without changing the accumulation or rounding.
// Side rows hold Q, `gap` other words, relative coefficients, sinks and gates.
__global__ void clip_prefill_queries(float *side, const float *scales, uint32_t *status,
                                     int heads, int dims, int gap, float q_limit,
                                     float rel_limit, int first, int log_floor,
                                     float log_alpha) {
    const int row = blockIdx.y, d = threadIdx.x;
    const bool relative = int(blockIdx.x) >= heads;
    const int head = blockIdx.x % heads;
    const int width = relative ? dims : kHeadDim;
    const int query_words = heads * kHeadDim;
    const int stride = 2 * query_words + gap + heads * (dims + 1);
    const int at = (relative ? query_words + gap : 0) + head * width + d;
    const int scale_at = (relative ? query_words : 0) + head * width + d;
    __shared__ float squares[kHeadDim], factor, tau;
    float raw = 0.f, x = 0.f;
    if (d < width) {
        raw = side[row * stride + at];
        x = __fmul_rn(raw, scales[scale_at]);
        squares[d] = __fmul_rn(x, x);
    }
    __syncthreads();
    if (d == 0) {
        float sum = 0.f;
        for (int i = 0; i < width; ++i) sum = __fadd_rn(sum, squares[i]);
        factor = fminf(1.f, __fdiv_rn(relative ? rel_limit : q_limit,
                                      __fadd_rn(__fsqrt_rn(sum), 1.e-6f)));
        tau = log_query_scale(first + row, log_floor, log_alpha);
    }
    __syncthreads();
    float result = __fmul_rn(x, factor);
    if (tau != 1.f) result = __fmul_rn(result, tau);
    // Test the raw projection too: clipping and sigmoid must not mask errors
    // that used to be caught during the projection readback.
    const int invalid = __syncthreads_or(!isfinite(raw) || !isfinite(x) || !isfinite(result));
    if (d == 0 && invalid) atomicOr(status + row, INFERENCE_CUDA_STATUS_INVALID_STEP_STATE);
    if (d < width) side[row * stride + at] = result;
}

// Whether the loaded code of the tensor-core tile kernel was built for sm_75 or later, the
// first PTX with the f16 MMA; older devices take the warp kernel. A failed query is cleared so
// the launch check below does not report it.
bool tensor_core_tile() {
    cudaFuncAttributes attr{};
    if (cudaFuncGetAttributes(&attr, snapshot_tile) != cudaSuccess) {
        (void)cudaGetLastError();
        return false;
    }
    return attr.ptxVersion >= 75;
}

bool valid(const inference_cuda_relative_attention_args &a) {
    if (a.kv_heads < 1 || a.query_heads < a.kv_heads || a.query_heads > kMaxHeads ||
        a.query_heads % a.kv_heads != 0 || a.relative_dims < 1 || a.relative_dims > kWarp ||
        a.relative_extent < 1 || a.relative_extent > kWordMax ||
        a.capacity < 1 || a.capacity > kWordMax) {
        return false;
    }
    const void *ranges[] = {a.out, a.keys, a.values, a.status, a.q, a.k,
                            a.v, a.relative, a.table, a.sinks, a.gates, a.pos};
    for (const void *p : ranges) {
        if (p == nullptr || uintptr_t(p) % 4 != 0) return false;
    }
    return true;
}

}  // namespace

extern "C" int inference_cuda_relative_attention_clip_queries(
    float *side, const float *scales, uint32_t *status, int rows, int heads,
    int dims, int gap, float q_limit, float rel_limit, int first, int log_floor,
    float log_alpha, void *stream, int *out_stage) {
    report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
    if (side == nullptr || scales == nullptr || status == nullptr || stream == nullptr ||
        (uintptr_t(side) | uintptr_t(scales) | uintptr_t(status)) % 4 != 0 ||
        rows < 1 || rows > 512 || heads < 1 || heads > kMaxHeads || dims < 1 || dims > kWarp ||
        gap < 0 ||
        !isfinite(q_limit) || q_limit <= 0.f || !isfinite(rel_limit) || rel_limit <= 0.f ||
        first < 0 || first > kWordMax - rows ||
        (log_floor > 0 && (!isfinite(log_alpha) || log_alpha < 0.f))) {
        return INFERENCE_CUDA_INVALID_ARGUMENT;
    }
    clip_prefill_queries<<<dim3(2 * heads, rows), kHeadDim, 0, (cudaStream_t)stream>>>(
        side, scales, status, heads, dims, gap, q_limit, rel_limit, first, log_floor, log_alpha);
    Failure failure;
    failure.record(cudaGetLastError(), INFERENCE_CUDA_STAGE_LAUNCH);
    report_stage(out_stage, failure.stage);
    return (int)failure.err;
}

extern "C" int inference_cuda_relative_attention_launch(
    const inference_cuda_relative_attention_args *in, void *stream, int *out_stage) {
    try {
        report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
        if (in == nullptr || stream == nullptr || !valid(*in)) return INFERENCE_CUDA_INVALID_ARGUMENT;
        Failure failure;
        const auto s = (cudaStream_t)stream;
        relative_attention<<<in->query_heads, kThreads, 0, s>>>(*in);
        failure.record(cudaGetLastError(), INFERENCE_CUDA_STAGE_LAUNCH);
        report_stage(out_stage, failure.stage);
        return (int)failure.err;
    } catch (...) {
        return INFERENCE_CUDA_NATIVE_EXCEPTION;
    }
}

extern "C" int inference_cuda_relative_attention_tile_launch(
    const Args *in, int first, int count, int input_stride, int query_stride,
    int side_stride, void *stream, int *out_stage) {
    try {
        report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
        if (in == nullptr || stream == nullptr || !valid(*in) ||
            count < 2 || count > snapshot::kLimit || !snapshot::valid_interval(first, count)) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        const int channels = in->query_heads * kHeadDim;
        auto valid_stride = [](int stride, int width) {
            return stride == 0 || (stride >= width && stride <= snapshot::kMaxStride);
        };
        if (!valid_stride(input_stride, in->kv_heads * kHeadDim) ||
            !valid_stride(query_stride, channels) || !valid_stride(side_stride, channels)) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        Failure failure;
        const auto s = (cudaStream_t)stream;
        if (tensor_core_tile()) {
            const dim3 grid(in->query_heads, (count + kTileQueries - 1) / kTileQueries);
            snapshot_tile<<<grid, kTileThreads, 0, s>>>(*in, first, count, input_stride,
                                                       query_stride, side_stride);
        } else {
            const int heads = block_heads(in->query_heads / in->kv_heads);
            const dim3 grid(in->query_heads / heads, count);
            switch (heads) {
                case 4: snapshot_queries<4><<<grid, kThreads, 0, s>>>(
                            *in, first, input_stride, query_stride, side_stride); break;
                case 3: snapshot_queries<3><<<grid, kThreads, 0, s>>>(
                            *in, first, input_stride, query_stride, side_stride); break;
                case 2: snapshot_queries<2><<<grid, kThreads, 0, s>>>(
                            *in, first, input_stride, query_stride, side_stride); break;
                default: snapshot_queries<1><<<grid, kThreads, 0, s>>>(
                            *in, first, input_stride, query_stride, side_stride); break;
            }
        }
        failure.record(cudaGetLastError(), INFERENCE_CUDA_STAGE_LAUNCH);
        if (failure.err == cudaSuccess) {
            // The stream, not CTA ordering, separates all snapshot reads from writes.
            snapshot_commit<<<min(count, in->capacity), kThreads, 0, s>>>(
                *in, first, count, input_stride);
            failure.record(cudaGetLastError(), INFERENCE_CUDA_STAGE_LAUNCH);
        }
        report_stage(out_stage, failure.stage);
        return (int)failure.err;
    } catch (...) {
        return INFERENCE_CUDA_NATIVE_EXCEPTION;
    }
}
