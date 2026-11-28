/* Entry and kernels of block_ops. See block_ops.h for the slot table of each operation.
 *
 * Numerics: f32 throughout with IEEE expf/tanhf/logf (this unit is built without
 * -use_fast_math). Reductions run in a fixed order (warp shuffle tree, then the warp
 * partials summed in index order), so a launch repeats bit for bit. q8_1 blocks follow the
 * vendored quantize rule: d = amax / 127, q = roundf(x / d), s = the raw sum of the inputs.
 * The chunk and single-row decode forms of the latent score (resident/mla_attn_chunk.cuh) alone
 * multiply f16 operands on tensor cores; their sums are f32 and in a fixed order too. */
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <math.h>
#include <stddef.h>
#include <stdint.h>

#include "abi.h"
#include "device.h"
#include "mma_f16.cuh"
#include "resident/block_ops.h"

#include "quant_blocks.h"
#include "resident/q8_1_store.cuh"

namespace {

constexpr int kQ8Block = 36;     /* bytes of one block_q8_1: half2 ds + 32 codes */
constexpr int kQ8Padding = 512;  /* values per row are padded to a multiple of this */

constexpr int kHcStreams = 4;
constexpr int kHcMixes = 17;     /* 4 pre + 4 post + 9 comb */

constexpr int kMlaRank = 512;
constexpr int kMlaRope = 64;
constexpr int kMlaWidth = kMlaRank + kMlaRope;
constexpr int kMlaGroup = 8;     /* query heads per attention block */
constexpr int kMlaTile = 32;     /* key positions per tile */
constexpr int kMlaRowHalves = kMlaWidth + 2;  /* padded smem row: 289 words, no bank conflicts */
constexpr int kMlaPartStride = kMlaRank + 4;  /* acc[512], m, l, two spare */

constexpr int kRdaDim = 128;

__device__ __forceinline__ float sigmoid_f(float x) { return 1.0f / (1.0f + expf(-x)); }

__device__ __forceinline__ float softplus_f(float x) {
    return x > 20.0f ? x : log1pf(expf(x));
}

__device__ __forceinline__ float warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

__device__ __forceinline__ float warp_max(float v) {
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}

/* Sum over the block; `scratch` holds one float per warp. Every thread gets the total. */
__device__ float block_sum(float v, float *scratch) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    v = warp_sum(v);
    __syncthreads();
    if (lane == 0) scratch[warp] = v;
    __syncthreads();
    float total = 0.0f;
    const int warps = blockDim.x >> 5;
    for (int i = 0; i < warps; ++i) total += scratch[i];
    return total;
}

__host__ __device__ size_t q8_row_bytes(long long k) {
    const long long padded = (k + kQ8Padding - 1) / kQ8Padding * kQ8Padding;
    return (size_t)(padded / 32) * kQ8Block;
}

/* Zeroes the padding blocks of one row after its `k / 32` live blocks. */
__device__ void zero_q8_tail(unsigned char *row, long long k) {
    const long long live = k / 32;
    const long long total = (k + kQ8Padding - 1) / kQ8Padding * kQ8Padding / 32;
    for (long long i = (long long)threadIdx.x; i < (total - live) * (kQ8Block / 4); i += blockDim.x)
        reinterpret_cast<unsigned *>(row + live * kQ8Block)[i] = 0u;
}

/* ---- mixed residual ---------------------------------------------------------------- */

__device__ void hc_read_solve(const float *tot, const float *base, const float *scale, int width,
                              float eps, float magnitude, float *s_pre, float *post_row,
                              float *comb_row) {
    const float rms = sqrtf(tot[kHcMixes] / (float)width + eps);
    float mixes[kHcMixes];
    for (int m = 0; m < kHcMixes; ++m) mixes[m] = tot[m] / rms;
    for (int s = 0; s < kHcStreams; ++s) {
        s_pre[s] = sigmoid_f(mixes[s] * scale[0] + base[s]);
        post_row[s] = magnitude * sigmoid_f(mixes[kHcStreams + s] * scale[1] + base[kHcStreams + s]);
    }
    /* z[j][q] of the 3x3 block, then e = H^T z H with the Helmert basis rows. */
    float z[3][3];
    for (int j = 0; j < 3; ++j)
        for (int q = 0; q < 3; ++q)
            z[j][q] = mixes[2 * kHcStreams + j * 3 + q] * scale[2] + base[2 * kHcStreams + j * 3 + q];
    float helmert[kHcStreams][3];
    for (int i = 0; i < kHcStreams; ++i)
        for (int j = 0; j < 3; ++j) {
            const float v = i <= j ? 1.0f : (i == j + 1 ? -(float)(j + 1) : 0.0f);
            helmert[i][j] = v / sqrtf((float)((j + 1) * (j + 2)));
        }
    float e[kHcStreams][kHcStreams];
    float rho = 0.0f;
    for (int i = 0; i < kHcStreams; ++i)
        for (int x = 0; x < kHcStreams; ++x) {
            float acc = 0.0f;
            for (int j = 0; j < 3; ++j)
                for (int q = 0; q < 3; ++q) acc += helmert[x][q] * z[j][q] * helmert[i][j];
            e[i][x] = acc;
            const float chart = 0.5f * (i == x ? 1.0f : 0.0f) + 0.5f / (float)kHcStreams;
            rho = fmaxf(rho, -acc / chart);
        }
    rho = 1.0f + fmaxf(rho, 0.0f);
    const float gamma = sigmoid_f(scale[3]);
    for (int i = 0; i < kHcStreams; ++i)
        for (int x = 0; x < kHcStreams; ++x) {
            const float ident = i == x ? 1.0f : 0.0f;
            const float phi = e[i][x] / rho + 0.5f * ident + 0.5f / (float)kHcStreams;
            comb_row[i + x * kHcStreams] = ident + gamma * (phi - ident);
        }
}

__global__ void __launch_bounds__(256) hc_read_kernel(
    const float *resid, const __half *fn, const float *base, const float *scale, float *branch,
    float *post_out, float *comb_out, int h, float eps, float magnitude) {
    __shared__ float red[kHcMixes + 1][8];
    __shared__ float s_pre[kHcStreams];
    const int t = blockIdx.x;
    const int width = kHcStreams * h;
    const float *inp = resid + (size_t)t * width;
    float dots[kHcMixes];
#pragma unroll
    for (int m = 0; m < kHcMixes; ++m) dots[m] = 0.0f;
    float ss = 0.0f;
    for (int r = threadIdx.x * 2; r < width; r += blockDim.x * 2) {
        const float2 v = *reinterpret_cast<const float2 *>(inp + r);
        ss += v.x * v.x + v.y * v.y;
#pragma unroll
        for (int m = 0; m < kHcMixes; ++m) {
            const float2 f =
                __half22float2(*reinterpret_cast<const __half2 *>(fn + (size_t)m * width + r));
            dots[m] += f.x * v.x + f.y * v.y;
        }
    }
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    ss = warp_sum(ss);
#pragma unroll
    for (int m = 0; m < kHcMixes; ++m) dots[m] = warp_sum(dots[m]);
    if (lane == 0) {
        red[kHcMixes][warp] = ss;
#pragma unroll
        for (int m = 0; m < kHcMixes; ++m) red[m][warp] = dots[m];
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        float tot[kHcMixes + 1];
        for (int m = 0; m <= kHcMixes; ++m) {
            float s = 0.0f;
            for (int w = 0; w < 8; ++w) s += red[m][w];
            tot[m] = s;
        }
        hc_read_solve(tot, base, scale, width, eps, magnitude, s_pre,
                      post_out + (size_t)t * kHcStreams, comb_out + (size_t)t * 16);
    }
    __syncthreads();
    for (int k = threadIdx.x; k < h; k += blockDim.x) {
        float acc = 0.0f;
        for (int s = 0; s < kHcStreams; ++s) acc += inp[s * h + k] * s_pre[s];
        branch[(size_t)t * h + k] = acc;
    }
}

/* Split pass 1 for small n (1..4): one block per (mix row or norm, token), exact same
 * reduction order as hc_read_kernel. Stages the 18 totals into comb_out[0..16] and
 * post_out[0..2] before hc_read_finish_kernel overwrites them. */
__global__ void __launch_bounds__(256) hc_read_dots_kernel(
    const float *resid, const __half *fn, float *post_out, float *comb_out, int h) {
    __shared__ float red[8];
    const int m = blockIdx.x, t = blockIdx.y;
    const int width = kHcStreams * h;
    const float *inp = resid + (size_t)t * width;
    float acc = 0.0f;
    if (m < kHcMixes) {
        const __half *row = fn + (size_t)m * width;
        for (int r = threadIdx.x * 2; r < width; r += blockDim.x * 2) {
            const float2 v = *reinterpret_cast<const float2 *>(inp + r);
            const float2 f = __half22float2(*reinterpret_cast<const __half2 *>(row + r));
            acc += f.x * v.x + f.y * v.y;
        }
    } else {
        for (int r = threadIdx.x * 2; r < width; r += blockDim.x * 2) {
            const float2 v = *reinterpret_cast<const float2 *>(inp + r);
            acc += v.x * v.x + v.y * v.y;
        }
    }
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    acc = warp_sum(acc);
    if (lane == 0) red[warp] = acc;
    __syncthreads();
    if (threadIdx.x == 0) {
        float s = 0.0f;
        for (int w = 0; w < 8; ++w) s += red[w];
        if (m < 16)
            comb_out[(size_t)t * 16 + m] = s;
        else
            post_out[(size_t)t * kHcStreams + (m - 16)] = s;
    }
}

__global__ void __launch_bounds__(256) hc_read_finish_kernel(
    const float *resid, const float *base, const float *scale, float *branch, float *post_out,
    float *comb_out, int h, float eps, float magnitude) {
    __shared__ float s_pre[kHcStreams];
    const int t = blockIdx.x;
    const int width = kHcStreams * h;
    const float *inp = resid + (size_t)t * width;
    if (threadIdx.x == 0) {
        float tot[kHcMixes + 1];
        float *comb_row = comb_out + (size_t)t * 16;
        float *post_row = post_out + (size_t)t * kHcStreams;
        for (int m = 0; m < 16; ++m) tot[m] = comb_row[m];
        tot[16] = post_row[0];
        tot[17] = post_row[1];
        hc_read_solve(tot, base, scale, width, eps, magnitude, s_pre, post_row, comb_row);
    }
    __syncthreads();
    for (int k = threadIdx.x; k < h; k += blockDim.x) {
        float acc = 0.0f;
        for (int s = 0; s < kHcStreams; ++s) acc += inp[s * h + k] * s_pre[s];
        branch[(size_t)t * h + k] = acc;
    }
}

__global__ void hc_mean_kernel(const float *resid, float *out, int row, int h) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= h) return;
    const float *inp = resid + (size_t)row * kHcStreams * h;
    float acc = 0.0f;
    for (int s = 0; s < kHcStreams; ++s) acc += inp[s * h + k];
    out[k] = acc / (float)kHcStreams;
}

__global__ void hc_broadcast_kernel(const unsigned *src, unsigned *dst, int streams, int h) {
    const int t = blockIdx.x, s = blockIdx.y;
    const unsigned *in = src + (size_t)t * h;
    unsigned *out = dst + ((size_t)t * streams + s) * h;
    for (int k = threadIdx.x; k < h; k += blockDim.x) out[k] = in[k];
}

/* ---- quantize -------------------------------------------------------------------------- */

__global__ void __launch_bounds__(256) gated_quant_kernel(const float *x, const float *g,
                                                          unsigned char *q8, float *y, int w) {
    const int t = blockIdx.x;
    unsigned char *qrow = q8 == NULL ? NULL : q8 + (size_t)t * q8_row_bytes(w);
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, warps = blockDim.x >> 5;
    for (int b = warp; b < w / 32; b += warps) {
        const size_t at = (size_t)t * w + (size_t)b * 32 + lane;
        float v = x[at];
        if (g != NULL) v *= sigmoid_f(g[at]);
        if (y != NULL) y[at] = v;
        if (qrow != NULL) store_q8_1_block(reinterpret_cast<block_q8_1 *>(qrow + (size_t)b * kQ8Block), v);
    }
    if (qrow != NULL) zero_q8_tail(qrow, w);
}

__global__ void __launch_bounds__(256) situ_quant_kernel(const float *gate, const float *up,
                                                         unsigned char *q8, float *y, int w,
                                                         int ld_gate, int ld_up, float beta,
                                                         float lb) {
    const int t = blockIdx.x;
    unsigned char *qrow = q8 == NULL ? NULL : q8 + (size_t)t * q8_row_bytes(w);
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, warps = blockDim.x >> 5;
    for (int b = warp; b < w / 32; b += warps) {
        const int i = b * 32 + lane;
        const float g = gate[(size_t)t * ld_gate + i];
        const float u = up[(size_t)t * ld_up + i];
        const float act = tanhf(g / beta) * beta * sigmoid_f(g);
        const float upq = lb > 0.0f ? tanhf(u / lb) * lb : u;
        if (y != NULL) y[(size_t)t * w + i] = act * upq;
        if (qrow != NULL) store_q8_1_block(reinterpret_cast<block_q8_1 *>(qrow + (size_t)b * kQ8Block), act * upq);
    }
    if (qrow != NULL) zero_q8_tail(qrow, w);
}

__global__ void fill_kernel(float *p, int count, float value) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) p[i] = value;
}

/* ---- inputs of the shared delta-rule scan ------------------------------------------ */

/* The shared sequential scan (prefill/gdn_scan_seq, CHANNEL_DECAY form) reads the conv rows as
 * [q | k | v] of width 3 * w, a projection row [z | beta logits] and one decay value per key
 * channel. One block per token finishes those from the layer's rows: v of the conv row scaled in
 * place by the per-head fold of its beta and delta logits, the decay exp(lower_bound *
 * sigmoid(-(g1 + dt_bias) * a)) of every channel, z copied from the output-gate rows and the beta
 * logits. */
__global__ void __launch_bounds__(256) rda_pack_kernel(
    const float *g1, const float *dt_bias, const float *a, const float *beta, const float *delta,
    const float *g2, float *qkv, float *decay, float *proj, int heads, float lower_bound) {
    __shared__ float fold[256];
    const int t = blockIdx.x;
    const int w = heads * kRdaDim;
    const size_t row = (size_t)t * w;
    const size_t proj_row = (size_t)t * (w + heads);
    if (threadIdx.x < heads) {
        const float bi = beta[(size_t)t * heads + threadIdx.x];
        const float de = bi + delta[(size_t)t * heads + threadIdx.x];
        fold[threadIdx.x] = expf(softplus_f(-bi) - softplus_f(-de));
        proj[proj_row + w + threadIdx.x] = bi;
    }
    __syncthreads();
    float *v = qkv + (size_t)t * 3 * w + 2 * w;
    for (int i = threadIdx.x; i < w; i += blockDim.x) {
        const int h = i / kRdaDim;
        v[i] = v[i] * fold[h];
        const float raw = (g1[row + i] + dt_bias[i]) * a[h];
        decay[row + i] = expf(lower_bound * sigmoid_f(-raw));
        proj[proj_row + i] = g2[row + i];
    }
}

/* ---- absorbed latent attention ------------------------------------------------------ */

/* qeff[t][h] = [ wk_b[h] . q_nope | q_rope ]: one block per (token, head), one warp per
 * latent row. */
__global__ void __launch_bounds__(128) mla_q_absorb_kernel(const float *qb, const float *wk_b,
                                                           float *qeff, int heads, int nope,
                                                           int rope, int rank) {
    extern __shared__ float qn[];
    const int t = blockIdx.x, h = blockIdx.y;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const float *qrow = qb + ((size_t)t * heads + h) * (nope + rope);
    for (int i = threadIdx.x; i < nope; i += blockDim.x) qn[i] = qrow[i];
    float *dst = qeff + ((size_t)t * heads + h) * (rank + rope);
    for (int i = threadIdx.x; i < rope; i += blockDim.x) dst[rank + i] = qrow[nope + i];
    __syncthreads();
    const float *wh = wk_b + (size_t)h * rank * nope;
    for (int c = warp; c < rank; c += blockDim.x >> 5) {
        const float4 wv = *reinterpret_cast<const float4 *>(wh + (size_t)c * nope + lane * 4);
        const float4 qv = *reinterpret_cast<const float4 *>(qn + lane * 4);
        float acc = wv.x * qv.x + wv.y * qv.y + wv.z * qv.z + wv.w * qv.w;
        acc = warp_sum(acc);
        if (lane == 0) dst[c] = acc;
    }
}

/* cache[pos0 + t] = [ rms_norm(kv[:rank]) * w | kv[rank:] ] as f16. */
__global__ void __launch_bounds__(256) mla_kv_prep_kernel(const float *kv, const float *w,
                                                          __half *cache, int rank, int rope,
                                                          int pos0, float eps) {
    __shared__ float scratch[32];
    const int t = blockIdx.x;
    const float *row = kv + (size_t)t * (rank + rope);
    float ss = 0.0f;
    for (int i = threadIdx.x; i < rank; i += blockDim.x) ss += row[i] * row[i];
    ss = block_sum(ss, scratch);
    const float scale = 1.0f / sqrtf(ss / (float)rank + eps);
    __half *dst = cache + (size_t)(pos0 + t) * (rank + rope);
    for (int i = threadIdx.x; i < rank; i += blockDim.x) dst[i] = __float2half(row[i] * scale * w[i]);
    for (int i = threadIdx.x; i < rope; i += blockDim.x) dst[rank + i] = __float2half(row[rank + i]);
}

/* Single-row MLA KV prep reading pos from device memory for graph-captured decode steps. */
__global__ void __launch_bounds__(256) mla_kv_prep_step_kernel(const float *kv, const float *w,
                                                               __half *cache, const int *pos_ptr,
                                                               int rank, int rope, int cells,
                                                               float eps) {
    __shared__ float scratch[32];
    const int pos = *pos_ptr;
    if (pos < 0 || pos >= cells) return;
    float ss = 0.0f;
    for (int i = threadIdx.x; i < rank; i += blockDim.x) ss += kv[i] * kv[i];
    ss = block_sum(ss, scratch);
    const float scale = 1.0f / sqrtf(ss / (float)rank + eps);
    __half *dst = cache + (size_t)pos * (rank + rope);
    for (int i = threadIdx.x; i < rank; i += blockDim.x) dst[i] = __float2half(kv[i] * scale * w[i]);
    for (int i = threadIdx.x; i < rope; i += blockDim.x) dst[rank + i] = __float2half(kv[rank + i]);
}

/* Fallback single-row decode latent attention for architectures older than sm_80 or without the
 * tensor-core kernel's shared-memory budget: same layout and math as mla_attn_kernel, reading pos
 * from pos_ptr when non-null. */
__global__ void __launch_bounds__(256)
mla_attn_decode_fallback_kernel(const float *qeff, const __half *cache, float *part,
                                const int *pos_ptr, int splits, int pos_host, int cells,
                                float scale) {
    extern __shared__ unsigned char smem_raw[];
    float *qs = reinterpret_cast<float *>(smem_raw);
    __half *kv = reinterpret_cast<__half *>(qs + kMlaGroup * kMlaWidth);
    float *pbuf = reinterpret_cast<float *>(kv + kMlaTile * kMlaRowHalves);
    const int g = blockIdx.x, sp = blockIdx.y;
    const int pos = pos_ptr != NULL ? *pos_ptr : pos_host;
    const int visible = (pos >= 0 && pos < cells) ? (pos + 1) : 0;
    const int per_split =
        visible > 0
            ? (((visible + splits - 1) / splits + kMlaTile - 1) / kMlaTile * kMlaTile)
            : 0;
    const int lo = sp * per_split;
    const int hi = min(lo + per_split, visible);
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const float *qsrc = qeff + (size_t)g * kMlaGroup * kMlaWidth;
    if (lo < hi) {
        for (int i = tid; i < kMlaGroup * kMlaWidth; i += 256) qs[i] = qsrc[i];
    }
    float m = -INFINITY, l = 0.0f;
    float acc[kMlaRank / 32];
#pragma unroll
    for (int i = 0; i < kMlaRank / 32; ++i) acc[i] = 0.0f;
    for (int base = lo; base < hi; base += kMlaTile) {
        const int cnt = min(kMlaTile, hi - base);
        __syncthreads();
        for (int idx = tid; idx < cnt * (kMlaWidth / 2); idx += 256) {
            const int p = idx / (kMlaWidth / 2), wd = idx % (kMlaWidth / 2);
            const unsigned v =
                reinterpret_cast<const unsigned *>(cache + (size_t)(base + p) * kMlaWidth)[wd];
            reinterpret_cast<unsigned *>(kv + p * kMlaRowHalves)[wd] = v;
        }
        __syncthreads();
        float sc = -INFINITY;
        if (lane < cnt) {
            const float *qh = qs + warp * kMlaWidth;
            const __half2 *kr = reinterpret_cast<const __half2 *>(kv + lane * kMlaRowHalves);
            float a = 0.0f;
#pragma unroll 4
            for (int d2 = 0; d2 < kMlaWidth / 2; ++d2) {
                const float2 kf = __half22float2(kr[d2]);
                a += qh[2 * d2] * kf.x + qh[2 * d2 + 1] * kf.y;
            }
            sc = a * scale;
        }
        const float mt = warp_max(sc);
        const float mnew = fmaxf(m, mt);
        const float corr = m == -INFINITY ? 0.0f : expf(m - mnew);
        const float p = lane < cnt ? expf(sc - mnew) : 0.0f;
        l = l * corr + warp_sum(p);
        m = mnew;
        pbuf[warp * kMlaTile + lane] = p;
#pragma unroll
        for (int i = 0; i < kMlaRank / 32; ++i) acc[i] *= corr;
        __syncwarp();
        for (int pp = 0; pp < cnt; ++pp) {
            const float pw = pbuf[warp * kMlaTile + pp];
            const __half *lat = kv + pp * kMlaRowHalves;
#pragma unroll
            for (int i = 0; i < kMlaRank / 32; ++i) acc[i] += pw * __half2float(lat[lane + 32 * i]);
        }
    }
    float *dst = part + (((size_t)g * splits + sp) * kMlaGroup + warp) * kMlaPartStride;
#pragma unroll
    for (int i = 0; i < kMlaRank / 32; ++i) dst[lane + 32 * i] = acc[i];
    if (lane == 0) {
        dst[kMlaRank] = m;
        dst[kMlaRank + 1] = l;
    }
}

/* Partial softmax attention of one (token, head group, key split). Writes the unnormalized
 * accumulator, the running max and the running sum per head. */
__global__ void __launch_bounds__(256) mla_attn_kernel(const float *qeff, const __half *cache,
                                                       float *part, int heads, int pos0,
                                                       int splits, int keys_per_split,
                                                       float scale) {
    extern __shared__ unsigned char smem_raw[];
    float *qs = reinterpret_cast<float *>(smem_raw);                 /* [8][576] */
    __half *kv = reinterpret_cast<__half *>(qs + kMlaGroup * kMlaWidth); /* [32][578] */
    float *pbuf = reinterpret_cast<float *>(kv + kMlaTile * kMlaRowHalves); /* [8][32] */
    const int t = blockIdx.x, g = blockIdx.y, sp = blockIdx.z;
    const int groups = heads / kMlaGroup;
    const int visible = pos0 + t + 1;
    const int lo = sp * keys_per_split;
    const int hi = min(lo + keys_per_split, visible);
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const float *qsrc = qeff + ((size_t)t * heads + (size_t)g * kMlaGroup) * kMlaWidth;
    for (int i = tid; i < kMlaGroup * kMlaWidth; i += 256) qs[i] = qsrc[i];
    float m = -INFINITY, l = 0.0f;
    float acc[kMlaRank / 32];
#pragma unroll
    for (int i = 0; i < kMlaRank / 32; ++i) acc[i] = 0.0f;
    for (int base = lo; base < hi; base += kMlaTile) {
        const int cnt = min(kMlaTile, hi - base);
        __syncthreads();
        for (int idx = tid; idx < cnt * (kMlaWidth / 2); idx += 256) {
            const int p = idx / (kMlaWidth / 2), wd = idx % (kMlaWidth / 2);
            const unsigned v = reinterpret_cast<const unsigned *>(cache + (size_t)(base + p) * kMlaWidth)[wd];
            reinterpret_cast<unsigned *>(kv + p * kMlaRowHalves)[wd] = v;
        }
        __syncthreads();
        float sc = -INFINITY;
        if (lane < cnt) {
            const float *qh = qs + warp * kMlaWidth;
            const __half2 *kr = reinterpret_cast<const __half2 *>(kv + lane * kMlaRowHalves);
            float a = 0.0f;
#pragma unroll 4
            for (int d2 = 0; d2 < kMlaWidth / 2; ++d2) {
                const float2 kf = __half22float2(kr[d2]);
                a += qh[2 * d2] * kf.x + qh[2 * d2 + 1] * kf.y;
            }
            sc = a * scale;
        }
        const float mt = warp_max(sc);
        const float mnew = fmaxf(m, mt);
        const float corr = m == -INFINITY ? 0.0f : expf(m - mnew);
        const float p = lane < cnt ? expf(sc - mnew) : 0.0f;
        l = l * corr + warp_sum(p);
        m = mnew;
        pbuf[warp * kMlaTile + lane] = p;
#pragma unroll
        for (int i = 0; i < kMlaRank / 32; ++i) acc[i] *= corr;
        __syncwarp();
        for (int pp = 0; pp < cnt; ++pp) {
            const float pw = pbuf[warp * kMlaTile + pp];
            const __half *lat = kv + pp * kMlaRowHalves;
#pragma unroll
            for (int i = 0; i < kMlaRank / 32; ++i) acc[i] += pw * __half2float(lat[lane + 32 * i]);
        }
    }
    float *dst = part + ((((size_t)t * groups + g) * splits + sp) * kMlaGroup + warp) * kMlaPartStride;
#pragma unroll
    for (int i = 0; i < kMlaRank / 32; ++i) dst[lane + 32 * i] = acc[i];
    if (lane == 0) {
        dst[kMlaRank] = m;
        dst[kMlaRank + 1] = l;
    }
}

#include "resident/mla_attn_chunk.cuh"

__global__ void __launch_bounds__(128) mla_reduce_kernel(const float *part, float *mix, int heads,
                                                         int splits) {
    const int t = blockIdx.x, h = blockIdx.y;
    const int groups = heads / kMlaGroup;
    const int g = h / kMlaGroup, hh = h % kMlaGroup;
    const float *base = part + (((size_t)t * groups + g) * splits) * kMlaGroup * kMlaPartStride +
                        (size_t)hh * kMlaPartStride;
    const size_t split_stride = (size_t)kMlaGroup * kMlaPartStride;
    float mx = -INFINITY;
    for (int s = 0; s < splits; ++s) mx = fmaxf(mx, base[s * split_stride + kMlaRank]);
    float total = 0.0f;
    for (int s = 0; s < splits; ++s) {
        const float ms = base[s * split_stride + kMlaRank];
        const float ls = base[s * split_stride + kMlaRank + 1];
        if (ms != -INFINITY || isnan(ls)) total += ls * expf(ms - mx);
    }
    for (int c = threadIdx.x; c < kMlaRank; c += blockDim.x) {
        float acc = 0.0f;
        for (int s = 0; s < splits; ++s) {
            const float ms = base[s * split_stride + kMlaRank];
            const float ls = base[s * split_stride + kMlaRank + 1];
            if (ms != -INFINITY || isnan(ls)) acc += base[s * split_stride + c] * expf(ms - mx);
        }
        mix[((size_t)t * heads + h) * kMlaRank + c] = acc / total;
    }
}

/* out[t][h][v] = wv_b[h][v] . mix[t][h]: one block per (token, head), one warp per row. */
__global__ void __launch_bounds__(128) mla_project_kernel(const float *mix, const float *wv_b,
                                                          float *out, int heads, int rank, int v) {
    extern __shared__ float mx[];
    const int t = blockIdx.x, h = blockIdx.y;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const float *m = mix + ((size_t)t * heads + h) * rank;
    for (int i = threadIdx.x; i < rank; i += blockDim.x) mx[i] = m[i];
    __syncthreads();
    const float *wh = wv_b + (size_t)h * v * rank;
    for (int r = warp; r < v; r += blockDim.x >> 5) {
        float acc = 0.0f;
        for (int c = lane * 4; c < rank; c += 128) {
            const float4 wv = *reinterpret_cast<const float4 *>(wh + (size_t)r * rank + c);
            acc += wv.x * mx[c] + wv.y * mx[c + 1] + wv.z * mx[c + 2] + wv.w * mx[c + 3];
        }
        acc = warp_sum(acc);
        if (lane == 0) out[((size_t)t * heads + h) * v + r] = acc;
    }
}

/* ---- entry ------------------------------------------------------------------------- */

bool aligned(const void *p, size_t a) { return (uintptr_t)p % a == 0; }

bool apart(const void *a, size_t ab, const void *b, size_t bb) {
    const uintptr_t x = (uintptr_t)a, y = (uintptr_t)b;
    return x <= y ? ab <= y - x : bb <= x - y;
}

struct Slots {
    const inference_cuda_block_ops_args *args;
    bool ok = true;
    bool written[INFERENCE_CUDA_BLOCK_OPS_PTRS] = {};
    bool used[INFERENCE_CUDA_BLOCK_OPS_PTRS] = {};
    size_t need[INFERENCE_CUDA_BLOCK_OPS_PTRS] = {};

    explicit Slots(const inference_cuda_block_ops_args *a) : args(a) {}

    /* A required slot of `bytes` bytes. */
    void use(int slot, long long bytes, bool write) {
        if (bytes <= 0 || args->ptr[slot] == NULL || args->bytes[slot] < (size_t)bytes ||
            !aligned(args->ptr[slot], 4)) {
            ok = false;
            return;
        }
        used[slot] = true;
        written[slot] = write;
        need[slot] = (size_t)bytes;
    }

    /* A slot that may be absent. */
    void maybe(int slot, long long bytes, bool write) {
        if (args->ptr[slot] != NULL) use(slot, bytes, write);
    }

    /* Slots that are not used must be NULL; written slots share no byte with another. */
    bool valid() const {
        if (!ok) return false;
        for (int i = 0; i < INFERENCE_CUDA_BLOCK_OPS_PTRS; ++i) {
            if (!used[i]) {
                if (args->ptr[i] != NULL) return false;
                continue;
            }
            if (!written[i]) continue;
            for (int j = 0; j < INFERENCE_CUDA_BLOCK_OPS_PTRS; ++j)
                if (j != i && used[j] && !apart(args->ptr[i], need[i], args->ptr[j], need[j]))
                    return false;
        }
        return true;
    }
};

int launch_status(int *out_stage) {
    Failure failure;
    failure.record(cudaGetLastError(), INFERENCE_CUDA_STAGE_LAUNCH);
    report_stage(out_stage, failure.stage);
    return (int)failure.err;
}

}  // namespace

extern "C" int inference_cuda_block_ops_launch(const inference_cuda_block_ops_args *args,
                                               void *stream, int *out_stage) {
    try {
        report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
        if (args == NULL || stream == NULL) return INFERENCE_CUDA_INVALID_ARGUMENT;
        const cudaStream_t s = (cudaStream_t)stream;
        const int *i = args->ints;
        const float *f = args->floats;
        void *const *p = args->ptr;
        Slots slots(args);
        switch (args->op) {
        case INFERENCE_CUDA_BLOCK_OP_HC_READ: {
            const long long n = i[0], h = i[1], w = 4LL * h;
            if (n < 1 || n > 65536 || h < 64 || h > 8192 || h % 2 != 0)
                return INFERENCE_CUDA_INVALID_ARGUMENT;
            slots.use(0, n * w * 4, false);
            slots.use(1, 17 * w * 2, false);
            slots.use(2, 17 * 4, false);
            slots.use(3, 4 * 4, false);
            slots.use(4, n * h * 4, true);
            slots.use(5, n * 4 * 4, true);
            slots.use(6, n * 16 * 4, true);
            if (!slots.valid() || !aligned(p[0], 8) || !aligned(p[1], 4))
                return INFERENCE_CUDA_INVALID_ARGUMENT;
            if (n <= 4) {
                hc_read_dots_kernel<<<dim3(kHcMixes + 1, (unsigned)n), 256, 0, s>>>(
                    (const float *)p[0], (const __half *)p[1], (float *)p[5], (float *)p[6],
                    (int)h);
                hc_read_finish_kernel<<<(unsigned)n, 256, 0, s>>>(
                    (const float *)p[0], (const float *)p[2], (const float *)p[3], (float *)p[4],
                    (float *)p[5], (float *)p[6], (int)h, f[0], f[1]);
            } else {
                hc_read_kernel<<<(unsigned)n, 256, 0, s>>>(
                    (const float *)p[0], (const __half *)p[1], (const float *)p[2],
                    (const float *)p[3], (float *)p[4], (float *)p[5], (float *)p[6], (int)h, f[0],
                    f[1]);
            }
            return launch_status(out_stage);
        }
        case INFERENCE_CUDA_BLOCK_OP_HC_MEAN: {
            const long long row = i[0], h = i[1];
            if (row < 0 || h < 32 || h > 8192) return INFERENCE_CUDA_INVALID_ARGUMENT;
            slots.use(0, (row + 1) * 4 * h * 4, false);
            slots.use(1, h * 4, true);
            if (!slots.valid()) return INFERENCE_CUDA_INVALID_ARGUMENT;
            hc_mean_kernel<<<(unsigned)((h + 255) / 256), 256, 0, s>>>(
                (const float *)p[0], (float *)p[1], (int)row, (int)h);
            return launch_status(out_stage);
        }
        case INFERENCE_CUDA_BLOCK_OP_HC_BROADCAST: {
            const long long n = i[0], streams = i[1], h = i[2];
            if (n < 1 || n > 65536 || streams < 1 || streams > 16 || h < 1 || h > 16384)
                return INFERENCE_CUDA_INVALID_ARGUMENT;
            slots.use(0, n * h * 4, false);
            slots.use(1, n * streams * h * 4, true);
            if (!slots.valid()) return INFERENCE_CUDA_INVALID_ARGUMENT;
            hc_broadcast_kernel<<<dim3((unsigned)n, (unsigned)streams), 256, 0, s>>>(
                (const unsigned *)p[0], (unsigned *)p[1], (int)streams, (int)h);
            return launch_status(out_stage);
        }
        case INFERENCE_CUDA_BLOCK_OP_RDA_PACK: {
            const long long n = i[0], heads = i[1], hd = i[2];
            if (n < 1 || n > 65536 || heads < 1 || heads > 256 || hd != kRdaDim)
                return INFERENCE_CUDA_INVALID_ARGUMENT;
            const long long w = heads * hd;
            slots.use(0, n * w * 4, false);
            slots.use(1, w * 4, false);
            slots.use(2, heads * 4, false);
            slots.use(3, n * heads * 4, false);
            slots.use(4, n * heads * 4, false);
            slots.use(5, n * w * 4, false);
            slots.use(6, n * 3 * w * 4, true);
            slots.use(7, n * w * 4, true);
            slots.use(8, n * (w + heads) * 4, true);
            if (!slots.valid()) return INFERENCE_CUDA_INVALID_ARGUMENT;
            rda_pack_kernel<<<(unsigned)n, 256, 0, s>>>(
                (const float *)p[0], (const float *)p[1], (const float *)p[2], (const float *)p[3],
                (const float *)p[4], (const float *)p[5], (float *)p[6], (float *)p[7],
                (float *)p[8], (int)heads, f[0]);
            return launch_status(out_stage);
        }
        case INFERENCE_CUDA_BLOCK_OP_MLA_Q_ABSORB: {
            const long long n = i[0], heads = i[1], nope = i[2], rope = i[3], rank = i[4];
            if (n < 1 || n > 65536 || heads < 1 || heads > 256 || nope != 128 || rope != kMlaRope ||
                rank != kMlaRank)
                return INFERENCE_CUDA_INVALID_ARGUMENT;
            slots.use(0, n * heads * (nope + rope) * 4, false);
            slots.use(1, heads * rank * nope * 4, false);
            slots.use(2, n * heads * (rank + rope) * 4, true);
            if (!slots.valid() || !aligned(p[1], 16)) return INFERENCE_CUDA_INVALID_ARGUMENT;
            mla_q_absorb_kernel<<<dim3((unsigned)n, (unsigned)heads), 128, (size_t)nope * 4, s>>>(
                (const float *)p[0], (const float *)p[1], (float *)p[2], (int)heads, (int)nope,
                (int)rope, (int)rank);
            return launch_status(out_stage);
        }
        case INFERENCE_CUDA_BLOCK_OP_MLA_KV_PREP: {
            const long long n = i[0], rank = i[1], rope = i[2], pos0 = i[3], cells = i[4];
            if (n < 1 || n > 65536 || rank != kMlaRank || rope != kMlaRope || pos0 < 0 ||
                pos0 + n > cells)
                return INFERENCE_CUDA_INVALID_ARGUMENT;
            slots.use(0, n * (rank + rope) * 4, false);
            slots.use(1, rank * 4, false);
            slots.use(2, cells * (rank + rope) * 2, true);
            if (!slots.valid()) return INFERENCE_CUDA_INVALID_ARGUMENT;
            mla_kv_prep_kernel<<<(unsigned)n, 256, 0, s>>>((const float *)p[0], (const float *)p[1],
                                                           (__half *)p[2], (int)rank, (int)rope,
                                                           (int)pos0, f[0]);
            return launch_status(out_stage);
        }
        case INFERENCE_CUDA_BLOCK_OP_MLA_KV_PREP_STEP: {
            const long long rank = i[0], rope = i[1], cells = i[2];
            if (rank != kMlaRank || rope != kMlaRope || cells < 1)
                return INFERENCE_CUDA_INVALID_ARGUMENT;
            slots.use(0, (rank + rope) * 4, false);
            slots.use(1, rank * 4, false);
            slots.use(2, cells * (rank + rope) * 2, true);
            slots.use(3, 4, false);
            if (!slots.valid()) return INFERENCE_CUDA_INVALID_ARGUMENT;
            mla_kv_prep_step_kernel<<<1, 256, 0, s>>>((const float *)p[0], (const float *)p[1],
                                                      (__half *)p[2], (const int *)p[3], (int)rank,
                                                      (int)rope, (int)cells, f[0]);
            return launch_status(out_stage);
        }
        case INFERENCE_CUDA_BLOCK_OP_MLA_ATTN_DECODE: {
            const long long heads = i[0], rank = i[1], rope = i[2], splits = i[3];
            const long long pos = i[4], cells = i[5];
            if (heads < kMlaGroup || heads > 256 || heads % kMlaGroup != 0 || rank != kMlaRank ||
                rope != kMlaRope || splits < 1 || splits > 64 || cells < 1 ||
                (p[3] == NULL && (pos < 0 || pos >= cells)))
                return INFERENCE_CUDA_INVALID_ARGUMENT;
            slots.use(0, heads * (rank + rope) * 4, false);
            slots.use(1, cells * (rank + rope) * 2, false);
            slots.use(2, (heads / kMlaGroup) * splits * kMlaGroup * kMlaPartStride * 4, true);
            slots.maybe(3, 4, false);
            if (!slots.valid() || !aligned(p[0], 8) || !aligned(p[1], 16) || !aligned(p[2], 8))
                return INFERENCE_CUDA_INVALID_ARGUMENT;
            int device = 0;
            if (cudaGetDevice(&device) != cudaSuccess || device < 0 || device >= 16) device = 0;
            static int arm[16] = {};
            if (arm[device] == 0) {
                Failure failure;
                cudaFuncAttributes attr = {};
                int limit = 0;
                failure.record(cudaFuncGetAttributes(&attr, mla_attn_decode_tc_kernel),
                               INFERENCE_CUDA_STAGE_LAUNCH);
                failure.record(cudaDeviceGetAttribute(
                                   &limit, cudaDevAttrMaxSharedMemoryPerBlockOptin, device),
                               INFERENCE_CUDA_STAGE_LAUNCH);
                const bool fits =
                    failure.ok() && attr.ptxVersion >= 80 && limit >= kMlaChunkSmem;
                if (fits)
                    failure.record(
                        cudaFuncSetAttribute(mla_attn_decode_tc_kernel,
                                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                                             kMlaChunkSmem),
                        INFERENCE_CUDA_STAGE_LAUNCH);
                report_stage(out_stage, failure.stage);
                if (!failure.ok()) return (int)failure.err;
                arm[device] = fits ? 1 : 2;
            }
            if (arm[device] == 1) {
                const unsigned head_tiles = (unsigned)((heads + 63) / 64);
                mla_attn_decode_tc_kernel<<<dim3(head_tiles, (unsigned)splits), kMlaChunkThreads,
                                            kMlaChunkSmem, s>>>(
                    (const float *)p[0], (const __half *)p[1], (float *)p[2], (const int *)p[3],
                    (int)heads, (int)splits, (int)pos, (int)cells, f[0]);
                return launch_status(out_stage);
            }
            const size_t smem = (size_t)kMlaGroup * kMlaWidth * 4 +
                                (size_t)kMlaTile * kMlaRowHalves * 2 +
                                (size_t)kMlaGroup * kMlaTile * 4;
            static bool fb_configured[16] = {};
            if (!fb_configured[device]) {
                Failure failure;
                failure.record(cudaFuncSetAttribute(mla_attn_decode_fallback_kernel,
                                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                    (int)smem),
                               INFERENCE_CUDA_STAGE_LAUNCH);
                report_stage(out_stage, failure.stage);
                if (!failure.ok()) return (int)failure.err;
                fb_configured[device] = true;
            }
            mla_attn_decode_fallback_kernel<<<dim3((unsigned)(heads / kMlaGroup), (unsigned)splits),
                                              256, smem, s>>>(
                (const float *)p[0], (const __half *)p[1], (float *)p[2], (const int *)p[3],
                (int)splits, (int)pos, (int)cells, f[0]);
            return launch_status(out_stage);
        }
        case INFERENCE_CUDA_BLOCK_OP_MLA_ATTN:
        case INFERENCE_CUDA_BLOCK_OP_MLA_ATTN_CHUNK: {
            const bool chunk = args->op == INFERENCE_CUDA_BLOCK_OP_MLA_ATTN_CHUNK;
            const long long n = i[0], heads = i[1], rank = i[2], rope = i[3], pos0 = i[4];
            const long long splits = i[5], kps = i[6];
            if (n < 1 || n > 65536 || heads < kMlaGroup || heads > 256 || heads % kMlaGroup != 0 ||
                rank != kMlaRank || rope != kMlaRope || pos0 < 0 || splits < 1 || splits > 64 ||
                kps < 32 || kps % 32 != 0 || splits * kps < pos0 + n)
                return INFERENCE_CUDA_INVALID_ARGUMENT;
            slots.use(0, n * heads * (rank + rope) * 4, false);
            slots.use(1, (pos0 + n) * (rank + rope) * 2, false);
            slots.use(2, n * (heads / kMlaGroup) * splits * kMlaGroup * kMlaPartStride * 4, true);
            if (!slots.valid() || !aligned(p[1], 4)) return INFERENCE_CUDA_INVALID_ARGUMENT;
            // The chunk kernel reads float pairs and copies cached rows 16 bytes at a time.
            if (chunk && (!aligned(p[0], 8) || !aligned(p[1], 16) || !aligned(p[2], 8)))
                return INFERENCE_CUDA_INVALID_ARGUMENT;
            int device = 0;
            if (cudaGetDevice(&device) != cudaSuccess || device < 0 || device >= 16) device = 0;
            if (chunk) {
                // Asked once per device: 1 the tensor-core kernel, 2 the row kernel. The loaded
                // image must be sm_80 or later and the device must grant the shared memory.
                static int arm[16] = {};
                if (arm[device] == 0) {
                    Failure failure;
                    cudaFuncAttributes attr = {};
                    int limit = 0;
                    failure.record(cudaFuncGetAttributes(&attr, mla_attn_chunk_kernel),
                                   INFERENCE_CUDA_STAGE_LAUNCH);
                    failure.record(cudaDeviceGetAttribute(
                                       &limit, cudaDevAttrMaxSharedMemoryPerBlockOptin, device),
                                   INFERENCE_CUDA_STAGE_LAUNCH);
                    const bool fits =
                        failure.ok() && attr.ptxVersion >= 80 && limit >= kMlaChunkSmem;
                    if (fits)
                        failure.record(
                            cudaFuncSetAttribute(mla_attn_chunk_kernel,
                                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                 kMlaChunkSmem),
                            INFERENCE_CUDA_STAGE_LAUNCH);
                    report_stage(out_stage, failure.stage);
                    if (!failure.ok()) return (int)failure.err;
                    arm[device] = fits ? 1 : 2;
                }
                if (arm[device] == 1) {
                    const unsigned tiles = (unsigned)((n + kMlaChunkTokens - 1) / kMlaChunkTokens);
                    const dim3 grid(tiles, (unsigned)(heads / kMlaGroup), (unsigned)splits);
                    mla_attn_chunk_kernel<<<grid, kMlaChunkThreads, kMlaChunkSmem, s>>>(
                        (const float *)p[0], (const __half *)p[1], (float *)p[2], (int)n,
                        (int)heads, (int)pos0, (int)splits, (int)kps, f[0]);
                    return launch_status(out_stage);
                }
            }
            const size_t smem = (size_t)kMlaGroup * kMlaWidth * 4 +
                                (size_t)kMlaTile * kMlaRowHalves * 2 +
                                (size_t)kMlaGroup * kMlaTile * 4;
            // The dynamic shared-memory limit is a per-device attribute of the function.
            static bool configured[16] = {};
            if (!configured[device]) {
                Failure failure;
                failure.record(cudaFuncSetAttribute(mla_attn_kernel,
                                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                    (int)smem),
                               INFERENCE_CUDA_STAGE_LAUNCH);
                report_stage(out_stage, failure.stage);
                if (!failure.ok()) return (int)failure.err;
                configured[device] = true;
            }
            mla_attn_kernel<<<dim3((unsigned)n, (unsigned)(heads / kMlaGroup), (unsigned)splits),
                              256, smem, s>>>((const float *)p[0], (const __half *)p[1],
                                              (float *)p[2], (int)heads, (int)pos0, (int)splits,
                                              (int)kps, f[0]);
            return launch_status(out_stage);
        }
        case INFERENCE_CUDA_BLOCK_OP_MLA_REDUCE: {
            const long long n = i[0], heads = i[1], rank = i[2], splits = i[3];
            if (n < 1 || n > 65536 || heads < kMlaGroup || heads > 256 || heads % kMlaGroup != 0 ||
                rank != kMlaRank || splits < 1 || splits > 64)
                return INFERENCE_CUDA_INVALID_ARGUMENT;
            slots.use(0, n * (heads / kMlaGroup) * splits * kMlaGroup * kMlaPartStride * 4, false);
            slots.use(1, n * heads * rank * 4, true);
            if (!slots.valid()) return INFERENCE_CUDA_INVALID_ARGUMENT;
            mla_reduce_kernel<<<dim3((unsigned)n, (unsigned)heads), 128, 0, s>>>(
                (const float *)p[0], (float *)p[1], (int)heads, (int)splits);
            return launch_status(out_stage);
        }
        case INFERENCE_CUDA_BLOCK_OP_MLA_PROJECT: {
            const long long n = i[0], heads = i[1], rank = i[2], v = i[3];
            if (n < 1 || n > 65536 || heads < 1 || heads > 256 || rank != kMlaRank || v < 1 ||
                v > 1024)
                return INFERENCE_CUDA_INVALID_ARGUMENT;
            slots.use(0, n * heads * rank * 4, false);
            slots.use(1, heads * v * rank * 4, false);
            slots.use(2, n * heads * v * 4, true);
            if (!slots.valid() || !aligned(p[1], 16)) return INFERENCE_CUDA_INVALID_ARGUMENT;
            mla_project_kernel<<<dim3((unsigned)n, (unsigned)heads), 128, (size_t)rank * 4, s>>>(
                (const float *)p[0], (const float *)p[1], (float *)p[2], (int)heads, (int)rank,
                (int)v);
            return launch_status(out_stage);
        }
        case INFERENCE_CUDA_BLOCK_OP_SITU_QUANT: {
            const long long rows = i[0], w = i[1], ldg = i[2], ldu = i[3];
            if (rows < 1 || rows > 1048576 || w < 32 || w > 32768 || w % 32 != 0 || ldg < w ||
                ldu < w)
                return INFERENCE_CUDA_INVALID_ARGUMENT;
            slots.use(0, ((rows - 1) * ldg + w) * 4, false);
            slots.use(1, ((rows - 1) * ldu + w) * 4, false);
            slots.maybe(2, rows * (long long)q8_row_bytes(w), true);
            slots.maybe(3, rows * w * 4, true);
            if (!slots.valid() || (p[2] == NULL && p[3] == NULL))
                return INFERENCE_CUDA_INVALID_ARGUMENT;
            situ_quant_kernel<<<(unsigned)rows, 256, 0, s>>>(
                (const float *)p[0], (const float *)p[1], (unsigned char *)p[2], (float *)p[3],
                (int)w, (int)ldg, (int)ldu, f[0], f[1]);
            return launch_status(out_stage);
        }
        case INFERENCE_CUDA_BLOCK_OP_GATED_QUANT: {
            const long long rows = i[0], w = i[1];
            if (rows < 1 || rows > 1048576 || w < 32 || w > 32768 || w % 32 != 0)
                return INFERENCE_CUDA_INVALID_ARGUMENT;
            slots.use(0, rows * w * 4, false);
            slots.maybe(1, rows * w * 4, false);
            slots.maybe(2, rows * (long long)q8_row_bytes(w), true);
            slots.maybe(3, rows * w * 4, true);
            if (!slots.valid() || (p[2] == NULL && p[3] == NULL))
                return INFERENCE_CUDA_INVALID_ARGUMENT;
            gated_quant_kernel<<<(unsigned)rows, 256, 0, s>>>(
                (const float *)p[0], (const float *)p[1], (unsigned char *)p[2], (float *)p[3],
                (int)w);
            return launch_status(out_stage);
        }
        case INFERENCE_CUDA_BLOCK_OP_FILL: {
            const long long count = i[0];
            if (count < 1 || count > (1LL << 30)) return INFERENCE_CUDA_INVALID_ARGUMENT;
            slots.use(0, count * 4, true);
            if (!slots.valid()) return INFERENCE_CUDA_INVALID_ARGUMENT;
            fill_kernel<<<(unsigned)((count + 255) / 256), 256, 0, s>>>((float *)p[0], (int)count,
                                                                       f[0]);
            return launch_status(out_stage);
        }
        default:
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
    } catch (...) {
        return INFERENCE_CUDA_NATIVE_EXCEPTION;
    }
}
