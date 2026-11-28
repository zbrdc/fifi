// The many-rows entries of the DeltaNet conv, the DeltaNet recurrence and the attention
// preparation of the resident decode step (contract: resident/multi_row.h). One launch advances n
// rows, each the single token of its own sequence; the state or the planes of a row come from a
// device table instead of one fixed address.
//
// Every kernel here is the single-token kernel of its entry with the row index in place of the
// token index and one extra indirection, written with the same expressions in the same order, so
// that row r of a launch of n rows carries the bytes of a launch of one row on row r's inputs. The
// device tests compare them bit for bit. The kv8 append and the attention decode entries live in
// kv8_append.cu and attn_decode_q8.cu, next to the quantizer and the split plan they share.
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>

#include "abi.h"
#include "device.h"
#include "resident/multi_row.h"

namespace {

constexpr int kStateDim = 128;  // DeltaNet head size (keys and values); kern::kStateDim
constexpr int kHeadDim = 256;   // full-attention head size; kern::kHeadDim
constexpr int kConvBlock = 256;
constexpr int kMaxChannels = 65536;
constexpr int kMaxLd = 1 << 20;
constexpr int kMaxKHeads = 64;
constexpr int kMaxVHeads = 128;
constexpr float kEps = 1e-6f;
constexpr int kMaxHeads = 256;
constexpr int kMaxKvHeads = 64;
constexpr int kMaxCells = (1 << 30) - 1;
constexpr int kStatusInvalidStepState = 4;

bool aligned(const void *p, size_t a) { return (uintptr_t)p % a == 0; }

bool rows_ok(int n) { return n >= 1 && n <= INFERENCE_CUDA_MULTI_ROW_MAX; }

template <bool ExactExp>
__device__ __forceinline__ float silu_f(float x) {
    if constexpr (ExactExp) {
        return x / (1.f + expf(-x));
    } else {
        return x / (1.f + __expf(-x));
    }
}
__device__ __forceinline__ float sigmoid_f(float x) { return 1.f / (1.f + __expf(-x)); }
__device__ __forceinline__ float softplus_f(float x) { return x > 20.f ? x : log1pf(__expf(x)); }
__device__ __forceinline__ float softplus_ieee_f(float x) { return x > 20.f ? x : log1pf(expf(x)); }

__device__ float warp_sum_f(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

// Sum of v over the block, returned to every thread (the vendored block_sum, line for line).
__device__ float block_sum_f(float v) {
    __shared__ float red[32];
    v = warp_sum_f(v);
    int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    __syncthreads();  // earlier callers may still be reading red
    if (lane == 0) red[warp] = v;
    __syncthreads();
    v = lane < (int)blockDim.x / 32 ? red[lane] : 0.f;
    return warp_sum_f(v);
}

// conv_silu of one token per row: the expression of conv_silu_rows_k (a rounded first product, then
// three fused multiply-adds in tap order), which the prefill tests pin to conv_silu_k bit for bit.
template <bool ExactExp>
__global__ void __launch_bounds__(kConvBlock)
conv_silu_multi_k(const float *__restrict__ proj, int ld, int n, const float4 *__restrict__ w,
                  float *const *states, float *__restrict__ out) {
    const int c = blockIdx.x * kConvBlock + threadIdx.x;
    if (c >= n) return;
    const int r = blockIdx.y;
    float *state = states[r];
    const float s0 = state[c], s1 = state[n + c], s2 = state[2 * n + c];
    const float4 wc = w[c];
    const float x = proj[(size_t)r * ld + c];
    const float v = __fmaf_rn(x, wc.w, __fmaf_rn(s2, wc.z, __fmaf_rn(s1, wc.y, __fmul_rn(s0, wc.x))));
    out[(size_t)r * n + c] = silu_f<ExactExp>(v);
    state[c] = s1;
    state[n + c] = s2;
    state[2 * n + c] = x;
}

struct DeltaMulti {
    const float *conv;
    int conv_ld;
    const float *proj;
    int ld, z_off, beta_off, alpha_off;
    const float *a;
    const float *dt_bias;
    const float *norm_w;
    float *const *states;
    float *out;
    int n_k_heads, n_v_heads;
    float eps;
};

// One block per (v head, row); thread j owns column j of the head's 128x128 state. The body is
// gated_delta_k of the vendored ops.cu for one token.
__global__ void __launch_bounds__(kStateDim) gated_delta_multi_k(DeltaMulti a) {
    constexpr int D = kStateDim;
    __shared__ float qs[D], ks[D];
    const int h = blockIdx.x, r = blockIdx.y, j = threadIdx.x, kh = h % a.n_k_heads;
    float *st = a.states[r] + (size_t)h * D * D;
    float s[D];
#pragma unroll
    for (int i = 0; i < D; ++i) s[i] = st[i * D + j];

    const float *c = a.conv + (size_t)r * a.conv_ld;
    const float *p = a.proj + (size_t)r * a.ld;
    float qv = c[kh * D + j], kv = c[(a.n_k_heads + kh) * D + j];
    float vv = c[2 * a.n_k_heads * D + h * D + j];
    // L2 norms, and q's 1/sqrt(D) attention scale.
    float qn = rsqrtf(block_sum_f(qv * qv) + a.eps) * rsqrtf((float)D);
    float kn = rsqrtf(block_sum_f(kv * kv) + a.eps);
    qs[j] = qv * qn;
    ks[j] = kv * kn;
    __syncthreads();
    float decay = __expf(softplus_f(p[a.alpha_off + h] + a.dt_bias[h]) * a.a[h]);
    float beta = sigmoid_f(p[a.beta_off + h]);
    float sk = 0.f;
#pragma unroll
    for (int i = 0; i < D; ++i) {
        s[i] *= decay;
        sk += s[i] * ks[i];
    }
    float d = (vv - sk) * beta, o = 0.f;
#pragma unroll
    for (int i = 0; i < D; ++i) {
        s[i] += ks[i] * d;
        o += s[i] * qs[i];
    }
    float rstd = rsqrtf(block_sum_f(o * o) / D + a.eps);
    float z = p[a.z_off + h * D + j];
    a.out[(size_t)r * a.n_v_heads * D + h * D + j] = o * rstd * a.norm_w[j] * silu_f<false>(z);
#pragma unroll
    for (int i = 0; i < D; ++i) st[i * D + j] = s[i];
}

// IEEE counterpart of gated_delta_ieee_m1_k in gdn_delta.cu for n rows: hoists decay, beta and
// silu(z) before loading s[D] so IEEE expf/log1pf temporaries never spill the state register file,
// and matches single-row gdn_delta_ieee_launch bit for bit.
__global__ void __launch_bounds__(kStateDim) gated_delta_multi_ieee_k(DeltaMulti a) {
    constexpr int D = kStateDim;
    __shared__ float qs[D], ks[D];
    __shared__ float sh_decay, sh_beta;
    const int h = blockIdx.x, r = blockIdx.y, j = threadIdx.x, kh = h % a.n_k_heads;
    const float *c = a.conv + (size_t)r * a.conv_ld;
    const float *p = a.proj + (size_t)r * a.ld;
    const float qv = c[kh * D + j];
    const float kv = c[(a.n_k_heads + kh) * D + j];
    const float vv = c[2 * a.n_k_heads * D + h * D + j];
    const float qn = rsqrtf(block_sum_f(qv * qv) + a.eps) * rsqrtf((float)D);
    const float kn = rsqrtf(block_sum_f(kv * kv) + a.eps);
    qs[j] = qv * qn;
    ks[j] = kv * kn;
    if (j == 0) {
        sh_decay = expf(softplus_ieee_f(p[a.alpha_off + h] + a.dt_bias[h]) * a.a[h]);
        sh_beta = 1.f / (1.f + expf(-p[a.beta_off + h]));
    }
    const float z = p[a.z_off + h * D + j];
    const float silu_z = z / (1.f + expf(-z));
    __syncthreads();
    const float decay = sh_decay;
    const float beta = sh_beta;
    float *st = a.states[r] + (size_t)h * D * D;
    float s[D];
#pragma unroll
    for (int i = 0; i < D; ++i) s[i] = st[i * D + j];
    float sk = 0.f;
#pragma unroll
    for (int i = 0; i < D; ++i) {
        s[i] *= decay;
        sk += s[i] * ks[i];
    }
    const float d = (vv - sk) * beta;
    float o = 0.f;
#pragma unroll
    for (int i = 0; i < D; ++i) {
        s[i] += ks[i] * d;
        o += s[i] * qs[i];
    }
#pragma unroll
    for (int i = 0; i < D; ++i) st[i * D + j] = s[i];
    const float rstd = rsqrtf(block_sum_f(o * o) / (float)D + a.eps);
    a.out[(size_t)r * a.n_v_heads * D + h * D + j] = o * rstd * a.norm_w[j] * silu_z;
}

struct PrepMulti {
    const float *proj;
    int ld, n_head, n_kv_head, n_rot, cells;
    const float *q_norm;
    const float *k_norm;
    float eps, rope_base;
    float *q, *k_rows, *v_rows;
    const inference_cuda_kv_row *rows;
};

// attn_prep_k (the f32 rows instance) for one token per row: block (head, row), thread per
// dimension. The position word and the status word are the row's own.
__global__ void attn_prep_multi_k(PrepMulti a) {
    constexpr int D = kHeadDim;
    __shared__ float x[D];
    const int head = blockIdx.x, t = blockIdx.y, d = threadIdx.x;
    const char *step = (const char *)a.rows[t].step;
    const long long first = *(const int *)(step + INFERENCE_CUDA_STEP_POS_OFFSET);
    if (first < 0 || first + 1 > a.rows[t].cells) {
        if (blockIdx.x == 0 && threadIdx.x == 0)
            atomicOr((int *)(step + INFERENCE_CUDA_STEP_STATUS_OFFSET), kStatusInvalidStepState);
        if (head < a.n_head) a.q[((size_t)t * a.n_head + head) * D + d] = __int_as_float(0x7fc00000);
        return;
    }
    const int pos = (int)first;
    const float *p = a.proj + (size_t)t * a.ld;
    const int kv_base = a.n_head * 2 * D;
    if (head >= a.n_head + a.n_kv_head) {
        const int vh = head - a.n_head - a.n_kv_head;
        a.v_rows[((size_t)t * a.n_kv_head + vh) * D + d] = p[kv_base + (a.n_kv_head + vh) * D + d];
        return;
    }
    const bool is_q = head < a.n_head;
    float v = is_q ? p[head * 2 * D + d] : p[kv_base + (head - a.n_head) * D + d];
    float rstd = rsqrtf(block_sum_f(v * v) / D + a.eps);
    x[d] = v * rstd * (is_q ? a.q_norm : a.k_norm)[d];
    __syncthreads();
    float out = x[d];
    const int half = a.n_rot / 2;
    if (d < a.n_rot) {
        const int i = d % half;
        float theta = pos * powf(a.rope_base, -2.f * i / a.n_rot), sn, cs;
        sincosf(theta, &sn, &cs);
        out = d < half ? x[d] * cs - x[d + half] * sn : x[d - half] * sn + x[d] * cs;
    }
    if (is_q)
        a.q[((size_t)t * a.n_head + head) * D + d] = out;
    else
        a.k_rows[((size_t)t * a.n_kv_head + head - a.n_head) * D + d] = out;
}

template <bool ExactExp>
int conv_multi_launch_impl(const inference_cuda_gdn_conv_multi_args *a, void *stream,
                           int *out_stage) {
    try {
        report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
        if (a == NULL || stream == NULL || a->proj == NULL || a->w == NULL || a->states == NULL ||
            a->out == NULL || !rows_ok(a->n) || a->channels < 1 || a->channels > kMaxChannels ||
            a->ld < a->channels || a->ld > kMaxLd || !aligned(a->proj, 4) || !aligned(a->out, 4) ||
            !aligned(a->w, 16) || !aligned(a->states, 8)) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        const dim3 grid((a->channels + kConvBlock - 1) / kConvBlock, a->n);
        conv_silu_multi_k<ExactExp><<<grid, kConvBlock, 0, (cudaStream_t)stream>>>(
            (const float *)a->proj, a->ld, a->channels, (const float4 *)a->w,
            (float *const *)a->states, (float *)a->out);
        Failure failure;
        failure.record(cudaGetLastError(), INFERENCE_CUDA_STAGE_LAUNCH);
        report_stage(out_stage, failure.stage);
        return (int)failure.err;
    } catch (...) {
        return INFERENCE_CUDA_NATIVE_EXCEPTION;
    }
}

template <bool ExactExp>
int delta_multi_launch_impl(const inference_cuda_gdn_delta_multi_args *a, void *stream,
                            int *out_stage) {
    try {
        report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
        if (a == NULL || stream == NULL || a->conv == NULL || a->proj == NULL || a->a == NULL ||
            a->dt_bias == NULL || a->norm_w == NULL || a->states == NULL || a->out == NULL ||
            !rows_ok(a->n) || a->n_k_heads < 1 || a->n_k_heads > kMaxKHeads || a->n_v_heads < 1 ||
            a->n_v_heads > kMaxVHeads || a->n_v_heads % a->n_k_heads != 0 || a->eps != kEps ||
            a->z_off < 0 || a->beta_off < 0 || a->alpha_off < 0 || !aligned(a->conv, 4) ||
            !aligned(a->proj, 4) || !aligned(a->a, 4) || !aligned(a->dt_bias, 4) ||
            !aligned(a->norm_w, 4) || !aligned(a->out, 4) || !aligned(a->states, 8)) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        const long conv_cols = (2L * a->n_k_heads + a->n_v_heads) * kStateDim;
        const long z_end = (long)a->z_off + (long)a->n_v_heads * kStateDim;
        const long b_end = (long)a->beta_off + a->n_v_heads, al_end = (long)a->alpha_off + a->n_v_heads;
        const long need_ld = z_end > b_end ? (z_end > al_end ? z_end : al_end) : (b_end > al_end ? b_end : al_end);
        if (a->conv_ld < conv_cols || a->conv_ld > (1 << 20) || a->ld < need_ld || a->ld > (1 << 22)) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        DeltaMulti k = {(const float *)a->conv, a->conv_ld, (const float *)a->proj, a->ld, a->z_off,
                        a->beta_off, a->alpha_off, (const float *)a->a, (const float *)a->dt_bias,
                        (const float *)a->norm_w, (float *const *)a->states, (float *)a->out,
                        a->n_k_heads, a->n_v_heads, a->eps};
        if constexpr (ExactExp) {
            gated_delta_multi_ieee_k<<<dim3(a->n_v_heads, a->n), kStateDim, 0, (cudaStream_t)stream>>>(k);
        } else {
            gated_delta_multi_k<<<dim3(a->n_v_heads, a->n), kStateDim, 0, (cudaStream_t)stream>>>(k);
        }
        Failure failure;
        failure.record(cudaGetLastError(), INFERENCE_CUDA_STAGE_LAUNCH);
        report_stage(out_stage, failure.stage);
        return (int)failure.err;
    } catch (...) {
        return INFERENCE_CUDA_NATIVE_EXCEPTION;
    }
}

}  // namespace

extern "C" int inference_cuda_gdn_conv_multi_launch(const inference_cuda_gdn_conv_multi_args *a,
                                                    void *stream, int *out_stage) {
    return conv_multi_launch_impl<false>(a, stream, out_stage);
}

extern "C" int inference_cuda_gdn_conv_multi_ieee_launch(const inference_cuda_gdn_conv_multi_args *a,
                                                         void *stream, int *out_stage) {
    return conv_multi_launch_impl<true>(a, stream, out_stage);
}

extern "C" int inference_cuda_gdn_delta_multi_launch(const inference_cuda_gdn_delta_multi_args *a,
                                                     void *stream, int *out_stage) {
    return delta_multi_launch_impl<false>(a, stream, out_stage);
}

extern "C" int inference_cuda_gdn_delta_multi_ieee_launch(const inference_cuda_gdn_delta_multi_args *a,
                                                          void *stream, int *out_stage) {
    return delta_multi_launch_impl<true>(a, stream, out_stage);
}

extern "C" int inference_cuda_attn_prep_multi_launch(const inference_cuda_attn_prep_multi_args *a,
                                                     void *stream, int *out_stage) {
    try {
        report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
        if (a == NULL || stream == NULL || a->proj == NULL || a->q_norm == NULL || a->k_norm == NULL ||
            a->k_rows == NULL || a->v_rows == NULL || a->q == NULL || a->rows == NULL ||
            !rows_ok(a->n) || a->head_dim != kHeadDim || a->n_head < 1 || a->n_head > kMaxHeads ||
            a->n_kv_head < 1 || a->n_kv_head > kMaxKvHeads || a->n_rot < 2 || a->n_rot > kHeadDim ||
            (a->n_rot & (a->n_rot - 1)) != 0 || a->cells < 1 || a->cells > kMaxCells ||
            !isfinite(a->eps) || !(a->eps > 0.0f) || !isfinite(a->rope_base) || !(a->rope_base > 1.0f) ||
            (long long)a->ld < ((long long)a->n_head * 2 + (long long)a->n_kv_head * 2) * kHeadDim ||
            !aligned(a->proj, 4) || !aligned(a->q_norm, 4) || !aligned(a->k_norm, 4) ||
            !aligned(a->q, 16) || !aligned(a->k_rows, 16) || !aligned(a->v_rows, 16) ||
            !aligned(a->rows, 8)) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        PrepMulti k = {(const float *)a->proj, a->ld, a->n_head, a->n_kv_head, a->n_rot, a->cells,
                       (const float *)a->q_norm, (const float *)a->k_norm, a->eps, a->rope_base,
                       (float *)a->q, (float *)a->k_rows, (float *)a->v_rows, a->rows};
        attn_prep_multi_k<<<dim3(a->n_head + 2 * a->n_kv_head, a->n), kHeadDim, 0, (cudaStream_t)stream>>>(k);
        Failure failure;
        failure.record(cudaGetLastError(), INFERENCE_CUDA_STAGE_LAUNCH);
        report_stage(out_stage, failure.stage);
        return (int)failure.err;
    } catch (...) {
        return INFERENCE_CUDA_NATIVE_EXCEPTION;
    }
}

namespace {

// The fp16 append of one token per row: block (kv head, row), thread per dimension. The planes,
// the position word and the status word are the row's own; the value is rounded to nearest even,
// as the single-sequence attn_prep rounds the rows it appends to an fp16 cache.
__global__ void kv16_append_multi_k(const float *k, const float *v, const inference_cuda_kv_row *rows,
                                    int n_kv_head, int head_dim) {
    const int h = blockIdx.x, t = blockIdx.y, d = threadIdx.x;
    const char *step = (const char *)rows[t].step;
    const long long pos = *(const int *)(step + INFERENCE_CUDA_STEP_POS_OFFSET);
    if (pos < 0 || pos >= rows[t].cells) {
        if (h == 0 && d == 0) atomicOr((int *)(step + INFERENCE_CUDA_STEP_STATUS_OFFSET), kStatusInvalidStepState);
        return;
    }
    const size_t src = ((size_t)t * n_kv_head + h) * head_dim + d;
    const size_t dst = ((size_t)pos * n_kv_head + h) * head_dim + d;
    ((__half *)rows[t].k_codes)[dst] = __float2half_rn(k[src]);
    ((__half *)rows[t].v_codes)[dst] = __float2half_rn(v[src]);
}

}  // namespace

extern "C" int inference_cuda_kv16_append_multi_launch(const inference_cuda_kv16_append_multi_args *a,
                                                       void *stream, int *out_stage) {
    try {
        report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
        if (a == NULL || stream == NULL || a->k == NULL || a->v == NULL || a->rows == NULL ||
            !rows_ok(a->n) || a->n_kv_head < 1 || a->n_kv_head > kMaxKvHeads || a->head_dim < 32 ||
            a->head_dim > 1024 || a->head_dim % 32 != 0 || a->cells < 1 || a->cells > kMaxCells ||
            !aligned(a->k, 4) || !aligned(a->v, 4) || !aligned(a->rows, 8)) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        kv16_append_multi_k<<<dim3(a->n_kv_head, a->n), a->head_dim, 0, (cudaStream_t)stream>>>(
            (const float *)a->k, (const float *)a->v, a->rows, a->n_kv_head, a->head_dim);
        Failure failure;
        failure.record(cudaGetLastError(), INFERENCE_CUDA_STAGE_LAUNCH);
        report_stage(out_stage, failure.stage);
        return (int)failure.err;
    } catch (...) {
        return INFERENCE_CUDA_NATIVE_EXCEPTION;
    }
}
