// Validated launches for the default vendored GDN kernel and the test-only state partition.
#include <cuda_runtime.h>
#include <stddef.h>
#include <stdint.h>

#include "abi.h"
#include "device.h"
#include "ops.h"
#include "resident/gdn_delta.h"
#include "resident/gdn_delta_partition.cuh"

namespace {

constexpr int kMaxM = 32768;
// Policy limits, not kernel bounds (the kernel launches n_v_heads blocks): Qwen3.6 and Qwen3-Next have 16 key and
// 32 value heads; raise these here, and in GDN_MAX_* of the wrapper, for a larger model.
constexpr int kMaxKHeads = 64;
constexpr int kMaxVHeads = 128;
constexpr float kEps = 1e-6f;  // one eps serves both norms; the GGUF rms eps is the f32 1e-6
constexpr size_t D = kern::kStateDim;

bool aligned(const void *p, size_t a) { return (uintptr_t)p % a == 0; }

struct Span {
    const void *p;
    size_t bytes;
};

bool apart(const Span &a, const Span &b) {
    uintptr_t pa = (uintptr_t)a.p, pb = (uintptr_t)b.p;
    return pa + a.bytes <= pb || pb + b.bytes <= pa;
}

bool shapes_ok(const inference_cuda_gdn_delta_args &a) {
    if (a.m < 1 || a.m > kMaxM || a.n_k_heads < 1 || a.n_k_heads > kMaxKHeads || a.n_v_heads < 1 ||
        a.n_v_heads > kMaxVHeads || a.n_v_heads % a.n_k_heads != 0 || a.eps != kEps) {
        return false;
    }
    // The offsets are bounded from above by ld (checked below, ld <= 1 << 22), so they have no limit of their own.
    if (a.z_off < 0 || a.beta_off < 0 || a.alpha_off < 0) {
        return false;
    }
    const long conv_cols = (2L * a.n_k_heads + a.n_v_heads) * (long)D;
    const long z_end = (long)a.z_off + (long)a.n_v_heads * (long)D;
    const long b_end = (long)a.beta_off + a.n_v_heads, al_end = (long)a.alpha_off + a.n_v_heads;
    const long need_ld = z_end > b_end ? (z_end > al_end ? z_end : al_end) : (b_end > al_end ? b_end : al_end);
    return a.conv_ld >= conv_cols && a.conv_ld <= (1 << 20) && a.ld >= need_ld && a.ld <= (1 << 22);
}

// Mirrors the Rust wrapper term by term; a term added here needs its twin there. The Rust side
// refuses everything this refuses, so a C refusal means a wrapper defect.
bool valid(const inference_cuda_gdn_delta_args &a, void *stream) {
    if (a.conv == NULL || a.proj == NULL || a.a == NULL || a.dt_bias == NULL || a.norm_w == NULL ||
        a.state == NULL || a.out == NULL || stream == NULL || !shapes_ok(a)) {
        return false;
    }
    if (!aligned(a.conv, 4) || !aligned(a.proj, 4) || !aligned(a.a, 4) || !aligned(a.dt_bias, 4) ||
        !aligned(a.norm_w, 4) || !aligned(a.state, 4) || !aligned(a.out, 4)) {
        return false;
    }
    const size_t m = (size_t)a.m, nv = (size_t)a.n_v_heads, nk = (size_t)a.n_k_heads;
    const size_t f = sizeof(float);
    const size_t conv_cols = (2 * nk + nv) * D;
    const size_t z_end = (size_t)a.z_off + nv * D, b_end = (size_t)a.beta_off + nv, al_end = (size_t)a.alpha_off + nv;
    size_t proj_cols = z_end > b_end ? z_end : b_end;
    proj_cols = proj_cols > al_end ? proj_cols : al_end;
    const Span state = {a.state, nv * D * D * f}, out = {a.out, m * nv * D * f};
    const Span reads[] = {{a.conv, ((m - 1) * (size_t)a.conv_ld + conv_cols) * f},
                          {a.proj, ((m - 1) * (size_t)a.ld + proj_cols) * f},
                          {a.a, nv * f},
                          {a.dt_bias, nv * f},
                          {a.norm_w, D * f}};
    if (!apart(state, out)) {
        return false;
    }
    for (const Span &r : reads) {
        if (!apart(state, r) || !apart(out, r)) {
            return false;
        }
    }
    return true;
}

__device__ __forceinline__ float warp_sum_ieee(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

__device__ __forceinline__ float block_sum_ieee(float v) {
    __shared__ float red[4];
    v = warp_sum_ieee(v);
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    __syncthreads();
    if (lane == 0) red[warp] = v;
    __syncthreads();
    return warp_sum_ieee(lane < 4 ? red[lane] : 0.f);
}

__device__ __forceinline__ float softplus_ieee(float x) {
    return x > 20.f ? x : log1pf(expf(x));
}

/* Single-token decode kernel: computes decay, beta, and silu(z) with IEEE expf/log1pf before
 * loading s[128] so the transcendental temporaries never overlap the state register file and
 * force local-memory spills across the 30 DeltaNet layers. Multi-token calls invoke one launch
 * per token so chunked and single-token calls are bit-identical by construction. */
__global__ void __launch_bounds__(kern::kStateDim) gated_delta_ieee_m1_k(kern::DeltaArgs a) {
    constexpr int kDim = kern::kStateDim;
    __shared__ float qs[kDim], ks[kDim];
    __shared__ float sh_decay, sh_beta;
    const int h = blockIdx.x;
    const int j = threadIdx.x;
    const int kh = h % a.n_k_heads;
    const float *c = a.conv;
    const float *p = a.proj;
    const float qv = c[kh * kDim + j];
    const float kv = c[(a.n_k_heads + kh) * kDim + j];
    const float vv = c[2 * a.n_k_heads * kDim + h * kDim + j];
    const float qn = rsqrtf(block_sum_ieee(qv * qv) + a.eps) * rsqrtf((float)kDim);
    const float kn = rsqrtf(block_sum_ieee(kv * kv) + a.eps);
    qs[j] = qv * qn;
    ks[j] = kv * kn;
    if (j == 0) {
        sh_decay = expf(softplus_ieee(p[a.alpha_off + h] + a.dt_bias[h]) * a.a[h]);
        sh_beta = 1.f / (1.f + expf(-p[a.beta_off + h]));
    }
    const float z = p[a.z_off + h * kDim + j];
    const float silu_z = z / (1.f + expf(-z));
    __syncthreads();
    const float decay = sh_decay;
    const float beta = sh_beta;
    float *st = a.state + (size_t)h * kDim * kDim;
    float s[kDim];
#pragma unroll
    for (int i = 0; i < kDim; ++i) s[i] = st[i * kDim + j];
    float sk = 0.f;
#pragma unroll
    for (int i = 0; i < kDim; ++i) {
        s[i] *= decay;
        sk += s[i] * ks[i];
    }
    const float d = (vv - sk) * beta;
    float o = 0.f;
#pragma unroll
    for (int i = 0; i < kDim; ++i) {
        s[i] += ks[i] * d;
        o += s[i] * qs[i];
    }
#pragma unroll
    for (int i = 0; i < kDim; ++i) st[i * kDim + j] = s[i];
    const float rstd = rsqrtf(block_sum_ieee(o * o) / (float)kDim + a.eps);
    a.out[(size_t)h * kDim + j] = o * rstd * a.norm_w[j] * silu_z;
}

cudaError_t launch_ieee(const kern::DeltaArgs &call, cudaStream_t stream) {
    kern::DeltaArgs step = call;
    step.m = 1;
    for (int t = 0; t < call.m; ++t) {
        step.conv = call.conv + (size_t)t * (size_t)call.conv_ld;
        step.proj = call.proj + (size_t)t * (size_t)call.ld;
        step.out = call.out + (size_t)t * (size_t)call.n_v_heads * kern::kStateDim;
        gated_delta_ieee_m1_k<<<step.n_v_heads, kern::kStateDim, 0, stream>>>(step);
        const cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            return err;
        }
    }
    return cudaSuccess;
}

}  // namespace

template <int Mode>
int delta_launch(const inference_cuda_gdn_delta_args *args,
                 void *stream, int *out_stage) {
    try {
        report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
        if (args == NULL || !valid(*args, stream)) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        // snap stays null: this entry does not use the speculative-rollback snapshots.
        kern::DeltaArgs call = {};
        call.conv = (const float *)args->conv;
        call.conv_ld = args->conv_ld;
        call.proj = (const float *)args->proj;
        call.ld = args->ld;
        call.z_off = args->z_off;
        call.beta_off = args->beta_off;
        call.alpha_off = args->alpha_off;
        call.a = (const float *)args->a;
        call.dt_bias = (const float *)args->dt_bias;
        call.norm_w = (const float *)args->norm_w;
        call.state = (float *)args->state;
        call.snap = NULL;
        call.out = (float *)args->out;
        call.m = args->m;
        call.n_k_heads = args->n_k_heads;
        call.n_v_heads = args->n_v_heads;
        call.eps = args->eps;
        Failure failure;
        cudaError_t err;
        if constexpr (Mode == 1) {
            err = gdn_partition::launch(call, (cudaStream_t)stream);
        } else if constexpr (Mode == 2) {
            err = launch_ieee(call, (cudaStream_t)stream);
        } else {
            err = kern::gated_delta(call, (cudaStream_t)stream);
        }
        failure.record(err, INFERENCE_CUDA_STAGE_LAUNCH);
        report_stage(out_stage, failure.stage);
        return (int)failure.err;
    } catch (...) {
        return INFERENCE_CUDA_NATIVE_EXCEPTION;
    }
}

extern "C" int inference_cuda_gdn_delta_launch(const inference_cuda_gdn_delta_args *args,
                                               void *stream, int *out_stage) {
    return delta_launch<0>(args, stream, out_stage);
}

extern "C" int inference_cuda_gdn_delta_partition_launch(const inference_cuda_gdn_delta_args *args,
                                                         void *stream, int *out_stage) {
    return delta_launch<1>(args, stream, out_stage);
}

extern "C" int inference_cuda_gdn_delta_ieee_launch(const inference_cuda_gdn_delta_args *args,
                                                    void *stream, int *out_stage) {
    return delta_launch<2>(args, stream, out_stage);
}
