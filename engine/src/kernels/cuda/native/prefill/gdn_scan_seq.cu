// CPF-DN-A K1/K2: the value-sliced sequential gated-delta scan over one chunk and the gated
// output RMSNorm (decision 0049 design row 9, P-a). Design guidance only, no code copied:
// the per-token math and its expression order are the vendored kern::gated_delta
// (vendor/xavier-qwen/src/ops.cu gated_delta_k, the user's own xavier-qwen code, RD-4V)
// restated in a value-sliced launch; the launch shape (grid (H, 1, S_v/4), block (32, 4),
// __launch_bounds__(128, 2), s_shard[4] in registers, a serial t loop) follows the design
// text and was read from the pinned llama.cpp gated_delta_net.cu for geometry only. fp32
// SIMT only: no MMA, no fp16, so the same source compiles unchanged for sm_72. Contract:
// prefill/gdn_scan_seq.h.
#include "prefill/gdn_scan_seq.h"

#include <cuda_runtime.h>
#include <stddef.h>
#include <stdint.h>

#include "abi.h"
#include "device.h"

namespace {

constexpr int kMaxM = 32768;         // the vendored kMaxM (gdn_conv.cu:16 / gdn_delta.cu:16)
constexpr int kMaxKHeads = 64;       // the RD-7b policy limits (gdn_delta.cu:14-15)
constexpr int kMaxVHeads = 128;
constexpr int kMaxLd = 1 << 22;      // gdn_delta.cu:41-52
constexpr int kMaxCtxTokens = (1 << 30) - 1;
constexpr float kEps = 1e-6f;        // the vendored pin (gdn_delta.cu:21)
constexpr int D = 128;               // the compiled head size (kern::kStateDim)
constexpr int kLanes = 32;           // one warp per value column
constexpr int kRowsPerLane = D / kLanes;  // 4 state rows per lane
constexpr int kSlices = D / 4;            // 32 slice warps per head: kSlices * 4 == D

struct ScanArgs {
    const float *conv, *proj, *a, *dt_bias, *norm_w;
    float *state, *o_out;
    int *status;
    int m, p0, ctx_tokens, conv_ld, ld, z_off, beta_off, alpha_off, n_k_heads, n_v_heads;
    float eps;
};

bool aligned(const void *p, size_t a) { return (uintptr_t)p % a == 0; }

struct Span {
    const void *p;
    size_t bytes;
};

bool apart(const Span &a, const Span &b) {
    uintptr_t pa = (uintptr_t)a.p, pb = (uintptr_t)b.p;
    return pa + a.bytes <= pb || pb + b.bytes <= pa;
}

/* The p0 bound the wrapper, this entry and the kernels all check (decision 0049 item 2),
 * the sum in 64 bits. */
bool bounds_ok(const inference_cuda_gdn_scan_seq_args &a) {
    return a.p0 >= 0 && (long long)a.p0 + a.m <= a.ctx_tokens;
}

/* Mirrors the Rust wrapper term by term; a term added here needs a row there. The Rust side
 * refuses everything this refuses, so a C refusal means a wrapper defect. */
bool valid(const inference_cuda_gdn_scan_seq_args &a, void *stream) {
    if (a.conv == NULL || a.proj == NULL || a.a == NULL || a.dt_bias == NULL ||
        a.norm_w == NULL || a.state == NULL || a.o_out == NULL || stream == NULL) {
        return false;
    }
    if (a.m < 1 || a.m > kMaxM || a.head_dim != D || a.n_k_heads < 1 ||
        a.n_k_heads > kMaxKHeads || a.n_v_heads < 1 || a.n_v_heads > kMaxVHeads ||
        a.n_v_heads % a.n_k_heads != 0 || a.eps != kEps) {
        return false;
    }
    const long conv_cols = (2L * a.n_k_heads + a.n_v_heads) * D;
    if ((long)a.conv_ld != conv_cols || a.z_off < 0 || a.beta_off < 0 || a.alpha_off < 0) {
        return false;
    }
    const long z_end = (long)a.z_off + (long)a.n_v_heads * D;
    const long b_end = (long)a.beta_off + a.n_v_heads, al_end = (long)a.alpha_off + a.n_v_heads;
    const long need_ld = z_end > b_end ? (z_end > al_end ? z_end : al_end)
                                       : (b_end > al_end ? b_end : al_end);
    if (a.ld < need_ld || a.ld > kMaxLd || a.ctx_tokens < 1 || a.ctx_tokens > kMaxCtxTokens) {
        return false;
    }
    if (!aligned(a.conv, 4) || !aligned(a.proj, 4) || !aligned(a.a, 4) ||
        !aligned(a.dt_bias, 4) || !aligned(a.norm_w, 4) || !aligned(a.state, 4) ||
        !aligned(a.o_out, 4) || (a.status != NULL && !aligned(a.status, 4))) {
        return false;
    }
    const size_t m = (size_t)a.m, nv = (size_t)a.n_v_heads, f = sizeof(float);
    size_t proj_cols = (size_t)z_end > (size_t)b_end ? (size_t)z_end : (size_t)b_end;
    proj_cols = proj_cols > (size_t)al_end ? proj_cols : (size_t)al_end;
    const Span state = {a.state, nv * D * D * f}, out = {a.o_out, m * nv * D * f};
    const Span reads[] = {{a.conv, m * (size_t)conv_cols * f},
                          {a.proj, ((m - 1) * (size_t)a.ld + proj_cols) * f},
                          {a.a, nv * f},
                          {a.dt_bias, nv * f},
                          {a.norm_w, (size_t)D * f}};
    if (!apart(state, out)) {
        return false;
    }
    for (const Span &r : reads) {
        if (!apart(state, r) || !apart(out, r)) {
            return false;
        }
    }
    if (a.status != NULL) {
        const Span status = {a.status, f};
        if (!apart(status, state) || !apart(status, out)) {
            return false;
        }
        for (const Span &r : reads) {
            if (!apart(status, r)) {
                return false;
            }
        }
    }
    return bounds_ok(a);
}

#ifdef INFERENCE_CUDA_FAULT_QUALIFICATION
/* valid() with `p0` replaced: the one field the sans-bounds check below must not depend
 * on. `p0` 0 passes the bound whenever m <= ctx_tokens, which every call that reaches the
 * fault entry satisfies (a real p0 in range implies it). */
bool valid_with_p0(const inference_cuda_gdn_scan_seq_args &a, void *stream, int p0_override) {
    inference_cuda_gdn_scan_seq_args b = a;
    b.p0 = p0_override;
    return valid(b, stream);
}

/* Every rule of the entry except the p0 bound, so the fault entry can reuse the check
 * while it deliberately violates that bound; the kernels' own device re-check is the code
 * under test. */
bool valid_sans_bounds(const inference_cuda_gdn_scan_seq_args &a, void *stream) {
    return valid_with_p0(a, stream, 0);
}
#endif

ScanArgs to_args(const inference_cuda_gdn_scan_seq_args *a) {
    ScanArgs k = {};
    k.conv = (const float *)a->conv;
    k.proj = (const float *)a->proj;
    k.a = (const float *)a->a;
    k.dt_bias = (const float *)a->dt_bias;
    k.norm_w = (const float *)a->norm_w;
    k.state = (float *)a->state;
    k.o_out = (float *)a->o_out;
    k.status = a->status;
    k.m = a->m;
    k.p0 = a->p0;
    k.ctx_tokens = a->ctx_tokens;
    k.conv_ld = a->conv_ld;
    k.ld = a->ld;
    k.z_off = a->z_off;
    k.beta_off = a->beta_off;
    k.alpha_off = a->alpha_off;
    k.n_k_heads = a->n_k_heads;
    k.n_v_heads = a->n_v_heads;
    k.eps = a->eps;
    return k;
}

__device__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

// Sum of v over the block, returned to every thread. The vendored reduction (ops.cu:20-29):
// identical order and barriers, re-stated because the vendored helper is file-local.
__device__ float block_sum(float v) {
    __shared__ float red[32];
    v = warp_sum(v);
    int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    __syncthreads();  // earlier callers may still be reading red
    if (lane == 0) red[warp] = v;
    __syncthreads();
    v = lane < (int)blockDim.x / 32 ? red[lane] : 0.f;
    return warp_sum(v);
}

// expf forms of the vendored gates (ops.cu:31-33 uses __expf): this file stays out of the
// fast-math groups, so every transcendental is the accurate one.
__device__ float silu_e(float x) { return x / (1.f + expf(-x)); }
__device__ float softplus_e(float x) { return x > 20.f ? x : log1pf(expf(x)); }

// K1: one warp owns four value columns of one head; lane l holds state rows
// l, l+32, l+64, l+96 of its column in registers for the whole chunk (state loaded once
// before the token loop, written back once after it).
__global__ void __launch_bounds__(kLanes * 4, 2) gdn_scan_seq_k(ScanArgs a) {
    const int h = blockIdx.x, kh = h % a.n_k_heads;
    const int lane = threadIdx.x, col = blockIdx.z * 4 + threadIdx.y;
    float *st = a.state + (size_t)h * D * D;
    // The device re-check of the launch arguments (decision 0049 item 2): block uniform,
    // before the state load. A refused launch loads no state, updates nothing, NaNs its
    // o_out columns for every t and ORs mask 8 into the optional status word (item 10).
    if ((long long)a.p0 < 0 || (long long)a.p0 + a.m > a.ctx_tokens) {
        if (a.status && blockIdx.x == 0 && blockIdx.z == 0 && threadIdx.x == 0 &&
            threadIdx.y == 0) {
            atomicOr(a.status, INFERENCE_CUDA_PREFILL_STATUS_BOUND_VIOLATION);
        }
        for (int t = 0; t < a.m; ++t) {
            a.o_out[((size_t)t * a.n_v_heads + h) * D + col] = __int_as_float(0x7fc00000);
        }
        return;
    }
    float s_shard[kRowsPerLane];
#pragma unroll
    for (int r = 0; r < kRowsPerLane; ++r) {
        s_shard[r] = st[(size_t)(r * kLanes + lane) * D + col];
    }
    for (int t = 0; t < a.m; ++t) {
        const float *c = a.conv + (size_t)t * a.conv_ld;
        const float *p = a.proj + (size_t)t * a.ld;
        float qv[kRowsPerLane], kv[kRowsPerLane];
#pragma unroll
        for (int r = 0; r < kRowsPerLane; ++r) {
            qv[r] = c[kh * D + r * kLanes + lane];
            kv[r] = c[(a.n_k_heads + kh) * D + r * kLanes + lane];
        }
        const float vv = c[2 * a.n_k_heads * D + h * D + col];
        // L2 norms in the ggml form (norm.cu:273), the norm and q's 1/sqrt(D) folded into
        // one multiplier (the vendored form of ops.cu:122). Every slice warp of the head
        // computes its own copy; the redundancy is L1-served.
        float sq = 0.f, ss = 0.f;
#pragma unroll
        for (int r = 0; r < kRowsPerLane; ++r) {
            sq += qv[r] * qv[r];
            ss += kv[r] * kv[r];
        }
        const float qmul = rsqrtf(fmaxf(warp_sum(sq), a.eps * a.eps)) * rsqrtf((float)D);
        const float kmul = rsqrtf(fmaxf(warp_sum(ss), a.eps * a.eps));
        float qhat[kRowsPerLane], khat[kRowsPerLane];
#pragma unroll
        for (int r = 0; r < kRowsPerLane; ++r) {
            qhat[r] = qv[r] * qmul;
            khat[r] = kv[r] * kmul;
        }
        // Gates from the projection row, in kernel; a is ssm_a as stored (no second exp).
        const float decay =
            expf(softplus_e(p[a.alpha_off + h] + a.dt_bias[h]) * a.a[h]);
        const float beta = 1.f / (1.f + expf(-p[a.beta_off + h]));
        // State update in the design's order: decay first, the sk dot against the DECAYED
        // state, then the rank-one write and the o dot (ops.cu:132-139).
        float sk_partial = 0.f;
#pragma unroll
        for (int r = 0; r < kRowsPerLane; ++r) {
            s_shard[r] *= decay;
            sk_partial += s_shard[r] * khat[r];
        }
        const float d = (vv - warp_sum(sk_partial)) * beta;
        float o_partial = 0.f;
#pragma unroll
        for (int r = 0; r < kRowsPerLane; ++r) {
            s_shard[r] += khat[r] * d;
            o_partial += s_shard[r] * qhat[r];
        }
        const float o = warp_sum(o_partial);
        if (lane == 0) a.o_out[((size_t)t * a.n_v_heads + h) * D + col] = o;
    }
#pragma unroll
    for (int r = 0; r < kRowsPerLane; ++r) {
        st[(size_t)(r * kLanes + lane) * D + col] = s_shard[r];
    }
}

// K2: the gated output RMSNorm of m rows, in place on the o_out rows K1 wrote (the vendored
// mean+eps form, ops.cu:141; the ggml L2 form applies to the q/k norms only). One block per
// (token, head), one thread per column; each block reads its D values before writing them.
__global__ void gdn_out_gate_k(ScanArgs a) {
    const int h = blockIdx.x, t = blockIdx.y, j = threadIdx.x;
    if ((long long)a.p0 < 0 || (long long)a.p0 + a.m > a.ctx_tokens) {
        if (a.status && blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0) {
            atomicOr(a.status, INFERENCE_CUDA_PREFILL_STATUS_BOUND_VIOLATION);
        }
        a.o_out[((size_t)t * a.n_v_heads + h) * D + j] = __int_as_float(0x7fc00000);
        return;
    }
    float *row = a.o_out + ((size_t)t * a.n_v_heads + h) * D;
    const float *p = a.proj + (size_t)t * a.ld;
    const float o = row[j];
    const float rstd = rsqrtf(block_sum(o * o) / D + a.eps);
    row[j] = o * rstd * a.norm_w[j] * silu_e(p[a.z_off + h * D + j]);
}

Failure launch_scan(const ScanArgs &k, void *stream) {
    Failure failure;
    gdn_scan_seq_k<<<dim3(k.n_v_heads, 1, kSlices), dim3(kLanes, 4, 1), 0,
                     (cudaStream_t)stream>>>(k);
    failure.record(cudaGetLastError(), INFERENCE_CUDA_STAGE_LAUNCH);
    if (failure.err == cudaSuccess) {
        gdn_out_gate_k<<<dim3(k.n_v_heads, k.m), D, 0, (cudaStream_t)stream>>>(k);
        failure.record(cudaGetLastError(), INFERENCE_CUDA_STAGE_LAUNCH);
    }
    return failure;
}

}  // namespace

extern "C" int inference_cuda_gdn_scan_seq_launch(
    const inference_cuda_gdn_scan_seq_args *args, void *stream, int *out_stage) {
    try {
        report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
        if (args == NULL || !valid(*args, stream)) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        const Failure failure = launch_scan(to_args(args), stream);
        report_stage(out_stage, failure.stage);
        return (int)failure.err;
    } catch (...) {
        return INFERENCE_CUDA_NATIVE_EXCEPTION;
    }
}

#ifdef INFERENCE_CUDA_FAULT_QUALIFICATION
extern "C" int inference_cuda_gdn_scan_seq_fault_bad_bounds_launch(
    const inference_cuda_gdn_scan_seq_args *args, void *stream, int *out_stage) {
    try {
        report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
        /* Every rule except the p0 bound holds, so every pointer and shape the guards'
         * NaN writes touch is valid; only the bound is violated, by force (the entry
         * overrides p0 below, after the sans-bounds validation). The kernels' device
         * re-check is the code under test. It is reached only when m <= ctx_tokens:
         * valid_sans_bounds checks the bound at p0 = 0, so a call with m > ctx_tokens is
         * refused here first, and a test of that shape would test this C refusal, not the
         * device guard. */
        if (args == NULL || !valid_sans_bounds(*args, stream)) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        ScanArgs k = to_args(args);
        k.p0 = k.ctx_tokens;  // ctx_tokens + m > ctx_tokens: the violation, p0 still an int
        const Failure failure = launch_scan(k, stream);
        report_stage(out_stage, failure.stage);
        return (int)failure.err;
    } catch (...) {
        return INFERENCE_CUDA_NATIVE_EXCEPTION;
    }
}
#endif
