/*
 * RD-5: the device router of the resident Qwen3.6 decode step (decision 0047).
 *
 * Design guidance only, no code copied. Idea from MegaCapybara v1.35 static RE,
 * the MegaCapybara static-RE lessons, C9 (route on the device so that the expert ids
 * never reach the host); design guidance only, no code copied. The code is original CUDA C for
 * sm_86 (sm_72 compiles), written from the shapes of this repository.
 *
 * Rule (the same as family/route.rs::route_experts with Router::SoftmaxTopK { renorm: true } and
 * as the qwen35moe graph of the pinned llama.cpp): p = softmax(x . W^T) over n_expert logits,
 * with the max, the exp, the sum and the divide all done in f32 here (the CPU path of
 * qwen36/host.rs takes the max in f32, the exp and the sum in f64, and rounds each p to f32);
 * take the top_k probabilities, larger first and the lower expert index first on a tie;
 * weight_i = p_i / (p_0 + ... + p_{k-1}), the sum taken sequentially in rank order. The
 * shared-expert gate is sigmoid(x . w_shared); it is optional (w_shared and shared_gate both NULL
 * for a family without a shared expert, MEL-A1). n_expert (up to 256) and top_k (up to 32) are
 * run-time values.
 *
 * Two launches on one stream:
 *   router_logits_k  grid ((rows + 1) / 2, m) with rows = n_expert + (w_shared ? 1 : 0), 64
 *                    threads. One warp per row of W; the extra row n_expert is w_shared. This is
 *                    the bandwidth part (2.1 MB of f32 weights per layer).
 *   router_select_k  grid m, 32 threads. One warp per row: softmax, top_k rounds of a warp-wide
 *                    best-of, renormalization, gate, fault check. No shared memory, no barrier.
 * Every reduction has a fixed order and there is no float atomic. The only atomic is an integer
 * OR of one status bit, which is order independent.
 *
 * A row is faulty when a logit is not finite, when the softmax denominator or the sum of the
 * selected probabilities is not a finite positive number, or when the gate is not finite. A
 * faulty row gets ids -1 (the expert kernels write NaN for an id outside [0, count) and read no
 * weight), weights 0, gate 0, and kRouterStatusBit is OR-ed into *status.
 * Differences from the CPU path, on purpose and on rows no healthy model produces: a -inf logit
 * is valid on the CPU (exp(-inf) = 0) and faulty here; a non-finite shared-gate dot is faulty
 * here, while the CPU computes sigmoid of it and returns NaN without an error.
 */
#include <cuda_runtime.h>
#include <stdint.h>

#include "abi.h"
#include "device.h"
#include "resident/router.h"


namespace {

constexpr int kRouterMaxExperts = 256;  // lane l owns experts l, l + 32, ... (8 per lane)
constexpr int kRouterMaxTop = 32;       // lane r keeps the winner of round r
constexpr int kRouterMaxRows = 4096;
constexpr int kRouterStatusBit = 1;     // OR-ed into *status for a faulty row
constexpr int kLogitChunk = 16;         // float4 loads in flight per operand and lane
constexpr int kLogitWarps = 2;          // rows of W per block; 2 was the fastest of 2, 4, 8
constexpr int kPer = kRouterMaxExperts / 32;

__device__ inline float warp_sum_r(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
__device__ inline float warp_max_r(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}

// One warp per row of w; row n_expert is w_shared. Lane l sums the float4 columns l, l + 32, ...
// with four running sums in column order, then the 32 lanes combine by an xor tree.
// logits is [m][n_expert + 1]; the last column is the shared-gate dot.
__global__ void __launch_bounds__(kLogitWarps * 32)
router_logits_k(const float *x, const float *w, const float *w_shared, float *logits, int hidden,
                int n_expert) {
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int e = blockIdx.x * kLogitWarps + warp, t = blockIdx.y;
    // Row n_expert is w_shared; without a shared gate it does not exist (w_shared is NULL), and the
    // whole warp returns, so the shuffles below always run with all 32 lanes.
    if (e > n_expert || (e == n_expert && w_shared == NULL)) {
        return;
    }
    const float4 *wr =
        reinterpret_cast<const float4 *>(e < n_expert ? w + (size_t)e * hidden : w_shared);
    const float4 *xr = reinterpret_cast<const float4 *>(x + (size_t)t * hidden);
    const int n4 = hidden / 4;
    float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
    for (int base = 0; base < n4; base += 32 * kLogitChunk) {
        float4 wv[kLogitChunk], xv[kLogitChunk];
#pragma unroll
        for (int j = 0; j < kLogitChunk; ++j) {
            const int i = base + j * 32 + lane;
            const bool in = i < n4;
            wv[j] = in ? __ldg(wr + i) : make_float4(0.f, 0.f, 0.f, 0.f);
            xv[j] = in ? __ldg(xr + i) : make_float4(0.f, 0.f, 0.f, 0.f);
        }
#pragma unroll
        for (int j = 0; j < kLogitChunk; ++j) {
            acc.x = fmaf(wv[j].x, xv[j].x, acc.x);
            acc.y = fmaf(wv[j].y, xv[j].y, acc.y);
            acc.z = fmaf(wv[j].z, xv[j].z, acc.z);
            acc.w = fmaf(wv[j].w, xv[j].w, acc.w);
        }
    }
    const float s = warp_sum_r((acc.x + acc.y) + (acc.z + acc.w));
    if (lane == 0) {
        logits[(size_t)t * (n_expert + 1) + e] = s;
    }
}

// True when (pa, ia) ranks before (pb, ib): larger probability first, then the lower index.
__device__ inline bool before(float pa, int ia, float pb, int ib) {
    return pa > pb || (pa == pb && ia < ib);
}

__global__ void __launch_bounds__(32)
router_select_k(const float *logits, int *ids, float *weights, float *shared_gate, int *status,
                int n_expert, int top_k) {  // shared_gate NULL: no shared-expert gate
    const int t = blockIdx.x, lane = threadIdx.x;
    const float *lg = logits + (size_t)t * (n_expert + 1);
    float p[kPer];
    bool bad = false;
    float mx = -INFINITY;
#pragma unroll
    for (int k = 0; k < kPer; ++k) {
        const int e = k * 32 + lane;
        p[k] = e < n_expert ? lg[e] : -INFINITY;
        bad |= e < n_expert && !isfinite(p[k]);
        mx = fmaxf(mx, p[k]);
    }
    mx = warp_max_r(mx);
    float zsum = 0.f;
#pragma unroll
    for (int k = 0; k < kPer; ++k) {
        p[k] = (k * 32 + lane) < n_expert ? expf(p[k] - mx) : 0.f;
        zsum += p[k];
    }
    const float z = warp_sum_r(zsum);
#pragma unroll
    for (int k = 0; k < kPer; ++k) {
        // -1: never a candidate. A padded expert has the highest index, so even at 0 it would lose
        // every tie against a real expert; the -1 is kept so that the order does not rest on that.
        p[k] = (k * 32 + lane) < n_expert ? p[k] / z : -1.f;
    }
    float my_p = 0.f, sum = 0.f;
    int my_i = -1;
    for (int r = 0; r < top_k; ++r) {
        float bp = -2.f;
        int bi = 0x7fffffff;
#pragma unroll
        for (int k = 0; k < kPer; ++k) {
            if (before(p[k], k * 32 + lane, bp, bi)) {
                bp = p[k];
                bi = k * 32 + lane;
            }
        }
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            const float op = __shfl_xor_sync(0xffffffffu, bp, o);
            const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
            if (before(op, oi, bp, bi)) {
                bp = op;
                bi = oi;
            }
        }
        sum += bp;  // rank order, sequential: the order of route_experts
        if (lane == r) {
            my_p = bp;
            my_i = bi;
        }
#pragma unroll
        for (int k = 0; k < kPer; ++k) {
            if (bi == k * 32 + lane) {
                p[k] = -1.f;  // the owner removes the winner
            }
        }
    }
    const float g = shared_gate != NULL ? lg[n_expert] : 0.f;
    bool fault = __any_sync(0xffffffffu, bad) || !isfinite(z) || z <= 0.f || !isfinite(sum) ||
                 sum <= 0.f || !isfinite(g);
    if (lane < top_k) {
        ids[(size_t)t * top_k + lane] = fault ? -1 : my_i;
        weights[(size_t)t * top_k + lane] = fault ? 0.f : my_p / sum;
    }
    if (lane == 0) {
        if (shared_gate != NULL) {
            shared_gate[t] = fault ? 0.f : 1.f / (1.f + expf(-g));
        }
        if (fault) {
            atomicOr(status, kRouterStatusBit);  // idempotent: the order of faulty rows is moot
        }
    }
}

bool aligned_to(const void *p, size_t n) { return ((uintptr_t)p % n) == 0; }

// A flat list of independent preconditions, one per line of the contract; splitting it would only
// scatter them. It cannot check buffer sizes or overlaps (the ABI carries no sizes): Rust does.
bool valid(const inference_cuda_router_args *a) {
    return a != NULL && a->x != NULL && a->w != NULL && a->logits != NULL && a->ids != NULL &&
           a->weights != NULL && a->status != NULL &&
           (a->w_shared == NULL) == (a->shared_gate == NULL) &&
           a->m >= 1 && a->m <= kRouterMaxRows && a->hidden >= 4 && a->hidden % 4 == 0 &&
           a->hidden <= 65536 && a->n_expert >= 1 && a->n_expert <= kRouterMaxExperts &&
           a->top_k >= 1 && a->top_k <= a->n_expert && a->top_k <= kRouterMaxTop &&
           aligned_to(a->x, 16) && aligned_to(a->w, 16) &&
           (a->w_shared == NULL || (aligned_to(a->w_shared, 16) && aligned_to(a->shared_gate, 4))) &&
           aligned_to(a->logits, 4) && aligned_to(a->ids, 4) && aligned_to(a->weights, 4) &&
           aligned_to(a->status, 4);
}

}  // namespace

extern "C" int inference_cuda_router_launch(const inference_cuda_router_args *a, void *stream,
                                            int *out_stage) {
    try {
        report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
        if (!valid(a)) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        const cudaStream_t s = (cudaStream_t)stream;
        Failure failure;
        const int rows = a->n_expert + (a->w_shared != NULL ? 1 : 0);  // the extra row is w_shared
        const dim3 grid((rows + kLogitWarps - 1) / kLogitWarps, a->m);
        router_logits_k<<<grid, kLogitWarps * 32, 0, s>>>(
            (const float *)a->x, (const float *)a->w, (const float *)a->w_shared,
            (float *)a->logits, a->hidden, a->n_expert);
        failure.record(cudaGetLastError(), INFERENCE_CUDA_STAGE_LAUNCH);
        if (failure.ok()) {
            router_select_k<<<a->m, 32, 0, s>>>((const float *)a->logits, (int *)a->ids,
                                                (float *)a->weights, (float *)a->shared_gate,
                                                (int *)a->status, a->n_expert, a->top_k);
            failure.record(cudaGetLastError(), INFERENCE_CUDA_STAGE_LAUNCH);
        }
        report_stage(out_stage, failure.stage);
        return (int)failure.err;
    } catch (...) {
        return INFERENCE_CUDA_NATIVE_EXCEPTION;
    }
}
