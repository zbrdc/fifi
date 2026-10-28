/*
 * RD-5: the device router of the resident Qwen3.6 decode step (decision 0047).
 *
 * Routing runs on the device so that the expert ids never reach the host. The code is original
 * CUDA C for
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
 *                    the bandwidth part (2.1 MB of f32 weights per layer), and it is the decode
 *                    kernel: every token re-reads W.
 *   router_logits_tiled_k  the same logits, bit for bit (see there for one sign of zero), for
 *                    m >= kTiledMinRows (a prefill chunk): a warp takes 4 rows of W against 8 tokens, so each load of W serves 8
 *                    tokens and each load of x serves 4 rows of W.
 *   router_select_k  grid m, 32 threads. One warp per row: softmax, top_k rounds of a warp-wide
 *                    best-of, renormalization, gate, fault check. No shared memory, no barrier.
 * Every reduction has a fixed order and there is no float atomic. The only atomic is an integer
 * OR of one status bit, which is order independent. Both logits kernels keep the order of
 * router_logits_k (see there), so a row's logits do not depend on the kernel or on the other rows
 * of the launch; a different order would move the logits by rounding and could flip an expert.
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

#include "../abi.h"
#include "../device.h"
#include "router.h"


namespace {

constexpr int kRouterMaxExperts = 256;  // lane l owns experts l, l + 32, ... (8 per lane)
constexpr int kRouterMaxTop = 32;       // lane r keeps the winner of round r
constexpr int kRouterMaxRows = 4096;
constexpr int kRouterStatusBit = 1;     // OR-ed into *status for a faulty row
constexpr int kLogitChunk = 16;         // float4 loads in flight per operand and lane
constexpr int kLogitWarps = 2;          // rows of W per block; 2 was the fastest of 2, 4, 8
constexpr int kPer = kRouterMaxExperts / 32;
// router_logits_tiled_k: a warp tile is kTileExperts rows of W by kTileTokens rows of x, so the 32
// lanes end with one output each; a block is kTileWarpsE by kTileWarpsT warp tiles. A step does 12
// float4 loads for 128 fmaf, so the load pipe rather than the fma rate sets the speed, and the
// tile is as large as the registers of a lane allow (128 running sums and 12 float4 operands).
// Of the 4 x 4, 4 x 8 and 8 x 4 tiles in 1 to 8 warps per block, 4 x 8 with 2 x 2 warps was within
// 5% of the best (8 x 4 with 2 x 4 warps, which fills the register file with one block).
constexpr int kTileExperts = 4;
constexpr int kTileTokens = 8;
constexpr int kTileWarpsE = 2;
constexpr int kTileWarpsT = 2;
constexpr int kTileThreads = kTileWarpsE * kTileWarpsT * 32;
// Below one tile of tokens the tiled kernel has too little to share: it was slower than
// router_logits_k at 2 to 4 rows and equal at 5, so decode-sized launches keep the decode kernel.
constexpr int kTiledMinRows = 8;

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
                int n_expert, int logit_stride) {
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
        logits[(size_t)t * logit_stride + e] = s;
    }
}

// Every lane holds a partial sum of each of 32 values, v[j] being its partial of value j. Returns
// the total of value `lane` for the lane `lane`, summed over the 32 lanes in the order of
// warp_sum_r: at each step o = 16, 8, 4, 2, 1 a lane adds its partner's copy (lane ^ o) to its own,
// but a lane keeps and adds only one half of the values (the upper half when it has bit o set) and
// sends the other half, so a step costs half the shuffles of the step before it and the 32 values
// take 31 shuffles instead of the 160 of 32 calls of warp_sum_r. An addition commutes, so the lane
// that keeps a value gets the same bits as either lane of the pair would in the butterfly.
__device__ inline float lane_sums32(float (&v)[32], int lane) {
    int kept = 32;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        kept >>= 1;
        const bool upper = (lane & o) != 0;
#pragma unroll
        for (int j = 0; j < 16; ++j) {
            if (j < kept) {
                const float sent = upper ? v[j] : v[j + kept];
                const float own = upper ? v[j + kept] : v[j];
                v[j] = own + __shfl_xor_sync(0xffffffffu, sent, o);
            }
        }
    }
    return v[0];
}

// One step of router_logits_tiled_k: float4 column i of kTileExperts rows of W and of kTileTokens
// rows of x, into the running sums of every pair. A column past the row (in false) adds the exact
// zero product the decode kernel adds there.
__device__ inline void tile_step(float4 (&acc)[kTileExperts][kTileTokens],
                                 const float4 *const (&wr)[kTileExperts],
                                 const float4 *const (&xr)[kTileTokens], int i, bool in) {
    float4 wv[kTileExperts], xv[kTileTokens];
#pragma unroll
    for (int r = 0; r < kTileExperts; ++r) {
        wv[r] = in ? __ldg(wr[r] + i) : make_float4(0.f, 0.f, 0.f, 0.f);
    }
#pragma unroll
    for (int s = 0; s < kTileTokens; ++s) {
        xv[s] = in ? __ldg(xr[s] + i) : make_float4(0.f, 0.f, 0.f, 0.f);
    }
#pragma unroll
    for (int r = 0; r < kTileExperts; ++r) {
#pragma unroll
        for (int s = 0; s < kTileTokens; ++s) {
            acc[r][s].x = fmaf(wv[r].x, xv[s].x, acc[r][s].x);
            acc[r][s].y = fmaf(wv[r].y, xv[s].y, acc[r][s].y);
            acc[r][s].z = fmaf(wv[r].z, xv[s].z, acc[r][s].z);
            acc[r][s].w = fmaf(wv[r].w, xv[s].w, acc[r][s].w);
        }
    }
}

// The prefill logits: the dot products of router_logits_k with the same bits, for m rows at once.
// A warp takes a tile of kTileExperts rows of W (row n_expert is w_shared) by kTileTokens rows of
// x, so W is read once per tile of tokens instead of once per token. The order of every dot
// product is the one of router_logits_k and must stay so: lane l keeps four running sums over the
// float4 columns l, l + 32, ... in column order with fmaf, adds them as (x + y) + (z + w), and the
// 32 lanes combine by the xor tree of warp_sum_r (lane_sums32 does 32 of these trees at once).
// A tile past the last row of W or x reads a valid row instead (the first row of W, the last of x)
// and stores nothing. logits is [m][n_expert + 1], as for router_logits_k.
// The logits equal those of router_logits_k bit for bit, with one theoretical exception. A running
// sum is -0 only when its first product underflows to -0, and the decode kernel adds a +0 product
// for every column past the row in its last chunk of 512 columns, which turns a -0 into +0, while
// this kernel adds them only in its last step of 32 columns. So when hidden / 4 is not a multiple
// of 512, a logit that is exactly zero can differ in the sign of its zero. At hidden 2048 neither
// kernel adds a padded column and they cannot differ.
__global__ void __launch_bounds__(kTileThreads)
router_logits_tiled_k(const float *x, const float *w, const float *w_shared, float *logits,
                      int hidden, int n_expert, int m, int logit_stride) {
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int e0 = (blockIdx.x * kTileWarpsE + warp % kTileWarpsE) * kTileExperts;
    const int t0 = (blockIdx.y * kTileWarpsT + warp / kTileWarpsE) * kTileTokens;
    const int rows = n_expert + (w_shared != NULL ? 1 : 0);
    if (e0 >= rows || t0 >= m) {
        return;  // the whole warp, so the shuffles below always run with all 32 lanes
    }
    const float4 *wr[kTileExperts], *xr[kTileTokens];
#pragma unroll
    for (int r = 0; r < kTileExperts; ++r) {
        const int e = e0 + r;
        const bool shared_row = e == n_expert && w_shared != NULL;
        wr[r] = reinterpret_cast<const float4 *>(
            e < n_expert ? w + (size_t)e * hidden : (shared_row ? w_shared : w));
    }
#pragma unroll
    for (int s = 0; s < kTileTokens; ++s) {
        xr[s] = reinterpret_cast<const float4 *>(x + (size_t)min(t0 + s, m - 1) * hidden);
    }
    float4 acc[kTileExperts][kTileTokens];
#pragma unroll
    for (int r = 0; r < kTileExperts; ++r) {
#pragma unroll
        for (int s = 0; s < kTileTokens; ++s) {
            acc[r][s] = make_float4(0.f, 0.f, 0.f, 0.f);
        }
    }
    const int n4 = hidden / 4;
    const int full = n4 / 32;  // columns 0 .. 32 * full - 1 have all 32 lanes inside the row
#pragma unroll 2
    for (int k = 0; k < full; ++k) {
        tile_step(acc, wr, xr, k * 32 + lane, true);
    }
    if (n4 % 32 != 0) {
        const int i = full * 32 + lane;
        tile_step(acc, wr, xr, i, i < n4);
    }
    float part[32];  // output s * kTileExperts + r: token t0 + s against row e0 + r
#pragma unroll
    for (int r = 0; r < kTileExperts; ++r) {
#pragma unroll
        for (int s = 0; s < kTileTokens; ++s) {
            part[s * kTileExperts + r] =
                (acc[r][s].x + acc[r][s].y) + (acc[r][s].z + acc[r][s].w);
        }
    }
    const float sum = lane_sums32(part, lane);
    const int e = e0 + lane % kTileExperts, t = t0 + lane / kTileExperts;
    if (e < rows && t < m) {
        logits[(size_t)t * logit_stride + e] = sum;
    }
}

// True when (pa, ia) ranks before (pb, ib): larger probability first, then the lower index.
__device__ inline bool before(float pa, int ia, float pb, int ib) {
    return pa > pb || (pa == pb && ia < ib);
}

template <bool Sigmoid>
__global__ void __launch_bounds__(32)
router_select_k(const float *logits, int *ids, float *weights, float *shared_gate, int *status,
                int n_expert, int top_k, int logit_stride) {
    const int t = blockIdx.x, lane = threadIdx.x;
    const float *lg = logits + (size_t)t * logit_stride;
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
        p[k] = (k * 32 + lane) < n_expert
            ? (Sigmoid ? 1.f / (1.f + expf(-p[k])) : expf(p[k] - mx)) : 0.f;
        zsum += p[k];
    }
    const float z = warp_sum_r(zsum);
#pragma unroll
    for (int k = 0; k < kPer; ++k) {
        // Sigmoid scores stay F32 and unnormalized until the selected-rank sum below.
        // Normalizing all scores here can create extra ties through F32 rounding.
        p[k] = (k * 32 + lane) < n_expert ? (Sigmoid ? p[k] : p[k] / z) : -1.f;
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

bool split_span(const void *pointer, size_t available, size_t needed, size_t alignment) {
    const uintptr_t address = reinterpret_cast<uintptr_t>(pointer);
    return pointer != NULL && address % alignment == 0 && available >= needed &&
           needed <= UINTPTR_MAX - address;
}

bool split_apart(const void *a, size_t an, const void *b, size_t bn) {
    return reinterpret_cast<uintptr_t>(a) + an <= reinterpret_cast<uintptr_t>(b) ||
           reinterpret_cast<uintptr_t>(b) + bn <= reinterpret_cast<uintptr_t>(a);
}

bool valid_split_logits(const inference_cuda_router_logits_args *a) {
    if (!a || a->m < 1 || a->m > kRouterMaxRows || a->hidden < 4 ||
        a->hidden > 65536 || a->hidden % 4 || a->n_expert < 1 ||
        a->n_expert > kRouterMaxExperts) return false;
    const size_t x = size_t(a->m) * a->hidden * 4;
    const size_t w = size_t(a->n_expert) * a->hidden * 4;
    const size_t y = size_t(a->m) * a->n_expert * 4;
    return split_span(a->x, a->x_bytes, x, 16) && split_span(a->w, a->w_bytes, w, 16) &&
           split_span(a->logits, a->logits_bytes, y, 4) &&
           split_apart(a->logits, y, a->x, x) && split_apart(a->logits, y, a->w, w);
}

bool valid_split_select(const inference_cuda_router_sigmoid_args *a) {
    if (!a || a->m < 1 || a->m > kRouterMaxRows || a->n_expert < 1 ||
        a->n_expert > kRouterMaxExperts || a->top_k < 1 ||
        a->top_k > a->n_expert || a->top_k > kRouterMaxTop) return false;
    const size_t logits = size_t(a->m) * a->n_expert * 4;
    const size_t picks = size_t(a->m) * a->top_k * 4;
    return split_span(a->logits, a->logits_bytes, logits, 4) &&
           split_span(a->ids, a->ids_bytes, picks, 4) &&
           split_span(a->weights, a->weights_bytes, picks, 4) &&
           split_span(a->status, a->status_bytes, 4, 4) &&
           split_apart(a->ids, picks, a->logits, logits) &&
           split_apart(a->weights, picks, a->logits, logits) &&
           split_apart(a->status, 4, a->logits, logits) &&
           split_apart(a->ids, picks, a->weights, picks) &&
           split_apart(a->status, 4, a->ids, picks) &&
           split_apart(a->status, 4, a->weights, picks);
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
        if (a->m >= kTiledMinRows) {
            const dim3 grid((rows + kTileExperts * kTileWarpsE - 1) / (kTileExperts * kTileWarpsE),
                            (a->m + kTileTokens * kTileWarpsT - 1) / (kTileTokens * kTileWarpsT));
            router_logits_tiled_k<<<grid, kTileThreads, 0, s>>>(
                (const float *)a->x, (const float *)a->w, (const float *)a->w_shared,
                (float *)a->logits, a->hidden, a->n_expert, a->m, a->n_expert + 1);
        } else {
            const dim3 grid((rows + kLogitWarps - 1) / kLogitWarps, a->m);
            router_logits_k<<<grid, kLogitWarps * 32, 0, s>>>(
                (const float *)a->x, (const float *)a->w, (const float *)a->w_shared,
                (float *)a->logits, a->hidden, a->n_expert, a->n_expert + 1);
        }
        failure.record(cudaGetLastError(), INFERENCE_CUDA_STAGE_LAUNCH);
        if (failure.ok()) {
            router_select_k<false><<<a->m, 32, 0, s>>>((const float *)a->logits, (int *)a->ids,
                                                (float *)a->weights, (float *)a->shared_gate,
                                                (int *)a->status, a->n_expert, a->top_k, a->n_expert + 1);
            failure.record(cudaGetLastError(), INFERENCE_CUDA_STAGE_LAUNCH);
        }
        report_stage(out_stage, failure.stage);
        return (int)failure.err;
    } catch (...) {
        return INFERENCE_CUDA_NATIVE_EXCEPTION;
    }
}

extern "C" int inference_cuda_router_logits_launch(const inference_cuda_router_logits_args *a,
                                                   void *stream, int *out_stage) {
    try {
        report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
        if (!stream || !valid_split_logits(a)) return INFERENCE_CUDA_INVALID_ARGUMENT;
        const cudaStream_t s = (cudaStream_t)stream;
        if (a->m >= kTiledMinRows) {
            const dim3 grid((a->n_expert + kTileExperts * kTileWarpsE - 1) / (kTileExperts * kTileWarpsE),
                            (a->m + kTileTokens * kTileWarpsT - 1) / (kTileTokens * kTileWarpsT));
            router_logits_tiled_k<<<grid, kTileThreads, 0, s>>>(
                a->x, a->w, NULL, a->logits, a->hidden, a->n_expert, a->m, a->n_expert);
        } else {
            const dim3 grid((a->n_expert + kLogitWarps - 1) / kLogitWarps, a->m);
            router_logits_k<<<grid, kLogitWarps * 32, 0, s>>>(
                a->x, a->w, NULL, a->logits, a->hidden, a->n_expert, a->n_expert);
        }
        Failure failure;
        failure.record(cudaGetLastError(), INFERENCE_CUDA_STAGE_LAUNCH);
        report_stage(out_stage, failure.stage);
        return int(failure.err);
    } catch (...) {
        return INFERENCE_CUDA_NATIVE_EXCEPTION;
    }
}

extern "C" int inference_cuda_router_sigmoid_select_launch(const inference_cuda_router_sigmoid_args *a,
                                                           void *stream, int *out_stage) {
    try {
        report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
        if (!stream || !valid_split_select(a)) return INFERENCE_CUDA_INVALID_ARGUMENT;
        router_select_k<true><<<a->m, 32, 0, (cudaStream_t)stream>>>(
            a->logits, a->ids, a->weights, NULL, a->status, a->n_expert, a->top_k, a->n_expert);
        Failure failure;
        failure.record(cudaGetLastError(), INFERENCE_CUDA_STAGE_LAUNCH);
        report_stage(out_stage, failure.stage);
        return int(failure.err);
    } catch (...) {
        return INFERENCE_CUDA_NATIVE_EXCEPTION;
    }
}
