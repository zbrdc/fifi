// CPF-DN-B K1/K2: the value-sliced sequential gated-delta scan over one chunk and the gated
// output RMSNorm (decision 0049 design row 9, P-a; K1 restructured by CPF-DN-B). Design
// guidance only, no code copied: the per-token math and its expression order are the vendored
// kern::gated_delta (vendor/xavier-qwen/src/ops.cu gated_delta_k, the user's own xavier-qwen
// code, RD-4V); the launch idea (the per-column loop reading q/k rows that are already
// L2-normalized and gates that are already computed, one warp per value column with 4 state
// rows per lane) was read from the pinned llama.cpp gated_delta_net.cu for geometry only —
// here the norms and gates are computed once per block into shared memory instead of once
// per column warp. K1: grid (H, 1, D/32), block (32, 8), one block owns 32 value columns
// and scans kTile-token tiles; the blocks share nothing (their state and o_out columns are
// disjoint), so a multi-wave launch is still correct (one wave on the 3090; sm_72/sm_89 run
// multi-wave). fp32 SIMT only: no MMA, no fp16, so the same source compiles unchanged for
// sm_72. Contract: prefill/gdn_scan_seq.h.
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
constexpr int kLanes = 32;           // threads per warp
// CPF-DN-B's launch (section 2.1): one block owns kBlockCols value columns of one head, so
// the grid is (n_v_heads, 1, D / kBlockCols); within a warp, kGroupLanes lanes share one
// column and the warp serves kLanes / kGroupLanes = 4 columns.
constexpr int kTile = 16;            // tokens staged per block iteration (section 2.2)
constexpr int kWarps = 8;            // warps per block (threadIdx.y)
constexpr int kGroupLanes = 8;       // lanes sharing one value column (g = lane & 7)
constexpr int kBlockCols = 32;       // value columns per block: kWarps * 4
static_assert(kTile == 2 * kWarps,
              "P1/P3 stage exactly the two slots j = w and j = w + kWarps, so any other "
              "kTile leaves slots unstaged (kTile = 32 would never stage 16-31)");
static_assert(kTile <= kLanes,
              "P2 stages one gate per lane of warp 1 (g_s[lane] under lane < len), so the "
              "tile can hold at most kLanes gate slots");
static_assert(kBlockCols == kLanes,
              "P3 stages one v column per lane (v_s[j][lane]), so the block owns exactly "
              "kLanes columns");
static_assert(
    kGroupLanes == 8,
    "g = lane & 7, cg = lane >> 3 and group_sum's xor tree (offsets 4, 2, 1) each assume "
    "one consecutive-8 lane octet per column");
static_assert(D % kBlockCols == 0, "the grid's z blocks tile the D columns exactly");
static_assert(D % kGroupLanes == 0, "the lane groups tile the D state rows exactly");
static_assert(kBlockCols == kWarps * kLanes / kGroupLanes, "one warp serves 4 columns");
static_assert(D / kGroupLanes == 4 * 4, "16 state entries per lane: s[4][4]");

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

// Sum of v over the kGroupLanes lanes of one column group. The xor offsets 4, 2 and 1 stay
// inside each consecutive-8 lane octet, so every lane of the group holds the same bits of
// the sum (IEEE addition commutes bitwise among the pairwise combines).
__device__ float group_sum(float v) {
#pragma unroll
    for (int o = 4; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
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

// K1 (CPF-DN-B): one block of kWarps warps owns kBlockCols = 32 value columns of one head
// (D / kBlockCols = 4 blocks per head, 128 blocks for Qwen3.6: one wave on the 3090). Within
// a warp, g = lane & 7 picks the row group and cg = lane >> 3 the column, so 8 lanes share
// one column and the warp serves 4 columns; each lane holds the 16 state entries at rows
// i = 32k + 4g + c (k, c in 0..3) of its column in registers for the whole chunk (state
// loaded once before the tile loop, written back once after it). The block-uniform tile
// loop stages kTile = 16 tokens per trip — P writes the L2-normalized q/k rows, the
// (decay, beta) gates and the v row to shared memory once per block — then a barrier, then
// S scans the slots with A's elementwise expressions, then a barrier before the next tile
// overwrites the stage. No thread leaves before the last barrier. The summation order of the
// two 128-term dots differs from CPF-DN-A's (four accumulators, a pairwise combine and a
// 3-level tree over the 8 lanes of a column, against one accumulator and a 5-level tree over
// 32 lanes); the rounding depth is 9 in both.
__global__ void __launch_bounds__(kLanes * kWarps, 2) gdn_scan_seq_k(ScanArgs a) {
    // qk_s is float4 so the type itself guarantees the 16-byte alignment of S's LDS.128
    // loads; P1's scalar stores go through the float view qk_f (element (j, plane, e) at
    // qk_f[(j * 2 + plane) * D + e]; S's float4 for rows 32k + 4g..+3 at index 8 * k + g).
    __shared__ float4 qk_s[kTile][2][D / 4];
    __shared__ float v_s[kTile][kBlockCols];
    __shared__ float2 g_s[kTile];
    float *const qk_f = reinterpret_cast<float *>(qk_s);
    const int h = blockIdx.x, kh = h % a.n_k_heads;
    const int lane = threadIdx.x, w = threadIdx.y;
    const int g = lane & 7, cg = lane >> 3;
    const int col = blockIdx.z * kBlockCols + w * 4 + cg;
    float *st = a.state + (size_t)h * D * D;
    // The device re-check of the launch arguments (decision 0049 item 2): block uniform,
    // before the state load and the first barrier. A refused launch loads no state, updates
    // nothing, NaNs its o_out columns for every t and ORs mask 8 into the optional status
    // word (item 10); the g == 0 lanes hold the block's 32 columns (one per warp), so every
    // NaN write hits one column exactly once.
    if ((long long)a.p0 < 0 || (long long)a.p0 + a.m > a.ctx_tokens) {
        if (a.status && blockIdx.x == 0 && blockIdx.z == 0 && threadIdx.x == 0 &&
            threadIdx.y == 0) {
            atomicOr(a.status, INFERENCE_CUDA_PREFILL_STATUS_BOUND_VIOLATION);
        }
        if (g == 0) {
            for (int t = 0; t < a.m; ++t) {
                a.o_out[((size_t)t * a.n_v_heads + h) * D + col] =
                    __int_as_float(0x7fc00000);
            }
        }
        return;
    }
    const float a_h = a.a[h], dt_h = a.dt_bias[h];
    float s[4][4];
#pragma unroll
    for (int k = 0; k < 4; ++k) {
#pragma unroll
        for (int c = 0; c < 4; ++c) {
            s[k][c] = st[(size_t)(32 * k + 4 * g + c) * D + col];
        }
    }
    for (int t0 = 0; t0 < a.m; t0 += kTile) {
        const int len = min(kTile, a.m - t0);
        // P1, the normalized q/k rows, once per block per token: warp w stages slots j = w
        // and j = w + kWarps in CPF-DN-A's lane order and A's expression order. Every global
        // load of the tile (P1's two slots, P3's two v values, P2's two logits) is issued
        // before the first shuffle of the norms, so no warp serializes a global round trip
        // between its loads (Amendment 1, MINOR 4).
        const float *row_w = a.conv + (size_t)(t0 + w) * a.conv_ld;
        // The slot-(w + kWarps) row pointer is clamped to the last conv row: on a short
        // last tile that row sits past the conv object, and even forming the pointer would
        // be undefined (review item n4); the clamped value is read only when live_w8,
        // where the clamp is a no-op (t0 + w + kWarps < m). The sibling row_w is
        // deliberately left unclamped: when the last tile holds fewer than 8 slots
        // (m = 1, 65), row_w of a dead warp is still formed past the conv object and is
        // never dereferenced — kept so that a lost live_w guard stays visible to memcheck
        // on exact-size uploads.
        const float *row_w8 =
            a.conv + (size_t)min(t0 + w + kWarps, a.m - 1) * a.conv_ld;
        const bool live_w = w < len, live_w8 = w + kWarps < len;
        float qv_w[4], kv_w[4], qv_w8[4], kv_w8[4];
        if (live_w) {
#pragma unroll
            for (int r = 0; r < 4; ++r) {
                qv_w[r] = row_w[kh * D + r * kLanes + lane];
                kv_w[r] = row_w[(a.n_k_heads + kh) * D + r * kLanes + lane];
            }
        }
        if (live_w8) {
#pragma unroll
            for (int r = 0; r < 4; ++r) {
                qv_w8[r] = row_w8[kh * D + r * kLanes + lane];
                kv_w8[r] = row_w8[(a.n_k_heads + kh) * D + r * kLanes + lane];
            }
        }
        // P3, the v row of the block's 32 columns: warp w stages slots j = w and j =
        // w + kWarps, one 128-byte coalesced row segment per warp instruction.
        const int v_base = 2 * a.n_k_heads * D + h * D + blockIdx.z * kBlockCols;
        float v_w = 0.f, v_w8 = 0.f;
        if (live_w) v_w = row_w[v_base + lane];
        if (live_w8) v_w8 = row_w8[v_base + lane];
        // P2, the gates, once per block per token: warp 1, one lane per slot (off warp 0,
        // whose P1/P3 loads must not queue behind the gate round trips).
        float alpha_l = 0.f, beta_l = 0.f;
        if (w == 1 && lane < len) {
            const float *p = a.proj + (size_t)(t0 + lane) * a.ld;
            alpha_l = p[a.alpha_off + h];
            beta_l = p[a.beta_off + h];
        }
        // P's reductions and shared stores.
        if (live_w) {
            // L2 norms in the ggml form (norm.cu:273), the norm and q's 1/sqrt(D) folded
            // into one multiplier (the vendored form of ops.cu:122): A's expressions
            // verbatim, now run once per slot instead of once per slice warp.
            float sq = 0.f, ss = 0.f;
#pragma unroll
            for (int r = 0; r < 4; ++r) {
                sq += qv_w[r] * qv_w[r];
                ss += kv_w[r] * kv_w[r];
            }
            const float qmul =
                rsqrtf(fmaxf(warp_sum(sq), a.eps * a.eps)) * rsqrtf((float)D);
            const float kmul = rsqrtf(fmaxf(warp_sum(ss), a.eps * a.eps));
#pragma unroll
            for (int r = 0; r < 4; ++r) {
                qk_f[(w * 2) * D + r * kLanes + lane] = qv_w[r] * qmul;
                qk_f[(w * 2 + 1) * D + r * kLanes + lane] = kv_w[r] * kmul;
            }
        }
        if (live_w8) {
            float sq = 0.f, ss = 0.f;
#pragma unroll
            for (int r = 0; r < 4; ++r) {
                sq += qv_w8[r] * qv_w8[r];
                ss += kv_w8[r] * kv_w8[r];
            }
            const float qmul =
                rsqrtf(fmaxf(warp_sum(sq), a.eps * a.eps)) * rsqrtf((float)D);
            const float kmul = rsqrtf(fmaxf(warp_sum(ss), a.eps * a.eps));
#pragma unroll
            for (int r = 0; r < 4; ++r) {
                qk_f[((w + kWarps) * 2) * D + r * kLanes + lane] = qv_w8[r] * qmul;
                qk_f[((w + kWarps) * 2 + 1) * D + r * kLanes + lane] = kv_w8[r] * kmul;
            }
        }
        if (w == 1 && lane < len) {
            // Gates from the projection row, in kernel; a is ssm_a as stored (no second
            // exp). A's expressions verbatim, with a[h] and dt_bias[h] read once before
            // the tile loop.
            g_s[lane] = make_float2(expf(softplus_e(alpha_l + dt_h) * a_h),
                                    1.f / (1.f + expf(-beta_l)));
        }
        if (live_w) v_s[w][lane] = v_w;
        if (live_w8) v_s[w + kWarps][lane] = v_w8;
        __syncthreads();  // P's writes are visible to S
        // S, the scan, per warp over the staged slots: the gates as one broadcast LDS.64, v
        // as one LDS.32, and k-hat/q-hat as 4 + 4 LDS.128 (float4 index 8 * k + g: the 8
        // lanes of a group read the same 16 bytes and the 4 groups of the warp read the
        // same 128 bytes per k). q-hat is loaded before the sk reduction so its latency
        // hides behind it. State update in the design's order: decay first, the sk dot
        // against the DECAYED state, then the rank-one write and the o dot (ops.cu:132-139).
        for (int j = 0; j < len; ++j) {
            const float2 gb = g_s[j];
            const float vv = v_s[j][w * 4 + cg];
            float khat[4][4], qhat[4][4];
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                const float4 kf = qk_s[j][1][8 * k + g];
                const float4 qf = qk_s[j][0][8 * k + g];
                khat[k][0] = kf.x;
                khat[k][1] = kf.y;
                khat[k][2] = kf.z;
                khat[k][3] = kf.w;
                qhat[k][0] = qf.x;
                qhat[k][1] = qf.y;
                qhat[k][2] = qf.z;
                qhat[k][3] = qf.w;
            }
            float acc[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
            for (int k = 0; k < 4; ++k) {
#pragma unroll
                for (int c = 0; c < 4; ++c) {
                    s[k][c] *= gb.x;
                    acc[c] += s[k][c] * khat[k][c];
                }
            }
            const float sk = group_sum((acc[0] + acc[1]) + (acc[2] + acc[3]));
            const float d = (vv - sk) * gb.y;
            float oacc[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
            for (int k = 0; k < 4; ++k) {
#pragma unroll
                for (int c = 0; c < 4; ++c) {
                    s[k][c] += khat[k][c] * d;
                    oacc[c] += s[k][c] * qhat[k][c];
                }
            }
            const float o = group_sum((oacc[0] + oacc[1]) + (oacc[2] + oacc[3]));
            if (g == 0) a.o_out[((size_t)(t0 + j) * a.n_v_heads + h) * D + col] = o;
        }
        __syncthreads();  // the next tile's P overwrites qk_s, g_s and v_s
    }
#pragma unroll
    for (int k = 0; k < 4; ++k) {
#pragma unroll
        for (int c = 0; c < 4; ++c) {
            st[(size_t)(32 * k + 4 * g + c) * D + col] = s[k][c];
        }
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
    gdn_scan_seq_k<<<dim3(k.n_v_heads, 1, D / kBlockCols), dim3(kLanes, kWarps, 1), 0,
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
