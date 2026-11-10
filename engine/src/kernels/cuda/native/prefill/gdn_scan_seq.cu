// The value-sliced sequential gated-delta scan over one chunk (gdn_scan_seq_k) and the gated
// output RMSNorm (gdn_out_gate_k). The per-token math and its expression order are the
// vendored kern::gated_delta (vendor/xavier-qwen/src/ops.cu gated_delta_k); the launch idea
// (the per-column loop reading q/k rows that are already L2-normalized and gates that are
// already computed, one warp per value column with 4 state rows per lane) was read from the
// pinned llama.cpp gated_delta_net.cu for geometry only — here the norms and gates are
// computed once per block into shared memory instead of once per column warp. The scan: grid
// (H, 1, D/32), block (32, 8), one block owns 32 value columns and scans kTile-token tiles;
// the blocks share nothing (their state and o_out columns are disjoint), so a multi-wave
// launch is still correct (one wave on the 3090; sm_72/sm_89 run multi-wave). fp32 SIMT
// only: no MMA, no fp16, so the same source compiles unchanged for sm_72. Contract:
// prefill/gdn_scan_seq.h. gdn_scan_seq_cols_k further down is the same scan with a block width of
// 8, 16 or 32 value columns (the launch above is its 32-column case); the per-column arithmetic is shared.
#include "prefill/gdn_scan_seq.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stddef.h>
#include <stdint.h>

#include "abi.h"
#include "device.h"
#include "resident/gdn_mutants.h"

namespace {

constexpr int kMaxM = 32768;         // the vendored kMaxM (gdn_conv.cu:16 / gdn_delta.cu:16)
constexpr int kMaxKHeads = 64;       // the decode scan's policy limits (gdn_delta.cu:14-15)
constexpr int kMaxVHeads = 128;
constexpr int kMaxLd = 1 << 22;      // gdn_delta.cu:41-52
constexpr int kMaxCtxTokens = (1 << 30) - 1;
constexpr float kEps = 1e-6f;        // the vendored pin (gdn_delta.cu:21)
constexpr int D = 128;               // the compiled head size (kern::kStateDim)
constexpr int kLanes = 32;           // threads per warp
// The launch: one block owns kBlockCols value columns of one head, so the grid is
// (n_v_heads, 1, D / kBlockCols); within a warp, kGroupLanes lanes share one column and the
// warp serves kLanes / kGroupLanes = 4 columns.
constexpr int kTile = 16;            // tokens staged per block iteration
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
#ifdef INFERENCE_CUDA_GDN_MUTANTS
    int mutant;  // the active mutant of resident/gdn_mutants.h; 0 is none
#endif
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

/* The p0 bound the wrapper, this entry and the kernels all check, the sum in 64 bits. */
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
        a.n_v_heads % a.n_k_heads != 0 || a.eps != kEps ||
        (a.form & ~INFERENCE_CUDA_GDN_FORM_ALL) != 0) {
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

#if defined(INFERENCE_CUDA_FAULT_QUALIFICATION) || defined(INFERENCE_CUDA_GDN_MUTANTS)
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
#ifdef INFERENCE_CUDA_GDN_MUTANTS
    k.mutant = gdn_mutant_current();
#endif
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

// The three form rules of the header, resolved at compile time so that each form is its own
// kernel instance and the form-0 instance is the code that existed before the bits.
template <bool kGrouped>
__device__ __forceinline__ int key_head(int h, const ScanArgs &a) {
    if constexpr (kGrouped) {
        return h / (a.n_v_heads / a.n_k_heads);
    } else {
        return h % a.n_k_heads;
    }
}

template <bool kAddEps>
__device__ __forceinline__ float l2_den(float sum, float eps) {
    if constexpr (kAddEps) {
        return sum + eps;
    } else {
        return fmaxf(sum, eps * eps);
    }
}

template <bool kSigmoid>
__device__ __forceinline__ float out_gate(float z) {
    if constexpr (kSigmoid) {
        return 1.f / (1.f + expf(-z));
    } else {
        return silu_e(z);
    }
}

// The scan: one block of kWarps warps owns kBlockCols = 32 value columns of one head
// (D / kBlockCols = 4 blocks per head, 128 blocks for Qwen3.6: one wave on the 3090). Within
// a warp, g = lane & 7 picks the row group and cg = lane >> 3 the column, so 8 lanes share
// one column and the warp serves 4 columns; each lane holds the 16 state entries at rows
// i = 32k + 4g + c (k, c in 0..3) of its column in registers for the whole chunk (state
// loaded once before the tile loop, written back once after it). The block-uniform tile
// loop stages kTile = 16 tokens per trip — P writes the L2-normalized q/k rows, the
// (decay, beta) gates and the v row to shared memory once per block — then a barrier, then
// S scans the slots with the header's elementwise expressions, then a barrier before the next
// tile overwrites the stage. No thread leaves before the last barrier. The summation order of
// the two 128-term dots differs from a one-warp-per-column scan's (four accumulators, a
// pairwise combine and a 3-level tree over the 8 lanes of a column, against one accumulator
// and a 5-level tree over 32 lanes); the rounding depth is 9 in both.
template <bool kGrouped, bool kAddEps>
__global__ void __launch_bounds__(kLanes * kWarps, 2) gdn_scan_seq_k(ScanArgs a) {
    // qk_s is float4 so the type itself guarantees the 16-byte alignment of S's LDS.128
    // loads; P1's scalar stores go through the float view qk_f (element (j, plane, e) at
    // qk_f[(j * 2 + plane) * D + e]; S's float4 for rows 32k + 4g..+3 at index 8 * k + g).
    __shared__ float4 qk_s[kTile][2][D / 4];
    __shared__ float v_s[kTile][kBlockCols];
    __shared__ float2 g_s[kTile];
    float *const qk_f = reinterpret_cast<float *>(qk_s);
    const int h = blockIdx.x, kh = key_head<kGrouped>(h, a);
    const int lane = threadIdx.x, w = threadIdx.y;
    const int g = lane & 7, cg = lane >> 3;
    const int col = blockIdx.z * kBlockCols + w * 4 + cg;
    float *st = a.state + (size_t)h * D * D;
    // The device re-check of the launch arguments: block uniform, before the state load and
    // the first barrier. A refused launch loads no state, updates nothing, NaNs its o_out
    // columns for every t and ORs mask 8 into the optional status word; the g == 0 lanes
    // hold the block's 32 columns (one per warp), so every NaN write hits one column exactly
    // once.
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
        // and j = w + kWarps, lane l holding elements 32r + l, in the header's expression
        // order. Every global load of the tile (P1's two slots, P3's two v values, P2's two
        // logits) is issued before the first shuffle of the norms, so no warp serializes a
        // global round trip between its loads.
        const float *row_w = a.conv + (size_t)(t0 + w) * a.conv_ld;
        // The slot-(w + kWarps) row pointer is clamped to the last conv row: on a short
        // last tile that row sits past the conv object, and even forming the pointer would
        // be undefined; the clamped value is read only when live_w8, where the clamp is a
        // no-op (t0 + w + kWarps < m). The sibling row_w is deliberately left unclamped:
        // when the last tile holds fewer than 8 slots (m = 1, 65), row_w of a dead warp is
        // still formed past the conv object and is never dereferenced — kept so that a lost
        // live_w guard stays visible to memcheck on exact-size uploads.
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
            // into one multiplier (the vendored form of ops.cu:122): the header's
            // expressions verbatim, run once per slot instead of once per slice warp.
            float sq = 0.f, ss = 0.f;
#pragma unroll
            for (int r = 0; r < 4; ++r) {
                sq += qv_w[r] * qv_w[r];
                ss += kv_w[r] * kv_w[r];
            }
            const float qmul = rsqrtf(l2_den<kAddEps>(warp_sum(sq), a.eps)) * rsqrtf((float)D);
            const float kmul = rsqrtf(l2_den<kAddEps>(warp_sum(ss), a.eps));
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
            const float qmul = rsqrtf(l2_den<kAddEps>(warp_sum(sq), a.eps)) * rsqrtf((float)D);
            const float kmul = rsqrtf(l2_den<kAddEps>(warp_sum(ss), a.eps));
#pragma unroll
            for (int r = 0; r < 4; ++r) {
                qk_f[((w + kWarps) * 2) * D + r * kLanes + lane] = qv_w8[r] * qmul;
                qk_f[((w + kWarps) * 2 + 1) * D + r * kLanes + lane] = kv_w8[r] * kmul;
            }
        }
        if (w == 1 && lane < len) {
            // Gates from the projection row, in kernel; a is ssm_a as stored (no second
            // exp). The header's expressions verbatim, with a[h] and dt_bias[h] read once
            // before the tile loop.
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
        // hides behind it. State update in the vendored order: decay first, the sk dot
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

// The same scan with a selectable block width: a block owns C = 4 * kW value columns of one
// head, one warp per 4 columns, so C = 32 is the launch above and C = 16 and C = 8 split each head
// into 8 and 16 blocks. Per column nothing changes: the lane mapping (g = lane & 7, cg = lane >> 3),
// the 16 state entries per lane, the order of every product and sum and the two group_sum trees
// are those of gdn_scan_seq_k, so the state and o_out bytes are the same for every C. What C
// changes is the grid (more, smaller blocks, which the scheduler can spread over the SMs more
// evenly) and the staging: a block stages the same kTile tokens with fewer warps, so each warp
// stages kTile / kW slots (slot = w + i * kW) and a narrower block re-reads each q/k row once
// per block of its head.
template <int kW, bool kGrouped, bool kAddEps>
__global__ void __launch_bounds__(kLanes * kW, kTile / kW) gdn_scan_seq_cols_k(ScanArgs a) {
    constexpr int kCols = kW * 4;
#ifdef INFERENCE_CUDA_GDN_MUTANTS
    const int mutant = a.mutant;
#else
    constexpr int mutant = 0;  // a mutant branch is the constant false and compiles away
#endif
    constexpr int kSlots = kTile / kW;  // slots staged by each warp
    static_assert(kTile % kW == 0 && kW >= 2 && kW <= 8, "kW warps must tile kTile and hold warp 1");
    __shared__ float4 qk_s[kTile][2][D / 4];
    __shared__ float v_s[kTile][kCols];
    __shared__ float2 g_s[kTile];
    float *const qk_f = reinterpret_cast<float *>(qk_s);
    const int h = blockIdx.x, kh = key_head<kGrouped>(h, a);
    const int lane = threadIdx.x, w = threadIdx.y;
    const int g = lane & 7, cg = lane >> 3;
    const int col = blockIdx.z * (mutant == 6 ? 32 : kCols) + w * 4 + cg;
    float *st = a.state + (size_t)h * D * D;
    if (mutant != 23 && ((long long)a.p0 < 0 || (long long)a.p0 + a.m > a.ctx_tokens)) {
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
        // Slot i of this warp is token t0 + w + i * kW. Every global load of the tile is issued
        // before the first shuffle of the norms, as in gdn_scan_seq_k. The row pointer is
        // clamped to the last conv row so that a dead slot never forms a pointer past the object.
        float qv[kSlots][4], kv[kSlots][4], vv_l[kSlots];
        const int v_base = 2 * a.n_k_heads * D + h * D + blockIdx.z * (mutant == 7 ? 32 : kCols);
#pragma unroll
        for (int i = 0; i < kSlots; ++i) {
            const int slot = w + i * kW;
            const float *row = a.conv + (size_t)min(t0 + slot, a.m - 1) * a.conv_ld;
            if (slot < len) {
#pragma unroll
                for (int r = 0; r < 4; ++r) {
                    qv[i][r] = row[kh * D + r * kLanes + lane];
                    kv[i][r] = row[(a.n_k_heads + kh) * D + r * kLanes + lane];
                }
            }
            vv_l[i] = 0.f;
            if (slot < len && lane < kCols) vv_l[i] = row[v_base + lane];
        }
        float alpha_l = 0.f, beta_l = 0.f;
        if (w == 1 && lane < len) {
            const float *p = a.proj + (size_t)(t0 + lane) * a.ld;
            alpha_l = p[a.alpha_off + h];
            beta_l = p[a.beta_off + h];
        }
#pragma unroll
        for (int i = 0; i < kSlots; ++i) {
            const int slot = w + i * kW;
            if (slot < len) {
                float sq = 0.f, ss = 0.f;
#pragma unroll
                for (int r = 0; r < 4; ++r) {
                    sq += qv[i][r] * qv[i][r];
                    ss += kv[i][r] * kv[i][r];
                }
                const float qmul =
                    rsqrtf(l2_den<kAddEps>(warp_sum(sq), a.eps)) * rsqrtf((float)D);
                const float kmul = rsqrtf(l2_den<kAddEps>(warp_sum(ss), a.eps));
#pragma unroll
                for (int r = 0; r < 4; ++r) {
                    qk_f[(slot * 2) * D + r * kLanes + lane] = qv[i][r] * qmul;
                    qk_f[(slot * 2 + 1) * D + r * kLanes + lane] = kv[i][r] * kmul;
                }
            }
        }
        if (w == 1 && lane < len) {
            g_s[lane] = make_float2(expf(softplus_e(alpha_l + dt_h) * a_h),
                                    1.f / (1.f + expf(-beta_l)));
        }
#pragma unroll
        for (int i = 0; i < kSlots; ++i) {
            const int slot = w + i * kW;
            if (slot < len && lane < kCols && !(mutant == 8 && i != 0)) v_s[slot][lane] = vv_l[i];
        }
        __syncthreads();
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
            const float sk = group_sum(mutant == 10 ? (acc[0] + acc[2]) + (acc[1] + acc[3])
                                                  : (acc[0] + acc[1]) + (acc[2] + acc[3]));
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
        if (mutant != 20) __syncthreads();
    }
#pragma unroll
    for (int k = 0; k < 4; ++k) {
#pragma unroll
        for (int c = 0; c < 4; ++c) {
            st[(size_t)(32 * k + 4 * g + c) * D + col] =
                (mutant == 11 && kW == 4 && h == 0) ? 0.f : s[k][c];
        }
    }
}

// The gated output RMSNorm of m rows, in place on the o_out rows gdn_scan_seq_k wrote (the
// vendored mean+eps form, ops.cu:141; the ggml L2 form applies to the q/k norms only). One
// block per (token, head), one thread per column; each block reads its D values before
// writing them.
template <bool kSigmoid>
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
    row[j] = o * rstd * a.norm_w[j] * out_gate<kSigmoid>(p[a.z_off + h * D + j]);
}

// ---- the chunk-parallel WY scan -----------------------------------------------------------
//
// The sequential kernels above evaluate the per-token recurrence in token order, so every
// token pays a D×D decay, a D×D rank-one write and two D-dots before the next token can
// start. gdn_scan_chunk_k evaluates the SAME recurrence (its scalar reference is
// kernels/gdn_reference.rs chunk_step, pinned there against step at f64 exactness, both
// directions of every form bit) one sub-chunk of kC tokens at a time: within a sub-chunk
// the token loop becomes the WY form — forward substitution over kC rank-one coefficient
// rows U (u_t = b_t·(v_t − B_t·Sᵀk_t) − Σ_{s<t} beta_t·(B_t/B_s)·(k_t·k_s)·u_s, with B the
// prefix product of the decays), the masked output sum (o_t = B_t·(q_tᵀS) + Σ_{s≤t}
// (B_t/B_s)·(q_t·k_s)·u_s against the sub-chunk's start state) and ONE decayed rank-kC
// state update per sub-chunk. The gates, the q/k L2 norms, the p0 guard, the state layout
// and the written bytes are those of the sequential kernels, so the same driver call and
// the same out-gate kernel (K2 above) serve it unchanged.
//
// Launch: grid (n_v_heads, 1, D / kCols), block (kLanes, kChunkWarps) = 512 threads; one
// block owns kCols value columns of one head and walks the m rows in kC-token
// sub-chunks. All hot reads are 128-bit: K̂/Q̂/S rows live with row stride D + 4 (a multiple
// of four floats, so rows stay 16-byte aligned and the +4 varies the bank across the eight
// lanes of an LDS.128 phase group), and every U/G/A/seen matrix is stored value-major with
// row stride kC + 4 for the same reason. Per sub-chunk: stage K̂/Q̂/V and the gates (P),
// the two kC×kC coefficient matrices (G), all kC state-seen dots as one parallel pass (M1:
// seen = B_t·Sᵀk̂_t for every (t, c) pair), the forward substitution as a lean serial pass
// (serial: one lane per value column, warps 0–1, each token one broadcast G read, one
// per-lane seen read and a t-term in-lane back sum — no cross-lane traffic at all), the
// outputs (M2: B_t·q̂ᵀS + the masked A·U sum, both operands float4 streams) and the single
// decayed rank-kC state update across all threads (update). The ratio B_t/B_s is a running
// product per Gram element, never a division: B_t itself underflows fp32 around 0.5^32
// while the ratios stay in range. No thread leaves before the barrier of the phase it reads.
//
// The rounding of this form differs from the sequential kernels' (deeper dot trees inside
// the solve against none in the token loop); both are pinned to the same f64 reference by
// their tests, each with its own derived bound. No mutants yet: the mutation bank targets
// the sequential kernels; adding the chunk instances is deferred until the kernel earns a
// production default.
constexpr int kC = 32;          // sub-chunk tokens (the solve's kC-step chain)
constexpr int kChunkWarps = 16; // warps per block (512 threads; the serial pass uses 2)
constexpr int kCols = 64;       // value columns per block (grid.z = D / kCols = 2: 32 blocks)
constexpr int kDpad = D + 4;    // row stride of the r-major rows (16-byte aligned, bank-varying)
constexpr int kCpad = kC + 4;   // row stride of the value-major matrices
constexpr int kHpad = 128;      // halves per r-major row: exactly D, XOR-swizzled 16B chunks
__device__ __forceinline__ int swr(int row, int r) {
    // chunk j of row k lives at chunk j ^ (k & 7): ldmatrix's 8-row loads span all 32
    // banks once each, with no pad column.
    return row * kHpad + (((r >> 3) ^ (row & 7)) << 3) + (r & 7);
}
constexpr int kCHpad = 40;      // halves per value-major split row of U: kC + 8, 80B bank-varying 16B rows
// The shared bytes of one launch: fp16 hi/lo split rows for K̂, Q̂ and the state slice S —
// the tensor-core form (a split pair reconstructs to ~2^-22 relative, so every fp32 reader
// below joins hi+lo exactly) — plus the fp32 value-major matrices U, G, A, seen and the
// three gate vectors.
constexpr size_t kChunkSmem =
    (4 * kC + 2 * kCols) * kHpad * sizeof(__half) +
    (2 * kC + 4 * kCols) * kCHpad * sizeof(__half) +
    (kC * kCpad + 3 * kC) * sizeof(float);
static_assert(kChunkSmem < 96 * 1024, "sm_72 caps opt-in shared memory at 96KB per block");
static_assert(kHpad >= D, "r-major half rows must hold a full D row");
static_assert(kHpad % 8 == 0, "swizzled r-major rows need 16B chunks");


// Split a float into an fp16 hi/lo pair (joint residual ~2^-22 relative) and join it back.
__device__ __forceinline__ void f2s(float x, __half *hi, __half *lo) {
    const __half h = __float2half_rn(x);
    *hi = h;
    *lo = __float2half_rn(x - __half2float(h));
}
__device__ __forceinline__ float j2(const __half *hi, const __half *lo, int i) {
    return __half2float(hi[i]) + __half2float(lo[i]);
}
__device__ __forceinline__ void f2s_at(float x, __half *hib, __half *lob, int i) {
    const __half h = __float2half_rn(x);
    hib[i] = h;
    lob[i] = __float2half_rn(x - __half2float(h));
}

#if __CUDA_ARCH__ >= 800
// The tensor-core primitives of the chunk kernel's matmul phases, in the house style of
// attn_prefill.cu (sm_80+; sm_72 builds never compile them and take the joined SIMT paths).
__device__ __forceinline__ unsigned smem_addr(const void *p) {
    return (unsigned)__cvta_generic_to_shared(p);
}
// The A fragment (16x16 halves, row-major source): lanes 0-7 name the rows of the
// (rows 0-7, cols 0-7) block, lanes 8-15 rows 8-15, lanes 16-23 the col+8 halves of rows
// 0-7, lanes 24-31 the same of rows 8-15.
template <int Stride>
__device__ __forceinline__ void ldm_a(const __half *m, int row, int col, unsigned (&r)[4]) {
    const int lr = row + (threadIdx.x & 7) + 8 * ((threadIdx.x >> 3) & 1);
    const int lc = col + 8 * (threadIdx.x >> 4);
    const unsigned addr = smem_addr(m + (size_t)lr * Stride + lc);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(addr));
}
// The swizzled variants for the r-major split family (K-hat, Q-hat, S).
template <int Stride>
__device__ __forceinline__ void ldm_as(const __half *m, int row, int col, unsigned (&r)[4]) {
    const int lr = row + (threadIdx.x & 7) + 8 * ((threadIdx.x >> 3) & 1);
    const int lc = col + 8 * (threadIdx.x >> 4);
    const unsigned addr = smem_addr(m + swr(lr, lc));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(addr));
}
// The B fragment (16x8 halves, from a row-major [n][k] source): the non-trans load already
// lands lane l on {B[2(l%4)][n0+l/4], B[2(l%4)+1][n0+l/4]} — the trans form is for [k][n]
// sources. Lanes 0-7 name source rows n0..n0+7 at k, lanes 8-15 the same rows at k+8.
template <int Stride>
__device__ __forceinline__ void ldm_b(const __half *m, int row, int col, unsigned (&r)[2]) {
    const int lr = row + (threadIdx.x & 7);
    const int lc = col + 8 * ((threadIdx.x >> 3) & 1);
    const unsigned addr = smem_addr(m + (size_t)lr * Stride + lc);
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
                 : "=r"(r[0]), "=r"(r[1])
                 : "r"(addr));
}
template <int Stride>
__device__ __forceinline__ void ldm_bs(const __half *m, int row, int col, unsigned (&r)[2]) {
    const int lr = row + (threadIdx.x & 7);
    const int lc = col + 8 * ((threadIdx.x >> 3) & 1);
    const unsigned addr = smem_addr(m + swr(lr, lc));
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
                 : "=r"(r[0]), "=r"(r[1])
                 : "r"(addr));
}
// The trans B load for a row-major [k][n] source (the update's K-hat rows), swizzled:
// the x2 load's two 8x8 tiles stack along k, so the 16 lanes address 16 source rows at
// one n chunk — not the same 8 rows at two chunks like the non-trans form.
template <int Stride>
__device__ __forceinline__ void ldm_bts(const __half *m, int row, int col, unsigned (&r)[2]) {
    const int lr = row + (threadIdx.x & 15);
    const unsigned addr = smem_addr(m + swr(lr, col));
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
                 : "=r"(r[0]), "=r"(r[1])
                 : "r"(addr));
}
// D (16x8 f32) += A (16x16 f16, row) * B (16x8 f16, col).
__device__ __forceinline__ void mma_16816(float *d, const unsigned *a, unsigned b0, unsigned b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
#endif

template <bool kGrouped, bool kAddEps>
__global__ void __launch_bounds__(kLanes * kChunkWarps, 1) gdn_scan_chunk_k(ScanArgs a) {
    extern __shared__ __align__(16) unsigned char sm[];  // the layout of kChunkSmem, in order
    __half *const kh_h = reinterpret_cast<__half *>(sm);  // K̂ hi [kC][kHpad]
    __half *const kh_l = kh_h + kC * kHpad;               // K̂ lo
    __half *const q_h = kh_l + kC * kHpad;                // Q̂ hi [kC][kHpad]
    __half *const q_l = q_h + kC * kHpad;                 // Q̂ lo
    __half *const S_h = q_l + kC * kHpad;                 // the state slice hi [kCols][kHpad]
    __half *const S_l = S_h + kCols * kHpad;              // the state slice lo
    __half *const U_h = S_l + kCols * kHpad;              // U hi [kCols][kCHpad], value-major: staged V, folded and solved in place
    __half *const U_l = U_h + kCols * kCHpad;             // U lo
    __half *const A_h = U_l + kCols * kCHpad;             // A hi [kC][kCHpad] (upper zero, rows ≥ len zero)
    __half *const A_l = A_h + kC * kCHpad;                // A lo
    __half *const UB_h = A_l + kC * kCHpad;              // tail-ratio-scaled U hi [kCols][kCHpad]
    __half *const UB_l = UB_h + kCols * kCHpad;          // ... lo (zero past the ragged tail)
    float *const G_s = reinterpret_cast<float *>(UB_l + kCols * kCHpad);  // G[t][s] = b_t·(B_t/B_s)·(k_t·k_s), s < t
    float *const dec_s = G_s + kC * kCpad;         // the decay gates a_t
    float *const b_s = dec_s + kC;                 // the beta gates
    float *const B_s = b_s + kC;                   // the prefix products B_t
    const int h = blockIdx.x, kh = key_head<kGrouped>(h, a);
    const int lane = threadIdx.x, w = threadIdx.y;
    const int tid = lane + kLanes * w;
    const int col0 = blockIdx.z * kCols;
    float *const st = a.state + (size_t)h * D * D;
    if ((long long)a.p0 < 0 || (long long)a.p0 + a.m > a.ctx_tokens) {
        if (a.status && blockIdx.x == 0 && blockIdx.z == 0 && tid == 0) {
            atomicOr(a.status, INFERENCE_CUDA_PREFILL_STATUS_BOUND_VIOLATION);
        }
        for (int i = tid; i < a.m * kCols; i += kLanes * kChunkWarps) {
            a.o_out[((size_t)(i / kCols) * a.n_v_heads + h) * D + col0 + i % kCols] =
                __int_as_float(0x7fc00000);
        }
        return;
    }
    const float a_h = a.a[h], dt_h = a.dt_bias[h];
    // The state slice loads and stores run over consecutive value columns per thread, so the
    // global traffic is coalesced; the shared layout is value-major because every hot phase
    // walks r or s with the lane pinned to one value column.
    for (int i = tid; i < D * kCols; i += kLanes * kChunkWarps) {
        const int c = i % kCols, r = i / kCols;
        f2s(st[(size_t)r * D + col0 + c], &S_h[swr(c, r)], &S_l[swr(c, r)]);
    }
    // A's strict upper triangle and its padding stay zero for the whole launch: M2 reads
    // A rows in 128-bit steps past the diagonal, and the zeros swallow the unwritten U
    // entries of a short tail. G's upper and padding stay zero for the same reason: the
    // serial pass reads its rows in 128-bit steps past the diagonal.
    for (int i = tid; i < kC * kCpad; i += kLanes * kChunkWarps) {
        const int t = i / kCpad, s = i % kCpad;
        if (s > t) G_s[i] = 0.f;
    }
    // U and the A splits stay zero outside the written window for the whole launch: the
    // phases that walk the full kC window pair those slots with zeros (finite halves),
    // and Gram overwrites the written lower triangle of the A splits every sub-chunk.
    for (int i = tid; i < kCols * kCHpad; i += kLanes * kChunkWarps) {
        U_h[i] = __half(0);
        U_l[i] = __half(0);
    }
    for (int i = tid; i < kC * kCHpad; i += kLanes * kChunkWarps) {
        A_h[i] = __half(0);
        A_l[i] = __half(0);
    }
    __syncthreads();
    for (int t0 = 0; t0 < a.m; t0 += kC) {
        const int len = min(kC, a.m - t0);
        // P, the staging: warp w stages slots j = w + i·kChunkWarps — the normalized q/k rows
        // (the same expressions as the sequential kernels, run once per slot), this block's
        // V columns into U (value-major), and (warp 0, one lane per slot) the gates with the
        // prefix products.
        for (int i = 0; i < kC / kChunkWarps; ++i) {
            const int j = w + i * kChunkWarps;
            const int row = min(t0 + j, a.m - 1);  // the clamp keeps the loads coalesced
            const float *rw = a.conv + (size_t)row * a.conv_ld;
            float qv[4], kv[4];
#pragma unroll
            for (int r = 0; r < 4; ++r) {
                qv[r] = rw[kh * D + r * kLanes + lane];
                kv[r] = rw[(a.n_k_heads + kh) * D + r * kLanes + lane];
            }
            float sq = 0.f, ss = 0.f;
#pragma unroll
            for (int r = 0; r < 4; ++r) {
                sq += qv[r] * qv[r];
                ss += kv[r] * kv[r];
            }
            const float qmul =
                rsqrtf(l2_den<kAddEps>(warp_sum(sq), a.eps)) * rsqrtf((float)D);
            const float kmul = rsqrtf(l2_den<kAddEps>(warp_sum(ss), a.eps));
#pragma unroll
            for (int r = 0; r < 4; ++r) {
                const int wq = swr(j, r * kLanes + lane);
                f2s(qv[r] * qmul, &q_h[wq], &q_l[wq]);
                const int wk = swr(j, r * kLanes + lane);
                f2s(kv[r] * kmul, &kh_h[wk], &kh_l[wk]);
            }
#pragma unroll
            for (int cv = 0; cv < (kCols + kLanes - 1) / kLanes; ++cv) {
                if (cv * kLanes + lane < kCols) {
                    f2s_at(
                        rw[2 * a.n_k_heads * D + h * D + col0 + cv * kLanes + lane], U_h, U_l,
                        (cv * kLanes + lane) * kCHpad + j);
                }
            }
        }
        if (w == 0) {
            // One lane per slot reads its projection row; lanes beyond len carry the
            // neutral 1 of the product scan, so every prefix below len is exact.
            float d = 1.f, bb = 0.f;
            if (lane < len) {
                const float *p = a.proj + (size_t)(t0 + lane) * a.ld;
                d = expf(softplus_e(p[a.alpha_off + h] + dt_h) * a_h);
                bb = 1.f / (1.f + expf(-p[a.beta_off + h]));
            }
            dec_s[lane] = d;
            b_s[lane] = bb;
            float pre = d;
#pragma unroll
            for (int o = 1; o < kLanes; o <<= 1) {
                const float up = __shfl_up_sync(0xffffffffu, pre, o);
                if (lane >= o) pre *= up;
            }
            B_s[lane] = pre;
        }
        __syncthreads();  // P's writes are visible to G, M1 and the serial pass
        // G, the two kC×kC coefficient matrices. sm_80+: both dots on the tensor cores —
        // warps 0-7 own the kk tiles (K-hat rows × K-hatᵀ), warps 8-15 the qk tiles (Q-hat
        // rows × K-hatᵀ; K-hatᵀ comes from the same row-major splits through the non-trans B
        // load), and each lane folds its own running ratio product — never a division: B_t
        // underflows fp32 long before the ratio does. sm_72 keeps the joined element walk.
#if __CUDA_ARCH__ >= 800
        {
            const int t0w = 16 * ((w & 7) >> 2), s0w = 8 * (w & 3);
            float d[4] = {0.f, 0.f, 0.f, 0.f};
            const __half *ar_h = (w < 8) ? kh_h : q_h;
            const __half *ar_l = (w < 8) ? kh_l : q_l;
            for (int r0 = 0; r0 < D; r0 += 16) {
                unsigned aH[4], aL[4], bH[2], bL[2];
                ldm_as<kHpad>(ar_h, t0w, r0, aH);
                ldm_as<kHpad>(ar_l, t0w, r0, aL);
                ldm_bs<kHpad>(kh_h, s0w, r0, bH);
                ldm_bs<kHpad>(kh_l, s0w, r0, bL);
                mma_16816(d, aH, bH[0], bH[1]);
                mma_16816(d, aH, bL[0], bL[1]);
                mma_16816(d, aL, bH[0], bH[1]);
            }
            const bool is_kk = w < 8;
            const int r0 = lane >> 2, cc = 2 * (lane & 3);
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const int t = t0w + r0 + 8 * (e >> 1);
                const int s = s0w + cc + (e & 1);
                if (s > t || t >= len) continue;
                float ratio = 1.f;
                for (int i = s + 1; i <= t; ++i) ratio *= dec_s[i];
                if (s != t) {
                    if (is_kk) {
                        G_s[t * kCpad + s] = b_s[t] * ratio * d[e];
                    } else {
                        f2s_at(ratio * d[e], A_h, A_l, t * kCHpad + s);
                    }
                } else if (!is_kk) {
                    f2s_at(d[e], A_h, A_l, t * kCHpad + t);
                }
            }
        }
#else
        for (int e = tid; e < kC * (kC + 1) / 2; e += kLanes * kChunkWarps) {
            int t, s;
            if (e < kC) {
                t = e;
                s = t;
            } else {
                const int k = e - kC;
                t = 1;
                // the strict-lower pairs in row order: (1,0), (2,0), (2,1), (3,0)… — t is the
                // smallest row whose cumulative count t(t+1)/2 exceeds k.
                while (t * (t + 1) / 2 <= k) ++t;
                s = k - t * (t - 1) / 2;
            }
            float ratio = 1.f;
            for (int i = s + 1; i <= t; ++i) ratio *= dec_s[i];
            float kk = 0.f, qk = 0.f;
            for (int r = 0; r < D; ++r) {
                const float ktr = j2(kh_h, kh_l, swr(t, r));
                const float ksr = j2(kh_h, kh_l, swr(s, r));
                kk += ktr * ksr;
                qk += j2(q_h, q_l, swr(t, r)) * ksr;
            }
            if (s != t) {
                G_s[t * kCpad + s] = b_s[t] * ratio * kk;
                f2s_at(ratio * qk, A_h, A_l, t * kCHpad + s);
            } else {
                f2s_at(qk, A_h, A_l, t * kCHpad + t);
            }
        }
#endif
        __syncthreads();  // G's writes are visible to the serial pass and M2
        // M1, every state-seen term of the sub-chunk in one pass. sm_80+: one tensor-core
        // tile per warp — warp w owns the 16x8 (t, c) block t0 = 16·(w>>3), c0 = 8·(w&7);
        // three mma products per 16-half k step cover the split (the lo·lo term sits below
        // the residual the EC_S derivation already counts). sm_72 keeps the joined SIMT walk.
#if __CUDA_ARCH__ >= 800
        {
            const int t0w = 16 * (w >> 3), c0w = 8 * (w & 7);
            float d[4] = {0.f, 0.f, 0.f, 0.f};
            for (int r0 = 0; r0 < D; r0 += 16) {
                unsigned aH[4], aL[4], bH[2], bL[2];
                ldm_as<kHpad>(kh_h, t0w, r0, aH);
                ldm_as<kHpad>(kh_l, t0w, r0, aL);
                ldm_bs<kHpad>(S_h, c0w, r0, bH);
                ldm_bs<kHpad>(S_l, c0w, r0, bL);
                mma_16816(d, aH, bH[0], bH[1]);
                mma_16816(d, aH, bL[0], bL[1]);
                mma_16816(d, aL, bH[0], bH[1]);
            }
            // D scatter: lane l owns rows l/4 and l/4+8, columns 2·(l%4) and +1.
            const int r0 = lane >> 2, cc = 2 * (lane & 3);
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const int t = t0w + r0 + 8 * (e >> 1);
                const int c = c0w + cc + (e & 1);
                if (t < len)
                    f2s_at(j2(U_h, U_l, c * kCHpad + t) - B_s[t] * d[e], U_h, U_l,
                           c * kCHpad + t);
            }
        }
#else
        {
            const int c = tid % kCols, tg = tid / kCols;
            const int tgs = kLanes * kChunkWarps / kCols;
            for (int i = 0; i < kC / tgs; ++i) {
                const int t = tg + tgs * i;
                if (t >= len) break;
                float acc = 0.f;
                for (int r = 0; r < D; ++r)
                    acc += j2(kh_h, kh_l, swr(t, r)) * j2(S_h, S_l, swr(c, r));
                f2s_at(j2(U_h, U_l, c * kCHpad + t) - B_s[t] * acc, U_h, U_l,
                       c * kCHpad + t);
            }
        }
#endif
        __syncthreads();  // M1's writes are visible to the serial pass
        // The forward substitution, one lane per value column (the first kCols/32 warps):
        // every read is the lane's own U/seen entry or a broadcast G coefficient, so the
        // kC-step chain carries no cross-lane traffic at all.
        const int cl = w * kLanes + lane;
        if (cl < kCols) {
            const int c = cl;
            for (int t = 0; t < len; ++t) {
                const float4 *g = reinterpret_cast<const float4 *>(G_s + t * kCpad);
                float4 back = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll 2
                for (int s = 0; s < t / 4; ++s) {
                    const float4 gs = g[s];
                    const int b4 = c * kCHpad + 4 * s;
                    back.x += gs.x * j2(U_h, U_l, b4);
                    back.y += gs.y * j2(U_h, U_l, b4 + 1);
                    back.z += gs.z * j2(U_h, U_l, b4 + 2);
                    back.w += gs.w * j2(U_h, U_l, b4 + 3);
                }
                float tail = 0.f;
                for (int s = 4 * (t / 4); s < t; ++s)
                    tail += G_s[t * kCpad + s] * j2(U_h, U_l, c * kCHpad + s);
                float b = back.x + back.y + back.z + back.w + tail;
                const float u = b_s[t] * j2(U_h, U_l, c * kCHpad + t) - b;
                f2s_at(u, U_h, U_l, c * kCHpad + t);
            }
        }
        __syncthreads();  // the solved U rows are visible to M2 and the update
        // M2, the outputs. sm_80+: both terms on the tensor cores in one warp tile — the
        // inter pass B_t·q̂ᵀS (M1's shape, Q̂ in the A slot) and the intra pass A·U over the
        // full kC window (A's zeroed upper swallows the masked tail) — fused into one
        // scatter, one global write per element. sm_72 keeps the joined SIMT walk.
#if __CUDA_ARCH__ >= 800
        {
            const int t0w = 16 * (w >> 3), c0w = 8 * (w & 7);
            float di[4] = {0.f, 0.f, 0.f, 0.f}, da[4] = {0.f, 0.f, 0.f, 0.f};
            for (int r0 = 0; r0 < D; r0 += 16) {
                unsigned aH[4], aL[4], bH[2], bL[2];
                ldm_as<kHpad>(q_h, t0w, r0, aH);
                ldm_as<kHpad>(q_l, t0w, r0, aL);
                ldm_bs<kHpad>(S_h, c0w, r0, bH);
                ldm_bs<kHpad>(S_l, c0w, r0, bL);
                mma_16816(di, aH, bH[0], bH[1]);
                mma_16816(di, aH, bL[0], bL[1]);
                mma_16816(di, aL, bH[0], bH[1]);
            }
            for (int s0 = 0; s0 < kC; s0 += 16) {
                unsigned aH[4], aL[4], bH[2], bL[2];
                ldm_a<kCHpad>(A_h, t0w, s0, aH);
                ldm_a<kCHpad>(A_l, t0w, s0, aL);
                ldm_b<kCHpad>(U_h, c0w, s0, bH);
                ldm_b<kCHpad>(U_l, c0w, s0, bL);
                mma_16816(da, aH, bH[0], bH[1]);
                mma_16816(da, aH, bL[0], bL[1]);
                mma_16816(da, aL, bH[0], bH[1]);
            }
            const int r0 = lane >> 2, cc = 2 * (lane & 3);
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const int t = t0w + r0 + 8 * (e >> 1);
                const int c = c0w + cc + (e & 1);
                if (t < len)
                    a.o_out[((size_t)(t0 + t) * a.n_v_heads + h) * D + col0 + c] =
                        B_s[t] * di[e] + da[e];
            }
        }
#else
        {
            const int c = tid % kCols, tg = tid / kCols;
            const int tgs = kLanes * kChunkWarps / kCols;
            for (int i = 0; i < kC / tgs; ++i) {
                const int t = tg + tgs * i;
                if (t >= len) break;
                float inter = 0.f;
                for (int r = 0; r < D; ++r)
                    inter += j2(q_h, q_l, swr(t, r)) * j2(S_h, S_l, swr(c, r));
                float intra = 0.f;
                for (int s = 0; s <= t; ++s)
                    intra += j2(A_h, A_l, t * kCHpad + s) * j2(U_h, U_l, c * kCHpad + s);
                a.o_out[((size_t)(t0 + t) * a.n_v_heads + h) * D + col0 + c] =
                    B_s[t] * inter + intra;
            }
        }
#endif
        __syncthreads();  // M2's reads of S, A and U are done before the update writes S
        // The sub-chunk's single state update: S ← B_end·S + Σ_s (B_end/B_s)·k_s u_sᵀ, with
        // the tail ratios staged into B_s by thread 0 (B_s's prefix products are spent).
        if (tid == 0) {
            float tail = 1.f;
            for (int s = len - 1; s >= 0; --s) {
                B_s[s] = tail;
                tail *= dec_s[s];
            }
        }
        __syncthreads();
        const float b_end = B_s[0] * dec_s[0];
#if __CUDA_ARCH__ >= 800
        // Stage the tail-ratio-scaled U as splits; rows past the ragged tail pair the
        // stale K-hat clones with exact zeros, so the full-window contraction is clean.
        for (int i = tid; i < kCols * kC; i += kLanes * kChunkWarps) {
            const int c = i / kC, s = i % kC;
            f2s_at(s < len ? B_s[s] * j2(U_h, U_l, c * kCHpad + s) : 0.f, UB_h, UB_l,
                   c * kCHpad + s);
        }
        __syncthreads();
        {
            // the update on the tensor cores: S ← b_end·S + K̂ᵀ·UB as m16n8k16 tiles —
            // warp w owns the c rows c0w = 16·(w>>2) and the r tiles r0w = 8·(w&3) + 32·i,
            // with UB's value-major rows in the A fragments and K-hat's [s][r] rows in the
            // trans B fragments. The epilogue folds b_end·S_old in and re-splits S in place.
            const int c0w = 16 * (w >> 2);
            for (int i = 0; i < D / 32; ++i) {
                const int r0w = 8 * (w & 3) + 32 * i;
                float d[4] = {0.f, 0.f, 0.f, 0.f};
                for (int s0 = 0; s0 < kC; s0 += 16) {
                    unsigned aH[4], aL[4], bH[2], bL[2];
                    ldm_a<kCHpad>(UB_h, c0w, s0, aH);
                    ldm_a<kCHpad>(UB_l, c0w, s0, aL);
                    ldm_bts<kHpad>(kh_h, s0, r0w, bH);
                    ldm_bts<kHpad>(kh_l, s0, r0w, bL);
                    mma_16816(d, aH, bH[0], bH[1]);
                    mma_16816(d, aH, bL[0], bL[1]);
                    mma_16816(d, aL, bH[0], bH[1]);
                }
                const int r0 = lane >> 2, cc = 2 * (lane & 3);
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    const int c = c0w + r0 + 8 * (e >> 1);
                    const int r = r0w + cc + (e & 1);
                    f2s_at(j2(S_h, S_l, swr(c, r)) * b_end + d[e], S_h, S_l, swr(c, r));
                }
            }
        }
#else
        {
            // thread (c, r): U and B are warp broadcasts, K̂'s column is the lane's own r —
            // one full-width shared read per term.
            const int r = tid & (D - 1), cg = tid >> 7;
            for (int i = 0; i < kCols / 4; ++i) {
                const int c = cg + 4 * i;
                float acc = 0.f;
                for (int s = 0; s < len; ++s) {
                    acc += B_s[s] * j2(kh_h, kh_l, swr(s, r)) * j2(U_h, U_l, c * kCHpad + s);
                }
                f2s(j2(S_h, S_l, swr(c, r)) * b_end + acc, &S_h[swr(c, r)],
                    &S_l[swr(c, r)]);
            }
        }
#endif
        __syncthreads();  // the update's writes are visible to the next sub-chunk's phases
    }
    for (int i = tid; i < D * kCols; i += kLanes * kChunkWarps) {
        const int c = i % kCols, r = i / kCols;
        st[(size_t)r * D + col0 + c] = j2(S_h, S_l, swr(c, r));
    }
}

#ifdef INFERENCE_CUDA_GDN_MUTANTS
bool host_mutant_nine(const ScanArgs &k) { return k.mutant == 9; }
#else
constexpr bool host_mutant_nine(const ScanArgs &) { return false; }
#endif

// The scan launch of `cols` value columns per block: 0 is the original kernel gdn_scan_seq_k,
// 8, 16 and 32 are gdn_scan_seq_cols_k (the caller has checked the value).
template <int kW, bool kGrouped, bool kAddEps>
void launch_cols(const ScanArgs &k, void *stream) {
    gdn_scan_seq_cols_k<kW, kGrouped, kAddEps>
        <<<dim3(k.n_v_heads, 1, D / (host_mutant_nine(k) ? 32 : 4 * kW)), dim3(kLanes, kW, 1), 0,
          (cudaStream_t)stream>>>(k);
}

template <bool kGrouped, bool kAddEps>
void launch_scan_kernel(const ScanArgs &k, int cols, void *stream) {
    switch (cols) {
    case 8: launch_cols<2, kGrouped, kAddEps>(k, stream); break;
    case 16: launch_cols<4, kGrouped, kAddEps>(k, stream); break;
    case 32: launch_cols<8, kGrouped, kAddEps>(k, stream); break;
    default:
        gdn_scan_seq_k<kGrouped, kAddEps>
            <<<dim3(k.n_v_heads, 1, D / kBlockCols), dim3(kLanes, kWarps, 1), 0,
               (cudaStream_t)stream>>>(k);
    }
}

template <bool kGrouped, bool kAddEps, bool kSigmoid>
Failure launch_form(const ScanArgs &k, int cols, void *stream) {
    Failure failure;
    launch_scan_kernel<kGrouped, kAddEps>(k, cols, stream);
    failure.record(cudaGetLastError(), INFERENCE_CUDA_STAGE_LAUNCH);
    if (failure.err == cudaSuccess) {
        gdn_out_gate_k<kSigmoid><<<dim3(k.n_v_heads, k.m), D, 0, (cudaStream_t)stream>>>(k);
        failure.record(cudaGetLastError(), INFERENCE_CUDA_STAGE_LAUNCH);
    }
    return failure;
}

template <bool kGrouped, bool kAddEps>
Failure launch_gate(const ScanArgs &k, int form, int cols, void *stream) {
    return (form & INFERENCE_CUDA_GDN_FORM_SIGMOID_GATE)
               ? launch_form<kGrouped, kAddEps, true>(k, cols, stream)
               : launch_form<kGrouped, kAddEps, false>(k, cols, stream);
}

template <bool kGrouped>
Failure launch_l2(const ScanArgs &k, int form, int cols, void *stream) {
    return (form & INFERENCE_CUDA_GDN_FORM_ADD_EPS_L2)
               ? launch_gate<kGrouped, true>(k, form, cols, stream)
               : launch_gate<kGrouped, false>(k, form, cols, stream);
}

/* The instance of `form` (bits already validated) with `cols` value columns per block (0: the
 * original kernel). */
Failure launch_scan(const ScanArgs &k, int form, int cols, void *stream) {
    return (form & INFERENCE_CUDA_GDN_FORM_GROUPED_HEADS)
               ? launch_l2<true>(k, form, cols, stream)
               : launch_l2<false>(k, form, cols, stream);
}

template <bool kGrouped, bool kAddEps, bool kSigmoid>
Failure launch_chunk_form(const ScanArgs &k, void *stream) {
    Failure failure;
    // The opt-in of the block's dynamic shared memory, once per instantiation: sm_72 and
    // later accept kChunkSmem (static_assert above); a failure surfaces as the launch
    // error below, which the wrapper maps to a DeviceFailure.
    static const bool opted_in = [] {
        cudaFuncSetAttribute(gdn_scan_chunk_k<kGrouped, kAddEps>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, (int)kChunkSmem);
        return true;
    }();
    (void)opted_in;
    gdn_scan_chunk_k<kGrouped, kAddEps>
        <<<dim3(k.n_v_heads, 1, D / kCols), dim3(kLanes, kChunkWarps), kChunkSmem,
           (cudaStream_t)stream>>>(k);
    failure.record(cudaGetLastError(), INFERENCE_CUDA_STAGE_LAUNCH);
    if (failure.err == cudaSuccess) {
        gdn_out_gate_k<kSigmoid><<<dim3(k.n_v_heads, k.m), D, 0, (cudaStream_t)stream>>>(k);
        failure.record(cudaGetLastError(), INFERENCE_CUDA_STAGE_LAUNCH);
    }
    return failure;
}

template <bool kGrouped, bool kAddEps>
Failure launch_chunk_gate(const ScanArgs &k, int form, void *stream) {
    return (form & INFERENCE_CUDA_GDN_FORM_SIGMOID_GATE)
               ? launch_chunk_form<kGrouped, kAddEps, true>(k, stream)
               : launch_chunk_form<kGrouped, kAddEps, false>(k, stream);
}

template <bool kGrouped>
Failure launch_chunk_l2(const ScanArgs &k, int form, void *stream) {
    return (form & INFERENCE_CUDA_GDN_FORM_ADD_EPS_L2)
               ? launch_chunk_gate<kGrouped, true>(k, form, stream)
               : launch_chunk_gate<kGrouped, false>(k, form, stream);
}

}  // namespace

extern "C" int inference_cuda_gdn_scan_chunk_launch(
    const inference_cuda_gdn_scan_seq_args *args, void *stream, int *out_stage) {
    try {
        report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
        if (args == NULL || !valid(*args, stream)) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        const ScanArgs k = to_args(args);
        const Failure failure = (args->form & INFERENCE_CUDA_GDN_FORM_GROUPED_HEADS)
                                    ? launch_chunk_l2<true>(k, args->form, stream)
                                    : launch_chunk_l2<false>(k, args->form, stream);
        report_stage(out_stage, failure.stage);
        return (int)failure.err;
    } catch (...) {
        return INFERENCE_CUDA_NATIVE_EXCEPTION;
    }
}

extern "C" int inference_cuda_gdn_scan_seq_launch(
    const inference_cuda_gdn_scan_seq_args *args, void *stream, int *out_stage) {
    try {
        report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
        if (args == NULL || !valid(*args, stream)) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        const Failure failure = launch_scan(to_args(args), args->form, 0, stream);
        report_stage(out_stage, failure.stage);
        return (int)failure.err;
    } catch (...) {
        return INFERENCE_CUDA_NATIVE_EXCEPTION;
    }
}

extern "C" int inference_cuda_gdn_scan_seq_cols_launch(
    const inference_cuda_gdn_scan_seq_args *args, void *stream, int *out_stage, int block_cols) {
    try {
        report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
        if (args == NULL || !valid(*args, stream) ||
            (block_cols != 8 && block_cols != 16 && block_cols != 32)) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        const Failure failure = launch_scan(to_args(args), args->form, block_cols, stream);
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
        const Failure failure = launch_scan(k, args->form, 0, stream);
        report_stage(out_stage, failure.stage);
        return (int)failure.err;
    } catch (...) {
        return INFERENCE_CUDA_NATIVE_EXCEPTION;
    }
}
#endif

#ifdef INFERENCE_CUDA_GDN_MUTANTS
/* The bound-violation launch of gdn_scan_seq_cols_k for the mutation tests: every rule of the entry holds except the
 * p0 bound, which is violated by force after validation, so the kernel's own device re-check is the code under test
 * (NaN rows, mask 8, no state load). The same shape as the fault-qualification entry of the original kernel, with a
 * block width. */
extern "C" int inference_cuda_gdn_scan_seq_cols_bad_bounds_launch(
    const inference_cuda_gdn_scan_seq_args *args, void *stream, int *out_stage, int block_cols) {
    try {
        report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
        if (args == NULL || !valid_sans_bounds(*args, stream) ||
            (block_cols != 8 && block_cols != 16 && block_cols != 32)) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        ScanArgs k = to_args(args);
        k.p0 = k.ctx_tokens;
        const Failure failure = launch_scan(k, args->form, block_cols, stream);
        report_stage(out_stage, failure.stage);
        return (int)failure.err;
    } catch (...) {
        return INFERENCE_CUDA_NATIVE_EXCEPTION;
    }
}
#endif
