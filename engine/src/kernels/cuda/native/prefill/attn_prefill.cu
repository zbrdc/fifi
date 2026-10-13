/* CPF-ATTN-A K2: the two kernels and the launch entry of attn_prefill, the flash-style
 * chunked-prefill attention (decision 0049 items 1, 2 and 10; the CPF-5a/5b kernel of design
 * section 3.7). Design guidance only, no code copied: the split rule, the visible/cache_row
 * span rules and the combine pattern restate the resident attn_decode (RD-8b, this tree);
 * the online-softmax recurrence, the query-tile geometry (8 tokens x 8 GQA heads = 64 rows,
 * key tiles of 32, K/V double-buffered in 96 KB of shared memory, the O accumulator in 128
 * registers per thread over 128 threads) follow design section 3.7. The device checks p0/m
 * again (item 2) and ORs bit 8 of the prefill status word (item 10).
 * Contract: prefill/attn_prefill.h. */
#include "prefill/attn_prefill.h"

#include <cuda_fp16.h>

#include <math.h>
#include <stdint.h>

#include "abi.h"
#include "device.h"

namespace {

constexpr int kHeadDim = 256;   // the only head size (the wrapper refuses every other value)
constexpr int kGroup = 8;       // query heads per kv head: the compiled group
constexpr int kTile = 32;       // keys per tile (the granule of a key range)
constexpr int kQTile = 8;       // tokens per query tile: the compiled tile
constexpr int kRows = kQTile * kGroup;  // 64 rows per query tile
constexpr int kMaxGridY = 65535;
constexpr int kMaxRows = (1 << 30) - 1;
constexpr int kMaxSplits = 128;
constexpr int kMaxMinKeys = 1 << 20;
/* Dynamic shared memory of the part kernel: the Q tile (64 x 256 fp16 = 32 KB) plus two
 * double-buffered K and V tiles (2 x 16 KB each) = 96 KB, above the 48 KB default (the entry
 * opts in once per process, the vendored gemm.cuh pattern). The sm_72 SIMT arm uses the Q
 * tile and one K/V stage (64 KB) of the same allocation. */
constexpr int kSmemBytes = kRows * kHeadDim * 2 + 2 * (2 * kTile * kHeadDim * 2);

struct Args {
    const float *q;
    const float *proj;  // NULL (no gate) or [m][ld]
    const __half *k_cache;
    const __half *v_cache;
    float *part;
    float *out;
    int *status;
    int ld, p0, m, n_head, n_kv_head, cache_rows, max_splits, min_keys, window, ring;
    float scale;
};

/* Key range plan from the number of keys alone: attn_decode's split_plan verbatim
 * (attn_decode.cu:66-72). n_split splits of `chunk` keys (a multiple of the tile) cover
 * n_keys; only n_active splits hold keys. Every block and the combine recompute it from the
 * launch arguments, so no device word carries it. */
__device__ inline void split_plan(int n_keys, int max_splits, int min_keys, int &chunk,
                                  int &n_active) {
    const int want = (n_keys + min_keys - 1) / min_keys;
    const int n_split = min(max(want, 1), max_splits);
    chunk = (((n_keys + n_split - 1) / n_split) + kTile - 1) / kTile * kTile;
    n_active = (n_keys + chunk - 1) / chunk;
}

/* Visible keys of the query tile whose rows sit at positions p_first .. n_keys - 1: the union
 * of the rows' windows (attn_decode.cu:81-84 per row), from the first row's oldest visible
 * position to the last row's own; first is the position of visible key 0. sees() then cuts
 * each row to its own window. */
__device__ inline void visible(const Args &a, int p_first, int n_keys, int &first, int &count) {
    first = a.window > 0 ? max(0, p_first + 1 - a.window) : 0;
    count = n_keys - first;
}

/* Whether the row at position p sees the key at position pos: causal, and with a window,
 * among the row's last `window` positions (rev2 §2 K2). p - window cannot overflow: p >= 0. */
__device__ inline bool sees(const Args &a, int pos, int p) {
    return pos <= p && (a.window == 0 || pos > p - a.window);
}

/* Cache row of the i-th visible key (attn_decode.cu:87-89). */
__device__ inline int cache_row(const Args &a, int first, int i) {
    return a.ring > 0 ? (first + i) % a.ring : first + i;
}

/* The device re-check of the launch arguments (decision 0049 item 2): block-uniform, before
 * the first barrier. ORs bit 8 once; the caller's writes (or their absence) are the callers'. */
__device__ inline bool bounds_ok(const Args &a) {
    return (long long)a.p0 >= 0 && (long long)a.p0 + a.m <= a.cache_rows;
}

__device__ inline void report_bound_violation(const Args &a) {
    if (a.status && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && threadIdx.x == 0)
        atomicOr(a.status, INFERENCE_CUDA_PREFILL_STATUS_BOUND_VIOLATION);
}

/* Two adjacent fp16 values as one 32-bit register (little-endian: the first in the low half),
 * the operand form of mma.sync. */
__device__ inline unsigned pack_hh(__half lo, __half hi) {
    __half2 pair = __halves2half2(lo, hi);
    return *reinterpret_cast<unsigned *>(&pair);
}

/* D (16x8 f32) += A (16x16 f16, row) * B (16x8 f16, col), the sm_80 tensor-core path. */
__device__ inline void mma_16816(float *d, const unsigned *a, unsigned b0, unsigned b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

/* One block: one query tile (kRows rows, row = token-of-tile * 8 + head-of-group) of one kv
 * head and one key range of the tile's visible span. Writes one partial cell. */
__global__ void __launch_bounds__(128) attn_prefill_part_k(Args a) {
    if (!bounds_ok(a)) {
        report_bound_violation(a);
        return;  // no partial
    }
    extern __shared__ __half smem[];
    __half *sq = smem;                                        // [64][256]
    __half *sk[2] = {smem + kRows * kHeadDim,
                     smem + kRows * kHeadDim + kTile * kHeadDim};
    __half *sv[2] = {smem + kRows * kHeadDim + 2 * kTile * kHeadDim,
                     smem + kRows * kHeadDim + 3 * kTile * kHeadDim};
    const int s = blockIdx.x, kvh = blockIdx.y, tile = blockIdx.z;
    const int t0 = tile * kQTile;
    const int tile_end = min(a.m, t0 + kQTile);
    const int n_keys = a.p0 + tile_end;  // keys 0..p0 + tile_end - 1 exist for this tile
    int first, count;
    visible(a, a.p0 + t0, n_keys, first, count);
    int chunk, n_active;
    split_plan(count, a.max_splits, a.min_keys, chunk, n_active);
    if (s >= n_active) return;
    const int k_begin = s * chunk, k_end = min(count, k_begin + chunk);
    const int max_p = n_keys - 1;  // the tile's last row's own position: the deepest key read

    /* The Q tile, rounded to fp16 once: rows past the chunk's end (m not a multiple of 8)
     * hold zeros; their partials are written but never read. */
    for (int i = threadIdx.x; i < kRows * kHeadDim; i += 128) {
        const int r = i / kHeadDim, d = i % kHeadDim;
        const int t = t0 + (r >> 3);
        sq[i] = t < tile_end
                    ? __float2half(a.q[((size_t)t * a.n_head + kvh * kGroup + (r & 7)) * kHeadDim + d])
                    : __half(0);
    }

    /* Loads the key tile at visible index base into buffer idx. Keys beyond the range hold
     * zeros, so a masked or out-of-range key contributes exactly nothing (0 * V = 0) and a
     * NaN cell no row may read never enters the tile. */
    const auto load_tile = [&](int base, int idx) {
        const int nvalid = min(kTile, k_end - base);
        for (int i = threadIdx.x; i < kTile * kHeadDim; i += 128) {
            const int j = i / kHeadDim, d = i % kHeadDim;
            __half k = __half(0), v = __half(0);
            if (j < nvalid) {
                const size_t at = ((size_t)cache_row(a, first, base + j) * a.n_kv_head + kvh) * kHeadDim + d;
                k = a.k_cache[at];
                v = a.v_cache[at];
            }
            sk[idx][i] = k;
            sv[idx][i] = v;
        }
    };

    /* The partial cell of this block: 64 stat pairs then, after every cell's stats, this
     * cell's 64 x 256 accumulators (the decode partial layout with the rows rounded to the
     * tile). */
    const size_t n_cells = (size_t)((a.m + kQTile - 1) / kQTile) * a.n_kv_head * a.max_splits;
    const size_t cell = ((size_t)tile * a.n_kv_head + kvh) * a.max_splits + s;
    float *const stats = a.part + cell * (kRows * 2);
    float *const accs = a.part + n_cells * (kRows * 2) + cell * ((size_t)kRows * kHeadDim);

#if __CUDA_ARCH__ >= 800
    /* Tensor-core arm (sm_80+): four warps, warp w owning rows w*16 + 0..15; per lane g =
     * lane >> 2, c = (lane & 3) * 2. S = scale * Q̂·K^T lives in four 16x8 fragments (the
     * tile's 32 keys), O in 32 fragments of 16x8 (the 256 dims); the row statistics are
     * combined over each row's four lanes with two shuffles. */
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int g = lane >> 2, c = (lane & 3) * 2;
    const int r0 = warp * 16 + g, r1 = r0 + 8;              // the lane's two rows
    const int p0_row0 = a.p0 + t0 + (r0 >> 3);              // their absolute positions
    const int p0_row1 = a.p0 + t0 + (r1 >> 3);
    float s_f[4][4], o_f[kHeadDim / 8][4];
#pragma unroll
    for (int nb = 0; nb < kHeadDim / 8; ++nb)
#pragma unroll
        for (int e = 0; e < 4; ++e) o_f[nb][e] = 0.f;
    float m_run[2] = {-INFINITY, -INFINITY}, l_run[2] = {0.f, 0.f};

    load_tile(k_begin, 0);
    int loaded = 1;
    for (int base = k_begin; base < k_end && first + base <= max_p; base += kTile) {
        const int next = base + kTile;
        if (next < k_end && first + next <= max_p) load_tile(next, loaded & 1);
        __syncthreads();  // this tile visible; the previous compute done
        const __half *K = sk[(loaded - 1) & 1], *V = sv[(loaded - 1) & 1];
        /* Scores: 16 k-steps of 16 dims; the A fragments read the Q tile, the B fragments
         * read the K tile at (key = nb*8 + g, dims k0 + c .. +1 / +8 .. +9). */
#pragma unroll
        for (int nb = 0; nb < 4; ++nb)
#pragma unroll
            for (int e = 0; e < 4; ++e) s_f[nb][e] = 0.f;
        for (int k0 = 0; k0 < kHeadDim; k0 += 16) {
            const unsigned a_f[4] = {
                pack_hh(sq[r0 * kHeadDim + k0 + c], sq[r0 * kHeadDim + k0 + c + 1]),
                pack_hh(sq[r1 * kHeadDim + k0 + c], sq[r1 * kHeadDim + k0 + c + 1]),
                pack_hh(sq[r0 * kHeadDim + k0 + c + 8], sq[r0 * kHeadDim + k0 + c + 9]),
                pack_hh(sq[r1 * kHeadDim + k0 + c + 8], sq[r1 * kHeadDim + k0 + c + 9])};
#pragma unroll
            for (int nb = 0; nb < 4; ++nb) {
                const int key = nb * kGroup + g;
                const unsigned b0 = pack_hh(K[key * kHeadDim + k0 + c], K[key * kHeadDim + k0 + c + 1]);
                const unsigned b1 = pack_hh(K[key * kHeadDim + k0 + c + 8], K[key * kHeadDim + k0 + c + 9]);
                mma_16816(s_f[nb], a_f, b0, b1);
            }
        }
        /* Scale, the row's mask (keys after its own position or before its window contribute
         * nothing) and the row max over the lane's eight keys, then over the row's four lanes. */
        float mx[2] = {-INFINITY, -INFINITY};
#pragma unroll
        for (int nb = 0; nb < 4; ++nb) {
            const int pos = first + base + nb * kGroup + c;
            s_f[nb][0] = sees(a, pos, p0_row0) ? s_f[nb][0] * a.scale : -INFINITY;
            s_f[nb][1] = sees(a, pos + 1, p0_row0) ? s_f[nb][1] * a.scale : -INFINITY;
            s_f[nb][2] = sees(a, pos, p0_row1) ? s_f[nb][2] * a.scale : -INFINITY;
            s_f[nb][3] = sees(a, pos + 1, p0_row1) ? s_f[nb][3] * a.scale : -INFINITY;
            mx[0] = fmaxf(mx[0], fmaxf(s_f[nb][0], s_f[nb][1]));
            mx[1] = fmaxf(mx[1], fmaxf(s_f[nb][2], s_f[nb][3]));
        }
#pragma unroll
        for (int o = 1; o <= 2; o <<= 1) {
            mx[0] = fmaxf(mx[0], __shfl_xor_sync(0xffffffffu, mx[0], o));
            mx[1] = fmaxf(mx[1], __shfl_xor_sync(0xffffffffu, mx[1], o));
        }
        const float m_new[2] = {fmaxf(m_run[0], mx[0]), fmaxf(m_run[1], mx[1])};
        /* A row that has seen no key yet (every key so far masked: the split starts after the
         * row's own position, review BLOCKER-1) keeps m_new = -inf; it exponentiates against 0
         * instead, so p = 0, corr = 0 and l, O stay 0 rather than exp(-inf - -inf) = NaN. */
        const float m_ref[2] = {m_new[0] == -INFINITY ? 0.f : m_new[0],
                                m_new[1] == -INFINITY ? 0.f : m_new[1]};
        float sum[2] = {0.f, 0.f};
#pragma unroll
        for (int nb = 0; nb < 4; ++nb) {
            s_f[nb][0] = expf(s_f[nb][0] - m_ref[0]);
            s_f[nb][1] = expf(s_f[nb][1] - m_ref[0]);
            s_f[nb][2] = expf(s_f[nb][2] - m_ref[1]);
            s_f[nb][3] = expf(s_f[nb][3] - m_ref[1]);
            sum[0] += s_f[nb][0] + s_f[nb][1];
            sum[1] += s_f[nb][2] + s_f[nb][3];
        }
#pragma unroll
        for (int o = 1; o <= 2; o <<= 1) {
            sum[0] += __shfl_xor_sync(0xffffffffu, sum[0], o);
            sum[1] += __shfl_xor_sync(0xffffffffu, sum[1], o);
        }
        const float corr[2] = {expf(m_run[0] - m_ref[0]), expf(m_run[1] - m_ref[1])};
        l_run[0] = l_run[0] * corr[0] + sum[0];
        l_run[1] = l_run[1] * corr[1] + sum[1];
        m_run[0] = m_new[0];
        m_run[1] = m_new[1];
        /* O = O * corr + P̂·V̂: the A fragments are the probabilities (rounded to fp16) of
         * this tile's 32 keys, register-direct from the S fragments; the B fragments read
         * the V tile at (key = 16*ks + c + {0,1}, dims nb*8 + g). */
#pragma unroll
        for (int nb = 0; nb < kHeadDim / 8; ++nb) {
            o_f[nb][0] *= corr[0];
            o_f[nb][1] *= corr[0];
            o_f[nb][2] *= corr[1];
            o_f[nb][3] *= corr[1];
        }
#pragma unroll
        for (int ks = 0; ks < 2; ++ks) {
            /* The A register order of m16n8k16 is (row, k-block): a0 row g keys c..c+1,
             * a1 row g+8 keys c..c+1, a2 row g keys 8+c..8+c+1, a3 row g+8 keys 8+c..8+c+1. */
            const unsigned pa[4] = {
                pack_hh(__float2half(s_f[2 * ks][0]), __float2half(s_f[2 * ks][1])),
                pack_hh(__float2half(s_f[2 * ks][2]), __float2half(s_f[2 * ks][3])),
                pack_hh(__float2half(s_f[2 * ks + 1][0]), __float2half(s_f[2 * ks + 1][1])),
                pack_hh(__float2half(s_f[2 * ks + 1][2]), __float2half(s_f[2 * ks + 1][3]))};
#pragma unroll
            for (int nb = 0; nb < kHeadDim / 8; ++nb) {
                const int key = 16 * ks + c, dim = nb * kGroup + g;
                const unsigned b0 = pack_hh(V[key * kHeadDim + dim], V[(key + 1) * kHeadDim + dim]);
                const unsigned b1 =
                    pack_hh(V[(key + 8) * kHeadDim + dim], V[(key + 9) * kHeadDim + dim]);
                mma_16816(o_f[nb], pa, b0, b1);
            }
        }
        __syncthreads();  // compute done before the next load overwrites this buffer
        ++loaded;
    }
    if ((lane & 3) == 0) {
        stats[r0 * 2] = m_run[0];
        stats[r0 * 2 + 1] = l_run[0];
        stats[r1 * 2] = m_run[1];
        stats[r1 * 2 + 1] = l_run[1];
    }
#pragma unroll
    for (int nb = 0; nb < kHeadDim / 8; ++nb) {
        const int dim = nb * kGroup + c;
        accs[(size_t)r0 * kHeadDim + dim] = o_f[nb][0];
        accs[(size_t)r0 * kHeadDim + dim + 1] = o_f[nb][1];
        accs[(size_t)r1 * kHeadDim + dim] = o_f[nb][2];
        accs[(size_t)r1 * kHeadDim + dim + 1] = o_f[nb][3];
    }
#else
    /* fp32 SIMT arm (sm_72, compile-only in this packet, decision 0049 item 14): 128 threads,
     * two per row (each owning one half of the row's 256 dims), the same loaders, masking and
     * part layout; the score halves join with one shuffle, so both halves of a row see the
     * same statistics in the same order. */
    const int row = threadIdx.x >> 1, half = threadIdx.x & 1;
    const int p_row = a.p0 + t0 + (row >> 3);
    float o_acc[kHeadDim / 2];
#pragma unroll
    for (int e = 0; e < kHeadDim / 2; ++e) o_acc[e] = 0.f;
    float m_run = -INFINITY, l_run = 0.f;
    for (int base = k_begin; base < k_end && first + base <= max_p; base += kTile) {
        load_tile(base, 0);
        __syncthreads();
        const __half *K = sk[0], *V = sv[0];
        const int nvalid = min(kTile, k_end - base);
        for (int j = 0; j < nvalid; ++j) {
            const bool vis = sees(a, first + base + j, p_row);
            float dot = 0.f;
#pragma unroll 8
            for (int e = 0; e < kHeadDim / 2; ++e)
                dot = fmaf(__half2float(sq[row * kHeadDim + half * (kHeadDim / 2) + e]),
                           __half2float(K[j * kHeadDim + half * (kHeadDim / 2) + e]), dot);
            dot += __shfl_xor_sync(0xffffffffu, dot, 1);
            const float s = vis ? dot * a.scale : -INFINITY;
            const float m_new = fmaxf(m_run, s);
            const float m_ref = m_new == -INFINITY ? 0.f : m_new;  // the MMA arm's no-key-yet rule
            const float p = expf(s - m_ref);
            const float corr = expf(m_run - m_ref);
            l_run = l_run * corr + p;
            m_run = m_new;
            const float ph = __half2float(__float2half(p));
#pragma unroll 8
            for (int e = 0; e < kHeadDim / 2; ++e)
                o_acc[e] = fmaf(ph, __half2float(V[j * kHeadDim + half * (kHeadDim / 2) + e]),
                                o_acc[e] * corr);
        }
        __syncthreads();
    }
    if (half == 0) {
        stats[row * 2] = m_run;
        stats[row * 2 + 1] = l_run;
    }
#pragma unroll 8
    for (int e = 0; e < kHeadDim / 2; ++e)
        accs[(size_t)row * kHeadDim + half * (kHeadDim / 2) + e] = o_acc[e];
#endif
}

/* Lane max over the 32 lanes, fixed order (the attn_decode reduce form). */
__device__ inline float warp_max_a(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}

/* One block per (query head, token): merges the partials of the token's tile's active splits
 * in split-index order (the attn_combine_k pattern, attn_decode.cu:273-327: exact block max,
 * index-order L and A sums), divides and applies the sigmoid output gate. */
__global__ void __launch_bounds__(kHeadDim) attn_prefill_combine_k(Args a) {
    __shared__ float s_w[kMaxSplits], s_l[kMaxSplits], s_red[kHeadDim / 32];
    const int h = blockIdx.x, t = blockIdx.y, d = threadIdx.x;
    const int kvh = h >> 3, g = h & 7;
    float *o = a.out + ((size_t)t * a.n_head + h) * kHeadDim + d;
    if (!bounds_ok(a)) {
        report_bound_violation(a);
        *o = __int_as_float(0x7fc00000);  // NaN: the caller's non-finite check ends the job
        return;
    }
    const int tile = t >> 3;
    const int tile_end = min(a.m, (tile + 1) * kQTile);
    const int n_keys = a.p0 + tile_end;
    int first, count;
    visible(a, a.p0 + tile * kQTile, n_keys, first, count);
    int chunk, n_active;
    split_plan(count, a.max_splits, a.min_keys, chunk, n_active);
    const size_t n_cells = (size_t)((a.m + kQTile - 1) / kQTile) * a.n_kv_head * a.max_splits;
    const size_t cell0 = ((size_t)tile * a.n_kv_head + kvh) * a.max_splits;
    const int rr = (t & (kQTile - 1)) * kGroup + g;  // the token's row of its tile's cell
    const float *stats = a.part + cell0 * (kRows * 2) + rr * 2;
    const float *accs = a.part + n_cells * (kRows * 2) + (cell0 * kRows + rr) * kHeadDim + d;
    constexpr int kStatsStep = kRows * 2;              // between the splits of one cell
    constexpr size_t kAccsStep = (size_t)kRows * kHeadDim;
    // Statistics: thread s < n_active reads split s; the block max is exact in any order.
    float ms = -INFINITY;
    if (d < n_active) {
        ms = stats[(size_t)d * kStatsStep];
        s_l[d] = stats[(size_t)d * kStatsStep + 1];
    }
    const float wm = warp_max_a(ms);
    if ((d & 31) == 0) s_red[d >> 5] = wm;
    __syncthreads();
    float M = s_red[0];
#pragma unroll
    for (int i = 1; i < kHeadDim / 32; ++i) M = fmaxf(M, s_red[i]);
    if (d < n_active) s_w[d] = expf(ms - M);
    __syncthreads();
    float L = 0.f;  // index order
    for (int s = 0; s < n_active; ++s) L = fmaf(s_l[s], s_w[s], L);
    float A = 0.f;  // index order; the loads do not depend on the weights, so they overlap
#pragma unroll 16
    for (int s = 0; s < n_active; ++s) A = fmaf(accs[(size_t)s * kAccsStep], s_w[s], A);
    const float sig = a.proj == NULL
                          ? 1.f
                          : 1.f / (1.f + expf(-a.proj[(size_t)t * a.ld + (size_t)h * 2 * kHeadDim + kHeadDim + d]));
    *o = (A / L) * sig;
}

}  // namespace

namespace {

bool aligned_to(const void *p, uintptr_t a) { return ((uintptr_t)p & (a - 1)) == 0; }

/* The wrapper mirrors every rule here, so the native entry cannot return INVALID_ARGUMENT
 * (which would latch the backend); a C caller gets the same refusal before any launch. */
bool valid(const inference_cuda_attn_prefill_args *a) {
    if (a == NULL || a->q == NULL || a->part == NULL || a->out == NULL) return false;
    if (a->kv_mode != INFERENCE_CUDA_PREFILL_KV_FP16) return false;
    if (a->k_cache == NULL || a->v_cache == NULL) return false;
    if (a->kv8_k_codes != NULL || a->kv8_k_scales != NULL || a->kv8_v_codes != NULL ||
        a->kv8_v_scales != NULL || a->kv8_capacity != 0)
        return false;
    if (!aligned_to(a->q, 16) || !aligned_to(a->k_cache, 16) || !aligned_to(a->v_cache, 16) ||
        !aligned_to(a->part, 16) || !aligned_to(a->out, 16))
        return false;
    if (a->proj != NULL && !aligned_to(a->proj, 4)) return false;
    if (a->status != NULL && !aligned_to(a->status, 4)) return false;
    const bool heads = a->n_head == kGroup * a->n_kv_head && a->n_kv_head >= 1 &&
                       a->n_kv_head <= 64;
    const bool gate_ok = a->proj == NULL || a->ld >= a->n_head * 2 * kHeadDim;
    // A ring needs a window, must hold the window plus the m - 1 later tokens of the chunk,
    // and must fit the cache (attn_decode.h:22-25). 64-bit sum: window may be INT_MAX.
    const bool ring_ok = a->window >= 0 && a->ring >= 0 &&
                         (a->ring == 0 || (a->window > 0 && a->ring <= a->cache_rows &&
                                           (long long)a->ring >=
                                               (long long)a->window + a->m - 1));
    const bool bounds = a->p0 >= 0 && (long long)a->p0 + a->m <= a->cache_rows;
    const bool plan = a->max_splits >= 1 && a->max_splits <= kMaxSplits &&
                      a->min_keys >= kTile && a->min_keys % kTile == 0 &&
                      a->min_keys <= kMaxMinKeys;
    const bool scale_ok = a->scale > 0.f && a->scale < 1e30f && isfinite(a->scale);
    return a->head_dim == kHeadDim && heads && gate_ok && ring_ok && bounds && plan &&
           scale_ok && a->m >= 1 && a->m <= kMaxGridY && a->cache_rows >= 1 &&
           a->cache_rows <= kMaxRows;
}

bool valid_sans_bounds(const inference_cuda_attn_prefill_args *a) {
    inference_cuda_attn_prefill_args b = *a;
    b.p0 = 0;
    b.m = 1;
    b.cache_rows = 2;
    return valid(&b) && a->m >= 1 && a->m <= kMaxGridY;
}

Args to_args(const inference_cuda_attn_prefill_args *a) {
    Args k = {};
    k.q = (const float *)a->q;
    k.proj = (const float *)a->proj;
    k.k_cache = (const __half *)a->k_cache;
    k.v_cache = (const __half *)a->v_cache;
    k.part = (float *)a->part;
    k.out = (float *)a->out;
    k.status = (int *)a->status;
    k.ld = a->ld;
    k.p0 = a->p0;
    k.m = a->m;
    k.n_head = a->n_head;
    k.n_kv_head = a->n_kv_head;
    k.cache_rows = a->cache_rows;
    k.max_splits = a->max_splits;
    k.min_keys = a->min_keys;
    k.window = a->window;
    k.ring = a->ring;
    k.scale = a->scale;
    return k;
}

/* Enqueues both kernels. The part kernel's 96 KB of dynamic shared memory need the opt-in
 * above the 48 KB default, set once per process (the vendored gemm.cuh pattern); every call
 * runs under the caller's dispatch guard, so the flag needs no lock of its own. */
Failure enqueue(const Args &k, void *stream) {
    static bool smem_armed = false;
    Failure failure;
    if (!smem_armed) {
        failure.record(cudaFuncSetAttribute(attn_prefill_part_k,
                                            cudaFuncAttributeMaxDynamicSharedMemorySize,
                                            kSmemBytes),
                       INFERENCE_CUDA_STAGE_LAUNCH);
        if (failure.ok()) smem_armed = true;
    }
    if (failure.ok()) {
        attn_prefill_part_k<<<dim3(k.max_splits, k.n_kv_head, (k.m + kQTile - 1) / kQTile),
                              128, kSmemBytes, (cudaStream_t)stream>>>(k);
        failure.record(cudaGetLastError(), INFERENCE_CUDA_STAGE_LAUNCH);
    }
    if (failure.ok()) {
        attn_prefill_combine_k<<<dim3(k.n_head, k.m), kHeadDim, 0, (cudaStream_t)stream>>>(k);
        failure.record(cudaGetLastError(), INFERENCE_CUDA_STAGE_LAUNCH);
    }
    return failure;
}

}  // namespace

/* Floats of `part` for the rule above; the Rust side pins the same numbers
 * (cpf_attn_part_bytes). */
extern "C" size_t inference_cuda_attn_prefill_part_floats(int m, int n_kv_head, int head_dim,
                                                          int max_splits) {
    return (size_t)((m + kQTile - 1) / kQTile) * kQTile * n_kv_head * max_splits * kGroup *
           (2 + head_dim);
}

extern "C" int inference_cuda_attn_prefill_launch(const inference_cuda_attn_prefill_args *a,
                                                  void *stream, int *out_stage) {
    try {
        report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
        if (!valid(a)) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        const Args k = to_args(a);
        const Failure failure = enqueue(k, stream);
        report_stage(out_stage, failure.stage);
        return (int)failure.err;
    } catch (...) {
        return INFERENCE_CUDA_NATIVE_EXCEPTION;
    }
}

#ifdef INFERENCE_CUDA_FAULT_QUALIFICATION
extern "C" int inference_cuda_attn_prefill_fault_bad_bounds_launch(
    const inference_cuda_attn_prefill_args *a, void *stream, int *out_stage) {
    try {
        report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
        /* Every rule except the p0 bound holds (m clamped to 1 so the shape rules stay
         * satisfiable); only the bound is violated, by force. The kernels' device re-checks
         * (no partial, NaN out, bit 8) are the code under test. */
        if (!valid_sans_bounds(a)) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        Args k = to_args(a);
        k.p0 = a->cache_rows;  // cache_rows + m > cache_rows: the bound violation
        const Failure failure = enqueue(k, stream);
        report_stage(out_stage, failure.stage);
        return (int)failure.err;
    } catch (...) {
        return INFERENCE_CUDA_NATIVE_EXCEPTION;
    }
}
#endif
