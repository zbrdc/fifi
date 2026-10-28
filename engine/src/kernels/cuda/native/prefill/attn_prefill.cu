/* CPF-ATTN-A K2 and CPF-ATTN-B: the kernels and the launch entry of attn_prefill, the
 * flash-style chunked-prefill attention (decision 0049 items 1, 2 and 10; the CPF-5a/5b kernel
 * of design section 3.7). The split rule, the
 * visible/cache_row span rules and the combine pattern restate the resident attn_decode
 * (RD-8b, this tree); the online-softmax recurrence and the query-tile geometry (8 tokens x 8
 * GQA heads = 64 rows, key tiles of 32) follow design section 3.7; the sm_80+ part kernel's
 * head_dim split with a fixed-order exchange, cp.async with an XOR swizzle, Q in registers and
 * ldmatrix follow the CPF-ATTN-B packet (Strata's micro-architecture and llama.cpp's MMA
 * kernel as described there, nothing read or copied). The device checks p0/m again (item 2)
 * and ORs mask 8 of the prefill status word (item 10).
 * Contract: prefill/attn_prefill.h. */
#include "prefill/attn_prefill.h"

#include <cuda_fp16.h>

#include <math.h>
#include <stdint.h>

#include "abi.h"
#include "device.h"

namespace {

constexpr int kHeadDim = 256;   // legacy geometry; launch templates select the head size
constexpr int kGroup = 8;       // head lanes packed into an MMA tile
constexpr int kTile = 32;       // keys per tile (the granule of a key range)
constexpr int kQTile = 8;       // tokens per query tile: the compiled tile
constexpr int kRows = kQTile * kGroup;  // 64 rows per query tile
constexpr int kMaxGridY = 65535;
constexpr int kMaxRows = (1 << 30) - 1;
constexpr int kMaxSplits = 128;
constexpr int kMaxMinKeys = 1 << 20;
/* Legacy D256 shared memory: 96 KB = 98,304 B, above the 48 KB default (the
 * entry opts in once per process, the vendored gemm.cuh pattern). The sm_72 SIMT kernel lays
 * out a Q tile (64 x 256 fp16 = 32 KB) and two K and V stages of 16 KB each, and uses the Q
 * tile and stage 0 of K (32-48 KB) and of V (64-80 KB), an 80 KB extent inside the request,
 * which sm_72's 96 KB per-block opt-in admits. */
constexpr int kSmemBytes = kRows * kHeadDim * 2 + 2 * (2 * kTile * kHeadDim * 2);
/* The head_dim-quarter kernel: 16 warps, two cp.async stages of K and V (16 KB planes of 32
 * keys x 512 B) and a 2 KB exchange slot per warp, the same 98,304 B request. */
constexpr int kThreadsD = 512;
constexpr int kPlaneBytes = kTile * kHeadDim * 2;
constexpr int kStageBytes = 2 * kPlaneBytes;
static_assert(2 * kStageBytes + (kThreadsD / 32) * 2048 == kSmemBytes, "dsplit shared memory");

struct Args {
    const float *q;
    const float *proj;  // NULL (no gate) or [m][ld]
    const float *sinks; // NULL or raw per-query-head logits
    const __half *k_cache;
    const __half *v_cache;
    float *part;
    float *out;
    int *status;
    int ld, p0, m, n_head, n_kv_head, head_dim, cache_rows, max_splits, min_keys, window, ring;
    float scale;
    int mode, key_origin, key_rows, part_index;  // 0 full, 1 segment part, 2 all-slot combine
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
    return pos <= p && (a.window == 0 || pos > p - a.window) &&
           (a.mode != 1 || pos < a.key_origin + a.key_rows);
}

/* Cache row of the i-th visible key (attn_decode.cu:87-89). */
__device__ inline int cache_row(const Args &a, int first, int i) {
    if (a.mode == 1) return first + i - a.key_origin;
    return a.ring > 0 ? (first + i) % a.ring : first + i;
}

/* The device re-check of the launch arguments (decision 0049 item 2): block-uniform, before
 * the first barrier. ORs mask 8 once; the caller's writes (or their absence) are the callers'. */
__device__ inline bool bounds_ok(const Args &a) {
    const long long end = (long long)a.p0 + a.m;
    if (a.p0 < 0 || a.m < 1 || end > (a.mode == 0 && a.ring == 0 ? a.cache_rows : kMaxRows)) return false;
    return a.mode != 1 ||
           (a.key_origin >= 0 && a.key_rows > 0 && a.key_rows <= a.cache_rows &&
            (long long)a.key_origin + a.key_rows <= end &&
            a.part_index >= 0 && a.part_index < a.max_splits);
}

__device__ inline void report_bound_violation(const Args &a) {
    if (a.status && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && threadIdx.x == 0)
        atomicOr(a.status, INFERENCE_CUDA_PREFILL_STATUS_BOUND_VIOLATION);
}

/* One block's share of the work: query tile `tile` (rows t0 .. tile_end - 1 of the chunk) of one
 * kv head, the keys k_begin .. k_end - 1 of the tile's visible span (visible key i at position
 * first + i), and its partial cell. */
struct Block {
    int t0, tile_end, n_keys, first, k_begin, k_end, max_p;
    float *stats, *accs;
};

/* Poison only visible scores; a NaN normalizer remains sticky across later healthy tiles. */
__device__ inline float normalizer_weight(bool visible, float score, float weight) {
    return visible && !isfinite(score) ? NAN : weight;
}

/* The block of split s, kv head kvh and query tile `tile`; false when the split holds no key
 * of the tile (the block then exits before any barrier). */
template <int kHeadDim, int kGroup>
__device__ inline bool block_of(const Args &a, int s, int kvh, int tile, Block &b) {
    const int kRows = kQTile * (kGroup == 0 ? a.n_head / a.n_kv_head : kGroup);
    b.t0 = tile * kQTile;
    b.tile_end = min(a.m, b.t0 + kQTile);
    b.n_keys = a.p0 + b.tile_end;  // keys 0..p0 + tile_end - 1 exist for this tile
    b.max_p = b.n_keys - 1;
    const size_t n_cells = (size_t)((a.m + kQTile - 1) / kQTile) * a.n_kv_head * a.max_splits;
    const int slot = a.mode == 1 ? a.part_index : s;
    const size_t cell = ((size_t)tile * a.n_kv_head + kvh) * a.max_splits + slot;
    b.stats = a.part + cell * (kRows * 2);
    b.accs = a.part + n_cells * (kRows * 2) + cell * ((size_t)kRows * kHeadDim);
    if (a.mode == 1) {
        b.first = a.key_origin;
        b.k_begin = 0;
        b.k_end = max(0, min(b.n_keys, a.key_origin + a.key_rows) - b.first);
        // Even an empty tile runs the existing zero-key recurrence and writes neutral partials.
        return true;
    }
    int count;
    visible(a, a.p0 + b.t0, b.n_keys, b.first, count);
    int chunk, n_active;
    split_plan(count, a.max_splits, a.min_keys, chunk, n_active);
    if (s >= n_active) return false;
    b.k_begin = s * chunk;
    b.k_end = min(count, b.k_begin + chunk);
    return true;
}

#if __CUDA_ARCH__ >= 800
/* D (16x8 f32) += A (16x16 f16, row) * B (16x8 f16, col), the sm_80 tensor-core path. */
__device__ inline void mma_16816(float *d, const unsigned *a, unsigned b0, unsigned b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
#else
/* The Q tile in shared memory, rounded to fp16 once (128 threads): rows past the chunk's end
 * (m not a multiple of 8) hold zeros; their partials are written but never read. */
template <int kHeadDim, int kGroup>
__device__ inline void load_q_tile(const Args &a, const Block &b, int kvh, __half *sq) {
    for (int i = threadIdx.x; i < kRows * kHeadDim; i += 128) {
        const int r = i / kHeadDim, d = i % kHeadDim;
        const int t = b.t0 + (r >> 3), g = r & 7;
        sq[i] = t < b.tile_end && g < kGroup
                    ? __float2half(a.q[((size_t)t * a.n_head + kvh * kGroup + g) * kHeadDim + d])
                    : __half(0);
    }
}

/* Loads the key tile at visible index base into sk/sv (128 threads). Keys beyond the range
 * hold zeros, so a masked or out-of-range key contributes exactly nothing (0 * V = 0) and a
 * NaN cell no row may read never enters the tile. */
template <int kHeadDim>
__device__ inline void load_kv_tile(const Args &a, const Block &b, int kvh, int base, __half *sk,
                                    __half *sv) {
    const int nvalid = min(kTile, b.k_end - base);
    for (int i = threadIdx.x; i < kTile * kHeadDim; i += 128) {
        const int j = i / kHeadDim, d = i % kHeadDim;
        __half k = __half(0), v = __half(0);
        if (j < nvalid) {
            const size_t at = ((size_t)cache_row(a, b.first, base + j) * a.n_kv_head + kvh) * kHeadDim + d;
            k = a.k_cache[at];
            v = a.v_cache[at];
        }
        sk[i] = k;
        sv[i] = v;
    }
}
#endif

/* The fp32 SIMT kernel of sm_72 (compile-only in this packet, decision 0049 item 14), the
 * same block and partial cell as attn_prefill_part_dsplit_k: 128 threads, two per row (each owning
 * one half of the row's 256 dims), the same loaders, masking and part layout; the score halves
 * join with one shuffle, so both halves of a row see the same statistics in the same order.
 * An sm_80+ build traps here (the entry launches the tensor-core kernel there). */
template <int kHeadDim, int kGroup>
__global__ void __launch_bounds__(128) attn_prefill_part_simt_k(Args a) {
#if __CUDA_ARCH__ >= 800
    __trap();
#else
    if (!bounds_ok(a)) {
        report_bound_violation(a);
        return;  // no partial
    }
    extern __shared__ __half smem[];
    __half *sq = smem;  // [64][256], then stage 0 of K at 32 KB and of V at 64 KB
    __half *sk = smem + kRows * kHeadDim, *sv = smem + kRows * kHeadDim + 2 * kTile * kHeadDim;
    const int s = blockIdx.x, kvh = blockIdx.y, tile = blockIdx.z;
    Block b;
    if (!block_of<kHeadDim, kGroup>(a, s, kvh, tile, b)) return;
    load_q_tile<kHeadDim, kGroup>(a, b, kvh, sq);
    const int row = threadIdx.x >> 1, half = threadIdx.x & 1;
    const int p_row = a.p0 + b.t0 + (row >> 3);
    float o_acc[kHeadDim / 2];
#pragma unroll
    for (int e = 0; e < kHeadDim / 2; ++e) o_acc[e] = 0.f;
    float m_run = -INFINITY, l_run = 0.f;
    for (int base = b.k_begin; base < b.k_end && b.first + base <= b.max_p; base += kTile) {
        load_kv_tile<kHeadDim>(a, b, kvh, base, sk, sv);
        __syncthreads();
        const __half *K = sk, *V = sv;
        const int nvalid = min(kTile, b.k_end - base);
        for (int j = 0; j < nvalid; ++j) {
            const bool vis = sees(a, b.first + base + j, p_row);
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
            l_run = l_run * corr + normalizer_weight(vis, s, p);
            m_run = m_new;
            const float ph = __half2float(__float2half(p));
#pragma unroll 8
            for (int e = 0; e < kHeadDim / 2; ++e)
                o_acc[e] = fmaf(ph, __half2float(V[j * kHeadDim + half * (kHeadDim / 2) + e]),
                                o_acc[e] * corr);
        }
        __syncthreads();
    }
    if ((row & 7) >= kGroup) return;
    const int out_row = (row >> 3) * kGroup + (row & 7);
    if (half == 0) {
        b.stats[out_row * 2] = m_run;
        b.stats[out_row * 2 + 1] = l_run;
    }
#pragma unroll 8
    for (int e = 0; e < kHeadDim / 2; ++e)
        b.accs[(size_t)out_row * kHeadDim + half * (kHeadDim / 2) + e] = o_acc[e];
#endif
}

#if __CUDA_ARCH__ >= 800
constexpr float kLog2e = 1.4426950408889634f;

__device__ inline float exp_score_delta(float delta) {
    return exp2f(__fmul_rn(delta, kLog2e));
}

/* The primitives of the head_dim-quarter kernel (sm_80+ only, so no device pass sees one
 * unreferenced). Stage planes are [32 keys][512 B]; the 16-byte chunk j of key k sits at
 * byte k * 512 + ((j ^ (k & 7)) << 4): the XOR stays inside a warp's 8-chunk quarter, and
 * the 8 row addresses of every 8x8 ldmatrix hit 8 distinct chunks of one 128-byte span. */
template <int kHeadDim>
__device__ inline unsigned swz(int key, int chunk) {
    return key * (2 * kHeadDim) + ((chunk ^ (key & 7)) << 4);
}

__device__ inline unsigned smem_addr(const void *p) {
    return (unsigned)__cvta_generic_to_shared(p);
}

/* 16 bytes global -> shared, bypassing L1; src_bytes 0 reads nothing and writes zeros. */
__device__ inline void cp_async16(unsigned dst, const void *src, int src_bytes) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst), "l"(src),
                 "r"(src_bytes));
}

__device__ inline void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }

__device__ inline void cp_async_wait_all() {
    asm volatile("cp.async.wait_group 0;\n" ::: "memory");
}

/* Four 8x8 b16 matrices; lanes 8i..8i+7 give the row addresses of matrix i. */
__device__ inline void ldsm_x4(unsigned addr, unsigned (&r)[4]) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(addr));
}

/* The same, transposed: lane T gets rows 2(T%4), 2(T%4)+1 of column T/4. */
__device__ inline void ldsm_x4_trans(unsigned addr, unsigned (&r)[4]) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(addr));
}

/* The named barrier of one row group's four warps (ids 1..4; 0 is __syncthreads). */
__device__ inline void bar_row_group(int rg) {
    asm volatile("bar.sync %0, 128;\n" ::"r"(1 + rg) : "memory");
}

/* Issues the K and V of the key tile at visible index base into the stage at byte `stage`
 * (512 threads; thread i copies keys (i >> 5) + {0, 16}, chunk i & 31). A key at or past
 * k_end copies src-size 0 (zeros, no global read) from the tile's first key row, a valid
 * row, so a cell no row may read never enters the tile. */
template <int kHeadDim>
__device__ inline void issue_kv_tile(const Args &a, const Block &b, int kvh, int base,
                                     unsigned stage) {
    constexpr int kPlaneBytes = kTile * kHeadDim * 2;
#pragma unroll
    for (int i = threadIdx.x; i < kTile * (kHeadDim / 8); i += kThreadsD) {
        const int key = i / (kHeadDim / 8), chunk = i % (kHeadDim / 8);
        const bool valid = base + key < b.k_end;
        const size_t row = (size_t)cache_row(a, b.first, base + (valid ? key : 0));
        const size_t at = (row * a.n_kv_head + kvh) * kHeadDim + chunk * 8;
        const unsigned dst = stage + swz<kHeadDim>(key, chunk);
        cp_async16(dst, a.k_cache + at, valid ? 16 : 0);
        cp_async16(dst + kPlaneBytes, a.v_cache + at, valid ? 16 : 0);
    }
    cp_async_commit();
}

/* The lane's Q̂ A-fragments for rows r0 = 16 rg + g (token t0 + 2 rg) and r1 = r0 + 8 (the
 * next token), head G kvh + g, dims (D/4) dq + 16 ks + {c, c+1, c+8, c+9}: rounded to fp16
 * once, as CPF-ATTN-A's q̂; a token at or past tile_end loads zeros and reads no q. */
template <int kHeadDim, int kGroup>
__device__ inline void load_q_frags(const Args &a, const Block &b, int kvh, int rg, int dq,
                                    unsigned (&qf)[4][4]) {
    const int lane = threadIdx.x & 31, g = lane >> 2, c = (lane & 3) * 2;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int t = b.t0 + 2 * rg + i;
#pragma unroll
        for (int ks = 0; ks < kHeadDim / 64; ++ks)
#pragma unroll
            for (int hi = 0; hi < 2; ++hi) {
                float2 v = make_float2(0.f, 0.f);
                if (t < b.tile_end && g < kGroup) {
                    const float *q = a.q + ((size_t)t * a.n_head + kvh * kGroup + g) * kHeadDim +
                                     (kHeadDim / 4) * dq + c;
                    v = *reinterpret_cast<const float2 *>(q + 16 * ks + 8 * hi);
                }
                const __half2 h = __floats2half2_rn(v.x, v.y);
                qf[ks][i + 2 * hi] = *reinterpret_cast<const unsigned *>(&h);
            }
    }
}
#endif

/* One block of the head_dim-quarter kernel (sm_80+): the block unit of CPF-ATTN-A's part kernel
 * (one query tile of 64 rows, one kv head, one split's keys; the same partial cell) over 16
 * warps. Warp w holds row group rg = w >> 2 (rows 16 rg .. +15) and head_dim quarter
 * dq = w & 3 (D/4 dimensions each): Q̂ in registers, its quarter of S = Q̂·K^T, and O for its
 * rows and dims. The four quarter partials of S are summed through shared memory in one
 * fixed order, so the four warps of a row group hold identical statistics and P̂.
 * K and V arrive by cp.async into two swizzled stages (one block barrier per tile) and
 * reach the MMAs by ldmatrix. Grid (q_tiles, n_kv_head, max_splits), the heaviest causal
 * tiles first; an sm_72 build traps here. */
template <int kHeadDim, int kGroup>
__global__ void __launch_bounds__(kThreadsD, 1) attn_prefill_part_dsplit_k(Args a) {
#if __CUDA_ARCH__ >= 800
    constexpr int kPlaneBytes = kTile * kHeadDim * 2;
    constexpr int kStageBytes = 2 * kPlaneBytes;
    if (!bounds_ok(a)) {
        report_bound_violation(a);
        return;  // no partial
    }
    extern __shared__ __align__(16) unsigned char dsmem[];
    const int q_tiles = (a.m + kQTile - 1) / kQTile;
    const int tile = q_tiles - 1 - (int)blockIdx.x, kvh = blockIdx.y, s = blockIdx.z;
    Block b;
    if (!block_of<kHeadDim, kGroup>(a, s, kvh, tile, b)) return;
    const unsigned stages = smem_addr(dsmem);
    float4 *const xch = reinterpret_cast<float4 *>(dsmem + 2 * kStageBytes);
    if (b.k_begin < b.k_end) issue_kv_tile<kHeadDim>(a, b, kvh, b.k_begin, stages);

    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int rg = warp >> 2, dq = warp & 3;
    const int g = lane >> 2, c = (lane & 3) * 2;
    unsigned qf[4][4];
    load_q_frags<kHeadDim, kGroup>(a, b, kvh, rg, dq, qf);
    const int p_row0 = a.p0 + b.t0 + 2 * rg, p_row1 = p_row0 + 1;  // rows r0, r1
    /* Raw sinks and real-key maxima share natural units; convert only normalized deltas.
     * Keep score products explicitly rounded in both mask paths (lane C MINOR-1). */
    const float score_scale = a.scale;
    /* ldmatrix lane roles: matrix mi = lane >> 3, row rr = lane & 7. K: key 16 np + 8 (mi >> 1)
     * + rr, chunk 8 dq + 2 ks + (mi & 1). V: key 16 ks + 8 (mi & 1) + rr, chunk 8 dq + 2 jp +
     * (mi >> 1). */
    const int mi = lane >> 3, rr = lane & 7;
    const int k_key = 8 * (mi >> 1) + rr, k_chunk = (kHeadDim / 32) * dq + (mi & 1);
    const int v_key = 8 * (mi & 1) + rr, v_chunk = (kHeadDim / 32) * dq + (mi >> 1);

    float o[kHeadDim / 32][4];
#pragma unroll
    for (int j = 0; j < kHeadDim / 32; ++j)
#pragma unroll
        for (int e = 0; e < 4; ++e) o[j][e] = 0.f;
    float m_run[2] = {-INFINITY, -INFINITY}, l_run[2] = {0.f, 0.f};

    int n = 0;
    for (int base = b.k_begin; base < b.k_end && b.first + base <= b.max_p; base += kTile, ++n) {
        cp_async_wait_all();
        __syncthreads();  // tile n visible; stage (n + 1) & 1 and the exchange free
        const int next = base + kTile;
        if (next < b.k_end && b.first + next <= b.max_p)
            issue_kv_tile<kHeadDim>(a, b, kvh, next, stages + ((n + 1) & 1) * kStageBytes);
        const unsigned kst = stages + (n & 1) * kStageBytes, vst = kst + kPlaneBytes;

        /* This quarter's S: 4 k16 steps x 4 n8 key blocks. */
        float sq[4][4];
#pragma unroll
        for (int nb = 0; nb < 4; ++nb)
#pragma unroll
            for (int e = 0; e < 4; ++e) sq[nb][e] = 0.f;
#pragma unroll
        for (int ks = 0; ks < kHeadDim / 64; ++ks)
#pragma unroll
            for (int np = 0; np < 2; ++np) {
                unsigned kb[4];
                ldsm_x4(kst + swz<kHeadDim>(16 * np + k_key, k_chunk + 2 * ks), kb);
                mma_16816(sq[2 * np], qf[ks], kb[0], kb[1]);
                mma_16816(sq[2 * np + 1], qf[ks], kb[2], kb[3]);
            }

        /* The exchange, one softmax per row group (design 3.10). Warp dq owns key block dq
         * (keys 8 dq .. +7) of its row group: it stores only the three blocks its siblings own
         * (X[w][nb != dq]) and keeps its own in registers. The row group's other regions sit
         * in own-block slots, which hold no S: the row maxima in X[4 rg][0], the row-sum
         * partials in X[4 rg + 1][1], and P̂ in X[4 rg + 2] bytes 0..1,279 (16 rows of 80 B).
         * P̂ covers X[4 rg + 2][0..1] and the first 256 B of X[4 rg + 2][2], the dq = 2 warp's
         * own-block slot: safe, because that slot holds no S and every S read is over before
         * X2, so after X2 only xmax and xsum must stay clear of P̂. */
        float4 own = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
        for (int nb = 0; nb < 4; ++nb) {
            const float4 v = make_float4(sq[nb][0], sq[nb][1], sq[nb][2], sq[nb][3]);
            if (nb == dq)
                own = v;
            else
                xch[(warp * 4 + nb) * 32 + lane] = v;
        }
        float *const xmax = reinterpret_cast<float *>(xch + (16 * rg) * 32);          // X[4rg][0]
        float *const xsum = reinterpret_cast<float *>(xch + (16 * rg + 5) * 32);      // X[4rg+1][1]
        unsigned char *const xp = reinterpret_cast<unsigned char *>(xch + (16 * rg + 8) * 32);
        bar_row_group(rg);  // X1: the S partials are in place
        /* The own block summed over the four quarters in quarter order ((S0 + S1) + S2) + S3:
         * the same values and order as the per-warp sum, so p and P̂ are bit-identical to it. */
        float4 sum4 = dq == 0 ? own : xch[((4 * rg) * 4 + dq) * 32 + lane];
#pragma unroll
        for (int q = 1; q < 4; ++q) {
            const float4 v = q == dq ? own : xch[((4 * rg + q) * 4 + dq) * 32 + lane];
            sum4.x += v.x;
            sum4.y += v.y;
            sum4.z += v.z;
            sum4.w += v.w;
        }
        /* Scale and mask in natural units; the fully visible tile skips the mask. */
        const bool full = b.first + base + kTile - 1 <= a.p0 + b.t0 &&
                          (a.window == 0 || b.first + base > b.max_p - a.window) &&
                          (a.mode != 1 || base + kTile <= b.k_end);
        const int pos = b.first + base + dq * 8 + c;
        const bool visible[4] = {full || sees(a, pos, p_row0), full || sees(a, pos + 1, p_row0),
                                 full || sees(a, pos, p_row1), full || sees(a, pos + 1, p_row1)};
        float s2[4];
        if (full) {
            s2[0] = __fmul_rn(sum4.x, score_scale);
            s2[1] = __fmul_rn(sum4.y, score_scale);
            s2[2] = __fmul_rn(sum4.z, score_scale);
            s2[3] = __fmul_rn(sum4.w, score_scale);
        } else {
            s2[0] = visible[0] ? __fmul_rn(sum4.x, score_scale) : -INFINITY;
            s2[1] = visible[1] ? __fmul_rn(sum4.y, score_scale) : -INFINITY;
            s2[2] = visible[2] ? __fmul_rn(sum4.z, score_scale) : -INFINITY;
            s2[3] = visible[3] ? __fmul_rn(sum4.w, score_scale) : -INFINITY;
        }
        float bm[2] = {fmaxf(s2[0], s2[1]), fmaxf(s2[2], s2[3])};
#pragma unroll
        for (int o2 = 1; o2 <= 2; o2 <<= 1) {
            bm[0] = fmaxf(bm[0], __shfl_xor_sync(0xffffffffu, bm[0], o2));
            bm[1] = fmaxf(bm[1], __shfl_xor_sync(0xffffffffu, bm[1], o2));
        }
        if (c == 0) {
            xmax[g * 4 + dq] = bm[0];
            xmax[(g + 8) * 4 + dq] = bm[1];
        }
        bar_row_group(rg);  // X2: the block maxima are in place
        const float4 mq0 = *reinterpret_cast<const float4 *>(xmax + g * 4);
        const float4 mq1 = *reinterpret_cast<const float4 *>(xmax + (g + 8) * 4);
        const float mx[2] = {fmaxf(fmaxf(mq0.x, mq0.y), fmaxf(mq0.z, mq0.w)),
                             fmaxf(fmaxf(mq1.x, mq1.y), fmaxf(mq1.z, mq1.w))};
        const float m_new[2] = {fmaxf(m_run[0], mx[0]), fmaxf(m_run[1], mx[1])};
        /* The no-key-yet rule (review BLOCKER-1, the B1 guard): exponentiate against 0 while
         * the max is still -inf, so p = 0 and corr = 0 instead of NaN (exp2f(-inf) = 0). */
        const float m_ref[2] = {m_new[0] == -INFINITY ? 0.f : m_new[0],
                                m_new[1] == -INFINITY ? 0.f : m_new[1]};
        const float p[4] = {exp_score_delta(s2[0] - m_ref[0]), exp_score_delta(s2[1] - m_ref[0]),
                            exp_score_delta(s2[2] - m_ref[1]), exp_score_delta(s2[3] - m_ref[1])};
        float ps[2] = {normalizer_weight(visible[0], s2[0], p[0]) + normalizer_weight(visible[1], s2[1], p[1]),
                       normalizer_weight(visible[2], s2[2], p[2]) + normalizer_weight(visible[3], s2[3], p[3])};
#pragma unroll
        for (int o2 = 1; o2 <= 2; o2 <<= 1) {
            ps[0] += __shfl_xor_sync(0xffffffffu, ps[0], o2);
            ps[1] += __shfl_xor_sync(0xffffffffu, ps[1], o2);
        }
        if (c == 0) {
            xsum[g * 4 + dq] = ps[0];
            xsum[(g + 8) * 4 + dq] = ps[1];
        }
        *reinterpret_cast<__half2 *>(xp + g * 80 + 16 * dq + 2 * c) = __floats2half2_rn(p[0], p[1]);
        *reinterpret_cast<__half2 *>(xp + (g + 8) * 80 + 16 * dq + 2 * c) =
            __floats2half2_rn(p[2], p[3]);
        const float corr[2] = {exp_score_delta(m_run[0] - m_ref[0]), exp_score_delta(m_run[1] - m_ref[1])};
        bar_row_group(rg);  // X3: the sum partials and P̂ are in place
        const float4 sq0 = *reinterpret_cast<const float4 *>(xsum + g * 4);
        const float4 sq1 = *reinterpret_cast<const float4 *>(xsum + (g + 8) * 4);
        l_run[0] = l_run[0] * corr[0] + (((sq0.x + sq0.y) + sq0.z) + sq0.w);
        l_run[1] = l_run[1] * corr[1] + (((sq1.x + sq1.y) + sq1.z) + sq1.w);
        m_run[0] = m_new[0];
        m_run[1] = m_new[1];

        /* O = O * corr + P̂·V̂ over this quarter's 64 dims: P̂ (16 rows x 32 keys) by ldmatrix
         * as A-fragments (matrix mi: rows 8 (mi & 1) .., keys 16 ks + 8 (mi >> 1) ..), V by
         * ldmatrix.trans, two n8 blocks per load. */
#pragma unroll
        for (int j = 0; j < kHeadDim / 32; ++j) {
            o[j][0] *= corr[0];
            o[j][1] *= corr[0];
            o[j][2] *= corr[1];
            o[j][3] *= corr[1];
        }
        const unsigned p_lane = smem_addr(xp) + (rr + 8 * (mi & 1)) * 80 + 16 * (mi >> 1);
#pragma unroll
        for (int ks = 0; ks < 2; ++ks) {
            unsigned pa[4];
            ldsm_x4(p_lane + 32 * ks, pa);
#pragma unroll
            for (int jp = 0; jp < kHeadDim / 64; ++jp) {
                unsigned vb[4];
                ldsm_x4_trans(vst + swz<kHeadDim>(16 * ks + v_key, v_chunk + 2 * jp), vb);
                mma_16816(o[2 * jp], pa, vb[0], vb[1]);
                mma_16816(o[2 * jp + 1], pa, vb[2], vb[3]);
            }
        }
    }
    // shortcut: GQA1 pads eight head lanes; retile when a measured GQA1 workload needs it.
    if (g >= kGroup) return;
    const int r0 = 2 * rg * kGroup + g, r1 = r0 + kGroup;
    /* Maxima already share the raw sink's natural units; no lossy log2 round-trip. */
    if (dq == 0 && (lane & 3) == 0) {
        b.stats[r0 * 2] = m_run[0];
        b.stats[r0 * 2 + 1] = l_run[0];
        b.stats[r1 * 2] = m_run[1];
        b.stats[r1 * 2 + 1] = l_run[1];
    }
#pragma unroll
    for (int j = 0; j < kHeadDim / 32; ++j) {
        const int dim = (kHeadDim / 4) * dq + 8 * j + c;
        *reinterpret_cast<float2 *>(b.accs + (size_t)r0 * kHeadDim + dim) = make_float2(o[j][0], o[j][1]);
        *reinterpret_cast<float2 *>(b.accs + (size_t)r1 * kHeadDim + dim) = make_float2(o[j][2], o[j][3]);
    }
#else
    __trap();
#endif
}

/* Lane max over the 32 lanes, fixed order (the attn_decode reduce form). */
__device__ inline float warp_max_a(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}


__device__ inline float affine_scalar(const void *ptr, int dtype, size_t offset) {
    const char *p = (const char *)ptr + offset;
    if (dtype == 2) return *(const float *)p;
    if (dtype == 1) return __int_as_float((unsigned int)*(const unsigned short *)p << 16);
    return __half2float(*(const __half *)p);
}

__device__ inline float q8_value(const inference_cuda_attn_kv_view &v, int row, int h, int d) {
    const size_t payload = (size_t)row * v.payload_time_stride_bytes +
                           (size_t)h * v.payload_head_stride_bytes + d;
    const float code = v.q8_signed ? (float)((const signed char *)v.payload)[payload]
                                   : (float)((const unsigned char *)v.payload)[payload];
    const size_t group = d / v.group_size;
    const float scale = affine_scalar(v.scales, v.scale_dtype,
        (size_t)row * v.scale_time_stride_bytes + (size_t)h * v.scale_head_stride_bytes +
        group * v.scale_group_stride_bytes);
    const float bias = v.bias == NULL ? 0.f : affine_scalar(v.bias, v.bias_dtype,
        (size_t)row * v.bias_time_stride_bytes + (size_t)h * v.bias_head_stride_bytes +
        group * v.bias_group_stride_bytes);
    return fmaf(scale, code, bias);
}

// A launch covers every query row; decode only visible keys, without an F16 cache copy.
template <int kHeadDim>
__global__ void attn_prefill_part_q8_k(Args a, inference_cuda_attn_kv_pair views) {
    constexpr int kThreads = kHeadDim < 128 ? 128 : kHeadDim;
    __shared__ float dot[kThreads / 32];
    if (!bounds_ok(a)) {
        report_bound_violation(a);
        return;
    }
    const int group = a.n_head / a.n_kv_head;
    const int h = blockIdx.y, kh = h / group, g = h % group, d = threadIdx.x;
    Block b;
    if (!block_of<kHeadDim, 0>(a, blockIdx.x, kh, blockIdx.z, b)) return;
    for (int t = b.t0; t < b.tile_end; ++t) {
        const float q = d < kHeadDim ? a.q[((size_t)t * a.n_head + h) * kHeadDim + d] : 0.f;
        float maximum = -INFINITY, normalizer = 0.f, accumulator = 0.f;
        for (int j = b.k_begin; j < b.k_end; ++j) {
            const int position = b.first + j, last = a.p0 + t;
            if (position > last || (a.window > 0 && position < last - a.window + 1)) continue;
            const int row = a.mode == 1 ? j : (a.ring > 0 ? position % a.ring : position);
            const float product = d < kHeadDim ? q * q8_value(views.k, row, kh, d) : 0.f;
            float partial = product;
            for (int shift = 16; shift > 0; shift >>= 1)
                partial += __shfl_down_sync(0xffffffff, partial, shift);
            if ((d & 31) == 0) dot[d >> 5] = partial;
            __syncthreads();
            float score = 0.f;
#pragma unroll
            for (int w = 0; w < kHeadDim / 32; ++w) score += dot[w];
            score *= a.scale;
            const float next = fmaxf(maximum, score);
            const float alpha = maximum == -INFINITY && next == -INFINITY ? 1.f : expf(maximum - next);
            const float weight = isfinite(score) ? expf(score - next) : __int_as_float(0x7fc00000);
            normalizer = fmaf(normalizer, alpha, weight);
            if (d < kHeadDim)
                accumulator = fmaf(accumulator, alpha, weight * q8_value(views.v, row, kh, d));
            maximum = next;
            __syncthreads();  // all warps must consume this dot before the next key
        }
        const int rr = (t - b.t0) * group + g;
        if (d == 0) {
            b.stats[rr * 2] = maximum;
            b.stats[rr * 2 + 1] = normalizer;
        }
        if (d < kHeadDim) b.accs[(size_t)rr * kHeadDim + d] = accumulator;
    }
}

/* One block per (query head, token): merges the partials of the token's tile's active splits
 * in split-index order (the attn_combine_k pattern, attn_decode.cu:273-327: exact block max,
 * index-order L and A sums), divides and applies the sigmoid output gate. */
template <int kHeadDim, int kGroup>
__global__ void __launch_bounds__(kHeadDim < 128 ? 128 : kHeadDim) attn_prefill_combine_k(Args a) {
    constexpr int kThreads = kHeadDim < 128 ? 128 : kHeadDim;
    const int group = kGroup == 0 ? a.n_head / a.n_kv_head : kGroup;
    const int kRows = kQTile * group;
    __shared__ float s_w[kMaxSplits], s_l[kMaxSplits], s_red[kThreads / 32];
    const int h = blockIdx.x, t = blockIdx.y, d = threadIdx.x;
    const int kvh = h / group, g = h % group;
    float *o = d < kHeadDim ? a.out + ((size_t)t * a.n_head + h) * kHeadDim + d : NULL;
    if (!bounds_ok(a)) {
        report_bound_violation(a);
        if (d < kHeadDim) *o = __int_as_float(0x7fc00000);
        return;
    }
    const int tile = t >> 3;
    const int tile_end = min(a.m, (tile + 1) * kQTile);
    const int n_keys = a.p0 + tile_end;
    int first, count;
    visible(a, a.p0 + tile * kQTile, n_keys, first, count);
    int chunk, n_active;
    if (a.mode == 2) n_active = a.max_splits;
    else split_plan(count, a.max_splits, a.min_keys, chunk, n_active);
    const size_t n_cells = (size_t)((a.m + kQTile - 1) / kQTile) * a.n_kv_head * a.max_splits;
    const size_t cell0 = ((size_t)tile * a.n_kv_head + kvh) * a.max_splits;
    const int rr = (t & (kQTile - 1)) * group + g;
    const float *stats = a.part + cell0 * (kRows * 2) + rr * 2;
    const float *accs = a.part + n_cells * (kRows * 2) + (cell0 * kRows + rr) * kHeadDim +
                        (d < kHeadDim ? d : 0);
    const int kStatsStep = kRows * 2;
    const size_t kAccsStep = (size_t)kRows * kHeadDim;
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
    for (int i = 1; i < kThreads / 32; ++i) M = fmaxf(M, s_red[i]);
    const float sink = a.sinks == NULL ? -INFINITY : a.sinks[h];
    if (a.sinks != NULL) M = fmaxf(M, sink);
    if (d < n_active)
        s_w[d] = ms == -INFINITY && s_l[d] == 0.f ? 0.f : expf(ms - M);
    __syncthreads();
    if (d >= kHeadDim) return;  // D64 still needs 128 threads to load every split statistic.
    float L = 0.f;  // index order
    for (int s = 0; s < n_active; ++s) L = fmaf(s_l[s], s_w[s], L);
    // Sink mass belongs to the merged query head, not each partition or KV head.
    if (a.sinks != NULL && sink != -INFINITY) L += expf(sink - M);
    float A = 0.f;  // index order; the loads do not depend on the weights, so they overlap
#pragma unroll 16
    for (int s = 0; s < n_active; ++s) A = fmaf(accs[(size_t)s * kAccsStep], s_w[s], A);
    const float sig = a.proj == NULL
                          ? 1.f
                          : 1.f / (1.f + expf(-a.proj[(size_t)t * a.ld + (size_t)h * 2 * kHeadDim + kHeadDim + d]));
    // Only a truly neutral empty row is zero; preserve non-finite partial failures.
    *o = (L == 0.f && A == 0.f ? 0.f : A / L) * sig;
}

}  // namespace

namespace {

bool aligned_to(const void *p, uintptr_t a) { return ((uintptr_t)p & (a - 1)) == 0; }

// Prove byte-strided read extents, including a zero stride used for scalar broadcasting.
bool read_span(const void *p, size_t cell, const size_t counts[3], const size_t strides[3]) {
    if (p == NULL || cell == 0) return false;
    const size_t limit = ~uintptr_t(0) - (uintptr_t)p;
    size_t span = cell;
    for (int i = 0; i < 3; ++i) {
        if (counts[i] == 0 || span > limit) return false;
        if (counts[i] > 1) {
            if (strides[i] > (limit - span) / (counts[i] - 1)) return false;
            span += (counts[i] - 1) * strides[i];
        }
    }
    return span <= limit;
}

bool scalar_plane(const void *p, int dtype, const size_t counts[3], const size_t strides[3],
                  bool required) {
    if (p == NULL) return !required && strides[0] == 0 && strides[1] == 0 && strides[2] == 0;
    if (dtype < 0 || dtype > 2) return false;
    const size_t bytes = dtype == 2 ? 4 : 2;
    if (!aligned_to(p, bytes)) return false;
    for (int i = 0; i < 3; ++i)
        if (counts[i] > 1 && strides[i] % bytes != 0) return false;
    return read_span(p, bytes, counts, strides);
}

bool q8_view(const inference_cuda_attn_kv_view &v, const inference_cuda_attn_prefill_args &a) {
    if (v.storage != INFERENCE_CUDA_ATTN_STORAGE_Q8_AFFINE || v.group_size <= 0 ||
        a.head_dim % v.group_size != 0 || (v.q8_signed != 0 && v.q8_signed != 1) ||
        v.q4_low_even != 0)
        return false;
    const size_t payload_counts[3] = {(size_t)a.cache_rows, (size_t)a.n_kv_head, 1};
    const size_t payload_strides[3] = {v.payload_time_stride_bytes, v.payload_head_stride_bytes, 0};
    const size_t counts[3] = {(size_t)a.cache_rows, (size_t)a.n_kv_head,
                              (size_t)a.head_dim / v.group_size};
    const size_t scale[3] = {v.scale_time_stride_bytes, v.scale_head_stride_bytes, v.scale_group_stride_bytes};
    const size_t bias[3] = {v.bias_time_stride_bytes, v.bias_head_stride_bytes, v.bias_group_stride_bytes};
    return read_span(v.payload, a.head_dim, payload_counts, payload_strides) &&
           scalar_plane(v.scales, v.scale_dtype, counts, scale, true) &&
           scalar_plane(v.bias, v.bias_dtype, counts, bias, false);
}

/* The wrapper mirrors every rule here, so the native entry cannot return INVALID_ARGUMENT
 * (which would latch the backend); a C caller gets the same refusal before any launch. */
bool valid(const inference_cuda_attn_prefill_args *a, bool segmented = false) {
    if (a == NULL || a->q == NULL || a->part == NULL || a->out == NULL ||
        a->k_cache == NULL || a->v_cache == NULL)
        return false;
    const bool tagged = a->kv_mode == INFERENCE_CUDA_PREFILL_KV_VIEW;
    if (tagged ? (a->kv_views == NULL || !aligned_to(a->kv_views, alignof(inference_cuda_attn_kv_pair)))
               : (a->kv_mode != INFERENCE_CUDA_PREFILL_KV_FP16 || a->kv_views != NULL))
        return false;
    if (a->kv8_k_codes != NULL || a->kv8_k_scales != NULL || a->kv8_v_codes != NULL ||
        a->kv8_v_scales != NULL || a->kv8_capacity != 0)
        return false;
    if (!aligned_to(a->q, 16) || !aligned_to(a->part, 16) || !aligned_to(a->out, 16)) return false;
    if (a->proj != NULL && !aligned_to(a->proj, 4)) return false;
    if (a->sinks != NULL && !aligned_to(a->sinks, 4)) return false;
    if (a->status != NULL && !aligned_to(a->status, 4)) return false;
    const bool heads = a->n_kv_head >= 1 && a->n_kv_head <= 64 &&
                       a->n_head >= 1 && a->n_head % a->n_kv_head == 0;
    const bool gate_ok = a->proj == NULL ||
                         (long long)a->ld >= (long long)a->n_head * 2 * a->head_dim;
    const bool ring_ok = a->window >= 0 && a->ring >= 0 &&
                         (a->ring == 0 || (a->window > 0 && a->ring <= a->cache_rows &&
                                           (long long)a->ring >= (long long)a->window + a->m - 1));
    const bool bounds = a->p0 >= 0 && (long long)a->p0 + a->m <=
                        (segmented || a->ring > 0 ? kMaxRows : a->cache_rows);
    const bool plan = a->max_splits >= 1 && a->max_splits <= kMaxSplits &&
                      a->min_keys >= kTile && a->min_keys % kTile == 0 &&
                      a->min_keys <= kMaxMinKeys;
    const bool scale_ok = a->scale > 0.f && a->scale < 1e30f && isfinite(a->scale);
    if (!(a->head_dim == 64 || a->head_dim == 128 || a->head_dim == 256) ||
        !heads || !gate_ok || !ring_ok || !bounds || !plan || !scale_ok ||
        (segmented && (a->window != 0 || a->ring != 0)) || a->m < 1 ||
        a->m > kMaxGridY || a->cache_rows < 1 || a->cache_rows > kMaxRows)
        return false;
    if (!tagged)
        return a->head_dim != 64 && (a->n_head == a->n_kv_head || a->n_head == kGroup * a->n_kv_head) &&
               aligned_to(a->k_cache, 16) && aligned_to(a->v_cache, 16);
    const inference_cuda_attn_kv_pair &views = *a->kv_views;
    const size_t floats = inference_cuda_attn_prefill_part_floats(
        a->m, a->n_kv_head, a->head_dim, a->max_splits, a->n_head / a->n_kv_head);
    return a->n_head <= kMaxGridY && floats > 0 && floats <= (64u << 20) / sizeof(float) &&
           a->k_cache == views.k.payload && a->v_cache == views.v.payload &&
           views.k.group_size == views.v.group_size && views.k.q8_signed == views.v.q8_signed &&
           q8_view(views.k, *a) && q8_view(views.v, *a);
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
    k.sinks = (const float *)a->sinks;
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
    k.head_dim = a->head_dim;
    k.cache_rows = a->cache_rows;
    k.max_splits = a->max_splits;
    k.min_keys = a->min_keys;
    k.window = a->window;
    k.ring = a->ring;
    k.scale = a->scale;
    return k;
}

/* The part kernel this process launches, and how. */
struct PartArm {
    int ptx;               // cudaFuncAttributes::ptxVersion of the loaded code
    void (*kernel)(Args);  // attn_prefill_part_dsplit_k (ptx >= 80) or attn_prefill_part_simt_k
    bool dsplit;           // the head_dim-quarter kernel: grid (q_tiles, n_kv_head, max_splits)
    int threads;
    int bytes;             // dynamic shared memory, opted in above the 48 KB default
    cudaError_t err;       // the first failure of the query or the opt-in
};

/* Selects by ptxVersion, the virtual arch the loaded code was compiled for (the __CUDA_ARCH__
 * its #if saw), not by the device's compute capability: a compute_72-only build JIT-compiled
 * on sm_86 still runs the SIMT kernel. Then opts the chosen kernel in to its dynamic shared
 * memory (the vendored gemm.cuh pattern). Both kernels request the same 98,304 B. */
template <int kHeadDim, int kGroup>
PartArm arm_part() {
    constexpr int simt_bytes = kRows * kHeadDim * 2 + 4 * kTile * kHeadDim * 2;
    PartArm arm = {0, attn_prefill_part_simt_k<kHeadDim, kGroup>, false, 128, simt_bytes,
                   cudaSuccess};
    cudaFuncAttributes attr = {};
    arm.err = cudaFuncGetAttributes(&attr, attn_prefill_part_dsplit_k<kHeadDim, kGroup>);
    if (arm.err != cudaSuccess) return arm;
    arm.ptx = attr.ptxVersion;
    if (arm.ptx >= 80) {
        arm.kernel = attn_prefill_part_dsplit_k<kHeadDim, kGroup>;
        arm.dsplit = true;
        arm.threads = kThreadsD;
        arm.bytes = 4 * kTile * kHeadDim * 2 + (kThreadsD / 32) * 2048;
    }
    arm.err = cudaFuncSetAttribute(arm.kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                   arm.bytes);
    return arm;
}

/* Enqueues both kernels. The part kernel is chosen and armed once per process, by a
 * function-local static with C++11 thread-safe initialisation (review m3). A failed arm is
 * cached too, so a transient failure stays permanent for the process; that assumes one
 * device per process, the engine's model. */
template <int kHeadDim, int kGroup>
Failure enqueue_typed(const Args &k, void *stream) {
    Failure failure;
    if (k.mode != 2) {
        static const PartArm arm = arm_part<kHeadDim, kGroup>();
        failure.record(arm.err, INFERENCE_CUDA_STAGE_LAUNCH);
        if (failure.ok()) {
            const int q_tiles = (k.m + kQTile - 1) / kQTile;
            const int splits = k.mode == 1 ? 1 : k.max_splits;
            const dim3 grid = arm.dsplit ? dim3(q_tiles, k.n_kv_head, splits)
                                         : dim3(splits, k.n_kv_head, q_tiles);
            arm.kernel<<<grid, arm.threads, arm.bytes, (cudaStream_t)stream>>>(k);
            failure.record(cudaGetLastError(), INFERENCE_CUDA_STAGE_LAUNCH);
        }
    }
    if (failure.ok() && k.mode != 1) {
        attn_prefill_combine_k<kHeadDim, kGroup><<<dim3(k.n_head, k.m), kHeadDim, 0,
                                                  (cudaStream_t)stream>>>(k);
        failure.record(cudaGetLastError(), INFERENCE_CUDA_STAGE_LAUNCH);
    }
    return failure;
}

#ifdef INFERENCE_CUDA_Q8_REFERENCE
/* Everything in this block is FIXTURE-ONLY (A ruling 01a1839f): the production
 * enqueue and every non-test code path are byte-identical to main; this ifdef compiles
 * away in every production build. The tiled step-2 kernel and its helpers exist only so
 * the same-ELF fixture can drive them through inference_cuda_attn_prefill_q8_tiled_launch. */
/* The group scalars depend only on (row, head, d/32). With group_size 32 a whole warp is one
 * group, so lane 0 can load them once per key and the per-element reloads disappear.
 * Addresses, conversions and the fmaf(scale, code, bias) formula stay identical to q8_value,
 * so every value is bit-identical; only the loads move (A ruling 01a18101). */
struct Q8GroupScalars {
    float scale, bias;
};

__device__ inline Q8GroupScalars q8_group_scalars(
    const inference_cuda_attn_kv_view &v, int row, int h, int group) {
    const float scale = affine_scalar(v.scales, v.scale_dtype,
        (size_t)row * v.scale_time_stride_bytes + (size_t)h * v.scale_head_stride_bytes +
        group * v.scale_group_stride_bytes);
    const float bias = v.bias == NULL ? 0.f : affine_scalar(v.bias, v.bias_dtype,
        (size_t)row * v.bias_time_stride_bytes + (size_t)h * v.bias_head_stride_bytes +
        group * v.bias_group_stride_bytes);
    return {scale, bias};
}

__device__ inline float q8_code(const inference_cuda_attn_kv_view &v, int row, int h, int d) {
    const size_t payload = (size_t)row * v.payload_time_stride_bytes +
                           (size_t)h * v.payload_head_stride_bytes + d;
    return v.q8_signed ? (float)((const signed char *)v.payload)[payload]
                       : (float)((const unsigned char *)v.payload)[payload];
}


/* Tiled q8 part kernel (mode 0; enqueue_q8 selects it when kHeadDim == 128, both planes'
 * group_size is 32 and G <= 8 — Opus design review MF5): one block per (query tile, kv head,
 * split), the F16 dsplit shape, so the G query-head blocks that share a kv head stop
 * re-dequanting the same keys. kTile keys are cooperatively dequantized to smem f32,
 * double-buffered: tile n+1's raw bytes are prefetched into registers before tile n is
 * consumed (byte-granular loads; q8 bytes cannot cp.async on the fixtures' odd strides —
 * SF2). Warp g = GQA head g (padding warps g >= group only keep the barriers, SF3) and scans
 * the tile's tokens; each token's softmax is per key TILE (SF1): kTile butterfly-reduced
 * scores sit in registers, one tile max, one rescale, one expf per key — the per-key serial
 * chain of the scalar kernel is gone. Masking/poison rules, the (maximum, normalizer,
 * accumulator) contract, cell layout rr = (t - t0) * group + g and index order match
 * attn_prefill_part_q8_k; no atomics. */
template <int kHeadDim>
__global__ void __launch_bounds__(256) attn_prefill_part_q8_tile_k(Args a,
                                                                  inference_cuda_attn_kv_pair views) {
    constexpr int kLanes = 32, kWarps = 8;
    constexpr int kDims = kHeadDim / kLanes;       // dims one lane owns
    constexpr int kTileKey = kTile;                // keys per smem tile (the plan granule)
    constexpr int kPlane = kTileKey * kHeadDim;    // floats in one k (or v) plane
    constexpr int kRun = 4, kRunsPerThread = kPlane / kRun / (kLanes * kWarps);
    constexpr int kRaw = kRunsPerThread * kRun;     // prefetch bytes per thread per plane
    constexpr int kStageBytes = 2 * kPlane * 4;     // [k plane][v plane]
    extern __shared__ __align__(16) unsigned char q8tile[];  // [2 stages][k][v]
    if (!bounds_ok(a)) {
        report_bound_violation(a);
        return;
    }
    const int group = a.n_head / a.n_kv_head;
    const int tile = blockIdx.x, kvh = blockIdx.y, s = blockIdx.z;
    const int lane = threadIdx.x & (kLanes - 1), g = threadIdx.x >> 5;
    Block b;
    if (!block_of<kHeadDim, 0>(a, s, kvh, tile, b)) return;  // off-plan split: nothing to write
    const int h = kvh * group + g;                 // this warp's query head (g may pad)
    const unsigned char *kp = (const unsigned char *)views.k.payload;
    const unsigned char *vp = (const unsigned char *)views.v.payload;
    const size_t kt = views.k.payload_time_stride_bytes, khs = views.k.payload_head_stride_bytes;
    const size_t vt = views.v.payload_time_stride_bytes, vhs = views.v.payload_head_stride_bytes;
    unsigned char rawk[kRaw], rawv[kRaw];
    /* Prefetch tile [base, base+kTileKey) into registers; masked tail keys keep stale bytes —
     * store_tile zeroes their smem, and the scan never reads past b.k_end. */
    auto load_tile = [&](int base) {
        const int end = min(base + kTileKey, b.k_end);
#pragma unroll
        for (int r = 0; r < kRunsPerThread; ++r) {
            const int e = (threadIdx.x + r * kLanes * kWarps) * kRun;
            const int j = base + e / kHeadDim, d0 = e % kHeadDim;
            if (j >= end) continue;
            const int position = b.first + j;
            const int row = a.mode == 1 ? j : (a.ring > 0 ? position % a.ring : position);
            const unsigned char *ksrc = kp + (size_t)row * kt + (size_t)kvh * khs + d0;
            const unsigned char *vsrc = vp + (size_t)row * vt + (size_t)kvh * vhs + d0;
#pragma unroll
            for (int i = 0; i < kRun; ++i) {
                rawk[r * kRun + i] = ksrc[i];
                rawv[r * kRun + i] = vsrc[i];
            }
        }
    };
    /* Dequant the PREFETCHED register bytes into one stage's k/v planes (SF1: consuming
     * rawk/rawv here is what makes the double buffer real — reloading the globals would
     * leave the prefetch dead code). With group_size 32 each 4-float run stays inside one
     * group (same-group locality; q8_group_scalars still loads per element). Stores are
     * float4-vectorized (SF3/Sol SF2): kRun 4 puts each lane's run at a 16 B stride, so
     * a quarter-warp phase's 8 lanes touch all 32 banks exactly once — the 8-float runs
     * of the previous respin left a measured-on-paper 2-way conflict, now removed. */
    auto store_tile = [&](int base, int stage) {
        const int end = min(base + kTileKey, b.k_end);
        float *kout = reinterpret_cast<float *>(q8tile + stage * kStageBytes);
        float *vout = kout + kPlane;
#pragma unroll
        for (int r = 0; r < kRunsPerThread; ++r) {
            const int e = (threadIdx.x + r * kLanes * kWarps) * kRun;
            const int j = base + e / kHeadDim, d0 = e % kHeadDim;
            float4 *ko4 = reinterpret_cast<float4 *>(kout + (size_t)(j - base) * kHeadDim + d0);
            float4 *vo4 = reinterpret_cast<float4 *>(vout + (size_t)(j - base) * kHeadDim + d0);
            if (j >= end) {
                const float4 zero = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
                for (int q = 0; q < kRun / 4; ++q) ko4[q] = vo4[q] = zero;
                continue;
            }
            const int position = b.first + j;
            const int row = a.mode == 1 ? j : (a.ring > 0 ? position % a.ring : position);
#pragma unroll
            for (int q = 0; q < kRun / 4; ++q) {
                float kf[4], vf[4];
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const int d = q * 4 + i;
                    const Q8GroupScalars ks =
                        q8_group_scalars(views.k, row, kvh, (d0 + d) >> 5);
                    const Q8GroupScalars vs =
                        q8_group_scalars(views.v, row, kvh, (d0 + d) >> 5);
                    const unsigned char kb = rawk[r * kRun + d], vb = rawv[r * kRun + d];
                    kf[i] = fmaf(ks.scale, views.k.q8_signed ? (float)(signed char)kb
                                                             : (float)kb,
                                 ks.bias);
                    vf[i] = fmaf(vs.scale, views.v.q8_signed ? (float)(signed char)vb
                                                             : (float)vb,
                                 vs.bias);
                }
                ko4[q] = make_float4(kf[0], kf[1], kf[2], kf[3]);
                vo4[q] = make_float4(vf[0], vf[1], vf[2], vf[3]);
            }
        }
    };
    for (int t = b.t0; t < b.tile_end; ++t) {
        float qf[kDims], acc[kDims], m_run = -INFINITY, l_run = 0.f;
#pragma unroll
        for (int i = 0; i < kDims; ++i) {
            qf[i] = g < group ? a.q[((size_t)t * a.n_head + h) * kHeadDim + lane + i * kLanes] : 0.f;
            acc[i] = 0.f;
        }
        const int last = a.p0 + t;
        /* Each token rescans the split's keys (its visibility window differs), so the tile
         * staging restarts per token: the GQA amortization (one dequant per kv head) stays;
         * m>1 cross-token reuse is future work, not the production shape. */
        load_tile(b.k_begin);
        store_tile(b.k_begin, 0);
        __syncthreads();
        for (int base = b.k_begin, stage = 0; base < b.k_end; base += kTileKey, stage ^= 1) {
            const bool more = base + kTileKey < b.k_end;
            if (more) load_tile(base + kTileKey);  // in flight while the warps scan this tile
            const float *const ks = reinterpret_cast<const float *>(q8tile + stage * kStageBytes);
            const float *const vs = ks + kPlane;
            float scores[kTileKey];
            unsigned vis_mask = 0;  // bit j: key j visible (register bitmask, no local mem)
            float tile_max = -INFINITY;
            if (g < group) {
                for (int j = 0; j < kTileKey; ++j) {
                    const int position = b.first + base + j;
                    /* Sol MF1 (owed Opus SF2): skip by VISIBILITY, never by score value —
                     * a visible key whose dot overflows to -inf must poison like the old
                     * kernel (attn_prefill_part_q8_k), not be skipped to a silent 0. */
                    const bool visible = base + j < b.k_end && position <= last &&
                                         (a.window <= 0 || position >= last - a.window + 1);
                    if (!visible) {
                        scores[j] = -INFINITY;  // invisible/out-of-range: skipped
                        continue;
                    }
                    vis_mask |= 1u << j;
                    float dot = 0.f;
#pragma unroll
                    for (int i = 0; i < kDims; ++i)
                        dot = fmaf(qf[i], ks[(size_t)j * kHeadDim + lane + i * kLanes], dot);
#pragma unroll
                    for (int shift = kLanes / 2; shift > 0; shift >>= 1)
                        dot += __shfl_xor_sync(0xffffffff, dot, shift);  // MF4: every lane scores
                    scores[j] = dot * a.scale;  // keep the visible value, even ±inf
                    tile_max = fmaxf(tile_max, scores[j]);
                }
            }            const float m_new = fmaxf(m_run, tile_max);
            const float alpha = m_run == -INFINITY && m_new == -INFINITY
                                    ? 1.f
                                    : expf(m_run - m_new);
            const float m_ref = m_new == -INFINITY ? 0.f : m_new;  // no-key-yet guard
            float l_add = 0.f, vacc[kDims];
#pragma unroll
            for (int i = 0; i < kDims; ++i) vacc[i] = 0.f;
            if (g < group) {
                for (int j = 0; j < kTileKey; ++j) {
                    if (!(vis_mask >> j & 1u)) continue;
                    const float w = isfinite(scores[j])
                                        ? expf(scores[j] - m_ref)
                                        : __int_as_float(0x7fc00000);  // visible non-finite poisons
                    l_add += w;  // key order, deterministic
#pragma unroll
                    for (int i = 0; i < kDims; ++i)
                        vacc[i] = fmaf(w, vs[(size_t)j * kHeadDim + lane + i * kLanes], vacc[i]);
                }
            }
            l_run = fmaf(l_run, alpha, l_add);
#pragma unroll
            for (int i = 0; i < kDims; ++i) acc[i] = fmaf(acc[i], alpha, vacc[i]);
            m_run = m_new;
            if (more) {
                store_tile(base + kTileKey, stage ^ 1);
                __syncthreads();  // next tile visible + this stage free
            }
        }
        if (g < group) {
            const int rr = (t - b.t0) * group + g;
            if (lane == 0) {
                b.stats[rr * 2] = m_run;
                b.stats[rr * 2 + 1] = l_run;
            }
#pragma unroll
            for (int i = 0; i < kDims; ++i)
                b.accs[(size_t)rr * kHeadDim + lane + i * kLanes] = acc[i];
        }
        /* MF2 (Opus bee8b4539): the LAST tile of a token has no barrier of its own (the
         * `more` barrier only fires between tiles), so the next token's store_tile into
         * stage 0 could overwrite data slower warps are still reading. Every warp —
         * padding warps included, hence the guard above instead of `continue` — must
         * reach this barrier once per token. */
        __syncthreads();
    }
}

/* Fixture-only tiled enqueue: the same admission and combine as enqueue_q8, but the part
 * kernel is attn_prefill_part_q8_tile_k<128>. Kept as a separate function so enqueue_q8
 * keeps main's exact text (byte-identity proof). */
template <int kHeadDim>
static Failure enqueue_q8_tiled(const Args &k, inference_cuda_attn_kv_pair views,
                                void *stream) {
    constexpr int threads = kHeadDim < 128 ? 128 : kHeadDim;
    constexpr int kTileBytes = 2 * 2 * kTile * kHeadDim * 4;  // [2 stages][k,v][32 keys][f32]
    Failure failure;
    /* Arm once per process (the arm_part pattern): the no-spill receipt (localSizeBytes
     * == 0) and the 64 KB dynamic-smem opt-in; a failed arm fails every launch, never
     * silently falls back. No 16-key fallback (dead code at the production block count). */
    static const bool armed = [] {
        cudaFuncAttributes attr = {};
        if (cudaFuncGetAttributes(&attr, attn_prefill_part_q8_tile_k<128>) != cudaSuccess)
            return false;
        if (attr.maxThreadsPerBlock < 256 || attr.localSizeBytes != 0) return false;
        return cudaFuncSetAttribute(attn_prefill_part_q8_tile_k<128>,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    kTileBytes) == cudaSuccess;
    }();
    if (!armed) {
        failure.record(cudaErrorInvalidValue, INFERENCE_CUDA_STAGE_LAUNCH);
        return failure;
    }
    attn_prefill_part_q8_tile_k<128><<<dim3((k.m + kQTile - 1) / kQTile, k.n_kv_head,
                                           k.max_splits),
                                      256, kTileBytes, (cudaStream_t)stream>>>(k, views);
    failure.record(cudaGetLastError(), INFERENCE_CUDA_STAGE_LAUNCH);
    if (failure.ok() && k.mode != 1) {
        attn_prefill_combine_k<kHeadDim, 0><<<dim3(k.n_head, k.m), threads, 0,
                                             (cudaStream_t)stream>>>(k);
        failure.record(cudaGetLastError(), INFERENCE_CUDA_STAGE_LAUNCH);
    }
    return failure;
}
#endif
template <int kHeadDim>
Failure enqueue_q8(const Args &k, inference_cuda_attn_kv_pair views, void *stream) {
    constexpr int threads = kHeadDim < 128 ? 128 : kHeadDim;
    Failure failure;
    if (k.mode != 2) {
        attn_prefill_part_q8_k<kHeadDim><<<dim3(k.mode == 1 ? 1 : k.max_splits, k.n_head,
                                               (k.m + kQTile - 1) / kQTile),
                                          threads, 0, (cudaStream_t)stream>>>(k, views);
        failure.record(cudaGetLastError(), INFERENCE_CUDA_STAGE_LAUNCH);
    }
    if (failure.ok() && k.mode != 1) {
        attn_prefill_combine_k<kHeadDim, 0><<<dim3(k.n_head, k.m), threads, 0,
                                             (cudaStream_t)stream>>>(k);
        failure.record(cudaGetLastError(), INFERENCE_CUDA_STAGE_LAUNCH);
    }
    return failure;
}

Failure enqueue(const Args &k, void *stream, const inference_cuda_attn_kv_pair *views) {
    if (views != NULL) {
        if (k.head_dim == 64) return enqueue_q8<64>(k, *views, stream);
        if (k.head_dim == 128) return enqueue_q8<128>(k, *views, stream);
        return enqueue_q8<256>(k, *views, stream);
    }
    if (k.head_dim == 128) {
        return k.n_head == k.n_kv_head ? enqueue_typed<128, 1>(k, stream)
                                       : enqueue_typed<128, 8>(k, stream);
    }
    return k.n_head == k.n_kv_head ? enqueue_typed<256, 1>(k, stream)
                                   : enqueue_typed<256, 8>(k, stream);
}

}  // namespace

/* Floats of `part` for the rule above; the Rust side pins the same numbers
 * (cpf_attn_part_bytes). */
extern "C" size_t inference_cuda_attn_prefill_part_floats(int m, int n_kv_head, int head_dim,
                                                          int max_splits, int group) {
    if (m < 1 || m > kMaxGridY || n_kv_head < 1 || n_kv_head > 64 ||
        (head_dim != 64 && head_dim != 128 && head_dim != 256) || group < 1 ||
        max_splits < 1 || max_splits > kMaxSplits)
        return 0;
    size_t floats = (size_t)((m + kQTile - 1) / kQTile);
    const size_t factors[] = {kQTile, (size_t)n_kv_head, (size_t)max_splits,
                              (size_t)group, (size_t)(2 + head_dim)};
    for (size_t factor : factors) {
        if (floats > SIZE_MAX / factor) return 0;
        floats *= factor;
    }
    return floats;
}

extern "C" int inference_cuda_attn_prefill_launch(const inference_cuda_attn_prefill_args *a,
                                                  void *stream, int *out_stage) {
    try {
        report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
        if (!valid(a)) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        const Args k = to_args(a);
        const Failure failure = enqueue(k, stream, a->kv_views);
        report_stage(out_stage, failure.stage);
        return (int)failure.err;
    } catch (...) {
        return INFERENCE_CUDA_NATIVE_EXCEPTION;
    }
}

extern "C" int inference_cuda_attn_prefill_part_launch(
    const inference_cuda_attn_prefill_args *a, int key_origin, int key_rows, int part_index,
    void *stream, int *out_stage) {
    try {
        report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
        if (stream == nullptr || !valid(a, true) ||
            inference_cuda_attn_prefill_part_floats(a->m, a->n_kv_head, a->head_dim,
                                                   a->max_splits, a->n_head / a->n_kv_head) >
                (64u << 20) / sizeof(float) ||
            key_origin < 0 || key_rows <= 0 || key_rows > a->cache_rows ||
            (long long)key_origin + key_rows > (long long)a->p0 + a->m ||
            part_index < 0 || part_index >= a->max_splits)
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        Args k = to_args(a);
        k.mode = 1;
        k.key_origin = key_origin;
        k.key_rows = key_rows;
        k.part_index = part_index;
        const Failure failure = enqueue(k, stream, a->kv_views);
        report_stage(out_stage, failure.stage);
        return (int)failure.err;
    } catch (...) {
        return INFERENCE_CUDA_NATIVE_EXCEPTION;
    }
}

extern "C" int inference_cuda_attn_prefill_combine_launch(
    const inference_cuda_attn_prefill_args *a, void *stream, int *out_stage) {
    try {
        report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
        if (stream == nullptr || !valid(a, true) ||
            inference_cuda_attn_prefill_part_floats(a->m, a->n_kv_head, a->head_dim,
                                                   a->max_splits, a->n_head / a->n_kv_head) >
                (64u << 20) / sizeof(float))
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        Args k = to_args(a);
        k.mode = 2;
        const Failure failure = enqueue(k, stream, a->kv_views);
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
         * (no partial, NaN out, mask 8) are the code under test. */
        if (!valid_sans_bounds(a)) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        Args k = to_args(a);
        k.p0 = -1;  // Invalid for both linear and absolute ring positions.
        const Failure failure = enqueue(k, stream, a->kv_views);
        report_stage(out_stage, failure.stage);
        return (int)failure.err;
    } catch (...) {
        return INFERENCE_CUDA_NATIVE_EXCEPTION;
    }
}
#endif
#ifdef INFERENCE_CUDA_Q8_REFERENCE
/* Same-ELF fixture entry (A ruling 01a18101): the OLD-path baseline. On this branch the
 * per-warp hoist is gone (A ruling 01a1839f), so the reference arm IS the default
 * production enqueue — identical admission, identical kernels. Kept as a separate symbol
 * so the fixture's arm routing and receipts stay stable. */
extern "C" int inference_cuda_attn_prefill_q8_reference_launch(
    const inference_cuda_attn_prefill_args *a, void *stream, int *out_stage) {
    try {
        report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
        if (!valid(a) || a->kv_views == NULL) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        const Args k = to_args(a);
        const Failure failure = enqueue(k, stream, a->kv_views);
        report_stage(out_stage, failure.stage);
        return (int)failure.err;
    } catch (...) {
        return INFERENCE_CUDA_NATIVE_EXCEPTION;
    }
}
#endif
#ifdef INFERENCE_CUDA_Q8_REFERENCE
/* Same-ELF fixture entry (MF1, Opus bee8b4539): the ONLY caller that can reach the tiled
 * kernel; production launches never do. Fixture builds only (A ruling 01a1839f: the
 * production enqueue is byte-identical to main; this entry drives the fixture-only
 * enqueue_q8_tiled). The entry REFUSES shapes the tiled predicate would reject, so a
 * fixture run can never silently measure the old kernel under a tiled label. */
extern "C" int inference_cuda_attn_prefill_q8_tiled_launch(
    const inference_cuda_attn_prefill_args *a, void *stream, int *out_stage) {
    try {
        report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
        if (!valid(a) || a->kv_views == NULL) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        const Args k = to_args(a);
        const inference_cuda_attn_kv_pair &views = *a->kv_views;
        if (k.head_dim != 128 || k.mode != 0 || k.n_head / k.n_kv_head > 8 ||
            views.k.group_size != 32 || views.v.group_size != 32) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;  // refuse silently-degraded routing
        }
        const Failure failure = enqueue_q8_tiled<128>(k, views, stream);
        report_stage(out_stage, failure.stage);
        return (int)failure.err;
    } catch (...) {
        return INFERENCE_CUDA_NATIVE_EXCEPTION;
    }
}
#endif
