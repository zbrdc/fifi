/* CPF-ATTN-A K2 and CPF-ATTN-B: the kernels and the launch entry of attn_prefill, the
 * flash-style chunked-prefill attention (decision 0049 items 1, 2 and 10; the CPF-5a/5b kernel
 * of design section 3.7). Design guidance only, no code copied: the split rule, the
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

constexpr int kHeadDim = 256;   // the only head size (the wrapper refuses every other value)
constexpr int kGroup = 8;       // query heads per kv head: the compiled group
constexpr int kTile = 32;       // keys per tile (the granule of a key range)
constexpr int kQTile = 8;       // tokens per query tile: the compiled tile
constexpr int kRows = kQTile * kGroup;  // 64 rows per query tile
constexpr int kMaxGridY = 65535;
constexpr int kMaxRows = (1 << 30) - 1;
constexpr int kMaxSplits = 128;
constexpr int kMaxMinKeys = 1 << 20;
/* Dynamic shared memory of both part kernels: 96 KB = 98,304 B, above the 48 KB default (the
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
 * the first barrier. ORs mask 8 once; the caller's writes (or their absence) are the callers'. */
__device__ inline bool bounds_ok(const Args &a) {
    return (long long)a.p0 >= 0 && (long long)a.p0 + a.m <= a.cache_rows;
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

/* The block of split s, kv head kvh and query tile `tile`; false when the split holds no key
 * of the tile (the block then exits before any barrier). */
__device__ inline bool block_of(const Args &a, int s, int kvh, int tile, Block &b) {
    b.t0 = tile * kQTile;
    b.tile_end = min(a.m, b.t0 + kQTile);
    b.n_keys = a.p0 + b.tile_end;  // keys 0..p0 + tile_end - 1 exist for this tile
    int count;
    visible(a, a.p0 + b.t0, b.n_keys, b.first, count);
    int chunk, n_active;
    split_plan(count, a.max_splits, a.min_keys, chunk, n_active);
    if (s >= n_active) return false;
    b.k_begin = s * chunk;
    b.k_end = min(count, b.k_begin + chunk);
    b.max_p = b.n_keys - 1;  // the tile's last row's own position: the deepest key read
    /* The partial cell: 64 stat pairs then, after every cell's stats, this cell's 64 x 256
     * accumulators (the decode partial layout with the rows rounded to the tile). */
    const size_t n_cells = (size_t)((a.m + kQTile - 1) / kQTile) * a.n_kv_head * a.max_splits;
    const size_t cell = ((size_t)tile * a.n_kv_head + kvh) * a.max_splits + s;
    b.stats = a.part + cell * (kRows * 2);
    b.accs = a.part + n_cells * (kRows * 2) + cell * ((size_t)kRows * kHeadDim);
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
__device__ inline void load_q_tile(const Args &a, const Block &b, int kvh, __half *sq) {
    for (int i = threadIdx.x; i < kRows * kHeadDim; i += 128) {
        const int r = i / kHeadDim, d = i % kHeadDim;
        const int t = b.t0 + (r >> 3);
        sq[i] = t < b.tile_end
                    ? __float2half(a.q[((size_t)t * a.n_head + kvh * kGroup + (r & 7)) * kHeadDim + d])
                    : __half(0);
    }
}

/* Loads the key tile at visible index base into sk/sv (128 threads). Keys beyond the range
 * hold zeros, so a masked or out-of-range key contributes exactly nothing (0 * V = 0) and a
 * NaN cell no row may read never enters the tile. */
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
    if (!block_of(a, s, kvh, tile, b)) return;
    load_q_tile(a, b, kvh, sq);
    const int row = threadIdx.x >> 1, half = threadIdx.x & 1;
    const int p_row = a.p0 + b.t0 + (row >> 3);
    float o_acc[kHeadDim / 2];
#pragma unroll
    for (int e = 0; e < kHeadDim / 2; ++e) o_acc[e] = 0.f;
    float m_run = -INFINITY, l_run = 0.f;
    for (int base = b.k_begin; base < b.k_end && b.first + base <= b.max_p; base += kTile) {
        load_kv_tile(a, b, kvh, base, sk, sv);
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
        b.stats[row * 2] = m_run;
        b.stats[row * 2 + 1] = l_run;
    }
#pragma unroll 8
    for (int e = 0; e < kHeadDim / 2; ++e)
        b.accs[(size_t)row * kHeadDim + half * (kHeadDim / 2) + e] = o_acc[e];
#endif
}

#if __CUDA_ARCH__ >= 800
constexpr float kLog2e = 1.4426950408889634f;  // scores in log2 units: exp2(s * c2) = exp(s * scale)
constexpr float kLn2 = 0.6931471805599453f;    // the stored max back in natural units

/* The primitives of the head_dim-quarter kernel (sm_80+ only, so no device pass sees one
 * unreferenced). Stage planes are [32 keys][512 B]; the 16-byte chunk j of key k sits at
 * byte k * 512 + ((j ^ (k & 7)) << 4): the XOR stays inside a warp's 8-chunk quarter, and
 * the 8 row addresses of every 8x8 ldmatrix hit 8 distinct chunks of one 128-byte span. */
__device__ inline unsigned swz(int key, int chunk) {
    return key * 512 + ((chunk ^ (key & 7)) << 4);
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
__device__ inline void issue_kv_tile(const Args &a, const Block &b, int kvh, int base,
                                     unsigned stage) {
    const int chunk = threadIdx.x & 31;
#pragma unroll
    for (int half = 0; half < 2; ++half) {
        const int key = (threadIdx.x >> 5) + 16 * half;
        const bool valid = base + key < b.k_end;
        const size_t row = (size_t)cache_row(a, b.first, base + (valid ? key : 0));
        const size_t at = (row * a.n_kv_head + kvh) * kHeadDim + chunk * 8;
        const unsigned dst = stage + swz(key, chunk);
        cp_async16(dst, a.k_cache + at, valid ? 16 : 0);
        cp_async16(dst + kPlaneBytes, a.v_cache + at, valid ? 16 : 0);
    }
    cp_async_commit();
}

/* The lane's Q̂ A-fragments for rows r0 = 16 rg + g (token t0 + 2 rg) and r1 = r0 + 8 (the
 * next token), head 8 kvh + g, dims 64 dq + 16 ks + {c, c+1, c+8, c+9}: rounded to fp16
 * once, as CPF-ATTN-A's q̂; a token at or past tile_end loads zeros and reads no q. */
__device__ inline void load_q_frags(const Args &a, const Block &b, int kvh, int rg, int dq,
                                    unsigned (&qf)[4][4]) {
    const int lane = threadIdx.x & 31, g = lane >> 2, c = (lane & 3) * 2;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int t = b.t0 + 2 * rg + i;
        const float *q = a.q + ((size_t)t * a.n_head + kvh * kGroup + g) * kHeadDim + 64 * dq + c;
#pragma unroll
        for (int ks = 0; ks < 4; ++ks)
#pragma unroll
            for (int hi = 0; hi < 2; ++hi) {
                float2 v = make_float2(0.f, 0.f);
                if (t < b.tile_end) v = *reinterpret_cast<const float2 *>(q + 16 * ks + 8 * hi);
                const __half2 h = __floats2half2_rn(v.x, v.y);
                qf[ks][i + 2 * hi] = *reinterpret_cast<const unsigned *>(&h);
            }
    }
}
#endif

/* One block of the head_dim-quarter kernel (sm_80+): the block unit of CPF-ATTN-A's part kernel
 * (one query tile of 64 rows, one kv head, one split's keys; the same partial cell) over 16
 * warps. Warp w holds row group rg = w >> 2 (rows 16 rg .. +15) and head_dim quarter
 * dq = w & 3 (dims 64 dq .. +63): Q̂ in registers, its quarter of S = Q̂·K^T, and O for its
 * rows and dims. The four quarter partials of S are summed through shared memory in one
 * fixed order, so the four warps of a row group hold identical statistics and P̂.
 * K and V arrive by cp.async into two swizzled stages (one block barrier per tile) and
 * reach the MMAs by ldmatrix. Grid (q_tiles, n_kv_head, max_splits), the heaviest causal
 * tiles first; an sm_72 build traps here. */
__global__ void __launch_bounds__(kThreadsD, 1) attn_prefill_part_dsplit_k(Args a) {
#if __CUDA_ARCH__ >= 800
    if (!bounds_ok(a)) {
        report_bound_violation(a);
        return;  // no partial
    }
    extern __shared__ __align__(16) unsigned char dsmem[];
    const int q_tiles = (a.m + kQTile - 1) / kQTile;
    const int tile = q_tiles - 1 - (int)blockIdx.x, kvh = blockIdx.y, s = blockIdx.z;
    Block b;
    if (!block_of(a, s, kvh, tile, b)) return;
    const unsigned stages = smem_addr(dsmem);
    float4 *const xch = reinterpret_cast<float4 *>(dsmem + 2 * kStageBytes);
    issue_kv_tile(a, b, kvh, b.k_begin, stages);

    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int rg = warp >> 2, dq = warp & 3;
    const int g = lane >> 2, c = (lane & 3) * 2;
    unsigned qf[4][4];
    load_q_frags(a, b, kvh, rg, dq, qf);
    const int p_row0 = a.p0 + b.t0 + 2 * rg, p_row1 = p_row0 + 1;  // rows r0, r1
    /* scale * log2(e), once per block. Every score product is __fmul_rn, in the masked and the
     * fast path alike: under -fmad=true a plain product could contract into the FFMA of
     * s2 - m_ref in one path only (lane C MINOR-1). */
    const float c2 = __fmul_rn(a.scale, kLog2e);
    /* ldmatrix lane roles: matrix mi = lane >> 3, row rr = lane & 7. K: key 16 np + 8 (mi >> 1)
     * + rr, chunk 8 dq + 2 ks + (mi & 1). V: key 16 ks + 8 (mi & 1) + rr, chunk 8 dq + 2 jp +
     * (mi >> 1). */
    const int mi = lane >> 3, rr = lane & 7;
    const int k_key = 8 * (mi >> 1) + rr, k_chunk = 8 * dq + (mi & 1);
    const int v_key = 8 * (mi & 1) + rr, v_chunk = 8 * dq + (mi >> 1);

    float o[8][4];
#pragma unroll
    for (int j = 0; j < 8; ++j)
#pragma unroll
        for (int e = 0; e < 4; ++e) o[j][e] = 0.f;
    float m_run[2] = {-INFINITY, -INFINITY}, l_run[2] = {0.f, 0.f};

    int n = 0;
    for (int base = b.k_begin; base < b.k_end && b.first + base <= b.max_p; base += kTile, ++n) {
        cp_async_wait_all();
        __syncthreads();  // tile n visible; stage (n + 1) & 1 and the exchange free
        const int next = base + kTile;
        if (next < b.k_end && b.first + next <= b.max_p)
            issue_kv_tile(a, b, kvh, next, stages + ((n + 1) & 1) * kStageBytes);
        const unsigned kst = stages + (n & 1) * kStageBytes, vst = kst + kPlaneBytes;

        /* This quarter's S: 4 k16 steps x 4 n8 key blocks. */
        float sq[4][4];
#pragma unroll
        for (int nb = 0; nb < 4; ++nb)
#pragma unroll
            for (int e = 0; e < 4; ++e) sq[nb][e] = 0.f;
#pragma unroll
        for (int ks = 0; ks < 4; ++ks)
#pragma unroll
            for (int np = 0; np < 2; ++np) {
                unsigned kb[4];
                ldsm_x4(kst + swz(16 * np + k_key, k_chunk + 2 * ks), kb);
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
        /* Scale and mask the block (log2 units; the fully visible tile skips the mask). */
        const bool full = b.first + base + kTile - 1 <= a.p0 + b.t0 &&
                          (a.window == 0 || b.first + base > b.max_p - a.window);
        float s2[4];
        if (full) {
            s2[0] = __fmul_rn(sum4.x, c2);
            s2[1] = __fmul_rn(sum4.y, c2);
            s2[2] = __fmul_rn(sum4.z, c2);
            s2[3] = __fmul_rn(sum4.w, c2);
        } else {
            const int pos = b.first + base + dq * 8 + c;
            s2[0] = sees(a, pos, p_row0) ? __fmul_rn(sum4.x, c2) : -INFINITY;
            s2[1] = sees(a, pos + 1, p_row0) ? __fmul_rn(sum4.y, c2) : -INFINITY;
            s2[2] = sees(a, pos, p_row1) ? __fmul_rn(sum4.z, c2) : -INFINITY;
            s2[3] = sees(a, pos + 1, p_row1) ? __fmul_rn(sum4.w, c2) : -INFINITY;
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
        const float p[4] = {exp2f(s2[0] - m_ref[0]), exp2f(s2[1] - m_ref[0]),
                            exp2f(s2[2] - m_ref[1]), exp2f(s2[3] - m_ref[1])};
        float ps[2] = {p[0] + p[1], p[2] + p[3]};
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
        const float corr[2] = {exp2f(m_run[0] - m_ref[0]), exp2f(m_run[1] - m_ref[1])};
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
        for (int j = 0; j < 8; ++j) {
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
            for (int jp = 0; jp < 4; ++jp) {
                unsigned vb[4];
                ldsm_x4_trans(vst + swz(16 * ks + v_key, v_chunk + 2 * jp), vb);
                mma_16816(o[2 * jp], pa, vb[0], vb[1]);
                mma_16816(o[2 * jp + 1], pa, vb[2], vb[3]);
            }
        }
    }
    const int r0 = 16 * rg + g, r1 = r0 + 8;
    /* The max goes out in natural units (m2 * ln2), so the combine's expf(ms - M) stays. */
    if (dq == 0 && (lane & 3) == 0) {
        b.stats[r0 * 2] = __fmul_rn(m_run[0], kLn2);
        b.stats[r0 * 2 + 1] = l_run[0];
        b.stats[r1 * 2] = __fmul_rn(m_run[1], kLn2);
        b.stats[r1 * 2 + 1] = l_run[1];
    }
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const int dim = 64 * dq + 8 * j + c;
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
PartArm arm_part() {
    PartArm arm = {0, attn_prefill_part_simt_k, false, 128, kSmemBytes, cudaSuccess};
    cudaFuncAttributes attr = {};
    arm.err = cudaFuncGetAttributes(&attr, attn_prefill_part_dsplit_k);
    if (arm.err != cudaSuccess) return arm;
    arm.ptx = attr.ptxVersion;
    if (arm.ptx >= 80) {
        arm.kernel = attn_prefill_part_dsplit_k;
        arm.dsplit = true;
        arm.threads = kThreadsD;
    }
    arm.err = cudaFuncSetAttribute(arm.kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                   arm.bytes);
    return arm;
}

/* Enqueues both kernels. The part kernel is chosen and armed once per process, by a
 * function-local static with C++11 thread-safe initialisation (review m3). A failed arm is
 * cached too, so a transient failure stays permanent for the process; that assumes one
 * device per process, the engine's model. */
Failure enqueue(const Args &k, void *stream) {
    static const PartArm arm = arm_part();
    Failure failure;
    failure.record(arm.err, INFERENCE_CUDA_STAGE_LAUNCH);
    if (failure.ok()) {
        const int q_tiles = (k.m + kQTile - 1) / kQTile;
        const dim3 grid = arm.dsplit ? dim3(q_tiles, k.n_kv_head, k.max_splits)
                                     : dim3(k.max_splits, k.n_kv_head, q_tiles);
        arm.kernel<<<grid, arm.threads, arm.bytes, (cudaStream_t)stream>>>(k);
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
         * (no partial, NaN out, mask 8) are the code under test. */
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
