/*
 * Embedding row gather of the resident Qwen3.6 decode step.
 *
 * The embedding table stays on the device and the token id is read there, so a captured step
 * needs no host-known token. The code is original CUDA C for sm_86 (sm_72 compiles), written from
 * the tile layout of this repository (vendored xavier-qwen repack.cu and mma884.cuh).
 *
 * What it computes: for token t, x[t][0 .. k-1] = row ids[t] of the tiled token embedding as f32. The
 * kernel dequantizes with the order of the CPU dequantizer (gguf_dequant.rs), every operation rounded
 * alone through __fmul_rn and __fsub_rn, so the debug device build (-G, which may contract) and the
 * release build give the same bits as the host. Because d * sc has at most 17 significant bits and
 * (d * sc) * q at most 21, both products are exact in f32 and the only rounding is the final subtract.
 *
 * Tile addressing (repack.cu pack_q4k / pack_q8_0, the same tiles the matmul reads). Row r of the
 * table lives in row group g = r / 8, row r % 8 of the tile (g * nb + ib), nb = k / 256, ib the
 * 256-column block. A lane of the 32-lane fragment layout is lane(r, qp) = (r & 3) | (qp << 2) |
 * ((r >> 2) << 4) for the row r % 8 and the quadpair qp (columns 64 qp .. 64 qp + 63 of the block).
 *   Q4_K tile, 72 vectors of 16 B: vector r % 8 is the head of row r % 8 (d, dmin, 12 scale bytes);
 *     vector 8 + 32 h + lane(r, qp) holds the 16 qs bytes 32 qp + 16 h .. +15 of that row. The low
 *     nibbles are columns 64 qp + 16 h + l, the high nibbles columns 64 qp + 32 + 16 h + l (l < 16),
 *     with the sub-blocks 2 qp (low) and 2 qp + 1 (high).
 *   Q8_0 tile, 136 vectors: the first 8 vectors are the 8 x 8 fp16 scales (row-major, 8 blocks of 32
 *     columns per row); vector 8 + 32 j + lane(r, qp) holds the 16 int8 quants of columns
 *     64 qp + 16 j .. +15, scale entry (r % 8) * 8 + (64 qp + 16 j) / 32.
 * One thread owns one 16-byte vector of one row: it loads the vector (and the head or the scale),
 * decodes 16 values (Q8_0) or 32 values (Q4_K, 16 low + 16 high) and stores them as float4.
 *
 * A token id outside [0, n_rows) (a poisoned step word is -1, an all-NaN argmax row gives -1) makes
 * every thread of the row store NaN and read no weight and no head, and thread 0 of that block ORs
 * status bit 4 (invalid step state, the existing bit; the host message names both causes) into the
 * step status word with one integer atomicOr, the pattern of the poison-guard patch of attn_prep.
 * The kernel writes no other bit, clears none and leaves the word alone for valid ids. The ids are
 * read once per thread.
 * No float atomic, no shared memory, no barrier; the same input twice gives equal bytes.
 */
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stddef.h>
#include <stdint.h>

#include "abi.h"
#include "device.h"
#include "resident/embed.h"
#include "resident/step_params.h"  // INFERENCE_CUDA_STATUS_INVALID_STEP_STATE

namespace {

constexpr int kTileK = 256;
constexpr int kQ4Vecs = 72;      // uint4 per Q4_K tile (mma884::Q4_K::kTileVecs)
constexpr int kQ8Vecs = 136;     // uint4 per Q8_0 tile (mma884::Q8_0::kTileVecs)
constexpr int kQ4Threads = 64;   // a row of k = 2048 has 8 blocks x 8 vectors
constexpr int kQ8Threads = 128;  // a row of k = 2048 has 8 blocks x 16 vectors
constexpr int kMaxTokens = 4096;
constexpr int kMaxRows = 1 << 28;
constexpr int kMaxK = 32768;
constexpr unsigned kNanBits = 0x7fc00000u;
constexpr int kMaskInvalidState = INFERENCE_CUDA_STATUS_INVALID_STEP_STATE;

__device__ __forceinline__ int lane_of(int r, int qp) { return (r & 3) | (qp << 2) | ((r >> 2) << 4); }

__device__ __forceinline__ float half_to_f32(unsigned bits) { return __half2float(__ushort_as_half((unsigned short)bits)); }

__device__ __forceinline__ unsigned byte_of(const uint4 &v, int i) {
    const unsigned w = (i >> 2) == 0 ? v.x : (i >> 2) == 1 ? v.y : (i >> 2) == 2 ? v.z : v.w;
    return (w >> (8 * (i & 3))) & 0xffu;
}

__device__ __forceinline__ void store4(float *dst, float a, float b, float c, float d) {
    *reinterpret_cast<float4 *>(dst) = make_float4(a, b, c, d);
}

__device__ __forceinline__ void store_nan16(float *dst) {
    const float nan = __uint_as_float(kNanBits);
    for (int i = 0; i < 16; i += 4) store4(dst + i, nan, nan, nan, nan);
}

/* ggml get_scale_min_k4 on the 12 scale bytes that follow d and dmin in the head (head bytes 4 .. 15). */
__device__ __forceinline__ void scale_min(int j, const uint4 &head, int &sc, int &mn) {
    if (j < 4) {
        sc = (int)(byte_of(head, 4 + j) & 63u);
        mn = (int)(byte_of(head, 8 + j) & 63u);
    } else {
        sc = (int)((byte_of(head, 8 + j) & 0xFu) | ((byte_of(head, j) >> 6) << 4));
        mn = (int)((byte_of(head, 8 + j) >> 4) | ((byte_of(head, 4 + j) >> 6) << 4));
    }
}

__global__ void __launch_bounds__(kQ4Threads)
    embed_q4k_k(const uint4 *tiles, const int *ids, float *x, int *status, int n_rows, int k) {
    const int t = blockIdx.x;
    const int id = ids[t];
    const bool ok = id >= 0 && id < n_rows;
    if (!ok && threadIdx.x == 0) {
        atomicOr(status, kMaskInvalidState);  // one thread, one integer OR: idempotent, order independent
    }
    const int nb = k / kTileK;
    float *row = x + (size_t)t * k;
    const int r = ok ? (id & 7) : 0;
    const size_t g = ok ? (size_t)(id >> 3) : 0;
    for (int w = threadIdx.x; w < nb * 8; w += kQ4Threads) {
        const int ib = w >> 3, qp = (w >> 1) & 3, h = w & 1;
        float *dst = row + ib * kTileK + 64 * qp + 16 * h;  // low columns; the high ones are dst + 32
        if (!ok) {
            store_nan16(dst);
            store_nan16(dst + 32);
            continue;
        }
        const uint4 *tile = tiles + (g * nb + ib) * kQ4Vecs;
        const uint4 head = tile[r];
        const uint4 qs = tile[8 + 32 * h + lane_of(r, qp)];
        const float d = half_to_f32(head.x & 0xffffu), dmin = half_to_f32(head.x >> 16);
        int sc_lo, mn_lo, sc_hi, mn_hi;
        scale_min(2 * qp, head, sc_lo, mn_lo);
        scale_min(2 * qp + 1, head, sc_hi, mn_hi);
        const float d_lo = __fmul_rn(d, (float)sc_lo), m_lo = __fmul_rn(dmin, (float)mn_lo);
        const float d_hi = __fmul_rn(d, (float)sc_hi), m_hi = __fmul_rn(dmin, (float)mn_hi);
        for (int l = 0; l < 16; l += 4) {
            float lo[4], hi[4];
            for (int c = 0; c < 4; ++c) {
                const unsigned b = byte_of(qs, l + c);
                lo[c] = __fsub_rn(__fmul_rn(d_lo, (float)(b & 0xFu)), m_lo);
                hi[c] = __fsub_rn(__fmul_rn(d_hi, (float)(b >> 4)), m_hi);
            }
            store4(dst + l, lo[0], lo[1], lo[2], lo[3]);
            store4(dst + 32 + l, hi[0], hi[1], hi[2], hi[3]);
        }
    }
}

__global__ void __launch_bounds__(kQ8Threads)
    embed_q80_k(const uint4 *tiles, const int *ids, float *x, int *status, int n_rows, int k) {
    const int t = blockIdx.x;
    const int id = ids[t];
    const bool ok = id >= 0 && id < n_rows;
    if (!ok && threadIdx.x == 0) {
        atomicOr(status, kMaskInvalidState);  // one thread, one integer OR: idempotent, order independent
    }
    const int nb = k / kTileK;
    float *row = x + (size_t)t * k;
    const int r = ok ? (id & 7) : 0;
    const size_t g = ok ? (size_t)(id >> 3) : 0;
    for (int w = threadIdx.x; w < nb * 16; w += kQ8Threads) {
        const int ib = w >> 4, qp = (w >> 2) & 3, j = w & 3;
        const int kk = 64 * qp + 16 * j;
        float *dst = row + ib * kTileK + kk;
        if (!ok) {
            store_nan16(dst);
            continue;
        }
        const uint4 *tile = tiles + (g * nb + ib) * kQ8Vecs;
        const uint4 vec = tile[8 + 32 * j + lane_of(r, qp)];
        const unsigned *scales = reinterpret_cast<const unsigned *>(tile);  // 8 x 8 fp16, two per word
        const int entry = r * 8 + (kk >> 5);
        const float d = half_to_f32((scales[entry >> 1] >> (16 * (entry & 1))) & 0xffffu);
        for (int l = 0; l < 16; l += 4) {
            float v[4];
            for (int c = 0; c < 4; ++c) v[c] = __fmul_rn(d, (float)(int)(int8_t)byte_of(vec, l + c));
            store4(dst + l, v[0], v[1], v[2], v[3]);
        }
    }
}

/* The same Q4_K gather against the file's own blocks (row-major, one block of
 * 144 bytes per 256 values: d, dmin, 12 scale bytes, then 128 qs bytes). A row of k values is
 * k / 256 consecutive blocks, so block ib of row id starts at uint4 index (id * nb + ib) * 9 and
 * every uint4 load is aligned (144 is a multiple of 16 and the base is 16-byte aligned). One
 * thread owns the same (block, quadpair qp, half h) unit as the tiled kernel: head = block[0]
 * (d, dmin and the 12 scale bytes, byte for byte the tile head) and qs = block[1 + 2 * qp + h]
 * (qs bytes 32 qp + 16 h .. +15 of the block; the low nibbles are columns 64 qp + 16 h + l, the
 * high nibbles 64 qp + 32 + 16 h + l). The arithmetic is the tiled kernel's, operation for
 * operation, so the two gathers are bit equal on the same weights. The row count needs no 8-row
 * rule: any n_rows in [1, 2^28] inclusive. */
__global__ void __launch_bounds__(kQ4Threads)
    embed_q4k_native_k(const uint4 *blocks, const int *ids, float *x, int *status, int n_rows, int k) {
    const int t = blockIdx.x;
    const int id = ids[t];
    const bool ok = id >= 0 && id < n_rows;
    if (!ok && threadIdx.x == 0) {
        atomicOr(status, kMaskInvalidState);  // one thread, one integer OR: idempotent, order independent
    }
    const int nb = k / kTileK;
    float *row = x + (size_t)t * k;
    for (int w = threadIdx.x; w < nb * 8; w += kQ4Threads) {
        const int ib = w >> 3, qp = (w >> 1) & 3, h = w & 1;
        float *dst = row + ib * kTileK + 64 * qp + 16 * h;  // low columns; the high ones are dst + 32
        if (!ok) {
            store_nan16(dst);
            store_nan16(dst + 32);
            continue;
        }
        const uint4 *block = blocks + ((size_t)id * nb + ib) * 9;
        const uint4 head = block[0];
        const uint4 qs = block[1 + 2 * qp + h];
        const float d = half_to_f32(head.x & 0xffffu), dmin = half_to_f32(head.x >> 16);
        int sc_lo, mn_lo, sc_hi, mn_hi;
        scale_min(2 * qp, head, sc_lo, mn_lo);
        scale_min(2 * qp + 1, head, sc_hi, mn_hi);
        const float d_lo = __fmul_rn(d, (float)sc_lo), m_lo = __fmul_rn(dmin, (float)mn_lo);
        const float d_hi = __fmul_rn(d, (float)sc_hi), m_hi = __fmul_rn(dmin, (float)mn_hi);
        for (int l = 0; l < 16; l += 4) {
            float lo[4], hi[4];
            for (int c = 0; c < 4; ++c) {
                const unsigned b = byte_of(qs, l + c);
                lo[c] = __fsub_rn(__fmul_rn(d_lo, (float)(b & 0xFu)), m_lo);
                hi[c] = __fsub_rn(__fmul_rn(d_hi, (float)(b >> 4)), m_hi);
            }
            store4(dst + l, lo[0], lo[1], lo[2], lo[3]);
            store4(dst + 32 + l, hi[0], hi[1], hi[2], hi[3]);
        }
    }
}

bool aligned_to(const void *p, size_t n) { return ((uintptr_t)p % n) == 0; }

/* Bytes of the tiles of n_rows x k for the type, 0 for a type that is not gathered. */
size_t tile_bytes(int type, int n_rows, int k) {
    const size_t tiles = (size_t)(n_rows / 8) * (size_t)(k / kTileK);
    if (type == INFERENCE_CUDA_TYPE_Q4_K) return tiles * kQ4Vecs * 16;
    if (type == INFERENCE_CUDA_TYPE_Q8_0) return tiles * kQ8Vecs * 16;
    return 0;
}

bool apart(const void *a, size_t abytes, const void *b, size_t bbytes) {
    const uintptr_t pa = (uintptr_t)a, pb = (uintptr_t)b;
    return pa + abytes <= pb || pb + bbytes <= pa;
}

/* Every term of this test has a twin in the Rust wrapper. */
bool valid(const inference_cuda_embed_args *a, void *stream) {
    if (a == NULL || stream == NULL || a->tiles == NULL || a->ids == NULL || a->x == NULL || a->status == NULL) return false;
    const bool type_ok = a->ggml_type == INFERENCE_CUDA_TYPE_Q4_K || a->ggml_type == INFERENCE_CUDA_TYPE_Q8_0;
    const bool m_ok = a->m >= 1 && a->m <= kMaxTokens;
    const bool rows_ok = a->n_rows >= 8 && a->n_rows <= kMaxRows && a->n_rows % 8 == 0;
    const bool k_ok = a->k >= kTileK && a->k <= kMaxK && a->k % kTileK == 0;
    if (!(type_ok && m_ok && rows_ok && k_ok)) return false;
    if (!(aligned_to(a->tiles, 16) && aligned_to(a->ids, 4) && aligned_to(a->x, 16) &&
          aligned_to(a->status, 4))) return false;
    const size_t x_bytes = (size_t)a->m * a->k * 4;
    const size_t tiles_bytes = tile_bytes(a->ggml_type, a->n_rows, a->k);
    return apart(a->x, x_bytes, a->ids, (size_t)a->m * 4) && apart(a->x, x_bytes, a->tiles, tiles_bytes) &&
           apart(a->status, 4, a->x, x_bytes) && apart(a->status, 4, a->ids, (size_t)a->m * 4) &&
           apart(a->status, 4, a->tiles, tiles_bytes);
}

/* Bytes of the native Q4_K blocks of n_rows x k. */
size_t native_blocks_bytes(int n_rows, int k) {
    return (size_t)n_rows * (size_t)(k / kTileK) * 144;
}

/* The native twin of valid(): Q4_K only, any row count from 1 up (no 8-row rule). */
bool valid_native(const inference_cuda_embed_native_args *a, void *stream) {
    if (a == NULL || stream == NULL || a->blocks == NULL || a->ids == NULL || a->x == NULL || a->status == NULL) return false;
    const bool m_ok = a->m >= 1 && a->m <= kMaxTokens;
    const bool rows_ok = a->n_rows >= 1 && a->n_rows <= kMaxRows;
    const bool k_ok = a->k >= kTileK && a->k <= kMaxK && a->k % kTileK == 0;
    if (!(m_ok && rows_ok && k_ok)) return false;
    if (!(aligned_to(a->blocks, 16) && aligned_to(a->ids, 4) && aligned_to(a->x, 16) &&
          aligned_to(a->status, 4))) return false;
    const size_t x_bytes = (size_t)a->m * a->k * 4;
    const size_t blocks_bytes = native_blocks_bytes(a->n_rows, a->k);
    return apart(a->x, x_bytes, a->ids, (size_t)a->m * 4) && apart(a->x, x_bytes, a->blocks, blocks_bytes) &&
           apart(a->status, 4, a->x, x_bytes) && apart(a->status, 4, a->ids, (size_t)a->m * 4) &&
           apart(a->status, 4, a->blocks, blocks_bytes);
}

}  // namespace

extern "C" int inference_cuda_embed_launch(const inference_cuda_embed_args *a, void *stream, int *out_stage) {
    try {
        report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
        if (!valid(a, stream)) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        const cudaStream_t s = (cudaStream_t)stream;
        const uint4 *tiles = (const uint4 *)a->tiles;
        if (a->ggml_type == INFERENCE_CUDA_TYPE_Q4_K) {
            embed_q4k_k<<<a->m, kQ4Threads, 0, s>>>(tiles, a->ids, a->x, a->status, a->n_rows, a->k);
        } else {
            embed_q80_k<<<a->m, kQ8Threads, 0, s>>>(tiles, a->ids, a->x, a->status, a->n_rows, a->k);
        }
        Failure failure;
        failure.record(cudaGetLastError(), INFERENCE_CUDA_STAGE_LAUNCH);
        report_stage(out_stage, failure.stage);
        return (int)failure.err;
    } catch (...) {
        return INFERENCE_CUDA_NATIVE_EXCEPTION;
    }
}

extern "C" int inference_cuda_embed_native_launch(const inference_cuda_embed_native_args *a, void *stream,
                                                  int *out_stage) {
    try {
        report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
        if (!valid_native(a, stream)) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        const cudaStream_t s = (cudaStream_t)stream;
        embed_q4k_native_k<<<a->m, kQ4Threads, 0, s>>>((const uint4 *)a->blocks, a->ids, a->x, a->status,
                                                        a->n_rows, a->k);
        Failure failure;
        failure.record(cudaGetLastError(), INFERENCE_CUDA_STAGE_LAUNCH);
        report_stage(out_stage, failure.stage);
        return (int)failure.err;
    } catch (...) {
        return INFERENCE_CUDA_NATIVE_EXCEPTION;
    }
}
