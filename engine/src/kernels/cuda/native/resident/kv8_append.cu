/* RD-8k: the KV8 append kernel of the resident decode step (decision 0047 item 6): quantizes the
 * post-norm, post-rope k rows and the v rows of one step to the frozen 8-bit KV format and
 * appends them to the code and scale planes.
 *
 * Design guidance only, no code copied: the layout idea (separate K and V code and scale arrays,
 * cell-major per KV head, one quantizer for decode and prefill) is from Strata's k34-kv8 layout
 * (`laguna_kv8_view.hpp`, research/kv/KV.md) and MegaCapybara C11 (int8 variant only); the code
 * is original CUDA C written from the shapes of this repository. The poison rule mirrors the
 * vendored attn_prep of RD-8a (this repository's own kernel).
 *
 * One thread per value, one warp per group of 32: the warp reduces |x| to the group amax, stores
 * the fp16 scale from lane 0 and the int8 code of its own value from every lane. The code is
 * quantized against the STORED fp16 scale (dequantized in f32), so the decode (RD-8l) and the
 * chunk appends read the same values. The kernel is one launch of grid (n_kv_head, rows) and
 * head_dim threads; all geometry is run-time.
 */
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stddef.h>
#include <stdint.h>

#include "abi.h"
#include "device.h"
#include "resident/kv8_append.h"

namespace {

constexpr int kGroup = 32;        // values per fp16 scale: the frozen group of the format
constexpr int kMaxM = 16;         // decode rows (the step's look-ahead)
constexpr int kMaxChunk = 65535;  // prefill rows (the grid's y extent)
constexpr int kMaxKvHeads = 64;
constexpr int kMinHead = 32;      // one warp quantizes one group
constexpr int kMaxHead = 1024;
constexpr int kMaxCells = (1 << 30) - 1;
// Mask 4 of StepParams.status (invalid step state, REPLY-47); equals
// INFERENCE_CUDA_STATUS_INVALID_STEP_STATE of resident/step_params.h, as attn_prep.cu.
constexpr int kStatusInvalidStepState = 4;

struct Args {
    const float *k;             // [m][n_kv_head][head_dim]
    const float *v;
    signed char *k_codes;       // [n_kv_head][cells][head_dim]
    unsigned short *k_scales;   // [n_kv_head][cells][head_dim / 32]
    signed char *v_codes;
    unsigned short *v_scales;
    const int *pos_dev;         // the decode entry: the device word; NULL for the chunk entry
    int pos0;                   // the chunk entry: the host base position
    int *status;
    int m, n_kv_head, head_dim, cells;
};

__device__ inline float warp_max_a(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}

__device__ inline float h2f(unsigned short bits) { return __half2float(__ushort_as_half(bits)); }

// scale = amax / 127 rounded to fp16 (half even). An all-zero group stores 0.
__device__ inline unsigned short quant_scale(float amax) {
    return __half_as_ushort(__float2half(amax / 127.f));
}

// code = round_half_even(x / s) clamped to [-127, 127], with s the STORED fp16 scale as f32.
// A group whose STORED scale is 0 (an all-zero group, or an amax below what an fp16 scale can
// carry) has codes 0: x / 0 is ±inf or NaN, and a float-to-int conversion of either is
// undefined (the zero-scale fix of the KV8-A review).
__device__ inline int quant_code(float x, float s) {
    if (s == 0.f) {
        return 0;
    }
    return max(-127, min(127, __float2int_rn(x / s)));
}

__global__ void kv8_append_k(Args a) {
    const int kvh = blockIdx.x, t = blockIdx.y, d = threadIdx.x;
    // Poison rule (RD-8a, REPLY-47): the position of token 0 is read from the device (decode) or
    // validated here (chunk) and tested, in 64 bits, against the cells before any other memory
    // access. The answer is block uniform (while no other stream writes the word), so the early
    // return happens before any barrier. A refused launch writes no code, no scale, and ORs
    // mask 4 into status (one thread, integer atomic). A valid launch touches neither.
    const long long base = a.pos_dev ? (long long)*a.pos_dev : (long long)a.pos0;
    if (base < 0 || base + a.m > a.cells) {
        if (a.status && blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0)
            atomicOr(a.status, kStatusInvalidStepState);
        return;
    }
    const int groups = a.head_dim / kGroup;
    const size_t cell = (size_t)(base + t);
    const size_t in_at = ((size_t)t * a.n_kv_head + kvh) * a.head_dim + d;
    const size_t code_at = (size_t)kvh * a.cells * a.head_dim + cell * a.head_dim + d;
    const size_t scale_at = (size_t)kvh * a.cells * groups + cell * groups + d / kGroup;
    const float kx = a.k[in_at], vx = a.v[in_at];
    const unsigned short ks = quant_scale(warp_max_a(fabsf(kx)));
    const unsigned short vs = quant_scale(warp_max_a(fabsf(vx)));
    a.k_codes[code_at] = (signed char)quant_code(kx, h2f(ks));
    a.v_codes[code_at] = (signed char)quant_code(vx, h2f(vs));
    if ((d & (kGroup - 1)) == 0) {
        a.k_scales[scale_at] = ks;
        a.v_scales[scale_at] = vs;
    }
}

bool aligned_to(const void *p, uintptr_t n) { return ((uintptr_t)p & (n - 1)) == 0; }

// The six data pointers every entry takes: f32 rows in, one code array and one scale array out,
// per K and V, each 4-byte aligned.
bool valid_data(const void *k, const void *v, const void *k_codes, const void *k_scales,
                const void *v_codes, const void *v_scales) {
    return k != NULL && v != NULL && k_codes != NULL && k_scales != NULL && v_codes != NULL &&
           v_scales != NULL && aligned_to(k, 4) && aligned_to(v, 4) && aligned_to(k_codes, 4) &&
           aligned_to(k_scales, 4) && aligned_to(v_codes, 4) && aligned_to(v_scales, 4);
}

// rows tokens, n_kv_head kv heads, head_dim values per head (a multiple of the group), cells
// plane cells per head with room for the rows.
bool valid_geometry(int rows, int n_kv_head, int head_dim, int cells) {
    return rows >= 1 && n_kv_head >= 1 && n_kv_head <= kMaxKvHeads && head_dim >= kMinHead &&
           head_dim <= kMaxHead && head_dim % kGroup == 0 && cells >= rows && cells <= kMaxCells;
}

Args make_args(const void *k, const void *v, void *k_codes, void *k_scales, void *v_codes,
               void *v_scales, int rows, int n_kv_head, int head_dim, int cells) {
    Args a = {};
    a.k = (const float *)k;
    a.v = (const float *)v;
    a.k_codes = (signed char *)k_codes;
    a.k_scales = (unsigned short *)k_scales;
    a.v_codes = (signed char *)v_codes;
    a.v_scales = (unsigned short *)v_scales;
    a.m = rows;
    a.n_kv_head = n_kv_head;
    a.head_dim = head_dim;
    a.cells = cells;
    return a;
}

Failure launch(const Args &a, cudaStream_t s) {
    Failure failure;
    kv8_append_k<<<dim3(a.n_kv_head, a.m), a.head_dim, 0, s>>>(a);
    failure.record(cudaGetLastError(), INFERENCE_CUDA_STAGE_LAUNCH);
    return failure;
}

}  // namespace

extern "C" int inference_cuda_kv8_append_launch(const inference_cuda_kv8_append_args *in,
                                                void *stream, int *out_stage) {
    try {
        report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
        const bool pointers = in != NULL && valid_data(in->k, in->v, in->k_codes, in->k_scales,
                                                       in->v_codes, in->v_scales) &&
                              in->pos != NULL && in->status != NULL && aligned_to(in->pos, 4) &&
                              aligned_to(in->status, 4);
        if (!pointers || !valid_geometry(in->m, in->n_kv_head, in->head_dim, in->cells) ||
            in->m > kMaxM) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        Args a = make_args(in->k, in->v, in->k_codes, in->k_scales, in->v_codes, in->v_scales,
                           in->m, in->n_kv_head, in->head_dim, in->cells);
        a.pos_dev = (const int *)in->pos;
        a.status = (int *)in->status;
        const Failure failure = launch(a, (cudaStream_t)stream);
        report_stage(out_stage, failure.stage);
        return (int)failure.err;
    } catch (...) {
        return INFERENCE_CUDA_NATIVE_EXCEPTION;
    }
}

extern "C" int inference_cuda_kv8_chunk_launch(const inference_cuda_kv8_chunk_args *in,
                                               void *stream, int *out_stage) {
    try {
        report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
        const bool pointers =
            in != NULL && valid_data(in->k, in->v, in->k_codes, in->k_scales, in->v_codes,
                                     in->v_scales);
        const bool position = in == NULL || (in->pos0 >= 0 && (long long)in->pos0 + in->t <= in->cells);
        if (!pointers || !position ||
            !valid_geometry(in->t, in->n_kv_head, in->head_dim, in->cells) ||
            in->t > kMaxChunk) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        Args a = make_args(in->k, in->v, in->k_codes, in->k_scales, in->v_codes, in->v_scales,
                           in->t, in->n_kv_head, in->head_dim, in->cells);
        a.pos0 = in->pos0;
        const Failure failure = launch(a, (cudaStream_t)stream);
        report_stage(out_stage, failure.stage);
        return (int)failure.err;
    } catch (...) {
        return INFERENCE_CUDA_NATIVE_EXCEPTION;
    }
}
