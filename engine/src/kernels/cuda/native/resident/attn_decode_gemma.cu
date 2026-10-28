/* Entry of attn_decode_gemma on an explicit stream. Glue only. */
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <stddef.h>
#include <stdint.h>
#include <cmath>

#include "abi.h"
#include "device.h"
#include "resident/attn_decode_gemma.h"

namespace {

bool aligned(const void *p, size_t a) { return (uintptr_t)p % a == 0; }
bool apart(const void *a, size_t ab, const void *b, size_t bb) {
    const uintptr_t x = (uintptr_t)a, y = (uintptr_t)b;
    return x <= y ? ab <= y - x : bb <= x - y;
}

__global__ void attn_decode_gemma_kernel(
    float *out, unsigned int *status, const float *q,
    const signed char *k_codes, const unsigned short *k_scales,
    const signed char *v_codes, const unsigned short *v_scales, const int *kv_len_word,
    int n_q_head, int n_kv_head, int head_dim, int cells, int window, float scale) {
    const int token = blockIdx.y;
    const int qh = blockIdx.x;
    const int tid = threadIdx.x;
    const int lane = tid % 32;
    const int warp = tid / 32;
    const int nwarps = (int)blockDim.x / 32;
    const int kvh = (int)(((long long)qh * n_kv_head) / n_q_head);
    const int64_t requested_len = (int64_t)*kv_len_word + token;
    const int out_of_range = (requested_len < 1 || requested_len > cells);
    const size_t out_base = ((size_t)token * n_q_head + qh) * (size_t)head_dim;
    if (out_of_range) {
        for (int d = tid; d < head_dim; d += blockDim.x)
            out[out_base + d] = __int_as_float(0x7fc00000);
        if (tid == 0 && status != NULL) atomicOr(status, 4u);
        return;
    }
    const int kv_len = (int)requested_len;
    const int visible = (window > 0 && window < kv_len) ? window : kv_len;
    const int first = kv_len - visible;
    const int groups = head_dim / 32;
    const int per_lane = head_dim / 32;
    float qv[16];
    for (int k = 0; k < per_lane; ++k) qv[k] = q[out_base + lane + 32 * k];
    float acc[16];
    for (int k = 0; k < per_lane; ++k) acc[k] = 0.0f;
    float run_m = -INFINITY;
    float run_l = 0.0f;
    for (int cell = first + warp; cell < kv_len; cell += nwarps) {
        const size_t plane = ((size_t)kvh * cells + cell) * (size_t)head_dim;
        const size_t sbase = ((size_t)kvh * cells + cell) * (size_t)groups;
        float partial = 0.0f;
        for (int k = 0; k < per_lane; ++k) {
            const float kscale = __half2float(__ushort_as_half(k_scales[sbase + k]));
            partial += qv[k] * ((float)k_codes[plane + lane + 32 * k] * kscale);
        }
        float score = partial;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) score += __shfl_xor_sync(0xffffffffu, score, o);
        score *= scale;
        const float new_m = fmaxf(run_m, score);
        const float p = expf(score - new_m);
        const float correction = (run_l == 0.0f) ? 0.0f : expf(run_m - new_m);
        run_l = run_l * correction + p;
        for (int k = 0; k < per_lane; ++k) {
            const float vscale = __half2float(__ushort_as_half(v_scales[sbase + k]));
            acc[k] = acc[k] * correction + p * ((float)v_codes[plane + lane + 32 * k] * vscale);
        }
        run_m = new_m;
    }
    // The combine keeps one staged value per lane per step: a [warps][32][16] array
    // would be 64 KB and exceed the static shared limit.
    __shared__ float sh_m[32];
    __shared__ float sh_l[32];
    __shared__ float sh_a[32][32];
    if (lane == 0) {
        sh_m[warp] = run_m;
        sh_l[warp] = run_l;
    }
    __syncthreads();
    float peak = 0.0f;
    float denom = 0.0f;
    if (warp == 0) {
        peak = -INFINITY;
        for (int w = 0; w < nwarps; ++w) peak = fmaxf(peak, sh_m[w]);
        for (int w = 0; w < nwarps; ++w) {
            const float wl = (sh_l[w] == 0.0f) ? 0.0f : expf(sh_m[w] - peak);
            denom += sh_l[w] * wl;
        }
    }
    for (int k = 0; k < per_lane; ++k) {
        sh_a[warp][lane] = acc[k];
        __syncthreads();
        if (warp == 0) {
            float combined = 0.0f;
            for (int w = 0; w < nwarps; ++w) {
                const float wl = (sh_l[w] == 0.0f) ? 0.0f : expf(sh_m[w] - peak);
                combined += sh_a[w][lane] * wl;
            }
            const int d = lane + 32 * k;
            if (d < head_dim) out[out_base + d] = combined / denom;
        }
        __syncthreads();
    }
}

}  // namespace

extern "C" int inference_cuda_attn_decode_gemma_launch(
    const inference_cuda_attn_decode_gemma_args *args, void *stream, int *out_stage) {
    try {
        report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
        if (args == NULL || stream == NULL) return INFERENCE_CUDA_INVALID_ARGUMENT;
        if (args->out == NULL || args->q == NULL || args->k_codes == NULL ||
            args->k_scales == NULL || args->v_codes == NULL || args->v_scales == NULL ||
            args->kv_len == NULL)
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        if (args->m < 1 || args->m > 16) return INFERENCE_CUDA_INVALID_ARGUMENT;
        if (args->n_q_head != 16) return INFERENCE_CUDA_INVALID_ARGUMENT;
        if (args->n_kv_head != 1 && args->n_kv_head != 8)
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        if (args->head_dim != 256 && args->head_dim != 512)
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        if (args->cells < 1 || args->window < 0 || args->window > args->cells)
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        if (!(args->scale > -1e30f && args->scale < 1e30f))
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        if (!std::isfinite(args->scale)) return INFERENCE_CUDA_INVALID_ARGUMENT;
        if (!aligned(args->out, 4) || !aligned(args->q, 4) || !aligned(args->kv_len, 4) ||
            !aligned(args->k_scales, 2) || !aligned(args->v_scales, 2) ||
            (args->status != NULL && !aligned(args->status, 4)))
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        const size_t plane = (size_t)args->n_kv_head * args->cells;
        const size_t codes_bytes = plane * args->head_dim;
        const size_t scales_bytes = plane * (args->head_dim / 32) * 2;
        const size_t q_bytes = (size_t)args->m * args->n_q_head * args->head_dim * 4;
        const size_t out_bytes = q_bytes;
        const void *reads[] = {args->q, args->k_codes, args->k_scales,
                               args->v_codes, args->v_scales, args->kv_len};
        const size_t sizes[] = {q_bytes, codes_bytes, scales_bytes,
                                codes_bytes, scales_bytes, sizeof(int)};
        for (int i = 0; i < 6; ++i) {
            if (!apart(args->out, out_bytes, reads[i], sizes[i]) ||
                (args->status != NULL && !apart(args->status, 4, reads[i], sizes[i])))
                return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        if (args->status != NULL && !apart(args->out, out_bytes, args->status, 4))
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        const dim3 grid((unsigned)args->n_q_head, (unsigned)args->m);
        Failure failure;
        attn_decode_gemma_kernel<<<grid, 256, 0, (cudaStream_t)stream>>>(
            (float *)args->out, (unsigned int *)args->status, (const float *)args->q,
            (const signed char *)args->k_codes, (const unsigned short *)args->k_scales,
            (const signed char *)args->v_codes, (const unsigned short *)args->v_scales,
            (const int *)args->kv_len, args->n_q_head, args->n_kv_head,
            args->head_dim, args->cells, args->window, args->scale);
        failure.record(cudaGetLastError(), INFERENCE_CUDA_STAGE_LAUNCH);
        report_stage(out_stage, failure.stage);
        return (int)failure.err;
    } catch (...) {
        return INFERENCE_CUDA_NATIVE_EXCEPTION;
    }
}
