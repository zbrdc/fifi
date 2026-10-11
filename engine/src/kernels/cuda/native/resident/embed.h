/*
 * RD-4c: launch entry of the embedding row gather (native/resident/embed.h, the per-kernel header of
 * the layout in kernels/cuda/resident/mod.rs; abi_resident.h stays closed). Launch-only: nothing here
 * allocates, copies or synchronizes; a fault in queued work surfaces at the next synchronize.
 * Include after abi.h in C++. Rust mirrors the struct in native/resident/embed.rs.
 *
 * x[t][0 .. k-1] = row ids[t] of the tiled token embedding, dequantized to f32, for t in [0, m).
 *   tiles      the tiled weight of n_rows x k (the layout of repack.cu / mma884.cuh: 8-row x 256-k tiles,
 *              groups of 8 rows in order), ggml_type INFERENCE_CUDA_TYPE_Q4_K (12) or _Q8_0 (8),
 *              16-byte aligned. The kernel reads only the tiles of the rows it is asked for.
 *   ids        a device array of m ints, 4-byte aligned, read when the kernel runs (the token_id word
 *              of StepParams, the first word of the argmax result pair, or a list of prompt tokens).
 *              One id outside [0, n_rows) writes a row of NaN, reads no weight (never out of bounds) and
 *              sets status bit 4. A prompt id known on the host is refused on the host (BadRequest) before it
 *              is uploaded; this device path is for ids the host cannot see.
 *   x          f32 [m][k], 16-byte aligned, written in full. It may not overlap ids, status or the tiles.
 *   status     one device i32, 4-byte aligned, the step status word (StepParams.status, offset 16;
 *              ResidentState::step_status()). When any id of the call is outside [0, n_rows) the kernel
 *              ORs INFERENCE_CUDA_STATUS_INVALID_STEP_STATE (4) into it with one integer atomicOr from
 *              one thread of the block that saw the id (idempotent, order independent; the RD-8a0
 *              pattern). It writes nothing else and never clears a bit, so the host cannot tell this cause from
 *              an invalid step state by the word alone (the host message names both). Valid ids leave it alone.
 *   n_rows     multiple of 8, 8 .. 2^28;  k  multiple of 256, 256 .. 32768;  m in [1, 4096].
 * stream must not be NULL. A rejected argument returns INFERENCE_CUDA_INVALID_ARGUMENT before any CUDA
 * call; otherwise the launch status of the kernel (one launch).
 * Numerics: bit equal to the host dequantizer (gguf_dequant.rs): Q4_K  d1 * q - min1 with d1 = d * sc,
 * min1 = dmin * m;  Q8_0  d * q. Each operation is rounded alone (no fused multiply-add), in debug and
 * in release device code alike.
 */
#ifndef INFERENCE_CUDA_RESIDENT_EMBED_H
#define INFERENCE_CUDA_RESIDENT_EMBED_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct inference_cuda_embed_args {
    const void *tiles;
    const int *ids;
    float *x;
    int *status;
    int ggml_type;
    int m;
    int n_rows;
    int k;
} inference_cuda_embed_args;

#ifdef __cplusplus
static_assert(sizeof(inference_cuda_embed_args) == 48 && offsetof(inference_cuda_embed_args, tiles) == 0 &&
                  offsetof(inference_cuda_embed_args, ids) == 8 && offsetof(inference_cuda_embed_args, x) == 16 &&
                  offsetof(inference_cuda_embed_args, status) == 24 &&
                  offsetof(inference_cuda_embed_args, ggml_type) == 32 && offsetof(inference_cuda_embed_args, m) == 36 &&
                  offsetof(inference_cuda_embed_args, n_rows) == 40 && offsetof(inference_cuda_embed_args, k) == 44,
              "inference_cuda_embed_args layout");
#else
_Static_assert(sizeof(inference_cuda_embed_args) == 48 && offsetof(inference_cuda_embed_args, tiles) == 0 &&
                   offsetof(inference_cuda_embed_args, ids) == 8 && offsetof(inference_cuda_embed_args, x) == 16 &&
                   offsetof(inference_cuda_embed_args, status) == 24 &&
                  offsetof(inference_cuda_embed_args, ggml_type) == 32 && offsetof(inference_cuda_embed_args, m) == 36 &&
                   offsetof(inference_cuda_embed_args, n_rows) == 40 && offsetof(inference_cuda_embed_args, k) == 44,
               "inference_cuda_embed_args layout");
#endif

int inference_cuda_embed_launch(const inference_cuda_embed_args *args, void *stream, int *out_stage);

#ifdef __cplusplus
}
#endif

#endif /* INFERENCE_CUDA_RESIDENT_EMBED_H */
