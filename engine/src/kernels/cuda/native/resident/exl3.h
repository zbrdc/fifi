#ifndef INFERENCE_CUDA_RESIDENT_EXL3_H
#define INFERENCE_CUDA_RESIDENT_EXL3_H
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* K,N are nonzero multiples of 128, <=2^22, with K*N<=2^32. Book 0/1/2
 * denotes 3INST/MCG/MUL1. Twice-bitrate is 2,4,...16, or MUL1's 3,5,7.
 * Trellis is input-tile-major packed u16; out is input-major [K,N] F16 bits.
 * Both regions must be sized, two-byte aligned, nonwrapping and disjoint.
 * Explicit non-NULL stream, launch only: no allocation, copy or synchronization.
 * Caller retains device owners through stream completion. Bad arguments return
 * INVALID_ARGUMENT before CUDA; asynchronous faults surface at completion. */
typedef struct inference_cuda_exl3_prep_args {
    const uint16_t *trellis;
    size_t trellis_bytes;
    uint16_t *out;
    size_t out_bytes;
    size_t k;
    size_t n;
    int bits_twice;
    int book;
} inference_cuda_exl3_prep_args;

#ifdef __cplusplus
static_assert(sizeof(inference_cuda_exl3_prep_args) == 56 &&
              offsetof(inference_cuda_exl3_prep_args, out) == 16 &&
              offsetof(inference_cuda_exl3_prep_args, k) == 32 &&
              offsetof(inference_cuda_exl3_prep_args, book) == 52, "EXL3 prep ABI");
#endif
int inference_cuda_exl3_prep(const inference_cuda_exl3_prep_args *args,
                             void *stream, int *out_stage);

/* y = H128((H128(x*suh) @ W'))*svh + bias, for rows 1..65536.
 * Vectors, global intermediates and accumulator storage are F32. XH:[rows,K] is scratch.
 * Legacy projections use F32 operands; midrows uses FP16 operands as described below.
 * Z holds split_k-1 F32[rows,N] banks for rows<=8, otherwise one bank.
 * Out:[rows,N] holds the last partial until publication. Bias is NULL/0 or F32[N].
 * Flag is a caller-initialized sticky u32: bit0 invalidates the ENTIRE output
 * after any nonfinite arithmetic. Check it after completion before publishing.
 * Read/read alias is allowed; every writable working span must be disjoint
 * from every other working span. Other lifetime/launch rules match prep. */
typedef struct inference_cuda_exl3_linear_args {
    const uint16_t *trellis; size_t trellis_bytes;
    const float *suh; size_t suh_bytes;
    const float *svh; size_t svh_bytes;
    const float *bias; size_t bias_bytes;
    const float *x; size_t x_bytes;
    float *xh; size_t xh_bytes;
    float *z; size_t z_bytes;
    float *out; size_t out_bytes;
    uint32_t *flag; size_t flag_bytes;
    size_t rows; size_t k; size_t n;
    int bits_twice; int book;
} inference_cuda_exl3_linear_args;
#ifdef __cplusplus
static_assert(sizeof(inference_cuda_exl3_linear_args) == 176 &&
              offsetof(inference_cuda_exl3_linear_args, flag) == 128 &&
              offsetof(inference_cuda_exl3_linear_args, rows) == 144 &&
              offsetof(inference_cuda_exl3_linear_args, book) == 172, "EXL3 linear ABI");
#endif
/* Legacy entrypoint selects split_k=2; the argument layout is unchanged. */
int inference_cuda_exl3_linear(const inference_cuda_exl3_linear_args *args,
                               void *stream, int *out_stage);
/* split_k must be 2,4,8; rows>8 retain the unsplit projection and one Z bank. */
int inference_cuda_exl3_linear_split(const inference_cuda_exl3_linear_args *args,
                                     uint32_t split_k, void *stream, int *out_stage);
/* Grouped split-2 small-row projection: 1..48 groups and 1..48 total pairs,
 * each group having 1..8 rows and shared k, n and flag. XH/Z/out must be
 * contiguous in group order; each Z slice holds one split-2 F32 bank.
 * Cross-group read/read aliases are allowed; writes must remain disjoint
 * from all reads and other writes. All groups validate before any enqueue;
 * invalid arguments return INVALID_ARGUMENT with STAGE_NONE. Stream and
 * buffer lifetime rules match prep. */
int inference_cuda_exl3_grouped(const inference_cuda_exl3_linear_args *calls,
                                size_t count, void *stream, int *out_stage);

/* Same contract and emissions as inference_cuda_exl3_grouped; the small
 * projection runs the per-warp shared-table gather variant with identical
 * arithmetic and reduction order (FIFI_FLINT_CUDA_GROUPED_GATHER, default
 * off; the default shuffle kernel stays untouched). */
int inference_cuda_exl3_grouped_gathered(const inference_cuda_exl3_linear_args *calls,
                                         size_t count, void *stream, int *out_stage);
/* Explicit tensor-core projection for rows9..144, one F32 Z bank.
 * FP16-rounded operands feed an F32 accumulator. Operand conversion and
 * tensor-core accumulation can round differently from legacy F32 arithmetic.
 * This is not an arbitrary FP32-equivalence guarantee; real-case qualification
 * is required. No default dispatch change. */
int inference_cuda_exl3_linear_midrows(const inference_cuda_exl3_linear_args *args,
                                       void *stream, int *out_stage);
#ifdef __cplusplus
}
#endif
#endif
