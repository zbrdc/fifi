/*
 * The mmid entries of the MMQ glue. Each entry is its own ABI in this one translation unit, with
 * its own argument struct; abi_resident.h stays closed.
 *
 * inference_cuda_mmq_ids_guard: copies the router ids [m][top_k] (contiguous, read only: the
 * router's tensor is never written) into ids_g [m][top_k]. A row is FLAGGED when an id lies
 * outside [0, n_experts) or two of its ids are equal; its copy is {0, 1, ..., top_k - 1}.
 * A single dummy value would itself repeat an id, and vendored mm_ids_helper turns a repeated id
 * into unwritten ids_dst and inverse entries that later serve as store indices (mmid.cu:80,
 * :83, :110, :113; mmq.cuh:470; quantize.cu:529-538). An unflagged row is copied unchanged.
 * Status: a flagged row whose every slot is -1 (the router's fault form, router.rs:24) ORs
 * INFERENCE_CUDA_MMQ_IDS_ROUTER_FAULT (mask 1: route_experts); any other flagged row ORs
 * INFERENCE_CUDA_MMQ_IDS_BAD_ROW (mask 8: prefill_chunk). Mask 8 outranks mask 1
 * (book.rs:96-101), so a chunk with both reads 9. status is NULL or the prefill status word,
 * never StepParams.status; a launch with no flagged row leaves it untouched.
 *
 * inference_cuda_mmq_build_ids: the guard above, ALWAYS enqueued first (no argument skips it),
 * then the vendored mm_ids_helper (mmid.cu:153-176) on ids_g, with si1 = top_k (contiguous rows)
 * and write_inverse. ids_dst [m * top_k] receives compact column -> token slot t * top_k + r
 * (mmid.cu:110), sorted by expert and by token within an expert; ids_src1 [m * top_k] the
 * inverse map, token slot -> compact column (mmid.cu:113); expert_bounds [n_experts + 1] the
 * first compact column of each expert and m * top_k at the end (mmid.cu:123, :129). Every
 * guarded row holds top_k distinct ids in range, so ids_src1 is a permutation of
 * [0, m * top_k) and ids_dst its inverse. Only the first m * top_k ints of ids_dst are written.
 * The helper takes 4 * m bytes of shared memory, which must fit the device's opt-in limit
 * (mmid.cu:146-147): the entry queries the device facts (mmq_facts.h) and refuses a larger m
 * before any launch.
 *
 * inference_cuda_mmq_mmid_launch: one grouped GEMM over a weight stack through the vendored
 * mul_mat_q_case (mmq.cuh:1500-1510) with the ids of an MmidIds. The geometry fields are the
 * vendored mmq_args (mmq.cuh:1326-1340) as the Rust fill_mmid_args fills them (the pin's
 * mmq.cu:255-261), and the entry refuses anything but this grouped form:
 *   type_x                     Q4_K (12) or Q6_K (14); a grouped Q8_0 or Q5_0 has no arm here
 *   ncols_x = K                a multiple of 256, 256..32768 (the K grid: the k loop
 *                              reads whole ITER_K = 256 rows with no k clamp, mmq.cuh:856)
 *   stride_row_x               K / 256, the blocks of one weight row
 *   nrows_x = N                the rows of one expert matrix, >= 1
 *   ncols_max = ncols_opt = m  the token count, within the shared shape rules
 *   ncols_dst = ncols_y        m * top_k, the compact columns
 *   nchannels_x = nchannels_y  n_experts: the expert is the channel index of the grid
 *   stride_channel_x           the per-expert stride in weight blocks, >= N * K / 256
 *   nrows_dst = ld             the dst row stride in floats, >= N
 *   nsamples_x = nsamples_y = 1; stride_channel_y, stride_channel_dst and the sample strides 0
 * The kernels compute indexes in int: ceil(m / 8) * ceil(N / 128) * n_experts * K / 256 < 2^30
 * (the stream-k limit of mmq.cuh:1402, bounded with the smallest tile, J = 8) and, in the same
 * rule, ceil(m / 8) * ceil(N / 128) * n_experts <= 21,474,836 (mmq.cuh:1399 computes
 * 100 * ntiles_dst in int for the tiling decision, and the J = 8 tile count is the largest the
 * launcher can pick), the weight index
 * (n_experts - 1) * stride_channel_x + N * K / 256 <= INT_MAX blocks and the dst index
 * (m * top_k - 1) * ld + N <= INT_MAX floats.
 * Capacities in byte: x >= that weight index in blocks of the type; y >= K / 128 * m * top_k *
 * 144 + 18,432 (the constant pad); ids_dst >= m * top_k + m + 127 ints (the stream-k fixup
 * gather of mmq.cuh:1295-1302); expert_bounds >= n_experts + 1 ints; dst >= the dst index in
 * floats; fixup >= nsm * 128 * 128 floats, nsm from the device facts. x, y and fixup are
 * 16-byte aligned, the rest 4. dst shares no byte with another range, and fixup none with x,
 * y, ids_dst or expert_bounds.
 * One more int index of the vendored kernels, bounded only by the shape caps: the y-tile copies
 * compute ncols_y * (kb0 * qk / 128) * 36 in int (mmq.cuh:859, :875), at most
 * 65,536 * 256 * 36 = 603,979,776 under m <= 2048, top_k <= 32 and K <= 32768. Raising a cap
 * reopens it, so it is recorded here rather than left implicit in the caps.
 * Output: dst[ids_dst[j] * ld + i] for compact column j and row i < N (mmq.cuh:470), so row
 * t * top_k + r holds pick r of token t. The ids must carry the values of an MmidIds (ids_dst a
 * permutation of [0, m * top_k), monotone bounds ending at m * top_k, at most m columns per
 * expert); this entry checks their capacities, the Rust type proves the values.
 *
 * Shape rules every entry shares: 1 <= n_experts <= 65535 (the z extent of the MMQ tiling grid),
 * 1 <= m <= 2048 (the chunk cap), 1 <= top_k <= min(n_experts, 32). Pointers are
 * 4-byte aligned, and every written range shares no byte with any other range of the call. A
 * violated rule is INFERENCE_CUDA_INVALID_ARGUMENT before any CUDA call. The launches are
 * asynchronous on stream and never synchronize.
 */
#ifndef INFERENCE_CUDA_PREFILL_MMQ_MMID_H
#define INFERENCE_CUDA_PREFILL_MMQ_MMID_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define INFERENCE_CUDA_MMQ_MMID_MAX_EXPERTS 65535
#define INFERENCE_CUDA_MMQ_MMID_MAX_TOKENS 2048
#define INFERENCE_CUDA_MMQ_MMID_MAX_TOP_K 32
/* Mask 1 of the prefill status word: a router fault row (every slot -1), route_experts. */
#define INFERENCE_CUDA_MMQ_IDS_ROUTER_FAULT 0x1
/* Mask 8 of the prefill status word: any other flagged row, prefill_chunk. */
#define INFERENCE_CUDA_MMQ_IDS_BAD_ROW 0x8

typedef struct inference_cuda_mmq_ids_guard_args {
    const int *ids; /* router ids [m][top_k], read only */
    int *ids_g;     /* the guarded copy [m][top_k] */
    int *status;    /* NULL or the prefill status word */
    int n_experts;
    int m;
    int top_k;
} inference_cuda_mmq_ids_guard_args;

int inference_cuda_mmq_ids_guard(const inference_cuda_mmq_ids_guard_args *args, void *stream,
                                 int *out_stage);

typedef struct inference_cuda_mmq_build_ids_args {
    const int *ids;     /* router ids [m][top_k], read only */
    int *ids_g;         /* the guarded copy [m][top_k] */
    int *ids_src1;      /* [m * top_k]: token slot -> compact column */
    int *ids_dst;       /* [m * top_k] written: compact column -> token slot */
    int *expert_bounds; /* [n_experts + 1] */
    int *status;        /* NULL or the prefill status word */
    int n_experts;
    int m;
    int top_k;
} inference_cuda_mmq_build_ids_args;

int inference_cuda_mmq_build_ids(const inference_cuda_mmq_build_ids_args *args, void *stream,
                                 int *out_stage);

typedef struct inference_cuda_mmq_mmid_args {
    const void *x;            /* the weight stack base, expert 0 */
    const void *y;            /* q8_1 compact columns, block_q8_1_mmq k-block-major */
    const int *ids_dst;       /* compact column -> token slot */
    const int *expert_bounds; /* [n_experts + 1] */
    float *dst;               /* rows of ld floats, one per token slot */
    float *fixup;             /* stream-k fixup scratch */
    size_t x_bytes;           /* the capacity of each range above */
    size_t y_bytes;
    size_t ids_dst_bytes;
    size_t expert_bounds_bytes;
    size_t dst_bytes;
    size_t fixup_bytes;
    int64_t type_x; /* the mmq_args geometry from here on, in its field order */
    int64_t ncols_x;
    int64_t nrows_x;
    int64_t ncols_dst;
    int64_t stride_row_x;
    int64_t ncols_y;
    int64_t nrows_dst;
    int64_t nchannels_x;
    int64_t nchannels_y;
    int64_t stride_channel_x;
    int64_t stride_channel_y;
    int64_t stride_channel_dst;
    int64_t nsamples_x;
    int64_t nsamples_y;
    int64_t stride_sample_x;
    int64_t stride_sample_y;
    int64_t stride_sample_dst;
    int64_t ncols_max;
    int64_t ncols_opt;
} inference_cuda_mmq_mmid_args;

int inference_cuda_mmq_mmid_launch(const inference_cuda_mmq_mmid_args *args, void *stream,
                                   int *out_stage);

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
static_assert(sizeof(inference_cuda_mmq_ids_guard_args) == 40 &&
                  offsetof(inference_cuda_mmq_ids_guard_args, status) == 16 &&
                  offsetof(inference_cuda_mmq_ids_guard_args, n_experts) == 24 &&
                  offsetof(inference_cuda_mmq_ids_guard_args, top_k) == 32,
              "inference_cuda_mmq_ids_guard_args layout");
static_assert(sizeof(inference_cuda_mmq_build_ids_args) == 64 &&
                  offsetof(inference_cuda_mmq_build_ids_args, expert_bounds) == 32 &&
                  offsetof(inference_cuda_mmq_build_ids_args, status) == 40 &&
                  offsetof(inference_cuda_mmq_build_ids_args, n_experts) == 48 &&
                  offsetof(inference_cuda_mmq_build_ids_args, top_k) == 56,
              "inference_cuda_mmq_build_ids_args layout");
static_assert(sizeof(inference_cuda_mmq_mmid_args) == 248 &&
                  offsetof(inference_cuda_mmq_mmid_args, fixup) == 40 &&
                  offsetof(inference_cuda_mmq_mmid_args, x_bytes) == 48 &&
                  offsetof(inference_cuda_mmq_mmid_args, fixup_bytes) == 88 &&
                  offsetof(inference_cuda_mmq_mmid_args, type_x) == 96 &&
                  offsetof(inference_cuda_mmq_mmid_args, nchannels_x) == 152 &&
                  offsetof(inference_cuda_mmq_mmid_args, ncols_max) == 232,
              "inference_cuda_mmq_mmid_args layout");
#else
_Static_assert(sizeof(inference_cuda_mmq_ids_guard_args) == 40 &&
                   offsetof(inference_cuda_mmq_ids_guard_args, status) == 16 &&
                   offsetof(inference_cuda_mmq_ids_guard_args, n_experts) == 24 &&
                   offsetof(inference_cuda_mmq_ids_guard_args, top_k) == 32,
               "inference_cuda_mmq_ids_guard_args layout");
_Static_assert(sizeof(inference_cuda_mmq_build_ids_args) == 64 &&
                   offsetof(inference_cuda_mmq_build_ids_args, expert_bounds) == 32 &&
                   offsetof(inference_cuda_mmq_build_ids_args, status) == 40 &&
                   offsetof(inference_cuda_mmq_build_ids_args, n_experts) == 48 &&
                   offsetof(inference_cuda_mmq_build_ids_args, top_k) == 56,
               "inference_cuda_mmq_build_ids_args layout");
_Static_assert(sizeof(inference_cuda_mmq_mmid_args) == 248 &&
                   offsetof(inference_cuda_mmq_mmid_args, fixup) == 40 &&
                   offsetof(inference_cuda_mmq_mmid_args, x_bytes) == 48 &&
                   offsetof(inference_cuda_mmq_mmid_args, fixup_bytes) == 88 &&
                   offsetof(inference_cuda_mmq_mmid_args, type_x) == 96 &&
                   offsetof(inference_cuda_mmq_mmid_args, nchannels_x) == 152 &&
                   offsetof(inference_cuda_mmq_mmid_args, ncols_max) == 232,
               "inference_cuda_mmq_mmid_args layout");
#endif

#endif /* INFERENCE_CUDA_PREFILL_MMQ_MMID_H */
