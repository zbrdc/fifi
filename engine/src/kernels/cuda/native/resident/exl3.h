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
    /* Producer-fusion arm: 0 stores f32 as before; 1 stores the bf16 roundtrip
     * (the same value the separate publish kernel writes) in the final
     * output_rotate, so the caller can skip that publish kernel. */
    uint32_t fused_publish;
} inference_cuda_exl3_linear_args;
#ifdef __cplusplus
static_assert(sizeof(inference_cuda_exl3_linear_args) == 184 &&
              offsetof(inference_cuda_exl3_linear_args, flag) == 128 &&
              offsetof(inference_cuda_exl3_linear_args, rows) == 144 &&
              offsetof(inference_cuda_exl3_linear_args, book) == 172 &&
              offsetof(inference_cuda_exl3_linear_args, fused_publish) == 176,
              "EXL3 linear ABI");
#endif
/* Legacy entrypoint selects split_k=2; the argument layout is unchanged. */
int inference_cuda_exl3_linear(const inference_cuda_exl3_linear_args *args,
                               void *stream, int *out_stage);
/* split_k must be 2,4,8; rows>8 retain the unsplit projection and one Z bank. */
int inference_cuda_exl3_linear_split(const inference_cuda_exl3_linear_args *args,
                                     uint32_t split_k, void *stream, int *out_stage);
/* Grouped split-2 small-row projection: 1..48 groups and 1..48 total pairs,
 * each group having 1..8 rows and shared k, n and flag. XH and Z must be
 * contiguous in group order; each Z slice holds one split-2 F32 bank. The
 * out extents may be scattered (the banked decode writes disjoint rows of
 * one activation buffer per routed expert) but must be mutually disjoint.
 * Cross-group read/read aliases are allowed; writes must remain disjoint
 * from all reads and other writes. All groups validate before any enqueue;
 * invalid arguments return INVALID_ARGUMENT with STAGE_NONE. Stream and
 * buffer lifetime rules match prep. */
int inference_cuda_exl3_grouped(const inference_cuda_exl3_linear_args *calls,
                                size_t count, void *stream, int *out_stage);

/* The grouped launch with the k split made explicit: 4 or 8 only (2 is refused;
 * stay on inference_cuda_exl3_grouped). Same views contract except each call's z
 * spans (split_k - 1) [rows, n] partial banks and the z strides between calls
 * follow that width. One-row calls, the plain one-tile path only: rows > 1,
 * the gather and fused variants refuse. The extra k partials fold left-to-right
 * in the output rotation - a different but fixed order than the split-2 fold,
 * so values differ from split 2 in the last ulp (not byte-identical; that is
 * the point: more blocks in flight for the fill-limited one-row decode). */
int inference_cuda_exl3_grouped_split(const inference_cuda_exl3_linear_args *calls,
                                      size_t count, uint32_t split_k, void *stream,
                                      int *out_stage);

/* Same contract and emissions as inference_cuda_exl3_grouped; the small
 * projection runs the per-warp shared-table gather variant with identical
 * arithmetic and reduction order (FIFI_FLINT_CUDA_GROUPED_GATHER, default
 * ON since 2026-10-28; any non-enable value rolls back to the shuffle kernel,
 * which stays untouched). */
int inference_cuda_exl3_grouped_gathered(const inference_cuda_exl3_linear_args *calls,
                                         size_t count, void *stream, int *out_stage);
/* Same contract as inference_cuda_exl3_grouped_gathered; each block covers `cols` (1, 2, 4 or 8)
 * adjacent 16-column tiles of n, so n must be a multiple of 16 * cols. Accumulation order, decode
 * and reduction match the gather entry, so every written byte is identical. Any other `cols`
 * or an n that is not a multiple is INVALID_ARGUMENT. */
int inference_cuda_exl3_grouped_wide(const inference_cuda_exl3_linear_args *calls,
                                     size_t count, void *stream, int *out_stage,
                                     unsigned cols);
/* inference_cuda_exl3_grouped_wide with cols 1 that runs a compile-time rate 10 or rate 6 body when every
 * group is book 2 at one of those rates, with the DP4A decode of book 2 when `dp4a` is nonzero.
 * Every written byte is identical to the cols 1 entry. */
int inference_cuda_exl3_grouped_wide_odd(const inference_cuda_exl3_linear_args *calls,
                                         size_t count, void *stream, int *out_stage,
                                         unsigned dp4a);
/* Id-indexed grouped projection (the routed experts of one decode token). calls[i] describes one
 * row of member 0 of a bank (trellis, suh, svh at member 0); index[i] names the member on the
 * device: ids[slot] is read when the kernels run, so the ids may be written by an earlier kernel of
 * the same stream and the launch can be captured in a graph. Member m lives at
 * trellis + m * trellis_stride bytes and suh/svh + m * scale_stride floats. An id outside
 * [0, members) skips its pair and sets bit0 of the flag. 1..16 calls of one row each, all of
 * one k, n and flag, spans contiguous in call order as for the grouped entries; `cols` as for
 * inference_cuda_exl3_grouped_wide. Bad arguments are INVALID_ARGUMENT before any launch. */
typedef struct inference_cuda_exl3_index_args {
    const int32_t *ids; size_t ids_bytes;
    uint32_t slot; uint32_t members;
    size_t trellis_stride;
    size_t scale_stride;
    /* Optional member table: when non-null it holds `members` u32 byte offsets (multiples of 4)
     * of the members from member 0, and replaces the strides (which are then ignored): member m
     * lives at trellis + offsets[m] bytes and suh/svh + offsets[m] / 4 floats. An entry of
     * 0xFFFFFFFF marks a member that is not resident: its pair is skipped and bit0 of the flag is
     * set, as for an id outside [0, members). */
    const uint32_t *offsets;
} inference_cuda_exl3_index_args;
#ifdef __cplusplus
static_assert(sizeof(inference_cuda_exl3_index_args) == 48 &&
              offsetof(inference_cuda_exl3_index_args, slot) == 16 &&
              offsetof(inference_cuda_exl3_index_args, trellis_stride) == 24 &&
              offsetof(inference_cuda_exl3_index_args, scale_stride) == 32 &&
              offsetof(inference_cuda_exl3_index_args, offsets) == 40,
              "EXL3 index ABI");
#endif
int inference_cuda_exl3_grouped_indexed(const inference_cuda_exl3_linear_args *calls,
                                        const inference_cuda_exl3_index_args *index,
                                        size_t count, void *stream, int *out_stage,
                                        unsigned cols);
/* The id-indexed launch with opt-in options, one tile per block. `options` bit 0 decodes book 2
 * with one packed byte dot product (DP4A), bit 1 takes up to 20 calls in one launch (else up to
 * 16), bit 2 pipelines the tile loop (the next tile's loads issued before the current one's
 * barriers); options 0 is refused (that is inference_cuda_exl3_grouped_indexed). Every byte
 * written equals the default launch of the same calls. */
int inference_cuda_exl3_grouped_indexed_option(const inference_cuda_exl3_linear_args *calls,
                                               const inference_cuda_exl3_index_args *index,
                                               size_t count, void *stream, int *out_stage,
                                               unsigned options);
/* The pointer-table batched id-indexed launch: one kernel per stage over every record of a
 * device-resident table (IndexedTablePair array), past the 4096-byte parameter block's 20-pair
 * cap. `k_units`/`n_tiles`/`n_units` are the batch's widest shapes (ceil(k/128), n/16,
 * ceil(n/128)); narrow records idle their tail blocks. Records carry their own k and n, so one
 * batch may mix shapes. `options` bit 0 is DP4A, bit 2 the pipelined tile loop; bits 1 and 3 do
 * not apply. Every byte written equals the option launch of the same calls one batch at a
 * time. The caller fills and validates the table; this launches only. */
struct IndexedTablePair;
int inference_cuda_exl3_indexed_table(const IndexedTablePair *pairs, size_t pairs_count,
                                      uint32_t k_units, uint32_t n_tiles, uint32_t n_units,
                                      uint32_t *flag, uint32_t fused_publish, void *stream,
                                      int *out_stage, unsigned options);
/* The host-side packer for the table launch: fills `out` (an array of `pairs_count`
 * IndexedTablePair the caller owns, `out_bytes` long) from the same call and index records
 * the option launch takes, validating per record with per-record spans and cross-record write
 * isolation. A count whose records do not fit `out_bytes` is refused, so a stale size constant
 * on one side cannot overwrite the other's buffer. Shapes may differ per record; k and n are
 * written into each record. */
int inference_cuda_exl3_indexed_table_pack(const inference_cuda_exl3_linear_args *calls,
                                           const inference_cuda_exl3_index_args *index,
                                           size_t pairs_count, size_t out_bytes,
                                           IndexedTablePair *out);
/* Test probe of the DP4A decode: per word, the byte sum and the DP4A result (2 * count words);
 * per state 0..65535, the book-2 decode and its DP4A form (2 * 65536 halves). */
int inference_cuda_exl3_book2_probe(const uint32_t *words, size_t count, uint32_t *sums,
                                    uint16_t *decoded, void *stream, int *out_stage);
/* Explicit tensor-core projection for rows9..144, one F32 Z bank.
 * FP16-rounded operands feed an F32 accumulator. Operand conversion and
 * tensor-core accumulation can round differently from legacy F32 arithmetic.
 * This is not an arbitrary FP32-equivalence guarantee; real-case qualification
 * is required. No default dispatch change. */
int inference_cuda_exl3_linear_midrows(const inference_cuda_exl3_linear_args *args,
                                       void *stream, int *out_stage);
/* Opt-in variant of inference_cuda_exl3_linear_midrows (FIFI_EXL3_TALL_HILO): same contract, geometry,
 * launch grid and emissions; the F32 activations are staged as an F16 high part and a scaled F16 residual
 * and multiplied against the same decoded weight tile into two F32 accumulators (sum_hi + sum_lo / 2048),
 * so the operand rounding error is about 2^-22 of each element instead of 2^-11. The entry above is
 * unchanged. Not an FP32-equivalence guarantee: the accumulation is the tensor core's. */
int inference_cuda_exl3_linear_midrows_hilo(const inference_cuda_exl3_linear_args *args,
                                            void *stream, int *out_stage);
/* Job-table projection for rows 1..256 per job: one launch per stage (input rotation,
 * tensor-core projection, output rotation) over any number of row blocks ("jobs"), each of
 * which reads one member of a bank of same-shaped, same-rate matrices. The prefill MoE runs
 * every expert of a chunk with it. The table is read from device memory when the kernels run, so a
 * caller may also write it on the device (no caller does yet).
 *
 * A job is {member, row0, rows}: rows row0..row0+rows of x, xh, z and out belong to member
 * `member` (0 <= member < members). Member m's trellis starts trellis_stride u16 elements, and
 * its suh and svh scale_stride floats, after member m-1's; a bank stores [suh | svh] per member,
 * so suh and svh are one buffer with svh at an offset of k floats and scale_stride = k + n. A
 * single matrix has members = 1 and strides of any value (ignored).
 *
 * x:[total_rows,K] is read (rotated input rows are written to xh:[total_rows,K] as F16, two
 * bytes each); z:[total_rows,N] holds one F32 projection bank; out receives row r, column c of job j at
 * out[(row0 + r) * out_ld + out_col + c], so a projection can land in its half of a wider row.
 * Arithmetic is the rows kernel's: FP16-rounded operands, F32 accumulation, for ALL row counts.
 * Rows of different jobs must be disjoint. A job outside its bounds (member, row0, rows,
 * rows > max_rows) is skipped and sets bit0 of flag, a caller-initialized sticky u32 that
 * invalidates the whole output; check it after completion. xh, z, out and flag must be disjoint
 * from every other range; trellis, suh, svh and the table may alias each other. The table is
 * jobs_count entries of 16 bytes in device memory, read when the kernels run. K, N, bits_twice and
 * book follow the linear ABI; jobs_count <= 4096; max_rows <= 256 bounds every job's rows and
 * picks the CTA (64, 128 or 256 rows); out_ld >= out_col + N. Other rules match the linear entry points. */
typedef struct inference_cuda_exl3_job {
    int32_t member;
    int32_t row0;
    int32_t rows;
    int32_t reserved; /* zero */
} inference_cuda_exl3_job;
typedef struct inference_cuda_exl3_jobs_args {
    const uint16_t *trellis; size_t trellis_bytes;
    const float *suh; size_t suh_bytes;
    const float *svh; size_t svh_bytes;
    const float *x; size_t x_bytes;
    uint16_t *xh; size_t xh_bytes; /* F16 bits */
    float *z; size_t z_bytes;
    float *out; size_t out_bytes;
    uint32_t *flag; size_t flag_bytes;
    const inference_cuda_exl3_job *jobs; size_t jobs_bytes;
    size_t trellis_stride; size_t scale_stride; size_t members;
    size_t k; size_t n;
    size_t total_rows; size_t out_ld; size_t out_col;
    uint32_t jobs_count; uint32_t max_rows;
    int bits_twice; int book;
    /* Row lists: row r of the table (the rows back to back, jobs in order) reads row in_rows[r] of x
     * and lands in row out_rows[r] of out; a null list is the identity. Then x holds x_rows rows
     * and out out_total_rows rows (both equal total_rows without a list). The lists are read, not
     * written; an entry out of range sets bit0 of flag and the row is dropped, and repeated
     * entries of out_rows are the caller's business (the last writer wins). */
    const uint32_t *in_rows; size_t in_rows_bytes;
    const uint32_t *out_rows; size_t out_rows_bytes;
    size_t x_rows; size_t out_total_rows;
} inference_cuda_exl3_jobs_args;

/* The kernel family a job-table call would run, without launching: 1 for the register-decode kernel
 * (switch on, compute capability major >= 8 running code compiled for sm_80 or more (ptxVersion), 16-byte aligned
 * input rows, trellis and member stride, K a multiple of 32), 0 for the shared-memory kernels.
 * A negative capability_major asks the current device and FIFI_EXL3_JOBS_REG (cached, as a launch
 * does); otherwise capability_major, ptx_version and enabled are taken as given, so a test needs
 * neither a device nor the environment. Returns INFERENCE_CUDA_INVALID_ARGUMENT for a null args
 * pointer. The call's other arguments are not validated here. */
int inference_cuda_exl3_jobs_kernel(const inference_cuda_exl3_jobs_args *args, int capability_major,
                                    int ptx_version, int enabled);
#ifdef __cplusplus
static_assert(sizeof(inference_cuda_exl3_job) == 16 &&
              sizeof(inference_cuda_exl3_jobs_args) == 272 &&
              offsetof(inference_cuda_exl3_jobs_args, flag) == 112 &&
              offsetof(inference_cuda_exl3_jobs_args, jobs) == 128 &&
              offsetof(inference_cuda_exl3_jobs_args, trellis_stride) == 144 &&
              offsetof(inference_cuda_exl3_jobs_args, k) == 168 &&
              offsetof(inference_cuda_exl3_jobs_args, total_rows) == 184 &&
              offsetof(inference_cuda_exl3_jobs_args, jobs_count) == 208 &&
              offsetof(inference_cuda_exl3_jobs_args, bits_twice) == 216 &&
              offsetof(inference_cuda_exl3_jobs_args, in_rows) == 224 &&
              offsetof(inference_cuda_exl3_jobs_args, x_rows) == 256, "EXL3 jobs ABI");
#endif
int inference_cuda_exl3_jobs(const inference_cuda_exl3_jobs_args *args,
                             void *stream, int *out_stage);
/* The same launches with an explicit fused-output-rotation request (the FIFI_EXL3_JOBS_FUSED_OUT
 * default is off); the fusion applies only when the register-decode kernel is taken. */
int inference_cuda_exl3_jobs_fused_out(const inference_cuda_exl3_jobs_args *args,
                                       void *stream, int *out_stage, int fused_out);
/* The same launches with both requests explicit: the fused output rotation (above) and the
 * warp-wide input rotation. The FIFI_EXL3_JOBS_WARP_ROTATE default is off. The wide kernel gives
 * the same xh bits as the one-thread-per-value kernel; it needs x and suh 16-byte aligned (and
 * the member stride of the scales a multiple of 4 floats), xh 8-byte aligned, and a call that
 * does not meet that runs the one-thread-per-value kernel whatever the request. */
int inference_cuda_exl3_jobs_modes(const inference_cuda_exl3_jobs_args *args,
                                   void *stream, int *out_stage, int fused_out, int wide_input);
/* Whether the warp-wide input rotation would run for a call with the request `enabled` (1) or
 * not (0), from the alignment rules alone; INFERENCE_CUDA_INVALID_ARGUMENT for a null args
 * pointer. */
int inference_cuda_exl3_jobs_wide_input_kernel(const inference_cuda_exl3_jobs_args *args, int enabled);
/* The reading of the FIFI_EXL3_JOBS_WARP_ROTATE text: 1 for exactly "1", 0 for "0" or a null
 * text, -1 for any other text (a launch treats -1 as off). */
int inference_cuda_exl3_jobs_wide_input_setting(const char *text);

/* The input rotation of two job-table calls that read the same rows, in one pass: call a is
 * rotated into its xh by its scales, call b into its xh by its scales, each by the arithmetic of
 * the warp-wide rotation, so both are byte-identical to two separate calls. The calls must share
 * x (pointer and row count), the row list, the table (pointer, count, max_rows), total_rows, K,
 * the member count and the flag, and meet the alignment rules of
 * inference_cuda_exl3_jobs_wide_input_kernel. The two xh ranges must be apart from each other
 * and from x, both scales, the table and the row list (these are checked here). Each xh against
 * the other call's trellis, svh, z, out and flag is the caller's duty (the Rust wrapper's
 * require_pair_apart checks it). Writes only the two xh ranges and the flag; the projection and
 * the output rotation are not run. A call that fails the checks is refused with
 * INFERENCE_CUDA_INVALID_ARGUMENT before any launch. */
int inference_cuda_exl3_jobs_rotate_pair(const inference_cuda_exl3_jobs_args *a,
                                         const inference_cuda_exl3_jobs_args *b,
                                         void *stream, int *out_stage);
/* 1 when inference_cuda_exl3_jobs_rotate_pair accepts the two calls (the consistency and alignment
 * rules above), 0 when it does not, INFERENCE_CUDA_INVALID_ARGUMENT for a null pointer. Looks at
 * pointers and integers only. */
int inference_cuda_exl3_jobs_pair_usable(const inference_cuda_exl3_jobs_args *a,
                                         const inference_cuda_exl3_jobs_args *b);
/* The projection and the output rotation of a call whose xh already holds the rotated input rows
 * (written by inference_cuda_exl3_jobs_rotate_pair on the same stream): the input rotation is
 * skipped. fused_out is the fused output rotation request, negative for the environment's. Calling
 * it on an xh that was not rotated projects whatever xh holds. */
int inference_cuda_exl3_jobs_rotated(const inference_cuda_exl3_jobs_args *a, void *stream,
                                     int *out_stage, int fused_out);

/* Member addressing of a job-table call, shared by every stage kernel that reads the table: the
 * trellis of member m starts trellis_stride u16 elements after member m-1's, its suh and svh
 * scale_stride floats after. A single matrix (members = 1) ignores the strides. */
#ifdef __cplusplus
#ifdef __CUDACC__
#define INFERENCE_CUDA_EXL3_HD __host__ __device__
#else
#define INFERENCE_CUDA_EXL3_HD
#endif
static inline INFERENCE_CUDA_EXL3_HD const uint16_t *
inference_cuda_exl3_member_trellis(const inference_cuda_exl3_jobs_args &p, int32_t member) {
    return p.trellis + (p.members > 1 ? size_t(member) * p.trellis_stride : 0);
}
static inline INFERENCE_CUDA_EXL3_HD const float *
inference_cuda_exl3_member_suh(const inference_cuda_exl3_jobs_args &p, int32_t member) {
    return p.suh + (p.members > 1 ? size_t(member) * p.scale_stride : 0);
}
static inline INFERENCE_CUDA_EXL3_HD const float *
inference_cuda_exl3_member_svh(const inference_cuda_exl3_jobs_args &p, int32_t member) {
    return p.svh + (p.members > 1 ? size_t(member) * p.scale_stride : 0);
}
#endif
#ifdef __cplusplus
}
#endif
#endif
