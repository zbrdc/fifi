/* The many-rows entries of the resident decode step (native/resident/multi_row.h): one launch
 * advances N rows, each row the single token of a sequence of its own. Included by multi_row.cu and
 * attn_decode_q8.cu and read by the Rust FFI mirror (native/resident/multi_row.rs); nothing else.
 *
 * Row r of a launch reads row r of every per-row input and writes row r of every per-row output
 * (f32 rows of the same shape as the single-token launches: proj [n][ld], conv [n][conv_ld],
 * out [n][...], q [n][n_head][256], k and v rows [n][n_kv_head][head_dim]). What differs per row
 * lives in a device table that the caller writes before the launch (a copy on the same stream):
 *   state tables   `void *const *`: n addresses, row r's recurrent or conv state of this layer
 *                  (the layout of the single-token entry: planar f32 [3][channels] for the conv,
 *                  f32 [n_v_heads][128][128] for the delta)
 *   kv row table   inference_cuda_kv_row[n]: row r's KV8 planes of this layer and its step block
 * The step block is the 32-byte StepParams of resident/step_params.h of that row's sequence: the
 * position word at byte 4 (attn_prep, kv8_append), kv_len at byte 8 (attn_decode_q8) and the
 * status word at byte 16 (mask 4 on a refused append). The words are read when the kernel runs, so
 * one captured graph serves every position of every row.
 *
 * Each entry computes row r exactly as the single-token entry of the same name does for that row
 * alone (m = 1): the output bytes of N rows in one launch are the output bytes of N single-row
 * launches. The rows must write disjoint bytes: distinct state addresses, distinct planes (the
 * rows of one sequence are not many rows of this entry; a K-row window of one sequence keeps the
 * m-token entries), distinct output rows. The host wrapper proves that for the addresses it is
 * given; the entries read the tables as written. A row whose position or kv_len is out of range
 * is refused alone, as in the single-token launch: its q or out row is NaN, no plane byte of it
 * is written and mask 4 reaches its own status word; the other rows are unaffected.
 *
 * Every entry enqueues on stream (not NULL) and returns INFERENCE_CUDA_INVALID_ARGUMENT before
 * any launch for a bad argument. Limits: 1 <= n <= 256 rows.
 */
#ifndef INFERENCE_CUDA_RESIDENT_MULTI_ROW_H
#define INFERENCE_CUDA_RESIDENT_MULTI_ROW_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define INFERENCE_CUDA_MULTI_ROW_MAX 256
#define INFERENCE_CUDA_STEP_POS_OFFSET 4
#define INFERENCE_CUDA_STEP_KV_LEN_OFFSET 8
#define INFERENCE_CUDA_STEP_STATUS_OFFSET 16

/* One row of a KV row table: 64 bytes, 8-aligned. Planes are those of kv8_append.h of this layer
 * of this row's sequence (int8 codes [n_kv_head][cells][head_dim], fp16 scales [n_kv_head][cells]
 * [head_dim / 32]); step is the row's StepParams block. */
typedef struct inference_cuda_kv_row {
    void *k_codes;
    void *k_scales;
    void *v_codes;
    void *v_scales;
    void *step;
    long long cells;      /* plane cells per head of THIS row's planes (rows of one launch may differ):
                           * every entry reads the row's own count for its bounds and plane strides;
                           * the `cells` argument of an entry only has to be valid (set it to the
                           * fewest of the rows) */
    void *reserved[2];
} inference_cuda_kv_row;

typedef struct inference_cuda_gdn_conv_multi_args {
    const void *proj;     /* f32 [n][ld], the first `channels` columns of a row are read */
    const void *w;        /* f32 [channels][4], 16-byte aligned, shared by every row */
    void *const *states;  /* device table: n addresses of planar f32 [3][channels] */
    void *out;            /* f32 [n][channels] */
    int n;
    int ld;
    int channels;
} inference_cuda_gdn_conv_multi_args;
int inference_cuda_gdn_conv_multi_launch(const inference_cuda_gdn_conv_multi_args *args,
                                         void *stream, int *out_stage);
int inference_cuda_gdn_conv_multi_ieee_launch(const inference_cuda_gdn_conv_multi_args *args,
                                              void *stream, int *out_stage);

typedef struct inference_cuda_gdn_delta_multi_args {
    const void *conv;     /* f32 [n][conv_ld] */
    const void *proj;     /* f32 [n][ld] */
    const void *a;
    const void *dt_bias;
    const void *norm_w;
    void *const *states;  /* device table: n addresses of f32 [n_v_heads][128][128] */
    void *out;            /* f32 [n][n_v_heads * 128] */
    int n;
    int conv_ld;
    int ld;
    int z_off;
    int beta_off;
    int alpha_off;
    int n_k_heads;
    int n_v_heads;
    float eps;            /* 1e-6 */
} inference_cuda_gdn_delta_multi_args;
int inference_cuda_gdn_delta_multi_launch(const inference_cuda_gdn_delta_multi_args *args,
                                          void *stream, int *out_stage);
int inference_cuda_gdn_delta_multi_ieee_launch(const inference_cuda_gdn_delta_multi_args *args,
                                               void *stream, int *out_stage);

/* attn_prep_rows for n rows of n sequences: row r takes the position word of rows[r].step. */
typedef struct inference_cuda_attn_prep_multi_args {
    const void *proj;     /* f32 [n][ld] */
    const void *q_norm;
    const void *k_norm;
    void *k_rows;         /* f32 [n][n_kv_head][256] */
    void *v_rows;
    void *q;              /* f32 [n][n_head][256] */
    const inference_cuda_kv_row *rows;
    int n;
    int ld;
    int n_head;           /* 1..256 */
    int n_kv_head;        /* 1..64 */
    int head_dim;         /* 256 */
    int n_rot;            /* a power of two, 2..256 */
    int cells;            /* plane cells per head of every row, 1..2^30 - 1 */
    float eps;
    float rope_base;
} inference_cuda_attn_prep_multi_args;
int inference_cuda_attn_prep_multi_launch(const inference_cuda_attn_prep_multi_args *args,
                                          void *stream, int *out_stage);

/* kv8_append for n rows of n sequences: row r appends at the position word of rows[r].step. */
typedef struct inference_cuda_kv8_append_multi_args {
    const void *k;        /* f32 [n][n_kv_head][head_dim] */
    const void *v;
    const inference_cuda_kv_row *rows;
    int n;
    int n_kv_head;
    int head_dim;         /* a multiple of 32, 32..1024 */
    int cells;
    int code_scale;       /* 1 = stored half, 2 = unrounded f32 */
    int ring;             /* 0: cell = position; R > 0: cell = position % R, R <= cells */
} inference_cuda_kv8_append_multi_args;
int inference_cuda_kv8_append_multi_launch(const inference_cuda_kv8_append_multi_args *args,
                                           void *stream, int *out_stage);

/* attn_decode_q8 for n rows of n sequences: row r sees the kv_len word of rows[r].step keys of its
 * own planes. The fields beyond rows are those of attn_decode_q8.h with m = n. */
typedef struct inference_cuda_attn_decode_q8_multi_args {
    const void *q;        /* f32 [n][n_head][head_dim] */
    const void *proj;     /* NULL or f32 [n][ld] */
    const inference_cuda_kv_row *rows;
    void *part;           /* inference_cuda_attn_decode_q8_part_floats(n, ...) floats, 16-aligned */
    void *out;            /* f32 [n][n_head * head_dim] */
    int n;
    int ld;
    int n_head;
    int n_kv_head;
    int head_dim;
    int cells;
    int max_splits;
    int min_keys;
    int window;
    int ring;
    float scale;
} inference_cuda_attn_decode_q8_multi_args;
int inference_cuda_attn_decode_q8_multi_launch(const inference_cuda_attn_decode_q8_multi_args *args,
                                               void *stream, int *out_stage);

/* attn_decode (the fp16 cache) for n rows of n sequences: row r sees the kv_len word of rows[r].step
 * keys of its own planes; rows[r].k_codes and rows[r].v_codes name the fp16 K and V planes
 * [cells][n_kv_head][head_dim] of attn_decode.h, 16-byte aligned as the single entry demands
 * (the other row fields are not read; the table itself is 8-aligned). The fields
 * beyond rows are those of attn_decode.h with m = n; `part` is
 * inference_cuda_attn_decode_part_floats(n, ...) floats, 16-aligned. */
typedef struct inference_cuda_attn_decode_multi_args {
    const void *q;        /* f32 [n][n_head][head_dim] */
    const void *proj;     /* NULL or f32 [n][ld] */
    const inference_cuda_kv_row *rows;
    void *part;
    void *out;            /* f32 [n][n_head * head_dim] */
    int n;
    int ld;
    int n_head;
    int n_kv_head;
    int head_dim;
    int cache_rows;
    int max_splits;
    int min_keys;
    int window;
    int ring;
    float scale;
} inference_cuda_attn_decode_multi_args;
int inference_cuda_attn_decode_multi_launch(const inference_cuda_attn_decode_multi_args *args,
                                            void *stream, int *out_stage);

/* The fp16 append for n rows of n sequences: row r writes its k and v rows (f32 [n][n_kv_head]
 * [head_dim], rounded to nearest even) to cell *pos of rows[r].k_codes and rows[r].v_codes, the
 * fp16 planes [cells][n_kv_head][head_dim], where pos is the position word of rows[r].step. A row
 * whose position is outside [0, cells) writes nothing and ORs mask 4 into its own status word. */
typedef struct inference_cuda_kv16_append_multi_args {
    const void *k;        /* f32 [n][n_kv_head][head_dim] */
    const void *v;
    const inference_cuda_kv_row *rows;
    int n;
    int n_kv_head;
    int head_dim;         /* 32..1024 */
    int cells;
} inference_cuda_kv16_append_multi_args;
int inference_cuda_kv16_append_multi_launch(const inference_cuda_kv16_append_multi_args *args,
                                            void *stream, int *out_stage);

/* attn_decode_gemma for n rows of n sequences (Bolt: n_q_head 8, n_kv_head 2): row r sees the
 * kv_len word of rows[r].step keys of its own planes; no output gate. The fields beyond rows are
 * those of attn_decode_gemma.h with m = n; the split-K path (split != 0) takes `part` of
 * inference_cuda_attn_decode_gemma_part_floats(n, ...) floats, 16-aligned. */
typedef struct inference_cuda_attn_decode_gemma_multi_args {
    const void *q;        /* f32 [n][n_q_head][head_dim] */
    const inference_cuda_kv_row *rows;
    void *part;           /* NULL when split == 0 */
    void *out;            /* f32 [n][n_q_head][head_dim] */
    int n;
    int n_q_head;
    int n_kv_head;
    int head_dim;
    int cells;
    int window;
    int split;
    int max_splits;
    int min_keys;
    int ring;
    float scale;
} inference_cuda_attn_decode_gemma_multi_args;
int inference_cuda_attn_decode_gemma_multi_launch(const inference_cuda_attn_decode_gemma_multi_args *args,
                                                  void *stream, int *out_stage);

#ifdef __cplusplus
static_assert(sizeof(inference_cuda_kv_row) == 64, "inference_cuda_kv_row is 64 bytes");
}
#endif

#endif
