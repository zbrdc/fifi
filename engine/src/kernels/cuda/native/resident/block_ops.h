/*
 * Launch entry of block_ops: the row operations of a hybrid decoder block that are not
 * shared with another resident kernel yet (the convolution is `gdn_conv`, the norm is `add_rmsnorm_q8`, the stream-mixing apply is `hyper_mix_apply`): the four-stream mixed residual with its stream-mixing matrix, a
 * inputs of the shared delta-rule scan (its decay per key channel), absorbed latent attention over one shared
 * key/value head, the SiTU activation with its q8_1 quantize.
 * Launch-only: nothing here allocates, copies or synchronizes. Every operation works on
 * `n` token rows at once, so decode (n = 1) and prefill chunks share the entry.
 *
 * One generic argument block carries up to 12 device ranges, 12 integers and 4 floats; the
 * meaning of each slot is fixed per operation and listed beside the operation code below.
 * The entry checks every slot an operation uses (non-NULL, 4-byte aligned, at least the
 * bytes the integers imply) before any CUDA call and returns INFERENCE_CUDA_INVALID_ARGUMENT
 * otherwise. Ranges the operation writes may not share a byte with any other slot.
 */
#ifndef INFERENCE_CUDA_RESIDENT_BLOCK_OPS_H
#define INFERENCE_CUDA_RESIDENT_BLOCK_OPS_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define INFERENCE_CUDA_BLOCK_OPS_PTRS 12
#define INFERENCE_CUDA_BLOCK_OPS_INTS 12
#define INFERENCE_CUDA_BLOCK_OPS_FLOATS 4

/* Operation codes. Slots are `p0..p11`, integers `i0..i11`, floats `f0..f3`.
 *
 * 1 HC_READ   mixed-residual read of one site. p0 resid f32 [n][4*h]; p1 fn f16 [17][4*h];
 *             p2 base f32 [17]; p3 scale f32 [4]; p4 branch f32 [n][h] (W); p5 post f32
 *             [n][4] (W); p6 comb f32 [n][16] (W). i0 n; i1 h. f0 epsilon; f1 magnitude.
 * 4 HC_MEAN   p0 resid f32 [rows][4*h]; p1 out f32 [h] (W). i0 row; i1 h.
 * 6 RDA_PACK  the inputs of the shared delta scan. p0 q; p1 k; p2 v; p3 g1; each f32
 *             [n][heads*hd]; p4 dt_bias f32 [heads*hd]; p5 a f32 [heads]; p6 beta f32
 *             [n][heads]; p7 delta f32 [n][heads]; p8 g2 f32 [n][heads*hd]; p9 qkv f32
 *             [n][3*heads*hd] (W: q, k, and v scaled by the fold of its beta and delta);
 *             p10 decay f32 [n][heads*hd] (W); p11 proj f32 [n][heads*hd + heads] (W: g2,
 *             then the beta logits). i0 n; i1 heads; i2 hd (128). f0 gate lower bound.
 * 7 MLA_Q_ABSORB p0 qb f32 [n][heads*(nope+rope)]; p1 wk_b f32 [heads][rank][nope];
 *             p2 qeff f32 [n][heads][rank+rope] (W). i0 n; i1 heads; i2 nope; i3 rope;
 *             i4 rank.
 * 8 MLA_KV_PREP p0 kv f32 [n][rank+rope]; p1 norm_w f32 [rank]; p2 cache f16
 *             [cells][rank+rope] (W). i0 n; i1 rank; i2 rope; i3 pos0; i4 cells.
 *             f0 epsilon.
 * 9 MLA_ATTN  p0 qeff f32 [n][heads][rank+rope]; p1 cache f16 [cells][rank+rope]; p2
 *             part f32 (W). i0 n; i1 heads (multiple of 8); i2 rank; i3 rope; i4 pos0;
 *             i5 splits; i6 keys_per_split (multiple of 32); f0 score scale. rank is 512
 *             and rope 64.
 * 10 MLA_REDUCE p0 part f32; p1 mix f32 [n][heads][rank] (W). i0 n; i1 heads; i2 rank;
 *             i3 splits.
 * 11 MLA_PROJECT p0 mix f32 [n][heads][rank]; p1 wv_b f32 [heads][v][rank]; p2 out f32
 *             [n][heads*v] (W). i0 n; i1 heads; i2 rank; i3 v.
 * 12 SITU_QUANT p0 gate f32 rows at ld_gate; p1 up f32 rows at ld_up; p2 q8_1 rows (W,
 *             optional); p3 y f32 [rows][w] (W, optional; one of p2 and p3 is present).
 *             i0 rows; i1 w; i2 ld_gate; i3 ld_up. f0 beta; f1 linear beta.
 * 13 GATED_QUANT p0 x f32 [rows][w]; p1 gate f32 [rows][w] (optional); p2 q8_1 rows (W,
 *             optional); p3 y f32 [rows][w] (W, optional; one of p2 and p3 is present).
 *             i0 rows; i1 w.
 * 15 FILL     p0 f32 [i0] (W). i0 count. f0 value.
 * 16 MLA_ATTN_CHUNK the slots, integers and partials of MLA_ATTN, for a chunk of rows:
 *             query tiles of 8 rows x 8 heads on tensor cores, the queries and the
 *             tile probabilities rounded to f16. p0 and p2 are 8-byte aligned, p1
 *             16-byte. A device or an image without the kernel runs MLA_ATTN.
 * 17 HC_BROADCAST copies each row into every residual stream, bit for bit. p0 src f32
 *             [n][h]; p1 dst f32 [n][streams][h] (W). i0 n; i1 streams; i2 h.
 * 18 MLA_ATTN_DECODE single-row shared-KV partial softmax attention across up to 64 heads per
 *             block on tensor cores (8 heads/block fallback on < sm_80). p0 qeff f32 [heads][576]
 *             (8B aligned); p1 cache f16 [cells][576] (16B aligned); p2 part f32
 *             [heads/8][splits][8][516] (W, 8B aligned); p3 optional pos_ptr i32 [1] (when
 *             non-null, reads pos from device memory). i0 heads (8..=256, multiple of 8);
 *             i1 rank (512); i2 rope (64); i3 splits (1..=64); i4 pos (used when p3 is null);
 *             i5 cells; f0 scale.
 * 19 MLA_KV_PREP_STEP single-row MLA_KV_PREP reading pos from device i32 word p3 (for graph
 *             capture). p0 kv f32 [576]; p1 w f32 [512]; p2 cache f16 [cells][576] (W);
 *             p3 pos_ptr i32 [1]. i0 rank (512); i1 rope (64); i2 cells; f0 eps.
 * 20 F16_GEMM batched projection Y[n][rows] = X[n][cols] * W[rows][cols]^T on f16 tensor cores
 *             (f32 accumulation; scalar fallback on < sm_80). p0 x f32 [n][cols];
 *             p1 w f16 [rows][cols]; p2 y f32 [n][rows] (W). i0 n; i1 rows; i2 cols.
 * 21 MLA_Q_ABSORB_CHUNK batched MLA_Q_ABSORB on tf32 tensor cores (f32 accumulation; falls back
 *             to MLA_Q_ABSORB on < sm_80). Same slots and integers as MLA_Q_ABSORB.
 * 22 MLA_PROJECT_CHUNK batched MLA_PROJECT on tf32 tensor cores (f32 accumulation; falls back
 *             to MLA_PROJECT on < sm_80). Same slots and integers as MLA_PROJECT.
 */
enum {
    INFERENCE_CUDA_BLOCK_OP_HC_READ = 1,
    INFERENCE_CUDA_BLOCK_OP_HC_MEAN = 4,
    INFERENCE_CUDA_BLOCK_OP_RDA_PACK = 6,
    INFERENCE_CUDA_BLOCK_OP_MLA_Q_ABSORB = 7,
    INFERENCE_CUDA_BLOCK_OP_MLA_KV_PREP = 8,
    INFERENCE_CUDA_BLOCK_OP_MLA_ATTN = 9,
    INFERENCE_CUDA_BLOCK_OP_MLA_REDUCE = 10,
    INFERENCE_CUDA_BLOCK_OP_MLA_PROJECT = 11,
    INFERENCE_CUDA_BLOCK_OP_SITU_QUANT = 12,
    INFERENCE_CUDA_BLOCK_OP_GATED_QUANT = 13,
    INFERENCE_CUDA_BLOCK_OP_FILL = 15,
    INFERENCE_CUDA_BLOCK_OP_MLA_ATTN_CHUNK = 16,
    INFERENCE_CUDA_BLOCK_OP_HC_BROADCAST = 17,
    INFERENCE_CUDA_BLOCK_OP_MLA_ATTN_DECODE = 18,
    INFERENCE_CUDA_BLOCK_OP_MLA_KV_PREP_STEP = 19,
    INFERENCE_CUDA_BLOCK_OP_F16_GEMM = 20,
    INFERENCE_CUDA_BLOCK_OP_MLA_Q_ABSORB_CHUNK = 21,
    INFERENCE_CUDA_BLOCK_OP_MLA_PROJECT_CHUNK = 22
};

typedef struct inference_cuda_block_ops_args {
    int op;
    int ints[INFERENCE_CUDA_BLOCK_OPS_INTS];
    float floats[INFERENCE_CUDA_BLOCK_OPS_FLOATS];
    void *ptr[INFERENCE_CUDA_BLOCK_OPS_PTRS]; /* NULL for an unused slot */
    size_t bytes[INFERENCE_CUDA_BLOCK_OPS_PTRS]; /* bytes of the range behind each slot */
} inference_cuda_block_ops_args;

#ifdef __cplusplus
static_assert(sizeof(inference_cuda_block_ops_args) == 264 && offsetof(inference_cuda_block_ops_args, ints) == 4 && offsetof(inference_cuda_block_ops_args, floats) == 52 && offsetof(inference_cuda_block_ops_args, ptr) == 72 && offsetof(inference_cuda_block_ops_args, bytes) == 168, "layout");
#else
_Static_assert(sizeof(inference_cuda_block_ops_args) == 264 && offsetof(inference_cuda_block_ops_args, ints) == 4 && offsetof(inference_cuda_block_ops_args, floats) == 52 && offsetof(inference_cuda_block_ops_args, ptr) == 72 && offsetof(inference_cuda_block_ops_args, bytes) == 168, "layout");
#endif

/* One launch on stream (must not be NULL). Returns INFERENCE_CUDA_INVALID_ARGUMENT before
 * any CUDA call for a bad argument, else the launch status. */
int inference_cuda_block_ops_launch(const inference_cuda_block_ops_args *args, void *stream,
                                    int *out_stage);

/* Mode-selectable block_ops launch: rounded_q8 (0 = legacy unrounded ds.y, 1 = rounded ds.y). */
int inference_cuda_block_ops_launch_mode(const inference_cuda_block_ops_args *args, void *stream,
                                         int *out_stage, int rounded_q8);

#ifdef __cplusplus
}
#endif

#endif /* INFERENCE_CUDA_RESIDENT_BLOCK_OPS_H */
