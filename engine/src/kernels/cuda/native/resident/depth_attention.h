/* Attention-residual depth stream, kept on the device: mix, seal and add.
 * One header per kernel file; abi_resident.h stays closed.
 *
 * State layout (f32, device memory), `width` floats per row:
 *   state = [ current | sealed[0] | sealed[1] | ... | sealed[blocks - 1] ]
 * A sealed row is the RMS-normalized copy of the row that was current when it was sealed.
 * `blocks` and `first_block` are kernel arguments: the launch sequence knows them.
 *
 * inference_cuda_depth_attention_mix
 *   Rows in this order: sealed[blocks - 1], sealed[blocks - 2], ..., sealed[first_block], then
 *   the normalized current row. Each row scores dot(row, query) / sqrt(width). The weights are the
 *   softmax of the scores (the peak is subtracted first) and `out` is their weighted mean. This is
 *   `Stream::mix(first_block, query)` of the host. Without a query (null) the output is the raw
 *   current row; with a `gamma` it is `norm(mix) * gamma` with `norm_eps`, as `CpuForward::norm`.
 *   Output rows are `out_stride` floats apart.
 *
 * inference_cuda_depth_attention_mix_jobs
 *   Up to 8 independent mixes in one launch, a block per job, each with its own state, query,
 *   output and sealed rows. Each output has the bits of the same mix launched on its own.
 *
 * inference_cuda_depth_attention_seal
 *   sealed[blocks] = current * rms_scale(current), then current = next. This is `Stream::seal(next)`.
 *
 * inference_cuda_depth_attention_add
 *   current = current + row, each sum rounded to nearest even. This is `Stream::add_current(row)`.
 *
 * inference_cuda_depth_attention_norm_rows
 *   dst = src * rms_scale(src) for up to 8 rows of one width in one launch, a block per row: the
 *   gate halves of the cross-pass controllers (`rms_norm_with_scale` of the host).
 *
 * RMS scale: (dot(x, x) / width + eps).sqrt().recip(), every operation rounded to nearest even.
 * Every row reduction is in the lane order of `lane_sum.cuh`, so results agree with the host's
 * ordered sums within floating-point error and are not bit-identical, while a row gives the same
 * bits whichever of these kernels, or of the other resident row norms of the same model, reduces it.
 * `health` is one float: left alone, or quiet NaN when an input or an output is not finite.
 *
 * Every entry takes the stream explicitly, returns INFERENCE_CUDA_INVALID_ARGUMENT for a bad
 * argument before any CUDA call, never falls back to the default stream and does not synchronize.
 * `out` and `health` must not overlap any other range.
 */
#ifndef INFERENCE_CUDA_RESIDENT_DEPTH_ATTENTION_H
#define INFERENCE_CUDA_RESIDENT_DEPTH_ATTENTION_H

#ifdef __cplusplus
extern "C" {
#endif

/* Each of mix, seal and add runs on `rows` states, `stride` rows of `width` floats apart, one
 * block per state; their other row operands (seal next, add addend) are packed `width` apart,
 * and mix outputs `out_stride` apart. `health` is one float for the whole launch. */
int inference_cuda_depth_attention_mix(const float *state, const float *query, const float *gamma,
                                      float *out, float *health, int first_block, int blocks,
                                      int width, float mix_eps, float norm_eps, int rows,
                                      int stride, int out_stride, void *stream, int *out_stage);

/* `rows` holds `state, query, out` of each of the `count` jobs and `extents` their `first_block,
 * blocks`. Every output and `health` share no byte with any row a job reads or with each other;
 * the jobs share `health`. */
int inference_cuda_depth_attention_mix_jobs(const void *const *rows, const int *extents, int count,
                                           float *health, int width, float eps, void *stream,
                                           int *out_stage);

int inference_cuda_depth_attention_seal(float *state, const float *next, float *health, int blocks,
                                       int width, float eps, int rows, int stride, void *stream,
                                       int *out_stage);

int inference_cuda_depth_attention_add(float *state, const float *row, float *health, int width,
                                      int rows, int stride, void *stream, int *out_stage);

/* `ranges` holds `src, dst, health` of each of the `count` rows. Every output shares no byte with
 * any source or other output, and every health word none with any source or output; rows may
 * share one health word. */
int inference_cuda_depth_attention_norm_rows(const void *const *ranges, int count, int width,
                                            float eps, void *stream, int *out_stage);

#ifdef __cplusplus
}
#endif

#endif /* INFERENCE_CUDA_RESIDENT_DEPTH_ATTENTION_H */
