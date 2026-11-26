/* Depth-residual stream of Inkling 2, kept on the device: mix, seal and add.
 * One header per kernel file; abi_resident.h stays closed.
 *
 * State layout (f32, device memory), `width` floats per row:
 *   state = [ current | sealed[0] | sealed[1] | ... | sealed[blocks - 1] ]
 * A sealed row is the RMS-normalized copy of the row that was current when it was sealed.
 * `blocks` and `first_block` are kernel arguments: the launch sequence knows them.
 *
 * inference_cuda_inkling_stream_mix
 *   Rows in this order: sealed[blocks - 1], sealed[blocks - 2], ..., sealed[first_block], then
 *   the normalized current row. Each row scores dot(row, query) / sqrt(width). The weights are the
 *   softmax of the scores (the peak is subtracted first) and `out` is their weighted mean. This is
 *   `Stream::mix(first_block, query)` of the host.
 *
 * inference_cuda_inkling_stream_mix_jobs
 *   `count` independent mixes of `inference_cuda_inkling_stream_mix` over one `width` and `eps`
 *   in one launch, one block per job: `ranges` holds `state, query, out, health` per job and
 *   `shape` its `first_block, blocks`. A job's output and health word share no byte with
 *   another job's row, output or health word, except that jobs may name the same health word.
 *
 * inference_cuda_inkling_stream_seal
 *   sealed[blocks] = current * rms_scale(current), then current = next. This is `Stream::seal(next)`.
 *
 * inference_cuda_inkling_stream_add
 *   current = current + row, each sum rounded to nearest even. This is `Stream::add_current(row)`.
 *
 * inference_cuda_inkling_stream_norm_rows
 *   `count` rows `dst = src * rms_scale(src)` in one launch, one block per row: the gate halves
 *   of a cross-pass controller (`rms_norm_with_scale` of the host). `ranges` holds `src, dst,
 *   health` per row, all over one `width` and `eps`. Each row's `dst` shares no byte with its
 *   `src`, and no row's `dst` or health word shares a byte with another row's ranges, except
 *   that rows may name the same health word.
 *
 * RMS scale: (dot(x, x) / width + eps).sqrt().recip(), every operation rounded to nearest even.
 * Reductions are reordered against the host's ordered sums (a block sum is a warp tree over the
 * lanes, then the warp partials in warp order; the softmax's exponentials run one per lane), so
 * results agree with the host within floating-point error and are not bit-identical.
 * `health` is one float: left alone, or quiet NaN when an input or an output is not finite.
 *
 * inference_cuda_inkling_zero_words / inference_cuda_inkling_copy_words
 *   Bookkeeping rows in one launch, one block per row: `rows[r]` holds `words[r]` floats to
 *   zero; a copy moves `words[r]` floats from `source[r]` to `destination[r]`, which share no byte.
 *
 * Every entry takes the stream explicitly, returns INFERENCE_CUDA_INVALID_ARGUMENT for a bad
 * argument before any CUDA call, never falls back to the default stream and does not synchronize.
 * `out` and `health` must not overlap any other range.
 */
#ifndef INFERENCE_CUDA_RESIDENT_INKLING_STREAM_H
#define INFERENCE_CUDA_RESIDENT_INKLING_STREAM_H

#ifdef __cplusplus
extern "C" {
#endif

int inference_cuda_inkling_zero_words(const void *const *rows, const int *words, int count,
                                      void *stream, int *out_stage);

int inference_cuda_inkling_copy_words(const void *const *source, void *const *destination,
                                      const int *words, int count, void *stream, int *out_stage);

int inference_cuda_inkling_stream_mix(const float *state, const float *query, float *out,
                                      float *health, int first_block, int blocks, int width,
                                      float eps, void *stream, int *out_stage);

/* Per job `state, query, out, health, addend, gamma, normed`: an addend row is added into the
 * current row of `state` first (the host's add_current); gamma and normed, both null or both
 * set, make the job also write `normed = out * rms_scale(out, norm_eps[j]) * gamma`. */
int inference_cuda_inkling_stream_mix_jobs(const void *const *ranges, const int *shape,
                                           const float *norm_eps, int count, int width, float eps,
                                           void *stream, int *out_stage);

int inference_cuda_inkling_stream_seal(float *state, const float *next, float *health, int blocks,
                                       int width, float eps, void *stream, int *out_stage);

int inference_cuda_inkling_stream_add(float *state, const float *row, float *health, int width,
                                      void *stream, int *out_stage);

int inference_cuda_inkling_stream_norm_rows(const void *const *ranges, int count, int width,
                                            float eps, void *stream, int *out_stage);

#ifdef __cplusplus
}
#endif

#endif /* INFERENCE_CUDA_RESIDENT_INKLING_STREAM_H */
