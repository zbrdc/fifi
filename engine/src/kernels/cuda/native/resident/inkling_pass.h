/* Row-group glue of the device-resident Inkling prefill pass. One header per kernel file;
 * abi_resident.h stays closed.
 *
 * Ranges are f32 in device memory unless stated. `rows` prompt rows share one depth stream
 * layout and one sidecar layout; `width` is the hidden size.
 *   stream state   sealed[blocks][rows][width] holds the sealed (normalized) rows oldest
 *                  first, current[rows][width] the current raw row of every prompt row.
 *   health[rows]   One word per row: left alone, or quiet NaN when an input, a weight, an
 *                  intermediate or an output of that row was not finite.
 *   status[rows]   u32 per row: bit 0 set when a prepared K or V value of that row was not
 *                  finite (the caller's attention status plane; it is ORed, not replaced).
 *
 * inference_cuda_inkling_pass_mix: `Stream::mix(first, query)` of family/inkling2/cpu_forward/
 *   depth_mix.rs for every row, one block per row. Mix row j is sealed row (blocks - 1 - j)
 * for j < blocks - first, then the current row scaled by its RMS norm with `eps`. Each score
 * is dot(row, query) / sqrt(width), the peak the maximum score, each weight
 * exp(score - peak); out[r * out_stride + c] = (sum over mix rows, in that order, of
 * weight * row[c]) / mass with mass the ordered sum of the weights.
 * inference_cuda_inkling_pass_seal: `Stream::seal(next)`: sealed[blocks][r] = current[r] * s
 *   with s = 1 / sqrt(dot(current[r], current[r]) / width + eps), then current[r] = next[r].
 * inference_cuda_inkling_pass_add: `Stream::add_current(row)`: current[r] += row[r].
 * inference_cuda_inkling_pass_prep: `prepare_attention_row` (family/inkling2/cpu_attention.rs)
 *   of the K and V planes of every row, taps T = 4, head width 128, `kv_heads` heads:
 *   raw[r * raw_stride + c] is the raw K of row r (c < kv_heads * 128) and the raw V at
 *   + kv_heads * 128. history_k, history_v hold the T - 1 raw rows before the group, oldest
 *   first (zero rows stand for the inputs a fresh sequence lacks). kernel_k, kernel_v hold
 *   kv_heads * 128 * T weights: with dynamic_taps = 0, kernel[c * T + t] is tap t of channel
 *   c and row r reads lag l (its own raw row for l = 0, the history for r - l < 0) as
 *   kernel[c * T + T - 1 - l]; with dynamic_taps = 1 (global attention), kernel[l * C + c]
 *   plus dynamic[r * dynamic_stride + (c / 128) * T + l]. For each channel, in order:
 *   acc = 0 + w_0 * x_0 + .. + w_3 * x_3;  conv = (acc + raw[c]) * (1 / sqrt(T + 1)).
 *   K heads then take the host's clip: scale = min(1, qk_limit / (sqrt(dot(head, head)) +
 *   1e-6)); V heads the RMS norm: scale = 1 / sqrt(dot(head, head) / 128 + eps). The result
 *   goes to side[r * side_stride + key_at + c] and side[r * side_stride + value_at + c].
 *   A non-finite value sets bit 0 of status[r].
 *
 * inference_cuda_inkling_pass_gate: `control_input_with_scales` of every row: gate[r] =
 *   [norm(in[r]) | norm(msg[r])] with `eps`, one block per row; `gate_stride` at least 2 * width.
 * inference_cuda_inkling_pass_apply: `apply_control` then `norm` of every row: with `control`,
 *   x[c] = (in[c] - tanh(forget[c]) * in[c]) + tanh(gain[c]) * msg[c] with gain and forget rows
 *   of `width` at `stride`, else x = in; out = x * rms(x, eps) * gamma. `out` may be `in` (in
 *   place); otherwise `in` is copied when `control` is 0.
 * inference_cuda_inkling_pass_tails: the last 3 rows of (the `past` valid rows of `old`, oldest
 *   first, then the `rows` rows of `src` at `stride`), oldest first, into `out` (3 rows).
 * inference_cuda_inkling_pass_positions: side[r * stride + pos_at] = u32(first + r).
 * inference_cuda_inkling_pass_zero: sets `count` u32 words to zero.
 *
 * Every entry takes the stream explicitly, returns INFERENCE_CUDA_INVALID_ARGUMENT for a bad
 * argument before any CUDA call, never falls back to the default stream and does not
 * synchronize.
 */
#ifndef INFERENCE_CUDA_RESIDENT_INKLING_PASS_H
#define INFERENCE_CUDA_RESIDENT_INKLING_PASS_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

int inference_cuda_inkling_pass_mix(const float *sealed, const float *current, const float *query,
                                    float *out, float *health, int rows, int blocks, int first,
                                    int width, int out_stride, float eps, void *stream,
                                    int *out_stage);

int inference_cuda_inkling_pass_seal(float *sealed, float *current, const float *next,
                                     float *health, int rows, int blocks, int width, float eps,
                                     void *stream, int *out_stage);

int inference_cuda_inkling_pass_add(float *current, const float *row, float *health, int rows,
                                    int width, void *stream, int *out_stage);

int inference_cuda_inkling_pass_prep(void *const *ranges, int rows, int raw_stride,
                                     int side_stride, int key_at, int value_at,
                                     int dynamic_stride, int kv_heads, float qk_limit, float eps,
                                     int dynamic_taps, void *stream, int *out_stage);

int inference_cuda_inkling_pass_gate(const float *in, const float *msg, float *gate, float *health,
                                     int rows, int width, int in_stride, int msg_stride,
                                     int gate_stride, float eps, void *stream, int *out_stage);

int inference_cuda_inkling_pass_apply(const float *in, const float *msg, const float *gain,
                                      const float *forget, const float *gamma, float *out,
                                      float *health, int rows, int width, int stride, float eps,
                                      int control, void *stream, int *out_stage);

int inference_cuda_inkling_pass_tails(const float *src, int rows, int stride, int width, int past,
                                      const float *old, float *out, void *stream, int *out_stage);

int inference_cuda_inkling_pass_positions(uint32_t *side, int rows, int stride, int pos_at,
                                          int first, void *stream, int *out_stage);

int inference_cuda_inkling_pass_zero(uint32_t *words, int count, void *stream, int *out_stage);

#ifdef __cplusplus
}
#endif

#endif /* INFERENCE_CUDA_RESIDENT_INKLING_PASS_H */
