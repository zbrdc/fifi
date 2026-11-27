/* Causal short convolution with a residual and a 1/sqrt(taps + 1) scale, over a history that
 * stays on the device. One header per kernel file; abi_resident.h stays closed.
 *
 * Ranges are f32 in device memory. With `taps` T and `width` W:
 *   kernel[W * T]            kernel[c * T + t] is the tap-t weight of channel c.
 *   history[(T - 1) * W]     Row 0 is the newest past input, row T - 2 the oldest.
 *   input[W], out[W]         The current row and its result.
 * For every channel c, in this order and with every operation rounded to nearest even:
 *   acc    = 0 + kernel[c*T + T - 1] * input[c] + kernel[c*T + T - 2] * history[0*W + c] + ...
 *          + kernel[c*T + 0] * history[(T - 2)*W + c]
 *   out[c] = (acc + input[c]) * (1 / sqrt(T + 1))
 * This is the host causal convolution without dynamic taps.
 * The host keeps fewer than T - 1 history rows at the start of a sequence; the device keeps
 * zero rows there. Each launch advances the history in place to [raw input, old row 0, ..],
 * dropping the oldest row.
 *
 * `health` is one float: left alone, or quiet NaN when an input, a kernel value, a history
 * value or an output is not finite.
 *
 * `state` is null, or the current row of a depth stream: then `state[c] += out[c]` (the host's
 * add_current, one rounded add) follows, and a non-finite sum is reported the same way. `state`
 * shares no byte with any other range.
 *
 * Every entry takes the stream explicitly, returns INFERENCE_CUDA_INVALID_ARGUMENT for a bad
 * argument before any CUDA call, never falls back to the default stream and does not
 * synchronize. `out` is either exactly `input` or shares no byte with it; `history`, `kernel`
 * and `health` share no byte with any other range.
 */
#ifndef INFERENCE_CUDA_RESIDENT_SHORT_CONV_H
#define INFERENCE_CUDA_RESIDENT_SHORT_CONV_H

#ifdef __cplusplus
extern "C" {
#endif

int inference_cuda_short_conv(const float *input, const float *kernel, float *history,
                                float *out, float *state, float *health, int width, int taps,
                                void *stream, int *out_stage);

#ifdef __cplusplus
}
#endif

#endif /* INFERENCE_CUDA_RESIDENT_SHORT_CONV_H */
