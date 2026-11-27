// Materialize the global convolution as taps for the relative prep. The host
// computes (x + sum(w_l * x_l)) / 3 for the 4-tap dynamic case. Local prep
// adds x and scales by 1/sqrt(5). Scale each global coefficient by sqrt(5)/3
// and fold the difference in the residual coefficient into lag zero. This is
// algebraic equality, not bit equality: real-weight row parity is required.
#include <cuda_runtime.h>
#include "device.h"
#include "abi.h"
namespace {
__global__ void effective_taps(const float *weights, const float *coeff,
                                       float *out, int width) {
    const int channel = blockIdx.x * blockDim.x + threadIdx.x;
    if (channel >= width) return;
    const int head = channel / 128;
    const float scale = sqrtf(5.0f) / 3.0f;
    for (int lag=0; lag<4; ++lag) {
        const float w = weights[lag * width + channel] + coeff[head * 4 + lag];
        out[channel * 4 + 3 - lag] = lag ? w * scale : (w + 1.f) * scale - 1.f;
    }
}
}
extern "C" int inference_cuda_effective_taps(void *const *p, int width,
    void *stream, int *out_stage) {
    effective_taps<<<(width+127)/128,128,0,(cudaStream_t)stream>>>(
        (const float *)p[0],(const float *)p[1],(float *)p[2],width);
    Failure failure;
    failure.record(cudaGetLastError(), INFERENCE_CUDA_STAGE_LAUNCH);
    report_stage(out_stage, failure.stage);
    return (int)failure.err;
}
