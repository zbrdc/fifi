#include <cstddef>
#include "mlx/stream.h"

extern "C" int inference_gpu_stream_count(std::size_t* count) noexcept {
  try {
    std::size_t result = 0;
    for (const auto& stream : mlx::core::get_streams()) {
      if (stream.device.type == mlx::core::Device::gpu && stream.device.index == 0) {
        ++result;
      }
    }
    *count = result;
    return 0;
  } catch (...) {
    return 1;
  }
}
