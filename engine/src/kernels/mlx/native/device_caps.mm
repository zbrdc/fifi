#import <Metal/Metal.h>
#include <cstddef>

extern "C" std::size_t inference_mlx_max_threadgroup_memory() {
  @autoreleasepool {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (device == nil) return 0;
    const std::size_t bytes = device.maxThreadgroupMemoryLength;
    [device release];
    return bytes;
  }
}
