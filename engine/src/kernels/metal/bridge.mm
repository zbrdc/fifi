// inference_metal bridge: the device and its queue, buffers, compute pipelines and command
// buffers, which Rust opens, fills a dispatch at a time, commits and waits for. Objective-C++,
// compiled by build.rs under the feature native-metal.
//
// Handles. Every handle Rust holds is an object the bridge retains for it: a +1 under automatic
// reference counting, released once by inference_metal_object_free. Rust never retains or
// releases on its own.
//
// Exception boundary. Rust declares every entry point as extern "C", so a foreign exception
// that leaves the bridge is undefined behavior in Rust. Every entry point is noexcept and runs
// its body through guarded(), which catches NSException, C++ and other exceptions, stores the
// text in the per-thread error slot and returns kStatusException. The autorelease pool drains
// there, after the handlers ran.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>

// One handle that carries the device and the queue every launch is encoded on. Objective-C
// declarations live at global scope, outside the anonymous namespace below.
@interface FifiMetalRuntime : NSObject
@property(readonly, nonatomic) id<MTLDevice> device;
@property(readonly, nonatomic) id<MTLCommandQueue> queue;
- (instancetype)initWithDevice:(id<MTLDevice>)device queue:(id<MTLCommandQueue>)queue;
@end

@implementation FifiMetalRuntime
- (instancetype)initWithDevice:(id<MTLDevice>)device queue:(id<MTLCommandQueue>)queue {
    self = [super init];
    if (self) {
        _device = device;
        _queue = queue;
    }
    return self;
}
@end

namespace {

constexpr int kStatusOk = 0;
constexpr int kStatusInvalid = 1;
constexpr int kStatusFailed = 2;
constexpr int kStatusCommandBuffer = 3;
constexpr int kStatusNoDevice = 4;
constexpr int kStatusException = 100;

constexpr size_t kErrorCapacity = 512;
thread_local char g_last_error[kErrorCapacity] = {0};

void set_error(const char *where, const char *detail) noexcept {
    snprintf(g_last_error, kErrorCapacity, "%s: %s", where, detail ? detail : "no detail");
}

void set_error_ns(const char *where, NSString *detail) noexcept {
    set_error(where, detail ? [detail UTF8String] : nullptr);
}

id<MTLDevice> device_of(void *runtime) {
    return runtime ? ((__bridge FifiMetalRuntime *)runtime).device : nil;
}

template <typename Body>
bool contain(const char *where, Body body) noexcept {
    try {
        @try {
            body();
            return true;
        } @catch (NSException *exception) {
            set_error_ns(where, exception.reason);
        }
    } catch (const std::exception &error) {
        set_error(where, error.what());
    } catch (...) {
        set_error(where, "an exception that carries no text");
    }
    return false;
}

template <typename Body>
int guarded(const char *where, Body body) noexcept {
    g_last_error[0] = 0;
    int status = kStatusFailed;
    bool completed = false;
    @autoreleasepool {
        completed = contain(where, [&] { status = body(); });
    }
    return completed ? status : kStatusException;
}

bool valid_grid(const size_t *grid, const size_t *per_threadgroup) {
    for (int i = 0; i < 3; ++i) {
        if (grid[i] == 0 || grid[i] > UINT32_MAX || per_threadgroup[i] == 0 ||
            per_threadgroup[i] > UINT32_MAX) {
            return false;
        }
    }
    return true;
}

bool is_open(id<MTLCommandBuffer> command) {
    const MTLCommandBufferStatus status = command.status;
    return status == MTLCommandBufferStatusNotEnqueued || status == MTLCommandBufferStatusEnqueued;
}

}  // namespace

extern "C" {

int inference_metal_last_error(char *buffer, size_t capacity) noexcept {
    if (buffer == nullptr || capacity == 0) {
        return kStatusInvalid;
    }
    snprintf(buffer, capacity, "%s", g_last_error);
    return kStatusOk;
}

int inference_metal_runtime_new(void **out) noexcept {
    return guarded("runtime_new", [&]() -> int {
        if (out == nullptr) {
            return kStatusInvalid;
        }
        *out = nullptr;
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (device == nil) {
            set_error("runtime_new", "no Metal device");
            return kStatusNoDevice;
        }
        id<MTLCommandQueue> queue = [device newCommandQueue];
        if (queue == nil) {
            set_error("runtime_new", "the device gave no command queue");
            return kStatusFailed;
        }
        FifiMetalRuntime *runtime = [[FifiMetalRuntime alloc] initWithDevice:device queue:queue];
        *out = (__bridge_retained void *)runtime;
        return kStatusOk;
    });
}

void inference_metal_object_free(void *object) noexcept {
    @autoreleasepool {
        id held = (__bridge_transfer id)object;
        (void)held;
    }
}

int inference_metal_buffer_new(void *runtime, size_t bytes, void **out) noexcept {
    return guarded("buffer_new", [&]() -> int {
        id<MTLDevice> device = device_of(runtime);
        if (device == nil || out == nullptr || bytes == 0) {
            return kStatusInvalid;
        }
        *out = nullptr;
        id<MTLBuffer> buffer =
            [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
        if (buffer == nil) {
            set_error("buffer_new", "the device gave no buffer");
            return kStatusFailed;
        }
        std::memset(buffer.contents, 0, buffer.length);
        *out = (__bridge_retained void *)buffer;
        return kStatusOk;
    });
}

uint8_t *inference_metal_buffer_contents(void *buffer) noexcept {
    @autoreleasepool {
        return buffer ? static_cast<uint8_t *>(((__bridge id<MTLBuffer>)buffer).contents)
                      : nullptr;
    }
}

int inference_metal_kernel_new(void *runtime, const char *source, const char *name, void **out,
                               size_t *max_threads) noexcept {
    return guarded("kernel_new", [&]() -> int {
        id<MTLDevice> device = device_of(runtime);
        if (device == nil || source == nullptr || name == nullptr || out == nullptr ||
            max_threads == nullptr) {
            return kStatusInvalid;
        }
        *out = nullptr;
        *max_threads = 0;
        NSError *error = nil;
        // Safe math, as the CUDA adapter's resident kernels compile without fast math, and the
        // default math mode follows the SDK's language version. The math mode exists from
        // macOS 15; the Boolean before it carries the same switch.
        MTLCompileOptions *options = [[MTLCompileOptions alloc] init];
        if (@available(macOS 15.0, *)) {
            options.mathMode = MTLMathModeSafe;
        } else {
            options.fastMathEnabled = NO;
        }
        id<MTLLibrary> library = [device newLibraryWithSource:[NSString stringWithUTF8String:source]
                                                      options:options
                                                        error:&error];
        if (library == nil) {
            // The compile log names the line and column of each error; the description alone
            // says only that the build failed. The SDK declares no constant for the log's key,
            // so its documented name stands here.
            NSString *logs = error.userInfo[@"compileLogs"];
            set_error_ns("kernel_new: the shader did not compile",
                         logs.length > 0 ? logs : error.localizedDescription);
            return kStatusFailed;
        }
        id<MTLFunction> function =
            [library newFunctionWithName:[NSString stringWithUTF8String:name]];
        if (function == nil) {
            set_error("kernel_new", "no function of that name in the shader");
            return kStatusInvalid;
        }
        id<MTLComputePipelineState> pipeline =
            [device newComputePipelineStateWithFunction:function error:&error];
        if (pipeline == nil) {
            set_error_ns("kernel_new: the pipeline did not build", error.localizedDescription);
            return kStatusFailed;
        }
        *out = (__bridge_retained void *)pipeline;
        *max_threads = static_cast<size_t>(pipeline.maxTotalThreadsPerThreadgroup);
        return kStatusOk;
    });
}

int inference_metal_command_new(void *runtime, void **out) noexcept {
    return guarded("command_new", [&]() -> int {
        FifiMetalRuntime *holder = (__bridge FifiMetalRuntime *)runtime;
        if (holder == nil || holder.queue == nil || out == nullptr) {
            return kStatusInvalid;
        }
        *out = nullptr;
        id<MTLCommandBuffer> command = [holder.queue commandBuffer];
        if (command == nil) {
            set_error("command_new", "the queue gave no command buffer");
            return kStatusFailed;
        }
        *out = (__bridge_retained void *)command;
        return kStatusOk;
    });
}

// One dispatch ends its own compute encoder, so a blit encoder can go between any two. A
// command buffer takes no dispatch after its commit.
int inference_metal_command_dispatch(void *held, void *kernel, void *const *buffers,
                                     const size_t *offsets, size_t count, const void *bytes,
                                     size_t bytes_len, const size_t *grid,
                                     const size_t *per_threadgroup, size_t repeats) noexcept {
    return guarded("command_dispatch", [&]() -> int {
        id<MTLCommandBuffer> command = (__bridge id<MTLCommandBuffer>)held;
        id<MTLComputePipelineState> pipeline = (__bridge id<MTLComputePipelineState>)kernel;
        if (command == nil || pipeline == nil || buffers == nullptr || offsets == nullptr ||
            count == 0 || count > 30 || bytes == nullptr || bytes_len == 0 || bytes_len > 4096 ||
            grid == nullptr || per_threadgroup == nullptr || repeats == 0 ||
            repeats > (1u << 20) || !valid_grid(grid, per_threadgroup)) {
            return kStatusInvalid;
        }
        const size_t group = per_threadgroup[0] * per_threadgroup[1] * per_threadgroup[2];
        if (group > static_cast<size_t>(pipeline.maxTotalThreadsPerThreadgroup)) {
            return kStatusInvalid;
        }
        for (size_t i = 0; i < count; ++i) {
            if (buffers[i] == nullptr || offsets[i] % 4 != 0) {
                return kStatusInvalid;
            }
        }
        if (!is_open(command)) {
            set_error("command_dispatch", "the command buffer is committed: it takes no dispatch");
            return kStatusInvalid;
        }
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        for (size_t i = 0; i < count; ++i) {
            [encoder setBuffer:(__bridge id<MTLBuffer>)buffers[i] offset:offsets[i] atIndex:i];
        }
        [encoder setBytes:bytes length:bytes_len atIndex:count];
        const MTLSize threads = MTLSizeMake(grid[0], grid[1], grid[2]);
        const MTLSize per_group =
            MTLSizeMake(per_threadgroup[0], per_threadgroup[1], per_threadgroup[2]);
        for (size_t r = 0; r < repeats; ++r) {
            [encoder dispatchThreads:threads threadsPerThreadgroup:per_group];
        }
        [encoder endEncoding];
        return kStatusOk;
    });
}

// Encodes a fill of `bytes` bytes with zero at byte `offset` of the buffer, behind the work
// encoded before it. A blit encoder, between the compute encoders of the dispatches around it.
int inference_metal_command_fill(void *held, void *buffer, size_t offset, size_t bytes) noexcept {
    return guarded("command_fill", [&]() -> int {
        id<MTLCommandBuffer> command = (__bridge id<MTLCommandBuffer>)held;
        id<MTLBuffer> target = (__bridge id<MTLBuffer>)buffer;
        if (command == nil || target == nil || bytes == 0) {
            return kStatusInvalid;
        }
        if (offset > target.length || bytes > target.length - offset) {
            return kStatusInvalid;
        }
        if (!is_open(command)) {
            set_error("command_fill", "the command buffer is committed: it takes no fill");
            return kStatusInvalid;
        }
        id<MTLBlitCommandEncoder> encoder = [command blitCommandEncoder];
        [encoder fillBuffer:target range:NSMakeRange(offset, bytes) value:0];
        [encoder endEncoding];
        return kStatusOk;
    });
}

int inference_metal_command_commit(void *held) noexcept {
    return guarded("command_commit", [&]() -> int {
        id<MTLCommandBuffer> command = (__bridge id<MTLCommandBuffer>)held;
        if (command == nil) {
            return kStatusInvalid;
        }
        if (!is_open(command)) {
            set_error("command_commit", "the command buffer is committed already");
            return kStatusInvalid;
        }
        [command commit];
        return kStatusOk;
    });
}

// Returns the GPU time of the command buffer in nanoseconds, 0 when the device reports none.
int inference_metal_command_wait(void *held, uint64_t *gpu_ns) noexcept {
    return guarded("command_wait", [&]() -> int {
        id<MTLCommandBuffer> command = (__bridge id<MTLCommandBuffer>)held;
        if (command == nil || gpu_ns == nullptr) {
            return kStatusInvalid;
        }
        if (is_open(command)) {
            set_error("command_wait", "the command buffer is not committed");
            return kStatusInvalid;
        }
        [command waitUntilCompleted];
        if (command.status != MTLCommandBufferStatusCompleted) {
            set_error_ns("command_wait: the command buffer did not complete",
                         command.error.localizedDescription);
            return kStatusCommandBuffer;
        }
        const CFTimeInterval start = command.GPUStartTime;
        const CFTimeInterval end = command.GPUEndTime;
        *gpu_ns = end > start ? static_cast<uint64_t>((end - start) * 1e9) : 0;
        return kStatusOk;
    });
}

}  // extern "C"
