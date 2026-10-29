// ane_bridge.mm — ANE decision-head bridge.
// Private AppleNeuralEngine APIs via objc_msgSend. One static MIL program per
// handle built only from numerically verified ops (matmul, bias add,
// sigmoid+mul SiLU), with these layout rules:
//   1. CONST tensors live in ONE FILE PER TENSOR (w1.bin, b1.bin, w2.bin,
//      b2.bin) — the pattern the probe's training pipeline used; a single
//      multi-chunk blob scrambles consts after the first.
//   2. The ANE reads each const's width axis circularly shifted by +64
//      (read col = (written col + 64) mod width); weights are pre-shifted by
//      -64 on write so the net result matches the CPU reference.
//   3. Input surface rows map to matmul rows as identity (channel-major);
//      output is read channel-major with stride S.
// Input is padded to spatial 32 (measured ANE minimum). Host-side softmax.
//
// Exception boundary. Rust declares every entry point as extern "C", so a
// foreign exception that leaves the bridge is undefined behavior in Rust.
// Every entry point is noexcept. The three working entry points run their
// body through guarded(), and ane_bridge_last_error only copies the error
// slot. The guard drains the autorelease pool after the handlers run, catches
// NSException, C++ and other exceptions, stores the text in the per-thread
// error slot, and returns kStatusException. Private selectors can vanish in a
// macOS update, and then objc_msgSend raises NSInvalidArgumentException.
// ensure_ane() checks every selector before first use.
// Build note: Objective-C++ compiles with -fobjc-arc-exceptions by default, so
// strong locals release when an exception unwinds. A clang++ -fobjc-arc probe
// and the fake-framework harness confirmed this. Plain Objective-C code does
// not do this and would leak on the exception paths.
#import <Foundation/Foundation.h>
#import <objc/runtime.h>
#import <objc/message.h>
#import <dlfcn.h>
#import <IOSurface/IOSurface.h>

#include <cerrno>
#include <chrono>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <memory>
#include <mutex>

#include <fcntl.h>
#include <limits.h>
#include <sys/file.h>
#include <unistd.h>

namespace {

// Status returned when a guarded body raised an exception.
constexpr int kStatusException = 100;

// How long a compile waits for another process that compiles the same model.
constexpr std::chrono::milliseconds kCompileLockTimeout{120000};
// Cleanup must finish, so it waits only briefly before it proceeds unlocked.
constexpr std::chrono::milliseconds kFreeLockTimeout{30000};

std::mutex g_bridge_mutex;

// Per-thread error text. A failed call and the caller's later retrieval stay
// associated even when another thread runs an operation in between.
const size_t kErrorCapacity = 512;
thread_local char g_last_error[kErrorCapacity] = {0};

void set_error_text(const char *message) noexcept {
    snprintf(g_last_error, kErrorCapacity, "%s", message ? message : "unknown error");
}

void set_error_detail(const char *where, const char *kind, const char *detail) noexcept {
    snprintf(g_last_error, kErrorCapacity, "%s: %s: %s", where, kind,
             detail ? detail : "no detail");
}

void record_objc_exception(const char *where, NSException *exception) noexcept {
    @try {
        const char *name = [[exception name] UTF8String];
        const char *reason = [[exception reason] UTF8String];
        snprintf(g_last_error, kErrorCapacity, "%s: Objective-C exception %s: %s", where,
                 name ? name : "unnamed", reason ? reason : "no reason");
    } @catch (...) {
        set_error_detail(where, "Objective-C exception", "description unreadable");
    }
}

// Runs body and converts every exception to a false result. Objective-C
// exceptions and C++ exceptions share one unwinder in a 64-bit process, so
// the C++ handlers also catch an Objective-C object that is not an
// NSException. The handlers store the text in the error slot when record is
// true. This function never throws.
template <typename Body>
bool contain(const char *where, bool record, Body body) noexcept {
    bool completed = false;
    try {
        @try {
            body();
            completed = true;
        } @catch (NSException *exception) {
            if (record) record_objc_exception(where, exception);
        }
    } catch (const std::exception &error) {
        if (record) set_error_detail(where, "C++ exception", error.what());
    } catch (...) {
        if (record) set_error_detail(where, "native exception", "unknown exception type");
    }
    return completed;
}

// The exception boundary of every bridge entry point. The body returns a
// status. An exception that escapes the body becomes kStatusException. The
// autorelease pool drains here, after the handlers ran, so no exception
// unwinds through a pool. The error slot starts empty for each call, so a
// failure never reports text from an earlier call.
template <typename Body>
int guarded(const char *where, Body body) noexcept {
    int status = kStatusException;
    @autoreleasepool {
        g_last_error[0] = '\0';
        if (!contain(where, true, [&] { status = body(); })) status = kStatusException;
    }
    return status;
}

// Runs body in its own autorelease pool and exception handlers, for a caller
// that holds g_bridge_mutex. The pool drains before that mutex goes, so an
// autoreleased framework object never frees while another handle runs. This
// function never throws. If the body throws, the handlers run inside the pool.
template <typename Body>
int pooled(const char *where, Body body) noexcept {
    int status = kStatusException;
    @autoreleasepool {
        if (!contain(where, true, [&] { status = body(); })) status = kStatusException;
    }
    return status;
}

Class cls(const char *name) { return NSClassFromString(@(name)); }

IOSurfaceRef make_surface(size_t bytes) {
    return IOSurfaceCreate((__bridge CFDictionaryRef)@{
        (id)kIOSurfaceWidth : @(bytes), (id)kIOSurfaceHeight : @1,
        (id)kIOSurfaceBytesPerElement : @1, (id)kIOSurfaceBytesPerRow : @(bytes),
        (id)kIOSurfaceAllocSize : @(bytes), (id)kIOSurfacePixelFormat : @0});
}

void set_error(const char *where, NSError *e) {
    NSString *description = e ? [e description] : @"unknown error";
    snprintf(g_last_error, kErrorCapacity, "%s: %s", where, [description UTF8String]);
}

struct FreeDeleter {
    void operator()(void *pointer) const noexcept { free(pointer); }
};
using HalfBuffer = std::unique_ptr<_Float16[], FreeDeleter>;

// const file with one chunk: header + values; optional per-row column shift
// (weights: identity; biases: -64 mod width so the add lines up on STORED channels)
NSData *const_file_shift(const _Float16 *values, size_t n, int width, int shift) {
    if (values == NULL || width <= 0 || n == 0 || n % (size_t)width != 0 ||
        shift <= -width || shift >= width) {
        set_error_text("decision head const packing received an invalid shape");
        return nil;
    }
    NSMutableData *data = [NSMutableData dataWithLength:64 + 64 + n * 2];
    if (data == nil) {
        set_error_text("decision head const allocation failed");
        return nil;
    }
    uint8_t *buf = (uint8_t *)[data mutableBytes];
    buf[0] = 0x01;
    buf[4] = 0x02;
    uint8_t *ch = buf + 64;
    ch[0] = 0xEF; ch[1] = 0xBE; ch[2] = 0xAD; ch[3] = 0xDE; ch[4] = 0x01; ch[10] = 0x08;
    _Float16 *dst = (_Float16 *)(ch + 64);
    for (size_t row = 0; row + width <= n; row += width) {
        for (int c = 0; c < width; c++) {
            int col = (c + shift) % width;
            if (col < 0) col += width;
            dst[row + (size_t)c] = values[row + (size_t)col];
        }
    }
    return (NSData *)data;
}
NSData *const_file(const _Float16 *values, size_t n, int width) {
    return const_file_shift(values, n, width, 0);
}

// padded second-layer width: next multiple of 32 at/after outputs (measured
// reshape aliasing corrupts surface channels at 260 % 32)
int out_w(int outputs) { return ((outputs + 64 + 31) / 32) * 32; }

NSString *mil_for(int input, int hidden, int outputs) {
    NSMutableString *m = [NSMutableString string];
    [m appendString:@"program(1.3)\n[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, {\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, {\"coremltools-version\", \"9.0\"}})]\n{\n"];
    [m appendFormat:@"    func main<ios18>(tensor<fp16, [1, %d, 1, 32]> x) {\n", input];
    [m appendString:@"        tensor<int32, [4]> r1 = const()[name=string(\"r1\"), val=tensor<int32, [4]>([1, 1, 1, 1])];\n"];
    [m appendString:@"        bool bF = const()[name=string(\"bF\"), val=bool(false)];\n"];
    [m appendString:@"        tensor<int32, [4]> pm = const()[name=string(\"pm\"), val=tensor<int32, [4]>([0, 1, 3, 2])];\n"];
    [m appendFormat:@"        tensor<int32, [4]> ra = const()[name=string(\"ra\"), val=tensor<int32, [4]>([1, 1, %d, 32])];\n", input];
    [m appendFormat:@"        tensor<fp16, [1, 1, %d, 32]> xr = reshape(shape=ra, x=x)[name=string(\"xr\")];\n", input];
    [m appendFormat:@"        tensor<fp16, [1, 1, 32, %d]> xnt = transpose(perm=pm, x=xr)[name=string(\"xnt\")];\n", input];
    [m appendFormat:@"        tensor<fp16, [1, 1, %d, %d]> W1c = const()[name=string(\"W1c\"), val=tensor<fp16, [1, 1, %d, %d]>(BLOBFILE(path=string(\"@model_path/weights/w1.bin\"), offset=uint64(64)))];\n", input, hidden, input, hidden];
    [m appendFormat:@"        tensor<fp16, [1, 1, 32, %d]> h1m = matmul(transpose_x=bF, transpose_y=bF, x=xnt, y=W1c)[name=string(\"h1m\")];\n", hidden];
    [m appendFormat:@"        tensor<fp16, [1, 1, %d, 32]> h1t = transpose(perm=pm, x=h1m)[name=string(\"h1t\")];\n", hidden];
    [m appendFormat:@"        tensor<int32, [4]> r2 = const()[name=string(\"r2\"), val=tensor<int32, [4]>([1, %d, 1, 32])];\n", hidden];
    [m appendFormat:@"        tensor<fp16, [1, %d, 1, 32]> h1f = reshape(shape=r2, x=h1t)[name=string(\"h1f\")];\n", hidden];
    [m appendFormat:@"        tensor<fp16, [1, %d, 1, 1]> b1 = const()[name=string(\"b1\"), val=tensor<fp16, [1, %d, 1, 1]>(BLOBFILE(path=string(\"@model_path/weights/b1.bin\"), offset=uint64(64)))];\n", hidden, hidden];
    [m appendFormat:@"        tensor<fp16, [1, %d, 1, 32]> hb = add(x=h1f, y=b1)[name=string(\"hb\")];\n", hidden];
    [m appendFormat:@"        tensor<fp16, [1, %d, 1, 32]> sg = sigmoid(x=hb)[name=string(\"sg\")];\n", hidden];
    [m appendFormat:@"        tensor<fp16, [1, %d, 1, 32]> si = mul(x=hb, y=sg)[name=string(\"si\")];\n", hidden];
    [m appendFormat:@"        tensor<int32, [4]> r3 = const()[name=string(\"r3\"), val=tensor<int32, [4]>([1, 1, %d, 32])];\n", hidden];
    [m appendFormat:@"        tensor<fp16, [1, 1, %d, 32]> sir = reshape(shape=r3, x=si)[name=string(\"sir\")];\n", hidden];
    [m appendFormat:@"        tensor<fp16, [1, 1, 32, %d]> sint = transpose(perm=pm, x=sir)[name=string(\"sint\")];\n", hidden];
    [m appendFormat:@"        tensor<fp16, [1, 1, %d, %d]> W2c = const()[name=string(\"W2c\"), val=tensor<fp16, [1, 1, %d, %d]>(BLOBFILE(path=string(\"@model_path/weights/w2.bin\"), offset=uint64(64)))];\n", hidden, out_w(outputs), hidden, out_w(outputs)];
    [m appendFormat:@"        tensor<fp16, [1, 1, 32, %d]> o2m = matmul(transpose_x=bF, transpose_y=bF, x=sint, y=W2c)[name=string(\"o2m\")];\n", out_w(outputs)];
    [m appendFormat:@"        tensor<fp16, [1, 1, %d, 32]> o2t = transpose(perm=pm, x=o2m)[name=string(\"o2t\")];\n", out_w(outputs)];
    [m appendFormat:@"        tensor<int32, [4]> r4 = const()[name=string(\"r4\"), val=tensor<int32, [4]>([1, %d, 1, 32])];\n", out_w(outputs)];
    [m appendFormat:@"        tensor<fp16, [1, %d, 1, 32]> o2f = reshape(shape=r4, x=o2t)[name=string(\"o2f\")];\n", out_w(outputs)];
    [m appendFormat:@"        tensor<fp16, [1, %d, 1, 1]> b2 = const()[name=string(\"b2\"), val=tensor<fp16, [1, %d, 1, 1]>(BLOBFILE(path=string(\"@model_path/weights/b2.bin\"), offset=uint64(64)))];\n", out_w(outputs), out_w(outputs)];
    [m appendFormat:@"        tensor<fp16, [1, %d, 1, 32]> lb = add(x=o2f, y=b2)[name=string(\"lb\")];\n", out_w(outputs)];
    [m appendString:@"    } -> (lb);\n}\n"];
    return [m copy];
}


// Owns one retained IOSurface until release().
class SurfaceRef {
  public:
    SurfaceRef() = default;
    explicit SurfaceRef(IOSurfaceRef ref) : ref_(ref) {}
    SurfaceRef(const SurfaceRef &) = delete;
    SurfaceRef &operator=(const SurfaceRef &) = delete;
    ~SurfaceRef() {
        if (ref_) CFRelease(ref_);
    }
    IOSurfaceRef get() const { return ref_; }
    IOSurfaceRef release() {
        IOSurfaceRef ref = ref_;
        ref_ = nullptr;
        return ref;
    }

  private:
    IOSurfaceRef ref_ = nullptr;
};

// One CPU lock on an IOSurface. The constructor records the lock status. The
// destructor unlocks a held surface, so an exception cannot leave it locked.
class SurfaceLock {
  public:
    SurfaceLock(IOSurfaceRef surface, IOSurfaceLockOptions options) noexcept
        : surface_(surface),
          options_(options),
          status_(IOSurfaceLock(surface, options, NULL)),
          held_(status_ == KERN_SUCCESS) {}
    SurfaceLock(const SurfaceLock &) = delete;
    SurfaceLock &operator=(const SurfaceLock &) = delete;
    ~SurfaceLock() {
        if (held_) IOSurfaceUnlock(surface_, options_, NULL);
    }
    bool held() const { return held_; }
    kern_return_t lock_status() const { return status_; }
    // Unlocks now and reports the unlock status.
    kern_return_t unlock() noexcept {
        if (!held_) return KERN_SUCCESS;
        held_ = false;
        return IOSurfaceUnlock(surface_, options_, NULL);
    }

  private:
    IOSurfaceRef surface_;
    IOSurfaceLockOptions options_;
    kern_return_t status_;
    bool held_;
};

void set_surface_error(const char *operation, const char *surface, kern_return_t status) noexcept {
    snprintf(g_last_error, kErrorCapacity, "%s failed on the %s surface with status %#x",
             operation, surface, (unsigned)status);
}

// Cross-process exclusion for the model temporary directory. The private API
// reads the model files from NSTemporaryDirectory()/<hexStringIdentifier>, a
// path it derives from the model content. Two processes that compile the same
// model use the same directory, and one process could remove it during the
// other's compile. A flock on one shared file serializes that window.
// This is a precaution. Only a fake framework has reproduced the race.
//
// Lock order. A call that needs both locks takes this file lock first and
// g_bridge_mutex second, and releases them in the reverse order. A wait for
// another process therefore never blocks predict on other handles.
class ProcessLock {
  public:
    ProcessLock() = default;
    ProcessLock(const ProcessLock &) = delete;
    ProcessLock &operator=(const ProcessLock &) = delete;
    ~ProcessLock() { release(); }

    // Takes the lock or stores the reason in the error slot. This function
    // never throws: the Foundation call that builds the path runs inside
    // contain().
    bool acquire(std::chrono::milliseconds timeout) noexcept {
        char path[PATH_MAX] = {0};
        // The caller can be a Rust thread with no autorelease pool, so the
        // Foundation work gets its own pool. The path is copied out before it ends.
        const bool found = contain("ANE bridge lock path", true, [&] {
            @autoreleasepool {
                NSString *file = [NSTemporaryDirectory()
                    stringByAppendingPathComponent:@"inference-engine-ane-bridge.lock"];
                snprintf(path, sizeof(path), "%s", [file fileSystemRepresentation]);
            }
        });
        if (!found) return false;
        fd_ = open(path, O_RDWR | O_CREAT | O_CLOEXEC | O_NOFOLLOW, 0600);
        if (fd_ < 0) {
            snprintf(g_last_error, kErrorCapacity, "ANE bridge lock file open failed: %s",
                     strerror(errno));
            return false;
        }
        const auto deadline = std::chrono::steady_clock::now() + timeout;
        for (;;) {
            if (flock(fd_, LOCK_EX | LOCK_NB) == 0) return true;
            const int reason = errno;
            if (reason != EWOULDBLOCK && reason != EINTR) {
                snprintf(g_last_error, kErrorCapacity, "ANE bridge lock failed: %s",
                         strerror(reason));
                release();
                return false;
            }
            if (std::chrono::steady_clock::now() >= deadline) {
                set_error_text("timed out waiting for another process that compiles an ANE model");
                release();
                return false;
            }
            usleep(20 * 1000);
        }
    }

    void release() noexcept {
        if (fd_ >= 0) {
            flock(fd_, LOCK_UN);
            close(fd_);
            fd_ = -1;
        }
    }

  private:
    int fd_ = -1;
};

// Unloads a model and swallows every exception. record keeps the text of an
// exception in the error slot. Cleanup after another failure passes false so
// the first failure stays visible.
bool unload_model(id model, bool record) noexcept {
    return contain("ANE model unload", record, [&] {
        NSError *unused = nil;
        ((BOOL(*)(id, SEL, unsigned int, NSError **))objc_msgSend)(
            model, @selector(unloadWithQoS:error:), 21, &unused);
    });
}

// Unloads the model when a failing path leaves the scope.
class ModelLoad {
  public:
    explicit ModelLoad(id model) : model_(model) {}
    ModelLoad(const ModelLoad &) = delete;
    ModelLoad &operator=(const ModelLoad &) = delete;
    ~ModelLoad() {
        if (armed_) unload_model(model_, false);
    }
    void disarm() { armed_ = false; }

  private:
    id model_;
    bool armed_ = true;
};

// Removes the model directory when the scope ends, or earlier on request.
class TempModelDir {
  public:
    explicit TempModelDir(NSString *path) : path_(path) {}
    TempModelDir(const TempModelDir &) = delete;
    TempModelDir &operator=(const TempModelDir &) = delete;
    ~TempModelDir() { remove(); }
    void remove() noexcept {
        if (path_ == nil) return;
        NSString *path = path_;
        path_ = nil;
        contain("ANE temporary directory removal", false, [&] {
            [[NSFileManager defaultManager] removeItemAtPath:path error:nil];
        });
    }

  private:
    NSString *path_;
};

bool release_quietly(void *object) noexcept {
    return contain("ANE object release", false, [&] { CFRelease(object); });
}

bool write_model_files(NSString *dir, NSData *mil, NSData *w1, NSData *b1, NSData *w2,
                       NSData *b2) {
    NSFileManager *fm = [NSFileManager defaultManager];
    NSError *e = nil;
    if (![fm createDirectoryAtPath:[dir stringByAppendingPathComponent:@"weights"]
            withIntermediateDirectories:YES
                             attributes:@{NSFilePosixPermissions : @0700}
                                  error:&e]) {
        set_error("model directory create", e);
        return false;
    }
    NSArray<NSString *> *names = @[
        @"model.mil", @"weights/w1.bin", @"weights/b1.bin", @"weights/w2.bin", @"weights/b2.bin"
    ];
    NSArray<NSData *> *contents = @[ mil, w1, b1, w2, b2 ];
    for (NSUInteger i = 0; i < names.count; i++) {
        if (![contents[i] writeToFile:[dir stringByAppendingPathComponent:names[i]]
                              options:NSDataWritingAtomic
                                error:&e]) {
            set_error("model file write", e);
            return false;
        }
    }
    return true;
}


typedef struct {
    void *model;    // CFBridgingRetain ownership
    void *request;  // CFBridgingRetain ownership
    IOSurfaceRef in_surface;
    IOSurfaceRef out_surface;
    int input;
    int hidden;
    int experts;
    int classes;
    int outputs;
} Aned;

// Releases every object that a handle owns, then the handle. It accepts a
// zero-filled handle and never throws, so any exit path releases the model.
struct AnedDeleter {
    void operator()(Aned *h) const noexcept {
        if (!h) return;
        if (h->model != NULL) release_quietly(h->model);
        if (h->request != NULL) release_quietly(h->request);
        if (h->in_surface != NULL) CFRelease(h->in_surface);
        if (h->out_surface != NULL) CFRelease(h->out_surface);
        free(h);
    }
};
using AnedHandle = std::unique_ptr<Aned, AnedDeleter>;

Class g_desc, g_inmem, g_request, g_io;
bool g_loaded = false;

// One private method that the bridge calls through objc_msgSend.
struct RequiredSelector {
    Class owner;
    const char *class_name;
    SEL selector;
    bool class_method;
};

// Checks the private classes and every selector before the first use. A
// renamed selector would otherwise raise NSInvalidArgumentException at the
// objc_msgSend call. The caller holds g_bridge_mutex.
bool ensure_ane(void) {
    if (g_loaded) return true;
    void *framework = dlopen(
        "/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine",
        RTLD_NOW);
    const char *load_error = framework ? NULL : dlerror();
    g_desc = cls("_ANEInMemoryModelDescriptor");
    g_inmem = cls("_ANEInMemoryModel");
    g_request = cls("_ANERequest");
    g_io = cls("_ANEIOSurfaceObject");
    if (!(g_desc && g_inmem && g_request && g_io)) {
        if (load_error) {
            set_error_detail("ANE private classes unavailable", "framework load failed",
                             load_error);
        } else {
            set_error_text("ANE private classes unavailable");
        }
        return false;
    }
    const RequiredSelector required[] = {
        {g_desc, "_ANEInMemoryModelDescriptor", @selector(modelWithMILText:weights:optionsPlist:),
         true},
        {g_inmem, "_ANEInMemoryModel", @selector(inMemoryModelWithDescriptor:), true},
        {g_inmem, "_ANEInMemoryModel", @selector(hexStringIdentifier), false},
        {g_inmem, "_ANEInMemoryModel", @selector(unloadWithQoS:error:), false},
        {g_inmem, "_ANEInMemoryModel", @selector(compileWithQoS:options:error:), false},
        {g_inmem, "_ANEInMemoryModel", @selector(loadWithQoS:options:error:), false},
        {g_inmem, "_ANEInMemoryModel", @selector(evaluateWithQoS:options:request:error:), false},
        {g_request, "_ANERequest",
         @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
         true},
        {g_io, "_ANEIOSurfaceObject", @selector(objectWithIOSurface:), true},
    };
    for (const RequiredSelector &item : required) {
        const bool available = item.class_method
                                   ? [item.owner respondsToSelector:item.selector]
                                   : [item.owner instancesRespondToSelector:item.selector];
        if (!available) {
            snprintf(g_last_error, kErrorCapacity,
                     "ANE private selector unavailable: %c[%s %s]", item.class_method ? '+' : '-',
                     item.class_name, sel_getName(item.selector));
            return false;
        }
    }
    g_loaded = true;
    return true;
}

// Argument checks that need no global state. They run before the file lock.
int check_new_args(int input, int hidden, int experts, int classes, const float *w1,
                   const float *b1, const float *w2, const float *b2, void **out) {
    if (out == NULL) {
        set_error_text("decision head output handle pointer is null");
        return 1;
    }
    *out = NULL;
    if (input < 1 || hidden < 1 || experts < 1 || classes < 1) {
        set_error_text("decision head dims must be positive");
        return 1;
    }
    if (w1 == NULL || b1 == NULL || w2 == NULL || b2 == NULL) {
        set_error_text("decision head weights must not be null");
        return 1;
    }
    if (hidden < 256) {
        set_error_text("decision head hidden must be at least 256");
        return 8;
    }
    long total_outputs = (long)experts + (long)classes;
    if (total_outputs > 4096) {
        set_error_text("decision head supports at most 4096 outputs");
        return 7;
    }
    return 0;
}

// The caller holds the file lock and g_bridge_mutex, and has checked the
// arguments. new_handle runs this body in a pool inside the mutex, and the
// file lock outlives that pool. The model and the other locals therefore
// release, and the pool drains, before either lock goes.
int new_body(int input, int hidden, int experts, int classes, const float *w1, const float *b1,
             const float *w2, const float *b2, void **out) {
    if (!ensure_ane()) return 2;
    int outputs = experts + classes;
    // Build fp16 const tensors in matmul layout:
    //   W1c[i][h] = w1[h][i] (transpose of the conv orientation);
    //   W2c[h][o] = w2[o][h].
    NSUInteger w1n = (NSUInteger)input * hidden;
    NSUInteger op = (NSUInteger)out_w(outputs);
    NSUInteger w2n = (NSUInteger)hidden * op;
    HalfBuffer f_w1((_Float16 *)calloc(w1n, sizeof(_Float16)));
    HalfBuffer f_b1((_Float16 *)calloc((NSUInteger)hidden, sizeof(_Float16)));
    HalfBuffer f_w2((_Float16 *)calloc(w2n, sizeof(_Float16)));
    HalfBuffer f_b2((_Float16 *)calloc(op, sizeof(_Float16)));
    if (!f_w1 || !f_b1 || !f_w2 || !f_b2) {
        set_error_text("decision head allocation failed");
        return 9;
    }
    for (int h = 0; h < hidden; h++)
        for (int i = 0; i < input; i++)
            f_w1[(size_t)i * hidden + h] = (_Float16)w1[(size_t)h * input + i];
    for (int h = 0; h < hidden; h++) f_b1[h] = (_Float16)b1[h];
    // ANE pairing: the second matmul pairs the first matmul's STORED output
    // (shifted +64 mod hidden) with W2c row h, so W2c rows are pre-shifted
    // by +64: written row rr holds logical row (rr - 64) mod hidden.
    for (int o = 0; o < outputs; o++) {
        for (int rr = 0; rr < hidden; rr++) {
            int src_row = (rr - 64) % hidden;
            if (src_row < 0) src_row += hidden;
            f_w2[(size_t)rr * op + o] = (_Float16)w2[(size_t)o * hidden + (size_t)src_row];
        }
    }
    // Biases are not pre-shifted. They add on stored channels, and the
    // measured parity test with nonzero biases confirms this alignment.
    for (int o = 0; o < outputs; o++) f_b2[o] = (_Float16)b2[o];
    NSData *b_w1 = const_file(f_w1.get(), w1n, hidden);
    NSData *b_b1 = const_file(f_b1.get(), (NSUInteger)hidden, hidden);
    NSData *b_w2 = const_file(f_w2.get(), w2n, (int)op);
    NSData *b_b2 = const_file(f_b2.get(), op, (int)op);
    if (b_w1 == nil || b_b1 == nil || b_w2 == nil || b_b2 == nil) return 10;
    f_w1.reset();
    f_b1.reset();
    f_w2.reset();
    f_b2.reset();
    NSError *e = nil;
    NSData *mil = [mil_for(input, hidden, outputs) dataUsingEncoding:NSUTF8StringEncoding];
    if (mil == nil) {
        set_error_text("ANE program text encoding failed");
        return 3;
    }
    NSDictionary *weights = @{
        @"@model_path/weights/w1.bin" : @{@"offset" : @0, @"data" : b_w1},
        @"@model_path/weights/b1.bin" : @{@"offset" : @0, @"data" : b_b1},
        @"@model_path/weights/w2.bin" : @{@"offset" : @0, @"data" : b_w2},
        @"@model_path/weights/b2.bin" : @{@"offset" : @0, @"data" : b_b2},
    };
    id desc = ((id(*)(Class, SEL, id, id, id))objc_msgSend)(
        g_desc, @selector(modelWithMILText:weights:optionsPlist:), mil, weights, nil);
    if (!desc) {
        set_error_text("ANE descriptor nil");
        return 3;
    }
    id mdl = ((id(*)(Class, SEL, id))objc_msgSend)(
        g_inmem, @selector(inMemoryModelWithDescriptor:), desc);
    if (mdl == nil) {
        set_error_text("ANE model creation failed");
        return 12;
    }
    ModelLoad model_load(mdl);
    id hx = ((id(*)(id, SEL))objc_msgSend)(mdl, @selector(hexStringIdentifier));
    if (hx == nil) {
        set_error_text("ANE model identifier unavailable");
        return 13;
    }
    // The identifier becomes one path component. Reject anything else.
    if (![hx isKindOfClass:[NSString class]] || [hx length] == 0 ||
        [hx hasPrefix:@"."] || [hx rangeOfString:@"/"].location != NSNotFound) {
        set_error_text("ANE model identifier is not a safe path component");
        return 13;
    }
    NSString *model_path = [NSTemporaryDirectory() stringByAppendingPathComponent:hx];
    TempModelDir model_dir(model_path);
    if (!write_model_files(model_path, mil, b_w1, b_b1, b_w2, b_b2)) return 16;
    if (!((BOOL(*)(id, SEL, unsigned int, id, NSError **))objc_msgSend)(
            mdl, @selector(compileWithQoS:options:error:), 21, @{}, &e)) {
        set_error("compile", e);
        return 4;
    }
    if (!((BOOL(*)(id, SEL, unsigned int, id, NSError **))objc_msgSend)(
            mdl, @selector(loadWithQoS:options:error:), 21, @{}, &e)) {
        set_error("load", e);
        return 5;
    }
    model_dir.remove();
    size_t in_bytes = (size_t)input * 32 * 2;
    size_t out_bytes = (size_t)out_w(outputs) * 32 * 2;
    SurfaceRef in_s(make_surface(in_bytes));
    SurfaceRef out_s(make_surface(out_bytes));
    if (!in_s.get() || !out_s.get()) {
        set_error_text("IOSurface create failed");
        return 6;
    }
    if (IOSurfaceGetAllocSize(in_s.get()) < in_bytes ||
        IOSurfaceGetAllocSize(out_s.get()) < out_bytes) {
        set_error_text("IOSurface allocation is smaller than requested");
        return 18;
    }
    id wI = ((id(*)(Class, SEL, IOSurfaceRef))objc_msgSend)(
        g_io, @selector(objectWithIOSurface:), in_s.get());
    id wO = ((id(*)(Class, SEL, IOSurfaceRef))objc_msgSend)(
        g_io, @selector(objectWithIOSurface:), out_s.get());
    if (wI == nil || wO == nil) {
        set_error_text("ANE surface object creation failed");
        return 14;
    }
    id req = ((id(*)(Class, SEL, id, id, id, id, id, id, id))objc_msgSend)(
        g_request,
        @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
        @[ wI ], @[ @0 ], @[ wO ], @[ @0 ], nil, nil, @0);
    if (req == nil) {
        set_error_text("ANE request creation failed");
        return 15;
    }
    AnedHandle h((Aned *)calloc(1, sizeof(Aned)));
    if (!h) {
        set_error_text("decision head handle allocation failed");
        return 11;
    }
    h->model = (void *)CFBridgingRetain(mdl);
    h->request = (void *)CFBridgingRetain(req);
    h->in_surface = in_s.release();
    h->out_surface = out_s.release();
    h->input = input;
    h->hidden = hidden;
    h->experts = experts;
    h->classes = classes;
    h->outputs = outputs;
    model_load.disarm();
    *out = h.release();
    return 0;
}

int predict_body(void *vp, const float *features, size_t feature_n, float *expert_out,
                 size_t expert_n, float *class_out, size_t class_n) {
    Aned *h = (Aned *)vp;
    if (!h) {
        set_error_text("decision head handle is null");
        return 1;
    }
    if (feature_n != (size_t)h->input) {
        set_error_text("feature count does not match the compiled input");
        return 2;
    }
    if (expert_n != (size_t)h->experts || class_n != (size_t)h->classes) {
        set_error_text("output count does not match the compiled head");
        return 3;
    }
    if (features == NULL || expert_out == NULL || class_out == NULL) {
        set_error_text("decision head buffers must not be null");
        return 3;
    }
    {
        SurfaceLock in_lock(h->in_surface, (IOSurfaceLockOptions)0);
        if (!in_lock.held()) {
            set_surface_error("IOSurfaceLock", "input", in_lock.lock_status());
            return 5;
        }
        uint8_t *base = (uint8_t *)IOSurfaceGetBaseAddress(h->in_surface);
        if (base == NULL) {
            set_error_text("input surface has no base address");
            return 5;
        }
        for (int c = 0; c < h->input; c++) {
            _Float16 *channel = (_Float16 *)(base + (size_t)c * 32 * 2);
            for (int col = 0; col < 32; col++) channel[col] = (_Float16)features[c];
        }
        kern_return_t unlocked = in_lock.unlock();
        if (unlocked != KERN_SUCCESS) {
            set_surface_error("IOSurfaceUnlock", "input", unlocked);
            return 5;
        }
    }
    NSError *e = nil;
    BOOL ok = ((BOOL(*)(id, SEL, unsigned int, id, id, NSError **))objc_msgSend)(
        (__bridge id)h->model, @selector(evaluateWithQoS:options:request:error:), 21, @{},
        (__bridge id)h->request, &e);
    if (!ok) {
        set_error("evaluate", e);
        return 4;
    }
    SurfaceLock out_lock(h->out_surface, kIOSurfaceLockReadOnly);
    if (!out_lock.held()) {
        set_surface_error("IOSurfaceLock", "output", out_lock.lock_status());
        return 6;
    }
    _Float16 *buf = (_Float16 *)IOSurfaceGetBaseAddress(h->out_surface);
    if (buf == NULL) {
        set_error_text("output surface has no base address");
        return 6;
    }
    // final readout: matmul channels are stored shifted +64 mod padded width,
    // so logical channel i lives at surface channel (i + 64) mod out_w.
    // The padded width keeps the low-channel artifact zone on padding, and
    // the parity test with nonzero biases covers this mapping.
    int ow = out_w(h->outputs);
    for (int i = 0; i < h->experts; i++) expert_out[i] = (float)buf[((i + 64) % ow) * 32];
    for (int i = 0; i < h->classes; i++)
        class_out[i] = (float)buf[(((h->experts + i) + 64) % ow) * 32];
    kern_return_t unlocked = out_lock.unlock();
    if (unlocked != KERN_SUCCESS) {
        set_surface_error("IOSurfaceUnlock", "output", unlocked);
        return 6;
    }
    return 0;
}

// The mutex-holding wrappers. Each takes g_bridge_mutex, then runs its body in
// a pool inside the mutex, so the pool drains and the body's locals release
// before the mutex goes. The outer guarded() keeps the file lock and its own
// pool around these wrappers.
int new_handle(int input, int hidden, int experts, int classes, const float *w1, const float *b1,
               const float *w2, const float *b2, void **out) {
    std::lock_guard<std::mutex> guard(g_bridge_mutex);
    return pooled("ane_bridge_new", [&]() -> int {
        return new_body(input, hidden, experts, classes, w1, b1, w2, b2, out);
    });
}

int predict_handle(void *vp, const float *features, size_t feature_n, float *expert_out,
                   size_t expert_n, float *class_out, size_t class_n) {
    std::lock_guard<std::mutex> guard(g_bridge_mutex);
    return pooled("ane_bridge_predict", [&]() -> int {
        return predict_body(vp, features, feature_n, expert_out, expert_n, class_out, class_n);
    });
}

// The caller holds the file lock when it can. The handle releases its objects
// while this function holds g_bridge_mutex, so the release cannot overlap a
// predict on another handle. If the mutex lock throws, the handle's deleter
// still releases the objects.
int free_handle(void *vp) {
    AnedHandle h((Aned *)vp);
    if (!h) return 0;
    std::lock_guard<std::mutex> guard(g_bridge_mutex);
    return pooled("ane_bridge_free", [&]() -> int {
        int status = 0;
        if (h->model != NULL && !unload_model((__bridge id)h->model, true)) {
            status = kStatusException;
        }
        h.reset();
        return status;
    });
}

}  // namespace

// Lock order for new and free: the file lock first, then g_bridge_mutex.
// The file lock lives outside guarded(), so it releases after the guard's
// autorelease pool drains. Nothing that the framework can remove from the
// shared model directory is released after the lock goes.
extern "C" int ane_bridge_new(int input, int hidden, int experts, int classes, const float *w1,
                              const float *b1, const float *w2, const float *b2,
                              void **out) noexcept {
    const int early = guarded("ane_bridge_new", [&]() -> int {
        return check_new_args(input, hidden, experts, classes, w1, b1, w2, b2, out);
    });
    if (early != 0) return early;
    ProcessLock process_lock;
    if (!process_lock.acquire(kCompileLockTimeout)) return 17;
    return guarded("ane_bridge_new", [&]() -> int {
        return new_handle(input, hidden, experts, classes, w1, b1, w2, b2, out);
    });
}

extern "C" int ane_bridge_predict(void *vp, const float *features, size_t feature_n,
                                  float *expert_out, size_t expert_n, float *class_out,
                                  size_t class_n) noexcept {
    return guarded("ane_bridge_predict", [&]() -> int {
        return predict_handle(vp, features, feature_n, expert_out, expert_n, class_out, class_n);
    });
}

extern "C" int ane_bridge_free(void *vp) noexcept {
    if (vp == NULL) return 0;
    // Cleanup must finish when another process holds the file lock, so a
    // timeout continues unlocked. guarded() clears the timeout text, because
    // the timeout is not an error of this call.
    ProcessLock process_lock;
    (void)process_lock.acquire(kFreeLockTimeout);
    return guarded("ane_bridge_free", [&]() -> int { return free_handle(vp); });
}

extern "C" int ane_bridge_last_error(char *buffer, size_t capacity) noexcept {
    if (!buffer || capacity == 0) return 1;
    snprintf(buffer, capacity, "%s", g_last_error);
    return 0;
}
