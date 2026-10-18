/*
 * P6-M1 bench: cost of one linear call, split by CUDA runtime call. Commit da3e219
 * measured inference_cuda_linear (three cudaMalloc/cudaFree pairs per call). Since
 * P6-workbuf it calls inference_cuda_linear_ws on caller-owned device buffers and
 * requires zero cudaMalloc and cudaFree hits per call.
 *
 * It links the unchanged engine files projector.cu, kernels.cu and repack.cu. The
 * linker option --wrap (see build.sh) sends the five runtime calls that run_linear
 * makes through the __wrap_ functions below. While g_rec.active is set they time
 * each call on the host; otherwise they pass straight through. The measured code is
 * the production code path, not a copy of it.
 *
 * Usage: p6m1_bench --mode cold|warm --events 0|1 --calls N --warmup W
 *                   --out FILE.csv --meta FILE.meta [--thermal PATH]
 *   cold: flush the L2 cache before each call and pick a random expert of a stack, so
 *         weights come from DRAM as in a decode step that touches 40 layers.
 *   warm: no flush, expert 0 every call.
 *   events 1: also record GPU-side windows with cudaEvent (adds three event records
 *         to every call, so the host totals of an events run are not the headline).
 * Exit codes: 0 ok, 2 error or bad usage, 3 thermal limit reached.
 */
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "abi.h"
#include "kernels.h"
#include "repack.h"

namespace {

int64_t now_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

/* Order of the phases is the order of the calls in run_linear (projector.cu). */
enum Phase {
    MALLOC_X, MALLOC_XF, MALLOC_Y, H2D, MEMSET, SYNC, D2H, FREE_Y, FREE_XF, FREE_X, PHASES
};

const char *const kPhaseNames[PHASES] = {
    "malloc_x_ns", "malloc_xf_ns", "malloc_y_ns", "h2d_ns",    "memset_ns",
    "sync_ns",     "d2h_ns",       "free_y_ns",   "free_xf_ns", "free_x_ns"};

struct Recorder {
    bool active = false;
    bool events = false;
    int malloc_seq = 0;
    int free_seq = 0;
    int64_t ns[PHASES] = {};
    int hits[PHASES] = {};
    cudaEvent_t after_h2d = nullptr;
    cudaEvent_t after_memset = nullptr;
    cudaEvent_t before_sync = nullptr;

    void reset() {
        malloc_seq = 0;
        free_seq = 0;
        for (int i = 0; i < PHASES; ++i) {
            ns[i] = 0;
            hits[i] = 0;
        }
    }

    void add(int phase, int64_t dt) {
        ns[phase] += dt;
        hits[phase] += 1;
    }
};

Recorder g_rec;

}  // namespace

extern "C" {

cudaError_t __real_cudaMalloc(void **ptr, size_t bytes);
cudaError_t __real_cudaFree(void *ptr);
cudaError_t __real_cudaMemcpy(void *dst, const void *src, size_t bytes, cudaMemcpyKind kind);
cudaError_t __real_cudaMemset(void *ptr, int value, size_t bytes);
cudaError_t __real_cudaDeviceSynchronize(void);

/* inference_cuda_linear_ws allocates nothing: measure_call rejects any hit here. */
cudaError_t __wrap_cudaMalloc(void **ptr, size_t bytes) {
    if (!g_rec.active) return __real_cudaMalloc(ptr, bytes);
    const int slot = std::min(g_rec.malloc_seq++, 2);
    const int64_t t0 = now_ns();
    const cudaError_t err = __real_cudaMalloc(ptr, bytes);
    g_rec.add(MALLOC_X + slot, now_ns() - t0);
    return err;
}

/* Same for cudaFree. */
cudaError_t __wrap_cudaFree(void *ptr) {
    if (!g_rec.active) return __real_cudaFree(ptr);
    const int slot = std::min(g_rec.free_seq++, 2);
    const int64_t t0 = now_ns();
    const cudaError_t err = __real_cudaFree(ptr);
    g_rec.add(FREE_Y + slot, now_ns() - t0);
    return err;
}

cudaError_t __wrap_cudaMemcpy(void *dst, const void *src, size_t bytes, cudaMemcpyKind kind) {
    if (!g_rec.active) return __real_cudaMemcpy(dst, src, bytes, kind);
    const int64_t t0 = now_ns();
    const cudaError_t err = __real_cudaMemcpy(dst, src, bytes, kind);
    const int64_t dt = now_ns() - t0;
    if (kind == cudaMemcpyHostToDevice) {
        g_rec.add(H2D, dt);
        if (g_rec.events) (void)cudaEventRecord(g_rec.after_h2d, 0);
    } else if (kind == cudaMemcpyDeviceToHost) {
        g_rec.add(D2H, dt);
    }
    return err;
}

cudaError_t __wrap_cudaMemset(void *ptr, int value, size_t bytes) {
    if (!g_rec.active) return __real_cudaMemset(ptr, value, bytes);
    const int64_t t0 = now_ns();
    const cudaError_t err = __real_cudaMemset(ptr, value, bytes);
    g_rec.add(MEMSET, now_ns() - t0);
    if (g_rec.events) (void)cudaEventRecord(g_rec.after_memset, 0);
    return err;
}

cudaError_t __wrap_cudaDeviceSynchronize(void) {
    if (!g_rec.active) return __real_cudaDeviceSynchronize();
    if (g_rec.events) (void)cudaEventRecord(g_rec.before_sync, 0);
    const int64_t t0 = now_ns();
    const cudaError_t err = __real_cudaDeviceSynchronize();
    g_rec.add(SYNC, now_ns() - t0);
    return err;
}

}  // extern "C"

namespace {

struct Shape {
    const char *id;
    const char *note;
    int ggml_type;
    int n_rows;
    int k;
    int experts;
};

/* Real Qwen3.6-35B-A3B shapes (family/gguf_model/inventory.rs, SURVEY.md section 6). */
const Shape kShapes[] = {
    {"attn_q", "attn_q, DeltaNet attn_qkv", INFERENCE_CUDA_TYPE_Q8_0, 8192, 2048, 1},
    {"attn_kv", "attn_k, attn_v, shared expert gate and up", INFERENCE_CUDA_TYPE_Q8_0, 512, 2048, 1},
    {"attn_o", "attn_output, ssm_out", INFERENCE_CUDA_TYPE_Q8_0, 2048, 4096, 1},
    {"shexp_down", "shared expert down", INFERENCE_CUDA_TYPE_Q8_0, 2048, 512, 1},
    {"attn_gate", "DeltaNet attn_gate", INFERENCE_CUDA_TYPE_Q8_0, 4096, 2048, 1},
    {"ssm_ab", "DeltaNet ssm_alpha, ssm_beta", INFERENCE_CUDA_TYPE_Q8_0, 32, 2048, 1},
    {"exp_gate_up", "routed expert gate and up", INFERENCE_CUDA_TYPE_Q4_K, 512, 2048, 256},
    {"exp_down", "routed expert down", INFERENCE_CUDA_TYPE_Q4_K, 2048, 512, 256},
    {"lm_head", "output.weight", INFERENCE_CUDA_TYPE_Q6_K, 248320, 2048, 1},
};

const int kThermalLimit = 75000;
const size_t kFlushBytes = 64u << 20;
const float kSentinel = 1.0e30f;

struct Options {
    std::string mode = "cold";
    int events = 0;
    int calls = 1000;
    int warmup = 50;
    std::string out;
    std::string meta;
    std::string thermal;
};

struct Sample {
    int expert = 0;
    int64_t total = 0;
    int64_t ns[PHASES] = {};
    int64_t other = 0;
    int64_t memset_win = -1;
    int64_t kernel_win = -1;
    int64_t floor_ns = 0;
};

[[noreturn]] void die(const char *fmt, ...) {
    va_list args;
    va_start(args, fmt);
    fprintf(stderr, "p6m1_bench: ");
    vfprintf(stderr, fmt, args);
    fprintf(stderr, "\n");
    va_end(args);
    exit(2);
}

void check(cudaError_t err, const char *what) {
    if (err != cudaSuccess) die("%s: %s", what, cudaGetErrorString(err));
}

int64_t elapsed_ns(cudaEvent_t from, cudaEvent_t to) {
    float ms = 0.0f;
    check(cudaEventElapsedTime(&ms, from, to), "cudaEventElapsedTime");
    return (int64_t)((double)ms * 1.0e6);
}

uint64_t next_random(uint64_t &state) {
    state ^= state << 13;
    state ^= state >> 7;
    state ^= state << 17;
    return state;
}

int read_thermal(const std::string &path) {
    FILE *file = fopen(path.c_str(), "r");
    if (file == nullptr) die("cannot open %s", path.c_str());
    int value = -1;
    if (fscanf(file, "%d", &value) != 1) die("cannot parse %s", path.c_str());
    fclose(file);
    return value;
}

/* Tile bytes of one expert: the tiled size of n_rows x k. */
size_t expert_bytes(const Shape &s) {
    return tiled_vecs((GgmlType)s.ggml_type, s.n_rows, s.k) * sizeof(uint4);
}

/* Random bytes stand in for the weights. Kernel time does not depend on the values;
 * numeric parity is gated by the engine tests, not here. */
void fill_random(void *device, size_t bytes, uint64_t seed) {
    const size_t chunk_words = (size_t)(8u << 20) / sizeof(uint64_t);
    std::vector<uint64_t> host(std::min(bytes / sizeof(uint64_t), chunk_words));
    uint64_t state = seed;
    for (size_t offset = 0; offset < bytes;) {
        const size_t n = std::min(host.size() * sizeof(uint64_t), bytes - offset);
        for (size_t i = 0; i < n / sizeof(uint64_t); ++i) host[i] = next_random(state);
        check(cudaMemcpy((unsigned char *)device + offset, host.data(), n, cudaMemcpyHostToDevice),
              "weight upload");
        offset += n;
    }
}

class Bench {
public:
    explicit Bench(const Options &opt) : opt_(opt) {}
    int run();

private:
    void setup_device();
    void write_meta_header();
    int run_shape(const Shape &s);
    Sample measure_call(const Shape &s, const void *weight);
    int64_t measure_floor(const Shape &s, const void *weight);
    void flush_l2();
    void write_rows(const Shape &s, const std::vector<Sample> &rows);

    Options opt_;
    FILE *csv_ = nullptr;
    FILE *meta_ = nullptr;
    void *flush_buf_ = nullptr;
    cudaEvent_t floor_start_ = nullptr;
    cudaEvent_t floor_end_ = nullptr;
    std::vector<float> x_;
    std::vector<float> y_;
    float *dev_x_ = nullptr;
    uint4 *dev_xf_ = nullptr;
    float *dev_y_ = nullptr;
    inference_cuda_workspace ws_ = {};
    uint64_t rng_ = 88172645463325252ull;
    int flush_count_ = 0;
};

void Bench::setup_device() {
    int count = 0;
    check(cudaGetDeviceCount(&count), "cudaGetDeviceCount");
    if (count < 1) die("no CUDA device");
    check(cudaSetDevice(0), "cudaSetDevice");
    check(cudaFree(nullptr), "context init");
    check(cudaEventCreate(&g_rec.after_h2d), "event");
    check(cudaEventCreate(&g_rec.after_memset), "event");
    check(cudaEventCreate(&g_rec.before_sync), "event");
    check(cudaEventCreate(&floor_start_), "event");
    check(cudaEventCreate(&floor_end_), "event");
    check(cudaMalloc(&flush_buf_, kFlushBytes), "flush buffer");
    g_rec.events = opt_.events != 0;
}

void Bench::write_meta_header() {
    cudaDeviceProp prop;
    check(cudaGetDeviceProperties(&prop, 0), "cudaGetDeviceProperties");
    int runtime = 0, driver = 0, clock_khz = 0, mem_khz = 0;
    check(cudaRuntimeGetVersion(&runtime), "runtime version");
    check(cudaDriverGetVersion(&driver), "driver version");
    check(cudaDeviceGetAttribute(&clock_khz, cudaDevAttrClockRate, 0), "clock rate");
    check(cudaDeviceGetAttribute(&mem_khz, cudaDevAttrMemoryClockRate, 0), "memory clock rate");
    fprintf(meta_, "device=%s\nsm=%d%d\nsm_count=%d\nl2_bytes=%d\nglobal_mem_bytes=%zu\n", prop.name,
            prop.major, prop.minor, prop.multiProcessorCount, prop.l2CacheSize, prop.totalGlobalMem);
    fprintf(meta_, "integrated=%d\nbus_width_bits=%d\nsm_clock_khz=%d\nmem_clock_khz=%d\n",
            prop.integrated, prop.memoryBusWidth, clock_khz, mem_khz);
    fprintf(meta_, "cuda_runtime=%d\ncuda_driver=%d\n", runtime, driver);
    fprintf(meta_, "mode=%s\nevents=%d\ncalls=%d\nwarmup=%d\nflush_bytes=%zu\n", opt_.mode.c_str(),
            opt_.events, opt_.calls, opt_.warmup, kFlushBytes);
    fflush(meta_);
}

void Bench::flush_l2() {
    check(cudaMemset(flush_buf_, ++flush_count_ & 0xff, kFlushBytes), "flush memset");
    check(cudaDeviceSynchronize(), "flush sync");
}

Sample Bench::measure_call(const Shape &s, const void *weight) {
    g_rec.reset();
    g_rec.active = true;
    int stage = 0;
    const int64_t t0 = now_ns();
    const int status = inference_cuda_linear_ws(s.ggml_type, weight, s.n_rows, s.k, x_.data(), 1,
                                                y_.data(), &ws_, &stage);
    const int64_t t1 = now_ns();
    g_rec.active = false;
    if (status != 0) die("inference_cuda_linear_ws %s: status %d stage %d", s.id, status, stage);
    Sample out;
    out.total = t1 - t0;
    int64_t phase_sum = 0;
    for (int p = 0; p < PHASES; ++p) {
        const bool allocation = p == MALLOC_X || p == MALLOC_XF || p == MALLOC_Y ||
                                p == FREE_Y || p == FREE_XF || p == FREE_X;
        const int want = allocation ? 0 : 1;
        if (g_rec.hits[p] != want)
            die("%s: phase %s ran %d times, expected %d", s.id, kPhaseNames[p], g_rec.hits[p], want);
        out.ns[p] = g_rec.ns[p];
        phase_sum += g_rec.ns[p];
    }
    out.other = out.total - phase_sum;
    if (g_rec.events) {
        out.memset_win = elapsed_ns(g_rec.after_h2d, g_rec.after_memset);
        out.kernel_win = elapsed_ns(g_rec.after_memset, g_rec.before_sync);
    }
    return out;
}

/* Pack and matmul only, on buffers that already exist: what one call would cost with a
 * persistent workspace. Includes the host launch gaps between the two kernels. */
int64_t Bench::measure_floor(const Shape &s, const void *weight) {
    check(cudaEventRecord(floor_start_, 0), "floor event");
    check(kern::pack_f32(dev_x_, 1, s.k, dev_xf_), "floor pack");
    check(kern::matmul((GgmlType)s.ggml_type, 1, (const uint4 *)weight, dev_xf_, dev_y_, s.n_rows,
                       s.n_rows, s.k),
          "floor matmul");
    check(cudaEventRecord(floor_end_, 0), "floor event");
    check(cudaEventSynchronize(floor_end_), "floor event sync");
    return elapsed_ns(floor_start_, floor_end_);
}

void Bench::write_rows(const Shape &s, const std::vector<Sample> &rows) {
    for (size_t i = 0; i < rows.size(); ++i) {
        const Sample &r = rows[i];
        fprintf(csv_, "%s,%s,%d,%zu,%d,%lld", s.id, opt_.mode.c_str(), opt_.events, i, r.expert,
                (long long)r.total);
        for (int p = 0; p < PHASES; ++p) fprintf(csv_, ",%lld", (long long)r.ns[p]);
        fprintf(csv_, ",%lld,%lld,%lld,%lld\n", (long long)r.other, (long long)r.memset_win,
                (long long)r.kernel_win, (long long)r.floor_ns);
    }
    fflush(csv_);
}

int Bench::run_shape(const Shape &s) {
    if (!opt_.thermal.empty()) {
        const int before = read_thermal(opt_.thermal);
        fprintf(meta_, "thermal_before shape=%s temp=%d\n", s.id, before);
        fflush(meta_);
        if (before >= kThermalLimit) {
            fprintf(meta_, "thermal_abort shape=%s temp=%d\n", s.id, before);
            fflush(meta_);
            return 3;
        }
    }
    const size_t per_expert = expert_bytes(s);
    if (per_expert == 0) die("%s: not a tiled shape", s.id);
    unsigned char *weights = nullptr;
    check(cudaMalloc((void **)&weights, per_expert * (size_t)s.experts), "weight alloc");
    fill_random(weights, per_expert * (size_t)s.experts, 0x9e3779b97f4a7c15ull);
    x_.assign((size_t)s.k, 0.0f);
    for (size_t i = 0; i < x_.size(); ++i) x_[i] = (float)((next_random(rng_) >> 40) % 2001) / 1000.0f - 1.0f;
    y_.assign((size_t)s.n_rows, kSentinel);
    check(cudaMalloc((void **)&dev_x_, x_.size() * sizeof(float)), "floor x");
    check(cudaMalloc((void **)&dev_xf_, kern::xf_vecs() * sizeof(uint4)), "floor xf");
    check(cudaMalloc((void **)&dev_y_, y_.size() * sizeof(float)), "floor y");
    check(cudaMemcpy(dev_x_, x_.data(), x_.size() * sizeof(float), cudaMemcpyHostToDevice), "floor x upload");
    check(cudaMemset(dev_xf_, 0, kern::xf_vecs() * sizeof(uint4)), "floor xf zero");
    ws_ = {dev_x_, x_.size() * sizeof(float), dev_xf_, kern::xf_vecs() * sizeof(uint4),
           dev_y_, y_.size() * sizeof(float)};
    const bool cold = opt_.mode == "cold";
    std::vector<Sample> rows;
    for (int i = 0; i < opt_.warmup + opt_.calls; ++i) {
        const int expert = (cold && s.experts > 1) ? (int)(next_random(rng_) % (uint64_t)s.experts) : 0;
        const void *weight = weights + per_expert * (size_t)expert;
        if (cold) flush_l2();
        Sample sample = measure_call(s, weight);
        sample.expert = expert;
        if (i == 0 && std::count(y_.begin(), y_.end(), kSentinel) == (long)y_.size())
            die("%s: y is untouched after the first call", s.id);
        if (cold) flush_l2();
        sample.floor_ns = measure_floor(s, weight);
        if (i >= opt_.warmup) rows.push_back(sample);
    }
    write_rows(s, rows);
    check(cudaFree(weights), "weight free");
    check(cudaFree(dev_x_), "floor x free");
    check(cudaFree(dev_xf_), "floor xf free");
    check(cudaFree(dev_y_), "floor y free");
    if (!opt_.thermal.empty()) fprintf(meta_, "thermal_after shape=%s temp=%d\n", s.id, read_thermal(opt_.thermal));
    fprintf(meta_, "done shape=%s rows=%zu expert_bytes=%zu\n", s.id, rows.size(), per_expert);
    fflush(meta_);
    return 0;
}

int Bench::run() {
    csv_ = fopen(opt_.out.c_str(), "w");
    meta_ = fopen(opt_.meta.c_str(), "w");
    if (csv_ == nullptr || meta_ == nullptr) die("cannot open output files");
    setup_device();
    write_meta_header();
    fprintf(csv_, "shape,mode,events,call,expert,total_ns");
    for (int p = 0; p < PHASES; ++p) fprintf(csv_, ",%s", kPhaseNames[p]);
    fprintf(csv_, ",other_ns,memset_win_ns,kernel_win_ns,floor_ns\n");
    for (const Shape &s : kShapes) {
        const int code = run_shape(s);
        if (code != 0) return code;
    }
    fclose(csv_);
    fclose(meta_);
    return 0;
}

bool parse_args(int argc, char **argv, Options &opt) {
    for (int i = 1; i + 1 < argc; i += 2) {
        const std::string key = argv[i];
        const std::string value = argv[i + 1];
        if (key == "--mode") opt.mode = value;
        else if (key == "--events") opt.events = atoi(value.c_str());
        else if (key == "--calls") opt.calls = atoi(value.c_str());
        else if (key == "--warmup") opt.warmup = atoi(value.c_str());
        else if (key == "--out") opt.out = value;
        else if (key == "--meta") opt.meta = value;
        else if (key == "--thermal") opt.thermal = value;
        else return false;
    }
    const bool pairs = (argc - 1) % 2 == 0;
    const bool mode_ok = opt.mode == "cold" || opt.mode == "warm";
    return pairs && mode_ok && (opt.events == 0 || opt.events == 1) && opt.calls > 0 &&
           opt.warmup >= 0 && !opt.out.empty() && !opt.meta.empty();
}

}  // namespace

int main(int argc, char **argv) {
    Options opt;
    if (!parse_args(argc, argv, opt)) {
        fprintf(stderr,
                "usage: p6m1_bench --mode cold|warm --events 0|1 --calls N --warmup W "
                "--out FILE.csv --meta FILE.meta [--thermal PATH]\n");
        return 2;
    }
    Bench bench(opt);
    return bench.run();
}
