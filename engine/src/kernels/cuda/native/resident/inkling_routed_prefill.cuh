// Launch-only Inkling Mini stage. The caller reserves one contiguous scratch area.
namespace {
struct RoutedTile {
    int expert, begin, end;
};
static_assert(sizeof(RoutedTile) == 3 * sizeof(int));
// One warp per expert first counts preceding slots, then performs stable compaction.
// Faulty ids map to expert zero for bounds safety and set the caller's status.
__global__ void routed_plan(const int *ids, int slots, int *bounds, int *order, int *inverse,
                            int *status) {
    int e = blockIdx.x, lane = threadIdx.x, base = 0;
    for (int f = 0; f < slots; f += 32) {
        int s = f + lane, id = s < slots ? ids[s] : 128;
        if (s < slots && unsigned(id) >= 128) {
            atomicOr(status, 1);
            id = 0;
        }
        base += __popc(__ballot_sync(~0u, id < e));
    }
    if (!lane) {
        bounds[e] = base;
        if (e == 127)
            bounds[128] = slots;
    }
    for (int f = 0; f < slots; f += 32) {
        int s = f + lane, id = s < slots ? ids[s] : 128;
        bool hit = s < slots && (unsigned(id) < 128 ? id : 0) == e;
        unsigned bits = __ballot_sync(~0u, hit);
        if (hit) {
            int pos = base + __popc(bits & ((1u << lane) - 1));
            order[pos] = s;
            inverse[s] = pos;
        }
        base += __popc(bits);
    }
}
__global__ void routed_tiles(const int *bounds, RoutedTile *tiles, int *count) {
    int e = blockIdx.x, b = 0;
    for (int j = 0; j < e; j++)
        b += (bounds[j + 1] - bounds[j] + 15) / 16;
    int lo = bounds[e], hi = bounds[e + 1], n = (hi - lo + 15) / 16;
    for (int i = threadIdx.x; i < n; i += 32)
        tiles[b + i] = {e, lo + 16 * i, min(hi, lo + 16 * (i + 1))};
    if (e == 127 && !threadIdx.x)
        *count = b + n;
}
template <int Type, int Block, int Bytes, bool Gate>
__global__ void routed_matrix(const RoutedTile *tiles, const int *nt, const int *order,
                              const float *input, const uint8_t *stack, float *output) {
#if __CUDA_ARCH__ >= 700
    using namespace nvcuda::wmma;
    if (blockIdx.y >= *nt)
        return;
    constexpr int K = Gate ? 2304 : 1152, N = 2304;
    auto tile = tiles[blockIdx.y];
    int t = threadIdx.x, col = blockIdx.x * 64;
    __shared__ __align__(32) __half x[16][24], w[64][24];
    __shared__ __align__(32) float result[16][64];
    WmmaAccumulator acc;
    fill_fragment(acc, 0.f);
    for (int k = 0; k < K; k += 16) {
        for (int i = t; i < 256; i += 128) {
            int r = tile.begin + i / 16;
            float v = 0;
            if (r < tile.end) {
                int row = Gate ? order[r] / 8 : r;
                v = input[row * K + k + i % 16];
            }
            x[i / 16][i % 16] = __float2half_rn(v);
        }
        for (int i = t; i < 1024; i += 128) {
            int n = col + i / 16;
            auto row = stack + (size_t(tile.expert) * N + n) * (K / Block * Bytes);
            w[i / 16][i % 16] = wmma_weight<Type, Block, Bytes>(row, k + i % 16);
        }
        __syncthreads();
        fragment<matrix_a, 16, 16, 16, __half, row_major> a;
        fragment<matrix_b, 16, 16, 16, __half, col_major> b;
        load_matrix_sync(a, &x[0][0], 24);
        load_matrix_sync(b, &w[t / 32 * 16][0], 24);
        mma_sync(acc, a, b, acc);
        __syncthreads();
    }
    store_matrix_sync(&result[0][t / 32 * 16], acc, 64, mem_row_major);
    __syncthreads();
    for (int i = t; i < 1024; i += 128)
        if (tile.begin + i / 64 < tile.end)
            output[(tile.begin + i / 64) * N + col + i % 64] = result[i / 64][i % 64];
#endif
}
__global__ void routed_activate(const int *ids, const int *order, const float *scales,
                                const float *gate, float *mid, int slots, float gc, float uc) {
    int p = blockIdx.y, k = blockIdx.x * 128 + threadIdx.x;
    if (p >= slots || k >= 1152)
        return;
    int e = ids[order[p]];
    float s = scales[unsigned(e) < 128 ? e : 0];
    mid[p * 1152 + k] =
        clipped_expert_activation(gate[p * 2304 + k] * s, gate[p * 2304 + 1152 + k] * s, gc, uc);
}
__global__ void routed_sum(const int *ids, const int *inv, const float *weights,
                           const float *scales, const float *down, float *y, int *status) {
    int r = blockIdx.y, k = blockIdx.x * 128 + threadIdx.x;
    float sum = 0;
    for (int j = 0; j < 8; j++) {
        int s = r * 8 + j, e = ids[s];
        sum = fmaf(down[inv[s] * 2304 + k] * scales[unsigned(e) < 128 ? e : 0], weights[s], sum);
    }
    if (!isfinite(sum))
        atomicOr(status, 1);
    y[r * 2304 + k] = sum;
}
} // namespace
// pointers: gate/down stacks, gate/down scales, input, ids, weights, output, scratch, status.
extern "C" int inference_inkling_routed(void *const *p, int rows, int gate, int down, float gc,
                                        float uc, void *stream, int *stage) {
    if (!p || !stream || rows < 1 || rows > 4096 ||
        !(gate == 16 || gate == 17 || gate == 19 || gate == 22) || !(down == 40 || down == 42))
        return INFERENCE_CUDA_INVALID_ARGUMENT;
    for (int i = 0; i < 10; i++)
        if (!p[i])
            return INFERENCE_CUDA_INVALID_ARGUMENT;
    int slots = rows * 8, max_tiles = (slots + 15) / 16 + 127;
    auto s = (cudaStream_t)stream;
    auto ids = (int *)p[5], bounds = (int *)p[8], order = bounds + 129, inv = order + slots,
         nt = inv + slots;
    auto tiles = (RoutedTile *)(nt + 1);
    auto tmp = (float *)(tiles + max_tiles), mid = tmp + slots * 2304;
    routed_plan<<<128, 32, 0, s>>>(ids, slots, bounds, order, inv, (int *)p[9]);
    routed_tiles<<<128, 32, 0, s>>>(bounds, tiles, nt);
    // The number of occupied tiles never exceeds the number of slots. Avoid
    // scheduling an expert-count-sized empty tail at verification/decode widths.
    dim3 grid(36, std::min(slots, max_tiles));
#define PROJECT(ID, B, Q, G, IN, OUT, W)                                                           \
    routed_matrix<ID, B, Q, G>                                                                     \
        <<<grid, 128, 0, s>>>(tiles, nt, order, IN, (const uint8_t *)p[W], OUT)
    switch (gate) {
    case 16:
        PROJECT(16, 256, 66, true, (float *)p[4], tmp, 0);
        break;
    case 17:
        PROJECT(17, 256, 74, true, (float *)p[4], tmp, 0);
        break;
    case 19:
        PROJECT(19, 256, 50, true, (float *)p[4], tmp, 0);
        break;
    case 22:
        PROJECT(22, 256, 82, true, (float *)p[4], tmp, 0);
        break;
    default:
        return INFERENCE_CUDA_INVALID_ARGUMENT;
    }
    routed_activate<<<dim3(9, slots), 128, 0, s>>>(ids, order, (float *)p[2], tmp, mid, slots, gc,
                                                   uc);
    if (down == 42) {
        PROJECT(42, 64, 18, false, mid, tmp, 1);
    } else if (down == 40) {
        PROJECT(40, 64, 36, false, mid, tmp, 1);
    } else
        return INFERENCE_CUDA_INVALID_ARGUMENT;
#undef PROJECT
    routed_sum<<<dim3(18, rows), 128, 0, s>>>(ids, inv, (float *)p[6], (float *)p[3], tmp,
                                              (float *)p[7], (int *)p[9]);
    Failure failure;
    failure.record(cudaGetLastError(), INFERENCE_CUDA_STAGE_LAUNCH);
    report_stage(stage, failure.stage);
    return (int)failure.err;
}
