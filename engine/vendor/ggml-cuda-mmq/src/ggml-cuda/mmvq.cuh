#include "common.cuh"

#include <cstdint>

#define MMVQ_MAX_BATCH_SIZE 8 // Max. batch size for which to use MMVQ kernels.

// V-MV1 (d): the device-side fusion struct of upstream common.cuh (pin lines 1578-1590), which the
// V-4 trim removed from the vendored common.cuh, restored here because both mul_mat_vec_q kernels
// take it by value; its ggml_glu_op field is the enum of the ggml.h shim (MATVEC-PAR S1 c1). The
// host struct (pin 1569-1577) is not restored: no kept host code reads tensors. The S1 launch
// entry passes an all-zero struct (no fusion); S3 may fuse swiglu into the matvec.
struct ggml_cuda_mm_fusion_args_device {
    const void * x_bias = nullptr;
    const void * gate = nullptr;
    const void * gate_bias = nullptr;
    const void * x_scale = nullptr;
    const void * gate_scale = nullptr;
    ggml_glu_op glu_op;
    float glu_limit = 0.0f;
};

// V-MV1 (a): the whole argument set of one MMVQ launch, in place of the deleted ggml_tensor host
// entries. nchannels_x is the number of experts of the weight stack on the ids path (the range the
// per-slot id is guarded against, V-MV1 A1) and 1 on the dense path; nchannels_y is 1 for one
// shared activation vector (gate/up) and n_used for one per used slot (down); dev replaces the
// deleted ggml_cuda_info() reads (V-3b pattern: mmq_args.dev).
struct fifi_mmvq_args {
    ggml_type type;              // one of Q4_K, Q5_0, Q6_K, Q8_0 (V-MV1 c)
    const void * vx;             // weight blocks, native GGUF row-major layout
    const void * vy_q8_1;        // activation, block_q8_1, MATRIX_ROW_PADDING-padded rows
    const int32_t * ids;         // per-slot expert ids read on the device, or null (dense path)
    float * dst;
    int64_t ncols_x;             // k
    int64_t nrows_x;             // rows of one expert (or of the dense weight)
    int64_t ncols_dst;           // activation columns, 1..MMVQ_MAX_BATCH_SIZE
    int64_t nchannels_x;         // experts of the stack (ids path) or 1
    int64_t nchannels_y;         // activation vectors (see above)
    int64_t nchannels_dst;       // used slots (ids path) or 1
    int64_t stride_row_x;        // weight blocks per row (ncols_x / qk)
    int64_t stride_col_y;        // q8_1 blocks per activation column
    int64_t stride_col_dst;      // floats per dst column; also the dst row bound
    int64_t stride_channel_x;    // weight blocks per expert
    int64_t stride_channel_y;    // q8_1 blocks per activation vector
    int64_t stride_channel_dst;  // floats per dst slot (A2: y_slot_stride, >= nrows_x)
    fifi_device_facts dev;       // the device the launch is for (fifi_check_device)
};

cudaError_t fifi_mmvq_launch(const fifi_mmvq_args & a, cudaStream_t stream);
