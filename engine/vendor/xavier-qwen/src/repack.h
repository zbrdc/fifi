// Load-time conversion of row-major ggml quant blocks into the 8-row x 256-k
// tiles the mma884 matmul formats read (see the format structs in mma884.cuh).
#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

#include <vector>

enum class GgmlType : uint32_t {
  F32 = 0, F16 = 1, Q4_0 = 2, Q8_0 = 8, Q4_K = 12, Q5_K = 13, Q6_K = 14
};

// Bytes of one row of k weights in ggml's layout; 0 for an unknown type.
size_t ggml_row_bytes(GgmlType type, int k);

// Whether mma884 has a matmul format for type.
bool is_tiled(GgmlType type);

// Whether to_q8_0 accepts type: matrices mma884 has no format for, which some
// fine-tunes mix in. Q4_0 converts exactly; F32/F16/Q5_K round to Q8_0.
bool converts_to_q8_0(GgmlType type);

// n_rows ggml rows of k weights at src, as ggml Q8_0 rows (k % 32 == 0); empty
// when converts_to_q8_0(type) is false or k % 256 != 0.
std::vector<uint8_t> to_q8_0(GgmlType type, const uint8_t* src, size_t n_rows, int k);

// uint4s of the tiled form of an n_rows x k matrix (n_rows % 8 == 0, k % 256 == 0);
// 0 when is_tiled(type) is false.
size_t tiled_vecs(GgmlType type, int n_rows, int k);

// Repacks n_rows ggml rows at src into tiled_vecs(type, n_rows, k) uint4s at dst.
// Row groups stay in order, so matrices stacked by rows repack to stacked tiles.
// Returns false, writing nothing, for an untiled type or a bad shape.
bool repack(GgmlType type, const uint8_t* src, int n_rows, int k, uint4* dst);
