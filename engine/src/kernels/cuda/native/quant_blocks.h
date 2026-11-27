/*
 * The Q8_1 block layout of the GGUF format, as the engine's own definition: two halves in
 * one 32-bit word (d, the scale, in the low half; s, the sum of the inputs, in the high
 * half), then 32 codes: 36 bytes. The engine's writer is resident/q8_1_store.cuh (the fused
 * producers); the vendored quantize_q8_1 writes the same block and the vendored dot products
 * read it. The vendored tree carries the same layout under the same name in ggml-common.h,
 * and no translation unit includes both: the three producers include this header, the glue
 * includes the vendored one.
 */
#ifndef INFERENCE_CUDA_QUANT_BLOCKS_H
#define INFERENCE_CUDA_QUANT_BLOCKS_H

#include <cuda_fp16.h>
#include <cstdint>

#define QK8_1 32
struct block_q8_1 {
  __half2 ds;
  int8_t qs[QK8_1];
};
static_assert(sizeof(block_q8_1) == 4 + QK8_1, "wrong q8_1 block size/padding");

#endif /* INFERENCE_CUDA_QUANT_BLOCKS_H */
