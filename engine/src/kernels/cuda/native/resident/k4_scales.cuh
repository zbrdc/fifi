/*
 * The 6-bit scale and minimum of sub-block g (0..7) of a Q4_K or Q5_K block: the format's
 * packing of the 12 scale bytes that follow f16 d and f16 dmin (block bytes 4..15). With
 * t = g mod 4, byte t of the first word is factor t, of the second word factor t + 4, of the
 * third word factor t + 8. weights/gguf_dequant.rs::scale_min_k4 is the rule on the host; the
 * device dequant tests compare a kernel's output with it.
 *
 * Two forms of one rule. The window decoders and the in-house Q4_K kernel (matvec_q4k.cu) hold
 * the bytes in memory. The dq matvec and the embed kernel receive the block's first 16 bytes as
 * one uint4 and read the factors by shifts from the words they have loaded, as they did before
 * this header, so their code does not change.
 */
#ifndef INFERENCE_CUDA_RESIDENT_K4_SCALES_CUH
#define INFERENCE_CUDA_RESIDENT_K4_SCALES_CUH

#include <cuda_runtime.h>

/* `scales` points at the first of the 12 bytes. */
__device__ __forceinline__ void scale_min_k4(const unsigned char *scales, int g, unsigned &sc,
                                             unsigned &mn) {
    if (g < 4) {
        sc = scales[g] & 63u;
        mn = scales[g + 4] & 63u;
    } else {
        sc = (scales[g + 4] & 0x0Fu) | ((unsigned)(scales[g - 4] >> 6) << 4);
        mn = (scales[g + 4] >> 4) | ((unsigned)(scales[g] >> 6) << 4);
    }
}

/* `head` holds the block's first 16 bytes: x is d | dmin, and y, z, w the 12 scale bytes. */
__device__ __forceinline__ void scale_min_k4(const uint4 &head, int g, unsigned &sc, unsigned &mn) {
    const unsigned t = (unsigned)(g & 3);
    const unsigned a = head.y >> (8 * t);
    const unsigned b = head.z >> (8 * t);
    const unsigned c = head.w >> (8 * t);
    if (g < 4) {
        sc = a & 63u;
        mn = b & 63u;
    } else {
        sc = (c & 0x0Fu) | ((a >> 2) & 0x30u);
        mn = ((c >> 4) & 0x0Fu) | ((b >> 2) & 0x30u);
    }
}

#endif
