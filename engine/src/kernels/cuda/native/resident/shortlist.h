#ifndef INFERENCE_CUDA_RESIDENT_SHORTLIST_H
#define INFERENCE_CUDA_RESIDENT_SHORTLIST_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
enum {
    INFERENCE_CUDA_SHORTLIST_K = 20,
    INFERENCE_CUDA_SHORTLIST_TILES = 16,
    INFERENCE_CUDA_SHORTLIST_PART_BYTES = 176,
    INFERENCE_CUDA_SHORTLIST_RESULT_OFFSET = 2816,
    INFERENCE_CUDA_SHORTLIST_SCRATCH_BYTES = 2992
};
typedef struct inference_cuda_shortlist_candidate {
    float logit;
    uint32_t id;
} inference_cuda_shortlist_candidate;
typedef struct inference_cuda_shortlist_result {
    inference_cuda_shortlist_candidate entries[20];
    uint32_t non_finite;
    uint32_t pad[3];
} inference_cuda_shortlist_result;
typedef struct inference_cuda_shortlist_args {
    const void *logits;
    void *scratch;
    int n;
} inference_cuda_shortlist_args;
/* Launch-only stable top-20: larger logit first, then lower id; original f32 bits retained.
 * logits: 4-byte-aligned f32[n], n in1..2^30. scratch:16-byte-aligned2992bytes,disjoint.
 * Result is176bytes atscratch+2816;firstmin(n,20) entries are valid whennon_finite==0.
 * Any non-finite input setsnon_finite=1;unusedentries haveidUINT32_MAX/logit-INFINITY.
 * No allocation,copy,synchronizationorstep-statuswrite. Explicitnon-nullstreamrequired. */
int inference_cuda_shortlist_launch(const inference_cuda_shortlist_args *args,
                                    void *stream, int *out_stage);
#ifdef __cplusplus
}
static_assert(sizeof(inference_cuda_shortlist_candidate)==8,"candidate layout");
static_assert(sizeof(inference_cuda_shortlist_result)==176 &&
              offsetof(inference_cuda_shortlist_result,non_finite)==160,"result layout");
static_assert(sizeof(inference_cuda_shortlist_args)==24 &&
              offsetof(inference_cuda_shortlist_args,n)==16,"argument layout");
#endif
#endif
