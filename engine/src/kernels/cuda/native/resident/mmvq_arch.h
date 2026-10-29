/*
 * Which built architecture the vendored MMVQ objects run on for a device. fifi_check_device
 * (vendored common.cuh) accepts a cc only when it is exactly one of the built architectures, so
 * an sm_89 GPU with a 72,86 build was refused although the sm_86 cubin runs on it. The two
 * wrappers clamp the device's cc to the architecture of the cubin that runs there before they
 * fill fifi_device_facts, and refuse when none does.
 */
#ifndef INFERENCE_CUDA_RESIDENT_MMVQ_ARCH_H
#define INFERENCE_CUDA_RESIDENT_MMVQ_ARCH_H

#ifdef __cplusplus
extern "C" {
#endif

/* cc and every entry of built are in ggml's internal encoding 100 * major + 10 * minor, the
 * encoding of __CUDA_ARCH_LIST__ (860 is sm_86). Returns the highest built architecture whose
 * cubin runs on a device of compute capability cc, or 0 when none does. A cubin runs on its own
 * architecture and, from Ampere on (major 8 and later), on a later minor of the same major
 * (sm_86 on 8.9); Volta and Turing cubins (major 7) run on no other minor (sm_72 on 7.5), and no
 * cubin runs across a major. Pure: no CUDA call. */
int inference_cuda_mmvq_runnable_arch(int cc, const int *built, int n_built);

#ifdef __cplusplus
}

/* The architecture list this translation unit is built for: nvcc defines __CUDA_ARCH_LIST__ in
 * every pass from the -gencode flags of build.rs (the vendored group also takes it from a generated
 * header, same set). 0 means the device cannot run any built cubin. */
static inline int inference_cuda_mmvq_built_arch(int cc) {
    static const int built[] = {__CUDA_ARCH_LIST__};
    return inference_cuda_mmvq_runnable_arch(cc, built, (int)(sizeof(built) / sizeof(built[0])));
}
#endif

#endif /* INFERENCE_CUDA_RESIDENT_MMVQ_ARCH_H */
