/*
 * Architecture clamp of the vendored MMVQ wrappers: see mmvq_arch.h.
 * Pure arithmetic over the architecture list, kept out of the wrappers so one definition serves
 * both and the CPU tests can drive it with any list.
 */
#include <stddef.h>

#include "resident/mmvq_arch.h"

namespace {

/* Cubins of this major and later run on a later minor of their own major (CUDA programming guide,
 * binary compatibility). The Volta/Turing family (major 7) is the exception: sm_72 does not run
 * on a 7.5 device. */
const int kForwardCompatibleMajor = 8;

bool cubin_runs_on(int arch, int cc) {
    if (arch == cc) {
        return true;
    }
    const int major = cc / 100;
    return arch < cc && arch / 100 == major && major >= kForwardCompatibleMajor;
}

}  // namespace

extern "C" int inference_cuda_mmvq_runnable_arch(int cc, const int *built, int n_built) {
    if (built == NULL || n_built < 1 || cc < 1) {
        return 0;
    }
    int best = 0;
    for (int i = 0; i < n_built; ++i) {
        if (built[i] > best && cubin_runs_on(built[i], cc)) {
            best = built[i];
        }
    }
    return best;
}
