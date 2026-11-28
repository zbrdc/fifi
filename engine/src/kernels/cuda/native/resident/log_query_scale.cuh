// The factor of a long-context attention's clipped query and relative coefficients at a
// position: 1 + alpha * ln((pos + 1) / floor) past the floor, and exactly 1 through it, so a
// context no longer than the floor keeps its bits. A floor of 0 or less turns it off. The log
// runs in double and is rounded once: --use_fast_math approximates only single precision.
#pragma once

#include <cmath>

__device__ inline float log_query_scale(int pos, int floor, float alpha) {
    if (floor <= 0 || pos < floor) return 1.f;
    const float n = float(static_cast<long long>(pos) + 1LL);
    const float ratio = __fdiv_rn(n, float(floor));
    return __fadd_rn(1.f, __fmul_rn(alpha, float(log(double(ratio)))));
}
