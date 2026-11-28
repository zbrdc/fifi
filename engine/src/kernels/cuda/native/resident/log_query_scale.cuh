// The factor of a long-context attention's clipped query and relative coefficients at a
// position: 1 + alpha * ln((pos + 1) / floor) past the floor, and exactly 1 through it, so a
// context no longer than the floor keeps its bits. A floor of 0 or less turns it off. The log
// runs in double and is rounded once: --use_fast_math approximates only single precision.
#pragma once

__device__ inline float log_query_scale(int pos, int floor, float alpha) {
    const int n = pos + 1;
    if (floor <= 0 || n <= floor) return 1.f;
    const float ratio = __fdiv_rn(float(n), float(floor));
    return __fadd_rn(1.f, __fmul_rn(alpha, float(log(double(ratio)))));
}
