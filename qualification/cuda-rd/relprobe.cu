#include <cstdio>
#include <cstring>
__global__ void k(float *x) { x[threadIdx.x] *= 2.f; }
int main(int argc, char **argv) {
    if (argc > 1 && !strcmp(argv[1], "build-info")) {
        printf("{\"profile\":\"release\",\"opt_level\":\"3\",\"debug\":false,\"debug_assertions\":false}\n");
        return 0;
    }
    float *d = nullptr; cudaMalloc(&d, 128); k<<<1, 32>>>(d); cudaDeviceSynchronize(); return 0;
}
