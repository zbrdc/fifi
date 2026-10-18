#!/usr/bin/env bash
# Builds the P6-M1 bench from the unchanged engine CUDA sources.
# Usage: build.sh <sm> <repo-root> <out-dir>   (sm: 86 for the RTX 3090, 72 for the Xavier)
# NVCC overrides the compiler (the Xavier needs /usr/local/cuda-12.2/bin/nvcc).
set -eu

arch="$1"
root="$2"
out="$3"
nvcc_bin="${NVCC:-nvcc}"
native="$root/engine/src/kernels/cuda/native"
xq="$root/engine/vendor/xavier-qwen/src"
here="$root/qualification/cuda-linear-split"

mkdir -p "$out"
flags=(-O2
    -gencode "arch=compute_${arch},code=sm_${arch}"
    -gencode "arch=compute_${arch},code=compute_${arch}"
    -I "$native" -I "$xq")

"$nvcc_bin" --version
for src in "$native/projector.cu" "$xq/kernels.cu" "$xq/repack.cu" "$here/bench.cu"; do
    name="$(basename "$src" .cu)"
    "$nvcc_bin" "${flags[@]}" -c "$src" -o "$out/$name.o"
    echo "compiled $name"
done

# --wrap routes the runtime calls of run_linear through the wrappers in bench.cu.
"$nvcc_bin" -o "$out/p6m1_bench" \
    "$out/projector.o" "$out/kernels.o" "$out/repack.o" "$out/bench.o" \
    -Xlinker --wrap=cudaMalloc -Xlinker --wrap=cudaFree -Xlinker --wrap=cudaMemcpy \
    -Xlinker --wrap=cudaMemset -Xlinker --wrap=cudaDeviceSynchronize
echo "built $out/p6m1_bench"
