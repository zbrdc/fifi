// The first read of the pages of a mapped file: one word every 4 KiB, one thread each. A
// dispatch that comes after finds the pages mapped for the GPU. Read by one kernel that
// marches, a page at a time, the pages cost that launch its wall time; read here, the faults
// come at once. The stride is below any page size, so every page is read.
//
// Every thread writes the word it read, so the read stands. A thread past the end reads
// nothing. The length is the Rust side's parameter, eight bytes.

#include <metal_stdlib>
using namespace metal;

constant constexpr uint kStride = 4096;

kernel void page_touch(device const uchar *bytes [[buffer(0)]],
                       constant ulong &len [[buffer(1)]],
                       device uint *touched [[buffer(2)]],
                       uint i [[thread_position_in_grid]]) {
  const ulong at = ulong(i) * kStride;
  if (at + 4 <= len) {
    touched[i] = *reinterpret_cast<device const uint *>(bytes + at);
  }
}
