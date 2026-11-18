#pragma once

// Host-only launch policy; callers validate their payload-specific bounds.
namespace fifi_ring {

constexpr int kValuesPerBlock = 256 * 8;
constexpr int kMaxBlocks = 64;

constexpr bool multiblock_enabled(const char *value) {
    return value != nullptr && value[0] == '1' && value[1] == '\0';
}

constexpr int payload_blocks(int count, bool enabled) {
    if (!enabled || count <= kValuesPerBlock) {
        return 1;
    }
    const int blocks = (count + kValuesPerBlock - 1) / kValuesPerBlock;
    return blocks < kMaxBlocks ? blocks : kMaxBlocks;
}

} // namespace fifi_ring
