//! Unit tests of the pure part of the op executor (test seam U13). CPU only. The launch wiring
//! against real wrappers is covered by the scratch device run of RD-9b3b.
use super::mixer::tail_extent;

#[test]
fn a_projection_writes_the_tail_of_its_slot_from_its_column() {
    // DeltaProj at one row: 12,352 floats; the z projection starts at column 8,192.
    assert_eq!(tail_extent(12352 * 4, 8192).expect("z"), (32768, 16640));
    assert_eq!(tail_extent(12352 * 4, 0).expect("qkv"), (0, 49408));
    assert_eq!(tail_extent(9216 * 4, 8704).expect("v"), (34816, 2048));
    assert!(
        tail_extent(12352 * 4, 12352).is_err(),
        "a column past the row"
    );
    assert!(tail_extent(16, usize::MAX).is_err(), "no overflow");
}
