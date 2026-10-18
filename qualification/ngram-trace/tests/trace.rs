use ngram_trace::{
    PAGE_BYTES_4K, PAGE_BYTES_16K, PageCounts, PleConfig, TraceError, page_counts, row_ids,
    trace_tokens,
};

const VOCABULARIES: [u64; 16] = [
    101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179,
];
const OFFSETS: [u64; 16] = [
    0, 101, 204, 311, 420, 533, 660, 791, 928, 1067, 1216, 1367, 1524, 1687, 1854, 2027,
];

fn fixture() -> PleConfig {
    PleConfig {
        eos_token_id: 97,
        position_multipliers: [3, 5, 7],
        head_vocab_sizes: VOCABULARIES,
        head_offsets: OFFSETS,
        table_rows: 2_206,
    }
}

#[test]
fn startup_padding_matches_independent_reference_rows_for_all_heads() {
    let expected = [
        79, 172, 259, 358, 451, 635, 750, 863, 930, 1158, 1297, 1418, 1545, 1688, 1998, 2147,
    ];
    assert_eq!(row_ids(&fixture(), &[2]), Ok(expected));
}

#[test]
fn token_positions_and_head_order_match_independent_reference_rows() {
    let expected = [
        13, 112, 211, 316, 421, 647, 774, 905, 991, 1130, 1279, 1430, 1587, 1750, 1917, 2090,
    ];
    assert_eq!(row_ids(&fixture(), &[11, 13, 17]), Ok(expected));
}

#[test]
fn predecessor_eos_resets_older_tokens_but_current_eos_keeps_its_context() {
    let config = fixture();
    let after_eos = [
        88, 181, 268, 367, 460, 644, 759, 872, 937, 1165, 1304, 1425, 1552, 1695, 2005, 2154,
    ];
    let current_eos = [
        98, 195, 290, 393, 494, 579, 698, 817, 940, 1208, 1355, 1500, 1651, 1810, 1971, 2138,
    ];
    assert_eq!(row_ids(&config, &[2, 97, 3]), Ok(after_eos));
    assert_eq!(row_ids(&config, &[2, 3, 97]), Ok(current_eos));
}

#[test]
fn positional_products_use_unsigned_64_bit_wrapping_before_xor() {
    let mut config = fixture();
    config.position_multipliers = [u64::MAX - 4, u64::MAX - 18, u64::MAX - 34];
    let expected = [
        76, 122, 234, 365, 468, 544, 663, 880, 943, 1083, 1278, 1431, 1651, 1840, 1888, 2153,
    ];
    assert_eq!(row_ids(&config, &[248_318, 248_319, 248_317]), Ok(expected));
}

#[test]
fn trace_tokens_emits_one_row_vector_per_token_and_preserves_eos_order() {
    let expected_start = [
        79, 172, 259, 358, 451, 635, 750, 863, 930, 1158, 1297, 1418, 1545, 1688, 1998, 2147,
    ];
    let expected_current_eos = [
        95, 192, 287, 390, 491, 576, 695, 814, 1004, 1083, 1220, 1492, 1619, 1762, 1899, 2042,
    ];
    let expected_after_eos = [
        88, 181, 268, 367, 460, 644, 759, 872, 937, 1165, 1304, 1425, 1552, 1695, 2005, 2154,
    ];
    assert_eq!(
        trace_tokens(&fixture(), &[2, 97, 3]),
        Ok(vec![
            expected_start,
            expected_current_eos,
            expected_after_eos
        ])
    );
}

#[test]
fn invalid_ranges_empty_histories_and_out_of_bounds_rows_fail_closed() {
    let mut config = fixture();
    config.head_vocab_sizes[3] = 0;
    assert_eq!(
        row_ids(&config, &[2]),
        Err(TraceError::InvalidHeadRange { head: 3 })
    );

    let mut config = fixture();
    config.head_offsets[15] = u64::MAX;
    assert_eq!(
        row_ids(&config, &[2]),
        Err(TraceError::InvalidHeadRange { head: 15 })
    );
    assert_eq!(row_ids(&fixture(), &[]), Err(TraceError::EmptyHistory));
    assert_eq!(
        page_counts(&[2_206], 192, 90, 2_206),
        Err(TraceError::RowOutOfBounds {
            row: 2_206,
            table_rows: 2_206,
        })
    );
}

#[test]
fn page_simulation_counts_every_page_spanned_from_the_file_offset() {
    assert_eq!(PAGE_BYTES_4K, 4_096);
    assert_eq!(PAGE_BYTES_16K, 16_384);
    assert_eq!(
        page_counts(&[43, 179], 192, 90, 2_206),
        Ok(PageCounts {
            pages_4k: 4,
            pages_16k: 2,
        })
    );
    assert_eq!(
        page_counts(&[43, 43], 192, 90, 2_206),
        Ok(PageCounts {
            pages_4k: 2,
            pages_16k: 1,
        })
    );
}

#[test]
fn page_simulation_rejects_invalid_extents_and_checked_address_overflow() {
    assert_eq!(
        page_counts(&[0], 192, 0, 2_206),
        Err(TraceError::InvalidRowBytes)
    );
    assert_eq!(
        page_counts(&[0], u64::MAX - 1, 90, 2_206),
        Err(TraceError::AddressOverflow)
    );
    assert_eq!(page_counts(&[], 192, 90, 0), Err(TraceError::EmptyTable));
}
