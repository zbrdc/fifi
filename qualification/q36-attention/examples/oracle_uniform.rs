use q36_attention_qualification::{Q8Rows, attention};

const QUERY_HEADS: usize = 16;
const KV_HEADS: usize = 2;
const HEAD_DIM: usize = 256;
const Q8_GROUP: usize = 64;
const WORDS_PER_ROW: usize = HEAD_DIM / 4;
const GROUPS_PER_ROW: usize = HEAD_DIM / Q8_GROUP;
const CAPACITY: usize = 16640;
const LONG_PREFIX_BOUND: f32 = 2e-4;

fn main() -> Result<(), &'static str> {
    let query = vec![0.0; QUERY_HEADS * HEAD_DIM];
    let key_codes = vec![0; KV_HEADS * CAPACITY * WORDS_PER_ROW];
    let key_scales = vec![0.0; KV_HEADS * CAPACITY * GROUPS_PER_ROW];
    let key_biases = vec![0.0; KV_HEADS * CAPACITY * GROUPS_PER_ROW];
    let keys = Q8Rows {
        capacity: CAPACITY,
        codes: &key_codes,
        scales: &key_scales,
        biases: &key_biases,
    };

    let value_codes = vec![0; KV_HEADS * CAPACITY * WORDS_PER_ROW];
    let value_scales = vec![0.0; KV_HEADS * CAPACITY * GROUPS_PER_ROW];
    let mut value_biases = vec![0.0; KV_HEADS * CAPACITY * GROUPS_PER_ROW];
    for kv_head in 0..KV_HEADS {
        let bias = if kv_head == 0 { 1.0 } else { -1.0 };
        for row in 0..CAPACITY {
            let start = (kv_head * CAPACITY + row) * GROUPS_PER_ROW;
            value_biases[start..start + GROUPS_PER_ROW].fill(bias);
        }
    }
    let values = Q8Rows {
        capacity: CAPACITY,
        codes: &value_codes,
        scales: &value_scales,
        biases: &value_biases,
    };

    for len in [16383, 16384, 16385] {
        let output = attention(&query, &keys, &values, len)?;
        assert_eq!(output.len(), QUERY_HEADS * HEAD_DIM);
        assert!(
            output.iter().all(|value| value.is_finite()),
            "output for length {len} contains a nonfinite value"
        );

        let max_abs_error = output
            .iter()
            .enumerate()
            .map(|(index, actual)| {
                let query_head = index / HEAD_DIM;
                let expected = if query_head < QUERY_HEADS / 2 {
                    1.0
                } else {
                    -1.0
                };
                (actual - expected).abs()
            })
            .fold(0.0_f32, f32::max);
        assert!(
            max_abs_error <= LONG_PREFIX_BOUND,
            "length {len} max absolute error {max_abs_error} exceeds {LONG_PREFIX_BOUND}"
        );
        println!(
            "{{\"len\":{len},\"capacity\":{CAPACITY},\"max_abs_error\":{max_abs_error:.9e},\"bound\":{LONG_PREFIX_BOUND:.1e}}}"
        );
    }

    Ok(())
}
