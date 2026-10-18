use q36_attention_qualification::{Q8Rows, attention};

const QUERY_HEADS: usize = 16;
const KV_HEADS: usize = 2;
const HEAD_DIM: usize = 256;
const CAPACITY: usize = 2;
const WORDS_PER_ROW: usize = 64;
const GROUPS_PER_ROW: usize = 4;

struct Q8Storage {
    capacity: usize,
    codes: Vec<u32>,
    scales: Vec<f32>,
    biases: Vec<f32>,
}

impl Q8Storage {
    fn zeroed(capacity: usize) -> Self {
        Self {
            capacity,
            codes: vec![0; KV_HEADS * capacity * WORDS_PER_ROW],
            scales: vec![0.0; KV_HEADS * capacity * GROUPS_PER_ROW],
            biases: vec![0.0; KV_HEADS * capacity * GROUPS_PER_ROW],
        }
    }

    fn view(&self) -> Q8Rows<'_> {
        Q8Rows {
            capacity: self.capacity,
            codes: &self.codes,
            scales: &self.scales,
            biases: &self.biases,
        }
    }

    fn set_bias_row(&mut self, kv_head: usize, row: usize, value: f32) {
        let start = (kv_head * self.capacity + row) * GROUPS_PER_ROW;
        self.biases[start..start + GROUPS_PER_ROW].fill(value);
    }
}

#[test]
fn attends_over_two_packed_q8_rows_with_grouped_query_heads() {
    let mut query = vec![0.0; QUERY_HEADS * HEAD_DIM];
    for head in 0..QUERY_HEADS {
        query[head * HEAD_DIM] = if head < 4 || (8..12).contains(&head) {
            16.0
        } else {
            -16.0
        };
    }

    let mut key_codes = vec![0; KV_HEADS * CAPACITY * WORDS_PER_ROW];
    key_codes[WORDS_PER_ROW] = 1;
    key_codes[(CAPACITY + 1) * WORDS_PER_ROW] = 1;
    let key_scales = vec![1.0; KV_HEADS * CAPACITY * GROUPS_PER_ROW];
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
    for (kv_head, rows) in [[2.0, 6.0], [-4.0, 8.0]].into_iter().enumerate() {
        for (row, value) in rows.into_iter().enumerate() {
            let start = (kv_head * CAPACITY + row) * GROUPS_PER_ROW;
            value_biases[start..start + GROUPS_PER_ROW].fill(value);
        }
    }
    let values = Q8Rows {
        capacity: CAPACITY,
        codes: &value_codes,
        scales: &value_scales,
        biases: &value_biases,
    };

    let result = attention(&query, &keys, &values, CAPACITY);
    assert!(result.is_ok(), "valid fixture returned {result:?}");
    let output = result.unwrap();
    assert_eq!(output.len(), QUERY_HEADS * HEAD_DIM);

    for head in 0..QUERY_HEADS {
        let expected = match head {
            0..=3 => 4.924_234_f32,
            4..=7 => 3.075_766_f32,
            8..=11 => 4.772_703_f32,
            _ => -0.772_702_9_f32,
        };
        for dimension in 0..HEAD_DIM {
            let actual = output[head * HEAD_DIM + dimension];
            assert!(
                actual.is_finite(),
                "output at [{head}, {dimension}] is not finite"
            );
            assert!(
                (actual - expected).abs() <= 1e-6,
                "output at [{head}, {dimension}] was {actual}, expected {expected}"
            );
        }
    }
}

#[test]
fn uniform_scores_ignore_poisoned_tail_rows() {
    let capacity = 3;
    let query = vec![0.0; QUERY_HEADS * HEAD_DIM];
    let key_codes = vec![0; KV_HEADS * capacity * WORDS_PER_ROW];
    let mut key_scales = vec![1.0; KV_HEADS * capacity * GROUPS_PER_ROW];
    let mut key_biases = vec![0.0; KV_HEADS * capacity * GROUPS_PER_ROW];
    let mut value_biases = vec![0.0; KV_HEADS * capacity * GROUPS_PER_ROW];
    let value_codes = vec![0; KV_HEADS * capacity * WORDS_PER_ROW];
    let value_scales = vec![0.0; KV_HEADS * capacity * GROUPS_PER_ROW];

    for kv_head in 0..KV_HEADS {
        for (row, value) in [2.0, 6.0].into_iter().enumerate() {
            let start = (kv_head * capacity + row) * GROUPS_PER_ROW;
            value_biases[start..start + GROUPS_PER_ROW].fill(value);
        }
        let tail = (kv_head * capacity + 2) * GROUPS_PER_ROW;
        key_scales[tail..tail + GROUPS_PER_ROW].fill(f32::NAN);
        key_biases[tail..tail + GROUPS_PER_ROW].fill(f32::NAN);
        value_biases[tail..tail + GROUPS_PER_ROW].fill(f32::NAN);
    }

    let keys = Q8Rows {
        capacity,
        codes: &key_codes,
        scales: &key_scales,
        biases: &key_biases,
    };
    let values = Q8Rows {
        capacity,
        codes: &value_codes,
        scales: &value_scales,
        biases: &value_biases,
    };
    let output = attention(&query, &keys, &values, 2).expect("active prefix is finite");

    assert_eq!(output.len(), QUERY_HEADS * HEAD_DIM);
    for (index, actual) in output.into_iter().enumerate() {
        assert!(actual.is_finite(), "output {index} is not finite");
        assert!((actual - 4.0).abs() <= 1e-6, "output {index} was {actual}");
    }
}

#[test]
fn packed_byte_lanes_and_affine_scales_decode() {
    const EXPECTED: [[[f32; 4]; 4]; 2] = [
        [
            [1.0, 2.0, 3.0, 4.0],
            [1.0, 3.0, 5.0, 7.0],
            [1.0, 5.0, 9.0, 13.0],
            [1.0, 9.0, 17.0, 25.0],
        ],
        [
            [4.0, 3.0, 2.0, 1.0],
            [7.0, 5.0, 3.0, 1.0],
            [13.0, 9.0, 5.0, 1.0],
            [25.0, 17.0, 9.0, 1.0],
        ],
    ];
    let query = vec![0.0; QUERY_HEADS * HEAD_DIM];
    let keys = Q8Storage::zeroed(CAPACITY);
    let mut values = Q8Storage::zeroed(CAPACITY);
    let words = [0x04030201, 0x01020304];
    let scales = [1.0, 2.0, 4.0, 8.0];
    let biases = [0.0, -1.0, -3.0, -7.0];

    for (kv_head, word) in words.into_iter().enumerate() {
        let row_start = kv_head * CAPACITY * WORDS_PER_ROW;
        values.codes[row_start..row_start + WORDS_PER_ROW].fill(word);
        let group_start = kv_head * CAPACITY * GROUPS_PER_ROW;
        values.scales[group_start..group_start + GROUPS_PER_ROW].copy_from_slice(&scales);
        values.biases[group_start..group_start + GROUPS_PER_ROW].copy_from_slice(&biases);
    }

    let output =
        attention(&query, &keys.view(), &values.view(), 1).expect("one active row is finite");
    for query_head in 0..QUERY_HEADS {
        let kv_head = query_head / 8;
        for dimension in 0..HEAD_DIM {
            let group = dimension / 64;
            let lane = dimension % 4;
            let actual = output[query_head * HEAD_DIM + dimension];
            let expected = EXPECTED[kv_head][group][lane];
            assert!(
                actual.is_finite(),
                "output at [{query_head}, {dimension}] is not finite"
            );
            assert!(
                (actual - expected).abs() <= 1e-6,
                "output at [{query_head}, {dimension}] was {actual}, expected {expected}"
            );
        }
    }
}

#[test]
fn scores_1000_and_1001_use_stable_softmax() {
    let mut query = vec![0.0; QUERY_HEADS * HEAD_DIM];
    for head in 0..QUERY_HEADS {
        query[head * HEAD_DIM] = 16.0;
    }
    let mut keys = Q8Storage::zeroed(CAPACITY);
    let mut values = Q8Storage::zeroed(CAPACITY);
    for kv_head in 0..KV_HEADS {
        keys.set_bias_row(kv_head, 0, 1000.0);
        keys.set_bias_row(kv_head, 1, 1001.0);
        values.set_bias_row(kv_head, 0, 2.0);
        values.set_bias_row(kv_head, 1, 6.0);
    }

    let output = attention(&query, &keys.view(), &values.view(), CAPACITY)
        .expect("finite large scores have a finite stable softmax");
    for (index, actual) in output.into_iter().enumerate() {
        assert!(actual.is_finite(), "output {index} is not finite");
        assert!(
            (actual - 4.924_234_f32).abs() <= 1e-6,
            "output {index} was {actual}"
        );
    }
}

#[test]
fn rejects_invalid_extents_and_lengths() {
    let query = vec![0.0; QUERY_HEADS * HEAD_DIM];
    let keys = Q8Storage::zeroed(CAPACITY);
    let values = Q8Storage::zeroed(CAPACITY);
    assert!(attention(&query, &keys.view(), &values.view(), 0).is_err());
    assert!(attention(&query, &keys.view(), &values.view(), CAPACITY + 1).is_err());
    assert!(attention(&query[..query.len() - 1], &keys.view(), &values.view(), 1).is_err());

    let short_keys = Q8Storage::zeroed(CAPACITY - 1);
    assert!(attention(&query, &short_keys.view(), &values.view(), 1).is_err());

    let mut short_codes = Q8Storage::zeroed(CAPACITY);
    short_codes.codes.pop();
    assert!(attention(&query, &short_codes.view(), &values.view(), 1).is_err());

    let mut short_scales = Q8Storage::zeroed(CAPACITY);
    short_scales.scales.pop();
    assert!(attention(&query, &keys.view(), &short_scales.view(), 1).is_err());

    let overflow = Q8Rows {
        capacity: usize::MAX,
        codes: &[],
        scales: &[],
        biases: &[],
    };
    assert!(attention(&query, &overflow, &overflow, 1).is_err());
}

#[test]
fn rejects_nonfinite_active_inputs_and_reconstructions() {
    let mut query = vec![0.0; QUERY_HEADS * HEAD_DIM];
    let mut keys = Q8Storage::zeroed(CAPACITY);
    let mut values = Q8Storage::zeroed(CAPACITY);

    query[0] = f32::NAN;
    assert!(attention(&query, &keys.view(), &values.view(), 1).is_err());
    query[0] = 0.0;

    keys.scales[0] = f32::NAN;
    assert!(attention(&query, &keys.view(), &values.view(), 1).is_err());
    keys.scales[0] = 0.0;

    keys.biases[0] = f32::INFINITY;
    assert!(attention(&query, &keys.view(), &values.view(), 1).is_err());
    keys.biases[0] = 0.0;

    values.scales[0] = f32::NAN;
    assert!(attention(&query, &keys.view(), &values.view(), 1).is_err());
    values.scales[0] = 0.0;

    values.biases[0] = f32::INFINITY;
    assert!(attention(&query, &keys.view(), &values.view(), 1).is_err());
    values.biases[0] = 0.0;

    keys.codes[0] = 255;
    keys.scales[0] = f32::MAX;
    assert!(attention(&query, &keys.view(), &values.view(), 1).is_err());
}
