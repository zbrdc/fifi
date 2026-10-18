const QUERY_HEADS: usize = 16;
const KV_HEADS: usize = 2;
const HEAD_DIM: usize = 256;
const QUERIES_PER_KV_HEAD: usize = 8;
const Q8_GROUP: usize = 64;
const Q8_WORDS_PER_ROW: usize = 64;
const SCORE_SCALE: f32 = 0.0625;

pub struct Q8Rows<'a> {
    pub capacity: usize,
    pub codes: &'a [u32],
    pub scales: &'a [f32],
    pub biases: &'a [f32],
}

pub fn attention(
    query: &[f32],
    keys: &Q8Rows<'_>,
    values: &Q8Rows<'_>,
    len: usize,
) -> Result<Vec<f32>, &'static str> {
    let output_len = checked_product(QUERY_HEADS, HEAD_DIM)?;
    validate_inputs(query, keys, values, len, output_len)?;
    for value in query {
        if !value.is_finite() {
            return Err("query contains a nonfinite value");
        }
    }

    let mut output = Vec::new();
    output
        .try_reserve_exact(output_len)
        .map_err(|_| "attention output allocation failed")?;
    output.resize(output_len, 0.0);

    let mut scores = Vec::new();
    scores
        .try_reserve_exact(len)
        .map_err(|_| "attention score allocation failed")?;
    scores.resize(len, 0.0);

    for query_head in 0..QUERY_HEADS {
        let query_start = checked_product(query_head, HEAD_DIM)?;
        let kv_head = query_head / QUERIES_PER_KV_HEAD;
        let mut maximum = f32::NEG_INFINITY;

        for position in 0..len {
            let mut dot = 0.0_f32;
            for dimension in 0..HEAD_DIM {
                let query_index = checked_sum(query_start, dimension)?;
                let query_value = *query
                    .get(query_index)
                    .ok_or("query extent does not match fixed geometry")?;
                let key_value = dequantize(keys, kv_head, position, dimension)?;
                let product = query_value * key_value;
                if !product.is_finite() {
                    return Err("attention dot product is not finite");
                }
                dot += product;
                if !dot.is_finite() {
                    return Err("attention dot product is not finite");
                }
            }

            let score = dot * SCORE_SCALE;
            if !score.is_finite() {
                return Err("attention score is not finite");
            }
            *scores
                .get_mut(position)
                .ok_or("attention score extent is invalid")? = score;
            if score > maximum {
                maximum = score;
            }
        }

        let mut normalization = 0.0_f32;
        for position in 0..len {
            let score = *scores
                .get(position)
                .ok_or("attention score extent is invalid")?;
            let shifted = score - maximum;
            if shifted.is_nan() || shifted == f32::INFINITY {
                return Err("shifted attention score is invalid");
            }
            let weight = shifted.exp();
            if !weight.is_finite() {
                return Err("attention normalization is not finite");
            }
            *scores
                .get_mut(position)
                .ok_or("attention score extent is invalid")? = weight;
            normalization += weight;
            if !normalization.is_finite() {
                return Err("attention normalization is not finite");
            }
        }
        if normalization <= 0.0 {
            return Err("attention normalization is not positive");
        }

        for position in 0..len {
            let weight = scores
                .get_mut(position)
                .ok_or("attention score extent is invalid")?;
            *weight /= normalization;
            if !weight.is_finite() {
                return Err("attention probability is not finite");
            }
        }

        for dimension in 0..HEAD_DIM {
            let output_index = checked_sum(query_start, dimension)?;
            let mut sum = 0.0_f32;
            for position in 0..len {
                let probability = *scores
                    .get(position)
                    .ok_or("attention score extent is invalid")?;
                let value = dequantize(values, kv_head, position, dimension)?;
                let product = probability * value;
                if !product.is_finite() {
                    return Err("attention output is not finite");
                }
                sum += product;
                if !sum.is_finite() {
                    return Err("attention output is not finite");
                }
            }
            *output
                .get_mut(output_index)
                .ok_or("attention output extent is invalid")? = sum;
        }
    }

    Ok(output)
}

fn validate_inputs(
    query: &[f32],
    keys: &Q8Rows<'_>,
    values: &Q8Rows<'_>,
    len: usize,
    output_len: usize,
) -> Result<(), &'static str> {
    if query.len() != output_len {
        return Err("query extent does not match fixed geometry");
    }
    if keys.capacity != values.capacity {
        return Err("key and value capacities differ");
    }
    if len == 0 || len > keys.capacity {
        return Err("attention length is empty or exceeds capacity");
    }

    let code_len = checked_product(checked_product(KV_HEADS, keys.capacity)?, Q8_WORDS_PER_ROW)?;
    let group_len = checked_product(checked_product(KV_HEADS, keys.capacity)?, Q8_GROUPS_PER_ROW)?;
    if keys.codes.len() != code_len
        || values.codes.len() != code_len
        || keys.scales.len() != group_len
        || keys.biases.len() != group_len
        || values.scales.len() != group_len
        || values.biases.len() != group_len
    {
        return Err("Q8 extents do not match capacity");
    }
    Ok(())
}

const Q8_GROUPS_PER_ROW: usize = 4;

fn dequantize(
    rows: &Q8Rows<'_>,
    kv_head: usize,
    position: usize,
    dimension: usize,
) -> Result<f32, &'static str> {
    let row = checked_sum(checked_product(kv_head, rows.capacity)?, position)?;
    let code_row = checked_product(row, Q8_WORDS_PER_ROW)?;
    let group_row = checked_product(row, Q8_GROUPS_PER_ROW)?;
    let word_index = checked_sum(code_row, dimension / 4)?;
    let group_index = checked_sum(group_row, dimension / Q8_GROUP)?;
    let word = *rows
        .codes
        .get(word_index)
        .ok_or("Q8 code extent is invalid")?;
    let scale = *rows
        .scales
        .get(group_index)
        .ok_or("Q8 scale extent is invalid")?;
    let bias = *rows
        .biases
        .get(group_index)
        .ok_or("Q8 bias extent is invalid")?;
    if !scale.is_finite() || !bias.is_finite() {
        return Err("active Q8 parameters contain a nonfinite value");
    }

    let shift = (dimension % 4) * 8;
    let code = ((word >> shift) & 0xff) as f32;
    let value = scale * code + bias;
    if !value.is_finite() {
        return Err("active Q8 value is not finite");
    }
    Ok(value)
}

fn checked_product(left: usize, right: usize) -> Result<usize, &'static str> {
    left.checked_mul(right)
        .ok_or("attention extent arithmetic overflow")
}

fn checked_sum(left: usize, right: usize) -> Result<usize, &'static str> {
    left.checked_add(right)
        .ok_or("attention index arithmetic overflow")
}
