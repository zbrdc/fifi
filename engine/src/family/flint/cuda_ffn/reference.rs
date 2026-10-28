//! Provisional parent-authored scalar reference; never a production prefill fallback.

#[derive(Clone, Debug)]
pub(super) struct Matrix {
    pub(super) rows: usize,
    pub(super) cols: usize,
    pub(super) values: Vec<f32>,
    pub(super) factor: f32,
}

#[derive(Clone, Debug)]
pub(super) struct Expert {
    pub(super) gate: Matrix,
    pub(super) up: Matrix,
    pub(super) down: Matrix,
}

#[derive(Clone, Debug)]
pub(super) struct Fixture {
    pub(super) hidden: usize,
    pub(super) router: Matrix,
    pub(super) experts: Vec<Expert>,
    pub(super) shared: Expert,
    pub(super) dense: Expert,
}

#[derive(Clone, Copy)]
pub(super) enum Kind {
    Dense,
    Moe,
}

#[derive(Default, Debug)]
pub(super) struct Expected {
    pub(super) output: Vec<f32>,
    pub(super) ids: Vec<i32>,
    pub(super) gates: Vec<f32>,
    pub(super) sorted_pairs: Vec<u32>,
    pub(super) inverse: Vec<u32>,
    pub(super) max_intermediate: f32,
}

pub(super) struct RealLayer<'a> {
    pub(super) q: &'a Matrix,
    pub(super) k: &'a Matrix,
    pub(super) v: &'a Matrix,
    pub(super) o: &'a Matrix,
    pub(super) pre_norm: &'a [f32],
    pub(super) post_norm: &'a [f32],
    pub(super) q_norm: &'a [f32],
    pub(super) k_norm: &'a [f32],
    pub(super) sinks: &'a [f32],
    pub(super) ffn: &'a Fixture,
    pub(super) kind: Kind,
    pub(super) sliding: bool,
}

pub(super) fn bf16(value: f32) -> Result<f32, String> {
    if !value.is_finite() {
        return Err("nonfinite BF16 input".into());
    }
    let bits = value.to_bits();
    let rounded = bits.wrapping_add(0x7fff + ((bits >> 16) & 1)) & 0xffff0000;
    let result = f32::from_bits(rounded);
    if !result.is_finite() {
        return Err("BF16 publication overflow".into());
    }
    Ok(result)
}

fn sigmoid(value: f32) -> f32 {
    1.0 / (1.0 + (-value).exp())
}

fn validate_matrix(matrix: &Matrix) -> Result<(), String> {
    if matrix.rows == 0
        || matrix.cols == 0
        || matrix.rows.checked_mul(matrix.cols) != Some(matrix.values.len())
        || !matrix.factor.is_finite()
        || matrix.factor <= 0.0
        || matrix.values.iter().any(|value| !value.is_finite())
    {
        return Err("invalid reference matrix".into());
    }
    Ok(())
}

fn validate_expert(expert: &Expert, hidden: usize) -> Result<(), String> {
    for matrix in [&expert.gate, &expert.up, &expert.down] {
        validate_matrix(matrix)?;
    }
    if expert.gate.cols != hidden
        || expert.up.cols != hidden
        || expert.gate.rows != expert.up.rows
        || expert.down.cols != expert.gate.rows
        || expert.down.rows != hidden
    {
        return Err("reference expert shapes disagree".into());
    }
    Ok(())
}

fn project(input: &[f32], matrix: &Matrix) -> Result<Vec<f32>, String> {
    if input.len() != matrix.cols {
        return Err("projection input width mismatch".into());
    }
    matrix
        .values
        .chunks_exact(matrix.cols)
        .map(|weight| {
            let sum = input
                .iter()
                .zip(weight)
                .fold(0.0f32, |sum, (&x, &w)| x.mul_add(w, sum));
            bf16(sum * matrix.factor)
        })
        .collect()
}

fn expert(input: &[f32], weights: &Expert) -> Result<(Vec<f32>, Vec<f32>), String> {
    let gate = project(input, &weights.gate)?;
    let up = project(input, &weights.up)?;
    let hidden: Vec<f32> = gate
        .iter()
        .zip(up)
        .map(|(&gate, up)| {
            let probability = bf16(sigmoid(gate))?;
            let silu = bf16(gate * probability)?;
            bf16(silu * up)
        })
        .collect::<Result<_, String>>()?;
    // No guard/undo is emulated: this is the independent, unscaled down reference.
    let down = project(&hidden, &weights.down)?;
    Ok((hidden, down))
}

pub(super) fn route(logits: &[f32]) -> Result<(Vec<usize>, Vec<f32>), String> {
    if logits.len() < 6 || logits.iter().any(|value| !value.is_finite()) {
        return Err("invalid raw router logits".into());
    }
    let scores: Vec<f32> = logits.iter().map(|&value| sigmoid(value)).collect();
    let mut ids: Vec<usize> = (0..scores.len()).collect();
    ids.sort_by(|&a, &b| scores[b].total_cmp(&scores[a]).then(a.cmp(&b)));
    ids.truncate(6);
    let mass = ids.iter().fold(0.0f32, |sum, &id| sum + scores[id]);
    if !mass.is_finite() || mass <= 0.0 {
        return Err("invalid selected router mass".into());
    }
    let gates = ids.iter().map(|&id| scores[id] / mass).collect();
    Ok((ids, gates))
}

fn dense_rows(fixture: &Fixture, input: &[f32]) -> Result<Expected, String> {
    let mut result = Expected::default();
    for token in input.chunks_exact(fixture.hidden) {
        let (hidden, down) = expert(token, &fixture.dense)?;
        result.max_intermediate = hidden
            .iter()
            .fold(result.max_intermediate, |max, v| max.max(v.abs()));
        result.output.extend(down);
    }
    Ok(result)
}

fn moe_rows(fixture: &Fixture, input: &[f32]) -> Result<Expected, String> {
    let mut router = fixture.router.clone();
    router.values = router
        .values
        .iter()
        .map(|&value| bf16(value))
        .collect::<Result<_, _>>()?;
    let mut result = Expected::default();
    for token in input.chunks_exact(fixture.hidden) {
        let logits = project(token, &router)?;
        let (ids, gates) = route(&logits)?;
        let mut total = vec![0.0f32; fixture.hidden];
        for (&id, &gate) in ids.iter().zip(&gates) {
            let (hidden, down) = expert(token, &fixture.experts[id])?;
            result.max_intermediate = hidden
                .iter()
                .fold(result.max_intermediate, |max, v| max.max(v.abs()));
            let gate = bf16(gate)?;
            for (sum, value) in total.iter_mut().zip(down) {
                *sum += bf16(value * gate)?;
            }
        }
        let (shared_hidden, shared) = expert(token, &fixture.shared)?;
        result.max_intermediate = shared_hidden
            .iter()
            .fold(result.max_intermediate, |max, v| max.max(v.abs()));
        for (sum, shared) in total.into_iter().zip(shared) {
            result.output.push(bf16(bf16(sum)? + shared)?);
        }
        result.ids.extend(ids.into_iter().map(|id| id as i32));
        result.gates.extend(gates);
    }
    let mut order: Vec<usize> = (0..result.ids.len()).collect();
    order.sort_by_key(|&pair| (result.ids[pair], pair));
    result.inverse = vec![0; order.len()];
    for (sorted, &original) in order.iter().enumerate() {
        result.inverse[original] = sorted as u32;
    }
    result.sorted_pairs = order.into_iter().map(|pair| pair as u32).collect();
    Ok(result)
}

pub(super) fn run(
    fixture: &Fixture,
    input: &[f32],
    rows: usize,
    kind: Kind,
) -> Result<Expected, String> {
    if rows == 0 || fixture.hidden == 0 || rows.checked_mul(fixture.hidden) != Some(input.len()) {
        return Err("invalid reference block geometry".into());
    }
    let input: Vec<f32> = input
        .iter()
        .map(|&value| bf16(value))
        .collect::<Result<_, _>>()?;
    match kind {
        Kind::Dense => {
            validate_expert(&fixture.dense, fixture.hidden)?;
            dense_rows(fixture, &input)
        }
        Kind::Moe => {
            validate_matrix(&fixture.router)?;
            if fixture.experts.len() < 6
                || fixture.experts.len() > 4096
                || fixture.router.rows != fixture.experts.len()
                || fixture.router.cols != fixture.hidden
                || fixture.router.factor != 1.0
            {
                return Err("invalid reference router geometry".into());
            }
            validate_expert(&fixture.shared, fixture.hidden)?;
            for weights in &fixture.experts {
                validate_expert(weights, fixture.hidden)?;
            }
            moe_rows(fixture, &input)
        }
    }
}

#[path = "reference_layer.rs"]
pub(super) mod layer;
#[path = "reference_real.rs"]
pub(super) mod real;

#[cfg(test)]
#[path = "reference_tests.rs"]
mod tests;
