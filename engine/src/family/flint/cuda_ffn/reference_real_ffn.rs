//! FP64 real-model reference; the strict synthetic F32 oracle stays separate.

use super::super::{
    Expected, Expert, Fixture, Kind, Matrix, bf16, validate_expert, validate_matrix,
};

pub(super) fn project(input: &[f32], matrix: &Matrix) -> Result<Vec<f32>, String> {
    validate_matrix(matrix)?;
    if input.len() != matrix.cols || input.iter().any(|value| !value.is_finite()) {
        return Err("invalid real projection input".into());
    }
    matrix
        .values
        .chunks_exact(matrix.cols)
        .map(|weight| {
            let sum = input
                .iter()
                .zip(weight)
                .map(|(&x, &w)| f64::from(x) * f64::from(w))
                .sum::<f64>();
            bf16((sum * f64::from(matrix.factor)) as f32)
        })
        .collect()
}

fn expert(input: &[f32], weights: &Expert) -> Result<(Vec<f32>, Vec<f32>), String> {
    let gate = project(input, &weights.gate)?;
    let up = project(input, &weights.up)?;
    let hidden = gate
        .iter()
        .zip(up)
        .map(|(&gate, up)| {
            let probability = bf16((1.0 / (1.0 + (-f64::from(gate)).exp())) as f32)?;
            let silu = bf16(gate * probability)?;
            bf16(silu * up)
        })
        .collect::<Result<Vec<_>, String>>()?;
    let down = project(&hidden, &weights.down)?;
    Ok((hidden, down))
}

fn route(logits: &[f32]) -> Result<(Vec<usize>, Vec<f32>), String> {
    if logits.len() < 6 || logits.iter().any(|value| !value.is_finite()) {
        return Err("invalid real router logits".into());
    }
    let scores = logits
        .iter()
        .map(|&value| 1.0 / (1.0 + (-f64::from(value)).exp()))
        .collect::<Vec<_>>();
    let mut ids = (0..scores.len()).collect::<Vec<_>>();
    ids.sort_by(|&a, &b| scores[b].total_cmp(&scores[a]).then(a.cmp(&b)));
    ids.truncate(6);
    let mass = ids.iter().map(|&id| scores[id]).sum::<f64>();
    if !mass.is_finite() || mass <= 0.0 {
        return Err("invalid real selected router mass".into());
    }
    let gates = ids.iter().map(|&id| (scores[id] / mass) as f32).collect();
    Ok((ids, gates))
}

fn track(result: &mut Expected, hidden: &[f32]) {
    for value in hidden {
        result.max_intermediate = result.max_intermediate.max(value.abs());
    }
}

fn moe_rows(
    fixture: &Fixture,
    input: &[f32],
    forced: Option<(&[i32], &[f32])>,
) -> Result<Expected, String> {
    validate_matrix(&fixture.router)?;
    if !(6..=4096).contains(&fixture.experts.len())
        || fixture.router.rows != fixture.experts.len()
        || fixture.router.cols != fixture.hidden
        || fixture.router.factor != 1.0
    {
        return Err("invalid real router geometry".into());
    }
    validate_expert(&fixture.shared, fixture.hidden)?;
    for weights in &fixture.experts {
        validate_expert(weights, fixture.hidden)?;
    }
    let mut router = fixture.router.clone();
    for value in &mut router.values {
        *value = bf16(*value)?;
    }
    if let Some((ids, gates)) = forced {
        validate_forced(fixture, input.len() / fixture.hidden, ids, gates)?;
    }
    let mut result = Expected::default();
    for (row, token) in input.chunks_exact(fixture.hidden).enumerate() {
        let (ids, gates) = if let Some((ids, gates)) = forced {
            let span = row * 6..(row + 1) * 6;
            (
                ids[span.clone()].iter().map(|&id| id as usize).collect(),
                gates[span].to_vec(),
            )
        } else {
            route(&project(token, &router)?)?
        };
        let mut total = vec![0.0_f64; fixture.hidden];
        for (&id, &gate) in ids.iter().zip(&gates) {
            let (hidden, down) = expert(token, &fixture.experts[id])?;
            track(&mut result, &hidden);
            let gate = bf16(gate)?;
            for (sum, value) in total.iter_mut().zip(down) {
                *sum += f64::from(bf16(value * gate)?);
            }
        }
        let (hidden, shared) = expert(token, &fixture.shared)?;
        track(&mut result, &hidden);
        for (sum, shared) in total.into_iter().zip(shared) {
            result.output.push(bf16(bf16(sum as f32)? + shared)?);
        }
        result.ids.extend(ids.into_iter().map(|id| id as i32));
        result.gates.extend(gates);
    }
    let mut order = (0..result.ids.len()).collect::<Vec<_>>();
    order.sort_by_key(|&pair| (result.ids[pair], pair));
    result.inverse = vec![0; order.len()];
    for (sorted, &original) in order.iter().enumerate() {
        result.inverse[original] = sorted as u32;
    }
    result.sorted_pairs = order.into_iter().map(|pair| pair as u32).collect();
    Ok(result)
}

fn validate_forced(
    fixture: &Fixture,
    rows: usize,
    ids: &[i32],
    gates: &[f32],
) -> Result<(), String> {
    if rows.checked_mul(6) != Some(ids.len())
        || gates.len() != ids.len()
        || ids
            .iter()
            .any(|&id| id < 0 || id as usize >= fixture.experts.len())
        || gates
            .iter()
            .any(|&gate| !gate.is_finite() || !(0.0..=1.0).contains(&gate))
    {
        return Err("invalid forced routing shape, expert ID or weight".into());
    }
    for (ids, gates) in ids.as_chunks::<6>().0.iter().zip(gates.as_chunks::<6>().0) {
        if ids
            .iter()
            .enumerate()
            .any(|(rank, id)| ids[..rank].contains(id))
            || gates.iter().map(|&gate| f64::from(gate)).sum::<f64>() <= 0.0
        {
            return Err("duplicate forced expert or zero routing mass".into());
        }
    }
    Ok(())
}

fn published_input(fixture: &Fixture, input: &[f32], rows: usize) -> Result<Vec<f32>, String> {
    if rows == 0 || fixture.hidden == 0 || rows.checked_mul(fixture.hidden) != Some(input.len()) {
        return Err("invalid real FFN block geometry".into());
    }
    input.iter().map(|&value| bf16(value)).collect()
}

pub(super) fn run_forced(
    fixture: &Fixture,
    input: &[f32],
    ids: &[i32],
    gates: &[f32],
) -> Result<Expected, String> {
    let rows = input.len().checked_div(fixture.hidden).unwrap_or(0);
    let input = published_input(fixture, input, rows)?;
    moe_rows(fixture, &input, Some((ids, gates)))
}

pub(super) fn run(
    fixture: &Fixture,
    input: &[f32],
    rows: usize,
    kind: Kind,
) -> Result<Expected, String> {
    let input = published_input(fixture, input, rows)?;
    if let Kind::Moe = kind {
        return moe_rows(fixture, &input, None);
    }
    validate_expert(&fixture.dense, fixture.hidden)?;
    let mut result = Expected::default();
    for token in input.chunks_exact(fixture.hidden) {
        let (hidden, down) = expert(token, &fixture.dense)?;
        track(&mut result, &hidden);
        result.output.extend(down);
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cancelling_fixture() -> Fixture {
        let dense = Expert {
            gate: Matrix {
                rows: 1,
                cols: 3,
                values: vec![1.0; 3],
                factor: 1.0,
            },
            up: Matrix {
                rows: 1,
                cols: 3,
                values: vec![0.0, 1.0, 0.0],
                factor: 1.0,
            },
            down: Matrix {
                rows: 3,
                cols: 1,
                values: vec![1.0; 3],
                factor: 1.0,
            },
        };
        Fixture {
            hidden: 3,
            router: Matrix {
                rows: 8,
                cols: 3,
                values: vec![0.0; 24],
                factor: 1.0,
            },
            experts: vec![dense.clone(); 8],
            shared: dense.clone(),
            dense,
        }
    }

    #[test]
    fn dense_ffn_retains_a_cancelling_gate_projection() {
        let actual = run(
            &cancelling_fixture(),
            &[16_777_216.0, 1.0, -16_777_216.0],
            1,
            Kind::Dense,
        )
        .unwrap();
        assert_eq!(actual.output, vec![0.73046875; 3]);
    }

    #[test]
    fn moe_keeps_top_six_tie_order_and_bf16_combine_publications() {
        let actual = run(
            &cancelling_fixture(),
            &[16_777_216.0, 1.0, -16_777_216.0],
            1,
            Kind::Moe,
        )
        .unwrap();
        assert_eq!(actual.ids, vec![0, 1, 2, 3, 4, 5]);
        assert_eq!(actual.gates, vec![1.0 / 6.0; 6]);
        assert_eq!(actual.sorted_pairs, vec![0, 1, 2, 3, 4, 5]);
        assert_eq!(actual.inverse, vec![0, 1, 2, 3, 4, 5]);
        assert_eq!(actual.output, vec![1.46875; 3]);
    }

    #[test]
    fn forced_routes_preserve_identity_and_change_the_selected_expert() {
        let mut fixture = cancelling_fixture();
        fixture.experts[7].down.values.fill(2.0);
        let input = [16_777_216.0, 1.0, -16_777_216.0];
        let original = run(&fixture, &input, 1, Kind::Moe).unwrap();
        let same = run_forced(&fixture, &input, &original.ids, &original.gates).unwrap();
        assert_eq!(same.output, original.output);
        assert_eq!(same.ids, original.ids);
        let ids = [7, 1, 2, 3, 4, 5];
        let changed = run_forced(&fixture, &input, &ids, &original.gates).unwrap();
        assert_eq!(changed.ids, ids);
        assert_eq!(changed.sorted_pairs, [1, 2, 3, 4, 5, 0]);
        assert_ne!(changed.output, original.output);
        for bad_ids in [[-1, 1, 2, 3, 4, 5], [8, 1, 2, 3, 4, 5], [0, 0, 2, 3, 4, 5]] {
            assert!(run_forced(&fixture, &input, &bad_ids, &original.gates).is_err());
        }
        assert!(run_forced(&fixture, &input, &ids[..5], &original.gates).is_err());
        assert!(run_forced(&fixture, &input, &ids, &[0.0; 6]).is_err());
        assert!(run_forced(&fixture, &input, &ids, &[f32::NAN; 6]).is_err());
        assert!(run_forced(&fixture, &input, &ids, &[2.0; 6]).is_err());
    }

    #[test]
    fn projection_does_not_lose_the_low_magnitude_term() {
        let matrix = Matrix {
            rows: 1,
            cols: 3,
            values: vec![1.0; 3],
            factor: 2.75,
        };
        assert_eq!(
            project(&[16_777_216.0, 1.0, -16_777_216.0], &matrix).unwrap(),
            vec![2.75],
        );
    }
}
