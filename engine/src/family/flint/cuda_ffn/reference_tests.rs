use super::*;

#[test]
fn bf16_reference_checks_ties_zero_wide_values_and_overflow() {
    assert_eq!(
        bf16(f32::from_bits(0x3f808000)).unwrap().to_bits(),
        0x3f800000
    );
    assert_eq!(
        bf16(f32::from_bits(0x3f818000)).unwrap().to_bits(),
        0x3f820000
    );
    assert_eq!(bf16(-0.0).unwrap().to_bits(), (-0.0f32).to_bits());
    assert_eq!(bf16(131072.0).unwrap(), 131072.0);
    for bad in [f32::MAX, f32::INFINITY, f32::NAN] {
        assert!(bf16(bad).is_err());
    }
}

#[test]
fn sigmoid_selection_uses_score_ties_not_logit_order() {
    let (ids, gates) = route(&[100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0]).unwrap();
    assert_eq!(ids, [0, 1, 2, 3, 4, 5]);
    assert_eq!(gates, [1.0 / 6.0; 6]);
    assert!(route(&[-1000.0; 8]).is_err());
    let mut invalid = [0.0; 8];
    invalid[7] = f32::INFINITY;
    assert!(route(&invalid).is_err());
}

#[test]
fn silu_requires_its_sigmoid_publication() {
    let differs = (0x3d00u32..0x4080).any(|bits| {
        let gate = f32::from_bits(bits << 16);
        let staged = bf16(gate * bf16(sigmoid(gate)).unwrap()).unwrap();
        let fused = bf16(gate * sigmoid(gate)).unwrap();
        staged.to_bits() != fused.to_bits()
    });
    assert!(differs);
}

#[test]
fn rank_sum_is_one_publication_not_six_bf16_additions() {
    let products = [256.0f32, 1.0, 1.0, 1.0, 1.0, -256.0];
    let one_sum = bf16(products.iter().sum()).unwrap();
    let six_sums = products
        .iter()
        .fold(0.0, |sum, &value| bf16(sum + value).unwrap());
    assert_eq!(bf16(one_sum + 2.0).unwrap(), 6.0);
    assert_eq!(bf16(six_sums + 2.0).unwrap(), 2.0);
}

fn matrix(value: f32) -> Matrix {
    Matrix {
        rows: 1,
        cols: 1,
        values: vec![value],
        factor: 1.0,
    }
}

fn weights() -> Expert {
    Expert {
        gate: matrix(16.0),
        up: matrix(2.0),
        down: matrix(1.0),
    }
}

fn fixture() -> Fixture {
    Fixture {
        hidden: 1,
        router: Matrix {
            rows: 8,
            cols: 1,
            values: vec![0.0; 8],
            factor: 1.0,
        },
        experts: vec![weights(); 8],
        shared: weights(),
        dense: weights(),
    }
}

#[test]
fn complete_dense_and_moe_oracles_have_ungated_shared_and_original_pair_order() {
    let f = fixture();
    for rows in [1, 7, 17, 512] {
        let dense = run(&f, &vec![1.0; rows], rows, Kind::Dense).unwrap();
        assert_eq!(dense.output, vec![32.0; rows]);
        assert!(dense.ids.is_empty() && dense.gates.is_empty());
        assert_eq!(dense.max_intermediate, 32.0);
        let moe = run(&f, &vec![1.0; rows], rows, Kind::Moe).unwrap();
        assert_eq!(moe.output, vec![64.0; rows]);
        assert_eq!(moe.ids, (0..rows).flat_map(|_| 0..6).collect::<Vec<_>>());
        assert_eq!(moe.gates, vec![1.0 / 6.0; rows * 6]);
        assert_eq!(moe.max_intermediate, 32.0);
        for (sorted, &original) in moe.sorted_pairs.iter().enumerate() {
            assert_eq!(moe.inverse[original as usize], sorted as u32);
        }
    }
}

#[test]
fn source_factor_is_once_and_bad_reference_inputs_refuse() {
    let mut f = fixture();
    f.dense.down.factor = 0.25;
    assert_eq!(run(&f, &[1.0], 1, Kind::Dense).unwrap().output, [8.0]);
    assert!(run(&f, &[], 1, Kind::Dense).is_err());
    assert!(run(&f, &[f32::NAN], 1, Kind::Dense).is_err());
    f.router.values.fill(-1000.0);
    assert!(run(&f, &[1.0], 1, Kind::Moe).is_err());
    f.dense.gate.values.clear();
    assert!(run(&f, &[1.0], 1, Kind::Dense).is_err());
}
