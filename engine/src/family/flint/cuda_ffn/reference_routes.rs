//! Routing attribution for the six sealed MoE miss cases; never a qualification oracle.

use super::*;
use crate::family::flint::cuda_ffn::real_model;
use serde_json::Value;
use std::path::PathBuf;

pub(in crate::family::flint::cuda_ffn) const PAIRS: [(usize, usize); 6] =
    [(2, 2), (8, 2), (10, 1), (15, 4), (26, 1), (27, 1)];

pub(in crate::family::flint::cuda_ffn) const TAIL_PAIRS: [(usize, usize); 8] = [
    (30, 3),
    (30, 6),
    (30, 7),
    (31, 1),
    (31, 7),
    (40, 0),
    (46, 6),
    (47, 0),
];

fn oracle() -> (real_model::Oracle, String) {
    let path = std::env::var("FIFI_FLINT_FULL_ORACLE").expect("frozen oracle");
    let sha = std::env::var("FIFI_FLINT_FULL_ORACLE_SHA256").expect("frozen oracle SHA256");
    (
        real_model::Oracle::parse(&std::fs::read(path).unwrap(), &sha).unwrap(),
        sha,
    )
}

fn directory(name: &str) -> PathBuf {
    let path = PathBuf::from(std::env::var(name).expect("routing evidence directory"));
    assert!(path.is_dir());
    path
}

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|value| value.to_bits()).collect()
}

pub(in crate::family::flint::cuda_ffn) fn decisions(
    data: &Value,
    rows: usize,
) -> Result<(Vec<i32>, Vec<f32>), String> {
    let ids: Vec<i32> = serde_json::from_value(data["ids"].clone()).map_err(|e| e.to_string())?;
    let gates: Vec<f32> =
        serde_json::from_value(data["gates"].clone()).map_err(|e| e.to_string())?;
    if !(1..=5).contains(&rows)
        || ids.len() != rows * 6
        || gates.len() != ids.len()
        || gates
            .iter()
            .any(|v| !v.is_finite() || !(0.0..=1.0).contains(v))
    {
        return Err("invalid route counts or gates".into());
    }
    for (ids, gates) in ids.as_chunks::<6>().0.iter().zip(gates.as_chunks::<6>().0) {
        if ids.iter().any(|id| !(0..256).contains(id))
            || ids.iter().collect::<std::collections::BTreeSet<_>>().len() != 6
            || gates.iter().sum::<f32>() <= 0.0
        {
            return Err("invalid expert range, duplicate, or empty gate row".into());
        }
    }
    Ok((ids, gates))
}

fn names(pairs: &[(usize, usize)], kind: &str) -> std::collections::BTreeSet<String> {
    assert!(!pairs.is_empty());
    assert!(
        pairs
            .iter()
            .all(|&(layer, case)| (2..49).contains(&layer) && case < 8)
    );
    let names = pairs
        .iter()
        .map(|(layer, case)| format!("{kind}-{layer}-p{:02}.json", case + 1))
        .collect::<std::collections::BTreeSet<_>>();
    assert_eq!(names.len(), pairs.len(), "duplicate routing case");
    names
}

pub(in crate::family::flint::cuda_ffn) fn manifest(
    root: &std::path::Path,
    kind: &str,
    pairs: &[(usize, usize)],
    oracle: (&real_model::Oracle, &str),
) {
    let expected = names(pairs, kind);
    let actual = std::fs::read_dir(root)
        .unwrap()
        .map(|entry| {
            let entry = entry.unwrap();
            assert!(entry.file_type().unwrap().is_file());
            entry.file_name().into_string().unwrap()
        })
        .collect::<std::collections::BTreeSet<_>>();
    assert_eq!(actual, expected, "incomplete or mixed routing collection");
    let files = expected
        .iter()
        .map(|name| {
            let bytes = std::fs::read(root.join(name)).unwrap();
            serde_json::json!({"name":name,"sha256":format!("{:x}",Sha256::digest(bytes))})
        })
        .collect::<Vec<_>>();
    let oracle_bytes = std::fs::read(std::env::var("FIFI_FLINT_FULL_ORACLE").unwrap()).unwrap();
    assert_eq!(format!("{:x}", Sha256::digest(oracle_bytes)), oracle.1);
    let executable = std::fs::read(std::env::current_exe().unwrap()).unwrap();
    real_model::write_new(
        &root.join("manifest.json"),
        &serde_json::json!({
            "schema_version":1,"diagnostic_only":true,"qualification":false,"kind":kind,
            "model_sha256":MODEL_SHA256,"prefixes_sha256":real_model::PREFIX_SHA256,
            "oracle_sha256":oracle.1,"executable_sha256":format!("{:x}",Sha256::digest(executable)),
            "cases":pairs,"case_count":pairs.len(),
            "row_count":pairs.iter().map(|&(_,case)|oracle.0.token_ids[case].len()).sum::<usize>(),
            "files":files,"complete":true
        }),
    );
}

#[test]
fn route_transport_rejects_counts_range_duplicates_and_empty_gates() {
    let valid = serde_json::json!({"ids":[0,1,2,3,4,255],"gates":[0.1,0.1,0.1,0.1,0.1,0.5]});
    assert!(decisions(&valid, 1).is_ok());
    for (field, value) in [
        ("ids", serde_json::json!([0, 1, 2, 3, 4])),
        ("ids", serde_json::json!([0, 1, 2, 3, 4, 256])),
        ("ids", serde_json::json!([0, 1, 2, 3, 4, 4])),
        ("gates", serde_json::json!([0, 0, 0, 0, 0, 0])),
        ("gates", serde_json::json!([1, 0, 0, 0, 0, -0.1])),
    ] {
        let mut invalid = valid.clone();
        invalid[field] = value;
        assert!(decisions(&invalid, 1).is_err());
    }
    assert!(decisions(&valid, 0).is_err());
    assert!(decisions(&valid, 2).is_err());
    assert_eq!(names(&TAIL_PAIRS, "device-routing").len(), 8);
    assert!(std::panic::catch_unwind(|| names(&[(49, 0)], "x")).is_err());
    assert!(std::panic::catch_unwind(|| names(&[(2, 0), (2, 0)], "x")).is_err());
}

fn published(data: &Value, field: &str, rows: usize) -> Vec<f32> {
    let bits: Vec<u32> = serde_json::from_value(data[field].clone()).unwrap();
    assert_eq!(bits.len(), rows * 2048);
    assert!(
        bits.iter()
            .all(|&v| v & 0xffff == 0 && f32::from_bits(v).is_finite())
    );
    bits.into_iter().map(f32::from_bits).collect()
}

#[test]
#[ignore = "requires pinned GGUF/oracle and an admitted CPU release run; six miss cases only"]
fn freeze_moe_miss_routing() {
    freeze_routing(&PAIRS);
}

#[test]
#[ignore = "requires admitted CPU release run; only eight new tail miss cases"]
fn freeze_tail_moe_miss_routing() {
    freeze_routing(&TAIL_PAIRS);
}

fn freeze_routing(pairs: &[(usize, usize)]) {
    let (oracle, sha) = oracle();
    let (file, header, identity) = real_model::model();
    let root = real_model::fresh_directory(
        "FIFI_FLINT_RCA_RESULT_DIR",
        &[PathBuf::from(
            std::env::var("FIFI_FLINT_FULL_ORACLE").unwrap(),
        )],
    );
    for &(index, case) in pairs {
        let layer = load(&file, &header, index as u32).unwrap();
        let state = &oracle.layers[index][case];
        let rows = oracle.token_ids[case].len();
        let input = state
            .input_bits
            .iter()
            .map(|&v| f32::from_bits(v))
            .collect::<Vec<_>>();
        let (residual, normalized) = super::super::prefix(&layer.view(), &input, rows).unwrap();
        let expected = super::super::run(&layer.view(), &input, rows).unwrap();
        assert_eq!(
            bits(&expected.output),
            state.output_bits,
            "original reference changed"
        );
        assert!(
            layer
                .ffn
                .router
                .values
                .iter()
                .all(|v| v.to_bits() & 0xffff == 0)
        );
        let logits = normalized
            .as_chunks::<2048>()
            .0
            .iter()
            .flat_map(|row| super::super::ffn::project(row, &layer.ffn.router).unwrap())
            .collect::<Vec<_>>();
        let scores = logits
            .iter()
            .map(|&v| 1.0 / (1.0 + (-f64::from(v)).exp()))
            .collect::<Vec<_>>();
        assert_eq!(logits.len(), rows * 256);
        assert_eq!(expected.ids.len(), rows * 6);
        verify_prepared_identity(&file, &identity).unwrap();
        real_model::write_new(
            &root.join(format!("reference-routing-{index}-p{:02}.json", case + 1)),
            &serde_json::json!({
                "schema_version":1,"diagnostic_only":true,"qualification":false,
                "layer":index,"case":format!("p{:02}",case+1),"rows":rows,"experts":256,
                "oracle_sha256":sha,"model_sha256":MODEL_SHA256,
                "prefixes_sha256":real_model::PREFIX_SHA256,"input_bits":state.input_bits,
                "residual_bits":bits(&residual),"post_norm_bits":bits(&normalized),
                "logits_bits":bits(&logits),"scores_fp64":scores,
                "ids":expected.ids,"gates":expected.gates,"bias":null,
                "original_output_bit_exact":true,
                "notes":"Original FP64 reference with its existing BF16/F16 publications. Counts cover this case only, not all31 rows of the layer."
            }),
        );
        println!(
            "RCA_REFERENCE_ROUTING layer={index} case={} rows={rows} original_bit_exact=true",
            case + 1
        );
    }
    manifest(&root, "reference-routing", pairs, (&oracle, &sha));
}

#[test]
#[ignore = "requires sealed reference+CUDA routing records and admitted CPU release run"]
fn force_cuda_routes_into_reference() {
    force_routes(&PAIRS);
}

#[test]
#[ignore = "requires admitted CPU run and new tail reference/device routing records"]
fn force_tail_cuda_routes_into_reference() {
    force_routes(&TAIL_PAIRS);
}

fn force_routes(pairs: &[(usize, usize)]) {
    let (oracle, sha) = oracle();
    let (file, header, identity) = real_model::model();
    let references = directory("FIFI_FLINT_ROUTE_REFERENCE");
    let devices = directory("FIFI_FLINT_ROUTE_DEVICE");
    let root = real_model::fresh_directory(
        "FIFI_FLINT_RCA_RESULT_DIR",
        &[
            references.clone(),
            devices.clone(),
            PathBuf::from(std::env::var("FIFI_FLINT_FULL_ORACLE").unwrap()),
        ],
    );
    for &(index, case) in pairs {
        let name = format!("{index}-p{:02}.json", case + 1);
        let reference_bytes =
            std::fs::read(references.join(format!("reference-routing-{name}"))).unwrap();
        let reference: Value = serde_json::from_slice(&reference_bytes).unwrap();
        let device_bytes = std::fs::read(devices.join(format!("device-routing-{name}"))).unwrap();
        let device: Value = serde_json::from_slice(&device_bytes).unwrap();
        let rows = oracle.token_ids[case].len();
        for data in [&reference, &device] {
            assert_eq!(data["layer"], index);
            assert_eq!(data["case"], format!("p{:02}", case + 1));
            assert_eq!(data["rows"], rows);
            assert_eq!(data["model_sha256"], MODEL_SHA256);
            assert_eq!(data["oracle_sha256"], sha);
            assert_eq!(
                data["input_bits"],
                serde_json::json!(oracle.layers[index][case].input_bits)
            );
        }
        for data in [&reference, &device] {
            assert_eq!(data["prefixes_sha256"], real_model::PREFIX_SHA256);
            assert_eq!(data["experts"], 256);
            assert_eq!(data["diagnostic_only"], true);
            assert_eq!(data["qualification"], false);
            decisions(data, rows).unwrap();
        }
        assert_eq!(device["saved_partial_bit_exact"], true);
        assert_eq!(device["own_route_identity_bit_exact"], true);
        assert_eq!(reference["original_output_bit_exact"], true);
        assert_eq!(
            bits(&published(&device, "forced_own_output_bits", rows)),
            bits(&published(&device, "baseline_output_bits", rows))
        );
        assert_eq!(
            device["reference_sha256"],
            format!("{:x}", Sha256::digest(&reference_bytes))
        );
        assert!(device["instrumented_matches_untraced"].as_bool().unwrap());
        let (ids, gates) = decisions(&device, rows).unwrap();
        let layer = load(&file, &header, index as u32).unwrap();
        let cpu_normalized = published(&reference, "post_norm_bits", rows);
        let cpu_residual = published(&reference, "residual_bits", rows);
        let cuda_normalized = published(&device, "post_norm_bits", rows);
        let cuda_residual = published(&device, "residual_bits", rows);
        let swapped =
            super::super::ffn::run_forced(&layer.ffn, &cpu_normalized, &ids, &gates).unwrap();
        let swapped = super::super::add(&cpu_residual, &swapped.output).unwrap();
        let local =
            super::super::ffn::run_forced(&layer.ffn, &cuda_normalized, &ids, &gates).unwrap();
        let local = super::super::add(&cuda_residual, &local.output).unwrap();
        let original_ids: Vec<i32> = serde_json::from_value(reference["ids"].clone()).unwrap();
        let original_gates: Vec<f32> = serde_json::from_value(reference["gates"].clone()).unwrap();
        let identity_control = super::super::ffn::run_forced(
            &layer.ffn,
            &cpu_normalized,
            &original_ids,
            &original_gates,
        )
        .unwrap();
        let identity_control = super::super::add(&cpu_residual, &identity_control.output).unwrap();
        assert_eq!(
            bits(&identity_control),
            oracle.layers[index][case].output_bits
        );
        verify_prepared_identity(&file, &identity).unwrap();
        assert_eq!(
            std::fs::read(references.join(format!("reference-routing-{name}"))).unwrap(),
            reference_bytes
        );
        assert_eq!(
            std::fs::read(devices.join(format!("device-routing-{name}"))).unwrap(),
            device_bytes
        );
        real_model::write_new(
            &root.join(format!("forced-reference-{name}")),
            &serde_json::json!({
                "schema_version":1,"diagnostic_only":true,"qualification":false,
                "layer":index,"case":format!("p{:02}",case+1),"rows":rows,
                "model_sha256":MODEL_SHA256,"oracle_sha256":sha,"prefixes_sha256":real_model::PREFIX_SHA256,
                "device_sha256":format!("{:x}",Sha256::digest(&device_bytes)),
                "reference_sha256":format!("{:x}",Sha256::digest(&reference_bytes)),
                "fp64_on_cpu_prefix_cuda_routes_bits":bits(&swapped),
                "fp64_on_cuda_prefix_cuda_routes_bits":bits(&local),"identity_forced_bit_exact":true
            }),
        );
        println!(
            "RCA_FORCED_REFERENCE layer={index} case={} identity_bit_exact=true",
            case + 1
        );
    }
    manifest(&root, "forced-reference", pairs, (&oracle, &sha));
}
