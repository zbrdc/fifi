use super::*;
use crate::family::flint::cuda_ffn::reference::real::gguf;
use std::io::BufReader;

#[test]
#[ignore = "requires the pinned real Flint GGUF; CPU-only, one expert, no CUDA device"]
fn real_prepared_selector_borrows_through_traced_without_cloning() {
    let file = File::open(std::env::var("FIFI_FLINT_GGUF").expect("pinned GGUF")).unwrap();
    let identity = serde_json::from_reader(
        File::open(std::env::var("FIFI_FLINT_MODEL_IDENTITY").expect("identity receipt")).unwrap(),
    )
    .unwrap();
    gguf::verify_prepared_identity(&file, &identity).unwrap();
    let header = GgufHeader::parse(
        &mut BufReader::new(&file),
        file.metadata().unwrap().len(),
        64 << 20,
    )
    .unwrap();
    let id = ExpertId {
        layer: 2,
        expert: 1,
    };
    let mut original = GgufSource::new(&file, &header).unwrap();
    let expected: [ExpertPayload; 3] =
        std::array::from_fn(|part| original.payload(id, part).unwrap());
    let mut prepared = streaming::PreparedSource::new(&file, &header, &[id], 16 << 20).unwrap();
    let pointers: Vec<_> = (0..3)
        .map(|part| {
            prepared
                .with_payload(id, part, |view| {
                    Ok((view.data.as_ptr(), view.scales.as_ptr()))
                })
                .unwrap()
        })
        .collect();
    let mut source = SelectedSource::Prepared(prepared);
    let mut stats = FetchStats::default();
    let mut traced = Traced {
        source: &mut source,
        stats: &mut stats,
    };
    for (part, expected) in expected.iter().enumerate() {
        traced
            .with_payload(id, part, |view| {
                assert_eq!(view, expected.as_view());
                assert_eq!(view.shape, expected.shape);
                assert_eq!(view.tensor_scale, expected.tensor_scale);
                assert_eq!(view.data.as_ptr(), pointers[part].0, "selector cloned data");
                assert_eq!(
                    view.scales.as_ptr(),
                    pointers[part].1,
                    "selector cloned scales"
                );
                Ok(())
            })
            .unwrap();
    }
    let error: Result<(), EngError> = traced.with_payload(id, 1, |_| {
        Err(EngError::BadRequest("prepared consumer sentinel".into()))
    });
    assert!(
        matches!(error, Err(EngError::BadRequest(message)) if message == "prepared consumer sentinel")
    );
    let error: Result<(), EngError> =
        traced.with_payload(id, 3, |_| panic!("invalid prepared part reached consumer"));
    assert!(matches!(error, Err(EngError::BadRequest(_))));
    let error: Result<(), EngError> = traced.with_payload(
        ExpertId {
            layer: id.layer,
            expert: id.expert + 1,
        },
        0,
        |_| panic!("missing prepared expert reached consumer"),
    );
    assert!(matches!(error, Err(EngError::BadRequest(_))));

    // GGUF intentionally materializes once per callback; it has no retained canonical pointer.
    let mut source = SelectedSource::Gguf(GgufSource::new(&file, &header).unwrap());
    let mut stats = FetchStats::default();
    let mut traced = Traced {
        source: &mut source,
        stats: &mut stats,
    };
    for (part, expected) in expected.iter().enumerate() {
        traced
            .with_payload(id, part, |view| {
                assert_eq!(view, expected.as_view());
                Ok(())
            })
            .unwrap();
    }
    assert_eq!(
        traced.stats.payload_bytes,
        expected
            .iter()
            .map(|part| part.data.len() + part.scales.len())
            .sum::<usize>()
    );
    let error: Result<(), EngError> = traced.with_payload(id, 0, |_| {
        Err(EngError::BadRequest("GGUF consumer sentinel".into()))
    });
    assert!(
        matches!(error, Err(EngError::BadRequest(message)) if message == "GGUF consumer sentinel")
    );
    let error: Result<(), EngError> =
        traced.with_payload(id, 3, |_| panic!("invalid GGUF part reached consumer"));
    assert!(matches!(error, Err(EngError::BadRequest(_))));
    gguf::verify_prepared_identity(&file, &identity).unwrap();
}

#[test]
fn prepared_switch_and_complete_id_plan_are_strict() {
    assert!(!prepared_flag(Err(std::env::VarError::NotPresent)).unwrap());
    assert!(!prepared_flag(Ok("0".into())).unwrap());
    assert!(prepared_flag(Ok("1".into())).unwrap());
    for value in ["", "2", "01", "false", "true", "1\n"] {
        assert!(matches!(
            prepared_flag(Ok(value.into())),
            Err(EngError::BadRequest(_))
        ));
    }
    assert!(matches!(
        prepared_flag(Err(std::env::VarError::NotUnicode("invalid".into()))),
        Err(EngError::BadRequest(_))
    ));
    let ids = prepared_ids();
    assert_eq!(ids.len(), 12_032);
    for (index, id) in ids.iter().enumerate() {
        assert_eq!(
            *id,
            ExpertId {
                layer: 2 + (index / 256) as u32,
                expert: (index % 256) as u32
            }
        );
    }
    assert_eq!(PREPARED_HOST_BUDGET, 30_064_771_072);
    let mut empty = [0u8; 0];
    let mut unwritable = std::io::Cursor::new(&mut empty[..]);
    assert!(matches!(
        first_step_marker(&mut unwritable),
        Err(EngError::Checkpoint(_))
    ));
}

fn prepare_records(output: &[u8]) -> Vec<serde_json::Value> {
    assert_eq!(output.last(), Some(&b'\n'));
    std::str::from_utf8(output)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line.strip_prefix("SELECTED_PREPARE ").unwrap()).unwrap())
        .collect()
}

fn assert_first_step_follows_prepare(progress: &mut Vec<u8>) {
    let records = prepare_records(progress);
    assert!(matches!(
        records.last().unwrap()["event"].as_str(),
        Some("end" | "disabled")
    ));
    let expected = [
        progress.as_slice(),
        b"SELECTED_STEP_BEGIN {\"position\":0}\n",
    ]
    .concat();
    first_step_marker(progress).unwrap();
    assert_eq!(*progress, expected);
}

#[test]
#[ignore = "requires the pinned real Flint GGUF; CPU-only, one expert, no CUDA device"]
fn real_selected_prepare_markers_and_errors_fail_closed() {
    let file = File::open(std::env::var("FIFI_FLINT_GGUF").expect("pinned GGUF")).unwrap();
    let identity = serde_json::from_reader(
        File::open(std::env::var("FIFI_FLINT_MODEL_IDENTITY").expect("identity receipt")).unwrap(),
    )
    .unwrap();
    gguf::verify_prepared_identity(&file, &identity).unwrap();
    let header = GgufHeader::parse(
        &mut BufReader::new(&file),
        file.metadata().unwrap().len(),
        64 << 20,
    )
    .unwrap();
    let id = ExpertId {
        layer: 2,
        expert: 1,
    };
    let budget = 16 << 20;
    let begin = serde_json::json!({
        "event":"begin", "prepared_enabled":true, "experts":1, "host_budget_bytes":budget,
    });
    let mut progress = Vec::new();
    let prepared = prepare_experts((&file, &header), &[id], budget, &mut progress).unwrap();
    let records = prepare_records(&progress);
    assert_eq!(records.len(), 2);
    let begin_len = progress.iter().position(|&byte| byte == b'\n').unwrap() + 1;
    let ns = records[1]["prepare_ns"].as_u64().unwrap();
    assert!(ns > 0);
    assert_eq!(
        records,
        [
            begin.clone(),
            serde_json::json!({
                "event":"end", "prepared_enabled":true, "experts":1, "host_budget_bytes":budget,
                "retained_bytes":prepared.retained_host_bytes(), "prepare_ns":ns,
            })
        ]
    );
    assert_first_step_follows_prepare(&mut progress);

    progress.clear();
    let source = SelectedSource::new((&file, &header), false, &mut progress).unwrap();
    assert!(!source.prepared_enabled());
    assert!(matches!(source, SelectedSource::Gguf(_)));
    assert_eq!(
        prepare_records(&progress),
        [serde_json::json!({
            "event":"disabled", "prepared_enabled":false, "experts":0,
            "retained_bytes":0, "prepare_ns":0,
        })]
    );
    assert_first_step_follows_prepare(&mut progress);

    progress.clear();
    let result = prepare_experts((&file, &header), &[id], 1, &mut progress);
    assert!(matches!(result, Err(EngError::MemoryPlan(_))));
    assert_eq!(
        prepare_records(&progress),
        [serde_json::json!({
            "event":"begin", "prepared_enabled":true, "experts":1, "host_budget_bytes":1,
        })]
    );

    progress.clear();
    let result = prepare_experts(
        (&file, &header),
        &[ExpertId {
            layer: 1,
            expert: 0,
        }],
        budget,
        &mut progress,
    );
    assert!(matches!(result, Err(EngError::BadRequest(_))));
    assert_eq!(prepare_records(&progress), std::slice::from_ref(&begin));

    progress.clear();
    let empty = File::open("/dev/null").unwrap();
    let result = prepare_experts((&empty, &header), &[id], budget, &mut progress);
    assert!(
        matches!(result, Err(EngError::Checkpoint(message)) if message.contains("Flint expert read"))
    );
    assert_eq!(prepare_records(&progress), [begin]);

    let mut empty = [0u8; 0];
    let mut unwritable = std::io::Cursor::new(&mut empty[..]);
    let result = prepare_experts((&file, &header), &[id], budget, &mut unwritable);
    assert!(
        matches!(result, Err(EngError::Checkpoint(message)) if message.contains("Flint prepare progress"))
    );
    let mut begin_only = vec![0u8; begin_len];
    let mut unwritable_end = std::io::Cursor::new(begin_only.as_mut_slice());
    let result = prepare_experts((&file, &header), &[id], budget, &mut unwritable_end);
    assert!(
        matches!(result, Err(EngError::Checkpoint(message)) if message.contains("Flint prepare progress"))
    );
    assert_eq!(prepare_records(&begin_only)[0]["event"], "begin");
    gguf::verify_prepared_identity(&file, &identity).unwrap();
}
