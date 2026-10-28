use super::super::source::GgufSource;
use crate::{
    EngError,
    expert_tier::{ExpertId, ExpertPayload, ExpertPayloadSource},
    family::flint::{cuda_ffn::reference::real::gguf, gguf_weights},
    weights::GgufHeader,
};
use std::{fs::File, io::BufReader, os::unix::fs::FileExt};

use super::super::PreparedSource;

#[test]
fn accounting_and_allocation_refuse_overflow() {
    assert!(matches!(
        super::reserve::<u8>(usize::MAX),
        Err(EngError::MemoryPlan(_))
    ));
    assert!(matches!(
        super::charge(0, usize::MAX, 2),
        Err(EngError::MemoryPlan(_))
    ));
    assert!(matches!(
        super::charge(u64::MAX, 1, 1),
        Err(EngError::MemoryPlan(_))
    ));
    assert_eq!(super::charge(3, 4, 5).unwrap(), 23);
    assert!(super::check_budget(23, 22).is_err());
    super::check_budget(23, 23).unwrap();
    super::check_budget(23, 24).unwrap();
}

#[test]
fn ids_require_unique_valid_routed_experts() {
    let a = ExpertId {
        layer: 2,
        expert: 1,
    };
    let b = ExpertId {
        layer: 48,
        expert: 255,
    };
    assert_eq!(super::ordered_ids(&[b, a]).unwrap(), [a, b]);
    assert!(super::ordered_ids(&[a, a]).is_err());
    for id in [
        ExpertId {
            layer: 1,
            expert: 0,
        },
        ExpertId {
            layer: 49,
            expert: 0,
        },
        ExpertId {
            layer: 2,
            expert: 256,
        },
    ] {
        assert!(super::ordered_ids(&[id]).is_err());
    }
}

fn real_expert() -> (File, GgufHeader, ExpertId, [ExpertPayload; 3]) {
    let id = ExpertId {
        layer: 2,
        expert: 1,
    };
    let (file, header, mut expected) = real_experts(&[id]);
    (file, header, id, expected.pop().unwrap())
}

fn real_experts(ids: &[ExpertId]) -> (File, GgufHeader, Vec<[ExpertPayload; 3]>) {
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
    let mut original = GgufSource::new(&file, &header).unwrap();
    let expected = ids
        .iter()
        .map(|&id| std::array::from_fn(|part| original.payload(id, part).unwrap()))
        .collect();

    // Copy only the requested experts; the model stays read-only.
    let scratch = tempfile::tempfile().unwrap();
    for &id in ids {
        let sources = gguf_weights::expert_sources(&header, id.layer, id.expert).unwrap();
        for source in [sources.gate, sources.up, sources.down] {
            for range in [source.data, source.scale] {
                let mut raw = vec![0; (range.end - range.start) as usize];
                file.read_exact_at(&mut raw, range.start).unwrap();
                scratch.write_all_at(&raw, range.start).unwrap();
            }
        }
    }
    gguf::verify_prepared_identity(&file, &identity).unwrap();
    (scratch, header, expected)
}

fn assert_borrowed_parts(
    prepared: &mut PreparedSource,
    id: ExpertId,
    expected: &[ExpertPayload; 3],
) {
    assert_eq!(prepared.parts(), 3);
    for _ in 0..2 {
        for (part, expected) in expected.iter().enumerate() {
            let data = prepared.entries[0].parts[part].data.as_ptr();
            let scales = prepared.entries[0].parts[part].scales.as_ptr();
            prepared
                .with_payload(id, part, |view| {
                    assert_eq!(view, expected.as_view());
                    assert_eq!(view.data.as_ptr(), data);
                    assert_eq!(view.scales.as_ptr(), scales);
                    Ok(())
                })
                .expect("prepared bytes must not reread the truncated source");
        }
    }
}

#[test]
#[ignore = "requires the pinned real Flint GGUF; CPU-only, one expert, no CUDA device"]
fn real_expert_survives_source_truncation_after_prepare() {
    let (scratch, header, id, expected) = real_expert();
    let mut prepared = PreparedSource::new(&scratch, &header, &[id], 16 << 20).unwrap();
    scratch.set_len(0).unwrap();
    drop(scratch);
    drop(header);
    assert_borrowed_parts(&mut prepared, id, &expected);
    let charged = prepared.retained_host_bytes();
    prepared.release_experts(&[id]).unwrap();
    prepared.release_experts(&[]).unwrap();
    assert!(matches!(
        prepared.release_layer(id.layer),
        Err(EngError::BadRequest(_))
    ));
    assert_eq!(prepared.retained_host_bytes(), charged);
    for (part, expected) in expected.iter().enumerate() {
        let mut owned = prepared.payload(id, part).unwrap();
        assert_eq!(&owned, expected);
        owned.data[0] ^= 0xff;
    }
    assert_borrowed_parts(&mut prepared, id, &expected);
    let mut calls = 0;
    let error = prepared
        .with_payload(id, 0, |_| {
            calls += 1;
            Err::<(), _>(EngError::BadRequest("callback sentinel".into()))
        })
        .unwrap_err();
    assert!(matches!(error, EngError::BadRequest(ref message) if message == "callback sentinel"));
    assert_eq!(calls, 1);
    assert!(matches!(
        prepared.payload(
            ExpertId {
                layer: 2,
                expert: 0
            },
            0
        ),
        Err(EngError::BadRequest(_))
    ));
    assert!(matches!(
        prepared.payload(id, 3),
        Err(EngError::BadRequest(_))
    ));
    let mut calls = 0;
    for (id, part) in [
        (
            ExpertId {
                layer: 2,
                expert: 0,
            },
            0,
        ),
        (id, 3),
        (id, usize::MAX),
    ] {
        let error = prepared.with_payload(id, part, |_| {
            calls += 1;
            Ok(())
        });
        assert!(matches!(error, Err(EngError::BadRequest(_))));
    }
    assert_eq!(calls, 0);
    assert_borrowed_parts(&mut prepared, id, &expected);
    assert_eq!(prepared.retained_host_bytes(), charged);
}

#[test]
#[ignore = "requires the pinned real Flint GGUF; CPU-only, one expert, no CUDA device"]
fn real_expert_budget_counts_retained_capacities_and_metadata() {
    let (scratch, header, id, _) = real_expert();
    assert!(matches!(
        PreparedSource::new(&scratch, &header, &[id, id], 16 << 20),
        Err(EngError::BadRequest(_))
    ));
    assert!(matches!(
        PreparedSource::new(
            &scratch,
            &header,
            &[ExpertId {
                layer: 49,
                expert: 0
            }],
            16 << 20
        ),
        Err(EngError::BadRequest(_))
    ));
    let prepared = PreparedSource::new(&scratch, &header, &[id], 16 << 20).unwrap();
    let bytes = std::mem::size_of::<PreparedSource>()
        + prepared.entries.capacity() * std::mem::size_of::<super::PreparedExpert>()
        + prepared
            .entries
            .iter()
            .flat_map(|entry| &entry.parts)
            .map(|part| part.data.capacity() + part.scales.capacity())
            .sum::<usize>();
    let charged = prepared.retained_host_bytes();
    assert_eq!(charged, bytes as u64);
    assert!(charged > 3 * super::PART_BYTES);
    drop(prepared);
    drop(PreparedSource::new(&scratch, &header, &[id], charged).unwrap());
    assert!(matches!(
        PreparedSource::new(&scratch, &header, &[id], charged - 1),
        Err(EngError::MemoryPlan(_))
    ));
    scratch.set_len(0).unwrap();
    assert!(matches!(
        PreparedSource::new(&scratch, &header, &[id], 0),
        Err(EngError::MemoryPlan(_))
    ));
    assert!(matches!(
        PreparedSource::new(&scratch, &header, &[id], 16 << 20),
        Err(EngError::Checkpoint(_))
    ));
}

#[test]
#[ignore = "requires the pinned real Flint GGUF; CPU-only, two experts, no CUDA device"]
fn real_reverse_ordered_experts_keep_lookup_and_accounting() {
    let ids = [
        ExpertId {
            layer: 48,
            expert: 255,
        },
        ExpertId {
            layer: 2,
            expert: 1,
        },
    ];
    let (scratch, header, expected) = real_experts(&ids);
    let mut prepared = PreparedSource::new(&scratch, &header, &ids, 16 << 20).unwrap();
    assert_eq!(prepared.entries.len(), 2);
    assert_eq!(prepared.entries[0].id, ids[1]);
    assert_eq!(prepared.entries[1].id, ids[0]);
    let bytes = std::mem::size_of::<PreparedSource>()
        + prepared.entries.capacity() * std::mem::size_of::<super::PreparedExpert>()
        + prepared
            .entries
            .iter()
            .flat_map(|entry| &entry.parts)
            .map(|part| part.data.capacity() + part.scales.capacity())
            .sum::<usize>();
    let charged = prepared.retained_host_bytes();
    assert_eq!(charged, bytes as u64);
    assert!(charged > 6 * super::PART_BYTES);
    for (&id, expected) in ids.iter().zip(&expected) {
        for (part, expected) in expected.iter().enumerate() {
            prepared
                .with_payload(id, part, |view| {
                    assert_eq!(view, expected.as_view());
                    Ok(())
                })
                .unwrap();
        }
    }
    drop(prepared);
    drop(PreparedSource::new(&scratch, &header, &ids, charged).unwrap());
    assert!(matches!(
        PreparedSource::new(&scratch, &header, &ids, charged - 1),
        Err(EngError::MemoryPlan(_))
    ));
}

#[test]
#[ignore = "requires the pinned real Flint GGUF; CPU-only, one expert, no CUDA device"]
fn real_prepare_refuses_each_truncated_part() {
    for part in 0..3 {
        let (scratch, header, id, _) = real_expert();
        let sources = gguf_weights::expert_sources(&header, id.layer, id.expert).unwrap();
        let source = [sources.gate, sources.up, sources.down]
            .into_iter()
            .nth(part)
            .unwrap();
        scratch.set_len(source.scale.end - 1).unwrap();
        let mut on_demand = GgufSource::new(&scratch, &header).unwrap();
        // Physical down/gate/up offsets differ from the preparation call order.
        assert!(matches!(
            on_demand.payload(id, part),
            Err(EngError::Checkpoint(ref message)) if message.starts_with("Flint expert read:")
        ));
        assert!(matches!(
            PreparedSource::new(&scratch, &header, &[id], 16 << 20),
            Err(EngError::Checkpoint(ref message)) if message.starts_with("Flint expert read:")
        ));
    }
}

#[test]
#[ignore = "requires the pinned real Flint GGUF; CPU-only, one expert, no CUDA device"]
fn real_prepare_refuses_each_nonfinite_part_after_valid_prefix() {
    for part in 0..3 {
        let (scratch, header, id, _) = real_expert();
        let sources = gguf_weights::expert_sources(&header, id.layer, id.expert).unwrap();
        let source = [sources.gate, sources.up, sources.down]
            .into_iter()
            .nth(part)
            .unwrap();
        scratch
            .write_all_at(&f32::NAN.to_le_bytes(), source.scale.start)
            .unwrap();
        let mut on_demand = GgufSource::new(&scratch, &header).unwrap();
        for earlier in 0..part {
            on_demand.payload(id, earlier).unwrap();
        }
        assert!(matches!(
            on_demand.payload(id, part),
            Err(EngError::Checkpoint(ref message)) if message.contains("tensor multiplier must be finite")
        ));
        assert!(matches!(
            PreparedSource::new(&scratch, &header, &[id], 16 << 20),
            Err(EngError::Checkpoint(ref message)) if message.contains("tensor multiplier must be finite")
        ));
    }
}
