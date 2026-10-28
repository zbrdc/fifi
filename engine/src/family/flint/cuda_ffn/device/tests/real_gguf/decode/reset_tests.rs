use super::*;
use crate::family::flint::cuda_ffn::real_model;

fn first_kv_rows(kv: &[LayerKv], stream: &DeviceStream) -> Vec<Vec<u8>> {
    kv.iter()
        .flat_map(|layer| [&layer.keys, &layer.values])
        .map(|plane| {
            let mut row = vec![0; 1024];
            plane.read_at(stream, 0, &mut row).unwrap();
            row
        })
        .collect()
}

#[test]
#[ignore = "requires fresh admitted GPU and pinned GGUF; real 49-owner post-write fault/reset"]
fn selected_post_write_failure_poison_resets_all_49_owners() {
    let (file, header, identity) = real_model::model();
    let prefixes = real_model::prefixes();
    let input = &prefixes[0].input_ids;
    let ctx = CudaContext::new().expect("required CUDA device");
    let stream = ctx.stream().unwrap();
    let mut decode = CudaDecode::new_selected(&ctx, &stream, (&file, &header), 8, 0).unwrap();
    let owners = |kv: &[LayerKv]| {
        kv.iter()
            .map(|layer| {
                (
                    std::ptr::from_ref(&layer.keys),
                    std::ptr::from_ref(&layer.values),
                )
            })
            .collect::<Vec<_>>()
    };
    let original_owners = owners(&decode.kv);
    for layer in &decode.kv {
        for plane in [&layer.keys, &layer.values] {
            plane.write_at(&stream, 0, &[0xff; 1024]).unwrap();
        }
    }
    decode.fail_after_layer = Some(48);
    assert!(
        matches!(decode.step(input[0]), Err(EngError::Checkpoint(message))
        if message == "injected after layer 48 KV write")
    );
    assert!(decode.poisoned);
    for layer in &mut decode.kv {
        assert_eq!(layer.cursor.committed_prefix(), 0);
        assert!(matches!(
            layer.cursor.begin_append(0, 1),
            Err(EngError::Backend { .. })
        ));
    }
    let written = first_kv_rows(&decode.kv, &stream);
    assert_eq!(written.len(), 98);
    assert!(written.iter().all(|row| {
        row.as_chunks::<2>()
            .0
            .iter()
            .all(|word| half::f16::from_bits(u16::from_ne_bytes(*word)).is_finite())
    }));
    assert!(matches!(
        decode.step(input[0]),
        Err(EngError::Backend { .. })
    ));
    let live = ctx.live_bytes();
    let destroyed = ctx.destroy_calls();
    decode.reset().unwrap();
    assert!(!decode.poisoned);
    assert_eq!(owners(&decode.kv), original_owners);
    assert_eq!(ctx.live_bytes(), live);
    assert_eq!(ctx.destroy_calls(), destroyed);
    assert!(
        decode
            .kv
            .iter()
            .all(|layer| layer.cursor.committed_prefix() == 0)
    );
    assert_eq!(first_kv_rows(&decode.kv, &stream), written);
    decode.fail_after_layer = None;
    let mut next = None;
    for &token in input {
        next = Some(decode.step(token).unwrap());
    }
    assert_eq!(next.unwrap() as usize, real_model::FIRST_TOKENS[0]);
    assert!(
        decode
            .kv
            .iter()
            .all(|layer| layer.cursor.committed_prefix() == input.len())
    );
    let stats = decode.fetch_stats();
    assert_eq!(stats.resident_experts, 0);
    assert_eq!(stats.cold_experts.len(), 47 * 6);
    assert_eq!(stats.payload_bytes, 47 * 6 * 1_769_472);
    for layer in 2..49 {
        let ids = stats
            .cold_experts
            .iter()
            .filter(|id| id.layer == layer)
            .map(|id| id.expert)
            .collect::<std::collections::BTreeSet<_>>();
        assert_eq!(ids.len(), 6);
        assert!(ids.iter().all(|&id| id < 256));
    }
    drop(decode);
    gguf::verify_prepared_identity(&file, &identity).unwrap();
}
