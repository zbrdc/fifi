use super::*;
use crate::family::flint::exl3_resident::{FlintResidentLimits, load_flint_resident};
use crate::family::project::cuda::Reserve;
use std::path::PathBuf;

fn token(position: usize) -> u32 {
    (630 + 31 * position) as u32
}

fn acceptance_bad_request_is_preserved(
    decoder: &Decoder<'_>,
    state: &mut State,
    stream: &DeviceStream,
) {
    let error = EngError::BadRequest("test acceptance refusal".into());
    let result: Result<()> = decoder.step(state, token(0), |_, _| Err(error.clone()));
    assert_eq!(result, Err(error));
    assert_eq!(state.fed(), 0);
    assert!(state.poisoned);
    decoder.ctx.ensure_usable().unwrap();
    state.reset(stream).unwrap();
}

fn failed_accept_requires_reset(decoder: &Decoder<'_>, state: &mut State, stream: &DeviceStream) {
    assert!(matches!(
        decoder.step(state, VOCAB as u32, |_, _| -> Result<()> {
            panic!("invalid token must refuse before acceptance")
        }),
        Err(EngError::BadRequest(_))
    ));
    assert_eq!(state.fed(), 0);
    assert!(!state.poisoned);
    assert!(
        state
            .kv
            .iter()
            .all(|cache| cache.cursor.committed_prefix() == 0)
    );
    acceptance_bad_request_is_preserved(decoder, state, stream);
    let pristine: Vec<Vec<u32>> = (0..4)
        .map(|position| {
            decoder
                .step(state, token(position), |_, logits| {
                    Ok(logits.iter().map(|value| value.to_bits()).collect())
                })
                .unwrap()
        })
        .collect();
    state.reset(stream).unwrap();
    for position in 0..3 {
        decoder.step(state, token(position), |_, _| Ok(())).unwrap();
    }
    let fed = state.fed();
    let failure = EngError::ModelOutput("injected post-KV acceptance failure".into());
    let mut accepted = false;
    let result: Result<()> = decoder.step(state, token(3), |_, _| {
        accepted = true;
        Err(failure.clone())
    });
    assert!(
        accepted,
        "failure must occur after all 49 layers and the head"
    );
    assert_eq!(result, Err(failure));
    assert_eq!(state.fed(), fed);
    assert!(state.poisoned);
    assert_eq!(state.kv.len(), 49);
    for cache in &mut state.kv {
        assert_eq!(cache.cursor.committed_prefix(), fed);
        assert!(matches!(
            cache.cursor.begin_append(fed, 1),
            Err(EngError::Backend {
                operation: "flint_kv_append",
                ..
            })
        ));
    }
    assert!(matches!(
        decoder.step(state, token(3), |_, _| -> Result<()> {
            panic!("poisoned decoder must refuse before acceptance")
        }),
        Err(EngError::BadRequest(_))
    ));
    assert_eq!(state.fed(), fed);
    decoder.ctx.ensure_usable().unwrap();
    state.reset(stream).unwrap();
    assert_eq!(state.fed(), 0);
    for (position, expected) in pristine.iter().enumerate() {
        decoder
            .step(state, token(position), |_, logits| {
                assert!(
                    logits
                        .iter()
                        .map(|value| value.to_bits())
                        .eq(expected.iter().copied()),
                    "reset replay differs at position {position}"
                );
                Ok(())
            })
            .unwrap();
    }
    println!(
        "real full49 post-KV acceptance failure: fed unchanged, all cursors poisoned, reset replay bitwise"
    );
}

fn ring_matches_linear(decoder: &Decoder<'_>, linear: &mut State, ring: &mut State) {
    for position in 0..1027 {
        let expected: Vec<u32> = decoder
            .step(linear, token(position), |_, logits| {
                Ok(logits.iter().map(|value| value.to_bits()).collect())
            })
            .unwrap();
        decoder
            .step(ring, token(position), |_, logits| {
                assert_eq!(logits.len(), expected.len());
                let mismatch = logits
                    .iter()
                    .zip(&expected)
                    .position(|(value, expected)| value.to_bits() != *expected);
                assert!(
                    mismatch.is_none(),
                    "ring/linear position {position}, first differing logit {mismatch:?}"
                );
                Ok(())
            })
            .unwrap();
        assert_eq!(linear.fed(), position + 1);
        assert_eq!(ring.fed(), position + 1);
        if [512, 513, 1025, 1026].contains(&position) {
            println!("real full49 ring513/linear bitwise at absolute position {position}");
        }
    }
    assert!(matches!(
        decoder.step(ring, token(1027), |_, _| Ok(())),
        Err(EngError::ContextLimit(_))
    ));
    assert_eq!(ring.fed(), 1027);
}

#[test]
#[ignore = "requires admitted GPU and full EXL3 weights; production Decoder fault/reset and ring proof"]
fn real_weights_failed_accept_reset_and_ring_boundaries() {
    let pack = PathBuf::from(std::env::var_os("FLINT_EXL3_PACK").expect("opt-in EXL3 pack"));
    let ctx = CudaContext::new().expect("real full49 proof requires CUDA");
    let stream = ctx.stream().unwrap();
    let linear_plan = KvPlan::new(1027, 1, false).unwrap();
    let ring_plan = KvPlan::new(1027, 1, true).unwrap();
    for layer in 0..49 {
        assert_eq!(
            ring_plan.cache_rows(layer).unwrap(),
            if FULL_LAYERS.contains(&(layer as u64)) {
                1027
            } else {
                513
            }
        );
    }
    let kv_bytes =
        linear_plan.bytes(4 * 128 * 2 * 2).unwrap() + ring_plan.bytes(4 * 128 * 2 * 2).unwrap();
    let hot_bytes = State::scratch_bytes(&linear_plan)
        .unwrap()
        .checked_add(State::scratch_bytes(&ring_plan).unwrap())
        .unwrap();
    let model = load_flint_resident(
        &ctx,
        &stream,
        &pack,
        FlintResidentLimits {
            max_staging_bytes: 2 << 30,
            max_weight_bytes: ctx.memory_info().unwrap().free as u64,
            reserve: Reserve {
                kv_cache: kv_bytes as u64,
                work: (512_u64 << 20).checked_add(hot_bytes).unwrap(),
                headroom: 512 << 20,
            },
        },
    )
    .unwrap();
    let decoder = Decoder::new(&ctx, &stream, &model).unwrap();
    let mut linear = State::new(&ctx, linear_plan).unwrap();
    failed_accept_requires_reset(&decoder, &mut linear, &stream);
    linear.reset(&stream).unwrap();
    let mut ring = State::new(&ctx, ring_plan).unwrap();
    ring_matches_linear(&decoder, &mut linear, &mut ring);
    stream.synchronize().unwrap();
    println!(
        "real full49 proof complete: every one of 1027 heads compared bitwise, no CPU-FP64 replay"
    );
}
