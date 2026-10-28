use super::*;
use crate::family::{
    project::cuda::WeightLayout,
    qwen36::cuda::{
        common_tests,
        dims::{KvMode, StepOptions},
        state::pinned::PrefixPinPool,
    },
};
use crate::kernels::cuda::resident::step_params::StepParams;

fn delta_equals(state: &ResidentState, stream: &DeviceStream, bytes: usize, value: u8) {
    let mut got = vec![0; bytes];
    state.delta.read_at(stream, 0, &mut got).unwrap();
    assert!(
        got.iter().all(|&byte| byte == value),
        "frozen Delta differs"
    );
}

fn prefix_equals(state: &ResidentState, stream: &DeviceStream, positions: usize, value: u8) {
    for layer in 0..state.layout.kv.layers {
        for range in state.layout.kv.prefix_ranges(layer, positions).unwrap() {
            let mut got = vec![0; range.bytes];
            state
                .resident_kv()
                .unwrap()
                .read_at(stream, range.device_offset, &mut got)
                .unwrap();
            assert!(got.iter().all(|&byte| byte == value), "KV prefix differs");
        }
    }
}

#[test]
#[ignore = "requires native CUDA; run explicitly under the GPU lease"]
fn device_tip_survives_reset_and_overwrite_forces_host_fallback() {
    let ctx = CudaContext::new().expect("native CUDA context");
    let stream = ctx.stream().unwrap();
    for mode in [KvMode::Fp16, KvMode::Int8G32] {
        let options = StepOptions {
            kv_mode: mode,
            context_tokens: 16,
            delta_slots: 1,
            forked_experts: false,
            tiered: None,
            layout: WeightLayout::Native,
        };
        let mut state =
            ResidentState::allocate(&ctx, &stream, &common_tests::dims(), &options).unwrap();
        let mut pool = PrefixPinPool::new(&ctx, state.layout, 2).unwrap();
        let mut tip = DeviceTip::new(&ctx, &stream, &state).unwrap();
        state
            .delta
            .write_at(&stream, 0, &vec![17; tip.bytes()])
            .unwrap();
        state
            .resident_kv()
            .unwrap()
            .write_at(&stream, 0, &vec![7; state.resident_kv().unwrap().bytes()])
            .unwrap();
        state.book.finish_overwrite(4);
        let mut pending = pool.begin(&ctx, &stream, &state, None).unwrap();
        let key = tip.capture(&ctx, &stream, &mut state).unwrap();
        pending.attach_device_tip(key.clone()).unwrap();
        state
            .delta
            .write_at(&stream, 0, &vec![29; tip.bytes()])
            .unwrap();
        state.book.finish_overwrite(8);
        let saved = pool.finish(&ctx, &stream, &state, pending).unwrap();
        state.reset(&stream).unwrap();
        assert!(
            pool.restore(&ctx, &stream, &mut state, &saved, Some(&tip))
                .unwrap()
        );
        assert_eq!(state.tokens(), 4);
        assert!(
            pool.can_recycle(&stream, &state, &saved),
            "device restore establishes live source"
        );
        delta_equals(&state, &stream, tip.bytes(), 17);
        prefix_equals(&state, &stream, 4, 7);
        let duplicate = tip.capture(&ctx, &stream, &mut state).unwrap();
        assert!(
            duplicate.same(&key),
            "an equal tip must preserve the stored device key"
        );

        state.reset(&stream).unwrap();
        state
            .write_step_params(&stream, &StepParams::new(1, 0).unwrap())
            .unwrap();
        state.enqueue_rows(1).unwrap();
        for layer in 0..state.layout.kv.layers {
            for range in state.layout.kv.prefix_ranges(layer, 1).unwrap() {
                state
                    .resident_kv()
                    .unwrap()
                    .write_at(&stream, range.device_offset, &vec![11; range.bytes])
                    .unwrap();
            }
        }
        state.commit_step(1, 0).unwrap();
        prefix_equals(&state, &stream, 1, 11);
        assert!(
            !pool
                .restore(&ctx, &stream, &mut state, &saved, Some(&tip))
                .unwrap()
        );
        delta_equals(&state, &stream, tip.bytes(), 17);
        prefix_equals(&state, &stream, 4, 7);
        assert!(
            pool.can_recycle(&stream, &state, &saved),
            "host fallback establishes live source"
        );

        let mut pending = pool.begin(&ctx, &stream, &state, None).unwrap();
        let key = tip.capture(&ctx, &stream, &mut state).unwrap();
        pending.attach_device_tip(key).unwrap();
        let second = pool.finish(&ctx, &stream, &state, pending).unwrap();
        state.reset(&stream).unwrap();
        state.begin_prefill().unwrap();
        state.revalidate(&stream, 1, 0).unwrap();
        assert!(
            !pool
                .restore(&ctx, &stream, &mut state, &second, Some(&tip))
                .unwrap()
        );
        prefix_equals(&state, &stream, 4, 7);

        let host = state.export_state(&stream).unwrap();
        let key = tip.capture(&ctx, &stream, &mut state).unwrap();
        state.import_state(&stream, &host).unwrap();
        assert!(!tip.restore(&ctx, &stream, &mut state, &key).unwrap());
        let key = tip.capture(&ctx, &stream, &mut state).unwrap();
        state.poison();
        state.reset(&stream).unwrap();
        assert!(!tip.restore(&ctx, &stream, &mut state, &key).unwrap());
    }
    println!("STATE_TIP_NATIVE_FALLBACK_PASS");
}
