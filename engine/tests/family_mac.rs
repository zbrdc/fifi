#[cfg(all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64"))]
mod recurrent_state {
    use std::sync::Mutex;

    use inference_engine::EngError;
    use inference_engine::family::{RecurrentLayout, RecurrentState};
    use inference_engine::kernels::MlxKernels;
    use inference_engine::tensor::Dtype;

    static GPU_TEST: Mutex<()> = Mutex::new(());

    fn hand_layout() -> RecurrentLayout {
        RecurrentLayout {
            heads: 2,
            rank: 3,
            conv_dim: 5,
            conv_width: 4,
            dtype: Dtype::F32,
        }
    }

    #[test]
    fn zero_state_reads_back_zero_and_carries_the_adapter_owner() {
        let _guard = GPU_TEST.lock().unwrap_or_else(|poison| poison.into_inner());
        let mut kernels = MlxKernels::try_new().expect("adapter opens");
        let layout = hand_layout();
        let state = RecurrentState::zeros(&mut kernels, &layout).expect("zero state imports");

        assert_eq!(state.layout(), layout);
        assert_eq!(state.owner_key(), kernels.owner_key());

        let conv = kernels.readback(state.conv()).expect("conv reads back");
        assert_eq!(conv.dtype(), Dtype::F32);
        assert_eq!(conv.shape(), &[5, 3]);
        assert!(
            conv.host_bytes()
                .expect("readback keeps host bytes")
                .iter()
                .all(|byte| *byte == 0)
        );

        let ssm = kernels.readback(state.ssm()).expect("ssm reads back");
        assert_eq!(ssm.dtype(), Dtype::F32);
        assert_eq!(ssm.shape(), &[2, 3, 3]);
        assert!(
            ssm.host_bytes()
                .expect("readback keeps host bytes")
                .iter()
                .all(|byte| *byte == 0)
        );
    }

    #[test]
    fn zero_state_rejects_a_zero_field() {
        let _guard = GPU_TEST.lock().unwrap_or_else(|poison| poison.into_inner());
        let mut kernels = MlxKernels::try_new().expect("adapter opens");
        let layout = RecurrentLayout {
            heads: 0,
            ..hand_layout()
        };
        assert!(matches!(
            RecurrentState::zeros(&mut kernels, &layout),
            Err(EngError::Checkpoint(_))
        ));
    }
}
