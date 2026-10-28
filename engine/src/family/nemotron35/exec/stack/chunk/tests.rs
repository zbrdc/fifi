use super::*;
use crate::family::nemotron35::{
    NemotronBlockKind, NemotronGeometry, NemotronMambaGeometry, NemotronMoeGeometry,
};

// Same tiny geometry and sparse-weight fixture style as nemotron35_stack and
// stack/cuda_tests, but every block contributes and token rows select different experts.
struct MixedSource {
    dense: HashMap<String, Tensor>,
    routed: Vec<u32>,
    fail_expert: Option<u32>,
}

impl MixedSource {
    fn new() -> Self {
        let mut source = Self {
            dense: HashMap::new(),
            routed: Vec::new(),
            fail_expert: None,
        };
        source.matrix(
            "embeddings.weight",
            3,
            8,
            &[
                (0, 0, 1.0),
                (0, 1, 0.2),
                (1, 1, 1.0),
                (1, 2, 0.2),
                (2, 2, 1.0),
            ],
        );
        source.put("norm_f.weight", &[8], &[1.0; 8]);
        source.dense.insert(
            "language_model.lm_head.weight".into(),
            tensor(
                &[3, 8],
                &[
                    1., 0., 0., 0., 0., 0., 0., 0., 0., 1.5, 0., 0., 0., 0., 0., 0., 0., 0., 0.75,
                    0., 0., 0., 0., 0.,
                ],
            )
            .unwrap(),
        );
        for block in 0..3 {
            source.put(&format!("layers.{block}.norm.weight"), &[8], &[1.0; 8]);
        }
        source.put("layers.0.mixer.norm.weight", &[8], &[1.0; 8]);
        let mut entries = Vec::new();
        for index in 0..8 {
            entries.push((index, index, 0.5));
            entries.push((8 + index, index, 0.5));
        }
        entries.extend([(16, 0, 0.5), (17, 1, 0.5), (18, 0, 0.1), (19, 1, 0.1)]);
        source.matrix("layers.0.mixer.in_proj.weight", 20, 8, &entries);
        source.matrix(
            "layers.0.mixer.out_proj.weight",
            8,
            8,
            &(0..8).map(|i| (i, i, 0.75)).collect::<Vec<_>>(),
        );
        source.put(
            "layers.0.mixer.conv1d.weight",
            &[10, 1, 2],
            &[0.25, 0.5].repeat(10),
        );
        source.put("layers.0.mixer.conv1d.bias", &[10], &[0.1; 10]);
        source.put("layers.0.mixer.dt_bias", &[2], &[0.1; 2]);
        source.put("layers.0.mixer.A_log", &[2], &[0.0; 2]);
        source.put("layers.0.mixer.D", &[2], &[0.5; 2]);
        source.matrix(
            "layers.1.mixer.q_proj.weight",
            8,
            8,
            &[(0, 0, 0.5), (4, 1, 0.5)],
        );
        source.matrix(
            "layers.1.mixer.k_proj.weight",
            4,
            8,
            &[(0, 0, 0.5), (1, 1, 0.5)],
        );
        source.matrix(
            "layers.1.mixer.v_proj.weight",
            4,
            8,
            &[(0, 0, 0.5), (1, 1, 0.5)],
        );
        source.matrix(
            "layers.1.mixer.o_proj.weight",
            8,
            8,
            &[(0, 0, 0.25), (1, 1, 0.25), (4, 4, 0.25)],
        );
        source.matrix(
            "layers.2.mixer.gate.weight",
            4,
            8,
            &[(0, 0, 1.0), (1, 1, 1.0), (2, 2, 1.0)],
        );
        source.put(
            "layers.2.mixer.gate.e_score_correction_bias",
            &[4],
            &[0.0; 4],
        );
        source.matrix(
            "layers.2.mixer.fc1_latent_proj.weight",
            2,
            8,
            &[(0, 0, 0.5), (0, 2, 0.25), (1, 1, 0.5)],
        );
        source.matrix(
            "layers.2.mixer.fc2_latent_proj.weight",
            8,
            2,
            &[(0, 0, 0.5), (1, 1, 0.5), (2, 0, 0.25)],
        );
        source
    }

    fn matrix(&mut self, name: &str, rows: u32, columns: u32, entries: &[(usize, usize, f32)]) {
        let mut values = vec![0.0; rows as usize * columns as usize];
        for &(row, column, value) in entries {
            values[row * columns as usize + column] = value;
        }
        self.put(name, &[rows, columns], &values);
    }

    fn put(&mut self, name: &str, shape: &[u32], values: &[f32]) {
        self.dense
            .insert(format!("{ROOT}{name}"), tensor(shape, values).unwrap());
    }
}

impl WeightSource for MixedSource {
    fn dense(&mut self, name: &str) -> Result<Tensor, EngError> {
        self.dense
            .get(name)
            .cloned()
            .ok_or_else(|| invalid("missing fixture weight"))
    }

    fn expert(
        &mut self,
        layer: u32,
        index: ExpertIndex,
        projection: ExpertProjection,
    ) -> Result<Tensor, EngError> {
        assert_eq!(layer, 2);
        match (index, projection) {
            (ExpertIndex::Routed(id), ExpertProjection::Up) => {
                self.routed.push(id);
                if self.fail_expert == Some(id) {
                    return Err(invalid("injected expert failure"));
                }
                tensor(&[1, 2], &[0.5 + id as f32 * 0.125, 0.25])
            }
            (ExpertIndex::Routed(id), ExpertProjection::Down) => {
                tensor(&[2, 1], &[0.5, 0.25 + id as f32 * 0.125])
            }
            (ExpertIndex::Shared, ExpertProjection::Up) => {
                tensor(&[1, 8], &[0.5, 0.25, 0., 0., 0., 0., 0., 0.])
            }
            (ExpertIndex::Shared, ExpertProjection::Down) => {
                tensor(&[8, 1], &[0.25, 0.5, 0.125, 0., 0., 0., 0., 0.])
            }
        }
    }
}

fn stack(capacity: u32) -> NemotronStack<MixedSource> {
    let spec = NemotronExecSpec::new(
        NemotronGeometry {
            hidden_size: 8,
            vocab_size: 3,
            num_attention_heads: 2,
            num_key_value_heads: 1,
            head_dim: 4,
            max_position_embeddings: 16,
            layer_norm_epsilon: 0.1,
        },
        NemotronMambaGeometry {
            groups: 1,
            state_size: 1,
            conv_kernel: 2,
            heads: 2,
            head_dim: 4,
            expand: 1,
            inner_size: 8,
            chunk_size: 1,
        },
        NemotronMoeGeometry {
            intermediate_size: 1,
            moe_intermediate_size: 1,
            latent_size: 2,
            shared_expert_intermediate_size: 1,
            routed_experts: 4,
            shared_experts: 1,
            experts_per_token: 1,
            n_group: 1,
            n_groups: 1,
            topk_group: 1,
            norm_topk_prob: true,
            routed_scaling_factor: 1.0,
        },
        vec![
            NemotronBlockKind::Mamba,
            NemotronBlockKind::Attention,
            NemotronBlockKind::Moe,
        ],
    )
    .unwrap();
    let state = NemotronState::new(&spec, capacity).unwrap();
    NemotronStack::new(spec, MixedSource::new(), state).unwrap()
}

fn assert_same(actual: &NemotronStack<MixedSource>, expected: &NemotronStack<MixedSource>) {
    assert_eq!(actual.state().position(), expected.state().position());
    assert_eq!(
        actual.state().total_state_bytes(),
        expected.state().total_state_bytes()
    );
    let (a_conv, a_ssm) = actual.state().mamba_state(0).unwrap();
    let (e_conv, e_ssm) = expected.state().mamba_state(0).unwrap();
    assert_eq!(a_conv.host_bytes(), e_conv.host_bytes());
    assert_eq!(a_ssm.host_bytes(), e_ssm.host_bytes());
    let key_bytes = |stack: &NemotronStack<_>| {
        // kv_width = 1 KV head x head_dim 4; only rows below the live
        // position are meaningful (failed chunks leave scratch above it).
        let live = stack.state().position() as usize * 4;
        stack
            .attention_keys(1)
            .unwrap()
            .get(..live)
            .unwrap_or(&[])
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<u8>>()
    };
    assert_eq!(key_bytes(actual), key_bytes(expected));
    assert_eq!(actual.logits(), expected.logits());
    assert_eq!(actual.last_prediction(), expected.last_prediction());
}

#[test]
fn nem35_chunk_mixed_nonzero_matches_sequential_and_decode() {
    for rows in [1, 2, 3, 16] {
        let mut sequential = stack(16);
        let mut chunked = stack(16);
        let tokens = [0, 1, 2, 1, 0];
        sequential.prefill(&tokens).unwrap();
        chunked.prefill_chunked(&tokens, rows).unwrap();
        assert_same(&chunked, &sequential);
        assert_eq!(sequential.source.routed, [0, 1, 2, 1, 0]);
        if rows == 16 {
            // One expert union for the whole chunk, not repeated one-token MoE.
            assert_eq!(chunked.source.routed, [0, 1, 2]);
        }
        // The raw hidden history is gone with incremental K/V; nonzero
        // projected keys still prove the mamba block fed the attention.
        let keys = chunked.attention_keys(1).unwrap();
        assert!(
            keys.iter().any(|&value| value != 0.0),
            "Mamba must contribute before attention"
        );
        let (_, ssm) = chunked.state().mamba_state(0).unwrap();
        assert!(
            read(ssm, &[2, 4, 1])
                .unwrap()
                .iter()
                .any(|v| v.abs() > 0.001)
        );
        for _ in 0..3 {
            assert_eq!(
                chunked.next_token().unwrap(),
                sequential.next_token().unwrap()
            );
            assert_same(&chunked, &sequential);
        }
        // Prefill can also continue a nonempty session after decode.
        sequential.prefill(&[2, 0, 1]).unwrap();
        chunked.prefill_chunked(&[2, 0, 1], rows).unwrap();
        assert_same(&chunked, &sequential);
    }
}

#[test]
fn nem35_chunk_layer_progress_is_flagged_and_layer_major() {
    let fixture = concat!(
        "family::nemotron35::exec::stack::chunk::tests",
        "::nem35_chunk_mixed_nonzero_matches_sequential_and_decode"
    );
    for enabled in [false, true] {
        let mut command = std::process::Command::new(std::env::current_exe().unwrap());
        command
            .args(["--exact", fixture, "--nocapture", "--test-threads=1"])
            .env_remove("NEM35_PHASE_TIMING");
        if enabled {
            command.env("NEM35_PHASE_TIMING", "1");
        }
        let output = command.output().unwrap();
        assert!(
            output.status.success(),
            "{}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(String::from_utf8_lossy(&output.stdout).contains("1 passed; 0 failed"));
        let stderr = String::from_utf8(output.stderr).unwrap();
        let records: Vec<serde_json::Value> = stderr
            .lines()
            .filter_map(|line| line.strip_prefix("nem35_layer "))
            .map(|json| serde_json::from_str(json).unwrap())
            .collect();
        if !enabled {
            assert!(records.is_empty(), "progress must be default-off");
            continue;
        }
        assert!(!records.is_empty(), "missing outer-layer progress records");
        assert_eq!(records.len() % 6, 0, "three paired layers per traversal");
        assert!(
            records
                .iter()
                .any(|record| record["rows"].as_u64() == Some(5))
        );
        for call in records.as_chunks::<6>().0 {
            for (layer, pair) in call.as_chunks::<2>().0.iter().enumerate() {
                assert_eq!(pair[0]["phase"], "begin");
                assert_eq!(pair[1]["phase"], "end");
                assert_eq!(pair[0]["layer"].as_u64(), Some(layer as u64));
                assert_eq!(pair[1]["layer"], pair[0]["layer"]);
                assert_eq!(pair[0]["rows"], call[0]["rows"]);
                assert_eq!(pair[1]["rows"], pair[0]["rows"]);
                assert!(pair[0]["elapsed_ns"].is_null());
                assert!(
                    pair[1]["elapsed_ns"]
                        .as_str()
                        .unwrap()
                        .parse::<u128>()
                        .is_ok()
                );
            }
        }
    }
}

#[test]
fn nem35_chunk_failure_keeps_state_logits_and_prediction_atomic() {
    let mut actual = stack(12);
    let mut expected = stack(12);
    actual.step(1).unwrap();
    expected.step(1).unwrap();
    actual.source.fail_expert = Some(2);
    assert!(actual.prefill_chunked(&[0, 2], 2).is_err());
    assert_same(&actual, &expected);
    actual.source.fail_expert = None;
    actual.prefill_chunked(&[0, 2], 2).unwrap();
    expected.prefill(&[0, 2]).unwrap();
    assert_same(&actual, &expected);
    assert_eq!(actual.next_token().unwrap(), expected.next_token().unwrap());
    assert_same(&actual, &expected);

    // A later failed chunk does not roll back earlier successfully committed chunks.
    let mut actual = stack(12);
    let mut expected = stack(12);
    actual.source.fail_expert = Some(2);
    assert!(actual.prefill_chunked(&[0, 1, 2], 2).is_err());
    expected.prefill(&[0, 1]).unwrap();
    assert_same(&actual, &expected);

    // Even a failure after every block (at the head) cannot publish partial state.
    let mut actual = stack(12);
    let expected = stack(12);
    actual.source.dense.remove("language_model.lm_head.weight");
    assert!(actual.prefill_chunked(&[0, 1], 2).is_err());
    assert_same(&actual, &expected);
}

#[test]
fn nem35_chunk_validates_whole_request_and_preserves_empty_prefill() {
    let mut actual = stack(4);
    let mut expected = stack(4);
    actual.prefill_chunked(&[], 2).unwrap();
    assert_same(&actual, &expected);
    actual.step(0).unwrap();
    expected.step(0).unwrap();
    for (tokens, rows) in [(&[1, 2, 0, 1][..], 2), (&[1, 3][..], 1), (&[1][..], 0)] {
        assert!(actual.prefill_chunked(tokens, rows).is_err());
        assert_same(&actual, &expected);
    }
    actual.prefill_chunked(&[], 2).unwrap();
    assert_same(&actual, &expected);
    actual.prefill_chunked(&[1, 2, 0], 2).unwrap();
    expected.prefill(&[1, 2, 0]).unwrap();
    assert_same(&actual, &expected);
    assert!(actual.prefill_chunked(&[0], 2).is_err());
    assert_same(&actual, &expected);
}

#[test]
fn nem35_chunk_rows_parser_is_strict_and_default_off() {
    assert_eq!(parse_rows(None).unwrap(), None);
    for (input, expected) in [
        ("0", None),
        ("00", None),
        ("1", Some(1)),
        ("512", Some(512)),
    ] {
        assert_eq!(parse_rows(Some(OsStr::new(input))).unwrap(), expected);
    }
    for input in [
        "",
        "-1",
        "+1",
        " 2",
        "2 ",
        "1.5",
        "abc",
        "999999999999999999999999999999999999",
    ] {
        assert!(parse_rows(Some(OsStr::new(input))).is_err(), "{input}");
    }
    #[cfg(unix)]
    {
        use std::os::unix::ffi::OsStrExt;
        assert!(parse_rows(Some(OsStr::from_bytes(&[0xff]))).is_err());
    }
}
