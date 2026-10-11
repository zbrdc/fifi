#![cfg(all(
    target_os = "macos",
    target_arch = "aarch64",
    feature = "native-fault-qualification"
))]

use std::{
    env,
    io::Read,
    process::{Command, Stdio},
    thread,
    time::{Duration, Instant},
};

use inference_engine::{
    EngError,
    cache::{DenseParts, KvParts, KvStore, LayerStateKind},
    family::attend_preprojected,
    kernels::{
        BinaryOp, MlxKernels, PackedWeight, QuantizationMode, QuantizationSpec, ReduceOp,
        ResidentPackedWeight, RopeSpec, RopeStep, UnaryOp,
    },
    sched::SessionBudget,
    tensor::{Dtype, ExecutionTensor, Tensor},
};

#[allow(unsafe_code)]
fn native_bytes() -> (usize, usize, usize) {
    let mut active = 0;
    let mut cache = 0;
    let mut peak = 0;
    assert_eq!(unsafe { mlx_sys::mlx_get_active_memory(&mut active) }, 0);
    assert_eq!(unsafe { mlx_sys::mlx_get_cache_memory(&mut cache) }, 0);
    assert_eq!(unsafe { mlx_sys::mlx_get_peak_memory(&mut peak) }, 0);
    (active, cache, peak)
}

#[allow(unsafe_code)]
fn reset_native_peak() {
    assert_eq!(unsafe { mlx_sys::mlx_reset_peak_memory() }, 0);
}

fn rss_kib() -> usize {
    let output = Command::new("/bin/ps")
        .args(["-o", "rss=", "-p", &std::process::id().to_string()])
        .output()
        .unwrap();
    assert!(output.status.success());
    String::from_utf8(output.stdout)
        .unwrap()
        .trim()
        .parse()
        .unwrap()
}

/// Collects a child pipe on its own thread, so a full pipe cannot block the child.
fn drain<R: Read + Send + 'static>(mut pipe: R) -> thread::JoinHandle<String> {
    thread::spawn(move || {
        let mut bytes = Vec::new();
        let _ = pipe.read_to_end(&mut bytes);
        String::from_utf8_lossy(&bytes).into_owned()
    })
}

/// Runs one test of this binary in a child process and returns its captured output.
///
/// A child that outlives `deadline` is killed, and the error carries its output so far.
fn run_child(test: &str, envs: &[(&str, &str)], deadline: Duration) -> Result<String, String> {
    let mut command = Command::new(env::current_exe().unwrap());
    command
        .args(["--exact", test, "--nocapture"])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    for (name, value) in envs {
        command.env(name, value);
    }
    let mut child = command.spawn().unwrap();
    let stdout = drain(child.stdout.take().unwrap());
    let stderr = drain(child.stderr.take().unwrap());
    let started = Instant::now();
    let status = loop {
        if let Some(status) = child.try_wait().unwrap() {
            break Some(status);
        }
        if started.elapsed() >= deadline {
            child.kill().unwrap();
            child.wait().unwrap();
            break None;
        }
        thread::sleep(Duration::from_millis(20));
    };
    let captured = format!("{} {}", stdout.join().unwrap(), stderr.join().unwrap());
    match status {
        // A renamed test makes the child run nothing and exit 0, so require one passed test.
        Some(status) if status.success() && captured.contains("1 passed") => Ok(captured),
        Some(status) if status.success() => Err(format!("child ran no test: {captured}")),
        Some(_) => Err(format!("child failed: {captured}")),
        None => Err(format!("child timed out after {deadline:?}: {captured}")),
    }
}

fn spawn_child(test: &str, envs: &[(&str, &str)]) {
    match run_child(test, envs, Duration::from_secs(180)) {
        Ok(output) => print!("{output}"),
        Err(message) => panic!("{message}"),
    }
}

#[test]
fn failed_retired_array_release_is_terminal() {
    if env::var_os("INFERENCE_NATIVE_RELEASE_CHILD").is_none() {
        spawn_child(
            "failed_retired_array_release_is_terminal",
            &[
                ("INFERENCE_NATIVE_RELEASE_CHILD", "1"),
                ("INFERENCE_MLX_FAIL_NEXT_FREE", "array"),
            ],
        );
        return;
    }

    let source = Tensor::try_from_host(Dtype::F32, vec![1], 1.0f32.to_le_bytes().to_vec()).unwrap();
    let mut kernels = MlxKernels::try_new().unwrap();
    let resident = kernels.import_tensor(&source).unwrap();
    let syncs_before_release = kernels.qualification_sync_count();
    drop(resident);
    assert_eq!(
        kernels.qualification_sync_count(),
        syncs_before_release,
        "the retire list releases an array without a synchronization"
    );
    // This import completes its own work. The flush that follows frees the retired array and fails.
    let second = kernels.import_tensor(&source);
    assert!(second.is_ok(), "the next import completes its own work");
    assert!(matches!(
        kernels.import_tensor(&source),
        Err(EngError::Backend {
            operation: "execute",
            ..
        })
    ));
    drop(second);
    drop(kernels);
    assert!(matches!(
        MlxKernels::try_new(),
        Err(EngError::Backend {
            operation: "initialize",
            ..
        })
    ));
}

#[test]
fn failed_stream_release_is_terminal() {
    if env::var_os("INFERENCE_NATIVE_RELEASE_CHILD").is_none() {
        spawn_child(
            "failed_stream_release_is_terminal",
            &[
                ("INFERENCE_NATIVE_RELEASE_CHILD", "1"),
                ("INFERENCE_MLX_FAIL_NEXT_FREE", "stream"),
            ],
        );
        return;
    }

    let kernels = MlxKernels::try_new().unwrap();
    drop(kernels);
    assert!(matches!(
        MlxKernels::try_new(),
        Err(EngError::Backend {
            operation: "initialize",
            ..
        })
    ));
}

#[test]
fn failed_device_release_is_terminal() {
    if env::var_os("INFERENCE_NATIVE_RELEASE_CHILD").is_none() {
        spawn_child(
            "failed_device_release_is_terminal",
            &[
                ("INFERENCE_NATIVE_RELEASE_CHILD", "1"),
                ("INFERENCE_MLX_FAIL_NEXT_FREE", "device"),
            ],
        );
        return;
    }

    let kernels = MlxKernels::try_new().unwrap();
    drop(kernels);
    assert!(matches!(
        MlxKernels::try_new(),
        Err(EngError::Backend {
            operation: "initialize",
            ..
        })
    ));
}

#[test]
fn healthy_native_release_allows_replacement() {
    if env::var_os("INFERENCE_NATIVE_RELEASE_CHILD").is_none() {
        spawn_child(
            "healthy_native_release_allows_replacement",
            &[("INFERENCE_NATIVE_RELEASE_CHILD", "1")],
        );
        return;
    }

    let source = Tensor::try_from_host(Dtype::F32, vec![1], 1.0f32.to_le_bytes().to_vec()).unwrap();
    let mut kernels = MlxKernels::try_new().unwrap();
    let resident = kernels.import_tensor(&source).unwrap();
    drop(resident);
    drop(kernels);
    drop(MlxKernels::try_new().unwrap());
}

fn small_f32(shape: Vec<u32>) -> Tensor {
    let values = vec![1.0f32; shape.iter().product::<u32>() as usize];
    Tensor::try_from_host(
        Dtype::F32,
        shape,
        values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect(),
    )
    .unwrap()
}

fn small_affine_weight() -> PackedWeight {
    let words = [0x76543210u32, 0xfedcba98]
        .into_iter()
        .flat_map(|word| std::iter::repeat_n(word, 8))
        .flat_map(u32::to_le_bytes)
        .collect();
    let sidecar = |values: [f32; 2]| {
        Tensor::try_from_host(
            Dtype::F32,
            vec![2, 1],
            values
                .into_iter()
                .flat_map(|value| value.to_le_bytes().to_vec())
                .collect(),
        )
        .unwrap()
    };
    PackedWeight::try_new(
        Tensor::try_from_host(Dtype::U32, vec![2, 8], words).unwrap(),
        sidecar([1.0, -2.0]),
        Some(sidecar([-8.0, 4.0])),
        vec![2, 64],
        QuantizationSpec {
            mode: QuantizationMode::Affine,
            bits: 4,
            group_size: 64,
        },
    )
    .unwrap()
}

#[test]
fn failed_matmul_synchronization_is_terminal() {
    if env::var_os("INFERENCE_NATIVE_FAILURE_CHILD").is_none() {
        spawn_child(
            "failed_matmul_synchronization_is_terminal",
            &[
                ("INFERENCE_NATIVE_FAILURE_CHILD", "1"),
                ("INFERENCE_MLX_FAIL_NEXT_SYNC", "1"),
            ],
        );
        return;
    }

    let weight = small_affine_weight();
    let activation = small_f32(vec![2, 64]);
    let mut kernels = MlxKernels::try_new().unwrap();
    assert!(matches!(
        kernels.dequant_matmul(&activation, &weight),
        Err(EngError::Backend {
            operation: "synchronize",
            ..
        })
    ));
    assert!(matches!(
        kernels.dequant_matmul(&activation, &weight),
        Err(EngError::Backend {
            operation: "execute",
            ..
        })
    ));
    drop(kernels);
    assert!(matches!(
        MlxKernels::try_new(),
        Err(EngError::Backend {
            operation: "initialize",
            ..
        })
    ));
}

fn populated_dense_cache(kernels: &mut MlxKernels, dim: u32) -> KvStore {
    let bytes = u64::from(dim) * 8;
    let mut cache = KvStore::try_new(
        LayerStateKind::DenseKv { heads: 1, dim },
        Dtype::F32,
        kernels.owner_key(),
        bytes,
    )
    .unwrap();
    let mut budget = SessionBudget::new(bytes);
    let key = kernels.import_tensor(&small_f32(vec![1, 1, dim])).unwrap();
    let value = kernels.import_tensor(&small_f32(vec![1, 1, dim])).unwrap();
    let growth = budget.acquire_growth(bytes).unwrap();
    cache
        .append(
            KvParts::Dense(DenseParts {
                key,
                value,
                steps: 1,
            }),
            growth,
        )
        .unwrap();
    cache
}

#[test]
fn failed_attention_synchronization_is_terminal() {
    if env::var_os("INFERENCE_NATIVE_FAILURE_CHILD").is_none() {
        spawn_child(
            "failed_attention_synchronization_is_terminal",
            &[
                ("INFERENCE_NATIVE_FAILURE_CHILD", "1"),
                ("INFERENCE_MLX_FAIL_NEXT_SYNC", "4"),
            ],
        );
        return;
    }

    let mut kernels = MlxKernels::try_new().unwrap();
    let query = kernels.import_tensor(&small_f32(vec![1, 1, 8])).unwrap();
    let cache = populated_dense_cache(&mut kernels, 8);
    let attended = attend_preprojected(&mut kernels, &query, &cache, 1).unwrap();
    assert!(matches!(
        kernels.readback(&attended),
        Err(EngError::Backend {
            operation: "synchronize",
            ..
        })
    ));
    assert!(matches!(
        attend_preprojected(&mut kernels, &query, &cache, 1),
        Err(EngError::Backend {
            operation: "execute",
            ..
        })
    ));
    drop(attended);
    drop(cache);
    drop(query);
    drop(kernels);
    assert!(matches!(
        MlxKernels::try_new(),
        Err(EngError::Backend {
            operation: "initialize",
            ..
        })
    ));
}

#[test]
fn failed_readback_synchronization_is_terminal() {
    if env::var_os("INFERENCE_NATIVE_FAILURE_CHILD").is_none() {
        spawn_child(
            "failed_readback_synchronization_is_terminal",
            &[
                ("INFERENCE_NATIVE_FAILURE_CHILD", "1"),
                ("INFERENCE_MLX_FAIL_NEXT_SYNC", "2"),
            ],
        );
        return;
    }

    let source = Tensor::try_from_host(Dtype::F32, vec![1], 1.0f32.to_le_bytes().to_vec()).unwrap();
    let mut kernels = MlxKernels::try_new().unwrap();
    let resident = kernels.import_tensor(&source).unwrap();
    assert!(matches!(
        kernels.readback(&resident),
        Err(EngError::Backend {
            operation: "synchronize",
            ..
        })
    ));
    assert!(matches!(
        kernels.readback(&resident),
        Err(EngError::Backend {
            operation: "execute",
            ..
        })
    ));
    drop(resident);
    drop(kernels);
    assert!(matches!(
        MlxKernels::try_new(),
        Err(EngError::Backend {
            operation: "initialize",
            ..
        })
    ));
}

#[test]
fn null_stream_synchronization_reports_api_validation_error() {
    if env::var_os("INFERENCE_NATIVE_NULL_STREAM_CHILD").is_none() {
        spawn_child(
            "null_stream_synchronization_reports_api_validation_error",
            &[
                ("INFERENCE_NATIVE_NULL_STREAM_CHILD", "1"),
                ("INFERENCE_MLX_NULL_STREAM_SYNC", "1"),
            ],
        );
        return;
    }

    let source = Tensor::try_from_host(Dtype::F32, vec![1], 1.0f32.to_le_bytes().to_vec()).unwrap();
    let mut kernels = MlxKernels::try_new().unwrap();
    let Err(EngError::Backend { operation, message }) = kernels.import_tensor(&source) else {
        panic!("a null stream synchronization must fail with Backend");
    };
    assert_eq!(operation, "synchronize");
    assert!(
        message.contains("expected a non-empty mlx_stream"),
        "unexpected captured message: {message}"
    );
    println!("api validation error, not a driver failure: {message}");
    assert!(matches!(
        kernels.import_tensor(&source),
        Err(EngError::Backend {
            operation: "execute",
            ..
        })
    ));
    drop(kernels);
    assert!(matches!(
        MlxKernels::try_new(),
        Err(EngError::Backend {
            operation: "initialize",
            ..
        })
    ));
}

#[test]
fn failed_import_synchronization_is_terminal() {
    if env::var_os("INFERENCE_NATIVE_FAILURE_CHILD").is_none() {
        spawn_child(
            "failed_import_synchronization_is_terminal",
            &[
                ("INFERENCE_NATIVE_FAILURE_CHILD", "1"),
                ("INFERENCE_MLX_FAIL_NEXT_SYNC", "1"),
            ],
        );
        return;
    }

    let source = small_f32(vec![RETAINED_ELEMENTS]);
    let mut kernels = MlxKernels::try_new().unwrap();
    reset_native_peak();
    let baseline = (native_bytes(), rss_kib());
    let failed = kernels.import_tensor(&source);
    assert!(matches!(
        failed,
        Err(EngError::Backend {
            operation: "synchronize",
            ..
        })
    ));
    // The copied array stays allocated because the import may still be in flight.
    let floor = baseline.0.0 + RETAINED_BYTES;
    assert_active_bytes_stay_at_least(floor, "after the failed import");
    let next = kernels.import_tensor(&source);
    assert!(matches!(
        next,
        Err(EngError::Backend {
            operation: "execute",
            ..
        })
    ));
    drop(kernels);
    assert!(matches!(
        MlxKernels::try_new(),
        Err(EngError::Backend {
            operation: "initialize",
            ..
        })
    ));
    assert_active_bytes_stay_at_least(floor, "after releasing the adapter");
    let retained = (native_bytes(), rss_kib());
    println!("baseline: {baseline:?}; retained after failure: {retained:?}");
}

const RETAINED_ELEMENTS: u32 = 1 << 20;
const RETAINED_BYTES: usize = RETAINED_ELEMENTS as usize * 4;

/// Resident inputs for one operation call.
struct Operands {
    tensors: Vec<ExecutionTensor>,
    weight: Option<ResidentPackedWeight>,
    /// A dense cache for the attention caller.
    cache: Option<KvStore>,
}

/// One resident operation that the retention tests drive through the public adapter.
struct Operation {
    label: &'static str,
    /// Native synchronizations that `prepare` and the calls before the failing one perform.
    /// The failing step synchronizes next.
    imports: usize,
    /// Output arrays of `RETAINED_BYTES` each that exist when the injected failure strikes.
    outputs: usize,
    /// The one-based native operation call of the process that takes an injected operation fault.
    fault_call: usize,
    prepare: fn(&mut MlxKernels) -> Operands,
    /// The operation drops its output on success. A success never occurs in these tests.
    call: fn(&mut MlxKernels, &mut Operands) -> Result<(), EngError>,
}

fn import_f32(kernels: &mut MlxKernels, shape: Vec<u32>) -> ExecutionTensor {
    kernels.import_tensor(&small_f32(shape)).unwrap()
}

fn prepare_unary(kernels: &mut MlxKernels) -> Operands {
    Operands {
        tensors: vec![import_f32(kernels, vec![RETAINED_ELEMENTS])],
        weight: None,
        cache: None,
    }
}

fn prepare_binary(kernels: &mut MlxKernels) -> Operands {
    Operands {
        tensors: (0..2)
            .map(|_| import_f32(kernels, vec![RETAINED_ELEMENTS]))
            .collect(),
        weight: None,
        cache: None,
    }
}

fn prepare_mm(kernels: &mut MlxKernels) -> Operands {
    Operands {
        tensors: vec![
            import_f32(kernels, vec![1, 1]),
            import_f32(kernels, vec![1, RETAINED_ELEMENTS]),
        ],
        weight: None,
        cache: None,
    }
}

fn large_affine_weight(rows: u32) -> PackedWeight {
    let sidecar = |value: f32| {
        let bytes = std::iter::repeat_n(value.to_le_bytes(), rows as usize)
            .flatten()
            .collect();
        Tensor::try_from_host(Dtype::F32, vec![rows, 1], bytes).unwrap()
    };
    PackedWeight::try_new(
        Tensor::try_from_host(Dtype::U32, vec![rows, 8], vec![0; rows as usize * 32]).unwrap(),
        sidecar(1.0),
        Some(sidecar(0.0)),
        vec![rows, 64],
        QuantizationSpec {
            mode: QuantizationMode::Affine,
            bits: 4,
            group_size: 64,
        },
    )
    .unwrap()
}

fn prepare_qmm(kernels: &mut MlxKernels) -> Operands {
    let weight = kernels
        .import_packed(&large_affine_weight(RETAINED_ELEMENTS))
        .unwrap();
    Operands {
        tensors: vec![import_f32(kernels, vec![1, 64])],
        weight: Some(weight),
        cache: None,
    }
}

const SILU: Operation = Operation {
    label: "SiLU",
    imports: 1,
    outputs: 1,
    fault_call: 1,
    prepare: prepare_unary,
    call: |kernels, operands| {
        kernels
            .silu(&operands.tensors[0])
            .and_then(|output| kernels.readback(&output).map(drop))
    },
};
const SOFTMAX: Operation = Operation {
    label: "softmax",
    imports: 1,
    outputs: 1,
    fault_call: 1,
    prepare: prepare_unary,
    call: |kernels, operands| {
        kernels
            .softmax(&operands.tensors[0])
            .and_then(|output| kernels.readback(&output).map(drop))
    },
};
const MUL: Operation = Operation {
    label: "elementwise multiply",
    imports: 2,
    outputs: 1,
    fault_call: 1,
    prepare: prepare_binary,
    call: |kernels, operands| {
        kernels
            .mul(&operands.tensors[0], &operands.tensors[1])
            .and_then(|output| kernels.readback(&output).map(drop))
    },
};
const ADD: Operation = Operation {
    label: "elementwise add",
    imports: 2,
    outputs: 1,
    fault_call: 1,
    prepare: prepare_binary,
    call: |kernels, operands| {
        kernels
            .add(&operands.tensors[0], &operands.tensors[1])
            .and_then(|output| kernels.readback(&output).map(drop))
    },
};
const SIGMOID: Operation = Operation {
    label: "sigmoid",
    imports: 1,
    outputs: 1,
    fault_call: 1,
    prepare: prepare_unary,
    call: |kernels, operands| {
        kernels
            .unary(UnaryOp::Sigmoid, &operands.tensors[0])
            .and_then(|output| kernels.readback(&output).map(drop))
    },
};
const SUBTRACT: Operation = Operation {
    label: "elementwise subtract",
    imports: 2,
    outputs: 1,
    fault_call: 1,
    prepare: prepare_binary,
    call: |kernels, operands| {
        kernels
            .binary(BinaryOp::Sub, &operands.tensors[0], &operands.tensors[1])
            .and_then(|output| kernels.readback(&output).map(drop))
    },
};
const MM: Operation = Operation {
    label: "matmul",
    imports: 2,
    outputs: 1,
    fault_call: 1,
    prepare: prepare_mm,
    call: |kernels, operands| {
        kernels
            .mm(&operands.tensors[0], &operands.tensors[1])
            .and_then(|output| kernels.readback(&output).map(drop))
    },
};
// The packed weight imports its payload, scales, and biases, and the activation imports once.
const QMM: Operation = Operation {
    label: "quantized matmul",
    imports: 4,
    outputs: 1,
    fault_call: 1,
    prepare: prepare_qmm,
    call: |kernels, operands| {
        kernels
            .qmm(&operands.tensors[0], operands.weight.as_ref().unwrap())
            .and_then(|output| kernels.readback(&output).map(drop))
    },
};

fn prepare_rms_norm(kernels: &mut MlxKernels) -> Operands {
    Operands {
        tensors: vec![
            import_f32(kernels, vec![1, RETAINED_ELEMENTS]),
            import_f32(kernels, vec![RETAINED_ELEMENTS]),
        ],
        weight: None,
        cache: None,
    }
}

const RMS_NORM: Operation = Operation {
    label: "RMS normalization",
    imports: 2,
    outputs: 1,
    fault_call: 1,
    prepare: prepare_rms_norm,
    call: |kernels, operands| {
        kernels
            .rms_norm(&operands.tensors[0], &operands.tensors[1], 1e-6)
            .and_then(|output| kernels.readback(&output).map(drop))
    },
};

fn prepare_rope(kernels: &mut MlxKernels) -> Operands {
    // One head of one token with a long head size, so each rotated output holds `RETAINED_BYTES`.
    Operands {
        tensors: vec![
            import_f32(kernels, vec![1, 1, RETAINED_ELEMENTS]),
            import_f32(kernels, vec![1, 1, RETAINED_ELEMENTS]),
        ],
        weight: None,
        cache: None,
    }
}

const ROPE_STEP: RopeStep = RopeStep {
    pos: 1,
    theta: 1e7,
    part: 64,
};

// The two imports synchronize first. The `q` readback synchronizes third and fails the first output.
const ROPE: Operation = Operation {
    label: "RoPE",
    imports: 2,
    outputs: 1,
    fault_call: 1,
    prepare: prepare_rope,
    call: |kernels, operands| {
        let [q, k] = &mut operands.tensors[..] else {
            panic!("rope needs q and k");
        };
        // Shares of the old tensors keep them alive, so no release synchronizes inside the call.
        let (old_q, old_k) = (q.share(), k.share());
        kernels.rope(q, k, &ROPE_STEP)?;
        kernels.readback(q)?;
        drop((old_q, old_k));
        Ok(())
    },
};
// The `q` readback synchronizes third. The `k` readback synchronizes fourth, so a failure there
// strands both evaluated outputs. A build fault in the `k` call leaves none.
// The `rope` and `rope_k` evaluation suites both stay (R-13). They strike the same point, the
// first `q` readback, but they select different build calls for the call, null, and mismatch
// faults, so each keeps its own coverage. `imports: 3` sets the synchronization ordinal to 4. It
// is not an import count.
const ROPE_K: Operation = Operation {
    imports: 3,
    outputs: 2,
    fault_call: 2,
    call: |kernels, operands| {
        let [q, k] = &mut operands.tensors[..] else {
            panic!("rope needs q and k");
        };
        let (old_q, old_k) = (q.share(), k.share());
        kernels.rope(q, k, &ROPE_STEP)?;
        kernels.readback(q)?;
        kernels.readback(k)?;
        drop((old_q, old_k));
        Ok(())
    },
    ..ROPE
};

// With one cache chunk, the attention entry builds eight native calls before `readback`. The fault
// selector counts only two: the SDPA call first and the output reshape second. The slice, join,
// batch shape, and layout calls stay outside it.
fn prepare_attention(kernels: &mut MlxKernels) -> Operands {
    let query = import_f32(kernels, vec![1, 1, RETAINED_ELEMENTS]);
    let cache = populated_dense_cache(kernels, RETAINED_ELEMENTS);
    Operands {
        tensors: vec![query],
        weight: None,
        cache: Some(cache),
    }
}

// The query, key, and value imports synchronize first. `readback` synchronizes fourth.
const ATTENTION: Operation = Operation {
    label: "attention",
    imports: 3,
    outputs: 1,
    fault_call: 1,
    prepare: prepare_attention,
    call: |kernels, operands| {
        attend_preprojected(
            kernels,
            &operands.tensors[0],
            operands.cache.as_ref().unwrap(),
            1,
        )
        .and_then(|output| kernels.readback(&output).map(drop))
    },
};
// The second selected call is the output reshape. It reports its own label for call and null
// faults. A mismatch there reaches the attention output check, which reports `attention`. A
// mismatch at the SDPA call stays hidden because the reshape restores the shape.
const ATTENTION_OUTPUT: Operation = Operation {
    label: "attention output shape",
    fault_call: 2,
    ..ATTENTION
};
const ATTENTION_OUTPUT_MISMATCH: Operation = Operation {
    fault_call: 2,
    ..ATTENTION
};

// One row of width W broadcast to two rows. Two rows of W/2 give an output of `RETAINED_BYTES`.
fn prepare_broadcast_rows(kernels: &mut MlxKernels) -> Operands {
    Operands {
        tensors: vec![import_f32(kernels, vec![1, RETAINED_ELEMENTS / 2])],
        weight: None,
        cache: None,
    }
}

const BROADCAST_ROWS: Operation = Operation {
    label: "broadcast rows",
    imports: 1,
    outputs: 1,
    fault_call: 1,
    prepare: prepare_broadcast_rows,
    call: |kernels, operands| {
        kernels
            .broadcast_rows(&operands.tensors[0], 2)
            .and_then(|output| kernels.readback(&output).map(drop))
    },
};

fn prepare_weighted_row_sum(kernels: &mut MlxKernels) -> Operands {
    Operands {
        tensors: vec![
            import_f32(kernels, vec![2, RETAINED_ELEMENTS]),
            import_f32(kernels, vec![2, 1, 1]),
        ],
        weight: None,
        cache: None,
    }
}

// The reshape of the weights is the first selected call, the multiply the second, and the sum the
// third. All three report the label `weighted row sum`, so the label cannot show which call took
// the fault. The call index selects it.
const WEIGHTED_ROW_SUM: Operation = Operation {
    label: "weighted row sum",
    imports: 2,
    outputs: 1,
    fault_call: 1,
    prepare: prepare_weighted_row_sum,
    call: |kernels, operands| {
        kernels
            .weighted_row_sum(&operands.tensors[0], &operands.tensors[1])
            .and_then(|output| kernels.readback(&output).map(drop))
    },
};
const WEIGHTED_ROW_SUM_MULTIPLY: Operation = Operation {
    fault_call: 2,
    ..WEIGHTED_ROW_SUM
};
const WEIGHTED_ROW_SUM_SUM: Operation = Operation {
    fault_call: 3,
    ..WEIGHTED_ROW_SUM
};

fn prepare_two_rows(kernels: &mut MlxKernels) -> Operands {
    Operands {
        tensors: vec![import_f32(kernels, vec![2, RETAINED_ELEMENTS])],
        weight: None,
        cache: None,
    }
}

fn prepare_half(kernels: &mut MlxKernels) -> Operands {
    Operands {
        tensors: vec![import_f32(kernels, vec![RETAINED_ELEMENTS / 2])],
        weight: None,
        cache: None,
    }
}

fn prepare_half_pair(kernels: &mut MlxKernels) -> Operands {
    Operands {
        tensors: vec![import_f32(kernels, vec![2, RETAINED_ELEMENTS / 2])],
        weight: None,
        cache: None,
    }
}

fn prepare_two_columns(kernels: &mut MlxKernels) -> Operands {
    Operands {
        tensors: vec![import_f32(kernels, vec![RETAINED_ELEMENTS, 2])],
        weight: None,
        cache: None,
    }
}

fn prepare_target_and_update(kernels: &mut MlxKernels) -> Operands {
    Operands {
        tensors: vec![
            import_f32(kernels, vec![RETAINED_ELEMENTS]),
            import_f32(kernels, vec![1]),
        ],
        weight: None,
        cache: None,
    }
}

// The mean over axis 0 of [2, N] allocates an [N] output of `RETAINED_BYTES`.
const MEAN: Operation = Operation {
    label: "mean",
    imports: 1,
    outputs: 1,
    fault_call: 1,
    prepare: prepare_two_rows,
    call: |kernels, operands| {
        kernels
            .reduce(ReduceOp::Mean, &operands.tensors[0], 0, false)
            .and_then(|output| kernels.readback(&output).map(drop))
    },
};
const MAX: Operation = Operation {
    label: "max",
    call: |kernels, operands| {
        kernels
            .reduce(ReduceOp::Max, &operands.tensors[0], 0, false)
            .and_then(|output| kernels.readback(&output).map(drop))
    },
    ..MEAN
};
// The contiguous copy of the transposed [N / 2, 2] view holds `RETAINED_BYTES`.
const TRANSPOSE: Operation = Operation {
    label: "transpose",
    imports: 1,
    outputs: 1,
    fault_call: 1,
    prepare: prepare_half_pair,
    call: |kernels, operands| {
        kernels
            .transpose(&operands.tensors[0], &[1, 0])
            .and_then(|output| kernels.readback(&output).map(drop))
    },
};
const EXPAND_DIMS: Operation = Operation {
    label: "expand dims",
    imports: 1,
    outputs: 1,
    fault_call: 1,
    prepare: prepare_unary,
    call: |kernels, operands| {
        kernels
            .expand_dims(&operands.tensors[0], 0)
            .and_then(|output| kernels.readback(&output).map(drop))
    },
};
// Repeating [N / 2] twice allocates an [N] output.
const REPEAT: Operation = Operation {
    label: "repeat",
    imports: 1,
    outputs: 1,
    fault_call: 1,
    prepare: prepare_half,
    call: |kernels, operands| {
        kernels
            .repeat(&operands.tensors[0], 2, 0)
            .and_then(|output| kernels.readback(&output).map(drop))
    },
};
// Column 0 of [N, 2] is a strided view; its contiguous copy holds `RETAINED_BYTES`.
const SLICE: Operation = Operation {
    label: "slice",
    imports: 1,
    outputs: 1,
    fault_call: 1,
    prepare: prepare_two_columns,
    call: |kernels, operands| {
        kernels
            .slice(&operands.tensors[0], &[0, 0], &[RETAINED_ELEMENTS, 1])
            .and_then(|output| kernels.readback(&output).map(drop))
    },
};
// The update writes into a copy of the [N] target, because the caller still holds the target.
const SLICE_UPDATE: Operation = Operation {
    label: "slice update",
    imports: 2,
    outputs: 1,
    fault_call: 1,
    prepare: prepare_target_and_update,
    call: |kernels, operands| {
        kernels
            .slice_update(&operands.tensors[0], &operands.tensors[1], &[0])
            .and_then(|output| kernels.readback(&output).map(drop))
    },
};
// `kv_append` (packet NATIVE-OPS-1) consumes its target, so each call imports a fresh one: the
// driver runs the closure twice, and the second call runs on the already terminal adapter, whose
// import rejects with `Backend{execute}`. The closure returns that error instead of unwrapping
// the import, so the driver observes the expected terminal rejection of the second call. The
// update import of `prepare` and the target import of the call synchronize before the readback.
fn prepare_kv_append(kernels: &mut MlxKernels) -> Operands {
    Operands {
        tensors: vec![import_f32(kernels, vec![1])],
        weight: None,
        cache: None,
    }
}

const KV_APPEND: Operation = Operation {
    label: "kv append",
    imports: 2,
    outputs: 1,
    fault_call: 1,
    prepare: prepare_kv_append,
    call: |kernels, operands| {
        let target = match kernels.import_tensor(&small_f32(vec![RETAINED_ELEMENTS])) {
            Ok(target) => target,
            Err(error) => return Err(error),
        };
        kernels
            .kv_append(target, &operands.tensors[0], &[0])
            .and_then(|output| kernels.readback(&output).map(drop))
    },
};
const CONCATENATE: Operation = Operation {
    label: "concatenate",
    imports: 2,
    outputs: 1,
    fault_call: 1,
    prepare: prepare_binary,
    call: |kernels, operands| {
        kernels
            .concatenate(&[&operands.tensors[0], &operands.tensors[1]], 0)
            .and_then(|output| kernels.readback(&output).map(drop))
    },
};

fn import_u32(kernels: &mut MlxKernels, shape: Vec<u32>, value: u32) -> ExecutionTensor {
    let count = shape.iter().product::<u32>() as usize;
    let host = Tensor::try_from_host(Dtype::U32, shape, value.to_le_bytes().repeat(count)).unwrap();
    kernels.import_tensor(&host).unwrap()
}

/// An affine 4-bit expert stack `[experts, rows, 64]` with group size 64; every value is 1.0.
fn affine_stack(experts: u32, rows: u32) -> PackedWeight {
    let words = (experts * rows * 8) as usize;
    let groups = (experts * rows) as usize;
    let ones: Vec<u8> = 1.0f32.to_le_bytes().repeat(groups);
    let zeros: Vec<u8> = 0.0f32.to_le_bytes().repeat(groups);
    PackedWeight::try_new(
        Tensor::try_from_host(
            Dtype::U32,
            vec![experts, rows, 8],
            0x1111_1111u32.to_le_bytes().repeat(words),
        )
        .unwrap(),
        Tensor::try_from_host(Dtype::F32, vec![experts, rows, 1], ones).unwrap(),
        Some(Tensor::try_from_host(Dtype::F32, vec![experts, rows, 1], zeros).unwrap()),
        vec![experts, rows, 64],
        QuantizationSpec {
            mode: QuantizationMode::Affine,
            bits: 4,
            group_size: 64,
        },
    )
    .unwrap()
}

// argmax over axis 0 of [2, N] returns a U32 [N] output of `RETAINED_BYTES`.
fn prepare_rows_for_argmax(kernels: &mut MlxKernels) -> Operands {
    Operands {
        tensors: vec![import_f32(kernels, vec![2, RETAINED_ELEMENTS])],
        weight: None,
        cache: None,
    }
}

const ARGMAX: Operation = Operation {
    label: "argmax",
    imports: 1,
    outputs: 1,
    fault_call: 1,
    prepare: prepare_rows_for_argmax,
    call: |kernels, operands| {
        kernels
            .argmax(&operands.tensors[0], 0, false)
            .and_then(|output| kernels.readback(&output).map(drop))
    },
};
// argsort of [N] returns a U32 [N] output.
const ARGSORT: Operation = Operation {
    label: "argsort",
    imports: 1,
    outputs: 1,
    fault_call: 1,
    prepare: prepare_unary,
    call: |kernels, operands| {
        kernels
            .argsort(&operands.tensors[0], 0)
            .and_then(|output| kernels.readback(&output).map(drop))
    },
};

fn prepare_u32_vector(kernels: &mut MlxKernels) -> Operands {
    Operands {
        tensors: vec![import_u32(kernels, vec![RETAINED_ELEMENTS], 1)],
        weight: None,
        cache: None,
    }
}

// Converting a U32 [N] resident returns an F32 [N] output of `RETAINED_BYTES`.
const U32_TO_F32: Operation = Operation {
    label: "convert",
    imports: 1,
    outputs: 1,
    fault_call: 1,
    prepare: prepare_u32_vector,
    call: |kernels, operands| {
        kernels
            .u32_to_f32(&operands.tensors[0])
            .and_then(|output| kernels.readback(&output).map(drop))
    },
};

// take of row 1 from [2, N] returns [1, N].
fn prepare_take(kernels: &mut MlxKernels) -> Operands {
    Operands {
        tensors: vec![
            import_f32(kernels, vec![2, RETAINED_ELEMENTS]),
            import_u32(kernels, vec![1], 1),
        ],
        weight: None,
        cache: None,
    }
}

const TAKE: Operation = Operation {
    label: "take",
    imports: 2,
    outputs: 1,
    fault_call: 2,
    prepare: prepare_take,
    call: |kernels, operands| {
        kernels
            .take(&operands.tensors[0], &operands.tensors[1], 0)
            .and_then(|output| kernels.readback(&output).map(drop))
    },
};
const TAKE_CLAMP: Operation = Operation {
    fault_call: 1,
    ..TAKE
};

// `take_along_axis` of one row of [2, N] on axis 0 with [1, N] U32 indices returns [1, N]:
// `RETAINED_ELEMENTS` values, as `prepare_take` builds for `take` (the indices carry the input
// rank here). The clamp of the indices is native call 1 and the gather is call 2 (packet
// Q36-PF-DEVROUTE fix1 K1).
fn prepare_take_along_axis(kernels: &mut MlxKernels) -> Operands {
    Operands {
        tensors: vec![
            import_f32(kernels, vec![2, RETAINED_ELEMENTS]),
            import_u32(kernels, vec![1, RETAINED_ELEMENTS], 1),
        ],
        weight: None,
        cache: None,
    }
}

const TAKE_ALONG_AXIS: Operation = Operation {
    label: "take_along_axis",
    imports: 2,
    outputs: 1,
    fault_call: 2,
    prepare: prepare_take_along_axis,
    call: |kernels, operands| {
        kernels
            .take_along_axis(&operands.tensors[0], &operands.tensors[1], 0)
            .and_then(|output| kernels.readback(&output).map(drop))
    },
};

// gather_qmm of x [1, 64, 64] with 16 indices over a [2, 1024, 64] stack returns [16, 64, 1024],
// which is `RETAINED_ELEMENTS` values.
fn prepare_gather_qmm(kernels: &mut MlxKernels) -> Operands {
    let stack = kernels.import_packed(&affine_stack(2, 1024)).unwrap();
    Operands {
        tensors: vec![
            import_f32(kernels, vec![1, 64, 64]),
            import_u32(kernels, vec![16], 1),
        ],
        weight: Some(stack),
        cache: None,
    }
}

// The packed stack imports its payload, scales, and biases; x and the indices import once each.
const GATHER_QMM: Operation = Operation {
    label: "gather quantized matmul",
    imports: 5,
    outputs: 1,
    fault_call: 2,
    prepare: prepare_gather_qmm,
    call: |kernels, operands| {
        kernels
            .gather_qmm(
                &operands.tensors[0],
                operands.weight.as_ref().unwrap(),
                &operands.tensors[1],
            )
            .and_then(|output| kernels.readback(&output).map(drop))
    },
};

// dequantize of a [16384, 64] packed weight returns `RETAINED_ELEMENTS` F32 values.
fn prepare_dequantize(kernels: &mut MlxKernels) -> Operands {
    let weight = kernels
        .import_packed(&large_affine_weight(RETAINED_ELEMENTS / 64))
        .unwrap();
    Operands {
        tensors: Vec::new(),
        weight: Some(weight),
        cache: None,
    }
}

const DEQUANTIZE: Operation = Operation {
    label: "dequantize",
    imports: 3,
    outputs: 1,
    fault_call: 1,
    prepare: prepare_dequantize,
    call: |kernels, operands| {
        kernels
            .dequantize(operands.weight.as_ref().unwrap())
            .and_then(|output| kernels.readback(&output).map(drop))
    },
};

// rope_scaled of [1024, 1, 1024] keeps the shape: `RETAINED_ELEMENTS` values.
fn prepare_rope_scaled(kernels: &mut MlxKernels) -> Operands {
    Operands {
        tensors: vec![import_f32(kernels, vec![1024, 1, 1024])],
        weight: None,
        cache: None,
    }
}

const ROPE_SCALED: Operation = Operation {
    label: "scaled RoPE",
    imports: 1,
    outputs: 1,
    fault_call: 1,
    prepare: prepare_rope_scaled,
    call: |kernels, operands| {
        let spec = RopeSpec {
            dims: 64,
            base: 10000.0,
            scale: 1.0,
            offset: 3,
        };
        kernels
            .rope_scaled(&operands.tensors[0], &spec)
            .and_then(|output| kernels.readback(&output).map(drop))
    },
};

// Queries [1, 16, 512, 128] give an output of `RETAINED_ELEMENTS` values; keys and values are
// [1, 4, 512, 128].
fn prepare_grouped_attention(kernels: &mut MlxKernels) -> Operands {
    Operands {
        tensors: vec![
            import_f32(kernels, vec![1, 16, 512, 128]),
            import_f32(kernels, vec![1, 4, 512, 128]),
            import_f32(kernels, vec![1, 4, 512, 128]),
        ],
        weight: None,
        cache: None,
    }
}

const GROUPED_ATTENTION: Operation = Operation {
    label: "grouped attention",
    imports: 3,
    outputs: 1,
    fault_call: 1,
    prepare: prepare_grouped_attention,
    call: |kernels, operands| {
        kernels
            .grouped_attention(
                &operands.tensors[0],
                &operands.tensors[1],
                &operands.tensors[2],
                0.088_388_35,
                false,
            )
            .and_then(|output| kernels.readback(&output).map(drop))
    },
};

// `grouped_attention_prefix` (packet NATIVE-OPS-1): the same output layout, and the one composite
// shim call covers the two prefix views and the fused attention, so it is selected call 1.
const GROUPED_ATTENTION_PREFIX: Operation = Operation {
    label: "grouped attention prefix",
    call: |kernels, operands| {
        kernels
            .grouped_attention_prefix(
                &operands.tensors[0],
                &operands.tensors[1],
                &operands.tensors[2],
                256,
                0.088_388_35,
                false,
            )
            .and_then(|output| kernels.readback(&output).map(drop))
    },
    ..GROUPED_ATTENTION
};

// `evaluate_async` (packet NATIVE-OPS-1) fills no output array and does not go through the
// counted selection hooks, so only its own un-counted submission fault
// (`INFERENCE_MLX_FAIL_ASYNC_EVAL`) applies to its submission; the evaluation, null, and
// mismatch faults have no point to strike (the output tensor stays lazy and readback-complete synchronizations are the
// blocking `evaluate` fault suite's own cases).
const EVALUATE_ASYNC: Operation = Operation {
    label: "evaluate",
    imports: 1,
    outputs: 1,
    fault_call: 1,
    prepare: prepare_unary,
    call: |kernels, operands| {
        let outputs: Vec<&ExecutionTensor> = operands.tensors.iter().collect();
        kernels.evaluate_async(&outputs)
    },
};

// `y` holds `RETAINED_ELEMENTS` values: 1,024 tokens, 8 value heads, and Dv = 128. Two key heads
// serve the value heads, and Dk = 32. The new state of 128 Ki values is a second output that the
// retained-bytes floor does not count.
fn prepare_gated_delta(kernels: &mut MlxKernels) -> Operands {
    Operands {
        tensors: vec![
            import_f32(kernels, vec![1, 1024, 2, 32]),
            import_f32(kernels, vec![1, 1024, 2, 32]),
            import_f32(kernels, vec![1, 1024, 8, 128]),
            import_f32(kernels, vec![1, 1024, 8]),
            import_f32(kernels, vec![1, 1024, 8]),
            import_f32(kernels, vec![1, 8, 128, 32]),
        ],
        weight: None,
        cache: None,
    }
}

const GATED_DELTA: Operation = Operation {
    label: "gated delta rule",
    imports: 6,
    outputs: 1,
    fault_call: 1,
    prepare: prepare_gated_delta,
    call: |kernels, operands| {
        let [q, k, v, g, beta, state] = &operands.tensors[..] else {
            panic!("gated delta needs six operands");
        };
        kernels
            .gated_delta(q, k, v, g, beta, state)
            .and_then(|output| kernels.readback(&output.y).map(drop))
    },
};

// The injected failure flags the adapter without creating in-flight work. Polling covers a
// release that MLX defers.
fn assert_active_bytes_stay_at_least(floor: usize, context: &str) {
    for _ in 0..20 {
        let (active, _, _) = native_bytes();
        assert!(
            active >= floor,
            "{context}: active {active} bytes fell below {floor} bytes"
        );
        thread::sleep(Duration::from_millis(10));
    }
}

/// A deterministic failure that a child process injects into the operation under test.
#[derive(Clone, Copy)]
enum Fault {
    /// The synchronization that completes the operation reports a failure. Lazy operations
    /// complete at `readback`. Eager operations complete inside the call.
    Synchronization,
    /// The selected native call returns an error status and produces no output.
    NativeCall,
    /// The selected native call reports success and leaves the output null.
    NullOutput,
    /// Evaluation runs and then reports a failure.
    Evaluation,
    /// The operation succeeds and returns an output with an extra unit axis.
    OutputMismatch,
    /// The un-counted submission fault of `evaluate_async`: its native submission call fails
    /// with no output. The submission takes part in no selection, so the counted operation
    /// faults never apply to it (guard.cpp documents why).
    Submission,
}

impl Operation {
    /// Every operation builds lazily. `readback` evaluates and synchronizes, so those faults
    /// carry the transfer labels. The other faults strike at the build call.
    fn expected_label(&self, fault: Fault) -> &'static str {
        match fault {
            Fault::Synchronization => "synchronize",
            Fault::Evaluation => "evaluate",
            _ => self.label,
        }
    }
}

/// The variable value that makes the operation's selected native call take the named failure.
fn operation_fault(name: &str, operation: &Operation) -> (&'static str, String) {
    let value = match operation.fault_call {
        1 => name.to_string(),
        call => format!("{name}:{call}"),
    };
    ("INFERENCE_MLX_FAULT_NEXT_OPERATION", value)
}

impl Fault {
    fn environment(self, operation: &Operation) -> (&'static str, String) {
        match self {
            // Each import synchronizes once. Each `readback` then synchronizes once. The call
            // closure reads each output back, so the failing step is its last readback.
            Self::Synchronization => (
                "INFERENCE_MLX_FAIL_NEXT_SYNC",
                (operation.imports + 1).to_string(),
            ),
            Self::NativeCall => operation_fault("call", operation),
            Self::NullOutput => operation_fault("null", operation),
            Self::Evaluation => operation_fault("evaluate", operation),
            Self::OutputMismatch => operation_fault("mismatch", operation),
            Self::Submission => ("INFERENCE_MLX_FAIL_ASYNC_EVAL", "1".to_string()),
        }
    }

    /// How many outputs `readback` has evaluated when the fault strikes. A failed build call
    /// leaves no evaluated output. A failed evaluation leaves the first output at least.
    /// A failed synchronization follows the evaluation of every read back output.
    fn evaluated_outputs(self, operation: &Operation) -> usize {
        match self {
            Self::NativeCall | Self::NullOutput | Self::OutputMismatch | Self::Submission => 0,
            Self::Evaluation => 1,
            Self::Synchronization => operation.outputs,
        }
    }
}

fn assert_failed_operation_is_terminal_and_retained(
    test: &str,
    operation: &Operation,
    fault: Fault,
) {
    if env::var_os("INFERENCE_NATIVE_FAILURE_CHILD").is_none() {
        let (name, value) = fault.environment(operation);
        spawn_child(
            test,
            &[("INFERENCE_NATIVE_FAILURE_CHILD", "1"), (name, &value)],
        );
        return;
    }

    let label = operation.expected_label(fault);
    let mut kernels = MlxKernels::try_new().unwrap();
    let mut operands = (operation.prepare)(&mut kernels);
    let (before, _, _) = native_bytes();
    let failed = (operation.call)(&mut kernels, &mut operands);
    assert!(
        failed.is_err(),
        "{label}: the injected failure did not fail the call"
    );
    // A failed selected call produces no output. Outputs of earlier calls stay retained.
    let retained_outputs = fault.evaluated_outputs(operation);
    let floor = before + RETAINED_BYTES * retained_outputs;
    assert_active_bytes_stay_at_least(floor, "after the failed call");
    let Err(EngError::Backend {
        operation: failed_label,
        message,
    }) = &failed
    else {
        panic!("{label}: expected Backend, got {:?}", failed.as_ref().err());
    };
    assert_eq!(*failed_label, label);
    if matches!(fault, Fault::OutputMismatch) {
        assert!(message.contains("output mismatch"), "message: {message}");
    }
    assert!(matches!(
        (operation.call)(&mut kernels, &mut operands),
        Err(EngError::Backend {
            operation: "execute",
            ..
        })
    ));
    drop(operands);
    drop(kernels);
    assert_active_bytes_stay_at_least(floor, "after releasing the adapter");
    assert!(matches!(
        MlxKernels::try_new(),
        Err(EngError::Backend {
            operation: "initialize",
            ..
        })
    ));
}

macro_rules! fault_test {
    ($test:expr, $operation:expr, $fault:expr) => {
        assert_failed_operation_is_terminal_and_retained($test, &$operation, $fault)
    };
}

macro_rules! sync_test {
    ($name:ident, $operation:expr) => {
        #[test]
        fn $name() {
            fault_test!(stringify!($name), $operation, Fault::Synchronization);
        }
    };
}

macro_rules! fault_case {
    ($module:ident, $name:ident, $operation:expr, $fault:expr) => {
        #[test]
        fn $name() {
            fault_test!(
                concat!(stringify!($module), "::", stringify!($name)),
                $operation,
                $fault
            );
        }
    };
}

// Each suite covers the four injected failures of one operation's dispatch path.
macro_rules! fault_suite {
    ($module:ident, $operation:expr) => {
        mod $module {
            use super::*;

            fault_case!(
                $module,
                native_call_is_terminal,
                $operation,
                Fault::NativeCall
            );
            fault_case!(
                $module,
                null_output_is_terminal,
                $operation,
                Fault::NullOutput
            );
            fault_case!(
                $module,
                evaluation_is_terminal_and_retains_output,
                $operation,
                Fault::Evaluation
            );
            fault_case!(
                $module,
                output_mismatch_is_terminal_and_retains_output,
                $operation,
                Fault::OutputMismatch
            );
        }
    };
}

sync_test!(
    failed_silu_synchronization_is_terminal_and_retains_output,
    SILU
);
sync_test!(
    failed_softmax_synchronization_is_terminal_and_retains_output,
    SOFTMAX
);
sync_test!(
    failed_mul_synchronization_is_terminal_and_retains_output,
    MUL
);
sync_test!(
    failed_add_synchronization_is_terminal_and_retains_output,
    ADD
);
sync_test!(failed_mm_synchronization_is_terminal_and_retains_output, MM);
sync_test!(
    failed_qmm_synchronization_is_terminal_and_retains_output,
    QMM
);

sync_test!(
    failed_rms_norm_synchronization_is_terminal_and_retains_output,
    RMS_NORM
);
sync_test!(
    failed_rope_q_synchronization_is_terminal_and_retains_output,
    ROPE
);
sync_test!(
    failed_rope_k_synchronization_is_terminal_and_retains_both_outputs,
    ROPE_K
);

sync_test!(
    failed_attention_synchronization_is_terminal_and_retains_output,
    ATTENTION
);
sync_test!(
    failed_broadcast_rows_synchronization_is_terminal_and_retains_output,
    BROADCAST_ROWS
);
sync_test!(
    failed_weighted_row_sum_synchronization_is_terminal_and_retains_output,
    WEIGHTED_ROW_SUM
);
sync_test!(
    failed_gated_delta_synchronization_is_terminal_and_retains_output,
    GATED_DELTA
);
sync_test!(
    failed_kv_append_synchronization_is_terminal_and_retains_output,
    KV_APPEND
);
sync_test!(
    failed_grouped_attention_prefix_synchronization_is_terminal_and_retains_output,
    GROUPED_ATTENTION_PREFIX
);

fault_suite!(silu, SILU);
fault_suite!(softmax, SOFTMAX);
fault_suite!(mul, MUL);
fault_suite!(add, ADD);
fault_suite!(sigmoid, SIGMOID);
fault_suite!(subtract, SUBTRACT);
fault_suite!(mm, MM);
fault_suite!(qmm, QMM);
fault_suite!(rms_norm, RMS_NORM);
fault_suite!(rope, ROPE);
fault_suite!(rope_k, ROPE_K);

// A mismatch at the SDPA call is not observable. The injected unit axis keeps the element count,
// and the output reshape (selected call 2) restores `[1, 1, dim]`. The mismatch case for attention
// strikes the reshape, in `attention_output_reshape`.
mod attention_sdpa {
    use super::*;

    fault_case!(
        attention_sdpa,
        native_call_is_terminal,
        ATTENTION,
        Fault::NativeCall
    );
    fault_case!(
        attention_sdpa,
        null_output_is_terminal,
        ATTENTION,
        Fault::NullOutput
    );
    fault_case!(
        attention_sdpa,
        evaluation_is_terminal_and_retains_output,
        ATTENTION,
        Fault::Evaluation
    );
}

fault_suite!(broadcast_rows, BROADCAST_ROWS);
fault_suite!(weighted_row_sum_reshape, WEIGHTED_ROW_SUM);
fault_suite!(weighted_row_sum_multiply, WEIGHTED_ROW_SUM_MULTIPLY);
fault_suite!(weighted_row_sum_sum, WEIGHTED_ROW_SUM_SUM);
fault_suite!(reduce_mean, MEAN);
fault_suite!(reduce_max, MAX);
fault_suite!(transpose_axes, TRANSPOSE);
fault_suite!(repeat_axis, REPEAT);
fault_suite!(slice_window, SLICE);
fault_suite!(slice_update_window, SLICE_UPDATE);
fault_suite!(kv_append_window, KV_APPEND);
fault_suite!(concatenate_parts, CONCATENATE);

// `expand_dims` returns a view of its input, so an evaluation adds no device memory and the
// retained-bytes floor of the evaluation case cannot observe the output.
mod expand_dims_axis {
    use super::*;

    fault_case!(
        expand_dims_axis,
        native_call_is_terminal,
        EXPAND_DIMS,
        Fault::NativeCall
    );
    fault_case!(
        expand_dims_axis,
        null_output_is_terminal,
        EXPAND_DIMS,
        Fault::NullOutput
    );
    fault_case!(
        expand_dims_axis,
        output_mismatch_is_terminal_and_retains_output,
        EXPAND_DIMS,
        Fault::OutputMismatch
    );
}

fault_suite!(argmax_axis, ARGMAX);
fault_suite!(argsort_axis, ARGSORT);
fault_suite!(u32_to_f32_convert, U32_TO_F32);
fault_suite!(take_gather, TAKE);
fault_suite!(take_index_clamp, TAKE_CLAMP);
fault_suite!(take_along_axis_gather, TAKE_ALONG_AXIS);
fault_suite!(gather_qmm_experts, GATHER_QMM);
fault_suite!(dequantize_weight, DEQUANTIZE);
fault_suite!(rope_scaled_rotation, ROPE_SCALED);
fault_suite!(grouped_attention_heads, GROUPED_ATTENTION);
fault_suite!(grouped_attention_prefix_heads, GROUPED_ATTENTION_PREFIX);
fault_suite!(gated_delta_rule, GATED_DELTA);

// Only the un-counted submission fault applies to the async submission of `evaluate_async`
// (packet NATIVE-OPS-1 fix1): the submission fills no output array, so the null, evaluation, and
// mismatch faults of the selection have no strike point on it, and the submission consumes no
// operation-fault slot (guard.cpp documents why).
#[test]
fn evaluate_async_native_call_failure_is_terminal_and_retains_output() {
    fault_test!(
        "evaluate_async_native_call_failure_is_terminal_and_retains_output",
        EVALUATE_ASYNC,
        Fault::Submission
    );
}

/// A counted fault aimed at the selected operation after an `evaluate_async` call reaches that
/// operation unconsumed (fix1 task 2a): the submission takes part in no selection, so `call:2`
/// strikes the second `silu` build while the `evaluate_async` between the builds succeeds. With
/// the old shim, which spent a selection slot on the submission, the `evaluate_async` call
/// itself failed instead and the aimed fault never struck.
#[test]
fn evaluate_async_consumes_no_operation_fault_slot() {
    if env::var_os("INFERENCE_NATIVE_FAILURE_CHILD").is_none() {
        spawn_child(
            "evaluate_async_consumes_no_operation_fault_slot",
            &[
                ("INFERENCE_NATIVE_FAILURE_CHILD", "1"),
                ("INFERENCE_MLX_FAULT_NEXT_OPERATION", "call:2"),
            ],
        );
        return;
    }

    let mut kernels = MlxKernels::try_new().unwrap();
    let x = import_f32(&mut kernels, vec![RETAINED_ELEMENTS]);
    let first = kernels.silu(&x).unwrap(); // Selected operation one.
    // The submission of `evaluate_async` is no selected operation and must leave the aimed
    // `call:2` untouched; the `unwrap` fails the test if it consumes the fault instead.
    kernels.evaluate_async(&[&first]).unwrap();
    let second = kernels.silu(&x); // Selected operation two: the fault strikes here.
    assert!(
        matches!(
            &second,
            Err(EngError::Backend {
                operation: "SiLU",
                ..
            })
        ),
        "the counted fault must strike the operation after the submission: {:?}",
        second.err()
    );
    assert!(matches!(
        kernels.silu(&x),
        Err(EngError::Backend {
            operation: "execute",
            ..
        })
    ));
    drop((first, x));
    drop(kernels);
    assert!(matches!(
        MlxKernels::try_new(),
        Err(EngError::Backend {
            operation: "initialize",
            ..
        })
    ));
}

/// The release of the consumed target handle inside `kv_append` follows decision 0022 (fix1
/// task 2b): the injected release failure makes the adapter terminal and the call reports
/// `Backend` with label `kv append` instead of returning `Ok` from a terminal adapter. The
/// taken handle and the built output stay retained.
#[test]
fn kv_append_release_failure_is_terminal_and_reported() {
    if env::var_os("INFERENCE_NATIVE_RELEASE_CHILD").is_none() {
        spawn_child(
            "kv_append_release_failure_is_terminal_and_reported",
            &[
                ("INFERENCE_NATIVE_RELEASE_CHILD", "1"),
                ("INFERENCE_MLX_FAIL_NEXT_FREE", "array"),
            ],
        );
        return;
    }

    let mut kernels = MlxKernels::try_new().unwrap();
    let update = import_f32(&mut kernels, vec![1]);
    let target = import_f32(&mut kernels, vec![RETAINED_ELEMENTS]);
    let (before, _, _) = native_bytes();
    assert!(matches!(
        kernels.kv_append(target, &update, &[0]),
        Err(EngError::Backend {
            operation: "kv append",
            ..
        })
    ));
    // The failed release retained the taken handle (decision 0022).
    assert_active_bytes_stay_at_least(before, "after the failed release");
    assert!(matches!(
        kernels.import_tensor(&small_f32(vec![1])),
        Err(EngError::Backend {
            operation: "execute",
            ..
        })
    ));
    drop((update, kernels));
    assert!(matches!(
        MlxKernels::try_new(),
        Err(EngError::Backend {
            operation: "initialize",
            ..
        })
    ));
}

mod attention_output_reshape {
    use super::*;

    fault_case!(
        attention_output_reshape,
        native_call_is_terminal,
        ATTENTION_OUTPUT,
        Fault::NativeCall
    );
    fault_case!(
        attention_output_reshape,
        null_output_is_terminal,
        ATTENTION_OUTPUT,
        Fault::NullOutput
    );
    fault_case!(
        attention_output_reshape,
        evaluation_is_terminal_and_retains_output,
        ATTENTION_OUTPUT,
        Fault::Evaluation
    );
    fault_case!(
        attention_output_reshape,
        output_mismatch_is_terminal_and_retains_output,
        ATTENTION_OUTPUT_MISMATCH,
        Fault::OutputMismatch
    );
}

// The completed-host operations keep their native arrays in the adapter until completion is known.
fn assert_failed_host_call_retains_pending<T>(
    test: &str,
    retained_bytes: usize,
    setup: fn() -> T,
    call: fn(&mut MlxKernels, &T) -> Result<Tensor, EngError>,
) {
    if env::var_os("INFERENCE_NATIVE_FAILURE_CHILD").is_none() {
        // The host call performs no import, so its own synchronization is the first.
        spawn_child(
            test,
            &[
                ("INFERENCE_NATIVE_FAILURE_CHILD", "1"),
                ("INFERENCE_MLX_FAIL_NEXT_SYNC", "1"),
            ],
        );
        return;
    }

    let inputs = setup();
    let mut kernels = MlxKernels::try_new().unwrap();
    let (before, _, _) = native_bytes();
    assert!(matches!(
        call(&mut kernels, &inputs),
        Err(EngError::Backend {
            operation: "synchronize",
            ..
        })
    ));
    let floor = before + retained_bytes;
    assert_active_bytes_stay_at_least(floor, "after the failed call");
    assert!(matches!(
        call(&mut kernels, &inputs),
        Err(EngError::Backend {
            operation: "execute",
            ..
        })
    ));
    drop(kernels);
    assert_active_bytes_stay_at_least(floor, "after releasing the adapter");
    assert!(matches!(
        MlxKernels::try_new(),
        Err(EngError::Backend {
            operation: "initialize",
            ..
        })
    ));
}

#[test]
fn failed_dequant_matmul_synchronization_retains_pending_arrays() {
    assert_failed_host_call_retains_pending(
        "failed_dequant_matmul_synchronization_retains_pending_arrays",
        RETAINED_ELEMENTS as usize * 32,
        || {
            (
                small_f32(vec![1, 64]),
                large_affine_weight(RETAINED_ELEMENTS),
            )
        },
        |kernels, (activation, weight)| kernels.dequant_matmul(activation, weight),
    );
}

#[test]
fn failed_dense_matmul_synchronization_retains_pending_arrays() {
    assert_failed_host_call_retains_pending(
        "failed_dense_matmul_synchronization_retains_pending_arrays",
        RETAINED_BYTES,
        || (small_f32(vec![1, 1]), small_f32(vec![1, RETAINED_ELEMENTS])),
        |kernels, (activation, weight)| kernels.matmul(activation, weight),
    );
}

#[test]
fn failed_embedding_rows_synchronization_retains_pending_arrays() {
    assert_failed_host_call_retains_pending(
        "failed_embedding_rows_synchronization_retains_pending_arrays",
        RETAINED_BYTES,
        || {
            let indices = Tensor::try_from_host(Dtype::U32, vec![1], vec![0; 4]).unwrap();
            (small_f32(vec![RETAINED_ELEMENTS, 1]), indices)
        },
        |kernels, (table, indices)| kernels.embedding_rows(table, indices),
    );
}

// A completed-host call recovers when its synchronization succeeds. See "Native failure policy"
// in the kernels design. The injected failure takes the first matching call only.
fn assert_host_failure_recovers(
    test: &str,
    fault: &str,
    label: &'static str,
    expected: &[f32],
    call: fn(&mut MlxKernels) -> Result<Tensor, EngError>,
) {
    if env::var_os("INFERENCE_NATIVE_FAILURE_CHILD").is_none() {
        spawn_child(
            test,
            &[
                ("INFERENCE_NATIVE_FAILURE_CHILD", "1"),
                ("INFERENCE_MLX_FAULT_NEXT_OPERATION", fault),
            ],
        );
        return;
    }

    let mut kernels = MlxKernels::try_new().unwrap();
    let failed = call(&mut kernels);
    assert!(
        matches!(&failed, Err(EngError::Backend { operation, .. }) if *operation == label),
        "expected Backend {label}, got {:?}",
        failed.as_ref().err()
    );
    let recovered = call(&mut kernels).expect("the adapter must serve the next call");
    drop(kernels);
    let mut replacement = MlxKernels::try_new().expect("a completed host failure is not terminal");
    let fresh = call(&mut replacement).unwrap();
    assert_eq!(recovered.shape(), fresh.shape());
    assert_eq!(recovered.host_bytes(), fresh.host_bytes());
    let bytes: Vec<u8> = expected
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect();
    assert_eq!(recovered.host_bytes(), Some(bytes.as_slice()));
}

#[test]
fn host_matmul_native_call_failure_is_recoverable() {
    assert_host_failure_recovers(
        "host_matmul_native_call_failure_is_recoverable",
        "call",
        "matmul",
        // One row of ones times a row of ones.
        &[1.0; 4],
        |kernels| kernels.matmul(&small_f32(vec![1, 1]), &small_f32(vec![1, 4])),
    );
}

#[test]
fn host_dequant_matmul_evaluation_failure_is_recoverable() {
    assert_host_failure_recovers(
        "host_dequant_matmul_evaluation_failure_is_recoverable",
        "evaluate",
        "evaluate",
        // Two rows of ones against the weight whose rows sum to -288 and -1216, as in native_matmul.rs.
        &[-288.0, -1216.0, -288.0, -1216.0],
        |kernels| kernels.dequant_matmul(&small_f32(vec![2, 64]), &small_affine_weight()),
    );
}

/// Runs the named test in a child process when the caller is the parent.
///
/// A poisoned adapter is sticky for the whole process, so tests that can poison it run in a child.
fn delegated_to_child(test: &str) -> bool {
    if env::var_os("INFERENCE_NATIVE_FAILURE_CHILD").is_some() {
        return false;
    }
    spawn_child(test, &[("INFERENCE_NATIVE_FAILURE_CHILD", "1")]);
    true
}

fn empty_f32(shape: Vec<u32>) -> Tensor {
    Tensor::try_from_host(Dtype::F32, shape, Vec::new()).unwrap()
}

fn assert_adapter_still_reads_back(kernels: &mut MlxKernels) {
    let source = small_f32(vec![3]);
    let resident = kernels.import_tensor(&source).unwrap();
    let back = kernels.readback(&resident).unwrap();
    assert_eq!(back.host_bytes(), source.host_bytes());
}

#[test]
fn zero_element_resident_reads_back_empty_and_leaves_the_adapter_usable() {
    if delegated_to_child("zero_element_resident_reads_back_empty_and_leaves_the_adapter_usable") {
        return;
    }
    let mut kernels = MlxKernels::try_new().unwrap();
    for shape in [vec![0], vec![0, 3], vec![2, 0]] {
        let resident = kernels.import_tensor(&empty_f32(shape.clone())).unwrap();
        let back = kernels.readback(&resident).unwrap();
        assert_eq!(back.dtype(), Dtype::F32);
        assert_eq!(back.shape(), shape);
        assert_eq!(back.byte_len(), 0);
    }
    assert_adapter_still_reads_back(&mut kernels);
}

#[test]
fn rms_norm_rejects_a_zero_element_input_and_leaves_the_adapter_usable() {
    if delegated_to_child("rms_norm_rejects_a_zero_element_input_and_leaves_the_adapter_usable") {
        return;
    }
    let mut kernels = MlxKernels::try_new().unwrap();
    let x = kernels.import_tensor(&empty_f32(vec![0, 3])).unwrap();
    let weight = import_f32(&mut kernels, vec![3]);
    let rejected = kernels.rms_norm(&x, &weight, 1e-6);
    assert!(
        matches!(rejected, Err(EngError::BadRequest(_))),
        "a zero-element input must fail with BadRequest, got {:?}",
        rejected.as_ref().err()
    );
    assert_adapter_still_reads_back(&mut kernels);
}

// The child body of the deadline case below. Without the environment variable it returns at once.
#[test]
fn hung_child_fixture() {
    if env::var_os("INFERENCE_NATIVE_HANG_CHILD").is_some() {
        println!("hung child started");
        thread::sleep(Duration::from_secs(20));
    }
}

#[test]
fn child_deadline_kills_a_hung_child_and_reports_its_output() {
    let started = Instant::now();
    let result = run_child(
        "hung_child_fixture",
        &[("INFERENCE_NATIVE_HANG_CHILD", "1")],
        Duration::from_secs(2),
    );
    let message = result.expect_err("a child that outlives its deadline must fail the run");
    assert!(message.contains("timed out"), "message: {message}");
    assert!(
        message.contains("hung child started"),
        "the message must carry the child output: {message}"
    );
    assert!(
        started.elapsed() < Duration::from_secs(10),
        "the deadline did not stop the child"
    );
}

#[test]
fn failed_vector_release_in_evaluate_is_terminal() {
    if env::var_os("INFERENCE_NATIVE_FAILURE_CHILD").is_none() {
        spawn_child(
            "failed_vector_release_in_evaluate_is_terminal",
            &[
                ("INFERENCE_NATIVE_FAILURE_CHILD", "1"),
                ("INFERENCE_MLX_FAIL_NEXT_FREE", "vector"),
            ],
        );
        return;
    }

    let mut kernels = MlxKernels::try_new().unwrap();
    let a = kernels.import_tensor(&small_f32(vec![2])).unwrap();
    let s = kernels.add(&a, &a).unwrap();
    assert!(matches!(
        kernels.evaluate(&[&s]),
        Err(EngError::Backend {
            operation: "evaluate",
            ..
        })
    ));
    assert!(matches!(
        kernels.import_tensor(&small_f32(vec![2])),
        Err(EngError::Backend {
            operation: "execute",
            ..
        })
    ));
    drop(s);
    drop(a);
    drop(kernels);
    assert!(matches!(
        MlxKernels::try_new(),
        Err(EngError::Backend {
            operation: "initialize",
            ..
        })
    ));
}

#[test]
fn failed_import_many_synchronization_is_terminal() {
    if env::var_os("INFERENCE_NATIVE_FAILURE_CHILD").is_none() {
        spawn_child(
            "failed_import_many_synchronization_is_terminal",
            &[
                ("INFERENCE_NATIVE_FAILURE_CHILD", "1"),
                ("INFERENCE_MLX_FAIL_NEXT_SYNC", "1"),
            ],
        );
        return;
    }

    let first = small_f32(vec![2]);
    let second = small_f32(vec![2]);
    let mut kernels = MlxKernels::try_new().unwrap();
    assert!(matches!(
        kernels.import_many(&[&first, &second]),
        Err(EngError::Backend {
            operation: "synchronize",
            ..
        })
    ));
    assert!(matches!(
        kernels.import_tensor(&first),
        Err(EngError::Backend {
            operation: "execute",
            ..
        })
    ));
    drop(kernels);
    assert!(matches!(
        MlxKernels::try_new(),
        Err(EngError::Backend {
            operation: "initialize",
            ..
        })
    ));
}

#[test]
fn failed_cap_flush_synchronization_is_terminal() {
    if env::var_os("INFERENCE_NATIVE_FAILURE_CHILD").is_none() {
        spawn_child(
            "failed_cap_flush_synchronization_is_terminal",
            &[
                ("INFERENCE_NATIVE_FAILURE_CHILD", "1"),
                ("INFERENCE_MLX_FAIL_NEXT_SYNC", "2"),
            ],
        );
        return;
    }

    // 17,000,000 F32 values are 68 MB, which exceeds the 64 MiB retire cap.
    let large = Tensor::try_from_host(Dtype::F32, vec![17_000_000], vec![0u8; 68_000_000]).unwrap();
    let mut kernels = MlxKernels::try_new().unwrap();
    let resident = kernels.import_tensor(&large).unwrap();
    // The import is synchronization 1. Dropping the resident runs the cap flush, which is synchronization 2.
    drop(resident);
    assert!(matches!(
        kernels.import_tensor(&small_f32(vec![2])),
        Err(EngError::Backend {
            operation: "execute",
            ..
        })
    ));
    drop(kernels);
    assert!(matches!(
        MlxKernels::try_new(),
        Err(EngError::Backend {
            operation: "initialize",
            ..
        })
    ));
}

#[test]
fn failed_last_share_synchronization_is_terminal() {
    if env::var_os("INFERENCE_NATIVE_FAILURE_CHILD").is_none() {
        spawn_child(
            "failed_last_share_synchronization_is_terminal",
            &[
                ("INFERENCE_NATIVE_FAILURE_CHILD", "1"),
                ("INFERENCE_MLX_FAIL_NEXT_SYNC", "3"),
            ],
        );
        return;
    }

    let mut kernels = MlxKernels::try_new().unwrap();
    let resident = kernels.import_tensor(&small_f32(vec![2])).unwrap();
    // The import is synchronization 1. Dropping the adapter is synchronization 2.
    drop(kernels);
    // Dropping the last share of the retire core is synchronization 3, which fails.
    drop(resident);
    assert!(matches!(
        MlxKernels::try_new(),
        Err(EngError::Backend {
            operation: "initialize",
            ..
        })
    ));
}
