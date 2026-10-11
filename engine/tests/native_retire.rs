#![cfg(all(
    target_os = "macos",
    target_arch = "aarch64",
    feature = "native-fault-qualification"
))]

use std::{
    env,
    io::Read,
    process::{Command, Stdio},
    sync::{Mutex, MutexGuard, PoisonError},
    thread,
    time::{Duration, Instant},
};

use inference_engine::{
    EngError,
    kernels::MlxKernels,
    tensor::{Dtype, Tensor},
};

static GPU_TEST: Mutex<()> = Mutex::new(());

/// Serializes the GPU tests. A panic in one test must not fail the others with a poison error.
fn serial() -> MutexGuard<'static, ()> {
    GPU_TEST.lock().unwrap_or_else(PoisonError::into_inner)
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn f32_tensor(shape: Vec<u32>, values: &[f32]) -> Tensor {
    Tensor::try_from_host(Dtype::F32, shape, f32_bytes(values)).unwrap()
}

fn filled_f32_tensor(elements: usize, value: f32) -> Tensor {
    let bytes = std::iter::repeat_n(value.to_le_bytes(), elements)
        .flatten()
        .collect();
    Tensor::try_from_host(Dtype::F32, vec![elements as u32], bytes).unwrap()
}

#[allow(unsafe_code)]
fn active_bytes() -> usize {
    let mut active = 0usize;
    assert_eq!(unsafe { mlx_sys::mlx_get_active_memory(&mut active) }, 0);
    active
}

/// Waits up to 200 ms for active memory to fall to at most `ceiling` bytes.
fn assert_active_falls_to(ceiling: usize, context: &str) {
    for _ in 0..20 {
        if active_bytes() <= ceiling {
            return;
        }
        thread::sleep(Duration::from_millis(10));
    }
    panic!(
        "{context}: active {} bytes stayed above {ceiling} bytes",
        active_bytes()
    );
}

/// Asserts that `MlxKernels::try_new` refuses admission because two adapter leases are held.
fn assert_admission_refused(context: &str) {
    match MlxKernels::try_new() {
        Err(EngError::Backend { message, .. }) => assert!(
            message.contains("two native adapter leases"),
            "{context}: unexpected message {message}"
        ),
        Err(other) => panic!("{context}: unexpected error {other:?}"),
        Ok(_) => panic!("{context}: a third adapter was admitted"),
    }
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
fn retire_costs_no_synchronization() {
    let _gpu = serial();
    let mut kernels = MlxKernels::try_new().unwrap();
    let resident = kernels
        .import_tensor(&f32_tensor(vec![4], &[1.0, 2.0, 3.0, 4.0]))
        .unwrap();
    let before = kernels.qualification_sync_count();
    drop(resident);
    assert_eq!(
        kernels.qualification_sync_count(),
        before,
        "a retire below the cap adds no synchronization"
    );
}

#[test]
fn import_many_imports_all_with_one_synchronization() {
    let _gpu = serial();
    let mut kernels = MlxKernels::try_new().unwrap();
    let sources = [
        f32_tensor(vec![2], &[1.0, 2.0]),
        f32_tensor(vec![2], &[3.0, 4.0]),
        f32_tensor(vec![2], &[5.0, 6.0]),
    ];
    let refs: Vec<&Tensor> = sources.iter().collect();
    let before = kernels.qualification_sync_count();
    let imported = kernels.import_many(&refs).unwrap();
    assert_eq!(
        kernels.qualification_sync_count(),
        before + 1,
        "import_many uses one synchronization"
    );
    assert_eq!(imported.len(), sources.len());
    for (resident, source) in imported.iter().zip(&sources) {
        let back = kernels.readback(resident).unwrap();
        assert_eq!(back.host_bytes(), source.host_bytes());
    }
}

#[test]
fn evaluate_completes_several_outputs() {
    let _gpu = serial();
    let mut kernels = MlxKernels::try_new().unwrap();
    let a = kernels
        .import_tensor(&f32_tensor(vec![2], &[1.0, 2.0]))
        .unwrap();
    let b = kernels
        .import_tensor(&f32_tensor(vec![2], &[10.0, 20.0]))
        .unwrap();
    let sum = kernels.add(&a, &b).unwrap();
    let product = kernels.mul(&a, &b).unwrap();
    kernels.evaluate(&[&sum, &product]).unwrap();
    let sum_back = kernels.readback(&sum).unwrap();
    assert_eq!(
        sum_back.host_bytes(),
        Some(f32_bytes(&[11.0, 22.0]).as_slice())
    );
    let product_back = kernels.readback(&product).unwrap();
    assert_eq!(
        product_back.host_bytes(),
        Some(f32_bytes(&[10.0, 40.0]).as_slice())
    );
}

#[test]
fn evaluate_rejects_empty_and_foreign_outputs() {
    let _gpu = serial();
    let mut kernels = MlxKernels::try_new().unwrap();
    let mut other = MlxKernels::try_new().unwrap();
    let foreign = other.import_tensor(&f32_tensor(vec![1], &[1.0])).unwrap();
    assert!(matches!(
        kernels.evaluate(&[]),
        Err(EngError::BadRequest(_))
    ));
    assert!(matches!(
        kernels.evaluate(&[&foreign]),
        Err(EngError::BadRequest(_))
    ));
    let own = kernels
        .import_tensor(&f32_tensor(vec![2], &[1.0, 2.0]))
        .unwrap();
    let back = kernels.readback(&own).unwrap();
    assert_eq!(back.host_bytes(), Some(f32_bytes(&[1.0, 2.0]).as_slice()));
}

#[test]
fn dropped_inputs_of_a_pending_graph_stay_valid() {
    let _gpu = serial();
    let mut kernels = MlxKernels::try_new().unwrap();
    let a = kernels
        .import_tensor(&f32_tensor(vec![2], &[1.0, 2.0]))
        .unwrap();
    let b = kernels
        .import_tensor(&f32_tensor(vec![2], &[10.0, 20.0]))
        .unwrap();
    let c = kernels.add(&a, &b).unwrap();
    drop(a);
    drop(b);
    let back = kernels.readback(&c).unwrap();
    assert_eq!(back.host_bytes(), Some(f32_bytes(&[11.0, 22.0]).as_slice()));
}

#[test]
fn retire_cap_forces_one_flush() {
    let _gpu = serial();
    let mut kernels = MlxKernels::try_new().unwrap();
    // 17,000,000 F32 values are 68,000,000 bytes, which exceeds the 64 MiB retire cap.
    let large = Tensor::try_from_host(Dtype::F32, vec![17_000_000], vec![0u8; 68_000_000]).unwrap();
    let resident = kernels.import_tensor(&large).unwrap();
    let before = kernels.qualification_sync_count();
    drop(resident);
    assert_eq!(
        kernels.qualification_sync_count(),
        before + 1,
        "exceeding the retire cap forces one flush synchronization"
    );
}

#[test]
fn releasing_the_last_resident_after_the_adapter_admits_a_new_adapter() {
    let _gpu = serial();
    let source = f32_tensor(vec![1], &[1.0]);
    let mut first = MlxKernels::try_new().unwrap();
    let resident = first.import_tensor(&source).unwrap();
    drop(first);
    let second = MlxKernels::try_new().unwrap();
    drop(second);
    drop(resident);
    let x = MlxKernels::try_new().unwrap();
    let y = MlxKernels::try_new().unwrap();
    drop(x);
    drop(y);
}

#[test]
fn failed_evaluate_synchronization_is_terminal() {
    let _gpu = serial();
    if env::var_os("INFERENCE_NATIVE_RETIRE_CHILD").is_none() {
        spawn_child(
            "failed_evaluate_synchronization_is_terminal",
            &[
                ("INFERENCE_NATIVE_RETIRE_CHILD", "1"),
                ("INFERENCE_MLX_FAIL_NEXT_SYNC", "2"),
            ],
        );
        return;
    }

    let mut kernels = MlxKernels::try_new().unwrap();
    let a = kernels
        .import_tensor(&f32_tensor(vec![2], &[1.0, 2.0]))
        .unwrap();
    let sum = kernels.add(&a, &a).unwrap();
    assert!(matches!(
        kernels.evaluate(&[&sum]),
        Err(EngError::Backend {
            operation: "synchronize",
            ..
        })
    ));
    assert!(matches!(
        kernels.evaluate(&[&sum]),
        Err(EngError::Backend {
            operation: "execute",
            ..
        })
    ));
}

#[test]
fn helper_filled_f32_tensor_has_the_requested_size() {
    let _gpu = serial();
    let tensor = filled_f32_tensor(5, 2.0);
    assert_eq!(tensor.shape(), &[5u32]);
    assert_eq!(tensor.byte_len(), 20);
}

#[test]
fn cap_flush_frees_memory_and_resets_the_counter() {
    let _gpu = serial();
    let mut kernels = MlxKernels::try_new().unwrap();
    // 17,000,000 F32 values are 68 MB, which exceeds the 64 MiB retire cap.
    let first = kernels
        .import_tensor(&filled_f32_tensor(17_000_000, 1.0))
        .unwrap();
    let floor = active_bytes();
    drop(first);
    assert_active_falls_to(
        floor - 60_000_000,
        "a retire past the cap frees the 68 MB array",
    );
    let second = kernels
        .import_tensor(&filled_f32_tensor(17_000_000, 1.0))
        .unwrap();
    let before = kernels.qualification_sync_count();
    drop(second);
    assert_eq!(
        kernels.qualification_sync_count(),
        before + 1,
        "the cap flush adds exactly one synchronization"
    );
}

#[test]
fn readback_flushes_retired_memory() {
    let _gpu = serial();
    let mut kernels = MlxKernels::try_new().unwrap();
    let small = kernels.import_tensor(&filled_f32_tensor(4, 1.0)).unwrap();
    let resident = kernels
        .import_tensor(&filled_f32_tensor(5_000_000, 1.0))
        .unwrap();
    let before = active_bytes();
    drop(resident);
    kernels.readback(&small).unwrap();
    assert_active_falls_to(
        before - 16_000_000,
        "readback flushes the retired 20 MB array",
    );
}

#[test]
fn evaluate_flushes_retired_memory() {
    let _gpu = serial();
    let mut kernels = MlxKernels::try_new().unwrap();
    let small = kernels.import_tensor(&filled_f32_tensor(4, 1.0)).unwrap();
    let resident = kernels
        .import_tensor(&filled_f32_tensor(5_000_000, 1.0))
        .unwrap();
    let before = active_bytes();
    drop(resident);
    kernels.evaluate(&[&small]).unwrap();
    assert_active_falls_to(
        before - 16_000_000,
        "evaluate flushes the retired 20 MB array",
    );
}

#[test]
fn import_flushes_retired_memory() {
    let _gpu = serial();
    let mut kernels = MlxKernels::try_new().unwrap();
    let resident = kernels
        .import_tensor(&filled_f32_tensor(5_000_000, 1.0))
        .unwrap();
    let before = active_bytes();
    drop(resident);
    kernels.import_tensor(&filled_f32_tensor(4, 1.0)).unwrap();
    assert_active_falls_to(
        before - 16_000_000,
        "an import flushes the retired 20 MB array",
    );
}

#[test]
fn import_many_flushes_retired_memory() {
    let _gpu = serial();
    let mut kernels = MlxKernels::try_new().unwrap();
    let resident = kernels
        .import_tensor(&filled_f32_tensor(5_000_000, 1.0))
        .unwrap();
    let before = active_bytes();
    drop(resident);
    kernels.import_many(&[&filled_f32_tensor(4, 1.0)]).unwrap();
    assert_active_falls_to(
        before - 16_000_000,
        "a batch import flushes the retired 20 MB array",
    );
}

#[test]
fn host_call_flushes_retired_memory() {
    let _gpu = serial();
    let mut kernels = MlxKernels::try_new().unwrap();
    let resident = kernels
        .import_tensor(&filled_f32_tensor(5_000_000, 1.0))
        .unwrap();
    let activation = f32_tensor(vec![1, 2], &[1.0, 2.0]);
    let weight = f32_tensor(vec![2, 2], &[1.0, 0.0, 0.0, 1.0]);
    let before = active_bytes();
    drop(resident);
    kernels.matmul(&activation, &weight).unwrap();
    assert_active_falls_to(
        before - 16_000_000,
        "a completed host call flushes the retired 20 MB array",
    );
}

#[test]
fn adapter_drop_flushes_retired_memory() {
    let _gpu = serial();
    let mut kernels = MlxKernels::try_new().unwrap();
    let resident = kernels
        .import_tensor(&filled_f32_tensor(5_000_000, 1.0))
        .unwrap();
    // The kept resident holds the retire core after the adapter drops, so only the adapter's flush can free `resident`.
    let keep = kernels.import_tensor(&filled_f32_tensor(4, 1.0)).unwrap();
    let before = active_bytes();
    drop(resident);
    drop(kernels);
    assert_active_falls_to(
        before - 16_000_000,
        "dropping the adapter flushes the retired 20 MB array",
    );
    drop(keep);
}

#[test]
fn retired_resident_holds_its_adapter_slot() {
    let _gpu = serial();
    let mut first = MlxKernels::try_new().unwrap();
    let resident = first.import_tensor(&filled_f32_tensor(4, 1.0)).unwrap();
    drop(first);
    let second = MlxKernels::try_new().unwrap();
    assert_admission_refused("the retired resident and the second adapter hold both slots");
    drop(second);
    let third = MlxKernels::try_new().unwrap();
    assert_admission_refused("the retired resident still holds one slot");
    drop(third);
    drop(resident);
    let x = MlxKernels::try_new().unwrap();
    let y = MlxKernels::try_new().unwrap();
    drop(x);
    drop(y);
}

#[test]
fn evaluate_uses_exactly_one_synchronization_on_a_deep_graph() {
    let _gpu = serial();
    let mut kernels = MlxKernels::try_new().unwrap();
    let a = kernels
        .import_tensor(&f32_tensor(vec![2], &[1.0, 2.0]))
        .unwrap();
    let b = kernels
        .import_tensor(&f32_tensor(vec![2], &[3.0, 4.0]))
        .unwrap();
    // c = a + b = [4, 6], d = c * a = [4, 12], e = d + d = [8, 24].
    let c = kernels.add(&a, &b).unwrap();
    let d = kernels.mul(&c, &a).unwrap();
    let e = kernels.add(&d, &d).unwrap();
    let before = kernels.qualification_sync_count();
    kernels.evaluate(&[&e]).unwrap();
    assert_eq!(
        kernels.qualification_sync_count(),
        before + 1,
        "evaluating a deep graph uses exactly one synchronization"
    );
    let back = kernels.readback(&e).unwrap();
    assert_eq!(back.host_bytes(), Some(f32_bytes(&[8.0, 24.0]).as_slice()));
}

#[test]
fn retire_stats_count_a_forced_flush_and_the_peak() {
    let _serial = serial();
    let mut kernels = MlxKernels::try_new().unwrap();
    // Three 32 MiB residents. Retiring the third makes 96 MiB, above the 64 MiB cap.
    let tensors: Vec<_> = (0..3)
        .map(|_| {
            kernels
                .import_tensor(&filled_f32_tensor(8 << 20, 1.0))
                .unwrap()
        })
        .collect();
    kernels.qualification_reset_retire_peak();
    let (flushes_before, _) = kernels.qualification_retire_stats();
    drop(tensors);
    let (flushes_after, peak) = kernels.qualification_retire_stats();
    assert_eq!(flushes_after - flushes_before, 1, "one forced flush");
    assert_eq!(peak, 3 * (32 << 20), "peak retired bytes");
}

#[test]
fn qualification_timing_counts_each_synchronizing_call() {
    let _serial = serial();
    let mut kernels = MlxKernels::try_new().unwrap();
    // A call before the reset, so a reset that keeps the counts fails even when this test runs alone.
    kernels.import_tensor(&f32_tensor(vec![1], &[0.0])).unwrap();
    kernels.qualification_reset_timing();
    assert_eq!(
        kernels.qualification_timing(),
        Default::default(),
        "reset sets every count and time to zero"
    );
    let x = kernels
        .import_tensor(&f32_tensor(vec![4], &[1.0, 2.0, 3.0, 4.0]))
        .unwrap();
    let both = kernels
        .import_many(&[
            &f32_tensor(vec![2], &[5.0, 6.0]),
            &f32_tensor(vec![2], &[7.0, 8.0]),
        ])
        .unwrap();
    let y = kernels.add(&both[0], &both[1]).unwrap();
    kernels.evaluate(&[&y]).unwrap();
    let host = kernels.readback(&x).unwrap();
    assert_eq!(host.host_bytes().unwrap().len(), 16);
    let t = kernels.qualification_timing();
    assert_eq!(
        (t.import_calls, t.readback_calls, t.evaluate_calls),
        (2, 1, 1),
        "one call each for import_tensor, import_many, evaluate, and readback"
    );
    assert!(
        t.import_ns > 0 && t.readback_ns > 0 && t.evaluate_ns > 0,
        "{t:?}"
    );
}

#[test]
fn set_retire_cap_returns_the_previous_cap_and_rejects_zero() {
    let _gpu = serial();
    let mut kernels = MlxKernels::try_new().unwrap();
    assert_eq!(
        kernels.set_retire_cap(128 << 20).unwrap(),
        64 << 20,
        "default cap"
    );
    assert_eq!(
        kernels.set_retire_cap(1 << 20).unwrap(),
        128 << 20,
        "raised cap"
    );
    assert!(
        matches!(kernels.set_retire_cap(0), Err(EngError::BadRequest(_))),
        "zero cap"
    );
    assert_eq!(
        kernels.set_retire_cap(64 << 20).unwrap(),
        1 << 20,
        "a rejected call changes nothing"
    );
}

#[test]
fn a_raised_retire_cap_defers_the_forced_flush() {
    let _gpu = serial();
    let mut kernels = MlxKernels::try_new().unwrap();
    kernels.set_retire_cap(128 << 20).unwrap();
    // 17,000,000 F32 values are 68,000,000 bytes: over 64 MiB, under 128 MiB.
    let resident = kernels
        .import_tensor(&filled_f32_tensor(17_000_000, 1.0))
        .unwrap();
    let before = kernels.qualification_sync_count();
    drop(resident);
    assert_eq!(
        kernels.qualification_sync_count(),
        before,
        "68 MB stays under a 128 MiB cap"
    );
}

#[test]
fn a_lowered_retire_cap_forces_a_flush_for_a_small_retire() {
    let _gpu = serial();
    let mut kernels = MlxKernels::try_new().unwrap();
    kernels.set_retire_cap(1 << 20).unwrap();
    // 524,288 F32 values are 2,097,152 bytes: over 1 MiB, far under 64 MiB.
    let resident = kernels
        .import_tensor(&filled_f32_tensor(524_288, 1.0))
        .unwrap();
    let before = kernels.qualification_sync_count();
    drop(resident);
    assert_eq!(
        kernels.qualification_sync_count(),
        before + 1,
        "2 MiB passes a 1 MiB cap"
    );
}
