use std::sync::Arc;
use std::time::{Duration, Instant};

use inference_engine::{
    EngError,
    kernels::{MlxKernels, PackedWeight, QuantizationMode, QuantizationSpec},
    tensor::{Dtype, Tensor},
    weights::{ResidentWeights, WeightSource},
};

use crate::fixtures::{
    CALLER_COUNT, FIXTURE_A, FIXTURE_B, JOBS_PER_CALLER, OUTPUT_DTYPE, OUTPUT_SHAPE, Output,
    Request, TOTAL_JOBS, expected_output, fixture_for,
};
use crate::probe::swap_used_mib;
use crate::protocol::{self, ComputeFailure, RunError};

const PROTOCOL_DEADLINE: Duration = Duration::from_secs(110);

unsafe extern "C" {
    fn inference_gpu_stream_count(count: *mut usize) -> i32;
}

#[derive(Debug)]
struct StageError {
    operation: &'static str,
    message: String,
}

fn stage<T>(
    operation: &'static str,
    result: Result<T, impl std::fmt::Debug>,
) -> Result<T, StageError> {
    result.map_err(|error| StageError {
        operation,
        message: format!("{error:?}"),
    })
}

fn gpu_streams() -> Result<usize, StageError> {
    let mut count = 0;
    // Safety: the pinned C++ observer is noexcept, writes one usize through this
    // stack pointer, and does not retain it. Every observation follows adapter
    // teardown with no pending native work. All spawned native owners have joined.
    let status = unsafe { inference_gpu_stream_count(&mut count) };
    if status != 0 {
        return Err(StageError {
            operation: "gpu stream count",
            message: format!("observer returned status {status}"),
        });
    }
    Ok(count)
}

struct Fixtures {
    weight: PackedWeight,
    activation_a: Tensor,
    activation_b: Tensor,
}

fn load_fixtures() -> Result<Fixtures, StageError> {
    let source = stage(
        "resident weights",
        ResidentWeights::from_bytes(Arc::from(
            include_bytes!("../../../engine/tests/fixtures/packed.safetensors").as_slice(),
        )),
    )?;
    let weight = stage(
        "packed weight",
        PackedWeight::try_new(
            source.tensor("weight").unwrap(),
            source.tensor("scales").unwrap(),
            Some(source.tensor("biases").unwrap()),
            vec![2, 64],
            QuantizationSpec {
                mode: QuantizationMode::Affine,
                bits: 4,
                group_size: 64,
            },
        ),
    )?;
    let mut values = vec![1.0f32; 64];
    values.push(1.0);
    values.extend([0.0; 63]);
    let bytes = values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect::<Vec<_>>();
    let activation_a = stage(
        "activation A",
        Tensor::try_from_host(Dtype::F32, vec![2, 64], bytes),
    )?;
    let activation_b = stage(
        "activation B",
        Tensor::try_from_host(Dtype::F32, vec![2, 64], vec![0; 512]),
    )?;
    Ok(Fixtures {
        weight,
        activation_a,
        activation_b,
    })
}

fn activation_for(fixture: u8, fixtures: &Fixtures) -> &Tensor {
    if fixture == FIXTURE_A {
        &fixtures.activation_a
    } else {
        &fixtures.activation_b
    }
}

fn to_output(result: &Tensor) -> Result<Output, ComputeFailure> {
    if result.dtype() != Dtype::F32 {
        return Err(ComputeFailure {
            operation: "matmul output",
            message: format!("unexpected dtype {:?}", result.dtype()),
        });
    }
    if result.shape() != OUTPUT_SHAPE {
        return Err(ComputeFailure {
            operation: "matmul output",
            message: format!("unexpected shape {:?}", result.shape()),
        });
    }
    let bytes = result.host_bytes().ok_or_else(|| ComputeFailure {
        operation: "matmul output",
        message: "output requires host storage".into(),
    })?;
    if bytes.len() != 16 {
        return Err(ComputeFailure {
            operation: "matmul output",
            message: format!("unexpected byte count {}", bytes.len()),
        });
    }
    let mut values = [0.0f32; 4];
    for (slot, word) in values.iter_mut().zip(bytes.as_chunks::<4>().0) {
        *slot = f32::from_le_bytes([word[0], word[1], word[2], word[3]]);
    }
    let _ = OUTPUT_DTYPE;
    Ok(Output {
        dtype: OUTPUT_DTYPE,
        shape: OUTPUT_SHAPE,
        values,
    })
}

fn compute_one(
    kernels: &mut MlxKernels,
    fixtures: &Fixtures,
    request: Request,
) -> Result<Output, ComputeFailure> {
    let result = kernels
        .dequant_matmul(activation_for(request.fixture, fixtures), &fixtures.weight)
        .map_err(|error| ComputeFailure {
            operation: "dequant matmul",
            message: format!("{error:?}"),
        })?;
    to_output(&result)
}

fn warmup() -> Result<(), StageError> {
    let fixtures = load_fixtures()?;
    let mut kernels = stage("warmup adapter", MlxKernels::try_new())?;
    for fixture in [FIXTURE_A, FIXTURE_B] {
        let request = Request {
            caller: 0,
            job: 0,
            fixture,
        };
        let output = stage(
            "warmup matmul",
            compute_one(&mut kernels, &fixtures, request),
        )?;
        stage(
            "warmup output",
            if output.same_bytes(&expected_output(fixture)) {
                Ok(())
            } else {
                Err(format!(
                    "warmup fixture {fixture} bytes mismatch: {:?}",
                    output.values
                ))
            },
        )?;
    }
    drop(kernels);
    Ok(())
}

fn serial_baseline() -> Result<Duration, StageError> {
    let fixtures = load_fixtures()?;
    let mut kernels = stage("baseline adapter", MlxKernels::try_new())?;
    let started = Instant::now();
    for caller in 0..CALLER_COUNT {
        for job in 0..JOBS_PER_CALLER {
            let fixture = fixture_for(caller, job);
            let request = Request {
                caller,
                job,
                fixture,
            };
            let output = stage(
                "baseline matmul",
                compute_one(&mut kernels, &fixtures, request),
            )?;
            if !output.same_bytes(&expected_output(fixture)) {
                return Err(StageError {
                    operation: "baseline output",
                    message: format!(
                        "caller {caller} job {job} bytes mismatch: {:?}",
                        output.values
                    ),
                });
            }
        }
    }
    let elapsed = started.elapsed();
    drop(kernels);
    drop(fixtures);
    Ok(elapsed)
}

fn describe_run_error(error: &RunError) -> String {
    format!("{error:?}")
}

fn concurrent_run(deadline: Instant) -> Result<protocol::RunStats, StageError> {
    let started = Instant::now();
    let stats = protocol::run_protocol(deadline, || {
        let fixtures = load_fixtures().expect("fixtures load on the owner thread");
        let mut kernels = MlxKernels::try_new().expect("adapter on the owner thread");
        move |request| compute_one(&mut kernels, &fixtures, request)
    })
    .map_err(|error| StageError {
        operation: "concurrent run",
        message: describe_run_error(&error),
    })?;
    println!(
        "{{\"stage\":\"concurrent\",\"total_jobs\":{},\"max_post_dequeue_queue_len\":{},\"elapsed_us\":{}}}",
        stats.total_jobs,
        stats.max_post_dequeue_queue_len,
        started.elapsed().as_micros()
    );
    Ok(stats)
}

fn gate_compute(
    caller: u8,
) -> Result<impl FnMut(Request) -> Result<Output, ComputeFailure>, ComputeFailure> {
    let fixtures = load_fixtures().map_err(|error| ComputeFailure {
        operation: error.operation,
        message: error.message,
    })?;
    let mut kernels = MlxKernels::try_new().map_err(|error| ComputeFailure {
        operation: "initialize",
        message: format!("caller {caller}: {error:?}"),
    })?;
    let thread = format!("{:?}", std::thread::current().id());
    println!("{{\"stage\":\"owner_initialized\",\"caller\":{caller},\"thread\":{thread:?}}}");
    Ok(move |request| compute_one(&mut kernels, &fixtures, request))
}

fn reject_third_owner() -> Result<(), ComputeFailure> {
    match MlxKernels::try_new() {
        Err(EngError::Backend {
            operation: "initialize",
            message,
        }) if message == "this process already holds two native adapter leases" => Ok(()),
        Ok(kernels) => {
            drop(kernels);
            Err(ComputeFailure {
                operation: "third owner",
                message: "a third native adapter unexpectedly succeeded".into(),
            })
        }
        Err(error) => Err(ComputeFailure {
            operation: "third owner",
            message: format!("unexpected constructor error: {error:?}"),
        }),
    }
}

fn two_thread_run(deadline: Instant) -> Result<(), StageError> {
    let started = Instant::now();
    let stats = stage(
        "two-thread run",
        crate::gate::run_gate(deadline, gate_compute, reject_third_owner),
    )?;
    let owner_threads = stats
        .owner_threads
        .iter()
        .map(|id| format!("{id:?}"))
        .collect::<Vec<_>>();
    println!(
        "{{\"stage\":\"two_threads\",\"total_jobs\":{},\"jobs_per_owner\":{JOBS_PER_CALLER},\"owner_threads\":{owner_threads:?},\"third_owner_rejected\":true,\"elapsed_us\":{}}}",
        stats.total_jobs,
        started.elapsed().as_micros()
    );
    Ok(())
}

fn check_gate_deadline(two_threads: bool, deadline: Instant) -> Result<(), StageError> {
    if two_threads && Instant::now() >= deadline {
        return Err(StageError {
            operation: "two-thread deadline",
            message: "the gate exceeded its monotonic deadline".into(),
        });
    }
    Ok(())
}

fn replacement() -> Result<(), StageError> {
    match MlxKernels::try_new() {
        Ok(kernels) => {
            drop(kernels);
            println!("{{\"stage\":\"replacement\",\"ok\":true}}");
            Ok(())
        }
        Err(error) => Err(StageError {
            operation: "replacement adapter",
            message: format!("{error:?}"),
        }),
    }
}

fn capture(operation: &'static str, args: &[&str]) -> Result<String, StageError> {
    let output = std::process::Command::new(args[0])
        .args(&args[1..])
        .output()
        .map_err(|error| StageError {
            operation,
            message: error.to_string(),
        })?;
    if !output.status.success() {
        return Err(StageError {
            operation,
            message: format!("exit status {}", output.status),
        });
    }
    Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
}

fn sample_memory(stage_name: &str) -> Result<(), StageError> {
    let pid = std::process::id().to_string();
    let rss_args = ["ps", "-o", "rss=", "-p", &pid];
    let rss_kib = capture("ps", &rss_args)?;
    let swap_report = capture("sysctl swap", &["sysctl", "-n", "vm.swapusage"])?;
    let swap_used = swap_used_mib(&swap_report).map_err(|message| StageError {
        operation: "sysctl swap",
        message,
    })?;
    let pressure_level = capture(
        "sysctl pressure",
        &["sysctl", "-n", "kern.memorystatus_vm_pressure_level"],
    )?;
    println!(
        "{{\"stage\":\"memory_{stage_name}\",\"rss_kib\":{rss_kib},\"swap_used_mib\":{swap_used:.2},\"pressure_level\":{pressure_level}}}"
    );
    Ok(())
}

fn run_stages(two_threads: bool) -> Result<(), StageError> {
    let started = Instant::now();
    let deadline = started + PROTOCOL_DEADLINE;
    sample_memory("before")?;
    warmup()?;
    let warmup_streams = gpu_streams()?;
    println!("{{\"stage\":\"warmup\",\"gpu_streams\":{warmup_streams}}}");
    let baseline_elapsed = serial_baseline()?;
    println!(
        "{{\"stage\":\"baseline\",\"jobs\":{TOTAL_JOBS},\"elapsed_us\":{}}}",
        baseline_elapsed.as_micros()
    );
    if two_threads {
        two_thread_run(deadline)?;
    } else {
        concurrent_run(deadline)?;
    }
    check_gate_deadline(two_threads, deadline)?;
    let final_streams = gpu_streams()?;
    if final_streams != warmup_streams {
        return Err(StageError {
            operation: "registration",
            message: format!(
                "final gpu stream count {final_streams} differs from warmup {warmup_streams}"
            ),
        });
    }
    println!(
        "{{\"stage\":\"registration\",\"gpu_streams\":{final_streams},\"warmup_gpu_streams\":{warmup_streams}}}"
    );
    replacement()?;
    sample_memory("after")?;
    check_gate_deadline(two_threads, deadline)
}

pub fn run(two_threads: bool) -> i32 {
    match run_stages(two_threads) {
        Ok(()) => 0,
        Err(error) => {
            eprintln!(
                "{{\"stage\":\"error\",\"operation\":{:?},\"message\":{:?}}}",
                error.operation, error.message
            );
            1
        }
    }
}
