use std::{ffi::c_void, process::Command, time::Instant};

use memmap2::MmapOptions;
use mlx_sys::{mlx_array, mlx_dtype__MLX_UINT8, mlx_stream};

mod adapter;

fn checked(status: i32, operation: &str) {
    if status != 0 {
        eprintln!("{operation} failed with status {status}");
        std::process::exit(2);
    }
}

fn handle(array: mlx_array, operation: &str) -> mlx_array {
    if array.ctx.is_null() {
        eprintln!("{operation} returned an empty handle");
        std::process::exit(2);
    }
    array
}

fn validate_u8_array(array: mlx_array, bytes: usize, operation: &str) {
    let valid = unsafe {
        mlx_sys::mlx_array_dtype(array) == mlx_dtype__MLX_UINT8
            && mlx_sys::mlx_array_ndim(array) == 1
            && mlx_sys::mlx_array_size(array) == bytes
            && mlx_sys::mlx_array_nbytes(array) == bytes
    };
    let mut contiguous = false;
    checked(
        unsafe { mlx_sys::_mlx_array_is_row_contiguous(&mut contiguous, array) },
        "check array layout",
    );
    if !valid || !contiguous {
        eprintln!("{operation} has an unexpected dtype, shape, or layout");
        std::process::exit(2);
    }
}

fn stream() -> mlx_stream {
    let stream = unsafe { mlx_sys::mlx_default_gpu_stream_new() };
    if stream.ctx.is_null() {
        eprintln!("default GPU stream is unavailable");
        std::process::exit(2);
    }
    checked(
        unsafe { mlx_sys::mlx_set_default_stream(stream) },
        "set default stream",
    );
    stream
}

fn limit_native_memory() -> usize {
    let mut previous = 0;
    checked(
        unsafe { mlx_sys::mlx_set_memory_limit(&mut previous, 3 << 30) },
        "set native memory limit",
    );
    previous
}

fn mlx_bytes() -> (usize, usize, usize) {
    let mut active = 0;
    let mut cache = 0;
    let mut peak = 0;
    checked(
        unsafe { mlx_sys::mlx_get_active_memory(&mut active) },
        "active memory",
    );
    checked(
        unsafe { mlx_sys::mlx_get_cache_memory(&mut cache) },
        "cache memory",
    );
    checked(
        unsafe { mlx_sys::mlx_get_peak_memory(&mut peak) },
        "peak memory",
    );
    (active, cache, peak)
}

fn rss_kib() -> usize {
    let output = Command::new("ps")
        .args(["-o", "rss=", "-p", &std::process::id().to_string()])
        .output()
        .expect("ps failed");
    if !output.status.success() {
        eprintln!("ps did not report resident memory");
        std::process::exit(2);
    }
    String::from_utf8(output.stdout)
        .expect("ps output is not UTF-8")
        .trim()
        .parse()
        .expect("ps resident memory is not an integer")
}

fn sample(stage: &str, elapsed_us: u128) {
    let (active, cache, peak) = mlx_bytes();
    println!(
        "{{\"stage\":\"{stage}\",\"elapsed_us\":{elapsed_us},\"rss_kib\":{},\"mlx_active_bytes\":{active},\"mlx_cache_bytes\":{cache},\"mlx_peak_bytes\":{peak}}}",
        rss_kib()
    );
}

fn parse_bytes() -> usize {
    let arg = std::env::args().nth(1).unwrap_or_else(|| {
        eprintln!("usage: native-import-qualification BYTES");
        std::process::exit(2);
    });
    let bytes = arg.parse::<usize>().unwrap_or_else(|_| {
        eprintln!("BYTES must be an integer");
        std::process::exit(2);
    });
    if bytes == 0 || bytes > 1 << 30 || bytes > i32::MAX as usize {
        eprintln!("BYTES must be between 1 and 1073741824");
        std::process::exit(2);
    }
    bytes
}

fn import_source(source: &[u8]) -> (mlx_array, u128) {
    let bytes = source.len();
    let shape = [i32::try_from(bytes).expect("shape fits i32")];
    let start = Instant::now();
    let input = handle(
        unsafe {
            mlx_sys::mlx_array_new_data(
                source.as_ptr().cast::<c_void>(),
                shape.as_ptr(),
                1,
                mlx_dtype__MLX_UINT8,
            )
        },
        "copying import",
    );
    (input, start.elapsed().as_micros())
}

fn check_copy(source: &mut [u8], input: mlx_array) {
    checked(unsafe { mlx_sys::mlx_array_eval(input) }, "evaluate input");
    validate_u8_array(input, source.len(), "imported array");
    let copied = unsafe { mlx_sys::mlx_array_data_uint8(input) };
    if copied.is_null() {
        eprintln!("imported data is unavailable");
        std::process::exit(2);
    }
    source[0] = 9;
    source[source.len() - 1] = 9;
    let first = unsafe { *copied };
    let last = unsafe { *copied.add(source.len() - 1) };
    println!(
        "{{\"stage\":\"copy_check\",\"different_pointer\":{},\"first\":{first},\"last\":{last}}}",
        copied != source.as_ptr()
    );
    sample("input_evaluated", 0);
    if first != 7 || last != 7 || copied == source.as_ptr() {
        eprintln!("the imported array did not retain an independent copy");
        std::process::exit(2);
    }
}

fn read_result(output: mlx_array, bytes: usize) {
    validate_u8_array(output, bytes, "GPU output");
    let result = unsafe { mlx_sys::mlx_array_data_uint8(output) };
    if result.is_null() {
        eprintln!("result data is unavailable");
        std::process::exit(2);
    }
    let first = unsafe { *result };
    let last = unsafe { *result.add(bytes - 1) };
    println!("{{\"stage\":\"result\",\"first\":{first},\"last\":{last}}}");
    if first != 8 || last != 8 {
        eprintln!("GPU output differs from the declared result");
        std::process::exit(2);
    }
    let start = Instant::now();
    let host = unsafe { std::slice::from_raw_parts(result, bytes) }.to_vec();
    sample("host_copy", start.elapsed().as_micros());
    let validation = Instant::now();
    if host.iter().any(|&value| value != 8) {
        eprintln!("host copy differs from the completed output");
        std::process::exit(2);
    }
    println!(
        "{{\"stage\":\"content_check\",\"elapsed_us\":{}}}",
        validation.elapsed().as_micros()
    );
}

fn execute_gpu(input: mlx_array, stream: mlx_stream, bytes: usize) {
    let one = [1u8];
    let scalar_shape = [1i32];
    let scalar = handle(
        unsafe {
            mlx_sys::mlx_array_new_data(
                one.as_ptr().cast::<c_void>(),
                scalar_shape.as_ptr(),
                1,
                mlx_dtype__MLX_UINT8,
            )
        },
        "scalar import",
    );
    let mut output = mlx_array {
        ctx: std::ptr::null_mut(),
    };
    let start = Instant::now();
    checked(
        unsafe { mlx_sys::mlx_add(&mut output, input, scalar, stream) },
        "GPU add",
    );
    handle(output, "GPU add");
    checked(
        unsafe { mlx_sys::mlx_array_eval(output) },
        "evaluate output",
    );
    checked(unsafe { mlx_sys::mlx_synchronize(stream) }, "synchronize");
    sample("gpu_complete", start.elapsed().as_micros());
    read_result(output, bytes);
    checked(unsafe { mlx_sys::mlx_array_free(output) }, "free output");
    checked(unsafe { mlx_sys::mlx_array_free(scalar) }, "free scalar");
    checked(unsafe { mlx_sys::mlx_array_free(input) }, "free input");
    sample("arrays_freed", 0);
}

fn run(bytes: usize) {
    let previous_limit = limit_native_memory();
    let mut source = MmapOptions::new()
        .len(bytes)
        .map_anon()
        .expect("anonymous map");
    source.fill(7);
    let stream = stream();
    println!(
        "{{\"stage\":\"identity\",\"bytes\":{bytes},\"source_kind\":\"anonymous_map\",\"dtype\":\"uint8\",\"native_limit_bytes\":{},\"previous_native_limit_bytes\":{previous_limit}}}",
        3usize << 30
    );
    sample("source_ready", 0);
    checked(unsafe { mlx_sys::mlx_reset_peak_memory() }, "reset peak");
    let (input, import_us) = import_source(&source);
    sample("imported", import_us);
    check_copy(&mut source, input);
    execute_gpu(input, stream, bytes);
    checked(
        unsafe { mlx_sys::mlx_stream_free(stream) },
        "free stream wrapper",
    );
    drop(source);
    checked(unsafe { mlx_sys::mlx_clear_cache() }, "clear native cache");
    sample("released", 0);
    std::thread::sleep(std::time::Duration::from_secs(2));
    sample("settled", 0);
}

fn main() {
    if std::env::args().nth(1).as_deref() == Some("adapter") {
        adapter::run();
    } else {
        run(parse_bytes());
    }
}
