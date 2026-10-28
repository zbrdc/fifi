#![cfg(all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64"))]
//! Page imports at the public `MlxKernels` interface (decision 0041, S2 plan Z1a).
//!
//! The Z0 spike showed that the pinned MLX uses the pages in place. That identity is not visible
//! through the public interface, so these tests check what callers rely on: correct bytes, no
//! synchronization, page reuse only after the consuming graph completes, no donation into lent
//! pages, owner checks, and rejection before native work.

use std::{
    sync::{Mutex, MutexGuard, PoisonError},
    thread,
    time::{Duration, Instant},
};

use inference_engine::{
    EngError,
    kernels::{
        HostPages, MlxKernels, PackedPages, PackedWeight, PageRange, QuantizationMode,
        QuantizationSpec, ResidentPackedWeight, UnaryOp,
    },
    tensor::{Dtype, ExecutionTensor, Tensor},
};

static_assertions::assert_not_impl_any!(HostPages: Send, Sync);

static GPU_TEST: Mutex<()> = Mutex::new(());

/// Serializes the GPU tests. A panic in one test must not fail the others with a poison error.
fn serial() -> MutexGuard<'static, ()> {
    GPU_TEST.lock().unwrap_or_else(PoisonError::into_inner)
}

const AFFINE4: QuantizationSpec = QuantizationSpec {
    mode: QuantizationMode::Affine,
    bits: 4,
    group_size: 64,
};

/// BF16 bit patterns of 2^-6 and 2^-7 times 1, 1.25, 1.5, and 1.75.
const SCALES: [u16; 8] = [
    0x3C80, 0x3CA0, 0x3CC0, 0x3CE0, 0x3C00, 0x3C20, 0x3C40, 0x3C60,
];
/// BF16 bit patterns of -0.25, -0.125, 0, 0.125, and 0.25.
const BIASES: [u16; 5] = [0xBE80, 0xBE00, 0x0000, 0x3E00, 0x3E80];

fn page() -> usize {
    let size = HostPages::page_size();
    assert!(size > 0, "the OS page size query failed");
    size
}

/// Bytes of `count` pages.
fn pages_of(count: usize) -> usize {
    count * page()
}

fn range(offset: usize, dtype: Dtype, shape: &[u32]) -> PageRange {
    PageRange {
        offset,
        dtype,
        shape: shape.to_vec(),
    }
}

fn byte_len(range: &PageRange) -> usize {
    let width = match range.dtype {
        Dtype::U8 => 1,
        Dtype::F16 | Dtype::Bf16 | Dtype::I16 => 2,
        Dtype::F32 | Dtype::U32 | Dtype::I32 => 4,
        Dtype::I64 => 8,
    };
    range
        .shape
        .iter()
        .map(|&dimension| usize::try_from(dimension).unwrap())
        .product::<usize>()
        * width
}

fn xorshift(mut state: u32) -> u32 {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    state
}

/// Deterministic bytes that differ per `seed`.
fn pattern(len: usize, seed: u32) -> Vec<u8> {
    let mut state = seed | 1;
    (0..len)
        .map(|_| {
            state = xorshift(state);
            state.to_le_bytes()[0]
        })
        .collect()
}

fn bf16_table(len: usize, table: &[u16], shift: usize) -> Vec<u8> {
    (0..len / 2)
        .flat_map(|index| table[(index + shift) % table.len()].to_le_bytes())
        .collect()
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn f32_values(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|word| f32::from_le_bytes(*word))
        .collect()
}

fn f32_tensor(shape: Vec<u32>, values: &[f32]) -> Tensor {
    Tensor::try_from_host(Dtype::F32, shape, f32_bytes(values)).unwrap()
}

fn u32_tensor(shape: Vec<u32>, values: &[u32]) -> Tensor {
    let bytes = values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect();
    Tensor::try_from_host(Dtype::U32, shape, bytes).unwrap()
}

/// Writes `bytes` at `offset`. The pages must not be in use.
fn write(pages: &mut HostPages, offset: usize, bytes: &[u8]) {
    let target = pages.try_write().expect("pages are free before the import");
    target[offset..offset + bytes.len()].copy_from_slice(bytes);
}

fn read(kernels: &mut MlxKernels, tensor: &ExecutionTensor) -> Vec<u8> {
    kernels
        .readback(tensor)
        .unwrap()
        .host_bytes()
        .unwrap()
        .to_vec()
}

/// A one-element resident whose `evaluate` gives one synchronization and one retire flush.
fn anchor(kernels: &mut MlxKernels) -> ExecutionTensor {
    kernels.import_tensor(&f32_tensor(vec![1], &[0.0])).unwrap()
}

/// Polls `try_write` until it succeeds or `timeout` passes. MLX may call the deleter on a
/// completion thread shortly after the flush.
fn writable_within(pages: &mut HostPages, timeout: Duration) -> bool {
    let deadline = Instant::now() + timeout;
    loop {
        if pages.try_write().is_some() {
            return true;
        }
        if Instant::now() >= deadline {
            return false;
        }
        thread::sleep(Duration::from_millis(1));
    }
}

/// `len` F32 values `(i % 17 - 8) / 8`.
fn ramp(len: usize) -> Vec<f32> {
    (0..len)
        .map(|index| (f32::from(u8::try_from(index % 17).unwrap()) - 8.0) / 8.0)
        .collect()
}

#[test]
fn import_pages_reads_back_each_range() {
    let _serial = serial();
    let mut pages = HostPages::try_new(pages_of(6)).unwrap();
    let ranges = [
        range(0, Dtype::U8, &[100]),
        range(pages_of(1), Dtype::U32, &[64]),
        range(pages_of(2), Dtype::Bf16, &[2, 50]),
        range(pages_of(3), Dtype::F32, &[3, 7]),
        range(pages_of(4), Dtype::I16, &[64]),
        range(pages_of(5), Dtype::I32, &[2, 32]),
    ];
    let contents: Vec<Vec<u8>> = ranges
        .iter()
        .zip(1..)
        .map(|(range, seed)| pattern(byte_len(range), seed))
        .collect();
    for (range, bytes) in ranges.iter().zip(&contents) {
        write(&mut pages, range.offset, bytes);
    }
    let mut kernels = MlxKernels::try_new().unwrap();

    let imported = kernels.import_pages(&pages, &ranges).unwrap();

    assert_eq!(imported.len(), ranges.len());
    for ((tensor, range), bytes) in imported.iter().zip(&ranges).zip(&contents) {
        assert_eq!(
            tensor.dtype(),
            range.dtype,
            "dtype at offset {}",
            range.offset
        );
        assert_eq!(
            tensor.shape(),
            range.shape,
            "shape at offset {}",
            range.offset
        );
        assert_eq!(
            &read(&mut kernels, tensor),
            bytes,
            "bytes at offset {}",
            range.offset
        );
    }
}

#[test]
fn pages_return_to_the_caller_only_after_the_flush() {
    let _serial = serial();
    let mut pages = HostPages::try_new(pages_of(1)).unwrap();
    write(&mut pages, 0, &pattern(64, 7));
    let mut kernels = MlxKernels::try_new().unwrap();
    let anchor = anchor(&mut kernels);
    let mut imported = kernels
        .import_pages(&pages, &[range(0, Dtype::U8, &[64])])
        .unwrap();
    assert!(pages.in_use(), "a live import uses the pages");

    assert!(
        pages.try_write().is_none(),
        "the pages stay lent while a resident over them lives"
    );
    drop(imported.pop());
    assert!(
        pages.try_write().is_none(),
        "the pin keeps the pages after the caller drops its resident"
    );
    pages.release();
    assert!(
        pages.try_write().is_none(),
        "a retired resident keeps the pages until the next synchronization"
    );
    kernels.evaluate(&[&anchor]).unwrap();
    assert!(
        writable_within(&mut pages, Duration::from_secs(1)),
        "the pages return within 1 s after the flush"
    );
    assert!(!pages.in_use());
}

#[test]
fn a_pending_graph_keeps_the_pages_until_it_completes() {
    let _serial = serial();
    let elements = page() / 4;
    let values = ramp(elements);
    let mut pages = HostPages::try_new(pages_of(1)).unwrap();
    write(&mut pages, 0, &f32_bytes(&values));
    let mut kernels = MlxKernels::try_new().unwrap();
    let anchor = anchor(&mut kernels);
    let shape = [u32::try_from(elements).unwrap()];
    let x = kernels
        .import_pages(&pages, &[range(0, Dtype::F32, &shape)])
        .unwrap()
        .pop()
        .unwrap();
    let negated = kernels.unary(UnaryOp::Neg, &x).unwrap();
    drop(x);
    pages.release();
    // This flush frees the import's handle and pin. Only the unevaluated graph holds the array.
    kernels.evaluate(&[&anchor]).unwrap();
    thread::sleep(Duration::from_millis(20));

    assert!(
        pages.try_write().is_none(),
        "the pages stay lent while an unevaluated graph reads them"
    );
    let expected: Vec<f32> = values.iter().map(|value| -value).collect();
    assert_eq!(f32_values(&read(&mut kernels, &negated)), expected);
    drop(negated);
    kernels.evaluate(&[&anchor]).unwrap();
    assert!(
        writable_within(&mut pages, Duration::from_secs(1)),
        "the pages return after the graph completes and its output is flushed"
    );
}

#[test]
fn lent_pages_are_never_donated() {
    let _serial = serial();
    let elements = page() / 4;
    let values = ramp(elements);
    let original = f32_bytes(&values);
    let mut pages = HostPages::try_new(pages_of(1)).unwrap();
    write(&mut pages, 0, &original);
    let mut kernels = MlxKernels::try_new().unwrap();
    let anchor = anchor(&mut kernels);
    let shape = [u32::try_from(elements).unwrap()];
    let x = kernels
        .import_pages(&pages, &[range(0, Dtype::F32, &shape)])
        .unwrap()
        .pop()
        .unwrap();
    let negated = kernels.unary(UnaryOp::Neg, &x).unwrap();
    drop(x);
    // This flush frees the caller's handle. Without the pin, only the graph would hold the array,
    // and MLX would donate the page buffer to the negation (Z0 case S9).
    kernels.evaluate(&[&anchor]).unwrap();

    let expected: Vec<f32> = values.iter().map(|value| -value).collect();
    assert_eq!(f32_values(&read(&mut kernels, &negated)), expected);
    drop(negated);
    pages.release();
    kernels.evaluate(&[&anchor]).unwrap();
    assert!(writable_within(&mut pages, Duration::from_secs(1)));
    let after = pages.try_write().unwrap();
    assert!(
        after[..original.len()] == original[..],
        "MLX wrote into the lent pages"
    );
}

#[test]
fn try_write_keeps_the_pins_of_a_pending_graph() {
    let _serial = serial();
    let elements = page() / 4;
    let values = ramp(elements);
    let original = f32_bytes(&values);
    let mut pages = HostPages::try_new(pages_of(1)).unwrap();
    write(&mut pages, 0, &original);
    let mut kernels = MlxKernels::try_new().unwrap();
    let anchor = anchor(&mut kernels);
    let shape = [u32::try_from(elements).unwrap()];
    let x = kernels
        .import_pages(&pages, &[range(0, Dtype::F32, &shape)])
        .unwrap()
        .pop()
        .unwrap();
    let negated = kernels.unary(UnaryOp::Neg, &x).unwrap();
    drop(x);

    // A caller that polls for free pages while the consumer is pending must not unpin them.
    for _ in 0..5 {
        assert!(pages.try_write().is_none(), "pinned pages are not writable");
        kernels.evaluate(&[&anchor]).unwrap();
        thread::sleep(Duration::from_millis(5));
    }
    let expected: Vec<f32> = values.iter().map(|value| -value).collect();
    assert_eq!(f32_values(&read(&mut kernels, &negated)), expected);
    drop(negated);
    pages.release();
    kernels.evaluate(&[&anchor]).unwrap();
    assert!(writable_within(&mut pages, Duration::from_secs(1)));
    let after = pages.try_write().unwrap();
    assert!(
        after[..original.len()] == original[..],
        "MLX wrote into pages that try_write had polled"
    );
}

/// A packed affine-4 layout at page multiples: one `[4, 128, 256]` stack, one `[128, 256]`
/// weight, and an F32 `[4]` plain range. Returns the parts and the layout.
fn packed_layout() -> (Vec<PackedPages>, PageRange, usize) {
    let mut cursor = 0;
    let mut place = |dtype: Dtype, shape: &[u32]| {
        let placed = range(cursor, dtype, shape);
        cursor += byte_len(&placed).div_ceil(page()) * page();
        placed
    };
    let stack = PackedPages {
        payload: place(Dtype::U32, &[4, 128, 32]),
        scales: place(Dtype::Bf16, &[4, 128, 4]),
        biases: Some(place(Dtype::Bf16, &[4, 128, 4])),
        logical_shape: vec![4, 128, 256],
        quantization: AFFINE4,
    };
    let matrix = PackedPages {
        payload: place(Dtype::U32, &[128, 32]),
        scales: place(Dtype::Bf16, &[128, 4]),
        biases: Some(place(Dtype::Bf16, &[128, 4])),
        logical_shape: vec![128, 256],
        quantization: AFFINE4,
    };
    let gates = place(Dtype::F32, &[4]);
    (vec![stack, matrix], gates, cursor)
}

fn fill_packed(pages: &mut HostPages, weights: &[PackedPages], gates: &PageRange) -> Vec<u8> {
    let bytes = pages.try_write().unwrap();
    for (index, weight) in weights.iter().enumerate() {
        let seed = u32::try_from(index).unwrap() + 11;
        let payload = &weight.payload;
        bytes[payload.offset..payload.offset + byte_len(payload)]
            .copy_from_slice(&pattern(byte_len(payload), seed));
        let scales = &weight.scales;
        bytes[scales.offset..scales.offset + byte_len(scales)].copy_from_slice(&bf16_table(
            byte_len(scales),
            &SCALES,
            index,
        ));
        let biases = weight.biases.as_ref().unwrap();
        bytes[biases.offset..biases.offset + byte_len(biases)].copy_from_slice(&bf16_table(
            byte_len(biases),
            &BIASES,
            index,
        ));
    }
    let gate_bytes = f32_bytes(&[0.5, 0.25, 0.125, 1.0]);
    bytes[gates.offset..gates.offset + 16].copy_from_slice(&gate_bytes);
    bytes.to_vec()
}

fn copied_weight(bytes: &[u8], weight: &PackedPages) -> PackedWeight {
    let tensor = |range: &PageRange| {
        let data = bytes[range.offset..range.offset + byte_len(range)].to_vec();
        Tensor::try_from_host(range.dtype, range.shape.clone(), data).unwrap()
    };
    PackedWeight::try_new(
        tensor(&weight.payload),
        tensor(&weight.scales),
        weight.biases.as_ref().map(tensor),
        weight.logical_shape.clone(),
        weight.quantization,
    )
    .unwrap()
}

/// `gather_qmm` over the stack and `qmm` over the matrix, read back.
fn run_packed(
    kernels: &mut MlxKernels,
    stack: &ResidentPackedWeight,
    matrix: &ResidentPackedWeight,
) -> (Vec<u8>, Vec<u8>) {
    let x = kernels
        .import_tensor(&f32_tensor(vec![1, 1, 256], &ramp(256)))
        .unwrap();
    let row = kernels
        .import_tensor(&f32_tensor(vec![1, 256], &ramp(256)))
        .unwrap();
    let idx = kernels
        .import_tensor(&u32_tensor(vec![4], &[3, 0, 2, 1]))
        .unwrap();
    let gathered = kernels.gather_qmm(&x, stack, &idx).unwrap();
    let product = kernels.qmm(&row, matrix).unwrap();
    (read(kernels, &gathered), read(kernels, &product))
}

#[test]
fn packed_pages_match_the_copying_import() {
    let _serial = serial();
    let (weights, gates, len) = packed_layout();
    let mut pages = HostPages::try_new(len).unwrap();
    let bytes = fill_packed(&mut pages, &weights, &gates);
    let mut kernels = MlxKernels::try_new().unwrap();

    let (lent, plain) = kernels
        .import_packed_pages(&pages, &weights, std::slice::from_ref(&gates))
        .unwrap();
    let copied_weights: Vec<PackedWeight> = weights
        .iter()
        .map(|weight| copied_weight(&bytes, weight))
        .collect();
    let refs: Vec<&PackedWeight> = copied_weights.iter().collect();
    let (copied, _) = kernels.import_packed_many_with(&refs, &[]).unwrap();

    assert_eq!(lent.len(), 2);
    assert_eq!(lent[0].logical_shape(), [4, 128, 256]);
    assert_eq!(lent[0].logical_elements(), 4 * 128 * 256);
    let from_pages = run_packed(&mut kernels, &lent[0], &lent[1]);
    let from_copies = run_packed(&mut kernels, &copied[0], &copied[1]);
    assert!(
        from_pages.0 == from_copies.0,
        "gather_qmm over pages differs from the copied stack"
    );
    assert!(
        from_pages.1 == from_copies.1,
        "qmm over pages differs from the copied weight"
    );
    assert_eq!(plain.len(), 1);
    assert_eq!(
        f32_values(&read(&mut kernels, &plain[0])),
        [0.5, 0.25, 0.125, 1.0]
    );
}

fn is_bad_request<T>(result: Result<T, EngError>) -> bool {
    matches!(result, Err(EngError::BadRequest(_)))
}

#[test]
fn invalid_ranges_are_rejected_before_native_work() {
    let _serial = serial();
    assert!(is_bad_request(HostPages::try_new(0)), "zero length");
    let mut pages = HostPages::try_new(pages_of(2)).unwrap();
    let contents = pattern(256, 3);
    write(&mut pages, 0, &contents);
    let mut kernels = MlxKernels::try_new().unwrap();
    let anchor = anchor(&mut kernels);
    let whole = u32::try_from(pages_of(2)).unwrap();

    let cases: [(&str, Vec<PageRange>); 6] = [
        (
            "an offset that is not a page multiple",
            vec![range(64, Dtype::U8, &[64])],
        ),
        (
            "a range past the end",
            vec![range(0, Dtype::U8, &[whole + 1])],
        ),
        (
            "an overflowing byte length",
            vec![range(0, Dtype::F32, &[u32::MAX, u32::MAX, u32::MAX])],
        ),
        (
            "a dimension above the native int",
            vec![range(0, Dtype::U8, &[1 << 31])],
        ),
        ("a zero-size range", vec![range(0, Dtype::U8, &[0])]),
        ("an empty slice", Vec::new()),
    ];
    for (name, ranges) in &cases {
        assert!(
            is_bad_request(kernels.import_pages(&pages, ranges)),
            "{name} must be BadRequest"
        );
    }
    let mut wrong_payload = PackedPages {
        payload: range(0, Dtype::U32, &[2, 8]),
        scales: range(page(), Dtype::Bf16, &[2, 1]),
        biases: Some(range(page(), Dtype::Bf16, &[2, 1])),
        logical_shape: vec![2, 64],
        quantization: AFFINE4,
    };
    assert!(
        !is_bad_request(kernels.import_packed_pages(&pages, &[wrong_payload.clone()], &[])),
        "the valid packed layout imports"
    );
    wrong_payload.payload.shape = vec![2, 16];
    assert!(
        is_bad_request(kernels.import_packed_pages(&pages, &[wrong_payload.clone()], &[])),
        "a payload shape that does not match the logical shape must be BadRequest"
    );
    wrong_payload.payload.shape = vec![2, 8];
    wrong_payload.biases = None;
    assert!(
        is_bad_request(kernels.import_packed_pages(&pages, &[wrong_payload], &[])),
        "affine pages without biases must be BadRequest"
    );

    let valid = kernels
        .import_pages(&pages, &[range(0, Dtype::U8, &[256])])
        .unwrap();
    assert_eq!(
        read(&mut kernels, &valid[0]),
        contents,
        "the adapter stays usable"
    );
    drop(valid);
    pages.release();
    kernels.evaluate(&[&anchor]).unwrap();
    assert!(writable_within(&mut pages, Duration::from_secs(1)));
}

#[test]
fn two_adapters_share_pages_and_keep_their_owners() {
    let _serial = serial();
    let mut pages = HostPages::try_new(pages_of(1)).unwrap();
    let values = ramp(64);
    write(&mut pages, 0, &f32_bytes(&values));
    let mut first = MlxKernels::try_new().unwrap();
    let mut second = MlxKernels::try_new().unwrap();
    let first_anchor = anchor(&mut first);
    let second_anchor = anchor(&mut second);
    let lent = [range(0, Dtype::F32, &[64])];
    let mine = first.import_pages(&pages, &lent).unwrap().pop().unwrap();
    let theirs = second.import_pages(&pages, &lent).unwrap().pop().unwrap();

    assert!(
        is_bad_request(second.unary(UnaryOp::Neg, &mine)),
        "a page resident belongs to the adapter that imported it"
    );
    assert_eq!(f32_values(&read(&mut first, &mine)), values);
    assert_eq!(f32_values(&read(&mut second, &theirs)), values);
    drop(mine);
    pages.release();
    first.evaluate(&[&first_anchor]).unwrap();
    thread::sleep(Duration::from_millis(20));
    assert!(
        pages.try_write().is_none(),
        "the second adapter's import still holds the pages"
    );
    drop(theirs);
    second.evaluate(&[&second_anchor]).unwrap();
    assert!(writable_within(&mut pages, Duration::from_secs(1)));
}

#[test]
#[cfg(feature = "native-fault-qualification")]
fn page_imports_do_not_synchronize() {
    let _serial = serial();
    let (weights, gates, len) = packed_layout();
    let mut pages = HostPages::try_new(len).unwrap();
    fill_packed(&mut pages, &weights, &gates);
    let mut kernels = MlxKernels::try_new().unwrap();

    let before = kernels.qualification_sync_count();
    let plain = kernels
        .import_pages(&pages, std::slice::from_ref(&gates))
        .unwrap();
    let (packed, _) = kernels.import_packed_pages(&pages, &weights, &[]).unwrap();
    assert_eq!(
        kernels.qualification_sync_count(),
        before,
        "page imports must not synchronize"
    );
    assert_eq!(packed.len(), 2);
    assert_eq!(
        f32_values(&read(&mut kernels, &plain[0])),
        [0.5, 0.25, 0.125, 1.0]
    );
    assert_eq!(kernels.qualification_sync_count(), before + 1);
}

#[test]
#[cfg(feature = "native-fault-qualification")]
fn aligned_page_imports_are_not_counted_as_copies() {
    let _serial = serial();
    let (weights, gates, len) = packed_layout();
    let mut pages = HostPages::try_new(len).unwrap();
    fill_packed(&mut pages, &weights, &gates);
    let mut kernels = MlxKernels::try_new().unwrap();

    let before = kernels.qualification_page_copies();
    let (packed, plain) = kernels
        .import_packed_pages(&pages, &weights, std::slice::from_ref(&gates))
        .unwrap();
    assert_eq!(packed.len() * 3 + plain.len(), 7);
    assert_eq!(
        kernels.qualification_page_copies(),
        before,
        "MLX copied an aligned page import"
    );
}

#[cfg(feature = "native-fault-qualification")]
mod faults {
    //! Injected failures of the page import shim (decision 0032). Each case runs in a child
    //! process, because a native failure is terminal for the whole process.

    use std::{
        env,
        io::Read,
        process::{Command, Stdio},
        thread,
        time::{Duration, Instant},
    };

    use super::{EngError, HostPages, MlxKernels, pages_of, range, serial};
    use inference_engine::tensor::Dtype;

    /// Collects a child pipe on its own thread, so a full pipe cannot block the child.
    fn drain<R: Read + Send + 'static>(mut pipe: R) -> thread::JoinHandle<String> {
        thread::spawn(move || {
            let mut bytes = Vec::new();
            let _ = pipe.read_to_end(&mut bytes);
            String::from_utf8_lossy(&bytes).into_owned()
        })
    }

    /// Runs one test of this binary in a child process and fails when the child fails.
    fn spawn_child(test: &str, envs: &[(&str, &str)]) {
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
            if started.elapsed() >= Duration::from_secs(180) {
                child.kill().unwrap();
                child.wait().unwrap();
                break None;
            }
            thread::sleep(Duration::from_millis(20));
        };
        let captured = format!("{} {}", stdout.join().unwrap(), stderr.join().unwrap());
        match status {
            // A renamed test makes the child run nothing and exit 0, so require one passed test.
            Some(status) if status.success() && captured.contains("1 passed") => {
                print!("{captured}");
            }
            Some(status) if status.success() => panic!("child ran no test: {captured}"),
            Some(_) => panic!("child failed: {captured}"),
            None => panic!("child timed out: {captured}"),
        }
    }

    /// The child side: the first selected operation of the process is the page import.
    fn injected_failure_is_terminal() {
        let mut pages = HostPages::try_new(pages_of(1)).unwrap();
        let mut kernels = MlxKernels::try_new().unwrap();
        let lent = [range(0, Dtype::U8, &[64])];

        let failed = kernels.import_pages(&pages, &lent);
        assert!(
            matches!(
                failed,
                Err(EngError::Backend {
                    operation: "page import",
                    ..
                })
            ),
            "the injected failure returns Backend page import"
        );
        assert!(
            matches!(
                kernels.import_pages(&pages, &lent),
                Err(EngError::Backend {
                    operation: "execute",
                    ..
                })
            ),
            "the adapter is terminal after a page import failure"
        );
        assert!(
            pages.try_write().is_none(),
            "a failed import keeps its share, so the pages never return"
        );
    }

    #[test]
    fn page_import_call_failure_is_terminal() {
        if env::var_os("INFERENCE_NATIVE_PAGES_CHILD").is_some() {
            injected_failure_is_terminal();
            return;
        }
        let _serial = serial();
        spawn_child(
            "faults::page_import_call_failure_is_terminal",
            &[
                ("INFERENCE_NATIVE_PAGES_CHILD", "1"),
                ("INFERENCE_MLX_FAULT_NEXT_OPERATION", "call"),
            ],
        );
    }

    #[test]
    fn page_import_null_failure_is_terminal() {
        if env::var_os("INFERENCE_NATIVE_PAGES_CHILD").is_some() {
            injected_failure_is_terminal();
            return;
        }
        let _serial = serial();
        spawn_child(
            "faults::page_import_null_failure_is_terminal",
            &[
                ("INFERENCE_NATIVE_PAGES_CHILD", "1"),
                ("INFERENCE_MLX_FAULT_NEXT_OPERATION", "null"),
            ],
        );
    }
}
