use std::io::{Cursor, Read, Seek, SeekFrom};

use inference_engine::{
    EngError,
    tensor::Dtype,
    weights::{
        AcquireRequest, CancelHandle, Completion, Priority, Selection, StreamedCheckpoint,
        StreamedWeights,
    },
};

const PAYLOAD: [u8; 6] = [10, 11, 12, 13, 14, 15];
const SECOND: [u8; 4] = [20, 21, 22, 23];

/// One safetensors shard: `U8` tensor `w` of shape `[2, 3]`, then `F16` tensor `v` of shape `[2]`.
fn shard() -> Vec<u8> {
    let header = br#"{"w":{"dtype":"U8","shape":[2,3],"data_offsets":[0,6]},"v":{"dtype":"F16","shape":[2],"data_offsets":[6,10]}}"#;
    let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
    bytes.extend_from_slice(header);
    bytes.extend_from_slice(&PAYLOAD);
    bytes.extend_from_slice(&SECOND);
    bytes
}

fn open(reader: impl Read + Seek + Send + 'static, byte_len: usize) -> StreamedWeights {
    StreamedWeights::open(Box::new(reader), byte_len as u64, 4096).unwrap()
}

fn demand(key: &str) -> AcquireRequest {
    AcquireRequest::new(
        key.into(),
        Selection::Whole,
        Priority::Demand,
        CancelHandle::new(),
    )
}

#[test]
fn present_key_completes_ready_with_exact_bytes_dtype_and_shape() {
    let bytes = shard();
    let store = open(Cursor::new(bytes.clone()), bytes.len());

    let ticket = store.acquire(demand("w")).unwrap();
    let completion = store.wait(ticket).unwrap();

    let Completion::Ready(tensor) = completion else {
        panic!("expected Ready, received {completion:?}");
    };
    assert_eq!(tensor.dtype(), Dtype::U8);
    assert_eq!(tensor.shape(), &[2, 3]);
    assert_eq!(tensor.host_bytes(), Some(PAYLOAD.as_slice()));
}

#[test]
fn tensor_with_nonzero_start_completes_ready_with_its_own_bytes_dtype_and_shape() {
    let bytes = shard();
    let store = open(Cursor::new(bytes.clone()), bytes.len());

    let ticket = store.acquire(demand("v")).unwrap();
    let completion = store.wait(ticket).unwrap();

    let Completion::Ready(tensor) = completion else {
        panic!("expected Ready, received {completion:?}");
    };
    assert_eq!(tensor.dtype(), Dtype::F16);
    assert_eq!(tensor.shape(), &[2]);
    assert_eq!(tensor.host_bytes(), Some(SECOND.as_slice()));
}

#[test]
fn absent_key_completes_missing() {
    let bytes = shard();
    let store = open(Cursor::new(bytes.clone()), bytes.len());

    let ticket = store.acquire(demand("absent")).unwrap();

    assert!(matches!(store.wait(ticket), Ok(Completion::Missing)));
}

/// A read dependency that serves the header and fails every read in the data region.
struct DataFailingReader {
    inner: Cursor<Vec<u8>>,
    data_start: u64,
}

impl Read for DataFailingReader {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        if self.inner.position() >= self.data_start {
            return Err(std::io::Error::other("injected read failure"));
        }
        self.inner.read(buf)
    }
}

impl Seek for DataFailingReader {
    fn seek(&mut self, position: SeekFrom) -> std::io::Result<u64> {
        self.inner.seek(position)
    }
}

#[test]
fn read_failure_returns_backend_weight_read_and_never_missing() {
    let bytes = shard();
    let data_start = 8 + u64::from_le_bytes(bytes[..8].try_into().unwrap());
    let len = bytes.len();
    let store = open(
        DataFailingReader {
            inner: Cursor::new(bytes),
            data_start,
        },
        len,
    );

    let outcome = store
        .acquire(demand("w"))
        .and_then(|ticket| store.wait(ticket));

    match outcome {
        Err(EngError::Backend { operation, message }) => {
            assert_eq!(operation, "weight_read");
            assert!(message.contains("injected read failure"), "{message}");
        }
        other => panic!("expected Backend weight_read, received {other:?}"),
    }
}

fn slice(key: &str, index: u32) -> AcquireRequest {
    AcquireRequest::new(
        key.into(),
        Selection::LeadingSlice { index },
        Priority::Demand,
        CancelHandle::new(),
    )
}

#[test]
fn leading_slice_returns_one_row_with_the_shape_after_the_leading_dimension() {
    let bytes = shard();
    let store = open(Cursor::new(bytes.clone()), bytes.len());

    let ticket = store.acquire(slice("w", 1)).unwrap();
    let completion = store.wait(ticket).unwrap();

    let Completion::Ready(tensor) = completion else {
        panic!("expected Ready, received {completion:?}");
    };
    assert_eq!(tensor.dtype(), Dtype::U8);
    assert_eq!(tensor.shape(), &[3]);
    assert_eq!(tensor.host_bytes(), Some([13_u8, 14, 15].as_slice()));

    let ticket = store.acquire(slice("w", 0)).unwrap();
    let completion = store.wait(ticket).unwrap();

    let Completion::Ready(tensor) = completion else {
        panic!("expected Ready, received {completion:?}");
    };
    assert_eq!(tensor.host_bytes(), Some([10_u8, 11, 12].as_slice()));
}

#[test]
fn leading_slice_of_a_one_dimensional_tensor_is_a_scalar() {
    let bytes = shard();
    let store = open(Cursor::new(bytes.clone()), bytes.len());

    let ticket = store.acquire(slice("v", 1)).unwrap();
    let completion = store.wait(ticket).unwrap();

    let Completion::Ready(tensor) = completion else {
        panic!("expected Ready, received {completion:?}");
    };
    assert_eq!(tensor.dtype(), Dtype::F16);
    assert!(tensor.shape().is_empty(), "{:?}", tensor.shape());
    assert_eq!(tensor.host_bytes(), Some([22_u8, 23].as_slice()));
}

#[test]
fn leading_slice_outside_the_stored_shape_is_a_bad_request() {
    let bytes = shard();
    let store = open(Cursor::new(bytes.clone()), bytes.len());

    let outcome = store
        .acquire(slice("w", 2))
        .and_then(|ticket| store.wait(ticket));

    assert!(
        matches!(outcome, Err(EngError::BadRequest(_))),
        "expected BadRequest, received {outcome:?}"
    );
}

#[test]
fn leading_slice_far_outside_the_stored_shape_is_a_bad_request() {
    let bytes = shard();
    let store = open(Cursor::new(bytes.clone()), bytes.len());

    let outcome = store
        .acquire(slice("w", 100))
        .and_then(|ticket| store.wait(ticket));

    assert!(
        matches!(outcome, Err(EngError::BadRequest(_))),
        "expected BadRequest, received {outcome:?}"
    );
}

#[test]
fn leading_slice_of_an_absent_key_is_missing() {
    let bytes = shard();
    let store = open(Cursor::new(bytes.clone()), bytes.len());

    let ticket = store.acquire(slice("absent", 0)).unwrap();

    assert!(matches!(store.wait(ticket), Ok(Completion::Missing)));
}

#[test]
fn leading_slice_read_failure_is_backend_weight_read() {
    let bytes = shard();
    let data_start = 8 + u64::from_le_bytes(bytes[..8].try_into().unwrap());
    let len = bytes.len();
    let store = open(
        DataFailingReader {
            inner: Cursor::new(bytes),
            data_start,
        },
        len,
    );

    let outcome = store
        .acquire(slice("w", 1))
        .and_then(|ticket| store.wait(ticket));

    match outcome {
        Err(EngError::Backend { operation, message }) => {
            assert_eq!(operation, "weight_read");
            assert!(message.contains("injected read failure"), "{message}");
        }
        other => panic!("expected Backend weight_read, received {other:?}"),
    }
}

#[test]
fn leading_slice_of_a_scalar_is_a_bad_request() {
    let bytes = shard_with(
        r#"{"s":{"dtype":"U8","shape":[],"data_offsets":[0,1]}}"#,
        &[5],
    );
    let store = open(Cursor::new(bytes.clone()), bytes.len());

    let outcome = store
        .acquire(slice("s", 0))
        .and_then(|ticket| store.wait(ticket));

    assert!(
        matches!(outcome, Err(EngError::BadRequest(_))),
        "expected BadRequest, received {outcome:?}"
    );
}

/// One safetensors shard with the given header JSON and payload bytes.
fn shard_with(header: &str, payload: &[u8]) -> Vec<u8> {
    let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
    bytes.extend_from_slice(header.as_bytes());
    bytes.extend_from_slice(payload);
    bytes
}

fn checkpoint(index: &str, shards: Vec<(&str, Vec<u8>)>) -> Result<StreamedCheckpoint, EngError> {
    checkpoint_with_budget(index, shards, 4096)
}

fn checkpoint_with_budget(
    index: &str,
    shards: Vec<(&str, Vec<u8>)>,
    max_header_bytes: u64,
) -> Result<StreamedCheckpoint, EngError> {
    let shards = shards
        .into_iter()
        .map(|(name, bytes)| {
            let len = bytes.len() as u64;
            let reader: Box<dyn inference_engine::weights::ShardIo + Send> =
                Box::new(Cursor::new(bytes));
            (name.to_string(), reader, len)
        })
        .collect();
    StreamedCheckpoint::open(index.as_bytes(), shards, max_header_bytes)
}

/// Shard `s1.safetensors` holds `a`, and shard `s2.safetensors` holds `b`.
fn two_shards() -> Vec<(&'static str, Vec<u8>)> {
    vec![
        (
            "s1.safetensors",
            shard_with(
                r#"{"a":{"dtype":"U8","shape":[2,2],"data_offsets":[0,4]}}"#,
                &[1, 2, 3, 4],
            ),
        ),
        (
            "s2.safetensors",
            shard_with(
                r#"{"b":{"dtype":"U8","shape":[3],"data_offsets":[0,3]}}"#,
                &[7, 8, 9],
            ),
        ),
    ]
}

const TWO_SHARD_INDEX: &str = r#"{"weight_map":{"a":"s1.safetensors","b":"s2.safetensors"}}"#;

#[test]
fn checkpoint_routes_each_key_to_its_shard() {
    let checkpoint = checkpoint(TWO_SHARD_INDEX, two_shards()).unwrap();

    let ticket = checkpoint.acquire(demand("a")).unwrap();
    let completion = checkpoint.wait(ticket).unwrap();
    let Completion::Ready(tensor) = completion else {
        panic!("expected Ready, received {completion:?}");
    };
    assert_eq!(tensor.shape(), &[2, 2]);
    assert_eq!(tensor.host_bytes(), Some([1_u8, 2, 3, 4].as_slice()));

    let ticket = checkpoint.acquire(slice("b", 2)).unwrap();
    let completion = checkpoint.wait(ticket).unwrap();
    let Completion::Ready(tensor) = completion else {
        panic!("expected Ready, received {completion:?}");
    };
    assert!(tensor.shape().is_empty(), "{:?}", tensor.shape());
    assert_eq!(tensor.host_bytes(), Some([9_u8].as_slice()));

    assert!(checkpoint.contains("a"));
    assert!(!checkpoint.contains("c"));
}

#[test]
fn checkpoint_reports_an_absent_key_as_missing() {
    let checkpoint = checkpoint(TWO_SHARD_INDEX, two_shards()).unwrap();

    let ticket = checkpoint.acquire(demand("c")).unwrap();

    assert!(matches!(checkpoint.wait(ticket), Ok(Completion::Missing)));
}

#[test]
fn stored_reports_dtype_and_shape() {
    let checkpoint = checkpoint(TWO_SHARD_INDEX, two_shards()).unwrap();

    assert_eq!(checkpoint.stored("a"), Some((Dtype::U8, vec![2, 2])));
    assert_eq!(checkpoint.stored("b"), Some((Dtype::U8, vec![3])));
    assert_eq!(checkpoint.stored("c"), None);
}

#[test]
fn header_over_budget_names_the_shard() {
    let outcome = checkpoint_with_budget(TWO_SHARD_INDEX, two_shards(), 8).map(|_| ());

    match outcome {
        Err(EngError::MemoryPlan(message)) => {
            assert!(message.contains("s1.safetensors"), "{message}");
        }
        other => panic!("expected MemoryPlan, received {other:?}"),
    }
}

#[test]
fn index_key_in_wrong_shard_is_a_checkpoint_error() {
    let outcome = checkpoint(
        r#"{"weight_map":{"a":"s2.safetensors","b":"s1.safetensors"}}"#,
        two_shards(),
    )
    .map(|_| ());

    match outcome {
        Err(EngError::Checkpoint(message)) => {
            assert!(message.contains("is missing from shard"), "{message}");
            assert!(message.contains("s2.safetensors"), "{message}");
        }
        other => panic!("expected Checkpoint, received {other:?}"),
    }
}

#[test]
fn index_key_missing_from_its_shard_is_a_checkpoint_error() {
    let s1 = shard_with(
        r#"{"a":{"dtype":"U8","shape":[2,2],"data_offsets":[0,4]}}"#,
        &[1, 2, 3, 4],
    );
    let s2 = shard_with(
        r#"{"c":{"dtype":"U8","shape":[3],"data_offsets":[0,3]}}"#,
        &[7, 8, 9],
    );
    let index =
        r#"{"weight_map":{"a":"s1.safetensors","b":"s1.safetensors","c":"s2.safetensors"}}"#;

    let outcome =
        checkpoint(index, vec![("s1.safetensors", s1), ("s2.safetensors", s2)]).map(|_| ());

    match outcome {
        Err(EngError::Checkpoint(message)) => {
            assert!(message.contains("b"), "{message}");
            assert!(message.contains("is missing from shard"), "{message}");
        }
        other => panic!("expected Checkpoint, received {other:?}"),
    }
}

#[test]
fn unindexed_tensor_is_a_checkpoint_error() {
    let s1 = shard_with(
        r#"{"a":{"dtype":"U8","shape":[2,2],"data_offsets":[0,4]},"x":{"dtype":"U8","shape":[1],"data_offsets":[4,5]}}"#,
        &[1, 2, 3, 4, 5],
    );

    let outcome = checkpoint(
        r#"{"weight_map":{"a":"s1.safetensors"}}"#,
        vec![("s1.safetensors", s1)],
    )
    .map(|_| ());

    match outcome {
        Err(EngError::Checkpoint(message)) => {
            assert!(message.contains("x"), "{message}");
            assert!(message.contains("unindexed"), "{message}");
        }
        other => panic!("expected Checkpoint, received {other:?}"),
    }
}

#[test]
fn shard_set_mismatch_is_a_checkpoint_error() {
    let only_s1 = two_shards().into_iter().take(1).collect();

    let outcome = checkpoint(TWO_SHARD_INDEX, only_s1).map(|_| ());

    assert!(
        matches!(outcome, Err(EngError::Checkpoint(_))),
        "expected Checkpoint, received {outcome:?}"
    );
}

#[test]
fn repeated_shard_is_a_checkpoint_error() {
    let s1 = two_shards().into_iter().next().unwrap().1;
    let twice = vec![("s1.safetensors", s1.clone()), ("s1.safetensors", s1)];

    let outcome = checkpoint(r#"{"weight_map":{"a":"s1.safetensors"}}"#, twice).map(|_| ());

    match outcome {
        Err(EngError::Checkpoint(message)) => {
            assert!(message.contains("supplied twice"), "{message}");
        }
        other => panic!("expected Checkpoint, received {other:?}"),
    }
}
