//! Behavior tests for the GGUF header and tensor table reader.
//! Fixtures are generated here from the GGUF v3 layout. No tensor payload is parsed.

use inference_engine::{
    EngError,
    weights::{
        GgufHeader, GgufType, MetaValue, SplitInfo, check_split_set, sha256_hex, sha256_stream,
        verify_sha256,
    },
};
use std::io::{self, Read};

const LIMIT: u64 = 1 << 20;

// (id, name, block elements, block bytes) from the pinned format reference.
const TYPES: &[(u32, &str, u64, u64)] = &[
    (0, "F32", 1, 4),
    (1, "F16", 1, 2),
    (2, "Q4_0", 32, 18),
    (3, "Q4_1", 32, 20),
    (6, "Q5_0", 32, 22),
    (7, "Q5_1", 32, 24),
    (8, "Q8_0", 32, 34),
    (10, "Q2_K", 256, 84),
    (11, "Q3_K", 256, 110),
    (12, "Q4_K", 256, 144),
    (13, "Q5_K", 256, 176),
    (14, "Q6_K", 256, 210),
    (16, "IQ2_XXS", 256, 66),
    (17, "IQ2_XS", 256, 74),
    (18, "IQ3_XXS", 256, 98),
    (19, "IQ1_S", 256, 50),
    (20, "IQ4_NL", 32, 18),
    (21, "IQ3_S", 256, 110),
    (22, "IQ2_S", 256, 82),
    (23, "IQ4_XS", 256, 136),
    (24, "I8", 1, 1),
    (25, "I16", 1, 2),
    (26, "I32", 1, 4),
    (27, "I64", 1, 8),
    (28, "F64", 1, 8),
    (29, "IQ1_M", 256, 56),
    (30, "BF16", 1, 2),
    (42, "Q2_0", 64, 18),
];

fn s(text: &str) -> Vec<u8> {
    s_raw(text.as_bytes())
}
fn s_raw(bytes: &[u8]) -> Vec<u8> {
    let mut out = (bytes.len() as u64).to_le_bytes().to_vec();
    out.extend_from_slice(bytes);
    out
}
fn kv(key: &str, ty: u32, payload: &[u8]) -> Vec<u8> {
    let mut out = s(key);
    out.extend_from_slice(&ty.to_le_bytes());
    out.extend_from_slice(payload);
    out
}
fn kv_u32(key: &str, v: u32) -> Vec<u8> {
    kv(key, 4, &v.to_le_bytes())
}
fn kv_u16(key: &str, v: u16) -> Vec<u8> {
    kv(key, 2, &v.to_le_bytes())
}
fn kv_i32(key: &str, v: i32) -> Vec<u8> {
    kv(key, 5, &v.to_le_bytes())
}
fn kv_str(key: &str, v: &str) -> Vec<u8> {
    kv(key, 8, &s(v))
}
fn ti(name: &str, dims: &[u64], ty: u32, off: u64) -> Vec<u8> {
    ti_raw(&s(name), dims.len() as u32, dims, ty, off)
}
fn ti_raw(name: &[u8], n_dims: u32, dims: &[u64], ty: u32, off: u64) -> Vec<u8> {
    let mut out = name.to_vec();
    out.extend_from_slice(&n_dims.to_le_bytes());
    for d in dims {
        out.extend_from_slice(&d.to_le_bytes());
    }
    out.extend_from_slice(&ty.to_le_bytes());
    out.extend_from_slice(&off.to_le_bytes());
    out
}

struct Gguf {
    magic: [u8; 4],
    version: u32,
    kv_count: Option<u64>,
    tensor_count: Option<u64>,
    kvs: Vec<Vec<u8>>,
    tensors: Vec<Vec<u8>>,
}

impl Gguf {
    fn new() -> Self {
        Self {
            magic: *b"GGUF",
            version: 3,
            kv_count: None,
            tensor_count: None,
            kvs: vec![],
            tensors: vec![],
        }
    }
    fn kv(mut self, entry: Vec<u8>) -> Self {
        self.kvs.push(entry);
        self
    }
    fn tensor(mut self, entry: Vec<u8>) -> Self {
        self.tensors.push(entry);
        self
    }
    /// Header bytes only, with no padding and no payload.
    fn header(&self) -> Vec<u8> {
        let mut out = self.magic.to_vec();
        out.extend_from_slice(&self.version.to_le_bytes());
        out.extend_from_slice(
            &self
                .tensor_count
                .unwrap_or(self.tensors.len() as u64)
                .to_le_bytes(),
        );
        out.extend_from_slice(&self.kv_count.unwrap_or(self.kvs.len() as u64).to_le_bytes());
        for k in &self.kvs {
            out.extend_from_slice(k);
        }
        for t in &self.tensors {
            out.extend_from_slice(t);
        }
        out
    }
    /// Header, zero padding to `align`, then `payload` zero bytes.
    fn file(&self, align: usize, payload: usize) -> Vec<u8> {
        let mut out = self.header();
        out.resize(out.len().div_ceil(align) * align, 0);
        out.resize(out.len() + payload, 0);
        out
    }
}

fn parse(bytes: &[u8]) -> Result<GgufHeader, EngError> {
    GgufHeader::from_bytes(bytes, bytes.len() as u64, LIMIT)
}
/// Asserts a `Checkpoint` error whose message contains `expect`.
fn assert_checkpoint(result: Result<GgufHeader, EngError>, expect: &str) {
    match result {
        Err(EngError::Checkpoint(message)) => assert!(
            message.contains(expect),
            "expected message with {expect:?}, got {message:?}"
        ),
        other => panic!("expected Checkpoint error with {expect:?}, got {other:?}"),
    }
}

#[test]
fn type_table_matches_the_pinned_reference() {
    for &(id, name, elems, bytes) in TYPES {
        let t = GgufType::from_id(id).unwrap_or_else(|e| panic!("{name}: {e:?}"));
        assert_eq!(t.id(), id, "{name}");
        assert_eq!(t.name(), name);
        assert_eq!(t.block_elems(), elems, "{name}");
        assert_eq!(t.block_bytes(), bytes, "{name}");
    }
}

#[test]
fn unknown_or_unsupported_type_ids_are_errors() {
    for id in [
        4,
        5,
        9,
        15,
        31,
        32,
        33,
        34,
        35,
        36,
        39,
        40,
        41,
        43,
        1000,
        u32::MAX,
    ] {
        match GgufType::from_id(id) {
            Err(EngError::Checkpoint(m)) => assert!(
                m.contains(&format!("unknown or unsupported tensor type id {id}")),
                "id {id}: {m}"
            ),
            other => panic!("id {id}: {other:?}"),
        }
    }
}

#[test]
fn valid_multi_type_file_records_ranges_and_sizes() {
    let mut g = Gguf::new().kv(kv_str("general.architecture", "test"));
    let mut off = 0u64;
    let mut expect = vec![];
    for &(id, name, elems, bytes) in TYPES {
        // Two blocks in a row, three rows.
        let dims = [elems * 2, 3];
        let size = 6 * bytes;
        g = g.tensor(ti(&format!("t.{name}"), &dims, id, off));
        expect.push((format!("t.{name}"), off, size));
        off = (off + size).div_ceil(32) * 32;
    }
    let file = g.file(32, off as usize);
    let h = parse(&file).unwrap();
    assert_eq!(h.version(), 3);
    assert_eq!(h.alignment(), 32);
    assert_eq!(h.header_len(), g.header().len() as u64);
    assert_eq!(h.data_start(), (g.header().len() as u64).div_ceil(32) * 32);
    assert_eq!(h.tensors().len(), TYPES.len());
    assert_eq!(h.file_len(), file.len() as u64);
    for &(id, name, elems, bytes) in TYPES {
        let t = h.tensor(&format!("t.{name}")).unwrap();
        assert_eq!(t.ggml_type(), GgufType::from_id(id).unwrap(), "{name}");
        assert_eq!(t.dims(), &[elems * 2, 3]);
        assert_eq!(t.element_count(), elems * 6, "{name}");
        assert_eq!(t.byte_len(), 6 * bytes, "{name}");
    }
    for (name, rel, size) in expect {
        let t = h.tensor(&name).unwrap();
        assert_eq!(t.data_offset(), rel);
        assert_eq!(t.abs_offset(), h.data_start() + rel);
        assert_eq!(t.byte_len(), size, "{name}");
        assert_eq!(t.abs_range(), t.abs_offset()..t.abs_offset() + size);
    }
    assert_eq!(
        h.meta("general.architecture"),
        Some(&MetaValue::String("test".into()))
    );
    assert!(h.split().is_none());
}

#[test]
fn element_count_and_dims_are_recorded() {
    let g = Gguf::new().tensor(ti("w", &[64, 5, 2], 42, 0));
    let h = parse(&g.file(32, 18 * 10)).unwrap();
    let t = h.tensor("w").unwrap();
    assert_eq!(t.dims(), &[64, 5, 2]);
    assert_eq!(t.element_count(), 640);
    assert_eq!(t.byte_len(), 180);
}

#[test]
fn ngram_table_size_matches_the_plan_without_payload() {
    // Plan section 3.3: 90 B rows, 320,001,536 rows.
    let g = Gguf::new().tensor(ti(
        "per_layer_token_embd.weight",
        &[160, 320_001_536],
        20,
        0,
    ));
    let header = g.header();
    let data_start = (header.len() as u64).div_ceil(32) * 32;
    let h = GgufHeader::from_bytes(&header, data_start + 28_800_138_240, LIMIT).unwrap();
    assert_eq!(h.tensors()[0].byte_len(), 28_800_138_240);
    assert_eq!(h.data_start(), data_start);
}

#[test]
fn custom_alignment_moves_the_data_start() {
    let g = Gguf::new()
        .kv(kv_u32("general.alignment", 64))
        .tensor(ti("a", &[32], 20, 0))
        .tensor(ti("b", &[32], 20, 64));
    let h = parse(&g.file(64, 128)).unwrap();
    assert_eq!(h.alignment(), 64);
    assert_eq!(h.data_start() % 64, 0);
    assert_eq!(h.tensor("b").unwrap().abs_offset(), h.data_start() + 64);
}

#[test]
fn bad_alignment_values_are_rejected() {
    for bad in [0u32, 3, 48, 100] {
        let g = Gguf::new().kv(kv_u32("general.alignment", bad));
        assert_checkpoint(parse(&g.file(32, 0)), "general.alignment");
    }
    let g = Gguf::new().kv(kv_str("general.alignment", "32"));
    assert_checkpoint(parse(&g.file(32, 0)), "general.alignment");
}

#[test]
fn bad_magic_and_version_are_rejected() {
    let mut g = Gguf::new();
    g.magic = *b"GGUX";
    assert_checkpoint(parse(&g.file(32, 0)), "bad magic");
    for v in [0u32, 1, 2, 4, 0x0300_0000] {
        let mut g = Gguf::new();
        g.version = v;
        assert_checkpoint(parse(&g.file(32, 0)), "unsupported version");
    }
}

#[test]
fn every_truncation_of_the_header_fails() {
    let g = Gguf::new()
        .kv(kv_str("general.architecture", "test"))
        .kv(kv_u32("x", 7))
        .kv(kv(
            "arr",
            9,
            &[
                &8u32.to_le_bytes()[..],
                &2u64.to_le_bytes(),
                &s("a"),
                &s("bc"),
            ]
            .concat(),
        ))
        .tensor(ti("w", &[64, 2], 42, 0));
    let file = g.file(32, 64);
    let header_len = g.header().len();
    for cut in 0..header_len {
        // The prefix is short, but the file length claims the full file.
        let r = GgufHeader::from_bytes(&file[..cut], file.len() as u64, LIMIT);
        match r {
            Err(EngError::Checkpoint(m)) => assert!(m.contains("truncated"), "cut {cut}: {m}"),
            other => panic!("cut {cut}: {other:?}"),
        }
    }
    // A file length below the header length also fails.
    assert_checkpoint(
        GgufHeader::from_bytes(&file, (header_len - 1) as u64, LIMIT),
        "truncated: header exceeds the file length",
    );
}

#[test]
fn zero_header_limit_is_a_bad_request() {
    let file = Gguf::new().file(32, 0);
    assert!(matches!(
        GgufHeader::from_bytes(&file, file.len() as u64, 0),
        Err(EngError::BadRequest(_))
    ));
}

#[test]
fn header_limit_is_enforced() {
    // The last field is a fixed-size u32, so the limit cuts it in the middle.
    let g = Gguf::new().kv(kv_u32("general.architecture", 7));
    let file = g.file(32, 0);
    let len = g.header().len() as u64;
    assert!(GgufHeader::from_bytes(&file, file.len() as u64, len).is_ok());
    assert_checkpoint(
        GgufHeader::from_bytes(&file, file.len() as u64, len - 1),
        "header exceeds the header byte limit",
    );
    // A string that crosses the limit fails at its length check.
    let g = Gguf::new().kv(kv_str("general.architecture", "test"));
    let file = g.file(32, 0);
    let len = g.header().len() as u64;
    assert_checkpoint(
        GgufHeader::from_bytes(&file, file.len() as u64, len - 1),
        "string value length exceeds the remaining header",
    );
}

#[test]
fn huge_counts_fail_before_allocation() {
    let mut g = Gguf::new();
    g.tensor_count = Some(u64::MAX);
    assert_checkpoint(parse(&g.file(32, 0)), "tensor count exceeds");
    let mut g = Gguf::new();
    g.kv_count = Some(u64::MAX / 2);
    assert_checkpoint(parse(&g.file(32, 0)), "metadata count exceeds");
    let mut g = Gguf::new();
    g.tensor_count = Some(LIMIT);
    assert_checkpoint(parse(&g.file(32, 0)), "tensor count exceeds");
}

#[test]
fn overlong_strings_are_rejected() {
    // Declared length far above the limit, with no bytes behind it.
    let mut bad = u64::MAX.to_le_bytes().to_vec();
    bad.extend_from_slice(b"k");
    let mut entry = bad.clone();
    entry.extend_from_slice(&4u32.to_le_bytes());
    entry.extend_from_slice(&0u32.to_le_bytes());
    assert_checkpoint(
        parse(&Gguf::new().kv(entry).file(32, 0)),
        "metadata key is longer than",
    );
    // A key above the name limit but inside the header limit.
    let long = "k".repeat(70_000);
    assert_checkpoint(
        parse(&Gguf::new().kv(kv_u32(&long, 1)).file(32, 0)),
        "metadata key is longer than 65535",
    );
    let long_name = "n".repeat(70_000);
    assert_checkpoint(
        parse(
            &Gguf::new()
                .tensor(ti(&long_name, &[32], 20, 0))
                .file(32, 64),
        ),
        "tensor name is longer than 63",
    );
    // A value string longer than the remaining header budget.
    let mut payload = (LIMIT * 2).to_le_bytes().to_vec();
    payload.extend_from_slice(b"x");
    assert_checkpoint(
        parse(&Gguf::new().kv(kv("v", 8, &payload)).file(32, 0)),
        "string value is longer than",
    );
}

#[test]
fn non_utf8_strings_are_rejected() {
    let key = s_raw(&[b'k', 0xff]);
    let mut entry = key;
    entry.extend_from_slice(&4u32.to_le_bytes());
    entry.extend_from_slice(&1u32.to_le_bytes());
    assert_checkpoint(
        parse(&Gguf::new().kv(entry).file(32, 0)),
        "metadata key is not valid UTF-8",
    );
    let value = kv("v", 8, &s_raw(&[0xc3, 0x28]));
    assert_checkpoint(
        parse(&Gguf::new().kv(value).file(32, 0)),
        "string value is not valid UTF-8",
    );
    let arr = kv(
        "a",
        9,
        &[
            &8u32.to_le_bytes()[..],
            &1u64.to_le_bytes(),
            &s_raw(&[0xff]),
        ]
        .concat(),
    );
    assert_checkpoint(
        parse(&Gguf::new().kv(arr).file(32, 0)),
        "array string is not valid UTF-8",
    );
    let name = ti_raw(&s_raw(&[0xff, 0xfe]), 1, &[32], 20, 0);
    assert_checkpoint(
        parse(&Gguf::new().tensor(name).file(32, 64)),
        "tensor name is not valid UTF-8",
    );
}

#[test]
fn duplicate_names_and_keys_are_rejected() {
    let g = Gguf::new().kv(kv_u32("k", 1)).kv(kv_u32("k", 2));
    assert_checkpoint(parse(&g.file(32, 0)), "duplicate metadata key k");
    let g = Gguf::new()
        .tensor(ti("w", &[32], 20, 0))
        .tensor(ti("w", &[32], 20, 32));
    assert_checkpoint(parse(&g.file(32, 128)), "duplicate tensor name w");
}

#[test]
fn every_metadata_value_type_parses() {
    let g = Gguf::new()
        .kv(kv("u8", 0, &[200]))
        .kv(kv("i8", 1, &[0xfe]))
        .kv(kv("u16", 2, &60000u16.to_le_bytes()))
        .kv(kv("i16", 3, &(-5i16).to_le_bytes()))
        .kv(kv("u32", 4, &4_000_000_000u32.to_le_bytes()))
        .kv(kv("i32", 5, &(-7i32).to_le_bytes()))
        .kv(kv("f32", 6, &1.5f32.to_le_bytes()))
        .kv(kv("bool", 7, &[1]))
        .kv(kv("str", 8, &s("héllo")))
        .kv(kv("u64", 10, &u64::MAX.to_le_bytes()))
        .kv(kv("i64", 11, &i64::MIN.to_le_bytes()))
        .kv(kv("f64", 12, &2.25f64.to_le_bytes()));
    let h = parse(&g.file(32, 0)).unwrap();
    assert_eq!(h.meta("u8"), Some(&MetaValue::U8(200)));
    assert_eq!(h.meta("i8"), Some(&MetaValue::I8(-2)));
    assert_eq!(h.meta("u16"), Some(&MetaValue::U16(60000)));
    assert_eq!(h.meta("i16"), Some(&MetaValue::I16(-5)));
    assert_eq!(h.meta("u32"), Some(&MetaValue::U32(4_000_000_000)));
    assert_eq!(h.meta("i32"), Some(&MetaValue::I32(-7)));
    assert_eq!(h.meta("f32"), Some(&MetaValue::F32(1.5)));
    assert_eq!(h.meta("bool"), Some(&MetaValue::Bool(true)));
    assert_eq!(h.meta("str"), Some(&MetaValue::String("héllo".into())));
    assert_eq!(h.meta("u64"), Some(&MetaValue::U64(u64::MAX)));
    assert_eq!(h.meta("i64"), Some(&MetaValue::I64(i64::MIN)));
    assert_eq!(h.meta("f64"), Some(&MetaValue::F64(2.25)));
    assert_eq!(h.metadata().len(), 12);
}

#[test]
fn arrays_and_nested_arrays_parse_and_record_their_extent() {
    // Array of u32, array of strings, empty array, array of arrays of u16.
    let a_u32 = kv(
        "a.u32",
        9,
        &[&4u32.to_le_bytes()[..], &3u64.to_le_bytes(), &[0u8; 12]].concat(),
    );
    let a_str = kv(
        "a.str",
        9,
        &[
            &8u32.to_le_bytes()[..],
            &2u64.to_le_bytes(),
            &s("a"),
            &s(""),
        ]
        .concat(),
    );
    let a_empty = kv(
        "a.empty",
        9,
        &[&0u32.to_le_bytes()[..], &0u64.to_le_bytes()].concat(),
    );
    let inner = |n: u64| {
        [
            &2u32.to_le_bytes()[..],
            &n.to_le_bytes(),
            &vec![0u8; 2 * n as usize],
        ]
        .concat()
    };
    let a_nest = kv(
        "a.nest",
        9,
        &[
            &9u32.to_le_bytes()[..],
            &2u64.to_le_bytes(),
            &inner(1),
            &inner(3),
        ]
        .concat(),
    );
    let g = Gguf::new().kv(a_u32).kv(a_str).kv(a_empty).kv(a_nest);
    let header = g.header();
    let h = parse(&g.file(32, 0)).unwrap();
    let get = |k: &str| match h.meta(k) {
        Some(MetaValue::Array(a)) => a.clone(),
        other => panic!("{k}: {other:?}"),
    };
    let a = get("a.u32");
    assert_eq!((a.elem_type, a.len), (4, 3));
    assert_eq!(a.byte_range.end - a.byte_range.start, 12);
    let a = get("a.str");
    assert_eq!((a.elem_type, a.len), (8, 2));
    assert_eq!(a.byte_range.end - a.byte_range.start, 8 + 1 + 8);
    let a = get("a.empty");
    assert_eq!((a.elem_type, a.len), (0, 0));
    assert!(a.byte_range.is_empty());
    let a = get("a.nest");
    assert_eq!((a.elem_type, a.len), (9, 2));
    assert_eq!(a.byte_range.end, header.len() as u64);
    // 2 inner headers (12 B each) plus 1 and 3 u16 values.
    assert_eq!(a.byte_range.end - a.byte_range.start, 24 + 2 + 6);
}

#[test]
fn bad_metadata_values_are_rejected() {
    assert_checkpoint(
        parse(&Gguf::new().kv(kv("k", 99, &[0; 8])).file(32, 0)),
        "unknown metadata value type 99",
    );
    assert_checkpoint(
        parse(&Gguf::new().kv(kv("k", 7, &[2])).file(32, 0)),
        "boolean value 2",
    );
    // Array length larger than the bytes that can follow.
    let a = kv(
        "a",
        9,
        &[&4u32.to_le_bytes()[..], &u64::MAX.to_le_bytes()].concat(),
    );
    assert_checkpoint(
        parse(&Gguf::new().kv(a).file(32, 0)),
        "array length exceeds",
    );
    let a = kv(
        "a",
        9,
        &[&4u32.to_le_bytes()[..], &(LIMIT).to_le_bytes()].concat(),
    );
    assert_checkpoint(
        parse(&Gguf::new().kv(a).file(32, 0)),
        "array length exceeds",
    );
    // Unknown element type inside an array.
    let a = kv(
        "a",
        9,
        &[&77u32.to_le_bytes()[..], &1u64.to_le_bytes(), &[0; 8]].concat(),
    );
    assert_checkpoint(
        parse(&Gguf::new().kv(a).file(32, 0)),
        "unknown array element type 77",
    );
    // Bool array with an invalid element.
    let a = kv(
        "a",
        9,
        &[&7u32.to_le_bytes()[..], &2u64.to_le_bytes(), &[1, 5]].concat(),
    );
    assert_checkpoint(
        parse(&Gguf::new().kv(a).file(32, 0)),
        "boolean array element",
    );
    // Empty key.
    assert_checkpoint(
        parse(&Gguf::new().kv(kv_u32("", 1)).file(32, 0)),
        "metadata key is empty",
    );
}

#[test]
fn excessive_array_nesting_is_rejected() {
    let mut payload = vec![];
    for _ in 0..40 {
        payload.extend_from_slice(&9u32.to_le_bytes());
        payload.extend_from_slice(&1u64.to_le_bytes());
    }
    payload.extend_from_slice(&0u32.to_le_bytes());
    payload.extend_from_slice(&0u64.to_le_bytes());
    let g = Gguf::new().kv(kv("deep", 9, &payload));
    assert_checkpoint(parse(&g.file(32, 0)), "nested too deeply");
}

#[test]
fn bad_tensor_entries_are_rejected() {
    let f = |t: Vec<u8>, payload: usize| parse(&Gguf::new().tensor(t).file(32, payload));
    assert_checkpoint(
        f(ti("w", &[32], 4, 0), 64),
        "unknown or unsupported tensor type id 4",
    );
    assert_checkpoint(
        f(ti("w", &[32], 9999, 0), 64),
        "unknown or unsupported tensor type id 9999",
    );
    assert_checkpoint(f(ti_raw(&s("w"), 0, &[], 0, 0), 64), "has 0 dimensions");
    assert_checkpoint(f(ti_raw(&s("w"), 5, &[1; 5], 0, 0), 64), "has 5 dimensions");
    assert_checkpoint(f(ti("w", &[32, 0], 20, 0), 64), "zero dimension");
    assert_checkpoint(f(ti("w", &[33], 20, 0), 64), "not a multiple of the");
    assert_checkpoint(
        f(ti("w", &[u64::MAX, u64::MAX], 0, 0), 64),
        "element count overflows",
    );
    assert_checkpoint(f(ti("w", &[1u64 << 62], 0, 0), 64), "byte size overflows");
    assert_checkpoint(
        f(ti("w", &[1, u64::MAX, 2], 0, 0), 64),
        "tensor w size overflow",
    );
    assert_checkpoint(f(ti("", &[32], 20, 0), 64), "tensor name is empty");
}

#[test]
fn tensor_ranges_are_checked_against_the_file() {
    let one = |rel: u64, payload: usize| {
        parse(
            &Gguf::new()
                .tensor(ti("w", &[32], 20, rel))
                .file(32, payload),
        )
    };
    assert!(one(0, 18).is_ok());
    assert_checkpoint(one(0, 17), "extends beyond");
    assert_checkpoint(one(32, 49), "extends beyond");
    assert_checkpoint(one(u64::MAX - 31, 64), "offset overflows");
    // The offset fits, but the offset plus the length does not.
    let header_len = Gguf::new().tensor(ti("w", &[256], 12, 0)).header().len() as u64;
    let data_start = header_len.div_ceil(32) * 32;
    let near_end = (u64::MAX - 31) - data_start; // data_start + near_end = 2^64 - 32
    assert_checkpoint(
        parse(
            &Gguf::new()
                .tensor(ti("w", &[256], 12, near_end))
                .file(32, 256),
        ),
        "range overflows",
    );
    for rel in [1u64, 8, 31, 33] {
        assert_checkpoint(one(rel, 128), "is not aligned to 32");
    }
}

#[test]
fn overlapping_tensors_are_rejected() {
    let g = Gguf::new()
        .tensor(ti("a", &[256], 12, 0)) // 144 B
        .tensor(ti("b", &[256], 12, 128));
    assert_checkpoint(parse(&g.file(32, 512)), "tensors a and b overlap");
    // Same range twice, and out-of-order table entries that do not overlap.
    let g = Gguf::new()
        .tensor(ti("a", &[256], 12, 0))
        .tensor(ti("b", &[256], 12, 0));
    assert_checkpoint(parse(&g.file(32, 512)), "overlap");
    let g = Gguf::new()
        .tensor(ti("b", &[256], 12, 192))
        .tensor(ti("a", &[256], 12, 0));
    assert!(parse(&g.file(32, 512)).is_ok());
    // Touching ranges are fine.
    let g = Gguf::new()
        .tensor(ti("a", &[32], 20, 0)) // 18 B
        .tensor(ti("b", &[32], 20, 32));
    assert!(parse(&g.file(32, 128)).is_ok());
}

#[test]
fn data_start_beyond_the_file_is_rejected() {
    let g = Gguf::new().kv(kv_str("general.architecture", "test"));
    let header = g.header();
    // File ends inside the alignment padding.
    let file_len = header.len() as u64;
    assert!(!header.len().is_multiple_of(32));
    assert_checkpoint(
        GgufHeader::from_bytes(&header, file_len, LIMIT),
        "data section starts beyond the end",
    );
}

#[test]
fn split_metadata_is_parsed_and_checked() {
    let ok = Gguf::new()
        .kv(kv_u16("split.no", 0))
        .kv(kv_u16("split.count", 2))
        .kv(kv_i32("split.tensors.count", 3))
        .tensor(ti("a", &[32], 20, 0))
        .tensor(ti("b", &[32], 20, 32));
    let h = parse(&ok.file(32, 128)).unwrap();
    assert_eq!(
        h.split(),
        Some(SplitInfo {
            count: 2,
            no: 0,
            tensors_count: Some(3)
        })
    );
    let bad: Vec<Gguf> = vec![
        Gguf::new()
            .kv(kv_u16("split.no", 2))
            .kv(kv_u16("split.count", 2)),
        Gguf::new().kv(kv_u16("split.count", 2)),
        Gguf::new().kv(kv_u16("split.no", 0)),
        Gguf::new()
            .kv(kv_u16("split.no", 0))
            .kv(kv_u16("split.count", 0)),
        Gguf::new()
            .kv(kv_str("split.no", "0"))
            .kv(kv_u16("split.count", 2)),
        Gguf::new()
            .kv(kv_u16("split.no", 0))
            .kv(kv_u16("split.count", 2))
            .kv(kv_i32("split.tensors.count", -1)),
        // Total below this shard's own tensor count.
        Gguf::new()
            .kv(kv_u16("split.no", 0))
            .kv(kv_u16("split.count", 2))
            .kv(kv_i32("split.tensors.count", 1))
            .tensor(ti("a", &[32], 20, 0))
            .tensor(ti("b", &[32], 20, 32)),
        // A single-file split must list exactly its own tensors.
        Gguf::new()
            .kv(kv_u16("split.no", 0))
            .kv(kv_u16("split.count", 1))
            .kv(kv_i32("split.tensors.count", 5))
            .tensor(ti("a", &[32], 20, 0)),
    ];
    let expect = [
        "split.no 2 is outside split.count 2",
        "must appear together",
        "must appear together",
        "is outside split.count 0",
        "split.no is not an integer",
        "split.tensors.count is negative",
        "disagrees with 2 tensors",
        "disagrees with 1 tensors",
    ];
    assert_eq!(bad.len(), expect.len());
    for (g, expect) in bad.iter().zip(expect) {
        assert_checkpoint(parse(&g.file(32, 128)), expect);
    }
}

#[test]
fn split_sets_must_be_complete_and_consistent() {
    let shard = |no: u16, count: u16, total: i32, names: &[&str]| {
        let mut g = Gguf::new()
            .kv(kv_u16("split.no", no))
            .kv(kv_u16("split.count", count))
            .kv(kv_i32("split.tensors.count", total));
        for (i, n) in names.iter().enumerate() {
            g = g.tensor(ti(n, &[32], 20, 32 * i as u64));
        }
        parse(&g.file(32, 256)).unwrap()
    };
    let s0 = shard(0, 2, 3, &["a", "b"]);
    let s1 = shard(1, 2, 3, &["c"]);
    check_split_set(&[&s0, &s1]).unwrap();
    check_split_set(&[&s1, &s0]).unwrap(); // order does not matter
    assert_checkpoint_unit(
        check_split_set(&[&s0]),
        "split set has 1 shards but split.count is 2",
    );
    assert_checkpoint_unit(
        check_split_set(&[&s0, &s0]),
        "split.no values do not cover each shard once",
    );
    // Both shards agree on a total that the listed tensors do not match.
    let a0 = shard(0, 2, 4, &["a", "b"]);
    let a1 = shard(1, 2, 4, &["c"]);
    assert_checkpoint_unit(
        check_split_set(&[&a0, &a1]),
        "split.tensors.count 4 differs from 3 listed tensors",
    );
    let wrong_total = shard(1, 2, 4, &["c"]);
    assert_checkpoint_unit(
        check_split_set(&[&s0, &wrong_total]),
        "shards disagree on split.tensors.count",
    );
    let wrong_count = shard(1, 3, 3, &["c"]);
    assert_checkpoint_unit(
        check_split_set(&[&s0, &wrong_count]),
        "shards disagree on split.count",
    );
    let dup_name = shard(1, 2, 3, &["a"]);
    assert_checkpoint_unit(
        check_split_set(&[&s0, &dup_name]),
        "tensor a appears in more than one shard",
    );
    let mixed_total = shard(1, 2, 2, &["c"]);
    assert_checkpoint_unit(
        check_split_set(&[&s0, &mixed_total]),
        "shards disagree on split.tensors.count",
    );
    // A header without split metadata is not a set member.
    let plain = parse(&Gguf::new().file(32, 0)).unwrap();
    assert_checkpoint_unit(check_split_set(&[&plain]), "shard has no split metadata");
    assert_checkpoint_unit(check_split_set(&[]), "split set is empty");
}

struct Guarded<'a> {
    data: &'a [u8],
    pos: usize,
    max: usize,
}
impl Read for Guarded<'_> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        assert!(
            self.pos + buf.len().min(self.data.len() - self.pos) <= self.max || buf.is_empty(),
            "read past the header: pos {} buf {}",
            self.pos,
            buf.len()
        );
        let n = buf.len().min(self.data.len() - self.pos);
        buf[..n].copy_from_slice(&self.data[self.pos..self.pos + n]);
        self.pos += n;
        Ok(n)
    }
}

#[test]
fn reader_never_reads_past_the_header() {
    let g = Gguf::new()
        .kv(kv_str("general.architecture", "test"))
        .kv(kv(
            "arr",
            9,
            &[&4u32.to_le_bytes()[..], &100u64.to_le_bytes(), &[0u8; 400]].concat(),
        ))
        .tensor(ti("w", &[64], 42, 0));
    let file = g.file(32, 1000);
    let header_len = g.header().len();
    let mut r = Guarded {
        data: &file,
        pos: 0,
        max: header_len,
    };
    let h = GgufHeader::parse(&mut r, file.len() as u64, LIMIT).unwrap();
    assert_eq!(r.pos, header_len);
    assert_eq!(h.header_len(), header_len as u64);
    let from_slice = parse(&file).unwrap();
    assert_eq!(from_slice.tensors(), h.tensors());
    assert_eq!(from_slice.metadata(), h.metadata());
}

#[test]
fn header_hash_covers_exactly_the_header_bytes() {
    let g = Gguf::new()
        .kv(kv_str("general.architecture", "test"))
        .tensor(ti("w", &[64], 42, 0));
    let file = g.file(32, 64);
    let h = parse(&file).unwrap();
    let header = &file[..h.header_len() as usize];
    let (digest, n) = sha256_stream(&mut &header[..]).unwrap();
    assert_eq!(n, header.len() as u64);
    assert_eq!(h.header_sha256(), digest);
    let hex = sha256_hex(&digest);
    h.check_header_sha256(&hex).unwrap();
    h.check_header_sha256(&hex.to_uppercase()).unwrap();
    // Payload bytes change nothing; a different header changes the hash.
    let mut other = file.clone();
    *other.last_mut().unwrap() = 9;
    assert_eq!(parse(&other).unwrap().header_sha256(), digest);
    let g2 = Gguf::new()
        .kv(kv_str("general.architecture", "tesu"))
        .tensor(ti("w", &[64], 42, 0));
    assert_ne!(parse(&g2.file(32, 64)).unwrap().header_sha256(), digest);
    let mut wrong = hex.clone();
    wrong.replace_range(0..1, if hex.starts_with('0') { "1" } else { "0" });
    assert_checkpoint_unit(h.check_header_sha256(&wrong), "header sha256");
    assert_checkpoint_unit(h.check_header_sha256("abc"), "64 hex digits");
    assert_checkpoint_unit(h.check_header_sha256(&"g".repeat(64)), "64 hex digits");
}

/// Asserts a `Checkpoint` error whose message contains `expect`.
fn assert_checkpoint_unit(r: Result<(), EngError>, expect: &str) {
    match r {
        Err(EngError::Checkpoint(message)) => assert!(
            message.contains(expect),
            "expected message with {expect:?}, got {message:?}"
        ),
        other => panic!("expected Checkpoint error with {expect:?}, got {other:?}"),
    }
}

#[test]
fn sha256_known_vectors_and_streaming() {
    let (d, n) = sha256_stream(&mut &b"abc"[..]).unwrap();
    assert_eq!(n, 3);
    assert_eq!(
        sha256_hex(&d),
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    );
    let (d, n) = sha256_stream(&mut &b""[..]).unwrap();
    assert_eq!(n, 0);
    assert_eq!(
        sha256_hex(&d),
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    );
    // More than one internal buffer, fed in odd chunks.
    struct Chunky<'a>(&'a [u8]);
    impl Read for Chunky<'_> {
        fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
            let n = buf.len().min(self.0.len()).min(777);
            buf[..n].copy_from_slice(&self.0[..n]);
            self.0 = &self.0[n..];
            Ok(n)
        }
    }
    let big: Vec<u8> = (0..3_000_017u32).map(|i| (i % 251) as u8).collect();
    let (a, na) = sha256_stream(&mut Chunky(&big)).unwrap();
    let (b, nb) = sha256_stream(&mut &big[..]).unwrap();
    assert_eq!((a, na), (b, nb));
    assert_eq!(na, big.len() as u64);
}

#[test]
fn verify_sha256_checks_length_and_digest() {
    let data = b"abc";
    let good = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";
    verify_sha256(&mut &data[..], 3, good).unwrap();
    verify_sha256(&mut &data[..], 3, &good.to_uppercase()).unwrap();
    assert_checkpoint_unit(
        verify_sha256(&mut &data[..], 4, good),
        "hashed 3 bytes but expected 4",
    );
    assert_checkpoint_unit(
        verify_sha256(&mut &data[..], 2, good),
        "hashed 3 bytes but expected 2",
    );
    assert_checkpoint_unit(
        verify_sha256(&mut &b"abd"[..], 3, good),
        "differs from the recorded",
    );
    assert_checkpoint_unit(
        verify_sha256(&mut &data[..], 3, &good[..63]),
        "64 hex digits",
    );
    assert_checkpoint_unit(
        verify_sha256(&mut &data[..], 3, &format!("{}z", &good[..63])),
        "64 hex digits",
    );
}

#[test]
fn read_errors_are_checkpoint_errors() {
    struct Failing;
    impl Read for Failing {
        fn read(&mut self, _: &mut [u8]) -> io::Result<usize> {
            Err(io::Error::other("disk"))
        }
    }
    assert_checkpoint(
        GgufHeader::parse(&mut Failing, 1 << 20, LIMIT),
        "header read: disk",
    );
    match sha256_stream(&mut Failing) {
        Err(EngError::Checkpoint(m)) => assert!(m.contains("hash read: disk"), "{m}"),
        other => panic!("{other:?}"),
    }
}

#[test]
fn tensor_names_follow_the_format_limit() {
    let at = |len: usize| {
        let name = "n".repeat(len);
        parse(&Gguf::new().tensor(ti(&name, &[32], 20, 0)).file(32, 64))
    };
    assert_eq!(at(63).unwrap().tensors()[0].name().len(), 63);
    assert_checkpoint(at(64), "tensor name is longer than 63 bytes");
    assert_checkpoint(at(65), "tensor name is longer than 63 bytes");
    // Metadata keys keep the larger limit.
    let key = "k".repeat(64);
    assert!(parse(&Gguf::new().kv(kv_u32(&key, 1)).file(32, 0)).is_ok());
}

#[test]
fn false_booleans_parse() {
    let h = parse(&Gguf::new().kv(kv("flag", 7, &[0])).file(32, 0)).unwrap();
    assert_eq!(h.meta("flag"), Some(&MetaValue::Bool(false)));
}

#[test]
fn split_numbers_accept_every_integer_type() {
    // (type id, payload for the value 1 or 2)
    let int = |ty: u32, v: u8| -> Vec<u8> {
        match ty {
            0 | 1 => vec![v],
            2 | 3 => u16::from(v).to_le_bytes().to_vec(),
            4 | 5 => u32::from(v).to_le_bytes().to_vec(),
            _ => u64::from(v).to_le_bytes().to_vec(),
        }
    };
    for (count_ty, no_ty) in [(0, 1), (3, 2), (5, 4), (10, 11), (11, 10)] {
        let g = Gguf::new()
            .kv(kv("split.count", count_ty, &int(count_ty, 2)))
            .kv(kv("split.no", no_ty, &int(no_ty, 1)));
        let h = parse(&g.file(32, 0)).unwrap();
        assert_eq!(
            h.split(),
            Some(SplitInfo {
                count: 2,
                no: 1,
                tensors_count: None
            }),
            "types {count_ty} {no_ty}"
        );
    }
    // Negative values of each signed type are rejected by name.
    for (ty, bytes) in [
        (1u32, (-1i8).to_le_bytes().to_vec()),
        (3, (-1i16).to_le_bytes().to_vec()),
        (5, (-1i32).to_le_bytes().to_vec()),
        (11, (-1i64).to_le_bytes().to_vec()),
    ] {
        let g = Gguf::new()
            .kv(kv("split.count", 2, &2u16.to_le_bytes()))
            .kv(kv("split.no", ty, &bytes));
        assert_checkpoint(parse(&g.file(32, 0)), "split.no is negative");
    }
}
