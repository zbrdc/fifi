//! GGUF v3 header and tensor table reader (slice G1).
//!
//! The reader consumes only the header bytes: the fixed prefix, the metadata
//! entries, and the tensor table. It never reads tensor payloads. Every
//! malformed input fails closed with `EngError::Checkpoint`.
//!
//! The layout follows the GGUF format specification. All integers are
//! little-endian. Tensor offsets are relative to the data section, which starts
//! at the header end rounded up to `general.alignment` (default 32).
//!
//! Deliberate differences from the llama.cpp reader (`gguf.cpp`):
//! - Metadata arrays may nest up to four levels deep. Deeper nesting fails.
//! - Tensor data offsets may leave gaps and may appear in any table order.
//!   Only overlap, misalignment, and out-of-file ranges fail.
//! - The data start is the padded header end, even when the file has no tensors.
//! - A tensor dimension of zero fails.
//! - Metadata keys may be up to 65,535 bytes. Tensor names follow the format
//!   limit and must be shorter than 64 bytes.

use std::{
    collections::HashSet,
    io::{ErrorKind, Read},
    ops::Range,
};

use sha2::{Digest, Sha256};

use crate::EngError;

const MAGIC: [u8; 4] = *b"GGUF";
const SUPPORTED_VERSION: u32 = 3;
const DEFAULT_ALIGNMENT: u64 = 32;
const MAX_KEY_BYTES: u64 = 65_535;
/// The format limit is `GGML_MAX_NAME` (64) including the terminator.
const MAX_TENSOR_NAME_BYTES: u64 = 63;
const STRING_CHUNK: u64 = 1 << 16;
const MAX_TENSOR_DIMS: u32 = 4;
/// Deepest array-in-array nesting accepted. The top-level array has depth 0.
const MAX_ARRAY_DEPTH: u32 = 4;
/// Smallest encoded metadata entry: key length, one key byte, type, one value byte.
const MIN_KV_BYTES: u64 = 8 + 1 + 4 + 1;
/// Smallest encoded tensor entry: name, one dimension, type, and offset.
const MIN_TENSOR_BYTES: u64 = 8 + 1 + 4 + 8 + 4 + 8;
const SKIP_CHUNK: usize = 8192;
const HASH_CHUNK: usize = 1 << 20;

fn bad<T>(message: impl Into<String>) -> Result<T, EngError> {
    Err(EngError::Checkpoint(format!("gguf: {}", message.into())))
}

/// A ggml tensor storage type that this reader can size.
///
/// Type ids and block geometry follow `ggml/include/ggml.h` and
/// `ggml/src/ggml-common.h` at llama.cpp commit `fb4b2737`.
/// NVFP4 also follows the Flint reference at `a37cc461`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GgufType {
    F32,
    F16,
    Bf16,
    F64,
    I8,
    I16,
    I32,
    I64,
    Q40,
    Q41,
    Q50,
    Q51,
    Q80,
    Nvfp4,
    Q20,
    Q2K,
    Q3K,
    Q4K,
    Q5K,
    Q6K,
    Iq2Xxs,
    Iq2Xs,
    Iq2S,
    Iq3Xxs,
    Iq3S,
    Iq1S,
    Iq1M,
    Iq4Nl,
    Iq4Xs,
}

impl GgufType {
    /// Maps a ggml type id to a type. An unknown or unsupported id is an error.
    pub fn from_id(id: u32) -> Result<Self, EngError> {
        Ok(match id {
            0 => Self::F32,
            1 => Self::F16,
            2 => Self::Q40,
            3 => Self::Q41,
            6 => Self::Q50,
            7 => Self::Q51,
            8 => Self::Q80,
            10 => Self::Q2K,
            11 => Self::Q3K,
            12 => Self::Q4K,
            13 => Self::Q5K,
            14 => Self::Q6K,
            16 => Self::Iq2Xxs,
            17 => Self::Iq2Xs,
            18 => Self::Iq3Xxs,
            19 => Self::Iq1S,
            20 => Self::Iq4Nl,
            21 => Self::Iq3S,
            22 => Self::Iq2S,
            23 => Self::Iq4Xs,
            24 => Self::I8,
            25 => Self::I16,
            26 => Self::I32,
            27 => Self::I64,
            28 => Self::F64,
            29 => Self::Iq1M,
            30 => Self::Bf16,
            40 => Self::Nvfp4,
            42 => Self::Q20,
            _ => return bad(format!("unknown or unsupported tensor type id {id}")),
        })
    }

    pub fn id(self) -> u32 {
        match self {
            Self::F32 => 0,
            Self::F16 => 1,
            Self::Q40 => 2,
            Self::Q41 => 3,
            Self::Q50 => 6,
            Self::Q51 => 7,
            Self::Q80 => 8,
            Self::Q2K => 10,
            Self::Q3K => 11,
            Self::Q4K => 12,
            Self::Q5K => 13,
            Self::Q6K => 14,
            Self::Iq2Xxs => 16,
            Self::Iq2Xs => 17,
            Self::Iq3Xxs => 18,
            Self::Iq1S => 19,
            Self::Iq4Nl => 20,
            Self::Iq3S => 21,
            Self::Iq2S => 22,
            Self::Iq4Xs => 23,
            Self::I8 => 24,
            Self::I16 => 25,
            Self::I32 => 26,
            Self::I64 => 27,
            Self::F64 => 28,
            Self::Iq1M => 29,
            Self::Bf16 => 30,
            Self::Nvfp4 => 40,
            Self::Q20 => 42,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::F32 => "F32",
            Self::F16 => "F16",
            Self::Bf16 => "BF16",
            Self::F64 => "F64",
            Self::I8 => "I8",
            Self::I16 => "I16",
            Self::I32 => "I32",
            Self::I64 => "I64",
            Self::Q40 => "Q4_0",
            Self::Q41 => "Q4_1",
            Self::Q50 => "Q5_0",
            Self::Q51 => "Q5_1",
            Self::Q80 => "Q8_0",
            Self::Nvfp4 => "NVFP4",
            Self::Q20 => "Q2_0",
            Self::Q2K => "Q2_K",
            Self::Q3K => "Q3_K",
            Self::Q4K => "Q4_K",
            Self::Q5K => "Q5_K",
            Self::Q6K => "Q6_K",
            Self::Iq2Xxs => "IQ2_XXS",
            Self::Iq2Xs => "IQ2_XS",
            Self::Iq2S => "IQ2_S",
            Self::Iq3Xxs => "IQ3_XXS",
            Self::Iq3S => "IQ3_S",
            Self::Iq1S => "IQ1_S",
            Self::Iq1M => "IQ1_M",
            Self::Iq4Nl => "IQ4_NL",
            Self::Iq4Xs => "IQ4_XS",
        }
    }

    /// Weights per block. A row length must be a multiple of this value.
    pub fn block_elems(self) -> u64 {
        match self {
            Self::F32
            | Self::F16
            | Self::Bf16
            | Self::F64
            | Self::I8
            | Self::I16
            | Self::I32
            | Self::I64 => 1,
            Self::Q40 | Self::Q41 | Self::Q50 | Self::Q51 | Self::Q80 | Self::Iq4Nl => 32,
            Self::Q20 | Self::Nvfp4 => 64,
            Self::Q2K
            | Self::Q3K
            | Self::Q4K
            | Self::Q5K
            | Self::Q6K
            | Self::Iq2Xxs
            | Self::Iq2Xs
            | Self::Iq2S
            | Self::Iq3Xxs
            | Self::Iq3S
            | Self::Iq1S
            | Self::Iq1M
            | Self::Iq4Xs => 256,
        }
    }

    /// Stored bytes per block.
    pub fn block_bytes(self) -> u64 {
        match self {
            Self::I8 => 1,
            Self::F16 | Self::Bf16 | Self::I16 => 2,
            Self::F32 | Self::I32 => 4,
            Self::F64 | Self::I64 => 8,
            Self::Q40 | Self::Q20 | Self::Iq4Nl => 18,
            Self::Q41 => 20,
            Self::Q50 => 22,
            Self::Q51 => 24,
            Self::Q80 => 34,
            Self::Nvfp4 => 36,
            Self::Q2K => 84,
            Self::Q3K | Self::Iq3S => 110,
            Self::Q4K => 144,
            Self::Q5K => 176,
            Self::Q6K => 210,
            Self::Iq2Xxs => 66,
            Self::Iq2Xs => 74,
            Self::Iq2S => 82,
            Self::Iq3Xxs => 98,
            Self::Iq1S => 50,
            Self::Iq1M => 56,
            Self::Iq4Xs => 136,
        }
    }
}

/// A metadata value. Arrays are validated in full but not retained.
#[derive(Debug, Clone, PartialEq)]
pub enum MetaValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    String(String),
    U64(u64),
    I64(i64),
    F64(f64),
    Array(ArrayMeta),
}

/// The extent of a validated metadata array.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArrayMeta {
    /// GGUF metadata type id of every element.
    pub elem_type: u32,
    /// Element count.
    pub len: u64,
    /// File byte range of the element data, after the element type and count.
    pub byte_range: Range<u64>,
}

/// Split-file metadata of one shard.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SplitInfo {
    /// `split.count`: the number of shards.
    pub count: u64,
    /// `split.no`: this shard's zero-based index.
    pub no: u64,
    /// `split.tensors.count`: the tensor total over all shards, when present.
    pub tensors_count: Option<u64>,
}

/// One tensor table entry with its checked byte range.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GgufTensor {
    name: String,
    dims: Vec<u64>,
    ggml_type: GgufType,
    data_offset: u64,
    abs_offset: u64,
    byte_len: u64,
    element_count: u64,
}

impl GgufTensor {
    pub fn name(&self) -> &str {
        &self.name
    }
    /// Dimensions in GGUF order: the first entry is the contiguous row length.
    pub fn dims(&self) -> &[u64] {
        &self.dims
    }
    pub fn ggml_type(&self) -> GgufType {
        self.ggml_type
    }
    /// Offset relative to the data section start.
    pub fn data_offset(&self) -> u64 {
        self.data_offset
    }
    /// Offset from the start of the file.
    pub fn abs_offset(&self) -> u64 {
        self.abs_offset
    }
    pub fn byte_len(&self) -> u64 {
        self.byte_len
    }
    pub fn element_count(&self) -> u64 {
        self.element_count
    }
    /// Absolute byte range in the file.
    pub fn abs_range(&self) -> Range<u64> {
        self.abs_offset..self.abs_offset + self.byte_len
    }
}

/// A parsed and checked GGUF header and tensor table.
#[derive(Debug, Clone)]
pub struct GgufHeader {
    version: u32,
    alignment: u64,
    data_start: u64,
    header_len: u64,
    file_len: u64,
    header_sha256: [u8; 32],
    metadata: Vec<(String, MetaValue)>,
    tensors: Vec<GgufTensor>,
    split: Option<SplitInfo>,
}

struct Src<'a, R: ?Sized> {
    inner: &'a mut R,
    pos: u64,
    max_header: u64,
    file_len: u64,
    hasher: Sha256,
}

impl<R: Read + ?Sized> Src<'_, R> {
    fn remaining(&self) -> u64 {
        self.max_header.min(self.file_len) - self.pos
    }

    fn fill(&mut self, buf: &mut [u8]) -> Result<(), EngError> {
        let end = self.pos.checked_add(buf.len() as u64);
        let Some(end) = end else {
            return bad("header offset overflow");
        };
        if end > self.max_header {
            return bad("header exceeds the header byte limit");
        }
        if end > self.file_len {
            return bad("truncated: header exceeds the file length");
        }
        self.inner.read_exact(buf).map_err(|error| {
            if error.kind() == ErrorKind::UnexpectedEof {
                EngError::Checkpoint("gguf: truncated header".into())
            } else {
                EngError::Checkpoint(format!("gguf: header read: {error}"))
            }
        })?;
        self.hasher.update(&*buf);
        self.pos = end;
        Ok(())
    }

    fn u32(&mut self) -> Result<u32, EngError> {
        let mut b = [0; 4];
        self.fill(&mut b)?;
        Ok(u32::from_le_bytes(b))
    }

    fn u64(&mut self) -> Result<u64, EngError> {
        let mut b = [0; 8];
        self.fill(&mut b)?;
        Ok(u64::from_le_bytes(b))
    }

    fn array<const N: usize>(&mut self) -> Result<[u8; N], EngError> {
        let mut b = [0; N];
        self.fill(&mut b)?;
        Ok(b)
    }

    fn string(&mut self, max_len: u64, what: &str) -> Result<String, EngError> {
        let len = self.u64()?;
        if len > max_len {
            return bad(format!("{what} is longer than {max_len} bytes"));
        }
        if len > self.remaining() {
            return bad(format!("{what} length exceeds the remaining header"));
        }
        // Read in bounded chunks so a declared length never forces one large allocation.
        let mut bytes = Vec::new();
        let mut left = len;
        while left > 0 {
            let n = left.min(STRING_CHUNK) as usize;
            let at = bytes.len();
            bytes.resize(at + n, 0);
            self.fill(&mut bytes[at..])?;
            left -= n as u64;
        }
        String::from_utf8(bytes).or_else(|_| bad(format!("{what} is not valid UTF-8")))
    }

    fn name(&mut self, what: &str, max_len: u64) -> Result<String, EngError> {
        let name = self.string(max_len, what)?;
        if name.is_empty() {
            return bad(format!("{what} is empty"));
        }
        Ok(name)
    }

    /// Reads and discards `len` fixed-size element bytes. Booleans must be 0 or 1.
    fn skip(&mut self, mut len: u64, check_bool: bool) -> Result<(), EngError> {
        let mut buf = [0u8; SKIP_CHUNK];
        while len > 0 {
            let n = len.min(SKIP_CHUNK as u64) as usize;
            self.fill(&mut buf[..n])?;
            if check_bool && buf[..n].iter().any(|&b| b > 1) {
                return bad("boolean array element is not 0 or 1");
            }
            len -= n as u64;
        }
        Ok(())
    }

    fn value(&mut self, ty: u32, depth: u32) -> Result<MetaValue, EngError> {
        Ok(match ty {
            0 => MetaValue::U8(self.array::<1>()?[0]),
            1 => MetaValue::I8(self.array::<1>()?[0] as i8),
            2 => MetaValue::U16(u16::from_le_bytes(self.array()?)),
            3 => MetaValue::I16(i16::from_le_bytes(self.array()?)),
            4 => MetaValue::U32(u32::from_le_bytes(self.array()?)),
            5 => MetaValue::I32(i32::from_le_bytes(self.array()?)),
            6 => MetaValue::F32(f32::from_le_bytes(self.array()?)),
            7 => match self.array::<1>()?[0] {
                0 => MetaValue::Bool(false),
                1 => MetaValue::Bool(true),
                other => return bad(format!("boolean value {other} is not 0 or 1")),
            },
            8 => MetaValue::String(self.string(self.max_header, "string value")?),
            9 => MetaValue::Array(self.array_value(depth)?),
            10 => MetaValue::U64(u64::from_le_bytes(self.array()?)),
            11 => MetaValue::I64(i64::from_le_bytes(self.array()?)),
            12 => MetaValue::F64(f64::from_le_bytes(self.array()?)),
            other => return bad(format!("unknown metadata value type {other}")),
        })
    }

    fn array_value(&mut self, depth: u32) -> Result<ArrayMeta, EngError> {
        if depth >= MAX_ARRAY_DEPTH {
            return bad("metadata arrays are nested too deeply");
        }
        let elem_type = self.u32()?;
        let len = self.u64()?;
        // Smallest encoded element size, which bounds the count by the header size.
        let (min_size, fixed) = match elem_type {
            0 | 1 | 7 => (1, true),
            2 | 3 => (2, true),
            4..=6 => (4, true),
            10..=12 => (8, true),
            8 => (8, false),
            9 => (12, false),
            other => return bad(format!("unknown array element type {other}")),
        };
        match len.checked_mul(min_size) {
            Some(bytes) if bytes <= self.remaining() => {}
            _ => return bad("array length exceeds the remaining header"),
        }
        let start = self.pos;
        if fixed {
            self.skip(len * min_size, elem_type == 7)?;
        } else {
            for _ in 0..len {
                if elem_type == 8 {
                    self.string(self.max_header, "array string")?;
                } else {
                    self.array_value(depth + 1)?;
                }
            }
        }
        Ok(ArrayMeta {
            elem_type,
            len,
            byte_range: start..self.pos,
        })
    }
}

fn nonneg_int(value: &MetaValue, key: &str) -> Result<u64, EngError> {
    let signed = |v: i64| u64::try_from(v).or_else(|_| bad(format!("{key} is negative")));
    match *value {
        MetaValue::U8(v) => Ok(v.into()),
        MetaValue::U16(v) => Ok(v.into()),
        MetaValue::U32(v) => Ok(v.into()),
        MetaValue::U64(v) => Ok(v),
        MetaValue::I8(v) => signed(v.into()),
        MetaValue::I16(v) => signed(v.into()),
        MetaValue::I32(v) => signed(v.into()),
        MetaValue::I64(v) => signed(v),
        _ => bad(format!("{key} is not an integer")),
    }
}

fn find<'a>(metadata: &'a [(String, MetaValue)], key: &str) -> Option<&'a MetaValue> {
    metadata.iter().find(|(k, _)| k == key).map(|(_, v)| v)
}

fn parse_tensor<R: Read + ?Sized>(src: &mut Src<'_, R>) -> Result<GgufTensor, EngError> {
    let name = src.name("tensor name", MAX_TENSOR_NAME_BYTES)?;
    let n_dims = src.u32()?;
    if !(1..=MAX_TENSOR_DIMS).contains(&n_dims) {
        return bad(format!("tensor {name} has {n_dims} dimensions"));
    }
    let mut dims = Vec::with_capacity(n_dims as usize);
    for _ in 0..n_dims {
        let dim = src.u64()?;
        if dim == 0 {
            return bad(format!("tensor {name} has a zero dimension"));
        }
        dims.push(dim);
    }
    let ggml_type = GgufType::from_id(src.u32()?)?;
    let data_offset = src.u64()?;
    let block = ggml_type.block_elems();
    if dims[0] % block != 0 {
        return bad(format!(
            "tensor {name} row length {} is not a multiple of the {} block size {block}",
            dims[0],
            ggml_type.name()
        ));
    }
    let mut rows = 1u64;
    for &dim in &dims[1..] {
        rows = rows
            .checked_mul(dim)
            .map_or_else(|| bad(format!("tensor {name} size overflow")), Ok)?;
    }
    let element_count = dims[0]
        .checked_mul(rows)
        .map_or_else(|| bad(format!("tensor {name} element count overflows")), Ok)?;
    let byte_len = (dims[0] / block)
        .checked_mul(rows)
        .and_then(|blocks| blocks.checked_mul(ggml_type.block_bytes()))
        .map_or_else(|| bad(format!("tensor {name} byte size overflows")), Ok)?;
    Ok(GgufTensor {
        name,
        dims,
        ggml_type,
        data_offset,
        abs_offset: 0,
        byte_len,
        element_count,
    })
}

fn parse_split(
    metadata: &[(String, MetaValue)],
    tensor_count: usize,
) -> Result<Option<SplitInfo>, EngError> {
    let count = find(metadata, "split.count");
    let no = find(metadata, "split.no");
    let total = find(metadata, "split.tensors.count");
    let (count, no) = match (count, no, total) {
        (None, None, None) => return Ok(None),
        (Some(count), Some(no), _) => (
            nonneg_int(count, "split.count")?,
            nonneg_int(no, "split.no")?,
        ),
        _ => return bad("split.count and split.no must appear together"),
    };
    if count == 0 || no >= count {
        return bad(format!("split.no {no} is outside split.count {count}"));
    }
    let tensors_count = total
        .map(|v| nonneg_int(v, "split.tensors.count"))
        .transpose()?;
    if let Some(total) = tensors_count {
        let own = tensor_count as u64;
        if total < own || (count == 1 && total != own) {
            return bad(format!(
                "split.tensors.count {total} disagrees with {own} tensors in this shard"
            ));
        }
    }
    Ok(Some(SplitInfo {
        count,
        no,
        tensors_count,
    }))
}

impl GgufHeader {
    /// Parses the header and tensor table from `reader`.
    ///
    /// `file_len` is the full file length. `max_header_bytes` bounds the header
    /// size, including counts and lengths declared inside it. The reader stops
    /// at the end of the tensor table and never reads tensor payloads.
    pub fn parse<R: Read + ?Sized>(
        reader: &mut R,
        file_len: u64,
        max_header_bytes: u64,
    ) -> Result<Self, EngError> {
        if max_header_bytes == 0 {
            return Err(EngError::BadRequest(
                "header byte limit must be nonzero".into(),
            ));
        }
        let mut src = Src {
            inner: reader,
            pos: 0,
            max_header: max_header_bytes,
            file_len,
            hasher: Sha256::new(),
        };
        let magic: [u8; 4] = src.array()?;
        if magic != MAGIC {
            return bad("bad magic");
        }
        let version = src.u32()?;
        if version != SUPPORTED_VERSION {
            return bad(format!("unsupported version {version}"));
        }
        let tensor_count = src.u64()?;
        let kv_count = src.u64()?;
        let fits = |count: u64, min: u64, src: &Src<'_, R>| {
            count
                .checked_mul(min)
                .is_some_and(|bytes| bytes <= src.remaining())
        };
        if !fits(tensor_count, MIN_TENSOR_BYTES, &src) {
            return bad("tensor count exceeds the remaining header");
        }
        if !fits(kv_count, MIN_KV_BYTES, &src) {
            return bad("metadata count exceeds the remaining header");
        }

        let mut metadata = Vec::with_capacity(kv_count.min(1024) as usize);
        let mut seen = HashSet::new();
        for _ in 0..kv_count {
            let key = src.name("metadata key", MAX_KEY_BYTES)?;
            if !seen.insert(key.clone()) {
                return bad(format!("duplicate metadata key {key}"));
            }
            let ty = src.u32()?;
            let value = src.value(ty, 0)?;
            metadata.push((key, value));
        }

        let mut tensors = Vec::with_capacity(tensor_count.min(4096) as usize);
        let mut names = HashSet::new();
        for _ in 0..tensor_count {
            let tensor = parse_tensor(&mut src)?;
            if !names.insert(tensor.name.clone()) {
                return bad(format!("duplicate tensor name {}", tensor.name));
            }
            tensors.push(tensor);
        }

        let header_len = src.pos;
        let header_sha256: [u8; 32] = src.hasher.finalize().into();

        let alignment = match find(&metadata, "general.alignment") {
            None => DEFAULT_ALIGNMENT,
            Some(MetaValue::U32(a)) if a.is_power_of_two() => u64::from(*a),
            Some(_) => return bad("general.alignment is not a nonzero power-of-two u32"),
        };
        let data_start = header_len
            .checked_add(alignment - 1)
            .map(|v| v / alignment * alignment)
            .map_or_else(|| bad("data start overflows"), Ok)?;
        if data_start > file_len {
            return bad("the data section starts beyond the end of the file");
        }
        for tensor in &mut tensors {
            if tensor.data_offset % alignment != 0 {
                return bad(format!(
                    "tensor {} offset {} is not aligned to {alignment}",
                    tensor.name, tensor.data_offset
                ));
            }
            let start = data_start.checked_add(tensor.data_offset).map_or_else(
                || bad(format!("tensor {} offset overflows", tensor.name)),
                Ok,
            )?;
            let end = start.checked_add(tensor.byte_len).map_or_else(
                || bad(format!("tensor {} range overflows", tensor.name)),
                Ok,
            )?;
            if end > file_len {
                return bad(format!(
                    "tensor {} extends beyond the end of the file",
                    tensor.name
                ));
            }
            tensor.abs_offset = start;
        }
        let mut order: Vec<usize> = (0..tensors.len()).collect();
        order.sort_by_key(|&i| tensors[i].abs_offset);
        for pair in order.windows(2) {
            let (a, b) = (&tensors[pair[0]], &tensors[pair[1]]);
            if a.abs_offset + a.byte_len > b.abs_offset {
                return bad(format!("tensors {} and {} overlap", a.name, b.name));
            }
        }

        let split = parse_split(&metadata, tensors.len())?;
        Ok(Self {
            version,
            alignment,
            data_start,
            header_len,
            file_len,
            header_sha256,
            metadata,
            tensors,
            split,
        })
    }

    /// Parses a header from a byte slice that holds at least the header bytes.
    pub fn from_bytes(
        bytes: &[u8],
        file_len: u64,
        max_header_bytes: u64,
    ) -> Result<Self, EngError> {
        let mut reader = bytes;
        Self::parse(&mut reader, file_len, max_header_bytes)
    }

    pub fn version(&self) -> u32 {
        self.version
    }
    pub fn alignment(&self) -> u64 {
        self.alignment
    }
    /// File offset where the data section starts.
    pub fn data_start(&self) -> u64 {
        self.data_start
    }
    /// Bytes from the file start to the end of the tensor table, before padding.
    pub fn header_len(&self) -> u64 {
        self.header_len
    }
    pub fn file_len(&self) -> u64 {
        self.file_len
    }
    /// SHA-256 over the first `header_len` bytes of the file.
    pub fn header_sha256(&self) -> [u8; 32] {
        self.header_sha256
    }
    pub fn metadata(&self) -> &[(String, MetaValue)] {
        &self.metadata
    }
    pub fn meta(&self, key: &str) -> Option<&MetaValue> {
        find(&self.metadata, key)
    }
    pub fn tensors(&self) -> &[GgufTensor] {
        &self.tensors
    }
    pub fn tensor(&self, name: &str) -> Option<&GgufTensor> {
        self.tensors.iter().find(|t| t.name == name)
    }
    pub fn split(&self) -> Option<SplitInfo> {
        self.split
    }

    /// Checks the header hash against a recorded lowercase or uppercase hex digest.
    pub fn check_header_sha256(&self, expected_hex: &str) -> Result<(), EngError> {
        let expected = parse_hex_digest(expected_hex)?;
        if expected != self.header_sha256 {
            return bad(format!(
                "header sha256 {} differs from the recorded {}",
                sha256_hex(&self.header_sha256),
                expected_hex.to_ascii_lowercase()
            ));
        }
        Ok(())
    }
}

/// Checks that parsed shard headers form one complete, consistent split set.
///
/// Every header needs split metadata. The shards must agree on `split.count`,
/// cover each index once, agree on any `split.tensors.count`, and share no
/// tensor name. The shard tensor counts must add up to that total.
pub fn check_split_set(headers: &[&GgufHeader]) -> Result<(), EngError> {
    if headers.is_empty() {
        return bad("split set is empty");
    }
    let infos: Vec<SplitInfo> = headers
        .iter()
        .map(|h| {
            h.split()
                .map_or_else(|| bad("shard has no split metadata"), Ok)
        })
        .collect::<Result<_, _>>()?;
    let count = infos[0].count;
    if infos.iter().any(|i| i.count != count) {
        return bad("shards disagree on split.count");
    }
    if headers.len() as u64 != count {
        return bad(format!(
            "split set has {} shards but split.count is {count}",
            headers.len()
        ));
    }
    let mut numbers: Vec<u64> = infos.iter().map(|i| i.no).collect();
    numbers.sort_unstable();
    if numbers.iter().enumerate().any(|(i, &n)| n != i as u64) {
        return bad("split.no values do not cover each shard once");
    }
    let totals: HashSet<Option<u64>> = infos.iter().map(|i| i.tensors_count).collect();
    if totals.len() != 1 {
        return bad("shards disagree on split.tensors.count");
    }
    let mut names = HashSet::new();
    let mut sum = 0u64;
    for header in headers {
        for tensor in header.tensors() {
            if !names.insert(tensor.name()) {
                return bad(format!(
                    "tensor {} appears in more than one shard",
                    tensor.name()
                ));
            }
            sum += 1;
        }
    }
    if let Some(Some(total)) = totals.into_iter().next()
        && total != sum
    {
        return bad(format!(
            "split.tensors.count {total} differs from {sum} listed tensors"
        ));
    }
    Ok(())
}

/// Formats a digest as 64 lowercase hex digits.
pub fn sha256_hex(digest: &[u8; 32]) -> String {
    use std::fmt::Write;
    digest.iter().fold(String::with_capacity(64), |mut out, b| {
        let _ = write!(out, "{b:02x}");
        out
    })
}

fn parse_hex_digest(hex: &str) -> Result<[u8; 32], EngError> {
    let bytes = hex.as_bytes();
    if bytes.len() != 64 || !bytes.iter().all(u8::is_ascii_hexdigit) {
        return bad("recorded sha256 is not 64 hex digits");
    }
    let nibble = |c: u8| (c as char).to_digit(16).unwrap_or(0) as u8;
    let mut out = [0u8; 32];
    for (i, pair) in bytes.chunks(2).enumerate() {
        out[i] = nibble(pair[0]) << 4 | nibble(pair[1]);
    }
    Ok(out)
}

/// Streams `reader` through SHA-256. Returns the digest and the byte count.
pub fn sha256_stream<R: Read + ?Sized>(reader: &mut R) -> Result<([u8; 32], u64), EngError> {
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; HASH_CHUNK];
    let mut total = 0u64;
    loop {
        match reader.read(&mut buf) {
            Ok(0) => break,
            Ok(n) => {
                hasher.update(&buf[..n]);
                total += n as u64;
            }
            Err(error) if error.kind() == ErrorKind::Interrupted => {}
            Err(error) => return bad(format!("hash read: {error}")),
        }
    }
    Ok((hasher.finalize().into(), total))
}

/// Streams `reader` and checks its length and SHA-256 against recorded values.
pub fn verify_sha256<R: Read + ?Sized>(
    reader: &mut R,
    expected_len: u64,
    expected_hex: &str,
) -> Result<(), EngError> {
    let expected = parse_hex_digest(expected_hex)?;
    let (digest, len) = sha256_stream(reader)?;
    if len != expected_len {
        return bad(format!("hashed {len} bytes but expected {expected_len}"));
    }
    if digest != expected {
        return bad(format!(
            "sha256 {} differs from the recorded {}",
            sha256_hex(&digest),
            expected_hex.to_ascii_lowercase()
        ));
    }
    Ok(())
}
