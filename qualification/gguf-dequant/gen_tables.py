#!/usr/bin/env python3
"""Generate engine/src/weights/gguf_tables.rs from the pinned ggml-common.h.

Usage: gen_tables.py <ggml-common.h> <out.rs>
The input is the file at llama.cpp commit fb4b2737a808a3fb7c2117a498f43815dc9be53e.
The script emits only the constant data. All decoding logic is written separately.
"""
import hashlib, re, struct, sys

src = open(sys.argv[1], encoding="utf-8").read()
out_path = sys.argv[2]
src_hash = hashlib.sha256(src.encode()).hexdigest()

TABLES = [
    ("uint8_t", "kmask_iq2xs", "KMASK_IQ2XS", "u8", "<B"),
    ("uint8_t", "ksigns_iq2xs", "KSIGNS_IQ2XS", "u8", "<B"),
    ("uint64_t", "iq2xxs_grid", "IQ2XXS_GRID", "u64", "<Q"),
    ("uint64_t", "iq2xs_grid", "IQ2XS_GRID", "u64", "<Q"),
    ("uint64_t", "iq2s_grid", "IQ2S_GRID", "u64", "<Q"),
    ("uint32_t", "iq3xxs_grid", "IQ3XXS_GRID", "u32", "<I"),
    ("uint32_t", "iq3s_grid", "IQ3S_GRID", "u32", "<I"),
    ("int8_t", "kvalues_iq4nl", "KVALUES_IQ4NL", "i8", "<b"),
]
HEADER = '''//! Constant lookup tables for the GGUF lattice quantization types.
//!
//! These eight tables are format data. They are copied from
//! `ggml/src/ggml-common.h` at llama.cpp commit
//! `fb4b2737a808a3fb7c2117a498f43815dc9be53e`
//! (file sha256 `{src_hash}`).
//! Each grid entry packs eight (or four) unsigned byte weights in little-endian order.
//! The dequantization logic that uses them is original code in `gguf_dequant.rs`.
//! The script `qualification/gguf-dequant/gen_tables.py` generated this file.
//!
//! llama.cpp and ggml are distributed under the MIT license:
//!
//! MIT License
//!
//! Copyright (c) 2023-2026 The ggml authors
//!
//! Permission is hereby granted, free of charge, to any person obtaining a copy
//! of this software and associated documentation files (the "Software"), to deal
//! in the Software without restriction, including without limitation the rights
//! to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
//! copies of the Software, and to permit persons to whom the Software is
//! furnished to do so, subject to the following conditions:
//!
//! The above copyright notice and this permission notice shall be included in all
//! copies or substantial portions of the Software.
//!
//! THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//! IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//! FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//! AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//! LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
//! OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
//! SOFTWARE.

'''
body = HEADER.format(src_hash=src_hash)
hashes = []
for ctype, cname, rname, rtype, fmt in TABLES:
    m = re.search(r"GGML_TABLE_BEGIN\(%s, %s, (\d+)\)(.*?)GGML_TABLE_END\(\)" % (ctype, cname), src, re.S)
    n = int(m.group(1))
    vals = [int(t, 0) for t in re.findall(r"-?0x[0-9a-fA-F]+|-?\d+", m.group(2))]
    assert len(vals) == n, (cname, len(vals), n)
    per = {"u8": 16, "i8": 16, "u32": 8, "u64": 4}[rtype]
    width = {"u8": 3, "i8": 4, "u32": 10, "u64": 18}[rtype]
    body += "pub(crate) const %s: [%s; %d] = [\n" % (rname, rtype, n)
    for i in range(0, n, per):
        row = vals[i:i + per]
        if rtype in ("u32", "u64"):
            s = ", ".join("0x%0*x" % (8 if rtype == "u32" else 16, v) for v in row)
        else:
            s = ", ".join(str(v) for v in row)
        body += "    " + s + ",\n"
    body += "];\n\n"
    raw = b"".join(struct.pack(fmt, v) for v in vals)
    hashes.append((rname, n, hashlib.sha256(raw).hexdigest()))
body += """#[cfg(test)]
mod tests {
    use super::*;
    use sha2::{Digest, Sha256};

    macro_rules! le_hex {
        ($table:expr) => {{
            let mut hasher = Sha256::new();
            for value in $table.iter() {
                hasher.update(value.to_le_bytes());
            }
            hasher
                .finalize()
                .iter()
                .map(|byte| format!("{byte:02x}"))
                .collect::<String>()
        }};
    }
"""
for rname, n, h in hashes:
    body += """
    /// The little-endian bytes of the table match the pinned upstream file.
    #[test]
    fn %s_matches_pinned_sha256() {
        assert_eq!(%s.len(), %d);
        assert_eq!(
            le_hex!(%s),
            "%s"
        );
    }
""" % (rname.lower(), rname, n, rname, h)
body += "}\n"
open(out_path, "w").write(body)
for h in hashes:
    print(*h)
print("source sha256", src_hash)
# Note: run `cargo fmt --all` in engine/ after generating; the committed file is rustfmt-formatted.
