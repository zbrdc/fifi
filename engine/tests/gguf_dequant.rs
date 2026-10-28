//! Bit-exact GGUF dequantization against fixtures from the pinned ggml CPU oracle.
//! Fixture provenance is in `tests/fixtures/gguf-dequant/manifest.json`.

use std::path::PathBuf;

use inference_engine::{
    EngError,
    weights::{GgufType, dequantize, dequantize_into, sha256_hex},
};
use serde_json::Value;
use sha2::{Digest, Sha256};

fn dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/gguf-dequant")
}

fn sha(bytes: &[u8]) -> String {
    sha256_hex(&Sha256::digest(bytes).into())
}

fn manifest() -> Value {
    serde_json::from_slice(&std::fs::read(dir().join("manifest.json")).unwrap()).unwrap()
}

fn ty_of(variant: &str) -> GgufType {
    match variant {
        "Q20" => GgufType::Q20,
        "Q4K" => GgufType::Q4K,
        "Q5K" => GgufType::Q5K,
        "Q6K" => GgufType::Q6K,
        "Q80" => GgufType::Q80,
        "Iq2Xxs" => GgufType::Iq2Xxs,
        "Iq2Xs" => GgufType::Iq2Xs,
        "Iq2S" => GgufType::Iq2S,
        "Iq3Xxs" => GgufType::Iq3Xxs,
        "Iq3S" => GgufType::Iq3S,
        "Iq4Nl" => GgufType::Iq4Nl,
        "Iq4Xs" => GgufType::Iq4Xs,
        "F32" => GgufType::F32,
        "F16" => GgufType::F16,
        "Bf16" => GgufType::Bf16,
        other => panic!("unknown variant {other}"),
    }
}

fn words(bytes: &[u8]) -> Vec<u32> {
    let (chunks, rest) = bytes.as_chunks::<4>();
    assert!(rest.is_empty());
    chunks.iter().map(|c| u32::from_le_bytes(*c)).collect()
}

/// Exact bits on aarch64. Elsewhere only one difference is accepted: the x86
/// default NaN (0xffc00000) where the arm64 oracle produced 0x7fc00000.
fn same(got: u32, want: u32) -> bool {
    if got == want {
        return true;
    }
    !cfg!(target_arch = "aarch64") && got == 0xffc0_0000 && want == 0x7fc0_0000
}

/// Synthetic, extra K-scale and real-tensor entries of the manifest.
fn entries(m: &Value) -> Vec<&Value> {
    ["types", "extra", "real"]
        .iter()
        .flat_map(|k| m[*k].as_array().unwrap().iter())
        .collect()
}

#[test]
fn manifest_lists_all_fourteen_types_and_hashes_match() {
    let m = manifest();
    assert_eq!(
        m["oracle_commit"],
        "fb4b2737a808a3fb7c2117a498f43815dc9be53e"
    );
    let types = m["types"].as_array().unwrap();
    assert_eq!(types.len(), 15);
    assert_eq!(m["extra"].as_array().unwrap().len(), 2);
    let real = m["real"].as_array().unwrap();
    assert_eq!(real.len(), 15, "one real-tensor entry per type");
    for r in real {
        assert!(r["tensor"].as_str().is_some_and(|t| !t.is_empty()));
        assert_eq!(r["shard_sha256"].as_str().unwrap().len(), 64);
        assert!(r["file_offset"].as_u64().unwrap() >= r["data_start"].as_u64().unwrap());
    }
    for t in entries(&m) {
        let name = t["name"].as_str().unwrap();
        let blocks = std::fs::read(dir().join(format!("{name}.blocks"))).unwrap();
        let out = std::fs::read(dir().join(format!("{name}.f32"))).unwrap();
        assert_eq!(sha(&blocks), t["input_sha256"].as_str().unwrap(), "{name}");
        assert_eq!(sha(&out), t["output_sha256"].as_str().unwrap(), "{name}");
        let ty = ty_of(t["variant"].as_str().unwrap());
        assert_eq!(ty.block_bytes(), t["block_bytes"].as_u64().unwrap());
        assert_eq!(ty.block_elems(), t["block_elems"].as_u64().unwrap());
    }
}

#[test]
fn every_type_matches_oracle_bit_for_bit() {
    let m = manifest();
    for t in entries(&m) {
        let name = t["name"].as_str().unwrap();
        let ty = ty_of(t["variant"].as_str().unwrap());
        let blocks = std::fs::read(dir().join(format!("{name}.blocks"))).unwrap();
        let want = words(&std::fs::read(dir().join(format!("{name}.f32"))).unwrap());
        let got = dequantize(ty, &blocks).unwrap_or_else(|e| panic!("{name}: {e:?}"));
        assert_eq!(got.len(), want.len(), "{name} length");
        let bad: Vec<usize> = (0..want.len())
            .filter(|&i| !same(got[i].to_bits(), want[i]))
            .collect();
        assert!(
            bad.is_empty(),
            "{name}: {} differing elements, first {:?}",
            bad.len(),
            bad.first().map(|&i| (i, got[i].to_bits(), want[i]))
        );
        let mut dst = vec![f32::from_bits(0x1234_5678); want.len()];
        dequantize_into(ty, &blocks, &mut dst).unwrap();
        assert!(
            (0..want.len()).all(|i| dst[i].to_bits() == got[i].to_bits()),
            "{name}: dequantize_into differs from dequantize"
        );
    }
}

const ALL: [GgufType; 15] = [
    GgufType::Q20,
    GgufType::Q4K,
    GgufType::Q5K,
    GgufType::Q6K,
    GgufType::Q80,
    GgufType::Iq2Xxs,
    GgufType::Iq2Xs,
    GgufType::Iq2S,
    GgufType::Iq3Xxs,
    GgufType::Iq3S,
    GgufType::Iq4Nl,
    GgufType::Iq4Xs,
    GgufType::F32,
    GgufType::F16,
    GgufType::Bf16,
];

#[test]
fn scale_fixtures_cover_every_six_bit_scale_and_min() {
    for (name, ty) in [
        ("q4_k_scales", GgufType::Q4K),
        ("q5_k_scales", GgufType::Q5K),
    ] {
        let blocks = std::fs::read(dir().join(format!("{name}.blocks"))).unwrap();
        let mut seen = [[[false; 64]; 8]; 2];
        for b in blocks.chunks_exact(ty.block_bytes() as usize) {
            let q = &b[4..16];
            for j in 0..8 {
                let (sc, mn) = if j < 4 {
                    (q[j] & 63, q[j + 4] & 63)
                } else {
                    (
                        (q[j + 4] & 15) | ((q[j - 4] >> 6) << 4),
                        (q[j + 4] >> 4) | ((q[j] >> 6) << 4),
                    )
                };
                seen[0][j][usize::from(sc)] = true;
                seen[1][j][usize::from(mn)] = true;
            }
        }
        assert!(seen.iter().flatten().flatten().all(|&x| x), "{name}");
    }
}

#[test]
fn input_length_not_a_block_multiple_is_an_error() {
    for ty in ALL {
        let bb = ty.block_bytes() as usize;
        let src = vec![0u8; bb + 1];
        let mut dst = vec![0f32; ty.block_elems() as usize];
        assert!(matches!(
            dequantize_into(ty, &src, &mut dst),
            Err(EngError::Checkpoint(_))
        ));
        assert!(dequantize(ty, &src).is_err(), "{ty:?}");
        assert!(dequantize(ty, &src[..bb - 1]).is_err(), "{ty:?}");
    }
}

#[test]
fn output_length_mismatch_is_an_error() {
    for ty in ALL {
        let src = vec![0u8; 2 * ty.block_bytes() as usize];
        let e = ty.block_elems() as usize;
        for n in [0, 2 * e - 1, 2 * e + 1, e] {
            let mut dst = vec![0f32; n];
            assert!(
                matches!(
                    dequantize_into(ty, &src, &mut dst),
                    Err(EngError::BadRequest(_))
                ),
                "{ty:?} {n}"
            );
        }
        let mut ok = vec![0f32; 2 * e];
        dequantize_into(ty, &src, &mut ok).unwrap();
    }
}

#[test]
fn empty_input_gives_empty_output() {
    for ty in ALL {
        assert!(dequantize(ty, &[]).unwrap().is_empty(), "{ty:?}");
        dequantize_into(ty, &[], &mut []).unwrap();
    }
}

#[test]
fn unsupported_type_is_an_error() {
    let mut dst = vec![0f32; 32];
    for ty in [GgufType::Q40, GgufType::Q2K, GgufType::Iq1S] {
        let src = vec![0u8; ty.block_bytes() as usize];
        assert!(matches!(
            dequantize_into(ty, &src, &mut dst),
            Err(EngError::Checkpoint(_))
        ));
    }
}

#[test]
fn scalar_formats_decode_known_values() {
    let v = dequantize(GgufType::F16, &[0x00, 0x3c, 0x00, 0xc0]).unwrap();
    assert_eq!(v, vec![1.0, -2.0]);
    let v = dequantize(GgufType::Bf16, &[0x80, 0x3f]).unwrap();
    assert_eq!(v, vec![1.0]);
    let v = dequantize(GgufType::F32, &1.5f32.to_le_bytes()).unwrap();
    assert_eq!(v, vec![1.5]);
}

#[test]
fn q5_0_decodes_a_hand_computed_block() {
    // d = 2.0; qh bits 0, 16 and 31 set; qs[j] = j | ((15 - j) << 4).
    let mut block = vec![0x00, 0x40, 0x01, 0x00, 0x01, 0x80];
    block.extend((0u8..16).map(|j| j | ((15 - j) << 4)));
    let v = dequantize(GgufType::Q50, &block).unwrap();
    let expect: [f32; 32] = [
        0.0, -30.0, -28.0, -26.0, -24.0, -22.0, -20.0, -18.0, -16.0, -14.0, -12.0, -10.0, -8.0,
        -6.0, -4.0, -2.0, 30.0, -4.0, -6.0, -8.0, -10.0, -12.0, -14.0, -16.0, -18.0, -20.0, -22.0,
        -24.0, -26.0, -28.0, -30.0, 0.0,
    ];
    let got: Vec<u32> = v.iter().map(|x| x.to_bits()).collect();
    let want: Vec<u32> = expect.iter().map(|x| x.to_bits()).collect();
    assert_eq!(got, want);
}

#[test]
fn q5_0_all_zero_block_decodes_to_negative_zero_bits() {
    // -16 * +0.0 is -0.0: a zero padding block is not +0.0 in a float reference.
    let v = dequantize(GgufType::Q50, &[0u8; 22]).unwrap();
    assert_eq!(v.len(), 32);
    assert!(v.iter().all(|x| x.to_bits() == 0x8000_0000), "{v:?}");
}
