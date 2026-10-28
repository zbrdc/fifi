//! The audit against the real Flint 4.0 pack. Ignored by default: it needs the 20 GB pack.
//!
//! Run with `FLINT_EXL3_PACK=<dir> cargo test --lib real_flint -- --ignored --nocapture`. It
//! reads headers and the small scale, marker, norm, sink and router ranges (about 230 MB in
//! all) and a sample of 17 trellis payloads; it never reads a whole shard. The expected counts
//! come from an independent Python header and range scan of the same revision.

use std::{
    fs::File,
    io::Read,
    path::{Path, PathBuf},
};

use sha2::{Digest, Sha256};

use super::{audit_published, fixture::PINNED_CONFIG_SHA256};
use crate::weights::sha256_hex;

/// `model.safetensors.index.json` of revision e7ab11242b0b30a6063f3559dda75fed6b9e12c6.
const INDEX_SHA256: &str = "07502a18eee9499d639f170bc4979697ff010f8d101b71c5e2220b051b53e127";
const SHARD_LENGTHS: [(&str, u64); 3] = [
    ("model-00001-of-00003.safetensors", 8_243_454_378),
    ("model-00002-of-00003.safetensors", 8_412_959_676),
    ("model-00003-of-00003.safetensors", 3_978_316_007),
];

/// The caps `Checkpoint::open` applies to these two files.
const CONFIG_LIMIT: u64 = 1024 * 1024;
const INDEX_LIMIT: u64 = 16 * 1024 * 1024;

/// Streams at most `limit` bytes of `path` into the hash; a longer file fails before the audit
/// opens the pack, as `Checkpoint::open` would refuse it.
fn bounded_sha256(path: &Path, limit: u64) -> String {
    let file = File::open(path).unwrap();
    bounded_sha256_of(file, limit, &path.display().to_string())
}

/// Reads at most `limit + 1` bytes from `reader`, so an endless or huge source stops early.
fn bounded_sha256_of(reader: impl Read, limit: u64, label: &str) -> String {
    let mut hasher = Sha256::new();
    let copied = std::io::copy(&mut reader.take(limit + 1), &mut hasher).unwrap();
    assert!(copied <= limit, "{label} exceeds {limit} bytes");
    let digest: [u8; 32] = hasher.finalize().into();
    sha256_hex(&digest)
}

/// An endless source that fails the test if asked for more than `allowed` bytes in total.
struct Tripwire {
    served: u64,
    allowed: u64,
}

impl Read for Tripwire {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        self.served += buf.len() as u64;
        assert!(self.served <= self.allowed, "tripwire: read past the bound");
        buf.fill(7);
        Ok(buf.len())
    }
}

#[test]
#[should_panic(expected = "endless exceeds 999 bytes")]
fn an_endless_source_is_cut_one_byte_past_the_limit() {
    let source = Tripwire {
        served: 0,
        allowed: 1000,
    };
    bounded_sha256_of(source, 999, "endless");
}

#[test]
fn the_bounded_hash_equals_a_full_hash_at_the_limit() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("metadata");
    std::fs::write(&path, vec![7u8; 1000]).unwrap();
    let full: [u8; 32] = Sha256::digest(vec![7u8; 1000]).into();
    assert_eq!(bounded_sha256(&path, 1000), sha256_hex(&full));
}

#[test]
#[should_panic(expected = "exceeds 999 bytes")]
fn the_bounded_hash_refuses_a_file_beyond_the_limit() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("metadata");
    std::fs::write(&path, vec![7u8; 1000]).unwrap();
    bounded_sha256(&path, 999);
}

#[test]
#[ignore = "reads the real pack: set FLINT_EXL3_PACK to the e7ab1124 directory"]
fn real_flint_4_0_pack_passes_the_audit() {
    let dir = PathBuf::from(std::env::var_os("FLINT_EXL3_PACK").expect("FLINT_EXL3_PACK"));
    assert_eq!(
        bounded_sha256(&dir.join("config.json"), CONFIG_LIMIT),
        PINNED_CONFIG_SHA256
    );
    assert_eq!(
        bounded_sha256(&dir.join("model.safetensors.index.json"), INDEX_LIMIT),
        INDEX_SHA256
    );
    let audit = audit_published(&dir).unwrap();
    eprintln!("{audit:#?}");

    assert_eq!(audit.tensors, 146_054);
    assert_eq!(audit.matrices.values().sum::<usize>(), 36_440);
    for role in ["experts.gate_proj", "experts.up_proj", "experts.down_proj"] {
        assert_eq!(audit.matrices[role], 47 * 256, "{role}");
    }
    let rates: Vec<_> = audit.bits_twice.iter().map(|(b, n)| (*b, *n)).collect();
    assert_eq!(rates, [(8, 36_439), (10, 1)]);
    for (name, length) in SHARD_LENGTHS {
        let shard = audit.shards.iter().find(|s| s.name == name).unwrap();
        assert_eq!(shard.file_len, length, "{name}");
        assert_eq!(8 + shard.header_len + shard.payload, length, "{name}");
    }
    assert_eq!(
        audit.shards.iter().map(|s| s.payload).sum::<u64>(),
        20_617_542_816
    );

    // No scale vector is entirely zero; zero and subnormal scale entries do occur.
    assert!(audit.dead_vectors.is_empty(), "{:?}", audit.dead_vectors);
    let up = &audit.scales["experts.up_proj.suh"];
    assert_eq!(
        (up.elements, up.subnormals, up.zeros),
        (24_641_536, 24_640_971, 0)
    );
    assert_eq!(up.subnormal_vectors, 11_871);
    let up_svh = &audit.scales["experts.up_proj.svh"];
    assert_eq!((up_svh.zeros, up_svh.subnormals), (386, 4));
    let gate_svh = &audit.scales["experts.gate_proj.svh"];
    assert_eq!((gate_svh.zeros, gate_svh.subnormals), (159, 79));
    let down_suh = &audit.scales["experts.down_proj.suh"];
    assert_eq!((down_suh.zeros, down_suh.subnormals), (4, 79));
    let router = &audit.scales["router"];
    assert_eq!(
        (router.elements, router.zeros, router.subnormals),
        (24_641_536, 313, 676_677)
    );
    // The sample: 4 attention, 1 dense, and 3 shared + 6 + 3 expert matrices.
    assert_eq!(audit.admitted.len(), 17);
    let shape_of = |prefix: &str| {
        let found = audit.admitted.iter().find(|a| a.prefix == prefix);
        let found = found.unwrap_or_else(|| panic!("{prefix} not admitted"));
        (found.shape, found.bits_twice)
    };
    assert_eq!(
        shape_of("model.layers.2.mlp.experts.0.up_proj"),
        ([2048, 512], 8)
    );
    assert_eq!(
        shape_of("model.layers.48.mlp.experts.255.down_proj"),
        ([512, 2048], 8)
    );
    assert_eq!(
        shape_of("model.layers.2.mlp.shared_expert.down_proj"),
        ([1024, 2048], 8)
    );
    assert_eq!(shape_of("model.layers.0.mlp.down_proj"), ([6144, 2048], 8));
    assert_eq!(
        shape_of("model.layers.0.self_attn.q_proj"),
        ([2048, 4096], 8)
    );
}
