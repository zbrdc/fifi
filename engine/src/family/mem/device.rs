//! The device policy table: every headroom, margin, floor and charge a plan keeps back on a card
//! (or on unified memory), once.
//!
//! A family supplies the byte costs of its own objects and reads what it keeps back from here.
//! Before there was one table, the same ideas were literals in a dozen files, and two stages of one
//! Qwen3.6 start disagreed about the headroom (the ladder promised slots against 256 MiB, the bank
//! admission charged 1 GiB). Each field names who reads it.

const MIB: u64 = 1 << 20;
const GIB: u64 = 1 << 30;

/// The device policy numbers. Read the fields through [`DEVICE`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct DevicePolicy {
    /// The flat per-part charge of chunked attention; equal to the kernel cap
    /// `CPF_ATTN_PART_CAP`, which a const assertion in the kernel plan checks.
    pub(crate) part_charge: u64,
    /// What the fixed CUDA admission keeps below measured free VRAM before any charge
    /// (`budget = free - margin`). Read by the Qwen3.6 fixed admission and by every estimate that
    /// mirrors it.
    pub(crate) admission_margin: u64,
    /// The spare the fixed admission leaves after every charge (the smallest chunk it admits).
    pub(crate) spare_min: u64,
    /// The measurement gap an estimate cannot close (probe free VRAM against the serving
    /// process's own post-load reading). Slack inside it is borderline, not a verdict.
    pub(crate) uncertainty_band: u64,
    /// What a plan adds for the allocations the serving process makes after it measured free
    /// memory (the prefill arena, the decode arena, eager module workspace) and, for an estimate
    /// that runs in another process, the serving context beyond a fresh probe's. Measured on the
    /// Qwen3.6 Q4_K_M server of the 2026-11-03 qualification (n=1); charged as one floor, never a
    /// precise term. Flash-Next borrows it for its own post-load allocations.
    pub(crate) runtime_overhead_floor: u64,
    /// The device bytes a Qwen3.6 weight load takes beyond the tensor bytes it plans: the kernel
    /// modules the start loads eagerly and the context's own workspace. Measured as the free bytes
    /// before the load less after it, less the planned tensor bytes, on the UD-IQ4_XS file on a
    /// 3090 (n=1): 269,850,112, rounded up to a whole MiB.
    pub(crate) load_unplanned: u64,
    /// The device bytes a legacy (non-fixed) admission keeps free: the Qwen3.6 admission
    /// headroom, its free-after-load floor, and the tiered bank admission's reserve.
    pub(crate) load_headroom: u64,
    /// Device bytes every Flint load keeps free beside its weights and named charges.
    pub(crate) flint_headroom: u64,
    /// Device bytes the Flint bank leaves free beyond its declared reserve.
    pub(crate) flint_bank_margin: u64,
    /// Device bytes the Bolt load keeps free beside its weights, KV cache and scratch.
    pub(crate) bolt_headroom: u64,
    /// VRAM kept free beyond the Qwen3.6 sparse planes when admitting them.
    pub(crate) sparse_headroom: u64,
    /// VRAM the expert tier leaves free after its resident experts and the dense weight cache.
    pub(crate) expert_tier_headroom: u64,
    /// Device bytes the Raon-1 load keeps free beside the weights, the scratch and the cache.
    pub(crate) raon_headroom: u64,
    /// Inkling controller admission keeps this for lazy staging and allocator rounding.
    pub(crate) inkling_controller_margin: u64,
    /// Free bytes the filesystem of an on-disk cache keeps beyond an entry's bytes: a write that
    /// would leave less is skipped, so a cache never takes the last bytes of a shared disk.
    pub(crate) disk_free_reserve: u64,
    /// Device bytes a step keeps free when it puts a token embedding table on the device for
    /// the token tail: with less left after the table, the table stays on the host.
    pub(crate) embedding_table_reserve: u64,
    /// The room family reserves leave for memory outside their tensor estimates on unified
    /// memory (the Mac prefill reserve and its derivation).
    pub(crate) unified_pressure_margin: u64,
}

/// The table.
pub(crate) const DEVICE: DevicePolicy = DevicePolicy {
    part_charge: 64 * MIB,
    admission_margin: 256 * MIB,
    spare_min: 64 * MIB,
    uncertainty_band: 128 * MIB,
    runtime_overhead_floor: 384 * MIB,
    load_unplanned: 258 * MIB,
    load_headroom: GIB,
    flint_headroom: 512 * MIB,
    flint_bank_margin: 160 * MIB,
    bolt_headroom: 512 * MIB,
    sparse_headroom: 2 * GIB,
    expert_tier_headroom: 3 * GIB,
    raon_headroom: 768 * MIB,
    inkling_controller_margin: 32 * MIB,
    disk_free_reserve: 512 * MIB,
    embedding_table_reserve: 256 * MIB,
    unified_pressure_margin: GIB,
};

#[cfg(test)]
mod tests {
    use super::*;

    /// The numbers every family measured and shipped with, byte for byte: moving them into the
    /// table changes no figure.
    #[test]
    fn the_table_holds_the_shipped_numbers() {
        assert_eq!(DEVICE.part_charge, 67_108_864);
        assert_eq!(DEVICE.admission_margin, 268_435_456);
        assert_eq!(DEVICE.spare_min, 67_108_864);
        assert_eq!(DEVICE.uncertainty_band, 134_217_728);
        assert_eq!(DEVICE.runtime_overhead_floor, 402_653_184);
        assert_eq!(DEVICE.load_unplanned, 270_532_608);
        assert_eq!(DEVICE.load_headroom, 1_073_741_824);
        assert_eq!(DEVICE.flint_headroom, 536_870_912);
        assert_eq!(DEVICE.flint_bank_margin, 167_772_160);
        assert_eq!(DEVICE.bolt_headroom, 536_870_912);
        assert_eq!(DEVICE.sparse_headroom, 2_147_483_648);
        assert_eq!(DEVICE.expert_tier_headroom, 3_221_225_472);
        assert_eq!(DEVICE.raon_headroom, 805_306_368);
        assert_eq!(DEVICE.inkling_controller_margin, 33_554_432);
        assert_eq!(DEVICE.disk_free_reserve, 536_870_912);
        assert_eq!(DEVICE.embedding_table_reserve, 268_435_456);
        assert_eq!(DEVICE.unified_pressure_margin, 1_073_741_824);
    }

    fn rust_files(dir: &std::path::Path, found: &mut Vec<std::path::PathBuf>) {
        for entry in std::fs::read_dir(dir).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() {
                rust_files(&path, found);
            } else if path.extension().is_some_and(|extension| extension == "rs") {
                found.push(path);
            }
        }
    }

    /// A headroom, margin or reserve constant in a family file is a second copy of this table.
    /// The scan refuses a `const` whose name says it keeps device bytes back and whose value is a
    /// bare shift literal.
    #[test]
    fn no_family_declares_a_device_headroom_literal() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
        let mut files = Vec::new();
        rust_files(&root, &mut files);
        let mut found = Vec::new();
        for file in files {
            let relative = file.strip_prefix(&root).unwrap().to_path_buf();
            if relative.starts_with("family/mem") {
                continue;
            }
            let text = std::fs::read_to_string(&file).unwrap();
            for (number, line) in text.lines().enumerate() {
                let line = line.trim();
                let declares = line.starts_with("pub const ")
                    || line.starts_with("pub(crate) const ")
                    || line.starts_with("pub(super) const ")
                    || line.starts_with("const ");
                let upper = line.to_ascii_uppercase();
                let named = [
                    "HEADROOM",
                    "MARGIN",
                    "RESERVE",
                    "OVERHEAD_FLOOR",
                    "SPARE_MIN",
                    "FREE_AFTER_LOAD",
                ]
                .iter()
                .any(|word| {
                    upper
                        .split('=')
                        .next()
                        .is_some_and(|name| name.contains(word))
                });
                let literal = line.split('=').nth(1).is_some_and(|value| {
                    let value = value.trim().trim_end_matches(';');
                    value.contains("<<")
                        && value
                            .chars()
                            .all(|c| c.is_ascii_digit() || " <_*usizeu64".contains(c))
                });
                if declares && named && literal {
                    found.push(format!("{}:{}: {line}", relative.display(), number + 1));
                }
            }
        }
        assert!(found.is_empty(), "{found:#?}");
    }
}
