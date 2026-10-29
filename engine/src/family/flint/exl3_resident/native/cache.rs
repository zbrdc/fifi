//! CPU cache policy used by the model-owned native pinned store.
//! This proves no VRAM freed, physical pinning, or GPU completion.

use crate::EngError;

pub(super) const EXPERTS: usize = 256;
pub(super) const SLOTS: usize = 224;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct ByteRange {
    pub(super) offset: usize,
    pub(super) bytes: usize,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct LayerLayout {
    source: ByteRange,
    slots: ByteRange,
    stride: usize,
}

pub(super) struct HostLayout {
    layers: Vec<LayerLayout>,
    host_bytes: usize,
    slot_bytes: usize,
}

impl HostLayout {
    // Strides come from auditing every expert's converted, padded pack, not expert zero.
    pub(super) fn new(strides: &[usize]) -> Result<Self, EngError> {
        if strides.is_empty() {
            return Err(cache_error("host layout needs at least one routed layer"));
        }
        let mut layers = Vec::new();
        layers
            .try_reserve_exact(strides.len())
            .map_err(|error| EngError::MemoryPlan(format!("Flint expert cache layout: {error}")))?;
        let mut host_bytes = 0usize;
        let mut slot_bytes = 0usize;
        for &stride in strides {
            if stride == 0 {
                return Err(cache_error("expert stride must be nonzero"));
            }
            let source_bytes = stride.checked_mul(EXPERTS).ok_or_else(cache_overflow)?;
            let device_bytes = stride.checked_mul(SLOTS).ok_or_else(cache_overflow)?;
            let source = ByteRange {
                offset: host_bytes,
                bytes: source_bytes,
            };
            let slots = ByteRange {
                offset: slot_bytes,
                bytes: device_bytes,
            };
            host_bytes = host_bytes
                .checked_add(source_bytes)
                .ok_or_else(cache_overflow)?;
            slot_bytes = slot_bytes
                .checked_add(device_bytes)
                .ok_or_else(cache_overflow)?;
            layers.push(LayerLayout {
                source,
                slots,
                stride,
            });
        }
        Ok(Self {
            layers,
            host_bytes,
            slot_bytes,
        })
    }

    pub(super) fn layer(&self, index: usize) -> Result<LayerLayout, EngError> {
        self.layers
            .get(index)
            .copied()
            .ok_or_else(|| cache_error("layer is out of range"))
    }

    pub(super) fn host_bytes(&self) -> usize {
        self.host_bytes
    }

    pub(super) fn slot_bytes(&self) -> usize {
        self.slot_bytes
    }
}

impl LayerLayout {
    pub(super) fn source(&self, expert: u16) -> Result<ByteRange, EngError> {
        self.element(self.source, usize::from(expert), EXPERTS)
    }

    pub(super) fn slot(&self, slot: u16) -> Result<ByteRange, EngError> {
        self.element(self.slots, usize::from(slot), SLOTS)
    }

    fn element(
        &self,
        region: ByteRange,
        index: usize,
        count: usize,
    ) -> Result<ByteRange, EngError> {
        if index >= count {
            return Err(cache_error("expert or slot is out of range"));
        }
        let relative = index.checked_mul(self.stride).ok_or_else(cache_overflow)?;
        let offset = region
            .offset
            .checked_add(relative)
            .ok_or_else(cache_overflow)?;
        Ok(ByteRange {
            offset,
            bytes: self.stride,
        })
    }
}

// A coverage/layout seal only. The native owner must retain the immutable PinnedBuf;
// this type neither owns bytes nor proves physical pinning or GPU completion.
pub(super) struct HostStore {
    layout: HostLayout,
}

impl HostStore {
    pub(super) fn build(
        layout: HostLayout,
        mut fill: impl FnMut(usize, u16, ByteRange) -> Result<usize, EngError>,
    ) -> Result<Self, EngError> {
        for (layer, geometry) in layout.layers.iter().enumerate() {
            for expert in 0..EXPERTS {
                let expert = u16::try_from(expert).map_err(|_| cache_overflow())?;
                let range = geometry.source(expert)?;
                if fill(layer, expert, range)? != range.bytes {
                    return Err(cache_error("host expert write has the wrong byte count"));
                }
            }
        }
        Ok(Self { layout })
    }

    pub(super) fn layout(&self) -> &HostLayout {
        &self.layout
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct MissCopy {
    pub(super) expert: u16,
    pub(super) slot: u16,
    pub(super) source: ByteRange,
    pub(super) destination: ByteRange,
}

impl MissCopy {
    const EMPTY: Self = Self {
        expert: 0,
        slot: 0,
        source: ByteRange {
            offset: 0,
            bytes: 0,
        },
        destination: ByteRange {
            offset: 0,
            bytes: 0,
        },
    };
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct PassKey {
    owner: u64,
    generation: u64,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct ReadyPass(PassKey);

#[derive(Clone, Copy, PartialEq, Eq)]
enum Phase {
    Initial,
    Idle,
    Prepared,
    Ready,
    Poisoned,
}

pub(super) struct LayerCache {
    geometry: LayerLayout,
    stream: u64,
    key: PassKey,
    phase: Phase,
    residents: [u16; SLOTS],
    slots: [Option<u16>; EXPERTS],
    recency: [u16; SLOTS],
    demanded: [bool; EXPERTS],
    copies: [MissCopy; SLOTS],
    copy_count: usize,
}

impl LayerCache {
    pub(super) fn new(store: &HostStore, layer: usize, stream: u64) -> Result<Self, EngError> {
        static NEXT_OWNER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
        let geometry = store.layout().layer(layer)?;
        let owner = NEXT_OWNER
            .fetch_update(
                std::sync::atomic::Ordering::Relaxed,
                std::sync::atomic::Ordering::Relaxed,
                |value| value.checked_add(1),
            )
            .map_err(|_| cache_overflow())?;
        let canonical = std::array::from_fn(|index| index as u16);
        Ok(Self {
            geometry,
            stream,
            key: PassKey {
                owner,
                generation: 0,
            },
            phase: Phase::Initial,
            residents: canonical,
            slots: Self::canonical_slots(),
            recency: canonical,
            demanded: [false; EXPERTS],
            copies: [MissCopy::EMPTY; SLOTS],
            copy_count: 0,
        })
    }

    fn canonical_slots() -> [Option<u16>; EXPERTS] {
        std::array::from_fn(|index| (index < SLOTS).then_some(index as u16))
    }

    fn check(&self, phase: Phase, stream: u64) -> Result<(), EngError> {
        if self.phase != phase || self.stream != stream {
            return Err(cache_error("wrong cache phase or stream"));
        }
        Ok(())
    }

    fn check_pass(&self, key: PassKey, phase: Phase, stream: u64) -> Result<(), EngError> {
        self.check(phase, stream)?;
        if key != self.key {
            return Err(cache_error("stale or foreign pass key"));
        }
        Ok(())
    }

    fn copy(&self, expert: u16, slot: u16) -> Result<MissCopy, EngError> {
        Ok(MissCopy {
            expert,
            slot,
            source: self.geometry.source(expert)?,
            destination: self.geometry.slot(slot)?,
        })
    }

    pub(super) fn publish_initial(
        &mut self,
        stream: u64,
        establish: impl FnOnce(&[MissCopy]) -> Result<(), EngError>,
    ) -> Result<(), EngError> {
        self.check(Phase::Initial, stream)?;
        for index in 0..SLOTS {
            self.copies[index] = self.copy(index as u16, index as u16)?;
        }
        self.copy_count = SLOTS;
        self.phase = Phase::Poisoned;
        establish(&self.copies[..self.copy_count])?;
        self.copy_count = 0;
        self.phase = Phase::Idle;
        Ok(())
    }

    pub(super) fn prepare(
        &mut self,
        ids: &[i32],
        max_copy_bytes: usize,
    ) -> Result<PassKey, EngError> {
        self.check(Phase::Idle, self.stream)?;
        let mut demanded = [false; EXPERTS];
        let mut count = 0;
        for &id in ids {
            let index = usize::try_from(id).map_err(|_| cache_error("unknown expert ID"))?;
            let entry = demanded
                .get_mut(index)
                .ok_or_else(|| cache_error("unknown expert ID"))?;
            if !*entry {
                count += 1;
                *entry = true;
            }
        }
        if count > SLOTS {
            return Err(cache_error("demanded union exceeds slot capacity"));
        }
        let generation = self
            .key
            .generation
            .checked_add(1)
            .ok_or_else(cache_overflow)?;
        let mut residents = self.residents;
        let mut slots = self.slots;
        let mut recency = self.recency;
        let mut copies = [MissCopy::EMPTY; SLOTS];
        let mut copy_count = 0usize;
        for &id in ids {
            let expert = id as u16;
            let slot = if let Some(slot) = slots[usize::from(expert)] {
                usize::from(slot)
            } else {
                let slot = recency
                    .iter()
                    .copied()
                    .find(|&slot| !demanded[usize::from(residents[usize::from(slot)])])
                    .ok_or_else(|| cache_error("no unprotected victim"))?;
                slots[usize::from(residents[usize::from(slot)])] = None;
                residents[usize::from(slot)] = expert;
                slots[usize::from(expert)] = Some(slot);
                copies[copy_count] = self.copy(expert, slot)?;
                copy_count += 1;
                usize::from(slot)
            };
            let position = recency
                .iter()
                .position(|&value| usize::from(value) == slot)
                .ok_or_else(|| cache_error("invalid recency order"))?;
            recency.copy_within(position + 1..SLOTS, position);
            recency[SLOTS - 1] = slot as u16;
        }
        let bytes = copy_count
            .checked_mul(self.geometry.stride)
            .ok_or_else(cache_overflow)?;
        if bytes > max_copy_bytes {
            return Err(cache_error("miss copies exceed byte bound"));
        }
        self.residents = residents;
        self.slots = slots;
        self.recency = recency;
        self.demanded = demanded;
        self.copies = copies;
        self.copy_count = copy_count;
        self.key.generation = generation;
        self.phase = Phase::Prepared;
        Ok(self.key)
    }

    pub(super) fn copy_ready(
        &mut self,
        pass: PassKey,
        stream: u64,
        establish: impl FnOnce(&[MissCopy]) -> Result<(), EngError>,
    ) -> Result<ReadyPass, EngError> {
        self.check_pass(pass, Phase::Prepared, stream)?;
        self.phase = Phase::Poisoned;
        establish(&self.copies[..self.copy_count])?;
        self.phase = Phase::Ready;
        Ok(ReadyPass(pass))
    }

    pub(super) fn slot(&self, ready: ReadyPass, expert: u16) -> Result<u16, EngError> {
        self.check_pass(ready.0, Phase::Ready, self.stream)?;
        if !self
            .demanded
            .get(usize::from(expert))
            .copied()
            .unwrap_or(false)
        {
            return Err(cache_error("expert is outside this pass"));
        }
        self.slots[usize::from(expert)].ok_or_else(|| cache_error("expert has no slot"))
    }

    pub(super) fn release<T>(
        &mut self,
        ready: ReadyPass,
        stream: u64,
        finish: impl FnOnce() -> Result<T, EngError>,
    ) -> Result<T, EngError> {
        self.check_pass(ready.0, Phase::Ready, stream)?;
        self.phase = Phase::Poisoned;
        let result = finish()?;
        self.demanded = [false; EXPERTS];
        self.copy_count = 0;
        self.phase = Phase::Idle;
        Ok(result)
    }

    pub(super) fn poison(&mut self) {
        self.phase = Phase::Poisoned;
    }

    pub(super) fn reset_after_drain(
        &mut self,
        stream: u64,
        drain: impl FnOnce() -> Result<(), EngError>,
    ) -> Result<(), EngError> {
        self.check(Phase::Poisoned, stream)?;
        let generation = self
            .key
            .generation
            .checked_add(1)
            .ok_or_else(cache_overflow)?;
        self.phase = Phase::Poisoned;
        drain()?;
        self.residents = std::array::from_fn(|index| index as u16);
        self.slots = Self::canonical_slots();
        self.recency = self.residents;
        self.demanded = [false; EXPERTS];
        self.copies = [MissCopy::EMPTY; SLOTS];
        self.copy_count = 0;
        self.key.generation = generation;
        self.phase = Phase::Initial;
        Ok(())
    }
}

fn cache_overflow() -> EngError {
    EngError::MemoryPlan("Flint expert cache byte count overflows".into())
}

fn cache_error(message: &str) -> EngError {
    EngError::BadRequest(format!("Flint expert cache: {message}"))
}

pub(crate) struct HostBudget {
    pub(super) mem_available: u64,
    // None means a confirmed unlimited cgroup, never a failed limit read.
    pub(super) cgroup_remaining: Option<u64>,
    pub(super) os_floor: u64,
}

impl HostBudget {
    pub(crate) fn probe() -> Result<Self, EngError> {
        Self::probe_in(
            std::path::Path::new("/proc"),
            std::path::Path::new("/sys/fs/cgroup"),
            std::process::id(),
        )
    }

    fn probe_in(
        proc: &std::path::Path,
        root: &std::path::Path,
        pid: u32,
    ) -> Result<Self, EngError> {
        let meminfo = read_admission_file(&proc.join("meminfo"))?;
        let membership = read_admission_file(&proc.join("self/cgroup"))?;
        let mounts = read_admission_file(&proc.join("self/mountinfo"))?;
        let group = cgroup_from_proc(root, &membership, &mounts, pid)?;
        let root = std::fs::canonicalize(root)
            .map_err(|error| EngError::MemoryPlan(format!("Flint cgroup root: {error}")))?;
        Self::from_meminfo(&meminfo, cgroup_remaining_in(&root, &group)?)
    }

    pub(super) fn from_meminfo(
        contents: &str,
        cgroup_remaining: Option<u64>,
    ) -> Result<Self, EngError> {
        let total = meminfo_bytes(contents, "MemTotal:")?;
        let available = meminfo_bytes(contents, "MemAvailable:")?;
        if total == 0 || available > total {
            return Err(EngError::MemoryPlan(
                "Flint expert cache memory snapshot has inconsistent totals.".into(),
            ));
        }
        Ok(Self {
            mem_available: available,
            cgroup_remaining,
            os_floor: total.div_ceil(10).max(8u64 << 30),
        })
    }

    pub(crate) fn admit(&self, required: u64) -> Result<(), EngError> {
        self.admit_for(required, "Flint expert cache")
    }

    /// [`admit`] with the caller's name in the refusal, so a prefix
    /// snapshot refusal does not masquerade as an expert-cache one
    /// (the refusal names the caller).
    pub(crate) fn admit_for(&self, required: u64, what: &str) -> Result<(), EngError> {
        let available = self
            .cgroup_remaining
            .map_or(self.mem_available, |remaining| {
                remaining.min(self.mem_available)
            });
        if required
            .checked_add(self.os_floor)
            .is_none_or(|needed| needed > available)
        {
            return Err(EngError::MemoryPlan(format!(
                "{what} needs {required} bytes plus {} bytes for the OS floor. Available host/cgroup budget: {available} bytes.",
                self.os_floor
            )));
        }
        Ok(())
    }
}

fn meminfo_bytes(contents: &str, name: &str) -> Result<u64, EngError> {
    let invalid = || {
        EngError::MemoryPlan(format!(
            "Flint expert cache needs one valid {name} value in kB from /proc/meminfo."
        ))
    };
    let mut matches = contents.lines().filter_map(|line| line.strip_prefix(name));
    let mut fields = matches.next().ok_or_else(invalid)?.split_whitespace();
    let bytes = fields
        .next()
        .and_then(|value| value.parse::<u64>().ok())
        .and_then(|kib| kib.checked_mul(1024))
        .ok_or_else(invalid)?;
    if fields.next() != Some("kB") || fields.next().is_some() || matches.next().is_some() {
        return Err(invalid());
    }
    Ok(bytes)
}

fn cgroup_remaining_bytes(maximum: &str, current: &str) -> Result<Option<u64>, EngError> {
    let invalid = || {
        EngError::MemoryPlan(
            "Flint expert cache needs memory.max as max or an unsigned byte count, and memory.current as an unsigned byte count."
                .into(),
        )
    };
    let parse = |value: &str| {
        let value = value.trim();
        if value.is_empty() || !value.bytes().all(|byte| byte.is_ascii_digit()) {
            return Err(invalid());
        }
        value.parse::<u64>().map_err(|_| invalid())
    };
    let current = parse(current)?;
    match maximum.trim() {
        "max" => Ok(None),
        maximum => Ok(Some(parse(maximum)?.saturating_sub(current))),
    }
}

fn read_admission_file(path: &std::path::Path) -> Result<String, EngError> {
    std::fs::read_to_string(path).map_err(|error| {
        EngError::MemoryPlan(format!(
            "Flint expert cache cannot read {}: {error}",
            path.display()
        ))
    })
}

// Discovery must verify the global controller root and reject hidden ancestors first.
fn cgroup_remaining_in(
    root: &std::path::Path,
    cgroup: &std::path::Path,
) -> Result<Option<u64>, EngError> {
    if !root.is_absolute()
        || !cgroup.is_absolute()
        || !cgroup.starts_with(root)
        || root
            .components()
            .chain(cgroup.components())
            .any(|component| component == std::path::Component::ParentDir)
    {
        return Err(EngError::MemoryPlan(
            "Flint expert cache cgroup must be inside its verified controller root.".into(),
        ));
    }
    let mut minimum: Option<u64> = None;
    let mut group = cgroup;
    while group != root {
        let parent = group.parent().ok_or_else(|| {
            EngError::MemoryPlan("Flint expert cache cgroup has no controller ancestor.".into())
        })?;
        if read_admission_file(&parent.join("cgroup.subtree_control"))?
            .split_whitespace()
            .any(|controller| controller == "memory")
            && let Some(remaining) = cgroup_remaining_bytes(
                &read_admission_file(&group.join("memory.max"))?,
                &read_admission_file(&group.join("memory.current"))?,
            )?
        {
            minimum = Some(minimum.map_or(remaining, |previous| previous.min(remaining)));
        }
        group = parent;
    }
    Ok(minimum)
}

fn verify_cgroup_root(root: &std::path::Path, mounts: &str) -> Result<(), EngError> {
    let invalid = || {
        EngError::MemoryPlan(
            "Flint expert cache needs one complete cgroup2 root with the memory controller.".into(),
        )
    };
    let point = root.to_str().ok_or_else(invalid)?;
    let mut mounted = false;
    for line in mounts.lines() {
        let Some((head, tail)) = line.split_once(" - ") else {
            continue;
        };
        let fields: Vec<_> = head.split_ascii_whitespace().collect();
        if fields.get(4) != Some(&point) {
            continue;
        }
        if mounted
            || fields.get(3) != Some(&"/")
            || tail.split_ascii_whitespace().next() != Some("cgroup2")
        {
            return Err(invalid());
        }
        mounted = true;
    }
    if !mounted
        || !read_admission_file(&root.join("cgroup.controllers"))?
            .split_whitespace()
            .any(|controller| controller == "memory")
    {
        return Err(invalid());
    }
    // These interfaces exist on namespace/subtree roots, but not on the global root.
    for name in ["cgroup.type", "memory.max", "memory.current"] {
        let path = root.join(name);
        match std::fs::metadata(&path) {
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Ok(_) => return Err(invalid()),
            Err(error) => {
                return Err(EngError::MemoryPlan(format!(
                    "Flint expert cache cannot verify {}: {error}",
                    path.display()
                )));
            }
        }
    }
    Ok(())
}

fn cgroup_from_proc(
    root: &std::path::Path,
    membership: &str,
    mounts: &str,
    pid: u32,
) -> Result<std::path::PathBuf, EngError> {
    let invalid = || {
        EngError::MemoryPlan(
            "Flint expert cache cannot map its process inside a complete unified cgroup tree."
                .into(),
        )
    };
    let mut lines = membership.lines();
    let path = std::path::Path::new(
        lines
            .next()
            .and_then(|line| line.strip_prefix("0::"))
            .ok_or_else(invalid)?,
    );
    if pid == 0
        || lines.next().is_some()
        || !root.is_absolute()
        || !path.is_absolute()
        || root
            .components()
            .chain(path.components())
            .any(|component| component == std::path::Component::ParentDir)
    {
        return Err(invalid());
    }
    let resolve = |path: &std::path::Path| {
        std::fs::canonicalize(path).map_err(|error| {
            EngError::MemoryPlan(format!(
                "Flint expert cache cannot resolve {}: {error}",
                path.display()
            ))
        })
    };
    let root = resolve(root)?;
    verify_cgroup_root(&root, mounts)?;
    let group = resolve(&root.join(path.strip_prefix("/").map_err(|_| invalid())?))?;
    if !group.starts_with(&root) {
        return Err(invalid());
    }
    if group != root && read_admission_file(&group.join("cgroup.type"))?.trim() != "domain" {
        return Err(invalid());
    }
    let mut found = false;
    for owner in read_admission_file(&group.join("cgroup.procs"))?.split_whitespace() {
        found |= owner.parse::<u32>().map_err(|_| invalid())? == pid;
    }
    if !found {
        return Err(invalid());
    }
    Ok(group)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn host_layout_sizes_all_256_sources_and_exactly_224_slots() {
        let layout = HostLayout::new(&[256, 512]);
        assert!(
            layout.is_ok(),
            "valid packed expert strides must produce a host layout"
        );
        let layout = layout.unwrap();
        assert_eq!(layout.host_bytes(), 256 * (256 + 512));
        assert_eq!(layout.slot_bytes(), 224 * (256 + 512));
    }

    #[test]
    fn host_store_seal_checks_coverage_bounds_and_partial_cleanup() {
        let layout = HostLayout::new(&[3, 5]).unwrap();
        // These are tiny pageable fixture bytes, never physical pinned storage.
        let mut bytes = vec![0u8; layout.host_bytes()];
        let mut writes = 0;
        let store = HostStore::build(layout, |layer, id, range| {
            assert_eq!(layer, writes / EXPERTS);
            assert_eq!(usize::from(id), writes % EXPERTS);
            bytes[range.offset..range.offset + range.bytes].fill(id as u8);
            writes += 1;
            Ok(range.bytes)
        });
        assert!(
            store.is_ok(),
            "all exact expert writes must seal the immutable layout"
        );
        let store = store.unwrap();
        assert_eq!(writes, 2 * EXPERTS);
        let second = store.layout().layer(1).unwrap();
        assert_eq!(
            second.source(255).unwrap(),
            ByteRange {
                offset: 256 * 3 + 255 * 5,
                bytes: 5
            }
        );
        assert_eq!(
            second.slot(223).unwrap(),
            ByteRange {
                offset: 224 * 3 + 223 * 5,
                bytes: 5
            }
        );
        assert!(second.source(256).is_err());
        assert!(second.slot(224).is_err());
        assert!(store.layout().layer(2).is_err());
        for layer in 0..2 {
            for id in 0..256 {
                let range = store.layout().layer(layer).unwrap().source(id).unwrap();
                assert!(
                    bytes[range.offset..range.offset + range.bytes]
                        .iter()
                        .all(|&byte| byte == id as u8)
                );
            }
        }
        for strides in [
            vec![],
            vec![0],
            vec![usize::MAX / EXPERTS + 1],
            vec![usize::MAX / EXPERTS; 2],
        ] {
            assert!(HostLayout::new(&strides).is_err());
        }
        assert!(
            HostStore::build(HostLayout::new(&[3]).unwrap(), |_, _, range| Ok(range
                .bytes
                - 1))
            .is_err()
        );
        let fixture_owner = std::rc::Rc::new(vec![0u8; 32]);
        let released = std::rc::Rc::downgrade(&fixture_owner);
        let mut partial_writes = 0;
        let failed = HostStore::build(HostLayout::new(&[3]).unwrap(), move |_, id, range| {
            assert_eq!(fixture_owner.len(), 32);
            partial_writes += 1;
            if id == 13 {
                assert_eq!(partial_writes, 14);
                return Err(cache_error("fixture acquisition failed"));
            }
            Ok(range.bytes)
        });
        assert!(failed.is_err());
        assert!(
            released.upgrade().is_none(),
            "failed construction must drop its owned fill resources"
        );
    }

    #[test]
    fn protected_lru_bounded_misses_preserve_duplicate_touch_order() {
        let store = HostStore::build(
            HostLayout::new(&[3]).unwrap(),
            |_, _, range| Ok(range.bytes),
        )
        .unwrap();
        let mut cache = LayerCache::new(&store, 0, 7).unwrap();
        cache
            .publish_initial(7, |copies| {
                assert_eq!(copies.len(), SLOTS);
                for (index, copy) in copies.iter().enumerate() {
                    assert_eq!(usize::from(copy.expert), index);
                    assert_eq!(usize::from(copy.slot), index);
                }
                Ok(())
            })
            .unwrap();
        for ids in [&[-1][..], &[256][..], &(0..225).collect::<Vec<_>>()[..]] {
            assert!(cache.prepare(ids, usize::MAX).is_err());
        }
        assert!(cache.prepare(&[224, 0, 224, 1], 2).is_err());
        let pass = cache.prepare(&[224, 0, 224, 1], 3).unwrap();
        assert!(cache.prepare(&[3], 3).is_err());
        assert!(
            cache
                .copy_ready(pass, 8, |_| panic!("wrong stream callback"))
                .is_err()
        );
        let ready = cache
            .copy_ready(pass, 7, |copies| {
                assert_eq!(
                    copies,
                    &[MissCopy {
                        expert: 224,
                        slot: 2,
                        source: ByteRange {
                            offset: 224 * 3,
                            bytes: 3
                        },
                        destination: ByteRange {
                            offset: 2 * 3,
                            bytes: 3
                        },
                    }]
                );
                Ok(())
            })
            .unwrap();
        assert_eq!(cache.slot(ready, 224).unwrap(), 2);
        assert_eq!(cache.slot(ready, 0).unwrap(), 0);
        assert_eq!(cache.slot(ready, 1).unwrap(), 1);
        assert!(cache.slot(ready, 2).is_err());
        cache.release(ready, 7, || Ok(())).unwrap();
        let mut ids = vec![225];
        ids.extend(3..224);
        ids.push(1);
        let next = cache.prepare(&ids, 3).unwrap();
        assert!(
            cache
                .copy_ready(pass, 7, |_| panic!("stale pass callback"))
                .is_err()
        );
        let ready = cache
            .copy_ready(next, 7, |copies| {
                assert_eq!(copies.len(), 1);
                assert_eq!(copies[0].expert, 225);
                assert_eq!(
                    copies[0].slot, 0,
                    "duplicate touches must make 224 newer than 0"
                );
                Ok(())
            })
            .unwrap();
        assert_eq!(cache.slot(ready, 225).unwrap(), 0);
        cache.release(ready, 7, || Ok(())).unwrap();
    }

    #[test]
    fn lifecycle_poison_drain_and_canonical_republication() {
        let store = HostStore::build(
            HostLayout::new(&[3]).unwrap(),
            |_, _, range| Ok(range.bytes),
        )
        .unwrap();
        let mut cache = LayerCache::new(&store, 0, 7).unwrap();
        let foreign = LayerCache::new(&store, 0, 7).unwrap().key;
        drop(store); // Geometry is owned; this seal never owned or pinned bytes.
        assert!(cache.prepare(&[0], 0).is_err());
        assert!(cache.slot(ReadyPass(cache.key), 0).is_err());
        assert!(
            cache
                .publish_initial(8, |_| panic!("wrong stream"))
                .is_err()
        );
        assert!(
            cache
                .reset_after_drain(7, || panic!("wrong phase"))
                .is_err()
        );
        let mut order = Vec::new();
        cache
            .publish_initial(7, |copies| {
                assert_eq!(copies.len(), SLOTS);
                order.push("initial");
                Ok(())
            })
            .unwrap();
        let before = (cache.slots, cache.residents, cache.recency, cache.key);
        for ids in [vec![-1], vec![256], (0..225).collect()] {
            assert!(cache.prepare(&ids, usize::MAX).is_err());
            assert_eq!(
                (cache.slots, cache.residents, cache.recency, cache.key),
                before
            );
        }
        assert!(cache.prepare(&[224], 2).is_err());
        assert_eq!(
            (cache.slots, cache.residents, cache.recency, cache.key),
            before
        );
        let pass = cache.prepare(&[224, 0, 224, 1], 3).unwrap();
        assert_eq!(cache.slots[2], None);
        assert_eq!(cache.slots[224], Some(2));
        assert!(cache.prepare(&[225], 3).is_err());
        assert!(
            cache
                .copy_ready(foreign, 7, |_| panic!("foreign key"))
                .is_err()
        );
        assert!(
            cache
                .copy_ready(pass, 8, |_| panic!("wrong stream"))
                .is_err()
        );
        let ready = cache
            .copy_ready(pass, 7, |_| {
                order.push("copy");
                Ok(())
            })
            .unwrap();
        assert!(cache.prepare(&[225], 3).is_err());
        assert!(
            cache
                .release::<()>(ReadyPass(foreign), 7, || panic!("foreign ready"))
                .is_err()
        );
        assert!(
            cache
                .release::<()>(ready, 8, || panic!("wrong finish stream"))
                .is_err()
        );
        assert_eq!(cache.slot(ready, 224).unwrap(), 2);
        let value = cache
            .release(ready, 7, || {
                order.push("finish");
                Ok(41)
            })
            .unwrap();
        assert_eq!(value, 41);
        assert_eq!(order, ["initial", "copy", "finish"]);
        assert!(
            cache
                .release::<()>(ready, 7, || panic!("repeated finish"))
                .is_err()
        );
        assert!(cache.slot(ready, 224).is_err());
        let next = cache.prepare(&[225], 3).unwrap();
        assert!(cache.copy_ready(pass, 7, |_| panic!("stale copy")).is_err());
        assert!(
            cache
                .copy_ready(next, 7, |_| Err(cache_error("copy failed")))
                .is_err()
        );
        assert!(cache.phase == Phase::Poisoned);
        let held = (
            cache.slots,
            cache.residents,
            cache.recency,
            cache.demanded,
            cache.copies,
            cache.copy_count,
            cache.key,
        );
        assert!(
            cache
                .reset_after_drain(8, || panic!("wrong drain stream"))
                .is_err()
        );
        assert!(
            cache
                .reset_after_drain(7, || Err(cache_error("drain failed")))
                .is_err()
        );
        assert!(cache.phase == Phase::Poisoned);
        assert_eq!(
            (
                cache.slots,
                cache.residents,
                cache.recency,
                cache.demanded,
                cache.copies,
                cache.copy_count,
                cache.key
            ),
            held
        );
        assert!(
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let _ = cache.reset_after_drain(7, || panic!("drain panic"));
            }))
            .is_err()
        );
        assert!(cache.phase == Phase::Poisoned);
        assert!(cache.prepare(&[0], 0).is_err());
        cache
            .reset_after_drain(7, || {
                order.push("drain");
                Ok(())
            })
            .unwrap();
        assert!(cache.key.generation > next.generation);
        assert!(
            cache.prepare(&[0], 0).is_err(),
            "drain must not publish canonical bytes"
        );
        assert!(cache.slot(ready, 224).is_err());
        cache
            .publish_initial(7, |copies| {
                for (id, copy) in copies.iter().enumerate() {
                    assert_eq!((copy.expert, copy.slot), (id as u16, id as u16));
                }
                order.push("reinitialize");
                Ok(())
            })
            .unwrap();
        assert_eq!(
            order,
            ["initial", "copy", "finish", "drain", "reinitialize"]
        );
        for id in 0..EXPERTS {
            assert_eq!(cache.slots[id], (id < SLOTS).then_some(id as u16));
        }
        let pass = cache.prepare(&[0], 0).unwrap();
        assert!(
            cache
                .copy_ready(next, 7, |_| panic!("pre-reset key"))
                .is_err()
        );
        let ready = cache
            .copy_ready(pass, 7, |copies| {
                assert!(copies.is_empty());
                Ok(())
            })
            .unwrap();
        assert!(cache.slot(ReadyPass(next), 0).is_err());
        assert!(
            cache
                .release(ready, 7, || Err::<(), _>(cache_error("finish failed")))
                .is_err()
        );
        assert!(cache.phase == Phase::Poisoned && cache.demanded[0]);
        assert!(cache.prepare(&[1], 0).is_err());
        assert!(
            cache
                .release::<()>(ready, 7, || panic!("failed finish retry"))
                .is_err()
        );
        cache.reset_after_drain(7, || Ok(())).unwrap();
        assert!(
            cache
                .publish_initial(7, |_| Err(cache_error("initial failed")))
                .is_err()
        );
        assert!(cache.phase == Phase::Poisoned && cache.copy_count == SLOTS);
        cache.reset_after_drain(7, || Ok(())).unwrap();
        cache.publish_initial(7, |_| Ok(())).unwrap();
        let pass = cache.prepare(&[0], 0).unwrap();
        assert!(
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let _ = cache.copy_ready(pass, 7, |_| panic!("copy panic"));
            }))
            .is_err()
        );
        assert!(cache.phase == Phase::Poisoned && cache.demanded[0]);
        cache.reset_after_drain(7, || Ok(())).unwrap();
        cache.publish_initial(7, |_| Ok(())).unwrap();
        let pass = cache.prepare(&[0], 0).unwrap();
        let ready = cache.copy_ready(pass, 7, |_| Ok(())).unwrap();
        assert!(
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let _ = cache.release::<()>(ready, 7, || panic!("finish panic"));
            }))
            .is_err()
        );
        assert!(cache.phase == Phase::Poisoned && cache.demanded[0]);
        cache.reset_after_drain(7, || Ok(())).unwrap();
        cache.publish_initial(7, |_| Ok(())).unwrap();
        cache.poison();
        cache.poison();
        assert!(cache.prepare(&[0], 0).is_err());
        cache.reset_after_drain(7, || Ok(())).unwrap();
        assert!(cache.phase == Phase::Initial);
    }

    #[test]
    fn captured_cgroup_roots_match_real_admission() {
        let captures: serde_json::Value =
            serde_json::from_str(include_str!("testdata/cgroup-views-20261028.json")).unwrap();
        let hostns: serde_json::Value =
            serde_json::from_str(include_str!("testdata/cgroup-hostns-20261028.json")).unwrap();
        let host_files = captures["host"]["root_file_names"].as_array().unwrap();
        for name in ["cgroup.type", "memory.max", "memory.current"] {
            assert!(!host_files.iter().any(|value| value.as_str() == Some(name)));
        }
        for (name, view, accepted) in [
            ("container", &captures["container"], false),
            ("host", &captures["host"], true),
            ("hostns", &hostns, true),
        ] {
            let root = tempfile::tempdir().unwrap();
            let directory = root.path().canonicalize().unwrap();
            let controllers = view["controllers"].as_str().unwrap();
            assert!(
                controllers
                    .split_whitespace()
                    .any(|value| value == "memory")
            );
            std::fs::write(directory.join("cgroup.controllers"), controllers).unwrap();
            for (file, key) in [
                ("cgroup.type", "cgroup_type"),
                ("memory.max", "memory_max"),
                ("memory.current", "memory_current"),
            ] {
                if let Some(contents) = view[key].as_str() {
                    std::fs::write(directory.join(file), contents).unwrap();
                }
            }
            let mounts = view["mountinfo"]
                .as_str()
                .unwrap()
                .replace("/sys/fs/cgroup", &directory.display().to_string());
            let result = verify_cgroup_root(&directory, &mounts);
            eprintln!("captured view {name}: accepted={}", result.is_ok());
            if accepted {
                assert!(result.is_ok(), "{name}: {result:?}");
            } else {
                assert!(matches!(result, Err(EngError::MemoryPlan(message))
                    if message.contains("one complete cgroup2 root")));
            }
        }
    }

    #[test]
    fn captured_cgroup_host_member_maps_to_nested_group() {
        let captures: serde_json::Value =
            serde_json::from_str(include_str!("testdata/cgroup-views-20261028.json")).unwrap();
        let view = &captures["host"];
        let root = tempfile::tempdir().unwrap();
        let directory = root.path().canonicalize().unwrap();
        std::fs::write(
            directory.join("cgroup.controllers"),
            view["controllers"].as_str().unwrap(),
        )
        .unwrap();
        let membership = view["membership"].as_str().unwrap();
        let group = directory.join(membership.trim().strip_prefix("0::/").unwrap());
        std::fs::create_dir_all(&group).unwrap();
        std::fs::write(
            group.join("cgroup.type"),
            view["member_type"].as_str().unwrap(),
        )
        .unwrap();
        std::fs::write(
            group.join("cgroup.procs"),
            view["member_procs"].as_str().unwrap(),
        )
        .unwrap();
        let mounts = view["mountinfo"]
            .as_str()
            .unwrap()
            .replace("/sys/fs/cgroup", &directory.display().to_string());
        let pid = u32::try_from(view["pid"].as_u64().unwrap()).unwrap();
        let result = cgroup_from_proc(&directory, membership, &mounts, pid);
        assert_eq!(result.unwrap(), group);
        eprintln!("captured host member: mapped=true pid={pid}");
    }

    fn discovery_fixture() -> (tempfile::TempDir, std::path::PathBuf, String) {
        let root = tempfile::tempdir().unwrap();
        let directory = root.path().canonicalize().unwrap();
        let group = directory.join("work");
        std::fs::create_dir(&group).unwrap();
        std::fs::write(directory.join("cgroup.controllers"), "memory\n").unwrap();
        std::fs::write(group.join("cgroup.type"), "domain\n").unwrap();
        std::fs::write(group.join("cgroup.procs"), "41\n42\n").unwrap();
        let mounts = format!(
            "36 29 0:33 / {} rw - cgroup2 cgroup rw\n",
            directory.display()
        );
        (root, group, mounts)
    }

    #[test]
    fn cgroup_discovery_resolves_own_pid_in_global_tree() {
        let (root, group, mounts) = discovery_fixture();
        let resolved = cgroup_from_proc(root.path(), "0::/work\n", &mounts, 42);
        assert!(
            resolved.is_ok(),
            "a complete global cgroup tree must resolve its own process group"
        );
        assert_eq!(resolved.unwrap(), group);
    }

    #[cfg(unix)]
    #[test]
    fn cgroup_discovery_matches_canonical_mount_paths() {
        let (root, group, mounts) = discovery_fixture();
        let aliases = tempfile::tempdir().unwrap();
        let alias = aliases.path().join("controller-root");
        std::os::unix::fs::symlink(root.path(), &alias).unwrap();
        let resolved = cgroup_from_proc(&alias, "0::/work\n", &mounts, 42).unwrap();
        assert_eq!(resolved, group);
    }

    #[test]
    fn cgroup_discovery_rejects_hidden_or_ambiguous_roots() {
        for (name, contents) in [
            ("cgroup.type", "domain\n"),
            ("memory.max", "max\n"),
            ("memory.current", "0\n"),
            ("cgroup.controllers", "cpu\n"),
        ] {
            let (root, _, mounts) = discovery_fixture();
            std::fs::write(root.path().join(name), contents).unwrap();
            let result = cgroup_from_proc(root.path(), "0::/work\n", &mounts, 42);
            assert!(
                matches!(result, Err(EngError::MemoryPlan(message))
                    if message.contains("complete cgroup2 root")),
                "must reject root interface {name}"
            );
        }
        for (source, filesystem, copies) in [
            ("/delegated", "cgroup2", 1),
            ("/", "tmpfs", 1),
            ("/", "cgroup2", 2),
        ] {
            let (root, group, _) = discovery_fixture();
            let mounts = format!(
                "36 29 0:33 {source} {} rw - {filesystem} cgroup rw\n",
                group.parent().unwrap().display()
            )
            .repeat(copies);
            let result = cgroup_from_proc(root.path(), "0::/work\n", &mounts, 42);
            assert!(
                matches!(result, Err(EngError::MemoryPlan(message))
                    if message.contains("complete cgroup2 root")),
                "must reject mount {source}/{filesystem}/{copies}"
            );
        }
    }

    #[test]
    fn cgroup_discovery_rejects_bad_process_mapping() {
        for (membership, pid, owners, kind) in [
            ("0::/../work\n", 42, "41\n42\n", "domain"),
            ("0::work\n", 42, "41\n42\n", "domain"),
            ("0::/work\n0::/work\n", 42, "41\n42\n", "domain"),
            ("1:memory:/work\n", 42, "41\n42\n", "domain"),
            ("0::/work\n", 0, "41\n42\n", "domain"),
            ("0::/work\n", 42, "41\n", "domain"),
            ("0::/work\n", 42, "41\n42\n", "threaded"),
            ("0::/work\n", 42, "41\n42\nnot-a-pid\n", "domain"),
        ] {
            let (root, group, mounts) = discovery_fixture();
            std::fs::write(group.join("cgroup.procs"), owners).unwrap();
            std::fs::write(group.join("cgroup.type"), kind).unwrap();
            let result = cgroup_from_proc(root.path(), membership, &mounts, pid);
            assert!(
                matches!(result, Err(EngError::MemoryPlan(message))
                    if message.contains("cannot map its process")),
                "must reject process mapping {membership:?}/{pid}/{owners:?}/{kind}"
            );
        }
    }

    #[test]
    fn cgroup_reader_rejects_missing_current_with_unlimited_max() {
        let root = tempfile::tempdir().unwrap();
        let child = root.path().join("child");
        std::fs::create_dir(&child).unwrap();
        std::fs::write(root.path().join("cgroup.controllers"), "memory\n").unwrap();
        std::fs::write(root.path().join("cgroup.subtree_control"), "memory\n").unwrap();
        std::fs::write(child.join("memory.max"), "max\n").unwrap();
        let result = cgroup_remaining_in(root.path(), &child);
        assert!(
            matches!(
                result,
                Err(EngError::MemoryPlan(message)) if message.contains("memory.current")
            ),
            "an unreadable active charge file must not become an unlimited budget"
        );
    }

    #[test]
    fn cgroup_reader_applies_finite_ancestor_charges() {
        let root = tempfile::tempdir().unwrap();
        let parent = root.path().join("parent");
        let child = parent.join("child");
        std::fs::create_dir_all(&child).unwrap();
        std::fs::write(root.path().join("cgroup.controllers"), "memory\n").unwrap();
        for directory in [root.path(), parent.as_path()] {
            std::fs::write(directory.join("cgroup.subtree_control"), "memory\n").unwrap();
        }
        for (directory, maximum, current) in [
            (&parent, "17179869184\n", "4294967296\n"),
            (&child, "68719476736\n", "1073741824\n"),
        ] {
            std::fs::write(directory.join("memory.max"), maximum).unwrap();
            std::fs::write(directory.join("memory.current"), current).unwrap();
        }
        let remaining = cgroup_remaining_in(root.path(), &child);
        assert!(
            remaining.is_ok(),
            "a readable hierarchy must honor finite ancestor charges"
        );
        assert_eq!(remaining.unwrap(), Some(12u64 << 30));
    }

    #[test]
    fn cgroup_snapshot_accounts_for_current_charges() {
        let remaining = cgroup_remaining_bytes("17179869184\n", "4294967296\n");
        assert!(
            remaining.is_ok(),
            "valid finite cgroup limit and current charges must produce a budget"
        );
        let remaining = remaining.unwrap();
        assert_eq!(remaining, Some(12u64 << 30));
        let budget = HostBudget::from_meminfo(
            "MemTotal: 33554432 kB\nMemAvailable: 25165824 kB\n",
            remaining,
        )
        .unwrap();
        assert!(budget.admit(4u64 << 30).is_ok());
        assert!(budget.admit((4u64 << 30) + 1).is_err());
    }

    #[test]
    fn cgroup_snapshot_distinguishes_unlimited_and_exhausted_limits() {
        for (maximum, current, remaining) in [
            ("max\n", "0\n", None),
            (" max\n", "18446744073709551615\n", None),
            ("0", "0", Some(0)),
            ("1024", "1024", Some(0)),
            ("4", "8", Some(0)),
            ("18446744073709551615", "0", Some(u64::MAX)),
            ("18446744073709551615", "18446744073709551615", Some(0)),
        ] {
            assert_eq!(
                cgroup_remaining_bytes(maximum, current).unwrap(),
                remaining,
                "{maximum:?} / {current:?}"
            );
        }
    }

    #[test]
    fn cgroup_snapshot_rejects_malformed_limits_and_current_charges() {
        for (maximum, current) in [
            ("", "0"),
            ("MAX", "0"),
            ("bad", "0"),
            ("1 kB", "0"),
            ("1 2", "0"),
            ("1.5", "0"),
            ("-1", "0"),
            ("+1", "0"),
            ("18446744073709551616", "0"),
            ("1", ""),
            ("1", "bad"),
            ("1", "1 kB"),
            ("1", "-1"),
            ("1", "+1"),
            ("1", "18446744073709551616"),
            ("max", ""),
            ("max", "bad"),
            ("max", "max"),
            ("max", "-1"),
            ("max", "18446744073709551616"),
        ] {
            assert!(
                cgroup_remaining_bytes(maximum, current).is_err(),
                "{maximum:?} / {current:?}"
            );
        }
    }

    #[test]
    fn host_snapshot_uses_mem_available_and_model_scaled_os_floor() {
        let result = HostBudget::from_meminfo(
            "MemTotal:       268435456 kB\nMemAvailable:    50331648 kB\n",
            None,
        );
        assert!(
            result.is_ok(),
            "valid memory snapshot must produce an admission budget"
        );
        let budget = result.unwrap();
        assert_eq!(budget.mem_available, 48u64 << 30);
        assert_eq!(budget.os_floor, 27_487_790_695);
        assert_eq!(budget.cgroup_remaining, None);
        assert!(budget.admit(256 * 47 * 1_603_584u64).is_ok());
    }

    #[test]
    fn host_snapshot_preserves_minimum_floor_and_finite_cgroup() {
        let text = "MemAvailable: 25165824 kB\nMemFree: 0 kB\nMemTotal: 33554432 kB\n";
        let floor = 8u64 << 30;
        let budget = HostBudget::from_meminfo(text, Some(floor + 1)).unwrap();
        assert_eq!(budget.os_floor, floor);
        assert_eq!(budget.cgroup_remaining, Some(floor + 1));
        assert!(budget.admit(1).is_ok());
        assert!(budget.admit(2).is_err());
        assert!(
            HostBudget::from_meminfo(text, Some(0))
                .unwrap()
                .admit(0)
                .is_err()
        );
        let empty = HostBudget::from_meminfo("MemTotal: 1 kB\nMemAvailable: 0 kB\n", None).unwrap();
        assert_eq!(empty.mem_available, 0);
        assert!(empty.admit(0).is_err());
    }

    #[test]
    fn host_snapshot_rejects_malformed_or_inconsistent_fields() {
        for text in [
            "",
            "MemTotal: 1 kB\n",
            "MemAvailable: 1 kB\n",
            "MemTotal: 0 kB\nMemAvailable: 0 kB\n",
            "MemTotal: 1 kB\nMemAvailable: 2 kB\n",
            "MemTotal: -1 kB\nMemAvailable: 0 kB\n",
            "MemTotal: 1 MB\nMemAvailable: 0 kB\n",
            "MemTotal: 1 kB\nMemAvailable: bad kB\n",
            "MemTotal: 1 kB\nMemAvailable: 0 MB\n",
            "MemTotal: 1 kB trailing\nMemAvailable: 0 kB\n",
            "MemTotal: 1 kB\nMemTotal: 1 kB\nMemAvailable: 0 kB\n",
            "MemTotal: 1 kB\nMemAvailable: 0 kB\nMemAvailable: 0 kB\n",
            "MemTotal: 18014398509481984 kB\nMemAvailable: 0 kB\n",
            "MemTotal: 1 kB\nMemAvailable: 18014398509481984 kB\n",
        ] {
            assert!(HostBudget::from_meminfo(text, None).is_err(), "{text:?}");
        }
    }

    #[test]
    fn host_budget_rejects_unrepresentable_payload_plus_floor() {
        let budget = HostBudget {
            mem_available: u64::MAX,
            cgroup_remaining: None,
            os_floor: 1,
        };
        assert!(budget.admit(u64::MAX).is_err());
        assert!(budget.admit(u64::MAX - 1).is_ok());
    }

    #[test]
    fn host_budget_accepts_exact_model_demand_only_after_os_floor() {
        let required = 256 * 47 * 1_603_584u64;
        let floor = 8u64 << 30;
        let exact = HostBudget {
            mem_available: required + floor,
            cgroup_remaining: Some(required + floor),
            os_floor: floor,
        };
        assert!(
            exact.admit(required).is_ok(),
            "exact model demand plus OS floor must admit"
        );
        assert!(
            HostBudget {
                mem_available: exact.mem_available - 1,
                ..exact
            }
            .admit(required)
            .is_err()
        );
        assert!(
            HostBudget {
                cgroup_remaining: Some(required + floor - 1),
                ..exact
            }
            .admit(required)
            .is_err()
        );
        assert!(
            HostBudget {
                cgroup_remaining: None,
                ..exact
            }
            .admit(required)
            .is_ok(),
            "an unlimited cgroup still obeys host availability and the OS floor"
        );
    }

    #[test]
    fn full_slot_union_exact_miss_budget_is_atomic_and_protects_demand() {
        let stride = 257;
        let store = HostStore::build(HostLayout::new(&[stride]).unwrap(), |_, _, range| {
            Ok(range.bytes)
        })
        .unwrap();
        let mut cache = LayerCache::new(&store, 0, 7).unwrap();
        cache.publish_initial(7, |_| Ok(())).unwrap();
        let ids: Vec<i32> = (32..256).collect();
        let before = (
            cache.slots,
            cache.residents,
            cache.recency,
            cache.key,
            cache.demanded,
            cache.copies,
            cache.copy_count,
        );
        assert!(cache.prepare(&ids, 32 * stride - 1).is_err());
        assert_eq!(
            (
                cache.slots,
                cache.residents,
                cache.recency,
                cache.key,
                cache.demanded,
                cache.copies,
                cache.copy_count
            ),
            before
        );
        let pass = cache.prepare(&ids, 32 * stride).unwrap();
        let expected: Vec<_> = (224..256)
            .map(|id| (id as u16, (id - 224) as u16))
            .collect();
        let ready = cache
            .copy_ready(pass, 7, |copies| {
                assert_eq!(copies.len(), 32);
                assert_eq!(
                    copies
                        .iter()
                        .map(|copy| (copy.expert, copy.slot))
                        .collect::<Vec<_>>(),
                    expected
                );
                let mut victims = std::collections::BTreeSet::new();
                for copy in copies {
                    assert!(victims.insert(copy.slot));
                    assert!((copy.slot as usize) < 32);
                    assert_eq!(copy.source, store.layout().layer(0)?.source(copy.expert)?);
                    assert_eq!(copy.destination, store.layout().layer(0)?.slot(copy.slot)?);
                    assert!(copy.source.offset + copy.source.bytes <= store.layout().host_bytes());
                    assert!(
                        copy.destination.offset + copy.destination.bytes
                            <= store.layout().slot_bytes()
                    );
                }
                Ok(())
            })
            .unwrap();
        for id in 32..224 {
            assert_eq!(cache.slot(ready, id).unwrap(), id);
        }
        for id in 224..256 {
            assert_eq!(cache.slot(ready, id).unwrap(), id - 224);
        }
        cache.release(ready, 7, || Ok(())).unwrap();
    }

    #[test]
    fn composed_host_probe_uses_verified_tree_and_fails_closed() {
        let (root, group, mounts) = discovery_fixture();
        let proc = tempfile::tempdir().unwrap();
        std::fs::create_dir(proc.path().join("self")).unwrap();
        std::fs::write(
            proc.path().join("meminfo"),
            "MemTotal: 33554432 kB\nMemAvailable: 25165824 kB\n",
        )
        .unwrap();
        std::fs::write(proc.path().join("self/cgroup"), "0::/work\n").unwrap();
        std::fs::write(proc.path().join("self/mountinfo"), mounts).unwrap();
        std::fs::write(root.path().join("cgroup.subtree_control"), "memory\n").unwrap();
        std::fs::write(group.join("memory.max"), "17179869184\n").unwrap();
        std::fs::write(group.join("memory.current"), "1073741824\n").unwrap();
        let budget = HostBudget::probe_in(proc.path(), root.path(), 42).unwrap();
        assert_eq!(budget.cgroup_remaining, Some(15u64 << 30));
        assert!(budget.admit(7u64 << 30).is_ok());
        assert!(budget.admit((7u64 << 30) + 1).is_err());
        std::fs::remove_file(group.join("memory.current")).unwrap();
        assert!(HostBudget::probe_in(proc.path(), root.path(), 42).is_err());
    }
}
