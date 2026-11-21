//! Host-RAM admission: the cgroup-aware guard the expert caches use.
//!
//! probe reads /proc/meminfo and the process's cgroup tree, and admit
//! refuses a payload that would break the OS floor or the enclosing cgroup
//! limit. Every refusal names the caller that refused; the frozen
//! families' exact texts are pinned by the full-text tests in this module.

use crate::EngError;

pub(crate) struct HostAdmission {
    // The total and the floor are read by the CUDA-build facts paths
    // (probe_or_bare and the admit pair); the plain host build only asks
    // for the available budget.
    #[allow(dead_code)]
    mem_total: u64,
    mem_available: u64,
    // None means a confirmed unlimited cgroup, never a failed limit read.
    cgroup_remaining: Option<u64>,
    #[allow(dead_code)]
    os_floor: u64,
}

impl HostAdmission {
    #[allow(
        dead_code,
        reason = "the callers are Linux-only; the macOS test build exercises probe_in"
    )]
    pub(crate) fn probe(label: &'static str, root_label: &'static str) -> Result<Self, EngError> {
        Self::probe_in(
            label,
            root_label,
            std::path::Path::new("/proc"),
            std::path::Path::new("/sys/fs/cgroup"),
            std::process::id(),
        )
    }

    #[cfg(all(feature = "native-cuda", target_os = "linux"))]
    pub(crate) fn cache_capacity(&self) -> u64 {
        self.available_budget().saturating_sub(self.os_floor)
    }

    /// The admission pair for facts paths that must not go dark inside a container: when the
    /// cgroup probe refuses (a namespace root, for example), the pair falls back to the bare
    /// `/proc/meminfo` figures and logs the refusal — those callers worked on the bare figure
    /// before, so they keep it. Planning that sizes memory uses [`HostAdmission::probe`] and
    /// refuses instead; an optional feature may plan zero on the refusal. The plain Linux
    /// build's Nemotron fit is one of these fall-back callers: a default start must keep
    /// serving where the bare figure always served.
    #[cfg(any(test, target_os = "linux"))]
    #[allow(
        dead_code,
        reason = "the callers are Linux-only; the macOS test build keeps the pair compiled"
    )]
    pub(crate) fn probe_or_bare(
        label: &'static str,
        root_label: &'static str,
    ) -> Result<(u64, u64), EngError> {
        match Self::probe(label, root_label) {
            Ok(admission) => Ok(admission.host_memory()),
            Err(refusal) => {
                let text = std::fs::read_to_string("/proc/meminfo").map_err(|error| {
                    EngError::MemoryPlan(format!(
                        "{label} cannot read /proc/meminfo for the bare fallback: {error}"
                    ))
                })?;
                Self::bare_fallback(label, root_label, &refusal, &text)
            }
        }
    }

    /// The bare path of [`probe_or_bare`]: the machine-wide figures of the given meminfo
    /// text, with one log line naming the refusal it replaced. Split out so a test can pin
    /// that an unverified budget falls back to a figure instead of refusing.
    #[cfg(any(test, target_os = "linux"))]
    #[allow(
        dead_code,
        reason = "the callers are Linux-only; the macOS test build keeps the pair compiled"
    )]
    pub(crate) fn bare_fallback(
        label: &'static str,
        root_label: &'static str,
        refusal: &EngError,
        meminfo: &str,
    ) -> Result<(u64, u64), EngError> {
        let bare = Self::from_meminfo(label, meminfo, None)?;
        eprintln!(
            "{root_label}: cgroup budget unknown ({refusal:?}); using the bare MemAvailable figure"
        );
        Ok(bare.host_memory())
    }

    fn probe_in(
        label: &'static str,
        root_label: &'static str,
        proc: &std::path::Path,
        root: &std::path::Path,
        pid: u32,
    ) -> Result<Self, EngError> {
        let meminfo = read_admission_file(label, &proc.join("meminfo"))?;
        let membership = read_admission_file(label, &proc.join("self/cgroup"))?;
        let mounts = read_admission_file(label, &proc.join("self/mountinfo"))?;
        let group = cgroup_from_proc(label, root, &membership, &mounts, pid)?;
        let root = std::fs::canonicalize(root).map_err(|error| root_error(root_label, error))?;
        Self::from_meminfo(label, &meminfo, cgroup_remaining_in(label, &root, &group)?)
    }

    /// Private to `probe_in` except for the host tests of the memory plans, which build the
    /// synthetic cgroup-clamped snapshot the CUDA start builds through `probe` (the same
    /// seam as `probe_in`).
    pub(crate) fn from_meminfo(
        label: &'static str,
        contents: &str,
        cgroup_remaining: Option<u64>,
    ) -> Result<Self, EngError> {
        let total = meminfo_bytes(label, contents, "MemTotal:")?;
        let available = meminfo_bytes(label, contents, "MemAvailable:")?;
        if total == 0 || available > total {
            return Err(EngError::MemoryPlan(format!(
                "{label} memory snapshot has inconsistent totals."
            )));
        }
        Ok(Self {
            mem_total: total,
            mem_available: available,
            cgroup_remaining,
            os_floor: total.div_ceil(10).max(8u64 << 30),
        })
    }

    /// The byte budget admission admits against: `MemAvailable` clamped by every enclosing
    /// cgroup's remaining bytes. The same figure [`HostAdmission::admit_for`] refuses against,
    /// so a host plan sized from it cannot disagree with the guard that checks it. Names no
    /// model family, so any family's host plan may reuse it.
    pub(crate) fn available_budget(&self) -> u64 {
        self.cgroup_remaining
            .map_or(self.mem_available, |remaining| {
                remaining.min(self.mem_available)
            })
    }

    /// `MemTotal` paired with [`HostAdmission::available_budget`], for host plans that carry
    /// both figures (the tier and bank plans read them as a pair).
    #[cfg_attr(
        not(test),
        allow(dead_code, reason = "the CUDA-build facts paths read the pair")
    )]
    pub(crate) fn host_memory(&self) -> (u64, u64) {
        (self.mem_total, self.available_budget())
    }

    /// [`admit_for`] with the caller's label as the refusing name.
    #[cfg_attr(
        not(test),
        allow(dead_code, reason = "the refusals are pinned by this module's tests")
    )]
    pub(crate) fn admit(&self, label: &'static str, required: u64) -> Result<(), EngError> {
        self.admit_for(required, label)
    }

    /// The admission check with the caller's name in the refusal, so a prefix
    /// snapshot refusal does not masquerade as another caller's
    /// (the refusal names the caller).
    #[cfg_attr(
        not(test),
        allow(dead_code, reason = "the refusals are pinned by this module's tests")
    )]
    pub(crate) fn admit_for(&self, required: u64, what: &str) -> Result<(), EngError> {
        let available = self.available_budget();
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

fn meminfo_bytes(label: &'static str, contents: &str, name: &str) -> Result<u64, EngError> {
    let invalid = || {
        EngError::MemoryPlan(format!(
            "{label} needs one valid {name} value in kB from /proc/meminfo."
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

fn cgroup_remaining_bytes(
    label: &'static str,
    maximum: &str,
    current: &str,
) -> Result<Option<u64>, EngError> {
    let invalid = || {
        EngError::MemoryPlan(format!(
            "{label} needs memory.max as max or an unsigned byte count, and memory.current as an unsigned byte count."
        ))
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

fn root_error(root_label: &str, error: std::io::Error) -> EngError {
    EngError::MemoryPlan(format!("{root_label} cgroup root: {error}"))
}

fn read_admission_file(label: &'static str, path: &std::path::Path) -> Result<String, EngError> {
    std::fs::read_to_string(path).map_err(|error| {
        EngError::MemoryPlan(format!("{label} cannot read {}: {error}", path.display()))
    })
}

// Discovery must verify the global controller root and reject hidden ancestors first.
fn cgroup_remaining_in(
    label: &'static str,
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
        return Err(EngError::MemoryPlan(format!(
            "{label} cgroup must be inside its verified controller root."
        )));
    }
    let mut minimum: Option<u64> = None;
    let mut group = cgroup;
    while group != root {
        let parent = group.parent().ok_or_else(|| {
            EngError::MemoryPlan(format!("{label} cgroup has no controller ancestor."))
        })?;
        if read_admission_file(label, &parent.join("cgroup.subtree_control"))?
            .split_whitespace()
            .any(|controller| controller == "memory")
            && let Some(remaining) = cgroup_remaining_bytes(
                label,
                &read_admission_file(label, &group.join("memory.max"))?,
                &read_admission_file(label, &group.join("memory.current"))?,
            )?
        {
            minimum = Some(minimum.map_or(remaining, |previous| previous.min(remaining)));
        }
        group = parent;
    }
    // The root of a cgroup namespace is the container's own cgroup: no parent is visible, so
    // its limit is the outermost one. The global root has no limit files.
    if root.join("memory.max").exists() {
        let remaining = cgroup_remaining_bytes(
            label,
            &read_admission_file(label, &root.join("memory.max"))?,
            &read_admission_file(label, &root.join("memory.current"))?,
        )?;
        if let Some(remaining) = remaining {
            minimum = Some(minimum.map_or(remaining, |previous| previous.min(remaining)));
        }
    }
    Ok(minimum)
}

fn verify_cgroup_root(
    label: &'static str,
    root: &std::path::Path,
    mounts: &str,
) -> Result<(), EngError> {
    let invalid = || {
        EngError::MemoryPlan(format!(
            "{label} needs one complete cgroup2 root with the memory controller."
        ))
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
        || !read_admission_file(label, &root.join("cgroup.controllers"))?
            .split_whitespace()
            .any(|controller| controller == "memory")
    {
        return Err(invalid());
    }
    // These interfaces exist on namespace/subtree roots (a container's own cgroup), but not on
    // the global root. All three together mean a namespace root, whose own limit
    // `cgroup_remaining_in` applies; some of them alone is an ambiguous tree.
    let mut present = 0;
    for name in ["cgroup.type", "memory.max", "memory.current"] {
        let path = root.join(name);
        match std::fs::metadata(&path) {
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Ok(_) => present += 1,
            Err(error) => {
                return Err(EngError::MemoryPlan(format!(
                    "{label} cannot verify {}: {error}",
                    path.display()
                )));
            }
        }
    }
    if present != 0 && present != 3 {
        return Err(invalid());
    }
    Ok(())
}

fn cgroup_from_proc(
    label: &'static str,
    root: &std::path::Path,
    membership: &str,
    mounts: &str,
    pid: u32,
) -> Result<std::path::PathBuf, EngError> {
    let invalid = || {
        EngError::MemoryPlan(format!(
            "{label} cannot map its process inside a complete unified cgroup tree."
        ))
    };
    // Legacy hierarchies such as net_cls can coexist with v2 memory accounting.
    let mut unified_path = None;
    for line in membership.lines() {
        let mut fields = line.splitn(3, ':');
        let id = fields.next().ok_or_else(invalid)?;
        let controllers = fields.next().ok_or_else(invalid)?;
        let path = std::path::Path::new(fields.next().ok_or_else(invalid)?);
        if !path.is_absolute() {
            return Err(invalid());
        }
        if id == "0" {
            if !controllers.is_empty() || unified_path.replace(path).is_some() {
                return Err(invalid());
            }
        } else if !id.bytes().all(|b| b.is_ascii_digit())
            || id.parse::<u32>().ok().is_none_or(|id| id == 0)
            || controllers
                .split(',')
                .any(|c| c.is_empty() || c == "memory" || c.contains(char::is_whitespace))
        {
            return Err(invalid());
        }
    }
    let path = unified_path.ok_or_else(invalid)?;
    if pid == 0
        || !root.is_absolute()
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
                "{label} cannot resolve {}: {error}",
                path.display()
            ))
        })
    };
    let root = resolve(root)?;
    verify_cgroup_root(label, &root, mounts)?;
    let group = resolve(&root.join(path.strip_prefix("/").map_err(|_| invalid())?))?;
    if !group.starts_with(&root) {
        return Err(invalid());
    }
    if group != root && read_admission_file(label, &group.join("cgroup.type"))?.trim() != "domain" {
        return Err(invalid());
    }
    let mut found = false;
    for owner in read_admission_file(label, &group.join("cgroup.procs"))?.split_whitespace() {
        found |= owner.parse::<u32>().map_err(|_| invalid())? == pid;
    }
    if !found {
        return Err(invalid());
    }
    Ok(group)
}

#[cfg(test)]
mod tests {
    const LABEL: &str = "the caller";
    const ROOT_LABEL: &str = "the caller";
    use super::*;

    fn refusal_text<T>(result: Result<T, EngError>) -> String {
        match result {
            Err(EngError::MemoryPlan(message)) => message,
            Err(other) => panic!("unexpected refusal variant: {other:?}"),
            Ok(_) => panic!("expected a refusal"),
        }
    }

    #[test]
    fn refusal_texts_carry_the_neutral_label_and_name_no_family() {
        let no_family = |message: &str| {
            assert!(message.contains(LABEL), "{message}");
            assert!(
                !message.contains("Flint") && !message.contains("Flash"),
                "{message}"
            );
        };
        // Every reachable refusal, produced, not quoted from the source.
        no_family(&refusal_text(HostAdmission::from_meminfo(
            LABEL,
            "MemTotal: 1 kB\nMemAvailable: 2 kB\n",
            None,
        )));
        no_family(&refusal_text(meminfo_bytes(LABEL, "junk", "MemAvailable:")));
        no_family(&refusal_text(cgroup_remaining_bytes(LABEL, "bogus", "0")));
        no_family(&refusal_text(read_admission_file(
            LABEL,
            std::path::Path::new("/nonexistent-admission-file"),
        )));
        no_family(&refusal_text(cgroup_remaining_in(
            LABEL,
            std::path::Path::new("/sys/fs/cgroup"),
            std::path::Path::new("/var"),
        )));
        no_family(&refusal_text(verify_cgroup_root(
            LABEL,
            std::path::Path::new("/definitely/not/here"),
            "",
        )));
        let (_root, _keep, directory, proc_dir) = namespace_root("max", "memory\n");
        no_family(&refusal_text(cgroup_from_proc(
            LABEL,
            &directory,
            "0::\nbroken",
            "",
            42,
        )));
        let mounts = std::fs::read_to_string(proc_dir.join("self/mountinfo")).unwrap();
        let looped = directory.join("cgroup.type");
        std::fs::remove_file(&looped).unwrap();
        std::os::unix::fs::symlink("cgroup.type", &looped).unwrap();
        no_family(&refusal_text(verify_cgroup_root(
            LABEL, &directory, &mounts,
        )));
        // The no-ancestor refusal is structurally unreachable: a cgroup that
        // starts_with(root) always walks up to root, so that branch stays
        // untriggered by design.
        // The root-label line only fires in the window between
        // cgroup_from_proc's canonicalize and probe_in's own, so it is
        // pinned through root_error rather than produced end to end.
        assert_eq!(
            refusal_text::<()>(Err(root_error(ROOT_LABEL, std::io::Error::other("x")))),
            "the caller cgroup root: x"
        );
        no_family(&refusal_text(cgroup_from_proc(
            LABEL,
            &directory,
            "0::/missing-group\n",
            &std::fs::read_to_string(proc_dir.join("self/mountinfo")).unwrap(),
            42,
        )));
        assert_eq!(
            refusal_text(HostAdmission::from_meminfo(
                LABEL,
                "MemTotal: 1 kB\nMemAvailable: 2 kB\n",
                None
            )),
            "the caller memory snapshot has inconsistent totals."
        );
        assert_eq!(
            refusal_text(meminfo_bytes(LABEL, "junk", "MemAvailable:")),
            "the caller needs one valid MemAvailable: value in kB from /proc/meminfo."
        );
        assert_eq!(
            refusal_text(cgroup_remaining_bytes(LABEL, "bogus", "0")),
            "the caller needs memory.max as max or an unsigned byte count, and memory.current as an unsigned byte count."
        );
        assert!(
            refusal_text(read_admission_file(
                LABEL,
                std::path::Path::new("/nonexistent-admission-file")
            ))
            .starts_with("the caller cannot read /nonexistent-admission-file: ")
        );
        assert_eq!(
            refusal_text(cgroup_remaining_in(
                LABEL,
                std::path::Path::new("/sys/fs/cgroup"),
                std::path::Path::new("/var")
            )),
            "the caller cgroup must be inside its verified controller root."
        );
        assert_eq!(
            refusal_text(verify_cgroup_root(
                LABEL,
                std::path::Path::new("/definitely/not/here"),
                ""
            )),
            "the caller needs one complete cgroup2 root with the memory controller."
        );
        assert_eq!(
            refusal_text(cgroup_from_proc(
                LABEL,
                std::path::Path::new("/sys/fs/cgroup"),
                "garbage",
                "",
                42
            )),
            "the caller cannot map its process inside a complete unified cgroup tree."
        );
    }

    #[test]
    fn flint_label_refusals_keep_the_frozen_texts_in_full() {
        const FLINT: &str = "Flint expert cache";
        assert_eq!(
            refusal_text(HostAdmission::from_meminfo(
                FLINT,
                "MemTotal: 1 kB\nMemAvailable: 2 kB\n",
                None
            )),
            "Flint expert cache memory snapshot has inconsistent totals."
        );
        assert_eq!(
            refusal_text(cgroup_remaining_in(
                FLINT,
                std::path::Path::new("/sys/fs/cgroup"),
                std::path::Path::new("/var")
            )),
            "Flint expert cache cgroup must be inside its verified controller root."
        );
        assert_eq!(
            refusal_text(verify_cgroup_root(
                FLINT,
                std::path::Path::new("/definitely/not/here"),
                ""
            )),
            "Flint expert cache needs one complete cgroup2 root with the memory controller."
        );
        assert_eq!(
            refusal_text(cgroup_from_proc(
                FLINT,
                std::path::Path::new("/sys/fs/cgroup"),
                "garbage",
                "",
                42
            )),
            "Flint expert cache cannot map its process inside a complete unified cgroup tree."
        );
        assert_eq!(
            refusal_text::<()>(Err(root_error("Flint", std::io::Error::other("x")))),
            "Flint cgroup root: x"
        );
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
            let result = verify_cgroup_root(LABEL, &directory, &mounts);
            eprintln!("captured view {name}: accepted={}", result.is_ok());
            if accepted {
                assert!(result.is_ok(), "{name}: {result:?}");
            } else {
                assert!(matches!(result, Err(EngError::MemoryPlan(message))
                    if message.contains("one complete cgroup2 root")));
            }
        }
    }

    #[cfg(all(feature = "native-cuda", target_os = "linux"))]
    #[test]
    fn an_unverified_budget_falls_back_to_the_bare_figure() {
        let refusal = EngError::MemoryPlan(String::from("admission: cannot map its process"));
        let (total, available) = HostAdmission::bare_fallback(
            LABEL,
            ROOT_LABEL,
            &refusal,
            "MemTotal: 33554432 kB\nMemAvailable: 25165824 kB\n",
        )
        .unwrap();
        // The bare figures: machine-wide, no cgroup minimum applied.
        assert_eq!((total, available), (32u64 << 30, 24u64 << 30));
    }

    #[test]
    fn captured_jetson_layout_refuses_cleanly_to_the_bare_fallback() {
        // The Xavier (L4T) runs hybrid cgroup v1 with a controller-less unified tree: the
        // membership's first line is a v1 controller line, not `0::`, and no memory limit
        // lives anywhere the unified reader looks. The capture pins that this layout refuses
        // with the clean "cannot map" error — the refusal `probe_or_bare` turns into the
        // bare `/proc/meminfo` fallback — before any filesystem access at all.
        let jetson: serde_json::Value =
            serde_json::from_str(include_str!("testdata/cgroup-jetson-20261118.json")).unwrap();
        let view = &jetson["jetson"];
        assert_eq!(view["unified_controllers"].as_str().unwrap(), "");
        let root = tempfile::tempdir().unwrap();
        let directory = root.path().canonicalize().unwrap();
        let result = cgroup_from_proc(
            LABEL,
            &directory,
            view["membership"].as_str().unwrap(),
            view["mountinfo"].as_str().unwrap(),
            std::process::id(),
        );
        assert!(
            matches!(&result, Err(EngError::MemoryPlan(message))
                if message.contains("cannot map its process")),
            "the Jetson layout must refuse, not resolve or panic: {result:?}"
        );
        // The refusal happens before the tree is read: the untouched directory proves it.
        assert_eq!(std::fs::read_dir(&directory).unwrap().count(), 0);
        eprintln!("captured jetson view: refused cleanly, bare fallback applies");
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
        let result = cgroup_from_proc(LABEL, &directory, membership, &mounts, pid);
        assert_eq!(result.unwrap(), group);
        eprintln!("captured host member: mapped=true pid={pid}");
    }

    /// The canonical controller root and its member group, plus the mounts that cover it.
    /// Paths are canonical so the strict ParentDir refusal in `cgroup_from_proc` judges the
    /// fixture tree, never the caller's `TMPDIR` spelling.
    fn discovery_fixture() -> (
        tempfile::TempDir,
        std::path::PathBuf,
        std::path::PathBuf,
        String,
    ) {
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
        (root, directory, group, mounts)
    }

    #[test]
    fn cgroup_discovery_resolves_own_pid_in_global_tree() {
        let (_root, directory, group, mounts) = discovery_fixture();
        let resolved = cgroup_from_proc(LABEL, &directory, "0::/work\n", &mounts, 42);
        assert!(
            resolved.is_ok(),
            "a complete global cgroup tree must resolve its own process group"
        );
        assert_eq!(resolved.unwrap(), group);
    }

    #[cfg(unix)]
    #[test]
    fn cgroup_discovery_matches_canonical_mount_paths() {
        let (_root, directory, group, mounts) = discovery_fixture();
        let aliases = tempfile::tempdir().unwrap();
        let alias = aliases
            .path()
            .canonicalize()
            .unwrap()
            .join("controller-root");
        std::os::unix::fs::symlink(&directory, &alias).unwrap();
        let resolved = cgroup_from_proc(LABEL, &alias, "0::/work\n", &mounts, 42).unwrap();
        assert_eq!(resolved, group);
    }

    #[cfg(unix)]
    #[test]
    fn a_root_spelled_through_parent_components_is_refused() {
        // The strictness is the point: a root that merely RESOLVES to the fixture
        // tree but is spelled with ".." (a TMPDIR like "/x/wt/../tmp") is refused
        // before canonicalization, so an ambiguous tree never becomes a budget.
        let (_root, directory, _group, mounts) = discovery_fixture();
        let dotted = directory.join("..").join(directory.file_name().unwrap());
        // The dotted spelling must resolve to the very fixture tree, so the
        // refusal below can only come from the strict ParentDir check.
        assert_eq!(
            std::fs::canonicalize(&dotted).unwrap(),
            std::fs::canonicalize(&directory).unwrap()
        );
        let result = cgroup_from_proc(LABEL, &dotted, "0::/work\n", &mounts, 42);
        assert!(
            matches!(&result, Err(EngError::MemoryPlan(message))
                if message.contains("cannot map its process")),
            "a root spelled with .. must be refused, got {result:?}"
        );
    }

    #[test]
    fn cgroup_discovery_rejects_hidden_or_ambiguous_roots() {
        for (name, contents) in [
            ("cgroup.type", "domain\n"),
            ("memory.max", "max\n"),
            ("memory.current", "0\n"),
            ("cgroup.controllers", "cpu\n"),
        ] {
            let (_root, directory, _, mounts) = discovery_fixture();
            std::fs::write(directory.join(name), contents).unwrap();
            let result = cgroup_from_proc(LABEL, &directory, "0::/work\n", &mounts, 42);
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
            let (_root, directory, group, _) = discovery_fixture();
            let mounts = format!(
                "36 29 0:33 {source} {} rw - {filesystem} cgroup rw\n",
                group.parent().unwrap().display()
            )
            .repeat(copies);
            let result = cgroup_from_proc(LABEL, &directory, "0::/work\n", &mounts, 42);
            assert!(
                matches!(result, Err(EngError::MemoryPlan(message))
                    if message.contains("complete cgroup2 root")),
                "must reject mount {source}/{filesystem}/{copies}"
            );
        }
    }

    /// A container's view: the cgroup root is the container's own cgroup, with limit files.
    /// Both returned paths are canonical so the strict ParentDir refusals judge the fixture
    /// tree, never the caller's `TMPDIR` spelling; the TempDirs only keep the trees alive.
    fn namespace_root(
        maximum: &str,
        controllers: &str,
    ) -> (
        tempfile::TempDir,
        tempfile::TempDir,
        std::path::PathBuf,
        std::path::PathBuf,
    ) {
        let root = tempfile::tempdir().unwrap();
        let directory = root.path().canonicalize().unwrap();
        for (name, contents) in [
            ("cgroup.controllers", controllers),
            ("cgroup.subtree_control", ""),
            ("cgroup.type", "domain\n"),
            ("cgroup.procs", "1\n42\n"),
            ("memory.max", maximum),
            ("memory.current", "2147483648\n"),
        ] {
            std::fs::write(directory.join(name), contents).unwrap();
        }
        let proc = tempfile::tempdir().unwrap();
        let proc_dir = proc.path().canonicalize().unwrap();
        std::fs::create_dir(proc_dir.join("self")).unwrap();
        std::fs::write(
            proc_dir.join("meminfo"),
            "MemTotal: 33554432 kB\nMemAvailable: 25165824 kB\n",
        )
        .unwrap();
        std::fs::write(proc_dir.join("self/cgroup"), "0::/\n").unwrap();
        std::fs::write(
            proc_dir.join("self/mountinfo"),
            format!(
                "36 29 0:33 / {} rw - cgroup2 cgroup rw\n",
                directory.display()
            ),
        )
        .unwrap();
        (root, proc, directory, proc_dir)
    }

    #[test]
    fn a_namespace_root_applies_its_own_limit_or_falls_back_to_available_memory() {
        let (_root, _proc, directory, proc_dir) = namespace_root("8589934592\n", "memory\n");
        let budget = HostAdmission::probe_in(LABEL, ROOT_LABEL, &proc_dir, &directory, 42).unwrap();
        assert_eq!(budget.cgroup_remaining, Some(6u64 << 30));
        assert_eq!(budget.os_floor, 8u64 << 30);
        // 6 GiB left in the cgroup, 8 GiB floor: nothing is admitted.
        assert!(budget.admit(LABEL, 1).is_err());
        let (_root, _proc, directory, proc_dir) = namespace_root("max\n", "memory\n");
        let budget = HostAdmission::probe_in(LABEL, ROOT_LABEL, &proc_dir, &directory, 42).unwrap();
        assert_eq!(budget.cgroup_remaining, None);
        // MemAvailable is 24 GiB; the floor is max(8 GiB, 10% of 32 GiB) = 8 GiB.
        assert!(budget.admit(LABEL, 16u64 << 30).is_ok());
        assert!(budget.admit(LABEL, (16u64 << 30) + 1).is_err());
    }

    #[test]
    fn a_root_with_some_but_not_all_namespace_files_is_refused() {
        let files = [
            ("cgroup.type", "domain\n"),
            ("memory.max", "max\n"),
            ("memory.current", "0\n"),
        ];
        for skipped in 0..files.len() {
            let (_root, directory, _group, mounts) = discovery_fixture();
            for (index, (name, contents)) in files.iter().enumerate() {
                if index != skipped {
                    std::fs::write(directory.join(name), contents).unwrap();
                }
            }
            let result = cgroup_from_proc(LABEL, &directory, "0::/work\n", &mounts, 42);
            assert!(
                matches!(result, Err(EngError::MemoryPlan(message))
                    if message.contains("complete cgroup2 root")),
                "two of three root files without {} must be refused",
                files[skipped].0
            );
        }
    }

    #[test]
    fn a_namespace_root_without_the_memory_controller_is_refused_clearly() {
        let (_root, _proc, directory, proc_dir) = namespace_root("max\n", "cpu\n");
        let result = HostAdmission::probe_in(LABEL, ROOT_LABEL, &proc_dir, &directory, 42);
        assert!(matches!(result, Err(EngError::MemoryPlan(message))
                if message.contains("complete cgroup2 root with the memory controller")));
    }

    #[test]
    fn hybrid_membership_uses_the_unified_memory_budget_in_either_order() {
        let (_root, _proc, directory, proc_dir) = namespace_root("8589934592\n", "memory\n");
        for membership in [
            "1:net_cls:/\n0::/\n",
            "0::/\n2:cpu,cpuacct:/work:bench\n",
            // A v1 line for a different controller whose path differs from the unified
            // path must not change which cgroup's limit is read.
            "0::/\n1:devices:/elsewhere\n",
        ] {
            std::fs::write(proc_dir.join("self/cgroup"), membership).unwrap();
            let budget =
                HostAdmission::probe_in(LABEL, ROOT_LABEL, &proc_dir, &directory, 42).unwrap();
            // Machine-wide MemAvailable is 24 GiB, but only 6 GiB remain in this group.
            assert_eq!(budget.available_budget(), 6u64 << 30);
        }
    }

    #[test]
    fn cgroup_discovery_rejects_bad_process_mapping() {
        for (membership, pid, owners, kind) in [
            ("0::/../work\n", 42, "41\n42\n", "domain"),
            ("0::work\n", 42, "41\n42\n", "domain"),
            ("0::/work\n0::/work\n", 42, "41\n42\n", "domain"),
            ("1:memory:/work\n", 42, "41\n42\n", "domain"),
            ("0::/work\n1:cpu,memory:/\n", 42, "42\n", "domain"),
            ("0::/work\n0:net_cls:/\n", 42, "42\n", "domain"),
            ("0::/work\n1::/\n", 42, "42\n", "domain"),
            ("0::/work\n1:cpu,:/\n", 42, "42\n", "domain"),
            ("0::/work\nx:cpu:/\n", 42, "42\n", "domain"),
            ("0::/work\n1:net_cls:relative\n", 42, "42\n", "domain"),
            ("0::/work\n", 0, "41\n42\n", "domain"),
            ("0::/work\n", 42, "41\n", "domain"),
            ("0::/work\n", 42, "41\n42\n", "threaded"),
            ("0::/work\n", 42, "41\n42\nnot-a-pid\n", "domain"),
        ] {
            let (_root, directory, group, mounts) = discovery_fixture();
            std::fs::write(group.join("cgroup.procs"), owners).unwrap();
            std::fs::write(group.join("cgroup.type"), kind).unwrap();
            let result = cgroup_from_proc(LABEL, &directory, membership, &mounts, pid);
            assert!(
                matches!(result, Err(EngError::MemoryPlan(message))
                    if message.contains("cannot map its process")),
                "must reject process mapping {membership:?}/{pid}/{owners:?}/{kind}"
            );
        }
    }

    #[test]
    fn cgroup_reader_rejects_missing_current_with_unlimited_max() {
        let _root = tempfile::tempdir().unwrap();
        let directory = _root.path().canonicalize().unwrap();
        let child = directory.join("child");
        std::fs::create_dir(&child).unwrap();
        std::fs::write(directory.join("cgroup.controllers"), "memory\n").unwrap();
        std::fs::write(directory.join("cgroup.subtree_control"), "memory\n").unwrap();
        std::fs::write(child.join("memory.max"), "max\n").unwrap();
        let result = cgroup_remaining_in(LABEL, &directory, &child);
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
        let _root = tempfile::tempdir().unwrap();
        let directory = _root.path().canonicalize().unwrap();
        let parent = directory.join("parent");
        let child = parent.join("child");
        std::fs::create_dir_all(&child).unwrap();
        std::fs::write(directory.join("cgroup.controllers"), "memory\n").unwrap();
        for dir in [directory.as_path(), parent.as_path()] {
            std::fs::write(dir.join("cgroup.subtree_control"), "memory\n").unwrap();
        }
        for (dir, maximum, current) in [
            (&parent, "17179869184\n", "4294967296\n"),
            (&child, "68719476736\n", "1073741824\n"),
        ] {
            std::fs::write(dir.join("memory.max"), maximum).unwrap();
            std::fs::write(dir.join("memory.current"), current).unwrap();
        }
        let remaining = cgroup_remaining_in(LABEL, &directory, &child);
        assert!(
            remaining.is_ok(),
            "a readable hierarchy must honor finite ancestor charges"
        );
        assert_eq!(remaining.unwrap(), Some(12u64 << 30));
    }

    #[test]
    fn cgroup_snapshot_accounts_for_current_charges() {
        let remaining = cgroup_remaining_bytes(LABEL, "17179869184\n", "4294967296\n");
        assert!(
            remaining.is_ok(),
            "valid finite cgroup limit and current charges must produce a budget"
        );
        let remaining = remaining.unwrap();
        assert_eq!(remaining, Some(12u64 << 30));
        let budget = HostAdmission::from_meminfo(
            LABEL,
            "MemTotal: 33554432 kB\nMemAvailable: 25165824 kB\n",
            remaining,
        )
        .unwrap();
        assert!(budget.admit(LABEL, 4u64 << 30).is_ok());
        assert!(budget.admit(LABEL, (4u64 << 30) + 1).is_err());
    }

    #[test]
    fn available_budget_is_the_cgroup_minimum_in_either_order() {
        // The plan-facing accessor must hand a plan the same number the guard refuses against:
        // MemAvailable clamped by the cgroup remaining bytes, whichever side is smaller, and
        // the bare host figure when the cgroup is unlimited.
        let unlimited = HostAdmission::from_meminfo(
            LABEL,
            "MemTotal: 67108864 kB\nMemAvailable: 25165824 kB\n",
            None,
        )
        .unwrap();
        assert_eq!(unlimited.available_budget(), 24 << 30);
        assert_eq!(unlimited.host_memory(), (64 << 30, 24 << 30));

        let cgroup_smaller = HostAdmission::from_meminfo(
            LABEL,
            "MemTotal: 67108864 kB\nMemAvailable: 25165824 kB\n",
            Some(12 << 30),
        )
        .unwrap();
        assert_eq!(cgroup_smaller.available_budget(), 12 << 30);

        let host_smaller = HostAdmission::from_meminfo(
            LABEL,
            "MemTotal: 67108864 kB\nMemAvailable: 8388608 kB\n",
            Some(12 << 30),
        )
        .unwrap();
        assert_eq!(host_smaller.available_budget(), 8 << 30);

        // One budget for plan and guard: the last byte the guard admits is the accessor's
        // figure minus the OS floor (8 GiB of the 64 GiB host).
        assert!(cgroup_smaller.admit(LABEL, 4u64 << 30).is_ok());
        assert!(cgroup_smaller.admit(LABEL, (4u64 << 30) + 1).is_err());
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
                cgroup_remaining_bytes(LABEL, maximum, current).unwrap(),
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
                cgroup_remaining_bytes(LABEL, maximum, current).is_err(),
                "{maximum:?} / {current:?}"
            );
        }
    }

    #[test]
    fn host_snapshot_uses_mem_available_and_model_scaled_os_floor() {
        let result = HostAdmission::from_meminfo(
            LABEL,
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
        assert!(budget.admit(LABEL, 256 * 47 * 1_603_584u64).is_ok());
    }

    #[test]
    fn host_snapshot_preserves_minimum_floor_and_finite_cgroup() {
        let text = "MemAvailable: 25165824 kB\nMemFree: 0 kB\nMemTotal: 33554432 kB\n";
        let floor = 8u64 << 30;
        let budget = HostAdmission::from_meminfo(LABEL, text, Some(floor + 1)).unwrap();
        assert_eq!(budget.os_floor, floor);
        assert_eq!(budget.cgroup_remaining, Some(floor + 1));
        assert!(budget.admit(LABEL, 1).is_ok());
        assert!(budget.admit(LABEL, 2).is_err());
        assert!(
            HostAdmission::from_meminfo(LABEL, text, Some(0))
                .unwrap()
                .admit(LABEL, 0)
                .is_err()
        );
        let empty =
            HostAdmission::from_meminfo(LABEL, "MemTotal: 1 kB\nMemAvailable: 0 kB\n", None)
                .unwrap();
        assert_eq!(empty.mem_available, 0);
        assert!(empty.admit(LABEL, 0).is_err());
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
            assert!(
                HostAdmission::from_meminfo(LABEL, text, None).is_err(),
                "{text:?}"
            );
        }
    }

    #[test]
    fn host_budget_rejects_unrepresentable_payload_plus_floor() {
        let budget = HostAdmission {
            mem_total: u64::MAX,
            mem_available: u64::MAX,
            cgroup_remaining: None,
            os_floor: 1,
        };
        assert!(budget.admit(LABEL, u64::MAX).is_err());
        assert!(budget.admit(LABEL, u64::MAX - 1).is_ok());
    }

    #[test]
    fn host_budget_accepts_exact_model_demand_only_after_os_floor() {
        let required = 256 * 47 * 1_603_584u64;
        let floor = 8u64 << 30;
        let exact = HostAdmission {
            mem_total: required + floor,
            mem_available: required + floor,
            cgroup_remaining: Some(required + floor),
            os_floor: floor,
        };
        assert!(
            exact.admit(LABEL, required).is_ok(),
            "exact model demand plus OS floor must admit"
        );
        assert!(
            HostAdmission {
                mem_available: exact.mem_available - 1,
                ..exact
            }
            .admit(LABEL, required)
            .is_err()
        );
        assert!(
            HostAdmission {
                cgroup_remaining: Some(required + floor - 1),
                ..exact
            }
            .admit(LABEL, required)
            .is_err()
        );
        assert!(
            HostAdmission {
                cgroup_remaining: None,
                ..exact
            }
            .admit(LABEL, required)
            .is_ok(),
            "an unlimited cgroup still obeys host availability and the OS floor"
        );
    }

    #[test]
    fn composed_host_probe_uses_verified_tree_and_fails_closed() {
        let (_root, directory, group, mounts) = discovery_fixture();
        let proc = tempfile::tempdir().unwrap();
        let proc_dir = proc.path().canonicalize().unwrap();
        std::fs::create_dir(proc_dir.join("self")).unwrap();
        std::fs::write(
            proc_dir.join("meminfo"),
            "MemTotal: 33554432 kB\nMemAvailable: 25165824 kB\n",
        )
        .unwrap();
        std::fs::write(proc_dir.join("self/cgroup"), "0::/work\n").unwrap();
        std::fs::write(proc_dir.join("self/mountinfo"), mounts).unwrap();
        std::fs::write(directory.join("cgroup.subtree_control"), "memory\n").unwrap();
        std::fs::write(group.join("memory.max"), "17179869184\n").unwrap();
        std::fs::write(group.join("memory.current"), "1073741824\n").unwrap();
        let budget = HostAdmission::probe_in(LABEL, ROOT_LABEL, &proc_dir, &directory, 42).unwrap();
        assert_eq!(budget.cgroup_remaining, Some(15u64 << 30));
        assert!(budget.admit(LABEL, 7u64 << 30).is_ok());
        assert!(budget.admit(LABEL, (7u64 << 30) + 1).is_err());
        std::fs::remove_file(group.join("memory.current")).unwrap();
        assert!(HostAdmission::probe_in(LABEL, ROOT_LABEL, &proc_dir, &directory, 42).is_err());
    }
}
