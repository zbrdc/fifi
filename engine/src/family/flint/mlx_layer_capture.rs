//! Test-only Flint hidden capture; no backend imports.
//!
//! Records receive selected host F32LE rows, not the original device storage.
//! The manifest retains the original dtype separately from that byte layout.

use sha2::{Digest, Sha256};
use std::{
    cell::{Cell, RefCell},
    collections::BTreeMap,
    fs::{self, OpenOptions},
    io::{self, Write},
    ops::Range,
    path::{Path, PathBuf},
    rc::Rc,
    sync::atomic::{AtomicU64, Ordering},
};

pub(super) const STAGES: &[&str] = &[
    "input",
    "norm-input",
    "q-projection",
    "k-projection",
    "v-projection",
    "q-post",
    "k-post",
    "attention-out",
    "attention-projection",
    "post-attention-residual",
    "norm-post-attention",
    "ffn-output",
    "output",
];

fn invalid(message: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, message)
}

pub(super) fn selected_rows(pos: u32, rows: u32) -> io::Result<Range<usize>> {
    let end = pos
        .checked_add(rows)
        .ok_or_else(|| invalid("position overflow"))?;
    let start = pos.max(2047);
    let end = end.min(2080);
    if start >= end {
        return Ok(0..0);
    }
    Ok(
        usize::try_from(start - pos).map_err(|_| invalid("row offset overflow"))?
            ..usize::try_from(end - pos).map_err(|_| invalid("row offset overflow"))?,
    )
}

struct Capture {
    file: String,
    dtype: String,
    rows: u32,
    width: u32,
}

struct Session {
    directory: PathBuf,
    pauses: Rc<Cell<usize>>,
    captures: BTreeMap<(usize, usize), Capture>,
    failed: bool,
}

thread_local! {
    static SESSION: RefCell<Option<Session>> = const { RefCell::new(None) };
}

pub(super) fn active() -> bool {
    SESSION.with(|slot| slot.borrow().as_ref().is_some_and(|s| s.pauses.get() == 0))
}

pub(super) struct Scope(Rc<Cell<usize>>);

impl Scope {
    pub(super) fn enter(base: &Path, label: &str) -> io::Result<Self> {
        if label.is_empty()
            || label == "."
            || label == ".."
            || !label
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || b"-_.".contains(&b))
        {
            return Err(invalid("label must be one safe path component"));
        }
        SESSION.with(|slot| {
            let mut slot = slot.borrow_mut();
            if slot.is_some() {
                return Err(invalid("capture scope already active"));
            }
            static NEXT: AtomicU64 = AtomicU64::new(0);
            let directory = loop {
                let path = base.join(format!(
                    "{label}-{}-{}",
                    std::process::id(),
                    NEXT.fetch_add(1, Ordering::Relaxed)
                ));
                match fs::create_dir(&path) {
                    Ok(()) => break path,
                    Err(error) if error.kind() == io::ErrorKind::AlreadyExists => continue,
                    Err(error) => return Err(error),
                }
            };
            fs::write(
                directory.join("manifest.json"),
                b"{\"complete\":false,\"captures\":[]}",
            )?;
            let pauses = Rc::new(Cell::new(0));
            *slot = Some(Session {
                directory,
                pauses: pauses.clone(),
                captures: BTreeMap::new(),
                failed: false,
            });
            Ok(Self(pauses))
        })
    }

    pub(super) fn finish(self) -> io::Result<serde_json::Value> {
        let session = SESSION.with(|slot| {
            let mut slot = slot.borrow_mut();
            if !slot
                .as_ref()
                .is_some_and(|s| Rc::ptr_eq(&s.pauses, &self.0))
            {
                return Err(invalid("capture scope is not current"));
            }
            Ok(slot.take().expect("checked current scope"))
        })?;
        if session.failed || session.captures.len() != 49 * STAGES.len() {
            return Err(invalid("capture is incomplete or has failed writes"));
        }
        let mut captures = Vec::with_capacity(session.captures.len());
        for ((layer, stage), capture) in &session.captures {
            if capture.rows != 33 {
                return Err(invalid("capture requires all 33 ordered rows"));
            }
            let raw = fs::read(session.directory.join(&capture.file))?;
            let expected = usize::try_from(capture.width)
                .ok()
                .and_then(|width| width.checked_mul(33))
                .and_then(|elements| elements.checked_mul(4))
                .ok_or_else(|| invalid("capture byte count overflow"))?;
            if raw.len() != expected {
                return Err(invalid("capture file length changed"));
            }
            captures.push(serde_json::json!({
                "layer": layer,
                "stage": STAGES[*stage],
                "file": capture.file,
                "sha256": format!("{:x}", Sha256::digest(&raw)),
                "original_dtype": capture.dtype,
                "rows": capture.rows,
                "width": capture.width,
                "shape": [capture.rows, capture.width],
            }));
        }
        let manifest = serde_json::json!({
            "complete": true, "logical_positions": [2047, 2080],
            "dtype": "F32LE", "captures": captures
        });
        let pending = session.directory.join("manifest.pending");
        fs::write(&pending, serde_json::to_vec_pretty(&manifest)?)?;
        fs::rename(pending, session.directory.join("manifest.json"))?;
        Ok(manifest)
    }
}

impl Drop for Scope {
    fn drop(&mut self) {
        SESSION.with(|slot| {
            let mut slot = slot.borrow_mut();
            if slot
                .as_ref()
                .is_some_and(|s| Rc::ptr_eq(&s.pauses, &self.0))
            {
                *slot = None;
            }
        });
    }
}

pub(super) struct Pause(Option<Rc<Cell<usize>>>);

pub(super) fn suspend() -> Pause {
    Pause(SESSION.with(|slot| {
        slot.borrow().as_ref().map(|session| {
            let pauses = session.pauses.clone();
            pauses.set(pauses.get().checked_add(1).expect("capture pause overflow"));
            pauses
        })
    }))
}

impl Drop for Pause {
    fn drop(&mut self) {
        if let Some(pauses) = &self.0 {
            pauses.set(pauses.get() - 1);
        }
    }
}

pub(super) fn record(
    layer: usize,
    pos: u32,
    stage: &str,
    dtype: &str,
    shape: &[u32],
    bytes: &[u8],
) -> io::Result<()> {
    SESSION.with(|slot| {
        let mut slot = slot.borrow_mut();
        let Some(session) = slot.as_mut().filter(|s| s.pauses.get() == 0) else {
            return Ok(());
        };
        if session.failed {
            return Err(invalid("capture has failed writes"));
        }
        let stage = STAGES
            .iter()
            .position(|&known| known == stage)
            .ok_or_else(|| invalid("unknown capture stage"))?;
        if layer >= 49 || shape.len() != 2 || shape[0] == 0 || shape[1] == 0 {
            return Err(invalid(
                "capture requires a valid layer and nonempty rank-two layout",
            ));
        }
        let end = pos
            .checked_add(shape[0])
            .ok_or_else(|| invalid("position overflow"))?;
        if pos < 2047 || end > 2080 {
            return Err(invalid("capture rows must be within 2047..2080"));
        }
        let expected = usize::try_from(shape[0])
            .ok()
            .zip(usize::try_from(shape[1]).ok())
            .and_then(|(rows, width)| rows.checked_mul(width))
            .and_then(|elements| elements.checked_mul(4))
            .ok_or_else(|| invalid("capture byte count overflow"))?;
        if bytes.len() != expected || dtype.is_empty() || dtype.chars().any(char::is_control) {
            return Err(invalid("invalid capture bytes or original dtype"));
        }
        if bytes
            .as_chunks::<4>()
            .0
            .iter()
            .any(|v| !f32::from_le_bytes(*v).is_finite())
        {
            return Err(invalid("capture values must be finite F32LE"));
        }
        let key = (layer, stage);
        let previous = session.captures.get(&key);
        let rows = previous.map_or(0, |capture| capture.rows);
        let next_pos = 2047_u32
            .checked_add(rows)
            .ok_or_else(|| invalid("position overflow"))?;
        if pos != next_pos
            || previous.is_some_and(|capture| capture.width != shape[1] || capture.dtype != dtype)
        {
            return Err(invalid(
                "capture rows must be ordered with constant width and dtype",
            ));
        }
        let file = format!("layer-{layer:02}-{}.f32le", STAGES[stage]);
        let written = (|| -> io::Result<()> {
            let mut options = OpenOptions::new();
            options.write(true);
            if previous.is_some() {
                options.append(true);
            } else {
                options.create_new(true);
            }
            options
                .open(session.directory.join(&file))?
                .write_all(bytes)
        })();
        if let Err(error) = written {
            session.failed = true;
            return Err(error);
        }
        session.captures.insert(
            key,
            Capture {
                file,
                dtype: dtype.to_owned(),
                rows: end - 2047,
                width: shape[1],
            },
        );
        Ok(())
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use sha2::{Digest, Sha256};
    use std::{
        fs,
        path::PathBuf,
        sync::atomic::{AtomicU64, Ordering},
    };

    struct Scratch(PathBuf);

    impl Scratch {
        fn new() -> Self {
            static NEXT: AtomicU64 = AtomicU64::new(0);
            let base = PathBuf::from(std::env::var_os("TMPDIR").expect("set task TMPDIR"));
            let path = base.join(format!(
                "flint-layer-capture-test-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            fs::create_dir(&path).expect("create unique test scratch");
            Self(path)
        }

        fn session(&self) -> PathBuf {
            let dirs: Vec<_> = fs::read_dir(&self.0)
                .unwrap()
                .map(|entry| entry.unwrap().path())
                .filter(|path| path.is_dir())
                .collect();
            assert_eq!(dirs.len(), 1, "one new session directory");
            dirs[0].clone()
        }
    }

    impl Drop for Scratch {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn values(rows: usize, width: usize) -> Vec<u8> {
        (0..rows * width)
            .flat_map(|index| (index as f32 + 0.5).to_le_bytes())
            .collect()
    }

    #[test]
    fn selected_rows_boundaries() {
        assert_eq!(selected_rows(1536, 512).unwrap(), 511..512);
        assert_eq!(selected_rows(2047, 1).unwrap(), 0..1);
        assert_eq!(selected_rows(2079, 1).unwrap(), 0..1);
        assert!(selected_rows(2080, 1).unwrap().is_empty());
        assert!(selected_rows(0, 512).unwrap().is_empty());
        assert_eq!(selected_rows(2040, 48).unwrap(), 7..40);
        assert!(selected_rows(2047, 0).unwrap().is_empty());
    }

    #[test]
    fn inactive_calls_do_not_validate_or_write() {
        assert!(!active());
        record(usize::MAX, u32::MAX, "../invalid", "invalid", &[], &[]).unwrap();
        {
            let _pause = suspend();
            assert!(!active());
        }
        assert!(!active());
    }

    #[test]
    fn pause_restores_session_and_is_thread_local() {
        let scratch = Scratch::new();
        let scope = Scope::enter(&scratch.0, "pause").unwrap();
        assert!(active());
        assert!(!std::thread::spawn(active).join().unwrap());
        {
            let _outer = suspend();
            assert!(!active());
            {
                let _inner = suspend();
                record(999, 0, "invalid", "invalid", &[], &[]).unwrap();
                assert!(!active());
            }
            assert!(!active());
        }
        assert!(active());
        record(0, 2047, "input", "BF16", &[1, 2], &values(1, 2)).unwrap();
        drop(scope);
        assert!(!active());
    }

    #[test]
    fn labels_are_single_safe_components_and_sessions_are_unique() {
        let scratch = Scratch::new();
        for label in ["", ".", "..", "../escape", "a/b", "a\\b", "/absolute"] {
            assert!(Scope::enter(&scratch.0, label).is_err(), "label {label:?}");
            assert!(!active());
        }
        let first = Scope::enter(&scratch.0, "capture").unwrap();
        let first_path = scratch.session();
        assert!(Scope::enter(&scratch.0, "nested").is_err());
        assert!(active());
        drop(first);
        let second = Scope::enter(&scratch.0, "capture").unwrap();
        let dirs: Vec<_> = fs::read_dir(&scratch.0)
            .unwrap()
            .map(|entry| entry.unwrap().path())
            .collect();
        assert_eq!(dirs.len(), 2);
        assert!(dirs.contains(&first_path));
        drop(second);
        assert!(!active());
    }

    #[test]
    fn writer_rejects_duplicate_gap_width_and_dtype_changes() {
        let scratch = Scratch::new();
        let scope = Scope::enter(&scratch.0, "ordered").unwrap();
        let row = values(1, 2);
        record(0, 2047, "input", "BF16", &[1, 2], &row).unwrap();
        assert!(record(0, 2047, "input", "BF16", &[1, 2], &row).is_err());
        assert!(record(0, 2049, "input", "BF16", &[1, 2], &row).is_err());
        assert!(record(0, 2048, "input", "BF16", &[1, 3], &values(1, 3)).is_err());
        assert!(record(0, 2048, "input", "F32", &[1, 2], &row).is_err());
        record(0, 2048, "input", "BF16", &[1, 2], &row).unwrap();
        assert!(scope.finish().is_err());
        assert!(!active());
    }

    #[test]
    fn writer_rejects_invalid_stage_layer_position_and_layout() {
        let scratch = Scratch::new();
        let scope = Scope::enter(&scratch.0, "invalid").unwrap();
        let row = values(1, 2);
        for stage in ["unknown", "../input", "input/output", ""] {
            assert!(record(0, 2047, stage, "BF16", &[1, 2], &row).is_err());
        }
        assert!(record(49, 2047, "input", "BF16", &[1, 2], &row).is_err());
        for pos in [2046, 2080, u32::MAX] {
            assert!(record(0, pos, "input", "BF16", &[1, 2], &row).is_err());
        }
        for shape in [vec![], vec![2], vec![1, 1, 2], vec![1, 0], vec![0, 2]] {
            assert!(record(0, 2047, "input", "BF16", &shape, &row).is_err());
        }
        assert!(record(0, 2079, "input", "BF16", &[2, 2], &values(2, 2)).is_err());
        assert!(record(0, 2047, "input", "BF16", &[1, 2], &row[..7]).is_err());
        assert!(record(0, 2047, "input", "BF16", &[1, 2], &values(2, 2)).is_err());
        drop(scope);
        assert!(!active());
    }

    #[test]
    fn finish_requires_every_layer_stage_and_all_33_rows() {
        for missing in ["layer", "stage", "row"] {
            let scratch = Scratch::new();
            let scope = Scope::enter(&scratch.0, missing).unwrap();
            for layer in 0..49 {
                for stage in STAGES {
                    if (missing == "layer" && layer == 48)
                        || (missing == "stage" && layer == 48 && *stage == "output")
                    {
                        continue;
                    }
                    let rows = if missing == "row" && layer == 48 && *stage == "output" {
                        32
                    } else {
                        33
                    };
                    record(
                        layer,
                        2047,
                        stage,
                        "BF16",
                        &[rows, 2],
                        &values(rows as usize, 2),
                    )
                    .unwrap();
                }
            }
            let session = scratch.session();
            assert!(scope.finish().is_err(), "missing {missing}");
            assert!(!active());
            let manifest: serde_json::Value =
                serde_json::from_slice(&fs::read(session.join("manifest.json")).unwrap()).unwrap();
            assert_eq!(manifest["complete"], false);
        }
    }

    #[test]
    fn complete_manifest_hashes_one_appended_raw_file_per_layer_stage() {
        let scratch = Scratch::new();
        let scope = Scope::enter(&scratch.0, "complete").unwrap();
        let session = scratch.session();
        let initial: serde_json::Value =
            serde_json::from_slice(&fs::read(session.join("manifest.json")).unwrap()).unwrap();
        assert_eq!(initial["complete"], false);
        let first = values(1, 2);
        let rest = values(32, 2);
        let expected = [first.as_slice(), rest.as_slice()].concat();
        for layer in 0..49 {
            for stage in STAGES {
                record(layer, 2047, stage, "BF16", &[1, 2], &first).unwrap();
                record(layer, 2048, stage, "BF16", &[32, 2], &rest).unwrap();
            }
        }
        let manifest = scope.finish().unwrap();
        assert!(!active());
        assert_eq!(manifest["complete"], true);
        assert_eq!(
            manifest["logical_positions"],
            serde_json::json!([2047, 2080])
        );
        assert_eq!(manifest["dtype"], "F32LE");
        let on_disk: serde_json::Value =
            serde_json::from_slice(&fs::read(session.join("manifest.json")).unwrap()).unwrap();
        assert_eq!(manifest, on_disk);
        let captures = manifest["captures"].as_array().unwrap();
        assert_eq!(captures.len(), 49 * STAGES.len());
        let mut identities = std::collections::HashSet::new();
        let mut files = std::collections::HashSet::new();
        for capture in captures {
            let layer = capture["layer"].as_u64().unwrap();
            let stage = capture["stage"].as_str().unwrap();
            assert!(layer < 49 && STAGES.contains(&stage));
            assert!(identities.insert((layer, stage)));
            assert_eq!(capture["original_dtype"], "BF16");
            assert_eq!(capture["rows"], 33);
            assert_eq!(capture["width"], 2);
            assert_eq!(capture["shape"], serde_json::json!([33, 2]));
            let file = capture["file"].as_str().unwrap();
            assert!(files.insert(file));
            let relative = Path::new(file);
            assert!(!relative.is_absolute());
            assert!(
                relative
                    .components()
                    .all(|part| matches!(part, std::path::Component::Normal(_)))
            );
            let raw = fs::read(session.join(relative)).unwrap();
            assert_eq!(raw, expected);
            assert_eq!(capture["sha256"], format!("{:x}", Sha256::digest(&raw)));
        }
        assert_eq!(
            fs::read_dir(session).unwrap().count(),
            49 * STAGES.len() + 1
        );
    }
}
