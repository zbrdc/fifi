//! The step's deferred prefix lifecycle. This is a child of step, so device owners stay private.
use super::super::{
    book::Health,
    prefix::{CudaSavedPrefix, PendingPrefix, entry_bytes},
    state::{
        device_tip::DeviceTip,
        pinned::{PinnedSnapshot, PrefixPinPool},
    },
};
use super::QwenResidentStep;
use crate::{
    EngError,
    family::session::{DecodeSession, PrefixSlot, SaveKind, StoreLimits, shadow_block_lcp_enabled},
};
use std::time::Instant;

const PIN_BYTES_ENV: &str = "Q36_CUDA_PINNED_PREFIX_BYTES";
const DEFAULT_PIN_BYTES_MAX: usize = 8usize << 30;
const PIN_MEMORY_HEADROOM: usize = 1usize << 30;

fn byte_limit(name: &str, value: Option<&std::ffi::OsStr>) -> Result<Option<usize>, EngError> {
    let Some(value) = value else { return Ok(None) };
    let bytes = value
        .to_str()
        .and_then(|value| value.trim().parse().ok())
        .ok_or_else(|| EngError::BadRequest(format!("{name} must be an unsigned byte count")))?;
    Ok(Some(bytes))
}

fn mem_available(contents: &str) -> Option<usize> {
    let value = contents
        .lines()
        .find_map(|line| line.strip_prefix("MemAvailable:"))?;
    let mut fields = value.split_whitespace();
    let kib = fields.next()?.parse::<usize>().ok()?;
    if fields.next()? != "kB" || fields.next().is_some() {
        return None;
    }
    kib.checked_mul(1024)
}

fn default_pin_bytes(checkpoint: usize) -> usize {
    checkpoint.saturating_mul(3).min(DEFAULT_PIN_BYTES_MAX)
}

fn planned_entries(checkpoint: usize, limit: usize, available: Option<usize>) -> usize {
    if checkpoint == 0 {
        return 0;
    }
    let retained = (limit / checkpoint).saturating_sub(1).min(2);
    if retained == 0 {
        return 0;
    }
    for retained in (1..=retained).rev() {
        let required = checkpoint
            .checked_mul(retained + 1)
            .and_then(|bytes| bytes.checked_add(PIN_MEMORY_HEADROOM));
        if required.is_some_and(|bytes| available.is_some_and(|free| free >= bytes)) {
            return retained;
        }
    }
    0
}

impl QwenResidentStep<'_> {
    /// Import refills the report; freeze lookup facts afterward, before later saves replace it.
    pub(crate) fn record_prefix_lookup(&mut self, prompt: &[u32], covered: usize) {
        self.prefill_report
            .observe_prefix_lookup(&self.prefix, prompt, covered);
    }

    pub(crate) fn enable_pinned_prefix(&mut self) -> Result<(), EngError> {
        if self.started || self.is_tiered() || self.prefix.key_len().is_some() {
            return Err(EngError::BadRequest(
                "pinned prefix needs a fresh non-tiered step".into(),
            ));
        }
        self.prefix = PrefixSlot::with_limits(StoreLimits {
            roots: 0,
            entries: 0,
        });
        self.prefix.set_shadow_enabled(shadow_block_lcp_enabled());
        let checkpoint = PrefixPinPool::checkpoint_bytes(self.state.layout())?;
        let limit = byte_limit(PIN_BYTES_ENV, std::env::var_os(PIN_BYTES_ENV).as_deref())?
            .unwrap_or_else(|| default_pin_bytes(checkpoint));
        let available = std::fs::read_to_string("/proc/meminfo")
            .ok()
            .and_then(|contents| mem_available(&contents));
        let entries = planned_entries(checkpoint, limit, available);
        if entries == 0 {
            eprintln!(
                "q36 cuda: pinned prefix refused, checkpoint {checkpoint} B, cap {limit} B, host headroom {PIN_MEMORY_HEADROOM} B, MemAvailable {available:?}; retained entries 0, bytes 0"
            );
            return Ok(());
        }
        let pool = match PrefixPinPool::new(self.context, *self.state.layout(), entries) {
            Ok(pool) => pool,
            Err(EngError::MemoryPlan(_)) => {
                eprintln!(
                    "q36 cuda: pinned prefix allocation refused; retained entries 0, bytes 0"
                );
                return Ok(());
            }
            Err(error) => return Err(error),
        };
        let retained = pool.retained_entries();
        self.prefix = PrefixSlot::with_limits(StoreLimits {
            roots: 0,
            entries: retained,
        })
        .with_recent_longer_protection();
        self.prefix.set_shadow_enabled(shadow_block_lcp_enabled());
        eprintln!(
            "q36 cuda: persistent pinned prefix, retained {retained}, staging 1, checkpoint {} B, total {} B; kill switches: Q36_CUDA_PINNED_PREFIX=0 (pageable), Q36_CUDA_STATE_ONLY_PREFIX=0 (pinned only)",
            pool.bytes() / (retained + 1),
            pool.bytes()
        );
        self.prefix_pins = Some(pool);
        Ok(())
    }

    /// The extra device slot is accounted by the constructor before admitting the prefill arena.
    pub(crate) fn enable_device_tip(&mut self) -> Result<usize, EngError> {
        if self.started || self.prefix_pins.is_none() || self.prefix.key_len().is_some() {
            return Err(EngError::BadRequest(
                "device tip needs a fresh pinned-prefix step".into(),
            ));
        }
        let tip = match DeviceTip::new(self.context, &self.stream, &self.state) {
            Ok(tip) => tip,
            Err(EngError::MemoryPlan(_)) => {
                eprintln!("q36 cuda: device tip allocation refused; pinned host fallback");
                return Ok(0);
            }
            Err(error) => return Err(error),
        };
        let bytes = tip.bytes();
        self.device_tip = Some(tip);
        eprintln!(
            "q36 cuda: state-only device tip, {bytes} B; disable: Q36_CUDA_STATE_ONLY_PREFIX=0"
        );
        Ok(bytes)
    }

    pub(crate) fn pinned_prefix_enabled(&self) -> bool {
        self.prefix_pins.is_some()
    }

    pub(crate) fn stage_pinned_prefix(
        &mut self,
        session: &DecodeSession,
        key: &[u32],
        kind: SaveKind,
        path: super::super::prefix::CudaPrefillPath,
    ) -> Result<bool, EngError> {
        let fed = self.tokens();
        let frozen = session.prepare_prefix_save(key, fed)?;
        self.complete_prefix_save()?;
        if self.prefix.limits().entries == 0 {
            return Ok(false);
        }
        let pool = self
            .prefix_pins
            .as_mut()
            .ok_or_else(|| EngError::BadRequest("pinned prefix was not enabled".into()))?;
        let recycled = if kind == SaveKind::Tip {
            self.prefix.take_tip_if(key, &path, |saved| match saved {
                CudaSavedPrefix::Pinned(saved) => {
                    pool.can_recycle(&self.stream, &self.state, saved)
                }
                _ => false,
            })
        } else {
            None
        };
        let recycled = match recycled {
            Some(CudaSavedPrefix::Pinned(saved)) => Some(saved),
            _ => None,
        };
        let mut state = match pool.begin(self.context, &self.stream, &self.state, recycled) {
            Ok(state) => state,
            Err(EngError::MemoryPlan(_)) => return Ok(false),
            Err(error) => return Err(error),
        };
        if let Some(device) = self.device_tip.as_mut() {
            let key = device.capture(self.context, &self.stream, &mut self.state)?;
            state.attach_device_tip(key)?;
        }
        self.pending_prefix = Some(PendingPrefix {
            session: frozen,
            fed,
            kind,
            path,
            state,
        });
        Ok(true)
    }

    pub(crate) fn complete_prefix_save(&mut self) -> Result<(), EngError> {
        let Some(pending) = self.pending_prefix.take() else {
            return Ok(());
        };
        // A failed model step must never publish a partly advanced snapshot.
        if self.state.health() != Health::Valid {
            return Ok(());
        }
        let began = Instant::now();
        let pool = self
            .prefix_pins
            .as_mut()
            .ok_or_else(|| EngError::BadRequest("pending prefix lost its pool".into()))?;
        let state = pool.finish(self.context, &self.stream, &self.state, pending.state)?;
        let positions = state.tokens();
        let copied_kv_bytes = state.copied_kv_bytes();
        let mut ready = Some(CudaSavedPrefix::Pinned(state));
        let saved = pending.session.save_prefix(
            &mut self.prefix,
            pending.session.history(),
            pending.path,
            pending.fed,
            pending.kind,
            || {
                ready
                    .take()
                    .ok_or_else(|| EngError::BadRequest("prefix already published".into()))
            },
        )?;
        if saved {
            let charge = entry_bytes(self.state.layout(), positions)
                .map_or_else(|error| format!("{error:?}"), |bytes| format!("{bytes} B"));
            eprintln!(
                "q36 cuda: prefix saved {positions} positions, {charge}, {:.1} ms, KV copied {copied_kv_bytes} B",
                began.elapsed().as_secs_f64() * 1000.0
            );
        }
        Ok(())
    }

    pub(crate) fn restore_pinned_prefix(&mut self, saved: &PinnedSnapshot) -> Result<(), EngError> {
        self.complete_prefix_save()?;
        let device = self
            .prefix_pins
            .as_mut()
            .ok_or_else(|| EngError::BadRequest("pinned snapshot lost its pool".into()))?
            .restore(
                self.context,
                &self.stream,
                &mut self.state,
                saved,
                self.device_tip.as_ref(),
            )?;
        if self.device_tip.is_some() {
            eprintln!(
                "q36 cuda: prefix restore source {}",
                if device { "device" } else { "host" }
            );
        }
        self.refill_prefill_report();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::family::{
        project::cuda::WeightLayout,
        qwen36::cuda::{
            common_tests,
            dims::{KvMode, StepOptions},
            state_layout::StateLayout,
        },
    };
    use std::ffi::OsStr;

    #[test]
    fn pin_admission_keeps_one_retained_entry_when_two_do_not_fit_host_memory() {
        let size = 1024;
        let limit = 3 * size;
        for available in [
            PIN_MEMORY_HEADROOM + 2 * size,
            PIN_MEMORY_HEADROOM + limit - 1,
        ] {
            assert_eq!(planned_entries(size, limit, Some(available)), 1);
        }
        assert_eq!(
            planned_entries(size, limit, Some(PIN_MEMORY_HEADROOM + 2 * size - 1)),
            0
        );
    }

    #[test]
    fn pin_admission_reserves_a_physical_staging_slot() {
        let size = 1024;
        assert_eq!(default_pin_bytes(size), 3 * size);
        assert_eq!(default_pin_bytes(4usize << 30), DEFAULT_PIN_BYTES_MAX);
        assert_eq!(default_pin_bytes(usize::MAX), DEFAULT_PIN_BYTES_MAX);
        let margin = PIN_MEMORY_HEADROOM;
        for (limit, available, expected) in [
            (0, Some(margin + 3 * size), 0),
            (2 * size - 1, Some(margin + 3 * size), 0),
            (2 * size, Some(margin + 2 * size), 1),
            (2 * size, Some(margin + 2 * size - 1), 0),
            (3 * size - 1, Some(margin + 3 * size), 1),
            (3 * size, Some(margin + 3 * size), 2),
            (8 * size, Some(margin + 3 * size), 2),
            (3 * size, Some(margin + 3 * size - 1), 1),
            (3 * size, None, 0),
        ] {
            assert_eq!(planned_entries(size, limit, available), expected);
        }
        for checkpoint in [0, usize::MAX, usize::MAX / 2] {
            assert_eq!(planned_entries(checkpoint, usize::MAX, Some(usize::MAX)), 0);
        }
        assert_eq!(
            planned_entries(usize::MAX / 3, usize::MAX, Some(usize::MAX)),
            1
        );
    }

    #[test]
    fn pin_admission_keeps_one_gib_available_beyond_the_pool() {
        let size = 1024;
        let headroom = 1usize << 30;
        for (retained, below) in [(1, 0), (2, 1)] {
            let bytes = (retained + 1) * size;
            assert_eq!(
                planned_entries(size, bytes, Some(headroom + bytes - 1)),
                below
            );
            assert_eq!(
                planned_entries(size, bytes, Some(headroom + bytes)),
                retained
            );
        }
    }

    #[test]
    fn pin_checkpoint_charge_is_the_allocated_layout_in_both_formats() {
        let mut saw_padding = false;
        for mode in [KvMode::Fp16, KvMode::Int8G32] {
            for slots in [1, 3] {
                let options = StepOptions {
                    kv_mode: mode,
                    context_tokens: 17,
                    delta_slots: slots,
                    forked_experts: false,
                    tiered: None,
                    layout: WeightLayout::Native,
                };
                let layout = StateLayout::new(&common_tests::dims(), &options).unwrap();
                let bytes = PrefixPinPool::checkpoint_bytes(&layout).unwrap();
                assert_eq!(bytes, layout.delta.total / slots + layout.kv.total);
                let payload = entry_bytes(&layout, 17).unwrap();
                assert!(bytes >= payload);
                saw_padding |= bytes > payload;
                assert!(bytes > entry_bytes(&layout, 1).unwrap());
                let mut invalid = layout;
                invalid.delta.slots = 0;
                assert!(PrefixPinPool::checkpoint_bytes(&invalid).is_err());
                invalid.delta.slots = 1;
                invalid.delta.total = usize::MAX;
                assert!(matches!(
                    PrefixPinPool::checkpoint_bytes(&invalid),
                    Err(EngError::MemoryPlan(_))
                ));
            }
        }
        assert!(
            saw_padding,
            "fixture must distinguish physical backing from exported payload"
        );
    }

    #[test]
    fn pin_byte_limits_reject_malformed_values_without_global_env_mutation() {
        assert_eq!(byte_limit(PIN_BYTES_ENV, None), Ok(None));
        assert_eq!(
            byte_limit(PIN_BYTES_ENV, Some(OsStr::new("0"))),
            Ok(Some(0))
        );
        assert_eq!(
            byte_limit(PIN_BYTES_ENV, Some(OsStr::new(" 1024 "))),
            Ok(Some(1024))
        );
        for value in ["", "-1", "1KiB", "garbage"] {
            assert!(byte_limit(PIN_BYTES_ENV, Some(OsStr::new(value))).is_err());
        }
        let overflow = format!("{}0", usize::MAX);
        assert!(byte_limit(PIN_BYTES_ENV, Some(OsStr::new(&overflow))).is_err());
        #[cfg(unix)]
        {
            use std::os::unix::ffi::OsStrExt;
            assert!(byte_limit(PIN_BYTES_ENV, Some(OsStr::from_bytes(&[0xff]))).is_err());
        }
    }

    #[test]
    fn unavailable_or_malformed_host_memory_refuses_retention() {
        assert_eq!(
            mem_available("MemTotal: 99 kB\nMemAvailable: 3072 kB\n"),
            Some(3 << 20)
        );
        for info in [
            "",
            "MemAvailable: -1 kB",
            "MemAvailable: 1 MB",
            "MemAvailable: 1 kB extra",
            "MemAvailable: nope kB",
        ] {
            assert_eq!(mem_available(info), None);
        }
        assert_eq!(
            mem_available(&format!("MemAvailable: {} kB", usize::MAX)),
            None
        );
    }
}
