//! Explicit options of the first CUDA serving profile; legacy MLX entries omit them.

use super::kv_plan::{KvPlan, KvStorage};
use crate::EngError;
use serde::Deserialize;

// Above eight rows EXL3 changes kernels and reduction order; this is a numeric cap.
pub(crate) const PREFILL_MAX_ROWS: usize = 8;

#[cfg(any(test, all(feature = "native-cuda", target_os = "linux")))]
fn default_on_selected(value: Option<&std::ffi::OsStr>) -> bool {
    match value {
        None => true,
        Some(value) => value
            .to_str()
            .is_some_and(|value| crate::tokenizer::env_allows_value(Some(value))),
    }
}

#[cfg(any(test, all(feature = "native-cuda", target_os = "linux")))]
pub(crate) fn late_health_selected(value: Option<&std::ffi::OsStr>) -> bool {
    default_on_selected(value)
}

#[cfg(any(test, all(feature = "native-cuda", target_os = "linux")))]
pub(crate) fn packed_pointwise_selected(value: Option<&std::ffi::OsStr>) -> bool {
    value
        .and_then(std::ffi::OsStr::to_str)
        .is_some_and(|value| crate::tokenizer::env_allows_value(Some(value)))
}

#[cfg(any(test, all(feature = "native-cuda", target_os = "linux")))]
pub(crate) fn grouped_gather_selected(value: Option<&std::ffi::OsStr>) -> bool {
    default_on_selected(value)
}

fn default_chunk_size() -> u32 {
    8
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct FlintOptions {
    pub(crate) context_limit: u32,
    pub(crate) kv_profile: String,
    #[serde(default = "default_chunk_size")]
    pub(crate) prefill_chunk_size: u32,
}

impl FlintOptions {
    pub(crate) fn plan(&self) -> Result<KvPlan, EngError> {
        let storage = match self.kv_profile.as_str() {
            "f16" => KvStorage::F16,
            "q8" => KvStorage::Q8G32,
            _ => {
                return Err(EngError::BadRequest(
                    "Flint CUDA requires explicit kv_profile f16 or q8".into(),
                ));
            }
        };
        if !(1..=PREFILL_MAX_ROWS).contains(&(self.prefill_chunk_size as usize)) {
            return Err(EngError::BadRequest(
                "Flint CUDA prefill_chunk_size must be in 1..=8".into(),
            ));
        }
        KvPlan::new(
            self.context_limit as usize,
            self.prefill_chunk_size as usize,
            true,
        )
        .map(|plan| plan.with_storage(storage))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packed_pointwise_selector_is_fail_closed_default_off() {
        use std::ffi::OsStr;

        for (value, expected) in [
            (None, false),
            (Some("1"), true),
            (Some(" TrUe "), true),
            (Some("\u{2003}On\u{00a0}"), true),
            (Some("0"), false),
            (Some("off"), false),
            (Some("false"), false),
            (Some(""), false),
            (Some("yes"), false),
            (Some("2"), false),
            (Some("true!"), false),
            (Some("on\0"), false),
        ] {
            assert_eq!(
                packed_pointwise_selected(value.map(OsStr::new)),
                expected,
                "{value:?}"
            );
        }
        #[cfg(unix)]
        {
            use std::os::unix::ffi::OsStrExt;
            assert!(!packed_pointwise_selected(Some(OsStr::from_bytes(
                b"on\xff"
            ))));
        }
    }

    #[test]
    fn grouped_gather_selector_table() {
        use std::ffi::OsStr;

        for (value, expected) in [
            (None, true),
            (Some("1"), true),
            (Some(" TrUe "), true),
            (Some("0"), false),
            (Some(""), false),
            (Some("off"), false),
            (Some(" \t\n"), false),
            (Some("2"), false),
            (Some("yes"), false),
        ] {
            assert_eq!(
                grouped_gather_selected(value.map(OsStr::new)),
                expected,
                "{value:?}"
            );
        }
        #[cfg(unix)]
        {
            use std::os::unix::ffi::OsStrExt;
            assert!(!grouped_gather_selected(Some(OsStr::from_bytes(b"on\xff"))));
        }
    }

    #[test]
    fn late_health_selector_is_fail_safe_default_on() {
        use std::ffi::OsStr;

        for (value, expected) in [
            (None, true),
            (Some("1"), true),
            (Some(" 1 "), true),
            (Some("true"), true),
            (Some("TRUE"), true),
            (Some(" TrUe "), true),
            (Some("on"), true),
            (Some("ON"), true),
            (Some("\tOn\r\n"), true),
            (Some("\u{2003}On\u{00a0}"), true),
            (Some("0"), false),
            (Some("off"), false),
            (Some(" OFF "), false),
            (Some("false"), false),
            (Some("FALSE"), false),
            (Some(""), false),
            (Some(" \t\n"), false),
            (Some("yes"), false),
            (Some("2"), false),
            (Some("true!"), false),
            (Some("on\0"), false),
            (Some("ＴＲＵＥ"), false),
            (Some("oＮ"), false),
        ] {
            assert_eq!(
                late_health_selected(value.map(OsStr::new)),
                expected,
                "{value:?}"
            );
        }
        #[cfg(unix)]
        {
            use std::os::unix::ffi::OsStrExt;
            assert!(!late_health_selected(Some(OsStr::from_bytes(b"on\xff"))));
        }
    }

    #[test]
    fn cuda_profile_requires_explicit_supported_format_and_valid_context() {
        for kv_profile in ["f16", "q8"] {
            for context in [0, 262_145] {
                assert!(
                    FlintOptions {
                        context_limit: context,
                        kv_profile: kv_profile.into(),
                        prefill_chunk_size: 1,
                    }
                    .plan()
                    .is_err()
                );
            }
        }
        for kv_profile in ["q4", "f32", "", "F16", "Q8"] {
            assert!(
                FlintOptions {
                    context_limit: 128,
                    kv_profile: kv_profile.into(),
                    prefill_chunk_size: 1,
                }
                .plan()
                .is_err()
            );
        }
        let plan = FlintOptions {
            context_limit: 8192,
            kv_profile: "f16".into(),
            prefill_chunk_size: 1,
        }
        .plan()
        .unwrap();
        assert_eq!(plan.context_rows(), 8192);
        assert_eq!(plan.cache_rows(0).unwrap(), 8192);
        assert_eq!(plan.cache_rows(2).unwrap(), 513);
        assert_eq!(plan.bytes(2048).unwrap(), (14 * 8192 + 35 * 513) * 2048);
    }

    #[test]
    fn chunk_option_defaults_to_eight_and_admits_bounded_ring() {
        assert_eq!(
            PREFILL_MAX_ROWS, 8,
            "raising the cap changes numeric kernels"
        );
        let serial: FlintOptions = serde_json::from_str(
            r#"{"context_limit":4096,"kv_profile":"f16","prefill_chunk_size":1}"#,
        )
        .unwrap();
        assert_eq!(serial.prefill_chunk_size, 1);
        assert_eq!(serial.plan().unwrap().cache_rows(2).unwrap(), 513);
        let batch: FlintOptions =
            serde_json::from_str(r#"{"context_limit":4096,"kv_profile":"f16"}"#).unwrap();
        assert_eq!(batch.prefill_chunk_size, 8);
        assert_eq!(batch.plan().unwrap().cache_rows(2).unwrap(), 520);
        for rows in 1..=8 {
            let options = FlintOptions {
                prefill_chunk_size: rows,
                ..serial.clone()
            };
            assert_eq!(
                options.plan().unwrap().cache_rows(2).unwrap(),
                512 + rows as usize
            );
        }
        assert_eq!(
            batch.plan().unwrap().bytes(2048).unwrap()
                - serial.plan().unwrap().bytes(2048).unwrap(),
            501_760
        );
        for rows in [0, 9, u32::MAX] {
            assert!(
                FlintOptions {
                    prefill_chunk_size: rows,
                    ..serial.clone()
                }
                .plan()
                .is_err()
            );
        }
    }

    #[test]
    fn cuda_q8_profile_is_opt_in_and_charges_payload_and_scales() {
        use crate::family::flint::kv_plan::KvStorage;

        let q8: FlintOptions =
            serde_json::from_str(r#"{"context_limit":4096,"kv_profile":"q8"}"#).unwrap();
        assert_eq!(q8.prefill_chunk_size, 8);
        let plan = q8.plan().unwrap();
        assert_eq!(plan.storage(), KvStorage::Q8G32);
        assert_eq!(plan.storage().row_bytes(), 1088);
        assert_eq!(plan.bytes(plan.storage().row_bytes()).unwrap(), 82_191_872);
        assert_eq!(plan.cache_rows(0).unwrap(), 4096);
        assert_eq!(plan.cache_rows(2).unwrap(), 520);

        let f16: FlintOptions =
            serde_json::from_str(r#"{"context_limit":4096,"kv_profile":"f16"}"#).unwrap();
        let control = f16.plan().unwrap();
        assert_eq!(control.storage(), KvStorage::F16);
        assert_eq!(
            control.bytes(control.storage().row_bytes()).unwrap(),
            154_714_112
        );
    }

    #[test]
    fn cuda_options_reject_missing_or_unknown_profile_keys() {
        for text in [
            r#"{"context_limit":128}"#,
            r#"{"context_limit":128,"kv_profile":"f16","typo":1}"#,
        ] {
            assert!(serde_json::from_str::<FlintOptions>(text).is_err());
        }
    }
}
