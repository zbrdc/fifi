//! `family::qwen36::cuda::gate`: the ONE public item of the stage 1 gate (decision 0047 item 11). Design guidance only,
//! no code copied (no MegaCapybara-derived design).
//!
//! RD-10a-17 wires the device loop of `run.rs` into `run_step_gate`.
//! The `pub` items of the whole `family::qwen36` tree are exactly `qwen36`, `cuda`, `gate`, `run_step_gate`,
//! `GateSettings` and `GateReport` (a source check, `public_items.py`, asserts it). The two types keep PRIVATE fields: the
//! binary builds a `GateSettings` from a JSON text (`str::parse`, the `FromStr` impl) and reads the report as text
//! (`to_string`, the `Display` impl). Trait impls carry no `pub` keyword, and the same check lists them too.
//! A `prefill` key is a later addition to the settings text, never a second function.
use super::dims::TieredPlan;
use crate::EngError;
use crate::family::project::cuda::WeightLayout;
use serde_json::Value;
use std::fmt;
use std::path::PathBuf;
use std::str::FromStr;

mod phases;
mod raw;
mod rows;
mod run;
mod steps;

/// What one GPU call is asked to do. Built only by parsing a JSON object with the keys named in `from_str`.
pub struct GateSettings {
    gguf: PathBuf,
    /// SHA-256 of the GGUF as checked by the run script (echoed into the raw file; the binary does not rehash 20 GB).
    gguf_sha256: String,
    /// Prompt names, token ids, and (forced mode only) the reference stream and the SHA-256 of its source file.
    names: Vec<String>,
    prompts: Vec<Vec<u32>>,
    forced: Vec<Vec<u32>>,
    forced_stream_sha256: Vec<String>,
    gen_tokens: u32,
    runs: u32,
    /// true = the captured graph path (RD-11a), false = the uncaptured launches of the same code.
    capture: bool,
    /// Stage gates: true. Anything else is refused.
    kv_fp16: bool,
    /// Wall-clock budget of the call in seconds (0 = none); on expiry the report is `incomplete`.
    deadline_secs: u32,
    /// The LOAD layout of the call (MATVEC-PAR S3b): "native" (the default when the key is
    /// absent, since the S3c qualification) or "tiled". It chooses only how the tensors are uploaded; the step's layout
    /// always follows the loaded weights.
    layout: WeightLayout,
    /// The tiered expert mode (OFF-2), or `None` for the plain native step; never on without the
    /// optional `tiered` member.
    tiered: Option<TieredPlan>,
}

/// The optional `tiered` member: `{"slots": n, "threads": n, "timeout_ms": n, "preload": bool}`,
/// `slots` required.
pub(super) fn tiered_member(v: &Value) -> Result<Option<TieredPlan>, EngError> {
    let bad = |what: &str| EngError::BadRequest(format!("gate settings: tiered {what}"));
    let Some(member) = v.get("tiered") else {
        return Ok(None);
    };
    let object = member.as_object().ok_or_else(|| bad("must be an object"))?;
    if let Some(key) = object
        .keys()
        .find(|key| !["slots", "threads", "timeout_ms", "preload"].contains(&key.as_str()))
    {
        return Err(bad(&format!("has the unknown member {key}")));
    }
    let number = |key: &str| -> Result<Option<u32>, EngError> {
        object
            .get(key)
            .map(|value| {
                value
                    .as_u64()
                    .and_then(|n| u32::try_from(n).ok())
                    .ok_or_else(|| bad(&format!("{key} is not a u32")))
            })
            .transpose()
    };
    let slots = number("slots")?.ok_or_else(|| bad("needs slots"))?;
    let threads = number("threads")?.unwrap_or(6);
    let mut plan = TieredPlan::new(slots as usize, threads as usize);
    if let Some(timeout_ms) = number("timeout_ms")? {
        plan.timeout_ms = timeout_ms;
    }
    if let Some(preload) = object.get("preload") {
        plan.preload = preload
            .as_bool()
            .ok_or_else(|| bad("preload is not a bool"))?;
    }
    plan.check()?;
    Ok(Some(plan))
}

/// The raw JSON of schema `cuda-rd-raw/2` (see `a5_eval.py`), without the identities only the binary knows.
pub struct GateReport {
    raw: String,
}

fn field<'a>(v: &'a Value, key: &str) -> Result<&'a Value, EngError> {
    v.get(key)
        .ok_or_else(|| EngError::BadRequest(format!("gate settings: missing {key}")))
}

fn string<'a>(v: &'a Value, key: &str) -> Result<&'a str, EngError> {
    field(v, key)?
        .as_str()
        .ok_or_else(|| EngError::BadRequest(format!("gate settings: {key} is not a string")))
}

fn strings(v: &Value, key: &str) -> Result<Vec<String>, EngError> {
    let bad = || EngError::BadRequest(format!("gate settings: {key} is not a list of strings"));
    field(v, key)?
        .as_array()
        .ok_or_else(bad)?
        .iter()
        .map(|s| s.as_str().map(str::to_owned).ok_or_else(bad))
        .collect()
}

fn id_lists(v: &Value, key: &str) -> Result<Vec<Vec<u32>>, EngError> {
    let bad = || EngError::BadRequest(format!("gate settings: {key} is not a list of id lists"));
    let id = |x: &Value| {
        x.as_u64()
            .and_then(|n| u32::try_from(n).ok())
            .ok_or_else(bad)
    };
    field(v, key)?
        .as_array()
        .ok_or_else(bad)?
        .iter()
        .map(|l| l.as_array().ok_or_else(bad)?.iter().map(id).collect())
        .collect()
}

fn number(v: &Value, key: &str) -> Result<u32, EngError> {
    field(v, key)?
        .as_u64()
        .and_then(|n| u32::try_from(n).ok())
        .ok_or_else(|| EngError::BadRequest(format!("gate settings: {key} is not a u32")))
}

fn flag(v: &Value, key: &str) -> Result<bool, EngError> {
    field(v, key)?
        .as_bool()
        .ok_or_else(|| EngError::BadRequest(format!("gate settings: {key} is not a boolean")))
}

impl FromStr for GateSettings {
    type Err = EngError;

    /// Keys: gguf, gguf_sha256, names, prompts, forced, forced_stream_sha256, gen_tokens, runs,
    /// capture, kv_fp16, deadline_secs, and the optional keys `layout` ("native", the default
    /// since the S3c qualification, or "tiled"; MATVEC-PAR S3b) and `tiered` (OFF-2; absent means
    /// off, refused with capture and with the tiled layout). A missing or mistyped key is a typed `BadRequest`, never a
    /// default.
    fn from_str(text: &str) -> Result<Self, EngError> {
        let v: Value = serde_json::from_str(text)
            .map_err(|e| EngError::BadRequest(format!("gate settings: {e}")))?;
        let s = GateSettings {
            gguf: PathBuf::from(string(&v, "gguf")?),
            gguf_sha256: string(&v, "gguf_sha256")?.to_owned(),
            names: strings(&v, "names")?,
            prompts: id_lists(&v, "prompts")?,
            forced: id_lists(&v, "forced")?,
            forced_stream_sha256: strings(&v, "forced_stream_sha256")?,
            gen_tokens: number(&v, "gen_tokens")?,
            runs: number(&v, "runs")?,
            capture: flag(&v, "capture")?,
            kv_fp16: flag(&v, "kv_fp16")?,
            deadline_secs: number(&v, "deadline_secs")?,
            // The ONE optional key (MATVEC-PAR S3b): absent means native (the S3c-qualified
            // default), anything but "tiled" and "native" is a typed error naming the key.
            layout: match v.get("layout") {
                None => WeightLayout::Native,
                Some(value) => match value.as_str() {
                    Some("tiled") => WeightLayout::Tiled,
                    Some("native") => WeightLayout::Native,
                    _ => {
                        return Err(EngError::BadRequest(
                            "gate settings: layout must be \"tiled\" or \"native\"".to_owned(),
                        ));
                    }
                },
            },
            tiered: tiered_member(&v)?,
        };
        if s.tiered.is_some() && s.capture {
            return Err(EngError::BadRequest(
                "gate settings: tiered and capture exclude each other: the tiered step is eager only"
                    .to_owned(),
            ));
        }
        if s.tiered.is_some() && s.layout != WeightLayout::Native {
            return Err(EngError::BadRequest(
                "gate settings: tiered needs the native layout: the bank stacks are native blocks"
                    .to_owned(),
            ));
        }
        if s.names.len() != s.prompts.len() || s.gguf_sha256.is_empty() || s.runs == 0 {
            return Err(EngError::BadRequest(
                "gate settings: inconsistent lists or zero runs".to_owned(),
            ));
        }
        Ok(s)
    }
}

impl fmt::Display for GateReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.raw)
    }
}

/// Runs the gate: the settings checks, the load, the memory admission and the device loop of
/// `run.rs` (RD-10a-17).
pub fn run_step_gate(settings: &GateSettings) -> Result<GateReport, EngError> {
    run::device_gate(settings)
}

#[cfg(test)]
mod tests {
    use super::*;

    const GOOD: &str = r#"{"gguf":"/m.gguf","gguf_sha256":"ab","names":["P1"],"prompts":[[1,2,3]],"forced":[],
        "forced_stream_sha256":[],"gen_tokens":128,"runs":1,"capture":true,"kv_fp16":true,"deadline_secs":270}"#;

    fn without(key: &str) -> String {
        let mut v: Value = serde_json::from_str(GOOD).unwrap();
        v.as_object_mut().unwrap().remove(key);
        v.to_string()
    }

    fn rejects_naming(text: &str, key: &str) {
        match text.parse::<GateSettings>() {
            Err(EngError::BadRequest(m)) => assert!(m.contains(key), "{key}: {m}"),
            Err(other) => panic!("{key}: expected BadRequest, got {other:?}"),
            Ok(_) => panic!("{key}: expected BadRequest, parsed Ok"),
        }
    }

    #[test]
    fn a_complete_settings_text_parses_into_the_private_fields() {
        let s: GateSettings = GOOD.parse().unwrap();
        assert_eq!(
            (s.names.len(), s.prompts[0].clone(), s.runs, s.gen_tokens),
            (1, vec![1, 2, 3], 1, 128)
        );
        assert!(s.capture && s.kv_fp16 && s.deadline_secs == 270);
    }

    #[test]
    fn every_missing_key_is_a_typed_error_not_a_default() {
        for key in [
            "gguf",
            "gguf_sha256",
            "names",
            "prompts",
            "forced",
            "forced_stream_sha256",
            "gen_tokens",
            "runs",
            "capture",
            "kv_fp16",
            "deadline_secs",
        ] {
            assert!(
                matches!(
                    without(key).parse::<GateSettings>(),
                    Err(EngError::BadRequest(_))
                ),
                "{key}"
            );
        }
    }

    #[test]
    fn wrong_types_and_inconsistent_lists_are_refused() {
        let bad = [
            GOOD.replace("\"runs\":1", "\"runs\":\"1\""),
            GOOD.replace("\"capture\":true", "\"capture\":1"),
            GOOD.replace("\"runs\":1", "\"runs\":0"),
            GOOD.replace("[[1,2,3]]", "[]"),
            "not json".to_owned(),
        ];
        for text in bad {
            assert!(
                matches!(text.parse::<GateSettings>(), Err(EngError::BadRequest(_))),
                "{text}"
            );
        }
    }

    #[test]
    fn a_mistyped_gguf_is_refused_as_bad_request_naming_the_key() {
        rejects_naming(&GOOD.replace("\"gguf\":\"/m.gguf\"", "\"gguf\":5"), "gguf");
    }

    #[test]
    fn a_mistyped_gguf_sha256_is_refused_as_bad_request_naming_the_key() {
        rejects_naming(
            &GOOD.replace("\"gguf_sha256\":\"ab\"", "\"gguf_sha256\":5"),
            "gguf_sha256",
        );
    }

    #[test]
    fn a_mistyped_names_is_refused_as_bad_request_naming_the_key() {
        rejects_naming(
            &GOOD.replace("\"names\":[\"P1\"]", "\"names\":\"P1\""),
            "names",
        );
    }

    #[test]
    fn a_mistyped_forced_is_refused_as_bad_request_naming_the_key() {
        rejects_naming(
            &GOOD.replace("\"forced\":[]", "\"forced\":\"none\""),
            "forced",
        );
    }

    #[test]
    fn a_mistyped_gen_tokens_is_refused_as_bad_request_naming_the_key() {
        rejects_naming(
            &GOOD.replace("\"gen_tokens\":128", "\"gen_tokens\":\"128\""),
            "gen_tokens",
        );
    }

    #[test]
    fn the_valid_example_still_parses_after_the_string_gate() {
        let s: GateSettings = GOOD.parse().unwrap();
        assert_eq!(s.gguf, PathBuf::from("/m.gguf"));
        assert_eq!(s.gguf_sha256, "ab");
    }

    #[test]
    fn the_layout_member_defaults_to_native_and_refuses_other_values() {
        let s: GateSettings = GOOD.parse().unwrap();
        assert_eq!(s.layout, WeightLayout::Native);
        let tiled = GOOD.replace("\"runs\":1", "\"runs\":1,\"layout\":\"tiled\"");
        assert_eq!(
            tiled.parse::<GateSettings>().unwrap().layout,
            WeightLayout::Tiled
        );
        for bad in [
            GOOD.replace("\"runs\":1", "\"runs\":1,\"layout\":\"bogus\""),
            GOOD.replace("\"runs\":1", "\"runs\":1,\"layout\":5"),
        ] {
            match bad.parse::<GateSettings>() {
                Ok(_) => panic!("a bad layout value must be refused: it parsed"),
                Err(EngError::BadRequest(message)) => {
                    assert!(message.contains("layout"), "{message}")
                }
                Err(other) => panic!("a bad layout value must be a BadRequest: {other:?}"),
            }
        }
    }

    #[test]
    fn the_report_displays_its_raw_text_and_refusals_precede_the_device() {
        assert_eq!(
            GateReport {
                raw: "{\"a\":1}".to_owned()
            }
            .to_string(),
            "{\"a\":1}"
        );
        // kv_fp16 is refused before a context exists, so this needs no device (U13).
        let refused = GOOD.replace("\"kv_fp16\":true", "\"kv_fp16\":false");
        assert!(matches!(
            run_step_gate(&refused.parse().unwrap()),
            Err(EngError::BadRequest(_))
        ));
    }

    fn with_member(member: &str) -> String {
        let mut v: Value = serde_json::from_str(GOOD).unwrap();
        v.as_object_mut()
            .unwrap()
            .insert("capture".into(), Value::Bool(false));
        v.as_object_mut()
            .unwrap()
            .insert("tiered".into(), serde_json::from_str(member).unwrap());
        v.to_string()
    }

    #[test]
    fn the_gate_settings_parse_tiered_and_it_is_off_without_the_member() {
        let plain: GateSettings = GOOD.parse().unwrap();
        assert_eq!(plain.tiered, None, "tiered is never on without the member");
        let tiered: GateSettings = with_member(r#"{"slots": 61, "timeout_ms": 5000}"#)
            .parse()
            .unwrap();
        let plan = tiered.tiered.expect("the member sets the plan");
        assert_eq!(
            (plan.slots_per_layer, plan.threads, plan.timeout_ms),
            (61, 6, 5000)
        );
        assert!(!plan.preload, "the banks start empty unless asked");
        let preloaded: GateSettings = with_member(r#"{"slots": 256, "preload": true}"#)
            .parse()
            .unwrap();
        assert!(preloaded.tiered.expect("the member sets the plan").preload);
    }

    #[test]
    fn the_tiered_member_is_checked_and_refused_with_capture_and_the_tiled_layout() {
        rejects_naming(&with_member(r#"{}"#), "needs slots");
        rejects_naming(&with_member(r#"{"slots": 0}"#), "slots_per_layer");
        rejects_naming(&with_member(r#"{"slots": 40, "threads": 99}"#), "threads");
        rejects_naming(
            &with_member(r#"{"slots": 40, "banks": 1}"#),
            "unknown member banks",
        );
        rejects_naming(&with_member(r#"[1]"#), "must be an object");
        rejects_naming(
            &with_member(r#"{"slots": 40, "preload": 1}"#),
            "preload is not a bool",
        );
        let mut capture: Value = serde_json::from_str(&with_member(r#"{"slots": 40}"#)).unwrap();
        capture["capture"] = Value::Bool(true);
        rejects_naming(&capture.to_string(), "exclude each other");
        let mut tiled: Value = serde_json::from_str(&with_member(r#"{"slots": 40}"#)).unwrap();
        tiled["layout"] = Value::String("tiled".into());
        rejects_naming(&tiled.to_string(), "needs the native layout");
    }
}
