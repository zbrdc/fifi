use std::{borrow::Cow, cell::Cell, collections::HashSet, fmt};

use safetensors::Dtype as StoredDtype;
use serde::de::{self, DeserializeSeed, Deserializer, MapAccess, SeqAccess, Unexpected, Visitor};

use crate::{EngError, tensor::Dtype};

const METADATA_KEY: &str = "__metadata__";

/// Checks safetensors header JSON against the rules that every load path shares.
///
/// The format forbids duplicate keys, and the upstream parser keeps only the last one.
/// This check reports a repeated object member name at any depth.
/// The format allows `__metadata__` only as a string-to-string map.
/// The upstream header type also reads `null` there as absent metadata, and MLX writes that form.
/// This check accepts `null` and rejects every other value that is not a string map.
/// `label` names the header in error messages.
///
/// The check streams over the JSON and builds no value tree. Its memory is the set of member
/// names of the objects that are open at one time, and the names borrow from `json`.
/// It keeps the `serde_json` recursion limit. It does not check the shape of a tensor entry.
pub(super) fn validate_header_json(json: &[u8], label: &str) -> Result<(), EngError> {
    let bad_metadata = Cell::new(false);
    let mut deserializer = serde_json::Deserializer::from_slice(json);
    let checker = Checker {
        top_level: Some(&bad_metadata),
    };
    match checker
        .deserialize(&mut deserializer)
        .and_then(|()| deserializer.end())
    {
        Ok(()) => Ok(()),
        Err(_) if bad_metadata.get() => Err(EngError::Checkpoint(format!(
            "{label} metadata is not a string map"
        ))),
        Err(error) => Err(EngError::Checkpoint(format!(
            "{label} is not valid JSON: {error}"
        ))),
    }
}

/// Parses header JSON into a value tree after `validate_header_json` accepts it.
///
/// The tree costs memory in proportion to the header size. Use `validate_header_json` alone
/// when the caller needs no tree.
pub(super) fn parse_header_json(json: &[u8], label: &str) -> Result<serde_json::Value, EngError> {
    validate_header_json(json, label)?;
    serde_json::from_slice(json)
        .map_err(|error| EngError::Checkpoint(format!("{label} is not valid JSON: {error}")))
}

/// Walks one JSON value, rejects repeated member names, and keeps no value.
///
/// `top_level` is set only for the header object. It carries the flag that marks a bad
/// `__metadata__` value.
struct Checker<'a> {
    top_level: Option<&'a Cell<bool>>,
}

impl<'de> DeserializeSeed<'de> for Checker<'_> {
    type Value = ();

    fn deserialize<D: Deserializer<'de>>(self, deserializer: D) -> Result<(), D::Error> {
        deserializer.deserialize_any(self)
    }
}

impl<'de> Visitor<'de> for Checker<'_> {
    type Value = ();

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("JSON with unique object member names")
    }

    fn visit_bool<E: de::Error>(self, _: bool) -> Result<(), E> {
        Ok(())
    }
    fn visit_i64<E: de::Error>(self, _: i64) -> Result<(), E> {
        Ok(())
    }
    fn visit_u64<E: de::Error>(self, _: u64) -> Result<(), E> {
        Ok(())
    }
    fn visit_f64<E: de::Error>(self, _: f64) -> Result<(), E> {
        Ok(())
    }
    fn visit_str<E: de::Error>(self, _: &str) -> Result<(), E> {
        Ok(())
    }
    fn visit_unit<E: de::Error>(self) -> Result<(), E> {
        Ok(())
    }

    fn visit_seq<A: SeqAccess<'de>>(self, mut sequence: A) -> Result<(), A::Error> {
        while sequence
            .next_element_seed(Checker { top_level: None })?
            .is_some()
        {}
        Ok(())
    }

    fn visit_map<A: MapAccess<'de>>(self, mut object: A) -> Result<(), A::Error> {
        let mut seen = HashSet::new();
        while let Some(is_metadata) = next_unique_name(&mut object, &mut seen)? {
            match self.top_level {
                Some(bad_metadata) if is_metadata => {
                    object.next_value_seed(StringMap {
                        bad_metadata,
                        inside_map: false,
                    })?;
                }
                _ => object.next_value_seed(Checker { top_level: None })?,
            }
        }
        Ok(())
    }
}

/// Accepts `null` or an object whose values are all strings, and rejects repeated names.
///
/// A rejected value sets `bad_metadata`, so the caller can tell it from a syntax error.
struct StringMap<'a> {
    bad_metadata: &'a Cell<bool>,
    inside_map: bool,
}

impl StringMap<'_> {
    fn reject<T, E: de::Error>(&self, unexpected: Unexpected<'_>) -> Result<T, E> {
        self.bad_metadata.set(true);
        Err(E::invalid_type(unexpected, self))
    }
}

impl<'de> DeserializeSeed<'de> for StringMap<'_> {
    type Value = ();

    fn deserialize<D: Deserializer<'de>>(self, deserializer: D) -> Result<(), D::Error> {
        deserializer.deserialize_any(self)
    }
}

impl<'de> Visitor<'de> for StringMap<'_> {
    type Value = ();

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("null or a map of strings")
    }

    fn visit_bool<E: de::Error>(self, value: bool) -> Result<(), E> {
        self.reject(Unexpected::Bool(value))
    }
    fn visit_i64<E: de::Error>(self, value: i64) -> Result<(), E> {
        self.reject(Unexpected::Signed(value))
    }
    fn visit_u64<E: de::Error>(self, value: u64) -> Result<(), E> {
        self.reject(Unexpected::Unsigned(value))
    }
    fn visit_f64<E: de::Error>(self, value: f64) -> Result<(), E> {
        self.reject(Unexpected::Float(value))
    }
    fn visit_str<E: de::Error>(self, value: &str) -> Result<(), E> {
        if self.inside_map {
            Ok(())
        } else {
            self.reject(Unexpected::Str(value))
        }
    }
    fn visit_unit<E: de::Error>(self) -> Result<(), E> {
        if self.inside_map {
            self.reject(Unexpected::Unit)
        } else {
            Ok(())
        }
    }
    fn visit_seq<A: SeqAccess<'de>>(self, _: A) -> Result<(), A::Error> {
        self.reject(Unexpected::Seq)
    }

    fn visit_map<A: MapAccess<'de>>(self, mut object: A) -> Result<(), A::Error> {
        if self.inside_map {
            return self.reject(Unexpected::Map);
        }
        let mut seen = HashSet::new();
        while next_unique_name(&mut object, &mut seen)?.is_some() {
            object.next_value_seed(StringMap {
                bad_metadata: self.bad_metadata,
                inside_map: true,
            })?;
        }
        Ok(())
    }
}

/// Reads the next member name and rejects a name that `seen` already holds.
///
/// It returns `None` at the end of the object, and otherwise whether the name is `__metadata__`.
fn next_unique_name<'de, A: MapAccess<'de>>(
    object: &mut A,
    seen: &mut HashSet<Cow<'de, str>>,
) -> Result<Option<bool>, A::Error> {
    let Some(name) = object.next_key_seed(NameSeed)? else {
        return Ok(None);
    };
    let is_metadata = name == METADATA_KEY;
    if let Some(repeated) = seen.replace(name) {
        return Err(de::Error::custom(format!(
            "duplicate JSON member {repeated:?}"
        )));
    }
    Ok(Some(is_metadata))
}

/// Reads a member name and borrows it from the input when it has no escapes.
struct NameSeed;

impl<'de> DeserializeSeed<'de> for NameSeed {
    type Value = Cow<'de, str>;

    fn deserialize<D: Deserializer<'de>>(self, deserializer: D) -> Result<Self::Value, D::Error> {
        deserializer.deserialize_str(self)
    }
}

impl<'de> Visitor<'de> for NameSeed {
    type Value = Cow<'de, str>;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("a member name")
    }

    fn visit_borrowed_str<E: de::Error>(self, value: &'de str) -> Result<Self::Value, E> {
        Ok(Cow::Borrowed(value))
    }
    fn visit_str<E: de::Error>(self, value: &str) -> Result<Self::Value, E> {
        Ok(Cow::Owned(value.to_owned()))
    }
}

/// Maps a stored safetensors dtype to the engine dtype. A dtype the engine cannot hold returns `None`.
pub(super) fn stored_dtype(stored: StoredDtype) -> Option<Dtype> {
    let mapped = match stored {
        StoredDtype::F32 => Dtype::F32,
        StoredDtype::F16 => Dtype::F16,
        StoredDtype::BF16 => Dtype::Bf16,
        StoredDtype::U8 => Dtype::U8,
        StoredDtype::U32 => Dtype::U32,
        StoredDtype::I64 => Dtype::I64,
        _ => return None,
    };
    Some(mapped)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn supported_stored_dtypes_map_to_engine_dtypes() {
        let cases = [
            (StoredDtype::F32, Dtype::F32),
            (StoredDtype::F16, Dtype::F16),
            (StoredDtype::BF16, Dtype::Bf16),
            (StoredDtype::U8, Dtype::U8),
            (StoredDtype::U32, Dtype::U32),
            (StoredDtype::I64, Dtype::I64),
        ];
        for (stored, expected) in cases {
            assert_eq!(stored_dtype(stored), Some(expected), "{stored}");
        }
    }

    #[test]
    fn other_stored_dtypes_have_no_engine_dtype() {
        let cases = [
            StoredDtype::BOOL,
            StoredDtype::I8,
            StoredDtype::I16,
            StoredDtype::U16,
            StoredDtype::I32,
            StoredDtype::F64,
            StoredDtype::U64,
            StoredDtype::F8_E4M3,
        ];
        for stored in cases {
            assert_eq!(stored_dtype(stored), None, "{stored}");
        }
    }

    #[test]
    fn unique_members_parse_into_a_value() {
        let value =
            parse_header_json(br#"{"a":{"x":1},"b":[{"x":2},{"x":3}]}  "#, "header").unwrap();
        assert_eq!(value["a"]["x"], 1);
        assert_eq!(value["b"][1]["x"], 3);
    }

    #[test]
    fn a_repeated_member_name_is_an_error_at_every_depth() {
        let cases = [
            (r#"{"a":1,"a":2}"#, "a"),
            (r#"{"a":{"b":1,"b":1}}"#, "b"),
            (r#"{"a":[{"c":null,"c":null}]}"#, "c"),
        ];
        for (json, name) in cases {
            let EngError::Checkpoint(message) =
                parse_header_json(json.as_bytes(), "the header").unwrap_err()
            else {
                panic!("expected a checkpoint error");
            };
            assert!(
                message.starts_with("the header is not valid JSON: "),
                "{message}"
            );
            assert!(
                message.contains(&format!("duplicate JSON member {name:?}")),
                "{message}"
            );
        }
    }

    #[test]
    fn invalid_json_is_a_checkpoint_error_with_the_subject() {
        let EngError::Checkpoint(message) = parse_header_json(b"{", "the header").unwrap_err()
        else {
            panic!("expected a checkpoint error");
        };
        assert!(
            message.starts_with("the header is not valid JSON: "),
            "{message}"
        );
    }

    #[test]
    fn metadata_must_be_absent_null_or_a_string_map() {
        for json in [
            r#"{}"#,
            r#"{"__metadata__":null}"#,
            r#"{"__metadata__":{}}"#,
            r#"{"__metadata__":{"a":"b"}}"#,
        ] {
            assert!(
                parse_header_json(json.as_bytes(), "the header").is_ok(),
                "{json}"
            );
        }
        for metadata in [
            "0",
            "true",
            r#""pt""#,
            r#"["pt"]"#,
            r#"{"a":1}"#,
            r#"{"a":null}"#,
            r#"{"a":{"b":"c"}}"#,
        ] {
            let json = format!(r#"{{"__metadata__":{metadata}}}"#);
            assert_eq!(
                parse_header_json(json.as_bytes(), "the header").unwrap_err(),
                EngError::Checkpoint("the header metadata is not a string map".into()),
                "{metadata}"
            );
        }
    }

    #[test]
    fn a_metadata_name_below_the_top_level_is_ordinary_data() {
        let json = br#"{"a":{"__metadata__":1}}"#;
        assert!(parse_header_json(json, "the header").is_ok());
    }

    fn message(result: Result<(), EngError>) -> String {
        match result.unwrap_err() {
            EngError::Checkpoint(message) => message,
            other => panic!("expected a checkpoint error, found {other:?}"),
        }
    }

    #[test]
    fn validation_accepts_well_formed_headers_of_any_shape() {
        for json in [
            r#"{}"#,
            r#"{"__metadata__":null,"a":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}}"#,
            r#"{"__metadata__":{"k":"v"},"a":{"x":[1,2.5,-3,true,null,"s",{"y":{}}]}}"#,
            r#"  {"a":1}  "#,
            r#"[1,2]"#,
            r#"7"#,
        ] {
            assert_eq!(validate_header_json(json.as_bytes(), "h"), Ok(()), "{json}");
        }
    }

    #[test]
    fn a_name_written_with_an_escape_repeats_the_plain_name() {
        let text = message(validate_header_json(br#"{"a":1,"\u0061":2}"#, "h"));
        assert!(
            text.starts_with("h is not valid JSON: duplicate JSON member \"a\""),
            "{text}"
        );
    }

    #[test]
    fn the_same_name_in_different_objects_is_not_a_repeat() {
        let json = br#"{"a":{"x":1},"b":{"x":2},"c":[{"x":3},{"x":4}]}"#;
        assert_eq!(validate_header_json(json, "h"), Ok(()));
    }

    #[test]
    fn repeated_names_inside_metadata_keep_the_duplicate_text() {
        let text = message(validate_header_json(
            br#"{"__metadata__":{"k":"a","k":"b"}}"#,
            "h",
        ));
        assert!(text.contains("duplicate JSON member \"k\""), "{text}");
        assert!(!text.contains("string map"), "{text}");
    }

    #[test]
    fn a_syntax_error_inside_metadata_is_not_reported_as_a_bad_map() {
        let text = message(validate_header_json(br#"{"__metadata__":{"k":tru}}"#, "h"));
        assert!(text.starts_with("h is not valid JSON: "), "{text}");
    }

    #[test]
    fn trailing_input_and_empty_input_are_errors() {
        for json in [&b"{} x"[..], b"{}{}", b"", b"   "] {
            let text = message(validate_header_json(json, "h"));
            assert!(text.starts_with("h is not valid JSON: "), "{text}");
        }
    }

    #[test]
    fn nesting_beyond_the_serde_json_limit_is_an_error() {
        let deep = format!("{}1{}", "[".repeat(200), "]".repeat(200));
        let text = message(validate_header_json(deep.as_bytes(), "h"));
        assert!(text.contains("recursion limit"), "{text}");
        let shallow = format!("{}1{}", "[".repeat(100), "]".repeat(100));
        assert_eq!(validate_header_json(shallow.as_bytes(), "h"), Ok(()));
    }

    #[test]
    fn member_names_repeat_at_the_end_of_a_long_object() {
        let mut json = String::from("{");
        for index in 0..5000 {
            json.push_str(&format!("\"t{index}\":{{}},"));
        }
        json.push_str("\"t4999\":{}}");
        let text = message(validate_header_json(json.as_bytes(), "h"));
        assert!(text.contains("duplicate JSON member \"t4999\""), "{text}");
    }
}
