//! The `tojson` filter of the reference renderer.
//!
//! `transformers` replaces the Jinja filter with
//! `tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False)`,
//! which calls `json.dumps`. This module ports that call for values that come from
//! JSON: null, booleans, numbers, strings, arrays, and objects. Decision 0029
//! records the choice. Object keys keep their insertion order unless `sort_keys`
//! is true. Floats print as Python `repr` does. Integers above `u64::MAX` parse
//! as floats (the crate does not enable `arbitrary_precision`), so they print as
//! floats where Python prints every digit. Decision 0029 records that divergence.

use minijinja::value::{Kwargs, Rest, Value, ValueKind, from_args};
use minijinja::{Error, ErrorKind};
use std::fmt::Write;

// Python would build a string of this many spaces. The engine refuses absurd sizes.
const MAX_INDENT: i64 = 1024;

const ARGUMENT_NAMES: [&str; 4] = ["ensure_ascii", "indent", "separators", "sort_keys"];

struct Format {
    ensure_ascii: bool,
    indent: Option<String>,
    item_separator: String,
    key_separator: String,
    sort_keys: bool,
}

fn python_error(kind: &str, message: &str) -> Error {
    Error::new(
        ErrorKind::InvalidOperation,
        format!("{kind}: tojson() {message}"),
    )
}

/// `{{ value | tojson(ensure_ascii, indent, separators, sort_keys) }}`.
/// Each argument takes its position or its name, as in Python.
pub(super) fn tojson_filter(value: Value, arguments: Rest<Value>) -> Result<Value, Error> {
    let format = Format::from_arguments(arguments.as_slice())?;
    let json = serde_json::to_value(&value).map_err(|error| {
        python_error(
            "TypeError",
            &format!("cannot serialize the value as JSON: {error}"),
        )
    })?;
    let mut out = String::new();
    write_value(&json, &format, 0, &mut out);
    Ok(Value::from(out))
}

impl Format {
    fn from_arguments(arguments: &[Value]) -> Result<Self, Error> {
        let (positional, keywords) = from_args::<(&[Value], Kwargs)>(arguments)?;
        if positional.len() > ARGUMENT_NAMES.len() {
            return Err(python_error(
                "TypeError",
                &format!(
                    "takes from 1 to 5 positional arguments but {} were given",
                    positional.len() + 1
                ),
            ));
        }
        let mut chosen: Vec<Option<Value>> = Vec::new();
        for (index, name) in ARGUMENT_NAMES.iter().enumerate() {
            let placed = positional.get(index);
            let named = if keywords.has(name) {
                Some(keywords.get::<Value>(name)?)
            } else {
                None
            };
            if placed.is_some() && named.is_some() {
                return Err(python_error(
                    "TypeError",
                    &format!("got multiple values for argument '{name}'"),
                ));
            }
            chosen.push(placed.cloned().or(named).filter(|value| !value.is_none()));
        }
        keywords
            .assert_all_used()
            .map_err(|_| python_error("TypeError", "got an unexpected keyword argument"))?;
        let indent = indent_text(chosen[1].as_ref())?;
        let (item_separator, key_separator) = separators(chosen[2].as_ref(), indent.is_some())?;
        Ok(Self {
            ensure_ascii: chosen[0].as_ref().is_some_and(Value::is_true),
            indent,
            item_separator,
            key_separator,
            sort_keys: chosen[3].as_ref().is_some_and(Value::is_true),
        })
    }
}

// An integer means that many spaces (none for a negative number). A string is used as is.
fn indent_text(value: Option<&Value>) -> Result<Option<String>, Error> {
    let Some(value) = value else {
        return Ok(None);
    };
    if let Some(text) = value.as_str() {
        return Ok(Some(text.to_string()));
    }
    let count = match value.kind() {
        ValueKind::Bool => Some(i64::from(value.is_true())),
        ValueKind::Number if value.is_integer() => value.as_i64(),
        _ => None,
    };
    match count {
        Some(count) if count > MAX_INDENT => Err(python_error(
            "ValueError",
            &format!("indent above {MAX_INDENT} is not supported"),
        )),
        Some(count) => Ok(Some(" ".repeat(usize::try_from(count).unwrap_or(0)))),
        None => Err(python_error(
            "TypeError",
            "indent must be None, an int, or a str",
        )),
    }
}

fn separators(value: Option<&Value>, indented: bool) -> Result<(String, String), Error> {
    let Some(value) = value else {
        let item = if indented { "," } else { ", " };
        return Ok((item.to_string(), ": ".to_string()));
    };
    if value.kind() != ValueKind::Seq {
        return Err(python_error(
            "TypeError",
            "separators must be a tuple of two strings",
        ));
    }
    let items: Vec<Value> = value.try_iter()?.collect();
    let [item, key] = items.as_slice() else {
        return Err(python_error(
            "ValueError",
            "separators must hold exactly two strings",
        ));
    };
    match (item.as_str(), key.as_str()) {
        (Some(item), Some(key)) => Ok((item.to_string(), key.to_string())),
        _ => Err(python_error("TypeError", "separators must hold strings")),
    }
}

fn write_value(value: &serde_json::Value, format: &Format, level: usize, out: &mut String) {
    match value {
        serde_json::Value::Null => out.push_str("null"),
        serde_json::Value::Bool(flag) => out.push_str(if *flag { "true" } else { "false" }),
        serde_json::Value::Number(number) => match number.as_f64().filter(|_| number.is_f64()) {
            Some(float) => out.push_str(&python_float(float)),
            None => {
                let _ = write!(out, "{number}");
            }
        },
        serde_json::Value::String(text) => write_string(text, format.ensure_ascii, out),
        serde_json::Value::Array(items) => {
            let rows: Vec<Row<'_>> = items.iter().map(|item| Row { key: None, item }).collect();
            write_container(('[', ']'), &rows, format, level, out);
        }
        serde_json::Value::Object(map) => {
            let mut rows: Vec<Row<'_>> = map
                .iter()
                .map(|(key, item)| Row {
                    key: Some(key.as_str()),
                    item,
                })
                .collect();
            if format.sort_keys {
                rows.sort_by(|left, right| left.key.cmp(&right.key));
            }
            write_container(('{', '}'), &rows, format, level, out);
        }
    }
}

// Python `float.__repr__`: the shortest digits that round-trip, in fixed notation
// for decimal exponents -4 to 15 and in exponent notation (two digits or more, with
// a sign) otherwise. The digits come from `serde_json`, whose shortest-digit
// algorithm (Ryu) picks the same digits as Python on 241,920 checked values.
// Rust `{:e}` differs from Python in 46 of them, at exact ties.
fn python_float(value: f64) -> String {
    if !value.is_finite() {
        // A JSON value never holds these. `serde_json` maps them to null.
        return "null".to_string();
    }
    let sign = if value.is_sign_negative() { "-" } else { "" };
    let Some((digits, exponent)) = shortest_digits(value) else {
        return value.to_string();
    };
    let body = if (-4..16).contains(&exponent) {
        fixed_notation(&digits, exponent)
    } else {
        let (first, rest) = digits.split_at(1);
        let point = if rest.is_empty() { "" } else { "." };
        let exponent_sign = if exponent < 0 { '-' } else { '+' };
        format!(
            "{first}{point}{rest}e{exponent_sign}{:02}",
            exponent.unsigned_abs()
        )
    };
    format!("{sign}{body}")
}

// The significant digits without trailing zeros, and the decimal exponent of the first digit.
fn shortest_digits(value: f64) -> Option<(String, i32)> {
    let text = serde_json::to_string(&value.abs()).ok()?;
    let (mantissa, exponent) = match text.split_once('e') {
        Some((mantissa, exponent)) => (mantissa, exponent.parse::<i32>().ok()?),
        None => (text.as_str(), 0),
    };
    let (whole, fraction) = mantissa.split_once('.').unwrap_or((mantissa, ""));
    let all = format!("{whole}{fraction}");
    let leading = i32::try_from(all.len() - all.trim_start_matches('0').len()).ok()?;
    let digits = all.trim_matches('0');
    if digits.is_empty() {
        return Some(("0".to_string(), 0));
    }
    let exponent = exponent + i32::try_from(whole.len()).ok()? - 1 - leading;
    Some((digits.to_string(), exponent))
}

fn fixed_notation(digits: &str, exponent: i32) -> String {
    match usize::try_from(exponent) {
        Ok(point) if digits.len() > point + 1 => {
            format!("{}.{}", &digits[..=point], &digits[point + 1..])
        }
        Ok(point) => format!("{digits}{}.0", "0".repeat(point + 1 - digits.len())),
        Err(_) => format!(
            "0.{}{digits}",
            "0".repeat(exponent.unsigned_abs() as usize - 1)
        ),
    }
}

struct Row<'a> {
    key: Option<&'a str>,
    item: &'a serde_json::Value,
}

// CPython `json.encoder`: with an indent, each item starts on a new line, and
// an empty container stays `[]` or `{}`.
fn write_container(
    brackets: (char, char),
    rows: &[Row<'_>],
    format: &Format,
    level: usize,
    out: &mut String,
) {
    out.push(brackets.0);
    if rows.is_empty() {
        out.push(brackets.1);
        return;
    }
    let line = |depth: usize, out: &mut String| {
        if let Some(indent) = &format.indent {
            out.push('\n');
            out.push_str(&indent.repeat(depth));
        }
    };
    line(level + 1, out);
    for (index, row) in rows.iter().enumerate() {
        if index > 0 {
            out.push_str(&format.item_separator);
            line(level + 1, out);
        }
        if let Some(key) = row.key {
            write_string(key, format.ensure_ascii, out);
            out.push_str(&format.key_separator);
        }
        write_value(row.item, format, level + 1, out);
    }
    line(level, out);
    out.push(brackets.1);
}

// Escapes `"`, `\`, and U+0000 to U+001F. With `ensure_ascii`, it also escapes
// every character outside U+0020 to U+007E (so DEL), as UTF-16 units.
fn write_string(text: &str, ensure_ascii: bool, out: &mut String) {
    out.push('"');
    for character in text.chars() {
        match character {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            ' '..='~' => out.push(character),
            other if other < ' ' || ensure_ascii => {
                let mut units = [0u16; 2];
                for unit in other.encode_utf16(&mut units).iter() {
                    let _ = write!(out, "\\u{unit:04x}");
                }
            }
            other => out.push(other),
        }
    }
    out.push('"');
}
