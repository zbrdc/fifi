//! Python `str` methods that chat templates call.
//!
//! The reference stack renders templates with Python, so a template can call
//! `startswith`, `endswith`, `split`, `lstrip`, and `rstrip` on a string. This
//! module ports those five methods with the Python argument rules: positional
//! and keyword forms, `None` defaults, tuple prefixes, character-index slices,
//! and the Unicode whitespace set of `str.isspace`. A call that raises a
//! `TypeError` or `ValueError` in Python returns an error here.

use minijinja::value::{Kwargs, Rest, Value, ValueKind, from_args};
use minijinja::{Error, ErrorKind};

#[derive(Clone, Copy)]
enum Side {
    Start,
    End,
}

/// Runs the named method, or returns `None` when the engine does not port it.
pub(super) fn call(text: &str, method: &str, args: &[Value]) -> Option<Result<Value, Error>> {
    let result = match method {
        "startswith" => affix(text, method, args, Side::Start),
        "endswith" => affix(text, method, args, Side::End),
        "split" => split(text, args),
        "lstrip" => strip(text, method, args, Side::Start),
        "rstrip" => strip(text, method, args, Side::End),
        _ => return None,
    };
    Some(result)
}

// `str.isspace` is true for Unicode category Zs or bidirectional class WS, B, or S.
// `char::is_whitespace` (White_Space) misses U+001C to U+001F, which are class B or S.
fn is_python_space(character: char) -> bool {
    character.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&character)
}

fn python_error(kind: &str, method: &str, message: &str) -> Error {
    Error::new(
        ErrorKind::InvalidOperation,
        format!("{kind}: {method}() {message}"),
    )
}

fn type_error(method: &str, message: &str) -> Error {
    python_error("TypeError", method, message)
}

fn positional_only<'a>(method: &str, args: &'a [Value]) -> Result<&'a [Value], Error> {
    let (positional, keywords) = from_args::<(&[Value], Kwargs)>(args)?;
    keywords
        .assert_all_used()
        .map_err(|_| type_error(method, "takes no keyword arguments"))?;
    Ok(positional)
}

fn optional_index(method: &str, value: Option<&Value>) -> Result<Option<i64>, Error> {
    match value {
        None => Ok(None),
        Some(item) if item.is_none() => Ok(None),
        Some(item) => item
            .as_i64()
            .map(Some)
            .ok_or_else(|| type_error(method, "slice indices must be integers or None")),
    }
}

fn affix(text: &str, method: &str, args: &[Value], side: Side) -> Result<Value, Error> {
    let positional = positional_only(method, args)?;
    let Some((wanted, bounds)) = positional.split_first() else {
        return Err(type_error(method, "takes at least 1 argument (0 given)"));
    };
    if bounds.len() > 2 {
        return Err(type_error(
            method,
            &format!("takes at most 3 arguments ({} given)", positional.len()),
        ));
    }
    let start = optional_index(method, bounds.first())?;
    let end = optional_index(method, bounds.get(1))?;
    let characters: Vec<char> = text.chars().collect();
    let (start, end) = adjust_indices(characters.len(), start, end);
    let matched = any_affix_matches(method, wanted, |candidate| {
        tail_match(&characters, candidate, start, end, side)
    })?;
    Ok(Value::from(matched))
}

// A tuple of affixes stops at the first match, so a bad element after a match
// is never seen. This is the CPython order. `minijinja` builds tuples and lists
// as the same sequence kind, so a list argument is accepted as a tuple. Python
// raises `TypeError` for a list. Accepting it is the closest behavior available.
fn any_affix_matches(
    method: &str,
    wanted: &Value,
    matches: impl Fn(&[char]) -> bool,
) -> Result<bool, Error> {
    if let Some(single) = wanted.as_str() {
        return Ok(matches(&single.chars().collect::<Vec<_>>()));
    }
    if wanted.kind() != ValueKind::Seq {
        return Err(type_error(
            method,
            &format!(
                "first arg must be str or a tuple of str, not {}",
                kind_name(wanted)
            ),
        ));
    }
    for item in wanted.try_iter()? {
        let Some(single) = item.as_str() else {
            return Err(type_error(
                method,
                &format!("tuple must only contain str, not {}", kind_name(&item)),
            ));
        };
        if matches(&single.chars().collect::<Vec<_>>()) {
            return Ok(true);
        }
    }
    Ok(false)
}

fn kind_name(value: &Value) -> String {
    format!("{:?}", value.kind()).to_lowercase()
}

// CPython ADJUST_INDICES: clamp `end`, then resolve negative indices from the end.
fn adjust_indices(length: usize, start: Option<i64>, end: Option<i64>) -> (usize, usize) {
    let length = i64::try_from(length).unwrap_or(i64::MAX);
    let mut end = end.unwrap_or(length);
    if end > length {
        end = length;
    } else if end < 0 {
        end = (end + length).max(0);
    }
    let mut start = start.unwrap_or(0);
    if start < 0 {
        start = (start + length).max(0);
    }
    (
        usize::try_from(start).unwrap_or(usize::MAX),
        usize::try_from(end).unwrap_or(0),
    )
}

fn tail_match(characters: &[char], wanted: &[char], start: usize, end: usize, side: Side) -> bool {
    if end < start || end - start < wanted.len() {
        return false;
    }
    match side {
        Side::Start => characters[start..start + wanted.len()] == *wanted,
        Side::End => characters[end - wanted.len()..end] == *wanted,
    }
}

fn split(text: &str, args: &[Value]) -> Result<Value, Error> {
    let (positional, keywords) = from_args::<(&[Value], Kwargs)>(args)?;
    if positional.len() > 2 {
        return Err(type_error(
            "split",
            &format!("takes at most 2 arguments ({} given)", positional.len()),
        ));
    }
    let separator = argument("sep", 0, positional, &keywords)?.filter(|value| !value.is_none());
    let limit = argument("maxsplit", 1, positional, &keywords)?;
    keywords
        .assert_all_used()
        .map_err(|_| type_error("split", "got an unexpected keyword argument"))?;
    let limit = match limit {
        None => None,
        Some(value) => match value.as_i64() {
            Some(count) => usize::try_from(count).ok(),
            None => {
                return Err(type_error(
                    "split",
                    &format!("{} cannot be interpreted as an integer", kind_name(&value)),
                ));
            }
        },
    };
    let parts = match separator {
        None => split_on_whitespace(text, limit),
        Some(value) => split_on_separator(text, &separator_text(&value)?, limit)?,
    };
    Ok(Value::from_iter(parts))
}

// Returns the value as given, including `None`. Only `sep` treats `None` as absent;
// `maxsplit=None` is a `TypeError` in Python. Naming an argument twice is an error.
fn argument(
    name: &str,
    index: usize,
    positional: &[Value],
    keywords: &Kwargs,
) -> Result<Option<Value>, Error> {
    let named = if keywords.has(name) {
        Some(keywords.get::<Value>(name)?)
    } else {
        None
    };
    let placed = positional.get(index);
    if named.is_some() && placed.is_some() {
        return Err(type_error(
            "split",
            &format!(
                "argument for split() given by name ('{name}') and position ({})",
                index + 1
            ),
        ));
    }
    Ok(placed.cloned().or(named))
}

fn separator_text(value: &Value) -> Result<String, Error> {
    match value.as_str() {
        Some("") => Err(python_error(
            "ValueError",
            "split",
            "got an empty separator",
        )),
        Some(text) => Ok(text.to_string()),
        None => Err(type_error(
            "split",
            &format!("must be str or None, not {}", kind_name(value)),
        )),
    }
}

fn split_on_separator(
    text: &str,
    separator: &str,
    limit: Option<usize>,
) -> Result<Vec<String>, Error> {
    Ok(match limit {
        None => text.split(separator).map(str::to_string).collect(),
        Some(count) => text
            .splitn(count.saturating_add(1), separator)
            .map(str::to_string)
            .collect(),
    })
}

// Runs of whitespace are one separator. The text yields no empty parts. After
// `limit` splits the rest keeps its trailing whitespace but loses its leading whitespace.
fn split_on_whitespace(text: &str, limit: Option<usize>) -> Vec<String> {
    let mut parts = Vec::new();
    let mut rest = text;
    loop {
        rest = rest.trim_start_matches(is_python_space);
        if rest.is_empty() {
            return parts;
        }
        if limit.is_some_and(|count| parts.len() >= count) {
            parts.push(rest.to_string());
            return parts;
        }
        let end = rest.find(is_python_space).unwrap_or(rest.len());
        parts.push(rest[..end].to_string());
        rest = &rest[end..];
    }
}

/// The `trim` filter of the reference renderer: `str(value).strip(chars)`.
/// `chars` takes its position or its name, as in Python.
pub(super) fn trim_filter(value: Value, arguments: Rest<Value>) -> Result<String, Error> {
    let (positional, keywords) = from_args::<(&[Value], Kwargs)>(arguments.as_slice())?;
    if positional.len() > 1 {
        return Err(type_error(
            "trim",
            &format!(
                "takes from 1 to 2 positional arguments but {} were given",
                positional.len() + 1
            ),
        ));
    }
    let named = if keywords.has("chars") {
        Some(keywords.get::<Value>("chars")?)
    } else {
        None
    };
    if named.is_some() && !positional.is_empty() {
        return Err(type_error(
            "trim",
            "got multiple values for argument 'chars'",
        ));
    }
    keywords
        .assert_all_used()
        .map_err(|_| type_error("trim", "got an unexpected keyword argument"))?;
    let chars = positional.first().or(named.as_ref());
    let text = if value.is_none() {
        "None".to_string()
    } else if value.kind() == ValueKind::Bool {
        if value.is_true() { "True" } else { "False" }.to_string()
    } else {
        value.to_string()
    };
    let trimmed = match chars_argument("strip", chars)? {
        None => text.trim_matches(is_python_space),
        Some(set) => text.trim_matches(|c| set.contains(c)),
    };
    Ok(trimmed.to_string())
}

// `None` or no argument means whitespace; a `str` is a set of characters.
fn chars_argument<'a>(method: &str, value: Option<&'a Value>) -> Result<Option<&'a str>, Error> {
    match value {
        None => Ok(None),
        Some(value) if value.is_none() => Ok(None),
        Some(value) => value.as_str().map(Some).ok_or_else(|| {
            type_error(
                method,
                &format!("arg must be None or str, not {}", kind_name(value)),
            )
        }),
    }
}

fn strip(text: &str, method: &str, args: &[Value], side: Side) -> Result<Value, Error> {
    let positional = positional_only(method, args)?;
    if positional.len() > 1 {
        return Err(type_error(
            method,
            &format!("expected at most 1 argument, got {}", positional.len()),
        ));
    }
    let characters = chars_argument(method, positional.first())?;
    let stripped = match (characters, side) {
        (None, Side::Start) => text.trim_start_matches(is_python_space),
        (None, Side::End) => text.trim_end_matches(is_python_space),
        (Some(set), Side::Start) => text.trim_start_matches(|c| set.contains(c)),
        (Some(set), Side::End) => text.trim_end_matches(|c| set.contains(c)),
    };
    Ok(Value::from(stripped))
}
