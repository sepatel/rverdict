//! Turns wire questions into model text.
//!
//! Structured values are rendered exactly as Von's reference server renders
//! them (Python's `json.dumps` and `str()`), because the published weights
//! were trained on that text and parity depends on it.

use std::fmt::Write;

use serde_json::{Map, Number, Value};

use crate::wire::{NoulCriteria, Question};

/// Default option texts for a `noul` without criteria.
pub const NOUL_TRUE: &str = "Yes, condition holds true.";
pub const NOUL_FALSE: &str = "No, condition is false.";

/// A question reduced to text: one instruction and one description per option.
#[derive(Debug, Clone, PartialEq)]
pub struct Rendered {
    pub instructions: String,
    pub options: Vec<String>,
    pub kind: RenderedKind,
}

#[derive(Debug, Clone, PartialEq)]
pub enum RenderedKind {
    /// Options are `[true, false]`. `explicit` is false when the defaults were
    /// used, which enables zero-shot debiasing.
    Noul { explicit: bool },
    /// Option keys in request order, parallel to `options`.
    Choice { keys: Vec<String> },
    /// Level descriptions, parallel to `options`.
    Score,
}

pub fn render(question: &Question) -> Rendered {
    let instructions = instructions_text(question.instructions());
    match question {
        Question::Noul { criteria, .. } => {
            let (yes, no) = noul_texts(criteria.as_ref());
            let explicit = yes.is_some() || no.is_some();
            Rendered {
                instructions,
                options: vec![
                    yes.unwrap_or_else(|| NOUL_TRUE.to_owned()),
                    no.unwrap_or_else(|| NOUL_FALSE.to_owned()),
                ],
                kind: RenderedKind::Noul { explicit },
            }
        }
        Question::Choice { criteria, .. } => {
            let options = criteria
                .iter()
                .map(|(key, desc)| match description_text(desc) {
                    Some(text) if !text.trim().is_empty() => text.trim().to_owned(),
                    _ => key.trim().to_owned(),
                })
                .collect();
            Rendered {
                instructions,
                options,
                kind: RenderedKind::Choice {
                    keys: criteria.keys().cloned().collect(),
                },
            }
        }
        Question::Score { criteria, .. } => Rendered {
            instructions,
            options: criteria.iter().map(level_text).collect(),
            kind: RenderedKind::Score,
        },
    }
}

fn noul_texts(criteria: Option<&NoulCriteria>) -> (Option<String>, Option<String>) {
    let text = |v: Option<&Value>| v.and_then(description_text).filter(|t| !t.is_empty());
    criteria.map_or((None, None), |c| {
        (text(c.yes.as_ref()), text(c.no.as_ref()))
    })
}

/// `json.dumps(v, sort_keys=isinstance(v, dict))` for structure, the text
/// itself for strings.
fn instructions_text(value: &Value) -> String {
    match value {
        Value::String(s) => s.clone(),
        Value::Object(_) => python_json(value, true),
        _ => python_json(value, false),
    }
}

/// Von's `_stringify_criteria`: `None` when absent, JSON for structure,
/// `str()` for scalars.
fn description_text(value: &Value) -> Option<String> {
    match value {
        Value::Null => None,
        Value::String(s) => Some(s.clone()),
        Value::Object(_) => Some(python_json(value, true)),
        Value::Array(_) => Some(python_json(value, false)),
        scalar => Some(python_str(scalar)),
    }
}

/// A score level: `{"what", "examples"}` objects become
/// `"<what> Examples: a, b"`, anything else its `str()`.
fn level_text(level: &Value) -> String {
    match level {
        Value::Object(map) => {
            let what = map.get("what").map(python_str).unwrap_or_default();
            let examples: Vec<String> = match map.get("examples") {
                Some(Value::Array(items)) => items.iter().map(python_str).collect(),
                _ => Vec::new(),
            };
            let suffix = if examples.is_empty() {
                String::new()
            } else {
                format!(" Examples: {}", examples.join(", "))
            };
            format!("{what}{suffix}").trim().to_owned()
        }
        other => python_str(other).trim().to_owned(),
    }
}

/// Von's `_format_state`: strings as-is, objects as `key: value` lines,
/// anything else its `str()`.
pub fn state_text(state: &Value) -> String {
    match state {
        Value::String(s) => s.clone(),
        Value::Object(map) => map
            .iter()
            .map(|(k, v)| format!("{k}: {}", python_str(v)))
            .collect::<Vec<_>>()
            .join("\n"),
        other => python_str(other),
    }
}

/// Python's `str(value)`: strings bare, everything else its `repr`.
pub fn python_str(value: &Value) -> String {
    match value {
        Value::String(s) => s.clone(),
        other => python_repr(other),
    }
}

fn python_repr(value: &Value) -> String {
    match value {
        Value::Null => "None".into(),
        Value::Bool(b) => if *b { "True" } else { "False" }.into(),
        Value::Number(n) => python_number(n),
        Value::String(s) => python_str_repr(s),
        Value::Array(items) => format!(
            "[{}]",
            items.iter().map(python_repr).collect::<Vec<_>>().join(", ")
        ),
        Value::Object(map) => format!(
            "{{{}}}",
            map.iter()
                .map(|(k, v)| format!("{}: {}", python_str_repr(k), python_repr(v)))
                .collect::<Vec<_>>()
                .join(", ")
        ),
    }
}

/// `repr()` of a Python `str`: single quotes unless the text contains a
/// single quote and no double quote.
fn python_str_repr(s: &str) -> String {
    let quote = if s.contains('\'') && !s.contains('"') {
        '"'
    } else {
        '\''
    };
    let mut out = String::with_capacity(s.len() + 2);
    out.push(quote);
    for c in s.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c == quote => {
                out.push('\\');
                out.push(c);
            }
            c if (c as u32) < 0x20 || c as u32 == 0x7f => {
                let _ = write!(out, "\\x{:02x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out.push(quote);
    out
}

/// `json.dumps` with Python's defaults: `", "`/`": "` separators and
/// `ensure_ascii=True`. `sort_keys` applies to every nested object.
pub fn python_json(value: &Value, sort_keys: bool) -> String {
    let mut out = String::new();
    write_json(&mut out, value, sort_keys);
    out
}

fn write_json(out: &mut String, value: &Value, sort_keys: bool) {
    match value {
        Value::Null => out.push_str("null"),
        Value::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        Value::Number(n) => out.push_str(&python_number(n)),
        Value::String(s) => write_json_string(out, s),
        Value::Array(items) => {
            out.push('[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                write_json(out, item, sort_keys);
            }
            out.push(']');
        }
        Value::Object(map) => {
            out.push('{');
            for (i, (k, v)) in entries(map, sort_keys).into_iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                write_json_string(out, k);
                out.push_str(": ");
                write_json(out, v, sort_keys);
            }
            out.push('}');
        }
    }
}

fn entries(map: &Map<String, Value>, sort: bool) -> Vec<(&String, &Value)> {
    let mut entries: Vec<_> = map.iter().collect();
    if sort {
        entries.sort_by(|a, b| a.0.cmp(b.0));
    }
    entries
}

fn write_json_string(out: &mut String, s: &str) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{08}' => out.push_str("\\b"),
            '\u{0c}' => out.push_str("\\f"),
            c if c.is_ascii() && (c as u32) >= 0x20 => out.push(c),
            c => {
                let mut units = [0u16; 2];
                for unit in c.encode_utf16(&mut units) {
                    let _ = write!(out, "\\u{unit:04x}");
                }
            }
        }
    }
    out.push('"');
}

/// Python's `repr` of an int or float.
fn python_number(n: &Number) -> String {
    if n.is_i64() || n.is_u64() {
        return n.to_string();
    }
    let f = n.as_f64().unwrap_or(f64::NAN);
    if !f.is_finite() {
        return if f.is_nan() {
            "NaN".into()
        } else if f > 0.0 {
            "Infinity".into()
        } else {
            "-Infinity".into()
        };
    }
    // Shortest round-trip digits, then Python's layout: positional for
    // exponents in [-4, 16), scientific with a signed two-digit exponent
    // otherwise, and always a decimal point in positional form.
    let sci = format!("{f:e}");
    let (mantissa, exponent) = sci.split_once('e').expect("{:e} always has an exponent");
    let exponent: i32 = exponent.parse().expect("{:e} exponent is an integer");
    if (-4..16).contains(&exponent) {
        let positional = format!("{f}");
        if positional.contains('.') {
            positional
        } else {
            format!("{positional}.0")
        }
    } else {
        let sign = if exponent < 0 { '-' } else { '+' };
        format!("{mantissa}e{sign}{:02}", exponent.abs())
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn json_matches_python_dumps() {
        let v = json!({"b": [1, 2.5, "é"], "a": {"z": null, "y": true}});
        assert_eq!(
            python_json(&v, true),
            r#"{"a": {"y": true, "z": null}, "b": [1, 2.5, "\u00e9"]}"#
        );
        assert_eq!(
            python_json(&json!([{"b": 1, "a": 2}]), false),
            r#"[{"b": 1, "a": 2}]"#
        );
    }

    #[test]
    fn floats_match_python_repr() {
        for (value, python) in [
            (1.0, "1.0"),
            (0.0001, "0.0001"),
            (1e-5, "1e-05"),
            (1e16, "1e+16"),
            (1e15, "1000000000000000.0"),
            (-2.5, "-2.5"),
        ] {
            assert_eq!(
                python_number(&Number::from_f64(value).unwrap()),
                python,
                "{value}"
            );
        }
    }

    #[test]
    fn state_objects_render_as_python_str_lines() {
        let state =
            json!({"from": "a@b.c", "flags": ["x", "it's"], "n": 3, "ok": false, "meta": null});
        assert_eq!(
            state_text(&state),
            "from: a@b.c\nflags: ['x', \"it's\"]\nn: 3\nok: False\nmeta: None"
        );
    }

    #[test]
    fn choice_falls_back_to_the_key_and_score_levels_join_examples() {
        let choice: Question = serde_json::from_value(json!({
            "type": "choice", "instructions": {"q": "Which?", "data": [1]},
            "criteria": {"billing ": null, "tech": {"k": 1}}
        }))
        .unwrap();
        let rendered = render(&choice);
        assert_eq!(rendered.instructions, r#"{"data": [1], "q": "Which?"}"#);
        assert_eq!(rendered.options, ["billing", r#"{"k": 1}"#]);

        let score: Question = serde_json::from_value(json!({
            "type": "score", "instructions": "How bad?",
            "criteria": ["Fine", {"what": "Broken", "examples": ["crash", "data loss"]}]
        }))
        .unwrap();
        assert_eq!(
            render(&score).options,
            ["Fine", "Broken Examples: crash, data loss"]
        );
    }
}
