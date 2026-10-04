use std::path::Path;

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::EvalError;

/// One labelled decision: a state, one wire-format question, and the answer
/// it should get. JevBench items and user-provided JSONL share this shape:
///
/// ```json
/// {"id": "e1", "state": "…", "question": {"type": "noul", "instructions": "…"}, "expected": "yes"}
/// ```
///
/// `expected` is `"yes"`/`"no"` (or a boolean) for a `noul`, an option key for
/// a `choice`, and a level index for a `score`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Task {
    pub id: String,
    /// A grouping for reports, such as a benchmark tier.
    #[serde(default)]
    pub subset: String,
    #[serde(default)]
    pub family: String,
    pub state: Value,
    pub question: Map<String, Value>,
    /// `None` when the item has no agreed answer; such items are not scored.
    #[serde(default)]
    pub expected: Option<Value>,
    /// The answer labels JevBench scores against; derived from the question
    /// when absent.
    #[serde(default, deserialize_with = "labels")]
    pub labels: Vec<String>,
}

fn labels<'de, D: serde::Deserializer<'de>>(d: D) -> Result<Vec<String>, D::Error> {
    let raw = Vec::<Value>::deserialize(d)?;
    Ok(raw.iter().map(label).collect())
}

/// Labels compare as strings; score levels are integers and nouls may be
/// booleans in the files.
pub fn label(value: &Value) -> String {
    match value {
        Value::String(s) => s.clone(),
        Value::Bool(true) => "yes".into(),
        Value::Bool(false) => "no".into(),
        other => other.to_string(),
    }
}

impl Task {
    fn kind(&self) -> &str {
        self.question
            .get("type")
            .and_then(Value::as_str)
            .unwrap_or("")
    }

    /// Answer labels in option order: `yes`/`no` for a noul (true first),
    /// option keys for a choice, level indices for a score.
    pub fn option_labels(&self) -> Vec<String> {
        match (self.kind(), self.question.get("criteria")) {
            ("noul", _) => vec!["yes".into(), "no".into()],
            ("choice", Some(Value::Object(criteria))) => criteria.keys().cloned().collect(),
            ("score", Some(Value::Array(levels))) => {
                (0..levels.len()).map(|i| i.to_string()).collect()
            }
            _ => Vec::new(),
        }
    }

    /// The scored label set: the file's own when given, else the options.
    pub fn label_set(&self) -> Vec<String> {
        if self.labels.is_empty() {
            self.option_labels()
        } else {
            self.labels.clone()
        }
    }

    /// Index of the expected answer among [`Task::option_labels`].
    pub fn expected_index(&self) -> Option<usize> {
        let expected = label(self.expected.as_ref()?);
        self.option_labels().iter().position(|l| *l == expected)
    }

    /// Reverses a choice question's option order, for order-invariance runs.
    pub fn reverse_choice_options(&mut self) {
        if let Some(Value::Object(criteria)) = self.question.get_mut("criteria") {
            let reversed: Map<String, Value> = std::mem::take(criteria).into_iter().rev().collect();
            *criteria = reversed;
        }
    }

    pub fn read_jsonl(path: &Path) -> Result<Vec<Self>, EvalError> {
        let text = std::fs::read_to_string(path).map_err(|source| EvalError::Io {
            path: path.to_owned(),
            source,
        })?;
        text.lines()
            .filter(|l| !l.trim().is_empty())
            .map(|l| Ok(serde_json::from_str(l)?))
            .collect()
    }

    pub fn write_jsonl(tasks: &[Self], path: &Path) -> Result<(), EvalError> {
        let mut text = String::new();
        for task in tasks {
            text.push_str(&serde_json::to_string(task)?);
            text.push('\n');
        }
        std::fs::write(path, text).map_err(|source| EvalError::Io {
            path: path.to_owned(),
            source,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn jevbench_lines_parse_including_their_own_group_field() {
        let line = r#"{"expected": "yes", "family": "fact", "group": null, "id": "easy-fact-00",
            "labels": ["no", "yes"], "split": "public", "state": "The order shipped.",
            "question": {"type": "noul", "instructions": "Has the order shipped?"}}"#;
        let task: Task = serde_json::from_str(line).unwrap();
        assert_eq!(task.label_set(), ["no", "yes"]);
        assert_eq!(task.expected_index(), Some(0));
    }
}
