//! Public datasets turned into labelled decisions, fetched at evaluation
//! time through the Hugging Face dataset viewer API and cached locally. They
//! are never committed or redistributed.

use std::path::{Path, PathBuf};

use serde_json::{Map, Value, json};

use crate::task::Task;
use crate::{EvalError, fnv};

/// Rows per dataset viewer request; the API's maximum.
const PAGE: usize = 100;

/// Fetches every row of `dataset`'s `split`, caching them as JSONL.
fn rows(dataset: &str, split: &str) -> Result<Vec<Map<String, Value>>, EvalError> {
    let path: PathBuf = rverdict_core::cache_root()
        .join("datasets")
        .join(dataset.replace('/', "__"))
        .join(format!("{split}.jsonl"));
    if !path.exists() {
        let mut lines = String::new();
        let mut offset = 0;
        loop {
            let url = format!(
                "https://datasets-server.huggingface.co/rows?dataset={dataset}&config=default&split={split}&offset={offset}&length={PAGE}"
            );
            let page: Value =
                serde_json::from_str(&reqwest::blocking::get(&url)?.error_for_status()?.text()?)?;
            let batch = page["rows"].as_array().cloned().unwrap_or_default();
            for row in &batch {
                lines.push_str(&row["row"].to_string());
                lines.push('\n');
            }
            offset += batch.len();
            if batch.len() < PAGE {
                break;
            }
        }
        write(&path, &lines)?;
    }
    let text = std::fs::read_to_string(&path).map_err(|source| EvalError::Io {
        path: path.clone(),
        source,
    })?;
    text.lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| Ok(serde_json::from_str(l)?))
        .collect()
}

fn write(path: &Path, text: &str) -> Result<(), EvalError> {
    let dir = path.parent().expect("cache paths have a parent");
    std::fs::create_dir_all(dir).map_err(|source| EvalError::Io {
        path: dir.to_owned(),
        source,
    })?;
    std::fs::write(path, text).map_err(|source| EvalError::Io {
        path: path.to_owned(),
        source,
    })
}

/// Enron-Spam (`SetFit/enron_spam`): real emails labelled spam or ham. Each
/// sampled email becomes two decisions: a zero-shot `noul` and a described
/// `choice`. The sample is the first `limit` emails in a stable hash order.
pub fn enron_spam(split: &str, limit: usize) -> Result<Vec<Task>, EvalError> {
    let mut rows = rows("SetFit/enron_spam", split)?;
    rows.sort_by_key(|r| fnv(&r["message_id"].to_string()));
    rows.truncate(limit);

    let mut tasks = Vec::with_capacity(rows.len() * 2);
    for row in rows {
        let id = row["message_id"].to_string().trim_matches('"').to_owned();
        let spam = row["label_text"].as_str() == Some("spam");
        let subject = row["subject"].as_str().unwrap_or_default();
        let message = row["message"].as_str().unwrap_or_default();
        let state = Value::String(format!("Subject: {subject}\n\n{message}"));
        tasks.push(task(
            format!("{id}-noul"),
            "noul",
            state.clone(),
            json!({"type": "noul", "instructions": "Is this email spam: unsolicited bulk, commercial or scam mail?"}),
            json!(if spam { "yes" } else { "no" }),
        ));
        tasks.push(task(
            format!("{id}-choice"),
            "choice",
            state,
            json!({"type": "choice", "instructions": "What kind of email is this?", "criteria": {
                "spam": "Unsolicited bulk, marketing or scam email",
                "legitimate": "A genuine personal or work email"
            }}),
            json!(if spam { "spam" } else { "legitimate" }),
        ));
    }
    Ok(tasks)
}

fn task(id: String, group: &str, state: Value, question: Value, expected: Value) -> Task {
    Task {
        id,
        subset: group.to_owned(),
        family: "enron_spam".into(),
        state,
        question: match question {
            Value::Object(map) => map,
            _ => Map::new(),
        },
        expected: Some(expected),
        labels: Vec::new(),
    }
}
