//! The System One wire format, as served by TypeSafe's `POST /v1/systemone`
//! and every compatible server (Cloudflare Workers AI, laya-serve, von,
//! strands-decider). Unknown fields are ignored.

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::OrderedMap;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Request {
    /// The content to evaluate: a string, or structured JSON.
    pub state: Value,
    /// Required by TypeSafe; optional here, where the loaded model answers.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    /// Question id → question. Order is preserved.
    pub questions: Map<String, Value>,
}

/// One typed question. `instructions` and option descriptions may be any
/// JSON value; [`crate::render`] turns them into model text.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum Question {
    Noul {
        instructions: Value,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        criteria: Option<NoulCriteria>,
    },
    Choice {
        instructions: Value,
        /// Option key → description, `null` when the key says enough.
        criteria: Map<String, Value>,
    },
    Score {
        instructions: Value,
        /// Ordered level descriptions, lowest first.
        criteria: Vec<Value>,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NoulCriteria {
    #[serde(default, rename = "true", skip_serializing_if = "Option::is_none")]
    pub yes: Option<Value>,
    #[serde(default, rename = "false", skip_serializing_if = "Option::is_none")]
    pub no: Option<Value>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Response {
    pub model: String,
    pub answers: OrderedMap<Answer>,
    pub usage: Usage,
    /// Present only when a state was cut to fit the context window.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub truncation: Option<Truncation>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum Answer {
    Noul {
        /// The committed yes/no probability.
        noul: f64,
        /// The calibrated probability before any decision rule; use this for
        /// confidence gating. Not part of TypeSafe's schema.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        noul_raw: Option<f64>,
    },
    Choice {
        choice: String,
        probabilities: OrderedMap<f64>,
        confidence: f64,
    },
    Score {
        score: f64,
        legend: OrderedMap<String>,
        probabilities: OrderedMap<f64>,
        confidence: f64,
    },
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Usage {
    pub input_tokens: usize,
    pub output_tokens: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Truncation {
    /// Tokens in the longest state that had to be cut.
    pub state_tokens: usize,
    /// Tokens kept from it.
    pub kept_tokens: usize,
    pub strategy: String,
    pub questions_affected: usize,
}

#[derive(Debug, thiserror::Error)]
#[error("question {id:?}: {reason}")]
pub struct InvalidQuestion {
    pub id: String,
    pub reason: String,
}

impl Request {
    /// Parses every question, rejecting the request on the first malformed
    /// one (served as HTTP 422).
    pub fn parse_questions(&self) -> Result<Vec<(String, Question)>, InvalidQuestion> {
        self.questions
            .iter()
            .map(|(id, raw)| {
                let invalid = |reason: String| InvalidQuestion {
                    id: id.clone(),
                    reason,
                };
                let question: Question =
                    serde_json::from_value(raw.clone()).map_err(|e| invalid(e.to_string()))?;
                question.validate().map_err(invalid)?;
                Ok((id.clone(), question))
            })
            .collect()
    }
}

impl Question {
    pub fn instructions(&self) -> &Value {
        match self {
            Self::Noul { instructions, .. }
            | Self::Choice { instructions, .. }
            | Self::Score { instructions, .. } => instructions,
        }
    }

    fn validate(&self) -> Result<(), String> {
        match self {
            Self::Choice { criteria, .. } if criteria.is_empty() => {
                Err("a choice needs at least one option".into())
            }
            Self::Choice { criteria, .. } if criteria.len() > 255 => {
                Err("a choice allows at most 255 options".into())
            }
            Self::Score { criteria, .. } if !(2..=10).contains(&criteria.len()) => Err(format!(
                "a score needs 2 to 10 levels, got {}",
                criteria.len()
            )),
            _ => Ok(()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn typesafe_example_request_parses_and_keeps_option_order() {
        let request: Request = serde_json::from_str(
            r#"{
                "state": "Help! My payouts have been failing for 3 days.",
                "model": "jev-latest",
                "questions": {
                    "department": {"type": "choice", "instructions": "Which team?",
                        "criteria": {"sales": "Pricing", "billing": "Payments", "technical": null}},
                    "frustration": {"type": "score", "instructions": "How frustrated?",
                        "criteria": ["Calm", "Frustrated", "Very angry"]},
                    "urgent": {"type": "noul", "instructions": "Urgent?",
                        "criteria": {"true": "Time-sensitive", "false": "No urgency"}},
                    "ignored_extra": {"type": "noul", "instructions": "x", "unknown": 1}
                }
            }"#,
        )
        .unwrap();
        let questions = request.parse_questions().unwrap();
        let Question::Choice { criteria, .. } = &questions[0].1 else {
            panic!("expected a choice");
        };
        assert_eq!(
            criteria.keys().collect::<Vec<_>>(),
            ["sales", "billing", "technical"]
        );
        assert_eq!(questions.len(), 4);
    }

    #[test]
    fn malformed_question_names_the_question() {
        let request: Request = serde_json::from_str(
            r#"{"state": "x", "questions": {"q1": {"type": "score", "instructions": "x", "criteria": ["one"]}}}"#,
        )
        .unwrap();
        let err = request.parse_questions().unwrap_err();
        assert_eq!(err.id, "q1");
    }
}
