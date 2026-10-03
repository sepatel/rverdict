//! Packs a question into the option-marker layout the weights were trained
//! on, ported from Von's `pack_sequence` and `_fit_state` (Apache-2.0).

use std::path::Path;
use std::sync::LazyLock;

use regex::Regex;
use rverdict_model::PackedSequence;
use tokenizers::Tokenizer;

use crate::EngineError;

const MASK: &str = "[MASK]";
const SEP: &str = "[SEP]";
/// Upper bound on state tokens before truncation, below the context window.
const MAX_STATE_TOKENS: usize = 8192;

pub struct Packer {
    tokenizer: Tokenizer,
    mask_id: u32,
    digit_split: bool,
    window: usize,
}

/// A state cut to fit the window.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Cut {
    pub state_tokens: usize,
    pub kept_tokens: usize,
}

impl Packer {
    pub fn new(tokenizer: &Path, digit_split: bool, window: usize) -> Result<Self, EngineError> {
        let tokenizer =
            Tokenizer::from_file(tokenizer).map_err(|e| EngineError::Tokenizer(e.to_string()))?;
        let mask_id = tokenizer
            .token_to_id(MASK)
            .ok_or_else(|| EngineError::Tokenizer("tokenizer has no [MASK] token".into()))?;
        Ok(Self {
            tokenizer,
            mask_id,
            digit_split,
            window,
        })
    }

    fn ids(&self, text: &str, specials: bool) -> Result<Vec<u32>, EngineError> {
        let encoding = self
            .tokenizer
            .encode(text, specials)
            .map_err(|e| EngineError::Tokenizer(e.to_string()))?;
        Ok(encoding.get_ids().to_vec())
    }

    fn decode(&self, ids: &[u32]) -> Result<String, EngineError> {
        let text = self
            .tokenizer
            .decode(ids, false)
            .map_err(|e| EngineError::Tokenizer(e.to_string()))?;
        Ok(clean_up_tokenization(&text))
    }

    /// `"<question> <state> [SEP] [MASK] <opt0> [MASK] <opt1> …"`, with
    /// special-token literals in user text broken so they cannot forge
    /// structure.
    fn text(&self, state: &str, question: &str, options: &[String]) -> String {
        let (state, question) = (neutralise(state), neutralise(question));
        let prefix = if question.is_empty() {
            state.trim().to_owned()
        } else {
            format!("{question} {state}").trim().to_owned()
        };
        let options = options
            .iter()
            .map(|o| format!("{MASK} {}", neutralise(o).trim()))
            .collect::<Vec<_>>()
            .join(" ");
        let packed = format!("{prefix} {SEP} {options}");
        if self.digit_split {
            split_digits(&packed)
        } else {
            packed
        }
    }

    pub fn pack(
        &self,
        state: &str,
        question: &str,
        options: &[String],
    ) -> Result<PackedSequence, EngineError> {
        let token_ids = self.ids(&self.text(state, question, options), true)?;
        let markers: Vec<usize> = token_ids
            .iter()
            .enumerate()
            .filter_map(|(i, &t)| (t == self.mask_id).then_some(i))
            .collect();
        if markers.len() != options.len() {
            return Err(EngineError::Tokenizer(format!(
                "packed {} option markers for {} options",
                markers.len(),
                options.len()
            )));
        }
        Ok(PackedSequence { token_ids, markers })
    }

    /// Tokens in `text` without special tokens, at least 1.
    pub fn count(&self, text: &str) -> Result<usize, EngineError> {
        Ok(self.ids(text, false)?.len().max(1))
    }

    /// Cuts the middle out of a state that would not fit alongside the
    /// question and options: headers and rules sit at the top, ledgers and
    /// events at the bottom, so 60% head and 40% tail are kept.
    pub fn fit_state(
        &self,
        state: &str,
        question: &str,
        options: &[String],
    ) -> Result<(String, Option<Cut>), EngineError> {
        let reserve = self.ids(&self.text("", question, options), true)?.len() + 8;
        let limit = MAX_STATE_TOKENS
            .min(self.window.saturating_sub(reserve))
            .max(16);
        let text = if self.digit_split {
            split_digits(state)
        } else {
            state.to_owned()
        };
        let ids = self.ids(&text, false)?;
        if ids.len() <= limit {
            return Ok((state.to_owned(), None));
        }
        let head = limit * 3 / 5;
        let tail = limit - head - 2;
        let fitted = format!(
            "{} ... {}",
            self.decode(&ids[..head])?,
            self.decode(&ids[ids.len() - tail..])?
        );
        Ok((
            fitted,
            Some(Cut {
                state_tokens: ids.len(),
                kept_tokens: limit,
            }),
        ))
    }
}

fn neutralise(text: &str) -> String {
    [MASK, SEP].iter().fold(text.to_owned(), |text, special| {
        let (head, tail) = special.split_at(1);
        text.replace(special, &format!("{head}\u{200d}{tail}"))
    })
}

/// `"2026"` → `"2 0 2 6"`: ModernBERT's BPE merges digit runs inconsistently.
fn split_digits(text: &str) -> String {
    static DIGITS: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\d+").expect("valid regex"));
    DIGITS
        .replace_all(text, |caps: &regex::Captures| {
            caps[0]
                .chars()
                .map(String::from)
                .collect::<Vec<_>>()
                .join(" ")
        })
        .into_owned()
}

/// `transformers`' `clean_up_tokenization_spaces`, which Von's tokenizer
/// config enables for decoding.
fn clean_up_tokenization(text: &str) -> String {
    [
        (" .", "."),
        (" ?", "?"),
        (" !", "!"),
        (" ,", ","),
        (" ' ", "'"),
        (" n't", "n't"),
        (" 'm", "'m"),
        (" 's", "'s"),
        (" 've", "'ve"),
        (" 're", "'re"),
    ]
    .iter()
    .fold(text.to_owned(), |text, (from, to)| text.replace(from, to))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn digit_runs_are_spaced_and_specials_are_broken() {
        assert_eq!(
            split_digits("Invoice 2026-0912, 42"),
            "Invoice 2 0 2 6-0 9 1 2, 4 2"
        );
        assert_eq!(
            neutralise("a [MASK] b [SEP]"),
            "a [\u{200d}MASK] b [\u{200d}SEP]"
        );
    }
}
