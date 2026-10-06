use crate::{Result, TokenizerError};
use fancy_regex::Regex;
use serde::{Deserialize, Serialize};
use std::ops::Range;

pub const GPT2_PATTERN: &str =
    r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum Pretokenizer {
    Raw,
    Gpt2 { pattern: String },
}

impl Pretokenizer {
    pub fn gpt2() -> Self {
        Self::Gpt2 {
            pattern: GPT2_PATTERN.into(),
        }
    }
}

pub(crate) fn compile(config: &Pretokenizer) -> Result<Option<Regex>> {
    match config {
        Pretokenizer::Raw => Ok(None),
        Pretokenizer::Gpt2 { pattern } if pattern == GPT2_PATTERN => Regex::new(pattern)
            .map(Some)
            .map_err(|e| TokenizerError::Preprocessing(e.to_string())),
        _ => Err(TokenizerError::InvalidModel(
            "only the canonical GPT-2 pattern is supported".into(),
        )),
    }
}

pub(crate) fn spans(text: &str, regex: Option<&Regex>) -> Result<Vec<Range<usize>>> {
    let mut spans = Vec::new();
    spans_into(text, regex, &mut spans)?;
    Ok(spans)
}

pub(crate) fn spans_into(text: &str, regex: Option<&Regex>, spans: &mut Vec<Range<usize>>) -> Result<()> {
    spans.clear();
    let Some(regex) = regex else {
        if !text.is_empty() { spans.push(0..text.len()); }
        return Ok(());
    };
    let mut end = 0;
    for matched in regex.find_iter(text) {
        let matched = matched.map_err(|e| TokenizerError::Preprocessing(e.to_string()))?;
        if matched.start() != end || matched.start() == matched.end() {
            return Err(TokenizerError::Preprocessing(
                "pattern skipped bytes or matched an empty span".into(),
            ));
        }
        spans.push(matched.range());
        end = matched.end();
    }
    if end != text.len() {
        return Err(TokenizerError::Preprocessing(
            "pattern did not cover all input bytes".into(),
        ));
    }
    Ok(())
}
