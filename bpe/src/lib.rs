//! A lossless, deterministic byte BPE reference implementation.
pub mod encode;
pub mod errors;
pub mod model;
pub mod pretokenize;
pub mod special;
pub mod trainer;

pub use errors::{Result, TokenizerError};
pub use model::{BpeModel, Merge, ModelData, Profile, Token, TokenId};
pub use pretokenize::{Pretokenizer, GPT2_PATTERN};
pub use special::{EncodeOptions, SpecialMode};
pub use trainer::{BpeTrainer, TrainConfig};

use fancy_regex::Regex;
use std::path::Path;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum Backend {
    #[default]
    Reference,
    /// Until Day 2, auto selects the reference implementation.
    Auto,
    Heap,
}

/// Immutable model and preprocessing compiled once per tokenizer.
pub struct Tokenizer {
    model: BpeModel,
    regex: Option<Regex>,
}

impl Tokenizer {
    pub fn new(model: BpeModel) -> Result<Self> {
        let regex = pretokenize::compile(model.pretokenizer())?;
        Ok(Self { model, regex })
    }
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        Self::new(BpeModel::load(path)?)
    }
    pub fn model(&self) -> &BpeModel {
        &self.model
    }
    pub fn pretoken_spans(&self, text: &str) -> Result<Vec<std::ops::Range<usize>>> {
        pretokenize::spans(text, self.regex.as_ref())
    }
    pub fn encode(&self, text: &str) -> Result<Vec<TokenId>> {
        self.encode_with(text, &EncodeOptions::default())
    }
    pub fn encode_with(&self, text: &str, options: &EncodeOptions) -> Result<Vec<TokenId>> {
        if options.backend == Backend::Heap {
            return Err(TokenizerError::UnsupportedBackend("heap".into()));
        }
        let mut ids = Vec::new();
        for piece in special::split(text, self.model.special_tokens(), &options.special)? {
            match piece {
                special::Piece::Special(id) => ids.push(id),
                special::Piece::Text(ordinary) => {
                    for span in pretokenize::spans(ordinary, self.regex.as_ref())? {
                        let initial = ordinary.as_bytes()[span]
                            .iter()
                            .map(|byte| self.model.base_id(*byte))
                            .collect();
                        ids.extend(encode::reference::encode(initial, &self.model));
                    }
                }
            }
        }
        Ok(ids)
    }
    /// Serial Day 1 batch: independent records, stable input order.
    pub fn encode_batch<S: AsRef<str>>(
        &self,
        texts: &[S],
        options: &EncodeOptions,
    ) -> Result<Vec<Vec<TokenId>>> {
        texts
            .iter()
            .map(|text| self.encode_with(text.as_ref(), options))
            .collect()
    }
    pub fn decode_bytes(&self, ids: &[TokenId]) -> Result<Vec<u8>> {
        self.model.decode_bytes(ids)
    }
    pub fn decode_utf8(&self, ids: &[TokenId]) -> Result<String> {
        self.model.decode_utf8(ids)
    }
    pub fn decode_lossy(&self, ids: &[TokenId]) -> Result<String> {
        Ok(String::from_utf8_lossy(&self.decode_bytes(ids)?).into_owned())
    }
}
