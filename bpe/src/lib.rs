//! Lossless byte BPE with interchangeable deterministic merge engines.
pub mod batch;
pub mod encode;
pub mod errors;
pub mod model;
pub mod pretokenize;
pub mod profile;
pub mod special;
pub mod trace;
pub mod trainer;

pub use batch::BatchEncoder;
pub use encode::CacheStats;
pub use errors::{Result, TokenizerError};
pub use model::{BpeModel, Merge, ModelData, Profile, Token, TokenId};
pub use pretokenize::{Pretokenizer, GPT2_PATTERN};
pub use special::{EncodeOptions, SpecialMode};
pub use trainer::{BpeTrainer, TrainConfig, TrainerBackend};

use fancy_regex::Regex;
use serde::{Deserialize, Serialize};
use std::{ops::Range, path::Path};

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Backend {
    #[default]
    Reference,
    /// Uses the scan unless the caller supplies a measured byte threshold.
    Auto,
    Heap,
}
impl std::str::FromStr for Backend {
    type Err = TokenizerError;
    fn from_str(value: &str) -> Result<Self> {
        match value {
            "reference" => Ok(Self::Reference),
            "heap" => Ok(Self::Heap),
            "auto" => Ok(Self::Auto),
            _ => Err(TokenizerError::UnsupportedBackend(value.into())),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct RuntimeConfig {
    pub heap_threshold: Option<usize>,
    pub reuse_buffers: bool,
    pub cache_capacity: usize,
    pub cache_bytes: usize,
    pub cache_max_piece_bytes: usize,
    /// Maximum retained element capacity per scratch vector (heap: four times).
    pub scratch_capacity_limit: usize,
}
impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            heap_threshold: None,
            reuse_buffers: true,
            cache_capacity: 0,
            cache_bytes: 8 * 1024 * 1024,
            cache_max_piece_bytes: 4096,
            scratch_capacity_limit: 65536,
        }
    }
}
impl RuntimeConfig {
    pub(crate) fn validate(&self) -> Result<()> {
        if self.heap_threshold.is_some_and(|value| value < 2) {
            return Err(TokenizerError::InvalidConfig("heap threshold must be at least two bytes".into()));
        }
        if self.cache_capacity > 0 && (self.cache_bytes == 0 || self.cache_max_piece_bytes == 0) {
            return Err(TokenizerError::InvalidConfig("enabled cache needs positive byte and piece limits".into()));
        }
        Ok(())
    }
}

#[derive(Default)]
pub(crate) struct WorkerState {
    scratch: encode::Scratch,
    spans: Vec<Range<usize>>,
    cache: encode::Cache,
}
impl WorkerState {
    pub(crate) fn stats(&self) -> CacheStats { self.cache.stats() }
    pub(crate) fn clear_cache(&mut self) { self.cache.clear(); }
    fn finish(&mut self, config: &RuntimeConfig) {
        let limit = if config.reuse_buffers { config.scratch_capacity_limit } else { 0 };
        self.scratch.trim(limit);
        if self.spans.capacity() > limit { self.spans = Vec::new(); }
    }
}

/// Immutable model and preprocessing compiled once per tokenizer.
pub struct Tokenizer {
    model: BpeModel,
    regex: Option<Regex>,
}

/// Reuses scratch and a bounded pre-token cache, tied to exactly one model.
pub struct EncodingSession<'a> {
    tokenizer: &'a Tokenizer,
    config: RuntimeConfig,
    state: WorkerState,
}
impl EncodingSession<'_> {
    pub fn encode(&mut self, text: &str, options: &EncodeOptions) -> Result<Vec<TokenId>> {
        self.tokenizer.encode_state(text, options, &self.config, &mut self.state)
    }
    pub fn clear_cache(&mut self) { self.state.clear_cache(); }
    pub fn cache_stats(&self) -> CacheStats { self.state.stats() }
}

impl Tokenizer {
    pub fn new(model: BpeModel) -> Result<Self> {
        let regex = pretokenize::compile(model.pretokenizer())?;
        Ok(Self { model, regex })
    }
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        Self::new(BpeModel::load(path)?)
    }
    pub fn model(&self) -> &BpeModel { &self.model }
    pub fn pretoken_spans(&self, text: &str) -> Result<Vec<Range<usize>>> {
        pretokenize::spans(text, self.regex.as_ref())
    }
    pub fn session(&self, config: RuntimeConfig) -> Result<EncodingSession<'_>> {
        config.validate()?;
        Ok(EncodingSession { tokenizer: self, config, state: WorkerState::default() })
    }
    pub fn encode(&self, text: &str) -> Result<Vec<TokenId>> {
        self.encode_with(text, &EncodeOptions::default())
    }
    pub fn encode_with(&self, text: &str, options: &EncodeOptions) -> Result<Vec<TokenId>> {
        self.session(RuntimeConfig::default())?.encode(text, options)
    }
    pub(crate) fn encode_state(&self, text: &str, options: &EncodeOptions, config: &RuntimeConfig, state: &mut WorkerState) -> Result<Vec<TokenId>> {
        let result = (|| {
            let mut ids = Vec::new();
            for piece in special::split(text, self.model.special_tokens(), &options.special)? {
                match piece {
                    special::Piece::Special(id) => ids.push(id),
                    special::Piece::Text(ordinary) => {
                        pretokenize::spans_into(ordinary, self.regex.as_ref(), &mut state.spans)?;
                        for span in &state.spans {
                            let bytes = &ordinary.as_bytes()[span.clone()];
                            let cached = config.cache_capacity > 0 && bytes.len() <= config.cache_max_piece_bytes;
                            if cached {
                                if let Some(tokens) = state.cache.get(bytes) {
                                    ids.extend_from_slice(tokens);
                                    continue;
                                }
                            }
                            let start = ids.len();
                            let backend = encode::selected(options.backend, bytes.len(), config);
                            if config.reuse_buffers {
                                state.scratch.encode(bytes, &self.model, backend, &mut ids);
                            } else if backend == Backend::Reference {
                                let initial = bytes.iter().map(|byte| self.model.base_id(*byte)).collect();
                                ids.extend(encode::reference::encode(initial, &self.model));
                            } else {
                                encode::Scratch::default().encode(bytes, &self.model, backend, &mut ids);
                            }
                            if cached {
                                state.cache.insert(bytes, &ids[start..], config.cache_capacity, config.cache_bytes);
                            }
                        }
                    }
                }
            }
            Ok(ids)
        })();
        state.finish(config);
        result
    }
    /// Serial, stable-order batch using one reusable worker.
    pub fn encode_batch<S: AsRef<str>>(&self, texts: &[S], options: &EncodeOptions) -> Result<Vec<Vec<TokenId>>> {
        if texts.is_empty() { self.encode_with("", options)?; }
        let mut session = self.session(RuntimeConfig::default())?;
        texts.iter().map(|text| session.encode(text.as_ref(), options)).collect()
    }
    pub fn batch_encoder(&self, threads: usize, config: RuntimeConfig) -> Result<BatchEncoder<'_>> {
        BatchEncoder::new(self, threads, config)
    }
    /// Tracing bypasses caches and records every actual merge, including on hits.
    pub fn trace(&self, text: &str, options: &EncodeOptions, config: &RuntimeConfig) -> Result<trace::EncodingTrace> {
        config.validate()?;
        let mut trace = trace::EncodingTrace { ids: Vec::new(), events: Vec::new() };
        let mut offset = 0;
        for piece in special::split(text, self.model.special_tokens(), &options.special)? {
            match piece {
                special::Piece::Special(id) => {
                    trace.ids.push(id);
                    offset += self.model.token_bytes(id)?.len();
                }
                special::Piece::Text(ordinary) => {
                    for span in self.pretoken_spans(ordinary)? {
                        let bytes = &ordinary.as_bytes()[span.clone()];
                        encode::trace_piece(bytes, &self.model, encode::selected(options.backend, bytes.len(), config), offset + span.start, &mut trace.ids, &mut trace.events);
                    }
                    offset += ordinary.len();
                }
            }
        }
        Ok(trace)
    }
    pub fn decode_bytes(&self, ids: &[TokenId]) -> Result<Vec<u8>> { self.model.decode_bytes(ids) }
    pub fn decode_utf8(&self, ids: &[TokenId]) -> Result<String> { self.model.decode_utf8(ids) }
    pub fn decode_lossy(&self, ids: &[TokenId]) -> Result<String> {
        Ok(String::from_utf8_lossy(&self.decode_bytes(ids)?).into_owned())
    }
}
