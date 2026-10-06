//! Diagnostic phase timings; never added to the normal encoding path.
use crate::{encode, Backend, Result, RuntimeConfig, Tokenizer};
use serde::Serialize;
use std::time::Instant;

#[derive(Debug, Serialize)]
pub struct PhaseProfile {
    pub preprocessing_seconds: f64,
    pub merge_seconds: f64,
    pub output_seconds: f64,
    pub pretokens: usize,
    pub tokens: usize,
}
/// Ordinary-text only. Merge includes construction of per-piece ID vectors;
/// output measures concatenating those vectors into a document result.
/// Small per-piece timers add overhead: use for diagnosis, not throughput claims.
pub fn measure(tokenizer: &Tokenizer, text: &str, backend: Backend, config: &RuntimeConfig) -> Result<PhaseProfile> {
    config.validate()?;
    let start = Instant::now();
    let spans = tokenizer.pretoken_spans(text)?;
    let preprocessing_seconds = start.elapsed().as_secs_f64();
    let mut scratch = encode::Scratch::default();
    let mut pieces = Vec::with_capacity(spans.len());
    let mut merge_seconds = 0.0;
    for span in &spans {
        let bytes = &text.as_bytes()[span.clone()];
        let engine = encode::selected(backend, bytes.len(), config);
        let start = Instant::now();
        let mut ids = Vec::new();
        scratch.encode(bytes, tokenizer.model(), engine, &mut ids);
        merge_seconds += start.elapsed().as_secs_f64();
        pieces.push(ids);
    }
    let start = Instant::now();
    let ids: Vec<_> = pieces.into_iter().flatten().collect();
    std::hint::black_box(&ids);
    let output_seconds = start.elapsed().as_secs_f64();
    let expected = tokenizer.encode(text)?;
    if ids != expected {
        return Err(crate::TokenizerError::InvalidConfig("profile disagrees with reference IDs".into()));
    }
    Ok(PhaseProfile { preprocessing_seconds, merge_seconds, output_seconds, pretokens: spans.len(), tokens: ids.len() })
}
