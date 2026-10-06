//! Merge events use absolute UTF-8 byte offsets in the original input.
use crate::{BpeModel, TokenId};
use serde::Serialize;

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct MergeEvent {
    pub start: usize,
    pub end: usize,
    pub left: TokenId,
    pub right: TokenId,
    pub out: TokenId,
    pub rank: u32,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct EncodingTrace {
    pub ids: Vec<TokenId>,
    pub events: Vec<MergeEvent>,
}

pub(crate) fn reference_piece(bytes: &[u8], model: &BpeModel, offset: usize, output: &mut Vec<TokenId>, events: &mut Vec<MergeEvent>) {
    let mut tokens: Vec<_> = bytes.iter().enumerate()
        .map(|(i, byte)| (model.base_id(*byte), offset + i, offset + i + 1)).collect();
    loop {
        let best = tokens.windows(2).filter_map(|p| model.rule((p[0].0, p[1].0))).min_by_key(|r| r.rank);
        let Some(rule) = best else { break };
        let mut merged = Vec::with_capacity(tokens.len());
        let mut i = 0;
        while i < tokens.len() {
            if i + 1 < tokens.len() && (tokens[i].0, tokens[i + 1].0) == (rule.left, rule.right) {
                events.push(MergeEvent { start: tokens[i].1, end: tokens[i + 1].2, left: rule.left, right: rule.right, out: rule.out, rank: rule.rank });
                merged.push((rule.out, tokens[i].1, tokens[i + 1].2));
                i += 2;
            } else {
                merged.push(tokens[i]);
                i += 1;
            }
        }
        tokens = merged;
    }
    output.extend(tokens.into_iter().map(|token| token.0));
}
