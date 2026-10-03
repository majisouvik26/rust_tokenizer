use crate::{
    encode::reference::merge_pair, pretokenize, BpeModel, Merge, Pretokenizer, Result, Token,
    TokenId, TokenizerError,
};
use base64::{engine::general_purpose::STANDARD, Engine};
use std::collections::{BTreeMap, HashMap};

#[derive(Debug, Clone)]
pub struct TrainConfig {
    pub vocab_size: usize,
    pub min_frequency: u64,
    pub pretokenizer: Pretokenizer,
}
impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            vocab_size: 8192,
            min_frequency: 2,
            pretokenizer: Pretokenizer::gpt2(),
        }
    }
}
pub struct BpeTrainer {
    config: TrainConfig,
}
impl BpeTrainer {
    pub fn new(config: TrainConfig) -> Self {
        Self { config }
    }
    pub fn train<S: AsRef<str>>(&self, records: &[S]) -> Result<BpeModel> {
        if !(256..=u32::MAX as usize).contains(&self.config.vocab_size)
            || self.config.min_frequency == 0
        {
            return Err(TokenizerError::InvalidConfig(
                "vocab_size must be >=256 and fit u32; min_frequency must be positive".into(),
            ));
        }
        let regex = pretokenize::compile(&self.config.pretokenizer)?;
        // Weighted deduplication; never concatenate record boundaries.
        let mut frequencies: BTreeMap<Vec<TokenId>, u64> = BTreeMap::new();
        for record in records {
            let text = record.as_ref();
            for span in pretokenize::spans(text, regex.as_ref())? {
                let ids = text.as_bytes()[span]
                    .iter()
                    .map(|byte| *byte as TokenId)
                    .collect();
                *frequencies.entry(ids).or_default() += 1;
            }
        }
        let mut chunks: Vec<_> = frequencies.into_iter().collect();
        let mut data = BpeModel::byte_only(self.config.pretokenizer.clone())?
            .data()
            .clone();
        let mut bytes: Vec<Vec<u8>> = (0..=255).map(|byte| vec![byte]).collect();
        let mut by_bytes: HashMap<Vec<u8>, TokenId> = bytes
            .iter()
            .cloned()
            .enumerate()
            .map(|(id, bytes)| (bytes, id as TokenId))
            .collect();
        while data.tokens.len() < self.config.vocab_size {
            let mut counts: BTreeMap<(TokenId, TokenId), u64> = BTreeMap::new();
            for (ids, frequency) in &chunks {
                for pair in ids.windows(2) {
                    *counts.entry((pair[0], pair[1])).or_default() += frequency;
                }
            }
            let best = counts
                .into_iter()
                .min_by(|(a_pair, a_count), (b_pair, b_count)| {
                    b_count.cmp(a_count).then_with(|| a_pair.cmp(b_pair))
                });
            let Some((pair, count)) = best else {
                break;
            };
            if count < self.config.min_frequency {
                break;
            }
            let mut joined = bytes[pair.0 as usize].clone();
            joined.extend_from_slice(&bytes[pair.1 as usize]);
            let out = if let Some(id) = by_bytes.get(&joined) {
                *id
            } else {
                let id = data.tokens.len() as TokenId;
                data.tokens.push(Token {
                    id,
                    bytes_b64: STANDARD.encode(&joined),
                });
                bytes.push(joined.clone());
                by_bytes.insert(joined, id);
                id
            };
            data.merges.push(Merge {
                left: pair.0,
                right: pair.1,
                out,
                rank: data.merges.len() as u32,
            });
            for (ids, _) in &mut chunks {
                *ids = merge_pair(ids, pair, out);
            }
        }
        BpeModel::from_data(data)
    }
}
