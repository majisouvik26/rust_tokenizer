//! Deterministic trainers share input preparation and canonical vocabulary rules.
mod incremental;
mod reference;
use crate::{
    pretokenize, BpeModel, Merge, ModelData, Pretokenizer, Result, Token, TokenId, TokenizerError,
};
use base64::{engine::general_purpose::STANDARD, Engine};
use serde::Serialize;
use std::collections::{BTreeMap, HashMap};

type Pair = (TokenId, TokenId);
type Chunk = (Vec<TokenId>, u64);

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum TrainerBackend {
    #[default]
    Reference,
    Incremental,
}
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
#[derive(Debug, Default, Serialize)]
pub struct TrainingReport {
    pub backend: TrainerBackend,
    pub unique_chunks: usize,
    pub total_chunks: u64,
    pub selected_pairs: usize,
    pub chunk_updates: usize,
    pub heap_rebuilds: usize,
    pub peak_heap_entries: usize,
}

pub struct BpeTrainer {
    config: TrainConfig,
}
impl BpeTrainer {
    pub fn new(config: TrainConfig) -> Self {
        Self { config }
    }
    pub fn train<S: AsRef<str>>(&self, records: &[S]) -> Result<BpeModel> {
        self.train_with(records, TrainerBackend::Reference)
    }
    pub fn train_with<S: AsRef<str>>(
        &self,
        records: &[S],
        backend: TrainerBackend,
    ) -> Result<BpeModel> {
        Ok(self.train_with_report(records, backend)?.0)
    }
    pub fn train_with_report<S: AsRef<str>>(
        &self,
        records: &[S],
        backend: TrainerBackend,
    ) -> Result<(BpeModel, TrainingReport)> {
        if !(256..=u32::MAX as usize).contains(&self.config.vocab_size)
            || self.config.min_frequency == 0
        {
            return Err(TokenizerError::InvalidConfig(
                "vocab_size must be >=256 and fit u32; min_frequency must be positive".into(),
            ));
        }
        let regex = pretokenize::compile(&self.config.pretokenizer)?;
        let mut frequencies: BTreeMap<Vec<TokenId>, u64> = BTreeMap::new();
        let mut report = TrainingReport {
            backend,
            ..Default::default()
        };
        for record in records {
            let text = record.as_ref();
            for span in pretokenize::spans(text, regex.as_ref())? {
                let ids = text.as_bytes()[span]
                    .iter()
                    .map(|byte| *byte as TokenId)
                    .collect();
                add(frequencies.entry(ids).or_default(), 1)?;
                add(&mut report.total_chunks, 1)?;
            }
        }
        let mut chunks: Vec<_> = frequencies.into_iter().collect();
        report.unique_chunks = chunks.len();
        let mut builder = Vocabulary::new(self.config.pretokenizer.clone())?;
        match backend {
            TrainerBackend::Reference => {
                reference::train(&mut chunks, &self.config, &mut builder, &mut report)?
            }
            TrainerBackend::Incremental => {
                incremental::train(&mut chunks, &self.config, &mut builder, &mut report)?
            }
        }
        Ok((BpeModel::from_data(builder.data)?, report))
    }
}
fn add(value: &mut u64, increment: u64) -> Result<()> {
    *value = value
        .checked_add(increment)
        .ok_or_else(|| TokenizerError::InvalidConfig("training count exceeds u64".into()))?;
    Ok(())
}
fn counts(ids: &[TokenId], frequency: u64) -> Result<BTreeMap<Pair, u64>> {
    let mut counts = BTreeMap::new();
    for pair in ids.windows(2) {
        add(counts.entry((pair[0], pair[1])).or_default(), frequency)?;
    }
    Ok(counts)
}
struct Vocabulary {
    data: ModelData,
    bytes: Vec<Vec<u8>>,
    by_bytes: HashMap<Vec<u8>, TokenId>,
    by_pair: HashMap<Pair, TokenId>,
}
impl Vocabulary {
    fn new(pretokenizer: Pretokenizer) -> Result<Self> {
        let bytes: Vec<Vec<u8>> = (0..=255).map(|byte| vec![byte]).collect();
        let by_bytes = bytes
            .iter()
            .cloned()
            .enumerate()
            .map(|(id, bytes)| (bytes, id as TokenId))
            .collect();
        Ok(Self {
            data: BpeModel::byte_only(pretokenizer)?.data().clone(),
            bytes,
            by_bytes,
            by_pair: HashMap::new(),
        })
    }
    fn output(&mut self, pair: Pair) -> Result<TokenId> {
        if let Some(out) = self.by_pair.get(&pair) {
            return Ok(*out);
        }
        let mut joined = self.bytes[pair.0 as usize].clone();
        joined.extend_from_slice(&self.bytes[pair.1 as usize]);
        let out = if let Some(id) = self.by_bytes.get(&joined) {
            *id
        } else {
            let id = self.data.tokens.len() as TokenId;
            self.data.tokens.push(Token {
                id,
                bytes_b64: STANDARD.encode(&joined),
            });
            self.bytes.push(joined.clone());
            self.by_bytes.insert(joined, id);
            id
        };
        let rank = u32::try_from(self.data.merges.len())
            .map_err(|_| TokenizerError::InvalidConfig("too many merge rules".into()))?;
        self.data.merges.push(Merge {
            left: pair.0,
            right: pair.1,
            out,
            rank,
        });
        self.by_pair.insert(pair, out);
        Ok(out)
    }
}
