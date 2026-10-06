use super::{add, Chunk, Result, TrainConfig, TrainingReport, Vocabulary};
use crate::encode::reference::merge_pair;
use std::collections::BTreeMap;

/// Full recount oracle. Every selected pair visits every unique weighted chunk.
pub(super) fn train(chunks: &mut [Chunk], config: &TrainConfig, vocabulary: &mut Vocabulary, report: &mut TrainingReport) -> Result<()> {
    while vocabulary.data.tokens.len() < config.vocab_size {
        let mut global = BTreeMap::new();
        for (ids, frequency) in chunks.iter() {
            for pair in ids.windows(2) { add(global.entry((pair[0], pair[1])).or_default(), *frequency)?; }
        }
        let best = global.into_iter().min_by(|(a_pair, a_count), (b_pair, b_count)| {
            b_count.cmp(a_count).then_with(|| a_pair.cmp(b_pair))
        });
        let Some((pair, count)) = best else { break };
        if count < config.min_frequency { break; }
        let out = vocabulary.output(pair)?;
        for (ids, _) in chunks.iter_mut() { *ids = merge_pair(ids, pair, out); }
        report.selected_pairs += 1;
        report.chunk_updates += chunks.len();
    }
    Ok(())
}
