use super::{add, counts, Chunk, Pair, Result, TrainConfig, TrainingReport, Vocabulary};
use crate::encode::reference::merge_pair;
use std::{
    cmp::Reverse,
    collections::{BTreeMap, BTreeSet, BinaryHeap},
};

#[derive(Default)]
struct PairState {
    count: u64,
    version: u64,
    chunks: BTreeSet<usize>,
}
type Candidate = (u64, Reverse<Pair>, u64);

/// Recount only affected chunks. Heap entries carry globally increasing versions,
/// preventing a removed/reintroduced pair from reviving an old candidate.
pub(super) fn train(
    chunks: &mut [Chunk],
    config: &TrainConfig,
    vocabulary: &mut Vocabulary,
    report: &mut TrainingReport,
) -> Result<()> {
    let mut pairs: BTreeMap<Pair, PairState> = BTreeMap::new();
    for (index, (ids, frequency)) in chunks.iter().enumerate() {
        for (pair, count) in counts(ids, *frequency)? {
            let entry = pairs.entry(pair).or_default();
            add(&mut entry.count, count)?;
            entry.chunks.insert(index);
        }
    }
    let mut heap: BinaryHeap<Candidate> = pairs
        .iter()
        .map(|(&pair, state)| (state.count, Reverse(pair), state.version))
        .collect();
    report.peak_heap_entries = heap.len();
    let mut version = 0;
    while vocabulary.data.tokens.len() < config.vocab_size {
        let Some((count, Reverse(pair), candidate_version)) = heap.pop() else {
            break;
        };
        let Some(state) = pairs.get(&pair) else {
            continue;
        };
        if state.version != candidate_version || state.count != count {
            continue;
        }
        if count < config.min_frequency {
            break;
        }
        let affected: Vec<_> = state.chunks.iter().copied().collect();
        let out = vocabulary.output(pair)?;
        let mut touched = BTreeSet::new();
        for index in affected {
            let (ids, frequency) = &mut chunks[index];
            for (old_pair, old_count) in counts(ids, *frequency)? {
                let old = pairs.get_mut(&old_pair).expect("indexed old pair");
                old.count -= old_count;
                old.chunks.remove(&index);
                touched.insert(old_pair);
            }
            *ids = merge_pair(ids, pair, out);
            for (new_pair, new_count) in counts(ids, *frequency)? {
                let new = pairs.entry(new_pair).or_default();
                add(&mut new.count, new_count)?;
                new.chunks.insert(index);
                touched.insert(new_pair);
            }
            report.chunk_updates += 1;
        }
        for pair in touched {
            if pairs[&pair].count == 0 {
                pairs.remove(&pair);
            } else {
                add(&mut version, 1)?;
                let state = pairs.get_mut(&pair).expect("touched pair");
                state.version = version;
                heap.push((state.count, Reverse(pair), state.version));
            }
        }
        report.peak_heap_entries = report.peak_heap_entries.max(heap.len());
        // Bound stale candidates relative to the current positive-count index.
        if heap.len() > pairs.len().saturating_mul(4) {
            heap = pairs
                .iter()
                .map(|(&pair, state)| (state.count, Reverse(pair), state.version))
                .collect();
            report.heap_rebuilds += 1;
        }
        report.selected_pairs += 1;
    }
    Ok(())
}
