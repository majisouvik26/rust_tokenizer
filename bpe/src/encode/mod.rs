pub mod reference;
mod cache;
mod heap;

pub use cache::CacheStats;
pub(crate) use cache::Cache;
use crate::{Backend, BpeModel, RuntimeConfig, TokenId};

#[derive(Default)]
pub(crate) struct Scratch {
    ids: Vec<TokenId>,
    merged: Vec<TokenId>,
    heap: heap::HeapScratch,
}

pub(crate) fn selected(backend: Backend, bytes: usize, config: &RuntimeConfig) -> Backend {
    match backend {
        Backend::Auto if config.heap_threshold.is_some_and(|threshold| bytes >= threshold) => Backend::Heap,
        Backend::Auto => Backend::Reference,
        other => other,
    }
}

impl Scratch {
    pub(crate) fn encode(&mut self, bytes: &[u8], model: &BpeModel, backend: Backend, output: &mut Vec<TokenId>) {
        match backend {
            Backend::Heap => self.heap.encode(bytes, model, output, 0, None),
            _ => {
                self.ids.clear();
                self.ids.extend(bytes.iter().map(|byte| model.base_id(*byte)));
                loop {
                    let best = self.ids.windows(2)
                        .filter_map(|pair| model.rule((pair[0], pair[1])))
                        .min_by_key(|rule| rule.rank);
                    let Some(best) = best else { break };
                    self.merged.clear();
                    reference::merge_pair_into(&self.ids, (best.left, best.right), best.out, &mut self.merged);
                    std::mem::swap(&mut self.ids, &mut self.merged);
                }
                output.extend_from_slice(&self.ids);
            }
        }
    }
    pub(crate) fn trim(&mut self, limit: usize) {
        if self.ids.capacity() > limit { self.ids = Vec::new(); }
        if self.merged.capacity() > limit { self.merged = Vec::new(); }
        self.heap.trim(limit);
    }
}

pub(crate) fn trace_piece(bytes: &[u8], model: &BpeModel, backend: Backend, offset: usize, output: &mut Vec<TokenId>, events: &mut Vec<crate::trace::MergeEvent>) {
    if backend == Backend::Heap {
        heap::HeapScratch::default().encode(bytes, model, output, offset, Some(events));
    } else {
        crate::trace::reference_piece(bytes, model, offset, output, events);
    }
}
