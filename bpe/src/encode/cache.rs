use crate::TokenId;
use serde::Serialize;
use std::collections::{HashMap, VecDeque};

#[derive(Debug, Clone, Copy, Default, Serialize)]
pub struct CacheStats {
    pub hits: u64,
    pub misses: u64,
    pub entries: usize,
    /// Includes both key copies and token ID payloads; excludes map overhead.
    pub payload_bytes: usize,
}

/// FIFO eviction, bounded by both entries and payload bytes, within one model.
#[derive(Default)]
pub(crate) struct Cache {
    entries: HashMap<Vec<u8>, Vec<TokenId>>,
    order: VecDeque<Vec<u8>>,
    stats: CacheStats,
}
impl Cache {
    pub(crate) fn get(&mut self, key: &[u8]) -> Option<&[TokenId]> {
        match self.entries.get(key) {
            Some(ids) => {
                self.stats.hits += 1;
                Some(ids)
            }
            None => {
                self.stats.misses += 1;
                None
            }
        }
    }
    pub(crate) fn insert(&mut self, key: &[u8], ids: &[TokenId], capacity: usize, budget: usize) {
        let cost = key
            .len()
            .saturating_mul(2)
            .saturating_add(ids.len().saturating_mul(4));
        if capacity == 0 || cost > budget || self.entries.contains_key(key) {
            return;
        }
        while self.entries.len() >= capacity
            || self.stats.payload_bytes.saturating_add(cost) > budget
        {
            let oldest = self.order.pop_front().expect("nonempty bounded cache");
            let removed = self.entries.remove(&oldest).expect("cached FIFO key");
            self.stats.payload_bytes -= oldest.len() * 2 + removed.len() * 4;
        }
        self.entries.insert(key.to_vec(), ids.to_vec());
        self.order.push_back(key.to_vec());
        self.stats.payload_bytes += cost;
        self.stats.entries = self.entries.len();
    }
    pub(crate) fn stats(&self) -> CacheStats {
        self.stats
    }
    pub(crate) fn clear(&mut self) {
        *self = Self::default();
    }
}
