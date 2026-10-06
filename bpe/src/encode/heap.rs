//! Rank-ordered local updates over an array-linked token sequence.
use crate::{trace::MergeEvent, BpeModel, TokenId};
use std::{cmp::Reverse, collections::BinaryHeap};

#[derive(Debug)]
struct Node {
    id: TokenId,
    prev: Option<usize>,
    next: Option<usize>,
    end: usize,
    generation: u64,
    alive: bool,
}

// Original left byte position breaks equal-rank ties, independently of heap order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct Candidate {
    rank: u32,
    left: usize,
    right: usize,
    left_generation: u64,
    right_generation: u64,
    out: TokenId,
}

#[derive(Default)]
pub(crate) struct HeapScratch {
    nodes: Vec<Node>,
    heap: BinaryHeap<Reverse<Candidate>>,
    round: Vec<Candidate>,
    changed: Vec<usize>,
}

impl HeapScratch {
    fn push(&mut self, left: usize, model: &BpeModel) {
        let a = &self.nodes[left];
        if !a.alive {
            return;
        }
        let Some(right) = a.next else { return };
        let b = &self.nodes[right];
        if let Some(rule) = model.rule((a.id, b.id)) {
            self.heap.push(Reverse(Candidate {
                rank: rule.rank,
                left,
                right,
                left_generation: a.generation,
                right_generation: b.generation,
                out: rule.out,
            }));
        }
    }

    fn valid(&self, candidate: Candidate) -> bool {
        let a = &self.nodes[candidate.left];
        let b = &self.nodes[candidate.right];
        a.alive
            && b.alive
            && a.next == Some(candidate.right)
            && b.prev == Some(candidate.left)
            && a.generation == candidate.left_generation
            && b.generation == candidate.right_generation
    }

    pub(crate) fn encode(
        &mut self,
        bytes: &[u8],
        model: &BpeModel,
        output: &mut Vec<TokenId>,
        offset: usize,
        mut events: Option<&mut Vec<MergeEvent>>,
    ) {
        self.nodes.clear();
        self.heap.clear();
        self.round.clear();
        self.changed.clear();
        self.nodes
            .extend(bytes.iter().enumerate().map(|(i, byte)| Node {
                id: model.base_id(*byte),
                prev: i.checked_sub(1),
                next: (i + 1 < bytes.len()).then_some(i + 1),
                end: i + 1,
                generation: 0,
                alive: true,
            }));
        for i in 0..self.nodes.len() {
            self.push(i, model);
        }
        while let Some(Reverse(first)) = self.heap.pop() {
            if !self.valid(first) {
                continue;
            }
            self.round.clear();
            self.round.push(first);
            // Freeze the whole selected-rank pass before adding new candidates.
            // Models can reuse an existing output ID: a merge may enable a lower
            // rank. The scan oracle still finishes this pass before that rank.
            while self
                .heap
                .peek()
                .is_some_and(|entry| entry.0.rank == first.rank)
            {
                self.round
                    .push(self.heap.pop().expect("peeked candidate").0);
            }
            self.changed.clear();
            for index in 0..self.round.len() {
                let candidate = self.round[index];
                if !self.valid(candidate) {
                    continue;
                }
                let (left, right) = (candidate.left, candidate.right);
                let previous = self.nodes[left].prev;
                let next = self.nodes[right].next;
                if let Some(trace) = events.as_deref_mut() {
                    trace.push(MergeEvent {
                        start: offset + left,
                        end: offset + self.nodes[right].end,
                        left: self.nodes[left].id,
                        right: self.nodes[right].id,
                        out: candidate.out,
                        rank: candidate.rank,
                    });
                }
                self.nodes[left].id = candidate.out;
                self.nodes[left].end = self.nodes[right].end;
                self.nodes[left].next = next;
                self.nodes[left].generation += 1;
                self.nodes[right].alive = false;
                self.nodes[right].generation += 1;
                if let Some(next) = next {
                    self.nodes[next].prev = Some(left);
                }
                if let Some(previous) = previous {
                    self.changed.push(previous);
                }
                self.changed.push(left);
            }
            // Duplicate refreshed entries are harmless: generations invalidate
            // all copies after their first successful merge.
            for index in 0..self.changed.len() {
                self.push(self.changed[index], model);
            }
        }
        if !self.nodes.is_empty() {
            let mut current = Some(0);
            while let Some(index) = current {
                output.push(self.nodes[index].id);
                current = self.nodes[index].next;
            }
        }
    }

    pub(crate) fn trim(&mut self, limit: usize) {
        if self.nodes.capacity() > limit
            || self.heap.capacity() > limit.saturating_mul(4)
            || self.round.capacity() > limit.saturating_mul(4)
            || self.changed.capacity() > limit.saturating_mul(4)
        {
            *self = Self::default();
        }
    }
}
