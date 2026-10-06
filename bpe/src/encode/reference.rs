use crate::{BpeModel, TokenId};

/// Merge non-overlapping occurrences left to right. New outputs are never
/// reconsidered during this pass. Encoder and trainer share this primitive.
pub fn merge_pair(ids: &[TokenId], pair: (TokenId, TokenId), out: TokenId) -> Vec<TokenId> {
    let mut merged = Vec::with_capacity(ids.len());
    merge_pair_into(ids, pair, out, &mut merged);
    merged
}

pub(crate) fn merge_pair_into(
    ids: &[TokenId],
    pair: (TokenId, TokenId),
    out: TokenId,
    merged: &mut Vec<TokenId>,
) {
    let mut i = 0;
    while i < ids.len() {
        if i + 1 < ids.len() && (ids[i], ids[i + 1]) == pair {
            merged.push(out);
            i += 2;
        } else {
            merged.push(ids[i]);
            i += 1;
        }
    }
}

/// Readable scan oracle: select lowest applicable rank, merge its occurrences,
/// repeat. Expected O(n^2) merge work in the worst case.
pub fn encode(mut ids: Vec<TokenId>, model: &BpeModel) -> Vec<TokenId> {
    loop {
        let best = ids
            .windows(2)
            .filter_map(|p| model.rule((p[0], p[1])))
            .min_by_key(|r| r.rank);
        let Some(best) = best else {
            return ids;
        };
        ids = merge_pair(&ids, (best.left, best.right), best.out);
    }
}
