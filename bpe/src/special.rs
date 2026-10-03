use crate::{Backend, Result, TokenId, TokenizerError};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone, Default)]
pub enum SpecialMode {
    #[default]
    Ordinary,
    /// Listed strings are atomic specials; others remain ordinary text.
    Allow(BTreeSet<String>),
    Reject,
}
#[derive(Debug, Clone, Default)]
pub struct EncodeOptions {
    pub backend: Backend,
    pub special: SpecialMode,
}
pub(crate) enum Piece<'a> {
    Text(&'a str),
    Special(TokenId),
}

pub(crate) fn split<'a>(
    text: &'a str,
    specials: &BTreeMap<String, TokenId>,
    mode: &SpecialMode,
) -> Result<Vec<Piece<'a>>> {
    if matches!(mode, SpecialMode::Ordinary) {
        return Ok(vec![Piece::Text(text)]);
    }
    if let SpecialMode::Allow(allowed) = mode {
        for name in allowed {
            if !specials.contains_key(name) {
                return Err(TokenizerError::UnknownSpecial(name.clone()));
            }
        }
    }
    let mut names: Vec<_> = specials
        .iter()
        .filter(|(name, _)| match mode {
            SpecialMode::Allow(allowed) => allowed.contains(*name),
            _ => true,
        })
        .collect();
    names.sort_by(|(a, _), (b, _)| b.len().cmp(&a.len()).then_with(|| a.cmp(b)));
    let mut pieces = Vec::new();
    let (mut start, mut position) = (0, 0);
    while position < text.len() {
        if let Some((name, id)) = names
            .iter()
            .find(|(name, _)| text[position..].starts_with(name.as_str()))
        {
            if matches!(mode, SpecialMode::Reject) {
                return Err(TokenizerError::DisallowedSpecial((*name).clone()));
            }
            if start < position {
                pieces.push(Piece::Text(&text[start..position]));
            }
            pieces.push(Piece::Special(**id));
            position += name.len();
            start = position;
        } else {
            position += text[position..]
                .chars()
                .next()
                .expect("valid character boundary")
                .len_utf8();
        }
    }
    if start < text.len() {
        pieces.push(Piece::Text(&text[start..]));
    }
    Ok(pieces)
}
