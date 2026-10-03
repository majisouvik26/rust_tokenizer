use crate::{pretokenize, Pretokenizer, Result, TokenizerError};
use base64::{engine::general_purpose::STANDARD, Engine};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, HashMap, HashSet},
    fs,
    path::Path,
};

pub type TokenId = u32;
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Profile {
    CustomByteBpe,
    Gpt2,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Token {
    pub id: TokenId,
    pub bytes_b64: String,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Merge {
    pub left: TokenId,
    pub right: TokenId,
    pub out: TokenId,
    pub rank: u32,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelData {
    pub format_version: u32,
    pub profile: Profile,
    pub pretokenizer: Pretokenizer,
    pub tokens: Vec<Token>,
    pub merges: Vec<Merge>,
    pub special_tokens: BTreeMap<String, TokenId>,
}

/// Construct only through validation. Ordinary IDs are contiguous. Imported
/// GPT-2 IDs are preserved. Special IDs live outside the ordinary vocabulary.
#[derive(Debug, Clone)]
pub struct BpeModel {
    data: ModelData,
    bytes: Vec<Vec<u8>>,
    base: [TokenId; 256],
    rules: HashMap<(TokenId, TokenId), Merge>,
    special_bytes: HashMap<TokenId, Vec<u8>>,
}

impl BpeModel {
    pub fn from_data(data: ModelData) -> Result<Self> {
        let invalid = |s: &str| TokenizerError::InvalidModel(s.into());
        if data.format_version != 1 {
            return Err(TokenizerError::UnsupportedVersion(data.format_version));
        }
        pretokenize::compile(&data.pretokenizer)?;
        if data.profile == Profile::Gpt2 && data.pretokenizer != Pretokenizer::gpt2() {
            return Err(invalid("GPT-2 requires canonical preprocessing"));
        }
        if data.tokens.len() < 256 || data.tokens.len() > u32::MAX as usize {
            return Err(invalid("vocabulary must cover 256 bytes and fit u32 IDs"));
        }
        let mut bytes = Vec::with_capacity(data.tokens.len());
        let mut unique = HashSet::new();
        let mut base = [TokenId::MAX; 256];
        let mut available = HashSet::new();
        for (index, token) in data.tokens.iter().enumerate() {
            if token.id as usize != index {
                return Err(invalid(
                    "tokens must have unique contiguous IDs in ID order",
                ));
            }
            let raw = STANDARD
                .decode(&token.bytes_b64)
                .map_err(|_| invalid("invalid token base64"))?;
            if STANDARD.encode(&raw) != token.bytes_b64 {
                return Err(invalid("base64 must be canonical"));
            }
            if raw.is_empty() || !unique.insert(raw.clone()) {
                return Err(invalid("ordinary token bytes must be nonempty and unique"));
            }
            if raw.len() == 1 {
                base[raw[0] as usize] = token.id;
                available.insert(token.id);
                if data.profile == Profile::CustomByteBpe && token.id != raw[0] as u32 {
                    return Err(invalid("custom base IDs must equal byte values"));
                }
            }
            bytes.push(raw);
        }
        if base.contains(&TokenId::MAX) {
            return Err(invalid("missing base byte"));
        }
        let mut rules = HashMap::new();
        for (index, rule) in data.merges.iter().enumerate() {
            if rule.rank as usize != index {
                return Err(invalid(
                    "merge ranks must be unique, contiguous, and in rank order",
                ));
            }
            if !available.contains(&rule.left) || !available.contains(&rule.right) {
                return Err(invalid(
                    "merge references a token not available before its rank",
                ));
            }
            let out = bytes
                .get(rule.out as usize)
                .ok_or_else(|| invalid("unknown merge output ID"))?;
            let mut joined = bytes[rule.left as usize].clone();
            joined.extend_from_slice(&bytes[rule.right as usize]);
            if out != &joined {
                return Err(invalid("merge output bytes differ from left + right"));
            }
            if rules
                .insert((rule.left, rule.right), rule.clone())
                .is_some()
            {
                return Err(invalid("duplicate merge pair"));
            }
            available.insert(rule.out);
        }
        if available.len() != bytes.len() {
            return Err(invalid("non-base token has no producing merge"));
        }
        let mut special_bytes = HashMap::new();
        for (name, id) in &data.special_tokens {
            if name.is_empty()
                || (*id as usize) < bytes.len()
                || special_bytes
                    .insert(*id, name.as_bytes().to_vec())
                    .is_some()
            {
                return Err(invalid("empty special string or colliding special ID"));
            }
            if unique.contains(name.as_bytes()) {
                return Err(invalid("special bytes collide with an ordinary token"));
            }
        }
        Ok(Self {
            data,
            bytes,
            base,
            rules,
            special_bytes,
        })
    }
    pub fn byte_only(pretokenizer: Pretokenizer) -> Result<Self> {
        Self::from_data(ModelData {
            format_version: 1,
            profile: Profile::CustomByteBpe,
            pretokenizer,
            tokens: (0..=255)
                .map(|b| Token {
                    id: b,
                    bytes_b64: STANDARD.encode([b as u8]),
                })
                .collect(),
            merges: vec![],
            special_tokens: BTreeMap::new(),
        })
    }
    pub fn data(&self) -> &ModelData {
        &self.data
    }
    pub fn pretokenizer(&self) -> &Pretokenizer {
        &self.data.pretokenizer
    }
    pub fn special_tokens(&self) -> &BTreeMap<String, TokenId> {
        &self.data.special_tokens
    }
    pub fn vocab_size(&self) -> usize {
        self.bytes.len()
    }
    pub fn merges(&self) -> &[Merge] {
        &self.data.merges
    }
    pub(crate) fn base_id(&self, byte: u8) -> TokenId {
        self.base[byte as usize]
    }
    pub(crate) fn rule(&self, pair: (TokenId, TokenId)) -> Option<&Merge> {
        self.rules.get(&pair)
    }
    pub fn token_bytes(&self, id: TokenId) -> Result<&[u8]> {
        self.bytes
            .get(id as usize)
            .or_else(|| self.special_bytes.get(&id))
            .map(Vec::as_slice)
            .ok_or(TokenizerError::UnknownToken(id))
    }
    pub fn decode_bytes(&self, ids: &[TokenId]) -> Result<Vec<u8>> {
        let mut output = Vec::new();
        for id in ids {
            output.extend_from_slice(self.token_bytes(*id)?);
        }
        Ok(output)
    }
    pub fn decode_utf8(&self, ids: &[TokenId]) -> Result<String> {
        Ok(String::from_utf8(self.decode_bytes(ids)?)?)
    }
    /// Fixed field order, ID/rank order, sorted specials. No volatile timestamps.
    pub fn canonical_json(&self) -> Result<Vec<u8>> {
        Ok(serde_json::to_vec(&self.data)?)
    }
    pub fn sha256(&self) -> Result<String> {
        Ok(format!("{:x}", Sha256::digest(self.canonical_json()?)))
    }
    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        fs::write(path, self.canonical_json()?)?;
        Ok(())
    }
    pub fn from_json(json: &[u8]) -> Result<Self> {
        Self::from_data(serde_json::from_slice(json)?)
    }
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        Self::from_json(&fs::read(path)?)
    }
}
