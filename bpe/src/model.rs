use std::collections::HashMap;
use crate::errors::TokenizerError;

pub struct BpeModel {
    pub merges: Vec<(String, String)>,
    pub vocab: HashMap<String, u32>,
    pub id_to_token: HashMap<u32, String>,
}

impl BpeModel {
    pub fn new() -> Self {
        BpeModel {
            merges: Vec::new(),
            vocab: HashMap::new(),
            id_to_token: HashMap::new(),
        }
    }

    pub fn add_merge(&mut self, pair: (String, String)) {
        let merged_token = format!("{}{}", pair.0, pair.1);
        let new_id = self.vocab.len() as u32;
        self.vocab.insert(merged_token.clone(), new_id);
        self.id_to_token.insert(new_id, merged_token);
        self.merges.push(pair);
    }

    pub fn encode(&self, text: &str) -> Result<Vec<u32>, TokenizerError> {
        let mut tokens: Vec<String> = text.chars().map(|c| c.to_string()).collect();
        tokens.push("</w>".to_string());

        let merge_map: HashMap<_, _> = self.merges.iter()
            .map(|(a, b)| ((a.clone(), b.clone()), format!("{}{}", a, b)))
            .collect();

        let mut changed = true;
        while changed {
            changed = false;
            let mut i = 0;
            while i < tokens.len() - 1 {
                let mut merged = None;
                for (pair, merged_token) in &merge_map {
                    if tokens[i] == pair.0 && tokens[i + 1] == pair.1 {
                        merged = Some(merged_token.clone());
                        break;
                    }
                }
                if let Some(mtoken) = merged {
                    tokens.splice(i..i + 2, [mtoken]);
                    changed = true;
                    i = 0;
                } else {
                    i += 1;
                }
            }
        }

        tokens.iter()
            .map(|t| self.vocab.get(t)
                .cloned()
                .ok_or_else(|| TokenizerError::VocabError(format!("Token '{}' not in vocabulary", t)))
            )
            .collect()
    }

    pub fn decode(&self, ids: &[u32]) -> Result<String, TokenizerError> {
        let tokens: Result<Vec<String>, TokenizerError> = ids.iter()
            .map(|id| self.id_to_token.get(id)
                .cloned()
                .ok_or_else(|| TokenizerError::VocabError(format!("ID '{}' not found", id)))
            )
            .collect();

        let s = tokens?.join("").replace("</w>", " ");
        Ok(s.trim().to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encoding_decoding() -> Result<(), TokenizerError> {
        let mut model = BpeModel::new();
        model.vocab.insert("a".into(), 0);
        model.vocab.insert("b".into(), 1);
        model.vocab.insert("</w>".into(), 2);
        model.vocab.insert("aa".into(), 3);
        model.vocab.insert("bb".into(), 4);
        model.id_to_token.insert(0, "a".into());
        model.id_to_token.insert(1, "b".into());
        model.id_to_token.insert(2, "</w>".into());
        model.id_to_token.insert(3, "aa".into());
        model.id_to_token.insert(4, "bb".into());
        model.merges = vec![("a".into(), "a".into()), ("b".into(), "b".into())];

        let encoded = model.encode("aaabbb")?;
        assert_eq!(encoded, vec![3, 0, 4, 1, 2]);

        let decoded = model.decode(&encoded)?;
        assert_eq!(decoded, "aaabbb");
        Ok(())
    }
}
