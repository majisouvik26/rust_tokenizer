use std::collections::HashMap;
use crate::{model::BpeModel, errors::TokenizerError};

pub struct BpeTrainer {
    pub num_merges: usize,
}

impl BpeTrainer {
    pub fn new(num_merges: usize) -> Self {
        BpeTrainer { num_merges }
    }

    pub fn train(&self, corpus: &str) -> Result<BpeModel, TokenizerError> {
        let mut words = self.preprocess(corpus);
        let mut model = BpeModel::new();
        self.initialize_vocab(&mut model, &words)?;

        for _ in 0..self.num_merges {
            let pairs = self.count_pairs(&words);
            if pairs.is_empty() {
                break;
            }

            let max_pair = pairs.into_iter()
                .max_by_key(|&(_, count)| count)
                .map(|(pair, _)| pair)
                .ok_or(TokenizerError::VocabError("No pairs found".into()))?;

            self.merge_pair(&max_pair, &mut words);
            model.add_merge(max_pair);
        }

        Ok(model)
    }

    fn preprocess(&self, corpus: &str) -> Vec<Vec<String>> {
        corpus.split_whitespace()
            .map(|word| {
                let mut chars: Vec<String> = word.chars().map(|c| c.to_string()).collect();
                chars.push("</w>".to_string());
                chars
            })
            .collect()
    }

    fn initialize_vocab(&self, model: &mut BpeModel, words: &[Vec<String>]) -> Result<(), TokenizerError> {
        let mut vocab_set = std::collections::HashSet::new();
        for word in words {
            for token in word {
                vocab_set.insert(token.clone());
            }
        }
        for token in vocab_set {
            let id = model.vocab.len() as u32;
            model.vocab.insert(token.clone(), id);
            model.id_to_token.insert(id, token);
        }
        Ok(())
    }

    fn count_pairs(&self, words: &[Vec<String>]) -> HashMap<(String, String), usize> {
        let mut counts = HashMap::new();
        for word in words {
            for i in 0..word.len() - 1 {
                let pair = (word[i].clone(), word[i + 1].clone());
                *counts.entry(pair).or_insert(0) += 1;
            }
        }
        counts
    }

    fn merge_pair(&self, pair: &(String, String), words: &mut [Vec<String>]) {
        let merged_token = format!("{}{}", pair.0, pair.1);
        for word in words.iter_mut() {
            let mut i = 0;
            while i < word.len() - 1 {
                if word[i] == pair.0 && word[i + 1] == pair.1 {
                    word.splice(i..i + 2, [merged_token.clone()]);
                    i = 0; // Restart to check for new pairs
                } else {
                    i += 1;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trainer() -> Result<(), TokenizerError> {
        let corpus = "aaabbb";
        let trainer = BpeTrainer::new(2);
        let model = trainer.train(corpus)?;

        assert_eq!(model.merges, vec![
            ("a".to_string(), "a".to_string()),
            ("b".to_string(), "b".to_string())
        ]);

        assert!(model.vocab.contains_key("aa"));
        assert!(model.vocab.contains_key("bb"));

        let encoded = model.encode(corpus)?;
        let decoded = model.decode(&encoded)?;
        assert_eq!(decoded, "aaabbb");
        Ok(())
    }
}