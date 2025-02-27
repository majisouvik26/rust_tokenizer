use std::fmt;

#[derive(Debug)]
pub enum TokenizerError {
    VocabError(String),
}

impl fmt::Display for TokenizerError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            TokenizerError::VocabError(msg) => write!(f, "Vocabulary error: {}", msg),
        }
    }
}

impl std::error::Error for TokenizerError {}