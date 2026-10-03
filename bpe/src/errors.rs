use thiserror::Error;

#[derive(Debug, Error)]
pub enum TokenizerError {
    #[error("invalid model: {0}")]
    InvalidModel(String),
    #[error("unsupported format version {0}; expected 1 (explicit migration required)")]
    UnsupportedVersion(u32),
    #[error("unknown token ID {0}")]
    UnknownToken(u32),
    #[error("unrecognized special token in allow list: {0:?}")]
    UnknownSpecial(String),
    #[error("recognized special token is forbidden: {0:?}")]
    DisallowedSpecial(String),
    #[error("invalid training configuration: {0}")]
    InvalidConfig(String),
    #[error("unsupported backend {0}; Day 1 implements reference only")]
    UnsupportedBackend(String),
    #[error("preprocessing failed: {0}")]
    Preprocessing(String),
    #[error("invalid UTF-8 after concatenating token bytes: {0}")]
    Utf8(#[from] std::string::FromUtf8Error),
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
}

pub type Result<T> = std::result::Result<T, TokenizerError>;
