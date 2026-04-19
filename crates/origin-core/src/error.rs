// SPDX-License-Identifier: AGPL-3.0-only
use serde::Serialize;

#[derive(Debug, thiserror::Error)]
pub enum OriginError {
    #[error("Vector DB error: {0}")]
    VectorDb(String),

    #[error("Embedding error: {0}")]
    Embedding(String),

    #[error("Indexer error: {0}")]
    Indexer(String),

    #[error("Source error: {source_name}: {message}")]
    Source {
        source_name: String,
        message: String,
    },

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("HTTP error: {0}")]
    #[allow(dead_code)]
    Http(String),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("LLM error: {0}")]
    Llm(String),

    #[error("Vision error: {0}")]
    Vision(String),

    #[error("Trigger error: {0}")]
    #[allow(dead_code)]
    Trigger(String),

    #[error("Router error: {0}")]
    #[allow(dead_code)]
    Router(String),

    #[error("Agent disabled: {0}")]
    AgentDisabled(String),

    #[error("{0}")]
    Generic(String),
}

// Tauri requires Serialize to pass errors across IPC
impl Serialize for OriginError {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_str(&self.to_string())
    }
}
