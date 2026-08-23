// SPDX-License-Identifier: Apache-2.0
use serde::Serialize;

#[derive(Debug, thiserror::Error)]
pub enum WenlanError {
    #[error("Vector DB error: {0}")]
    VectorDb(String),

    #[error("Embedding error: {0}")]
    Embedding(String),

    #[error("Source error: {source_name}: {message}")]
    Source {
        source_name: String,
        message: String,
    },

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("LLM error: {0}")]
    Llm(String),

    #[error("Agent disabled: {0}")]
    AgentDisabled(String),

    #[error("Validation error: {0}")]
    Validation(String),

    #[error("Conflict: {0}")]
    Conflict(String),

    #[error("Page draft id already exists with a different snapshot: {0}")]
    PageDraftIdConflict(String),

    #[error("Not found: {0}")]
    NotFound(String),

    #[error("{0}")]
    Generic(String),
}

// Tauri requires Serialize to pass errors across IPC
impl Serialize for WenlanError {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_str(&self.to_string())
    }
}
