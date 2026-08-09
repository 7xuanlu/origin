// SPDX-License-Identifier: AGPL-3.0-only
//! App-local error type. Mirrors the variants used by app code (file watcher,
//! MCP config, sensor, sources/sync). origin-core's OriginError is no longer
//! imported after the dep removal in Phase 5-D PR2.
use serde::Serialize;

#[derive(Debug, thiserror::Error)]
pub enum AppError {
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

    #[error("Vision error: {0}")]
    Vision(String),

    #[error("{0}")]
    Generic(String),
}

// Tauri requires Serialize to pass errors across IPC
impl Serialize for AppError {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_str(&self.to_string())
    }
}
