// SPDX-License-Identifier: AGPL-3.0-only
//! Shared Obsidian vault sync logic used by both Tauri commands and REST API.
//!
//! In the thin-client architecture the heavy lifting (DB sync state, embedding,
//! chunking) happens inside the origin-server daemon.  This module delegates
//! to `POST /api/sources/{id}/sync` on the daemon.

use crate::error::OriginError;
use crate::state::AppState;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::sync::Arc;
use tokio::sync::RwLock;

use super::Source;

/// Stats returned from a source sync operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyncStats {
    pub files_found: usize,
    pub ingested: usize,
    pub skipped: usize,
    pub errors: usize,
    /// Categorized error detail when `errors > 0`. Known values:
    /// "google_drive_offline", "file_read_errors".
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error_detail: Option<String>,
}

/// Compute SHA-256 hex digest of a string.
pub fn content_hash(content: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(content.as_bytes());
    format!("{:x}", hasher.finalize())
}

/// Sync an Obsidian vault by delegating to the daemon's sync endpoint.
///
/// The daemon owns the DB, embeddings, and sync-state tracking.
/// This is a thin proxy: `POST /api/sources/{id}/sync`.
pub async fn sync_obsidian_vault(
    source: &Source,
    state: &Arc<RwLock<AppState>>,
) -> Result<SyncStats, OriginError> {
    let client = {
        let s = state.read().await;
        s.client.clone()
    };

    let path = format!("/api/sources/{}/sync", source.id);
    let stats: SyncStats = client
        .post_empty(&path)
        .await
        .map_err(|e| OriginError::Http(format!("daemon sync failed: {}", e)))?;

    Ok(stats)
}
