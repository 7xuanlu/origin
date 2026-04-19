// SPDX-License-Identifier: Apache-2.0
//! REST API endpoints for source management.

use crate::error::ServerError;
use crate::state::ServerState;
use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::Json,
};
use origin_core::sources::obsidian::{has_any_markdown, note_to_documents, scan_vault};
use origin_core::sources::Source;
use origin_types::sources::{SourceType, SyncStatus};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::RwLock;

// ===== Request/Response Types =====

#[derive(Debug, Deserialize)]
pub struct AddSourceRequest {
    pub source_type: String,
    pub path: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyncStatsResponse {
    pub files_found: usize,
    pub ingested: usize,
    pub skipped: usize,
    pub errors: usize,
    /// Categorized detail when errors > 0. Known values:
    /// "google_drive_offline", "file_read_errors".
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error_detail: Option<String>,
}

/// Detect Google Drive File Provider paths on macOS. Files at these paths are
/// often "online-only" placeholders — `std::fs::read_to_string()` blocks and
/// eventually times out while the OS tries to download them on demand.
fn is_google_drive_path(path: &std::path::Path) -> bool {
    path.to_string_lossy()
        .contains("/Library/CloudStorage/GoogleDrive-")
}

// ===== Handlers =====

/// GET /api/sources
pub async fn handle_list_sources() -> Json<Vec<Source>> {
    let config = origin_core::config::load_config();
    Json(config.sources)
}

/// POST /api/sources
pub async fn handle_add_source(
    State(state): State<Arc<RwLock<ServerState>>>,
    Json(body): Json<AddSourceRequest>,
) -> Result<Json<Source>, ServerError> {
    let path = PathBuf::from(&body.path);
    if !path.exists() {
        return Err(ServerError::ValidationError(format!(
            "Path does not exist: {}",
            path.display()
        )));
    }
    if !path.is_dir() {
        return Err(ServerError::ValidationError(
            "Path is not a directory".to_string(),
        ));
    }

    let st = match body.source_type.as_str() {
        "obsidian" => {
            // Accept any folder of markdown files, Obsidian vault or not.
            // Frontend detects .obsidian/ for cosmetic badge purposes.
            // `has_any_markdown` short-circuits on the first match instead
            // of walking the entire vault, so registration is fast even on
            // very large knowledge bases.
            if !has_any_markdown(&path) {
                return Err(ServerError::ValidationError(format!(
                    "No markdown files found in: {}",
                    path.display()
                )));
            }
            SourceType::Obsidian
        }
        "directory" => SourceType::Directory,
        other => {
            return Err(ServerError::ValidationError(format!(
                "Unknown source type: {}",
                other
            )));
        }
    };

    let dirname = path
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_else(|| "dir".to_string());
    let slug = origin_core::export::obsidian::slugify(&dirname);
    let id = format!("{}-{}", st.as_str(), slug);

    let mut config = origin_core::config::load_config();
    if config.sources.iter().any(|s| s.path == path) {
        return Err(ServerError::ValidationError(format!(
            "Source already registered for path: {}",
            path.display()
        )));
    }

    let source = Source {
        id,
        source_type: st.clone(),
        path: path.clone(),
        status: SyncStatus::Active,
        last_sync: None,
        file_count: 0,
        memory_count: 0,
        last_sync_errors: 0,
        last_sync_error_detail: None,
    };

    config.sources.push(source.clone());
    origin_core::config::save_config(&config)?;

    if st == SourceType::Directory {
        let mut s = state.write().await;
        if !s.watch_paths.contains(&path) {
            s.watch_paths.push(path);
        }
    }

    Ok(Json(source))
}

/// DELETE /api/sources/{id}
pub async fn handle_remove_source(
    State(state): State<Arc<RwLock<ServerState>>>,
    Path(id): Path<String>,
) -> Result<StatusCode, ServerError> {
    let mut config = origin_core::config::load_config();
    let source = config
        .sources
        .iter()
        .find(|s| s.id == id)
        .cloned()
        .ok_or_else(|| ServerError::NotFound(format!("Source not found: {}", id)))?;

    config.sources.retain(|s| s.id != id);
    origin_core::config::save_config(&config)?;

    {
        let mut s = state.write().await;
        s.watch_paths.retain(|p| p != &source.path);
    }

    let s = state.read().await;
    if let Some(ref db) = s.db {
        let _ = db.delete_all_sync_state(&id).await;
    }

    Ok(StatusCode::NO_CONTENT)
}

// ===== Helpers =====

fn content_hash(content: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(content.as_bytes());
    format!("{:x}", hasher.finalize())
}

/// POST /api/sources/{id}/sync — Trigger a sync for a source.
///
/// Scans the source directory for markdown files, compares mtime and content
/// hash against stored sync state, and upserts changed documents through
/// the quality gate.
pub async fn handle_sync_source(
    State(state): State<Arc<RwLock<ServerState>>>,
    Path(id): Path<String>,
) -> Result<Json<SyncStatsResponse>, ServerError> {
    // Look up the source in config.
    let config = origin_core::config::load_config();
    let source = config
        .sources
        .iter()
        .find(|s| s.id == id)
        .cloned()
        .ok_or_else(|| ServerError::NotFound(format!("Source not found: {}", id)))?;

    if source.source_type != SourceType::Obsidian {
        return Err(ServerError::ValidationError(format!(
            "Sync is only supported for Obsidian sources, got: {:?}",
            source.source_type
        )));
    }

    // Clone the DB Arc out of the state guard, then drop the guard.
    let db = {
        let s = state.read().await;
        s.db.clone().ok_or(ServerError::DbNotInitialized)?
    };

    let md_files = scan_vault(&source.path);
    let mut ingested: usize = 0;
    let mut skipped: usize = 0;
    let mut errors: usize = 0;
    // Per-file error counters used for the GDrive threshold. Kept separate
    // from `errors` (which also tallies per-chunk upsert failures) so the
    // threshold comparison stays in consistent "files" units.
    let mut file_errors: usize = 0;
    let mut gdrive_errors: usize = 0;

    for file_path in &md_files {
        let file_key = file_path.to_string_lossy().to_string();
        let is_gdrive = is_google_drive_path(file_path);

        // Read file metadata + content.
        let metadata = match std::fs::metadata(file_path) {
            Ok(m) => m,
            Err(e) => {
                tracing::warn!("[sync] stat failed for {}: {}", file_path.display(), e);
                errors += 1;
                file_errors += 1;
                if is_gdrive {
                    gdrive_errors += 1;
                }
                continue;
            }
        };
        let mtime_ns = metadata
            .modified()
            .ok()
            .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
            .map(|d| d.as_nanos() as i64)
            .unwrap_or(0);

        // Check sync state — skip if mtime unchanged.
        let existing = db.get_sync_state(&id, &file_key).await.ok().flatten();
        if let Some(ref ss) = existing {
            if ss.mtime_ns == mtime_ns {
                skipped += 1;
                continue;
            }
        }

        let content = match std::fs::read_to_string(file_path) {
            Ok(c) => c,
            Err(e) => {
                tracing::warn!("[sync] read failed for {}: {}", file_path.display(), e);
                errors += 1;
                file_errors += 1;
                if is_gdrive {
                    gdrive_errors += 1;
                }
                continue;
            }
        };

        // Hash check — skip if content unchanged despite mtime change.
        let hash = content_hash(&content);
        if let Some(ref ss) = existing {
            if ss.content_hash == hash {
                // mtime changed but content didn't — update mtime only.
                let _ = db.upsert_sync_state(&id, &file_key, mtime_ns, &hash).await;
                skipped += 1;
                continue;
            }
        }

        let mtime_secs = mtime_ns / 1_000_000_000;
        let docs = note_to_documents(&id, file_path, &content, mtime_secs);
        if docs.is_empty() {
            // MOC or otherwise empty — still record sync state.
            let _ = db.upsert_sync_state(&id, &file_key, mtime_ns, &hash).await;
            skipped += 1;
            continue;
        }

        // Filter through quality gate (acquire a brief read guard, then drop).
        let filtered: Vec<_> = {
            let s = state.read().await;
            docs.into_iter()
                .filter(|d| s.quality_gate.check_content(&d.content).admitted)
                .collect()
        };
        // Guard dropped here — safe to .await below.

        // If everything was filtered out (pure boilerplate), DO NOT mark the
        // file as synced. Leaving sync_state untouched means a future sync
        // will re-read the file — giving us a chance to ingest it if the
        // quality gate rules are later loosened. If we upserted here, the
        // hash match would make the file permanently invisible until the
        // user edits it. (Invariant from PR #57.)
        if filtered.is_empty() {
            skipped += 1;
            continue;
        }

        let count = filtered.len();
        match db.upsert_documents(filtered).await {
            Ok(_) => {
                ingested += count;
                let _ = db.upsert_sync_state(&id, &file_key, mtime_ns, &hash).await;
            }
            Err(e) => {
                tracing::error!("[sync] upsert failed for {}: {}", file_path.display(), e);
                errors += count;
            }
        }
    }

    // Categorize errors for user-facing display. If most of the per-file
    // errors came from Google Drive online-only files, surface that
    // specifically so the user knows the fix is "make files available
    // offline in Finder". We compare per-file counts (not the mixed
    // files+chunks `errors` total) so a single upsert failure on a
    // multi-chunk file doesn't skew the threshold.
    let error_detail: Option<String> = if errors == 0 {
        None
    } else if file_errors > 0 && gdrive_errors * 2 >= file_errors {
        Some("google_drive_offline".to_string())
    } else {
        Some("file_read_errors".to_string())
    };

    tracing::info!(
        "[sync] {} complete: {} files, {} ingested, {} skipped, {} errors ({:?})",
        id,
        md_files.len(),
        ingested,
        skipped,
        errors,
        error_detail
    );

    // Update source metadata in config. This is the canonical write path —
    // the Tauri-side `sync_registered_source` command used to also write here
    // which double-counted `memory_count`; it now skips the write.
    let mut config = origin_core::config::load_config();
    if let Some(src) = config.sources.iter_mut().find(|s| s.id == id) {
        src.last_sync = Some(chrono::Utc::now().timestamp());
        src.file_count = md_files.len() as u64;
        src.memory_count = src.memory_count.saturating_add(ingested as u64);
        src.last_sync_errors = errors as u64;
        src.last_sync_error_detail = error_detail.clone();
    }
    let _ = origin_core::config::save_config(&config);

    Ok(Json(SyncStatsResponse {
        files_found: md_files.len(),
        ingested,
        skipped,
        errors,
        error_detail,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn content_hash_is_deterministic() {
        assert_eq!(content_hash("hello"), content_hash("hello"));
        assert_ne!(content_hash("hello"), content_hash("world"));
    }

    #[tokio::test]
    async fn handle_sync_source_returns_not_found_for_missing_id() {
        use crate::state::ServerState;
        let state = Arc::new(RwLock::new(ServerState::default()));

        let result =
            handle_sync_source(State(state), Path("nonexistent-source-id".to_string())).await;

        match result {
            Err(ServerError::NotFound(msg)) => {
                assert!(msg.contains("nonexistent-source-id"));
            }
            _ => panic!("Expected NotFound error"),
        }
    }
}
