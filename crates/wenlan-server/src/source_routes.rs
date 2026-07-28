// SPDX-License-Identifier: Apache-2.0
//! REST API endpoints for source management.

use crate::error::ServerError;
use crate::state::ServerState;
use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::Json,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::RwLock;
use wenlan_core::sources::directory::{is_reserved_ingest_root, scan_directory};
use wenlan_core::sources::obsidian::{has_any_markdown, note_to_documents, scan_vault};
use wenlan_core::sources::Source;
use wenlan_types::sources::{SourceType, SyncStatus};

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
    /// Set when background document enrichment is paused (LLM failure, awaiting
    /// a backoff retry): the pause reason. Additive + optional so older clients
    /// deserialize cleanly and a `None` is omitted from the wire.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub paused: Option<String>,
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
    let config = wenlan_core::config::load_config();
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

    let mut config = wenlan_core::config::load_config();
    let st = match body.source_type.as_str() {
        "obsidian" => {
            if !path.is_dir() {
                return Err(ServerError::ValidationError(
                    "Path is not a directory".to_string(),
                ));
            }
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
        "directory" => {
            if !path.is_dir() && !path.is_file() {
                return Err(ServerError::ValidationError(
                    "Path is not a file or directory".to_string(),
                ));
            }
            let knowledge_path = config.knowledge_path_or_default();
            if is_reserved_ingest_root(&path, &knowledge_path) {
                return Err(ServerError::ValidationError(format!(
                    "Path is a reserved ingest root and cannot be registered: {}",
                    path.display()
                )));
            }
            SourceType::Directory
        }
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
    let slug = wenlan_core::export::obsidian::slugify(&dirname);
    let id = format!("{}-{}", st.as_str(), slug);

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
    wenlan_core::config::save_config(&config)?;

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
    let mut config = wenlan_core::config::load_config();
    let source = config
        .sources
        .iter()
        .find(|s| s.id == id)
        .cloned()
        .ok_or_else(|| ServerError::NotFound(format!("Source not found: {}", id)))?;

    config.sources.retain(|s| s.id != id);
    wenlan_core::config::save_config(&config)?;

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
    content_hash_bytes(content.as_bytes())
}

/// A Directory source root is "live" when it exists and is reachable: a
/// directory we can actually enumerate, or a present single file. An
/// unreadable directory (`read_dir` errors) is deliberately NOT live —
/// otherwise `scan_directory` would return an empty set and the deletion diff
/// would reap every tracked file, i.e. a gone-root masquerading as gone-files
/// (§4/§5). Symlinks resolve through `metadata`.
fn directory_root_is_live(path: &std::path::Path) -> bool {
    match std::fs::metadata(path) {
        Ok(m) if m.is_dir() => std::fs::read_dir(path).is_ok(),
        Ok(m) => m.is_file(),
        Err(_) => false,
    }
}

/// Mark a source `Unavailable` in config because its root is missing/unreadable.
/// Deletes nothing (root-gone != file-gone) and stamps `last_sync` so the UI
/// shows the check happened. Auto-recovers: the next sync against a live root
/// flips it back to `Active` in `finalize_sync`.
fn mark_source_unavailable(id: &str, reason: &str) {
    let mut config = wenlan_core::config::load_config();
    if let Some(src) = config.sources.iter_mut().find(|s| s.id == id) {
        src.status = SyncStatus::Unavailable(reason.to_string());
        src.last_sync = Some(chrono::Utc::now().timestamp());
    }
    let _ = wenlan_core::config::save_config(&config);
}

fn content_hash_bytes(content: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(content);
    format!("{:x}", hasher.finalize())
}

/// Finalize a sync by categorizing errors, updating source metadata in config,
/// fetching paused status, and building the response.
/// Extracted to avoid duplication between Directory and Obsidian sync branches.
#[allow(clippy::too_many_arguments)]
async fn finalize_sync(
    db: Arc<wenlan_core::db::MemoryDB>,
    config_id: &str,
    files_found: usize,
    ingested: usize,
    skipped: usize,
    errors: usize,
    file_errors: usize,
    gdrive_errors: usize,
) -> Result<SyncStatsResponse, ServerError> {
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
        config_id,
        files_found,
        ingested,
        skipped,
        errors,
        error_detail
    );

    // Update source metadata in config. This is the canonical write path —
    // the Tauri-side `sync_registered_source` command used to also write here
    // which double-counted `memory_count`; it now skips the write.
    let mut config = wenlan_core::config::load_config();
    if let Some(src) = config.sources.iter_mut().find(|s| s.id == config_id) {
        src.last_sync = Some(chrono::Utc::now().timestamp());
        src.file_count = files_found as u64;
        src.memory_count = src.memory_count.saturating_add(ingested as u64);
        src.last_sync_errors = errors as u64;
        src.last_sync_error_detail = error_detail.clone();
        // Recovery: a source that had gone Unavailable (missing/unreadable root)
        // flips back to Active once a sync completes against a live root. Only
        // touch Unavailable — never clobber a user's explicit Paused state.
        if matches!(src.status, SyncStatus::Unavailable(_)) {
            src.status = SyncStatus::Active;
        }
    }
    let _ = wenlan_core::config::save_config(&config);

    // Surface any paused background document-enrichment so a sync caller sees
    // the queue is stalled on an LLM failure (waiting for a backoff retry).
    let paused = db
        .document_enrichment_queue_status()
        .await
        .ok()
        .and_then(|q| q.paused_reason);

    Ok(SyncStatsResponse {
        files_found,
        ingested,
        skipped,
        errors,
        error_detail,
        paused,
    })
}

/// Core Directory-source sync: root-guard + cheap mtime/hash diff + deletion
/// propagation + rename optimization, enqueueing changed files for background
/// document enrichment. This is the ONE shared routine so the HTTP handler
/// (`handle_sync_source`) and the background scheduler (§4) never re-implement
/// the diff. The caller guarantees `source.source_type == SourceType::Directory`.
///
/// Unlike the Obsidian branch, this path needs no `ServerState` (no quality
/// gate): it depends only on the DB, the source, and the resolved
/// `knowledge_path` from config, so it stays framework-light and testable.
pub(crate) async fn sync_directory_source(
    db: Arc<wenlan_core::db::MemoryDB>,
    source: &Source,
    config: &wenlan_core::config::Config,
) -> Result<SyncStatsResponse, ServerError> {
    let id = source.id.clone();

    // Root-guard (§4/§5): a missing/unreadable root means "source
    // unavailable", NOT "every file deleted". Mark it unavailable, diff
    // nothing, delete zero rows, and return early — a later sync that finds
    // the root live auto-recovers it. For a single-file source the root IS
    // the file, so a deleted file lands here too: never auto-deleted, the
    // user removes the source explicitly.
    let root = source.path.clone();
    let root_display = root.display().to_string();
    let files = tokio::task::spawn_blocking(move || {
        directory_root_is_live(&root).then(|| scan_directory(&root))
    })
    .await
    .map_err(|error| {
        ServerError::Internal(format!(
            "directory filesystem task failed for source {id} at {root_display}: {error}"
        ))
    })?;
    let Some(files) = files else {
        mark_source_unavailable(&id, "source path is missing or unreadable");
        return Ok(SyncStatsResponse {
            files_found: 0,
            ingested: 0,
            skipped: 0,
            errors: 0,
            error_detail: None,
            paused: None,
        });
    };

    let knowledge_path = config.knowledge_path_or_default();
    let scanned: std::collections::HashSet<String> = files
        .iter()
        .map(|p| p.to_string_lossy().to_string())
        .collect();

    // Deletion propagation & rename optimization setup (§4/§5):
    // - The root is live, so any tracked file NOT in the current scan vanished.
    // - Capture content_hash of vanished files for rename optimization.
    // - If a new file's hash matches a vanished file's, rebind chunks instead of
    //   delete+enqueue (pure-DB op: UPDATE memories.source_id old→new).
    // - For non-matching files, delete normally and enqueue for re-enrichment.
    //
    // Build a hash->vanished_file map first, before enqueuing new files.
    let mut vanished_by_hash: std::collections::HashMap<String, (String, String)> =
        std::collections::HashMap::new(); // hash -> (old_doc_source_id, old_path)
    let mut files_to_delete: Vec<(String, String)> = Vec::new(); // (path, doc_source_id)

    if let Ok(tracked) = db.list_sync_state_paths(&id).await {
        for tracked_path in tracked {
            if scanned.contains(&tracked_path) {
                continue;
            }
            // Fetch the vanished file's sync state to get its content_hash.
            if let Ok(Some(sync_state)) = db.get_sync_state(&id, &tracked_path).await {
                let doc_source_id = wenlan_core::sources::directory::document_source_id(
                    &id,
                    std::path::Path::new(&tracked_path),
                    Some(&knowledge_path),
                );
                // Record this vanished file's hash for the optimization.
                vanished_by_hash.insert(
                    sync_state.content_hash.clone(),
                    (doc_source_id.clone(), tracked_path.clone()),
                );
                files_to_delete.push((tracked_path.clone(), doc_source_id));
            }
        }
    }

    let mut ingested: usize = 0;
    let mut skipped: usize = 0;
    let mut errors: usize = 0;
    let mut file_errors: usize = 0;
    let mut gdrive_errors: usize = 0;
    let mut renamed_files: std::collections::HashSet<String> = std::collections::HashSet::new();

    for file_path in &files {
        let file_key = file_path.to_string_lossy().to_string();
        let is_gdrive = is_google_drive_path(file_path);

        let metadata_path = file_path.clone();
        let metadata =
            match tokio::task::spawn_blocking(move || std::fs::metadata(metadata_path)).await {
                Ok(Ok(metadata)) => metadata,
                Ok(Err(e)) => {
                    tracing::warn!("[sync] stat failed for {}: {}", file_path.display(), e);
                    errors += 1;
                    file_errors += 1;
                    if is_gdrive {
                        gdrive_errors += 1;
                    }
                    continue;
                }
                Err(e) => {
                    tracing::warn!("[sync] stat task failed for {}: {}", file_path.display(), e);
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

        let existing = db.get_sync_state(&id, &file_key).await.ok().flatten();
        if let Some(ref ss) = existing {
            if ss.mtime_ns == mtime_ns {
                skipped += 1;
                continue;
            }
        }

        // The mtime fast path above keeps unchanged files out of the read/hash phase.
        let read_path = file_path.clone();
        let hash = match tokio::task::spawn_blocking(move || {
            std::fs::read(read_path).map(|bytes| content_hash_bytes(&bytes))
        })
        .await
        {
            Ok(Ok(hash)) => hash,
            Ok(Err(e)) => {
                tracing::warn!("[sync] read failed for {}: {}", file_path.display(), e);
                errors += 1;
                file_errors += 1;
                if is_gdrive {
                    gdrive_errors += 1;
                }
                continue;
            }
            Err(e) => {
                tracing::warn!("[sync] read task failed for {}: {}", file_path.display(), e);
                errors += 1;
                file_errors += 1;
                if is_gdrive {
                    gdrive_errors += 1;
                }
                continue;
            }
        };
        if let Some(ref ss) = existing {
            if ss.content_hash == hash {
                let _ = db.upsert_sync_state(&id, &file_key, mtime_ns, &hash).await;
                skipped += 1;
                continue;
            }
        }

        // Rename optimization: check if this file's hash matches a vanished file's.
        if let Some((old_doc_source_id, old_path)) = vanished_by_hash.remove(&hash) {
            // This file has the same content as a vanished file — rebind instead of re-enqueue.
            let new_doc_source_id = wenlan_core::sources::directory::document_source_id(
                &id,
                file_path,
                Some(&knowledge_path),
            );
            let old_source_page_id =
                wenlan_core::document_enrichment::source_page_id(&id, &old_path);
            let new_source_page_id =
                wenlan_core::document_enrichment::source_page_id(&id, &file_key);

            // UPDATE memories: rebind chunks from old_doc_source_id to new_doc_source_id.
            match db
                .rebind_source_id_with_source_page(
                    "memory",
                    old_doc_source_id.as_str(),
                    &new_doc_source_id,
                    &old_source_page_id,
                    &new_source_page_id,
                )
                .await
            {
                Ok(()) => {
                    // Sync/queue bookkeeping is path-keyed rather than
                    // document-id-keyed, so it stays outside rebind_source_id.
                    for outcome in [
                        db.delete_sync_state(&id, &old_path).await,
                        db.dequeue_document(&id, &old_path).await,
                    ] {
                        if let Err(error) = outcome {
                            tracing::warn!(
                                "[sync] rename cleanup failed for {}: {}",
                                old_path,
                                error
                            );
                            errors += 1;
                            file_errors += 1;
                        }
                    }
                    if let Err(error) = db.upsert_sync_state(&id, &file_key, mtime_ns, &hash).await
                    {
                        tracing::warn!(
                            "[sync] renamed sync_state failed for {}: {}",
                            file_path.display(),
                            error
                        );
                        errors += 1;
                        file_errors += 1;
                    }

                    // Mark this file as renamed (don't delete it later).
                    renamed_files.insert(old_path);
                    // Skipped instead of ingested: chunks reused, not re-enriched.
                    skipped += 1;
                    continue;
                }
                Err(error) => {
                    tracing::warn!(
                        "[sync] rebind yielded for {} (candidate {}): {}; falling back to enqueue",
                        file_path.display(),
                        old_path,
                        error
                    );
                }
            }
        }

        // Normal enqueue (no rename match).
        match db.enqueue_document(&id, &file_key, Some(&hash)).await {
            Ok(_) => {
                ingested += 1;
            }
            Err(e) => {
                tracing::error!("[sync] enqueue failed for {}: {}", file_path.display(), e);
                errors += 1;
            }
        }
    }

    // Clean up vanished files that were not renamed (sources matched earlier).
    for (path, doc_source_id) in files_to_delete {
        if !renamed_files.contains(&path) {
            let _ = db.delete_by_source_id("memory", &doc_source_id).await;
            let _ = db.delete_sync_state(&id, &path).await;
            let _ = db.dequeue_document(&id, &path).await;
        }
    }

    finalize_sync(
        db,
        &id,
        files.len(),
        ingested,
        skipped,
        errors,
        file_errors,
        gdrive_errors,
    )
    .await
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
    let config = wenlan_core::config::load_config();
    let source = config
        .sources
        .iter()
        .find(|s| s.id == id)
        .cloned()
        .ok_or_else(|| ServerError::NotFound(format!("Source not found: {}", id)))?;

    // Clone the DB Arc out of the state guard, then drop the guard.
    let db = {
        let s = state.read().await;
        s.db.clone().ok_or(ServerError::DbNotInitialized)?
    };

    if source.source_type == SourceType::Directory {
        return sync_directory_source(db, &source, &config).await.map(Json);
    }

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

    finalize_sync(
        db,
        &id,
        md_files.len(),
        ingested,
        skipped,
        errors,
        file_errors,
        gdrive_errors,
    )
    .await
    .map(Json)
}

#[cfg(test)]
mod tests {
    use super::*;
    use wenlan_core::config::Config;
    use wenlan_core::events::NoopEmitter;
    use wenlan_core::sources::SyncStatus;

    struct DataDirGuard {
        previous: Option<std::ffi::OsString>,
        _tmp: tempfile::TempDir,
    }

    impl DataDirGuard {
        fn new() -> Self {
            let tmp = tempfile::tempdir().unwrap();
            let previous = std::env::var_os("WENLAN_DATA_DIR");
            std::env::set_var("WENLAN_DATA_DIR", tmp.path());
            Self {
                previous,
                _tmp: tmp,
            }
        }
    }

    impl Drop for DataDirGuard {
        fn drop(&mut self) {
            match &self.previous {
                Some(value) => std::env::set_var("WENLAN_DATA_DIR", value),
                None => std::env::remove_var("WENLAN_DATA_DIR"),
            }
        }
    }

    fn mtime_ns(path: &std::path::Path) -> i64 {
        std::fs::metadata(path)
            .unwrap()
            .modified()
            .unwrap()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as i64
    }

    #[test]
    fn content_hash_is_deterministic() {
        assert_eq!(content_hash("hello"), content_hash("hello"));
        assert_ne!(content_hash("hello"), content_hash("world"));
    }

    #[test]
    fn sync_stats_response_defaults_and_skips_paused_when_absent() {
        // Older responses omit `paused` entirely → deserializes to None.
        let json = r#"{"files_found":1,"ingested":1,"skipped":0,"errors":0}"#;
        let parsed: SyncStatsResponse = serde_json::from_str(json).unwrap();
        assert!(parsed.paused.is_none());
        // A None paused detail is omitted from the wire.
        let out = serde_json::to_string(&parsed).unwrap();
        assert!(
            !out.contains("paused"),
            "None paused must be omitted: {out}"
        );

        // A paused detail round-trips.
        let with = SyncStatsResponse {
            files_found: 0,
            ingested: 0,
            skipped: 0,
            errors: 0,
            error_detail: None,
            paused: Some("analysis LLM failed".to_string()),
        };
        let s = serde_json::to_string(&with).unwrap();
        assert!(
            s.contains("\"paused\":\"analysis LLM failed\""),
            "paused detail must serialize: {s}"
        );
        let back: SyncStatsResponse = serde_json::from_str(&s).unwrap();
        assert_eq!(back.paused.as_deref(), Some("analysis LLM failed"));
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

    #[tokio::test]
    async fn handle_add_source_accepts_single_file_directory_source() {
        let _lock = crate::TEST_DATA_DIR_LOCK
            .get_or_init(|| tokio::sync::Mutex::new(()))
            .lock()
            .await;
        let _env = DataDirGuard::new();
        let source_root = tempfile::tempdir().unwrap();
        let file_path = source_root.path().join("paper.txt");
        std::fs::write(
            &file_path,
            "folder source registration can point at one file",
        )
        .unwrap();
        let state = Arc::new(RwLock::new(ServerState::default()));

        let Json(source) = handle_add_source(
            State(state.clone()),
            Json(AddSourceRequest {
                source_type: "directory".to_string(),
                path: file_path.to_string_lossy().to_string(),
            }),
        )
        .await
        .expect("single-file directory source should register");

        assert_eq!(source.source_type, SourceType::Directory);
        assert_eq!(source.path, file_path);
        assert!(state.read().await.watch_paths.contains(&file_path));
    }

    #[tokio::test]
    async fn handle_add_source_rejects_reserved_pages_directory() {
        let _lock = crate::TEST_DATA_DIR_LOCK
            .get_or_init(|| tokio::sync::Mutex::new(()))
            .lock()
            .await;
        let env = DataDirGuard::new();
        let pages_path = env._tmp.path().join("pages");
        std::fs::create_dir(&pages_path).unwrap();
        let config = Config {
            knowledge_path: Some(pages_path.clone()),
            ..Config::default()
        };
        wenlan_core::config::save_config(&config).unwrap();
        let state = Arc::new(RwLock::new(ServerState::default()));

        let result = handle_add_source(
            State(state),
            Json(AddSourceRequest {
                source_type: "directory".to_string(),
                path: pages_path.to_string_lossy().to_string(),
            }),
        )
        .await;

        match result {
            Err(ServerError::ValidationError(msg)) => {
                assert!(
                    msg.contains("reserved"),
                    "reserved-root rejection should explain the guard: {msg}"
                );
            }
            other => panic!("expected reserved-root ValidationError, got {other:?}"),
        }
    }

    /// Seed a fully-enriched folder document: its chunks under the canonical
    /// `document_source_id` plus a matching `source_sync_state` row (mtime +
    /// content hash), exactly as the write path leaves them. Returns the
    /// `doc_source_id` so callers can assert chunk presence/absence.
    async fn seed_document(
        db: &wenlan_core::db::MemoryDB,
        source_id: &str,
        file_path: &std::path::Path,
        knowledge_path: &std::path::Path,
        content: &str,
    ) -> String {
        let doc_source_id = wenlan_core::sources::directory::document_source_id(
            source_id,
            file_path,
            Some(knowledge_path),
        );
        let doc = wenlan_core::sources::RawDocument {
            source: "memory".to_string(),
            source_id: doc_source_id.clone(),
            title: file_path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("doc")
                .to_string(),
            content: content.to_string(),
            last_modified: chrono::Utc::now().timestamp(),
            source_agent: Some("folder".to_string()),
            ..Default::default()
        };
        db.upsert_documents(vec![doc]).await.unwrap();
        let file_key = file_path.to_string_lossy().to_string();
        db.upsert_sync_state(
            source_id,
            &file_key,
            mtime_ns(file_path),
            &content_hash(content),
        )
        .await
        .unwrap();
        doc_source_id
    }

    fn register_directory_source(id: &str, path: &std::path::Path) {
        let source = Source {
            id: id.to_string(),
            source_type: SourceType::Directory,
            path: path.to_path_buf(),
            status: SyncStatus::Active,
            last_sync: None,
            file_count: 0,
            memory_count: 0,
            last_sync_errors: 0,
            last_sync_error_detail: None,
        };
        wenlan_core::config::save_config(&Config {
            sources: vec![source],
            ..Config::default()
        })
        .unwrap();
    }

    async fn new_test_db() -> (Arc<wenlan_core::db::MemoryDB>, tempfile::TempDir) {
        let db_dir = tempfile::tempdir().unwrap();
        let db = Arc::new(
            wenlan_core::db::MemoryDB::new(db_dir.path(), Arc::new(NoopEmitter))
                .await
                .unwrap(),
        );
        (db, db_dir)
    }

    fn loaded_source_status(id: &str) -> SyncStatus {
        wenlan_core::config::load_config()
            .sources
            .into_iter()
            .find(|s| s.id == id)
            .expect("source in config")
            .status
    }

    #[tokio::test]
    async fn handle_sync_source_deletes_vanished_file_under_live_root() {
        let _lock = crate::TEST_DATA_DIR_LOCK
            .get_or_init(|| tokio::sync::Mutex::new(()))
            .lock()
            .await;
        let _env = DataDirGuard::new();

        let source_root = tempfile::tempdir().unwrap();
        let kept_path = source_root.path().join("kept.txt");
        let gone_path = source_root.path().join("gone.txt");
        std::fs::write(&kept_path, "This file stays put on the live root.").unwrap();
        std::fs::write(&gone_path, "This file will vanish from the live root.").unwrap();

        let source_id = "directory-notes".to_string();
        register_directory_source(&source_id, source_root.path());
        let knowledge_path = wenlan_core::config::load_config().knowledge_path_or_default();

        let (db, _db_dir) = new_test_db().await;
        let kept_doc = seed_document(
            &db,
            &source_id,
            &kept_path,
            &knowledge_path,
            "This file stays put on the live root.",
        )
        .await;
        let gone_doc = seed_document(
            &db,
            &source_id,
            &gone_path,
            &knowledge_path,
            "This file will vanish from the live root.",
        )
        .await;
        // A still-pending enrichment for the vanished file must be dequeued so
        // the worker can never re-materialize its chunks after deletion.
        db.enqueue_document(&source_id, &gone_path.to_string_lossy(), Some("hash-gone"))
            .await
            .unwrap();

        // The file disappears from a LIVE root -> its chunks must be reaped.
        std::fs::remove_file(&gone_path).unwrap();

        let state = Arc::new(RwLock::new(ServerState {
            db: Some(db.clone()),
            ..ServerState::default()
        }));
        let _ = handle_sync_source(State(state), Path(source_id.clone()))
            .await
            .expect("directory sync should succeed on a live root");

        assert!(
            db.get_memories_by_source_id("memory", &gone_doc)
                .await
                .unwrap()
                .is_empty(),
            "vanished file's chunks must be deleted"
        );
        assert!(
            !db.get_memories_by_source_id("memory", &kept_doc)
                .await
                .unwrap()
                .is_empty(),
            "surviving file's chunks must be retained"
        );
        assert!(
            db.get_sync_state(&source_id, &gone_path.to_string_lossy())
                .await
                .unwrap()
                .is_none(),
            "vanished file's sync_state must be cleared"
        );
        assert!(
            db.get_sync_state(&source_id, &kept_path.to_string_lossy())
                .await
                .unwrap()
                .is_some(),
            "surviving file's sync_state must remain"
        );
        assert!(
            db.get_queue_entry(&source_id, &gone_path.to_string_lossy())
                .await
                .unwrap()
                .is_none(),
            "vanished file's pending enrichment must be dequeued"
        );
        assert!(
            matches!(loaded_source_status(&source_id), SyncStatus::Active),
            "a live-root sync leaves the source Active"
        );
    }

    #[tokio::test]
    async fn handle_sync_source_missing_root_marks_unavailable_and_deletes_nothing() {
        let _lock = crate::TEST_DATA_DIR_LOCK
            .get_or_init(|| tokio::sync::Mutex::new(()))
            .lock()
            .await;
        let _env = DataDirGuard::new();

        let parent = tempfile::tempdir().unwrap();
        let root = parent.path().join("live_root");
        std::fs::create_dir(&root).unwrap();
        let file_path = root.join("note.txt");
        std::fs::write(&file_path, "Chunks that must survive a missing root.").unwrap();

        let source_id = "directory-notes".to_string();
        register_directory_source(&source_id, &root);
        let knowledge_path = wenlan_core::config::load_config().knowledge_path_or_default();

        let (db, _db_dir) = new_test_db().await;
        let doc = seed_document(
            &db,
            &source_id,
            &file_path,
            &knowledge_path,
            "Chunks that must survive a missing root.",
        )
        .await;

        // Root vanishes (renamed/removed). root-gone != file-gone.
        std::fs::remove_dir_all(&root).unwrap();

        let state = Arc::new(RwLock::new(ServerState {
            db: Some(db.clone()),
            ..ServerState::default()
        }));
        let Json(stats) = handle_sync_source(State(state), Path(source_id.clone()))
            .await
            .expect("sync against a missing root should not error");

        assert_eq!(
            stats.files_found, 0,
            "no files scanned under a missing root"
        );
        assert_eq!(stats.ingested, 0);
        assert!(
            !db.get_memories_by_source_id("memory", &doc)
                .await
                .unwrap()
                .is_empty(),
            "a missing root must delete ZERO chunks"
        );
        assert!(
            db.get_sync_state(&source_id, &file_path.to_string_lossy())
                .await
                .unwrap()
                .is_some(),
            "a missing root must leave sync_state intact"
        );
        assert!(
            matches!(loaded_source_status(&source_id), SyncStatus::Unavailable(_)),
            "a missing root marks the source Unavailable"
        );
    }

    #[tokio::test]
    async fn handle_sync_source_single_file_root_missing_marks_unavailable_no_delete() {
        let _lock = crate::TEST_DATA_DIR_LOCK
            .get_or_init(|| tokio::sync::Mutex::new(()))
            .lock()
            .await;
        let _env = DataDirGuard::new();

        let parent = tempfile::tempdir().unwrap();
        let file_path = parent.path().join("paper.txt");
        std::fs::write(&file_path, "A single-file source's only document.").unwrap();

        let source_id = "directory-paper".to_string();
        register_directory_source(&source_id, &file_path);
        let knowledge_path = wenlan_core::config::load_config().knowledge_path_or_default();

        let (db, _db_dir) = new_test_db().await;
        let doc = seed_document(
            &db,
            &source_id,
            &file_path,
            &knowledge_path,
            "A single-file source's only document.",
        )
        .await;

        // For a single-file source the root IS the file: a deleted file is a
        // missing root, which must be treated as "unavailable", never auto-deleted.
        std::fs::remove_file(&file_path).unwrap();

        let state = Arc::new(RwLock::new(ServerState {
            db: Some(db.clone()),
            ..ServerState::default()
        }));
        let _ = handle_sync_source(State(state), Path(source_id.clone()))
            .await
            .expect("single-file sync against a missing file should not error");

        assert!(
            !db.get_memories_by_source_id("memory", &doc)
                .await
                .unwrap()
                .is_empty(),
            "single-file source must never auto-delete on a missing root"
        );
        assert!(
            matches!(loaded_source_status(&source_id), SyncStatus::Unavailable(_)),
            "single-file missing root marks the source Unavailable"
        );
    }

    #[tokio::test]
    async fn handle_sync_source_recovers_when_root_reappears() {
        let _lock = crate::TEST_DATA_DIR_LOCK
            .get_or_init(|| tokio::sync::Mutex::new(()))
            .lock()
            .await;
        let _env = DataDirGuard::new();

        let parent = tempfile::tempdir().unwrap();
        let root = parent.path().join("live_root");
        std::fs::create_dir(&root).unwrap();
        let orig_path = root.join("orig.txt");
        std::fs::write(&orig_path, "Original document before the root vanished.").unwrap();

        let source_id = "directory-notes".to_string();
        register_directory_source(&source_id, &root);
        let knowledge_path = wenlan_core::config::load_config().knowledge_path_or_default();

        let (db, _db_dir) = new_test_db().await;
        seed_document(
            &db,
            &source_id,
            &orig_path,
            &knowledge_path,
            "Original document before the root vanished.",
        )
        .await;

        // Root vanishes -> unavailable, deletes nothing.
        std::fs::remove_dir_all(&root).unwrap();
        let state = Arc::new(RwLock::new(ServerState {
            db: Some(db.clone()),
            ..ServerState::default()
        }));
        let _ = handle_sync_source(State(state.clone()), Path(source_id.clone()))
            .await
            .expect("sync against a missing root should not error");
        assert!(
            matches!(loaded_source_status(&source_id), SyncStatus::Unavailable(_)),
            "root removal marks the source Unavailable"
        );

        // Root reappears with a fresh file -> next sync resyncs and recovers.
        std::fs::create_dir(&root).unwrap();
        let fresh_path = root.join("fresh.txt");
        std::fs::write(&fresh_path, "A fresh file after the root came back online.").unwrap();

        let Json(stats) = handle_sync_source(State(state), Path(source_id.clone()))
            .await
            .expect("sync after the root reappears should succeed");

        assert!(
            matches!(loaded_source_status(&source_id), SyncStatus::Active),
            "a completed sync on a live root flips Unavailable back to Active"
        );
        assert_eq!(
            stats.ingested, 1,
            "the fresh file is enqueued on the recovering sync"
        );
        assert!(
            db.get_queue_entry(&source_id, &fresh_path.to_string_lossy())
                .await
                .unwrap()
                .is_some(),
            "the fresh file is queued for enrichment"
        );
    }

    #[tokio::test]
    async fn handle_sync_source_enqueues_changed_directory_files_and_skips_unchanged() {
        let _lock = crate::TEST_DATA_DIR_LOCK
            .get_or_init(|| tokio::sync::Mutex::new(()))
            .lock()
            .await;
        let _env = DataDirGuard::new();
        let source_root = tempfile::tempdir().unwrap();
        let changed_path = source_root.path().join("changed.txt");
        let unchanged_path = source_root.path().join("unchanged.txt");
        std::fs::write(
            &changed_path,
            "This changed file has enough text for folder sync queueing.",
        )
        .unwrap();
        let unchanged_content = "This unchanged file already has matching sync state.";
        std::fs::write(&unchanged_path, unchanged_content).unwrap();

        let source_id = "directory-notes".to_string();
        let source = Source {
            id: source_id.clone(),
            source_type: SourceType::Directory,
            path: source_root.path().to_path_buf(),
            status: SyncStatus::Active,
            last_sync: None,
            file_count: 0,
            memory_count: 0,
            last_sync_errors: 0,
            last_sync_error_detail: None,
        };
        wenlan_core::config::save_config(&Config {
            sources: vec![source],
            ..Config::default()
        })
        .unwrap();

        let db_dir = tempfile::tempdir().unwrap();
        let db = Arc::new(
            wenlan_core::db::MemoryDB::new(db_dir.path(), Arc::new(NoopEmitter))
                .await
                .unwrap(),
        );
        let unchanged_key = unchanged_path.to_string_lossy().to_string();
        db.upsert_sync_state(
            &source_id,
            &unchanged_key,
            mtime_ns(&unchanged_path),
            &content_hash(unchanged_content),
        )
        .await
        .unwrap();
        let state = Arc::new(RwLock::new(ServerState {
            db: Some(db.clone()),
            ..ServerState::default()
        }));

        let Json(stats) = handle_sync_source(State(state), Path(source_id.clone()))
            .await
            .expect("directory sync should succeed");

        assert_eq!(stats.files_found, 2);
        assert_eq!(stats.ingested, 1);
        assert_eq!(stats.skipped, 1);
        assert_eq!(stats.errors, 0);

        let changed_key = changed_path.to_string_lossy().to_string();
        let queued = db
            .get_queue_entry(&source_id, &changed_key)
            .await
            .unwrap()
            .expect("changed file should be enqueued");
        assert_eq!(queued.status, "pending");
        assert_eq!(queued.source_id, source_id);
        assert_eq!(queued.file_path, changed_key);
        assert!(queued.content_hash.is_some());
        assert!(
            db.get_queue_entry("directory-notes", &unchanged_key)
                .await
                .unwrap()
                .is_none(),
            "unchanged file should not be enqueued"
        );
        assert!(
            db.get_sync_state("directory-notes", &queued.file_path)
                .await
                .unwrap()
                .is_none(),
            "sync_state must not be written at enqueue time"
        );
    }

    #[tokio::test]
    async fn handle_sync_source_rename_optimization_rebinds_chunks_on_content_match() {
        // Rename optimization: if a new file's content_hash equals a just-deleted
        // doc's, reuse enrichment (rebind chunks to the new path) instead of
        // re-running. This test verifies the OPTIMIZED behavior: a renamed file
        // with identical content should NOT be re-enqueued, and chunks should be
        // re-pointed to the new path by updating memories.source_id.
        let _lock = crate::TEST_DATA_DIR_LOCK
            .get_or_init(|| tokio::sync::Mutex::new(()))
            .lock()
            .await;
        let _env = DataDirGuard::new();

        let source_root = tempfile::tempdir().unwrap();
        let orig_path = source_root.path().join("document.txt");
        let renamed_path = source_root.path().join("document_renamed.txt");
        let content =
            "This is the original document content that will be preserved despite the rename.";
        std::fs::write(&orig_path, content).unwrap();

        let source_id = "directory-notes".to_string();
        register_directory_source(&source_id, source_root.path());
        let knowledge_path = wenlan_core::config::load_config().knowledge_path_or_default();

        let (db, _db_dir) = new_test_db().await;
        let orig_doc_source_id = wenlan_core::sources::directory::document_source_id(
            &source_id,
            &orig_path,
            Some(&knowledge_path),
        );
        let orig_doc = seed_document(&db, &source_id, &orig_path, &knowledge_path, content).await;

        // BEFORE rename: original file exists with its enriched chunks.
        let orig_chunks = db
            .get_memories_by_source_id("memory", &orig_doc)
            .await
            .unwrap();
        assert!(
            !orig_chunks.is_empty(),
            "original file should have enriched chunks"
        );
        let orig_chunk_count = orig_chunks.len();
        let orig_chunk_ids: Vec<String> =
            orig_chunks.iter().map(|chunk| chunk.id.clone()).collect();
        let original_key = orig_path.to_string_lossy().to_string();
        let old_source_page_id =
            wenlan_core::document_enrichment::source_page_id(&source_id, &original_key);
        wenlan_core::post_write::page_write(
            &db,
            wenlan_core::post_write::PageWrite::Create {
                page_id: Some(&old_source_page_id),
                req: wenlan_types::requests::CreateConceptRequest {
                    title: "Original Source Page".to_string(),
                    content: "Source Page body survives an identical-content rename.".to_string(),
                    summary: None,
                    entity_id: None,
                    space: None.into(),
                    source_memory_ids: orig_chunk_ids,
                    creation_kind: Some("source".to_string()),
                    workspace: None,
                },
                agent: "test",
                knowledge_path: None,
                page_min_cluster_size: 1,
                page_match_threshold: 0.0,
                citations_json: None,
            },
        )
        .await
        .unwrap();

        // Rename the file: old vanishes, new appears with identical content.
        std::fs::remove_file(&orig_path).unwrap();
        std::fs::write(&renamed_path, content).unwrap();

        let renamed_key = renamed_path.to_string_lossy().to_string();
        let new_doc_source_id = wenlan_core::sources::directory::document_source_id(
            &source_id,
            &renamed_path,
            Some(&knowledge_path),
        );
        let new_source_page_id =
            wenlan_core::document_enrichment::source_page_id(&source_id, &renamed_key);

        let state = Arc::new(RwLock::new(ServerState {
            db: Some(db.clone()),
            ..ServerState::default()
        }));
        let Json(stats) = handle_sync_source(State(state), Path(source_id.clone()))
            .await
            .expect("directory sync should succeed");

        // OPTIMIZED BEHAVIOR:
        // - Renamed file is NOT enqueued (chunks are reused, not re-enriched).
        // - Original doc's chunks are deleted (old path cleaned up).
        // - New doc_source_id has the chunks (re-pointed via UPDATE memories.source_id).
        // - sync_state is updated to the new path.

        assert_eq!(
            stats.ingested, 0,
            "renamed file should NOT be enqueued (optimization: chunks reused)"
        );

        // Original doc_source_id has no chunks (they were transferred/re-pointed).
        assert!(
            db.get_memories_by_source_id("memory", &orig_doc_source_id)
                .await
                .unwrap()
                .is_empty(),
            "original doc_source_id should have no chunks (re-pointed to new path)"
        );

        // New doc_source_id has the chunks (re-pointed from the old path).
        let new_chunks = db
            .get_memories_by_source_id("memory", &new_doc_source_id)
            .await
            .unwrap();
        assert_eq!(
            new_chunks.len(),
            orig_chunk_count,
            "renamed file should have same chunks re-pointed to the new path"
        );
        assert!(
            db.get_page(&old_source_page_id).await.unwrap().is_none(),
            "the path-derived SOURCE Page id must not remain orphaned"
        );
        let renamed_page = db
            .get_page(&new_source_page_id)
            .await
            .unwrap()
            .expect("SOURCE Page identity should move with the file");
        assert_eq!(renamed_page.version, 1);

        // New sync_state exists with the new path.
        assert!(
            db.get_sync_state(&source_id, &renamed_key)
                .await
                .unwrap()
                .is_some(),
            "renamed file's sync_state should be created"
        );

        // Renamed file should NOT be in the queue (optimization: not re-enqueued).
        assert!(
            db.get_queue_entry(&source_id, &renamed_key)
                .await
                .unwrap()
                .is_none(),
            "renamed file should NOT be queued (chunks reused, not re-enriched)"
        );
    }

    #[tokio::test]
    async fn rename_optimization_consumes_each_vanished_hash_only_once() {
        let _lock = crate::TEST_DATA_DIR_LOCK
            .get_or_init(|| tokio::sync::Mutex::new(()))
            .lock()
            .await;
        let _env = DataDirGuard::new();

        let source_root = tempfile::tempdir().unwrap();
        let original_path = source_root.path().join("original.txt");
        let first_new_path = source_root.path().join("copy_a.txt");
        let second_new_path = source_root.path().join("copy_b.txt");
        let content = "One vanished document cannot be rebound into two new files.";
        std::fs::write(&original_path, content).unwrap();

        let source_id = "directory-duplicate-hash".to_string();
        register_directory_source(&source_id, source_root.path());
        let knowledge_path = wenlan_core::config::load_config().knowledge_path_or_default();
        let (db, _db_dir) = new_test_db().await;
        seed_document(&db, &source_id, &original_path, &knowledge_path, content).await;

        std::fs::remove_file(&original_path).unwrap();
        std::fs::write(&first_new_path, content).unwrap();
        std::fs::write(&second_new_path, content).unwrap();

        let state = Arc::new(RwLock::new(ServerState {
            db: Some(db.clone()),
            ..ServerState::default()
        }));
        let Json(stats) = handle_sync_source(State(state), Path(source_id.clone()))
            .await
            .expect("directory sync should succeed");

        assert_eq!(
            stats.errors, 0,
            "duplicate content is valid, not a sync error"
        );
        assert_eq!(
            stats.ingested, 1,
            "one copy reuses chunks and the other must enter normal enrichment"
        );
        let mut rebound_files = 0usize;
        let mut queued_files = 0usize;
        for path in [&first_new_path, &second_new_path] {
            let doc_source_id = wenlan_core::sources::directory::document_source_id(
                &source_id,
                path,
                Some(&knowledge_path),
            );
            if !db
                .get_memories_by_source_id("memory", &doc_source_id)
                .await
                .unwrap()
                .is_empty()
            {
                rebound_files += 1;
            }
            if db
                .get_queue_entry(&source_id, &path.to_string_lossy())
                .await
                .unwrap()
                .is_some()
            {
                queued_files += 1;
            }
        }
        assert_eq!(rebound_files, 1);
        assert_eq!(queued_files, 1);
    }

    #[tokio::test]
    async fn occupied_rename_target_falls_back_to_normal_enrichment() {
        let _lock = crate::TEST_DATA_DIR_LOCK
            .get_or_init(|| tokio::sync::Mutex::new(()))
            .lock()
            .await;
        let _env = DataDirGuard::new();

        let source_root = tempfile::tempdir().unwrap();
        let vanished_path = source_root.path().join("vanished.txt");
        let target_path = source_root.path().join("target.txt");
        let moved_content = "Content moving onto an already tracked target path.";
        let old_target_content = "The target previously held different content.";
        std::fs::write(&vanished_path, moved_content).unwrap();
        std::fs::write(&target_path, old_target_content).unwrap();

        let source_id = "directory-occupied-target".to_string();
        register_directory_source(&source_id, source_root.path());
        let knowledge_path = wenlan_core::config::load_config().knowledge_path_or_default();
        let (db, _db_dir) = new_test_db().await;
        let vanished_doc = seed_document(
            &db,
            &source_id,
            &vanished_path,
            &knowledge_path,
            moved_content,
        )
        .await;
        let target_doc = seed_document(
            &db,
            &source_id,
            &target_path,
            &knowledge_path,
            old_target_content,
        )
        .await;

        std::fs::remove_file(&vanished_path).unwrap();
        std::fs::write(&target_path, moved_content).unwrap();
        let target_key = target_path.to_string_lossy().to_string();
        db.upsert_sync_state(
            &source_id,
            &target_key,
            -1,
            &content_hash(old_target_content),
        )
        .await
        .unwrap();

        let state = Arc::new(RwLock::new(ServerState {
            db: Some(db.clone()),
            ..ServerState::default()
        }));
        let Json(stats) = handle_sync_source(State(state), Path(source_id.clone()))
            .await
            .expect("directory sync should succeed");

        assert_eq!(stats.errors, 0);
        assert_eq!(stats.ingested, 1, "the occupied target must be enqueued");
        assert!(
            db.get_memories_by_source_id("memory", &vanished_doc)
                .await
                .unwrap()
                .is_empty(),
            "the vanished source is cleaned through canonical forget"
        );
        let target_chunks = db
            .get_memories_by_source_id("memory", &target_doc)
            .await
            .unwrap();
        assert_eq!(target_chunks.len(), 1, "target chunks must not be merged");
        assert_eq!(target_chunks[0].content, old_target_content);
        let queued = db
            .get_queue_entry(&source_id, &target_key)
            .await
            .unwrap()
            .expect("target should wait for bounded enrichment");
        assert_eq!(queued.content_hash, Some(content_hash(moved_content)));
    }
}
