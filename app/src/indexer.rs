// SPDX-License-Identifier: AGPL-3.0-only
use crate::error::OriginError;
use crate::sources::local_files::LocalFilesSource;
use crate::sources::RawDocument;
use crate::state::AppState;
use notify_debouncer_full::{new_debouncer, DebounceEventResult, Debouncer};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;

/// Type alias for the file system watcher.
pub type FileWatcher = Debouncer<notify::RecommendedWatcher, notify_debouncer_full::FileIdMap>;

/// Request body for the daemon's `/api/ingest/text` endpoint.
#[derive(Serialize)]
struct IngestTextRequest {
    source: String,
    source_id: String,
    title: String,
    content: String,
    url: Option<String>,
    metadata: Option<HashMap<String, String>>,
}

/// Response from the daemon's `/api/ingest/text` endpoint.
#[derive(Deserialize)]
struct IngestResponse {
    chunks_created: usize,
    #[allow(dead_code)]
    document_id: String,
}

/// Ingest a batch of raw documents by posting each to the daemon.
pub async fn ingest_documents(
    mut docs: Vec<RawDocument>,
    state: &Arc<RwLock<AppState>>,
) -> Result<usize, OriginError> {
    if docs.is_empty() {
        return Ok(0);
    }

    // Use a single timestamp for both document metadata and activity tracking
    // so that docs always fall within the activity window they trigger.
    let now = chrono::Utc::now().timestamp();
    for doc in &mut docs {
        doc.last_modified = now;
    }

    // Bulk imports (Obsidian vaults, batch directory syncs) should not
    // emit toast notifications — they'd spam the user with dozens of popups.
    // Detected by source_agent "obsidian" (set by the obsidian connector).
    // Use all() so a mixed batch (one obsidian doc + non-obsidian docs)
    // still emits the toast for the non-obsidian ones.
    let is_bulk_import = !docs.is_empty()
        && docs
            .iter()
            .all(|d| d.source_agent.as_deref() == Some("obsidian"));

    let count = docs.len();

    // Clone the client so we can release the state lock before making HTTP calls.
    let client = {
        let mut s = state.write().await;
        s.index_status.files_total += count as u64;
        s.client.clone()
    };

    // Post each document to the daemon's ingest endpoint.
    let mut total_chunks: usize = 0;
    let mut last_error: Option<String> = None;

    for doc in docs {
        let req = IngestTextRequest {
            source: doc.source,
            source_id: doc.source_id,
            title: doc.title,
            content: doc.content,
            url: doc.url,
            metadata: if doc.metadata.is_empty() {
                None
            } else {
                Some(doc.metadata)
            },
        };

        match client
            .post_json::<IngestTextRequest, IngestResponse>("/api/ingest/text", &req)
            .await
        {
            Ok(resp) => {
                total_chunks += resp.chunks_created;
            }
            Err(e) => {
                log::error!("Failed to ingest document: {}", e);
                last_error = Some(e);
            }
        }
    }

    // Update state with results
    {
        let mut s = state.write().await;
        if last_error.is_some() {
            s.index_status.last_error = last_error.clone();
        } else {
            s.index_status.files_indexed += count as u64;
            s.index_status.last_error = None;
            s.last_ingestion_at = now;
            s.touch_activity(now);
            // Suppress the toast for bulk Obsidian imports — see
            // is_bulk_import detection above.
            if !is_bulk_import {
                s.emit_capture_event(crate::state::CaptureEvent {
                    source: "local_files".to_string(),
                    source_id: String::new(),
                    summary: format!(
                        "{} file{} indexed",
                        count,
                        if count == 1 { "" } else { "s" }
                    ),
                    chunks: total_chunks,
                    processing: false,
                });
            }
            // Space classification is handled by the daemon.
        }
        log::info!("Ingested {} documents ({} chunks)", count, total_chunks);
    }

    if let Some(err) = last_error {
        Err(OriginError::VectorDb(err))
    } else {
        Ok(total_chunks)
    }
}

/// Create a file watcher that ingests changed files into the vector database.
pub fn create_file_watcher(state: Arc<RwLock<AppState>>) -> Result<FileWatcher, OriginError> {
    let debouncer = new_debouncer(
        Duration::from_secs(2),
        None,
        move |result: DebounceEventResult| {
            let state = state.clone();
            match result {
                Ok(events) => {
                    let mut changed_paths: Vec<PathBuf> = Vec::new();

                    for event in events {
                        for path in &event.paths {
                            if path.is_file()
                                && LocalFilesSource::is_indexable(path)
                                && !changed_paths.contains(path)
                            {
                                changed_paths.push(path.clone());
                            }
                        }
                    }

                    if !changed_paths.is_empty() {
                        tauri::async_runtime::spawn(async move {
                            let docs: Vec<RawDocument> = changed_paths
                                .iter()
                                .filter_map(|p| LocalFilesSource::read_file(p).ok())
                                .collect();

                            if !docs.is_empty() {
                                if let Err(e) = ingest_documents(docs, &state).await {
                                    log::error!("File watcher ingestion error: {}", e);
                                }
                            }
                        });
                    }
                }
                Err(errors) => {
                    for e in errors {
                        log::error!("File watcher error: {}", e);
                    }
                }
            }
        },
    )
    .map_err(|e| OriginError::Indexer(e.to_string()))?;

    log::info!("File watcher created");
    Ok(debouncer)
}

/// Directories to skip when setting up file watchers (same as scan_directory).
const WATCH_SKIP_DIRS: &[&str] = &[
    ".git",
    "node_modules",
    "target",
    ".next",
    "__pycache__",
    ".venv",
    "venv",
    "dist",
    "build",
    ".cache",
    ".svn",
    ".hg",
    "vendor",
    "Pods",
    ".gradle",
    ".idea",
];

/// Add a path to an existing file watcher.
///
/// Instead of a single recursive watch (which monitors build artifacts, .git, etc.
/// and exhausts file descriptors), we walk the directory tree ourselves and skip
/// ignored directories, watching each valid directory individually.
pub fn watch_path(watcher: &mut FileWatcher, path: &Path) -> Result<(), OriginError> {
    let skip: std::collections::HashSet<&str> = WATCH_SKIP_DIRS.iter().copied().collect();
    let mut count = 0;

    watch_path_filtered(watcher, path, &skip, &mut count)?;

    log::info!("Watching {} directories under {}", count, path.display());
    Ok(())
}

fn watch_path_filtered(
    watcher: &mut FileWatcher,
    dir: &Path,
    skip: &std::collections::HashSet<&str>,
    count: &mut usize,
) -> Result<(), OriginError> {
    // Watch this directory (non-recursive)
    watcher
        .watch(dir, notify::RecursiveMode::NonRecursive)
        .map_err(|e| OriginError::Indexer(format!("Failed to watch {}: {}", dir.display(), e)))?;
    *count += 1;

    // Recurse into child directories, skipping ignored ones
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return Ok(()),
    };

    for entry in entries.flatten() {
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }

        let name = entry.file_name();
        let name_str = name.to_string_lossy();

        // Skip dotfiles/dirs and known heavy directories
        if name_str.starts_with('.') || skip.contains(name_str.as_ref()) {
            continue;
        }

        watch_path_filtered(watcher, &path, skip, count)?;
    }

    Ok(())
}

/// Sync a specific source by name.
pub async fn sync_source(
    source_name: &str,
    state: &Arc<RwLock<AppState>>,
) -> Result<usize, OriginError> {
    // Set is_running early so the UI reflects progress immediately
    {
        let mut s = state.write().await;
        s.index_status.is_running = true;
        s.index_status.last_error = None;
    }

    let result = sync_source_inner(source_name, state).await;

    // Always clear is_running and report errors
    {
        let mut s = state.write().await;
        s.index_status.is_running = false;
        if let Err(ref e) = result {
            s.index_status.last_error = Some(e.to_string());
        }
    }

    result
}

async fn sync_source_inner(
    source_name: &str,
    state: &Arc<RwLock<AppState>>,
) -> Result<usize, OriginError> {
    let docs = if source_name == "local_files" {
        // Clone watch_paths under a brief read lock, then scan without holding any lock
        let watch_paths = {
            let s = state.read().await;
            if s.watch_paths.is_empty() {
                return Ok(0);
            }
            s.watch_paths.clone()
        };

        // Blocking FS scan on a dedicated thread — no state lock held
        tokio::task::spawn_blocking(move || {
            let mut docs = Vec::new();
            for path in &watch_paths {
                let files = LocalFilesSource::scan_directory(path);
                for file_path in files {
                    match LocalFilesSource::read_file(&file_path) {
                        Ok(doc) => docs.push(doc),
                        Err(e) => {
                            log::warn!("Skipping file {}: {}", file_path.display(), e);
                        }
                    }
                }
            }
            docs
        })
        .await
        .map_err(|e| OriginError::Indexer(format!("Scan task failed: {}", e)))?
    } else {
        // For non-local sources, use write lock so fetch_updates can mutate (token refresh)
        let mut s = state.write().await;
        let source = s
            .sources
            .get_mut(source_name)
            .ok_or_else(|| OriginError::Source {
                source_name: source_name.to_string(),
                message: "Source not found".to_string(),
            })?;

        if !source.is_connected().await {
            return Err(OriginError::Source {
                source_name: source_name.to_string(),
                message: "Source is not connected".to_string(),
            });
        }

        source.fetch_updates().await?
    };

    ingest_documents(docs, state).await
}
