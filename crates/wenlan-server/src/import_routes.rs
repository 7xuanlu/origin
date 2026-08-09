// SPDX-License-Identifier: Apache-2.0
use crate::error::ServerError;
use crate::route_registry::{get, post, TrackedRouter};
use crate::state::{ServerState, SharedState};
use axum::{extract::State, response::Json};
use std::sync::Arc;
use tokio::sync::RwLock;
use uuid::Uuid;
use wenlan_types::import::{ImportChatExportRequest, ImportChatExportResponse};
use wenlan_types::requests::ImportMemoriesRequest;
use wenlan_types::responses::ImportMemoriesResponse;
use wenlan_types::WriteSpaceSource;

pub(crate) fn register(router: TrackedRouter<SharedState>) -> TrackedRouter<SharedState> {
    router
        .route("/api/import/memories", post(handle_import_memories))
        .route("/api/import/chat-export", post(handle_chat_export_import))
        .route("/api/import/state", get(handle_list_pending_imports))
}

/// POST /api/import/memories
pub async fn handle_import_memories(
    State(state): State<Arc<RwLock<ServerState>>>,
    crate::space_header::SpaceHeader(header_space): crate::space_header::SpaceHeader,
    Json(req): Json<ImportMemoriesRequest>,
) -> Result<Json<ImportMemoriesResponse>, ServerError> {
    let valid_sources = ["chatgpt", "claude", "other"];
    if !valid_sources.contains(&req.source.as_str()) {
        return Err(ServerError::ValidationError(format!(
            "Invalid source '{}'. Must be one of: chatgpt, claude, other",
            req.source
        )));
    }

    let (db, confidence_cfg) = {
        let s = state.read().await;
        (
            s.db.clone().ok_or(ServerError::DbNotInitialized)?,
            s.tuning.confidence.clone(),
        )
    };
    let _space_write_guard = db.lock_space_writes().await;
    let resolved = db
        .resolve_write_space(&req.space, header_space.as_deref())
        .await?;
    let result = wenlan_core::importer::import_memories_no_llm_in_space(
        &db,
        &req.content,
        &req.source,
        req.label.as_deref(),
        &confidence_cfg,
        &resolved,
    )
    .await?;
    let persisted = db.finalize_write_space(&resolved).await?;
    let space_source = if persisted.space_name.is_some() {
        persisted.source
    } else {
        WriteSpaceSource::Uncategorized
    };

    Ok(Json(ImportMemoriesResponse {
        imported: result.imported,
        skipped: result.skipped,
        breakdown: result.breakdown,
        entities_created: result.entities_created,
        observations_added: result.observations_added,
        relations_created: result.relations_created,
        batch_id: result.batch_id,
        space: persisted.space_name,
        space_source: Some(space_source),
    }))
}

// ---------------------------------------------------------------------------
// POST /api/import/chat-export
// ---------------------------------------------------------------------------

/// Maximum ZIP file size we accept (512 MB).
const MAX_IMPORT_ZIP_SIZE: u64 = 512 * 1024 * 1024;

/// POST /api/import/chat-export
///
/// Reads a chat-export ZIP from disk, auto-detects the vendor, bulk-ingests
/// all new conversations as raw memories. Automatic enrichment is admitted
/// later by the global ambient scheduler.
pub async fn handle_chat_export_import(
    State(state): State<Arc<RwLock<ServerState>>>,
    Json(req): Json<ImportChatExportRequest>,
) -> Result<Json<ImportChatExportResponse>, ServerError> {
    // 1. Validate file size before reading (I4).
    let meta = tokio::fs::metadata(&req.path)
        .await
        .map_err(|e| ServerError::ChatImport(format!("failed to stat {}: {e}", req.path)))?;
    if meta.len() > MAX_IMPORT_ZIP_SIZE {
        return Err(ServerError::ChatImport(format!(
            "file too large ({} bytes, max {})",
            meta.len(),
            MAX_IMPORT_ZIP_SIZE
        )));
    }

    // 2. Read ZIP from disk (async I/O to avoid blocking Tokio worker — I1).
    let bytes = tokio::fs::read(&req.path)
        .await
        .map_err(|e| ServerError::ChatImport(format!("failed to read {}: {e}", req.path)))?;

    // 3. Dispatch parse — auto-detect vendor.
    let batch = wenlan_core::chat_import::dispatch_parse(&bytes)
        .map_err(|e| ServerError::ChatImport(format!("parse failed: {e}")))?;

    // 4. Snapshot Arc<MemoryDB> out of the RwLock guard before any awaits.
    let db = {
        let guard = state.read().await;
        guard
            .db
            .as_ref()
            .ok_or_else(|| ServerError::ChatImport("database not initialized".into()))?
            .clone()
    };

    // 5. Create import_state row.
    let import_id = format!("imp_{}", Uuid::new_v4());
    db.start_import_state(&import_id, batch.vendor, &req.path)
        .await
        .map_err(|e| ServerError::ChatImport(format!("start_import_state: {e}")))?;
    db.update_import_state_stage(
        &import_id,
        wenlan_core::chat_import::bulk_ingest::ImportStage::StageA,
        Some(batch.conversations.len() as i64),
        Some(0),
    )
    .await
    .map_err(|e| ServerError::ChatImport(format!("update_import_state_stage: {e}")))?;

    // 6. Bulk ingest conversations. On failure, mark import_state as Error (I3).
    // TODO: Replace NoopEmitter with a broadcast-channel emitter that forwards
    // progress events to /ws/updates so the Tauri app can display live progress.
    let emitter: Arc<dyn wenlan_core::events::EventEmitter> =
        Arc::new(wenlan_core::events::NoopEmitter);
    let result = match wenlan_core::chat_import::bulk_ingest::bulk_import_conversations(
        db.clone(),
        &batch.conversations,
        emitter,
        &import_id,
    )
    .await
    {
        Ok(r) => r,
        Err(e) => {
            let msg = format!("bulk ingest: {e}");
            let _ = db
                .update_import_state_stage_with_error(
                    &import_id,
                    wenlan_core::chat_import::bulk_ingest::ImportStage::Error,
                    None,
                    None,
                    Some(&msg),
                )
                .await;
            return Err(ServerError::ChatImport(msg));
        }
    };

    // 7. The import is complete once its durable batch is searchable. Automatic
    //    classification/extraction remains a global ambient backlog represented
    //    by NULL fields plus versioned enrichment receipts; import_state does
    //    not attempt to track that unrelated, potentially multi-day work.
    db.update_import_state_stage(
        &import_id,
        wenlan_core::chat_import::bulk_ingest::ImportStage::Done,
        None,
        Some(result.conversations_ingested as i64),
    )
    .await
    .map_err(|e| ServerError::ChatImport(format!("update_import_state_stage: {e}")))?;

    let vendor_str = batch.vendor.as_str().to_string();
    Ok(Json(ImportChatExportResponse {
        import_id,
        vendor: vendor_str,
        conversations_total: batch.conversations.len(),
        conversations_new: result.conversations_ingested,
        conversations_skipped_existing: result.conversations_skipped_existing,
        memories_stored: result.memories_stored,
    }))
}

// ---------------------------------------------------------------------------
// GET /api/import/state
// ---------------------------------------------------------------------------

/// GET /api/import/state — list non-terminal imports (pending/in-progress).
pub async fn handle_list_pending_imports(
    State(state): State<Arc<RwLock<ServerState>>>,
) -> Result<Json<Vec<wenlan_types::import::PendingImport>>, ServerError> {
    let db = {
        let guard = state.read().await;
        guard
            .db
            .as_ref()
            .ok_or_else(|| ServerError::ChatImport("database not initialized".into()))?
            .clone()
    };
    let rows = db
        .list_pending_imports()
        .await
        .map_err(|e| ServerError::ChatImport(format!("list_pending_imports: {e}")))?;
    Ok(Json(
        rows.into_iter()
            .map(|s| wenlan_types::import::PendingImport {
                id: s.id,
                vendor: s.vendor.as_str().to_string(),
                stage: s.stage.as_str().to_string(),
                source_path: s.source_path,
                processed_conversations: s.processed_conversations,
                total_conversations: s.total_conversations,
            })
            .collect(),
    ))
}

#[cfg(test)]
mod chat_export_route_tests {
    use async_trait::async_trait;
    use axum::body::Body;
    use axum::http::Request;
    use std::io::Write;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use tower::ServiceExt;

    use crate::state::ServerState;
    use std::sync::Arc;
    use tokio::sync::{Notify, RwLock};

    struct BlockingProvider {
        calls: AtomicUsize,
        called: Notify,
        release: Notify,
    }

    #[async_trait]
    impl wenlan_core::llm_provider::LlmProvider for BlockingProvider {
        async fn generate(
            &self,
            _request: wenlan_core::llm_provider::LlmRequest,
        ) -> Result<String, wenlan_core::llm_provider::LlmError> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            self.called.notify_one();
            self.release.notified().await;
            Ok("{}".to_string())
        }

        fn is_available(&self) -> bool {
            true
        }

        fn name(&self) -> &str {
            "import-handoff-test"
        }

        fn backend(&self) -> wenlan_core::llm_provider::LlmBackend {
            wenlan_core::llm_provider::LlmBackend::Api
        }

        fn kind(&self) -> &'static str {
            "mock"
        }
    }

    fn write_claude_export(path: &std::path::Path) {
        const CONVERSATIONS: &str = r#"[
          {
            "uuid": "route-conv-1",
            "name": "Scheduler handoff",
            "summary": "A short summary",
            "created_at": "2026-04-01T10:00:00.000Z",
            "updated_at": "2026-04-01T10:05:00.000Z",
            "account": {"uuid": "acc-1"},
            "chat_messages": [
              {
                "uuid": "msg-1",
                "text": "How should enrichment run?",
                "content": [{"type": "text", "text": "How should enrichment run?"}],
                "sender": "human",
                "created_at": "2026-04-01T10:00:00.000Z",
                "updated_at": "2026-04-01T10:00:00.000Z",
                "attachments": [],
                "files": [],
                "parent_message_uuid": "00000000-0000-4000-8000-000000000000"
              },
              {
                "uuid": "msg-2",
                "text": "Only through the ambient scheduler.",
                "content": [{"type": "text", "text": "Only through the ambient scheduler."}],
                "sender": "assistant",
                "created_at": "2026-04-01T10:00:05.000Z",
                "updated_at": "2026-04-01T10:00:05.000Z",
                "attachments": [],
                "files": [],
                "parent_message_uuid": "msg-1"
              }
            ]
          }
        ]"#;
        let file = std::fs::File::create(path).unwrap();
        let mut zip = zip::ZipWriter::new(file);
        let options: zip::write::FileOptions<()> =
            zip::write::FileOptions::default().compression_method(zip::CompressionMethod::Stored);
        zip.start_file("users.json", options).unwrap();
        zip.write_all(b"[]").unwrap();
        zip.start_file("conversations.json", options).unwrap();
        zip.write_all(CONVERSATIONS.as_bytes()).unwrap();
        zip.finish().unwrap();
    }

    #[tokio::test]
    async fn post_chat_export_with_missing_file_returns_400() {
        let state = Arc::new(RwLock::new(ServerState::default()));
        let app = crate::router::build_router(state);
        let body = serde_json::json!({ "path": "/tmp/definitely-not-a-real-zip-xyz.zip" });
        let req = Request::builder()
            .method("POST")
            .uri("/api/import/chat-export")
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_vec(&body).unwrap()))
            .unwrap();
        let response = app.oneshot(req).await.unwrap();
        assert_eq!(response.status(), 400);
    }

    #[tokio::test]
    async fn chat_import_finishes_ingest_without_starting_enrichment() {
        let dir = tempfile::tempdir().unwrap();
        let archive_path = dir.path().join("claude-export.zip");
        write_claude_export(&archive_path);
        let db = Arc::new(
            wenlan_core::db::MemoryDB::new(
                &dir.path().join("db"),
                Arc::new(wenlan_core::events::NoopEmitter),
            )
            .await
            .unwrap(),
        );
        let provider = Arc::new(BlockingProvider {
            calls: AtomicUsize::new(0),
            called: Notify::new(),
            release: Notify::new(),
        });
        let state = Arc::new(RwLock::new(ServerState {
            db: Some(db.clone()),
            llm: Some(provider.clone()),
            ..Default::default()
        }));

        let response = super::handle_chat_export_import(
            axum::extract::State(state),
            axum::Json(wenlan_types::import::ImportChatExportRequest {
                path: archive_path.to_string_lossy().into_owned(),
            }),
        )
        .await
        .unwrap()
        .0;

        let import = db
            .load_import_state(&response.import_id)
            .await
            .unwrap()
            .expect("import state");
        assert_eq!(
            import.stage,
            wenlan_core::chat_import::bulk_ingest::ImportStage::Done
        );
        assert!(db.list_pending_imports().await.unwrap().is_empty());
        assert!(db.get_classification_candidate(3).await.unwrap().is_some());
        assert!(
            tokio::time::timeout(
                std::time::Duration::from_millis(100),
                provider.called.notified()
            )
            .await
            .is_err(),
            "chat import must leave enrichment to the ambient scheduler"
        );
        assert_eq!(provider.calls.load(Ordering::SeqCst), 0);
    }
}
