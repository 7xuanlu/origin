// SPDX-License-Identifier: Apache-2.0
use crate::error::ServerError;
use crate::state::ServerState;
use axum::{extract::State, response::Json};
use origin_types::import::{ImportChatExportRequest, ImportChatExportResponse};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use uuid::Uuid;

#[derive(Debug, Deserialize)]
pub struct ImportMemoriesRequest {
    pub source: String,
    pub content: String,
    #[serde(default)]
    pub label: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct ImportMemoriesResponse {
    pub imported: usize,
    pub skipped: usize,
    pub breakdown: HashMap<String, usize>,
    pub entities_created: usize,
    pub observations_added: usize,
    pub relations_created: usize,
    pub batch_id: String,
}

/// POST /api/import/memories
pub async fn handle_import_memories(
    State(state): State<Arc<RwLock<ServerState>>>,
    Json(req): Json<ImportMemoriesRequest>,
) -> Result<Json<ImportMemoriesResponse>, ServerError> {
    let valid_sources = ["chatgpt", "claude", "other"];
    if !valid_sources.contains(&req.source.as_str()) {
        return Err(ServerError::ValidationError(format!(
            "Invalid source '{}'. Must be one of: chatgpt, claude, other",
            req.source
        )));
    }

    let result = {
        let s = state.read().await;
        let db = s.db.as_ref().ok_or(ServerError::DbNotInitialized)?;
        let confidence_cfg = &s.tuning.confidence;
        origin_core::importer::import_memories_no_llm(
            db,
            &req.content,
            &req.source,
            req.label.as_deref(),
            confidence_cfg,
        )
        .await
        .map_err(|e| ServerError::Internal(e.to_string()))?
    };

    Ok(Json(ImportMemoriesResponse {
        imported: result.imported,
        skipped: result.skipped,
        breakdown: result.breakdown,
        entities_created: result.entities_created,
        observations_added: result.observations_added,
        relations_created: result.relations_created,
        batch_id: result.batch_id,
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
/// all new conversations as raw memories, and spawns a background refinery
/// steep to classify + enrich them.
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
    let batch = origin_core::chat_import::dispatch_parse(&bytes)
        .map_err(|e| ServerError::ChatImport(format!("parse failed: {e}")))?;

    // 4. Snapshot Arc<MemoryDB> + LLM providers out of the RwLock guard before any awaits.
    let (db, llm, api_llm, synthesis_llm) = {
        let guard = state.read().await;
        let db = guard
            .db
            .as_ref()
            .ok_or_else(|| ServerError::ChatImport("database not initialized".into()))?
            .clone();
        (
            db,
            guard.llm.clone(),
            guard.api_llm.clone(),
            guard.synthesis_llm.clone(),
        )
    };

    // 5. Create import_state row.
    let import_id = format!("imp_{}", Uuid::new_v4());
    db.start_import_state(&import_id, batch.vendor, &req.path)
        .await
        .map_err(|e| ServerError::ChatImport(format!("start_import_state: {e}")))?;
    db.update_import_state_stage(
        &import_id,
        origin_core::chat_import::bulk_ingest::ImportStage::StageA,
        Some(batch.conversations.len() as i64),
        Some(0),
    )
    .await
    .map_err(|e| ServerError::ChatImport(format!("update_import_state_stage: {e}")))?;

    // 6. Bulk ingest conversations. On failure, mark import_state as Error (I3).
    // TODO: Replace NoopEmitter with a broadcast-channel emitter that forwards
    // progress events to /ws/updates so the Tauri app can display live progress.
    let emitter: Arc<dyn origin_core::events::EventEmitter> =
        Arc::new(origin_core::events::NoopEmitter);
    let result = match origin_core::chat_import::bulk_ingest::bulk_import_conversations(
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
                    origin_core::chat_import::bulk_ingest::ImportStage::Error,
                    None,
                    None,
                    Some(&msg),
                )
                .await;
            return Err(ServerError::ChatImport(msg));
        }
    };

    // 7. Mark StageB and spawn background steep.
    db.update_import_state_stage(
        &import_id,
        origin_core::chat_import::bulk_ingest::ImportStage::StageB,
        None,
        Some(result.conversations_ingested as i64),
    )
    .await
    .map_err(|e| ServerError::ChatImport(format!("update_import_state_stage: {e}")))?;

    // Spawn background refinery loop that keeps steeping until all imported
    // memories are classified. Only then does import_state transition to Done.
    // This keeps the import visible via GET /api/import/state while the
    // refinery is still working through the batch.
    let db_for_steep = db.clone();
    let import_id_for_steep = import_id.clone();
    tokio::spawn(async move {
        loop {
            let steep_result = origin_core::refinery::trigger_steep_now(
                db_for_steep.clone(),
                llm.as_ref(),
                api_llm.as_ref(),
                synthesis_llm.as_ref(),
            )
            .await;
            if let Err(e) = steep_result {
                let msg = format!("steep failed: {e}");
                tracing::error!("[chat-import] {msg}");
                let _ = db_for_steep
                    .update_import_state_stage_with_error(
                        &import_id_for_steep,
                        origin_core::chat_import::bulk_ingest::ImportStage::Error,
                        None,
                        None,
                        Some(&msg),
                    )
                    .await;
                return;
            }
            // Check if all import memories are classified.
            let remaining = db_for_steep.count_unclassified_imports().await.unwrap_or(0);
            if remaining == 0 {
                let _ = db_for_steep
                    .update_import_state_stage(
                        &import_id_for_steep,
                        origin_core::chat_import::bulk_ingest::ImportStage::Done,
                        None,
                        None,
                    )
                    .await;
                return;
            }
            // Brief pause between steep cycles to avoid busy-looping.
            tokio::time::sleep(std::time::Duration::from_secs(2)).await;
        }
    });

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
) -> Result<Json<Vec<origin_types::import::PendingImport>>, ServerError> {
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
            .map(|s| origin_types::import::PendingImport {
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
    use axum::body::Body;
    use axum::http::Request;
    use tower::ServiceExt;

    use crate::state::ServerState;
    use std::sync::Arc;
    use tokio::sync::RwLock;

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
}
