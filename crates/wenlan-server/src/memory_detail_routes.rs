// SPDX-License-Identifier: Apache-2.0
use crate::error::ServerError;
use crate::route_registry::{get, TrackedRouter};
use crate::state::{ServerState, SharedState};
use axum::{
    extract::{Path, State},
    response::Json,
};
use std::sync::Arc;
use tokio::sync::RwLock;

pub(crate) fn register(router: TrackedRouter<SharedState>) -> TrackedRouter<SharedState> {
    router
        .route("/api/capture-stats", get(handle_capture_stats))
        .route("/api/memory/{id}/detail", get(handle_get_memory_detail))
        .route("/api/memory/by-ids", get(handle_get_memories_by_ids))
}

// =====================================================================
// Batch 5 — Capture stats and memory detail
// =====================================================================

/// GET /api/capture-stats
pub async fn handle_capture_stats(
    State(state): State<Arc<RwLock<ServerState>>>,
) -> Result<Json<serde_json::Value>, ServerError> {
    let db = {
        let s = state.read().await;
        s.db.clone().ok_or(ServerError::DbNotInitialized)?
    };
    let count = db.count().await.unwrap_or(0);
    Ok(Json(serde_json::json!({
        "total_chunks": count,
    })))
}

/// GET /api/memory/{id}/detail
pub async fn handle_get_memory_detail(
    State(state): State<Arc<RwLock<ServerState>>>,
    crate::space_header::SpaceHeader(header_space): crate::space_header::SpaceHeader,
    Path(id): Path<String>,
) -> Result<Json<wenlan_types::responses::MemoryDetailResponse>, ServerError> {
    let db = {
        let s = state.read().await;
        s.db.clone().ok_or(ServerError::DbNotInitialized)?
    };
    let scope = crate::read_scope::effective_read_scope(&db, None, header_space.as_deref()).await?;
    let memory = db
        .get_memory_detail_scoped(&id, &scope)
        .await
        .map_err(|e| ServerError::Internal(e.to_string()))?
        .ok_or_else(|| ServerError::NotFound("memory not found".to_string()))?;
    Ok(Json(wenlan_types::responses::MemoryDetailResponse {
        memory: Some(memory),
    }))
}

/// GET /api/memory/by-ids?ids=mem_a,mem_b,...
///
/// Batch-fetch multiple memories by source_id in a single round trip.
/// The response preserves input order; missing ids are silently omitted.
/// Used by ConceptDetail to load all source memories at once.
pub async fn handle_get_memories_by_ids(
    State(state): State<Arc<RwLock<ServerState>>>,
    crate::space_header::SpaceHeader(header_space): crate::space_header::SpaceHeader,
    axum::extract::Query(params): axum::extract::Query<std::collections::HashMap<String, String>>,
) -> Result<Json<wenlan_types::responses::PinnedMemoriesResponse>, ServerError> {
    let db = {
        let s = state.read().await;
        s.db.clone().ok_or(ServerError::DbNotInitialized)?
    };
    let scope = crate::read_scope::effective_read_scope(&db, None, header_space.as_deref()).await?;
    let ids: Vec<String> = params
        .get("ids")
        .map(|s| {
            s.split(',')
                .filter(|p| !p.is_empty())
                .map(str::to_string)
                .collect()
        })
        .unwrap_or_default();
    let memories = db
        .get_memories_by_source_ids_scoped(&ids, &scope)
        .await
        .map_err(|e| ServerError::Internal(e.to_string()))?;
    Ok(Json(wenlan_types::responses::PinnedMemoriesResponse {
        memories,
    }))
}
