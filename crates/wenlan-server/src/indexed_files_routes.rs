// SPDX-License-Identifier: Apache-2.0
use crate::error::ServerError;
use crate::route_registry::{delete, get, post, put, TrackedRouter};
use crate::state::{ServerState, SharedState};
use axum::{
    extract::{Path, State},
    response::Json,
};
use std::sync::Arc;
use tokio::sync::RwLock;

pub(crate) fn register(router: TrackedRouter<SharedState>) -> TrackedRouter<SharedState> {
    router
        .route("/api/indexed-files", get(handle_list_indexed_files))
        .route("/api/chunks/{source_id}", get(handle_get_chunks))
        .route("/api/chunks/{id}/update", put(handle_update_chunk))
        .route(
            "/api/chunks/time-range",
            delete(handle_delete_by_time_range),
        )
        .route("/api/chunks/delete-bulk", post(handle_delete_bulk))
}

// =====================================================================
// Batch 2 — Indexed files / chunks
// =====================================================================

/// GET /api/indexed-files
pub async fn handle_list_indexed_files(
    State(state): State<Arc<RwLock<ServerState>>>,
    crate::space_header::SpaceHeader(header_space): crate::space_header::SpaceHeader,
) -> Result<Json<wenlan_types::responses::IndexedFilesResponse>, ServerError> {
    let db = {
        let s = state.read().await;
        s.db.clone().ok_or(ServerError::DbNotInitialized)?
    };
    let scope = crate::read_scope::effective_read_scope(&db, None, header_space.as_deref()).await?;
    let files = db
        .list_indexed_files_scoped(&scope)
        .await
        .map_err(|e| ServerError::Internal(e.to_string()))?;
    Ok(Json(wenlan_types::responses::IndexedFilesResponse {
        files,
    }))
}

/// GET /api/chunks/{source_id}
pub async fn handle_get_chunks(
    State(state): State<Arc<RwLock<ServerState>>>,
    crate::space_header::SpaceHeader(header_space): crate::space_header::SpaceHeader,
    Path(source_id): Path<String>,
) -> Result<Json<Vec<wenlan_core::db::MemoryDetail>>, ServerError> {
    let db = {
        let s = state.read().await;
        s.db.clone().ok_or(ServerError::DbNotInitialized)?
    };
    let scope = crate::read_scope::effective_read_scope(&db, None, header_space.as_deref()).await?;
    let chunks = db
        .get_chunks_scoped(&source_id, &scope)
        .await
        .map_err(|e| ServerError::Internal(e.to_string()))?
        .ok_or_else(|| ServerError::NotFound("memory not found".to_string()))?;
    Ok(Json(chunks))
}

/// PUT /api/chunks/{id}/update
pub async fn handle_update_chunk(
    State(state): State<Arc<RwLock<ServerState>>>,
    Path(id): Path<String>,
    Json(req): Json<wenlan_types::requests::UpdateChunkRequest>,
) -> Result<Json<wenlan_types::responses::SuccessResponse>, ServerError> {
    let db = {
        let s = state.read().await;
        s.db.clone().ok_or(ServerError::DbNotInitialized)?
    };
    db.update_memory(&id, &req.content)
        .await
        .map_err(|e| ServerError::Internal(e.to_string()))?;
    Ok(Json(wenlan_types::responses::SuccessResponse { ok: true }))
}

/// DELETE /api/chunks/time-range
pub async fn handle_delete_by_time_range(
    State(state): State<Arc<RwLock<ServerState>>>,
    Json(req): Json<wenlan_types::requests::DeleteByTimeRangeRequest>,
) -> Result<Json<wenlan_types::responses::DeleteCountResponse>, ServerError> {
    let db = {
        let s = state.read().await;
        s.db.clone().ok_or(ServerError::DbNotInitialized)?
    };
    let deleted = db
        .delete_by_time_range(req.start, req.end)
        .await
        .map_err(|e| ServerError::Internal(e.to_string()))?;
    Ok(Json(wenlan_types::responses::DeleteCountResponse {
        deleted,
    }))
}

/// POST /api/chunks/delete-bulk
pub async fn handle_delete_bulk(
    State(state): State<Arc<RwLock<ServerState>>>,
    Json(req): Json<wenlan_types::requests::BulkDeleteRequest>,
) -> Result<Json<wenlan_types::responses::DeleteCountResponse>, ServerError> {
    let db = {
        let s = state.read().await;
        s.db.clone().ok_or(ServerError::DbNotInitialized)?
    };
    let mut deleted = 0usize;
    for item in &req.items {
        if db
            .delete_by_source_id(&item.source, &item.source_id)
            .await
            .is_ok()
        {
            deleted += 1;
        }
    }
    Ok(Json(wenlan_types::responses::DeleteCountResponse {
        deleted,
    }))
}
