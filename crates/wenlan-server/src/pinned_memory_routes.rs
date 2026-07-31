// SPDX-License-Identifier: Apache-2.0
use crate::error::ServerError;
use crate::route_registry::{get, post, TrackedRouter};
use crate::state::{ServerState, SharedState};
use axum::{
    extract::{Path, State},
    response::Json,
};
use std::sync::Arc;
use tokio::sync::RwLock;

pub(crate) fn register(router: TrackedRouter<SharedState>) -> TrackedRouter<SharedState> {
    router
        .route("/api/memory/pinned", get(handle_list_pinned_memories))
        .route("/api/memory/{id}/pin", post(handle_pin_memory))
        .route("/api/memory/{id}/unpin", post(handle_unpin_memory))
}

/// GET /api/memory/pinned
pub async fn handle_list_pinned_memories(
    State(state): State<Arc<RwLock<ServerState>>>,
    crate::space_header::SpaceHeader(header_space): crate::space_header::SpaceHeader,
) -> Result<Json<wenlan_types::responses::PinnedMemoriesResponse>, ServerError> {
    let db = {
        let s = state.read().await;
        s.db.clone().ok_or(ServerError::DbNotInitialized)?
    };
    let scope = crate::read_scope::effective_read_scope(&db, None, header_space.as_deref()).await?;
    let memories = db
        .list_pinned_memories_scoped(&scope)
        .await
        .map_err(|e| ServerError::Internal(e.to_string()))?;
    Ok(Json(wenlan_types::responses::PinnedMemoriesResponse {
        memories,
    }))
}

/// POST /api/memory/{id}/pin
pub async fn handle_pin_memory(
    State(state): State<Arc<RwLock<ServerState>>>,
    Path(id): Path<String>,
) -> Result<Json<wenlan_types::responses::SuccessResponse>, ServerError> {
    let db = {
        let s = state.read().await;
        s.db.clone().ok_or(ServerError::DbNotInitialized)?
    };
    db.pin_memory(&id)
        .await
        .map_err(|e| ServerError::Internal(e.to_string()))?;
    Ok(Json(wenlan_types::responses::SuccessResponse { ok: true }))
}

/// POST /api/memory/{id}/unpin
pub async fn handle_unpin_memory(
    State(state): State<Arc<RwLock<ServerState>>>,
    Path(id): Path<String>,
) -> Result<Json<wenlan_types::responses::SuccessResponse>, ServerError> {
    let db = {
        let s = state.read().await;
        s.db.clone().ok_or(ServerError::DbNotInitialized)?
    };
    db.unpin_memory(&id)
        .await
        .map_err(|e| ServerError::Internal(e.to_string()))?;
    Ok(Json(wenlan_types::responses::SuccessResponse { ok: true }))
}
