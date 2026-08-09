// SPDX-License-Identifier: Apache-2.0
use crate::error::ServerError;
use crate::route_registry::{get, TrackedRouter};
use crate::state::{ServerState, SharedState};
use axum::{
    extract::{Path, State},
    response::Json,
};
use serde::Deserialize;
use std::sync::Arc;
use tokio::sync::RwLock;

pub(crate) fn register_history(router: TrackedRouter<SharedState>) -> TrackedRouter<SharedState> {
    router.route(
        "/api/memory/{id}/revisions",
        get(handle_get_memory_revisions),
    )
}

pub(crate) fn register_pending(router: TrackedRouter<SharedState>) -> TrackedRouter<SharedState> {
    router
        .route(
            "/api/memory/pending-revisions",
            get(handle_list_pending_revisions),
        )
        .route(
            "/api/memory/pending-revision/{source_id}",
            get(handle_get_pending_revision),
        )
}

#[derive(Debug, Deserialize)]
pub struct PendingRevisionsQuery {
    #[serde(default = "default_pending_revisions_limit")]
    pub limit: usize,
}

fn default_pending_revisions_limit() -> usize {
    50
}

/// GET /api/memory/pending-revisions?limit=N
pub async fn handle_list_pending_revisions(
    State(state): State<Arc<RwLock<ServerState>>>,
    crate::space_header::SpaceHeader(header_space): crate::space_header::SpaceHeader,
    axum::extract::Query(q): axum::extract::Query<PendingRevisionsQuery>,
) -> Result<Json<Vec<wenlan_types::responses::PendingRevisionItem>>, ServerError> {
    let db = {
        let s = state.read().await;
        s.db.clone().ok_or(ServerError::DbNotInitialized)?
    };
    let scope = crate::read_scope::effective_read_scope(&db, None, header_space.as_deref()).await?;
    let limit = q.limit.clamp(1, 500);
    let items = db
        .list_pending_revisions_scoped(limit, &scope)
        .await
        .map_err(|e| ServerError::Internal(e.to_string()))?;
    Ok(Json(items))
}

/// GET /api/memory/pending-revision/{source_id}
pub async fn handle_get_pending_revision(
    State(state): State<Arc<RwLock<ServerState>>>,
    crate::space_header::SpaceHeader(header_space): crate::space_header::SpaceHeader,
    Path(source_id): Path<String>,
) -> Result<Json<Option<wenlan_core::db::PendingRevision>>, ServerError> {
    let db = {
        let s = state.read().await;
        s.db.clone().ok_or(ServerError::DbNotInitialized)?
    };
    let scope = crate::read_scope::effective_read_scope(&db, None, header_space.as_deref()).await?;
    let revision = db
        .get_pending_revision_for_scoped(&source_id, &scope)
        .await
        .map_err(|e| ServerError::Internal(e.to_string()))?
        .ok_or_else(|| ServerError::NotFound("memory not found".to_string()))?;
    Ok(Json(Some(revision)))
}

/// GET /api/memory/{id}/revisions
///
/// Walk the supersede chain for a memory and return all revision entries.
pub async fn handle_get_memory_revisions(
    State(state): State<Arc<RwLock<ServerState>>>,
    crate::space_header::SpaceHeader(header_space): crate::space_header::SpaceHeader,
    Path(id): Path<String>,
) -> Result<Json<wenlan_types::responses::ListMemoryRevisionsResponse>, ServerError> {
    let db = {
        let s = state.read().await;
        s.db.clone().ok_or(ServerError::DbNotInitialized)?
    };
    let scope = crate::read_scope::effective_read_scope(&db, None, header_space.as_deref()).await?;
    let entries = db
        .walk_supersede_chain_scoped(&id, &scope)
        .await
        .map_err(|e| ServerError::Internal(e.to_string()))?
        .ok_or_else(|| ServerError::NotFound("memory not found".to_string()))?;
    let chain_depth = entries.last().map(|e| e.depth).unwrap_or(0);
    Ok(Json(wenlan_types::responses::ListMemoryRevisionsResponse {
        current_source_id: id,
        chain_depth,
        entries,
    }))
}
