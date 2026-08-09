// SPDX-License-Identifier: Apache-2.0
use crate::error::ServerError;
use crate::route_registry::{get, post, TrackedRouter};
use crate::state::{ServerState, SharedState};
use axum::{
    extract::{Path, State},
    response::Json,
};
use serde::Deserialize;
use std::sync::Arc;
use tokio::sync::RwLock;

pub(crate) fn register(router: TrackedRouter<SharedState>) -> TrackedRouter<SharedState> {
    router
        .route("/api/snapshots", get(handle_list_snapshots))
        .route(
            "/api/snapshots/{id}/captures",
            get(handle_get_snapshot_captures),
        )
        .route(
            "/api/snapshots/{id}/captures-with-content",
            get(handle_get_snapshot_captures_with_content),
        )
        .route("/api/snapshots/{id}/delete", post(handle_delete_snapshot))
}

/// POST /api/snapshots/{id}/delete
pub async fn handle_delete_snapshot(
    State(state): State<Arc<RwLock<ServerState>>>,
    Path(id): Path<String>,
) -> Result<Json<wenlan_types::responses::SuccessResponse>, ServerError> {
    let db = {
        let s = state.read().await;
        s.db.clone().ok_or(ServerError::DbNotInitialized)?
    };
    db.delete_snapshot(&id)
        .await
        .map_err(|e| ServerError::Internal(e.to_string()))?;
    Ok(Json(wenlan_types::responses::SuccessResponse { ok: true }))
}

// ===== Session Snapshots =====

#[derive(Debug, Deserialize)]
pub struct SnapshotsQuery {
    #[serde(default = "default_snapshots_limit")]
    pub limit: usize,
}

fn default_snapshots_limit() -> usize {
    10
}

/// GET /api/snapshots?limit=N
///
/// Returns the N most recent session snapshots (default 10).
pub async fn handle_list_snapshots(
    State(state): State<Arc<RwLock<ServerState>>>,
    axum::extract::Query(query): axum::extract::Query<SnapshotsQuery>,
) -> Result<Json<Vec<wenlan_types::SessionSnapshot>>, ServerError> {
    let db = {
        let s = state.read().await;
        s.db.clone().ok_or(ServerError::DbNotInitialized)?
    };
    let rows = db
        .get_recent_snapshots(query.limit)
        .await
        .map_err(|e| ServerError::Internal(e.to_string()))?;
    let snapshots = rows
        .into_iter()
        .map(|r| wenlan_types::SessionSnapshot {
            id: r.id,
            activity_id: r.activity_id,
            started_at: r.started_at,
            ended_at: r.ended_at,
            primary_apps: r.primary_apps,
            summary: r.summary,
            tags: r.tags,
            capture_count: r.capture_count as u64,
        })
        .collect();
    Ok(Json(snapshots))
}

/// GET /api/snapshots/{id}/captures
///
/// Returns capture metadata (no full text) for a snapshot.
pub async fn handle_get_snapshot_captures(
    State(state): State<Arc<RwLock<ServerState>>>,
    Path(id): Path<String>,
    crate::space_header::SpaceHeader(header_space): crate::space_header::SpaceHeader,
) -> Result<Json<Vec<wenlan_types::SnapshotCapture>>, ServerError> {
    let db = {
        let s = state.read().await;
        s.db.clone().ok_or(ServerError::DbNotInitialized)?
    };
    let scope = crate::read_scope::effective_read_scope(&db, None, header_space.as_deref()).await?;
    let rows = db.get_captures_for_snapshot_scoped(&id, &scope).await?;
    let captures = rows
        .into_iter()
        .map(|c| wenlan_types::SnapshotCapture {
            source_id: c.source_id,
            app_name: c.app_name,
            window_title: c.window_title,
            timestamp: c.timestamp,
            source: c.source,
        })
        .collect();
    Ok(Json(captures))
}

/// GET /api/snapshots/{id}/captures-with-content
///
/// Returns captures for a snapshot plus their full chunked text and LLM
/// summary.
pub async fn handle_get_snapshot_captures_with_content(
    State(state): State<Arc<RwLock<ServerState>>>,
    Path(id): Path<String>,
    crate::space_header::SpaceHeader(header_space): crate::space_header::SpaceHeader,
) -> Result<Json<Vec<wenlan_types::SnapshotCaptureWithContent>>, ServerError> {
    let db = {
        let s = state.read().await;
        s.db.clone().ok_or(ServerError::DbNotInitialized)?
    };
    let scope = crate::read_scope::effective_read_scope(&db, None, header_space.as_deref()).await?;
    let captures = db
        .get_snapshot_captures_with_content_scoped(&id, &scope)
        .await?;
    Ok(Json(captures))
}
