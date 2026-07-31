// SPDX-License-Identifier: Apache-2.0
use crate::error::ServerError;
use crate::route_registry::{get, post, put, TrackedRouter};
use crate::state::{ServerState, SharedState};
use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::Json,
};
use serde::Deserialize;
use std::sync::Arc;
use tokio::sync::RwLock;
use wenlan_types::requests::SetDefaultSpaceRequest;
use wenlan_types::responses::DefaultSpaceResponse;

pub(crate) async fn registered_request_space(
    db: &wenlan_core::db::MemoryDB,
    requested: &Option<String>,
    context: &str,
) -> Result<Option<String>, ServerError> {
    let proposed = requested
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty());
    let registered = db
        .registered_space_or_none(requested.as_deref())
        .await
        .map_err(|e| ServerError::Internal(e.to_string()))?;
    if registered.is_none() {
        if let Some(space) = proposed {
            tracing::warn!(
                "[memory] ignoring unregistered space {:?} for {}; using unscoped fallback",
                space,
                context
            );
        }
    }
    Ok(registered)
}

pub(crate) fn register_core(router: TrackedRouter<SharedState>) -> TrackedRouter<SharedState> {
    router
        .route(
            "/api/spaces",
            get(handle_list_spaces).post(handle_create_space),
        )
        .route(
            "/api/spaces/default",
            get(handle_get_default_space)
                .put(handle_set_default_space)
                .delete(handle_clear_default_space),
        )
        .route(
            "/api/spaces/{name}",
            put(handle_update_space).delete(handle_delete_space),
        )
        .route("/api/spaces/{from}/move-to/{to}", post(handle_move_space))
}

// ===== Space CRUD Handlers =====

#[derive(Debug, Deserialize)]
pub struct CreateSpaceRequest {
    pub name: String,
    pub description: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct UpdateSpaceRequest {
    pub new_name: Option<String>,
    pub description: Option<String>,
}

pub async fn handle_get_default_space(
    State(state): State<Arc<RwLock<ServerState>>>,
) -> Result<Json<DefaultSpaceResponse>, ServerError> {
    let db = {
        let state = state.read().await;
        state.db.clone().ok_or(ServerError::DbNotInitialized)?
    };
    Ok(Json(DefaultSpaceResponse {
        space: db.get_default_space().await?,
    }))
}

pub async fn handle_set_default_space(
    State(state): State<Arc<RwLock<ServerState>>>,
    Json(request): Json<SetDefaultSpaceRequest>,
) -> Result<Json<DefaultSpaceResponse>, ServerError> {
    let db = {
        let state = state.read().await;
        state.db.clone().ok_or(ServerError::DbNotInitialized)?
    };
    Ok(Json(DefaultSpaceResponse {
        space: Some(db.set_default_space(&request.space_id).await?),
    }))
}

pub async fn handle_clear_default_space(
    State(state): State<Arc<RwLock<ServerState>>>,
) -> Result<StatusCode, ServerError> {
    let db = {
        let state = state.read().await;
        state.db.clone().ok_or(ServerError::DbNotInitialized)?
    };
    db.clear_default_space().await?;
    Ok(StatusCode::NO_CONTENT)
}

pub async fn handle_list_spaces(
    State(state): State<Arc<RwLock<ServerState>>>,
) -> Result<Json<Vec<wenlan_core::db::Space>>, ServerError> {
    let db = {
        let s = state.read().await;
        s.db.clone().ok_or(ServerError::DbNotInitialized)?
    };
    let spaces = db
        .list_spaces()
        .await
        .map_err(|e| ServerError::Internal(e.to_string()))?;
    Ok(Json(spaces))
}

pub async fn handle_create_space(
    State(state): State<Arc<RwLock<ServerState>>>,
    Json(req): Json<CreateSpaceRequest>,
) -> Result<Json<wenlan_core::db::Space>, ServerError> {
    let db = {
        let s = state.read().await;
        s.db.clone().ok_or(ServerError::DbNotInitialized)?
    };
    let space = db
        .create_space(&req.name, req.description.as_deref(), false)
        .await
        .map_err(|e| ServerError::Internal(e.to_string()))?;
    Ok(Json(space))
}

pub async fn handle_update_space(
    State(state): State<Arc<RwLock<ServerState>>>,
    Path(name): Path<String>,
    Json(req): Json<UpdateSpaceRequest>,
) -> Result<Json<wenlan_core::db::Space>, ServerError> {
    let db = {
        let s = state.read().await;
        s.db.clone().ok_or(ServerError::DbNotInitialized)?
    };
    let new_name = req.new_name.as_deref().unwrap_or(&name);
    let space = db
        .update_space(&name, new_name, req.description.as_deref())
        .await
        .map_err(|e| ServerError::Internal(e.to_string()))?;
    Ok(Json(space))
}

pub async fn handle_delete_space(
    State(state): State<Arc<RwLock<ServerState>>>,
    Path(name): Path<String>,
) -> Result<Json<serde_json::Value>, ServerError> {
    let db = {
        let s = state.read().await;
        s.db.clone().ok_or(ServerError::DbNotInitialized)?
    };
    db.delete_space(&name, "keep")
        .await
        .map_err(|e| ServerError::Internal(e.to_string()))?;
    Ok(Json(serde_json::json!({"deleted": name})))
}

pub async fn handle_move_space(
    State(state): State<Arc<RwLock<ServerState>>>,
    Path((from, to)): Path<(String, String)>,
) -> Result<Json<serde_json::Value>, ServerError> {
    let db = {
        let s = state.read().await;
        s.db.clone().ok_or(ServerError::DbNotInitialized)?
    };
    let affected = db
        .reassign_memories_space(&from, &to)
        .await
        .map_err(|e| ServerError::Internal(e.to_string()))?;
    Ok(Json(serde_json::json!({"affected": affected})))
}

pub(crate) fn register_extended(router: TrackedRouter<SharedState>) -> TrackedRouter<SharedState> {
    router
        .route("/api/spaces/{name}/pin", post(handle_pin_space))
        .route("/api/spaces/{name}/confirm", post(handle_confirm_space))
        .route("/api/spaces/reorder", post(handle_reorder_space))
        .route("/api/spaces/{name}/star", post(handle_toggle_space_starred))
        .route(
            "/api/documents/{source_id}/space",
            post(handle_set_document_space),
        )
}

// =====================================================================
// Batch 4 — Space CRUD
// =====================================================================

/// POST /api/spaces/{name}/pin — toggle space pinned (starred) state
pub async fn handle_pin_space(
    State(state): State<Arc<RwLock<ServerState>>>,
    Path(name): Path<String>,
) -> Result<Json<wenlan_types::responses::SuccessResponse>, ServerError> {
    let db = {
        let s = state.read().await;
        s.db.clone().ok_or(ServerError::DbNotInitialized)?
    };
    // pin_space maps to toggle_space_starred in the DB
    db.toggle_space_starred(&name)
        .await
        .map_err(|e| ServerError::Internal(e.to_string()))?;
    Ok(Json(wenlan_types::responses::SuccessResponse { ok: true }))
}

/// POST /api/spaces/{name}/confirm
pub async fn handle_confirm_space(
    State(state): State<Arc<RwLock<ServerState>>>,
    Path(name): Path<String>,
) -> Result<Json<wenlan_types::responses::SuccessResponse>, ServerError> {
    let db = {
        let s = state.read().await;
        s.db.clone().ok_or(ServerError::DbNotInitialized)?
    };
    db.confirm_space(&name)
        .await
        .map_err(|e| ServerError::Internal(e.to_string()))?;
    Ok(Json(wenlan_types::responses::SuccessResponse { ok: true }))
}

/// POST /api/spaces/reorder
pub async fn handle_reorder_space(
    State(state): State<Arc<RwLock<ServerState>>>,
    Json(req): Json<wenlan_types::requests::ReorderSpaceRequest>,
) -> Result<Json<wenlan_types::responses::SuccessResponse>, ServerError> {
    let db = {
        let s = state.read().await;
        s.db.clone().ok_or(ServerError::DbNotInitialized)?
    };
    db.reorder_space(&req.name, req.new_order)
        .await
        .map_err(|e| ServerError::Internal(e.to_string()))?;
    Ok(Json(wenlan_types::responses::SuccessResponse { ok: true }))
}

/// POST /api/spaces/{name}/star — toggle starred state
pub async fn handle_toggle_space_starred(
    State(state): State<Arc<RwLock<ServerState>>>,
    Path(name): Path<String>,
) -> Result<Json<serde_json::Value>, ServerError> {
    let db = {
        let s = state.read().await;
        s.db.clone().ok_or(ServerError::DbNotInitialized)?
    };
    let starred = db
        .toggle_space_starred(&name)
        .await
        .map_err(|e| ServerError::Internal(e.to_string()))?;
    Ok(Json(serde_json::json!({ "starred": starred })))
}

/// POST /api/documents/{source_id}/space — assign a document to a space (domain)
pub async fn handle_set_document_space(
    State(state): State<Arc<RwLock<ServerState>>>,
    Path(source_id): Path<String>,
    Json(req): Json<wenlan_types::requests::SetDocumentSpaceRequest>,
) -> Result<Json<wenlan_types::responses::SuccessResponse>, ServerError> {
    let db = {
        let s = state.read().await;
        s.db.clone().ok_or(ServerError::DbNotInitialized)?
    };
    let requested_space = Some(req.space_name);
    let registered_space =
        registered_request_space(&db, &requested_space, "set_document_space").await?;
    db.update_memory_space_opt(&source_id, registered_space.as_deref())
        .await
        .map_err(|e| ServerError::Internal(e.to_string()))?;
    Ok(Json(wenlan_types::responses::SuccessResponse { ok: true }))
}
