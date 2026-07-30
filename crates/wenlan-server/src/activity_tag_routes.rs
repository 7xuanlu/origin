// SPDX-License-Identifier: Apache-2.0
use crate::error::ServerError;
use crate::route_registry::{delete, get, put, TrackedRouter};
use crate::state::{ServerState, SharedState};
use axum::{
    extract::{Path, State},
    response::Json,
};
use serde::Deserialize;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

pub(crate) fn register(router: TrackedRouter<SharedState>) -> TrackedRouter<SharedState> {
    router
        .route("/api/activities", get(handle_list_activities))
        .route("/api/tags", get(handle_list_tags))
        .route("/api/tags/{name}", delete(handle_delete_tag))
        .route("/api/suggest-tags", get(handle_suggest_tags))
        .route(
            "/api/documents/{source_id}/tags",
            put(handle_set_document_tags),
        )
}

// =====================================================================
// Batch 5 — Activity and tags
// =====================================================================

/// GET /api/activities
pub async fn handle_list_activities(
    State(state): State<Arc<RwLock<ServerState>>>,
    crate::space_header::SpaceHeader(header_space): crate::space_header::SpaceHeader,
    axum::extract::Query(params): axum::extract::Query<HashMap<String, String>>,
) -> Result<Json<wenlan_types::responses::ActivityResponse>, ServerError> {
    let db = {
        let s = state.read().await;
        s.db.clone().ok_or(ServerError::DbNotInitialized)?
    };
    let limit: usize = params
        .get("limit")
        .and_then(|v| v.parse().ok())
        .unwrap_or(50);
    let agent_name = params.get("agent_name").cloned();
    let since: Option<i64> = params.get("since").and_then(|v| v.parse().ok());
    let scope = crate::read_scope::effective_read_scope(&db, None, header_space.as_deref()).await?;
    let activities = db
        .list_agent_activity_scoped(limit, agent_name.as_deref(), since, &scope)
        .await
        .map_err(|e| ServerError::Internal(e.to_string()))?;
    // Page-bearing through `detail`, which four write sites format a page title
    // into at write time. The row keeps no page id, so the adapter refuses the
    // sentence rather than ruling on a subject it cannot name — which is also
    // why no grant is threaded here.
    let activities = wenlan_core::truth_adapter::redact_page_activity_detail(&db, activities)
        .await
        .map_err(|e| ServerError::Internal(e.to_string()))?;
    Ok(Json(wenlan_types::responses::ActivityResponse {
        activities,
    }))
}

/// GET /api/tags
pub async fn handle_list_tags(
    State(state): State<Arc<RwLock<ServerState>>>,
    crate::space_header::SpaceHeader(header_space): crate::space_header::SpaceHeader,
) -> Result<Json<wenlan_types::responses::TagsResponse>, ServerError> {
    let db = {
        let s = state.read().await;
        s.db.clone().ok_or(ServerError::DbNotInitialized)?
    };
    let scope = crate::read_scope::effective_read_scope(&db, None, header_space.as_deref()).await?;
    let projection = db
        .list_tags_scoped(&scope)
        .await
        .map_err(|e| ServerError::Internal(e.to_string()))?;
    Ok(Json(wenlan_types::responses::TagsResponse {
        tags: projection.tags,
        document_tags: projection.document_tags,
    }))
}

/// DELETE /api/tags/{name}
pub async fn handle_delete_tag(
    State(state): State<Arc<RwLock<ServerState>>>,
    Path(name): Path<String>,
) -> Result<Json<wenlan_types::responses::SuccessResponse>, ServerError> {
    let db = {
        let s = state.read().await;
        s.db.clone().ok_or(ServerError::DbNotInitialized)?
    };
    db.delete_tag(&name)
        .await
        .map_err(|e| ServerError::Internal(e.to_string()))?;
    Ok(Json(wenlan_types::responses::SuccessResponse { ok: true }))
}

/// PUT /api/documents/{source_id}/tags
pub async fn handle_set_document_tags(
    State(state): State<Arc<RwLock<ServerState>>>,
    Path(source_id): Path<String>,
    Json(req): Json<wenlan_types::requests::SetDocumentTagsRequest>,
) -> Result<Json<wenlan_types::responses::TagsResponse>, ServerError> {
    let db = {
        let s = state.read().await;
        s.db.clone().ok_or(ServerError::DbNotInitialized)?
    };
    let source = document_tag_source(&req).to_string();
    let tags = db
        .set_document_tags(&source, &source_id, req.tags)
        .await
        .map_err(|e| ServerError::Internal(e.to_string()))?;
    Ok(Json(wenlan_types::responses::TagsResponse {
        tags,
        document_tags: HashMap::new(),
    }))
}

fn document_tag_source(req: &wenlan_types::requests::SetDocumentTagsRequest) -> &str {
    req.source
        .as_deref()
        .map(str::trim)
        .filter(|source| !source.is_empty())
        .unwrap_or("memory")
}

#[derive(Debug, Deserialize)]
pub struct SuggestTagsQuery {
    pub source: String,
    pub source_id: String,
    /// Optional caller-side hint — the Tauri app passes the name of the
    /// active application at the document's timestamp (activities are
    /// tracked in-process there, not in the DB).
    #[serde(default)]
    pub activity_app: Option<String>,
}

/// GET /api/suggest-tags?source=...&source_id=...&activity_app=...
///
/// Returns candidate tag names derived from a document's chunked content
/// and title, optionally augmented with a caller-supplied activity app
/// name, and with already-assigned tags filtered out.
pub async fn handle_suggest_tags(
    State(state): State<Arc<RwLock<ServerState>>>,
    crate::space_header::SpaceHeader(header_space): crate::space_header::SpaceHeader,
    axum::extract::Query(query): axum::extract::Query<SuggestTagsQuery>,
) -> Result<Json<wenlan_types::responses::TagsResponse>, ServerError> {
    // Read phase: clone db Arc, drop guard, then fetch existing tags from DB.
    let db = {
        let s = state.read().await;
        s.db.clone().ok_or(ServerError::DbNotInitialized)?
    };
    let scope = crate::read_scope::effective_read_scope(&db, None, header_space.as_deref()).await?;
    let (existing, chunks) = db
        .get_tag_suggestion_inputs_scoped(&query.source, &query.source_id, &scope)
        .await
        .map_err(|e| ServerError::Internal(e.to_string()))?
        .ok_or_else(|| ServerError::NotFound("memory not found".to_string()))?;

    let chunk_contents: Vec<String> = chunks.iter().map(|c| c.content.clone()).collect();
    let title = chunks.first().map(|c| c.title.clone()).unwrap_or_default();

    let mut tags = wenlan_core::tags::suggest_tags_for_document(&chunk_contents, &title, &existing);

    // Merge caller-side activity hint (app name), respecting dedup + the
    // "not already assigned" filter.
    if let Some(app) = query.activity_app {
        let normalized = app.trim().to_lowercase();
        if !normalized.is_empty()
            && !existing.iter().any(|t| t == &normalized)
            && !tags.iter().any(|t| t == &normalized)
        {
            tags.push(normalized);
            tags.sort();
        }
    }

    Ok(Json(wenlan_types::responses::TagsResponse {
        tags,
        document_tags: HashMap::new(),
    }))
}

#[cfg(test)]
mod tag_route_tests {
    use super::*;

    #[test]
    fn document_tag_source_prefers_request_source() {
        let req = wenlan_types::requests::SetDocumentTagsRequest {
            source: Some("manual".to_string()),
            tags: vec!["rust".to_string()],
        };

        assert_eq!(document_tag_source(&req), "manual");
    }

    #[test]
    fn document_tag_source_defaults_to_memory_for_old_payloads() {
        let req = wenlan_types::requests::SetDocumentTagsRequest {
            source: None,
            tags: vec!["rust".to_string()],
        };

        assert_eq!(document_tag_source(&req), "memory");
    }
}
