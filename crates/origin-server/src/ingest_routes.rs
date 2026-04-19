// SPDX-License-Identifier: Apache-2.0
use crate::error::ServerError;
use crate::state::ServerState;
use axum::{
    extract::{Path, State},
    response::Json,
};
use origin_types::sources::RawDocument;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

// ===== Request/Response Types =====

#[derive(Debug, Deserialize)]
pub struct IngestTextRequest {
    pub source: String,
    pub source_id: String,
    pub title: String,
    pub content: String,
    pub url: Option<String>,
    pub metadata: Option<HashMap<String, String>>,
}

#[derive(Debug, Deserialize)]
pub struct IngestWebpageRequest {
    pub url: String,
    pub title: String,
    pub content: String,
    pub metadata: Option<HashMap<String, String>>,
}

#[derive(Debug, Deserialize)]
pub struct IngestMemoryRequest {
    pub source: String,
    pub source_id: String,
    pub title: String,
    pub content: String,
    pub url: Option<String>,
    pub tags: Option<Vec<String>>,
    pub metadata: Option<HashMap<String, String>>,
}

#[derive(Debug, Serialize)]
pub struct IngestResponse {
    pub chunks_created: usize,
    pub document_id: String,
}

#[derive(Debug, Serialize)]
pub struct DeleteResponse {
    pub deleted: bool,
}

// ===== Route Handlers =====

/// POST /api/ingest/text
pub async fn handle_ingest_text(
    State(state): State<Arc<RwLock<ServerState>>>,
    Json(req): Json<IngestTextRequest>,
) -> Result<Json<IngestResponse>, ServerError> {
    let document_id = req.source_id.clone();

    let doc = RawDocument {
        source: req.source,
        source_id: req.source_id,
        title: req.title,
        summary: None,
        content: req.content,
        url: req.url,
        last_modified: chrono::Utc::now().timestamp(),
        metadata: req.metadata.unwrap_or_default(),
        memory_type: None,
        source_agent: None,
        domain: None,
        confidence: None,
        confirmed: None,
        supersedes: None,
        pending_revision: false,
        ..Default::default()
    };

    let chunks_created = {
        let s = state.read().await;
        let db = s.db.as_ref().ok_or(ServerError::DbNotInitialized)?;
        db.upsert_documents(vec![doc])
            .await
            .map_err(|e| ServerError::IngestFailed(e.to_string()))?
    };

    Ok(Json(IngestResponse {
        chunks_created,
        document_id,
    }))
}

/// POST /api/ingest/webpage
pub async fn handle_ingest_webpage(
    State(state): State<Arc<RwLock<ServerState>>>,
    Json(req): Json<IngestWebpageRequest>,
) -> Result<Json<IngestResponse>, ServerError> {
    let document_id = req.url.clone();

    let mut metadata = req.metadata.unwrap_or_default();
    if let Some(domain) = req
        .url
        .split("://")
        .nth(1)
        .and_then(|rest| rest.split('/').next())
    {
        metadata.insert("domain".to_string(), domain.to_string());
    }

    let doc = RawDocument {
        source: "webpage".to_string(),
        source_id: req.url.clone(),
        title: req.title,
        summary: None,
        content: req.content,
        url: Some(req.url),
        last_modified: chrono::Utc::now().timestamp(),
        metadata,
        memory_type: None,
        source_agent: None,
        domain: None,
        confidence: None,
        confirmed: None,
        supersedes: None,
        pending_revision: false,
        ..Default::default()
    };

    let chunks_created = {
        let s = state.read().await;
        let db = s.db.as_ref().ok_or(ServerError::DbNotInitialized)?;
        db.upsert_documents(vec![doc])
            .await
            .map_err(|e| ServerError::IngestFailed(e.to_string()))?
    };

    Ok(Json(IngestResponse {
        chunks_created,
        document_id,
    }))
}

/// POST /api/ingest/memory
pub async fn handle_ingest_memory(
    State(state): State<Arc<RwLock<ServerState>>>,
    Json(req): Json<IngestMemoryRequest>,
) -> Result<Json<IngestResponse>, ServerError> {
    let document_id = req.source_id.clone();

    let mut metadata = req.metadata.unwrap_or_default();
    if let Some(tags) = req.tags {
        metadata.insert("tags".to_string(), tags.join(","));
    }

    let doc = RawDocument {
        source: req.source,
        source_id: req.source_id,
        title: req.title,
        summary: None,
        content: req.content,
        url: req.url,
        last_modified: chrono::Utc::now().timestamp(),
        metadata,
        memory_type: None,
        source_agent: None,
        domain: None,
        confidence: None,
        confirmed: None,
        supersedes: None,
        pending_revision: false,
        ..Default::default()
    };

    let chunks_created = {
        let s = state.read().await;
        let db = s.db.as_ref().ok_or(ServerError::DbNotInitialized)?;
        db.upsert_documents(vec![doc])
            .await
            .map_err(|e| ServerError::IngestFailed(e.to_string()))?
    };

    Ok(Json(IngestResponse {
        chunks_created,
        document_id,
    }))
}

/// DELETE /api/documents/:source/:source_id
pub async fn handle_delete_document(
    State(state): State<Arc<RwLock<ServerState>>>,
    Path((source, source_id)): Path<(String, String)>,
) -> Result<Json<DeleteResponse>, ServerError> {
    let s = state.read().await;
    let db = s.db.as_ref().ok_or(ServerError::DbNotInitialized)?;
    db.delete_by_source_id(&source, &source_id)
        .await
        .map_err(|e| ServerError::Internal(e.to_string()))?;
    Ok(Json(DeleteResponse { deleted: true }))
}
