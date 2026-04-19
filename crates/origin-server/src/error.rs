// SPDX-License-Identifier: Apache-2.0
use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use serde_json::json;

/// Server-specific errors
#[derive(Debug)]
pub enum ServerError {
    DbNotInitialized,
    SearchFailed(String),
    IngestFailed(String),
    Internal(String),
    AgentDisabled(String),
    NotFound(String),
    ValidationError(String),
    QualityGateRejected {
        reason: String,
        detail: String,
        similar_to: Option<String>,
    },
    ChatImport(String),
}

impl std::fmt::Display for ServerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ServerError::DbNotInitialized => write!(f, "Database not initialized"),
            ServerError::SearchFailed(msg) => write!(f, "Search failed: {}", msg),
            ServerError::IngestFailed(msg) => write!(f, "Ingest failed: {}", msg),
            ServerError::Internal(msg) => write!(f, "Internal error: {}", msg),
            ServerError::AgentDisabled(msg) => write!(f, "Agent disabled: {}", msg),
            ServerError::NotFound(msg) => write!(f, "Not found: {}", msg),
            ServerError::ValidationError(msg) => write!(f, "Validation error: {}", msg),
            ServerError::QualityGateRejected { detail, .. } => {
                write!(f, "Quality gate rejected: {}", detail)
            }
            ServerError::ChatImport(msg) => write!(f, "Chat import failed: {}", msg),
        }
    }
}

impl std::error::Error for ServerError {}

impl IntoResponse for ServerError {
    fn into_response(self) -> Response {
        if let ServerError::QualityGateRejected {
            reason,
            detail,
            similar_to,
        } = self
        {
            let body = Json(json!({
                "status": "rejected",
                "reason": reason,
                "detail": detail,
                "similar_to": similar_to,
            }));
            return (StatusCode::UNPROCESSABLE_ENTITY, body).into_response();
        }

        let (status, error_message) = match self {
            ServerError::DbNotInitialized => (
                StatusCode::SERVICE_UNAVAILABLE,
                "Database not initialized".to_string(),
            ),
            ServerError::SearchFailed(msg) => (StatusCode::BAD_REQUEST, msg),
            ServerError::IngestFailed(msg) => (StatusCode::BAD_REQUEST, msg),
            ServerError::Internal(msg) => (StatusCode::INTERNAL_SERVER_ERROR, msg),
            ServerError::AgentDisabled(msg) => (StatusCode::FORBIDDEN, msg),
            ServerError::NotFound(msg) => (StatusCode::NOT_FOUND, msg),
            ServerError::ValidationError(msg) => (StatusCode::UNPROCESSABLE_ENTITY, msg),
            ServerError::ChatImport(msg) => (StatusCode::BAD_REQUEST, msg),
            ServerError::QualityGateRejected { .. } => unreachable!(),
        };

        let body = Json(json!({
            "error": error_message,
        }));

        (status, body).into_response()
    }
}

impl From<origin_core::OriginError> for ServerError {
    fn from(err: origin_core::OriginError) -> Self {
        match err {
            origin_core::OriginError::AgentDisabled(msg) => ServerError::AgentDisabled(msg),
            other => ServerError::Internal(other.to_string()),
        }
    }
}
