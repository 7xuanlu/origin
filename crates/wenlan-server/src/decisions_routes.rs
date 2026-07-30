// SPDX-License-Identifier: Apache-2.0
use crate::error::ServerError;
use crate::route_registry::{get, TrackedRouter};
use crate::state::{ServerState, SharedState};
use axum::{extract::State, response::Json};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

pub(crate) fn register(router: TrackedRouter<SharedState>) -> TrackedRouter<SharedState> {
    router
        .route("/api/decisions", get(handle_list_decisions))
        .route("/api/decisions/domains", get(handle_list_decision_domains))
}

// =====================================================================
// Batch 6 — Decisions
// =====================================================================

/// GET /api/decisions
pub async fn handle_list_decisions(
    State(state): State<Arc<RwLock<ServerState>>>,
    crate::space_header::SpaceHeader(header_space): crate::space_header::SpaceHeader,
    axum::extract::Query(params): axum::extract::Query<HashMap<String, String>>,
) -> Result<Json<wenlan_types::responses::DecisionsResponse>, ServerError> {
    let db = {
        let s = state.read().await;
        s.db.clone().ok_or(ServerError::DbNotInitialized)?
    };
    let space = params
        .get("space")
        .or_else(|| params.get("domain"))
        .cloned();
    let scope =
        crate::read_scope::effective_read_scope(&db, space.as_deref(), header_space.as_deref())
            .await?;
    let limit: usize = params
        .get("limit")
        .and_then(|v| v.parse().ok())
        .unwrap_or(100);
    let decisions = db
        .list_memories_scoped(&scope, Some("decision"), None, None, limit)
        .await
        .map_err(|e| ServerError::Internal(e.to_string()))?;
    Ok(Json(wenlan_types::responses::DecisionsResponse {
        decisions,
    }))
}

/// GET /api/decisions/domains
/// (Path kept as "domains" for back-compat; will rename to "spaces" in PR-A+1.)
pub async fn handle_list_decision_domains(
    State(state): State<Arc<RwLock<ServerState>>>,
) -> Result<Json<wenlan_types::responses::DecisionDomainsResponse>, ServerError> {
    let db = {
        let s = state.read().await;
        s.db.clone().ok_or(ServerError::DbNotInitialized)?
    };
    let domains = db
        .list_decision_spaces()
        .await
        .map_err(|e| ServerError::Internal(e.to_string()))?;
    Ok(Json(wenlan_types::responses::DecisionDomainsResponse {
        domains,
    }))
}
