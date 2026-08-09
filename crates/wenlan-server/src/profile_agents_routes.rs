// SPDX-License-Identifier: Apache-2.0
use crate::error::ServerError;
use crate::route_registry::{get, TrackedRouter};
use crate::state::{ServerState, SharedState};
use axum::{
    extract::{Path, State},
    response::Json,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::RwLock;

// ===== Profile Types =====

#[derive(Debug, Serialize)]
pub struct ProfileResponse {
    pub id: String,
    pub name: String,
    pub display_name: Option<String>,
    pub email: Option<String>,
    pub bio: Option<String>,
    pub avatar_path: Option<String>,
    pub created_at: i64,
    pub updated_at: i64,
}

#[derive(Debug, Deserialize)]
pub struct UpdateProfileRequest {
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub display_name: Option<String>,
    #[serde(default)]
    pub email: Option<String>,
    #[serde(default)]
    pub bio: Option<String>,
    #[serde(default)]
    pub avatar_path: Option<String>,
}

// ===== Agent Types =====

#[derive(Debug, Serialize)]
pub struct AgentResponse {
    pub id: String,
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub display_name: Option<String>,
    pub agent_type: String,
    pub description: Option<String>,
    pub enabled: bool,
    pub trust_level: String,
    pub last_seen_at: Option<i64>,
    pub memory_count: i64,
    pub created_at: i64,
    pub updated_at: i64,
}

#[derive(Debug, Deserialize)]
pub struct UpdateAgentRequest {
    #[serde(default)]
    pub agent_type: Option<String>,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub enabled: Option<bool>,
    #[serde(default)]
    pub trust_level: Option<String>,
    #[serde(default)]
    pub display_name: Option<String>,
}

pub(crate) fn register(router: TrackedRouter<SharedState>) -> TrackedRouter<SharedState> {
    router
        .route(
            "/api/profile",
            get(handle_get_profile).put(handle_update_profile),
        )
        .route("/api/agents", get(handle_list_agents))
        .route(
            "/api/agents/{name}",
            get(handle_get_agent)
                .put(handle_update_agent)
                .delete(handle_delete_agent),
        )
}

// ===== Profile Handlers =====

pub async fn handle_get_profile(
    State(state): State<Arc<RwLock<ServerState>>>,
) -> Result<Json<ProfileResponse>, ServerError> {
    let s = state.read().await;
    let db = s.db.as_ref().ok_or(ServerError::DbNotInitialized)?;
    let profile = db
        .get_profile()
        .await
        .map_err(|e| ServerError::Internal(e.to_string()))?;
    match profile {
        Some(p) => Ok(Json(ProfileResponse {
            id: p.id,
            name: p.name,
            display_name: p.display_name,
            email: p.email,
            bio: p.bio,
            avatar_path: p.avatar_path,
            created_at: p.created_at,
            updated_at: p.updated_at,
        })),
        None => Err(ServerError::NotFound("No profile found".to_string())),
    }
}

pub async fn handle_update_profile(
    State(state): State<Arc<RwLock<ServerState>>>,
    Json(req): Json<UpdateProfileRequest>,
) -> Result<Json<ProfileResponse>, ServerError> {
    let s = state.read().await;
    let db = s.db.as_ref().ok_or(ServerError::DbNotInitialized)?;
    let profile = db
        .get_profile()
        .await
        .map_err(|e| ServerError::Internal(e.to_string()))?
        .ok_or(ServerError::NotFound("No profile found".to_string()))?;
    db.update_profile(
        &profile.id,
        req.name.as_deref(),
        req.display_name.as_deref(),
        req.email.as_deref(),
        req.bio.as_deref(),
        req.avatar_path.as_deref(),
    )
    .await
    .map_err(|e| ServerError::Internal(e.to_string()))?;
    let updated = db
        .get_profile()
        .await
        .map_err(|e| ServerError::Internal(e.to_string()))?
        .ok_or(ServerError::NotFound("No profile found".to_string()))?;
    Ok(Json(ProfileResponse {
        id: updated.id,
        name: updated.name,
        display_name: updated.display_name,
        email: updated.email,
        bio: updated.bio,
        avatar_path: updated.avatar_path,
        created_at: updated.created_at,
        updated_at: updated.updated_at,
    }))
}

// ===== Agent Handlers =====

pub async fn handle_list_agents(
    State(state): State<Arc<RwLock<ServerState>>>,
) -> Result<Json<Vec<AgentResponse>>, ServerError> {
    let s = state.read().await;
    let db = s.db.as_ref().ok_or(ServerError::DbNotInitialized)?;
    let agents = db
        .list_agents()
        .await
        .map_err(|e| ServerError::Internal(e.to_string()))?;
    Ok(Json(agents.into_iter().map(agent_to_response).collect()))
}

pub async fn handle_get_agent(
    State(state): State<Arc<RwLock<ServerState>>>,
    Path(name): Path<String>,
) -> Result<Json<AgentResponse>, ServerError> {
    let s = state.read().await;
    let db = s.db.as_ref().ok_or(ServerError::DbNotInitialized)?;
    let agent = db
        .get_agent(&name)
        .await
        .map_err(|e| ServerError::Internal(e.to_string()))?
        .ok_or(ServerError::NotFound(format!("Agent '{}' not found", name)))?;
    Ok(Json(agent_to_response(agent)))
}

pub async fn handle_update_agent(
    State(state): State<Arc<RwLock<ServerState>>>,
    Path(name): Path<String>,
    Json(req): Json<UpdateAgentRequest>,
) -> Result<Json<AgentResponse>, ServerError> {
    let s = state.read().await;
    let db = s.db.as_ref().ok_or(ServerError::DbNotInitialized)?;
    db.get_agent(&name)
        .await
        .map_err(|e| ServerError::Internal(e.to_string()))?
        .ok_or(ServerError::NotFound(format!("Agent '{}' not found", name)))?;
    db.update_agent(
        &name,
        req.agent_type.as_deref(),
        req.description.as_deref(),
        req.enabled,
        req.trust_level.as_deref(),
        req.display_name.as_deref(),
    )
    .await
    .map_err(|e| ServerError::Internal(e.to_string()))?;
    let updated = db
        .get_agent(&name)
        .await
        .map_err(|e| ServerError::Internal(e.to_string()))?
        .ok_or(ServerError::NotFound(format!("Agent '{}' not found", name)))?;
    Ok(Json(agent_to_response(updated)))
}

pub async fn handle_delete_agent(
    State(state): State<Arc<RwLock<ServerState>>>,
    Path(name): Path<String>,
) -> Result<Json<serde_json::Value>, ServerError> {
    let s = state.read().await;
    let db = s.db.as_ref().ok_or(ServerError::DbNotInitialized)?;
    db.delete_agent(&name)
        .await
        .map_err(|e| ServerError::Internal(e.to_string()))?;
    Ok(Json(serde_json::json!({ "deleted": name })))
}

fn agent_to_response(a: wenlan_core::db::AgentConnection) -> AgentResponse {
    AgentResponse {
        id: a.id,
        name: a.name,
        display_name: a.display_name,
        agent_type: a.agent_type,
        description: a.description,
        enabled: a.enabled,
        trust_level: a.trust_level,
        last_seen_at: a.last_seen_at,
        memory_count: a.memory_count,
        created_at: a.created_at,
        updated_at: a.updated_at,
    }
}
