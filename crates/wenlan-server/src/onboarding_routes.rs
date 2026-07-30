// SPDX-License-Identifier: Apache-2.0
//! HTTP handlers for onboarding milestone state.
//!
//! Three endpoints back the post-wizard UX:
//! - `GET /api/onboarding/milestones` — list all recorded milestones
//! - `POST /api/onboarding/milestones/{id}/acknowledge` — mark a milestone as seen
//! - `POST /api/onboarding/reset` — clear all milestones (dev/demo only)
//!
//! Handlers follow the canonical snapshot-and-drop pattern: clone the
//! `Arc<MemoryDB>` out of the `RwLock<ServerState>` read guard, drop the
//! guard, then call the async DB method. Never hold the guard across
//! `.await` (see CLAUDE.md).

use crate::error::ServerError;
use crate::route_registry::{get, post, TrackedRouter};
use crate::state::SharedState;
use axum::{
    extract::{Path, State},
    http::StatusCode,
    Json,
};
use std::str::FromStr;
use wenlan_core::onboarding::{MilestoneId, MilestoneRecord};

pub(crate) fn register(router: TrackedRouter<SharedState>) -> TrackedRouter<SharedState> {
    router
        .route("/api/onboarding/milestones", get(handle_list_milestones))
        .route(
            "/api/onboarding/milestones/{id}/acknowledge",
            post(handle_acknowledge_milestone),
        )
        .route("/api/onboarding/reset", post(handle_reset_milestones))
}

pub async fn handle_list_milestones(
    State(state): State<SharedState>,
) -> Result<Json<Vec<MilestoneRecord>>, ServerError> {
    let db = {
        let guard = state.read().await;
        guard
            .db
            .as_ref()
            .ok_or(ServerError::DbNotInitialized)?
            .clone()
    }; // guard dropped here
    let list = db.list_milestones().await.map_err(ServerError::from)?;
    Ok(Json(list))
}

pub async fn handle_acknowledge_milestone(
    State(state): State<SharedState>,
    Path(id): Path<String>,
) -> Result<StatusCode, ServerError> {
    let milestone = MilestoneId::from_str(&id)
        .map_err(|e| ServerError::ValidationError(format!("invalid milestone id: {}", e)))?;
    let db = {
        let guard = state.read().await;
        guard
            .db
            .as_ref()
            .ok_or(ServerError::DbNotInitialized)?
            .clone()
    }; // guard dropped here
    db.acknowledge_milestone(milestone)
        .await
        .map_err(ServerError::from)?;
    Ok(StatusCode::NO_CONTENT)
}

pub async fn handle_reset_milestones(
    State(state): State<SharedState>,
) -> Result<StatusCode, ServerError> {
    let db = {
        let guard = state.read().await;
        guard
            .db
            .as_ref()
            .ok_or(ServerError::DbNotInitialized)?
            .clone()
    }; // guard dropped here
    db.reset_onboarding_milestones()
        .await
        .map_err(ServerError::from)?;
    Ok(StatusCode::NO_CONTENT)
}
