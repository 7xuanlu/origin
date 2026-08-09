// SPDX-License-Identifier: Apache-2.0
use crate::error::ServerError;
use crate::route_registry::{get, post, TrackedRouter};
use crate::state::{ServerState, SharedState};
use axum::{extract::State, response::Json};
use std::sync::Arc;
use tokio::sync::RwLock;

pub(crate) fn register(router: TrackedRouter<SharedState>) -> TrackedRouter<SharedState> {
    router
        .route("/api/profile/narrative", get(handle_get_profile_narrative))
        .route(
            "/api/profile/narrative/regenerate",
            post(handle_regenerate_narrative),
        )
}

/// GET /api/profile/narrative
///
/// Cache-first: returns the cached narrative immediately if present, so the
/// profile page loads instantly instead of waiting on an LLM call every time.
/// Falls through to `generate_narrative` (which writes to cache on success)
/// when the cache is empty. Explicit regeneration still goes through
/// `/api/profile/narrative/regenerate`.
pub async fn handle_get_profile_narrative(
    State(state): State<Arc<RwLock<ServerState>>>,
) -> Result<Json<wenlan_core::narrative::NarrativeResponse>, ServerError> {
    let (db, llm, prompts, tuning) = {
        let s = state.read().await;
        let db = s.db.clone().ok_or(ServerError::DbNotInitialized)?;
        let llm = s.llm.clone();
        let prompts = s.prompts.clone();
        let tuning = s.tuning.narrative.clone();
        (db, llm, prompts, tuning)
    };

    // 1. Try the cache first. If we have a stored narrative with content,
    //    return it immediately — no LLM round-trip on page load.
    if let Ok(Some((content, generated_at, memory_count))) = db.get_cached_narrative().await {
        if !content.is_empty() {
            return Ok(Json(wenlan_core::narrative::NarrativeResponse {
                content,
                generated_at,
                is_stale: false,
                memory_count,
            }));
        }
    }

    // 2. Nothing cached — generate fresh (this call also writes to the cache
    //    so subsequent loads are instant).
    let narrative =
        wenlan_core::narrative::generate_narrative(&db, llm.as_deref(), &prompts, &tuning)
            .await
            .map_err(|e| ServerError::Internal(e.to_string()))?;
    Ok(Json(narrative))
}

/// POST /api/profile/narrative/regenerate
pub async fn handle_regenerate_narrative(
    State(state): State<Arc<RwLock<ServerState>>>,
) -> Result<Json<wenlan_core::narrative::NarrativeResponse>, ServerError> {
    // Same as get, but always regenerates (no cache)
    let (db, llm, prompts, tuning) = {
        let s = state.read().await;
        let db = s.db.clone().ok_or(ServerError::DbNotInitialized)?;
        let llm = s.llm.clone();
        let prompts = s.prompts.clone();
        let tuning = s.tuning.narrative.clone();
        (db, llm, prompts, tuning)
    };
    let narrative =
        wenlan_core::narrative::generate_narrative(&db, llm.as_deref(), &prompts, &tuning)
            .await
            .map_err(|e| ServerError::Internal(e.to_string()))?;
    Ok(Json(narrative))
}
