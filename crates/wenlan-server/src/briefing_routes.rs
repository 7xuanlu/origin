// SPDX-License-Identifier: Apache-2.0
use crate::error::ServerError;
use crate::route_registry::{get, TrackedRouter};
use crate::state::{ServerState, SharedState};
use axum::{extract::State, response::Json};
use std::sync::Arc;
use tokio::sync::RwLock;

pub(crate) fn register(router: TrackedRouter<SharedState>) -> TrackedRouter<SharedState> {
    router.route("/api/briefing", get(handle_get_briefing))
}

/// GET /api/briefing
pub async fn handle_get_briefing(
    State(state): State<Arc<RwLock<ServerState>>>,
    crate::space_header::SpaceHeader(header_space): crate::space_header::SpaceHeader,
) -> Result<Json<wenlan_core::briefing::BriefingResponse>, ServerError> {
    let (db, llm, prompts, tuning) = {
        let s = state.read().await;
        let db = s.db.clone().ok_or(ServerError::DbNotInitialized)?;
        let llm = s.llm.clone();
        let prompts = s.prompts.clone();
        let tuning = s.tuning.briefing.clone();
        (db, llm, prompts, tuning)
    };
    let scope = crate::read_scope::effective_read_scope(&db, None, header_space.as_deref()).await?;
    let briefing = wenlan_core::briefing::generate_briefing_scoped(
        &db,
        llm.as_deref(),
        &prompts,
        &tuning,
        &scope,
    )
    .await?;
    Ok(Json(briefing))
}
