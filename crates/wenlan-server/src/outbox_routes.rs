// SPDX-License-Identifier: Apache-2.0

use crate::route_registry::{get, post, TrackedRouter};
use crate::state::SharedState;
use axum::{extract::State, response::Json};
use serde::Serialize;

pub(crate) fn register(router: TrackedRouter<SharedState>) -> TrackedRouter<SharedState> {
    router
        .route("/api/outbox/drain", post(handle_drain))
        .route("/api/outbox/status", get(handle_status))
}

pub async fn handle_drain(
    State(state): State<SharedState>,
) -> Json<wenlan_types::OutboxDrainReport> {
    Json(crate::outbox::drain_once(state).await)
}

#[derive(Debug, Serialize)]
pub struct OutboxStatusResponse {
    pub queued: u32,
    pub failed: u32,
}

pub async fn handle_status() -> Json<OutboxStatusResponse> {
    Json(OutboxStatusResponse {
        queued: crate::outbox::queued_files()
            .map(|files| files.len() as u32)
            .unwrap_or(0),
        failed: crate::outbox::failed_files()
            .map(|files| files.len() as u32)
            .unwrap_or(0),
    })
}
