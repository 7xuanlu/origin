// SPDX-License-Identifier: Apache-2.0
//! Operator-facing HTTP surface for the ambient enrichment scheduler:
//! `POST /api/ambient/sweep` forces one bounded, synchronous pass over every
//! ambient job type, bypassing the idle/resource gate (not each job's own
//! per-call slice bound); `GET /api/ambient/status` reports pending-work
//! counts, per-phase last-run timestamps, and whether the passive scheduler's
//! idle gate currently admits work. Business logic for both lives in
//! `crate::scheduler` (via the `ambient_admin` submodule) — this file only
//! owns HTTP framing.

use axum::{extract::State, Json};
use std::sync::Arc;
use tokio::sync::RwLock;

use crate::error::ServerError;
use crate::route_registry::{get, post, TrackedRouter};
use crate::scheduler::{AmbientStatusReport, AmbientSweepReport};
use crate::state::{ServerState, SharedState};

pub(crate) fn register(router: TrackedRouter<SharedState>) -> TrackedRouter<SharedState> {
    router
        .route("/api/ambient/status", get(handle_ambient_status))
        .route("/api/ambient/sweep", post(handle_ambient_sweep))
}

/// POST /api/ambient/sweep
pub async fn handle_ambient_sweep(
    State(state): State<Arc<RwLock<ServerState>>>,
) -> Result<Json<AmbientSweepReport>, ServerError> {
    let (db, llm, api_llm, external_llm, prompts, refinery_cfg, distillation_cfg, ambient_run_lock) = {
        let s = state.read().await;
        (
            s.db.clone().ok_or(ServerError::DbNotInitialized)?,
            s.llm.clone(),
            s.api_llm.clone(),
            s.external_llm.clone(),
            s.prompts.clone(),
            s.tuning.refinery.clone(),
            s.tuning.distillation.clone(),
            s.ambient_run_lock.clone(),
        )
    };
    // Routing consent and the knowledge root are read fresh from config, same
    // as the passive scheduler tick and `handle_sync_source` — neither is
    // cached in `ServerState`.
    let runtime_config = wenlan_core::config::load_config();
    let everyday_pin =
        wenlan_core::refinery::EverydaySource::parse(runtime_config.everyday_source.as_deref());
    let knowledge_path = runtime_config.knowledge_path_or_default();

    // Hold the ambient-run lock for the whole lap so this force-sweep and
    // the passive scheduler's own ambient tick can never execute jobs at the
    // same time (doubled LLM contention, possible double-claiming of the
    // same document/page). Unlike the scheduler tick, which only
    // `try_lock`s and skips its turn, the endpoint *waits* here: a caller
    // that explicitly asked for a sweep wants the work done, not a fast
    // failure to retry later, and this is exactly the trade the CLI's
    // generous request timeout is sized for. Holding a `tokio::sync::Mutex`
    // guard across `.await` is fine — the repo's no-guard-across-await
    // invariant is specific to `RwLock`.
    let _ambient_run_guard = ambient_run_lock.lock().await;

    let report = crate::scheduler::force_ambient_sweep(
        &db,
        llm.as_ref(),
        api_llm.as_ref(),
        external_llm.as_ref(),
        everyday_pin,
        &prompts,
        &refinery_cfg,
        &distillation_cfg,
        Some(knowledge_path.as_path()),
    )
    .await;
    Ok(Json(report))
}

/// GET /api/ambient/status
pub async fn handle_ambient_status(
    State(state): State<Arc<RwLock<ServerState>>>,
) -> Result<Json<AmbientStatusReport>, ServerError> {
    let (db, gate) = {
        let s = state.read().await;
        let db = s.db.clone().ok_or(ServerError::DbNotInitialized)?;
        // A `std::sync::Mutex` lock is synchronous and released before this
        // statement ends — never held across an `.await`. Bound to its own
        // `let` (rather than inlined into the tuple below) so the guard
        // drops here, before `s` itself drops at the end of this block.
        let gate = s.ambient_gate.lock().unwrap().clone();
        (db, gate)
    };
    let report = crate::scheduler::ambient_status(&db, gate)
        .await
        .map_err(ServerError::from)?;
    Ok(Json(report))
}
