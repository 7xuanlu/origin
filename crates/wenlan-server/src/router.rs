// SPDX-License-Identifier: Apache-2.0
//! Axum router construction — wires all HTTP and WebSocket routes.

use crate::lifecycle::ShutdownHandle;
pub use crate::route_registry::AppRouter;
use crate::route_registry::{get, TrackedRouter};
use crate::state::SharedState;
use crate::{
    activity_tag_routes, ambient_routes, brief_routes, briefing_routes, community_routes,
    config_routes, decisions_routes, entity_graph_routes, import_routes, indexed_files_routes,
    ingest_routes, knowledge_routes, lint_routes, memory_detail_routes, memory_revision_routes,
    memory_routes, onboarding_routes, outbox_routes, page_map_routes, page_routes,
    pinned_memory_routes, profile_agents_routes, profile_narrative_routes, refinery_routes,
    repair_routes, routes, security, snapshot_routes, source_routes, spaces_routes, truth_guard,
    websocket,
};
use tower_http::cors::{AllowOrigin, Any, CorsLayer};
use wenlan_core::truth_manifest::Builder;

/// Build the shared application router with all routes.
pub fn build_router(state: SharedState) -> AppRouter {
    let shutdown = {
        state
            .try_read()
            .expect("build_router requires unlocked initial ServerState")
            .shutdown
            .clone()
    };
    build_router_with_shutdown(state, shutdown)
}

/// Build the router with an out-of-band lifecycle handle. The daemon uses this
/// form so `/api/shutdown` never waits behind the workload-state lock.
pub fn build_router_with_shutdown(state: SharedState, shutdown: ShutdownHandle) -> AppRouter {
    // Reflect CORS only for local origins so a legit localhost/Tauri browser
    // can read responses; every other origin is refused. The real protection
    // is `security::guard_local_only` below — this just stops browsers from
    // exposing responses to non-local sites. Native clients (app/CLI/MCP over
    // reqwest) send no Origin and are unaffected.
    let cors = CorsLayer::new()
        .allow_origin(AllowOrigin::predicate(|origin, _parts| {
            origin
                .to_str()
                .map(crate::security::origin_is_local)
                .unwrap_or(false)
        }))
        .allow_methods(Any)
        .allow_headers(Any);

    let router = repair_routes::register(lint_routes::register(TrackedRouter::new(Builder::Main)));
    let router = routes::register(router);
    let router = ambient_routes::register(router);
    let router = brief_routes::register(router);
    let router = outbox_routes::register(router);
    let router = community_routes::register(router);
    let router = ingest_routes::register(router);
    let router = import_routes::register(router);
    let router = memory_routes::register_core(router);
    let router = entity_graph_routes::register_writes(router);
    let router = profile_agents_routes::register(router);
    let router = entity_graph_routes::register_reads(router);
    let router = spaces_routes::register_core(router);
    let router = page_routes::register(router);

    let router = page_map_routes::register(router);

    let router = memory_revision_routes::register_history(router);
    let router = refinery_routes::register(router);
    let router = source_routes::register(router);
    let router = config_routes::register(router);
    let router = indexed_files_routes::register(router);
    let router = entity_graph_routes::register_crud(router);
    let router = spaces_routes::register_extended(router);
    let router = activity_tag_routes::register(router);
    let router = memory_detail_routes::register(router);
    let router = decisions_routes::register(router);
    let router = briefing_routes::register(router);
    let router = profile_narrative_routes::register(router);
    let router = pinned_memory_routes::register(router);
    let router = memory_revision_routes::register_pending(router);
    let router = snapshot_routes::register(router);
    let router = page_routes::register_manual_edit(router);
    let router = knowledge_routes::register(router);
    let router = onboarding_routes::register(router);
    let router = websocket::register(router);

    router
        .finish()
        // Innermost of the three, and `route_layer` rather than `layer`: it
        // resolves the route's marker shape from `MatchedPath`, which only
        // exists after routing.
        .route_layer(axum::middleware::from_fn_with_state(
            truth_guard::TruthGuardState {
                state: state.clone(),
                builder: Builder::Main,
            },
            truth_guard::guard,
        ))
        .layer(cors)
        // Outermost: reject cross-origin browser traffic before CORS/handlers.
        .layer(axum::middleware::from_fn(security::guard_local_only))
        .layer(axum::Extension(shutdown))
        .with_state(state)
}

/// Build the fail-closed router used while one exact repair owns the writer
/// fence. Preparation and every ordinary product mutation stay unreachable;
/// the process can only inspect health, rerun lint, apply the approved
/// manifest, and verify its receipts.
pub fn build_repair_router(state: SharedState) -> AppRouter {
    let cors = CorsLayer::new()
        .allow_origin(AllowOrigin::predicate(|origin, _parts| {
            origin
                .to_str()
                .map(crate::security::origin_is_local)
                .unwrap_or(false)
        }))
        .allow_methods(Any)
        .allow_headers(Any);

    repair_routes::register_execution(lint_routes::register(TrackedRouter::new(Builder::Repair)))
        .route("/api/health", get(routes::handle_health))
        .route("/api/status", get(routes::handle_status))
        .finish_restricted()
        .route_layer(axum::middleware::from_fn_with_state(
            truth_guard::TruthGuardState {
                state: state.clone(),
                builder: Builder::Repair,
            },
            truth_guard::guard,
        ))
        .layer(cors)
        .layer(axum::middleware::from_fn(security::guard_local_only))
        .with_state(state)
}

#[cfg(test)]
mod repair_only_tests {
    use super::*;
    use axum::{
        body::Body,
        http::{Method, Request, StatusCode},
    };
    use std::sync::Arc;
    use tokio::sync::RwLock;
    use tower::ServiceExt;

    async fn status(method: Method, path: &str) -> StatusCode {
        let state = Arc::new(RwLock::new(crate::state::ServerState::new()));
        build_repair_router(state)
            .oneshot(
                Request::builder()
                    .method(method)
                    .uri(path)
                    .header("content-type", "application/json")
                    .body(Body::from("{}"))
                    .unwrap(),
            )
            .await
            .unwrap()
            .status()
    }

    #[tokio::test]
    async fn repair_only_router_exposes_only_read_only_diagnostics_and_exact_execution() {
        assert_eq!(status(Method::GET, "/api/health").await, StatusCode::OK);
        assert_ne!(
            status(Method::GET, "/api/lint").await,
            StatusCode::NOT_FOUND
        );
        assert_ne!(
            status(Method::POST, "/api/repairs/apply").await,
            StatusCode::NOT_FOUND
        );
        assert_ne!(
            status(Method::POST, "/api/repairs/verify").await,
            StatusCode::NOT_FOUND
        );

        for path in [
            "/api/repairs/plan",
            "/api/repairs/prepare",
            "/api/memory/store",
            "/api/pages",
            "/api/config",
        ] {
            assert_eq!(
                status(Method::POST, path).await,
                StatusCode::NOT_FOUND,
                "repair-only router unexpectedly exposed {path}"
            );
        }
    }
}
