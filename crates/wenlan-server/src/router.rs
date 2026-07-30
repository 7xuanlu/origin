// SPDX-License-Identifier: Apache-2.0
//! Axum router construction — wires all HTTP and WebSocket routes.

use crate::lifecycle::ShutdownHandle;
pub use crate::route_registry::AppRouter;
use crate::route_registry::{delete, get, patch, post, put, TrackedRouter};
use crate::state::SharedState;
use crate::{
    brief_routes, community_routes, config_routes, import_routes, ingest_routes, knowledge_routes,
    lint_routes, memory_routes, onboarding_routes, page_map_routes, refinery_routes, repair_routes,
    routes, security, source_routes, truth_guard, websocket,
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
    let router = brief_routes::register(router);
    let router = community_routes::register(router);

    let router = router
        .route(
            "/api/memory/recent",
            get(memory_routes::handle_recent_memories),
        )
        .route(
            "/api/memory/unconfirmed",
            get(memory_routes::handle_list_unconfirmed_memories),
        );
    let router = ingest_routes::register(router);
    let router = import_routes::register(router);

    let router = router
        // Memory CRUD
        .route(
            "/api/memory/store",
            post(memory_routes::handle_store_memory),
        )
        .route(
            "/api/memory/search",
            post(memory_routes::handle_search_memory),
        )
        .route(
            "/api/memory/confirm/{source_id}",
            post(memory_routes::handle_confirm_memory),
        )
        .route(
            "/api/memory/list",
            post(memory_routes::handle_list_memories),
        )
        .route(
            "/api/memory/delete/{source_id}",
            delete(memory_routes::handle_delete_memory),
        )
        // Memory reclassify
        .route(
            "/api/memory/reclassify/{source_id}",
            post(memory_routes::handle_reclassify_memory),
        )
        // Enrichment status
        .route(
            "/api/memory/{source_id}/enrichment-status",
            get(memory_routes::handle_get_enrichment_status),
        )
        // Pending revisions
        .route(
            "/api/memory/revision/{id}/accept",
            post(memory_routes::handle_accept_revision),
        )
        .route(
            "/api/memory/revision/{id}/dismiss",
            post(memory_routes::handle_dismiss_revision),
        )
        // Contradiction flags
        .route(
            "/api/memory/contradiction/{source_id}/dismiss",
            post(memory_routes::handle_dismiss_contradiction),
        )
        // Knowledge graph
        .route(
            "/api/memory/entities",
            post(memory_routes::handle_create_entity),
        )
        .route(
            "/api/memory/relations",
            post(memory_routes::handle_create_relation),
        )
        .route(
            "/api/memory/observations",
            post(memory_routes::handle_add_observation),
        )
        .route(
            "/api/memory/link-entity",
            post(memory_routes::handle_link_entity),
        )
        // Profile & Agents
        .route(
            "/api/profile",
            get(memory_routes::handle_get_profile).put(memory_routes::handle_update_profile),
        )
        .route("/api/agents", get(memory_routes::handle_list_agents))
        .route(
            "/api/agents/{name}",
            get(memory_routes::handle_get_agent)
                .put(memory_routes::handle_update_agent)
                .delete(memory_routes::handle_delete_agent),
        )
        // Knowledge graph retrieval + stats
        .route(
            "/api/memory/entities/list",
            post(memory_routes::handle_list_entities),
        )
        .route(
            "/api/memory/entities/search",
            post(memory_routes::handle_search_entities),
        )
        .route(
            "/api/memory/entities/{entity_id}",
            get(memory_routes::handle_get_entity_detail),
        )
        .route(
            "/api/memory/stats",
            get(memory_routes::handle_get_memory_stats),
        )
        .route("/api/home-stats", get(memory_routes::handle_get_home_stats))
        // Entity suggestions
        .route(
            "/api/memory/entity-suggestions",
            get(memory_routes::handle_get_entity_suggestions),
        )
        // Nurture cards
        .route(
            "/api/memory/nurture",
            get(memory_routes::handle_get_nurture_cards),
        )
        // Spaces
        .route(
            "/api/spaces",
            get(memory_routes::handle_list_spaces).post(memory_routes::handle_create_space),
        )
        .route(
            "/api/spaces/default",
            get(memory_routes::handle_get_default_space)
                .put(memory_routes::handle_set_default_space)
                .delete(memory_routes::handle_clear_default_space),
        )
        .route(
            "/api/spaces/{name}",
            put(memory_routes::handle_update_space).delete(memory_routes::handle_delete_space),
        )
        .route(
            "/api/spaces/{from}/move-to/{to}",
            post(memory_routes::handle_move_space),
        )
        // Pages (legacy SQL tables still named "concepts" — see db.rs)
        .route(
            "/api/pages",
            get(memory_routes::handle_list_pages).post(memory_routes::handle_create_page),
        )
        .route(
            "/api/pages/search",
            post(memory_routes::handle_search_pages),
        )
        .route(
            "/api/pages/export",
            post(memory_routes::handle_export_pages),
        )
        .route(
            "/api/pages/recent-changes",
            get(routes::handle_recent_page_changes),
        )
        .route(
            "/api/pages/orphan-links",
            get(memory_routes::handle_list_orphan_links),
        )
        .route(
            "/api/pages/{id}/export",
            post(memory_routes::handle_export_page),
        )
        .route(
            "/api/pages/{id}",
            get(memory_routes::handle_get_page)
                .put(memory_routes::handle_refresh_page)
                .delete(memory_routes::handle_delete_page),
        )
        .route(
            "/api/pages/{id}/sources",
            get(memory_routes::handle_get_page_sources),
        )
        .route(
            "/api/pages/{id}/links",
            get(memory_routes::handle_get_page_links),
        )
        .route(
            "/api/pages/{id}/archive",
            post(memory_routes::handle_archive_page),
        )
        .route(
            "/api/pages/{id}/revisions",
            get(memory_routes::handle_get_page_revisions),
        )
        // Page Map (mind-map v1, stage 2)
        .route(
            "/api/pages/{id}/map",
            get(page_map_routes::handle_get_page_map)
                .delete(page_map_routes::handle_reset_page_map),
        )
        .route(
            "/api/pages/{id}/map/layout",
            put(page_map_routes::handle_put_page_map_layout),
        )
        .route(
            "/api/pages/{id}/map/nodes",
            post(page_map_routes::handle_create_map_node),
        )
        .route(
            "/api/pages/{id}/map/nodes/{node_id}",
            patch(page_map_routes::handle_patch_map_node)
                .delete(page_map_routes::handle_delete_map_node),
        )
        .route(
            "/api/pages/{id}/map/edges",
            post(page_map_routes::handle_create_map_edge),
        )
        .route(
            "/api/pages/{id}/map/edges/{edge_id}",
            patch(page_map_routes::handle_patch_map_edge)
                .delete(page_map_routes::handle_delete_map_edge),
        )
        .route(
            "/api/pages/{id}/map/improve",
            post(page_map_routes::handle_improve_page_map),
        )
        .route(
            "/api/memory/{id}/revisions",
            get(memory_routes::handle_get_memory_revisions),
        )
        // Rejections
        .route(
            "/api/memory/rejections",
            get(memory_routes::handle_get_rejections),
        );
    let router = refinery_routes::register(router);
    let router = source_routes::register(router);
    let router = config_routes::register(router);

    let router = router
        // Indexed files / chunks (batch 2)
        .route(
            "/api/indexed-files",
            get(memory_routes::handle_list_indexed_files),
        )
        .route(
            "/api/chunks/{source_id}",
            get(memory_routes::handle_get_chunks),
        )
        .route(
            "/api/chunks/{id}/update",
            put(memory_routes::handle_update_chunk),
        )
        .route(
            "/api/chunks/time-range",
            delete(memory_routes::handle_delete_by_time_range),
        )
        .route(
            "/api/chunks/delete-bulk",
            post(memory_routes::handle_delete_bulk),
        )
        // Entity/observation CRUD (batch 3)
        .route(
            "/api/memory/entities/{id}/confirm",
            put(memory_routes::handle_confirm_entity),
        )
        .route(
            "/api/memory/entities/{id}/delete",
            delete(memory_routes::handle_delete_entity),
        )
        .route(
            "/api/memory/entities/{entity_id}/observations",
            post(memory_routes::handle_add_entity_observation),
        )
        .route(
            "/api/memory/observations/{id}",
            put(memory_routes::handle_update_observation)
                .delete(memory_routes::handle_delete_observation),
        )
        .route(
            "/api/memory/observations/{id}/confirm",
            put(memory_routes::handle_confirm_observation),
        )
        // Space extended (batch 4)
        .route(
            "/api/spaces/{name}/pin",
            post(memory_routes::handle_pin_space),
        )
        .route(
            "/api/spaces/{name}/confirm",
            post(memory_routes::handle_confirm_space),
        )
        .route(
            "/api/spaces/reorder",
            post(memory_routes::handle_reorder_space),
        )
        .route(
            "/api/spaces/{name}/star",
            post(memory_routes::handle_toggle_space_starred),
        )
        .route(
            "/api/documents/{source_id}/space",
            post(memory_routes::handle_set_document_space),
        )
        // Activity, tags, capture stats, memory detail (batch 5)
        .route(
            "/api/activities",
            get(memory_routes::handle_list_activities),
        )
        .route("/api/tags", get(memory_routes::handle_list_tags))
        .route("/api/tags/{name}", delete(memory_routes::handle_delete_tag))
        .route("/api/suggest-tags", get(memory_routes::handle_suggest_tags))
        .route(
            "/api/documents/{source_id}/tags",
            put(memory_routes::handle_set_document_tags),
        )
        .route(
            "/api/capture-stats",
            get(memory_routes::handle_capture_stats),
        )
        .route(
            "/api/memory/{id}/detail",
            get(memory_routes::handle_get_memory_detail),
        )
        .route(
            "/api/memory/by-ids",
            get(memory_routes::handle_get_memories_by_ids),
        )
        .route(
            "/api/memory/{id}/versions",
            get(memory_routes::handle_get_version_chain),
        )
        .route(
            "/api/memory/{id}/update",
            put(memory_routes::handle_update_memory),
        )
        .route(
            "/api/memory/{id}/stability",
            put(memory_routes::handle_set_stability),
        )
        .route(
            "/api/memory/{id}/correct",
            post(memory_routes::handle_correct_memory),
        )
        // Decisions, briefing, working memory, profile narrative, pinned (batch 6)
        .route("/api/decisions", get(memory_routes::handle_list_decisions))
        // HTTP path stays "domains" for one-release back-compat; rename to "spaces" in PR-A+1.
        .route(
            "/api/decisions/domains",
            get(memory_routes::handle_list_decision_domains),
        )
        .route("/api/briefing", get(memory_routes::handle_get_briefing))
        .route(
            "/api/profile/narrative",
            get(memory_routes::handle_get_profile_narrative),
        )
        .route(
            "/api/profile/narrative/regenerate",
            post(memory_routes::handle_regenerate_narrative),
        )
        .route(
            "/api/memory/pinned",
            get(memory_routes::handle_list_pinned_memories),
        )
        .route(
            "/api/memory/{id}/pin",
            post(memory_routes::handle_pin_memory),
        )
        .route(
            "/api/memory/{id}/unpin",
            post(memory_routes::handle_unpin_memory),
        )
        .route(
            "/api/memory/pending-revisions",
            get(memory_routes::handle_list_pending_revisions),
        )
        .route(
            "/api/memory/pending-revision/{source_id}",
            get(memory_routes::handle_get_pending_revision),
        )
        .route("/api/snapshots", get(memory_routes::handle_list_snapshots))
        .route(
            "/api/snapshots/{id}/captures",
            get(memory_routes::handle_get_snapshot_captures),
        )
        .route(
            "/api/snapshots/{id}/captures-with-content",
            get(memory_routes::handle_get_snapshot_captures_with_content),
        )
        .route(
            "/api/snapshots/{id}/delete",
            post(memory_routes::handle_delete_snapshot),
        )
        .route(
            "/api/memory/{id}/update-page",
            post(memory_routes::handle_update_page),
        );
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
