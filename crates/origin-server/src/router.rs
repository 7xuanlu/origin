// SPDX-License-Identifier: Apache-2.0
//! Axum router construction — wires all HTTP and WebSocket routes.

use crate::state::SharedState;
use crate::{
    config_routes, import_routes, ingest_routes, knowledge_routes, memory_routes,
    onboarding_routes, routes, source_routes, websocket,
};
use axum::{
    routing::{delete, get, post, put},
    Router,
};
use tower_http::cors::{Any, CorsLayer};

/// Build the shared application router with all routes.
pub fn build_router(state: SharedState) -> Router {
    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    Router::new()
        // General
        .route("/api/health", get(routes::handle_health))
        .route("/api/status", get(routes::handle_status))
        .route("/api/search", post(routes::handle_search))
        .route("/api/context", post(routes::handle_context))
        .route("/api/chat-context", post(routes::handle_chat_context))
        .route("/api/ping", get(routes::handle_ping))
        .route("/api/shutdown", post(routes::handle_shutdown))
        .route("/api/debug/pipeline", get(routes::handle_pipeline_status))
        .route(
            "/api/retrievals/recent",
            get(routes::handle_recent_retrievals),
        )
        .route(
            "/api/memory/recent",
            get(memory_routes::handle_recent_memories),
        )
        .route(
            "/api/memory/unconfirmed",
            get(memory_routes::handle_list_unconfirmed_memories),
        )
        .route("/api/concepts/recent", get(routes::handle_recent_concepts))
        .route("/api/steep", post(routes::handle_steep))
        .route("/api/distill", post(routes::handle_distill))
        .route("/api/distill/{concept_id}", post(routes::handle_redistill))
        .route(
            "/api/eval/distillation",
            get(routes::handle_distillation_eval),
        )
        // Ingest
        .route("/api/ingest/text", post(ingest_routes::handle_ingest_text))
        .route(
            "/api/ingest/webpage",
            post(ingest_routes::handle_ingest_webpage),
        )
        .route(
            "/api/ingest/memory",
            post(ingest_routes::handle_ingest_memory),
        )
        .route(
            "/api/documents/{source}/{source_id}",
            delete(ingest_routes::handle_delete_document),
        )
        // Import
        .route(
            "/api/import/memories",
            post(import_routes::handle_import_memories),
        )
        .route(
            "/api/import/chat-export",
            post(import_routes::handle_chat_export_import),
        )
        .route(
            "/api/import/state",
            get(import_routes::handle_list_pending_imports),
        )
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
        .route(
            "/api/memory/{source_id}/enrichment-status/retry",
            post(memory_routes::handle_retry_enrichment),
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
        .route(
            "/api/memory/entity-suggestions/{id}/approve",
            post(memory_routes::handle_approve_entity_suggestion),
        )
        .route(
            "/api/memory/entity-suggestions/{id}/dismiss",
            post(memory_routes::handle_dismiss_entity_suggestion),
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
            "/api/spaces/{name}",
            put(memory_routes::handle_update_space).delete(memory_routes::handle_delete_space),
        )
        // Concepts
        .route(
            "/api/concepts",
            get(memory_routes::handle_list_concepts).post(memory_routes::handle_create_concept),
        )
        .route(
            "/api/concepts/search",
            post(memory_routes::handle_search_concepts),
        )
        .route(
            "/api/concepts/export",
            post(memory_routes::handle_export_concepts),
        )
        .route(
            "/api/concepts/recent-changes",
            get(routes::handle_recent_concept_changes),
        )
        .route(
            "/api/concepts/{id}/export",
            post(memory_routes::handle_export_concept),
        )
        .route(
            "/api/concepts/{id}",
            get(memory_routes::handle_get_concept).delete(memory_routes::handle_delete_concept),
        )
        .route(
            "/api/concepts/{id}/sources",
            get(memory_routes::handle_get_concept_sources),
        )
        .route(
            "/api/concepts/{id}/archive",
            post(memory_routes::handle_archive_concept),
        )
        // Rejections
        .route(
            "/api/memory/rejections",
            get(memory_routes::handle_get_rejections),
        )
        // Sources
        .route(
            "/api/sources",
            get(source_routes::handle_list_sources).post(source_routes::handle_add_source),
        )
        .route(
            "/api/sources/{id}",
            delete(source_routes::handle_remove_source),
        )
        .route(
            "/api/sources/{id}/sync",
            post(source_routes::handle_sync_source),
        )
        // Config
        .route(
            "/api/config",
            get(config_routes::handle_get_config).put(config_routes::handle_update_config),
        )
        .route(
            "/api/config/skip-apps",
            get(config_routes::handle_get_skip_apps).put(config_routes::handle_update_skip_apps),
        )
        // On-device model (list + download-and-load)
        .route(
            "/api/on-device-model",
            get(config_routes::handle_get_on_device_model),
        )
        .route(
            "/api/on-device-model/download",
            post(config_routes::handle_download_on_device_model),
        )
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
            "/api/memory/{id}/update-concept",
            post(memory_routes::handle_update_concept),
        )
        // Knowledge directory
        .route(
            "/api/knowledge/path",
            get(knowledge_routes::handle_get_knowledge_path),
        )
        .route(
            "/api/knowledge/count",
            get(knowledge_routes::handle_get_knowledge_count),
        )
        .route(
            "/api/knowledge/recent-relations",
            get(knowledge_routes::handle_list_recent_relations),
        )
        // Onboarding milestones
        .route(
            "/api/onboarding/milestones",
            get(onboarding_routes::handle_list_milestones),
        )
        .route(
            "/api/onboarding/milestones/{id}/acknowledge",
            post(onboarding_routes::handle_acknowledge_milestone),
        )
        .route(
            "/api/onboarding/reset",
            post(onboarding_routes::handle_reset_milestones),
        )
        // WebSocket
        .route("/ws/updates", get(websocket::handle_ws_upgrade))
        .layer(cors)
        .with_state(state)
}
