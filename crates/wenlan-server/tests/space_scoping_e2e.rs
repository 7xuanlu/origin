// SPDX-License-Identifier: Apache-2.0

mod common;
mod space_scoping;

#[tokio::test]
async fn global_routes_ignore_space_selectors() {
    space_scoping::global_cases::global_routes_ignore_space_header().await;
}

#[tokio::test]
async fn wave_1_rejects_unknown_selectors() {
    space_scoping::retrieval_cases::unknown_selectors_are_rejected().await;
}

#[tokio::test]
async fn wave_1_honors_primary_and_header_precedence() {
    space_scoping::retrieval_cases::primary_and_header_precedence().await;
}

#[tokio::test]
async fn wave_1_filters_collections_before_limit() {
    space_scoping::retrieval_cases::collections_filter_before_limit().await;
}

#[tokio::test]
async fn wave_1_handles_blank_fallback_and_reserved_collision() {
    space_scoping::retrieval_cases::blank_primary_falls_back_and_reserved_collision_rejects().await;
}

#[tokio::test]
async fn wave_1_ranked_routes_exclude_cross_scope_rows() {
    space_scoping::retrieval_cases::ranked_routes_exclude_cross_scope_rows().await;
}

#[test]
fn wave_1_registry_matches_completed_catalog_contracts() {
    space_scoping::case_runner::assert_wave_1_catalog_contract();
}

#[tokio::test]
async fn wave_2_records_reject_unknown_selectors() {
    space_scoping::record_cases::unknown_selectors_are_rejected().await;
}

#[tokio::test]
async fn wave_2_records_hide_mismatched_direct_ids() {
    space_scoping::record_cases::direct_routes_do_not_disclose_mismatched_ids().await;
}

#[tokio::test]
async fn wave_2_records_filter_collections_before_materialization() {
    space_scoping::record_cases::collections_filter_before_materialization().await;
}

#[tokio::test]
async fn wave_2_records_scope_history_and_chunk_source_priority() {
    space_scoping::record_cases::history_and_chunk_source_priority_are_scoped().await;
}

#[test]
fn wave_2_records_registry_matches_completed_catalog_contracts() {
    space_scoping::record_cases::registry_matches_catalog();
}

#[tokio::test]
async fn wave_2_derived_rejects_unknown_selectors() {
    space_scoping::derived_cases::unknown_selectors_are_rejected().await;
}

#[tokio::test]
async fn wave_2_derived_filters_projection_owners() {
    space_scoping::derived_cases::projections_exclude_cross_scope_and_orphan_owners().await;
}

#[tokio::test]
async fn wave_2_derived_filters_events_before_limit() {
    space_scoping::derived_cases::event_scope_is_applied_before_limit().await;
}

#[test]
fn wave_2_derived_registry_matches_completed_catalog_contracts() {
    space_scoping::derived_cases::registry_matches_completed_contracts();
}

#[tokio::test]
async fn wave_2_parent_collections_reject_unknown_selectors() {
    space_scoping::parent_cases::unknown_selectors_are_rejected().await;
}

#[tokio::test]
async fn wave_2_parent_collections_isolate_briefing_cache() {
    space_scoping::parent_cases::scoped_briefing_does_not_read_or_write_global_cache().await;
}

#[tokio::test]
async fn wave_2_parent_collections_scope_snapshot_membership() {
    space_scoping::parent_cases::snapshot_parent_collections_are_scoped().await;
}

#[test]
fn wave_2_parent_collections_registry_matches_completed_contracts() {
    space_scoping::parent_cases::registry_matches_completed_contracts();
}

#[tokio::test]
async fn wave_3_pages_reject_unknown_selectors() {
    space_scoping::page_cases::unknown_selectors_are_rejected().await;
}

#[tokio::test]
async fn wave_3_pages_scope_collections_and_precedence() {
    space_scoping::page_cases::collections_and_precedence_are_scoped().await;
}

#[tokio::test]
async fn wave_3_pages_filter_ranked_candidates_before_limit() {
    space_scoping::page_cases::ranked_candidates_are_filtered_before_limit().await;
}

#[tokio::test]
async fn wave_3_pages_gate_direct_and_child_routes() {
    space_scoping::page_cases::direct_and_child_routes_are_gated().await;
}

#[test]
fn wave_3_pages_registry_matches_completed_contracts() {
    space_scoping::page_cases::registry_matches_completed_contracts();
}

#[tokio::test]
async fn wave_4_knowledge_rejects_unknown_selectors() {
    space_scoping::knowledge_cases::unknown_selectors_are_rejected().await;
}

#[tokio::test]
async fn wave_4_knowledge_scopes_entity_collections_and_search() {
    space_scoping::knowledge_cases::entity_collections_and_search_are_scoped().await;
}

#[tokio::test]
async fn wave_4_knowledge_gates_detail_and_relation_endpoints() {
    space_scoping::knowledge_cases::detail_and_relation_endpoints_are_scoped().await;
}

#[test]
fn wave_4_knowledge_registry_closes_all_scope_contracts() {
    space_scoping::knowledge_cases::registry_matches_completed_contracts();
}
