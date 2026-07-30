// SPDX-License-Identifier: Apache-2.0
//! Library target for integration-testing the HTTP router.
//!
//! Only the modules needed by `tests/` are re-exported here. The binary
//! entry-point (`main.rs`) continues to own the daemon lifecycle.

pub mod activity_tag_routes;
pub mod brief_files;
pub mod brief_routes;
pub mod briefing_routes;
pub mod community_routes;
pub mod config_routes;
pub mod decisions_routes;
pub mod entity_graph_routes;
pub mod error;
mod host_activity;
pub mod import_routes;
pub mod indexed_files_routes;
pub mod ingest_batcher;
pub mod ingest_routes;
pub mod knowledge_routes;
pub mod lifecycle;
pub mod lint_routes;
pub mod maintenance_coordinator;
pub mod memory_detail_routes;
pub mod memory_routes;
pub mod onboarding_routes;
pub mod page_map_routes;
pub mod profile_agents_routes;
pub mod read_scope;
pub mod refinery_routes;
pub mod reflection_debounce;
pub mod repair_routes;
mod route_registry;
pub mod router;
pub mod routes;
pub mod runtime_observation;
pub mod scheduler;
pub mod security;
pub mod sensitive_read_routes;
pub mod source_routes;
pub mod space_header;
pub mod spaces_routes;
pub mod state;
pub mod truth_guard;
pub mod websocket;

#[cfg(test)]
#[path = "brief_files_test.rs"]
mod brief_files_test;

#[cfg(test)]
#[path = "lint_endpoint_test.rs"]
mod lint_endpoint_test;

#[cfg(test)]
#[path = "repair_endpoint_test.rs"]
mod repair_endpoint_test;

#[cfg(test)]
#[path = "truth_error_seam_test.rs"]
mod truth_error_seam_test;

/// Shared mutex for tests that mutate the process-wide `WENLAN_DATA_DIR` env
/// var. Rust tests run in parallel by default, so any test that swaps this env
/// var must hold this single crate-level lock for the full guard lifetime.
#[cfg(test)]
pub(crate) static TEST_DATA_DIR_LOCK: std::sync::OnceLock<tokio::sync::Mutex<()>> =
    std::sync::OnceLock::new();

// cmd_* modules contain main-only CLI logic; not re-exported.
