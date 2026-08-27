// SPDX-License-Identifier: Apache-2.0
//! Core business logic for the Wenlan memory system.
//!
//! This crate contains memory storage, retrieval, embeddings, LLM processing,
//! and all non-UI logic. It is being extracted from the Tauri app crate
//! incrementally; the current set of modules is the zero-dependency slice
//! moved in phase 3a.

pub use wenlan_types;

pub mod briefing;
pub mod cache;
pub mod chat_import;
pub mod chunker;
pub mod citations;
pub mod claim_judge;
pub mod classify;
pub mod community_grouping;
pub mod community_partition;
pub mod community_routing;
pub mod config;
pub mod contradiction;
pub mod db;
pub mod decay;
mod derived_artifact_state;
pub mod document_enrichment;
#[cfg(test)]
mod drift_guard;
pub mod edge_grounding;
pub mod engine;
pub mod env_compat;
pub mod error;
pub mod eval;
pub mod events;
pub mod export;
pub mod extract;
pub mod faithfulness;
pub mod importer;
pub mod ingest;
pub mod kg;
pub mod kg_quality;
pub mod lint;
pub mod llm_provider;
pub mod m6;
pub mod maintenance;
pub mod migrate_rename;
pub mod migrations;
pub mod narrative;
pub mod on_device_models;
pub mod onboarding;
pub mod origin;
pub mod page_map_improve;
pub mod page_projection_tracker;
pub mod pages;
pub mod post_ingest;
pub mod post_write;
pub mod presence;
pub mod privacy;
pub mod prompts;
pub mod provenance;
pub mod quality_gate;
pub mod read_scope;
pub mod reconcile;
pub mod refinery;
pub mod repair;
pub mod repair_plan;
pub mod reranker;
pub(crate) mod retrieval;
pub mod router;
pub mod schema;
pub mod sources;
pub mod space_context;
pub mod synthesis;
pub mod tags;
pub(crate) mod temporal_query;
pub mod topic_match;
pub mod truth_adapter;
pub mod truth_contract;
pub mod truth_manifest;
pub mod tuning;
pub mod vocab;

// Re-exports for convenience.
pub use error::WenlanError;
pub use events::{EventEmitter, NoopEmitter};

/// Crate version. Release builds report the bare `CARGO_PKG_VERSION`; local
/// source builds append a `+g<sha8>` dev suffix (emitted by build.rs) so the
/// version-drift nudges can tell a dev daemon from a released one.
pub fn version() -> &'static str {
    concat!(env!("CARGO_PKG_VERSION"), env!("WENLAN_VERSION_SUFFIX"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn version_is_set() {
        assert!(!version().is_empty());
    }

    #[test]
    fn version_keeps_semver_prefix() {
        // The dev suffix (if any) is appended, never replaces the base version.
        assert!(version().starts_with(env!("CARGO_PKG_VERSION")));
    }

    // `wenlan-server --version` reports its own crate version via clap and
    // never calls this function, so it cannot prove the WENLAN_RELEASE_VERSION
    // marker build.rs reads (issue #606). This test is the only surface that
    // does: it prints the built-in version for a human comparing the three
    // build scenarios, and asserts the durable shape either way — the bare
    // crate version for a release build, or that version plus a `+g<sha8>`
    // dev suffix for a local build.
    #[test]
    fn version_matches_bare_or_dev_suffix_shape() {
        let v = version();
        eprintln!("wenlan_core::version() = {v}");
        let base = env!("CARGO_PKG_VERSION");
        assert!(
            v == base
                || (v.starts_with(base)
                    && v[base.len()..].starts_with("+g")
                    && v.len() == base.len() + 10),
            "version {v:?} is neither the bare crate version {base:?} nor a `+g<sha8>` dev build"
        );
    }
}
