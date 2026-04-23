// SPDX-License-Identifier: Apache-2.0
//! Core business logic for the Origin memory system.
//!
//! This crate contains memory storage, retrieval, embeddings, LLM processing,
//! and all non-UI logic. It is being extracted from the Tauri app crate
//! incrementally; the current set of modules is the zero-dependency slice
//! moved in phase 3a.

pub use origin_types;

pub mod access_tracker;
pub mod activity;
pub mod briefing;
pub mod cache;
pub mod chat_import;
pub mod chunker;
pub mod classify;
pub mod concepts;
pub mod config;
pub mod context_packager;
pub mod contradiction;
pub mod db;
pub mod decay;
pub mod engine;
pub mod error;
pub mod eval;
pub mod events;
pub mod export;
pub mod extract;
pub mod importer;
pub mod kg_quality;
pub mod llm_classifier;
pub mod llm_provider;
pub mod memory_schema;
pub mod merge;
pub mod narrative;
pub mod on_device_models;
pub mod onboarding;
pub mod post_ingest;
pub mod privacy;
pub mod prompts;
pub mod quality_gate;
pub mod refinery;
pub mod rerank;
pub mod router;
pub mod schema;
pub mod sources;
pub mod spaces;
pub mod system_info;
pub mod tags;
pub mod topic_match;
pub mod tuning;
pub mod working_memory;

// Re-exports for convenience.
pub use error::OriginError;
pub use events::{EventEmitter, NoopEmitter};

/// Crate version.
pub fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn version_is_set() {
        assert!(!version().is_empty());
    }
}
