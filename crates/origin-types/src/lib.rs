// SPDX-License-Identifier: Apache-2.0
//! Shared types for the Origin memory system.
//!
//! This crate provides lightweight type definitions shared across
//! origin-core, origin-server, and the Tauri app. Dependencies are
//! limited to serde and serde_json -- no heavy runtime deps.

pub mod entities;
pub mod import;
pub mod memory;
pub mod requests;
pub mod responses;
pub mod sources;

// Re-export commonly used types at crate root for convenience.
pub use entities::{
    Entity, EntityDetail, EntitySearchResult, EntitySuggestion, Observation, RecentRelation,
    Relation, RelationWithEntity,
};
pub use memory::{
    ActivityBadge, ActivityKind, AgentActivityRow, AgentConnection, ConceptChange,
    ConceptChangeKind, DomainInfo, HomeStats, IndexedFileInfo, MemoryItem, MemoryStats,
    MemoryVersionItem, Profile, RecentActivityItem, RejectionRecord, RetrievalEvent, SearchResult,
    SessionSnapshot, SnapshotCapture, SnapshotCaptureWithContent, Space, TopMemory, TypeBreakdown,
};
pub use sources::{MemoryType, RawDocument, SourceType, StabilityTier, SyncStatus};

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

    #[test]
    fn memory_type_roundtrip() {
        for variant in [
            MemoryType::Identity,
            MemoryType::Preference,
            MemoryType::Decision,
            MemoryType::Fact,
            MemoryType::Goal,
        ] {
            let s = variant.to_string();
            let parsed: MemoryType = s.parse().unwrap();
            assert_eq!(parsed, variant);
        }
    }

    #[test]
    fn search_result_serializes() {
        let sr = SearchResult {
            id: "1".into(),
            content: "test".into(),
            source: "memory".into(),
            source_id: "mem_abc".into(),
            title: "Test".into(),
            url: None,
            chunk_index: 0,
            last_modified: 1000,
            score: 0.9,
            chunk_type: None,
            language: None,
            semantic_unit: None,
            memory_type: Some("fact".into()),
            domain: None,
            source_agent: None,
            confidence: Some(0.8),
            confirmed: Some(true),
            stability: None,
            supersedes: None,
            summary: None,
            entity_id: None,
            entity_name: None,
            quality: None,
            is_archived: false,
            is_recap: false,
            structured_fields: None,
            retrieval_cue: None,
            source_text: None,
            raw_score: 0.0,
        };
        let json = serde_json::to_string(&sr).unwrap();
        assert!(json.contains("mem_abc"));
        // Verify skip_serializing_if works: None fields should be absent
        assert!(!json.contains("entity_id"));
    }

    #[test]
    fn raw_document_default() {
        let doc = RawDocument::default();
        assert_eq!(doc.enrichment_status, "raw");
        assert_eq!(doc.supersede_mode, "hide");
        assert!(!doc.pending_revision);
        assert!(!doc.is_recap);
    }

    #[test]
    fn stability_tier_mapping() {
        use sources::stability_tier;
        assert_eq!(stability_tier(Some("identity")), StabilityTier::Protected);
        assert_eq!(stability_tier(Some("preference")), StabilityTier::Protected);
        assert_eq!(stability_tier(Some("fact")), StabilityTier::Standard);
        assert_eq!(stability_tier(Some("decision")), StabilityTier::Standard);
        assert_eq!(stability_tier(Some("goal")), StabilityTier::Ephemeral);
        assert_eq!(stability_tier(None), StabilityTier::Ephemeral);
    }
}

#[cfg(test)]
mod retrieval_event_tests {
    use super::*;

    #[test]
    fn retrieval_event_roundtrips() {
        let e = RetrievalEvent {
            timestamp_ms: 1_700_000_000_000,
            agent_name: "claude-code".into(),
            query: Some("origin positioning".into()),
            concept_titles: vec!["Origin positioning".into(), "Daemon architecture".into()],
            concept_ids: vec![],
            memory_snippets: vec![],
        };
        let s = serde_json::to_string(&e).unwrap();
        let back: RetrievalEvent = serde_json::from_str(&s).unwrap();
        assert_eq!(back.agent_name, "claude-code");
        assert_eq!(back.concept_titles.len(), 2);
        assert_eq!(back.query.as_deref(), Some("origin positioning"));
    }

    #[test]
    fn retrieval_event_omits_none_query() {
        let e = RetrievalEvent {
            timestamp_ms: 1_700_000_000_000,
            agent_name: "claude-code".into(),
            query: None,
            concept_titles: vec![],
            concept_ids: vec![],
            memory_snippets: vec![],
        };
        let s = serde_json::to_string(&e).unwrap();
        assert!(
            !s.contains("\"query\""),
            "expected None query to be skipped on the wire, got: {s}",
        );
        let back: RetrievalEvent = serde_json::from_str(&s).unwrap();
        assert_eq!(back.query, None);
        assert!(back.concept_titles.is_empty());
    }

    #[test]
    fn concept_change_roundtrips() {
        let c = ConceptChange {
            concept_id: "concept_abc".into(),
            title: "Wiki-style prose concepts".into(),
            change_kind: ConceptChangeKind::Revised,
            changed_at_ms: 1_700_000_000_000,
        };
        let s = serde_json::to_string(&c).unwrap();
        assert!(
            s.contains("\"change_kind\":\"revised\""),
            "expected snake_case change_kind on the wire, got: {s}",
        );
        let back: ConceptChange = serde_json::from_str(&s).unwrap();
        assert_eq!(back.change_kind, ConceptChangeKind::Revised);
    }
}
