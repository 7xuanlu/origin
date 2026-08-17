// SPDX-License-Identifier: Apache-2.0
//! Shared types for the Wenlan memory system.
//!
//! This crate provides lightweight type definitions shared across
//! wenlan-core, wenlan-server, and the Tauri app. Dependencies are
//! limited to serde and serde_json -- no heavy runtime deps.

pub mod brand;
pub mod brief;
pub mod briefing;
pub mod communities;
pub mod entities;
pub mod events;
pub mod import;
pub mod lint;
pub mod memory;
pub mod memory_type;
pub mod narrative;
pub mod onboarding;
pub mod outbox;
pub mod page_map;
pub mod pages;
pub mod repair;
pub mod repair_plan;
pub mod requests;
pub mod responses;
pub mod sources;
pub mod space_context;
pub mod system_info;
pub mod working_memory;

// Re-export commonly used types at crate root for convenience.
pub use brief::*;
pub use briefing::{BriefingResponse, ContradictionItem};
pub use communities::{
    CommunityListResponse, CommunityMember, CommunityMemberCursor, CommunityMembersResponse,
    CommunityPageAssignment, CommunityPageAssignmentsResponse, CommunityProposalAcceptResponse,
    CommunityProposalAction, CommunityProposalPayload, CommunityProposalRejectResponse,
    CommunityProposalSummary, CommunityReadScope, CommunitySummary, ListCommunityProposalsResponse,
    PageCommunityAssignmentState,
};
pub use entities::{
    Entity, EntityDetail, EntitySearchResult, EntitySuggestion, GraphMemoryLink, GraphMemoryNode,
    GraphPageLink, GraphPageNode, GraphRef, GraphRelation, KnowledgeGraphResponse, Observation,
    RecentRelation, Relation, RelationWithEntity,
};
pub use lint::{LintCheckResult, LintQuery, LintReport};
pub use memory::{
    ActivityBadge, ActivityKind, AgentActivityRow, AgentConnection, DomainInfo,
    EnrichmentStatusResponse, EnrichmentStepStatus, HomeStats, IndexedFileInfo, MemoryItem,
    MemoryStats, MemoryVersionItem, PageChange, PageChangeKind, Profile, RecentActivityItem,
    RejectionRecord, RetrievalEvent, SearchResult, SessionSnapshot, SnapshotCapture,
    SnapshotCaptureWithContent, Space, TopMemory, TypeBreakdown,
};
pub use memory_type::{MEMORY_TYPE_CAPTURE_DESCRIPTION, MEMORY_TYPE_FILTER_DESCRIPTION};
pub use narrative::NarrativeResponse;
pub use outbox::{
    OutboxDrainDetail, OutboxDrainReport, OutboxEnvelope, OutboxPayload, OUTBOX_SCHEMA,
};
pub use pages::{Page, PageEvidence};
pub use repair::*;
pub use repair_plan::*;
pub use requests::{
    AcceptRefinementRequest, CreatePageDraftRequest, PageDraftVersionRequest,
    UpdatePageDraftRequest,
};
pub use responses::{
    ContradictionDismissResponse, ExportStats, ListMemoryRevisionsResponse,
    ListPageRevisionsResponse, ListRefinementsResponse, MemoryDetail, MemoryRevisionEntry,
    OnDeviceModelEntry, OnDeviceModelResponse, OrphanLink, OrphanLinksResponse, PageChangelogEntry,
    PageDraftResponse, PageWriteResponse, PendingRevision, PendingRevisionItem, ProposalAction,
    RefinementCardAction, RefinementPayload, RefinementProposalSummary, RejectRefinementResponse,
    RevisionAcceptResponse, RevisionDismissResponse,
};
pub use sources::{MemoryType, RawDocument, SourceType, StabilityTier, SyncStatus};
pub use space_context::{WriteOutcome, WriteSpaceSource, WriteSpaceTarget};

use serde::{Deserialize, Serialize};

/// A single revision entry in a memory's changelog (topic-key upsert history).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChangelogEntry {
    pub version: i64,
    /// Unix timestamp of when this revision was written.
    pub at: i64,
    /// Human-readable one-liner describing what changed. May be empty when
    /// the LLM delta hasn't been generated yet (async fill-in).
    pub delta: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source_agent: Option<String>,
    /// The source_id of the incoming memory that triggered this upsert.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub incoming_source_id: Option<String>,
}

/// A link between a page and one of its source memories.
/// (Backed by the `concept_sources` SQL table; rename deferred for back-compat.)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PageSource {
    pub page_id: String,
    pub memory_source_id: String,
    /// Unix timestamp of when this link was created.
    pub linked_at: i64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub link_reason: Option<String>,
}

/// Page source enriched with the memory's metadata (for the API response).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PageSourceWithMemory {
    pub source: PageSource,
    pub memory: Option<crate::memory::MemoryItem>,
}

/// Crate version.
pub fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

#[cfg(test)]
#[path = "repair_tests.rs"]
mod repair_tests;

#[cfg(test)]
#[path = "repair_plan_tests.rs"]
mod repair_plan_tests;

#[cfg(test)]
#[path = "entities_freeze_tests.rs"]
mod entities_freeze_tests;

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
            MemoryType::Lesson,
            MemoryType::Gotcha,
            MemoryType::Fact,
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
            space: None,
            source_agent: None,
            confidence: Some(0.8),
            confirmed: Some(true),
            stability: None,
            supersedes: None,
            summary: None,
            entity_id: None,
            entity_name: None,
            quality: None,
            importance: None,
            event_date: None,
            is_archived: false,
            is_recap: false,
            structured_fields: None,
            retrieval_cue: None,
            source_text: None,
            content_hash: None,
            raw_score: 0.0,
            version: 0,
            pending_revision: false,
            merged_from: None,
            last_delta_summary: None,
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
        assert_eq!(stability_tier(Some("lesson")), StabilityTier::Standard);
        assert_eq!(stability_tier(Some("gotcha")), StabilityTier::Standard);
        // Deprecated: legacy "goal" rows still in DB pre-migration map to
        // Protected via Identity fold (aspirations = identity).
        assert_eq!(stability_tier(Some("goal")), StabilityTier::Protected);
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
            page_titles: vec!["Wenlan positioning".into(), "Daemon architecture".into()],
            page_ids: vec![],
            memory_snippets: vec![],
        };
        let s = serde_json::to_string(&e).unwrap();
        let back: RetrievalEvent = serde_json::from_str(&s).unwrap();
        assert_eq!(back.agent_name, "claude-code");
        assert_eq!(back.page_titles.len(), 2);
        assert_eq!(back.query.as_deref(), Some("origin positioning"));
    }

    #[test]
    fn retrieval_event_omits_none_query() {
        let e = RetrievalEvent {
            timestamp_ms: 1_700_000_000_000,
            agent_name: "claude-code".into(),
            query: None,
            page_titles: vec![],
            page_ids: vec![],
            memory_snippets: vec![],
        };
        let s = serde_json::to_string(&e).unwrap();
        assert!(
            !s.contains("\"query\""),
            "expected None query to be skipped on the wire, got: {s}",
        );
        let back: RetrievalEvent = serde_json::from_str(&s).unwrap();
        assert_eq!(back.query, None);
        assert!(back.page_titles.is_empty());
    }

    #[test]
    fn page_change_roundtrips() {
        let c = PageChange {
            page_id: "page_abc".into(),
            title: "Wiki-style prose pages".into(),
            change_kind: PageChangeKind::Revised,
            changed_at_ms: 1_700_000_000_000,
        };
        let s = serde_json::to_string(&c).unwrap();
        assert!(
            s.contains("\"change_kind\":\"revised\""),
            "expected snake_case change_kind on the wire, got: {s}",
        );
        let back: PageChange = serde_json::from_str(&s).unwrap();
        assert_eq!(back.change_kind, PageChangeKind::Revised);
    }
}
