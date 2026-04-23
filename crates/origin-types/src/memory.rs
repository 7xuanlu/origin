// SPDX-License-Identifier: Apache-2.0
//! Core memory data types — search results, items, stats, profiles, agents, spaces.

use serde::{Deserialize, Serialize};

/// A search result from hybrid (vector + FTS) search.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    pub id: String,
    pub content: String,
    pub source: String,
    pub source_id: String,
    pub title: String,
    pub url: Option<String>,
    pub chunk_index: i32,
    pub last_modified: i64,
    pub score: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub chunk_type: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub language: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub semantic_unit: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub memory_type: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub domain: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source_agent: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub confidence: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub confirmed: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stability: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub supersedes: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub summary: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub entity_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub entity_name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub quality: Option<String>,
    #[serde(default)]
    pub is_archived: bool,
    #[serde(default)]
    pub is_recap: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub structured_fields: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub retrieval_cue: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source_text: Option<String>,
    /// Raw RRF score before normalization -- for absolute relevance gating.
    #[serde(default)]
    pub raw_score: f32,
}

/// A full memory item with all metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryItem {
    pub source_id: String,
    pub title: String,
    pub content: String,
    pub summary: Option<String>,
    pub memory_type: Option<String>,
    pub domain: Option<String>,
    pub source_agent: Option<String>,
    pub confidence: Option<f32>,
    pub confirmed: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stability: Option<String>,
    pub pinned: bool,
    pub supersedes: Option<String>,
    pub last_modified: i64,
    pub chunk_count: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub entity_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub quality: Option<String>,
    #[serde(default)]
    pub is_recap: bool,
    pub enrichment_status: String,
    pub supersede_mode: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub structured_fields: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub retrieval_cue: Option<String>,
    pub access_count: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source_text: Option<String>,
    #[serde(default = "default_version")]
    pub version: i64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub changelog: Option<String>,
}

fn default_version() -> i64 {
    1
}

/// A single item in a version chain.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryVersionItem {
    pub source_id: String,
    pub title: String,
    pub content: String,
    pub memory_type: Option<String>,
    pub confirmed: bool,
    pub supersedes: Option<String>,
    pub last_modified: i64,
}

/// Aggregate memory statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryStats {
    pub total: u64,
    pub new_today: u64,
    pub confirmed: u64,
    pub domains: Vec<DomainInfo>,
    #[serde(default)]
    pub by_type: Vec<TypeBreakdown>,
    #[serde(default)]
    pub entity_linked: u64,
    #[serde(default)]
    pub enrichment_pending: u64,
}

/// Count breakdown by memory type.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TypeBreakdown {
    pub memory_type: String,
    pub count: u64,
}

/// Count breakdown by domain.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DomainInfo {
    pub name: String,
    pub count: u64,
}

/// File/document info as shown in list views.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexedFileInfo {
    pub source_id: String,
    pub title: String,
    pub source: String,
    pub url: Option<String>,
    pub chunk_count: u64,
    pub last_modified: i64,
    pub summary: Option<String>,
    #[serde(default)]
    pub processing: bool,
    pub memory_type: Option<String>,
    pub domain: Option<String>,
    pub source_agent: Option<String>,
    pub confidence: Option<f32>,
    pub confirmed: Option<bool>,
    pub stability: Option<String>,
    pub pinned: bool,
}

/// Home dashboard statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HomeStats {
    pub total: u64,
    pub new_today: u64,
    pub confirmed: u64,
    pub total_ingested: u64,
    pub active_insights: u64,
    pub distilled_today: u64,
    pub distilled_all: u64,
    pub sources_archived: u64,
    pub times_served_today: u64,
    pub words_saved_today: u64,
    pub times_served_week: u64,
    pub words_saved_week: u64,
    pub times_served_all: u64,
    pub words_saved_all: u64,
    pub corrections_active: u64,
    pub top_memories: Vec<TopMemory>,
}

/// A top-accessed memory for the home dashboard.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopMemory {
    pub source_id: String,
    pub content: String,
    pub memory_type: Option<String>,
    pub domain: Option<String>,
    pub times_retrieved: u64,
}

/// A session snapshot — a compact summary of a contiguous working session.
///
/// Wire format for `GET /api/snapshots`. Mirrors `SessionSnapshotRow` in
/// origin-core/db.rs but lives here so origin-types can stay the single
/// source of truth for HTTP boundary shapes (and because capture_count is
/// serialized as a JSON number rather than a Rust `usize`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionSnapshot {
    pub id: String,
    pub activity_id: String,
    pub started_at: i64,
    pub ended_at: i64,
    pub primary_apps: Vec<String>,
    pub summary: String,
    pub tags: Vec<String>,
    pub capture_count: u64,
}

/// A single capture belonging to a snapshot.
///
/// Wire format for `GET /api/snapshots/{id}/captures`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SnapshotCapture {
    pub source_id: String,
    pub app_name: String,
    pub window_title: String,
    pub timestamp: i64,
    pub source: String,
}

/// A snapshot capture enriched with full chunk content + LLM summary.
///
/// Wire format for `GET /api/snapshots/{id}/captures-with-content`. The
/// frontend uses this to render the snapshot detail panel.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SnapshotCaptureWithContent {
    pub source_id: String,
    pub app_name: String,
    pub window_title: String,
    pub timestamp: i64,
    pub source: String,
    pub content: String,
    pub summary: Option<String>,
}

/// User profile.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Profile {
    pub id: String,
    pub name: String,
    pub display_name: Option<String>,
    pub email: Option<String>,
    pub bio: Option<String>,
    pub avatar_path: Option<String>,
    pub created_at: i64,
    pub updated_at: i64,
}

/// An agent connection record.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentConnection {
    pub id: String,
    /// Canonical technical identifier (lowercase, hyphen-case). Matches the
    /// `x-agent-name` HTTP header sent by the client. This is the value used
    /// for attribution and filtering — treat it as the primary key.
    pub name: String,
    /// Human-readable name shown in UI. If None, the frontend falls back to
    /// `KNOWN_CLIENT_DISPLAY_NAMES[name]` and then to `name` itself.
    #[serde(default)]
    pub display_name: Option<String>,
    pub agent_type: String,
    pub description: Option<String>,
    pub enabled: bool,
    pub trust_level: String,
    pub last_seen_at: Option<i64>,
    pub memory_count: i64,
    pub created_at: i64,
    pub updated_at: i64,
}

/// An agent activity log entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentActivityRow {
    pub id: i64,
    pub timestamp: i64,
    pub agent_name: String,
    pub action: String,
    pub memory_ids: Option<String>,
    pub query: Option<String>,
    pub detail: Option<String>,
    pub memory_titles: Vec<String>,
}

/// A space (domain grouping).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Space {
    pub id: String,
    pub name: String,
    pub description: Option<String>,
    pub suggested: bool,
    pub starred: bool,
    pub sort_order: i64,
    pub memory_count: u64,
    pub entity_count: u64,
    pub created_at: f64,
    pub updated_at: f64,
}

/// A rejected memory entry for quality gate diagnostics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RejectionRecord {
    pub id: String,
    pub content: String,
    pub source_agent: Option<String>,
    pub rejection_reason: String,
    pub rejection_detail: Option<String>,
    pub similarity_score: Option<f64>,
    pub similar_to_source_id: Option<String>,
    pub created_at: i64,
}

/// An event describing when an agent retrieved concepts/memories from Origin.
///
/// Backs Zone 4 of the home page ("Where Claude leaned on you") — a proof
/// surface showing which concepts were pulled into an agent's context and
/// when, giving the user evidence their curated knowledge is in use.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RetrievalEvent {
    pub timestamp_ms: i64,
    pub agent_name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub query: Option<String>,
    #[serde(default)]
    pub concept_titles: Vec<String>,
    /// Stable concept IDs corresponding 1:1 with `concept_titles`.
    /// Used by the UI to navigate directly by ID rather than doing a
    /// fragile title lookup. Empty on legacy events recorded before this
    /// field was added; the UI falls back to the title-match path in that case.
    #[serde(default)]
    pub concept_ids: Vec<String>,
    #[serde(default)]
    pub memory_snippets: Vec<String>,
}

/// The kind of change that happened to a concept.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ConceptChangeKind {
    Created,
    Revised,
    Merged,
}

/// A change event for a concept — feeds the home page delta zones.
///
/// Backs Zones 1 and 3 of the home page: surfacing newly created, revised,
/// or merged concepts so the user sees their knowledge base evolving.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ConceptChange {
    pub concept_id: String,
    pub title: String,
    pub change_kind: ConceptChangeKind,
    pub changed_at_ms: i64,
}

/// The kind of item in a recent-activity feed entry.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ActivityKind {
    Concept,
    Memory,
}

/// A badge summarising what changed since the user last saw an item.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ActivityBadge {
    New,
    Revised,
    Refined,
    Growing { added: u32 },
    NeedsReview,
    None,
}

/// A single entry in the home-page recent-activity feed.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct RecentActivityItem {
    pub kind: ActivityKind,
    pub id: String,
    pub title: String,
    pub snippet: Option<String>,
    pub timestamp_ms: u64,
    pub badge: ActivityBadge,
}

#[cfg(test)]
mod tests {
    #[test]
    fn activity_badge_serializes_as_tagged_enum() {
        use super::ActivityBadge;
        assert_eq!(
            serde_json::to_string(&ActivityBadge::New).unwrap(),
            r#"{"kind":"new"}"#
        );
        assert_eq!(
            serde_json::to_string(&ActivityBadge::Refined).unwrap(),
            r#"{"kind":"refined"}"#
        );
        assert_eq!(
            serde_json::to_string(&ActivityBadge::Revised).unwrap(),
            r#"{"kind":"revised"}"#
        );
        assert_eq!(
            serde_json::to_string(&ActivityBadge::NeedsReview).unwrap(),
            r#"{"kind":"needs_review"}"#
        );
        assert_eq!(
            serde_json::to_string(&ActivityBadge::None).unwrap(),
            r#"{"kind":"none"}"#
        );
        assert_eq!(
            serde_json::to_string(&ActivityBadge::Growing { added: 3 }).unwrap(),
            r#"{"kind":"growing","added":3}"#
        );
    }

    #[test]
    fn recent_activity_item_round_trips() {
        use super::{ActivityBadge, ActivityKind, RecentActivityItem};
        let item = RecentActivityItem {
            kind: ActivityKind::Memory,
            id: "mem_abc".into(),
            title: "A memory".into(),
            snippet: Some("Snippet text".into()),
            timestamp_ms: 1_776_000_000_000,
            badge: ActivityBadge::New,
        };
        let s = serde_json::to_string(&item).unwrap();
        let back: RecentActivityItem = serde_json::from_str(&s).unwrap();
        assert_eq!(back.id, "mem_abc");
        assert!(matches!(back.kind, ActivityKind::Memory));
        assert!(matches!(back.badge, ActivityBadge::New));
    }

    #[test]
    fn retrieval_event_includes_memory_snippets() {
        use super::RetrievalEvent;
        let evt = RetrievalEvent {
            timestamp_ms: 1,
            agent_name: "claude-code".into(),
            query: None,
            concept_titles: vec![],
            concept_ids: vec![],
            memory_snippets: vec!["The first line of the memory".into()],
        };
        let s = serde_json::to_string(&evt).unwrap();
        assert!(s.contains("memory_snippets"));
    }
}
