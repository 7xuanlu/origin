// SPDX-License-Identifier: Apache-2.0
//! API response types for all HTTP endpoints.

use crate::entities::{Entity, EntitySearchResult};
use crate::memory::{IndexedFileInfo, MemoryItem, MemoryStats, SearchResult};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ===== Memory CRUD =====

#[derive(Debug, Serialize, Deserialize)]
pub struct StoreMemoryResponse {
    pub source_id: String,
    pub chunks_created: usize,
    /// Memory type at the moment of persistence. If caller did not supply
    /// one and enrichment is pending, this is a placeholder (`"fact"`) —
    /// check `enrichment` field to know whether to expect it to change.
    pub memory_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub entity_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub quality: Option<String>,
    /// Schema-validation issues — actionable by the agent.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub warnings: Vec<String>,
    /// How structured fields were populated. "agent" | "llm" | "none" | "unknown" (forward-compat default).
    #[serde(default = "default_extraction_method")]
    pub extraction_method: String,
    /// Enrichment state for the memory. `"pending"` when background
    /// classification + entity extraction + concept linking will run;
    /// `"not_needed"` when no LLM is available and the memory stays as
    /// caller-supplied. Machine-readable — Tauri app uses this to drive
    /// polling / live-update UI, MCP callers can choose to relay state.
    /// Defaulted for backward compatibility with pre-async-enrichment clients.
    #[serde(default)]
    pub enrichment: String,
    /// Prose cue for caller agents — safe to relay to the user verbatim.
    /// Communicates that Origin is compiling the memory into reusable
    /// context in the background, so callers don't treat `None` enriched
    /// fields as failure. Empty when the store completed fully sync.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub hint: String,
}

fn default_extraction_method() -> String {
    "unknown".to_string()
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SearchMemoryResponse {
    pub results: Vec<SearchResult>,
    pub took_ms: f64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ListMemoriesResponse {
    pub memories: Vec<IndexedFileInfo>,
}

/// Shared wire format for any `deleted: bool` response.
///
/// Reused by:
/// - `DELETE /api/memory/delete/{id}` (server/memory.rs)
/// - `DELETE /api/documents/{source}/{source_id}` (server/ingest.rs)
#[derive(Debug, Serialize, Deserialize)]
pub struct DeleteResponse {
    pub deleted: bool,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ConfirmResponse {
    pub confirmed: bool,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ReclassifyMemoryResponse {
    pub source_id: String,
    pub memory_type: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct MemoryStatsResponse {
    pub stats: MemoryStats,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct NurtureCardsResponse {
    pub cards: Vec<MemoryItem>,
}

// ===== General search/context =====

#[derive(Debug, Serialize, Deserialize)]
pub struct HealthResponse {
    pub status: String,
    pub db_initialized: bool,
    pub version: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct StatusResponse {
    pub is_running: bool,
    pub files_indexed: u64,
    pub files_total: u64,
    pub sources_connected: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SearchResponse {
    pub results: Vec<SearchResult>,
    pub took_ms: f64,
}

#[doc(hidden)]
#[derive(Debug, Serialize, Deserialize)]
pub struct ContextSuggestion {
    pub content: String,
    pub score: f32,
    pub source: String,
}

#[doc(hidden)]
#[derive(Debug, Serialize, Deserialize)]
pub struct ContextResponse {
    pub suggestions: Vec<ContextSuggestion>,
    pub took_ms: f64,
}

#[derive(Debug, Default, Serialize, Deserialize)]
pub struct TierTokenEstimates {
    pub tier1_identity: usize,
    pub tier2_project: usize,
    pub tier3_relevant: usize,
    pub total: usize,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ProfileContext {
    pub narrative: String,
    pub identity: Vec<String>,
    pub preferences: Vec<String>,
    pub goals: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct KnowledgeContext {
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub concepts: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub decisions: Vec<String>,
    #[serde(default)]
    pub relevant_memories: Vec<SearchResult>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub graph_context: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ChatContextResponse {
    pub context: String,
    pub profile: ProfileContext,
    pub knowledge: KnowledgeContext,
    pub took_ms: f64,
    pub token_estimates: TierTokenEstimates,
}

// ===== Profile & Agents =====

#[derive(Debug, Serialize, Deserialize)]
pub struct ProfileResponse {
    pub id: String,
    pub name: String,
    pub display_name: Option<String>,
    pub email: Option<String>,
    pub bio: Option<String>,
    pub avatar_path: Option<String>,
    pub created_at: i64,
    pub updated_at: i64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct AgentResponse {
    pub id: String,
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
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

// ===== Knowledge graph =====

#[derive(Debug, Serialize, Deserialize)]
pub struct CreateEntityResponse {
    pub id: String,
}

#[doc(hidden)]
#[derive(Debug, Serialize, Deserialize)]
pub struct CreateRelationResponse {
    pub id: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct AddObservationResponse {
    pub id: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ListEntitiesResponse {
    pub entities: Vec<Entity>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SearchEntitiesResponse {
    pub results: Vec<EntitySearchResult>,
}

// ===== Import =====

#[derive(Debug, Serialize, Deserialize)]
pub struct ImportMemoriesResponse {
    pub imported: usize,
    pub skipped: usize,
    pub breakdown: HashMap<String, usize>,
    pub entities_created: usize,
    pub observations_added: usize,
    pub relations_created: usize,
    pub batch_id: String,
}

// ===== Steep =====

/// How loud Origin should be about a phase's output.
#[doc(hidden)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Nudge {
    Silent,
    Ambient,
    Notable,
    Wow,
}

/// Result of a single phase within a steep cycle.
#[doc(hidden)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhaseResult {
    pub name: String,
    pub duration_ms: u64,
    pub items_processed: usize,
    pub error: Option<String>,
    pub nudge: Nudge,
    pub headline: Option<String>,
}

#[doc(hidden)]
#[derive(Debug, Serialize, Deserialize)]
pub struct SteepResponse {
    pub memories_decayed: u64,
    pub recaps_generated: u32,
    pub distilled: u32,
    pub pending_remaining: u32,
    pub phases: Vec<PhaseResult>,
}

// ===== Config =====

#[derive(Debug, Serialize, Deserialize)]
pub struct ConfigResponse {
    pub skip_apps: Vec<String>,
    pub skip_title_patterns: Vec<String>,
    pub private_browsing_detection: bool,
    pub setup_completed: bool,
    pub clipboard_enabled: bool,
    pub screen_capture_enabled: bool,
    pub selection_capture_enabled: bool,
    pub remote_access_enabled: bool,
}

// ===== Indexed files / chunks =====

#[derive(Debug, Serialize, Deserialize)]
pub struct IndexedFilesResponse {
    pub files: Vec<IndexedFileInfo>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct DeleteCountResponse {
    pub deleted: usize,
}

// ===== Entity / Observation =====

#[derive(Debug, Serialize, Deserialize)]
pub struct SuccessResponse {
    pub ok: bool,
}

// ===== Memory detail =====

#[derive(Debug, Serialize, Deserialize)]
pub struct MemoryDetailResponse {
    pub memory: Option<MemoryItem>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct VersionChainResponse {
    pub versions: Vec<crate::memory::MemoryVersionItem>,
}

// ===== Tags =====

#[derive(Debug, Serialize, Deserialize)]
pub struct TagsResponse {
    pub tags: Vec<String>,
}

// ===== Activity =====

#[derive(Debug, Serialize, Deserialize)]
pub struct ActivityResponse {
    pub activities: Vec<crate::memory::AgentActivityRow>,
}

// ===== Decisions =====

#[derive(Debug, Serialize, Deserialize)]
pub struct DecisionsResponse {
    pub decisions: Vec<MemoryItem>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct DecisionDomainsResponse {
    pub domains: Vec<String>,
}

// ===== Pinned =====

#[derive(Debug, Serialize, Deserialize)]
pub struct PinnedMemoriesResponse {
    pub memories: Vec<MemoryItem>,
}

// ===== Ingest =====

#[derive(Debug, Serialize, Deserialize)]
pub struct IngestResponse {
    pub chunks_created: usize,
    pub document_id: String,
}

// Note: ingest's `DELETE /api/documents/{source}/{source_id}` reuses the
// `DeleteResponse { deleted: bool }` defined above — same wire format.

// ===== Concept Export =====

#[derive(Debug, Deserialize, Serialize)]
pub struct ExportConceptResponse {
    pub path: String,
}

// ===== Knowledge Directory =====

#[derive(Debug, Deserialize, Serialize)]
pub struct KnowledgePathResponse {
    pub path: String,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct KnowledgeCountResponse {
    pub count: u64,
}

// ===== Sources =====

#[doc(hidden)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyncStatsResponse {
    pub files_found: usize,
    pub ingested: usize,
    pub skipped: usize,
    pub errors: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn store_memory_response_deserializes_without_extraction_method() {
        // Forward-compat: older server responses (pre-D9) omit extraction_method entirely.
        let json = r#"{
            "source_id": "mem_abc",
            "chunks_created": 3,
            "memory_type": "fact"
        }"#;
        let parsed: StoreMemoryResponse = serde_json::from_str(json).unwrap();
        assert_eq!(parsed.source_id, "mem_abc");
        assert_eq!(parsed.chunks_created, 3);
        assert_eq!(parsed.memory_type, "fact");
        assert_eq!(parsed.extraction_method, "unknown");
        assert!(parsed.warnings.is_empty());
    }

    #[test]
    fn store_memory_response_deserializes_with_all_fields() {
        let json = r#"{
            "source_id": "mem_abc",
            "chunks_created": 3,
            "memory_type": "fact",
            "warnings": ["decision memory missing claim"],
            "extraction_method": "llm"
        }"#;
        let parsed: StoreMemoryResponse = serde_json::from_str(json).unwrap();
        assert_eq!(parsed.warnings.len(), 1);
        assert_eq!(parsed.extraction_method, "llm");
    }

    #[test]
    fn store_memory_response_exposes_enrichment_and_hint() {
        // Post-async-refactor shape: the daemon returns immediately after
        // upsert and reports deferred enrichment via `enrichment` + `hint`.
        let json = r#"{
            "source_id": "mem_xyz",
            "chunks_created": 1,
            "memory_type": "fact",
            "warnings": [],
            "extraction_method": "unknown",
            "enrichment": "pending",
            "hint": "Stored. Origin is compiling classification + concept links in the background (~2s). Recall will surface the enriched form shortly."
        }"#;
        let parsed: StoreMemoryResponse = serde_json::from_str(json).unwrap();
        assert_eq!(parsed.enrichment, "pending");
        assert!(parsed.hint.contains("compiling"));
    }

    #[test]
    fn store_memory_response_defaults_enrichment_for_older_responses() {
        // Backward-compat: existing clients (origin-mcp, Tauri app) that
        // deserialize pre-async-refactor responses must keep working.
        let json = r#"{
            "source_id": "mem_old",
            "chunks_created": 1,
            "memory_type": "fact"
        }"#;
        let parsed: StoreMemoryResponse = serde_json::from_str(json).unwrap();
        assert_eq!(parsed.enrichment, ""); // default
        assert_eq!(parsed.hint, ""); // default
    }

    #[test]
    fn store_memory_response_roundtrips_not_needed_state() {
        // Daemon reports `not_needed` when no LLM is available. Hint is empty
        // (skip_serializing_if) so the JSON shrinks accordingly.
        let response = StoreMemoryResponse {
            source_id: "mem_no_llm".into(),
            chunks_created: 1,
            memory_type: "fact".into(),
            entity_id: None,
            quality: None,
            warnings: vec![],
            extraction_method: "none".into(),
            enrichment: "not_needed".into(),
            hint: String::new(),
        };
        let json = serde_json::to_string(&response).unwrap();
        assert!(json.contains("\"enrichment\":\"not_needed\""));
        assert!(
            !json.contains("\"hint\""),
            "empty hint must be skipped on the wire, got: {json}"
        );
        let parsed: StoreMemoryResponse = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.enrichment, "not_needed");
        assert_eq!(parsed.hint, "");
    }

    #[test]
    fn chat_context_response_roundtrips_with_empty_knowledge_sections() {
        let response = ChatContextResponse {
            context: "context".into(),
            profile: ProfileContext {
                narrative: "n".into(),
                identity: vec![],
                preferences: vec![],
                goals: vec![],
            },
            knowledge: KnowledgeContext {
                concepts: vec![],
                decisions: vec![],
                relevant_memories: vec![],
                graph_context: vec![],
            },
            took_ms: 1.0,
            token_estimates: TierTokenEstimates {
                tier1_identity: 1,
                tier2_project: 2,
                tier3_relevant: 3,
                total: 6,
            },
        };

        let json = serde_json::to_string(&response).unwrap();
        let parsed: ChatContextResponse = serde_json::from_str(&json).unwrap();
        assert!(parsed.knowledge.concepts.is_empty());
        assert!(parsed.knowledge.decisions.is_empty());
        assert!(parsed.knowledge.relevant_memories.is_empty());
        assert!(parsed.knowledge.graph_context.is_empty());
    }
}
