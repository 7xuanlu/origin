// SPDX-License-Identifier: Apache-2.0
//! API response types for all HTTP endpoints.

use crate::entities::{Entity, EntitySearchResult};
use crate::memory::{IndexedFileInfo, MemoryItem, MemoryStats, SearchResult};
use crate::pages::Page;
use crate::repair::RepairDigest;
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
    /// Enrichment state for the memory. `"pending"` when the quiet/cooldown-
    /// gated ambient scheduler has authorized derived work remaining;
    /// `"paused"` when no source is pinned or the pinned source is unavailable.
    /// `"not_needed"` remains accepted as a legacy wire value from older
    /// daemons. Machine-readable — Tauri app uses this to drive
    /// polling / live-update UI, MCP callers can choose to relay state.
    /// Defaulted for backward compatibility with older clients.
    #[serde(default)]
    pub enrichment: String,
    /// Prose cue for caller agents — safe to relay to the user verbatim.
    /// Communicates that Wenlan is compiling the memory into reusable
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
    /// Distilled pages surfaced by the page channel during reranked search.
    /// Absent when no page rows were returned (back-compat: old daemons never
    /// set this field; old consumers that don't read it are unaffected).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub supplemental_pages: Option<Vec<SearchResult>>,
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

/// Cross-encoder reranker state, surfaced on `/api/status` so operators can see
/// whether an opt-in reranker is actually wired vs. silently degraded.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum RerankerStatus {
    /// `WENLAN_RERANKER_ENABLED` was not `1` — no reranker requested.
    #[default]
    Disabled,
    /// Reranker initialized and wired.
    Active { model_id: String },
    /// Reranker was requested but init failed (e.g. model download error);
    /// search silently falls back to embedding+FTS ordering.
    Failed { reason: String },
}

/// Background document-enrichment queue state, surfaced on `/api/status` so
/// operators can see whether folder-ingest enrichment is progressing, idle, or
/// paused on an LLM failure (waiting for a backoff retry). Mirrors the
/// [`RerankerStatus`] tagged-enum shape.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum QueueStatus {
    /// No documents are queued for enrichment (nothing pending, in-progress, or paused).
    #[default]
    Idle,
    /// Documents are queued or being enriched. `pending` counts rows not yet done.
    Active { pending: u64 },
    /// At least one document is paused after an LLM failure. `reason` is that
    /// pause's failure reason, `next_retry_at` the earliest Unix-secs retry time
    /// (the scheduler auto-resumes once it elapses — no daemon restart needed),
    /// and `pending` counts rows not yet done.
    Paused {
        reason: String,
        pending: u64,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        next_retry_at: Option<i64>,
    },
}

/// Observable runtime selected for the local llama.cpp model.
///
/// This is additive to `/api/status`: older daemons omit it and deserialize as
/// `backend = "disabled"`. A GPU failure is visible through
/// `fallback_reason` even when the provider recovered and is serving on CPU.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OnDeviceInferenceStatus {
    #[serde(default = "default_disabled_backend")]
    pub backend: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub device: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub device_index: Option<usize>,
    #[serde(default)]
    pub gpu_layers: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fallback_reason: Option<String>,
}

fn default_disabled_backend() -> String {
    "disabled".to_string()
}

impl Default for OnDeviceInferenceStatus {
    fn default() -> Self {
        Self {
            backend: default_disabled_backend(),
            device: None,
            device_index: None,
            gpu_layers: 0,
            fallback_reason: None,
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct StatusResponse {
    pub is_running: bool,
    pub files_indexed: u64,
    pub files_total: u64,
    pub sources_connected: Vec<String>,
    /// Background document-enrichment queue state (folder-ingest). Additive:
    /// defaults to `Idle` so older daemons (which omit it) deserialize cleanly.
    #[serde(default)]
    pub queue: QueueStatus,
    /// Compile-routing queue depth (spec §3.1/§7): clusters the last routed
    /// compile left pending because no lane (cloud or healthy on-device) was
    /// available. `Active { pending }` when nonzero, `Idle` otherwise; never
    /// `Paused` (no retry/backoff concept for this gauge). Additive: defaults
    /// to `Idle` so older daemons (which omit it) deserialize cleanly.
    #[serde(default)]
    pub compile_queue: QueueStatus,
    /// Reranker on the DEEP path (`/api/memory/search` with `rerank=true`). Legacy
    /// field — for `WENLAN_RERANKER_ENABLED=1` it is the configured model, exactly as before.
    #[serde(default)]
    pub reranker: RerankerStatus,
    /// Reranker on the LIGHT paths — quick (`/api/search`) + context
    /// (`/api/context`). Populated when `WENLAN_RERANKER_MODE` is `lite`/`full`.
    /// Additive: defaults to `Disabled` so older daemons (which omit it) deserialize cleanly.
    #[serde(default)]
    pub reranker_light: RerankerStatus,
    /// Resolved reranker mode: `"off"` | `"lite"` | `"full"`. Empty string for older
    /// daemons that predate `WENLAN_RERANKER_MODE`.
    #[serde(default)]
    pub reranker_mode: String,
    /// llama.cpp runtime used by the selected on-device provider. Additive:
    /// defaults to `disabled` for older daemons and for installations without
    /// a local model.
    #[serde(default)]
    pub on_device_inference: OnDeviceInferenceStatus,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SearchResponse {
    pub results: Vec<SearchResult>,
    pub took_ms: f64,
    /// Distilled pages surfaced by the shared visibility gate on `/api/search`.
    /// Absent when no page rows passed the space-scope + effective-tier gate
    /// (back-compat: old daemons never set this field; old consumers that
    /// don't read it are unaffected). Mirrors `SearchMemoryResponse`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub supplemental_pages: Option<Vec<SearchResult>>,
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
    /// Deprecated: goal taxonomy folded into Identity by migration 45 (Phase 0).
    /// Always empty — daemon does not emit goal-typed memories. Field stays for
    /// wire backward compat; will be removed in 0.4.
    #[deprecated(
        since = "0.3.2",
        note = "Goal taxonomy folded into Identity by migration 45 (Phase 0). \
                Always empty. Will be removed in 0.4."
    )]
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub goals: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct KnowledgeContext {
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub pages: Vec<String>,
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
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub warnings: Vec<String>,
}

#[doc(hidden)]
#[derive(Debug, Serialize, Deserialize)]
pub struct CreateRelationResponse {
    pub id: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub warnings: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct AddObservationResponse {
    pub id: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub warnings: Vec<String>,
}

#[doc(hidden)]
#[derive(Debug, Serialize, Deserialize)]
pub struct CreatePageResponse {
    pub id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub attached_to: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub warnings: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ListEntitiesResponse {
    pub entities: Vec<Entity>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SearchEntitiesResponse {
    pub results: Vec<EntitySearchResult>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SearchPagesResponse {
    pub pages: Vec<Page>,
}

/// Wikilink graph centered on a single page. Outbound = labels parsed
/// out of this page's body; `target_page_id` is `None` for orphans.
/// Inbound = active pages whose body cites this title.
#[derive(Debug, Serialize, Deserialize)]
pub struct PageLinksResponse {
    pub outbound: Vec<PageLinkOutbound>,
    pub inbound: Vec<PageLinkInbound>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct PageLinkOutbound {
    pub label: String,
    /// `None` when the resolver couldn't find a matching active page —
    /// surfaces in the orphan-by-count feed via /api/pages/orphan-links.
    pub target_page_id: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct PageLinkInbound {
    pub source_page_id: String,
    pub label: String,
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

/// How loud Wenlan should be about a phase's output.
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
    pub remote_access_enabled: bool,
    /// Anthropic model used for fast/routine tasks (e.g. classification, tagging).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub routine_model: Option<String>,
    /// Anthropic model used for synthesis tasks (e.g. distillation, narrative).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub synthesis_model: Option<String>,
    /// Base URL for an OpenAI-compatible external LLM endpoint.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub external_llm_endpoint: Option<String>,
    /// Model identifier to use with the external LLM endpoint.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub external_llm_model: Option<String>,
    /// Whether an external-LLM API key is stored. The key value itself is
    /// never serialized anywhere.
    #[serde(default)]
    pub external_llm_api_key_configured: bool,
    /// Everyday-job source pin (raw config value): `"anthropic"` | `"external"`
    /// | `"on_device"`, or absent/null when unpinned. Lets the app show the
    /// pin the user chose distinctly from the resolved source.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub everyday_source: Option<String>,
    /// Synthesis-job source pin (raw config value): `"anthropic"` |
    /// `"external"`, or absent/null when unpinned.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub synthesis_source: Option<String>,
    /// Whether the proactive Page-Map suggestion phase is enabled. Default true.
    #[serde(default)]
    pub page_map_auto_suggest: bool,
}

// ===== On-device model =====

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OnDeviceModelEntry {
    pub id: String,
    pub display_name: String,
    pub param_count: String,
    pub ram_required_gb: f64,
    pub file_size_gb: f64,
    pub cached: bool,
}

/// Response envelope for `GET /api/on-device-model`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OnDeviceModelResponse {
    /// ID of the model currently loaded in the daemon, if any.
    pub loaded: Option<String>,
    /// ID the user has selected in config; may differ from loaded when a
    /// download is pending or a restart is needed.
    pub selected: Option<String>,
    /// All available models with per-model cache/download state.
    pub models: Vec<OnDeviceModelEntry>,
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

fn is_false(value: &bool) -> bool {
    !*value
}

#[derive(Debug, Serialize, Deserialize)]
pub struct PageWriteResponse {
    pub ok: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub revision_card_id: Option<String>,
    #[serde(default, skip_serializing_if = "is_false")]
    pub gated: bool,
}

/// Page draft create, update, and publish response envelope.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PageDraftResponse {
    pub page: Page,
}

// ===== Memory detail =====

#[derive(Debug, Serialize, Deserialize)]
pub struct MemoryDetailResponse {
    pub memory: Option<MemoryItem>,
}

/// Detailed chunk-level view of a stored memory, returned by `/api/chunks/{source_id}`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryDetail {
    pub id: String,
    pub content: String,
    pub title: String,
    pub source_id: String,
    pub chunk_index: i32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub chunk_type: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub language: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub semantic_unit: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub byte_start: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub byte_end: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub summary: Option<String>,
}

/// A pending revision waiting for human approval (Protected tier supersede).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PendingRevision {
    pub source_id: String,
    pub content: String,
    pub source_agent: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct VersionChainResponse {
    pub versions: Vec<crate::memory::MemoryVersionItem>,
}

// ===== Tags =====

#[derive(Debug, Serialize, Deserialize)]
pub struct TagsResponse {
    pub tags: Vec<String>,
    #[serde(default)]
    pub document_tags: HashMap<String, Vec<String>>,
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
    /// Kept as `domains` for one-release back-compat with callers of
    /// `/api/decisions/domains`; rename to `spaces` in PR-A+1.
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

/// Statistics from a bulk page export operation (POST /api/pages/export).
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct ExportStats {
    pub exported: usize,
    pub skipped: usize,
    pub failed: usize,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct ExportPageResponse {
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

// ===== Revision history =====

/// One entry in a memory's supersede chain, returned by `/api/memory/{id}/revisions`.
///
/// `depth = 0` is the current (most-recent) memory; higher depths are older
/// predecessors. `delta_summary` is `None` for the deepest entry (no predecessor
/// to diff against) and computed heuristically for all shallower entries.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryRevisionEntry {
    pub source_id: String,
    pub depth: i64,
    pub title: String,
    pub content_preview: String,
    pub last_modified: i64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source_agent: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub supersede_mode: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub delta_summary: Option<String>,
}

/// Response envelope for `/api/memory/{id}/revisions`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ListMemoryRevisionsResponse {
    pub current_source_id: String,
    pub chain_depth: i64,
    pub entries: Vec<MemoryRevisionEntry>,
}

/// One entry in a page's version changelog, returned by `/api/pages/{id}/revisions`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PageChangelogEntry {
    pub version: i64,
    pub at: i64,
    pub edited_by: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub delta_summary: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub incoming_source_ids: Option<Vec<String>>,
    /// Human-readable summary of citation verification for this revision
    /// (e.g. "3 verified, 1 unverified, 2 stripped"). None for revisions that
    /// didn't touch citations.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub citations_summary: Option<String>,
}

/// Response envelope for `/api/pages/{id}/revisions`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ListPageRevisionsResponse {
    pub page_id: String,
    pub current_version: i64,
    pub user_edited: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stale_reason: Option<String>,
    pub entries: Vec<PageChangelogEntry>,
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

// ===== Refinement proposals =====

/// The action type for a background-refinery proposal.
///
/// Used as the `action` tag in [`RefinementPayload`].
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ProposalAction {
    EntityMerge,
    RelationConflict,
    DetectContradiction,
    SuggestEntity,
    DedupMerge,
    PageMerge,
    CrossSpaceDiscovery,
    PageKeepOrArchive,
    LintRepairReview,
    VocabPromote,
    /// ponytail: deserialize-only catch-all. Lets a stale client decode a
    /// newer daemon's action tag instead of failing the whole list. NEVER
    /// constructed or serialized by the daemon (it only ever holds real
    /// variants), so serialize-of-Unknown never happens in practice.
    #[serde(other)]
    Unknown,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum RefinementCardAction {
    Accept,
    Dismiss,
    PickSpace,
}

/// Tagged-union payload emitted by the background refinery.
///
/// Each variant carries exactly the fields needed for that action type.
/// Decoded at the route boundary so downstream consumers (MCP wrappers,
/// agent skills) can pattern-match instead of inspecting raw JSON strings.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
#[serde(tag = "action", rename_all = "snake_case")]
pub enum RefinementPayload {
    EntityMerge {
        existing_id: String,
        new_id: String,
        similarity: f64,
    },
    RelationConflict {
        existing_id: String,
        new_id: String,
        from: String,
        to: String,
        old_type: String,
        new_type: String,
    },
    DetectContradiction,
    SuggestEntity {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        name_hint: Option<String>,
    },
    DedupMerge,
    PageMerge {
        left_page_id: String,
        right_page_id: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        similarity: Option<f64>,
        source_overlap: usize,
        source_overlap_ratio: f64,
    },
    CrossSpaceDiscovery {
        memory_count: usize,
        spaces: Vec<String>,
        allowed_actions: Vec<RefinementCardAction>,
    },
    PageKeepOrArchive {
        page_id: String,
        source_count: usize,
        allowed_actions: Vec<RefinementCardAction>,
    },
    LintRepairReview {
        check_id: String,
        occurrence_digest: RepairDigest,
        owner_binding_digest: RepairDigest,
        issue: String,
        choices: Vec<String>,
        suggested_research_queries: Vec<String>,
    },
    VocabPromote {
        kind: String,
        old_value: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        category: Option<String>,
    },
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct RefinementProposalSummary {
    pub id: String,
    pub action: ProposalAction,
    pub source_ids: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub payload: Option<RefinementPayload>,
    pub confidence: f64,
    pub created_at: String,
}

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub struct ListRefinementsResponse {
    pub proposals: Vec<RefinementProposalSummary>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct RejectRefinementResponse {
    pub id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AcceptRefinementResponse {
    pub id: String,
    pub action_applied: String,
}

#[cfg(test)]
mod refinement_wire_tests {
    use super::*;

    #[test]
    fn proposal_action_serde_round_trip() {
        let cases = [
            ("\"entity_merge\"", ProposalAction::EntityMerge),
            ("\"relation_conflict\"", ProposalAction::RelationConflict),
            (
                "\"detect_contradiction\"",
                ProposalAction::DetectContradiction,
            ),
            ("\"suggest_entity\"", ProposalAction::SuggestEntity),
            ("\"dedup_merge\"", ProposalAction::DedupMerge),
            (
                "\"cross_space_discovery\"",
                ProposalAction::CrossSpaceDiscovery,
            ),
            (
                "\"page_keep_or_archive\"",
                ProposalAction::PageKeepOrArchive,
            ),
            ("\"vocab_promote\"", ProposalAction::VocabPromote),
        ];
        for (json, expected) in cases {
            let parsed: ProposalAction = serde_json::from_str(json).unwrap();
            assert_eq!(parsed, expected, "deserialize {json}");
            let back = serde_json::to_string(&expected).unwrap();
            assert_eq!(back, json, "serialize {expected:?}");
        }
    }

    #[test]
    fn vocab_promote_payload_round_trips() {
        let p = RefinementPayload::VocabPromote {
            kind: "relation".into(),
            old_value: "design_inspiration".into(),
            category: None,
        };
        let json = serde_json::to_string(&p).unwrap();
        let back: RefinementPayload = serde_json::from_str(&json).unwrap();
        assert_eq!(p, back);
        assert!(json.contains("\"action\":\"vocab_promote\""));
    }

    #[test]
    fn proposal_action_unknown_future_variant_deserializes() {
        // A stale client must decode a NEWER daemon's action tag to Unknown,
        // never error the whole ListRefinementsResponse.
        let parsed: ProposalAction = serde_json::from_str("\"future_unshipped_action\"").unwrap();
        assert_eq!(parsed, ProposalAction::Unknown);
        let parsed2: ProposalAction = serde_json::from_str("\"totally_new_action\"").unwrap();
        assert_eq!(parsed2, ProposalAction::Unknown);
    }

    #[test]
    fn refinement_payload_entity_merge_round_trip() {
        let json =
            r#"{"action":"entity_merge","existing_id":"e1","new_id":"e2","similarity":0.87}"#;
        let parsed: RefinementPayload = serde_json::from_str(json).unwrap();
        match parsed {
            RefinementPayload::EntityMerge {
                ref existing_id,
                ref new_id,
                similarity,
            } => {
                assert_eq!(existing_id, "e1");
                assert_eq!(new_id, "e2");
                assert!((similarity - 0.87).abs() < 1e-9);
            }
            _ => panic!("expected EntityMerge variant"),
        }
        let back = serde_json::to_value(&parsed).unwrap();
        assert_eq!(back["action"], "entity_merge");
        assert_eq!(back["existing_id"], "e1");
    }

    #[test]
    fn refinement_payload_dedup_merge_no_fields() {
        let json = r#"{"action":"dedup_merge"}"#;
        let parsed: RefinementPayload = serde_json::from_str(json).unwrap();
        assert!(matches!(parsed, RefinementPayload::DedupMerge));
    }

    #[test]
    fn refinement_payload_cross_space_discovery_round_trip() {
        let json = r#"{"action":"cross_space_discovery","memory_count":3,"spaces":["personal","work"],"allowed_actions":["dismiss","pick_space"]}"#;
        let parsed: RefinementPayload = serde_json::from_str(json).unwrap();
        match parsed {
            RefinementPayload::CrossSpaceDiscovery {
                memory_count,
                ref spaces,
                ref allowed_actions,
            } => {
                assert_eq!(memory_count, 3);
                assert_eq!(spaces, &vec!["personal".to_string(), "work".to_string()]);
                assert_eq!(
                    allowed_actions,
                    &vec![
                        RefinementCardAction::Dismiss,
                        RefinementCardAction::PickSpace
                    ]
                );
            }
            _ => panic!("expected CrossSpaceDiscovery"),
        }
        let back = serde_json::to_value(&parsed).unwrap();
        assert_eq!(back["action"], "cross_space_discovery");
        assert_eq!(back["memory_count"], 3);
    }

    #[test]
    fn refinement_payload_page_keep_or_archive_round_trip() {
        let json = r#"{"action":"page_keep_or_archive","page_id":"page_stub","source_count":1,"allowed_actions":["dismiss","accept"]}"#;
        let parsed: RefinementPayload = serde_json::from_str(json).unwrap();
        match parsed {
            RefinementPayload::PageKeepOrArchive {
                ref page_id,
                source_count,
                ref allowed_actions,
            } => {
                assert_eq!(page_id, "page_stub");
                assert_eq!(source_count, 1);
                assert_eq!(
                    allowed_actions,
                    &vec![RefinementCardAction::Dismiss, RefinementCardAction::Accept]
                );
            }
            _ => panic!("expected PageKeepOrArchive"),
        }
        let back = serde_json::to_value(&parsed).unwrap();
        assert_eq!(back["action"], "page_keep_or_archive");
        assert_eq!(back["source_count"], 1);
    }

    #[test]
    fn refinement_payload_relation_conflict_round_trip() {
        let json = r#"{"action":"relation_conflict","existing_id":"r1","new_id":"r2","from":"e_a","to":"e_b","old_type":"works_at","new_type":"founded"}"#;
        let parsed: RefinementPayload = serde_json::from_str(json).unwrap();
        match parsed {
            RefinementPayload::RelationConflict {
                ref existing_id,
                ref new_id,
                ref from,
                ref to,
                ref old_type,
                ref new_type,
            } => {
                assert_eq!(existing_id, "r1");
                assert_eq!(new_id, "r2");
                assert_eq!(from, "e_a");
                assert_eq!(to, "e_b");
                assert_eq!(old_type, "works_at");
                assert_eq!(new_type, "founded");
            }
            _ => panic!("expected RelationConflict"),
        }
        let back = serde_json::to_value(&parsed).unwrap();
        assert_eq!(back["from"], "e_a");
        assert_eq!(back["to"], "e_b");
    }

    #[test]
    fn refinement_payload_detect_contradiction_unit_variant() {
        let json = r#"{"action":"detect_contradiction"}"#;
        let parsed: RefinementPayload = serde_json::from_str(json).unwrap();
        assert!(matches!(parsed, RefinementPayload::DetectContradiction));
    }

    #[test]
    fn refinement_payload_suggest_entity_with_name_hint() {
        let json = r#"{"action":"suggest_entity","name_hint":"PostgreSQL"}"#;
        let parsed: RefinementPayload = serde_json::from_str(json).unwrap();
        match parsed {
            RefinementPayload::SuggestEntity { ref name_hint } => {
                assert_eq!(name_hint.as_deref(), Some("PostgreSQL"));
            }
            _ => panic!("expected SuggestEntity"),
        }
    }

    #[test]
    fn refinement_payload_suggest_entity_without_name_hint() {
        let json = r#"{"action":"suggest_entity"}"#;
        let parsed: RefinementPayload = serde_json::from_str(json).unwrap();
        assert!(matches!(
            parsed,
            RefinementPayload::SuggestEntity { name_hint: None }
        ));
    }

    #[test]
    fn list_refinements_response_round_trip() {
        let resp = ListRefinementsResponse {
            proposals: vec![RefinementProposalSummary {
                id: "ref_1".into(),
                action: ProposalAction::EntityMerge,
                source_ids: vec!["a".into(), "b".into()],
                payload: Some(RefinementPayload::EntityMerge {
                    existing_id: "a".into(),
                    new_id: "b".into(),
                    similarity: 0.86,
                }),
                confidence: 0.86,
                created_at: "2026-05-12T00:00:00Z".into(),
            }],
        };
        let json = serde_json::to_string(&resp).unwrap();
        let parsed: ListRefinementsResponse = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.proposals.len(), 1);
        assert_eq!(parsed.proposals[0].id, "ref_1");
        assert!(matches!(
            parsed.proposals[0].action,
            ProposalAction::EntityMerge
        ));
    }

    #[test]
    fn reject_refinement_response_round_trip() {
        let resp = RejectRefinementResponse { id: "ref_x".into() };
        let json = serde_json::to_string(&resp).unwrap();
        let parsed: RejectRefinementResponse = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.id, "ref_x");
    }
}

#[cfg(test)]
mod on_device_model_response_tests {
    use super::*;

    #[test]
    fn on_device_model_response_preserves_selected_loaded_and_models() {
        let response = OnDeviceModelResponse {
            loaded: Some("qwen3-4b".to_string()),
            selected: Some("qwen3-4b".to_string()),
            models: vec![OnDeviceModelEntry {
                id: "qwen3-4b".to_string(),
                display_name: "Qwen3 4B".to_string(),
                param_count: "4B".to_string(),
                ram_required_gb: 6.0,
                file_size_gb: 2.7,
                cached: true,
            }],
        };

        let value = serde_json::to_value(&response).unwrap();

        assert_eq!(value["loaded"], "qwen3-4b");
        assert_eq!(value["selected"], "qwen3-4b");
        assert_eq!(value["models"][0]["id"], "qwen3-4b");
        assert_eq!(value["models"][0]["cached"], true);

        let parsed: OnDeviceModelResponse = serde_json::from_value(value).unwrap();
        assert_eq!(parsed.loaded.as_deref(), Some("qwen3-4b"));
        assert_eq!(parsed.selected.as_deref(), Some("qwen3-4b"));
        assert_eq!(parsed.models.len(), 1);
        assert!(parsed.models[0].cached);
    }

    #[test]
    fn on_device_model_response_allows_null_loaded_and_selected() {
        let parsed: OnDeviceModelResponse =
            serde_json::from_str(r#"{"loaded":null,"selected":null,"models":[]}"#).unwrap();

        assert!(parsed.loaded.is_none());
        assert!(parsed.selected.is_none());
        assert!(parsed.models.is_empty());
    }
}

/// One orphaned page link label aggregated across sources.
///
/// `count` is how many distinct source pages reference this label
/// without a matching target page existing.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct OrphanLink {
    pub label: String,
    pub count: i64,
}

/// Response for `GET /api/pages/orphan-links`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct OrphanLinksResponse {
    pub min_count: usize,
    pub orphan_labels: Vec<OrphanLink>,
}

/// One pending revision awaiting human accept/dismiss.
///
/// `target_source_id` is the memory being revised; pass it to
/// `accept_pending_revision` or `dismiss_pending_revision`.
/// `revision_source_id` is the staged revision row itself, exposed
/// for diagnostics and round-tripping (not for the action call).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PendingRevisionItem {
    pub target_source_id: String,
    pub revision_source_id: String,
    pub revision_content: String,
    pub source_agent: Option<String>,
    pub last_modified: i64,
    /// Doc file source_id that grounds a doc-grounded revision (L3); None for
    /// other revision producers. Read from structured_fields.grounded_in.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub grounded_in: Option<String>,
}

/// Response returned by `POST /api/memory/revision/{id}/accept`.
/// Carries the now-consumed revision row id so agents can correlate with
/// their `list_pending_revisions` cache.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RevisionAcceptResponse {
    pub target_source_id: String,
    pub revision_source_id: String,
    pub wrote: bool,
}

/// Response returned by `POST /api/memory/revision/{id}/dismiss`.
/// `wrote: true` always (404 on missing).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RevisionDismissResponse {
    pub target_source_id: String,
    pub wrote: bool,
}

/// Response returned by `POST /api/memory/contradiction/{source_id}/dismiss`.
/// `wrote: true` is best-effort: the daemon's underlying DB method silently
/// no-ops when no rows match. Wrapper cannot distinguish dismiss-of-existing
/// from dismiss-of-nothing without an extra SELECT (out of scope).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ContradictionDismissResponse {
    pub source_id: String,
    pub wrote: bool,
}

#[cfg(test)]
mod mutation_response_tests {
    use super::*;

    #[test]
    fn revision_accept_response_serializes_byte_identical() {
        let r = RevisionAcceptResponse {
            target_source_id: "mem_target".into(),
            revision_source_id: "mem_rev".into(),
            wrote: true,
        };
        assert_eq!(
            serde_json::to_string(&r).unwrap(),
            r#"{"target_source_id":"mem_target","revision_source_id":"mem_rev","wrote":true}"#
        );
    }

    #[test]
    fn revision_dismiss_response_serializes_byte_identical() {
        let r = RevisionDismissResponse {
            target_source_id: "mem_target".into(),
            wrote: true,
        };
        assert_eq!(
            serde_json::to_string(&r).unwrap(),
            r#"{"target_source_id":"mem_target","wrote":true}"#
        );
    }

    #[test]
    fn contradiction_dismiss_response_serializes_byte_identical() {
        let r = ContradictionDismissResponse {
            source_id: "mem_abc".into(),
            wrote: true,
        };
        assert_eq!(
            serde_json::to_string(&r).unwrap(),
            r#"{"source_id":"mem_abc","wrote":true}"#
        );
    }
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
        // The daemon returns immediately after upsert and reports quiet
        // deferred enrichment via `enrichment` + `hint`.
        let json = r#"{
            "source_id": "mem_xyz",
            "chunks_created": 1,
            "memory_type": "fact",
            "warnings": [],
            "extraction_method": "unknown",
            "enrichment": "pending",
            "hint": "Stored. Recall is available now; Wenlan will quietly enrich classification and page links in the background."
        }"#;
        let parsed: StoreMemoryResponse = serde_json::from_str(json).unwrap();
        assert_eq!(parsed.enrichment, "pending");
        assert!(parsed.hint.contains("quietly enrich"));
    }

    #[test]
    fn store_memory_response_defaults_enrichment_for_older_responses() {
        // Backward-compat: existing clients (wenlan-mcp, Tauri app) that
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
    fn tags_response_defaults_document_tags_for_older_responses() {
        let json = r#"{"tags":["rust","tauri"]}"#;
        let parsed: TagsResponse = serde_json::from_str(json).unwrap();

        assert_eq!(parsed.tags, vec!["rust", "tauri"]);
        assert!(parsed.document_tags.is_empty());
    }

    #[test]
    fn tags_response_deserializes_document_tag_map() {
        let json = r#"{
            "tags":["rust","tauri"],
            "document_tags":{"memory::mem1":["rust"],"page::page1":["tauri"]}
        }"#;
        let parsed: TagsResponse = serde_json::from_str(json).unwrap();

        assert_eq!(
            parsed.document_tags.get("memory::mem1"),
            Some(&vec!["rust".to_string()])
        );
        assert_eq!(
            parsed.document_tags.get("page::page1"),
            Some(&vec!["tauri".to_string()])
        );
    }

    #[test]
    fn store_memory_response_roundtrips_not_needed_state() {
        // Keep accepting the legacy state emitted by older daemons.
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
    fn store_memory_response_roundtrips_paused_state_with_hint() {
        let response = StoreMemoryResponse {
            source_id: "mem_paused".into(),
            chunks_created: 1,
            memory_type: "fact".into(),
            entity_id: None,
            quality: None,
            warnings: vec![],
            extraction_method: "none".into(),
            enrichment: "paused".into(),
            hint: "Stored; choose a model source to enable enrichment.".into(),
        };
        let json = serde_json::to_string(&response).unwrap();
        assert!(json.contains("\"enrichment\":\"paused\""));
        let parsed: StoreMemoryResponse = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.enrichment, "paused");
        assert!(parsed.hint.contains("choose a model source"));
    }

    #[test]
    fn chat_context_response_roundtrips_with_empty_knowledge_sections() {
        // ProfileContext.goals is deprecated; constructing it directly here
        // for wire roundtrip coverage until 0.4 drops the field entirely.
        #[allow(deprecated)]
        let profile = ProfileContext {
            narrative: "n".into(),
            identity: vec![],
            preferences: vec![],
            goals: vec![],
        };
        let response = ChatContextResponse {
            context: "context".into(),
            profile,
            knowledge: KnowledgeContext {
                pages: vec![],
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
        assert!(parsed.knowledge.pages.is_empty());
        assert!(parsed.knowledge.decisions.is_empty());
        assert!(parsed.knowledge.relevant_memories.is_empty());
        assert!(parsed.knowledge.graph_context.is_empty());
    }

    #[test]
    fn orphan_links_response_golden_string() {
        let resp = OrphanLinksResponse {
            min_count: 2,
            orphan_labels: vec![OrphanLink {
                label: "Rust".to_string(),
                count: 3,
            }],
        };
        let s = serde_json::to_string(&resp).unwrap();
        assert_eq!(
            s,
            r#"{"min_count":2,"orphan_labels":[{"label":"Rust","count":3}]}"#
        );
    }

    #[test]
    fn orphan_links_response_empty_round_trip() {
        let resp = OrphanLinksResponse {
            min_count: 1,
            orphan_labels: vec![],
        };
        let decoded: OrphanLinksResponse =
            serde_json::from_str(&serde_json::to_string(&resp).unwrap()).unwrap();
        assert_eq!(decoded, resp);
    }

    #[test]
    fn pending_revision_item_round_trip() {
        let item = PendingRevisionItem {
            target_source_id: "mem_target".into(),
            revision_source_id: "mem_rev".into(),
            revision_content: "new body".into(),
            source_agent: Some("claude-code".into()),
            last_modified: 1_715_000_000,
            grounded_in: None,
        };
        let json = serde_json::to_value(&item).unwrap();
        assert_eq!(json["target_source_id"], "mem_target");
        assert_eq!(json["revision_source_id"], "mem_rev");
        assert_eq!(json["revision_content"], "new body");
        let decoded: PendingRevisionItem = serde_json::from_value(json).unwrap();
        assert_eq!(decoded, item);
    }
}

#[cfg(test)]
mod queue_status_tests {
    use super::*;

    #[test]
    fn status_response_defaults_queue_to_idle_when_absent() {
        // Old daemons omit `queue` entirely — it must default to Idle so the
        // wire change stays additive (a new client reads an old response).
        let json =
            r#"{"is_running":true,"files_indexed":0,"files_total":0,"sources_connected":[]}"#;
        let parsed: StatusResponse = serde_json::from_str(json).unwrap();
        assert_eq!(parsed.queue, QueueStatus::Idle);
        assert_eq!(
            parsed.on_device_inference,
            OnDeviceInferenceStatus::default()
        );
    }

    #[test]
    fn on_device_inference_status_round_trips_vulkan_device_and_fallback() {
        let status = OnDeviceInferenceStatus {
            backend: "vulkan".into(),
            device: Some("NVIDIA GeForce RTX 3060 Laptop GPU".into()),
            device_index: Some(2),
            gpu_layers: 99,
            fallback_reason: None,
        };
        let json = serde_json::to_string(&status).unwrap();
        let parsed: OnDeviceInferenceStatus = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed, status);
        assert!(json.contains("\"backend\":\"vulkan\""));
        assert!(json.contains("RTX 3060"));

        let fallback: OnDeviceInferenceStatus = serde_json::from_str(
            r#"{"backend":"cpu","gpu_layers":0,"fallback_reason":"Vulkan context creation failed"}"#,
        )
        .unwrap();
        assert_eq!(fallback.backend, "cpu");
        assert_eq!(
            fallback.fallback_reason.as_deref(),
            Some("Vulkan context creation failed")
        );
    }

    #[test]
    fn queue_status_paused_round_trips_with_reason_and_retry() {
        let s = QueueStatus::Paused {
            reason: "analysis LLM failed".into(),
            pending: 2,
            next_retry_at: Some(1_712_678_400),
        };
        let json = serde_json::to_string(&s).unwrap();
        assert!(json.contains("\"state\":\"paused\""), "got: {json}");
        assert!(
            json.contains("\"reason\":\"analysis LLM failed\""),
            "got: {json}"
        );
        assert!(json.contains("\"next_retry_at\":1712678400"), "got: {json}");
        assert_eq!(serde_json::from_str::<QueueStatus>(&json).unwrap(), s);
    }

    #[test]
    fn queue_status_active_round_trips() {
        let s = QueueStatus::Active { pending: 3 };
        let json = serde_json::to_string(&s).unwrap();
        assert!(json.contains("\"state\":\"active\""), "got: {json}");
        assert!(json.contains("\"pending\":3"), "got: {json}");
        assert_eq!(serde_json::from_str::<QueueStatus>(&json).unwrap(), s);
    }

    #[test]
    fn queue_status_idle_serializes_state_only() {
        let json = serde_json::to_string(&QueueStatus::Idle).unwrap();
        assert_eq!(json, r#"{"state":"idle"}"#);
    }
}

#[cfg(test)]
mod reranker_status_tests {
    use super::*;

    #[test]
    fn status_response_defaults_reranker_to_disabled() {
        // Old daemons omit reranker AND the newer reranker_light/reranker_mode fields
        // entirely — all three must default cleanly (additive wire change, no break).
        let json =
            r#"{"is_running":true,"files_indexed":0,"files_total":0,"sources_connected":[]}"#;
        let parsed: StatusResponse = serde_json::from_str(json).unwrap();
        assert_eq!(parsed.reranker, RerankerStatus::Disabled);
        assert_eq!(parsed.reranker_light, RerankerStatus::Disabled);
        assert_eq!(parsed.reranker_mode, "");
    }

    #[test]
    fn status_response_roundtrips_per_path_reranker() {
        let s = StatusResponse {
            is_running: true,
            files_indexed: 0,
            files_total: 0,
            sources_connected: vec![],
            queue: QueueStatus::Idle,
            compile_queue: QueueStatus::Idle,
            reranker: RerankerStatus::Active {
                model_id: "BGERerankerBase".into(),
            },
            reranker_light: RerankerStatus::Active {
                model_id: "JINARerankerV1TurboEn".into(),
            },
            reranker_mode: "full".into(),
            on_device_inference: OnDeviceInferenceStatus::default(),
        };
        let json = serde_json::to_string(&s).unwrap();
        let parsed: StatusResponse = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.reranker, s.reranker);
        assert_eq!(parsed.reranker_light, s.reranker_light);
        assert_eq!(parsed.reranker_mode, "full");
    }

    #[test]
    fn reranker_status_active_roundtrips() {
        let s = RerankerStatus::Active {
            model_id: "BGERerankerBase".into(),
        };
        let json = serde_json::to_string(&s).unwrap();
        assert_eq!(serde_json::from_str::<RerankerStatus>(&json).unwrap(), s);
        assert!(json.contains("\"state\":\"active\""));
    }
}

#[cfg(test)]
mod search_memory_response_tests {
    use super::SearchMemoryResponse;

    /// Old daemon responses (no `supplemental_pages` key) must deserialize
    /// successfully with `supplemental_pages == None`.  This locks in the
    /// back-compat guarantee: clients talking to an older daemon never see a
    /// deserialization error.
    #[test]
    fn back_compat_missing_supplemental_pages_is_none() {
        let json = r#"{"results":[],"took_ms":1.0}"#;
        let resp: SearchMemoryResponse = serde_json::from_str(json).expect("should deserialize");
        assert!(
            resp.supplemental_pages.is_none(),
            "should be None when key absent"
        );
        assert_eq!(resp.took_ms, 1.0);
    }

    /// `supplemental_pages` absent means the field is omitted on the wire
    /// (skip_serializing_if = "Option::is_none").
    #[test]
    fn none_supplemental_pages_not_serialized() {
        let resp = SearchMemoryResponse {
            results: vec![],
            took_ms: 2.0,
            supplemental_pages: None,
        };
        let json = serde_json::to_string(&resp).expect("serialize");
        assert!(
            !json.contains("supplemental_pages"),
            "None field must be omitted from wire: {}",
            json
        );
    }

    /// When pages are present they round-trip correctly.
    #[test]
    fn some_supplemental_pages_round_trips() {
        let json = r#"{"results":[],"took_ms":0.5,"supplemental_pages":[]}"#;
        let resp: SearchMemoryResponse = serde_json::from_str(json).expect("deserialize");
        assert!(
            resp.supplemental_pages.is_some(),
            "supplemental_pages should be Some"
        );
        assert!(
            resp.supplemental_pages.unwrap().is_empty(),
            "empty array should deserialize to empty vec"
        );
    }
}

#[cfg(test)]
mod search_response_tests {
    use super::SearchResponse;

    /// Old daemon responses (no `supplemental_pages` key) must deserialize
    /// successfully with `supplemental_pages == None`. Back-compat guarantee:
    /// a client built against the new wire type can still read an older
    /// `/api/search` response that predates the additive page field.
    #[test]
    fn back_compat_missing_supplemental_pages_is_none() {
        let json = r#"{"results":[],"took_ms":1.0}"#;
        let resp: SearchResponse = serde_json::from_str(json).expect("should deserialize");
        assert!(
            resp.supplemental_pages.is_none(),
            "should be None when key absent"
        );
        assert_eq!(resp.took_ms, 1.0);
    }

    /// `supplemental_pages == None` is omitted on the wire
    /// (skip_serializing_if = "Option::is_none").
    #[test]
    fn none_supplemental_pages_not_serialized() {
        let resp = SearchResponse {
            results: vec![],
            took_ms: 2.0,
            supplemental_pages: None,
        };
        let json = serde_json::to_string(&resp).expect("serialize");
        assert!(
            !json.contains("supplemental_pages"),
            "None field must be omitted from wire: {}",
            json
        );
    }

    /// When pages are present they round-trip correctly.
    #[test]
    fn some_supplemental_pages_round_trips() {
        let json = r#"{"results":[],"took_ms":0.5,"supplemental_pages":[]}"#;
        let resp: SearchResponse = serde_json::from_str(json).expect("deserialize");
        assert!(
            resp.supplemental_pages.is_some(),
            "supplemental_pages should be Some"
        );
        assert!(
            resp.supplemental_pages.unwrap().is_empty(),
            "empty array should deserialize to empty vec"
        );
    }
}
