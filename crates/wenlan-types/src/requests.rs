// SPDX-License-Identifier: Apache-2.0
//! API request types for all HTTP endpoints.

use crate::WriteSpaceTarget;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ===== Memory CRUD =====

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StoreMemoryRequest {
    pub content: String,
    #[serde(default)]
    pub memory_type: Option<String>,
    #[serde(
        default,
        alias = "domain",
        skip_serializing_if = "WriteSpaceTarget::is_inherit"
    )]
    pub space: WriteSpaceTarget,
    #[serde(default)]
    pub source_agent: Option<String>,
    #[serde(default)]
    pub title: Option<String>,
    #[serde(default)]
    pub confidence: Option<f32>,
    #[serde(default)]
    pub supersedes: Option<String>,
    /// Entity name for resolution (e.g. "Alice", "PostgreSQL")
    #[serde(default)]
    pub entity: Option<String>,
    /// Direct entity ID (bypasses name resolution)
    #[serde(default)]
    pub entity_id: Option<String>,
    #[serde(default)]
    pub structured_fields: Option<serde_json::Value>,
    #[serde(default)]
    pub retrieval_cue: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SearchMemoryRequest {
    pub query: String,
    #[serde(default = "default_limit")]
    pub limit: usize,
    #[serde(default)]
    pub memory_type: Option<String>,
    #[serde(default, alias = "domain")]
    pub space: Option<String>,
    #[serde(default)]
    pub source_agent: Option<String>,
    /// When `true` AND the daemon has a reranker wired (via
    /// `WENLAN_RERANKER_ENABLED=1`), results pass through a cross-encoder
    /// reranker after the embedding+FTS hybrid step. When `true` but no
    /// reranker is available, the daemon logs a warning and falls back to
    /// the plain hybrid ordering. Default `false` to preserve current
    /// behavior for callers that don't opt in.
    #[serde(default)]
    pub rerank: bool,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ListMemoriesRequest {
    #[serde(default)]
    pub memory_type: Option<String>,
    #[serde(default, alias = "domain")]
    pub space: Option<String>,
    #[serde(default = "default_list_limit")]
    pub limit: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub confirmed: Option<bool>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ConfirmRequest {
    #[serde(default = "default_confirmed")]
    pub confirmed: bool,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ReclassifyMemoryRequest {
    pub memory_type: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ImportMemoriesRequest {
    pub source: String,
    pub content: String,
    #[serde(default)]
    pub label: Option<String>,
    #[serde(default, skip_serializing_if = "WriteSpaceTarget::is_inherit")]
    pub space: WriteSpaceTarget,
}

// ===== General search/context =====

#[derive(Debug, Serialize, Deserialize)]
pub struct SearchRequest {
    pub query: String,
    #[serde(default = "default_limit")]
    pub limit: usize,
    pub source_filter: Option<String>,
    #[serde(default, alias = "domain")]
    pub space: Option<String>,
}

#[doc(hidden)]
#[derive(Debug, Serialize, Deserialize)]
pub struct ContextRequest {
    pub current_file: String,
    pub cursor_prefix: String,
    #[serde(default = "default_limit")]
    pub limit: usize,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ChatContextRequest {
    #[serde(default)]
    pub query: Option<String>,
    #[serde(default)]
    pub conversation_id: Option<String>,
    #[serde(default = "default_max_chunks")]
    pub max_chunks: usize,
    #[serde(default)]
    pub relevance_threshold: Option<f64>,
    /// Deprecated: goal-typed memories were folded into identity by migration 45
    /// (Phase 0). Daemon ignores this field — no goal-load path exists anymore.
    /// Field stays for wire backward compat; will be removed in 0.4.
    #[deprecated(
        since = "0.3.2",
        note = "Goal taxonomy folded into Identity by migration 45 (Phase 0). \
                Daemon ignores this field. Will be removed in 0.4."
    )]
    #[serde(default = "default_true")]
    pub include_goals: bool,
    #[serde(default, alias = "domain")]
    pub space: Option<String>,
}

// ===== Knowledge graph =====

#[derive(Debug, Serialize, Deserialize)]
pub struct CreateEntityRequest {
    pub name: String,
    pub entity_type: String,
    #[serde(
        default,
        alias = "domain",
        skip_serializing_if = "WriteSpaceTarget::is_inherit"
    )]
    pub space: WriteSpaceTarget,
    #[serde(default)]
    pub source_agent: Option<String>,
    #[serde(default)]
    pub confidence: Option<f32>,
}

#[doc(hidden)]
#[derive(Debug, Serialize, Deserialize)]
pub struct CreateRelationRequest {
    pub from_entity: String,
    pub to_entity: String,
    pub relation_type: String,
    #[serde(default)]
    pub source_agent: Option<String>,
    #[serde(default)]
    pub confidence: Option<f64>,
    #[serde(default)]
    pub explanation: Option<String>,
    #[serde(default)]
    pub source_memory_id: Option<String>,
    /// Verbatim source-memory quote the relation was extracted from (M3g
    /// span capture). Daemon-internal (KG extraction) only -- the
    /// `/api/memory/relations` wire route strips this to `None` before the
    /// core call, so an agent-triggered request can never set it.
    #[serde(default)]
    pub span: Option<String>,
    /// Extraction model id/version that produced `span` (§6.6 versioning).
    /// Daemon-internal only -- stripped on the wire route, same as `span`.
    #[serde(default)]
    pub model_version: Option<String>,
    /// `extract_knowledge_graph` prompt version that produced `span`.
    /// Daemon-internal only -- stripped on the wire route, same as `span`.
    #[serde(default)]
    pub prompt_version: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct AddObservationRequest {
    pub entity_id: String,
    pub content: String,
    #[serde(default)]
    pub source_agent: Option<String>,
    #[serde(default)]
    pub confidence: Option<f32>,
}

#[doc(hidden)]
#[derive(Debug, Serialize, Deserialize)]
pub struct LinkEntityRequest {
    pub source_id: String,
    pub entity_id: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ListEntitiesRequest {
    #[serde(default)]
    pub entity_type: Option<String>,
    #[serde(default, alias = "domain")]
    pub space: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SearchEntitiesRequest {
    pub query: String,
    #[serde(default = "default_entity_search_limit")]
    pub limit: usize,
}

// ===== Profile & Agents =====

#[derive(Debug, Serialize, Deserialize)]
pub struct UpdateProfileRequest {
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub display_name: Option<String>,
    #[serde(default)]
    pub email: Option<String>,
    #[serde(default)]
    pub bio: Option<String>,
    #[serde(default)]
    pub avatar_path: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct UpdateAgentRequest {
    #[serde(default)]
    pub agent_type: Option<String>,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub enabled: Option<bool>,
    #[serde(default)]
    pub trust_level: Option<String>,
    /// Empty string clears the field; None leaves it unchanged.
    #[serde(default)]
    pub display_name: Option<String>,
}

// ===== Spaces =====

#[derive(Debug, Serialize, Deserialize)]
pub struct CreateSpaceRequest {
    pub name: String,
    pub description: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct UpdateSpaceRequest {
    pub new_name: Option<String>,
    pub description: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SetDefaultSpaceRequest {
    pub space_id: String,
}

// ===== Concepts =====

#[doc(hidden)]
#[derive(Debug, Serialize, Deserialize)]
pub struct CreateConceptRequest {
    pub title: String,
    pub content: String,
    #[serde(default)]
    pub summary: Option<String>,
    #[serde(default)]
    pub entity_id: Option<String>,
    #[serde(
        default,
        alias = "domain",
        skip_serializing_if = "WriteSpaceTarget::is_inherit"
    )]
    pub space: WriteSpaceTarget,
    #[serde(default)]
    pub source_memory_ids: Vec<String>,
    #[serde(default)]
    pub creation_kind: Option<String>,
    /// Dedicated workspace axis (P3), the authoritative axis for page
    /// filtering. When Some, persisted to `pages.workspace`. Distinct from
    /// `space`, the legacy scope input defaulted from the `X-Origin-Space`
    /// header when the body omits it (see `handle_create_page`).
    #[serde(default)]
    pub workspace: Option<String>,
}

/// First durable snapshot for a human-authored Page draft.
///
/// The client only sends this request once either `title` or `content` is
/// meaningful. Both fields default to empty so title-first and body-first
/// writing flows share one wire shape.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CreatePageDraftRequest {
    /// Stable client-generated id used to make an ambiguous create retry safe.
    pub draft_id: String,
    pub title: String,
    pub content: String,
    pub space: Option<String>,
    space_provided: bool,
}

impl CreatePageDraftRequest {
    /// Build a request with an explicit `space` value.
    ///
    /// Passing `None` serializes as `"space": null`.
    pub fn new(draft_id: String, title: String, content: String, space: Option<String>) -> Self {
        Self {
            draft_id,
            title,
            content,
            space,
            space_provided: true,
        }
    }

    /// Build a request that omits `space`, allowing the server to inherit the
    /// `X-Wenlan-Space` request header.
    pub fn new_inheriting_header_space(draft_id: String, title: String, content: String) -> Self {
        Self {
            draft_id,
            title,
            content,
            space: None,
            space_provided: false,
        }
    }

    /// Whether the JSON body contained a `space` key.
    ///
    /// This distinguishes an omitted key (inherit the request header) from an
    /// explicit `null` (clear the header-provided Space).
    pub fn space_was_provided(&self) -> bool {
        self.space_provided
    }
}

impl Serialize for CreatePageDraftRequest {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;

        let field_count = if self.space_provided { 4 } else { 3 };
        let mut state = serializer.serialize_struct("CreatePageDraftRequest", field_count)?;
        state.serialize_field("draft_id", &self.draft_id)?;
        state.serialize_field("title", &self.title)?;
        state.serialize_field("content", &self.content)?;
        if self.space_provided {
            state.serialize_field("space", &self.space)?;
        }
        state.end()
    }
}

impl<'de> Deserialize<'de> for CreatePageDraftRequest {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct Wire {
            draft_id: String,
            #[serde(default)]
            title: String,
            #[serde(default)]
            content: String,
            #[serde(default, deserialize_with = "double_option")]
            space: Option<Option<String>>,
        }

        let wire = Wire::deserialize(deserializer)?;
        let (space_provided, space) = match wire.space {
            Some(space) => (true, space),
            None => (false, None),
        };
        Ok(Self {
            draft_id: wire.draft_id,
            title: wire.title,
            content: wire.content,
            space,
            space_provided,
        })
    }
}

/// Complete replacement snapshot for an existing Page draft.
#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct UpdatePageDraftRequest {
    pub expected_version: i64,
    pub title: String,
    pub content: String,
    pub space: Option<String>,
}

impl<'de> Deserialize<'de> for UpdatePageDraftRequest {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct Wire {
            expected_version: i64,
            title: String,
            content: String,
            #[serde(default, deserialize_with = "double_option")]
            space: Option<Option<String>>,
        }

        let wire = Wire::deserialize(deserializer)?;
        let space = wire
            .space
            .ok_or_else(|| serde::de::Error::missing_field("space"))?;
        Ok(Self {
            expected_version: wire.expected_version,
            title: wire.title,
            content: wire.content,
            space,
        })
    }
}

/// Optimistic-concurrency body shared by draft publish and discard.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PageDraftVersionRequest {
    pub expected_version: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "action", rename_all = "snake_case")]
pub enum AcceptRefinementRequest {
    Accept {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        notes: Option<String>,
    },
    PickSpace {
        space: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        notes: Option<String>,
    },
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SearchPagesRequest {
    pub query: String,
    #[serde(default)]
    pub limit: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub page_type: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub space: Option<String>,
}

// ===== Ingest =====

#[derive(Debug, Serialize, Deserialize)]
pub struct IngestTextRequest {
    pub source: String,
    pub source_id: String,
    pub title: String,
    pub content: String,
    pub url: Option<String>,
    pub metadata: Option<HashMap<String, String>>,
}

#[doc(hidden)]
#[derive(Debug, Serialize, Deserialize)]
pub struct IngestWebpageRequest {
    pub url: String,
    pub title: String,
    pub content: String,
    pub metadata: Option<HashMap<String, String>>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct IngestMemoryRequest {
    pub source: String,
    pub source_id: String,
    pub title: String,
    pub content: String,
    pub url: Option<String>,
    pub tags: Option<Vec<String>>,
    pub metadata: Option<HashMap<String, String>>,
}

// ===== Sources =====

#[doc(hidden)]
#[derive(Debug, Serialize, Deserialize)]
pub struct AddSourceRequest {
    pub source_type: String,
    /// Filesystem path as a string. Kept as `String` (not `PathBuf`) because
    /// this is an HTTP wire format — the handler converts it to `PathBuf`.
    pub path: String,
}

// ===== Config =====

/// Distinguishes an omitted JSON field (outer None) from an explicit `null`
/// (Some(None)). Used for presence-sensitive wire fields.
fn double_option<'de, T, D>(de: D) -> Result<Option<Option<T>>, D::Error>
where
    T: serde::Deserialize<'de>,
    D: serde::Deserializer<'de>,
{
    serde::Deserialize::deserialize(de).map(Some)
}

#[derive(Debug, Serialize, Deserialize)]
pub struct UpdateConfigRequest {
    #[serde(default)]
    pub skip_apps: Option<Vec<String>>,
    #[serde(default)]
    pub skip_title_patterns: Option<Vec<String>>,
    #[serde(default)]
    pub private_browsing_detection: Option<bool>,
    #[serde(default)]
    pub setup_completed: Option<bool>,
    #[serde(default)]
    pub clipboard_enabled: Option<bool>,
    #[serde(default)]
    pub screen_capture_enabled: Option<bool>,
    #[serde(default)]
    pub remote_access_enabled: Option<bool>,
    /// Anthropic model used for fast/routine tasks (e.g. classification, tagging).
    #[serde(default)]
    pub routine_model: Option<String>,
    /// Anthropic model used for synthesis tasks (e.g. distillation, narrative).
    #[serde(default)]
    pub synthesis_model: Option<String>,
    /// Base URL for an OpenAI-compatible external LLM endpoint.
    #[serde(default)]
    pub external_llm_endpoint: Option<String>,
    /// Model identifier to use with the external LLM endpoint.
    #[serde(default)]
    pub external_llm_model: Option<String>,
    /// API key for the external endpoint. Tri-state: omitted = preserve stored
    /// key; `null` or `""` = clear; non-empty = replace. Never echoed back.
    #[serde(
        default,
        deserialize_with = "double_option",
        skip_serializing_if = "Option::is_none"
    )]
    pub external_llm_api_key: Option<Option<String>>,
    /// Per-job source pin for everyday work: `"anthropic"` | `"external"` |
    /// `"on_device"`. Omitted = preserve; `""` = clear; other values are
    /// validated by the config route.
    #[serde(default)]
    pub everyday_source: Option<String>,
    /// Per-job source pin for synthesis: `"anthropic"` | `"external"`
    /// (`"on_device"` only when the compile gate is set). Omitted = preserve;
    /// `""` = clear; validated by the config route.
    #[serde(default)]
    pub synthesis_source: Option<String>,
    /// Gates the proactive Page-Map suggestion phase in the scheduler. Omitted =
    /// preserve stored value; present = set. Never gates the explicit improve route.
    #[serde(default)]
    pub page_map_auto_suggest: Option<bool>,
}

// ===== Chunks / indexed files =====

#[derive(Debug, Serialize, Deserialize)]
pub struct DeleteByTimeRangeRequest {
    pub start: i64,
    pub end: i64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct BulkDeleteItem {
    pub source: String,
    pub source_id: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct BulkDeleteRequest {
    pub items: Vec<BulkDeleteItem>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct UpdateChunkRequest {
    pub content: String,
}

// ===== Entity / Observation CRUD =====

#[derive(Debug, Serialize, Deserialize)]
pub struct ConfirmEntityRequest {
    #[serde(default = "default_confirmed")]
    pub confirmed: bool,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct AddEntityObservationRequest {
    pub content: String,
    #[serde(default)]
    pub source_agent: Option<String>,
    #[serde(default)]
    pub confidence: Option<f32>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct UpdateObservationRequest {
    pub content: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ConfirmObservationRequest {
    #[serde(default = "default_confirmed")]
    pub confirmed: bool,
}

// ===== Spaces extended =====

#[derive(Debug, Serialize, Deserialize)]
pub struct ReorderSpaceRequest {
    pub name: String,
    pub new_order: i64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SetDocumentSpaceRequest {
    pub space_name: String,
}

// ===== Tags =====

#[derive(Debug, Serialize, Deserialize)]
pub struct SetDocumentTagsRequest {
    #[serde(default)]
    pub source: Option<String>,
    pub tags: Vec<String>,
}

// ===== Memory update =====

#[derive(Debug, Serialize, Deserialize)]
pub struct UpdateMemoryRequest {
    #[serde(default)]
    pub content: Option<String>,
    #[serde(default, alias = "domain")]
    pub space: Option<String>,
    #[serde(default)]
    pub confirmed: Option<bool>,
    #[serde(default)]
    pub memory_type: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SetStabilityRequest {
    pub stability: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct CorrectMemoryRequest {
    pub correction_prompt: String,
}

// ===== Concepts update =====

#[derive(Debug, Serialize, Deserialize)]
pub struct UpdatePageRequest {
    pub content: String,
    /// Source memory IDs to associate with this page version.
    /// Omitted or empty by HTTP callers that preserve existing sources;
    /// always populated by `post_write::update_page`.
    #[serde(default)]
    pub source_memory_ids: Vec<String>,
    /// Optimistic-concurrency guard for the M0 write gate. `Some(v)` lands the
    /// write only while the stored page is still at version `v`; a mismatch is
    /// refused instead of overwriting whatever landed in between.
    ///
    /// Omitted by legacy clients, in which case the server guards on the version
    /// it loaded to make the ownership decision — so the decision and the write
    /// always describe the same row.
    #[serde(default)]
    pub expected_version: Option<i64>,
    /// Retry identity. Sent together, `caller_id` and `operation_id` make the
    /// write replayable: the same pair with the same request replays the
    /// stored response instead of writing again, and the same pair with a
    /// different request is a conflict. This is what turns "the response was
    /// lost, the client retried" into a no-op rather than a second version.
    ///
    /// Either one alone is ignored — an operation id is only meaningful
    /// within the caller that minted it.
    #[serde(default)]
    pub caller_id: Option<String>,
    #[serde(default)]
    pub operation_id: Option<String>,
}

/// Body for `PUT /api/pages/{id}` — agent-side refresh of a stale page.
///
/// Distinct from `UpdatePageRequest` (manual content edit via POST) because:
///  - `source_memory_ids` is replaced, not preserved.
///  - `summary` is optionally updated.
///  - The handler clears `stale_reason` in the same transaction.
///
/// v1 deliberately excludes title / entity_id / space changes — slug rename
/// has its own concurrent-read failure mode and lands as a separate route.
#[derive(Debug, Serialize, Deserialize)]
pub struct RefreshPageRequest {
    pub content: String,
    pub source_memory_ids: Vec<String>,
    #[serde(default)]
    pub summary: Option<String>,
}

// ===== Concept Export =====

/// Request body for `POST /api/pages/export` (bulk export all pages to an Obsidian vault).
#[derive(Debug, Deserialize, Serialize)]
pub struct ExportPagesRequest {
    pub vault_path: Option<String>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct ExportPageRequest {
    pub vault_path: String,
}

// ===== LLM test =====

/// `POST /api/llm/test` — probe an OpenAI-compatible LLM endpoint with a 1-shot prompt.
/// Used by the app settings UI to validate a custom endpoint before saving.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestLlmRequest {
    pub endpoint: String,
    pub model: String,
    /// Optional override prompt. Defaults to "Say 'hello' and nothing else." server-side.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt: Option<String>,
    /// Optional bearer key for this probe only — not persisted.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub api_key: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestLlmResponse {
    pub response: String,
}

// ===== On-device model =====

/// Body for `POST /api/on-device-model/download`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OnDeviceModelRequest {
    pub model_id: String,
}

// ===== Default value functions =====

fn default_limit() -> usize {
    10
}

fn default_list_limit() -> usize {
    100
}

fn default_confirmed() -> bool {
    true
}

fn default_max_chunks() -> usize {
    5
}

fn default_true() -> bool {
    true
}

fn default_entity_search_limit() -> usize {
    20
}

#[cfg(test)]
mod search_pages_page_type_test {
    use super::*;

    #[test]
    fn search_pages_request_accepts_page_type() {
        let json = r#"{"query":"foo","limit":10,"page_type":"recap"}"#;
        let parsed: SearchPagesRequest = serde_json::from_str(json).unwrap();
        assert_eq!(parsed.page_type.as_deref(), Some("recap"));
    }

    #[test]
    fn search_pages_request_page_type_optional() {
        let json = r#"{"query":"foo","limit":10}"#;
        let parsed: SearchPagesRequest = serde_json::from_str(json).unwrap();
        assert!(parsed.page_type.is_none());
    }

    #[test]
    fn search_pages_request_accepts_optional_space() {
        let json = r#"{"query":"foo","space":"work"}"#;
        let parsed: SearchPagesRequest = serde_json::from_str(json).unwrap();
        assert_eq!(parsed.space.as_deref(), Some("work"));

        let omitted: SearchPagesRequest = serde_json::from_str(r#"{"query":"foo"}"#).unwrap();
        assert!(omitted.space.is_none());
    }
}

#[cfg(test)]
mod list_memories_confirmed_test {
    use super::*;

    #[test]
    fn confirmed_false_serializes_to_json() {
        let req = ListMemoriesRequest {
            memory_type: None,
            space: None,
            limit: 20,
            confirmed: Some(false),
        };
        let s = serde_json::to_string(&req).unwrap();
        assert!(s.contains("\"confirmed\":false"), "got: {s}");
    }

    #[test]
    fn confirmed_none_skips_serialization() {
        let req = ListMemoriesRequest {
            memory_type: None,
            space: None,
            limit: 20,
            confirmed: None,
        };
        let s = serde_json::to_string(&req).unwrap();
        assert!(!s.contains("confirmed"), "got: {s}");
    }
}

#[cfg(test)]
mod set_document_tags_test {
    use super::*;

    #[test]
    fn set_document_tags_request_accepts_optional_source() {
        let req: SetDocumentTagsRequest =
            serde_json::from_str(r#"{"source":"manual","tags":["rust"]}"#).unwrap();

        assert_eq!(req.source.as_deref(), Some("manual"));
        assert_eq!(req.tags, vec!["rust"]);
    }

    #[test]
    fn set_document_tags_request_defaults_source_for_old_payloads() {
        let req: SetDocumentTagsRequest = serde_json::from_str(r#"{"tags":["rust"]}"#).unwrap();

        assert!(req.source.is_none());
        assert_eq!(req.tags, vec!["rust"]);
    }
}

#[cfg(test)]
mod on_device_model_request_test {
    use super::*;

    #[test]
    fn on_device_model_request_round_trips_model_id() {
        let req: OnDeviceModelRequest = serde_json::from_str(r#"{"model_id":"qwen3-4b"}"#).unwrap();

        assert_eq!(req.model_id, "qwen3-4b");
        assert_eq!(
            serde_json::to_string(&req).unwrap(),
            r#"{"model_id":"qwen3-4b"}"#
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn update_config_request_external_key_tristate() {
        let r: UpdateConfigRequest = serde_json::from_str("{}").unwrap();
        assert_eq!(r.external_llm_api_key, None); // omitted
        let r: UpdateConfigRequest =
            serde_json::from_str(r#"{"external_llm_api_key":null}"#).unwrap();
        assert_eq!(r.external_llm_api_key, Some(None)); // explicit null
        let r: UpdateConfigRequest =
            serde_json::from_str(r#"{"external_llm_api_key":"sk-x"}"#).unwrap();
        assert_eq!(r.external_llm_api_key, Some(Some("sk-x".to_string())));
    }

    #[test]
    fn test_llm_request_api_key_optional() {
        let r: TestLlmRequest =
            serde_json::from_str(r#"{"endpoint":"http://x","model":"m"}"#).unwrap();
        assert!(r.api_key.is_none());
    }
}

// ===== UI presence (M5 D7) =====

/// A capability the app's Tauri backend minted for one gesture, as it crosses
/// the wire.
///
/// Binding spec: `docs/plans/2026-07-27-m5-presence-threat-model.md`. `nonce`
/// and `mac` are lowercase hex of the app's raw bytes; the daemon needs the
/// raw nonce to recompute the HMAC, and stores only its digest.
///
/// **Deserialize only, on purpose.** The app's own `PresenceCapability`
/// refuses to implement `Serialize` so no Tauri command can hand a capability
/// to JavaScript (T1). This is the same guarantee facing the other way: with no
/// `Serialize`, a receipt, a log line, or an error body physically cannot carry
/// one, so §7's redaction contract holds by construction rather than by every
/// future call site remembering it.
#[derive(Clone, Deserialize)]
pub struct PresenceCapability {
    pub protocol_version: u32,
    /// `attest_claim` or `review_page`. A string rather than an enum so an
    /// unknown action is a refusal the daemon words, not a deserialize error
    /// that echoes the submitted value back.
    pub action: String,
    pub target_ids: Vec<String>,
    /// Digest of the exact content the gesture was made against (T6).
    pub base_digest: String,
    pub caller_id: String,
    pub operation_id: String,
    pub minted_at: u64,
    pub expires_at: u64,
    pub nonce: String,
    pub mac: String,
}

/// Redacted by construction (§7): the derived `Debug` would print the HMAC and
/// the raw nonce into the first log line that ever formatted a request.
impl std::fmt::Debug for PresenceCapability {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PresenceCapability")
            .field("protocol_version", &self.protocol_version)
            .field("action", &self.action)
            .field("target_ids", &self.target_ids)
            .field("base_digest", &self.base_digest)
            .field("caller_id", &self.caller_id)
            .field("operation_id", &self.operation_id)
            .field("minted_at", &self.minted_at)
            .field("expires_at", &self.expires_at)
            .field("nonce", &"<redacted>")
            .field("mac", &"<redacted>")
            .finish()
    }
}

/// Mark one page as human-reviewed.
///
/// The capability already names the page and the exact content the human read,
/// so there is nothing else to send. In particular the reviewed version is not
/// a field: the daemon derives it from the row whose content matches
/// `base_digest`, rather than believing a caller about which version it saw.
#[derive(Debug, Clone, Deserialize)]
pub struct ReviewPageRequest {
    pub presence: PresenceCapability,
}
