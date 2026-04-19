// SPDX-License-Identifier: Apache-2.0
//! API request types for all HTTP endpoints.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ===== Memory CRUD =====

#[derive(Debug, Serialize, Deserialize)]
pub struct StoreMemoryRequest {
    pub content: String,
    #[serde(default)]
    pub memory_type: Option<String>,
    #[serde(default)]
    pub domain: Option<String>,
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
    #[serde(default)]
    pub domain: Option<String>,
    #[serde(default)]
    pub source_agent: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ListMemoriesRequest {
    #[serde(default)]
    pub memory_type: Option<String>,
    #[serde(default)]
    pub domain: Option<String>,
    #[serde(default = "default_list_limit")]
    pub limit: usize,
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
}

// ===== General search/context =====

#[derive(Debug, Serialize, Deserialize)]
pub struct SearchRequest {
    pub query: String,
    #[serde(default = "default_limit")]
    pub limit: usize,
    pub source_filter: Option<String>,
    #[serde(default)]
    pub domain: Option<String>,
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
    #[serde(default = "default_true")]
    pub include_goals: bool,
    #[serde(default)]
    pub domain: Option<String>,
}

// ===== Knowledge graph =====

#[derive(Debug, Serialize, Deserialize)]
pub struct CreateEntityRequest {
    pub name: String,
    pub entity_type: String,
    #[serde(default)]
    pub domain: Option<String>,
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
    #[serde(default)]
    pub domain: Option<String>,
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
    #[serde(default)]
    pub domain: Option<String>,
    #[serde(default)]
    pub source_memory_ids: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SearchConceptsRequest {
    pub query: String,
    #[serde(default)]
    pub limit: Option<usize>,
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
    pub selection_capture_enabled: Option<bool>,
    #[serde(default)]
    pub remote_access_enabled: Option<bool>,
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
    pub tags: Vec<String>,
}

// ===== Memory update =====

#[derive(Debug, Serialize, Deserialize)]
pub struct UpdateMemoryRequest {
    #[serde(default)]
    pub content: Option<String>,
    #[serde(default)]
    pub domain: Option<String>,
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
pub struct UpdateConceptRequest {
    pub content: String,
}

// ===== Concept Export =====

#[derive(Debug, Deserialize, Serialize)]
pub struct ExportConceptRequest {
    pub vault_path: String,
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
