// SPDX-License-Identifier: AGPL-3.0-only
//! Knowledge compilation: concepts are structured documents compiled from memory clusters.

use serde::{Deserialize, Serialize};

/// A compiled knowledge concept — structured, cross-referenced, backed by source memories.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Concept {
    pub id: String,
    pub title: String,
    pub summary: Option<String>,
    pub content: String,
    pub entity_id: Option<String>,
    pub domain: Option<String>,
    /// Kept for dual-write transition; prefer concept_sources join table for new reads.
    pub source_memory_ids: Vec<String>,
    pub version: i64,
    pub status: String,
    pub created_at: String,
    pub last_compiled: String,
    pub last_modified: String,
    /// How many source memories were updated since last distillation.
    pub sources_updated_count: i64,
    /// Why this concept is stale: "source_updated" | "source_conflict" | None.
    pub stale_reason: Option<String>,
    /// True if a human has edited this concept's content directly.
    pub user_edited: bool,
}

impl Concept {
    pub fn new_id() -> String {
        format!("concept_{}", uuid::Uuid::new_v4())
    }
}
