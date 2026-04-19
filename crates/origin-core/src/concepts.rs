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
    pub source_memory_ids: Vec<String>,
    pub version: i64,
    pub status: String,
    pub created_at: String,
    pub last_compiled: String,
    pub last_modified: String,
}

impl Concept {
    pub fn new_id() -> String {
        format!("concept_{}", uuid::Uuid::new_v4())
    }
}
