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
    /// Relevance score from search (0.0-1.0). Only populated by `search_concepts`;
    /// zero for persisted/non-search contexts.
    #[serde(default, skip_serializing_if = "is_zero_f32")]
    pub relevance_score: f32,
}

fn is_zero_f32(v: &f32) -> bool {
    *v == 0.0
}

impl Concept {
    pub fn new_id() -> String {
        format!("concept_{}", uuid::Uuid::new_v4())
    }
}

/// Filter concepts by source overlap with search results.
///
/// A concept is contextually relevant if the memories it was compiled from
/// overlap with the memories that search_memory returned for this query.
/// This is the strongest relevance signal: it answers "is this concept about
/// the thing I'm searching for?" rather than relying on embedding similarity
/// (which we proved doesn't discriminate between good and garbage concepts).
///
/// `min_overlap`: minimum number of search result source_ids that must appear
/// in the concept's `source_memory_ids`. Recommended: 2 (filters noise while
/// keeping concepts with genuine topical overlap).
pub fn filter_concepts_by_source_overlap(
    concepts: &[Concept],
    search_result_source_ids: &std::collections::HashSet<String>,
    min_overlap: usize,
) -> Vec<Concept> {
    concepts
        .iter()
        .filter(|c| {
            let overlap = c
                .source_memory_ids
                .iter()
                .filter(|sid| search_result_source_ids.contains(sid.as_str()))
                .count();
            overlap >= min_overlap
        })
        .cloned()
        .collect()
}
