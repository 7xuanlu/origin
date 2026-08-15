// SPDX-License-Identifier: Apache-2.0
//! Knowledge graph types -- entities, observations, relations.

use serde::{Deserialize, Serialize};

/// A knowledge graph entity.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Entity {
    pub id: String,
    pub name: String,
    pub entity_type: String,
    #[serde(default, alias = "domain")]
    pub space: Option<String>,
    pub source_agent: Option<String>,
    pub confidence: Option<f32>,
    pub confirmed: bool,
    pub created_at: i64,
    pub updated_at: i64,
}

/// An entity search result with distance score.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntitySearchResult {
    pub entity: Entity,
    pub distance: f32,
}

/// Full entity detail including observations and relations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityDetail {
    pub entity: Entity,
    pub observations: Vec<Observation>,
    pub relations: Vec<RelationWithEntity>,
}

/// An observation attached to an entity.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Observation {
    pub id: String,
    pub entity_id: String,
    pub content: String,
    pub source_agent: Option<String>,
    pub confidence: Option<f32>,
    pub confirmed: bool,
    pub created_at: i64,
}

/// A relation between two entities.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Relation {
    pub id: String,
    pub from_entity: String,
    pub to_entity: String,
    pub relation_type: String,
    pub source_agent: Option<String>,
    pub created_at: i64,
}

/// A relation with resolved entity info (for detail views).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelationWithEntity {
    pub id: String,
    pub relation_type: String,
    pub direction: String,
    pub entity_id: String,
    pub entity_name: String,
    pub entity_type: String,
    pub source_agent: Option<String>,
    pub created_at: i64,
}

/// A relation with both entity names resolved, for the home page connections feed.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecentRelation {
    pub id: String,
    pub from_entity_id: String,
    pub relation_type: String,
    pub to_entity_id: String,
    pub from_entity_name: String,
    pub to_entity_name: String,
    /// Unix seconds (same unit as the `created_at` column in the `relations` table).
    pub created_at_ms: i64,
}

/// One bulk read of the whole knowledge graph for a read scope: every entity
/// the scope can see, every live relation whose BOTH endpoints are in that
/// entity set, and the memories linked to at least one of those entities.
///
/// Exists so the desktop Graph view can draw the complete graph from ONE
/// request instead of fanning out per-entity detail fetches (which capped the
/// drawn graph at the first 20 entities and rendered every other connected
/// entity as an isolate).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgeGraphResponse {
    /// Same rows `/api/memory/entities/list` returns for this scope.
    pub entities: Vec<Entity>,
    /// Every live entity<->entity relation with both endpoints in `entities`.
    pub relations: Vec<GraphRelation>,
    /// Only memories with at least one link into `entities`.
    pub memories: Vec<GraphMemoryNode>,
    /// memory_id <-> entity_id; both endpoints are present above.
    pub memory_links: Vec<GraphMemoryLink>,
}

/// A relation edge as the bulk graph read returns it: both endpoints by id,
/// no resolved neighbour names (the caller already holds every entity row).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphRelation {
    pub id: String,
    pub from_entity: String,
    pub to_entity: String,
    pub relation_type: String,
    pub source_agent: Option<String>,
    pub created_at: i64,
}

/// A memory drawn as a graph node.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphMemoryNode {
    pub source_id: String,
    pub title: String,
    pub memory_type: Option<String>,
    pub space: Option<String>,
    pub confirmed: bool,
    pub last_modified: i64,
}

/// A memory-to-entity link, drawn as an edge.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphMemoryLink {
    pub memory_id: String,
    pub entity_id: String,
}

/// A pending entity suggestion from the refinement queue.
#[derive(Debug, Serialize, Deserialize)]
pub struct EntitySuggestion {
    pub id: String,
    pub entity_name: Option<String>,
    pub source_ids: Vec<String>,
    pub confidence: f64,
    pub created_at: String,
}
