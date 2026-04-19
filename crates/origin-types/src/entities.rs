// SPDX-License-Identifier: Apache-2.0
//! Knowledge graph types -- entities, observations, relations.

use serde::{Deserialize, Serialize};

/// A knowledge graph entity.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Entity {
    pub id: String,
    pub name: String,
    pub entity_type: String,
    pub domain: Option<String>,
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

/// A pending entity suggestion from the refinement queue.
#[derive(Debug, Serialize, Deserialize)]
pub struct EntitySuggestion {
    pub id: String,
    pub entity_name: Option<String>,
    pub source_ids: Vec<String>,
    pub confidence: f64,
    pub created_at: String,
}
