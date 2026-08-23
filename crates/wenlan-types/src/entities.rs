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
    /// Former names this entity has absorbed via a merge or an explicit
    /// alias declaration (lowercase). Empty for an entity with no aliases.
    #[serde(default)]
    pub aliases: Vec<String>,
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
    /// Memories linked to at least one entity in `entities`, plus the ones
    /// cited by a page in `pages`.
    pub memories: Vec<GraphMemoryNode>,
    /// memory_id <-> entity_id; both endpoints are present above.
    pub memory_links: Vec<GraphMemoryLink>,
    /// Wiki pages: everything that is not an entity shadow page.
    pub pages: Vec<GraphPageNode>,
    /// Typed edges with a page on at least one end. Every endpoint id is
    /// present in the collection its `kind` names.
    pub page_links: Vec<GraphPageLink>,
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

/// A wiki page drawn as a graph node.
///
/// "Wiki page" is every page that is not an entity's `kind='entity'` dual-write
/// shadow: the distilled/authored/research/source pages a human would call a
/// page. Entity shadows stay out because the entity itself is already a node.
///
/// No `community_id`: neither store can answer it for a wiki page.
/// `pages.community_id` is written only onto `kind='entity'` shadows by
/// `detect_communities`, and `page_community_assignments` is a fenced routing
/// table whose only trusted reader (`list_community_page_assignments`) gates
/// every row behind five freshness joins and a `state` check. Reading either
/// one raw would report a stale, held or dropped assignment as a fact, so the
/// map derives a page's region from what it links to instead.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphPageNode {
    pub id: String,
    pub title: String,
    /// The page's space NAME, matching the vocabulary `Entity.space` and
    /// `GraphMemoryNode.space` speak (scoping itself keys on `workspace`).
    pub space: Option<String>,
    pub creation_kind: String,
    /// Set when the page is *about* an entity, which is a different thing from
    /// being that entity's shadow page.
    pub entity_id: Option<String>,
    /// RFC 3339, as `pages.last_modified` stores it.
    pub last_modified: String,
}

/// One endpoint of a [`GraphPageLink`]: which collection to look the id up in.
///
/// `kind` is `"page"`, `"entity"` or `"memory"`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphRef {
    pub kind: String,
    pub id: String,
}

/// A typed edge with a page on at least one end.
///
/// `link_type` is `"wikilink"` (a resolved `[[link]]`, page->page or
/// page->entity), `"about"` (`pages.entity_id`), or `"cites"` (a page->memory
/// citation edge).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphPageLink {
    pub from: GraphRef,
    pub to: GraphRef,
    pub link_type: String,
}
