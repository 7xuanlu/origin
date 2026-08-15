// SPDX-License-Identifier: AGPL-3.0-only
import type { Entity, EntityDetail, KnowledgeGraph, RelationWithEntity } from "../tauri";

// The renderer-neutral graph every view consumes. Daemon response shapes are
// translated into this once (here); no view reads raw relation records for
// drawing. Keeps renderers (SVG Focus now, Reagraph Atlas later) as leaves.
//
// Parallel edges: distinct relation ids with identical (source, type, target)
// intentionally stay distinct edges here — collapsing them is a view decision.
// ponytail: Atlas (src/lib/graph/atlas.ts) keeps them distinct and lets sigma
// draw them overlapped; the Focus ego view collapses per neighbor+direction
// instead (FocusGraph's deriveNeighbors) — renderers make different calls on
// purpose, this file doesn't referee it.

export type GraphNodeKind = "entity" | "memory" | "page";

export interface GraphNode {
  id: string;
  kind: GraphNodeKind;
  name: string;
  /** Daemon vocabulary string, verbatim (e.g. "person", "technology"). */
  entityType: string;
  /** null = unknown (matches the space convention below). */
  confirmed: boolean | null;
  /** Number of distinct edges touching this node in THIS model (a self-loop counts once, not twice). */
  degree: number;
  /** Entity's space (falling back to domain), matching the entity page's
   *  `space ?? domain` precedent. null for relation-only synthesized
   *  neighbors, whose owning space this model never learns — the M6
   *  cartography partition (see cartography.ts) treats that as its own
   *  bucket rather than guessing. */
  space: string | null;
  createdAt: number;
  updatedAt: number;
}

export interface GraphEdge {
  id: string;
  /** Semantic origin — direction is normalized so source→target is the verb's subject→object. */
  source: string;
  target: string;
  /** relation_type verb, verbatim. */
  type: string;
  /** null until wenlan-types exposes confidence on RelationWithEntity. */
  confidence: number | null;
  createdAt: number;
}

export interface GraphModel {
  nodes: GraphNode[];
  edges: GraphEdge[];
  /** Honesty metadata: how many entities we actually pulled relations for. */
  coverage: { relationsFetchedFor: number; totalEntities: number };
}

// wenlan-types 0.12.0 carries no confidence field; read it defensively so
// the day the daemon adds it the model lights up with zero call-site
// changes. Never fabricate a value — absent stays null.
function confidenceOf(rel: RelationWithEntity): number | null {
  const value = (rel as { confidence?: number | null }).confidence;
  return typeof value === "number" ? value : null;
}

/**
 * The space an entity is grouped under: its own `space`, else its `domain`.
 * `||`, not `??` — an entity carrying an empty-string space still has a domain
 * worth grouping under, and `??` would keep the "" and read as unscoped
 * downstream (cartography's isUnscopedSpace treats it as falsy). Every place
 * that derives a space from an entity goes through here so the rules agree.
 */
export function entitySpace(entity: Pick<Entity, "space" | "domain">): string | null {
  return entity.space || entity.domain || null;
}

function nodeFromEntity(entity: Entity): GraphNode {
  return {
    id: entity.id,
    kind: "entity",
    name: entity.name,
    entityType: entity.entity_type,
    confirmed: entity.confirmed,
    degree: 0,
    space: entitySpace(entity),
    createdAt: entity.created_at,
    updatedAt: entity.updated_at,
  };
}

// A neighbor that appears only inside a relation (not in the entities list) —
// we know only what the relation record carries. confirmed is unknown here
// (null), not false: with the daemon's top-20 detail-fetch cap, synthesized
// neighbors are often real confirmed entities, and claiming false would be
// fabrication. Timestamps borrow the relation's (the entity's own are
// unavailable).
function nodeFromRelation(rel: RelationWithEntity): GraphNode {
  return {
    id: rel.entity_id,
    kind: "entity",
    name: rel.entity_name,
    entityType: rel.entity_type,
    confirmed: null,
    degree: 0,
    space: null,
    createdAt: rel.created_at,
    updatedAt: rel.created_at,
  };
}

// direction is relative to the home entity whose detail carried this relation.
// "incoming" means neighbor→home; "outgoing" means home→neighbor.
function edgeFromRelation(homeId: string, rel: RelationWithEntity): GraphEdge {
  const incoming = rel.direction === "incoming";
  const source = incoming ? rel.entity_id : homeId;
  const target = incoming ? homeId : rel.entity_id;
  const type = rel.relation_type;
  // A relation with no id (daemon gap) still needs a stable, non-empty id for
  // renderer keys — fall back to the same composite used to dedupe it below.
  const id = rel.id || `${source}:${type}:${target}`;
  return {
    id,
    source,
    target,
    type,
    confidence: confidenceOf(rel),
    createdAt: rel.created_at,
  };
}

/**
 * Fold a set of entities and their fetched details into one GraphModel.
 * Edges are direction-normalized and deduped; degree is computed over the
 * deduped set; neighbor-only entities are synthesized as nodes. No filtering
 * (caps / orphan-dropping) happens here — those are per-view decisions.
 */
export function buildGraphModel(entities: Entity[], details: EntityDetail[]): GraphModel {
  const nodes = new Map<string, GraphNode>();
  for (const entity of entities) nodes.set(entity.id, nodeFromEntity(entity));

  const edges = new Map<string, GraphEdge>();
  // Composites already registered — by an id-bearing edge or an idless one —
  // so a later idless mirror of the SAME relation (its id stripped on the
  // other endpoint) is recognized as a duplicate instead of double-counted.
  const seenComposites = new Set<string>();
  for (const detail of details) {
    const homeId = detail.entity.id;
    // The home entity should be in `entities`, but seed from the detail if not
    // so its edges always have both endpoints present.
    if (!nodes.has(homeId)) nodes.set(homeId, nodeFromEntity(detail.entity));

    for (const rel of detail.relations) {
      if (!nodes.has(rel.entity_id)) nodes.set(rel.entity_id, nodeFromRelation(rel));

      const edge = edgeFromRelation(homeId, rel);
      const composite = `${edge.source}:${edge.type}:${edge.target}`;
      if (rel.id) {
        // Dedupe by relation id (the same relation surfaces on both
        // endpoints' details, each carrying its real id).
        const key = `id:${rel.id}`;
        if (!edges.has(key)) edges.set(key, edge);
        seenComposites.add(composite);
      } else if (!seenComposites.has(composite)) {
        // No id (daemon gap): fall back to the endpoints+verb composite.
        // Distinct explicit ids that happen to share a composite still stay
        // distinct (parallel-edge policy, see module doc) — they never reach
        // this branch.
        seenComposites.add(composite);
        edges.set(`k:${composite}`, edge);
      }
    }
  }

  for (const edge of edges.values()) {
    const source = nodes.get(edge.source);
    if (source) source.degree += 1;
    // A self-loop (source === target) is one relation touching the node
    // once, not twice.
    if (edge.target !== edge.source) {
      const target = nodes.get(edge.target);
      if (target) target.degree += 1;
    }
  }

  // Coverage counts unique entities we fetched relations FOR, not the raw
  // array length — a duplicate detail for the same entity shouldn't inflate
  // the "N of M" honesty chip.
  const uniqueDetailEntityIds = new Set(details.map((d) => d.entity.id));

  return {
    nodes: Array.from(nodes.values()),
    edges: Array.from(edges.values()),
    coverage: { relationsFetchedFor: uniqueDetailEntityIds.size, totalEntities: entities.length },
  };
}

/**
 * The `entityType` every memory node carries. Not a daemon vocabulary word —
 * memories have no entity type — so palette/atlas can key their one muted
 * swatch and fixed size off it without a `kind` lookup at every call site.
 */
export const MEMORY_NODE_TYPE = "memory";

/** Edge verb for a memory -> entity link. Not a daemon relation type. */
export const MEMORY_EDGE_TYPE = "mentions";

/** Graph-node id for a memory. Prefixed so it can never collide with an
 *  entity id, and so click routing can tell the two apart. */
export function memoryNodeId(sourceId: string): string {
  return `mem:${sourceId}`;
}

/** The memory `source_id` behind a memory node id, or null for anything else. */
export function memorySourceId(nodeId: string): string | null {
  return nodeId.startsWith("mem:") ? nodeId.slice(4) : null;
}

/**
 * Narrow a bulk graph read to one space: entities by `entitySpace`, memories
 * by their own space, and relations/links to whatever survives. A relation
 * whose other endpoint was filtered out is DROPPED, not synthesized — the
 * bulk read carries every in-scope entity, so a missing endpoint means out of
 * scope rather than not-fetched-yet.
 */
export function filterKnowledgeGraph(graph: KnowledgeGraph, space: string | null): KnowledgeGraph {
  if (!space) return graph;
  const entities = graph.entities.filter((entity) => entitySpace(entity) === space);
  const kept = new Set(entities.map((entity) => entity.id));
  const memories = graph.memories.filter((memory) => memory.space === space);
  const keptMemories = new Set(memories.map((memory) => memory.source_id));
  return {
    entities,
    relations: graph.relations.filter(
      (relation) => kept.has(relation.from_entity) && kept.has(relation.to_entity),
    ),
    memories,
    memory_links: graph.memory_links.filter(
      (link) => kept.has(link.entity_id) && keptMemories.has(link.memory_id),
    ),
  };
}

/**
 * Fold one bulk graph read into a GraphModel: every entity as a node, every
 * relation as an edge, and every memory that links to a present entity as its
 * own node joined to those entities by `mentions` edges. Degree counts the
 * memory edges too, so an entity with only memories attached is not an
 * isolate. No filtering happens here — that is a per-view decision
 * (filterKnowledgeGraph above for the space filter, degree-0 hiding in
 * AtlasView).
 */
export function buildKnowledgeGraphModel(graph: KnowledgeGraph): GraphModel {
  const nodes = new Map<string, GraphNode>();
  for (const entity of graph.entities) nodes.set(entity.id, nodeFromEntity(entity));

  const edges = new Map<string, GraphEdge>();
  const seenComposites = new Set<string>();
  for (const relation of graph.relations) {
    // Both endpoints must be present. The daemon already guarantees it; a
    // dangling endpoint here would mean a filter ran, and inventing the
    // missing entity is exactly the fabrication the old top-20 fan-out was
    // forced into.
    if (!nodes.has(relation.from_entity) || !nodes.has(relation.to_entity)) continue;
    const type = relation.relation_type;
    const composite = `${relation.from_entity}:${type}:${relation.to_entity}`;
    const edge: GraphEdge = {
      id: relation.id || composite,
      source: relation.from_entity,
      target: relation.to_entity,
      type,
      confidence: null,
      createdAt: relation.created_at,
    };
    if (relation.id) {
      const key = `id:${relation.id}`;
      if (!edges.has(key)) edges.set(key, edge);
      seenComposites.add(composite);
    } else if (!seenComposites.has(composite)) {
      seenComposites.add(composite);
      edges.set(`k:${composite}`, edge);
    }
  }

  // Only memories with at least one link to a present entity become nodes —
  // a memory node with no edge would be an isolate the view hides anyway.
  const linksByMemory = new Map<string, string[]>();
  for (const link of graph.memory_links) {
    if (!nodes.has(link.entity_id)) continue;
    const list = linksByMemory.get(link.memory_id);
    if (list) list.push(link.entity_id);
    else linksByMemory.set(link.memory_id, [link.entity_id]);
  }
  for (const memory of graph.memories) {
    const linked = linksByMemory.get(memory.source_id);
    if (!linked) continue;
    const id = memoryNodeId(memory.source_id);
    if (nodes.has(id)) continue;
    nodes.set(id, {
      id,
      kind: "memory",
      name: memory.title,
      entityType: MEMORY_NODE_TYPE,
      confirmed: memory.confirmed,
      degree: 0,
      space: memory.space || null,
      createdAt: memory.last_modified,
      updatedAt: memory.last_modified,
    });
    for (const entityId of [...new Set(linked)].sort()) {
      const key = `mem:${memory.source_id}:${entityId}`;
      edges.set(key, {
        id: key,
        source: id,
        target: entityId,
        type: MEMORY_EDGE_TYPE,
        confidence: null,
        createdAt: memory.last_modified,
      });
    }
  }

  for (const edge of edges.values()) {
    const source = nodes.get(edge.source);
    if (source) source.degree += 1;
    if (edge.target !== edge.source) {
      const target = nodes.get(edge.target);
      if (target) target.degree += 1;
    }
  }

  return {
    nodes: Array.from(nodes.values()),
    edges: Array.from(edges.values()),
    // One read covers every entity in scope, so coverage is total — the
    // honesty chip has nothing left to confess.
    coverage: { relationsFetchedFor: graph.entities.length, totalEntities: graph.entities.length },
  };
}

/** 1-hop ego graph: the center entity plus its direct neighbors. */
export function buildEgoModel(detail: EntityDetail): GraphModel {
  return buildGraphModel([detail.entity], [detail]);
}
