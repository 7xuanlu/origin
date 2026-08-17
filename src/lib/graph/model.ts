// SPDX-License-Identifier: AGPL-3.0-only
import type {
  Entity,
  EntityDetail,
  GraphRef,
  KnowledgeGraph,
  RelationWithEntity,
} from "../tauri";

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
  /** How many underlying facts this one drawn edge stands for. Only the
   *  synthesized shared-source edges set it (the number of memories two pages
   *  both cite); every other edge is one fact and leaves it undefined. */
  weight?: number;
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
        // An idless mirror folded EARLIER sits under the composite key; the
        // id-bearing copy wins regardless of detail order.
        edges.delete(`k:${composite}`);
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
 * The `entityType` every wiki-page node carries — same reason as
 * MEMORY_NODE_TYPE above: one swatch, keyed without a `kind` lookup.
 */
export const PAGE_NODE_TYPE = "page";

/** Edge verbs for the typed page links, plus the synthesized one. Not
 *  daemon relation types. */
export const WIKILINK_EDGE_TYPE = "wikilink";
export const CITES_EDGE_TYPE = "cites";
/** Two pages that cite the same memory, drawn directly page→page because the
 *  memory node itself is hidden. */
export const SHARED_SOURCE_EDGE_TYPE = "shared_source";

/** A memory cited by more than this many visible pages is a hub source — a
 *  session log or a bulk import, not a shared topic. Pairing every page that
 *  touches it makes a clique (real data: one memory cited by 29 pages, 406
 *  pairs from that memory alone), so a hub source synthesizes NO pairwise
 *  edges at all. The memory is still there to see when the memory layer is on. */
export const SHARED_SOURCE_MAX_FANOUT = 6;

/** How many shared-source edges one page may keep, strongest first. Without
 *  the cut 33 real pages carried 20+ overlap neighbours (max 45) and the page
 *  layer collapsed into a single blob. */
export const SHARED_SOURCE_TOP_K = 4;

/** Graph-node id for a wiki page. Prefixed for the same reason memories are:
 *  no collision with an entity id, and click routing can tell them apart. */
export function pageNodeId(pageId: string): string {
  return `page:${pageId}`;
}

/** The page id behind a page node id, or null for anything else. */
export function pageIdOf(nodeId: string): string | null {
  return nodeId.startsWith("page:") ? nodeId.slice(5) : null;
}

/** Which node kinds the map is currently drawing. Off means the nodes and
 *  every edge touching them never enter the model at all. */
export interface GraphLayers {
  entity: boolean;
  page: boolean;
  memory: boolean;
}

/** Wiki pages and entities on, memories off — the round-2 default view. */
export const DEFAULT_LAYERS: GraphLayers = { entity: true, page: true, memory: false };

/**
 * Narrow a bulk graph read to one space: entities by `entitySpace`, memories
 * and pages by their own space, and relations/links to whatever survives. A
 * relation whose other endpoint was filtered out is DROPPED, not synthesized
 * — the bulk read carries every in-scope entity, so a missing endpoint means
 * out of scope rather than not-fetched-yet.
 */
export function filterKnowledgeGraph(graph: KnowledgeGraph, space: string | null): KnowledgeGraph {
  if (!space) return graph;
  const entities = graph.entities.filter((entity) => entitySpace(entity) === space);
  const kept = new Set(entities.map((entity) => entity.id));
  const memories = graph.memories.filter((memory) => memory.space === space);
  const keptMemories = new Set(memories.map((memory) => memory.source_id));
  const pages = graph.pages.filter((page) => page.space === space);
  const keptPages = new Set(pages.map((page) => page.id));
  const endpointKept = (ref: GraphRef): boolean =>
    ref.kind === "page"
      ? keptPages.has(ref.id)
      : ref.kind === "entity"
        ? kept.has(ref.id)
        : keptMemories.has(ref.id);
  return {
    entities,
    relations: graph.relations.filter(
      (relation) => kept.has(relation.from_entity) && kept.has(relation.to_entity),
    ),
    memories,
    memory_links: graph.memory_links.filter(
      (link) => kept.has(link.entity_id) && keptMemories.has(link.memory_id),
    ),
    pages,
    page_links: graph.page_links.filter(
      (link) => endpointKept(link.from) && endpointKept(link.to),
    ),
  };
}

/**
 * Fold one bulk graph read into a GraphModel: every entity as a node, every
 * relation as an edge, every wiki page as its own node joined by its typed
 * links (`wikilink` / `about` / `cites`), and every memory that links to a
 * present entity or is cited by a present page as its own node. Degree counts
 * all of them, so an entity with only memories attached is not an isolate.
 *
 * `layers` decides which KINDS exist at all. A hidden layer's nodes and every
 * edge touching them are dropped BEFORE degree is computed, so "unconnected"
 * means unconnected among the layers actually on screen. With the memory layer
 * off, two pages citing the same memory are joined directly by one synthesized
 * `shared_source` edge instead — the llm_wiki look, where shared sources are
 * the page graph's second edge type. With memories on, the memory node itself
 * shows that path and nothing is synthesized.
 *
 * No other filtering happens here — that is a per-view decision
 * (filterKnowledgeGraph above for the space filter, degree-0 hiding in
 * AtlasView).
 */
export function buildKnowledgeGraphModel(
  graph: KnowledgeGraph,
  options?: { layers?: GraphLayers },
): GraphModel {
  const layers = options?.layers ?? { entity: true, page: true, memory: true };
  const nodes = new Map<string, GraphNode>();
  if (layers.entity) {
    for (const entity of graph.entities) nodes.set(entity.id, nodeFromEntity(entity));
  }

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

  // Pages join the map before memories so a page's citations can pull a
  // memory in that no entity links to. `last_modified` is RFC 3339 on a page
  // (Unix seconds on a memory); parse it to the same Unix-seconds unit every
  // other node carries, and fall back to 0 rather than NaN on a bad string.
  const pageTimestamp = (value: string): number => {
    const parsed = Date.parse(value);
    return Number.isFinite(parsed) ? Math.round(parsed / 1000) : 0;
  };
  const pageNodeIds = new Map<string, string>();
  if (layers.page) {
    for (const page of graph.pages) {
      const id = pageNodeId(page.id);
      if (nodes.has(id)) continue;
      pageNodeIds.set(page.id, id);
      nodes.set(id, {
        id,
        kind: "page",
        name: page.title,
        entityType: PAGE_NODE_TYPE,
        // A page carries no confirmed axis of its own on this wire shape.
        confirmed: null,
        degree: 0,
        space: page.space || null,
        createdAt: pageTimestamp(page.last_modified),
        updatedAt: pageTimestamp(page.last_modified),
      });
    }
  }

  // Which memories earn a node: linked to a present entity, or cited by a
  // present page. Collected first, because a citation alone is enough.
  const linksByMemory = new Map<string, string[]>();
  if (layers.memory && layers.entity) {
    for (const link of graph.memory_links) {
      if (!nodes.has(link.entity_id)) continue;
      const list = linksByMemory.get(link.memory_id);
      if (list) list.push(link.entity_id);
      else linksByMemory.set(link.memory_id, [link.entity_id]);
    }
  }
  // page id -> the memory ids it cites, for the memory nodes below and for
  // the shared-source synthesis when the memory layer is off.
  const citedByPage = new Map<string, string[]>();
  for (const link of graph.page_links) {
    if (link.link_type !== CITES_EDGE_TYPE) continue;
    if (!pageNodeIds.has(link.from.id)) continue;
    const list = citedByPage.get(link.from.id);
    if (list) list.push(link.to.id);
    else citedByPage.set(link.from.id, [link.to.id]);
  }
  const citedMemories = new Set<string>();
  for (const cited of citedByPage.values()) for (const id of cited) citedMemories.add(id);

  if (layers.memory) {
    for (const memory of graph.memories) {
      const linked = linksByMemory.get(memory.source_id);
      if (!linked && !citedMemories.has(memory.source_id)) continue;
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
      for (const entityId of [...new Set(linked ?? [])].sort()) {
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
  }

  // The typed page links. An endpoint whose layer is off simply isn't in
  // `nodes`, which is what drops the edge with it.
  const graphNodeId = (ref: GraphRef): string =>
    ref.kind === "page" ? pageNodeId(ref.id) : ref.kind === "memory" ? memoryNodeId(ref.id) : ref.id;
  for (const link of graph.page_links) {
    const source = graphNodeId(link.from);
    const target = graphNodeId(link.to);
    if (!nodes.has(source) || !nodes.has(target) || source === target) continue;
    const key = `pl:${link.link_type}:${source}:${target}`;
    if (edges.has(key)) continue;
    edges.set(key, {
      id: key,
      source,
      target,
      type: link.link_type,
      confidence: null,
      createdAt: nodes.get(source)!.createdAt,
    });
  }

  // Shared sources, only while the memory layer is off: one undirected edge
  // per page pair, weighted by how many memories they both cite. Page-only on
  // purpose — doing the same for entities would detonate the hubs.
  if (!layers.memory && layers.page) {
    const pagesByMemory = new Map<string, string[]>();
    for (const [pageId, cited] of citedByPage) {
      for (const memoryId of new Set(cited)) {
        const list = pagesByMemory.get(memoryId);
        if (list) list.push(pageId);
        else pagesByMemory.set(memoryId, [pageId]);
      }
    }
    const shared = new Map<string, number>();
    for (const pageIds of pagesByMemory.values()) {
      const sorted = [...new Set(pageIds)].sort();
      // Hub source: too many pages cite it for the overlap to mean anything.
      if (sorted.length > SHARED_SOURCE_MAX_FANOUT) continue;
      for (let i = 0; i < sorted.length; i++) {
        for (let j = i + 1; j < sorted.length; j++) {
          const key = `${sorted[i]}|${sorted[j]}`;
          shared.set(key, (shared.get(key) ?? 0) + 1);
        }
      }
    }

    // Thin each page down to its strongest overlaps. A pair survives if it is
    // in EITHER endpoint's top-K (union, not intersection) — a page whose only
    // partner ranks low on that partner's own list still keeps its one edge,
    // which is what stops the cut from stranding pages.
    const incident = new Map<string, { partner: string; weight: number }[]>();
    const noteIncident = (page: string, partner: string, weight: number) => {
      const list = incident.get(page);
      if (list) list.push({ partner, weight });
      else incident.set(page, [{ partner, weight }]);
    };
    for (const [pair, weight] of shared) {
      const [a, b] = pair.split("|");
      noteIncident(a, b, weight);
      noteIncident(b, a, weight);
    }
    const keptPairs = new Set<string>();
    for (const [page, partners] of incident) {
      const ranked = [...partners].sort(
        (x, y) => y.weight - x.weight || (x.partner < y.partner ? -1 : x.partner > y.partner ? 1 : 0),
      );
      for (const { partner } of ranked.slice(0, SHARED_SOURCE_TOP_K)) {
        keptPairs.add(page < partner ? `${page}|${partner}` : `${partner}|${page}`);
      }
    }

    for (const [pair, weight] of shared) {
      if (!keptPairs.has(pair)) continue;
      const [a, b] = pair.split("|");
      const source = pageNodeId(a);
      const target = pageNodeId(b);
      const key = `ss:${source}:${target}`;
      if (edges.has(key)) continue;
      edges.set(key, {
        id: key,
        source,
        target,
        type: SHARED_SOURCE_EDGE_TYPE,
        confidence: null,
        createdAt: nodes.get(source)!.createdAt,
        weight,
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

/**
 * Smallest connected component the map draws by default. Real data has 160
 * components with the default layers on: one of 288 nodes, then 18, 14, 8 …
 * and 137 of four nodes or fewer. Those crumbs carry no structure of their
 * own, and a row of them would be longer than the shelf they sit on. They go
 * behind the "small groups hidden" chip, which reveals them on demand.
 */
export const MIN_COMPONENT_SIZE = 5;

/**
 * Node id -> how many nodes are in its connected component, by union-find over
 * the model's own edges (direction ignored; an edge with a missing endpoint is
 * not adjacency). A degree-0 node reports 1, so the isolate rule this
 * generalises falls out of the same number.
 */
export function componentSizeByNode(model: GraphModel): Map<string, number> {
  const parent = new Map<string, string>();
  for (const node of model.nodes) parent.set(node.id, node.id);

  const find = (id: string): string => {
    let root = id;
    while (parent.get(root) !== root) root = parent.get(root)!;
    // Path compression, iterative — a 3,300-node chain would blow a recursive
    // stack, and this runs on every layer flip.
    let walk = id;
    while (parent.get(walk) !== root) {
      const next = parent.get(walk)!;
      parent.set(walk, root);
      walk = next;
    }
    return root;
  };

  for (const edge of model.edges) {
    if (!parent.has(edge.source) || !parent.has(edge.target)) continue;
    const a = find(edge.source);
    const b = find(edge.target);
    if (a !== b) parent.set(a, b);
  }

  const sizes = new Map<string, number>();
  for (const node of model.nodes) {
    const root = find(node.id);
    sizes.set(root, (sizes.get(root) ?? 0) + 1);
  }
  const byNode = new Map<string, number>();
  for (const node of model.nodes) byNode.set(node.id, sizes.get(find(node.id))!);
  return byNode;
}

/**
 * The sub-model the map actually draws: every node whose connected component
 * has at least MIN_COMPONENT_SIZE members, plus the edges between them. Edges
 * are filtered too — a dropped node's edges would otherwise point at nothing.
 *
 * `showSmallGroups` is the reader's own call, made through the toolbar chip:
 * with it on, nothing is dropped and the small components join the shelf.
 */
export function drawableModel(model: GraphModel, showSmallGroups = false): GraphModel {
  if (showSmallGroups) return model;
  const kept = drawableByComponentSize(model);
  if (kept === null || kept.size === model.nodes.length) return model;
  return {
    ...model,
    nodes: model.nodes.filter((node) => kept.has(node.id)),
    edges: model.edges.filter((edge) => kept.has(edge.source) && kept.has(edge.target)),
  };
}

/**
 * The nodes big enough to draw by default, or null when NOTHING clears the
 * bar. A small group is small relative to a core; with no component of
 * MIN_COMPONENT_SIZE anywhere there is no core, and hiding every node would
 * leave a young graph of a few pairs staring at a blank map behind a chip.
 * So in that case nothing is small and nothing is hidden.
 */
function drawableByComponentSize(model: GraphModel): Set<string> | null {
  const sizes = componentSizeByNode(model);
  const kept = new Set<string>();
  for (const node of model.nodes) {
    if ((sizes.get(node.id) ?? 0) >= MIN_COMPONENT_SIZE) kept.add(node.id);
  }
  return kept.size === 0 ? null : kept;
}

/**
 * How many nodes sit in components too small to draw by default — the number
 * on the "small groups hidden" chip. Computed from the FULL model, so it does
 * not change when the reader turns the small groups on; the chip has to keep
 * saying how many there are in order to offer them back.
 */
export function smallGroupNodeCount(model: GraphModel): number {
  const kept = drawableByComponentSize(model);
  if (kept === null) return 0;
  return model.nodes.length - kept.size;
}

/** 1-hop ego graph: the center entity plus its direct neighbors. */
export function buildEgoModel(detail: EntityDetail): GraphModel {
  return buildGraphModel([detail.entity], [detail]);
}
