// SPDX-License-Identifier: AGPL-3.0-only
import { describe, it, expect } from "vitest";
import type {
  Entity,
  EntityDetail,
  GraphMemoryNode,
  GraphPageLink,
  GraphPageNode,
  GraphRef,
  GraphRelation,
  KnowledgeGraph,
  RelationWithEntity,
} from "../tauri";
import {
  buildEgoModel,
  buildGraphModel,
  buildKnowledgeGraphModel,
  filterKnowledgeGraph,
  memorySourceId,
  pageIdOf,
  pageNodeId,
  componentSizeByNode,
  drawableModel,
  DEFAULT_LAYERS,
  MIN_COMPONENT_SIZE,
  SHARED_SOURCE_MAX_FANOUT,
  SHARED_SOURCE_TOP_K,
} from "./model";

function makeEntity(o: Partial<Entity> = {}): Entity {
  return {
    id: o.id ?? "E",
    name: o.name ?? "Center",
    entity_type: o.entity_type ?? "concept",
    domain: o.domain ?? null,
    space: o.space ?? null,
    source_agent: o.source_agent ?? null,
    confidence: o.confidence ?? null,
    confirmed: o.confirmed ?? true,
    created_at: o.created_at ?? 100,
    updated_at: o.updated_at ?? 200,
  };
}

function makeRel(
  o: Partial<RelationWithEntity> & { confidence?: number | null } = {},
): RelationWithEntity {
  return {
    id: o.id ?? "r1",
    relation_type: o.relation_type ?? "knows",
    direction: o.direction ?? "outgoing",
    entity_id: o.entity_id ?? "B",
    entity_name: o.entity_name ?? "Bob",
    entity_type: o.entity_type ?? "person",
    source_agent: o.source_agent ?? null,
    created_at: o.created_at ?? 150,
    ...(o.confidence !== undefined ? { confidence: o.confidence } : {}),
  } as RelationWithEntity;
}

function makeDetail(entity: Entity, relations: RelationWithEntity[]): EntityDetail {
  return { entity, observations: [], relations };
}

describe("buildGraphModel / buildEgoModel", () => {
  it("normalizes an outgoing relation to center→neighbor", () => {
    const model = buildEgoModel(
      makeDetail(makeEntity({ id: "E" }), [
        makeRel({ id: "r1", direction: "outgoing", entity_id: "B" }),
      ]),
    );
    const edge = model.edges.find((e) => e.id === "r1")!;
    expect(edge.source).toBe("E");
    expect(edge.target).toBe("B");
  });

  it("normalizes an incoming relation to neighbor→center", () => {
    const model = buildEgoModel(
      makeDetail(makeEntity({ id: "E" }), [
        makeRel({ id: "r2", direction: "incoming", entity_id: "A" }),
      ]),
    );
    const edge = model.edges.find((e) => e.id === "r2")!;
    expect(edge.source).toBe("A");
    expect(edge.target).toBe("E");
  });

  it("dedupes the same relation surfaced on both endpoints' details", () => {
    const a = makeEntity({ id: "A", name: "A" });
    const b = makeEntity({ id: "B", name: "B" });
    const detailA = makeDetail(a, [
      makeRel({ id: "r1", direction: "outgoing", entity_id: "B", entity_name: "B" }),
    ]);
    const detailB = makeDetail(b, [
      makeRel({ id: "r1", direction: "incoming", entity_id: "A", entity_name: "A" }),
    ]);
    const model = buildGraphModel([a, b], [detailA, detailB]);
    expect(model.edges).toHaveLength(1);
    expect(model.edges[0].source).toBe("A");
    expect(model.edges[0].target).toBe("B");
  });

  it("falls back to the endpoints+verb key when a relation has no id, and exports it as the edge id", () => {
    const e = makeEntity({ id: "E" });
    const detail = makeDetail(e, [
      makeRel({ id: "", direction: "outgoing", entity_id: "B", relation_type: "uses" }),
      makeRel({ id: "", direction: "outgoing", entity_id: "B", relation_type: "uses" }),
    ]);
    const model = buildGraphModel([e], [detail]);
    expect(model.edges).toHaveLength(1);
    expect(model.edges[0].id).toBe("E:uses:B");
  });

  it("counts a self-loop once toward degree, not twice", () => {
    const model = buildEgoModel(
      makeDetail(makeEntity({ id: "E" }), [
        makeRel({ id: "r1", direction: "outgoing", entity_id: "E", entity_name: "E" }),
      ]),
    );
    expect(model.nodes.find((n) => n.id === "E")!.degree).toBe(1);
  });

  it("dedupes a mirrored relation whose id was stripped on the other endpoint", () => {
    const a = makeEntity({ id: "A", name: "A" });
    const b = makeEntity({ id: "B", name: "B" });
    const detailA = makeDetail(a, [
      makeRel({ id: "r1", direction: "outgoing", entity_id: "B", entity_name: "B", relation_type: "uses" }),
    ]);
    const detailB = makeDetail(b, [
      makeRel({ id: "", direction: "incoming", entity_id: "A", entity_name: "A", relation_type: "uses" }),
    ]);
    const model = buildGraphModel([a, b], [detailA, detailB]);
    expect(model.edges).toHaveLength(1);
    expect(model.nodes.find((n) => n.id === "A")!.degree).toBe(1);
    expect(model.nodes.find((n) => n.id === "B")!.degree).toBe(1);
  });

  it("counts unique detail entities for coverage, not raw detail-array length", () => {
    const e = makeEntity({ id: "E" });
    const model = buildGraphModel([e], [makeDetail(e, []), makeDetail(e, [])]);
    expect(model.coverage.relationsFetchedFor).toBe(1);
  });

  it("computes degree over the deduped edge set", () => {
    const model = buildEgoModel(
      makeDetail(makeEntity({ id: "E" }), [
        makeRel({ id: "r1", direction: "outgoing", entity_id: "B" }),
        makeRel({ id: "r2", direction: "outgoing", entity_id: "C" }),
        makeRel({ id: "r3", direction: "incoming", entity_id: "D" }),
      ]),
    );
    expect(model.nodes.find((n) => n.id === "E")!.degree).toBe(3);
    expect(model.nodes.find((n) => n.id === "B")!.degree).toBe(1);
    expect(model.nodes.find((n) => n.id === "D")!.degree).toBe(1);
  });

  it("passes coverage counts through", () => {
    const entities = [
      makeEntity({ id: "A" }),
      makeEntity({ id: "B" }),
      makeEntity({ id: "C" }),
    ];
    const details = [makeDetail(entities[0], []), makeDetail(entities[1], [])];
    const model = buildGraphModel(entities, details);
    expect(model.coverage.relationsFetchedFor).toBe(2);
    expect(model.coverage.totalEntities).toBe(3);
  });

  it("synthesizes a node for a neighbor missing from the entities list, with confirmed unknown (null) not fabricated false", () => {
    const e = makeEntity({ id: "E", confirmed: true });
    const detail = makeDetail(e, [
      makeRel({
        id: "r1",
        direction: "outgoing",
        entity_id: "NEW",
        entity_name: "Newcomer",
        entity_type: "organization",
        created_at: 777,
      }),
    ]);
    const model = buildGraphModel([e], [detail]);
    const synth = model.nodes.find((n) => n.id === "NEW")!;
    expect(synth).toBeDefined();
    expect(synth.kind).toBe("entity");
    expect(synth.name).toBe("Newcomer");
    expect(synth.entityType).toBe("organization");
    expect(synth.confirmed).toBeNull();
    expect(synth.createdAt).toBe(777);

    // A listed entity (fetched from the daemon, not synthesized) keeps its
    // real boolean — only relation-only neighbors get the unknown null.
    const home = model.nodes.find((n) => n.id === "E")!;
    expect(home.confirmed).toBe(true);
  });

  it("reads confidence when the relation carries it, else null", () => {
    const model = buildEgoModel(
      makeDetail(makeEntity({ id: "E" }), [
        makeRel({ id: "r1", direction: "outgoing", entity_id: "B", confidence: 0.42 }),
        makeRel({ id: "r2", direction: "outgoing", entity_id: "C" }),
      ]),
    );
    expect(model.edges.find((e) => e.id === "r1")!.confidence).toBe(0.42);
    expect(model.edges.find((e) => e.id === "r2")!.confidence).toBeNull();
  });

  it("reads space from the entity, falling back to domain, else null", () => {
    const withSpace = makeEntity({ id: "E", space: "Work", domain: "legacy" });
    const withSpaceModel = buildGraphModel([withSpace], [makeDetail(withSpace, [])]);
    expect(withSpaceModel.nodes.find((n) => n.id === "E")!.space).toBe("Work");

    const domainOnly = makeEntity({ id: "F", space: null, domain: "legacy" });
    const domainOnlyModel = buildGraphModel([domainOnly], [makeDetail(domainOnly, [])]);
    expect(domainOnlyModel.nodes.find((n) => n.id === "F")!.space).toBe("legacy");

    const neither = makeEntity({ id: "G", space: null, domain: null });
    const neitherModel = buildGraphModel([neither], [makeDetail(neither, [])]);
    expect(neitherModel.nodes.find((n) => n.id === "G")!.space).toBeNull();

    // An empty space string is not a space. It has to fall through to the
    // domain, or the node reads as unscoped and loses its grouping.
    const emptySpace = makeEntity({ id: "H", space: "", domain: "legacy" });
    const emptySpaceModel = buildGraphModel([emptySpace], [makeDetail(emptySpace, [])]);
    expect(emptySpaceModel.nodes.find((n) => n.id === "H")!.space).toBe("legacy");

    const bothEmpty = makeEntity({ id: "I", space: "", domain: "" });
    const bothEmptyModel = buildGraphModel([bothEmpty], [makeDetail(bothEmpty, [])]);
    expect(bothEmptyModel.nodes.find((n) => n.id === "I")!.space).toBeNull();
  });

  it("leaves a relation-synthesized neighbor's space unknown (null), never guessed from the home entity", () => {
    const e = makeEntity({ id: "E", space: "Work" });
    const detail = makeDetail(e, [
      makeRel({ id: "r1", direction: "outgoing", entity_id: "NEW", entity_name: "Newcomer" }),
    ]);
    const model = buildGraphModel([e], [detail]);
    expect(model.nodes.find((n) => n.id === "NEW")!.space).toBeNull();
  });

  it("buildEgoModel keeps full center entity data and 1/1 coverage", () => {
    const e = makeEntity({ id: "E", name: "Origin", confirmed: true, entity_type: "project" });
    const model = buildEgoModel(
      makeDetail(e, [makeRel({ id: "r1", direction: "outgoing", entity_id: "B" })]),
    );
    const center = model.nodes.find((n) => n.id === "E")!;
    expect(center.confirmed).toBe(true);
    expect(center.entityType).toBe("project");
    expect(center.kind).toBe("entity");
    expect(model.coverage).toEqual({ relationsFetchedFor: 1, totalEntities: 1 });
  });
});

function makeGraphRelation(o: Partial<GraphRelation> = {}): GraphRelation {
  return {
    id: o.id ?? "r1",
    from_entity: o.from_entity ?? "A",
    to_entity: o.to_entity ?? "B",
    relation_type: o.relation_type ?? "knows",
    source_agent: o.source_agent ?? null,
    created_at: o.created_at ?? 150,
  };
}

function makeGraphMemory(o: Partial<GraphMemoryNode> = {}): GraphMemoryNode {
  return {
    source_id: o.source_id ?? "m1",
    title: o.title ?? "A memory",
    memory_type: o.memory_type ?? "fact",
    space: o.space ?? null,
    confirmed: o.confirmed ?? true,
    last_modified: o.last_modified ?? 300,
  };
}

function makeGraph(o: Partial<KnowledgeGraph> = {}): KnowledgeGraph {
  return {
    entities: o.entities ?? [],
    relations: o.relations ?? [],
    memories: o.memories ?? [],
    memory_links: o.memory_links ?? [],
    pages: o.pages ?? [],
    page_links: o.page_links ?? [],
  };
}

function makePage(o: Partial<GraphPageNode> = {}): GraphPageNode {
  return {
    id: o.id ?? "p1",
    title: o.title ?? "A page",
    space: o.space ?? null,
    creation_kind: o.creation_kind ?? "distilled",
    entity_id: o.entity_id ?? null,
    last_modified: o.last_modified ?? "2026-08-15T00:00:00Z",
  };
}

function pageLink(
  from: [GraphRef["kind"], string],
  to: [GraphRef["kind"], string],
  link_type: GraphPageLink["link_type"],
): GraphPageLink {
  return { from: { kind: from[0], id: from[1] }, to: { kind: to[0], id: to[1] }, link_type };
}

const ALL_LAYERS = { entity: true, page: true, memory: true };

describe("buildKnowledgeGraphModel", () => {
  it("folds entities and relations without inventing endpoints", () => {
    const model = buildKnowledgeGraphModel(
      makeGraph({
        entities: [makeEntity({ id: "A" }), makeEntity({ id: "B" })],
        relations: [
          makeGraphRelation({ id: "r1", from_entity: "A", to_entity: "B" }),
          // Dangling: the daemon never ships this, and a filtered read means
          // the endpoint is out of scope — synthesizing it would fabricate.
          makeGraphRelation({ id: "r2", from_entity: "A", to_entity: "GONE" }),
        ],
      }),
    );
    expect(model.nodes.map((n) => n.id).sort()).toEqual(["A", "B"]);
    expect(model.edges.map((e) => e.id)).toEqual(["r1"]);
    expect(model.nodes.find((n) => n.id === "A")!.degree).toBe(1);
  });

  it("draws a memory as its own node joined to every entity it links to", () => {
    const model = buildKnowledgeGraphModel(
      makeGraph({
        entities: [makeEntity({ id: "A" }), makeEntity({ id: "B" })],
        memories: [makeGraphMemory({ source_id: "m1", title: "Shared" })],
        memory_links: [
          { memory_id: "m1", entity_id: "B" },
          { memory_id: "m1", entity_id: "A" },
        ],
      }),
    );
    const memory = model.nodes.find((n) => n.id === "mem:m1")!;
    expect(memory.kind).toBe("memory");
    expect(memory.entityType).toBe("memory");
    expect(memory.name).toBe("Shared");
    // Two links, so the memory has degree 2 and neither entity is an isolate.
    expect(memory.degree).toBe(2);
    expect(model.nodes.find((n) => n.id === "A")!.degree).toBe(1);
    expect(model.nodes.find((n) => n.id === "B")!.degree).toBe(1);
    expect(model.edges.every((e) => e.type === "mentions")).toBe(true);
  });

  it("drops a memory whose links all point outside the entity set", () => {
    const model = buildKnowledgeGraphModel(
      makeGraph({
        entities: [makeEntity({ id: "A" })],
        memories: [makeGraphMemory({ source_id: "m1" })],
        memory_links: [{ memory_id: "m1", entity_id: "GONE" }],
      }),
    );
    expect(model.nodes.map((n) => n.id)).toEqual(["A"]);
    expect(model.edges).toEqual([]);
  });

  it("prefixes memory ids so a memory can never collide with an entity", () => {
    const model = buildKnowledgeGraphModel(
      makeGraph({
        entities: [makeEntity({ id: "m1", name: "Entity called m1" })],
        memories: [makeGraphMemory({ source_id: "m1", title: "Memory called m1" })],
        memory_links: [{ memory_id: "m1", entity_id: "m1" }],
      }),
    );
    expect(model.nodes.find((n) => n.id === "m1")!.name).toBe("Entity called m1");
    expect(model.nodes.find((n) => n.id === "mem:m1")!.name).toBe("Memory called m1");
    expect(memorySourceId("mem:m1")).toBe("m1");
    expect(memorySourceId("m1")).toBeNull();
  });

  it("reports total coverage — one read, nothing left unfetched", () => {
    const model = buildKnowledgeGraphModel(
      makeGraph({ entities: [makeEntity({ id: "A" }), makeEntity({ id: "B" })] }),
    );
    expect(model.coverage).toEqual({ relationsFetchedFor: 2, totalEntities: 2 });
  });
});

describe("filterKnowledgeGraph", () => {
  const work = makeEntity({ id: "A", space: "Work" });
  const personal = makeEntity({ id: "B", space: "Personal" });
  const graph = makeGraph({
    entities: [work, personal],
    relations: [makeGraphRelation({ id: "r1", from_entity: "A", to_entity: "B" })],
    memories: [
      makeGraphMemory({ source_id: "m1", space: "Work" }),
      makeGraphMemory({ source_id: "m2", space: "Personal" }),
    ],
    memory_links: [
      { memory_id: "m1", entity_id: "A" },
      { memory_id: "m2", entity_id: "A" },
    ],
  });

  it("returns the graph untouched when no space is selected", () => {
    expect(filterKnowledgeGraph(graph, null)).toBe(graph);
  });

  it("drops a cross-space relation rather than keeping a half-visible edge", () => {
    const scoped = filterKnowledgeGraph(graph, "Work");
    expect(scoped.entities.map((e) => e.id)).toEqual(["A"]);
    expect(scoped.relations).toEqual([]);
  });

  it("scopes memories by their own space, not by the entity they link to", () => {
    const scoped = filterKnowledgeGraph(graph, "Work");
    // m2 is a Personal memory hanging off a Work entity: out of scope.
    expect(scoped.memories.map((m) => m.source_id)).toEqual(["m1"]);
    expect(scoped.memory_links).toEqual([{ memory_id: "m1", entity_id: "A" }]);
  });

  it("falls back to domain when an entity carries no space", () => {
    const legacy = makeEntity({ id: "C", space: "", domain: "Work" });
    const scoped = filterKnowledgeGraph(makeGraph({ entities: [legacy] }), "Work");
    expect(scoped.entities.map((e) => e.id)).toEqual(["C"]);
  });
});

describe("buildKnowledgeGraphModel — wiki pages", () => {
  const seed = () =>
    makeGraph({
      entities: [makeEntity({ id: "E1" }), makeEntity({ id: "E2" })],
      memories: [makeGraphMemory({ source_id: "m1" }), makeGraphMemory({ source_id: "m2" })],
      pages: [
        makePage({ id: "pa", title: "Page A", entity_id: "E1" }),
        makePage({ id: "pb", title: "Page B" }),
      ],
      page_links: [
        pageLink(["page", "pa"], ["page", "pb"], "wikilink"),
        pageLink(["page", "pa"], ["entity", "E1"], "about"),
        pageLink(["page", "pa"], ["entity", "E2"], "wikilink"),
        pageLink(["page", "pa"], ["memory", "m1"], "cites"),
        pageLink(["page", "pb"], ["memory", "m1"], "cites"),
        pageLink(["page", "pb"], ["memory", "m2"], "cites"),
      ],
    });

  it("draws pages as prefixed nodes joined by their typed links", () => {
    const model = buildKnowledgeGraphModel(seed(), { layers: ALL_LAYERS });
    const page = model.nodes.find((n) => n.id === pageNodeId("pa"))!;
    expect(page.kind).toBe("page");
    expect(page.name).toBe("Page A");
    expect(page.entityType).toBe("page");
    expect(pageIdOf(page.id)).toBe("pa");
    // A page id can never collide with an entity id.
    expect(model.nodes.some((n) => n.id === "pa")).toBe(false);

    const types = model.edges
      .filter((e) => e.source.startsWith("page:") || e.target.startsWith("page:"))
      .map((e) => `${e.source}->${e.target}:${e.type}`)
      .sort();
    expect(types).toEqual([
      "page:pa->E1:about",
      "page:pa->E2:wikilink",
      "page:pa->mem:m1:cites",
      "page:pa->page:pb:wikilink",
      "page:pb->mem:m1:cites",
      "page:pb->mem:m2:cites",
    ]);
    // Degree counts every one of them.
    expect(page.degree).toBe(4);
  });

  it("pulls in a memory that only a page cites", () => {
    // No memory_links at all — round 1 would have dropped both memories.
    const model = buildKnowledgeGraphModel(seed(), { layers: ALL_LAYERS });
    expect(model.nodes.filter((n) => n.kind === "memory").map((n) => n.id).sort()).toEqual([
      "mem:m1",
      "mem:m2",
    ]);
  });

  it("drops a hidden layer's nodes and edges before computing degree", () => {
    const model = buildKnowledgeGraphModel(seed(), {
      layers: { entity: false, page: true, memory: false },
    });
    expect(model.nodes.map((n) => n.id).sort()).toEqual(["page:pa", "page:pb"]);
    // Only the wikilink and the synthesized shared-source edge survive: the
    // about/entity-wikilink edges lost their entity, the cites lost their
    // memory.
    const page = model.nodes.find((n) => n.id === "page:pa")!;
    expect(page.degree).toBe(2);
  });

  it("synthesizes one weighted shared-source edge per page pair while memories are off", () => {
    const model = buildKnowledgeGraphModel(seed(), {
      layers: { entity: true, page: true, memory: false },
    });
    const shared = model.edges.filter((e) => e.type === "shared_source");
    expect(shared).toHaveLength(1);
    expect(shared[0].source).toBe("page:pa");
    expect(shared[0].target).toBe("page:pb");
    // Both pages cite m1; only pb cites m2, so the pair shares exactly one.
    expect(shared[0].weight).toBe(1);
    expect(model.nodes.some((n) => n.kind === "memory")).toBe(false);
  });

  it("synthesizes nothing while memories are on — the memory node shows the path", () => {
    const model = buildKnowledgeGraphModel(seed(), { layers: ALL_LAYERS });
    expect(model.edges.some((e) => e.type === "shared_source")).toBe(false);
  });

  it("weights a shared-source edge by how many memories the pair shares", () => {
    const graph = seed();
    graph.page_links.push(pageLink(["page", "pa"], ["memory", "m2"], "cites"));
    const model = buildKnowledgeGraphModel(graph, {
      layers: { entity: false, page: true, memory: false },
    });
    expect(model.edges.find((e) => e.type === "shared_source")!.weight).toBe(2);
  });

  it("defaults to pages and entities on, memories off", () => {
    expect(DEFAULT_LAYERS).toEqual({ entity: true, page: true, memory: false });
    const model = buildKnowledgeGraphModel(seed(), { layers: DEFAULT_LAYERS });
    expect(model.nodes.some((n) => n.kind === "memory")).toBe(false);
    expect(model.nodes.some((n) => n.kind === "page")).toBe(true);
    expect(model.nodes.some((n) => n.kind === "entity")).toBe(true);
  });
});

describe("filterKnowledgeGraph — pages", () => {
  it("keeps only the space's pages and the links whose endpoints both survive", () => {
    const filtered = filterKnowledgeGraph(
      makeGraph({
        entities: [makeEntity({ id: "E1", space: "work" })],
        pages: [
          makePage({ id: "pa", space: "work" }),
          makePage({ id: "pb", space: "personal" }),
        ],
        page_links: [
          pageLink(["page", "pa"], ["page", "pb"], "wikilink"),
          pageLink(["page", "pa"], ["entity", "E1"], "about"),
        ],
      }),
      "work",
    );
    expect(filtered.pages.map((p) => p.id)).toEqual(["pa"]);
    expect(filtered.page_links).toHaveLength(1);
    expect(filtered.page_links[0].link_type).toBe("about");
  });
});

describe("buildKnowledgeGraphModel — shared-source thinning", () => {
  const PAGES_OFF_MEMORIES = { entity: false, page: true, memory: false };

  /** `citations` maps a memory id to the page ids that cite it. */
  function citing(citations: Record<string, string[]>): KnowledgeGraph {
    const pageIds = [...new Set(Object.values(citations).flat())].sort();
    return makeGraph({
      pages: pageIds.map((id) => makePage({ id, title: id })),
      memories: Object.keys(citations).map((id) => makeGraphMemory({ source_id: id })),
      page_links: Object.entries(citations).flatMap(([memoryId, pages]) =>
        pages.map((pageId) => pageLink(["page", pageId], ["memory", memoryId], "cites")),
      ),
    });
  }

  const sharedPairs = (graph: KnowledgeGraph): string[] =>
    buildKnowledgeGraphModel(graph, { layers: PAGES_OFF_MEMORIES })
      .edges.filter((e) => e.type === "shared_source")
      .map((e) => `${e.source}|${e.target}`)
      .sort();

  it("pairs the pages of a memory cited by exactly the fan-out limit", () => {
    const pages = Array.from({ length: SHARED_SOURCE_MAX_FANOUT }, (_, i) => `p${i}`);
    expect(sharedPairs(citing({ m: pages })).length).toBeGreaterThan(0);
  });

  it("synthesizes nothing for a hub source cited by more pages than the limit", () => {
    // Session logs and bulk imports are cited by everything; pairing them
    // turns the page layer into one clique (real data: 29 pages on one memory).
    const pages = Array.from({ length: SHARED_SOURCE_MAX_FANOUT + 1 }, (_, i) => `p${i}`);
    expect(sharedPairs(citing({ m: pages }))).toEqual([]);
  });

  it("keeps a page's single partner even when that partner is already full", () => {
    // "hub" already has SHARED_SOURCE_TOP_K partners it shares two memories
    // with, so "solo" (one shared memory) is off the bottom of hub's list —
    // but solo's own list has room, and either endpoint is enough.
    const citations: Record<string, string[]> = { weak: ["hub", "solo"] };
    for (let i = 0; i < SHARED_SOURCE_TOP_K; i++) {
      citations[`strong${i}a`] = ["hub", `p${i}`];
      citations[`strong${i}b`] = ["hub", `p${i}`];
    }
    expect(sharedPairs(citing(citations))).toContain("page:hub|page:solo");
  });

  it("drops a pair only when NEITHER endpoint ranks it in its top-K", () => {
    // Six pages, every pair sharing exactly one memory: each page keeps its
    // four lowest-id partners, which leaves E and F off each other's list and
    // off nobody else's — exactly one of the fifteen pairs is cut.
    const ids = ["a", "b", "c", "d", "e", "f"];
    const citations: Record<string, string[]> = {};
    for (let i = 0; i < ids.length; i++) {
      for (let j = i + 1; j < ids.length; j++) citations[`m${ids[i]}${ids[j]}`] = [ids[i], ids[j]];
    }
    const pairs = sharedPairs(citing(citations));
    expect(pairs).toHaveLength(14);
    expect(pairs).not.toContain("page:e|page:f");
  });

  it("ranks by weight first, so a heavy overlap outranks a lexically earlier light one", () => {
    // "z" shares two memories with "hub"; a0..a3 share one each. Only z and
    // the three lexically-earliest light partners make hub's top-4, and a3
    // survives anyway because hub is a3's only partner.
    const citations: Record<string, string[]> = { z1: ["hub", "z"], z2: ["hub", "z"] };
    for (let i = 0; i < 4; i++) citations[`light${i}`] = ["hub", `a${i}`];
    expect(sharedPairs(citing(citations))).toContain("page:hub|page:z");
  });
});

describe("componentSizeByNode / drawableModel", () => {
  const model = (nodeIds: string[], edges: [string, string][]) => ({
    nodes: nodeIds.map((id) => ({
      id,
      kind: "entity" as const,
      name: id,
      entityType: "concept",
      confirmed: null,
      degree: edges.filter(([s, t]) => s === id || t === id).length,
      space: null,
      createdAt: 0,
      updatedAt: 0,
    })),
    edges: edges.map(([source, target], i) => ({
      id: `e${i}`,
      source,
      target,
      type: "knows",
      confidence: null,
      createdAt: 0,
    })),
    coverage: { relationsFetchedFor: nodeIds.length, totalEntities: nodeIds.length },
  });

  it("reports every member of a component with the same size", () => {
    const sizes = componentSizeByNode(
      model(["a", "b", "c", "d", "e"], [["a", "b"], ["b", "c"], ["d", "e"]]),
    );
    expect([...sizes.values()].sort()).toEqual([2, 2, 3, 3, 3]);
    expect(sizes.get("a")).toBe(3);
    expect(sizes.get("d")).toBe(2);
  });

  it("reports 1 for a node with no edges at all", () => {
    expect(componentSizeByNode(model(["lonely"], [])).get("lonely")).toBe(1);
  });

  it("follows edges regardless of direction", () => {
    const sizes = componentSizeByNode(model(["a", "b", "c"], [["b", "a"], ["c", "b"]]));
    expect(sizes.get("c")).toBe(3);
  });

  it("drops components below MIN_COMPONENT_SIZE, and their edges with them", () => {
    const drawable = drawableModel(
      model(["a", "b", "c", "d", "e", "f"], [["a", "b"], ["b", "c"], ["d", "e"]]),
    );
    expect(drawable.nodes.map((n) => n.id)).toEqual(["a", "b", "c"]);
    // "d"-"e" would otherwise dangle off nodes the graph no longer has.
    expect(drawable.edges.map((e) => e.id)).toEqual(["e0", "e1"]);
    expect(MIN_COMPONENT_SIZE).toBe(3);
  });

  it("returns the model untouched when every component is big enough", () => {
    const full = model(["a", "b", "c"], [["a", "b"], ["b", "c"]]);
    expect(drawableModel(full)).toBe(full);
  });
});
