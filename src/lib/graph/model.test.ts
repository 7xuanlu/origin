// SPDX-License-Identifier: AGPL-3.0-only
import { describe, it, expect } from "vitest";
import type {
  Entity,
  EntityDetail,
  GraphMemoryNode,
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
  };
}

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
