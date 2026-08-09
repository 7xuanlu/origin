// SPDX-License-Identifier: AGPL-3.0-only
import { describe, it, expect } from "vitest";
import type { Entity, EntityDetail, RelationWithEntity } from "../tauri";
import { buildGraphModel, buildEgoModel } from "./model";

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

  it("reads community_id when the entity carries it, else null", () => {
    const withCid = { ...makeEntity({ id: "E" }), community_id: 7 } as Entity;
    const withCidModel = buildGraphModel([withCid], [makeDetail(withCid, [])]);
    expect(withCidModel.nodes.find((n) => n.id === "E")!.communityId).toBe(7);
    const plain = makeEntity({ id: "F" });
    const plainModel = buildGraphModel([plain], [makeDetail(plain, [])]);
    expect(plainModel.nodes.find((n) => n.id === "F")!.communityId).toBeNull();
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
