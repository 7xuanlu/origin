// SPDX-License-Identifier: AGPL-3.0-only
import { describe, expect, it } from "vitest";
import { generateBakeoffGraph } from "./synthetic";

describe("generateBakeoffGraph", () => {
  it("is deterministic: same seed + same n produces a byte-identical model", () => {
    const a = generateBakeoffGraph(500, 7);
    const b = generateBakeoffGraph(500, 7);
    expect(a).toEqual(b);
  });

  it("a different seed changes the model", () => {
    const a = generateBakeoffGraph(500, 7);
    const b = generateBakeoffGraph(500, 8);
    expect(a).not.toEqual(b);
  });

  it("produces exactly n nodes", () => {
    const model = generateBakeoffGraph(250, 1);
    expect(model.nodes).toHaveLength(250);
  });

  it("only uses the 5 palette entity types, plus page nodes", () => {
    const model = generateBakeoffGraph(500, 1);
    const types = new Set(model.nodes.map((n) => n.entityType));
    for (const t of types) {
      expect(["project", "technology", "organization", "person", "concept", "page"]).toContain(t);
    }
    expect(model.nodes.some((n) => n.kind === "page")).toBe(true);
    expect(model.nodes.some((n) => n.kind === "entity")).toBe(true);
  });

  it("groups nodes into roughly sqrt(n) communities, all set (non-null)", () => {
    const model = generateBakeoffGraph(1000, 1);
    const communities = new Set(model.nodes.map((n) => n.communityId));
    expect(model.nodes.every((n) => n.communityId !== null)).toBe(true);
    expect(communities.size).toBe(Math.round(Math.sqrt(1000)));
  });

  it("skews degree toward hubs (max degree well above the mean)", () => {
    const model = generateBakeoffGraph(1000, 1);
    const degrees = model.nodes.map((n) => n.degree);
    const mean = degrees.reduce((s, d) => s + d, 0) / degrees.length;
    const max = Math.max(...degrees);
    expect(max).toBeGreaterThan(mean * 3);
  });

  it("emits deterministic x/y positions for the synchronous-layout fallback", () => {
    const a = generateBakeoffGraph(100, 3);
    const b = generateBakeoffGraph(100, 3);
    expect(a.nodes.every((n) => Number.isFinite(n.x) && Number.isFinite(n.y))).toBe(true);
    expect(a.nodes.map((n) => [n.x, n.y])).toEqual(b.nodes.map((n) => [n.x, n.y]));
  });

  it("reports full coverage", () => {
    const model = generateBakeoffGraph(300, 1);
    expect(model.coverage).toEqual({ relationsFetchedFor: 300, totalEntities: 300 });
  });
});
