// SPDX-License-Identifier: AGPL-3.0-only
import { describe, it, expect, vi } from "vitest";
import type Graph from "graphology";
import type { ForceLink, SimulationLinkDatum } from "d3-force";
import type { GraphModel, GraphNode, GraphEdge } from "./model";
import type { GraphPalette } from "./palette";
import {
  buildAtlasGraph,
  runAtlasLayout,
  createAtlasSimulation,
  nonSimulatedIds,
  satellitePlan,
  placeSatellites,
  shelveComponents,
  SHELF_GAP,
  SHELF_TOP_GAP,
  SHELF_WIDTH_FACTOR,
  hoverStateFor,
  nodeDisplay,
  edgeDisplay,
  drawPageRings,
  drawRadialNodeLabel,
} from "./atlas";
import type { HoverState, AtlasSimNode } from "./atlas";

const PALETTE: GraphPalette = {
  project: "#111111",
  tool: "#222222",
  org: "#333333",
  person: "#444444",
  concept: "#555555",
  neutral: "#666666",
  edge: "#777777",
  edgeStrong: "#888888",
  label: "#999999",
  labelMuted: "#aaaaaa",
  // Black surface keeps the composite math legible: composited channel is
  // just slotChannel * alpha.
  surface: "#000000",
  hull: "rgba(1,2,3,0.05)",
  hullBorder: "rgba(1,2,3,0.16)",
  graticule: "rgba(4,5,6,0.13)",
  bridge: "#bbbbbb",
  memory: "#cccccc",
  page: "#dddddd",
};

function node(overrides: Partial<GraphNode> = {}): GraphNode {
  return {
    id: overrides.id ?? "n1",
    kind: "entity",
    name: overrides.name ?? "Node",
    entityType: overrides.entityType ?? "concept",
    // NOT `?? true`: null is a real value here (relation-derived neighbors),
    // and ?? would silently promote it to confirmed.
    confirmed: "confirmed" in overrides ? (overrides.confirmed as boolean | null) : true,
    degree: overrides.degree ?? 0,
    space: "space" in overrides ? (overrides.space as string | null) : null,
    createdAt: overrides.createdAt ?? 100,
    updatedAt: overrides.updatedAt ?? 200,
  };
}

function edge(overrides: Partial<GraphEdge> = {}): GraphEdge {
  return {
    id: overrides.id ?? "e1",
    source: overrides.source ?? "n1",
    target: overrides.target ?? "n2",
    type: overrides.type ?? "knows",
    confidence: overrides.confidence ?? null,
    createdAt: overrides.createdAt ?? 100,
  };
}

function makeModel(nodes: GraphNode[], edges: GraphEdge[] = []): GraphModel {
  return { nodes, edges, coverage: { relationsFetchedFor: nodes.length, totalEntities: nodes.length } };
}

describe("buildAtlasGraph", () => {
  it("carries every model node and edge into the graphology graph", () => {
    const model = makeModel(
      [node({ id: "a" }), node({ id: "b" }), node({ id: "c" })],
      [edge({ id: "e1", source: "a", target: "b" }), edge({ id: "e2", source: "b", target: "c" })],
    );
    const graph = buildAtlasGraph(model, PALETTE);
    expect(graph.order).toBe(3);
    expect(graph.size).toBe(2);
  });

  it("is deterministic: the same model produces identical node attributes and positions", () => {
    const model = makeModel(
      [node({ id: "a", degree: 3 }), node({ id: "b", degree: 1 })],
      [edge({ id: "e1", source: "a", target: "b" })],
    );
    const g1 = buildAtlasGraph(model, PALETTE);
    const g2 = buildAtlasGraph(model, PALETTE);
    expect(g1.getNodeAttributes("a")).toEqual(g2.getNodeAttributes("a"));
    expect(g1.getNodeAttributes("b")).toEqual(g2.getNodeAttributes("b"));
  });

  it("scales node size monotonically with degree", () => {
    const model = makeModel([
      node({ id: "a", degree: 0 }),
      node({ id: "b", degree: 1 }),
      node({ id: "c", degree: 4 }),
      node({ id: "d", degree: 9 }),
    ]);
    const graph = buildAtlasGraph(model, PALETTE);
    const sizes = ["a", "b", "c", "d"].map((id) => graph.getNodeAttribute(id, "size") as number);
    expect(sizes[0]).toBeLessThan(sizes[1]);
    expect(sizes[1]).toBeLessThan(sizes[2]);
    expect(sizes[2]).toBeLessThan(sizes[3]);
  });

  it("fills nodes with the stability-tiered composite of their slot color over the surface", () => {
    const model = makeModel([
      node({ id: "p", entityType: "project", confirmed: true }),
      node({ id: "t", entityType: "technology", confirmed: false }),
      node({ id: "x", entityType: "place", confirmed: null }), // unknown type -> neutral
    ]);
    const graph = buildAtlasGraph(model, PALETTE);
    // Confirmed: project #111111 at 0.9 over #000000 → 0x11 * 0.9 = 15 → #0f0f0f.
    expect(graph.getNodeAttribute("p", "color")).toBe("#0f0f0f");
    // Unconfirmed: tool #222222 at 0.5 → 0x22 * 0.5 = 17 → #111111.
    expect(graph.getNodeAttribute("t", "color")).toBe("#111111");
    // Unknown status (relation-derived): neutral #666666 at 0.5 → #333333.
    expect(graph.getNodeAttribute("x", "color")).toBe("#333333");
  });

  it("gives a confirmed node a larger size base than an unconfirmed one at equal degree, capped at 12", () => {
    const model = makeModel([
      node({ id: "conf", confirmed: true, degree: 2 }),
      node({ id: "unconf", confirmed: false, degree: 2 }),
      node({ id: "unknown", confirmed: null, degree: 2 }),
      node({ id: "hub", confirmed: true, degree: 300 }),
    ]);
    const graph = buildAtlasGraph(model, PALETTE);
    // base + 1.6 * log2(1 + degree); log2(3) = 1.585.
    const growth = 1.6 * Math.log2(3);
    expect(graph.getNodeAttribute("conf", "size")).toBeCloseTo(4 + growth, 10);
    expect(graph.getNodeAttribute("unconf", "size")).toBeCloseTo(3 + growth, 10);
    expect(graph.getNodeAttribute("unknown", "size")).toBeCloseTo(3 + growth, 10);
    expect(graph.getNodeAttribute("hub", "size")).toBe(12);
  });

  it("keeps a wiki page on the entity scale and a memory below it, capped", () => {
    const model = makeModel([
      node({ id: "page", entityType: "page", confirmed: null, degree: 3 }),
      node({ id: "entity", entityType: "concept", confirmed: false, degree: 3 }),
      node({ id: "memory", entityType: "memory", confirmed: true, degree: 3 }),
      node({ id: "memhub", entityType: "memory", confirmed: true, degree: 300 }),
    ]);
    const graph = buildAtlasGraph(model, PALETTE);
    const growth = 1.6 * Math.log2(4);
    expect(graph.getNodeAttribute("page", "size")).toBeCloseTo(3 + growth, 10);
    expect(graph.getNodeAttribute("entity", "size")).toBeCloseTo(3 + growth, 10);
    // Memories are context: they start lowest and are capped well under the
    // entity/page ceiling, so a much-cited memory can never dominate.
    expect(graph.getNodeAttribute("memory", "size")).toBe(4);
    expect(graph.getNodeAttribute("memhub", "size")).toBe(4);
  });

  it("draws a shared-source edge thinner than a real link and keeps the verb on the edge", () => {
    const model = makeModel(
      [node({ id: "a" }), node({ id: "b" })],
      [
        { id: "w", source: "a", target: "b", type: "wikilink", confidence: null, createdAt: 1 },
        {
          id: "s",
          source: "a",
          target: "b",
          type: "shared_source",
          confidence: null,
          createdAt: 1,
          weight: 2,
        },
      ],
    );
    const graph = buildAtlasGraph(model, PALETTE);
    expect(graph.getEdgeAttribute("w", "size")).toBe(1.5);
    expect(graph.getEdgeAttribute("s", "size")).toBe(0.8);
    expect(graph.getEdgeAttribute("s", "edgeType")).toBe("shared_source");
  });

  it("rings every page node just outside its disc, in the page ink", () => {
    const calls: string[] = [];
    const ctx = {
      save: () => calls.push("save"),
      restore: () => calls.push("restore"),
      beginPath: () => calls.push("beginPath"),
      stroke: () => calls.push("stroke"),
      arc: (x: number, y: number, r: number) => calls.push(`arc:${x},${y},${r}`),
      strokeStyle: "",
      lineWidth: 0,
    } as unknown as CanvasRenderingContext2D;
    drawPageRings(ctx, [{ x: 10, y: 20, size: 5 }], PALETTE);
    expect(ctx.strokeStyle).toBe(PALETTE.page);
    expect(ctx.lineWidth).toBe(1);
    // Radius is the disc plus a 2px gap, so the ring reads as a separate mark
    // rather than a slightly fatter dot — round 3 tightened both the gap and
    // the stroke so dense page clusters stop merging into one teal mass.
    expect(calls).toContain("arc:10,20,7");
    expect(calls.filter((c) => c === "stroke")).toHaveLength(1);
  });

  it("draws nothing when there are no pages on the map", () => {
    let touched = false;
    const ctx = { save: () => (touched = true) } as unknown as CanvasRenderingContext2D;
    drawPageRings(ctx, [], PALETTE);
    expect(touched).toBe(false);
  });

  it("stores confirmed on the node so theme recoloring can recompute the tiered fill", () => {
    const graph = buildAtlasGraph(makeModel([node({ id: "a", confirmed: null })]), PALETTE);
    expect(graph.getNodeAttribute("a", "confirmed")).toBeNull();
  });

  it("colors edges with the palette's quiet edge tone, size 1.5 (CSS px — old graph's exact stroke)", () => {
    const model = makeModel(
      [node({ id: "a" }), node({ id: "b" })],
      [edge({ id: "e1", source: "a", target: "b" })],
    );
    const graph = buildAtlasGraph(model, PALETTE);
    expect(graph.getEdgeAttribute("e1", "color")).toBe(PALETTE.edge);
    expect(graph.getEdgeAttribute("e1", "size")).toBe(1.5);
  });

  it("paints an edge between two real regions in the amber bridge ink, slightly thinner", () => {
    // Two triangles joined by a1–b1 — the canonical bridge fixture.
    const model = makeModel(
      ["a1", "a2", "a3", "b1", "b2", "b3"].map((id) => node({ id })),
      [
        edge({ id: "ea1", source: "a1", target: "a2" }),
        edge({ id: "ea2", source: "a2", target: "a3" }),
        edge({ id: "ea3", source: "a3", target: "a1" }),
        edge({ id: "eb1", source: "b1", target: "b2" }),
        edge({ id: "eb2", source: "b2", target: "b3" }),
        edge({ id: "eb3", source: "b3", target: "b1" }),
        edge({ id: "bridge", source: "a1", target: "b1" }),
      ],
    );
    const communities = new Map<string, string>([
      ["a1", "0"],
      ["a2", "0"],
      ["a3", "0"],
      ["b1", "1"],
      ["b2", "1"],
      ["b3", "1"],
    ]);
    const graph = buildAtlasGraph(model, PALETTE, communities);
    expect(graph.getEdgeAttribute("bridge", "color")).toBe(PALETTE.bridge);
    expect(graph.getEdgeAttribute("bridge", "size")).toBe(1.4);
    expect(graph.getEdgeAttribute("bridge", "bridge")).toBe(true);
    // Intra-region edges keep the quiet tone.
    expect(graph.getEdgeAttribute("ea1", "color")).toBe(PALETTE.edge);
    expect(graph.getEdgeAttribute("ea1", "size")).toBe(1.5);
    expect(graph.getEdgeAttribute("ea1", "bridge")).toBe(false);
  });

  it("treats every edge as normal when no community map is passed", () => {
    const model = makeModel(
      [node({ id: "a" }), node({ id: "b" })],
      [edge({ id: "e1", source: "a", target: "b" })],
    );
    const graph = buildAtlasGraph(model, PALETTE);
    expect(graph.getEdgeAttribute("e1", "color")).toBe(PALETTE.edge);
    expect(graph.getEdgeAttribute("e1", "bridge")).toBe(false);
  });

  it("keeps distinct parallel relations between the same pair as distinct edges", () => {
    // GraphModel's parallel-edge policy (see model.ts) keeps these as two
    // separate edges — a non-multi graph would throw adding the second one.
    const model = makeModel(
      [node({ id: "a" }), node({ id: "b" })],
      [
        edge({ id: "e1", source: "a", target: "b", type: "founded" }),
        edge({ id: "e2", source: "a", target: "b", type: "mentors" }),
      ],
    );
    const graph = buildAtlasGraph(model, PALETTE);
    expect(graph.size).toBe(2);
  });

  it("seeds finite deterministic positions before any layout has run", () => {
    const model = makeModel([node({ id: "a" }), node({ id: "b" }), node({ id: "c" })]);
    const graph = buildAtlasGraph(model, PALETTE);
    for (const id of ["a", "b", "c"]) {
      expect(Number.isFinite(graph.getNodeAttribute(id, "x"))).toBe(true);
      expect(Number.isFinite(graph.getNodeAttribute(id, "y"))).toBe(true);
    }
  });
});

describe("runAtlasLayout", () => {
  it("leaves every node with finite coordinates after layout", () => {
    const model = makeModel(
      [node({ id: "a" }), node({ id: "b" }), node({ id: "c" }), node({ id: "d" })],
      [
        edge({ id: "e1", source: "a", target: "b" }),
        edge({ id: "e2", source: "b", target: "c" }),
        edge({ id: "e3", source: "c", target: "d" }),
      ],
    );
    const graph = buildAtlasGraph(model, PALETTE);
    runAtlasLayout(graph);
    graph.forEachNode((_id, attrs) => {
      expect(Number.isFinite(attrs.x)).toBe(true);
      expect(Number.isFinite(attrs.y)).toBe(true);
    });
  });

  it("is deterministic: laying out identically-built graphs lands on the same positions", () => {
    const model = makeModel(
      [node({ id: "a" }), node({ id: "b" }), node({ id: "c" })],
      [edge({ id: "e1", source: "a", target: "b" }), edge({ id: "e2", source: "b", target: "c" })],
    );
    const g1 = buildAtlasGraph(model, PALETTE);
    const g2 = buildAtlasGraph(model, PALETTE);
    runAtlasLayout(g1);
    runAtlasLayout(g2);
    for (const id of ["a", "b", "c"]) {
      expect(g1.getNodeAttribute(id, "x")).toBeCloseTo(g2.getNodeAttribute(id, "x") as number, 10);
      expect(g1.getNodeAttribute(id, "y")).toBeCloseTo(g2.getNodeAttribute(id, "y") as number, 10);
    }
  });
});

describe("createAtlasSimulation", () => {
  function starGraph(): Graph {
    // Hub "h" with four spokes, laid out once so the spokes start near the hub.
    const model = makeModel(
      [node({ id: "h" }), node({ id: "s1" }), node({ id: "s2" }), node({ id: "s3" }), node({ id: "s4" })],
      [
        edge({ id: "e1", source: "h", target: "s1" }),
        edge({ id: "e2", source: "h", target: "s2" }),
        edge({ id: "e3", source: "h", target: "s3" }),
        edge({ id: "e4", source: "h", target: "s4" }),
      ],
    );
    const graph = buildAtlasGraph(model, PALETTE);
    runAtlasLayout(graph);
    return graph;
  }

  it("pulls a neighbor closer to a hub displaced via fx/fy", () => {
    const graph = starGraph();
    const sim = createAtlasSimulation(graph);
    const simNodes = sim.nodes();
    const hub = simNodes.find((n) => n.id === "h")!;
    const neighbor = simNodes.find((n) => n.id === "s1")!;

    const newHub = { x: hub.x! + 300, y: hub.y! + 300 };
    const distBefore = Math.hypot(neighbor.x! - newHub.x, neighbor.y! - newHub.y);

    hub.fx = newHub.x;
    hub.fy = newHub.y;
    sim.alpha(1);
    sim.tick(30);

    const distAfter = Math.hypot(neighbor.x! - newHub.x, neighbor.y! - newHub.y);
    expect(distAfter).toBeLessThan(distBefore);
  });

  it("excludes isolates from the simulation entirely — the ring-hold is structural, not fx/fy", () => {
    const model = makeModel(
      [node({ id: "a", degree: 1 }), node({ id: "b", degree: 1 }), node({ id: "iso" })],
      [edge({ id: "e1", source: "a", target: "b" })],
    );
    const graph = buildAtlasGraph(model, PALETTE);
    const sim = createAtlasSimulation(graph);
    expect(sim.nodes().some((n) => n.id === "iso")).toBe(false);
  });

  it("EQUILIBRIUM INVARIANT: settles to near-zero alpha at creation, and reheating without a drag barely moves the connected cluster", () => {
    const graph = starGraph();
    const sim = createAtlasSimulation(graph);

    expect(sim.alpha()).toBeLessThanOrEqual(0.01);

    const bboxDiagonal = () => {
      let minX = Infinity;
      let maxX = -Infinity;
      let minY = Infinity;
      let maxY = -Infinity;
      for (const n of sim.nodes()) {
        minX = Math.min(minX, n.x!);
        maxX = Math.max(maxX, n.x!);
        minY = Math.min(minY, n.y!);
        maxY = Math.max(maxY, n.y!);
      }
      return Math.hypot(maxX - minX, maxY - minY);
    };

    const before = bboxDiagonal();
    // Reheat WITHOUT touching fx/fy on anything — no drag in progress, so a
    // sim already at its own equilibrium should barely move.
    sim.alphaTarget(0);
    sim.alpha(0.3);
    sim.tick(60);
    const after = bboxDiagonal();

    expect(Math.abs(after - before) / before).toBeLessThan(0.03);
  });

  it("collapses parallel edges between the same pair to a single sim link", () => {
    const model = makeModel(
      [node({ id: "a" }), node({ id: "b" })],
      [
        edge({ id: "e1", source: "a", target: "b", type: "founded" }),
        edge({ id: "e2", source: "a", target: "b", type: "mentors" }),
      ],
    );
    const graph = buildAtlasGraph(model, PALETTE);
    const sim = createAtlasSimulation(graph);
    const linkForce = sim.force<ForceLink<AtlasSimNode, SimulationLinkDatum<AtlasSimNode>>>("link");
    expect(linkForce?.links()).toHaveLength(1);
  });

  it("syncs ticked positions back onto the graphology graph for connected nodes", () => {
    const graph = starGraph();
    const before = {
      x: graph.getNodeAttribute("s1", "x") as number,
      y: graph.getNodeAttribute("s1", "y") as number,
    };

    const sim = createAtlasSimulation(graph);
    sim.alpha(1);
    sim.tick(30);

    const after = {
      x: graph.getNodeAttribute("s1", "x") as number,
      y: graph.getNodeAttribute("s1", "y") as number,
    };
    expect(after).not.toEqual(before);
  });

  it("invokes onTick after every writeback — settle, then the pack pass, then once per manual tick", () => {
    const graph = starGraph();
    const onTick = vi.fn();
    const sim = createAtlasSimulation(graph, onTick);
    // The settle runs as a single wrapped tick(220) call → one writeback;
    // round 4's component packing runs the same writeback path afterwards so
    // satellites follow their moved anchors → a second.
    expect(onTick).toHaveBeenCalledTimes(2);
    sim.tick(1);
    expect(onTick).toHaveBeenCalledTimes(3);
  });
});

describe("hoverStateFor", () => {
  it("returns the empty state when nothing is hovered", () => {
    const model = makeModel(
      [node({ id: "a" }), node({ id: "b" })],
      [edge({ id: "e1", source: "a", target: "b" })],
    );
    const graph = buildAtlasGraph(model, PALETTE);
    const state = hoverStateFor(graph, null);
    expect(state.hovered).toBeNull();
    expect(state.neighbors.size).toBe(0);
  });

  it("collects the exact neighbor set for a hovered node with edges", () => {
    const model = makeModel(
      [node({ id: "a" }), node({ id: "b" }), node({ id: "c" }), node({ id: "d" })],
      [edge({ id: "e1", source: "a", target: "b" }), edge({ id: "e2", source: "a", target: "c" })],
    );
    const graph = buildAtlasGraph(model, PALETTE);
    const state = hoverStateFor(graph, "a");
    expect(state.hovered).toBe("a");
    expect(state.neighbors).toEqual(new Set(["b", "c"]));
  });

  it("returns an empty neighbor set for an isolated node", () => {
    const model = makeModel([node({ id: "a" }), node({ id: "iso" })]);
    const graph = buildAtlasGraph(model, PALETTE);
    const state = hoverStateFor(graph, "iso");
    expect(state.neighbors.size).toBe(0);
  });
});

describe("nodeDisplay", () => {
  const attrs = { label: "Alice", color: "#123456", size: 8 };

  it("passes attrs through unchanged when nothing is hovered", () => {
    const state: HoverState = { hovered: null, neighbors: new Set() };
    expect(nodeDisplay(state, "a", attrs, PALETTE)).toEqual(attrs);
  });

  it("keeps the hovered node's own color, forces its label, and puts it on top", () => {
    const state: HoverState = { hovered: "a", neighbors: new Set(["b"]) };
    const result = nodeDisplay(state, "a", attrs, PALETTE);
    expect(result.color).toBe(attrs.color);
    expect(result.label).toBe(attrs.label);
    expect(result.forceLabel).toBe(true);
    expect(result.zIndex).toBe(2);
  });

  it("keeps a neighbor's own color and label, at zIndex 1", () => {
    const state: HoverState = { hovered: "a", neighbors: new Set(["b"]) };
    const result = nodeDisplay(state, "b", attrs, PALETTE);
    expect(result.color).toBe(attrs.color);
    expect(result.label).toBe(attrs.label);
    expect(result.forceLabel).toBeUndefined();
    expect(result.zIndex).toBe(1);
  });

  it("mutes and blanks everyone else, at zIndex 0", () => {
    const state: HoverState = { hovered: "a", neighbors: new Set(["b"]) };
    const result = nodeDisplay(state, "c", attrs, PALETTE);
    expect(result.color).toBe(PALETTE.edge);
    expect(result.label).toBe("");
    expect(result.zIndex).toBe(0);
  });
});

describe("edgeDisplay", () => {
  const attrs = { color: "#abcdef", size: 1 };

  it("passes attrs through unchanged when nothing is hovered", () => {
    const state: HoverState = { hovered: null, neighbors: new Set() };
    expect(edgeDisplay(state, "e1", "a", "b", attrs, PALETTE)).toEqual(attrs);
  });

  it("emphasizes an edge incident to the hovered node as source", () => {
    const state: HoverState = { hovered: "a", neighbors: new Set(["b"]) };
    const result = edgeDisplay(state, "e1", "a", "b", attrs, PALETTE);
    expect(result.color).toBe(PALETTE.edgeStrong);
    expect(result.zIndex).toBe(1);
    expect(result.hidden).toBeUndefined();
  });

  it("emphasizes an edge incident to the hovered node as target", () => {
    const state: HoverState = { hovered: "b", neighbors: new Set(["a"]) };
    const result = edgeDisplay(state, "e1", "a", "b", attrs, PALETTE);
    expect(result.color).toBe(PALETTE.edgeStrong);
    expect(result.zIndex).toBe(1);
  });

  it("hides a non-incident edge while hovering", () => {
    const state: HoverState = { hovered: "a", neighbors: new Set(["b"]) };
    const result = edgeDisplay(state, "e2", "b", "c", attrs, PALETTE);
    expect(result.hidden).toBe(true);
  });

  it("emphasizes both parallel edges between the hovered node and a neighbor", () => {
    const state: HoverState = { hovered: "a", neighbors: new Set(["b"]) };
    const r1 = edgeDisplay(state, "e1", "a", "b", attrs, PALETTE);
    const r2 = edgeDisplay(state, "e2", "a", "b", attrs, PALETTE);
    expect(r1.color).toBe(PALETTE.edgeStrong);
    expect(r2.color).toBe(PALETTE.edgeStrong);
  });
});

describe("drawRadialNodeLabel", () => {
  function mockCtx() {
    return {
      font: "",
      fillStyle: "",
      globalAlpha: 1,
      textAlign: "",
      textBaseline: "",
      fillText: vi.fn(),
    };
  }

  // One node whose GRAPH position we place per case; the drawer's viewport
  // data stays fixed at (100, 50) size 4 → pad 12.
  function graphWithNodeAt(gx: number, gy: number) {
    const graph = buildAtlasGraph(makeModel([node({ id: "n1", name: "Alice" })]), PALETTE);
    graph.setNodeAttribute("n1", "x", gx);
    graph.setNodeAttribute("n1", "y", gy);
    return graph;
  }
  const data = { key: "n1", label: "Alice", size: 4, x: 100, y: 50 };
  const settings = { labelColor: { color: "#abcdef" } };

  it("places the label INWARD (left of the node) for a node right of the graph center", () => {
    const ctx = mockCtx();
    drawRadialNodeLabel(ctx as any, data, settings, graphWithNodeAt(10, 0));
    expect(ctx.textAlign).toBe("right");
    expect(ctx.textBaseline).toBe("middle");
    expect(ctx.fillText).toHaveBeenCalledWith("Alice", 88, 50);
  });

  it("places the label right of the node for a node left of the graph center", () => {
    const ctx = mockCtx();
    drawRadialNodeLabel(ctx as any, data, settings, graphWithNodeAt(-10, 0));
    expect(ctx.textAlign).toBe("left");
    expect(ctx.fillText).toHaveBeenCalledWith("Alice", 112, 50);
  });

  it("places the label below the node for a node above center — graph +y is SCREEN-UP in sigma", () => {
    const ctx = mockCtx();
    drawRadialNodeLabel(ctx as any, data, settings, graphWithNodeAt(0, 10));
    expect(ctx.textAlign).toBe("center");
    expect(ctx.textBaseline).toBe("top");
    // Viewport y grows downward: +pad = below the node on screen = inward.
    expect(ctx.fillText).toHaveBeenCalledWith("Alice", 100, 62);
  });

  it("places the label above the node for a node below center", () => {
    const ctx = mockCtx();
    drawRadialNodeLabel(ctx as any, data, settings, graphWithNodeAt(0, -10));
    expect(ctx.textBaseline).toBe("bottom");
    expect(ctx.fillText).toHaveBeenCalledWith("Alice", 100, 38);
  });

  it("draws 12px system-font ink from settings.labelColor at 85% alpha, restored after", () => {
    const ctx = mockCtx();
    let alphaAtDraw = 0;
    ctx.fillText.mockImplementation(() => {
      alphaAtDraw = ctx.globalAlpha;
    });
    drawRadialNodeLabel(ctx as any, data, settings, graphWithNodeAt(10, 0));
    expect(ctx.font).toBe("12px -apple-system, sans-serif");
    expect(ctx.fillStyle).toBe("#abcdef");
    expect(alphaAtDraw).toBe(0.85);
    expect(ctx.globalAlpha).toBe(1); // restored — the labels canvas is shared
  });

  it("draws nothing for an empty label (hover reducer blanks dimmed nodes)", () => {
    const ctx = mockCtx();
    drawRadialNodeLabel(ctx as any, { ...data, label: "" }, settings, graphWithNodeAt(10, 0));
    expect(ctx.fillText).not.toHaveBeenCalled();
  });
});

describe("shared-source edges and node size", () => {
  it("sizes a page from its asserted links only, ignoring shared-source overlap", () => {
    // Same drawn degree (3), but two of "overlap"'s edges are inferred
    // overlap rather than asserted links, so it draws at asserted-degree 1.
    const model = makeModel(
      [
        node({ id: "asserted", entityType: "page", confirmed: null, degree: 3 }),
        node({ id: "overlap", entityType: "page", confirmed: null, degree: 3 }),
        node({ id: "x" }),
        node({ id: "y" }),
        node({ id: "z" }),
      ],
      [
        edge({ id: "a1", source: "asserted", target: "x", type: "wikilink" }),
        edge({ id: "a2", source: "asserted", target: "y", type: "wikilink" }),
        edge({ id: "a3", source: "asserted", target: "z", type: "cites" }),
        edge({ id: "o1", source: "overlap", target: "x", type: "wikilink" }),
        edge({ id: "o2", source: "overlap", target: "y", type: "shared_source" }),
        edge({ id: "o3", source: "overlap", target: "z", type: "shared_source" }),
      ],
    );
    const graph = buildAtlasGraph(model, PALETTE);
    expect(graph.getNodeAttribute("asserted", "size")).toBeCloseTo(3 + 1.6 * Math.log2(4), 10);
    expect(graph.getNodeAttribute("overlap", "size")).toBeCloseTo(3 + 1.6 * Math.log2(2), 10);
  });

  it("never sizes a node below its base, however many shared-source edges touch it", () => {
    const model = makeModel(
      [node({ id: "p", entityType: "page", confirmed: null, degree: 1 }), node({ id: "q", entityType: "page", confirmed: null, degree: 1 })],
      [edge({ id: "s", source: "p", target: "q", type: "shared_source" })],
    );
    const graph = buildAtlasGraph(model, PALETTE);
    expect(graph.getNodeAttribute("p", "size")).toBe(3);
  });
});

describe("nonSimulatedIds and satellites", () => {
  function leafModel(): GraphModel {
    return makeModel(
      [
        node({ id: "hub", degree: 3 }),
        node({ id: "peer", degree: 1 }),
        node({ id: "leaf1", entityType: "memory", confirmed: null, degree: 1 }),
        node({ id: "leaf2", entityType: "memory", confirmed: null, degree: 1 }),
        node({ id: "busy", entityType: "memory", confirmed: null, degree: 2 }),
      ],
      [
        edge({ id: "e1", source: "hub", target: "peer" }),
        edge({ id: "e2", source: "leaf1", target: "hub", type: "mentions" }),
        edge({ id: "e3", source: "leaf2", target: "hub", type: "mentions" }),
        edge({ id: "e4", source: "busy", target: "hub", type: "mentions" }),
        edge({ id: "e5", source: "busy", target: "peer", type: "mentions" }),
      ],
    );
  }

  it("excludes degree-1 memories and isolates, but keeps every other node", () => {
    const graph = buildAtlasGraph(
      makeModel(
        [...leafModel().nodes, node({ id: "iso" })],
        leafModel().edges,
      ),
      PALETTE,
    );
    expect(nonSimulatedIds(graph).sort()).toEqual(["iso", "leaf1", "leaf2"]);
  });

  it("does not treat a degree-1 ENTITY as a satellite — only memories orbit", () => {
    const graph = buildAtlasGraph(leafModel(), PALETTE);
    expect(nonSimulatedIds(graph)).not.toContain("peer");
  });

  it("orbits each leaf around its one neighbour at the neighbour's radius plus a gap", () => {
    const graph = buildAtlasGraph(leafModel(), PALETTE);
    graph.setNodeAttribute("hub", "x", 40);
    graph.setNodeAttribute("hub", "y", -15);
    const plan = satellitePlan(graph);
    expect(plan.map((s) => s.id).sort()).toEqual(["leaf1", "leaf2"]);
    placeSatellites(graph, plan);
    const hubSize = graph.getNodeAttribute("hub", "size") as number;
    for (const id of ["leaf1", "leaf2"]) {
      const dx = (graph.getNodeAttribute(id, "x") as number) - 40;
      const dy = (graph.getNodeAttribute(id, "y") as number) + 15;
      expect(Math.hypot(dx, dy)).toBeCloseTo(hubSize + 6, 10);
    }
    // Two leaves on one anchor sit on opposite sides of it, not on top of
    // each other.
    expect(plan[0].angle).not.toBe(plan[1].angle);
  });

  it("keeps leaf memories out of the simulation and its links", () => {
    const graph = buildAtlasGraph(leafModel(), PALETTE);
    const sim = createAtlasSimulation(graph);
    expect(sim.nodes().map((n) => n.id).sort()).toEqual(["busy", "hub", "peer"]);
    const linkForce = sim.force<ForceLink<AtlasSimNode, SimulationLinkDatum<AtlasSimNode>>>("link");
    // hub-peer, busy-hub, busy-peer: the two leaf edges are drawn but never
    // simulated, so d3 is never asked to resolve an endpoint it does not own.
    expect(linkForce?.links()).toHaveLength(3);
  });

  it("carries a dragged anchor's leaves along on every tick writeback", () => {
    const graph = buildAtlasGraph(leafModel(), PALETTE);
    runAtlasLayout(graph);
    const sim = createAtlasSimulation(graph);
    const before = graph.getNodeAttribute("leaf1", "x") as number;
    // Exactly what AtlasView's mousemovebody does: write the pointer position
    // straight onto the graph AND pin the sim node there.
    const hubX = graph.getNodeAttribute("hub", "x") as number;
    graph.setNodeAttribute("hub", "x", hubX + 500);
    const hub = sim.nodes().find((n) => n.id === "hub")!;
    hub.fx = hubX + 500;
    hub.fy = hub.y;
    sim.alpha(0.3);
    sim.tick(5);
    const after = graph.getNodeAttribute("leaf1", "x") as number;
    expect(after - before).toBeCloseTo(500, 6);
  });

  it("is deterministic: the same graph plans the same orbits twice", () => {
    const g1 = buildAtlasGraph(leafModel(), PALETTE);
    const g2 = buildAtlasGraph(leafModel(), PALETTE);
    expect(satellitePlan(g1)).toEqual(satellitePlan(g2));
  });

  /** One entity with `count` leaf memories hanging off it — the real capture's
   *  worst anchor carries 374. */
  function haloModel(count: number): GraphModel {
    const nodes: GraphNode[] = [node({ id: "hub", degree: count })];
    const edges: GraphEdge[] = [];
    for (let i = 0; i < count; i += 1) {
      const id = `leaf${String(i).padStart(3, "0")}`;
      nodes.push(node({ id, entityType: "memory", confirmed: null, degree: 1 }));
      edges.push(edge({ id: `m${i}`, source: id, target: "hub", type: "mentions" }));
    }
    return makeModel(nodes, edges);
  }

  function haloGraph(count: number): Graph {
    const graph = buildAtlasGraph(haloModel(count), PALETTE);
    graph.setNodeAttribute("hub", "x", 0);
    graph.setNodeAttribute("hub", "y", 0);
    return graph;
  }

  it("fills SHELLS as the leaf count grows instead of crowding one circle", () => {
    const rings = (count: number) =>
      new Set(satellitePlan(haloGraph(count)).map((s) => s.radius.toFixed(6))).size;
    // Two leaves fit on the first ring; forty cannot.
    expect(rings(2)).toBe(1);
    expect(rings(40)).toBeGreaterThan(1);
    expect(rings(120)).toBeGreaterThan(rings(40));
  });

  it("never lets two satellite discs of one halo touch, at 40 leaves", () => {
    const graph = haloGraph(40);
    const plan = satellitePlan(graph);
    placeSatellites(graph, plan);
    const leafSize = graph.getNodeAttribute("leaf000", "size") as number;
    // Two discs plus a unit of sky. On one circle at the anchor's radius the
    // forty leaves sit ~1.6 units apart — a solid donut.
    const floor = 2 * leafSize + 1;
    let closest = Infinity;
    for (let i = 0; i < plan.length; i += 1) {
      for (let j = i + 1; j < plan.length; j += 1) {
        const a = plan[i] as { id: string };
        const b = plan[j] as { id: string };
        closest = Math.min(
          closest,
          Math.hypot(
            (graph.getNodeAttribute(a.id, "x") as number) - (graph.getNodeAttribute(b.id, "x") as number),
            (graph.getNodeAttribute(a.id, "y") as number) - (graph.getNodeAttribute(b.id, "y") as number),
          ),
        );
      }
    }
    expect(closest).toBeGreaterThanOrEqual(floor);
  });

  it("leaves the leaf under the pointer where the drag put it", () => {
    const graph = buildAtlasGraph(leafModel(), PALETTE);
    runAtlasLayout(graph);
    const sim = createAtlasSimulation(graph);
    // What AtlasView's mousemovebody does for a satellite: write the pointer
    // position straight onto the graph. It is not a sim node, so there is no
    // pin to set — the sim is told by id instead.
    sim.setDraggingId("leaf1");
    graph.setNodeAttribute("leaf1", "x", 900);
    graph.setNodeAttribute("leaf1", "y", -900);
    sim.alpha(0.3);
    sim.tick(5);
    expect(graph.getNodeAttribute("leaf1", "x")).toBe(900);
    expect(graph.getNodeAttribute("leaf1", "y")).toBe(-900);
    // Its sibling is still riding its orbit, so the writeback did run.
    expect(graph.getNodeAttribute("leaf2", "x")).not.toBe(900);
  });

  it("puts a released leaf back on its orbit — the exemption is for the drag only", () => {
    const graph = buildAtlasGraph(leafModel(), PALETTE);
    runAtlasLayout(graph);
    const sim = createAtlasSimulation(graph);
    graph.setNodeAttribute("leaf1", "x", 900);
    sim.alpha(0.3);
    sim.tick(1);
    expect(graph.getNodeAttribute("leaf1", "x")).not.toBe(900);
  });
});

describe("simulation forces (round 4)", () => {
  function pagePair(): GraphModel {
    return makeModel(
      [
        node({ id: "p1", entityType: "page", degree: 2 }),
        node({ id: "p2", entityType: "page", degree: 2 }),
        node({ id: "p3", entityType: "page", degree: 2 }),
      ],
      [
        edge({ id: "s1", source: "p1", target: "p2", type: "shared_source" }),
        edge({ id: "w1", source: "p2", target: "p3", type: "wikilink" }),
      ],
    );
  }

  it("gives each sim node a collision radius wider than its disc, and pages wider still", () => {
    const model = makeModel(
      [node({ id: "e", degree: 1 }), node({ id: "p", entityType: "page", degree: 1 })],
      [edge({ id: "e1", source: "e", target: "p", type: "about" })],
    );
    const graph = buildAtlasGraph(model, PALETTE);
    const sim = createAtlasSimulation(graph);
    sim.stop();
    const byId = new Map(sim.nodes().map((n) => [n.id, n]));
    const entity = byId.get("e")!;
    const page = byId.get("p")!;
    expect(entity.radius).toBe((graph.getNodeAttribute("e", "size") as number) + 2);
    // Page adds its detached ring (gap 2 + width 1) on top of the same pad.
    expect(page.radius).toBe((graph.getNodeAttribute("p", "size") as number) + 3 + 2);
  });

  it("registers a collide force, so two discs dropped on the same spot separate", () => {
    const graph = buildAtlasGraph(pagePair(), PALETTE);
    // Stack p1 and p2 exactly, which is what the page blob looked like.
    for (const id of ["p1", "p2"]) {
      graph.setNodeAttribute(id, "x", 0);
      graph.setNodeAttribute(id, "y", 0);
    }
    const sim = createAtlasSimulation(graph);
    sim.stop();
    const byId = new Map(sim.nodes().map((n) => [n.id, n]));
    const gap = Math.hypot(byId.get("p1")!.x! - byId.get("p2")!.x!, byId.get("p1")!.y! - byId.get("p2")!.y!);
    expect(sim.force("collide")).toBeDefined();
    expect(gap).toBeGreaterThan(byId.get("p1")!.radius);
  });

  it("gives each link the rest length its verb calls for", () => {
    const graph = buildAtlasGraph(pagePair(), PALETTE);
    const sim = createAtlasSimulation(graph);
    sim.stop();
    const linkForce = sim.force<ForceLink<AtlasSimNode, SimulationLinkDatum<AtlasSimNode>>>("link");
    const links = linkForce!.links() as unknown as { source: AtlasSimNode; target: AtlasSimNode }[];
    const distance = linkForce!.distance() as unknown as (link: unknown) => number;
    const strength = linkForce!.strength() as unknown as (link: unknown) => number;
    const shared = links.find((l) => l.source.id === "p1" || l.target.id === "p1")!;
    const wiki = links.find((l) => l.source.id === "p3" || l.target.id === "p3")!;
    expect(distance(shared)).toBe(70);
    expect(strength(shared)).toBeCloseTo(0.15, 6);
    expect(distance(wiki)).toBe(50);
    expect(strength(wiki)).toBeCloseTo(0.5, 6);
  });

  it("leaves an unlisted verb on d3's own distance and 1/min-degree strength", () => {
    const model = makeModel(
      [node({ id: "a", degree: 2 }), node({ id: "b", degree: 1 }), node({ id: "c", degree: 1 })],
      [
        edge({ id: "e1", source: "a", target: "b", type: "knows" }),
        edge({ id: "e2", source: "a", target: "c", type: "knows" }),
      ],
    );
    const graph = buildAtlasGraph(model, PALETTE);
    const sim = createAtlasSimulation(graph);
    sim.stop();
    const linkForce = sim.force<ForceLink<AtlasSimNode, SimulationLinkDatum<AtlasSimNode>>>("link");
    const links = linkForce!.links() as unknown as { source: AtlasSimNode; target: AtlasSimNode }[];
    const distance = linkForce!.distance() as unknown as (link: unknown) => number;
    const strength = linkForce!.strength() as unknown as (link: unknown) => number;
    // a has two links, b and c one each → 1/min(2,1) = 1, d3's own answer.
    expect(distance(links[0])).toBe(30);
    expect(strength(links[0])).toBeCloseTo(1, 6);
  });

  it("keeps the strongest verb when parallel edges collapse to one spring", () => {
    const model = makeModel(
      [node({ id: "p1", entityType: "page", degree: 2 }), node({ id: "p2", entityType: "page", degree: 2 })],
      [
        edge({ id: "s1", source: "p1", target: "p2", type: "shared_source" }),
        edge({ id: "w1", source: "p1", target: "p2", type: "wikilink" }),
      ],
    );
    const graph = buildAtlasGraph(model, PALETTE);
    const sim = createAtlasSimulation(graph);
    sim.stop();
    const linkForce = sim.force<ForceLink<AtlasSimNode, SimulationLinkDatum<AtlasSimNode>>>("link");
    const links = linkForce!.links();
    const distance = linkForce!.distance() as unknown as (link: unknown) => number;
    expect(links).toHaveLength(1);
    // The wikilink is the asserted link, so it sets the spring — not the
    // shared-source edge that happens to run beside it.
    expect(distance(links[0])).toBe(50);
  });
});

describe("shelveComponents", () => {
  /** One chain per requested size, so component sizes are exactly as asked
   *  and every node is an entity (a leaf MEMORY would be a satellite, which
   *  the shelf measures through its anchor instead). */
  function componentsModel(sizes: number[]): GraphModel {
    const nodes: GraphNode[] = [];
    const edges: GraphEdge[] = [];
    sizes.forEach((size, c) => {
      for (let i = 0; i < size; i += 1) {
        nodes.push(node({ id: `c${c}n${i}` }));
        if (i > 0) {
          edges.push(edge({ id: `c${c}e${i}`, source: `c${c}n${i - 1}`, target: `c${c}n${i}` }));
        }
      }
    });
    return makeModel(nodes, edges);
  }

  function shelved(sizes: number[]) {
    const graph = buildAtlasGraph(componentsModel(sizes), PALETTE);
    runAtlasLayout(graph);
    const placement = shelveComponents(graph);
    return { graph, placement };
  }

  /** The drawn extent of one component: every node's disc. */
  function box(graph: Graph, ids: string[]) {
    const pad = (id: string) => graph.getNodeAttribute(id, "size") as number;
    const xs = ids.map((id) => graph.getNodeAttribute(id, "x") as number);
    const ys = ids.map((id) => graph.getNodeAttribute(id, "y") as number);
    return {
      minX: Math.min(...ids.map((id, i) => (xs[i] as number) - pad(id))),
      maxX: Math.max(...ids.map((id, i) => (xs[i] as number) + pad(id))),
      minY: Math.min(...ids.map((id, i) => (ys[i] as number) - pad(id))),
      maxY: Math.max(...ids.map((id, i) => (ys[i] as number) + pad(id))),
    };
  }

  it("centres the largest component on the origin and returns it first", () => {
    const { graph, placement } = shelved([9, 6, 5]);
    expect(placement[0]).toHaveLength(9);
    const core = box(graph, placement[0] as string[]);
    expect((core.minX + core.maxX) / 2).toBeCloseTo(0, 6);
    expect((core.minY + core.maxY) / 2).toBeCloseTo(0, 6);
  });

  it("puts every other component BELOW the core, never around it", () => {
    const { graph, placement } = shelved([9, 6, 5, 5]);
    const core = box(graph, placement[0] as string[]);
    for (const ids of placement.slice(1)) {
      const shelf = box(graph, ids);
      // Graph +y is screen-up, so "below" is smaller y. The whole shelf box
      // clears the core's bottom edge by at least SHELF_TOP_GAP.
      expect(shelf.maxY).toBeLessThanOrEqual(core.minY - SHELF_TOP_GAP + 1e-6);
    }
    const first = box(graph, placement[1] as string[]);
    expect(first.maxY).toBeCloseTo(core.minY - SHELF_TOP_GAP, 6);
  });

  it("orders the shelf largest-first, left to right, one SHELF_GAP apart", () => {
    const { graph, placement } = shelved([12, 7, 6, 5]);
    const shelf = placement.slice(1);
    expect(shelf.map((ids) => ids.length)).toEqual([7, 6, 5]);
    const boxes = shelf.map((ids) => box(graph, ids));
    for (let i = 1; i < boxes.length; i += 1) {
      const left = boxes[i - 1] as { maxX: number };
      const right = boxes[i] as { minX: number };
      expect(right.minX - left.maxX).toBeCloseTo(SHELF_GAP, 6);
    }
  });

  it("centres each shelf row under the core", () => {
    const { graph, placement } = shelved([12, 7, 6, 5]);
    const boxes = placement.slice(1).map((ids) => box(graph, ids));
    const rowLeft = Math.min(...boxes.map((b) => b.minX));
    const rowRight = Math.max(...boxes.map((b) => b.maxX));
    expect((rowLeft + rowRight) / 2).toBeCloseTo(0, 6);
  });

  it("wraps to another row once a row would run wider than the core allows", () => {
    // Eight shelf components against a narrow core: one row cannot hold them.
    const { graph, placement } = shelved([6, 5, 5, 5, 5, 5, 5, 5, 5]);
    const core = box(graph, placement[0] as string[]);
    const boxes = placement.slice(1).map((ids) => box(graph, ids));
    const rows = new Map<string, { minX: number; maxX: number }[]>();
    for (const b of boxes) {
      const key = b.maxY.toFixed(6);
      const list = rows.get(key);
      if (list) list.push(b);
      else rows.set(key, [b]);
    }
    expect(rows.size).toBeGreaterThan(1);
    const budget = SHELF_WIDTH_FACTOR * (core.maxX - core.minX);
    for (const row of rows.values()) {
      const width = Math.max(...row.map((b) => b.maxX)) - Math.min(...row.map((b) => b.minX));
      // A row may only exceed the budget when a single component does.
      const widest = Math.max(...row.map((b) => b.maxX - b.minX));
      expect(width).toBeLessThanOrEqual(Math.max(budget, widest) + 1e-6);
    }
  });

  it("never overlaps two components", () => {
    const { graph, placement } = shelved([9, 6, 5, 5, 5, 5, 5]);
    const boxes = placement.map((ids) => box(graph, ids));
    for (let i = 0; i < boxes.length; i += 1) {
      for (let j = i + 1; j < boxes.length; j += 1) {
        const a = boxes[i] as { minX: number; maxX: number; minY: number; maxY: number };
        const b = boxes[j] as { minX: number; maxX: number; minY: number; maxY: number };
        const overlaps =
          a.minX < b.maxX && b.minX < a.maxX && a.minY < b.maxY && b.minY < a.maxY;
        expect(overlaps).toBe(false);
      }
    }
  });

  it("moves a component rigidly — internal distances survive", () => {
    const { graph, placement } = shelved([9, 5]);
    const ids = placement[1] as string[];
    const graph2 = buildAtlasGraph(componentsModel([9, 5]), PALETTE);
    runAtlasLayout(graph2);
    const at = (g: Graph, id: string) => ({
      x: g.getNodeAttribute(id, "x") as number,
      y: g.getNodeAttribute(id, "y") as number,
    });
    for (let i = 1; i < ids.length; i += 1) {
      const before = Math.hypot(
        at(graph2, ids[i] as string).x - at(graph2, ids[0] as string).x,
        at(graph2, ids[i] as string).y - at(graph2, ids[0] as string).y,
      );
      const after = Math.hypot(
        at(graph, ids[i] as string).x - at(graph, ids[0] as string).x,
        at(graph, ids[i] as string).y - at(graph, ids[0] as string).y,
      );
      expect(after).toBeCloseTo(before, 6);
    }
  });

  it("puts lone nodes last, on rows of their own", () => {
    const { graph, placement } = shelved([9, 5, 1, 1, 3]);
    // Largest-first among the grouped ones, then the singletons.
    expect(placement.slice(1).map((ids) => ids.length)).toEqual([5, 3, 1, 1]);
    const grouped = placement.slice(1, 3).map((ids) => box(graph, ids));
    const singles = placement.slice(3).map((ids) => box(graph, ids));
    const groupedBottom = Math.min(...grouped.map((b) => b.minY));
    for (const single of singles) {
      expect(single.maxY).toBeLessThanOrEqual(groupedBottom + 1e-6);
    }
    // And the two lone nodes share their own row.
    expect((singles[0] as { maxY: number }).maxY).toBeCloseTo(
      (singles[1] as { maxY: number }).maxY,
      6,
    );
  });

  it("keeps a shelved component clear of the core's satellite halo", () => {
    const leaves = 40;
    const nodes: GraphNode[] = [];
    const edges: GraphEdge[] = [];
    for (let i = 0; i < 6; i += 1) {
      nodes.push(node({ id: `k${i}` }));
      if (i > 0) edges.push(edge({ id: `ke${i}`, source: `k${i - 1}`, target: `k${i}` }));
    }
    for (let i = 0; i < leaves; i += 1) {
      const id = `leaf${String(i).padStart(3, "0")}`;
      nodes.push(node({ id, entityType: "memory", confirmed: null, degree: 1 }));
      edges.push(edge({ id: `m${i}`, source: id, target: "k0", type: "mentions" }));
    }
    for (let i = 0; i < 5; i += 1) {
      nodes.push(node({ id: `s${i}` }));
      if (i > 0) edges.push(edge({ id: `se${i}`, source: `s${i - 1}`, target: `s${i}` }));
    }
    const graph = buildAtlasGraph(makeModel(nodes, edges), PALETTE);
    // Placed by hand instead of laid out: the whole core sits on one line, so
    // the ONLY thing that can reach below it is k0's halo.
    for (let i = 0; i < 6; i += 1) {
      graph.setNodeAttribute(`k${i}`, "x", i * 40);
      graph.setNodeAttribute(`k${i}`, "y", 0);
    }
    for (let i = 0; i < 5; i += 1) {
      graph.setNodeAttribute(`s${i}`, "x", i * 40);
      graph.setNodeAttribute(`s${i}`, "y", 500);
    }
    const placement = shelveComponents(graph);
    placeSatellites(graph, satellitePlan(graph));
    expect(placement[0]).toContain("k0");
    const shelf = box(graph, ["s0", "s1", "s2", "s3", "s4"]);
    const leafSize = graph.getNodeAttribute("leaf000", "size") as number;
    let lowest = Infinity;
    for (let i = 0; i < leaves; i += 1) {
      lowest = Math.min(
        lowest,
        (graph.getNodeAttribute(`leaf${String(i).padStart(3, "0")}`, "y") as number) - leafSize,
      );
    }
    // The halo hangs well below the core's own discs, and the shelf still
    // clears its lowest leaf by the full SHELF_TOP_GAP.
    expect(lowest).toBeLessThan(-(graph.getNodeAttribute("k0", "size") as number) * 2);
    expect(shelf.maxY).toBeLessThanOrEqual(lowest - SHELF_TOP_GAP + 1e-6);
  });

  it("is deterministic: the same graph shelves the same way twice", () => {
    const first = shelved([9, 6, 5, 1]);
    const second = shelved([9, 6, 5, 1]);
    expect(second.placement).toEqual(first.placement);
    for (const ids of first.placement) {
      expect(box(second.graph, ids)).toEqual(box(first.graph, ids));
    }
  });

  it("does nothing to an empty graph", () => {
    const graph = buildAtlasGraph(makeModel([]), PALETTE);
    expect(shelveComponents(graph)).toEqual([]);
  });

  it("holds the shelved components still through a drag reheat", () => {
    const graph = buildAtlasGraph(componentsModel([9, 6, 5]), PALETTE);
    runAtlasLayout(graph);
    const sim = createAtlasSimulation(graph);
    const placement = [
      ["c1n0", "c1n1", "c1n2", "c1n3", "c1n4", "c1n5"],
      ["c2n0", "c2n1", "c2n2", "c2n3", "c2n4"],
    ];
    const before = placement.map((ids) => box(graph, ids));
    // Exactly AtlasView's downNode: pin the pressed core node, jump alpha to
    // 0.3 and reheat.
    const pressed = sim.nodes().find((n) => n.id === "c0n0") as {
      fx?: number | null;
      fy?: number | null;
      x?: number;
      y?: number;
    };
    pressed.fx = pressed.x;
    pressed.fy = pressed.y;
    sim.alpha(0.3).alphaTarget(0.3);
    sim.tick(60);
    placement.forEach((ids, i) => {
      expect(box(graph, ids)).toEqual(before[i]);
    });
  });

  it("holds the core in place through a reheat instead of chasing the shelf's mass", () => {
    const graph = buildAtlasGraph(componentsModel([24, 6, 6, 6, 6]), PALETTE);
    runAtlasLayout(graph);
    const sim = createAtlasSimulation(graph);
    const pressed = sim.nodes().find((n) => n.fx == null) as {
      fx?: number | null;
      fy?: number | null;
      x?: number;
      y?: number;
    };
    const rest = sim.nodes().filter((n) => n.fx == null && n !== pressed);
    const centroid = () => ({
      x: rest.reduce((sum, n) => sum + (n.x ?? 0), 0) / rest.length,
      y: rest.reduce((sum, n) => sum + (n.y ?? 0), 0) / rest.length,
    });
    const span = Math.max(...rest.map((n) => Math.hypot(n.x ?? 0, n.y ?? 0)));
    const before = centroid();

    pressed.fx = pressed.x;
    pressed.fy = pressed.y;
    sim.alpha(0.3).alphaTarget(0.3);
    sim.tick(60);

    const after = centroid();
    // The shelf is pinned BELOW the core, so a centre force that averaged
    // every node would see a centroid below the origin every tick and walk
    // the core upward to compensate — the two zones would drift apart.
    // 18 units on a 430-unit span as written. d3's own forceCenter, which
    // averages the pinned shelf in too, gives 268; averaging only the free
    // nodes but pulling them to a centroid of (0,0) rather than to where the
    // shelf pass left them gives 54.
    expect(Math.hypot(after.x - before.x, after.y - before.y)).toBeLessThan(span * 0.08);
  });
});
