// SPDX-License-Identifier: AGPL-3.0-only
import { describe, it, expect, vi } from "vitest";
import Graph from "graphology";
import type { GraphModel, GraphNode, GraphEdge } from "./model";
import type { GraphPalette } from "./palette";
import {
  communitiesFor,
  bridgeEdgeTest,
  convexHull,
  communityRegions,
  graticuleRadii,
  cartographyScene,
  drawCartography,
} from "./cartography";

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
  surface: "#000000",
  hull: "rgba(1,2,3,0.05)",
  hullBorder: "rgba(1,2,3,0.16)",
  graticule: "rgba(4,5,6,0.13)",
  bridge: "#bbbbbb",
};

function node(id: string, overrides: Partial<GraphNode> = {}): GraphNode {
  return {
    id,
    kind: "entity",
    name: overrides.name ?? id,
    entityType: "concept",
    confirmed: true,
    degree: 0,
    communityId: "communityId" in overrides ? (overrides.communityId as number | null) : null,
    createdAt: 100,
    updatedAt: 200,
  };
}

function edge(id: string, source: string, target: string): GraphEdge {
  return { id, source, target, type: "knows", confidence: null, createdAt: 100 };
}

function modelOf(nodes: GraphNode[], edges: GraphEdge[]): GraphModel {
  return {
    nodes,
    edges,
    coverage: { relationsFetchedFor: nodes.length, totalEntities: nodes.length },
  };
}

/** Two triangles (a1-a2-a3, b1-b2-b3) joined by the single edge a1-b1 —
 *  the canonical two-regions-one-bridge map. */
function twoTriangles(): GraphModel {
  return modelOf(
    ["a1", "a2", "a3", "b1", "b2", "b3"].map((id) => node(id)),
    [
      edge("ea1", "a1", "a2"),
      edge("ea2", "a2", "a3"),
      edge("ea3", "a3", "a1"),
      edge("eb1", "b1", "b2"),
      edge("eb2", "b2", "b3"),
      edge("eb3", "b3", "b1"),
      edge("bridge", "a1", "b1"),
    ],
  );
}

describe("communitiesFor", () => {
  it("splits two hub-bridged triangles into two communities instead of leaking across the bridge", () => {
    const communities = communitiesFor(twoTriangles());
    const a = [communities.get("a1"), communities.get("a2"), communities.get("a3")];
    const b = [communities.get("b1"), communities.get("b2"), communities.get("b3")];
    expect(new Set(a).size).toBe(1);
    expect(new Set(b).size).toBe(1);
    expect(a[0]).not.toBe(b[0]);
  });

  it("climbs spokes to their hub: two disjoint stars are two communities, isolates stay singletons", () => {
    const m = modelOf(
      ["hubA", "s1", "s2", "s3", "hubB", "t1", "t2", "t3", "alone"].map((id) => node(id)),
      [
        edge("e1", "hubA", "s1"),
        edge("e2", "hubA", "s2"),
        edge("e3", "hubA", "s3"),
        edge("e4", "hubB", "t1"),
        edge("e5", "hubB", "t2"),
        edge("e6", "hubB", "t3"),
      ],
    );
    const communities = communitiesFor(m);
    expect(communities.get("s1")).toBe(communities.get("hubA"));
    expect(communities.get("s3")).toBe(communities.get("hubA"));
    expect(communities.get("t2")).toBe(communities.get("hubB"));
    expect(communities.get("hubA")).not.toBe(communities.get("hubB"));
    const others = new Set([communities.get("hubA"), communities.get("hubB")]);
    expect(others.has(communities.get("alone"))).toBe(false);
  });

  it("ignores parallel edges and self-loops when ranking degree for the climb", () => {
    // "big" has 2 distinct neighbors but 4 edge records; "hub" has 3 distinct.
    // Spoke "shared" touches both — it must climb to hub, not to the
    // parallel-inflated big.
    const m = modelOf(
      ["hub", "h1", "h2", "shared", "big", "b1"].map((id) => node(id)),
      [
        edge("e1", "hub", "h1"),
        edge("e2", "hub", "h2"),
        edge("e3", "hub", "shared"),
        edge("e4", "big", "b1"),
        edge("e5", "big", "b1"),
        edge("e6", "big", "big"),
        edge("e7", "big", "shared"),
        edge("e8", "big", "shared"),
      ],
    );
    const communities = communitiesFor(m);
    expect(communities.get("shared")).toBe(communities.get("hub"));
  });

  it("uses daemon community ids verbatim when any node carries one, and never groups the nulls", () => {
    const m = modelOf(
      [
        node("x1", { communityId: 7 }),
        node("x2", { communityId: 7 }),
        node("x3", { communityId: 9 }),
        node("u1"),
        node("u2"),
      ],
      // Heavily connected nulls — a fallback detector would group them; the
      // daemon path must NOT.
      [edge("e1", "u1", "u2"), edge("e2", "u1", "x1")],
    );
    const communities = communitiesFor(m);
    expect(communities.get("x1")).toBe(7);
    expect(communities.get("x2")).toBe(7);
    expect(communities.get("x3")).toBe(9);
    expect(communities.get("u1")).not.toBe(communities.get("u2"));
    expect([7, 9]).not.toContain(communities.get("u1"));
    expect([7, 9]).not.toContain(communities.get("u2"));
  });
});

describe("bridgeEdgeTest", () => {
  it("flags only edges spanning two regions of at least 3 members", () => {
    const communities = communitiesFor(twoTriangles());
    const isBridge = bridgeEdgeTest(communities);
    expect(isBridge("a1", "b1")).toBe(true);
    expect(isBridge("a1", "a2")).toBe(false);
    expect(isBridge("b2", "b3")).toBe(false);
  });

  it("does not dress an edge into a sub-region islet as a bridge", () => {
    // Community 0 has 3 members, community 1 only 2.
    const communities = new Map<string, number>([
      ["r1", 0],
      ["r2", 0],
      ["r3", 0],
      ["p1", 1],
      ["p2", 1],
    ]);
    const isBridge = bridgeEdgeTest(communities);
    expect(isBridge("r1", "p1")).toBe(false);
    expect(isBridge("r1", "missing")).toBe(false);
  });
});

describe("convexHull", () => {
  it("drops interior points and returns the outer ring", () => {
    const hull = convexHull([
      { x: 0, y: 0 },
      { x: 10, y: 0 },
      { x: 10, y: 10 },
      { x: 0, y: 10 },
      { x: 5, y: 5 },
    ]);
    expect(hull).toHaveLength(4);
    expect(hull).toEqual(
      expect.arrayContaining([
        { x: 0, y: 0 },
        { x: 10, y: 0 },
        { x: 10, y: 10 },
        { x: 0, y: 10 },
      ]),
    );
  });

  it("returns what it can for degenerate input", () => {
    expect(convexHull([{ x: 1, y: 2 }])).toEqual([{ x: 1, y: 2 }]);
    expect(
      convexHull([
        { x: 0, y: 0 },
        { x: 4, y: 4 },
        { x: 2, y: 2 },
      ]),
    ).toHaveLength(2);
  });
});

function graphOf(
  nodes: { id: string; x: number; y: number; label: string }[],
  edges: [string, string][] = [],
): Graph {
  const graph = new Graph({ multi: true });
  for (const n of nodes) graph.addNode(n.id, { x: n.x, y: n.y, label: n.label });
  edges.forEach(([source, target], i) => graph.addEdgeWithKey(`e${i}`, source, target));
  return graph;
}

describe("communityRegions", () => {
  it("hulls communities of 3+, names them after the highest-degree member, and sorts largest-first", () => {
    const graph = graphOf(
      [
        { id: "a1", x: 0, y: 0, label: "Wenlan" },
        { id: "a2", x: 10, y: 0, label: "Tauri" },
        { id: "a3", x: 0, y: 10, label: "React" },
        { id: "a4", x: 10, y: 10, label: "Rust" },
        { id: "b1", x: 50, y: 50, label: "Claude Code" },
        { id: "b2", x: 60, y: 50, label: "Skills" },
        { id: "b3", x: 50, y: 60, label: "Anthropic" },
        { id: "tiny", x: 99, y: 99, label: "Islet" },
      ],
      [
        ["a1", "a2"],
        ["a1", "a3"],
        ["a1", "a4"],
        ["b1", "b2"],
        ["b1", "b3"],
      ],
    );
    const communities = new Map<string, number>([
      ["a1", 0],
      ["a2", 0],
      ["a3", 0],
      ["a4", 0],
      ["b1", 1],
      ["b2", 1],
      ["b3", 1],
      ["tiny", 2],
    ]);
    const regions = communityRegions(graph, communities);
    expect(regions).toHaveLength(2);
    expect(regions[0].name).toBe("Wenlan");
    expect(regions[0].memberCount).toBe(4);
    expect(regions[0].hull).toHaveLength(4);
    expect(regions[1].name).toBe("Claude Code");
    expect(regions[1].memberCount).toBe(3);
  });

  it("skips community members sigma no longer has a node for", () => {
    const graph = graphOf([
      { id: "a1", x: 0, y: 0, label: "A" },
      { id: "a2", x: 1, y: 0, label: "B" },
    ]);
    const communities = new Map<string, number>([
      ["a1", 0],
      ["a2", 0],
      ["ghost", 0],
    ]);
    // 3 mapped members but only 2 present -> below MIN_REGION_SIZE.
    expect(communityRegions(graph, communities)).toHaveLength(0);
  });
});

describe("graticuleRadii", () => {
  it("spaces three rings evenly out to 5% past the farthest node", () => {
    expect(graticuleRadii(300)).toEqual([105, 210, 315]);
  });

  it("draws nothing for an empty or single-origin graph", () => {
    expect(graticuleRadii(0)).toEqual([]);
    expect(graticuleRadii(-5)).toEqual([]);
  });
});

describe("cartographyScene", () => {
  it("measures the farthest node from the origin", () => {
    const graph = graphOf([
      { id: "a", x: 3, y: 4, label: "A" },
      { id: "b", x: -1, y: 0, label: "B" },
    ]);
    const scene = cartographyScene(graph, new Map([["a", 0], ["b", 1]]));
    expect(scene.maxNodeRadius).toBe(5);
    expect(scene.regions).toHaveLength(0);
  });
});

interface StrokeCall {
  strokeStyle: string;
  lineWidth: number;
  dash: number[];
}

function mockCtx() {
  const strokes: StrokeCall[] = [];
  const fills: string[] = [];
  const texts: { text: string; x: number; y: number; font: string; fillStyle: string }[] = [];
  let dash: number[] = [];
  const ctx = {
    strokeStyle: "",
    fillStyle: "",
    lineWidth: 0,
    lineJoin: "",
    lineCap: "",
    font: "",
    letterSpacing: "",
    textAlign: "",
    textBaseline: "",
    save: vi.fn(),
    restore: vi.fn(),
    beginPath: vi.fn(),
    closePath: vi.fn(),
    moveTo: vi.fn(),
    lineTo: vi.fn(),
    bezierCurveTo: vi.fn(),
    arc: vi.fn(),
    setLineDash: vi.fn((d: number[]) => {
      dash = d;
    }),
    stroke: vi.fn(() => {
      strokes.push({ strokeStyle: ctx.strokeStyle, lineWidth: ctx.lineWidth, dash });
    }),
    fill: vi.fn(() => {
      fills.push(ctx.fillStyle);
    }),
    fillText: vi.fn((text: string, x: number, y: number) => {
      texts.push({ text, x, y, font: ctx.font, fillStyle: ctx.fillStyle });
    }),
  };
  return { ctx: ctx as unknown as CanvasRenderingContext2D, strokes, fills, texts };
}

describe("drawCartography", () => {
  const identity = (pos: { x: number; y: number }) => pos;

  function sceneWithRegions() {
    const graph = graphOf(
      [
        { id: "a1", x: 100, y: 100, label: "Wenlan" },
        { id: "a2", x: 200, y: 100, label: "Tauri" },
        { id: "a3", x: 150, y: 200, label: "React" },
      ],
      [
        ["a1", "a2"],
        ["a1", "a3"],
      ],
    );
    const communities = new Map<string, number>([
      ["a1", 0],
      ["a2", 0],
      ["a3", 0],
    ]);
    return cartographyScene(graph, communities);
  }

  it("draws three dashed graticule rings in the graticule ink", () => {
    const { ctx, strokes } = mockCtx();
    drawCartography(ctx, sceneWithRegions(), identity, PALETTE);
    const rings = strokes.filter((s) => s.dash.length === 2);
    expect(rings).toHaveLength(3);
    expect(rings[0]).toMatchObject({ strokeStyle: PALETTE.graticule, lineWidth: 1, dash: [1, 7] });
    // Ring radii come from the farthest node (|(150,200)| ≈ 250) — the rings
    // are the arcs centered on the projected origin (identity → (0,0));
    // hull-corner arcs sit at hull vertices and never at the origin here.
    const arc = ctx.arc as ReturnType<typeof vi.fn>;
    const rings2 = arc.mock.calls.filter((c) => c[0] === 0 && c[1] === 0);
    expect(rings2).toHaveLength(3);
    expect(rings2[2][2]).toBeCloseTo(Math.hypot(150, 200) * 1.05, 5);
  });

  it("paints each region as ONE smooth pad-expanded blob: a uniform wash fill plus a 1px border stroke", () => {
    const { ctx, strokes, fills } = mockCtx();
    drawCartography(ctx, sceneWithRegions(), identity, PALETTE);
    const hullStrokes = strokes.filter((s) => s.dash.length === 0);
    // Exactly one stroke — a second pass over the same translucent path is
    // the stacked-band bug the single expanded blob exists to avoid.
    expect(hullStrokes).toHaveLength(1);
    expect(hullStrokes[0]).toMatchObject({ strokeStyle: PALETTE.hullBorder, lineWidth: 1 });
    expect(fills).toEqual([PALETTE.hull]);
    // Continuously curving silhouette: one cubic Bézier per hull vertex, no
    // straight edge runs (lineTo) and no corner arcs.
    const bezier = ctx.bezierCurveTo as ReturnType<typeof vi.fn>;
    expect(bezier).toHaveBeenCalledTimes(3);
    expect(ctx.lineTo).not.toHaveBeenCalled();
    // The spline INTERPOLATES the expanded ring, so every curve endpoint
    // sits exactly HULL_PAD out from its raw hull vertex — the no-sag
    // clearance guarantee.
    const raw = [
      { x: 100, y: 100 },
      { x: 200, y: 100 },
      { x: 150, y: 200 },
    ];
    for (const call of bezier.mock.calls) {
      const end = { x: call[4], y: call[5] };
      const nearest = Math.min(...raw.map((v) => Math.hypot(end.x - v.x, end.y - v.y)));
      expect(nearest).toBeCloseTo(26, 6);
    }
  });

  it("names the dominant region in 16px italic serif muted ink above the hull", () => {
    const { ctx, texts } = mockCtx();
    void ctx;
    drawCartography(ctx, sceneWithRegions(), identity, PALETTE);
    expect(texts).toHaveLength(1);
    expect(texts[0].text).toBe("Wenlan");
    expect(texts[0].font).toBe("italic 500 16px Fraunces, Georgia, serif");
    expect(texts[0].fillStyle).toBe(PALETTE.labelMuted);
    // Above the raw hull's top edge (y=100) by pad 26 + 14 — the extra 6px
    // clears the smooth blob's outward bow between expanded vertices.
    expect(texts[0].y).toBe(100 - 26 - 14);
    // Centered on the hull's x centroid.
    expect(texts[0].x).toBe(150);
  });

  it("drops to 13px for secondary regions", () => {
    const graph = graphOf(
      [
        { id: "a1", x: 0, y: 0, label: "Alpha" },
        { id: "a2", x: 10, y: 0, label: "A2" },
        { id: "a3", x: 5, y: 10, label: "A3" },
        { id: "a4", x: 5, y: 5, label: "A4" },
        { id: "b1", x: 100, y: 100, label: "Beta" },
        { id: "b2", x: 110, y: 100, label: "B2" },
        { id: "b3", x: 105, y: 110, label: "B3" },
      ],
      [
        ["a1", "a2"],
        ["a1", "a3"],
        ["a1", "a4"],
        ["b1", "b2"],
        ["b1", "b3"],
      ],
    );
    const communities = new Map<string, number>([
      ["a1", 0],
      ["a2", 0],
      ["a3", 0],
      ["a4", 0],
      ["b1", 1],
      ["b2", 1],
      ["b3", 1],
    ]);
    const { ctx, texts } = mockCtx();
    drawCartography(ctx, cartographyScene(graph, communities), identity, PALETTE);
    expect(texts.map((t) => t.text)).toEqual(["Alpha", "Beta"]);
    expect(texts[0].font).toContain("16px");
    expect(texts[1].font).toContain("13px");
  });

  it("scales graticule radii by the projection's px-per-unit", () => {
    const { ctx } = mockCtx();
    const half = (pos: { x: number; y: number }) => ({ x: pos.x / 2, y: pos.y / 2 });
    const graph = graphOf([{ id: "a", x: 300, y: 0, label: "A" }]);
    drawCartography(ctx, cartographyScene(graph, new Map([["a", 0]])), half, PALETTE);
    const arc = ctx.arc as ReturnType<typeof vi.fn>;
    // maxNodeRadius 300 → outer ring 315 graph units → 157.5 at half scale.
    expect(arc.mock.calls[2][2]).toBeCloseTo(157.5, 5);
  });
});
