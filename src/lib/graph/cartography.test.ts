// SPDX-License-Identifier: AGPL-3.0-only
import { describe, it, expect, vi } from "vitest";
import Graph from "graphology";
import type { GraphModel, GraphNode, GraphEdge } from "./model";
import type { GraphPalette } from "./palette";
import type { SpaceCartography } from "./community";
import {
  communitiesFor,
  communityRegions,
  cartographyScene,
  drawRegionNames,
  placeRegionLabels,
  MAX_REGION_LABELS,
  MIN_LABELLED_SPAN_PX,
  REGION_NAME_LIFT_MIN_PX,
} from "./cartography";
import type { CartographyScene, Region } from "./cartography";

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
  graticule: "rgba(4,5,6,0.13)",
  bridge: "#bbbbbb",
  memory: "#cccccc",
  page: "#cccccc",
};

function node(id: string, overrides: Partial<GraphNode> = {}): GraphNode {
  return {
    id,
    kind: "entity",
    name: overrides.name ?? id,
    entityType: "concept",
    confirmed: true,
    degree: 0,
    space: "space" in overrides ? (overrides.space as string | null) : null,
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
  const noCartography = new Map<string, SpaceCartography>();

  it("splits two hub-bridged triangles into two communities instead of leaking across the bridge", () => {
    const communities = communitiesFor(twoTriangles(), noCartography);
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
    const communities = communitiesFor(m, noCartography);
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
    const communities = communitiesFor(m, noCartography);
    expect(communities.get("shared")).toBe(communities.get("hub"));
  });

  it("uses the daemon's community ids verbatim for a ready space, never falling back to the climb", () => {
    const m = modelOf(
      [
        node("x1", { space: "Work" }),
        node("x2", { space: "Work" }),
        node("x3", { space: "Work" }),
        node("u1", { space: "Work" }),
        node("u2", { space: "Work" }),
      ],
      // Heavily connected — a fallback climb would group these; the ready
      // daemon read must win regardless of edge shape.
      [edge("e1", "u1", "u2"), edge("e2", "u1", "x1")],
    );
    const cartographyBySpace = new Map<string, SpaceCartography>([
      [
        "Work",
        {
          status: "ready",
          memberCommunityId: new Map([
            ["x1", "c7"],
            ["x2", "c7"],
            ["x3", "c9"],
          ]),
        },
      ],
    ]);
    const communities = communitiesFor(m, cartographyBySpace);
    expect(communities.get("x1")).toBe("durable:sWork:c7");
    expect(communities.get("x2")).toBe("durable:sWork:c7");
    expect(communities.get("x3")).toBe("durable:sWork:c9");
    // A member the daemon's read never covered gets its own singleton under
    // the separate `unassigned:` family (never nested under `durable:` —
    // see communitiesFor's collision note).
    expect(communities.get("u1")).toBe("unassigned:sWork:u1");
    expect(communities.get("u2")).toBe("unassigned:sWork:u2");
    expect(communities.get("u1")).not.toBe(communities.get("u2"));
  });

  it("keeps a not-ready space on the client-side fallback climb, namespaced by space", () => {
    const m = modelOf(
      [node("a1", { space: "Work" }), node("a2", { space: "Work" }), node("a3", { space: "Work" })],
      [edge("e1", "a1", "a2"), edge("e2", "a2", "a3"), edge("e3", "a3", "a1")],
    );
    const cartographyBySpace = new Map<string, SpaceCartography>([["Work", { status: "fallback" }]]);
    const communities = communitiesFor(m, cartographyBySpace);
    for (const id of ["a1", "a2", "a3"]) {
      expect(communities.get(id)).toMatch(/^fallback:sWork:/);
    }
  });

  it("resolves each space independently in one render: one ready, one still on the fallback climb", () => {
    const m = modelOf(
      [
        node("r1", { space: "Ready" }),
        node("r2", { space: "Ready" }),
        node("f1", { space: "Fallback" }),
        node("f2", { space: "Fallback" }),
        node("f3", { space: "Fallback" }),
      ],
      [edge("e1", "f1", "f2"), edge("e2", "f2", "f3"), edge("e3", "f3", "f1")],
    );
    const cartographyBySpace = new Map<string, SpaceCartography>([
      [
        "Ready",
        {
          status: "ready",
          memberCommunityId: new Map([
            ["r1", "c1"],
            ["r2", "c1"],
          ]),
        },
      ],
    ]);
    const communities = communitiesFor(m, cartographyBySpace);
    expect(communities.get("r1")).toBe("durable:sReady:c1");
    expect(communities.get("r2")).toBe("durable:sReady:c1");
    expect(communities.get("f1")).toMatch(/^fallback:sFallback:/);
  });

  it("never merges two spaces' fallback climbs even when both land on the same local index and an edge crosses the boundary", () => {
    const m = modelOf(
      [
        node("hubA", { space: "Alpha" }),
        node("sA1", { space: "Alpha" }),
        node("sA2", { space: "Alpha" }),
        node("hubB", { space: "Beta" }),
        node("sB1", { space: "Beta" }),
        node("sB2", { space: "Beta" }),
      ],
      [
        edge("ea1", "hubA", "sA1"),
        edge("ea2", "hubA", "sA2"),
        edge("eb1", "hubB", "sB1"),
        edge("eb2", "hubB", "sB2"),
        // Crosses the space boundary — must not factor into either climb.
        edge("cross", "hubA", "hubB"),
      ],
    );
    const communities = communitiesFor(m, noCartography);
    expect(communities.get("hubA")).toBe("fallback:sAlpha:0");
    expect(communities.get("sA1")).toBe("fallback:sAlpha:0");
    expect(communities.get("hubB")).toBe("fallback:sBeta:0");
    expect(communities.get("hubA")).not.toBe(communities.get("hubB"));
  });

  it("never collides a daemon community id containing 'unassigned:' with the generated unassigned-singleton id", () => {
    const m = modelOf([node("e1", { space: "Work" }), node("e2", { space: "Work" })], []);
    const cartographyBySpace = new Map<string, SpaceCartography>([
      [
        "Work",
        {
          status: "ready",
          // e1 carries a daemon-assigned id that happens to spell out the
          // OLD unassigned marker; e2's own read never covered it, so it
          // gets the generated unassigned singleton — the two must never
          // resolve to the same string.
          memberCommunityId: new Map([["e1", "unassigned:e2"]]),
        },
      ],
    ]);
    const communities = communitiesFor(m, cartographyBySpace);
    expect(communities.get("e1")).not.toBe(communities.get("e2"));
  });

  it("never collides two different (space, communityId) pairs that concatenate to the same string", () => {
    const withPair = (space: string, communityId: string) => {
      const m = modelOf([node("x", { space })], []);
      const cartographyBySpace = new Map<string, SpaceCartography>([
        [space, { status: "ready", memberCommunityId: new Map([["x", communityId]]) }],
      ]);
      return communitiesFor(m, cartographyBySpace).get("x");
    };
    // ("a", "b:c") and ("a:b", "c") concatenate to the identical raw string
    // under an unescaped join — they must resolve to distinct ids.
    expect(withPair("a", "b:c")).not.toBe(withPair("a:b", "c"));
  });

  it("never lets a daemon's empty-string-space cartography reach the unscoped bucket", () => {
    const m = modelOf(
      [
        node("a1", { space: "Work" }),
        node("a2", { space: "Work" }),
        node("a3", { space: "Work" }),
        node("n1", { space: null }),
      ],
      [edge("e1", "a1", "a2"), edge("e2", "a2", "a3"), edge("e3", "a3", "a1")],
    );
    // Even if a daemon reported cartography under the literal empty string
    // (the space GraphNode.space can never actually equal, but the sentinel
    // must be immune to it regardless), a null-space node must never pick
    // it up as if it were durable.
    const cartographyBySpace = new Map<string, SpaceCartography>([
      ["Work", { status: "ready", memberCommunityId: new Map([["a1", "c1"], ["a2", "c1"], ["a3", "c1"]]) }],
      ["", { status: "ready", memberCommunityId: new Map([["n1", "c9"]]) }],
    ]);
    const communities = communitiesFor(m, cartographyBySpace);
    expect(communities.get("a1")).toMatch(/^durable:/);
    expect(communities.get("n1")).toMatch(/^fallback:/);
    expect(communities.get("n1")).not.toBe(communities.get("a1"));
  });

  // The daemon does not forbid NUL-bearing space names (create_space rejects
  // only its own UUID sentinel), so ANY in-band magic value standing for "no
  // space" is forgeable by a real space literally named it. "No space" is
  // therefore represented out of band — see spaceSegment.
  const FORGED_SENTINEL = " unscoped";

  /** Two triangles joined by one cross edge: the first in `space`, the second
   *  unscoped (space null). */
  function scopedTriangleBesideUnscoped(space: string | null): GraphModel {
    return modelOf(
      [
        node("s1", { space }),
        node("s2", { space }),
        node("s3", { space }),
        node("n1", { space: null }),
        node("n2", { space: null }),
        node("n3", { space: null }),
      ],
      [
        edge("es1", "s1", "s2"),
        edge("es2", "s2", "s3"),
        edge("es3", "s3", "s1"),
        edge("en1", "n1", "n2"),
        edge("en2", "n2", "n3"),
        edge("en3", "n3", "n1"),
        edge("cross", "s1", "n1"),
      ],
    );
  }

  it("never bridges a real space named exactly the old no-space sentinel to the unscoped bucket", () => {
    const communities = communitiesFor(
      scopedTriangleBesideUnscoped(FORGED_SENTINEL),
      new Map<string, SpaceCartography>(),
    );
    expect(communities.get("s1")).not.toBe(communities.get("n1"));
    // Cross-space: the space segment (2nd id component) must differ too.
    expect(communities.get("s1")!.split(":")[1]).not.toBe(communities.get("n1")!.split(":")[1]);
  });

  it("still honors durable cartography for a space named exactly the old no-space sentinel", () => {
    const cartographyBySpace = new Map<string, SpaceCartography>([
      [
        FORGED_SENTINEL,
        {
          status: "ready",
          memberCommunityId: new Map([
            ["s1", "c1"],
            ["s2", "c1"],
            ["s3", "c1"],
          ]),
        },
      ],
    ]);
    const communities = communitiesFor(
      scopedTriangleBesideUnscoped(FORGED_SENTINEL),
      cartographyBySpace,
    );
    // The real space keeps its daemon assignment; the unscoped nodes stay on
    // the client climb and never pick that assignment up.
    expect(communities.get("s1")).toMatch(/^durable:/);
    expect(communities.get("s1")).toBe(communities.get("s3"));
    expect(communities.get("n1")).toMatch(/^fallback:/);
    expect(communities.get("s1")!.split(":")[1]).not.toBe(communities.get("n1")!.split(":")[1]);
  });

  it("pools null-space and empty-string-space nodes into ONE unscoped bucket that may bridge itself", () => {
    const m = modelOf(
      [
        node("p1", { space: null }),
        node("p2", { space: null }),
        node("p3", { space: null }),
        node("q1", { space: "" }),
        node("q2", { space: "" }),
        node("q3", { space: "" }),
      ],
      [
        edge("ep1", "p1", "p2"),
        edge("ep2", "p2", "p3"),
        edge("ep3", "p3", "p1"),
        edge("eq1", "q1", "q2"),
        edge("eq2", "q2", "q3"),
        edge("eq3", "q3", "q1"),
        edge("cross", "p1", "q1"),
      ],
    );
    const communities = communitiesFor(m, new Map<string, SpaceCartography>());
    // One bucket: different communities but the SAME space segment, so two
    // unscoped regions CAN bridge each other (unlike a true cross-space pair).
    expect(communities.get("p1")).not.toBe(communities.get("q1"));
    expect(communities.get("p1")!.split(":")[1]).toBe(communities.get("q1")!.split(":")[1]);
  });

  function memoryNode(id: string, space: string | null = null): GraphNode {
    return { ...node(id), kind: "memory", entityType: "memory", space };
  }

  function mentions(id: string, memory: string, entity: string): GraphEdge {
    return { id, source: memory, target: entity, type: "mentions", confidence: null, createdAt: 100 };
  }

  function pageNode(id: string, space: string | null = null): GraphNode {
    return { ...node(id), kind: "page", entityType: "page", space };
  }

  function about(id: string, page: string, entity: string): GraphEdge {
    return { id, source: page, target: entity, type: "about", confidence: null, createdAt: 100 };
  }

  function wikilink(id: string, from: string, to: string): GraphEdge {
    return { id, source: from, target: to, type: "wikilink", confidence: null, createdAt: 100 };
  }

  it("gives a wiki page the community of the entity it links to", () => {
    const m = modelOf(
      [node("a1", { space: "Work" }), node("a2", { space: "Work" }), pageNode("page:p1", "Work")],
      [edge("ea1", "a1", "a2"), about("ab1", "page:p1", "a2")],
    );
    const communities = communitiesFor(m, new Map<string, SpaceCartography>());
    expect(communities.get("page:p1")).toBe(communities.get("a2"));
  });

  it("propagates a community from page to page when a page links no entity", () => {
    // p1 is anchored to the region through a2; p2 only wikilinks p1.
    const m = modelOf(
      [
        node("a1", { space: "Work" }),
        node("a2", { space: "Work" }),
        pageNode("page:p1", "Work"),
        pageNode("page:p2", "Work"),
      ],
      [
        edge("ea1", "a1", "a2"),
        about("ab1", "page:p1", "a2"),
        wikilink("wl1", "page:p2", "page:p1"),
      ],
    );
    const communities = communitiesFor(m, new Map<string, SpaceCartography>());
    expect(communities.get("page:p2")).toBe(communities.get("a2"));
  });

  it("leaves a page-only map with no regions rather than inventing them", () => {
    // Entities toggled off: nothing to inherit from, so no page is assigned
    // and the region count is honestly zero.
    const m = modelOf(
      [pageNode("page:p1", "Work"), pageNode("page:p2", "Work")],
      [wikilink("wl1", "page:p1", "page:p2")],
    );
    const communities = communitiesFor(m, new Map<string, SpaceCartography>());
    expect(communities.size).toBe(0);
  });

  it("never drags a scoped entity into the unscoped bucket via a spaceless page", () => {
    const m = modelOf(
      [node("a1", { space: "Work" }), node("a2", { space: "Work" }), pageNode("page:p1", null)],
      [edge("ea1", "a1", "a2"), about("ab1", "page:p1", "a1")],
    );
    const communities = communitiesFor(m, new Map<string, SpaceCartography>());
    for (const id of ["a1", "a2", "page:p1"]) {
      expect(communities.get(id)!.split(":")[1]).not.toBe("u");
    }
  });

  it("gives a memory the community of the entity it links to", () => {
    const m = modelOf(
      [node("a1", { space: "Work" }), node("a2", { space: "Work" }), memoryNode("mem:m1")],
      [edge("ea1", "a1", "a2"), mentions("lm1", "mem:m1", "a2")],
    );
    const communities = communitiesFor(m, new Map<string, SpaceCartography>());
    expect(communities.get("mem:m1")).toBe(communities.get("a2"));
  });

  it("breaks a multi-entity tie on the lowest entity id, deterministically", () => {
    // Two disjoint Work regions; the memory touches one node in each, so the
    // answer is only stable if the tie-break is by id rather than edge order.
    const m = modelOf(
      [
        node("a1", { space: "Work" }),
        node("a2", { space: "Work" }),
        node("b1", { space: "Work" }),
        node("b2", { space: "Work" }),
        memoryNode("mem:m1"),
      ],
      [
        edge("ea", "a1", "a2"),
        edge("eb", "b1", "b2"),
        mentions("lm1", "mem:m1", "b1"),
        mentions("lm2", "mem:m1", "a1"),
      ],
    );
    const communities = communitiesFor(m, new Map<string, SpaceCartography>());
    expect(communities.get("mem:m1")).toBe(communities.get("a1"));
    expect(communities.get("mem:m1")).not.toBe(communities.get("b1"));
  });

  it("never drags a scoped entity into the unscoped bucket via a spaceless memory", () => {
    const m = modelOf(
      [node("a1", { space: "Work" }), node("a2", { space: "Work" }), memoryNode("mem:m1", null)],
      [edge("ea1", "a1", "a2"), mentions("lm1", "mem:m1", "a1")],
    );
    const communities = communitiesFor(m, new Map<string, SpaceCartography>());
    // "u" is the unscoped segment (see communitiesFor's spaceSegment).
    for (const id of ["a1", "a2", "mem:m1"]) {
      expect(communities.get(id)!.split(":")[1]).not.toBe("u");
    }
  });

  it("inherits a DURABLE community too, without ever being a durable member", () => {
    const cartography = new Map<string, SpaceCartography>([
      [
        "Work",
        {
          status: "ready",
          memberCommunityId: new Map([
            ["a1", "c1"],
            ["a2", "c1"],
          ]),
        } as SpaceCartography,
      ],
    ]);
    const m = modelOf(
      [node("a1", { space: "Work" }), node("a2", { space: "Work" }), memoryNode("mem:m1", "Work")],
      [edge("ea1", "a1", "a2"), mentions("lm1", "mem:m1", "a1")],
    );
    const communities = communitiesFor(m, cartography);
    expect(communities.get("a1")).toContain("durable:");
    expect(communities.get("mem:m1")).toBe(communities.get("a1"));
  });

  it("leaves a memory out of every region when its entities are not partitioned", () => {
    const m = modelOf(
      [memoryNode("mem:m1"), memoryNode("mem:m2")],
      [mentions("lm1", "mem:m1", "mem:m2")],
    );
    const communities = communitiesFor(m, new Map<string, SpaceCartography>());
    expect(communities.has("mem:m1")).toBe(false);
    expect(communities.has("mem:m2")).toBe(false);
  });
});

function graphOf(
  nodes: { id: string; x: number; y: number; label: string; entityType?: string }[],
  edges: [string, string][] = [],
): Graph {
  const graph = new Graph({ multi: true });
  for (const n of nodes) {
    graph.addNode(n.id, { x: n.x, y: n.y, label: n.label, ...(n.entityType ? { entityType: n.entityType } : {}) });
  }
  edges.forEach(([source, target], i) => graph.addEdgeWithKey(`e${i}`, source, target));
  return graph;
}

describe("communityRegions", () => {
  it("measures communities of 3+, names them after the highest-degree member, and sorts largest-first", () => {
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
    const communities = new Map<string, string>([
      ["a1", "0"],
      ["a2", "0"],
      ["a3", "0"],
      ["a4", "0"],
      ["b1", "1"],
      ["b2", "1"],
      ["b3", "1"],
      ["tiny", "2"],
    ]);
    const regions = communityRegions(graph, communities);
    expect(regions).toHaveLength(2);
    expect(regions[0].name).toBe("Wenlan");
    expect(regions[0].memberCount).toBe(4);
    expect(regions[0].centroid).toEqual({ x: 5, y: 5 });
    expect(regions[0].bounds).toEqual({ minX: 0, maxX: 10, minY: 0, maxY: 10 });
    expect(regions[1].name).toBe("Claude Code");
    expect(regions[1].memberCount).toBe(3);
  });

  it("skips community members sigma no longer has a node for", () => {
    const graph = graphOf([
      { id: "a1", x: 0, y: 0, label: "A" },
      { id: "a2", x: 1, y: 0, label: "B" },
    ]);
    const communities = new Map<string, string>([
      ["a1", "0"],
      ["a2", "0"],
      ["ghost", "0"],
    ]);
    // 3 mapped members but only 2 present -> below MIN_REGION_SIZE.
    expect(communityRegions(graph, communities)).toHaveLength(0);
  });
});

describe("cartographyScene", () => {
  it("is the named regions and nothing else — no terrain points, nothing to paint under the nodes", () => {
    const graph = graphOf([
      { id: "a", x: 3, y: 4, label: "A" },
      { id: "b", x: -1, y: 0, label: "B" },
      { id: "c", x: 0, y: 1, label: "C" },
      { id: "m", x: 7, y: 7, label: "M", entityType: "memory" },
    ]);
    const scene = cartographyScene(graph, new Map([["a", "0"], ["b", "0"], ["c", "0"]]));
    expect(Object.keys(scene)).toEqual(["regions"]);
    expect(scene.regions.map((r) => r.name)).toEqual(["A"]);
  });
});

function mockCtx() {
  const fills: string[] = [];
  const haloTexts: { text: string; x: number; y: number; strokeStyle: string; lineWidth: number; lineJoin: string }[] = [];
  const texts: { text: string; x: number; y: number; font: string; fillStyle: string; textBaseline: string }[] = [];
  /** "stroke:<name>" / "fill:<name>" in call order — lets a test assert the
   *  halo is drawn before the fill for each label. */
  const drawOrder: string[] = [];
  /** What the context looked like at each measureText call — the placement
   *  pass has to measure in the same font AND tracking it later draws with. */
  const measures: { text: string; font: string; letterSpacing: string }[] = [];
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
    arc: vi.fn(),
    setLineDash: vi.fn(),
    stroke: vi.fn(),
    fill: vi.fn(() => {
      fills.push(ctx.fillStyle);
    }),
    drawImage: vi.fn(),
    strokeText: vi.fn((text: string, x: number, y: number) => {
      haloTexts.push({ text, x, y, strokeStyle: ctx.strokeStyle, lineWidth: ctx.lineWidth, lineJoin: ctx.lineJoin });
      drawOrder.push(`stroke:${text}`);
    }),
    fillText: vi.fn((text: string, x: number, y: number) => {
      texts.push({ text, x, y, font: ctx.font, fillStyle: ctx.fillStyle, textBaseline: ctx.textBaseline });
      drawOrder.push(`fill:${text}`);
    }),
    // The real 2D context measures the label; jsdom's has no text engine, so
    // the mock reports a deterministic 7px per character at the label size.
    measureText: vi.fn((text: string) => {
      measures.push({ text, font: ctx.font, letterSpacing: ctx.letterSpacing });
      return { width: text.length * 7 };
    }),
  };
  return { ctx: ctx as unknown as CanvasRenderingContext2D, fills, haloTexts, texts, drawOrder, measures };
}

describe("drawRegionNames", () => {
  const identity = (pos: { x: number; y: number }) => pos;

  /** A region with no real hull — just enough to place a label: a centroid
   *  and a bounds box `spanX` wide (must clear MIN_LABELLED_SPAN_PX to earn
   *  a name). */
  function region(name: string, centroidX: number, centroidY: number, spanX = 200): Region {
    return {
      name,
      memberCount: 3,
      centroid: { x: centroidX, y: centroidY },
      bounds: {
        minX: centroidX - spanX / 2,
        maxX: centroidX + spanX / 2,
        minY: centroidY - 10,
        maxY: centroidY + 10,
      },
    };
  }

  it("draws the dominant region at 15px and the next at 12px, lifted above the projected centroid", () => {
    const { ctx, texts } = mockCtx();
    const scene: CartographyScene = {
      regions: [region("Wenlan", 100, 100), region("Tauri", 600, 300)],
    };
    drawRegionNames(ctx, scene, identity, PALETTE);
    expect(texts.map((t) => t.text)).toEqual(["Wenlan", "Tauri"]);
    expect(texts[0].font).toBe("italic 500 15px Fraunces, Georgia, serif");
    expect(texts[1].font).toBe("italic 500 12px Fraunces, Georgia, serif");
    expect(texts[0].fillStyle).toBe(PALETTE.labelMuted);
    // The fixtures are 20px tall, so the lift clamps to its minimum.
    expect(texts[0].x).toBe(100);
    expect(texts[0].y).toBe(100 - REGION_NAME_LIFT_MIN_PX);
    expect(texts[1].x).toBe(600);
    expect(texts[1].y).toBe(300 - REGION_NAME_LIFT_MIN_PX);
    expect(texts[0].textBaseline).toBe("middle");
  });

  it("strokes a surface-coloured halo before the fill, for every drawn name", () => {
    const { ctx, haloTexts, drawOrder } = mockCtx();
    const scene: CartographyScene = {
      regions: [region("Wenlan", 100, 100), region("Tauri", 600, 300)],
    };
    drawRegionNames(ctx, scene, identity, PALETTE);
    expect(haloTexts).toEqual([
      { text: "Wenlan", x: 100, y: 100 - REGION_NAME_LIFT_MIN_PX, strokeStyle: PALETTE.surface, lineWidth: 3, lineJoin: "round" },
      { text: "Tauri", x: 600, y: 300 - REGION_NAME_LIFT_MIN_PX, strokeStyle: PALETTE.surface, lineWidth: 3, lineJoin: "round" },
    ]);
    expect(drawOrder).toEqual(["stroke:Wenlan", "fill:Wenlan", "stroke:Tauri", "fill:Tauri"]);
  });
});

describe("placeRegionLabels", () => {
  const identity = (pos: { x: number; y: number }) => pos;
  // A fixed per-character rate keeps the overlap arithmetic in the tests exact.
  const measure = (text: string, size: number) => text.length * (size / 2);

  /** A `width`-wide, 20-tall region with its bounds' top-left at (x, y);
   *  the centroid sits at the box's own centre. */
  function boxRegion(name: string, x: number, y: number, width: number, members = 3): Region {
    return {
      name,
      memberCount: members,
      centroid: { x: x + width / 2, y: y + 10 },
      bounds: { minX: x, maxX: x + width, minY: y, maxY: y + 20 },
    };
  }

  const sceneOf = (regions: Region[]): CartographyScene => ({ regions });

  it("skips a region whose bounds are a speck on screen", () => {
    const wide = boxRegion("Wide", 0, 0, MIN_LABELLED_SPAN_PX);
    const speck = boxRegion("Speck", 0, 500, MIN_LABELLED_SPAN_PX - 1);
    const placed = placeRegionLabels(sceneOf([wide, speck]), identity, measure);
    expect(placed.map((label) => label.name)).toEqual(["Wide"]);
  });

  it("keeps the first of two regions whose label boxes would collide", () => {
    // Same x span, 6px apart vertically: the second box lands inside the
    // first one's 4px pad, so only the larger region keeps its name.
    const big = boxRegion("Big", 0, 100, 200, 9);
    const small = boxRegion("Small", 0, 106, 200, 4);
    const placed = placeRegionLabels(sceneOf([big, small]), identity, measure);
    expect(placed.map((label) => label.name)).toEqual(["Big"]);
  });

  it("draws both when the same two regions are pulled far enough apart", () => {
    const big = boxRegion("Big", 0, 100, 200, 9);
    const small = boxRegion("Small", 0, 400, 200, 4);
    const placed = placeRegionLabels(sceneOf([big, small]), identity, measure);
    expect(placed.map((label) => label.name)).toEqual(["Big", "Small"]);
    // First placed is the dominant one; every other label drops to 12px.
    expect(placed[0].size).toBe(15);
    expect(placed[1].size).toBe(12);
  });

  it("walks regions in the order they arrive, so the largest wins a contested spot", () => {
    // Region order is communityRegions' largest-first order; reversing the
    // input reverses which of a colliding pair survives.
    const a = boxRegion("A", 0, 100, 200, 9);
    const b = boxRegion("B", 0, 106, 200, 4);
    expect(placeRegionLabels(sceneOf([a, b]), identity, measure).map((l) => l.name)).toEqual(["A"]);
    expect(placeRegionLabels(sceneOf([b, a]), identity, measure).map((l) => l.name)).toEqual(["B"]);
  });

  it("never draws more than MAX_REGION_LABELS names, however many fit", () => {
    const regions = Array.from({ length: MAX_REGION_LABELS + 10 }, (_, i) =>
      boxRegion(`R${i}`, 0, i * 300, 200),
    );
    const placed = placeRegionLabels(sceneOf(regions), identity, measure);
    expect(placed).toHaveLength(MAX_REGION_LABELS);
    expect(placed[placed.length - 1].name).toBe(`R${MAX_REGION_LABELS - 1}`);
  });

  it("reads region width in SCREEN px, so zooming in reveals more names", () => {
    // 80 graph units wide: a speck at 1x, comfortably labelled at 2x (160px).
    const scene = sceneOf([boxRegion("Zoomed", 0, 0, 80)]);
    const doubled = (pos: { x: number; y: number }) => ({ x: pos.x * 2, y: pos.y * 2 });
    expect(placeRegionLabels(scene, identity, measure)).toHaveLength(0);
    expect(placeRegionLabels(scene, doubled, measure)).toHaveLength(1);
  });

  it("sets the name above the region's projected centroid, clear of the hub and its own label", () => {
    // 20px tall: the lift clamps to REGION_NAME_LIFT_MIN_PX.
    const placed = placeRegionLabels(sceneOf([boxRegion("Mid", 40, 200, 140)]), identity, measure);
    expect(placed[0].x).toBe(110);
    expect(placed[0].y).toBe(210 - REGION_NAME_LIFT_MIN_PX);
    // A tall region lifts by a share of its on-screen height instead.
    const tall: Region = {
      name: "Tall",
      memberCount: 5,
      centroid: { x: 100, y: 200 },
      bounds: { minX: 0, maxX: 200, minY: 100, maxY: 300 },
    };
    const [label] = placeRegionLabels(sceneOf([tall]), identity, measure);
    expect(label.y).toBe(200 - 0.3 * 200);
  });
});

describe("region label measurement includes its tracking", () => {
  const identity = (pos: { x: number; y: number }) => pos;

  /** Two regions side by side, each named with a 25-character name and each
   *  wide enough on screen to clear MIN_LABELLED_SPAN_PX. At a flat 7px per
   *  character the two label boxes clear each other; with the artifact's
   *  0.14em tracking added they overlap. So whether the second name survives
   *  says exactly which width the placement pass measured. `gap` is the
   *  distance between the two centroids. */
  function twoWideRegions(gap: number): CartographyScene {
    const region = (name: string, centroidX: number): Region => ({
      name,
      memberCount: 3,
      centroid: { x: centroidX, y: 0 },
      bounds: { minX: centroidX - 75, maxX: centroidX + 75, minY: -10, maxY: 10 },
    });
    return {
      regions: [region("Deterministic Fixture Set", 0), region("Reproducible Layout Notes", gap)],
    };
  }

  /** mockCtx reports a flat 7px per character; a real canvas adds the
   *  letterSpacing after every character, so the mock has to as well for the
   *  width to mean anything. */
  function trackingAwareCtx() {
    const harness = mockCtx();
    const ctx = harness.ctx as unknown as {
      letterSpacing: string;
      measureText: (text: string) => { width: number };
    };
    ctx.measureText = (text: string) => ({
      width: text.length * (7 + (parseFloat(ctx.letterSpacing) || 0)),
    });
    return harness;
  }

  it("sets the same letter spacing for measuring that it later draws with", () => {
    const { ctx, texts, measures } = mockCtx();
    drawRegionNames(ctx, twoWideRegions(200), identity, PALETTE);
    expect(measures.length).toBeGreaterThan(0);
    for (const call of measures) {
      // 15px label -> 2.1px, 12px label -> 1.7px. Never the 0 that measuring
      // in the bare font would leave behind.
      const size = Number(/(\d+)px/.exec(call.font)?.[1]);
      expect(call.letterSpacing).toBe(`${(size * 0.14).toFixed(1)}px`);
    }
    // Whatever survived placement is drawn with that same tracking.
    expect(texts.length).toBeGreaterThan(0);
  });

  it("drops a second name whose box only fits when the tracking is ignored", () => {
    const { ctx, texts } = trackingAwareCtx();
    drawRegionNames(ctx, twoWideRegions(200), identity, PALETTE);
    expect(texts.map((t) => t.text)).toEqual(["Deterministic Fixture Set"]);
  });

  it("keeps both names when they are far enough apart for the tracked width", () => {
    const { ctx, texts } = trackingAwareCtx();
    // Push the right region 200px further out: now even the tracked boxes
    // clear, so the thinning is about real width and not a blanket rule.
    drawRegionNames(ctx, twoWideRegions(400), identity, PALETTE);
    expect(texts.map((t) => t.text).sort()).toEqual([
      "Deterministic Fixture Set",
      "Reproducible Layout Notes",
    ]);
  });
});
