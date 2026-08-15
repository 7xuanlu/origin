// SPDX-License-Identifier: AGPL-3.0-only
import { describe, it, expect, vi } from "vitest";
import Graph from "graphology";
import type { GraphModel, GraphNode, GraphEdge } from "./model";
import type { GraphPalette } from "./palette";
import type { SpaceCartography } from "./community";
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
    expect(bridgeEdgeTest(communities)("s1", "n1")).toBe(false);
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
    expect(bridgeEdgeTest(communities)("s1", "n1")).toBe(false);
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
    // One bucket, so two unscoped regions CAN bridge each other.
    expect(communities.get("p1")).not.toBe(communities.get("q1"));
    expect(bridgeEdgeTest(communities)("p1", "q1")).toBe(true);
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

describe("bridgeEdgeTest", () => {
  it("flags only edges spanning two regions of at least 3 members", () => {
    const communities = communitiesFor(twoTriangles(), new Map<string, SpaceCartography>());
    const isBridge = bridgeEdgeTest(communities);
    expect(isBridge("a1", "b1")).toBe(true);
    expect(isBridge("a1", "a2")).toBe(false);
    expect(isBridge("b2", "b3")).toBe(false);
  });

  it("does not dress an edge into a sub-region islet as a bridge", () => {
    // Community 0 has 3 members, community 1 only 2.
    const communities = new Map<string, string>([
      ["r1", "0"],
      ["r2", "0"],
      ["r3", "0"],
      ["p1", "1"],
      ["p2", "1"],
    ]);
    const isBridge = bridgeEdgeTest(communities);
    expect(isBridge("r1", "p1")).toBe(false);
    expect(isBridge("r1", "missing")).toBe(false);
  });

  it("never bridges across a space boundary, even between two real regions of the required size", () => {
    // Two 3-member regions, one per space, joined by a cross-space edge —
    // namespacing alone makes their community ids differ (see
    // communitiesFor), so without an explicit same-space check this reads
    // as two distinct real regions and gets flagged a bridge.
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
        edge("cross", "hubA", "hubB"),
      ],
    );
    const communities = communitiesFor(m, new Map<string, SpaceCartography>());
    const isBridge = bridgeEdgeTest(communities);
    expect(isBridge("hubA", "hubB")).toBe(false);
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
    expect(regions[0].hull).toHaveLength(4);
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
    const scene = cartographyScene(graph, new Map([["a", "0"], ["b", "1"]]));
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
    const communities = new Map<string, string>([
      ["a1", "0"],
      ["a2", "0"],
      ["a3", "0"],
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
    const communities = new Map<string, string>([
      ["a1", "0"],
      ["a2", "0"],
      ["a3", "0"],
      ["a4", "0"],
      ["b1", "1"],
      ["b2", "1"],
      ["b3", "1"],
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
    drawCartography(ctx, cartographyScene(graph, new Map([["a", "0"]])), half, PALETTE);
    const arc = ctx.arc as ReturnType<typeof vi.fn>;
    // maxNodeRadius 300 → outer ring 315 graph units → 157.5 at half scale.
    expect(arc.mock.calls[2][2]).toBeCloseTo(157.5, 5);
  });
});
