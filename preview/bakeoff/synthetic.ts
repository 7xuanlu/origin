// SPDX-License-Identifier: AGPL-3.0-only
// Seeded synthetic GraphModel generator for the renderer bake-off (task #17).
// Same seed + same n must produce a byte-identical model — see synthetic.test.ts.
import type { GraphEdge, GraphModel, GraphNode } from "../../src/lib/graph/model";

// A GraphNode with a precomputed position. Every adapter can render this
// geometry directly when its own layout can't complete synchronously (the
// spec's fallback path) — it's a real GraphNode (imported type), just with
// x/y layered on, not a parallel shape.
export interface BakeoffNode extends GraphNode {
  x: number;
  y: number;
}

export interface BakeoffModel extends Omit<GraphModel, "nodes"> {
  nodes: BakeoffNode[];
}

// The 5 daemon entity-type strings the palette actually colors (see
// slotForEntityType in src/lib/graph/palette.ts) — anything else, including
// "page" below, resolves to the neutral slot.
const ENTITY_TYPES = ["project", "technology", "organization", "person", "concept"] as const;

const BASE_TIME = 1_700_000_000_000;

// Deterministic PRNG (mulberry32) — Math.random() would break the "same seed
// ⇒ same model" requirement.
function mulberry32(seed: number): () => number {
  let a = seed | 0;
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * N nodes across the 5 palette entity types plus ~10% "page" kind nodes,
 * grouped into ~sqrt(n) communities. Degree skew comes from intra-community
 * preferential attachment (each new member links to an existing one weighted
 * by current degree); ~2% bridge edges keep communities from becoming
 * disconnected islands. x/y are a deterministic circular layout, used as the
 * seed for sigma's forceAtlas2 and as every adapter's synchronous-layout
 * fallback.
 */
export function generateBakeoffGraph(n: number, seed = 1): BakeoffModel {
  const rand = mulberry32(seed);
  const communityCount = Math.max(1, Math.round(Math.sqrt(n)));
  const byCommunity: number[][] = Array.from({ length: communityCount }, () => []);

  const nodes: BakeoffNode[] = [];
  for (let i = 0; i < n; i++) {
    const communityId = i % communityCount;
    const isPage = rand() < 0.1;
    const entityType = isPage ? "page" : ENTITY_TYPES[Math.floor(rand() * ENTITY_TYPES.length)];
    const angle = rand() * Math.PI * 2;
    const radius = 50 + rand() * 400;
    nodes.push({
      id: `bo-${i}`,
      kind: isPage ? "page" : "entity",
      name: `${isPage ? "Page" : entityType} ${i}`,
      entityType,
      confirmed: null,
      degree: 0,
      communityId,
      createdAt: BASE_TIME - i * 60_000,
      updatedAt: BASE_TIME - i * 30_000,
      x: Math.cos(angle) * radius,
      y: Math.sin(angle) * radius,
    });
    byCommunity[communityId].push(i);
  }

  const edges: GraphEdge[] = [];
  let edgeSeq = 0;
  const addEdge = (a: number, b: number) => {
    if (a === b) return;
    edges.push({
      id: `bo-e-${edgeSeq++}`,
      source: nodes[a].id,
      target: nodes[b].id,
      type: "relates_to",
      confidence: null,
      createdAt: BASE_TIME,
    });
    nodes[a].degree += 1;
    nodes[b].degree += 1;
  };

  for (const members of byCommunity) {
    for (let k = 1; k < members.length; k++) {
      const candidates = members.slice(0, k);
      const weights = candidates.map((idx) => nodes[idx].degree + 1);
      const total = weights.reduce((sum, w) => sum + w, 0);
      let pick = rand() * total;
      let target = candidates[0];
      for (let c = 0; c < candidates.length; c++) {
        pick -= weights[c];
        if (pick <= 0) {
          target = candidates[c];
          break;
        }
      }
      addEdge(members[k], target);
    }
  }

  const bridgeCount = Math.max(1, Math.round(n * 0.02));
  for (let i = 0; i < bridgeCount; i++) {
    const a = Math.floor(rand() * n);
    const b = Math.floor(rand() * n);
    if (nodes[a].communityId === nodes[b].communityId) continue;
    addEdge(a, b);
  }

  return {
    nodes,
    edges,
    coverage: { relationsFetchedFor: n, totalEntities: n },
  };
}
