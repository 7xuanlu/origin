// SPDX-License-Identifier: AGPL-3.0-only
import type Graph from "graphology";
import type { GraphModel } from "./model";
import type { GraphPalette } from "./palette";
import type { SpaceCartography } from "./community";

// The Atlas cartography layer (design-mockup artifact, screen 01 —
// "Cartography, not physics"): named community regions wrapped in translucent
// hulls, dashed orientation rings, and amber bridge edges between regions.
// Everything here is pure math / pure canvas drawing; AtlasView owns the
// underlay canvas and the sigma afterRender wiring.

/** Minimum members before a community earns a hull + name — a 1-2 node
 *  "region" is noise, not geography. */
export const MIN_REGION_SIZE = 3;

/** Screen-px padding between a member node and its hull edge. */
const HULL_PAD = 26;

function pushInto<K, V>(map: Map<K, V[]>, key: K, value: V): void {
  const list = map.get(key);
  if (list) list.push(value);
  else map.set(key, [value]);
}

/**
 * "No space" is represented OUT OF BAND, never as a magic string. The daemon
 * forbids only its own UUID sentinel in `create_space`, so ANY in-band value
 * — NUL-prefixed included — is forgeable by a real space literally named it,
 * and a forged one lands in the null bucket: a cross-space bridge, which
 * D13/App-PR forbids.
 *
 * The space component of every community id is therefore TAG-DISCRIMINATED:
 * a real space renders as `s` + encodeURIComponent(space), the unscoped
 * bucket as the bare token `u`. No real segment can spell `u` (they all start
 * with `s`), and encodeURIComponent is injective, so two segments compare
 * equal iff they came from the same space. Neither tag can contain an
 * unescaped ":", so the 3-segment split below still reads the space back
 * whole.
 */
const UNSCOPED_SEGMENT = "u";

/** Everything that lands in the unscoped bucket: null, undefined, and the
 *  empty string alike — an empty space string is no more a space than a
 *  missing one. AtlasView asks this same question of raw entities for the
 *  fallback badge, so the rule lives here rather than in two places. */
export function isUnscopedSpace(space: string | null | undefined): boolean {
  return !space;
}

function spaceSegment(space: string | null | undefined): string {
  return isUnscopedSpace(space) ? UNSCOPED_SEGMENT : `s${encodeURIComponent(space as string)}`;
}

/**
 * Steepest-ascent peak-climbing over ONE space's nodes: every node follows
 * its highest-degree same-space neighbor (strictly higher than its own
 * degree) upward until a local degree peak, and the peak is the community.
 * Deterministic (degree ties break on the smaller id), terminating (degree
 * strictly increases along a climb), and it can't leak across a hub–hub
 * bridge the way deterministic label propagation does: two equal-degree hubs
 * are each their own peak. ponytail: crude next to Louvain, but zero deps
 * and hub-shaped like this data. Returns node id -> a LOCAL (unqualified,
 * still colliding across spaces) community id — callers namespace it.
 */
function climbFallback(nodeIds: string[], edges: { source: string; target: string }[]): Map<string, string> {
  const idsInPartition = new Set(nodeIds);
  // Distinct-neighbor adjacency: parallel edges and self-loops must not
  // inflate the degree that drives the climb; edges leaving the partition
  // (a different space, or the unscoped bucket) are simply not adjacency.
  const adjacency = new Map<string, Set<string>>();
  for (const id of nodeIds) adjacency.set(id, new Set());
  for (const edge of edges) {
    if (edge.source === edge.target) continue;
    if (!idsInPartition.has(edge.source) || !idsInPartition.has(edge.target)) continue;
    adjacency.get(edge.source)?.add(edge.target);
    adjacency.get(edge.target)?.add(edge.source);
  }
  const degree = (id: string) => adjacency.get(id)?.size ?? 0;

  const stepUp = (id: string): string => {
    let best: string | null = null;
    for (const neighbor of adjacency.get(id) ?? []) {
      if (degree(neighbor) <= degree(id)) continue;
      if (
        best === null ||
        degree(neighbor) > degree(best) ||
        (degree(neighbor) === degree(best) && neighbor < best)
      ) {
        best = neighbor;
      }
    }
    return best ?? id;
  };

  const peakOf = new Map<string, string>();
  const climb = (id: string): string => {
    const cached = peakOf.get(id);
    if (cached !== undefined) return cached;
    const next = stepUp(id);
    const peak = next === id ? id : climb(next);
    peakOf.set(id, peak);
    return peak;
  };

  const peaks = [...new Set(nodeIds.map(climb))].sort();
  const peakIndex = new Map(peaks.map((peak, i) => [peak, i]));
  return new Map(nodeIds.map((id) => [id, String(peakIndex.get(peakOf.get(id)!))]));
}

/**
 * Node id -> opaque community id, partitioned per space (D13/App-PR: no
 * region or bridge edge may span spaces). Each space is handled
 * independently and resolves to exactly one of three id families, never
 * more than one per space:
 *
 * - READY (`cartographyBySpace.get(space)?.status === "ready"`): the
 *   daemon's own community_id, namespaced `durable:<spaceSegment>:
 *   <enc(community_id)>`.
 * - a member the daemon's read didn't cover gets its own singleton under a
 *   SEPARATE top-level family, `unassigned:<spaceSegment>:<enc(node id)>` —
 *   never nested under `durable:`, so a daemon-assigned community_id that
 *   happens to spell out the literal string "unassigned:<id>" can't forge
 *   one (see the collision note below).
 * - anything else (not ready — no durable data published yet, or a
 *   partial-error): the client-side fallback climb, namespaced
 *   `fallback:<spaceSegment>:<enc(local id)>`.
 *
 * The space component is the tag-discriminated segment above (`s<enc>` or
 * `u`); community_id and node id are encodeURIComponent-escaped before
 * joining. That does two things:
 *   1. Collision-proofs the join itself — encodeURIComponent never leaves a
 *      literal ":" unescaped, so `<prefix>:<encSpace>:<encRest>` always
 *      splits back into exactly 3 parts; two different (space, id) pairs
 *      can never concatenate to the same raw string the way an unescaped
 *      join would (`("a","b:c")` vs `("a:b","c")`).
 *   2. Keeps the three top-level prefixes (`durable:`, `unassigned:`,
 *      `fallback:`) reserved: since escaped, space/id components can never
 *      contain an unescaped ":", no daemon-controlled string can forge a
 *      different family's prefix.
 * Together, two different spaces' — or families' — community/local ids can
 * never compare equal, so bridgeEdgeTest and communityRegions can't
 * accidentally group members across a space boundary purely from id
 * collision (bridgeEdgeTest also checks the space explicitly — see below —
 * since namespacing alone only prevents accidental MERGING, not a
 * cross-space pair from being flagged a bridge).
 */
export function communitiesFor(
  model: GraphModel,
  cartographyBySpace: Map<string, SpaceCartography>,
): Map<string, string> {
  // null keys the one unscoped bucket — an out-of-band key no space string
  // can occupy, so the partition itself is unforgeable, not just the ids.
  const bySpace = new Map<string | null, string[]>();
  // Memory nodes are never partitioned or climbed: they are not durable
  // community members, and a spaceless memory hanging off a scoped entity
  // would otherwise drag that entity's region into the unscoped bucket.
  // They inherit an entity's community below instead.
  const memoryNodeIds: string[] = [];
  for (const node of model.nodes) {
    if (node.kind === "memory") {
      memoryNodeIds.push(node.id);
      continue;
    }
    pushInto(bySpace, isUnscopedSpace(node.space) ? null : node.space, node.id);
  }

  const result = new Map<string, string>();
  for (const [space, nodeIds] of bySpace) {
    const segment = spaceSegment(space);
    const cartography = space !== null ? cartographyBySpace.get(space) : undefined;
    if (cartography?.status === "ready" && cartography.memberCommunityId) {
      for (const id of nodeIds) {
        const communityId = cartography.memberCommunityId.get(id);
        result.set(
          id,
          communityId !== undefined
            ? `durable:${segment}:${encodeURIComponent(communityId)}`
            : `unassigned:${segment}:${encodeURIComponent(id)}`,
        );
      }
      continue;
    }
    const local = climbFallback(nodeIds, model.edges);
    for (const [id, localId] of local) {
      result.set(id, `fallback:${segment}:${encodeURIComponent(localId)}`);
    }
  }

  // A memory joins the region of the entity it links to. Ties break on the
  // lowest entity id so the answer is deterministic, and a memory whose
  // entities all landed outside the partition simply gets no community —
  // which keeps it out of every region size and every bridge test.
  if (memoryNodeIds.length > 0) {
    const memories = new Set(memoryNodeIds);
    const anchorOf = new Map<string, string>();
    for (const edge of model.edges) {
      const [memory, entity] = memories.has(edge.source)
        ? [edge.source, edge.target]
        : memories.has(edge.target)
          ? [edge.target, edge.source]
          : [null, null];
      if (memory === null || entity === null || memories.has(entity)) continue;
      const current = anchorOf.get(memory);
      if (current === undefined || entity < current) anchorOf.set(memory, entity);
    }
    for (const [memory, entity] of anchorOf) {
      const community = result.get(entity);
      if (community !== undefined) result.set(memory, community);
    }
  }
  return result;
}

/** Extract the tagged space segment from a namespaced community id
 *  (`durable:<segment>:…` / `fallback:<segment>:…` / `unassigned:<segment>:…`)
 *  — every family shares this exact 3-segment shape (see communitiesFor), so
 *  a single split index reads the space regardless of prefix. Compared as the
 *  tagged segment itself: `s<enc>` is injective in the space and `u` is
 *  unreachable from any real name, so two ids carry the same space iff their
 *  segments are equal, and unscoped only ever matches unscoped. */
function communitySpace(id: string): string {
  return id.split(":")[1] ?? "";
}

/**
 * Per-edge bridge test, sizes precomputed once. A bridge spans two DIFFERENT
 * communities, in the SAME space, that are both real regions
 * (>= MIN_REGION_SIZE members) — an edge poking out of a 2-node islet is
 * not map furniture, and a cross-space edge is never a bridge no matter how
 * large the two communities are (D13/App-PR: no bridge may span spaces).
 * Community ids are space-namespaced (see communitiesFor), which makes two
 * same-space communities compare unequal when they should, but a
 * cross-space PAIR also compares unequal — namespacing alone can't tell
 * "different community" from "different space", so the space check below
 * is required, not redundant.
 */
export function bridgeEdgeTest(
  communities: Map<string, string>,
): (source: string, target: string) => boolean {
  const sizes = new Map<string, number>();
  for (const community of communities.values()) {
    sizes.set(community, (sizes.get(community) ?? 0) + 1);
  }
  return (source, target) => {
    const a = communities.get(source);
    const b = communities.get(target);
    return (
      a !== undefined &&
      b !== undefined &&
      a !== b &&
      communitySpace(a) === communitySpace(b) &&
      (sizes.get(a) ?? 0) >= MIN_REGION_SIZE &&
      (sizes.get(b) ?? 0) >= MIN_REGION_SIZE
    );
  };
}

export interface Region {
  /** Convex hull of member GRAPH positions, counter-clockwise. */
  hull: { x: number; y: number }[];
  /** Highest-degree member's name — the region's label. */
  name: string;
  memberCount: number;
}

/**
 * Andrew's monotone chain convex hull. Returns the hull counter-clockwise
 * without repeating the first point; degenerate inputs (<3 points, collinear)
 * return what they can — the fat-stroke drawing below renders a 2-point hull
 * as a capsule, which is fine.
 */
export function convexHull(points: { x: number; y: number }[]): { x: number; y: number }[] {
  const pts = [...points].sort((a, b) => a.x - b.x || a.y - b.y);
  if (pts.length <= 2) return pts;
  const cross = (
    o: { x: number; y: number },
    a: { x: number; y: number },
    b: { x: number; y: number },
  ) => (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x);
  const lower: { x: number; y: number }[] = [];
  for (const p of pts) {
    while (lower.length >= 2 && cross(lower[lower.length - 2], lower[lower.length - 1], p) <= 0)
      lower.pop();
    lower.push(p);
  }
  const upper: { x: number; y: number }[] = [];
  for (let i = pts.length - 1; i >= 0; i--) {
    const p = pts[i];
    while (upper.length >= 2 && cross(upper[upper.length - 2], upper[upper.length - 1], p) <= 0)
      upper.pop();
    upper.push(p);
  }
  lower.pop();
  upper.pop();
  return lower.concat(upper);
}

/**
 * Regions worth naming: communities with >= MIN_REGION_SIZE members, hulled
 * over their CURRENT graph positions (so hulls flex live with a drag) and
 * named after their highest-degree member (degree ties break alphabetically,
 * then by id — deterministic). Sorted largest-first so the caller can give
 * the dominant region the bigger type.
 */
/** The member a region is named after: highest degree, ties broken by
 *  smaller name then smaller id — one rule shared by the drawn region labels
 *  and the insight rail so the two can never disagree. */
export function regionLeader<T extends { id: string; name: string; degree: number }>(
  members: T[],
): T {
  let hub = members[0];
  for (const m of members) {
    if (
      m.degree > hub.degree ||
      (m.degree === hub.degree && (m.name < hub.name || (m.name === hub.name && m.id < hub.id)))
    ) {
      hub = m;
    }
  }
  return hub;
}

export function communityRegions(graph: Graph, communities: Map<string, string>): Region[] {
  const members = new Map<string, string[]>();
  for (const [id, community] of communities) {
    if (!graph.hasNode(id)) continue;
    pushInto(members, community, id);
  }
  const regions: Region[] = [];
  for (const ids of members.values()) {
    if (ids.length < MIN_REGION_SIZE) continue;
    const hubId = regionLeader(
      ids.map((id) => ({
        id,
        name: graph.getNodeAttribute(id, "label") as string,
        degree: graph.degree(id),
      })),
    ).id;
    regions.push({
      hull: convexHull(
        ids.map((id) => ({
          x: graph.getNodeAttribute(id, "x") as number,
          y: graph.getNodeAttribute(id, "y") as number,
        })),
      ),
      name: graph.getNodeAttribute(hubId, "label") as string,
      memberCount: ids.length,
    });
  }
  return regions.sort((a, b) => b.memberCount - a.memberCount || (a.name < b.name ? -1 : 1));
}

/** The three dashed orientation rings: even steps out to just past the
 *  farthest node. Zero/negative extent -> no rings. */
export function graticuleRadii(maxNodeRadius: number): number[] {
  if (maxNodeRadius <= 0) return [];
  const outer = maxNodeRadius * 1.05;
  return [outer / 3, (outer * 2) / 3, outer];
}

export interface CartographyScene {
  regions: Region[];
  /** Graph-space distance of the farthest node from the origin (forceCenter
   *  pins the cluster there), for the graticule. */
  maxNodeRadius: number;
}

/** Everything drawCartography needs, computed once per paint from live state. */
export function cartographyScene(graph: Graph, communities: Map<string, string>): CartographyScene {
  let maxNodeRadius = 0;
  graph.forEachNode((_id, attrs) => {
    maxNodeRadius = Math.max(maxNodeRadius, Math.hypot(attrs.x as number, attrs.y as number));
  });
  return { regions: communityRegions(graph, communities), maxNodeRadius };
}

/**
 * Trace the hull as a smooth organic blob, the artifact's hull silhouette:
 * a closed Catmull-Rom spline through the hull vertices pushed outward by
 * `pad` along their corner bisectors, emitted as cubic Béziers. The spline
 * INTERPOLATES its points (unlike a midpoint-quadratic, which sags halfway
 * back toward the polygon at sparse corners), so every node keeps >= pad
 * clearance at the vertices while the segments between them bow gently
 * outward — continuously curving everywhere, no straight runs. One closed
 * path -> ONE translucent fill stays uniform (a two-pass fat stroke stacks
 * inks into a donut band). Degenerate hulls: 1 distinct point becomes a
 * circle, 2 a capsule.
 */
function traceSmoothHull(
  ctx: CanvasRenderingContext2D,
  hull: { x: number; y: number }[],
  pad: number,
): void {
  const pts = hull.filter(
    (p, i) =>
      i === 0 || Math.hypot(p.x - hull[i - 1].x, p.y - hull[i - 1].y) > 1e-6,
  );
  if (
    pts.length > 1 &&
    Math.hypot(pts[0].x - pts[pts.length - 1].x, pts[0].y - pts[pts.length - 1].y) <= 1e-6
  ) {
    pts.pop();
  }
  if (pts.length === 0) return;
  if (pts.length === 1) {
    ctx.moveTo(pts[0].x + pad, pts[0].y);
    ctx.arc(pts[0].x, pts[0].y, pad, 0, 2 * Math.PI);
    return;
  }
  if (pts.length === 2) {
    // Capsule: two half-circle caps joined by the (auto-drawn) side lines.
    const [a, b] = pts;
    const angle = Math.atan2(b.y - a.y, b.x - a.x);
    ctx.moveTo(a.x + pad * Math.cos(angle + Math.PI / 2), a.y + pad * Math.sin(angle + Math.PI / 2));
    ctx.arc(a.x, a.y, pad, angle + Math.PI / 2, angle - Math.PI / 2);
    ctx.arc(b.x, b.y, pad, angle - Math.PI / 2, angle + Math.PI / 2);
    return;
  }

  const n = pts.length;
  const cx = pts.reduce((s, p) => s + p.x, 0) / n;
  const cy = pts.reduce((s, p) => s + p.y, 0) / n;
  // Push each vertex out along its corner bisector (the two adjacent edge
  // normals averaged), signed away from the centroid — winding-proof
  // (projection flips graph-space CCW to screen CW).
  const expanded = pts.map((p, i) => {
    const prev = pts[(i + n - 1) % n];
    const next = pts[(i + 1) % n];
    const edgeNormal = (a: { x: number; y: number }, b: { x: number; y: number }) => {
      const len = Math.hypot(b.x - a.x, b.y - a.y);
      return { x: (b.y - a.y) / len, y: -(b.x - a.x) / len };
    };
    const n1 = edgeNormal(prev, p);
    const n2 = edgeNormal(p, next);
    let bx = n1.x + n2.x;
    let by = n1.y + n2.y;
    const blen = Math.hypot(bx, by);
    if (blen < 1e-6) {
      bx = p.x - cx;
      by = p.y - cy;
    }
    const len = Math.hypot(bx, by) || 1;
    bx /= len;
    by /= len;
    if (bx * (p.x - cx) + by * (p.y - cy) < 0) {
      bx = -bx;
      by = -by;
    }
    return { x: p.x + bx * pad, y: p.y + by * pad };
  });

  // Closed uniform Catmull-Rom through the expanded ring, as cubic Béziers
  // (the standard CR->Bézier handles: p1 + (p2-p0)/6 and p2 - (p3-p1)/6).
  ctx.moveTo(expanded[0].x, expanded[0].y);
  for (let i = 0; i < n; i++) {
    const p0 = expanded[(i + n - 1) % n];
    const p1 = expanded[i];
    const p2 = expanded[(i + 1) % n];
    const p3 = expanded[(i + 2) % n];
    ctx.bezierCurveTo(
      p1.x + (p2.x - p0.x) / 6,
      p1.y + (p2.y - p0.y) / 6,
      p2.x - (p3.x - p1.x) / 6,
      p2.y - (p3.y - p1.y) / 6,
      p2.x,
      p2.y,
    );
  }
}

/**
 * Paint the cartography underlay in VIEWPORT space. `project` maps graph
 * coords to viewport CSS px (AtlasView passes sigma's graphToViewport).
 * Draw order: graticule (deepest) -> hull blobs -> region names. Each hull
 * is a smooth Catmull-Rom blob around the pad-expanded vertices, filled
 * once with the translucent wash and stroked once with a 1px border — the
 * artifact's exact hull anatomy (fill --kg-hull, stroke --kg-hull-border,
 * stroke-width 1, continuously curving Q-spline silhouette).
 */
export function drawCartography(
  ctx: CanvasRenderingContext2D,
  scene: CartographyScene,
  project: (pos: { x: number; y: number }) => { x: number; y: number },
  palette: GraphPalette,
): void {
  const center = project({ x: 0, y: 0 });
  const edge = project({ x: scene.maxNodeRadius, y: 0 });
  const pxPerUnit =
    scene.maxNodeRadius > 0
      ? Math.hypot(edge.x - center.x, edge.y - center.y) / scene.maxNodeRadius
      : 0;

  ctx.save();
  ctx.strokeStyle = palette.graticule;
  ctx.lineWidth = 1;
  ctx.setLineDash([1, 7]);
  for (const radius of graticuleRadii(scene.maxNodeRadius)) {
    ctx.beginPath();
    ctx.arc(center.x, center.y, radius * pxPerUnit, 0, 2 * Math.PI);
    ctx.stroke();
  }
  ctx.setLineDash([]);

  for (const region of scene.regions) {
    const screenHull = region.hull.map(project);
    if (screenHull.length === 0) continue;
    ctx.beginPath();
    traceSmoothHull(ctx, screenHull, HULL_PAD);
    ctx.closePath();
    ctx.fillStyle = palette.hull;
    ctx.fill();
    ctx.strokeStyle = palette.hullBorder;
    ctx.lineWidth = 1;
    ctx.stroke();
  }

  // Region names last, above their hulls: italic serif with wide tracking
  // (the artifact's .region style), centered above the hull's top edge. The
  // dominant region gets 16px, the rest 13px (.region.minor).
  scene.regions.forEach((region, i) => {
    const screenHull = region.hull.map(project);
    if (screenHull.length === 0) return;
    const cx = screenHull.reduce((s, p) => s + p.x, 0) / screenHull.length;
    const top = Math.min(...screenHull.map((p) => p.y));
    const size = i === 0 ? 16 : 13;
    ctx.font = `italic 500 ${size}px Fraunces, Georgia, serif`;
    // Wide tracking is part of the artifact spec; jsdom's mock ctx simply
    // ignores the property.
    ctx.letterSpacing = `${(size * 0.14).toFixed(1)}px`;
    ctx.fillStyle = palette.labelMuted;
    ctx.textAlign = "center";
    ctx.textBaseline = "bottom";
    // The smooth blob can bow a touch past the raw hull top between two
    // expanded vertices, so the name gets pad + 14 of lift, not pad + 8.
    ctx.fillText(region.name, cx, top - HULL_PAD - 14);
    ctx.letterSpacing = "0px";
  });
  ctx.restore();
}
