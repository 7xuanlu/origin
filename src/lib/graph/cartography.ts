// SPDX-License-Identifier: AGPL-3.0-only
import type Graph from "graphology";
import type { GraphModel } from "./model";
import type { GraphPalette } from "./palette";
import type { SpaceCartography } from "./community";

// The Atlas cartography layer — place names, nothing else: named community
// regions get an italic name drawn above the nodes, and the map's shape comes
// from the nodes themselves. There is deliberately no terrain, wash, hull or
// halo under anything — every version of one (hulls, per-node washes, a
// density field) read as a shadow or an aura around the points. Everything
// here is pure math / pure canvas drawing; AtlasView owns the overlay canvas
// and the sigma afterRender wiring.

/** Minimum members before a community earns a name — a 1-2 node "region" is
 *  noise, not geography. */
export const MIN_REGION_SIZE = 3;

/** A region whose members span less than this on screen is a speck: its name
 *  would be noise rather than orientation, so it goes unnamed until you zoom
 *  in far enough for the span to cross the bar. */
export const MIN_LABELLED_SPAN_PX = 90;

/** A place name sits ABOVE its region's centroid, not on it: the centroid is
 *  where the hub node and its own label are, and a name on top of a name was
 *  the first thing the eye caught. The lift is a share of the region's
 *  on-screen height, clamped so a flat region still clears its hub and a
 *  tall one does not float off into the sea. */
export const REGION_NAME_LIFT_FRACTION = 0.3;
export const REGION_NAME_LIFT_MIN_PX = 18;
export const REGION_NAME_LIFT_MAX_PX = 60;

/** Hard ceiling on drawn region names per paint, at any zoom. Real data has
 *  66 regions with memories off and 148 with them on; at fit-to-screen a map
 *  needs a handful of place names, not a gazetteer. Counts only names that
 *  land in the viewport: zoomed in, a dozen big regions off to the side must
 *  not spend the cap and leave the regions in view nameless. */
export const MAX_REGION_LABELS = 12;
/** A name closer than this (CSS px) to a member that is NOT the region's hub
 *  reads as that node's caption. Such a name moves onto its hub instead. */
export const REGION_NAME_KEEP_OUT_PX = 36;

/** Clear air demanded around a candidate label before it may sit next to an
 *  already-placed one. */
const LABEL_BOX_PAD = 4;

/** The one region-label font (the artifact's `.region` style), sized per call
 *  — shared by the width measurement and the actual draw so the two agree. */
function regionLabelFont(size: number): string {
  return `italic 500 ${size}px Fraunces, Georgia, serif`;
}

/** The tracking that goes with that font. Wide tracking is part of the
 *  artifact's `.region` style and it WIDENS the text, so the placement pass
 *  has to measure with it applied — measuring without it made every accepted
 *  label box about 14% per character too narrow, which let two names the
 *  overlap test had cleared still collide on screen. */
function regionLabelTracking(size: number): string {
  return `${(size * 0.14).toFixed(1)}px`;
}

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
 * region may span spaces). Each space is handled
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
 * never compare equal, so communityRegions can't accidentally group members
 * across a space boundary purely from id collision.
 */
export function communitiesFor(
  model: GraphModel,
  cartographyBySpace: Map<string, SpaceCartography>,
): Map<string, string> {
  // null keys the one unscoped bucket — an out-of-band key no space string
  // can occupy, so the partition itself is unforgeable, not just the ids.
  const bySpace = new Map<string | null, string[]>();
  // Memory AND wiki-page nodes are never partitioned or climbed: neither is a
  // durable community member (the daemon assigns communities to entities
  // only), and a spaceless one hanging off a scoped entity would otherwise
  // drag that entity's region into the unscoped bucket. Both inherit a
  // community below instead.
  const inheritorIds: string[] = [];
  for (const node of model.nodes) {
    if (node.kind === "memory" || node.kind === "page") {
      inheritorIds.push(node.id);
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

  // A memory or page joins the region of the partitioned node (an entity) it
  // links to. Ties break on the lowest neighbor id so the answer is
  // deterministic, and one whose neighbors all landed outside the partition
  // simply gets no community — which keeps it out of every region size.
  //
  // Pages get a second chance the memories never needed: a page that links no
  // entity but wikilinks another page inherits from that page, the lowest-id
  // neighbor that has a community, propagated until nothing new resolves. In
  // the pages-only view (entities toggled off) nothing is anchored at all and
  // the map honestly draws no regions rather than inventing them.
  if (inheritorIds.length > 0) {
    const inheritors = new Set(inheritorIds);
    const anchorOf = new Map<string, string>();
    const peers = new Map<string, string[]>();
    for (const edge of model.edges) {
      const bothInherit = inheritors.has(edge.source) && inheritors.has(edge.target);
      if (bothInherit) {
        pushInto(peers, edge.source, edge.target);
        pushInto(peers, edge.target, edge.source);
        continue;
      }
      const [inheritor, anchor] = inheritors.has(edge.source)
        ? [edge.source, edge.target]
        : inheritors.has(edge.target)
          ? [edge.target, edge.source]
          : [null, null];
      if (inheritor === null || anchor === null) continue;
      const current = anchorOf.get(inheritor);
      if (current === undefined || anchor < current) anchorOf.set(inheritor, anchor);
    }
    for (const [inheritor, anchor] of anchorOf) {
      const community = result.get(anchor);
      if (community !== undefined) result.set(inheritor, community);
    }
    // Propagate between inheritors until nothing changes. Each pass can only
    // ADD assignments, and there are finitely many nodes, so it terminates.
    let changed = true;
    while (changed) {
      changed = false;
      for (const id of inheritorIds) {
        if (result.has(id)) continue;
        let best: string | null = null;
        for (const peer of peers.get(id) ?? []) {
          if (!result.has(peer)) continue;
          if (best === null || peer < best) best = peer;
        }
        if (best === null) continue;
        result.set(id, result.get(best)!);
        changed = true;
      }
    }
  }
  return result;
}

/** The member a region is named after: highest degree, ties broken by
 *  smaller name then smaller id — deterministic, so the drawn region labels
 *  never flicker between members. */
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

export interface Region {
  /** Highest-degree member's name — the region's label. */
  name: string;
  /** The member the region is named after: its GRAPH position and CSS-px
   *  radius. When sigma draws this node's own label the place is already
   *  named, so the region name yields; and when the lifted centroid would
   *  sit on some other member, the name sits above this node instead. */
  hub: { id: string; x: number; y: number; size: number };
  /** Every other member's GRAPH position and CSS-px radius — what the name
   *  keeps REGION_NAME_KEEP_OUT_PX clear of. */
  otherMembers: { x: number; y: number; size: number }[];
  memberCount: number;
  /** Mean of member GRAPH positions — where the name sits. */
  centroid: { x: number; y: number };
  /** Axis-aligned bounds of member GRAPH positions — how much of the screen
   *  the region spans decides whether its name is worth drawing. */
  bounds: { minX: number; maxX: number; minY: number; maxY: number };
  /** True when every member is an island node (see atlas.ts's
   *  shelveComponents) — the name is held back while the islands are dim. */
  island: boolean;
}

/**
 * Regions worth naming: communities with >= MIN_REGION_SIZE members (memory
 * dust excepted), measured over their CURRENT graph positions (so names
 * follow a drag) and named after their highest-degree member. Sorted largest-first so the caller can give
 * the dominant region the bigger type.
 */
export function communityRegions(graph: Graph, communities: Map<string, string>): Region[] {
  const members = new Map<string, string[]>();
  for (const [id, community] of communities) {
    if (!graph.hasNode(id)) continue;
    // A memory is dust on its anchor (atlas.ts's satellitePlan), not a
    // place: it neither counts toward a region nor widens one, so the names
    // on the map are the same with the memory layer on and off.
    if (graph.getNodeAttribute(id, "dustOf") !== undefined) continue;
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
    let sumX = 0;
    let sumY = 0;
    const bounds = { minX: Infinity, maxX: -Infinity, minY: Infinity, maxY: -Infinity };
    let island = true;
    const hub = { id: hubId, x: 0, y: 0, size: 0 };
    const otherMembers: Region["otherMembers"] = [];
    for (const id of ids) {
      const x = graph.getNodeAttribute(id, "x") as number;
      const y = graph.getNodeAttribute(id, "y") as number;
      // Node sizes are true CSS px at every zoom (AtlasView's
      // zoomToSizeRatioFunction), so a radius here is a screen radius.
      const size = (graph.getNodeAttribute(id, "size") as number | undefined) ?? 0;
      if (id === hubId) {
        hub.x = x;
        hub.y = y;
        hub.size = size;
      } else {
        otherMembers.push({ x, y, size });
      }
      if (graph.getNodeAttribute(id, "island") !== true) island = false;
      sumX += x;
      sumY += y;
      bounds.minX = Math.min(bounds.minX, x);
      bounds.maxX = Math.max(bounds.maxX, x);
      bounds.minY = Math.min(bounds.minY, y);
      bounds.maxY = Math.max(bounds.maxY, y);
    }
    regions.push({
      name: graph.getNodeAttribute(hubId, "label") as string,
      hub,
      otherMembers,
      memberCount: ids.length,
      centroid: { x: sumX / ids.length, y: sumY / ids.length },
      bounds,
      island,
    });
  }
  return regions.sort((a, b) => b.memberCount - a.memberCount || (a.name < b.name ? -1 : 1));
}

export interface CartographyScene {
  regions: Region[];
}

/** Everything the place-name paint needs, computed once per paint from live
 *  state (positions and communities). */
export function cartographyScene(graph: Graph, communities: Map<string, string>): CartographyScene {
  return { regions: communityRegions(graph, communities) };
}

/** One region name that earned its place on this paint, in viewport CSS px. */
export interface PlacedLabel {
  name: string;
  /** Horizontal centre of the text (drawn with textAlign "center"). */
  x: number;
  /** Vertical centre of the text (drawn with textBaseline "middle"). */
  y: number;
  /** Font size in px: 15 for the first label placed, 12 for every other. */
  size: number;
}

/**
 * Which region names actually get drawn, and where. Pure and injectable so it
 * can be unit-tested without a canvas: `project` maps graph coords to viewport
 * CSS px and `measure(text, size)` reports the text width the real 2D context
 * would report.
 *
 * Regions arrive largest-first (communityRegions sorts them) and are walked in
 * that order, so the biggest region always wins a contested spot. A name sits
 * above its region's centroid (see REGION_NAME_LIFT_*), like a place name
 * set on the land rather than on the town. A region is skipped when its
 * members span a speck on screen (narrower than MIN_LABELLED_SPAN_PX) or when
 * its label box would touch one already placed; placement stops at
 * MAX_REGION_LABELS. Because on-screen span grows with zoom, more names
 * appear as you approach. With a `viewport` (CSS px), a name whose box lies
 * wholly outside it is skipped without counting toward the cap.
 *
 * A region is named after its hub node, so the name never doubles that
 * node's own label: a hub in `labelledNodes` (the nodes sigma drew a label
 * for this paint) costs its region the name, again without counting toward
 * the cap. And a name that would land within REGION_NAME_KEEP_OUT_PX of a
 * member that is not the hub — where it reads as that node's caption —
 * sits just above the hub instead.
 */
export function placeRegionLabels(
  scene: CartographyScene,
  project: (pos: { x: number; y: number }) => { x: number; y: number },
  measure: (text: string, size: number) => number,
  viewport?: { width: number; height: number },
  labelledNodes?: ReadonlySet<string>,
): PlacedLabel[] {
  const placed: PlacedLabel[] = [];
  const boxes: { left: number; right: number; top: number; bottom: number }[] = [];
  for (const region of scene.regions) {
    if (placed.length >= MAX_REGION_LABELS) break;
    // The hub's own label is on screen: the place already has its name.
    if (labelledNodes?.has(region.hub.id)) continue;
    const { bounds } = region;
    // Project all four corners: the camera may rotate, so a graph-space width
    // is not a screen width.
    const corners = [
      project({ x: bounds.minX, y: bounds.minY }),
      project({ x: bounds.maxX, y: bounds.minY }),
      project({ x: bounds.minX, y: bounds.maxY }),
      project({ x: bounds.maxX, y: bounds.maxY }),
    ];
    const xs = corners.map((p) => p.x);
    const ys = corners.map((p) => p.y);
    if (Math.max(...xs) - Math.min(...xs) < MIN_LABELLED_SPAN_PX) continue;
    const centroid = project(region.centroid);
    const lift = Math.min(
      REGION_NAME_LIFT_MAX_PX,
      Math.max(REGION_NAME_LIFT_MIN_PX, REGION_NAME_LIFT_FRACTION * (Math.max(...ys) - Math.min(...ys))),
    );
    // Viewport y grows downward, so "above" is smaller y.
    let at = { x: centroid.x, y: centroid.y - lift };
    const size = placed.length === 0 ? 15 : 12;
    const halfWidth = measure(region.name, size) / 2;
    const boxAround = (p: { x: number; y: number }) => ({
      left: p.x - halfWidth,
      right: p.x + halfWidth,
      top: p.y - size / 2,
      bottom: p.y + size / 2,
    });
    let box = boxAround(at);
    // Beside some other member the name reads as that node's caption (the
    // region "sqlite" once sat right over the node "tally"). The name is the
    // hub's, so put it on the hub — just above its disc, like its own label.
    const captionsAnotherNode = region.otherMembers.some((member) => {
      const p = project(member);
      const reach = member.size + REGION_NAME_KEEP_OUT_PX;
      return (
        p.x - reach < box.right &&
        box.left < p.x + reach &&
        p.y - reach < box.bottom &&
        box.top < p.y + reach
      );
    });
    if (captionsAnotherNode) {
      const hub = project(region.hub);
      at = { x: hub.x, y: hub.y - region.hub.size - LABEL_BOX_PAD - size / 2 };
      box = boxAround(at);
    }
    if (
      viewport &&
      (box.right < 0 || box.left > viewport.width || box.bottom < 0 || box.top > viewport.height)
    ) {
      continue;
    }
    const overlaps = boxes.some(
      (other) =>
        other.left < box.right + LABEL_BOX_PAD &&
        box.left - LABEL_BOX_PAD < other.right &&
        other.top < box.bottom + LABEL_BOX_PAD &&
        box.top - LABEL_BOX_PAD < other.bottom,
    );
    if (overlaps) continue;
    boxes.push(box);
    placed.push({ name: region.name, x: at.x, y: at.y, size });
  }
  return placed;
}

/** Halo stroke around a place name, in CSS px — ground-coloured, so the name
 *  stays legible over the nodes it sits among. */
const LABEL_HALO_WIDTH = 3;

/**
 * Paint the region names ABOVE the nodes (AtlasView's overlay canvas), in
 * VIEWPORT space: italic serif with wide tracking (the artifact's .region
 * style), centred on the region's centroid, each with a ground-coloured halo
 * — only the ones that survive the placement pass (see placeRegionLabels).
 */
export function drawRegionNames(
  ctx: CanvasRenderingContext2D,
  scene: CartographyScene,
  project: (pos: { x: number; y: number }) => { x: number; y: number },
  palette: GraphPalette,
  viewport?: { width: number; height: number },
  labelledNodes?: ReadonlySet<string>,
): void {
  const measure = (text: string, size: number): number => {
    ctx.font = regionLabelFont(size);
    ctx.letterSpacing = regionLabelTracking(size);
    const width = ctx.measureText(text).width;
    ctx.letterSpacing = "0px";
    return width;
  };
  ctx.save();
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.lineJoin = "round";
  ctx.lineWidth = LABEL_HALO_WIDTH;
  ctx.strokeStyle = palette.surface;
  ctx.fillStyle = palette.labelMuted;
  for (const label of placeRegionLabels(scene, project, measure, viewport, labelledNodes)) {
    ctx.font = regionLabelFont(label.size);
    // Same tracking the measurement used; jsdom's mock ctx simply ignores it.
    ctx.letterSpacing = regionLabelTracking(label.size);
    ctx.strokeText(label.name, label.x, label.y);
    ctx.fillText(label.name, label.x, label.y);
    ctx.letterSpacing = "0px";
  }
  ctx.restore();
}
