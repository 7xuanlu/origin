// SPDX-License-Identifier: AGPL-3.0-only
import type Graph from "graphology";
import { MEMORY_NODE_TYPE, type GraphModel } from "./model";
import type { GraphPalette } from "./palette";
import type { SpaceCartography } from "./community";

// The Atlas cartography layer — a relief map, not a diagram: the knowledge
// nodes (pages and entities) are summed into a density field, and only where
// that field is dense enough does terrain ink appear — land under the
// crowds, open sea around a lone point — with no outline anywhere. Named
// community regions sit on that terrain as place names drawn above the
// nodes. Everything here is pure math / pure canvas drawing; AtlasView owns
// the two helper canvases and the sigma afterRender wiring.

/** Minimum members before a community earns a name — a 1-2 node "region" is
 *  noise, not geography. */
export const MIN_REGION_SIZE = 3;

/** Radius of one node's density kernel in screen px. Screen-constant on
 *  purpose: zoomed out, neighbouring kernels sum into one landmass; zoomed
 *  in, the crowd spreads out, the sum falls under the land threshold and the
 *  map resolves into bare points — detail instead of blur. */
export const TERRAIN_RADIUS_PX = 34;

/** The density field is sampled at one cell per this many CSS px, then
 *  upscaled with bilinear smoothing onto the underlay. Coarse is the point:
 *  a 1440x900 viewport is a 360x225 field, cheap to read back and shade on
 *  every sigma frame, and the smoothing is what gives the coast a soft edge. */
export const TERRAIN_CELL_PX = 4;

/** One kernel's peak value in the field's alpha channel (0-1). Kernels are
 *  summed with the `lighter` blend, so a pixel's alpha is KERNEL_PEAK times
 *  the local node density until it saturates at 1 — four nodes deep. */
export const KERNEL_PEAK = 0.25;

/** Density (in nodes, KERNEL_PEAK units) where land starts and where it
 *  reaches full ink. LAND_MIN is above 1 on purpose: a lone node's kernel
 *  peaks at exactly 1, so it never shows — no halo, no drop shadow, no aura
 *  around a single point. Two nodes close together just break the surface;
 *  three or more within a kernel radius are solid land. */
export const LAND_MIN = 1.1;
export const LAND_FULL = 2.6;

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
 *  needs a handful of place names, not a gazetteer. */
export const MAX_REGION_LABELS = 12;

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
  memberCount: number;
  /** Mean of member GRAPH positions — where the name sits. */
  centroid: { x: number; y: number };
  /** Axis-aligned bounds of member GRAPH positions — how much of the screen
   *  the region spans decides whether its name is worth drawing. */
  bounds: { minX: number; maxX: number; minY: number; maxY: number };
}

/**
 * Regions worth naming: communities with >= MIN_REGION_SIZE members, measured
 * over their CURRENT graph positions (so names follow a drag) and named after
 * their highest-degree member. Sorted largest-first so the caller can give
 * the dominant region the bigger type.
 */
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
    let sumX = 0;
    let sumY = 0;
    const bounds = { minX: Infinity, maxX: -Infinity, minY: Infinity, maxY: -Infinity };
    for (const id of ids) {
      const x = graph.getNodeAttribute(id, "x") as number;
      const y = graph.getNodeAttribute(id, "y") as number;
      sumX += x;
      sumY += y;
      bounds.minX = Math.min(bounds.minX, x);
      bounds.maxX = Math.max(bounds.maxX, x);
      bounds.minY = Math.min(bounds.minY, y);
      bounds.maxY = Math.max(bounds.maxY, y);
    }
    regions.push({
      name: graph.getNodeAttribute(hubId, "label") as string,
      memberCount: ids.length,
      centroid: { x: sumX / ids.length, y: sumY / ids.length },
      bounds,
    });
  }
  return regions.sort((a, b) => b.memberCount - a.memberCount || (a.name < b.name ? -1 : 1));
}

export interface CartographyScene {
  regions: Region[];
  /** GRAPH positions of every drawn KNOWLEDGE node — pages and entities. Each
   *  one adds a kernel to the density field. Memories are left out: they are
   *  context hung around the subjects, and with ~2,000 of them on the map
   *  every crowd of memories around a page turned the field solid, so the
   *  whole core became one saturated blot instead of landmasses. */
  points: { x: number; y: number }[];
}

/** Everything the two cartography paints need, computed once per paint from
 *  live state. */
export function cartographyScene(graph: Graph, communities: Map<string, string>): CartographyScene {
  const points: { x: number; y: number }[] = [];
  graph.forEachNode((_id, attrs) => {
    if (attrs.entityType === MEMORY_NODE_TYPE) return;
    points.push({ x: attrs.x as number, y: attrs.y as number });
  });
  return { regions: communityRegions(graph, communities), points };
}

/**
 * The offscreen density field plus the one kernel stamped into it. Built
 * once per AtlasView mount (AtlasView keeps it across frames); null where
 * there is no 2D canvas to draw into (jsdom) — drawCartography then falls
 * back to a flat disc per point, the same picture at test granularity.
 */
export interface TerrainField {
  canvas: HTMLCanvasElement;
  ctx: CanvasRenderingContext2D;
  /** One node's kernel: white, alpha KERNEL_PEAK at the centre falling to 0
   *  at TERRAIN_RADIUS_PX, in FIELD cells. Stamped with the `lighter` blend
   *  so overlapping kernels sum in the alpha channel. */
  kernel: HTMLCanvasElement;
}

export function terrainField(): TerrainField | null {
  if (typeof document === "undefined") return null;
  const canvas = document.createElement("canvas");
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  if (!ctx || typeof ctx.createRadialGradient !== "function") return null;
  const kernel = document.createElement("canvas");
  const radius = TERRAIN_RADIUS_PX / TERRAIN_CELL_PX;
  const side = Math.ceil(radius * 2);
  kernel.width = side;
  kernel.height = side;
  const kctx = kernel.getContext("2d");
  if (!kctx) return null;
  const c = side / 2;
  // Roughly (1 - d/R)^2: the gradient is linear between stops, so a middle
  // stop at a quarter of the peak bends it toward the bell the sum wants.
  const gradient = kctx.createRadialGradient(c, c, 0, c, c, radius);
  gradient.addColorStop(0, `rgba(255, 255, 255, ${KERNEL_PEAK})`);
  gradient.addColorStop(0.5, `rgba(255, 255, 255, ${KERNEL_PEAK * 0.25})`);
  gradient.addColorStop(1, "rgba(255, 255, 255, 0)");
  kctx.fillStyle = gradient;
  kctx.fillRect(0, 0, side, side);
  return { canvas, ctx, kernel };
}

/** Parse a CSS `rgb()`/`rgba()` ink into channels; anything else (jsdom's
 *  empty computed styles, a hex swatch in a test) shades as opaque black so
 *  the failure is visible rather than silent. */
export function parseInk(ink: string): { r: number; g: number; b: number; a: number } {
  const m = /^rgba?\(\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)(?:\s*,\s*([\d.]+))?\s*\)/.exec(ink);
  if (!m) return { r: 0, g: 0, b: 0, a: 1 };
  return { r: Number(m[1]), g: Number(m[2]), b: Number(m[3]), a: m[4] === undefined ? 1 : Number(m[4]) };
}

/** Smooth ramp from 0 at LAND_MIN to 1 at LAND_FULL over the summed density
 *  (in nodes). Pure, exported for the tests: this IS the coastline. */
export function landCover(density: number): number {
  const t = Math.min(1, Math.max(0, (density - LAND_MIN) / (LAND_FULL - LAND_MIN)));
  return t * t * (3 - 2 * t);
}

/**
 * Turn the summed field (alpha = KERNEL_PEAK x density, white) into terrain
 * ink in place: every pixel becomes the ink's colour at the ink's alpha
 * times its land cover. Operates on the raw RGBA buffer so it can run on a
 * plain array in tests and on getImageData's buffer in the browser.
 */
export function shadeField(data: Uint8ClampedArray, ink: { r: number; g: number; b: number; a: number }): void {
  for (let i = 0; i < data.length; i += 4) {
    const density = data[i + 3]! / 255 / KERNEL_PEAK;
    const cover = density <= LAND_MIN ? 0 : landCover(density);
    data[i] = ink.r;
    data[i + 1] = ink.g;
    data[i + 2] = ink.b;
    data[i + 3] = Math.round(255 * ink.a * cover);
  }
}

/**
 * Paint the terrain underlay in VIEWPORT space. `project` maps graph coords
 * to viewport CSS px (AtlasView passes sigma's graphToViewport); `viewport`
 * is the canvas size in CSS px. Every knowledge node stamps one kernel into
 * the coarse density field; the field is then read back, shaded through
 * landCover (below LAND_MIN nothing, LAND_FULL and up the full terrain ink)
 * and drawn onto the underlay scaled up with smoothing. Land only where
 * nodes crowd; a single point casts no halo. Nothing is outlined — the map's
 * shapes come from density alone.
 */
export function drawCartography(
  ctx: CanvasRenderingContext2D,
  scene: CartographyScene,
  project: (pos: { x: number; y: number }) => { x: number; y: number },
  palette: GraphPalette,
  viewport: { width: number; height: number },
  field: TerrainField | null,
): void {
  const r = TERRAIN_RADIUS_PX;
  if (!field) {
    // No 2D canvas (jsdom): one flat disc per point stands in for the field.
    ctx.save();
    ctx.fillStyle = palette.terrain;
    for (const point of scene.points) {
      const at = project(point);
      if (at.x < -r || at.y < -r || at.x > viewport.width + r || at.y > viewport.height + r) continue;
      ctx.beginPath();
      ctx.arc(at.x, at.y, r, 0, 2 * Math.PI);
      ctx.fill();
    }
    ctx.restore();
    return;
  }
  const cols = Math.max(1, Math.ceil(viewport.width / TERRAIN_CELL_PX));
  const rows = Math.max(1, Math.ceil(viewport.height / TERRAIN_CELL_PX));
  if (field.canvas.width !== cols || field.canvas.height !== rows) {
    field.canvas.width = cols;
    field.canvas.height = rows;
  }
  const fctx = field.ctx;
  fctx.globalCompositeOperation = "source-over";
  fctx.clearRect(0, 0, cols, rows);
  fctx.globalCompositeOperation = "lighter";
  const kr = field.kernel.width / 2;
  let stamped = 0;
  for (const point of scene.points) {
    const at = project(point);
    if (at.x < -r || at.y < -r || at.x > viewport.width + r || at.y > viewport.height + r) continue;
    fctx.drawImage(field.kernel, at.x / TERRAIN_CELL_PX - kr, at.y / TERRAIN_CELL_PX - kr);
    stamped += 1;
  }
  fctx.globalCompositeOperation = "source-over";
  if (stamped === 0) return;
  const image = fctx.getImageData(0, 0, cols, rows);
  shadeField(image.data, parseInk(palette.terrain));
  fctx.putImageData(image, 0, 0);
  ctx.save();
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = "high";
  ctx.drawImage(field.canvas, 0, 0, cols, rows, 0, 0, cols * TERRAIN_CELL_PX, rows * TERRAIN_CELL_PX);
  ctx.restore();
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
 * appear as you approach.
 */
export function placeRegionLabels(
  scene: CartographyScene,
  project: (pos: { x: number; y: number }) => { x: number; y: number },
  measure: (text: string, size: number) => number,
): PlacedLabel[] {
  const placed: PlacedLabel[] = [];
  const boxes: { left: number; right: number; top: number; bottom: number }[] = [];
  for (const region of scene.regions) {
    if (placed.length >= MAX_REGION_LABELS) break;
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
    const at = { x: centroid.x, y: centroid.y - lift };
    const size = placed.length === 0 ? 15 : 12;
    const halfWidth = measure(region.name, size) / 2;
    const box = {
      left: at.x - halfWidth,
      right: at.x + halfWidth,
      top: at.y - size / 2,
      bottom: at.y + size / 2,
    };
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
  for (const label of placeRegionLabels(scene, project, measure)) {
    ctx.font = regionLabelFont(label.size);
    // Same tracking the measurement used; jsdom's mock ctx simply ignores it.
    ctx.letterSpacing = regionLabelTracking(label.size);
    ctx.strokeText(label.name, label.x, label.y);
    ctx.fillText(label.name, label.x, label.y);
    ctx.letterSpacing = "0px";
  }
  ctx.restore();
}
