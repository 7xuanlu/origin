// SPDX-License-Identifier: AGPL-3.0-only
import Graph from "graphology";
import forceAtlas2 from "graphology-layout-forceatlas2";
import {
  forceSimulation,
  forceLink,
  forceManyBody,
  forceCollide,
  type Simulation,
  type SimulationNodeDatum,
} from "d3-force";
import type { GraphModel } from "./model";
import {
  MEMORY_NODE_TYPE,
  PAGE_NODE_TYPE,
  SHARED_SOURCE_EDGE_TYPE,
  WIKILINK_EDGE_TYPE,
  CITES_EDGE_TYPE,
  MEMORY_EDGE_TYPE,
} from "./model";
import { compositeOver, nodeFillFor, type GraphPalette } from "./palette";

const MIN_NODE_SIZE = 3;
const MAX_NODE_SIZE = 14;
/** Wiki pages sit on the same scale as entities. Round 3 pulled the base from
 *  3.5 down to the unconfirmed-entity 3: with shared-source overlap no longer
 *  counting toward size (see buildAtlasGraph), a page reads as a subject
 *  without the whole page layer drawing a size step above the entities. */
const PAGE_NODE_BASE = 3;
/** Memory nodes start below the smallest entity and stay there: they are
 *  context around the subjects, and a memory linked to six entities must not
 *  outgrow the entities themselves. */
const MEMORY_NODE_BASE = 1.5;
const MEMORY_MAX_SIZE = 3;
/** Px added per doubling of degree. Linear growth (round 1's degree * 0.5)
 *  saturated the cap on real data — 1,600 entities where the hubs run to
 *  hundreds of links drew as one uniform field of 8s. log2 keeps the hubs
 *  visibly bigger while leaving the long tail distinguishable; the gain and
 *  cap went up together (1.6/12 -> 1.9/14) so a hub reads as a landmark on
 *  the relief map, not just a slightly larger dot. */
const DEGREE_GAIN = 1.9;

// Base by stability/kind (confirmed 4, unconfirmed 3, page 3, memory 1.5)
// plus DEGREE_GAIN per doubling of degree, capped. Size still encodes
// confirmation at degree 0, as it did before.
function nodeSizeFor(confirmed: boolean | null, degree: number, entityType?: string): number {
  const growth = DEGREE_GAIN * Math.log2(1 + degree);
  if (entityType === MEMORY_NODE_TYPE) {
    return Math.min(MEMORY_MAX_SIZE, MEMORY_NODE_BASE + growth);
  }
  const base =
    entityType === PAGE_NODE_TYPE ? PAGE_NODE_BASE : confirmed === true ? 4 : MIN_NODE_SIZE;
  return Math.min(MAX_NODE_SIZE, base + growth);
}

/** Edge stroke by verb, in CSS px. Edges are the quietest ink on the map — a
 *  hairline at rest, so the points carry the picture; a
 *  shared-source edge (an inferred overlap, not an asserted link) is thinner
 *  still. Hover emphasis recolours, it does not thicken. */
export function edgeSizeFor(type: string): number {
  return type === SHARED_SOURCE_EDGE_TYPE ? 0.6 : 1;
}

/**
 * GraphModel -> a graphology instance sigma can render directly. Positions
 * seed on a deterministic circle (node array order), so the result is
 * reproducible without running a layout; runAtlasLayout refines it from
 * there. `multi: true` because GraphModel's parallel-edge policy keeps
 * distinct relations between the same pair as distinct edges (see model.ts)
 * — a simple graph would throw adding the second one.
 */
export function buildAtlasGraph(model: GraphModel, palette: GraphPalette): Graph {
  const graph = new Graph({ multi: true });
  // ponytail: the old graph's confirmed-glow halo (r+3 disc at 0.1 alpha) is
  // skipped — it needs a custom WebGL node program in sigma; the tiered fills
  // and size base carry the confirmed/unconfirmed distinction instead.

  // Shared-source edges stand for an inferred overlap, not an asserted link,
  // so they must not inflate a page's disc: size is computed from the degree
  // MINUS however many shared-source edges touch the node. Done here rather
  // than as a second field on GraphNode — the model type stays untouched, and
  // the one place that draws discs is the one place that has to know.
  const sharedSourceIncident = new Map<string, number>();
  for (const edge of model.edges) {
    if (edge.type !== SHARED_SOURCE_EDGE_TYPE) continue;
    sharedSourceIncident.set(edge.source, (sharedSourceIncident.get(edge.source) ?? 0) + 1);
    if (edge.target !== edge.source) {
      sharedSourceIncident.set(edge.target, (sharedSourceIncident.get(edge.target) ?? 0) + 1);
    }
  }

  // Seed circle over the nodes the layout will move — memories are not among
  // them (see nonSimulatedIds: a memory rides its anchor), so they are left
  // off the circle and out of its count. That is what keeps the seed, and so
  // the whole layout, identical with the memory layer on and off.
  const seeded = model.nodes.filter((node) => node.entityType !== MEMORY_NODE_TYPE);
  const n = seeded.length;
  const seedIndex = new Map(seeded.map((node, i) => [node.id, i]));
  model.nodes.forEach((node) => {
    const i = seedIndex.get(node.id);
    const angle = i === undefined ? 0 : (2 * Math.PI * i) / Math.max(n, 1);
    const sizingDegree = Math.max(0, node.degree - (sharedSourceIncident.get(node.id) ?? 0));
    graph.addNode(node.id, {
      label: node.name,
      size: nodeSizeFor(node.confirmed, sizingDegree, node.entityType),
      color: nodeFillFor(node.entityType, node.confirmed, palette),
      entityType: node.entityType,
      // Kept on the node so the theme-flip recolor (AtlasView) can recompute
      // the stability-tiered fill without re-reading the model.
      confirmed: node.confirmed,
      x: i === undefined ? 0 : Math.cos(angle),
      y: i === undefined ? 0 : Math.sin(angle),
    });
  });

  // ponytail: parallel edges between the same pair draw fully overlapped —
  // a view decision (see model.ts's parallel-edge note), fine at round-1 scale.
  for (const edge of model.edges) {
    graph.addEdgeWithKey(edge.id, edge.source, edge.target, {
      // Kept on the edge so the hover reducer can read the verb without
      // re-reading the model.
      edgeType: edge.type,
      // Rendered 1:1 in CSS px (AtlasView pins zoomToSizeRatioFunction to 1).
      // Needs minEdgeThickness lowered in AtlasView — sigma's default floor
      // (1.7) silently bumps a hairline back up.
      size: edgeSizeFor(edge.type),
      color: palette.edge,
    });
  }
  return graph;
}

/** Iteration budget for FA2: 600 on anything up to 200 nodes (unchanged from
 *  round 2 at demo scale), decaying to the 60 floor by ~2,000 nodes. The whole
 *  layout is synchronous on the main thread, so the budget has to shrink as
 *  the per-iteration cost grows or the memory layer freezes the UI. */
function layoutIterations(order: number): number {
  return Math.min(600, Math.max(60, Math.floor(120_000 / Math.max(order, 1))));
}

/** Same shape for the d3 settle: the full 220 pre-paint ticks up to ~720
 *  nodes, then down to the 60 floor. */
function settleTicks(order: number): number {
  return Math.min(220, Math.max(60, Math.floor(160_000 / Math.max(order, 1))));
}

/**
 * Force-directed refinement of the seeded circle, synchronous.
 *
 * Non-simulated nodes (leaf memories, isolates — see nonSimulatedIds) are laid
 * out on a scratch copy that leaves them out entirely, then the positions are
 * copied back. On real data that is ~1,300 of ~3,300 nodes and their edges
 * removed from the O(n log n) repulsion, and they get their real positions
 * from placeSatellites afterwards anyway.
 *
 * ponytail: still sync FA2. If this ever blocks past ~3s again, the next step
 * is graphology-layout-forceatlas2's worker entry plus a "laying out" state.
 */
export function runAtlasLayout(graph: Graph): void {
  const excluded = new Set(nonSimulatedIds(graph));
  if (excluded.size === 0) {
    forceAtlas2.assign(graph, {
      iterations: layoutIterations(graph.order),
      settings: forceAtlas2.inferSettings(graph),
    });
    return;
  }

  const core = new Graph({ multi: true });
  graph.forEachNode((id, attrs) => {
    if (excluded.has(id)) return;
    core.addNode(id, { x: attrs.x, y: attrs.y, size: attrs.size });
  });
  graph.forEachEdge((key, _attrs, source, target) => {
    if (excluded.has(source) || excluded.has(target)) return;
    core.addEdgeWithKey(key, source, target, {});
  });
  if (core.order === 0) return;
  forceAtlas2.assign(core, {
    iterations: layoutIterations(core.order),
    settings: forceAtlas2.inferSettings(core),
  });
  core.forEachNode((id, attrs) => {
    graph.setNodeAttribute(id, "x", attrs.x);
    graph.setNodeAttribute(id, "y", attrs.y);
  });
}

/** Graph-space distance from the anchor disc's EDGE to a first-ring leaf
 *  memory's CENTRE (the leaf's own radius, 1.5-3 px, comes out of it, so the
 *  edge-to-edge clearance is 7-8.5). Went 6 -> 10 with the memory discs
 *  shrinking: at 6 the first ring of discs touched the page disc and read
 *  as a grey ring round a teal one, a donut, on the real map with memories
 *  on. */
const SATELLITE_GAP = 10;

/**
 * Nodes no force layout runs on: degree-0 isolates (round 1's ring) plus EVERY
 * memory, whatever its degree. A memory is evidence about a subject, not a
 * place of its own: it rides the subject it says most about (see
 * satelliteAnchor) instead of pulling subjects together with its own springs.
 * That is what keeps the map identical with the memory layer on and off — on
 * real data the layer adds 2,000 memories and 4,910 edges, and simulating
 * them re-laid the whole map into a pile every time the chip was pressed.
 */
export function nonSimulatedIds(graph: Graph): string[] {
  const ids: string[] = [];
  graph.forEachNode((id, attrs) => {
    if (graph.degree(id) === 0 || attrs.entityType === MEMORY_NODE_TYPE) ids.push(id);
  });
  return ids;
}

/** The node a memory rides: its most-connected non-memory neighbour, ties to
 *  the smaller id so the answer never depends on edge order. A memory linked
 *  to three entities sits by the hub among them, where the eye looks for it;
 *  its other links are drawn on hover only (see edgeDisplay). */
export function satelliteAnchor(graph: Graph, id: string): string | undefined {
  let best: string | undefined;
  let bestDegree = -1;
  for (const neighbour of graph.neighbors(id)) {
    if (graph.getNodeAttribute(neighbour, "entityType") === MEMORY_NODE_TYPE) continue;
    const degree = graph.degree(neighbour);
    if (degree > bestDegree || (degree === bestDegree && best !== undefined && neighbour < best)) {
      best = neighbour;
      bestDegree = degree;
    }
  }
  return best;
}

/** Where one memory sits relative to the node it rides. */
export interface Satellite {
  id: string;
  anchor: string;
  angle: number;
  radius: number;
  /** 0-based place in the anchor's halo, inner ring first, ids ascending
   *  within a ring. The zoom tiers (see dustVisibleCount) show the first N. */
  rank: number;
}

/** Smallest arc a satellite may claim on its ring: two disc diameters plus a
 *  pixel, so consecutive leaves read as separate dots with clear sky between
 *  them rather than a bead chain. */
function satelliteMinArc(leafSize: number): number {
  return 4 * leafSize + 1;
}

/** Radial step from one shell to the next: a disc diameter plus 3, so a leaf
 *  on the outer ring cannot touch the one it sits behind. */
function satelliteRingStep(leafSize: number): number {
  return 2 * leafSize + 3;
}

/**
 * Deterministic orbits for the memories: each hangs off its anchor (see
 * satelliteAnchor), sorted by id so the answer never depends on iteration
 * order. Isolates are skipped — they have no anchor to orbit.
 *
 * Round 5: leaves fill SHELLS, not a single circle. A ring takes as many
 * leaves as fit at satelliteMinArc spacing and the rest start a new ring
 * satelliteRingStep further out. One circle was fine for the handful of
 * leaves a test fixture has, but the real capture has an entity anchoring
 * 374 of them — at radius anchorSize + SATELLITE_GAP that is 0.2 graph units
 * of arc each, drawn as a solid donut of overlapping discs.
 */
export function satellitePlan(graph: Graph): Satellite[] {
  const leavesByAnchor = new Map<string, string[]>();
  for (const id of nonSimulatedIds(graph)) {
    if (graph.degree(id) === 0) continue;
    const anchor = satelliteAnchor(graph, id);
    if (anchor === undefined) continue;
    const list = leavesByAnchor.get(anchor);
    if (list) list.push(id);
    else leavesByAnchor.set(anchor, [id]);
  }
  const plan: Satellite[] = [];
  for (const [anchor, leaves] of leavesByAnchor) {
    const sorted = [...leaves].sort();
    // One spacing for the whole halo, taken from the widest leaf in it: a
    // per-leaf spacing would make ring capacity depend on which leaves
    // happened to land on that ring.
    let leafSize = 0;
    for (const id of sorted) {
      leafSize = Math.max(leafSize, (graph.getNodeAttribute(id, "size") as number) ?? 0);
    }
    const minArc = satelliteMinArc(leafSize);
    const step = satelliteRingStep(leafSize);
    let radius = (graph.getNodeAttribute(anchor, "size") as number) + SATELLITE_GAP;
    let placed = 0;
    while (placed < sorted.length) {
      // At least one per ring even when the anchor disc is tiny, or a small
      // circumference would stall the loop.
      const capacity = Math.max(1, Math.floor((2 * Math.PI * radius) / minArc));
      const count = Math.min(capacity, sorted.length - placed);
      for (let i = 0; i < count; i += 1) {
        plan.push({
          id: sorted[placed + i] as string,
          anchor,
          angle: (2 * Math.PI * i) / count,
          radius,
          rank: placed + i,
        });
      }
      placed += count;
      radius += step;
    }
  }
  return plan;
}

/** Write a satellite plan onto the graph. Cheap enough (two trig calls per
 *  leaf) to re-run on every tick writeback, which is what makes a dragged
 *  entity carry its memories along.
 *
 *  `skipId` is the node the pointer is currently holding. A satellite is not
 *  a sim node, so a drag moves it by writing the graph directly; without this
 *  exemption the next writeback would put it straight back on its orbit and
 *  the leaf would look unmovable while the sim is warm. */
export function placeSatellites(graph: Graph, plan: Satellite[], skipId?: string | null): void {
  for (const satellite of plan) {
    if (satellite.id === skipId) continue;
    const ax = graph.getNodeAttribute(satellite.anchor, "x") as number;
    const ay = graph.getNodeAttribute(satellite.anchor, "y") as number;
    graph.setNodeAttribute(satellite.id, "x", ax + satellite.radius * Math.cos(satellite.angle));
    graph.setNodeAttribute(satellite.id, "y", ay + satellite.radius * Math.sin(satellite.angle));
  }
}

/** Write the halo bookkeeping the reducers read onto the graph: each memory's
 *  `dustRank` and `dustOf` (its anchor), and each anchor's `dustCount`. The
 *  plan is the authority; this just makes it reachable from a node's attrs
 *  at paint time without a lookup. */
export function annotateDust(graph: Graph, plan: Satellite[]): void {
  const counts = new Map<string, number>();
  for (const satellite of plan) {
    graph.mergeNodeAttributes(satellite.id, { dustRank: satellite.rank, dustOf: satellite.anchor });
    counts.set(satellite.anchor, (counts.get(satellite.anchor) ?? 0) + 1);
  }
  for (const [anchor, count] of counts) graph.setNodeAttribute(anchor, "dustCount", count);
}

/**
 * How many of an anchor's memories are drawn at a given zoom, as a multiple
 * of the zoom the map opened at (`zoomIn` = mount ratio / current ratio, so 1
 * at the opening view and 2 when the viewer has zoomed in twice). At the
 * opening view an anchor shows a pinch of dust — six dots — and a count of
 * the rest (AtlasView's overlay); closer in, a full first ring and more;
 * closer still, everything. Hovering the anchor shows everything at any zoom
 * (see nodeDisplay).
 */
export function dustVisibleCount(zoomIn: number): number {
  if (zoomIn < 2) return 6;
  if (zoomIn < 4) return 18;
  return Infinity;
}

/** How many of an anchor's memories a hover reveals at most. Everything
 *  would be the honest answer, but the busiest anchor on real data carries
 *  698 and at the opening zoom they pack into a solid disc; the first rings
 *  say "a lot" without the blob, and the count beside the anchor says how
 *  many. Zoom in past 4x to see them all. */
export const HOVER_DUST_MAX = 48;

/** How many neighbors a hover names outright (`forceLabel`). Beyond this the
 *  label grid decides, so a hub with dozens of memories stays readable. */
export const NEIGHBOR_LABEL_MAX = 12;

/** Zoomed in this much past the opening view, the islands (every component
 *  but the core) are drawn solid with their names and edges; before that they
 *  sit dim at the rim so the core is the one thing the eye lands on. */
export const ISLANDS_SOLID_ZOOM = 2;

/** What the reducers need to know about the camera, refreshed by AtlasView
 *  before every render. */
export interface LodState {
  dustVisible: number;
  islandsSolid: boolean;
}

/** The opening view's state: a pinch of dust, islands dim. */
export const OPENING_LOD: LodState = { dustVisible: dustVisibleCount(1), islandsSolid: false };

export function lodFor(zoomIn: number): LodState {
  return { dustVisible: dustVisibleCount(zoomIn), islandsSolid: zoomIn >= ISLANDS_SOLID_ZOOM };
}

/** Clearance between the core's box and the nearest island, and between any
 *  two islands, in graph units — a moat, so an island never reads as a
 *  peninsula of the core or of its neighbour. */
export const ISLAND_GAP = 34;
/** Islands ring the core on a slight ellipse: screens are wider than tall, so
 *  a ring squashed to this fraction on y fills the frame instead of piling
 *  islands into the top and bottom margins. */
export const ISLAND_RING_SQUASH = 0.8;
/** How far out an island probes per step while looking for clear water. */
const ISLAND_STEP = 4;
const GOLDEN_ANGLE = Math.PI * (3 - Math.sqrt(5));

/** One component's drawn extent, in graph units. */
interface ComponentBox {
  ids: string[];
  minX: number;
  maxX: number;
  minY: number;
  maxY: number;
}

const boxWidth = (box: ComponentBox): number => box.maxX - box.minX;
const boxHeight = (box: ComponentBox): number => box.maxY - box.minY;

/** Connected components over the DRAWN graph, in graph node order, with one
 *  rule for memories: a memory belongs to its ANCHOR's component and joins
 *  nothing else. Its other edges are not adjacency here — a memory that
 *  mentions two entities from two components is evidence about both, not a
 *  bridge between them, and treating it as one would merge two islands the
 *  moment the memory layer is turned on. `anchorOf` is the satellite plan's
 *  memory -> anchor map. */
function graphComponents(graph: Graph, anchorOf: Map<string, string>): string[][] {
  const parent = new Map<string, string>();
  graph.forEachNode((id) => parent.set(id, id));
  const find = (start: string): string => {
    let root = start;
    while (parent.get(root) !== root) root = parent.get(root) as string;
    let walk = start;
    while (parent.get(walk) !== root) {
      const next = parent.get(walk) as string;
      parent.set(walk, root);
      walk = next;
    }
    return root;
  };
  graph.forEachEdge((_key, _attrs, source, target) => {
    if (anchorOf.has(source) || anchorOf.has(target)) return;
    const a = find(source);
    const b = find(target);
    if (a !== b) parent.set(a, b);
  });
  for (const [memory, anchor] of anchorOf) {
    const a = find(memory);
    const b = find(anchor);
    if (a !== b) parent.set(a, b);
  }
  const groups = new Map<string, string[]>();
  graph.forEachNode((id) => {
    const root = find(id);
    const list = groups.get(root);
    if (list) list.push(id);
    else groups.set(root, [id]);
  });
  return [...groups.values()];
}

/** Bounding box of a component's ink: every node's disc. Satellites are
 *  skipped — a memory's dust is drawn at most a few dots deep at the opening
 *  view and never counts toward the room a component takes, which is what
 *  keeps the memory layer from reshaping the map. */
function measureComponent(graph: Graph, ids: string[], satellites: Set<string>): ComponentBox {
  let minX = Infinity;
  let maxX = -Infinity;
  let minY = Infinity;
  let maxY = -Infinity;
  for (const id of ids) {
    if (satellites.has(id)) continue;
    const x = graph.getNodeAttribute(id, "x") as number;
    const y = graph.getNodeAttribute(id, "y") as number;
    const pad = (graph.getNodeAttribute(id, "size") as number) ?? 0;
    minX = Math.min(minX, x - pad);
    maxX = Math.max(maxX, x + pad);
    minY = Math.min(minY, y - pad);
    maxY = Math.max(maxY, y + pad);
  }
  if (minX === Infinity) {
    // A component of nothing but satellites cannot happen (a satellite's
    // anchor is in its own component), but a zero box beats a NaN one.
    return { ids, minX: 0, maxX: 0, minY: 0, maxY: 0 };
  }
  return { ids, minX, maxX, minY, maxY };
}

/** A placed island: centre and the radius of the circle it claims. */
interface Island {
  x: number;
  y: number;
  r: number;
}

/** Shortest distance from a point to an axis-aligned box (0 inside it). */
function distanceToBox(x: number, y: number, box: ComponentBox): number {
  const dx = Math.max(box.minX - x, 0, x - box.maxX);
  const dy = Math.max(box.minY - y, 0, y - box.maxY);
  return Math.hypot(dx, dy);
}

/**
 * Round 7. Islands, not wings.
 *
 * The biggest component is the CORE: it keeps the layout the sim gave it and
 * is recentred so its bbox centre is the origin. Every other component is an
 * ISLAND — a circle the size of its own half-diagonal — placed largest first
 * on a golden-angle spiral round the core: each walks out along its own angle
 * until it clears the core's box and every island already placed by
 * ISLAND_GAP. Big islands therefore sit nearest the core and the singletons
 * (lone nodes, shown only when the reader turns the small groups on) end up
 * on the outer rim. The ring is squashed to ISLAND_RING_SQUASH on y so it
 * fills a landscape viewport.
 *
 * A full ring was tried once before (round 4) and dropped because it read as
 * a false core-and-periphery hierarchy and shrank the core; the wings that
 * replaced it (round 6) were rejected by the reader outright. The difference
 * now is zoom: at the opening view the islands are drawn dim, no names, no
 * edges (see LodState), so the eye lands on the core and the islands are a
 * rim the viewer knows is there. Zoom in past ISLANDS_SOLID_ZOOM and they
 * come up solid.
 *
 * Returns the node ids of each component in placement order — the core first,
 * so the caller can tell it from the islands and hold each kind the way it
 * needs (rigid centroid for the core, springs for the islands).
 *
 * Pure in the sense that matters here: it reads x/y/size off the graph and
 * writes x/y back, touching nothing else and consulting no clock or random.
 */
export function shelveComponents(graph: Graph): string[][] {
  if (graph.order === 0) return [];
  const anchorOf = new Map(satellitePlan(graph).map((satellite) => [satellite.id, satellite.anchor]));
  const satellites = new Set(anchorOf.keys());
  // Ranked by the nodes that take room — satellites count for nothing here
  // either, or the memory layer could promote an island to core.
  const weight = (box: ComponentBox) => box.ids.filter((id) => !satellites.has(id)).length;
  const boxes = graphComponents(graph, anchorOf)
    .map((ids) => measureComponent(graph, ids, satellites))
    .sort((a, b) => weight(b) - weight(a) || ((a.ids[0] as string) < (b.ids[0] as string) ? -1 : 1));

  const translate = (box: ComponentBox, dx: number, dy: number) => {
    for (const id of box.ids) {
      graph.setNodeAttribute(id, "x", (graph.getNodeAttribute(id, "x") as number) + dx);
      graph.setNodeAttribute(id, "y", (graph.getNodeAttribute(id, "y") as number) + dy);
    }
    box.minX += dx;
    box.maxX += dx;
    box.minY += dy;
    box.maxY += dy;
  };

  const core = boxes[0] as ComponentBox;
  translate(core, -(core.minX + core.maxX) / 2, -(core.minY + core.maxY) / 2);

  const placed: Island[] = [];
  boxes.slice(1).forEach((box, i) => {
    const r = Math.hypot(boxWidth(box), boxHeight(box)) / 2;
    const angle = i * GOLDEN_ANGLE;
    // Start at the core's nearer half-extent: nothing closer can clear it.
    let d = Math.min(boxWidth(core), boxHeight(core)) / 2;
    for (;;) {
      const x = Math.cos(angle) * d;
      const y = Math.sin(angle) * d * ISLAND_RING_SQUASH;
      const clearOfCore = distanceToBox(x, y, core) >= r + ISLAND_GAP;
      if (clearOfCore && placed.every((p) => Math.hypot(p.x - x, p.y - y) >= p.r + r + ISLAND_GAP)) {
        placed.push({ x, y, r });
        translate(box, x - (box.minX + box.maxX) / 2, y - (box.minY + box.maxY) / 2);
        break;
      }
      d += ISLAND_STEP;
    }
  });

  return [core.ids, ...boxes.slice(1).map((box) => box.ids)];
}

/** The sim plus the one thing the view has to tell it: which node the pointer
 *  is holding right now. Satellites are not sim nodes, so the writeback needs
 *  to be told to leave the dragged one alone (see placeSatellites). */
export interface AtlasSimulation extends Simulation<AtlasSimNode, undefined> {
  setDraggingId(id: string | null): void;
  /** Put every unheld shelf component back on its slot right now, instead of
   *  over the cooling tail. The release path uses it when there will be no
   *  cooling tail (see shelfAnchorForce's `settle`). */
  settleShelf(): void;
}

export interface AtlasSimNode extends SimulationNodeDatum {
  id: string;
  /** Collision radius: the drawn disc plus COLLIDE_PAD. */
  radius: number;
}

export interface AtlasSimLink {
  source: string | AtlasSimNode;
  target: string | AtlasSimNode;
  /** The graph edge's verb, which sets this link's rest length and pull. */
  type: string;
}

/** Node-to-node repulsion, matching the retired ConstellationMap feel. Named
 *  because the shelf relax pass (relaxShelf) has to run the same charge over
 *  its scratch copy or a component would relax to a different shape there
 *  than it holds in the live simulation. */
const CHARGE_STRENGTH = -40;
/** Collide runs at 0.7 rather than 1 so a collision resolves over a few ticks
 *  instead of snapping, which reads calmer. */
const COLLIDE_STRENGTH = 0.7;

/** Breathing room between two discs that are not otherwise pushed apart. Two
 *  px is enough to stop the overlap that made page clusters read as one blob
 *  without visibly loosening the rest of the map. */
const COLLIDE_PAD = 2;
/** Rest length and pull per edge verb; anything not listed keeps d3's own
 *  defaults (distance 30, strength 1/min(degree)). A shared-source edge is an
 *  inferred overlap, so it sits long and slack — it should suggest that two
 *  pages are near each other, not staple them together. A wikilink is an
 *  asserted link between pages, so it is shorter and firmer, but still longer
 *  than a relation so page clusters read as a stratum, not a clump. */
const LINK_LAYOUT: Record<string, { distance: number; strength: number }> = {
  [SHARED_SOURCE_EDGE_TYPE]: { distance: 70, strength: 0.15 },
  [WIKILINK_EDGE_TYPE]: { distance: 50, strength: 0.5 },
};

/** Which verb survives when parallel edges between one pair collapse to a
 *  single spring: the most strongly asserted one wins, so a wikilink is never
 *  loosened by a shared-source edge that happens to run beside it. */
function linkPriority(type: string): number {
  if (type === WIKILINK_EDGE_TYPE) return 3;
  if (type === CITES_EDGE_TYPE || type === MEMORY_EDGE_TYPE) return 1;
  if (type === SHARED_SOURCE_EDGE_TYPE) return 0;
  // Entity relations and page->entity `about` links: asserted, unremarkable.
  return 2;
}

function linkEndId(end: string | AtlasSimNode): string {
  return typeof end === "string" ? end : end.id;
}

/** Rest length for one link's verb — LINK_LAYOUT's, or d3's own 30. */
function linkDistanceFor(link: AtlasSimLink): number {
  return LINK_LAYOUT[link.type]?.distance ?? 30;
}

/** Pull for one link's verb. d3's default strength is 1/min(degree) over the
 *  LINK graph, computed privately inside forceLink — overriding .strength()
 *  would throw that away for every verb, so the same formula is rebuilt here
 *  from a precomputed degree map and used for anything LINK_LAYOUT does not
 *  name. */
function linkStrengthFor(link: AtlasSimLink, degree: Map<string, number>): number {
  const named = LINK_LAYOUT[link.type];
  if (named) return named.strength;
  const source = degree.get(linkEndId(link.source)) ?? 1;
  const target = degree.get(linkEndId(link.target)) ?? 1;
  return 1 / Math.min(source, target);
}

/** One shelf group's rigid centroid anchor: every free (non-fx/fy) member is
 *  shifted uniformly so the group's own centroid sits at `targetX/targetY`. */
interface CenterGroup {
  targetX: number;
  targetY: number;
}

/**
 * `forceCenter(0, 0)`, but blind to pinned nodes, PER GROUP, and able to hold
 * a target other than the origin.
 *
 * d3's own centre force averages EVERY node and shifts them all by the
 * offset. Pinned nodes snap straight back to fx/fy afterwards, so with a
 * shelf hanging below the core the average sits below the origin every tick
 * and the whole correction lands on the core: on the real capture a
 * drag-reheat slid the core 776 units up a 3,939-unit span, pulling the two
 * zones apart. Averaging the FREE nodes only cuts that to 371.
 *
 * `retarget()` removes the rest. shelveComponents centres each component by
 * its BOUNDING BOX, which is not where its centroid is; asking the force for
 * a centroid of exactly (0,0) therefore slides a component by the difference
 * on the first reheat. Retargeting after the shelf is laid out tells the
 * force to hold each component exactly where the composition put it, which
 * leaves only the small amount a component genuinely relaxes by.
 *
 * Round 5's live shelf generalizes this from one group to many (see
 * `setGroups`). On the live simulation only the CORE is held this way: a
 * rigid hold corrects position but not velocity, so a group carries whatever
 * momentum the external field gave it into the next tick — negligible across
 * the core's thousands of free nodes, but the dominant error on a handful of
 * nodes sitting thousands of units from the core's mass (measured at capture
 * scale: 28 units of centroid drift over a 60-tick drag reheat). Shelved
 * components are held by `shelfAnchorForce` instead. The many-group form is
 * still what `relaxShelf`'s scratch simulation uses, where every group is a
 * shelf component and there is no distant core to be pushed by.
 *
 * Before `setGroups` is ever called (the pre-shelf settle), every node falls
 * into one implicit group targeting the origin — exactly the old single-group
 * behavior. After it is called, a node no group names is left alone.
 */
function groupCenterForce(): {
  (alpha: number): void;
  initialize(nodes: AtlasSimNode[]): void;
  setGroups(placement: string[][]): void;
  retarget(): void;
} {
  let nodes: AtlasSimNode[] = [];
  // Null until setGroups runs: every node is then in one implicit group at the
  // origin. Once groups are installed, an id no group names is skipped — on
  // the live simulation that is every shelved node, held by shelfAnchorForce.
  let groupOf: Map<string, CenterGroup> | null = null;
  const defaultGroup: CenterGroup = { targetX: 0, targetY: 0 };
  const groupFor = (id: string): CenterGroup | undefined =>
    groupOf === null ? defaultGroup : groupOf.get(id);

  // One pass over all nodes, bucketed by group — O(nodes) regardless of how
  // many components are on the shelf, rather than O(nodes * groups).
  const centroids = (): Map<CenterGroup, { x: number; y: number; free: number }> => {
    const sums = new Map<CenterGroup, { x: number; y: number; free: number }>();
    for (const node of nodes) {
      if (node.fx != null || node.fy != null) continue;
      const group = groupFor(node.id);
      if (!group) continue;
      const entry = sums.get(group) ?? { x: 0, y: 0, free: 0 };
      entry.x += node.x ?? 0;
      entry.y += node.y ?? 0;
      entry.free += 1;
      sums.set(group, entry);
    }
    for (const entry of sums.values()) {
      entry.x /= entry.free;
      entry.y /= entry.free;
    }
    return sums;
  };

  const force = () => {
    const sums = centroids();
    for (const node of nodes) {
      if (node.fx != null || node.fy != null) continue;
      const group = groupFor(node.id);
      if (!group) continue;
      const centroid = sums.get(group);
      if (!centroid) continue;
      node.x = (node.x ?? 0) - (centroid.x - group.targetX);
      node.y = (node.y ?? 0) - (centroid.y - group.targetY);
    }
  };
  force.initialize = (ns: AtlasSimNode[]) => {
    nodes = ns;
  };
  // Installs one group per placement entry. An id no entry names is left to
  // whatever else holds it — on the live simulation, the shelf springs.
  force.setGroups = (placement: string[][]) => {
    const installed = new Map<string, CenterGroup>();
    groupOf = installed;
    for (const ids of placement) {
      const group: CenterGroup = { targetX: 0, targetY: 0 };
      for (const id of ids) installed.set(id, group);
    }
  };
  force.retarget = () => {
    for (const [group, centroid] of centroids()) {
      group.targetX = centroid.x;
      group.targetY = centroid.y;
    }
  };
  return force;
}

/** How hard a shelved node is pulled back to the slot the packing gave it.
 *  0.1 is d3's own forceX/forceY default: an order of magnitude weaker than a
 *  link (strength 1), so a hand on one node still drags its neighbors along
 *  instead of fighting the anchor, and a released node eases back to its slot
 *  instead of snapping. */
const SHELF_ANCHOR_STRENGTH = 0.1;

/** What fraction of a component's remaining centroid offset is closed per
 *  tick while no hand is on it. Deliberately NOT scaled by alpha, unlike every
 *  other force here: alpha is a finite budget (it decays 3%/tick toward
 *  alphaMin, so the whole cooling tail after a release sums to about 10), and
 *  an alpha-scaled return of this shape completes only ~53% of the journey
 *  before the simulation goes to sleep — measured as a 77.5-unit residual
 *  offset, enough for a released pair to sit inside the core's box. Unscaled,
 *  0.1 closes the offset geometrically (0.925 per tick after velocity decay)
 *  and is done long before alphaMin. It is the same rate as the springs, so a
 *  release reads as one motion rather than two. */
const SHELF_RETURN_RATE = 0.1;

/** One shelved node's slot, and which component it belongs to. */
interface ShelfAnchor {
  x: number;
  y: number;
  group: number;
}

/**
 * What holds the shelf together once no component is frozen with fx/fy.
 *
 * Two parts, and both are needed:
 *
 * - A per-node SPRING toward the exact position the packing gave that node.
 *   Unlike a centroid hold this resists rotation and shear as well as
 *   translation, which is what the packed rows actually need: an elongated
 *   component in the core's repulsion field is a body in a radial field, and
 *   it swings to point at the core unless something pulls each end back. Its
 *   centroid barely moves while it does that, so a centroid hold does not
 *   even see it — on the 24/6/5 fixture a five-node chain raised its top edge
 *   9.1 units toward the core over 120 ticks under a centroid-only hold, and
 *   0.75 with these springs. Soft, so a drag still wins locally.
 * - Control of each component's BULK velocity, for components with no hand on
 *   them: whatever link and charge just handed the component as a whole is
 *   replaced by exactly the velocity that carries its centroid back to the
 *   slot, `offset * SHELF_RETURN_RATE`. Only the uniform part is touched — the
 *   part that translates a component without changing its shape — so every
 *   relative motion (a drag pulling a neighbor, collide opening a knot) is
 *   left alone. This is what a spring this soft cannot do on its own: at
 *   capture scale the core's mass accelerates a distant component by roughly
 *   11 units per tick, which a 0.1 spring only cancels hundreds of units away
 *   from the slot. Measured together at capture scale: 28.5 units of drift
 *   under the centroid hold, 0.65 under these two.
 *
 *   Steering home rather than simply zeroing the bulk velocity is what makes a
 *   RELEASE land. Zeroing removes the spring's own return velocity again on
 *   the following tick, leaving only one alpha-scaled impulse per tick to do
 *   the whole journey — and alpha runs out first: a 200-unit drag on a shelved
 *   pair used to settle 77.5 units from its slot, overlapping the core's box
 *   by 13. With the bulk velocity steered home it lands within a unit.
 *
 * A component the pointer is holding is left entirely alone: with one node
 * pinned, "the group's bulk velocity" is mostly the drag response itself, and
 * overriding it would clamp the very neighbors that are supposed to follow.
 * The hand is the constraint while it is down; this force takes over on
 * release. When the release skips the cooling tail altogether (reduced
 * motion), `settle` does the same correction in one step.
 */
function shelfAnchorForce(): {
  (alpha: number): void;
  initialize(nodes: AtlasSimNode[]): void;
  setPlacement(shelf: string[][]): void;
  settle(): void;
} {
  let nodes: AtlasSimNode[] = [];
  let anchorOf = new Map<string, ShelfAnchor>();
  let groupCount = 0;

  interface Bulk {
    /** Summed velocity and summed offset-to-anchor over the group's free members. */
    vx: number;
    vy: number;
    dx: number;
    dy: number;
    free: number;
  }

  /** Per group: how its free members are moving, how far they are from their
   *  slots, and whether the pointer is holding one of them. */
  const measure = (): { bulk: Bulk[]; handOn: boolean[] } => {
    const bulk = Array.from({ length: groupCount }, () => ({ vx: 0, vy: 0, dx: 0, dy: 0, free: 0 }));
    const handOn = new Array<boolean>(groupCount).fill(false);
    for (const node of nodes) {
      const anchor = anchorOf.get(node.id);
      if (!anchor) continue;
      if (node.fx != null || node.fy != null) {
        handOn[anchor.group] = true;
        continue;
      }
      const entry = bulk[anchor.group] as Bulk;
      entry.vx += node.vx ?? 0;
      entry.vy += node.vy ?? 0;
      entry.dx += anchor.x - (node.x ?? 0);
      entry.dy += anchor.y - (node.y ?? 0);
      entry.free += 1;
    }
    return { bulk, handOn };
  };

  const force = (alpha: number) => {
    if (anchorOf.size === 0) return;
    const { bulk, handOn } = measure();
    for (const node of nodes) {
      const anchor = anchorOf.get(node.id);
      if (!anchor) continue;
      if (node.fx != null || node.fy != null) continue;
      const entry = bulk[anchor.group] as Bulk;
      if (!handOn[anchor.group] && entry.free > 0) {
        // Shift every member by (wanted bulk velocity - current bulk velocity),
        // both means over the group's free set.
        node.vx = (node.vx ?? 0) + (entry.dx * SHELF_RETURN_RATE - entry.vx) / entry.free;
        node.vy = (node.vy ?? 0) + (entry.dy * SHELF_RETURN_RATE - entry.vy) / entry.free;
      }
      node.vx = (node.vx ?? 0) + (anchor.x - (node.x ?? 0)) * SHELF_ANCHOR_STRENGTH * alpha;
      node.vy = (node.vy ?? 0) + (anchor.y - (node.y ?? 0)) * SHELF_ANCHOR_STRENGTH * alpha;
    }
  };
  force.initialize = (ns: AtlasSimNode[]) => {
    nodes = ns;
  };
  /** Anchor every named node where it stands right now — call it straight
   *  after the pack, whose output IS the arrangement to hold. */
  force.setPlacement = (shelf: string[][]) => {
    const byId = new Map(nodes.map((node) => [node.id, node]));
    anchorOf = new Map();
    groupCount = shelf.length;
    shelf.forEach((ids, group) => {
      for (const id of ids) {
        const node = byId.get(id);
        if (!node) continue;
        anchorOf.set(id, { x: node.x ?? 0, y: node.y ?? 0, group });
      }
    });
  };
  /** The cooling tail's whole job, done in one step: rigidly translate every
   *  unheld component back onto its slot and stop it dead. For the release
   *  path under reduced motion, where there IS no cooling tail — the
   *  simulation is stopped on mouseup, so a component would otherwise keep
   *  whatever displacement the drag gave it, up to and including sitting on
   *  top of the core. Rigid, so the shape the drag produced survives; only the
   *  offset that breaks the two zones apart is undone. */
  force.settle = () => {
    if (anchorOf.size === 0) return;
    const { bulk, handOn } = measure();
    for (const node of nodes) {
      const anchor = anchorOf.get(node.id);
      if (!anchor) continue;
      if (node.fx != null || node.fy != null) continue;
      const entry = bulk[anchor.group] as Bulk;
      if (handOn[anchor.group] || entry.free === 0) continue;
      node.x = (node.x ?? 0) + entry.dx / entry.free;
      node.y = (node.y ?? 0) + entry.dy / entry.free;
      node.vx = 0;
      node.vy = 0;
    }
  };
  return force;
}

/**
 * The knot-opening relax pass, run on a scratch simulation that owns the
 * SHELVED nodes only.
 *
 * Every component keeps its own rigid centroid hold here (groupCenterForce
 * over the shelf placement), so charge, collide and links can spread a
 * component that settled into a tangle without any of them wandering off its
 * slot. Relaxed positions are copied back onto `nodes` — the caller re-packs
 * afterwards, because a component that opened up no longer fits its old row.
 *
 * Scratch rather than the live simulation for two reasons: the core is
 * already settled, so re-running it buys nothing and costs the whole O(core)
 * charge (the live-sim version of this pass doubled Atlas's synchronous load
 * time, 543 ms to ~1,000 ms on the real capture); and with the core absent
 * there is no distant mass to push the shelf around while it relaxes.
 *
 * Returns the scratch simulation, which is stopped and whose only remaining
 * use is to show a caller (or a test) exactly which nodes it owned.
 */
export function relaxShelf(
  nodes: AtlasSimNode[],
  links: AtlasSimLink[],
  placement: string[][],
  linkDegree: Map<string, number>,
): Simulation<AtlasSimNode, undefined> {
  const shelfPlacement = placement.slice(1);
  const shelfIds = new Set(shelfPlacement.flat());
  // Fresh node and link objects: d3 stamps `index` onto every node it
  // simulates and rewrites a link's endpoints from id to node reference, so
  // handing it the live simulation's own objects would corrupt the live
  // collide and link forces.
  const scratch: AtlasSimNode[] = [];
  const relaxed = new Map<string, AtlasSimNode>();
  for (const node of nodes) {
    if (!shelfIds.has(node.id)) continue;
    const copy: AtlasSimNode = { id: node.id, x: node.x, y: node.y, radius: node.radius };
    scratch.push(copy);
    relaxed.set(node.id, copy);
  }
  const scratchLinks: AtlasSimLink[] = [];
  for (const link of links) {
    const source = linkEndId(link.source);
    const target = linkEndId(link.target);
    if (!shelfIds.has(source) || !shelfIds.has(target)) continue;
    scratchLinks.push({ source, target, type: link.type });
  }

  const center = groupCenterForce();
  const sim = forceSimulation(scratch)
    .force(
      "link",
      forceLink<AtlasSimNode, AtlasSimLink>(scratchLinks)
        .id((d) => d.id)
        .distance(linkDistanceFor)
        .strength((link) => linkStrengthFor(link, linkDegree)),
    )
    .force("charge", forceManyBody<AtlasSimNode>().strength(CHARGE_STRENGTH))
    .force("center", center)
    .force(
      "collide",
      forceCollide<AtlasSimNode>().radius((d) => d.radius).strength(COLLIDE_STRENGTH).iterations(1),
    )
    .alphaDecay(0.03)
    .velocityDecay(0.25)
    // forceSimulation starts its own animation frame loop on construction;
    // this pass is synchronous and ticks by hand.
    .stop();
  center.setGroups(shelfPlacement);
  center.retarget();
  sim.alpha(0.3);
  sim.tick(settleTicks(scratch.length));

  for (const node of nodes) {
    const copy = relaxed.get(node.id);
    if (!copy) continue;
    node.x = copy.x;
    node.y = copy.y;
  }
  return sim;
}

/** d3-force simulation over the live graphology graph — the interaction engine.
 *  Sim nodes are the drawable graph MINUS the nodes nonSimulatedIds names:
 *  degree-0 isolates and every memory. Neither is simulated —
 *  isolates are not drawn at all once drawableModel drops components under
 *  MIN_COMPONENT_SIZE, and a memory rides its anchor as a satellite.
 *  Matches the retired ConstellationMap feel: charge -40,
 *  centre-on-origin (groupCenterForce), alphaDecay 0.03, velocityDecay 0.25, d3-default link
 *  force. Parallel edges collapse to one link per undirected pair (d3 sums
 *  pull per link; sigma still RENDERS every parallel edge). Settles
 *  synchronously to its own equilibrium before returning — a FA2 seed handed
 *  straight to a fresh sim explodes toward the sim's roomier rest state on
 *  first drag; settling here means the graph the caller paints is already at
 *  rest, and a drag only flexes it (see round 5 spec). Every tick writes sim
 *  x/y back into the graph (sigma auto-repaints on attr change). */
export function createAtlasSimulation(
  graph: Graph,
  onTick?: () => void,
): AtlasSimulation {
  const excluded = new Set(nonSimulatedIds(graph));
  const satellites = satellitePlan(graph);
  annotateDust(graph, satellites);
  const nodes: AtlasSimNode[] = [];
  graph.forEachNode((id, attrs) => {
    if (excluded.has(id)) return;
    nodes.push({
      id,
      x: attrs.x as number,
      y: attrs.y as number,
      radius: (attrs.size as number) + COLLIDE_PAD,
    });
  });

  const linkByPair = new Map<string, AtlasSimLink>();
  graph.forEachEdge((_edge, attrs, source, target) => {
    // A link to a node the sim doesn't own would make d3 throw looking the
    // endpoint up; a memory's edges are drawn (on hover) but never simulated.
    if (excluded.has(source) || excluded.has(target)) return;
    const pairKey = [source, target].sort().join("|");
    const type = (attrs.edgeType as string) ?? "";
    const existing = linkByPair.get(pairKey);
    if (!existing) {
      linkByPair.set(pairKey, { source, target, type });
      return;
    }
    if (linkPriority(type) > linkPriority(existing.type)) existing.type = type;
  });
  const links = [...linkByPair.values()];

  // Degrees over the LINK graph, which is what d3's own 1/min(degree) default
  // strength is computed from (see linkStrengthFor).
  const linkDegree = new Map<string, number>();
  for (const link of links) {
    const source = linkEndId(link.source);
    const target = linkEndId(link.target);
    linkDegree.set(source, (linkDegree.get(source) ?? 0) + 1);
    linkDegree.set(target, (linkDegree.get(target) ?? 0) + 1);
  }

  const center = groupCenterForce();
  const shelfAnchor = shelfAnchorForce();
  const sim = forceSimulation(nodes)
    .force(
      "link",
      forceLink<AtlasSimNode, AtlasSimLink>(links)
        .id((d) => d.id)
        .distance(linkDistanceFor)
        .strength((link) => linkStrengthFor(link, linkDegree)),
    )
    .force("charge", forceManyBody<AtlasSimNode>().strength(CHARGE_STRENGTH))
    .force("center", center)
    // After link and charge, before collide: the shelf springs cancel the
    // bulk velocity those two just handed a shelved component, and collide
    // then gets the last word on discs that would otherwise overlap.
    .force("shelf", shelfAnchor)
    // Keeps discs off each other.
    .force(
      "collide",
      forceCollide<AtlasSimNode>().radius((d) => d.radius).strength(COLLIDE_STRENGTH).iterations(1),
    )
    .alphaDecay(0.03)
    .velocityDecay(0.25);

  // onTick runs after every position writeback so the caller can PAINT in
  // the same frame the physics stepped. Relying on sigma's graph-event
  // scheduled render instead paints every tick one frame late: d3's timer
  // and sigma's scheduler are separate rAF queues, and a render requested
  // mid-frame only runs on the next one — a constant extra frame of drag
  // latency (the old force-graph loop ticked and painted together).
  let draggingId: string | null = null;
  const writeBack = () => {
    for (const node of nodes) {
      if (node.fx != null && node.fy != null) continue;
      graph.setNodeAttribute(node.id, "x", node.x);
      graph.setNodeAttribute(node.id, "y", node.y);
    }
    // Memories ride their anchor, so they are re-placed after every
    // position writeback — including mid-drag, which is what carries a
    // dragged entity's memories along with it. The one exception is a memory
    // the pointer is holding: that one is being positioned by hand.
    placeSatellites(graph, satellites, draggingId);
    onTick?.();
  };
  sim.on("tick", writeBack);

  // d3's own per-frame loop (driven by restart()) calls its internal tick
  // step directly, bypassing this method — wrapping it only affects manual
  // callers, which keeps sim.tick() synchronous AND graph-synced for tests
  // without double-writing during live, restart()-driven dragging.
  const rawTick = sim.tick.bind(sim);
  sim.tick = (iterations?: number) => {
    rawTick(iterations);
    writeBack();
    return sim;
  };

  // Settle to equilibrium synchronously before first paint (≈ full alpha
  // decay at 0.03 when the budget allows it) — see the doc comment above.
  // Budgeted by the nodes the sim owns, not the graph: memories are not
  // simulated, and counting them would change the settle with the layer.
  sim.tick(settleTicks(nodes.length));

  // Round 5: the settle leaves every small component on roughly one circle
  // around the core (charge pushes out, forceCenter pulls in, and they
  // balance at about the same radius for all of them). Pack them as islands
  // round the core instead (see shelveComponents), then copy those positions
  // back INTO the sim so a later drag flexes the real layout rather than
  // snapping to the ring.
  let placement = shelveComponents(graph);
  for (const node of nodes) {
    node.x = graph.getNodeAttribute(node.id, "x") as number;
    node.y = graph.getNodeAttribute(node.id, "y") as number;
  }

  // Live shelf: instead of freezing every shelved component with fx/fy, open
  // the ones that settled into a tangled knot. Charge and collide inside a
  // component are what spread it; the core is already settled and would only
  // make the pass cost more, so this runs on a shelf-only scratch simulation
  // (see relaxShelf) whose results are copied back onto the sim nodes.
  relaxShelf(nodes, links, placement, linkDegree);
  for (const node of nodes) {
    graph.setNodeAttribute(node.id, "x", node.x);
    graph.setNodeAttribute(node.id, "y", node.y);
  }

  // Relaxing changes a component's bbox (that is the point — a knot spreading
  // out grows its own box), so the row packing above is stale. Re-shelve:
  // rigid translation preserves the shape the relax pass just found, it only
  // restores which row and how much space each component gets.
  placement = shelveComponents(graph);
  for (const node of nodes) {
    node.x = graph.getNodeAttribute(node.id, "x") as number;
    node.y = graph.getNodeAttribute(node.id, "y") as number;
    // The settle and the relax both left velocity behind, and the pack just
    // teleported everything anyway — a later reheat must start from rest
    // rather than resume momentum aimed at positions that no longer exist.
    node.vx = 0;
    node.vy = 0;
  }
  // The core keeps the rigid centroid hold it has always had; every island
  // is held at the slot this pack just gave it by its own springs.
  center.setGroups(placement.slice(0, 1));
  center.retarget();
  shelfAnchor.setPlacement(placement.slice(1));
  // Which nodes are islands, for the reducers (dim at the opening view).
  const islandIds = new Set(placement.slice(1).flat());
  graph.forEachNode((id) => {
    graph.setNodeAttribute(id, "island", islandIds.has(id));
  });

  // Same writeback path as a tick, so satellites re-place around their moved
  // anchors and the caller's onTick marks the cartography scene dirty.
  writeBack();

  sim.alpha(0);
  sim.stop();

  const atlasSim = sim as AtlasSimulation;
  atlasSim.setDraggingId = (id: string | null) => {
    draggingId = id;
  };
  atlasSim.settleShelf = () => {
    shelfAnchor.settle();
    writeBack();
  };
  return atlasSim;
}

export interface HoverState {
  hovered: string | null;
  neighbors: Set<string>;
}

/** Hovered node id plus its neighbor set, or the empty state when nothing is hovered. */
export function hoverStateFor(graph: Graph, hovered: string | null): HoverState {
  if (hovered === null) return { hovered: null, neighbors: new Set() };
  return { hovered, neighbors: new Set(graph.neighbors(hovered)) };
}

// sigma's own nodeReducer/edgeReducer types resolve `data` to graphology's
// permissive `Attributes` (`{[name: string]: any}`) for a default-generic
// Graph, and expect a return assignable to `Partial<NodeDisplayData>` /
// `Partial<EdgeDisplayData>` — neither of which this package exposes without
// pulling in graphology-types as a new direct dependency. `any` matches what
// sigma already passes through, so it typechecks both ways without one.

/** Dim fill for an island at the opening view: the node's own ink at 30%
 *  over the ground. Cached per (ink, ground) pair — the reducer runs per node
 *  per frame. */
const dimFillCache = new Map<string, string>();
function dimFill(color: string, surface: string): string {
  const key = `${color}|${surface}`;
  let dim = dimFillCache.get(key);
  if (dim === undefined) {
    dim = compositeOver(color, surface, 0.3);
    dimFillCache.set(key, dim);
  }
  return dim;
}

/**
 * Node display override for sigma's nodeReducer — pure so it's unit-testable
 * without a sigma renderer. Two concerns, zoom first, then hover:
 *
 * - ZOOM (`lod`). A memory is dust on its anchor: only the first
 *   `lod.dustVisible` of an anchor's memories are drawn (by `dustRank`),
 *   unless the anchor is hovered, which shows the first HOVER_DUST_MAX. An island node (`island`) is drawn dim and nameless until
 *   `lod.islandsSolid`.
 * - HOVER, the round-2 spec: no-hover passthrough, the hovered node itself,
 *   its neighbors, everyone else muted and blanked. Neighbors are also named
 *   (`forceLabel`) when there are at most NEIGHBOR_LABEL_MAX of them: the
 *   whole point of lighting them is to show what the node connects to, and
 *   small nodes never earn a label from sigma's size threshold on their own.
 *   Past the cap the label grid decides, so a hub with dozens of memories
 *   does not turn into a wall of text.
 */
export function nodeDisplay(
  state: HoverState,
  nodeId: string,
  attrs: Record<string, any>,
  palette: GraphPalette,
  lod: LodState = OPENING_LOD,
): Record<string, any> {
  const dustRank = attrs.dustRank as number | undefined;
  if (dustRank !== undefined && dustRank >= lod.dustVisible) {
    const anchorHovered = state.hovered === attrs.dustOf && dustRank < Math.max(lod.dustVisible, HOVER_DUST_MAX);
    if (!anchorHovered && state.hovered !== nodeId) return { ...attrs, hidden: true };
  }
  let base = attrs;
  if (attrs.island && !lod.islandsSolid) {
    base = { ...attrs, color: dimFill(attrs.color as string, palette.surface), label: "" };
  }
  if (state.hovered === null) return base;
  if (nodeId === state.hovered) return { ...attrs, forceLabel: true, zIndex: 2 };
  if (state.neighbors.has(nodeId)) {
    return state.neighbors.size <= NEIGHBOR_LABEL_MAX ? { ...attrs, forceLabel: true, zIndex: 1 } : { ...attrs, zIndex: 1 };
  }
  return { ...base, color: palette.edge, label: "", zIndex: 0 };
}

/**
 * Edge display override for sigma's edgeReducer. A memory's thread to its
 * anchor is drawn whenever the memory itself is (the zoom's `dustVisible`
 * tier, see nodeDisplay), so a shown dot is never loose; its other edges, and
 * a hidden memory's thread, wait for a hover of the memory or either endpoint.
 * An island's edges are hidden while the island is dim. Then the hover rule:
 * edges incident to the hovered node get emphasized, everything else hides.
 */
export function edgeDisplay(
  state: HoverState,
  _edgeId: string,
  source: string,
  target: string,
  attrs: Record<string, any>,
  palette: GraphPalette,
  lod: LodState = OPENING_LOD,
  endpointAttrs?: { source: Record<string, any>; target: Record<string, any> },
): Record<string, any> {
  const incident = state.hovered !== null && (source === state.hovered || target === state.hovered);
  if (incident) return { ...attrs, color: palette.edgeStrong, zIndex: 1 };
  if (state.hovered !== null) return { ...attrs, hidden: true };
  if (endpointAttrs) {
    const { source: s, target: t } = endpointAttrs;
    if (s.dustRank !== undefined || t.dustRank !== undefined) {
      const [dust, dustId, other] = s.dustRank !== undefined ? [s, source, target] : [t, target, source];
      const shown = (dust.dustRank as number) < lod.dustVisible;
      if (!shown || dust.dustOf !== other || dustId === other) return { ...attrs, hidden: true };
    }
    if ((s.island || t.island) && !lod.islandsSolid) return { ...attrs, hidden: true };
  }
  return attrs;
}

/** The node-label typeface: the app's body face, with the system font behind
 *  it for the rare machine that has not loaded it. */
export const NODE_LABEL_FONT = '12px "Instrument Sans", -apple-system, sans-serif';
/** Ground-coloured halo stroke behind a node label, in CSS px — the label
 *  stays legible where it crosses an edge or a neighbouring disc. */
const NODE_LABEL_HALO_WIDTH = 3;

/**
 * Sector-radial label drawer, ported from the old canvas graph: 12px body
 * font at 85% ink, placed left/right/above/below the node by its angle from
 * the graph center (0,0 — forceCenter pins the cluster there) so labels face
 * INWARD toward the cluster instead of expanding the bbox. Graph y is negated
 * for the angle: sigma renders graph +y screen-up while the old canvas
 * rendered it screen-down, and inward placement must track the on-screen
 * quadrant, not the raw coordinate. `halo`, when given, is the ground colour
 * stroked behind the text first. Wired into sigma via
 * settings.defaultDrawNodeLabel (data carries the node key plus viewport
 * x/y/size — see sigma's renderLabels call site).
 */
export function drawRadialNodeLabel(
  context: CanvasRenderingContext2D,
  data: Record<string, any>,
  settings: Record<string, any>,
  graph: Graph,
  halo?: string,
): void {
  if (!data.label) return;
  const pad = (data.size as number) + 8;
  const gx = graph.getNodeAttribute(data.key, "x") as number;
  const gy = graph.getNodeAttribute(data.key, "y") as number;
  const angle = Math.atan2(-gy, gx);
  const sector = Math.round((angle + Math.PI) / (Math.PI / 2)) % 4;

  let x = data.x as number;
  let y = data.y as number;
  if (sector === 0 || sector === 2) {
    const isRight = sector === 0;
    context.textAlign = isRight ? "left" : "right";
    context.textBaseline = "middle";
    x += isRight ? pad : -pad;
  } else {
    const isBelow = sector === 1;
    context.textAlign = "center";
    context.textBaseline = isBelow ? "top" : "bottom";
    y += isBelow ? pad : -pad;
  }

  context.font = NODE_LABEL_FONT;
  context.fillStyle = settings.labelColor?.color ?? "#000000";
  context.globalAlpha = 0.85;
  if (halo) {
    context.lineJoin = "round";
    context.lineWidth = NODE_LABEL_HALO_WIDTH;
    context.strokeStyle = halo;
    context.strokeText(data.label, x, y);
  }
  context.fillText(data.label, x, y);
  context.globalAlpha = 1;
}
