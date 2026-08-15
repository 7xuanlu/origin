// SPDX-License-Identifier: AGPL-3.0-only
import Graph from "graphology";
import forceAtlas2 from "graphology-layout-forceatlas2";
import {
  forceSimulation,
  forceLink,
  forceManyBody,
  forceCenter,
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
import { nodeFillFor, type GraphPalette } from "./palette";
import { bridgeEdgeTest } from "./cartography";

const MIN_NODE_SIZE = 3;
const MAX_NODE_SIZE = 12;
/** Wiki pages sit on the same scale as entities. Round 3 pulled the base from
 *  3.5 down to the unconfirmed-entity 3: with shared-source overlap no longer
 *  counting toward size (see buildAtlasGraph), a page reads as a subject
 *  without the whole page layer drawing a size step above the entities. */
const PAGE_NODE_BASE = 3;
/** Memory nodes start below the smallest entity and stay there: they are
 *  context around the subjects, and a memory linked to six entities must not
 *  outgrow the entities themselves. */
const MEMORY_NODE_BASE = 2;
const MEMORY_MAX_SIZE = 4;
/** Px added per doubling of degree. Linear growth (round 1's degree * 0.5)
 *  saturated the cap on real data — 1,600 entities where the hubs run to
 *  hundreds of links drew as one uniform field of 8s. log2 keeps the hubs
 *  visibly bigger while leaving the long tail distinguishable. */
const DEGREE_GAIN = 1.6;

// Base by stability/kind (confirmed 4, unconfirmed 3, page 3, memory 2)
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

/** Edge ink and stroke by verb. Shared-source edges are the lightest thing on
 *  the map (they stand for an inferred overlap, not an asserted link); a
 *  bridge stays amber and a hair thinner, per the artifact. */
export function edgeSizeFor(type: string, bridge: boolean): number {
  if (type === SHARED_SOURCE_EDGE_TYPE) return 0.8;
  return bridge ? 1.4 : 1.5;
}

/**
 * GraphModel -> a graphology instance sigma can render directly. Positions
 * seed on a deterministic circle (node array order), so the result is
 * reproducible without running a layout; runAtlasLayout refines it from
 * there. `multi: true` because GraphModel's parallel-edge policy keeps
 * distinct relations between the same pair as distinct edges (see model.ts)
 * — a simple graph would throw adding the second one.
 */
export function buildAtlasGraph(
  model: GraphModel,
  palette: GraphPalette,
  communities?: Map<string, string>,
): Graph {
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

  const n = model.nodes.length;
  model.nodes.forEach((node, i) => {
    const angle = (2 * Math.PI * i) / Math.max(n, 1);
    const sizingDegree = Math.max(0, node.degree - (sharedSourceIncident.get(node.id) ?? 0));
    graph.addNode(node.id, {
      label: node.name,
      size: nodeSizeFor(node.confirmed, sizingDegree, node.entityType),
      color: nodeFillFor(node.entityType, node.confirmed, palette),
      entityType: node.entityType,
      // Kept on the node so the theme-flip recolor (AtlasView) can recompute
      // the stability-tiered fill without re-reading the model.
      confirmed: node.confirmed,
      x: Math.cos(angle),
      y: Math.sin(angle),
    });
  });

  // ponytail: parallel edges between the same pair draw fully overlapped —
  // a view decision (see model.ts's parallel-edge note), fine at round-1 scale.
  const isBridge = communities ? bridgeEdgeTest(communities) : () => false;
  for (const edge of model.edges) {
    // Cross-region edges are the map's bridges: amber, a hair thinner
    // (the artifact's 1.4 stroke), flagged so the theme-flip recolor keeps
    // them amber. ponytail: the artifact dashes them too — sigma's stock
    // edge programs can't dash; custom WebGL program if the solid amber
    // isn't distinct enough.
    const bridge = isBridge(edge.source, edge.target);
    graph.addEdgeWithKey(edge.id, edge.source, edge.target, {
      // Kept on the edge so the theme-flip and cartography recolors can
      // recompute ink and stroke without re-reading the model.
      edgeType: edge.type,
      // Rendered 1:1 in CSS px (AtlasView pins zoomToSizeRatioFunction to 1),
      // calibrated to the old canvas graph's exact stroke: lineWidth 1 at its
      // fixed k=1.499 zoom ≈ 1.5 CSS px. Needs minEdgeThickness lowered in
      // AtlasView — sigma's default floor (1.7) silently bumps this back up.
      size: edgeSizeFor(edge.type, bridge),
      color: bridge ? palette.bridge : palette.edge,
      bridge,
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

/**
 * Parks degree-0 isolates on a deterministic ring just outside the connected
 * cluster instead of wherever FA2's gravity-only diffusion (or the d3 sim's
 * settle) left them: quiet periphery, honest bbox. Computed from the graph's
 * CURRENT connected-node bbox at call time.
 *
 * TEST-ONLY as of round 3: drawableModel drops every component smaller than
 * MIN_COMPONENT_SIZE before the layout runs, so a degree-0 isolate never
 * reaches the map and there is no ring left to park. Nothing in the render
 * path calls this; it is kept because the ring geometry is still the
 * documented answer if isolates are ever drawn again.
 */
export function placeIsolateRing(graph: Graph): void {
  const isolates = isolateIds(graph);
  const isolateSet = new Set(isolates);
  let minX = Infinity;
  let maxX = -Infinity;
  let minY = Infinity;
  let maxY = -Infinity;
  graph.forEachNode((id, attrs) => {
    if (isolateSet.has(id)) return;
    minX = Math.min(minX, attrs.x as number);
    maxX = Math.max(maxX, attrs.x as number);
    minY = Math.min(minY, attrs.y as number);
    maxY = Math.max(maxY, attrs.y as number);
  });
  // No isolates, or nothing BUT isolates (the seed circle is already fine).
  if (isolates.length === 0 || minX === Infinity) return;
  const cx = (minX + maxX) / 2;
  const cy = (minY + maxY) / 2;
  const radius = Math.max(maxX - minX, maxY - minY, 1) * 0.65;
  isolates.forEach((id, i) => {
    const angle = (2 * Math.PI * i) / isolates.length;
    graph.setNodeAttribute(id, "x", cx + radius * Math.cos(angle));
    graph.setNodeAttribute(id, "y", cy + radius * Math.sin(angle));
  });
}

/** Degree-0 node ids — the round-1 isolate ring that gravity would otherwise
 *  pull inward during a live layout step. */
export function isolateIds(graph: Graph): string[] {
  const isolates: string[] = [];
  graph.forEachNode((id) => {
    if (graph.degree(id) === 0) isolates.push(id);
  });
  return isolates;
}

/** Graph-space clearance between a leaf memory and the disc it orbits. */
const SATELLITE_GAP = 6;

/**
 * Nodes no force layout runs on: degree-0 isolates (round 1's ring) plus every
 * degree-1 MEMORY, which round 3 hangs off its one neighbour as a satellite
 * instead. On real data 1,308 of 1,865 memories are such leaves; simulating
 * them buys nothing — a single link has exactly one rest position — and it
 * doubles the cost of every layout step.
 */
export function nonSimulatedIds(graph: Graph): string[] {
  const ids: string[] = [];
  graph.forEachNode((id, attrs) => {
    const degree = graph.degree(id);
    if (degree === 0 || (degree === 1 && attrs.entityType === MEMORY_NODE_TYPE)) ids.push(id);
  });
  return ids;
}

/** Where one leaf memory sits relative to the node it hangs off. */
export interface Satellite {
  id: string;
  anchor: string;
  angle: number;
  radius: number;
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
 * Deterministic orbits for the leaf memories: each hangs off its one
 * neighbour, sorted by id so the answer never depends on iteration order.
 * Isolates are skipped — they have no anchor to orbit.
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
    const anchor = graph.neighbors(id)[0];
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

/** Graph-space clearance between two packed components, and between the core
 *  and the first ring of them. Big enough that the gap reads as a gap at
 *  fit-to-screen zoom, small enough that the cloud still hugs the core. */
export const PACK_GAP = 24;
/** How much further out the greedy search pushes a component each time the
 *  current spiral radius still collides with something already placed. */
export const PACK_STEP = 12;
/** 137.5 degrees in radians — the phyllotaxis angle. Successive components
 *  land on different sides of the core instead of stacking along one arm. */
const GOLDEN_ANGLE = Math.PI * (3 - Math.sqrt(5));

interface PackedComponent {
  ids: string[];
  cx: number;
  cy: number;
  /** Distance from the component's bbox centre to its furthest node EDGE. */
  radius: number;
}

/** Connected components over the simulated nodes only, in graph node order. */
function simulatedComponents(graph: Graph, simulated: Set<string>): string[][] {
  const parent = new Map<string, string>();
  for (const id of simulated) parent.set(id, id);
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
    if (!simulated.has(source) || !simulated.has(target)) return;
    const a = find(source);
    const b = find(target);
    if (a !== b) parent.set(a, b);
  });
  const groups = new Map<string, string[]>();
  for (const id of simulated) {
    const root = find(id);
    const list = groups.get(root);
    if (list) list.push(id);
    else groups.set(root, [id]);
  }
  return [...groups.values()];
}

/** How far each anchor's satellite halo reaches past the anchor's own centre:
 *  the outermost shell radius plus that leaf's disc. Leaves are not simulated,
 *  so they are invisible to the component bbox unless it is told about them —
 *  and a 374-leaf halo reaches ~130 units, far more than PACK_GAP covers. */
function satelliteReach(plan: Satellite[], sizeOf: (id: string) => number): Map<string, number> {
  const reach = new Map<string, number>();
  for (const satellite of plan) {
    const outer = satellite.radius + sizeOf(satellite.id);
    reach.set(satellite.anchor, Math.max(reach.get(satellite.anchor) ?? 0, outer));
  }
  return reach;
}

function measureComponent(
  graph: Graph,
  ids: string[],
  reach: Map<string, number>,
): PackedComponent {
  let minX = Infinity;
  let maxX = -Infinity;
  let minY = Infinity;
  let maxY = -Infinity;
  for (const id of ids) {
    const x = graph.getNodeAttribute(id, "x") as number;
    const y = graph.getNodeAttribute(id, "y") as number;
    minX = Math.min(minX, x);
    maxX = Math.max(maxX, x);
    minY = Math.min(minY, y);
    maxY = Math.max(maxY, y);
  }
  const cx = (minX + maxX) / 2;
  const cy = (minY + maxY) / 2;
  let radius = 0;
  for (const id of ids) {
    const x = graph.getNodeAttribute(id, "x") as number;
    const y = graph.getNodeAttribute(id, "y") as number;
    const size = (graph.getNodeAttribute(id, "size") as number) ?? 0;
    radius = Math.max(radius, Math.hypot(x - cx, y - cy) + Math.max(size, reach.get(id) ?? 0));
  }
  return { ids, cx, cy, radius };
}

/**
 * Round 4, section A. After the sim settles, charge repulsion plus
 * forceCenter has pushed every small component out to roughly the same
 * distance from the core — twenty-odd islands on one circle, which reads as
 * the round-1 ring of dots at a coarser grain.
 *
 * This rearranges them: the biggest component is recentred on the origin and
 * keeps the layout the sim gave it, and every other component is translated
 * RIGIDLY (never re-laid-out) onto a phyllotaxis spiral around it. Each one
 * starts at the smallest radius that clears the core and is pushed out in
 * PACK_STEP increments until its bounding disc clears every component already
 * placed, so no two overlap. Order is size-descending with an id tie-break,
 * so the same graph always packs the same way.
 *
 * A component's measured radius covers its satellite halo too (see
 * satelliteReach): leaves are not simulated, so a bbox over sim nodes alone
 * would let one component's halo sit inside its neighbour's gap.
 *
 * Returns the node ids of each component in placement order — largest first,
 * so the caller can tell the core from the packed islands.
 *
 * Pure in the sense that matters here: it reads x/y/size off the graph and
 * writes x/y back, touching nothing else and consulting no clock or random.
 */
export function packComponents(graph: Graph, simulatedIds: string[]): string[][] {
  const simulated = new Set(simulatedIds);
  if (simulated.size === 0) return [];
  const reach = satelliteReach(satellitePlan(graph), (id) =>
    (graph.getNodeAttribute(id, "size") as number) ?? 0,
  );
  const components = simulatedComponents(graph, simulated)
    .map((ids) => measureComponent(graph, ids, reach))
    .sort((a, b) => b.ids.length - a.ids.length || (a.ids[0] < b.ids[0] ? -1 : 1));

  const translate = (component: PackedComponent, tx: number, ty: number) => {
    const dx = tx - component.cx;
    const dy = ty - component.cy;
    for (const id of component.ids) {
      graph.setNodeAttribute(id, "x", (graph.getNodeAttribute(id, "x") as number) + dx);
      graph.setNodeAttribute(id, "y", (graph.getNodeAttribute(id, "y") as number) + dy);
    }
    component.cx = tx;
    component.cy = ty;
  };

  const core = components[0];
  translate(core, 0, 0);
  const placed: PackedComponent[] = [core];

  for (let i = 1; i < components.length; i += 1) {
    const component = components[i];
    const angle = i * GOLDEN_ANGLE;
    let r = core.radius + PACK_GAP + component.radius;
    let x = 0;
    let y = 0;
    // Greedy: step outward along this arm until the component's disc clears
    // every disc already placed. Bounded by the loop over `placed`, which is
    // at most a couple of dozen components on real data.
    for (;;) {
      x = r * Math.cos(angle);
      y = r * Math.sin(angle);
      const clash = placed.some(
        (other) => Math.hypot(x - other.cx, y - other.cy) < other.radius + component.radius + PACK_GAP,
      );
      if (!clash) break;
      r += PACK_STEP;
    }
    translate(component, x, y);
    placed.push(component);
  }
  return placed.map((component) => component.ids);
}

/** The sim plus the one thing the view has to tell it: which node the pointer
 *  is holding right now. Satellites are not sim nodes, so the writeback needs
 *  to be told to leave the dragged one alone (see placeSatellites). */
export interface AtlasSimulation extends Simulation<AtlasSimNode, undefined> {
  setDraggingId(id: string | null): void;
}

export interface AtlasSimNode extends SimulationNodeDatum {
  id: string;
  /** Collision radius: the drawn disc, plus a page's ring, plus COLLIDE_PAD. */
  radius: number;
}

interface AtlasSimLink {
  source: string;
  target: string;
  /** The graph edge's verb, which sets this link's rest length and pull. */
  type: string;
}

/** Breathing room between two discs that are not otherwise pushed apart. Two
 *  px is enough to stop the overlap that made page clusters read as one blob
 *  without visibly loosening the rest of the map. */
const COLLIDE_PAD = 2;
/** Rest length and pull per edge verb; anything not listed keeps d3's own
 *  defaults (distance 30, strength 1/min(degree)). A shared-source edge is an
 *  inferred overlap, so it sits long and slack — it should suggest that two
 *  pages are near each other, not staple them together. A wikilink is an
 *  asserted link between pages, so it is shorter and firmer, but still longer
 *  than a relation because page discs carry a ring. */
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

/** d3-force simulation over the live graphology graph — the interaction engine.
 *  Sim nodes are the drawable graph MINUS the nodes nonSimulatedIds names:
 *  degree-0 isolates and every degree-1 memory. Neither is simulated —
 *  isolates are not drawn at all once drawableModel drops components under
 *  MIN_COMPONENT_SIZE, and a leaf memory rides its anchor as a satellite.
 *  Matches the retired ConstellationMap feel: charge -40,
 *  forceCenter(0, 0), alphaDecay 0.03, velocityDecay 0.25, d3-default link
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
  const nodes: AtlasSimNode[] = [];
  graph.forEachNode((id, attrs) => {
    if (excluded.has(id)) return;
    // A page is drawn as a disc plus a detached ring, so its footprint is
    // wider than its `size` — collide against the ring or the rings overlap.
    const ring =
      attrs.entityType === PAGE_NODE_TYPE ? PAGE_RING_GAP + PAGE_RING_WIDTH : 0;
    nodes.push({
      id,
      x: attrs.x as number,
      y: attrs.y as number,
      radius: (attrs.size as number) + ring + COLLIDE_PAD,
    });
  });

  const linkByPair = new Map<string, AtlasSimLink>();
  graph.forEachEdge((_edge, attrs, source, target) => {
    // A link to a node the sim doesn't own would make d3 throw looking the
    // endpoint up; a leaf memory's one edge is drawn but never simulated.
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

  // d3's default link strength is 1/min(degree) over the LINK graph, computed
  // privately inside forceLink — overriding .strength() would throw that away
  // for every verb, so the same formula is rebuilt here and used for anything
  // LINK_LAYOUT does not name.
  const linkDegree = new Map<string, number>();
  for (const link of links) {
    linkDegree.set(link.source, (linkDegree.get(link.source) ?? 0) + 1);
    linkDegree.set(link.target, (linkDegree.get(link.target) ?? 0) + 1);
  }

  const sim = forceSimulation(nodes)
    .force(
      "link",
      forceLink<AtlasSimNode, AtlasSimLink>(links)
        .id((d) => d.id)
        .distance((link) => LINK_LAYOUT[link.type]?.distance ?? 30)
        .strength((link) => {
          const named = LINK_LAYOUT[link.type];
          if (named) return named.strength;
          const source = linkDegree.get(linkEndId(link.source)) ?? 1;
          const target = linkDegree.get(linkEndId(link.target)) ?? 1;
          return 1 / Math.min(source, target);
        }),
    )
    .force("charge", forceManyBody<AtlasSimNode>().strength(-40))
    .force("center", forceCenter(0, 0))
    // Keeps discs off each other. Strength 0.7 (not 1) so the collision
    // resolves over a few ticks instead of snapping, which reads calmer.
    .force("collide", forceCollide<AtlasSimNode>().radius((d) => d.radius).strength(0.7).iterations(1))
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
    // Leaf memories ride their anchor, so they are re-placed after every
    // position writeback — including mid-drag, which is what carries a
    // dragged entity's memories along with it. The one exception is a leaf
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
  sim.tick(settleTicks(graph.order));

  // Round 4: the settle leaves every small component on roughly one circle
  // around the core (charge pushes out, forceCenter pulls in, and they
  // balance at about the same radius for all of them). Pack them into a
  // spiral instead, then copy the packed positions back INTO the sim so a
  // later drag flexes the packed layout rather than snapping to the ring.
  const packedComponents = packComponents(graph, nodes.map((node) => node.id));
  // Everything outside the biggest component is PINNED where the pack put it.
  // Without this the first drag undoes the packing: downNode reheats the sim
  // to alpha 0.3 and charge plus forceCenter push the islands straight back
  // out to the one ring the pack just broke up, and nothing re-packs them.
  // An island has no edge into the core, so it has nothing to flex toward
  // anyway — holding it still costs the layout nothing.
  const core = new Set(packedComponents[0] ?? []);
  for (const node of nodes) {
    node.x = graph.getNodeAttribute(node.id, "x") as number;
    node.y = graph.getNodeAttribute(node.id, "y") as number;
    if (!core.has(node.id)) {
      node.fx = node.x;
      node.fy = node.y;
    }
  }
  // Same writeback path as a tick, so satellites re-place around their moved
  // anchors and the caller's onTick marks the cartography scene dirty.
  writeBack();

  sim.alpha(0);
  sim.stop();

  const atlasSim = sim as AtlasSimulation;
  atlasSim.setDraggingId = (id: string | null) => {
    draggingId = id;
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

/**
 * Node display override for the hover reducer — pure so it's unit-testable
 * without a sigma renderer. Four cases, pinned by the round-2 spec: no-hover
 * passthrough, the hovered node itself, its neighbors, everyone else.
 */
export function nodeDisplay(
  state: HoverState,
  nodeId: string,
  attrs: Record<string, any>,
  palette: GraphPalette,
): Record<string, any> {
  if (state.hovered === null) return attrs;
  if (nodeId === state.hovered) return { ...attrs, forceLabel: true, zIndex: 2 };
  if (state.neighbors.has(nodeId)) return { ...attrs, zIndex: 1 };
  return { ...attrs, color: palette.edge, label: "", zIndex: 0 };
}

/**
 * Edge display override for the hover reducer — edges incident to the
 * hovered node get emphasized, everything else hides.
 */
export function edgeDisplay(
  state: HoverState,
  _edgeId: string,
  source: string,
  target: string,
  attrs: Record<string, any>,
  palette: GraphPalette,
): Record<string, any> {
  if (state.hovered === null) return attrs;
  if (source === state.hovered || target === state.hovered) {
    return { ...attrs, color: palette.edgeStrong, zIndex: 1 };
  }
  return { ...attrs, hidden: true };
}

/**
 * Sector-radial label drawer, ported verbatim from the old canvas graph:
 * 12px system font at 85% ink, placed left/right/above/below the node by its
 * angle from the graph center (0,0 — forceCenter pins the cluster there) so
 * labels face INWARD toward the cluster instead of expanding the bbox.
 * Graph y is negated for the angle: sigma renders graph +y screen-up while
 * the old canvas rendered it screen-down, and inward placement must track
 * the on-screen quadrant, not the raw coordinate. Wired into sigma via
 * settings.defaultDrawNodeLabel (data carries the node key plus viewport
 * x/y/size — see sigma's renderLabels call site).
 */
export function drawRadialNodeLabel(
  context: CanvasRenderingContext2D,
  data: Record<string, any>,
  settings: Record<string, any>,
  graph: Graph,
): void {
  if (!data.label) return;
  const pad = (data.size as number) + 8;
  const gx = graph.getNodeAttribute(data.key, "x") as number;
  const gy = graph.getNodeAttribute(data.key, "y") as number;
  const angle = Math.atan2(-gy, gx);
  const sector = Math.round((angle + Math.PI) / (Math.PI / 2)) % 4;

  context.font = "12px -apple-system, sans-serif";
  context.fillStyle = settings.labelColor?.color ?? "#000000";
  context.globalAlpha = 0.85;
  if (sector === 0 || sector === 2) {
    const isRight = sector === 0;
    context.textAlign = isRight ? "left" : "right";
    context.textBaseline = "middle";
    context.fillText(data.label, data.x + (isRight ? pad : -pad), data.y);
  } else {
    const isBelow = sector === 1;
    context.textAlign = "center";
    context.textBaseline = isBelow ? "top" : "bottom";
    context.fillText(data.label, data.x, data.y + (isBelow ? pad : -pad));
  }
  context.globalAlpha = 1;
}

/** Ring stroke width in CSS px. Thinned from 1.5 in round 3: at real page
 *  density the fatter ring merged neighbouring pages into one teal mass. */
const PAGE_RING_WIDTH = 1;
/** Clear air between the disc and its ring — without the gap the ring reads as
 *  a slightly fatter dot rather than a different kind of thing. Tightened from
 *  3 for the same density reason as the stroke above. */
const PAGE_RING_GAP = 2;

/**
 * The halo ring that marks a wiki page. Sigma v3.0.3 ships only disc node
 * programs (`node-circle` / `node-point`, both borderless), and no `@sigma/*`
 * package is installed, so a square marker would mean either a new dependency
 * or a hand-written WebGL program — neither was in scope. The ring is drawn on
 * a plain 2D canvas stacked ABOVE sigma instead, the same technique the
 * cartography underlay uses below it. `positions` are viewport CSS px.
 */
export function drawPageRings(
  ctx: CanvasRenderingContext2D,
  positions: { x: number; y: number; size: number }[],
  palette: GraphPalette,
): void {
  if (positions.length === 0) return;
  ctx.save();
  ctx.strokeStyle = palette.page;
  ctx.lineWidth = PAGE_RING_WIDTH;
  for (const { x, y, size } of positions) {
    ctx.beginPath();
    ctx.arc(x, y, size + PAGE_RING_GAP, 0, 2 * Math.PI);
    ctx.stroke();
  }
  ctx.restore();
}
