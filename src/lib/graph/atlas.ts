// SPDX-License-Identifier: AGPL-3.0-only
import Graph from "graphology";
import forceAtlas2 from "graphology-layout-forceatlas2";
import {
  forceSimulation,
  forceLink,
  forceManyBody,
  forceCenter,
  type Simulation,
  type SimulationNodeDatum,
} from "d3-force";
import type { GraphModel } from "./model";
import { MEMORY_NODE_TYPE, PAGE_NODE_TYPE, SHARED_SOURCE_EDGE_TYPE } from "./model";
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
 * CURRENT connected-node bbox at call time — round 5 calls this AFTER the
 * sim settles (see createAtlasSimulation) so the ring tracks the graph's
 * rest-state extent, not FA2's raw seed packing.
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

/**
 * Deterministic orbits for the leaf memories: each sits at its anchor's disc
 * radius plus SATELLITE_GAP, spaced evenly by its index among that anchor's
 * own leaves (sorted by id, so the answer never depends on iteration order).
 * Isolates are skipped — they have no anchor to orbit.
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
    const radius = (graph.getNodeAttribute(anchor, "size") as number) + SATELLITE_GAP;
    const sorted = [...leaves].sort();
    sorted.forEach((id, i) => {
      plan.push({ id, anchor, angle: (2 * Math.PI * i) / sorted.length, radius });
    });
  }
  return plan;
}

/** Write a satellite plan onto the graph. Cheap enough (two trig calls per
 *  leaf) to re-run on every tick writeback, which is what makes a dragged
 *  entity carry its memories along. */
export function placeSatellites(graph: Graph, plan: Satellite[]): void {
  for (const satellite of plan) {
    const ax = graph.getNodeAttribute(satellite.anchor, "x") as number;
    const ay = graph.getNodeAttribute(satellite.anchor, "y") as number;
    graph.setNodeAttribute(satellite.id, "x", ax + satellite.radius * Math.cos(satellite.angle));
    graph.setNodeAttribute(satellite.id, "y", ay + satellite.radius * Math.sin(satellite.angle));
  }
}

export interface AtlasSimNode extends SimulationNodeDatum {
  id: string;
}

interface AtlasSimLink {
  source: string;
  target: string;
}

/** d3-force simulation over the live graphology graph — the interaction engine.
 *  Sim nodes are the CONNECTED subgraph only (degree > 0) — isolates hold
 *  their round-1 ring position structurally (see placeIsolateRing) and are
 *  never simulated. Matches the retired ConstellationMap feel: charge -40,
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
): Simulation<AtlasSimNode, undefined> {
  const excluded = new Set(nonSimulatedIds(graph));
  const satellites = satellitePlan(graph);
  const nodes: AtlasSimNode[] = [];
  graph.forEachNode((id, attrs) => {
    if (excluded.has(id)) return;
    nodes.push({ id, x: attrs.x as number, y: attrs.y as number });
  });

  const seenPairs = new Set<string>();
  const links: AtlasSimLink[] = [];
  graph.forEachEdge((_edge, _attrs, source, target) => {
    // A link to a node the sim doesn't own would make d3 throw looking the
    // endpoint up; a leaf memory's one edge is drawn but never simulated.
    if (excluded.has(source) || excluded.has(target)) return;
    const pairKey = [source, target].sort().join("|");
    if (seenPairs.has(pairKey)) return;
    seenPairs.add(pairKey);
    links.push({ source, target });
  });

  const sim = forceSimulation(nodes)
    .force("link", forceLink<AtlasSimNode, AtlasSimLink>(links).id((d) => d.id))
    .force("charge", forceManyBody<AtlasSimNode>().strength(-40))
    .force("center", forceCenter(0, 0))
    .alphaDecay(0.03)
    .velocityDecay(0.25);

  // onTick runs after every position writeback so the caller can PAINT in
  // the same frame the physics stepped. Relying on sigma's graph-event
  // scheduled render instead paints every tick one frame late: d3's timer
  // and sigma's scheduler are separate rAF queues, and a render requested
  // mid-frame only runs on the next one — a constant extra frame of drag
  // latency (the old force-graph loop ticked and painted together).
  const writeBack = () => {
    for (const node of nodes) {
      if (node.fx != null && node.fy != null) continue;
      graph.setNodeAttribute(node.id, "x", node.x);
      graph.setNodeAttribute(node.id, "y", node.y);
    }
    // Leaf memories ride their anchor, so they are re-placed after every
    // position writeback — including mid-drag, which is what carries a
    // dragged entity's memories along with it.
    placeSatellites(graph, satellites);
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
  sim.alpha(0);
  sim.stop();
  return sim;
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
