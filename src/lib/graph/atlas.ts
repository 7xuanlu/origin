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
import { nodeFillFor, type GraphPalette } from "./palette";
import { bridgeEdgeTest } from "./cartography";

const MIN_NODE_SIZE = 3;
const MAX_NODE_SIZE = 8;

// The old canvas graph's exact radius scale: a stability base (confirmed 4,
// everything else 3) plus half a px per connection, capped at 8 — finer than
// the sqrt scale it replaced, and size itself encodes confirmation.
function nodeSizeFor(confirmed: boolean | null, degree: number): number {
  const base = confirmed === true ? 4 : MIN_NODE_SIZE;
  return Math.min(MAX_NODE_SIZE, base + degree * 0.5);
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
  communities?: Map<string, number>,
): Graph {
  const graph = new Graph({ multi: true });
  // ponytail: the old graph's confirmed-glow halo (r+3 disc at 0.1 alpha) is
  // skipped — it needs a custom WebGL node program in sigma; the tiered fills
  // and size base carry the confirmed/unconfirmed distinction instead.
  const n = model.nodes.length;
  model.nodes.forEach((node, i) => {
    const angle = (2 * Math.PI * i) / Math.max(n, 1);
    graph.addNode(node.id, {
      label: node.name,
      size: nodeSizeFor(node.confirmed, node.degree),
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
      // Rendered 1:1 in CSS px (AtlasView pins zoomToSizeRatioFunction to 1),
      // calibrated to the old canvas graph's exact stroke: lineWidth 1 at its
      // fixed k=1.499 zoom ≈ 1.5 CSS px. Needs minEdgeThickness lowered in
      // AtlasView — sigma's default floor (1.7) silently bumps this back up.
      size: bridge ? 1.4 : 1.5,
      color: bridge ? palette.bridge : palette.edge,
      bridge,
    });
  }
  return graph;
}

/**
 * Force-directed refinement of the seeded circle, synchronous — fine at
 * today's top-20-cap scale.
 * ponytail: sync FA2, move to a graphology worker when the bulk relations
 * endpoint lands and node counts stop being top-20-capped.
 */
export function runAtlasLayout(graph: Graph): void {
  const iterations = Math.min(600, Math.max(100, graph.order * 6));
  forceAtlas2.assign(graph, { iterations, settings: forceAtlas2.inferSettings(graph) });
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
  const isolates = new Set(isolateIds(graph));
  const nodes: AtlasSimNode[] = [];
  graph.forEachNode((id, attrs) => {
    if (isolates.has(id)) return;
    nodes.push({ id, x: attrs.x as number, y: attrs.y as number });
  });

  const seenPairs = new Set<string>();
  const links: AtlasSimLink[] = [];
  graph.forEachEdge((_edge, _attrs, source, target) => {
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
  // decay at 0.03) — see the doc comment above.
  sim.tick(220);
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
