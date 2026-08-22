// SPDX-License-Identifier: AGPL-3.0-only
import { useEffect, useMemo, useRef, useState } from "react";
import type { KeyboardEvent as ReactKeyboardEvent } from "react";
import { useQuery } from "@tanstack/react-query";
import { useTranslation } from "react-i18next";
import Graph from "graphology";
import Sigma from "sigma";
import { getKnowledgeGraph } from "../../lib/tauri";
import type { Entity, KnowledgeGraph } from "../../lib/tauri";
import {
  buildKnowledgeGraphModel,
  entitySpace,
  filterKnowledgeGraph,
  memorySourceId,
  pageIdOf,
  drawableModel,
  smallGroupNodeCount,
  DEFAULT_LAYERS,
  MEMORY_NODE_TYPE,
  PAGE_NODE_TYPE,
} from "../../lib/graph/model";
import type { GraphLayers, GraphModel, GraphNode } from "../../lib/graph/model";
import {
  buildAtlasGraph,
  runAtlasLayout,
  createAtlasSimulation,
  hoverStateFor,
  nodeDisplay,
  edgeDisplay,
  drawRadialNodeLabel,
} from "../../lib/graph/atlas";
import type { HoverState, AtlasSimulation } from "../../lib/graph/atlas";
import type { CartographyScene } from "../../lib/graph/cartography";
import {
  communitiesFor,
  cartographyScene,
  drawRegionNames,
  isUnscopedSpace,
  MIN_REGION_SIZE,
} from "../../lib/graph/cartography";
import { useGraphPalette, colorForEntityType, nodeFillFor } from "../../lib/graph/palette";
import type { GraphPalette } from "../../lib/graph/palette";
import { fetchCartographyForSpaces, aggregateCartographyStatus } from "../../lib/graph/community";
import type { SpaceCartography } from "../../lib/graph/community";

// One shared empty map for the unresolved query. An inline `new Map()` default
// mints a fresh identity on every render, and this map feeds the memoized
// community climb and the place-name overlay — a new identity re-runs the climb
// and repaints every edge each render until the fetch lands.
const EMPTY_CARTOGRAPHY: Map<string, SpaceCartography> = new Map();

// Same reason as EMPTY_CARTOGRAPHY: a stable identity for the unresolved
// graph query, so the model memo below doesn't rebuild on every render.
const EMPTY_GRAPH: KnowledgeGraph = {
  entities: [],
  relations: [],
  memories: [],
  memory_links: [],
  pages: [],
  page_links: [],
};

/** Where the layer choice survives a reload. */
const LAYERS_STORAGE_KEY = "atlas.layers";

/** Where the small-groups choice is remembered across reloads. */
export const SMALL_GROUPS_STORAGE_KEY = "atlas.smallGroups";

/** Read the persisted small-groups choice. Only a stored `true` turns them
 *  on; anything malformed — bad JSON, a number, a string — leaves them
 *  hidden, which is the default the map is designed around. */
export function readStoredSmallGroups(raw: string | null): boolean {
  if (raw === null) return false;
  try {
    return JSON.parse(raw) === true;
  } catch {
    return false;
  }
}

/** Read the persisted layer choice. Anything malformed — bad JSON, a
 *  non-object, a non-boolean field, or all three off — falls back to the
 *  default rather than half-applying a broken value. */
export function readStoredLayers(raw: string | null): GraphLayers {
  if (raw === null) return DEFAULT_LAYERS;
  try {
    const parsed: unknown = JSON.parse(raw);
    if (typeof parsed !== "object" || parsed === null) return DEFAULT_LAYERS;
    const record = parsed as Record<string, unknown>;
    const layers: GraphLayers = {
      entity: record.entity,
      page: record.page,
      memory: record.memory,
    } as GraphLayers;
    if (
      typeof layers.entity !== "boolean" ||
      typeof layers.page !== "boolean" ||
      typeof layers.memory !== "boolean"
    ) {
      return DEFAULT_LAYERS;
    }
    if (!layers.entity && !layers.page && !layers.memory) return DEFAULT_LAYERS;
    return layers;
  } catch {
    return DEFAULT_LAYERS;
  }
}

// Same 5-slot legend as the retired canvas graph (ConstellationMap): place,
// event, and unknown types fold to neutral and get no swatch; concept is
// labeled "Theme" to match the product copy.
const LEGEND_ITEMS: { label: string; key: string }[] = [
  { label: "Project", key: "project" },
  { label: "Technology", key: "technology" },
  { label: "Organization", key: "organization" },
  { label: "Person", key: "person" },
  { label: "Theme", key: "concept" },
  { label: "Wiki page", key: PAGE_NODE_TYPE },
  { label: "Memory", key: MEMORY_NODE_TYPE },
];

/** What a click on a drawn node resolves to. Memories carry their own
 *  `source_id`, not the prefixed graph-node id. */
export type AtlasNodeTarget =
  | { kind: "entity"; id: string }
  | { kind: "memory"; id: string }
  | { kind: "page"; id: string };

/** The node id a click landed on, resolved to a navigable target. */
function targetForNode(nodeId: string): AtlasNodeTarget {
  const sourceId = memorySourceId(nodeId);
  if (sourceId !== null) return { kind: "memory", id: sourceId };
  const pageId = pageIdOf(nodeId);
  if (pageId !== null) return { kind: "page", id: pageId };
  return { kind: "entity", id: nodeId };
}

interface AtlasViewProps {
  onNodeClick?: (target: AtlasNodeTarget) => void;
  // Initial framing: center this entity with its neighborhood emphasized
  // (EntityDetail's overlay "Atlas" mode). Applied instantly on mount — a
  // starting frame, not a transition — so no camera animation.
  focusEntityId?: string;
  // Main.tsx's Graph view passes navigateBack; renders a back button as the
  // first toolbar item (a floating one would sit on the search box).
  onBack?: () => void;
}

// jsdom has no matchMedia; treat its absence as "no preference" rather than
// throwing (see the mouseup wiring below, which is exercised by tests that
// don't stub it).
function prefersReducedMotion(): boolean {
  if (typeof window === "undefined" || typeof window.matchMedia !== "function") return false;
  return window.matchMedia("(prefers-reduced-motion: reduce)").matches;
}

/**
 * sigma-rendered whole-graph view — the shipped Graph tab (Main.tsx) and the
 * entity overlay's "Atlas" mode. Replaced the canvas ConstellationMap; the
 * query keys keep the "constellation-" prefix so nothing else invalidates.
 */
export default function AtlasView({ onNodeClick, focusEntityId, onBack }: AtlasViewProps) {
  const { t } = useTranslation();
  const palette = useGraphPalette();
  const containerRef = useRef<HTMLDivElement>(null);
  const sigmaRef = useRef<Sigma | null>(null);
  const graphRef = useRef<Graph | null>(null);
  const simRef = useRef<AtlasSimulation | null>(null);
  // Reducer inputs, read from refs so hover/theme changes repaint without a
  // React re-render or a renderer rebuild (see the mount effect below).
  const hoverStateRef = useRef<HoverState>({ hovered: null, neighbors: new Set() });
  const paletteRef = useRef<GraphPalette>(palette);
  // Node-drag state: which node (if any) is being dragged, and whether the
  // pointer actually moved during the current press — the latter gates
  // clickNode so a drag-release doesn't also fire entity navigation.
  const draggedNodeRef = useRef<string | null>(null);
  const movedDuringPressRef = useRef(false);
  // Cached cartography scene (the named regions). Rebuilding it
  // on every afterRender meant a plain camera pan or a hover re-measured all
  // 66 regions; the scene only actually changes when node positions or
  // communities do, so paints mark it dirty and the afterRender handler
  // rebuilds only then.
  const sceneRef = useRef<CartographyScene | null>(null);
  const sceneDirtyRef = useRef(true);

  // ONE read for the whole graph. This used to be two queries — every entity,
  // then a detail fetch per entity capped at the first 20 — which drew every
  // connected entity outside that top 20 as an isolate.
  const {
    data: graph = EMPTY_GRAPH,
    isLoading: graphLoading,
    isError: graphError,
    refetch: refetchGraph,
  } = useQuery({
    queryKey: ["knowledge-graph"],
    queryFn: () => getKnowledgeGraph(),
    refetchInterval: 120_000,
  });
  const entities = graph.entities;

  // Same field precedence as the entity page (EntityDetail's space-then-domain
  // rule), through model.ts's entitySpace so the list, the filter, and the
  // graph nodes cannot disagree about which space an entity belongs to.
  const spaces = useMemo(
    () =>
      Array.from(
        new Set(entities.map((e: Entity) => entitySpace(e)).filter((s): s is string => !!s)),
      ).sort(),
    [entities],
  );
  const [spaceFilter, setSpaceFilter] = useState<string | null>(null);

  // Which node kinds are drawn. Wiki pages and entities on, memories off by
  // default: memories outnumber everything else and bury the map. Persisted
  // across reloads; a malformed stored value falls back to the default.
  const [layers, setLayers] = useState<GraphLayers>(() => {
    if (typeof window === "undefined") return DEFAULT_LAYERS;
    try {
      return readStoredLayers(window.localStorage.getItem(LAYERS_STORAGE_KEY));
    } catch {
      return DEFAULT_LAYERS;
    }
  });
  const toggleLayer = (key: keyof GraphLayers) => {
    const next = { ...layers, [key]: !layers[key] };
    // The last lit chip can't be turned off — an empty map is not a view.
    if (!next.entity && !next.page && !next.memory) return;
    setLayers(next);
    try {
      window.localStorage.setItem(LAYERS_STORAGE_KEY, JSON.stringify(next));
    } catch {
      // Private mode / quota: the choice still applies for this session.
    }
  };
  // Components of fewer than MIN_COMPONENT_SIZE nodes are off the map until
  // the reader asks for them; the chip below is the ask. Persisted like the
  // layer choice, and a malformed stored value falls back to hidden.
  const [showSmallGroups, setShowSmallGroups] = useState<boolean>(() => {
    if (typeof window === "undefined") return false;
    try {
      return readStoredSmallGroups(window.localStorage.getItem(SMALL_GROUPS_STORAGE_KEY));
    } catch {
      return false;
    }
  });
  const toggleSmallGroups = () => {
    const next = !showSmallGroups;
    setShowSmallGroups(next);
    try {
      window.localStorage.setItem(SMALL_GROUPS_STORAGE_KEY, JSON.stringify(next));
    } catch {
      // Private mode / quota: the choice still applies for this session.
    }
  };

  const onlyLayerOn = (key: keyof GraphLayers) =>
    layers[key] && Object.values(layers).filter(Boolean).length === 1;

  // D13/App-PR readiness, one fetch-and-classify per known space (community.ts):
  // cursor-paginated to exhaustion, generation-checked, never declared ready off
  // a partial read. Keyed on ALL known spaces regardless of spaceFilter, so
  // one space's trouble stays visible while another is being viewed; the
  // unscoped half of the badge is read off the filtered model instead (below).
  const { data: cartographyBySpace = EMPTY_CARTOGRAPHY } = useQuery({
    queryKey: ["constellation-cartography", spaces],
    queryFn: () => fetchCartographyForSpaces(spaces),
    enabled: spaces.length > 0,
    refetchInterval: 120_000,
  });
  // Scoping filters the model INPUTS: the space's own entities and memories,
  // and only the relations/links whose endpoints both survive. Regions,
  // counts, and insights all re-derive from the scoped model.
  const model = useMemo(
    () => buildKnowledgeGraphModel(filterKnowledgeGraph(graph, spaceFilter), { layers }),
    [graph, spaceFilter, layers],
  );

  // What sigma actually draws. Nodes in a connected component smaller than
  // MIN_COMPONENT_SIZE are left out (model.ts) unless the reader turns them
  // on: on real data ~960 of 1,600 entities have no relation at all, and
  // another ~196 sit in small groups of four or fewer. Both buried the map.
  // The count is shown on the toolbar chip that turns them back on.
  // Overlay entry point: a focused entity that sits in a small group would
  // otherwise vanish with it and the mount focus below would silently not
  // fire, so the small groups are drawn for that mount whatever the chip says.
  const visibleModel = useMemo<GraphModel>(() => {
    const drawable = drawableModel(model, showSmallGroups);
    if (
      focusEntityId &&
      !drawable.nodes.some((n) => n.id === focusEntityId) &&
      model.nodes.some((n) => n.id === focusEntityId)
    ) {
      return drawableModel(model, true);
    }
    return drawable;
  }, [model, showSmallGroups, focusEntityId]);
  // Counted off the FULL model, so the chip keeps its number when the groups
  // are showing and can offer to hide them again.
  const smallGroupCount = useMemo(() => smallGroupNodeCount(model), [model]);

  // Anything in cartography.ts's unscoped bucket is drawn on the fallback
  // climb, so the badge must never read all-durable while such a node is on
  // the map. Asked of the RENDERED model — the very nodes communitiesFor
  // partitions — so the badge and the drawn cartography cannot disagree.
  // Reading the raw entity list instead misses the case the model CREATES
  // rather than carries: under a space filter a relation to another space's
  // entity keeps its endpoint while that entity is filtered away, so
  // buildGraphModel synthesizes it with no space at all. Going through the
  // model also covers the two unfiltered ways in — a relation-only neighbor,
  // and an entity whose own space is null or empty.
  // Memory and wiki-page nodes are exempt: both inherit their community from
  // an entity (cartography.ts), so a spaceless one is not a node drawn on the
  // fallback climb and must not hold the badge back.
  const hasUnscopedFallback = useMemo(
    () =>
      model.nodes.some(
        (n: GraphNode) => n.kind === "entity" && isUnscopedSpace(n.space),
      ),
    [model],
  );
  const cartographyStatus = useMemo(
    () => aggregateCartographyStatus(cartographyBySpace, hasUnscopedFallback),
    [cartographyBySpace, hasUnscopedFallback],
  );
  const communities = useMemo(
    () => communitiesFor(model, cartographyBySpace),
    [model, cartographyBySpace],
  );
  // Mirrors `communities` for the mount effect's afterRender closure (see
  // the cartography-refresh effect below, by the theme-flip effect) — a
  // space's durable status arriving or regressing must repaint the place
  // names without tearing down the sim/camera, so the paint reads this
  // ref at PAINT time instead of closing over the `communities` value that
  // was current when the sigma renderer was built.
  const communitiesRef = useRef<Map<string, string>>(communities);

  // Region count for the toolbar count line — membership only, so it agrees
  // with the regions the cartography scene actually names without needing
  // node positions. Counted over the DRAWN nodes for the same reason:
  // communities come from the full model, but a hidden small group's
  // community is not on the map.
  const regionCount = useMemo(() => {
    const groups = new Map<string, GraphNode[]>();
    for (const node of visibleModel.nodes) {
      const community = communities.get(node.id);
      if (community === undefined) continue;
      const list = groups.get(community);
      if (list) list.push(node);
      else groups.set(community, [node]);
    }
    let count = 0;
    for (const members of groups.values()) {
      if (members.length >= MIN_REGION_SIZE) count += 1;
    }
    return count;
  }, [visibleModel, communities]);

  // Toolbar search (artifact screen 01): type → listbox of entity names,
  // Enter/click → camera fly + the same emphasis hover applies.
  const searchInputRef = useRef<HTMLInputElement>(null);
  const [query, setQuery] = useState("");
  const [activeIndex, setActiveIndex] = useState(0);
  const [searchFocused, setSearchFocused] = useState(false);
  // Regions on/off governs only the place-name overlay canvas; the count line
  // keeps reporting regions either way.
  // Ref mirror so the sigma mount effect (which recreates the overlay per
  // model) can apply the current choice without re-running on toggle.
  const [showRegions, setShowRegions] = useState(true);
  const showRegionsRef = useRef(true);
  const overlayRef = useRef<HTMLCanvasElement | null>(null);

  const matches = useMemo(() => {
    const needle = query.trim().toLowerCase();
    if (!needle) return [];
    return model.nodes.filter((node) => node.name.toLowerCase().includes(needle)).slice(0, 8);
  }, [model, query]);

  // ⌘K / Ctrl+K jumps to the search box from anywhere in the window.
  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") {
        e.preventDefault();
        searchInputRef.current?.focus();
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, []);

  const focusEntity = (nodeId: string) => {
    setQuery("");
    setActiveIndex(0);
    searchInputRef.current?.blur();
    const renderer = sigmaRef.current;
    const drawn = graphRef.current;
    // A degree-0 node is not on the map at all (see visibleModel), but search
    // can still name it, so focusing opens the node's own page rather than
    // flying the camera to nothing.
    if (!drawn || !drawn.hasNode(nodeId)) {
      if (model.nodes.some((node) => node.id === nodeId)) onNodeClick?.(targetForNode(nodeId));
      return;
    }
    const graph = drawn;
    if (!renderer) return;
    // Same emphasis as hovering the node: its neighborhood stays lit, the
    // rest dims. Cleared naturally by the next enter/leaveNode.
    hoverStateRef.current = hoverStateFor(graph, nodeId);
    const display = renderer.getNodeDisplayData(nodeId);
    if (display) {
      const camera = renderer.getCamera();
      // Ratio only ever shrinks (zooms in) — landing further out than the
      // current view would read as the map running away from the match.
      const state = { x: display.x, y: display.y, ratio: Math.min(camera.ratio, 1) };
      if (prefersReducedMotion()) camera.setState(state);
      else camera.animate(state, { duration: 450 });
    }
    renderer.refresh();
  };

  const onSearchKeyDown = (e: ReactKeyboardEvent<HTMLInputElement>) => {
    if (e.key === "ArrowDown") {
      e.preventDefault();
      setActiveIndex((i) => Math.min(i + 1, Math.max(matches.length - 1, 0)));
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      setActiveIndex((i) => Math.max(i - 1, 0));
    } else if (e.key === "Enter") {
      const match = matches[activeIndex] ?? matches[0];
      if (match) focusEntity(match.id);
    } else if (e.key === "Escape") {
      setQuery("");
      searchInputRef.current?.blur();
    }
  };

  // Mount/rebuild sigma whenever the model changes. `palette` is read here
  // (fresh at build time) but deliberately not a dependency — a theme flip
  // recolors the existing graph in place (below) instead of tearing down and
  // remounting the whole renderer.
  useEffect(() => {
    const container = containerRef.current;
    // Guarded on the FULL model, not the drawn one: a graph whose components
    // are all smaller than MIN_COMPONENT_SIZE draws nothing but must still
    // mount, so the map frame and the "N unconnected or paired, hidden" chip
    // explain the emptiness instead of the view silently showing a blank slot.
    if (!container || model.nodes.length === 0) return;

    const graph = buildAtlasGraph(visibleModel, palette);
    runAtlasLayout(graph);
    graphRef.current = graph;
    sceneRef.current = null;
    sceneDirtyRef.current = true;

    // Same-frame paint per physics step (see createAtlasSimulation's onTick
    // note). sigmaRef is still null during the synchronous settle ticks, so
    // the 220 pre-paint steps don't render.
    const sim = createAtlasSimulation(graph, () => {
      sceneDirtyRef.current = true;
      sigmaRef.current?.refresh();
    });
    simRef.current = sim;
    if (import.meta.env.DEV) {
      // Preview/debug handle only — stripped from prod builds.
      (window as unknown as Record<string, unknown>).__ATLAS_SIM = sim;
    }
    // Place-name overlay — a plain 2D canvas appended AFTER sigma mounts
    // (below) so it stacks ABOVE sigma's canvases: region names sit on top
    // of the nodes like names on a map, never under them. Redrawn on every
    // afterRender, so the names follow drags and camera moves for free. It
    // is the only thing the Atlas paints outside sigma — nothing is drawn
    // under the nodes.
    const overlay = document.createElement("canvas");
    overlay.dataset.testid = "atlas-region-names";
    overlay.style.position = "absolute";
    overlay.style.inset = "0";
    overlay.style.width = "100%";
    overlay.style.height = "100%";
    overlay.style.pointerEvents = "none";
    overlay.style.display = showRegionsRef.current ? "" : "none";
    overlayRef.current = overlay;

    const renderer = new Sigma(graph, container, {
      // Only nodes at least this big carry a label. With the log2 size scale
      // (atlas.ts) that is roughly degree >= 5 for an entity and >= 6 for a
      // page, so the zoomed-out map shows hub names only; sigma's own label
      // grid reveals the rest as you zoom in.
      labelRenderedSizeThreshold: 8,
      // Round 4: with memories on, the size threshold alone still let dozens
      // of labels pile on top of each other. Sigma buckets the viewport into
      // a grid of labelGridCellSize screen px and keeps
      // ceil(labelDensity / cameraRatio^2) labels per cell, biggest node
      // first. At 0.04 that is exactly ONE name per cell for any camera ratio
      // above 0.2 — i.e. one per 120 px of screen at every zoom the map
      // normally sits at — and the cell size is what decides how coarse that
      // thinning is. Zooming in reveals more names because the same graph
      // area then spans more cells, not because the per-cell count rises.
      // 200 px cells: a handful of names at fit zoom, read as landmarks,
      // with room for a radial label to extend without touching the next.
      labelDensity: 0.04,
      labelGridCellSize: 200,
      // Default camera fit maps the graph bbox edge-to-edge on the tighter
      // axis, half-clipping the extreme nodes; give the map a margin.
      stagePadding: 40,
      // Sigma's default label ink is black regardless of theme; pass the
      // resolved text token instead (updated on theme flip below).
      labelColor: { color: palette.label },
      // Off by default in sigma — without it, the zIndex values nodeDisplay/
      // edgeDisplay return are computed but never affect paint order.
      zIndex: true,
      // Sigma's default hover renderer (drawDiscNodeHover) paints a hardcoded
      // #FFF label box — unreadable under the dark theme's light label ink.
      // The hovered label already renders theme-correct on the labels layer
      // (nodeDisplay forces it; renderLabels doesn't skip the hovered node),
      // so the box is pure loss. No-op it.
      defaultDrawNodeHover: () => {},
      // Node/edge sizes are true CSS px at every zoom. The default divides
      // sizes by sqrt(camera ratio), which shrinks items badly once the
      // density cap below zooms the camera out ~5x from fit.
      zoomToSizeRatioFunction: () => 1,
      // Edges are a 1 px hairline (0.6 for shared-source); sigma's default
      // floor of 1.7 would silently bump them back up.
      minEdgeThickness: 0.5,
      // 12px body-font labels placed radially around the node, facing the
      // cluster center, over a ground-coloured halo — sigma's default is
      // 14px Arial pinned to the right.
      defaultDrawNodeLabel: (
        ctx: CanvasRenderingContext2D,
        data: Record<string, any>,
        s: Record<string, any>,
      ) => drawRadialNodeLabel(ctx, data, s, graph, paletteRef.current.surface),
      nodeReducer: (node, attrs) => nodeDisplay(hoverStateRef.current, node, attrs, paletteRef.current),
      edgeReducer: (edge, attrs) => {
        const [source, target] = graph.extremities(edge);
        return edgeDisplay(hoverStateRef.current, edge, source, target, attrs, paletteRef.current);
      },
    });
    sigmaRef.current = renderer;
    container.appendChild(overlay);
    if (import.meta.env.DEV) {
      // Preview/debug handle only — stripped from prod builds.
      (window as unknown as Record<string, unknown>).__ATLAS_SIGMA = renderer;
    }

    const drawOverlay = (scene: CartographyScene) => {
      const ctx = overlay.getContext("2d");
      if (!ctx) return; // jsdom
      const { width, height } = renderer.getDimensions();
      const dpr = window.devicePixelRatio || 1;
      if (overlay.width !== width * dpr || overlay.height !== height * dpr) {
        overlay.width = width * dpr;
        overlay.height = height * dpr;
      }
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, width, height);
      drawRegionNames(ctx, scene, (pos) => renderer.graphToViewport(pos), paletteRef.current, { width, height });
    };
    // One handler paints the overlay from one scene — rebuilt only when a
    // paint marked it dirty — and sigma sees a single afterRender listener.
    renderer.on("afterRender", () => {
      if (sceneDirtyRef.current || sceneRef.current === null) {
        sceneRef.current = cartographyScene(graph, communitiesRef.current);
        sceneDirtyRef.current = false;
      }
      drawOverlay(sceneRef.current);
    });
    // Default zoom: sigma's fit stretches a small cluster edge-to-edge no
    // matter how big the container (7.3 px/graph-unit in preview) — links
    // render ~5x longer than the old graph's ("too wide"). A fixed density
    // cap at the old graph's exact 1.5 px/unit overshot the other way: in a
    // large container the cluster filled <20% of the view ("too far away").
    // The liked reference — the old Graph tab — sits at ~60% fill of the
    // smaller container axis, so target that fill, clamped to [1.5, 3]
    // px/unit: never denser than the old graph's spacing floor, never back
    // to the fit sprawl on huge screens. Only ever zoom OUT from fit.
    const o = renderer.graphToViewport({ x: 0, y: 0 });
    const u = renderer.graphToViewport({ x: 1, y: 0 });
    const pxPerUnit = Math.hypot(u.x - o.x, u.y - o.y);
    const bbox = renderer.getBBox();
    const span = Math.max(bbox.x[1] - bbox.x[0], bbox.y[1] - bbox.y[0]);
    const { width, height } = renderer.getDimensions();
    const targetDensity = Math.min(3, Math.max(1.5, (0.6 * Math.min(width, height)) / span));
    if (pxPerUnit > targetDensity) {
      const camera = renderer.getCamera();
      camera.setState({ ratio: camera.ratio * (pxPerUnit / targetDensity) });
    }
    // Overlay entry point: land already centered on the focused entity with
    // the same emphasis the search fly applies. setState, never animate —
    // this is the first frame the user sees, not a camera move.
    if (focusEntityId && graph.hasNode(focusEntityId)) {
      hoverStateRef.current = hoverStateFor(graph, focusEntityId);
      const display = renderer.getNodeDisplayData(focusEntityId);
      if (display) {
        const camera = renderer.getCamera();
        camera.setState({ x: display.x, y: display.y, ratio: Math.min(camera.ratio, 1) });
      }
    }
    // First paint of THIS renderer, and the only thing that draws the
    // overlay canvas above at all. Sigma's constructor render already happened
    // before the afterRender listeners existed, and the simulation rests at
    // alpha 0 straight out of createAtlasSimulation (atlas.ts settles it
    // synchronously, then stops), so no tick will schedule one either. This
    // used to be left to the cartography effect below — which only runs when
    // `communities` changes, and `communities` is derived from the FULL
    // model. Toggling the small-groups chip rebuilds the renderer off
    // `visibleModel` WITHOUT changing `communities`, so nothing ever painted:
    // the fresh canvas stayed at its untouched 300x150 default and every
    // region name vanished from the map.
    renderer.refresh();
    renderer.on("clickNode", ({ node }) => {
      // A moved drag must not also navigate on release.
      if (movedDuringPressRef.current) return;
      onNodeClick?.(targetForNode(node));
    });
    // Hover is LOCKED while a drag is live: our drag doesn't capture the
    // pointer (sigma's captor keeps picking), so sweeping the grabbed node
    // across other hit areas would fire enter/leave mid-drag — the graph
    // dims/undims, labels pop, and the cursor flickers grabbing→pointer→
    // default. force-graph never shows this (d3-drag captures the pointer,
    // hover is inert mid-drag), and the flashing reads as jank.
    renderer.on("enterNode", ({ node }) => {
      if (draggedNodeRef.current) return;
      hoverStateRef.current = hoverStateFor(graph, node);
      container.style.cursor = "pointer";
      renderer.refresh();
    });
    renderer.on("leaveNode", () => {
      if (draggedNodeRef.current) return;
      hoverStateRef.current = hoverStateFor(graph, null);
      container.style.cursor = "default";
      renderer.refresh();
    });

    // Node drag — sigma v3's mouse-manipulation pattern (see mouse.d.ts /
    // sigma.esm.js MouseCaptor): downNode starts it, the captor's own
    // mousemovebody/mouseup/mousedown carry the rest. Physics now come from
    // the d3-force simulation (see atlas.ts's createAtlasSimulation) instead
    // of a stepped FA2 loop — downNode pins the pressed node and reheats the
    // sim; mousemovebody drags that pin along with the pointer; mouseup
    // releases it and lets alpha decay naturally (or stops outright under
    // reduced motion).
    renderer.on("downNode", ({ node }) => {
      draggedNodeRef.current = node;
      movedDuringPressRef.current = false;
      graph.setNodeAttribute(node, "highlighted", true);
      container.style.cursor = "grabbing";
      // A leaf memory is not a sim node — the writeback would put it back on
      // its orbit on the next tick unless the sim knows a hand is on it.
      sim.setDraggingId(node);
      const simNode = sim.nodes().find((n) => n.id === node);
      // Leaf memories aren't sim members (see nonSimulatedIds) — dragging one
      // is pure direct manipulation via mousemovebody's graphology writes, so
      // there's nothing here to pin or reheat; setDraggingId above is what
      // stops the writeback snapping it back onto its orbit.
      if (simNode) {
        simNode.fx = simNode.x;
        simNode.fy = simNode.y;
        // alpha JUMPS to the target instead of ramping: the sim rests at
        // alpha 0, and alphaTarget alone climbs at only 3%/tick — neighbor
        // forces stay near-zero for the first ~1/3s of a drag, which reads
        // as lag (measured 3x early neighbor response with the jump). Safe
        // on a settled sim: the equilibrium-invariant test reheats to 0.3
        // and pins bbox drift < 3%.
        sim.alpha(0.3).alphaTarget(0.3).restart();
      }
    });
    const mouseCaptor = renderer.getMouseCaptor();
    mouseCaptor.on("mousedown", () => {
      // Freeze the camera frame so dragging a boundary node doesn't re-fit it.
      if (!renderer.getCustomBBox()) renderer.setCustomBBox(renderer.getBBox());
    });
    mouseCaptor.on("mousemovebody", (e) => {
      const draggedNode = draggedNodeRef.current;
      if (!draggedNode) return;
      movedDuringPressRef.current = true;
      const pos = renderer.viewportToGraph(e);
      graph.setNodeAttribute(draggedNode, "x", pos.x);
      graph.setNodeAttribute(draggedNode, "y", pos.y);
      // Written straight onto the graph, outside the sim's writeback, so the
      // cartography scene has to be told the positions moved.
      sceneDirtyRef.current = true;
      // Instant response between ticks — the dragged node's own position
      // isn't waiting on the next sim tick; its neighbors flow toward this
      // pin as the sim (reheated on downNode) keeps ticking.
      const simNode = sim.nodes().find((n) => n.id === draggedNode);
      if (simNode) {
        simNode.fx = pos.x;
        simNode.fy = pos.y;
      }
      // Sigma's own click suppression (draggedEvents vs. draggedEventsTolerance)
      // never sees this drag — preventSigmaDefault short-circuits handleMove
      // before that counter increments — so movedDuringPressRef above is what
      // actually guards clickNode.
      e.preventSigmaDefault();
      e.original.preventDefault();
      e.original.stopPropagation();
    });
    mouseCaptor.on("mouseup", () => {
      const draggedNode = draggedNodeRef.current;
      if (draggedNode) {
        graph.setNodeAttribute(draggedNode, "highlighted", false);
        const simNode = sim.nodes().find((n) => n.id === draggedNode);
        // Every component is live now (see atlas.ts's groupCenterForce) — its
        // own anchor holds it at its shelf slot, so releasing fx/fy here
        // never hands a shelved node back to unconstrained charge/forceCenter.
        if (simNode) {
          simNode.fx = null;
          simNode.fy = null;
        }
        draggedNodeRef.current = null;
      }
      sim.setDraggingId(null);
      container.style.cursor = hoverStateRef.current.hovered ? "pointer" : "default";
      // Natural decay is the inertia tail; reduced motion skips it outright.
      // That tail is also what walks a released shelf component back onto its
      // slot, so skipping it needs the same correction applied in one step —
      // otherwise a component dragged toward the core just stays there.
      if (prefersReducedMotion()) {
        sim.settleShelf();
        sim.stop();
      } else sim.alphaTarget(0);
    });

    // Direct wheel zoom. Sigma's default quantizes the gesture into 1.7x
    // steps eased over 250ms and DROPS any wheel event landing within
    // zoomDuration/5 = 50ms of the last accepted one — a trackpad's
    // 60-120 events/s collapse to ~20 animated lurches, which reads as a
    // low refresh rate (measured: paint cadence stays 120fps; only the
    // camera moves in steps). The old graph's d3-zoom applies every delta
    // 1:1 in the same frame; do the same, with d3-zoom's own delta scale,
    // zooming toward the cursor.
    mouseCaptor.on("wheel", (e) => {
      e.preventSigmaDefault();
      const we = e.original as WheelEvent;
      // d3-zoom wheelDelta: pixel-mode deltas x0.002, line-mode x0.05,
      // page-mode x1, and pinch (ctrlKey wheel on mac) x10. Camera ratio
      // is inverse scale, so positive deltaY (scroll down) grows it.
      const scale = we.deltaMode === 1 ? 0.05 : we.deltaMode ? 1 : 0.002;
      const factor = Math.pow(2, we.deltaY * scale * (we.ctrlKey ? 10 : 1));
      const camera = renderer.getCamera();
      const newRatio = camera.getBoundedRatio(camera.ratio * factor);
      if (newRatio === camera.ratio) return;
      camera.setState(renderer.getViewportZoomedState({ x: e.x, y: e.y }, newRatio));
    });

    return () => {
      sim.stop();
      simRef.current = null;
      sigmaRef.current = null;
      graphRef.current = null;
      renderer.kill();
      // Sigma removes its own canvases; the overlay is ours to remove.
      overlay.remove();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [visibleModel]);

  // Theme flip: recolor the live graph and repaint — no remount. Also keeps
  // paletteRef current so nodeReducer/edgeReducer (read at paint time) see
  // the new theme without the renderer being rebuilt.
  useEffect(() => {
    paletteRef.current = palette;
    const graph = graphRef.current;
    const renderer = sigmaRef.current;
    if (!graph || !renderer) return;
    graph.updateEachNodeAttributes((_id, attrs) => ({
      ...attrs,
      color: nodeFillFor(attrs.entityType, attrs.confirmed, palette),
    }));
    graph.updateEachEdgeAttributes((_id, attrs) => ({
      ...attrs,
      color: palette.edge,
    }));
    renderer.setSetting("labelColor", { color: palette.label });
    // refresh() re-fires afterRender, which repaints the place names with
    // the palette paletteRef now carries.
    renderer.refresh();
  }, [palette]);

  // Cartography arriving or regressing (fallback -> ready, or ready -> a
  // partial-error) must repaint the place names WITHOUT tearing down the
  // sim/camera — the mount effect above only rebuilds on `visibleModel`
  // changing, so a `cartographyBySpace` refetch that flips a space's status
  // never reaches the renderer otherwise. Same "repaint in place" shape as
  // the theme-flip effect: point communitiesRef at the fresh map (the paint
  // callbacks read it), then refresh.
  useEffect(() => {
    // On the mount pass communitiesRef already holds this very map (it is
    // seeded with it), and the mount effect has just painted with it — so
    // there is nothing to repaint and the scene would be rebuilt twice.
    // Only an actual change to the map is work.
    if (communitiesRef.current === communities) return;
    communitiesRef.current = communities;
    // Region membership is derived from the communities map, so a status flip
    // invalidates the cached scene even though nothing moved.
    sceneDirtyRef.current = true;
    sigmaRef.current?.refresh();
  }, [communities]);

  useEffect(() => {
    const overlay = overlayRef.current;
    if (!overlay) return;
    overlay.style.display = showRegions ? "" : "none";
    // The canvas keeps its last frame while hidden; repaint on re-show so it
    // matches wherever the camera and drags went in the meantime.
    if (showRegions) sigmaRef.current?.refresh();
  }, [showRegions]);

  const statusStyle = {
    height: "100%",
    width: "100%",
    display: "flex",
    flexDirection: "column" as const,
    alignItems: "center",
    justifyContent: "center",
    gap: 10,
    background: "var(--mem-surface)",
    fontFamily: "var(--mem-font-body)",
  };

  // Honest states: a dead daemon must never look like an empty graph.
  if (graphError) {
    return (
      <div data-testid="atlas-view" style={statusStyle}>
        <p className="entity-empty" style={{ color: "var(--mem-status-danger-text)" }}>
          {t("constellationMap.loadError")}
        </p>
        <button
          type="button"
          className="memory-detail-text-button"
          onClick={() => {
            refetchGraph();
          }}
        >
          {t("constellationMap.retry")}
        </button>
      </div>
    );
  }

  if (graphLoading) {
    return (
      <div data-testid="atlas-view" style={statusStyle}>
        <span className="entity-empty">{t("constellationMap.loading")}</span>
      </div>
    );
  }

  // Nothing to draw only when the daemon has neither entities NOR wiki pages:
  // a knowledge base made purely of pages is a real map, not an empty one.
  if (entities.length === 0 && graph.pages.length === 0) {
    return (
      <div data-testid="atlas-view" style={statusStyle}>
        <span className="entity-empty">{t("constellationMap.empty")}</span>
      </div>
    );
  }

  // Counts describe what is actually ON THE MAP: layer on, and connected to
  // something. What the degree-0 filter took out is reported by its own chip,
  // and a layer that is off contributes nothing rather than a zero.
  const drawn = visibleModel.nodes;
  const pageCount = drawn.filter((node) => node.kind === "page").length;
  const memoryCount = drawn.filter((node) => node.kind === "memory").length;
  const entityCount = drawn.length - pageCount - memoryCount;
  // A kind appears when its layer is on and it actually contributed nodes.
  // Entities are the exception and always report, zero included — the round-1
  // line did, and "0 entities" is the honest answer to an empty entity layer.
  const countLine = [
    ...(layers.page && pageCount > 0 ? [t("atlas.countPages", { count: pageCount })] : []),
    ...(layers.entity ? [t("atlas.countEntities", { count: entityCount })] : []),
    ...(layers.memory && memoryCount > 0
      ? [t("atlas.countMemories", { count: memoryCount })]
      : []),
    ...(regionCount > 0 ? [t("atlas.countRegions", { count: regionCount })] : []),
  ].join(" · ");
  const dropdownOpen = searchFocused && query.trim().length > 0;

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100%", width: "100%" }}>
      {/* Toolbar — artifact screen 01: ⌘K search + mono count line. The
          filter chips and Atlas|Focus segment wait for their features. */}
      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: 10,
          padding: "12px 16px",
          borderBottom: "1px solid var(--mem-border)",
          flexWrap: "wrap",
          background: "var(--mem-surface)",
          fontFamily: "var(--mem-font-body)",
        }}
      >
        {onBack && (
          <button
            type="button"
            onClick={onBack}
            className="flex items-center gap-1.5 rounded-md transition-colors duration-150 hover:bg-[var(--mem-hover)]"
            style={{
              color: "var(--mem-text-secondary)",
              fontSize: 12,
              fontFamily: "var(--mem-font-body)",
              background: "none",
              border: "none",
              cursor: "pointer",
              padding: "6px 8px",
            }}
          >
            <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M19 12H5M12 19l-7-7 7-7"/></svg>
            {t("main.back")}
          </button>
        )}
        <div style={{ position: "relative", flex: "0 1 300px", minWidth: 250 }}>
          <div
            style={{
              display: "flex",
              alignItems: "center",
              gap: 8,
              background: "var(--mem-bg)",
              border: `1px solid ${searchFocused ? "var(--mem-accent-indigo-border)" : "var(--mem-border)"}`,
              borderRadius: "var(--mem-radius-md)",
              padding: "7px 12px",
            }}
          >
            <input
              ref={searchInputRef}
              type="text"
              role="combobox"
              aria-expanded={dropdownOpen}
              aria-controls="atlas-search-listbox"
              aria-activedescendant={
                dropdownOpen && matches.length > 0 ? `atlas-search-option-${activeIndex}` : undefined
              }
              aria-label={t("atlas.searchLabel")}
              placeholder={t("atlas.searchPlaceholder")}
              value={query}
              onChange={(e) => {
                setQuery(e.target.value);
                setActiveIndex(0);
              }}
              onFocus={() => setSearchFocused(true)}
              onBlur={() => setSearchFocused(false)}
              onKeyDown={onSearchKeyDown}
              style={{
                flex: 1,
                minWidth: 0,
                background: "transparent",
                border: "none",
                outline: "none",
                font: "400 13px var(--mem-font-body)",
                color: "var(--mem-text)",
                padding: 0,
              }}
            />
            <kbd
              style={{
                font: "400 10px var(--mem-font-mono)",
                color: "var(--mem-text-tertiary)",
                border: "1px solid var(--mem-border)",
                borderRadius: 4,
                padding: "1px 5px",
              }}
            >
              ⌘K
            </kbd>
          </div>
          {dropdownOpen && (
            <ul
              id="atlas-search-listbox"
              role="listbox"
              style={{
                position: "absolute",
                top: "calc(100% + 6px)",
                left: 0,
                right: 0,
                margin: 0,
                padding: 4,
                listStyle: "none",
                background: "var(--mem-surface)",
                border: "1px solid var(--mem-popover-border, var(--mem-border))",
                borderRadius: "var(--mem-radius-md)",
                boxShadow: "0 8px 24px rgba(0, 0, 0, 0.12)",
                zIndex: 20,
                maxHeight: 280,
                overflowY: "auto",
              }}
            >
              {matches.map((node, index) => (
                <li
                  key={node.id}
                  id={`atlas-search-option-${index}`}
                  role="option"
                  aria-selected={index === activeIndex}
                  // preventDefault keeps the input's blur from closing the
                  // list before this row's click lands.
                  onMouseDown={(e) => e.preventDefault()}
                  onClick={() => focusEntity(node.id)}
                  onMouseEnter={() => setActiveIndex(index)}
                  style={{
                    display: "flex",
                    alignItems: "center",
                    gap: 8,
                    padding: "6px 10px",
                    borderRadius: "var(--mem-radius-sm)",
                    fontSize: 13,
                    color: "var(--mem-text)",
                    cursor: "pointer",
                    background: index === activeIndex ? "var(--mem-hover)" : "transparent",
                  }}
                >
                  <span
                    style={{
                      width: 8,
                      height: 8,
                      borderRadius: "50%",
                      flexShrink: 0,
                      backgroundColor:
                        node.entityType === MEMORY_NODE_TYPE
                          ? palette.memory
                          : node.entityType === PAGE_NODE_TYPE
                            ? palette.page
                            : colorForEntityType(node.entityType, palette),
                      opacity: 0.85,
                    }}
                  />
                  <span style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                    {node.name}
                  </span>
                </li>
              ))}
              {matches.length === 0 && (
                <li style={{ padding: "6px 10px", fontSize: 12, color: "var(--mem-text-tertiary)" }}>
                  {t("atlas.noMatches")}
                </li>
              )}
            </ul>
          )}
        </div>
        {/* Layer chips — which node kinds the map draws. The last lit chip is
            disabled: an empty map is not a view. */}
        {(
          [
            { key: "page" as const, label: t("atlas.layer.page") },
            { key: "entity" as const, label: t("atlas.layer.entity") },
            { key: "memory" as const, label: t("atlas.layer.memory") },
          ]
        ).map(({ key, label }) => {
          const on = layers[key];
          const locked = onlyLayerOn(key);
          return (
            <button
              key={key}
              type="button"
              aria-pressed={on}
              disabled={locked}
              onClick={() => toggleLayer(key)}
              style={{
                fontSize: 12,
                color: on ? "var(--mem-text)" : "var(--mem-text-secondary)",
                border: `1px solid ${on ? "var(--mem-distilled-border)" : "var(--mem-border)"}`,
                borderRadius: "var(--mem-radius-full)",
                padding: "4px 12px",
                background: on ? "var(--mem-indigo-bg)" : "transparent",
                cursor: locked ? "default" : "pointer",
                opacity: locked ? 0.7 : 1,
                fontFamily: "inherit",
              }}
            >
              {label}
            </button>
          );
        })}
        <button
          type="button"
          aria-pressed={showRegions}
          onClick={() => {
            showRegionsRef.current = !showRegions;
            setShowRegions(!showRegions);
          }}
          style={{
            fontSize: 12,
            color: showRegions ? "var(--mem-text)" : "var(--mem-text-secondary)",
            border: `1px solid ${showRegions ? "var(--mem-distilled-border)" : "var(--mem-border)"}`,
            borderRadius: "var(--mem-radius-full)",
            padding: "4px 12px",
            background: showRegions ? "var(--mem-indigo-bg)" : "transparent",
            cursor: "pointer",
            fontFamily: "inherit",
          }}
        >
          {t("atlas.regionsToggle")}
        </button>
        {spaces.length > 0 && (
          <select
            aria-label={t("atlas.spaceLabel")}
            value={spaceFilter ?? ""}
            onChange={(event) => setSpaceFilter(event.target.value || null)}
            style={{
              fontSize: 12,
              color: spaceFilter ? "var(--mem-text)" : "var(--mem-text-secondary)",
              border: `1px solid ${spaceFilter ? "var(--mem-distilled-border)" : "var(--mem-border)"}`,
              borderRadius: "var(--mem-radius-full)",
              padding: "4px 10px",
              background: spaceFilter ? "var(--mem-indigo-bg)" : "transparent",
              cursor: "pointer",
              fontFamily: "inherit",
            }}
          >
            <option value="">{t("atlas.spaceAll")}</option>
            {spaces.map((space) => (
              <option key={space} value={space}>
                {space}
              </option>
            ))}
          </select>
        )}
        {cartographyStatus && (
          <span
            role={cartographyStatus === "partial-error" ? "alert" : undefined}
            style={{
              fontSize: 11,
              fontFamily: "var(--mem-font-mono)",
              color:
                cartographyStatus === "partial-error"
                  ? "var(--mem-danger)"
                  : "var(--mem-text-tertiary)",
              border: `1px solid ${cartographyStatus === "partial-error" ? "var(--mem-danger)" : "var(--mem-border)"}`,
              borderRadius: "var(--mem-radius-full)",
              padding: "3px 10px",
            }}
          >
            {t(
              cartographyStatus === "ready"
                ? "atlas.cartographyReady"
                : cartographyStatus === "partial-error"
                  ? "atlas.cartographyPartialError"
                  : "atlas.cartographyFallback",
            )}
          </span>
        )}
        {smallGroupCount > 0 && (
          <button
            type="button"
            onClick={toggleSmallGroups}
            aria-pressed={showSmallGroups}
            aria-label={t(showSmallGroups ? "atlas.hideSmallGroups" : "atlas.showSmallGroups")}
            style={{
              font: "400 11px var(--mem-font-mono)",
              color: showSmallGroups ? "var(--mem-text-secondary)" : "var(--mem-text-tertiary)",
              background: "transparent",
              border: "1px solid var(--mem-border)",
              borderRadius: "var(--mem-radius-full)",
              padding: "3px 10px",
              cursor: "pointer",
            }}
          >
            {showSmallGroups
              ? t("atlas.hideSmallGroups")
              : t("atlas.smallGroupsHidden", { count: smallGroupCount })}
          </button>
        )}
        <span
          style={{
            marginLeft: "auto",
            font: "400 11px var(--mem-font-mono)",
            color: "var(--mem-text-tertiary)",
          }}
        >
          {countLine}
        </span>
      </div>

      <div style={{ position: "relative", flex: 1, minHeight: 0 }}>
      <div ref={containerRef} data-testid="atlas-view" style={{ height: "100%", width: "100%" }} />

      {/* Legend — top-right, same furniture as the old canvas graph (minus
          the memories/pages/labels toggles Atlas doesn't have yet). */}
      <div
        style={{
          position: "absolute",
          top: 10,
          right: 10,
          display: "flex",
          flexDirection: "column",
          gap: 5,
          padding: "6px 10px",
          fontSize: 10,
          fontFamily: "var(--mem-font-body)",
          color: "var(--mem-text-tertiary)",
          background: "var(--mem-surface)",
          border: "1px solid var(--mem-border)",
          borderRadius: 6,
          opacity: 0.85,
          pointerEvents: "none",
        }}
      >
        {LEGEND_ITEMS.map(({ label, key }) => (
          <div key={key} style={{ display: "flex", alignItems: "center", gap: 5 }}>
            <span
              style={{
                display: "inline-block",
                width: 8,
                height: 8,
                borderRadius: "50%",
                backgroundColor:
                  key === MEMORY_NODE_TYPE
                    ? palette.memory
                    : key === PAGE_NODE_TYPE
                      ? palette.page
                      : colorForEntityType(key, palette),
                opacity: 0.7,
                flexShrink: 0,
              }}
            />
            <span>{label}</span>
          </div>
        ))}
        <div style={{ display: "flex", alignItems: "center", gap: 5 }}>
          <span
            style={{
              display: "inline-block",
              width: 12,
              height: 0,
              borderTop: "1px solid var(--mem-text-tertiary)",
              opacity: 0.5,
              flexShrink: 0,
            }}
          />
          <span>{t("constellationMap.legendConnection")}</span>
        </div>
      </div>

      {/* Hint chip — bottom-left, artifact's map affordance line. */}
      <span
        style={{
          position: "absolute",
          left: 14,
          bottom: 12,
          font: "400 10.5px var(--mem-font-mono)",
          color: "var(--mem-text-tertiary)",
          background: "var(--mem-surface)",
          border: "1px solid var(--mem-border)",
          borderRadius: "var(--mem-radius-full)",
          padding: "4px 11px",
          pointerEvents: "none",
          opacity: 0.9,
        }}
      >
        {t("atlas.hint")}
      </span>
      </div>
    </div>
  );
}
