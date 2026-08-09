// SPDX-License-Identifier: AGPL-3.0-only
import { useEffect, useMemo, useRef, useState } from "react";
import type { KeyboardEvent as ReactKeyboardEvent } from "react";
import { useQuery } from "@tanstack/react-query";
import { useTranslation } from "react-i18next";
import Graph from "graphology";
import Sigma from "sigma";
import type { Simulation } from "d3-force";
import { listEntities, getEntityDetail } from "../../lib/tauri";
import type { Entity, EntityDetail } from "../../lib/tauri";
import { buildGraphModel } from "../../lib/graph/model";
import type { GraphNode } from "../../lib/graph/model";
import {
  buildAtlasGraph,
  runAtlasLayout,
  createAtlasSimulation,
  placeIsolateRing,
  hoverStateFor,
  nodeDisplay,
  edgeDisplay,
  drawRadialNodeLabel,
} from "../../lib/graph/atlas";
import type { HoverState, AtlasSimNode } from "../../lib/graph/atlas";
import {
  communitiesFor,
  cartographyScene,
  drawCartography,
  bridgeEdgeTest,
  regionLeader,
  MIN_REGION_SIZE,
} from "../../lib/graph/cartography";
import { useGraphPalette, colorForEntityType, nodeFillFor } from "../../lib/graph/palette";
import type { GraphPalette } from "../../lib/graph/palette";

// Same 5-slot legend as the retired canvas graph (ConstellationMap): place,
// event, and unknown types fold to neutral and get no swatch; concept is
// labeled "Theme" to match the product copy.
const LEGEND_ITEMS: { label: string; key: string }[] = [
  { label: "Project", key: "project" },
  { label: "Technology", key: "technology" },
  { label: "Organization", key: "organization" },
  { label: "Person", key: "person" },
  { label: "Theme", key: "concept" },
];

interface AtlasViewProps {
  onNodeClick?: (entityId: string) => void;
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
  const simRef = useRef<Simulation<AtlasSimNode, undefined> | null>(null);
  // Reducer inputs, read from refs so hover/theme changes repaint without a
  // React re-render or a renderer rebuild (see the mount effect below).
  const hoverStateRef = useRef<HoverState>({ hovered: null, neighbors: new Set() });
  const paletteRef = useRef<GraphPalette>(palette);
  // Node-drag state: which node (if any) is being dragged, and whether the
  // pointer actually moved during the current press — the latter gates
  // clickNode so a drag-release doesn't also fire entity navigation.
  const draggedNodeRef = useRef<string | null>(null);
  const movedDuringPressRef = useRef(false);

  const {
    data: entities = [],
    isLoading: entitiesLoading,
    isError: entitiesError,
    refetch: refetchEntities,
  } = useQuery({
    queryKey: ["constellation-entities"],
    queryFn: () => listEntities(),
    refetchInterval: 120_000,
  });

  const top20Ids = useMemo(
    () => entities.slice(0, 20).map((e: Entity) => e.id),
    [entities],
  );

  const {
    data: details = [],
    isLoading: detailsLoading,
    isError: detailsError,
    refetch: refetchDetails,
  } = useQuery({
    queryKey: ["constellation-relations", top20Ids],
    queryFn: async () => {
      const settled = await Promise.allSettled(top20Ids.map((id) => getEntityDetail(id)));
      const succeeded = settled
        .filter((r): r is PromiseFulfilledResult<EntityDetail> => r.status === "fulfilled")
        .map((r) => r.value);
      // One flaky detail fetch shouldn't blank the whole graph — only a
      // total wipeout is a real outage worth the full error screen.
      if (succeeded.length === 0) {
        throw new Error("All entity detail fetches failed");
      }
      return succeeded;
    },
    enabled: top20Ids.length > 0,
    refetchInterval: 300_000,
    staleTime: 120_000,
  });

  // Same field precedence as the entity page (EntityDetail's `space ?? domain`).
  const spaces = useMemo(
    () =>
      Array.from(
        new Set(entities.map((e: Entity) => e.space ?? e.domain).filter((s): s is string => !!s)),
      ).sort(),
    [entities],
  );
  const [spaceFilter, setSpaceFilter] = useState<string | null>(null);

  // Scoping filters the model INPUTS: the space's own entities plus whatever
  // bridge neighbors their relations synthesize (those carry no space of their
  // own). Regions, counts, and insights all re-derive from the scoped model.
  const model = useMemo(() => {
    if (!spaceFilter) return buildGraphModel(entities, details);
    return buildGraphModel(
      entities.filter((e: Entity) => (e.space ?? e.domain) === spaceFilter),
      details.filter((d: EntityDetail) => (d.entity.space ?? d.entity.domain) === spaceFilter),
    );
  }, [entities, details, spaceFilter]);
  const communities = useMemo(() => communitiesFor(model), [model]);

  // Region count + names for the toolbar and rail — membership only, so it
  // agrees with the hulls drawCartography actually draws without needing
  // node positions; names share regionLeader with the drawn labels.
  const regionInfo = useMemo(() => {
    const groups = new Map<number, GraphNode[]>();
    for (const node of model.nodes) {
      const community = communities.get(node.id);
      if (community === undefined) continue;
      const list = groups.get(community);
      if (list) list.push(node);
      else groups.set(community, [node]);
    }
    const names = new Map<number, string>();
    for (const [community, members] of groups) {
      if (members.length >= MIN_REGION_SIZE) names.set(community, regionLeader(members).name);
    }
    return { count: names.size, names };
  }, [model, communities]);

  // Insight rail (artifact screen 01) — only cards whose data is real today:
  // isolates as the gap signal, cross-region bridges, this week's relations.
  // Every action is a live focusEntity fly, no dead links.
  const insights = useMemo(() => {
    const cards: { key: string; title: string; body: string; focusId: string }[] = [];
    const nodeName = new Map(model.nodes.map((n) => [n.id, n.name]));

    const isolates = model.nodes.filter((n) => n.degree === 0);
    if (isolates.length > 0) {
      const first = isolates[0];
      cards.push({
        key: "gap",
        title: t("atlas.rail.gapTitle"),
        body:
          isolates.length === 1
            ? t("atlas.rail.gapOne", { name: first.name })
            : t("atlas.rail.gapMore", { name: first.name, count: isolates.length - 1 }),
        focusId: first.id,
      });
    }

    const isBridge = bridgeEdgeTest(communities);
    const bridge = model.edges.find((e) => isBridge(e.source, e.target));
    if (bridge) {
      const pairKey = (s: string, t2: string) => {
        const a = communities.get(s)!;
        const b = communities.get(t2)!;
        return a < b ? `${a}:${b}` : `${b}:${a}`;
      };
      const pair = pairKey(bridge.source, bridge.target);
      const pairCount = model.edges.filter(
        (e) => isBridge(e.source, e.target) && pairKey(e.source, e.target) === pair,
      ).length;
      cards.push({
        key: "bridge",
        title: t("atlas.rail.bridgeTitle"),
        body: t("atlas.rail.bridgeBody", {
          count: pairCount,
          a: regionInfo.names.get(communities.get(bridge.source)!),
          b: regionInfo.names.get(communities.get(bridge.target)!),
          link: `${nodeName.get(bridge.source)} → ${nodeName.get(bridge.target)}`,
        }),
        focusId: bridge.source,
      });
    }

    // createdAt is Unix seconds (daemon convention — see EntityDetail's * 1000)
    const weekAgo = Date.now() / 1000 - 7 * 24 * 60 * 60;
    const recent = model.edges.filter((e) => e.createdAt >= weekAgo);
    if (recent.length > 0) {
      const gained = new Map<string, number>();
      for (const e of recent) {
        gained.set(e.source, (gained.get(e.source) ?? 0) + 1);
        if (e.target !== e.source) gained.set(e.target, (gained.get(e.target) ?? 0) + 1);
      }
      let topId = "";
      let topN = 0;
      for (const [id, n] of gained) {
        const name = nodeName.get(id) ?? "";
        const topName = nodeName.get(topId) ?? "";
        if (n > topN || (n === topN && name < topName)) {
          topId = id;
          topN = n;
        }
      }
      cards.push({
        key: "week",
        title: t("atlas.rail.weekTitle"),
        body: t("atlas.rail.weekBody", {
          count: recent.length,
          name: nodeName.get(topId),
          gained: topN,
        }),
        focusId: topId,
      });
    }
    return cards;
  }, [model, communities, regionInfo, t]);

  // Toolbar search (artifact screen 01): type → listbox of entity names,
  // Enter/click → camera fly + the same emphasis hover applies.
  const searchInputRef = useRef<HTMLInputElement>(null);
  const [query, setQuery] = useState("");
  const [activeIndex, setActiveIndex] = useState(0);
  const [searchFocused, setSearchFocused] = useState(false);
  // Regions on/off governs only the cartography underlay (hulls + names +
  // graticule); the count line and rail keep reporting regions either way.
  // Ref mirror so the sigma mount effect (which recreates the underlay per
  // model) can apply the current choice without re-running on toggle.
  const [showRegions, setShowRegions] = useState(true);
  const showRegionsRef = useRef(true);
  const underlayRef = useRef<HTMLCanvasElement | null>(null);

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
    const graph = graphRef.current;
    if (!renderer || !graph || !graph.hasNode(nodeId)) return;
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
    if (!container || model.nodes.length === 0) return;

    const graph = buildAtlasGraph(model, palette, communities);
    runAtlasLayout(graph);
    graphRef.current = graph;

    // Same-frame paint per physics step (see createAtlasSimulation's onTick
    // note). sigmaRef is still null during the synchronous settle ticks, so
    // the 220 pre-paint steps don't render.
    const sim = createAtlasSimulation(graph, () => sigmaRef.current?.refresh());
    simRef.current = sim;
    if (import.meta.env.DEV) {
      // Preview/debug handle only — stripped from prod builds.
      (window as unknown as Record<string, unknown>).__ATLAS_SIM = sim;
    }
    placeIsolateRing(graph);

    // Cartography underlay (hulls, region names, graticule) — a plain 2D
    // canvas appended BEFORE sigma mounts so sigma's own canvases stack above
    // it. Redrawn on every afterRender, so hulls flex live with drags and
    // track camera moves for free.
    const underlay = document.createElement("canvas");
    underlay.dataset.testid = "atlas-cartography";
    underlay.style.position = "absolute";
    underlay.style.inset = "0";
    underlay.style.width = "100%";
    underlay.style.height = "100%";
    underlay.style.display = showRegionsRef.current ? "" : "none";
    container.appendChild(underlay);
    underlayRef.current = underlay;

    const renderer = new Sigma(graph, container, {
      labelRenderedSizeThreshold: 6,
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
      // Edges are 1.5 CSS px (the old graph's stroke); sigma's default floor
      // of 1.7 would silently bump them back up.
      minEdgeThickness: 1,
      // 12px system-font labels placed radially around the node, facing the
      // cluster center — sigma's default is 14px Arial pinned to the right.
      defaultDrawNodeLabel: (
        ctx: CanvasRenderingContext2D,
        data: Record<string, any>,
        s: Record<string, any>,
      ) => drawRadialNodeLabel(ctx, data, s, graph),
      nodeReducer: (node, attrs) => nodeDisplay(hoverStateRef.current, node, attrs, paletteRef.current),
      edgeReducer: (edge, attrs) => {
        const [source, target] = graph.extremities(edge);
        return edgeDisplay(hoverStateRef.current, edge, source, target, attrs, paletteRef.current);
      },
    });
    sigmaRef.current = renderer;
    if (import.meta.env.DEV) {
      // Preview/debug handle only — stripped from prod builds.
      (window as unknown as Record<string, unknown>).__ATLAS_SIGMA = renderer;
    }

    const drawUnderlay = () => {
      const ctx = underlay.getContext("2d");
      if (!ctx) return; // jsdom
      const { width, height } = renderer.getDimensions();
      const dpr = window.devicePixelRatio || 1;
      if (underlay.width !== width * dpr || underlay.height !== height * dpr) {
        underlay.width = width * dpr;
        underlay.height = height * dpr;
      }
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, width, height);
      drawCartography(
        ctx,
        cartographyScene(graph, communities),
        (pos) => renderer.graphToViewport(pos),
        paletteRef.current,
      );
    };
    renderer.on("afterRender", drawUnderlay);
    drawUnderlay();
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
      renderer.refresh();
    }
    renderer.on("clickNode", ({ node }) => {
      // A moved drag must not also navigate on release.
      if (movedDuringPressRef.current) return;
      onNodeClick?.(node);
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
      const simNode = sim.nodes().find((n) => n.id === node);
      // Isolates aren't sim members (see createAtlasSimulation) — an isolate
      // drag is pure direct manipulation via mousemovebody's graphology
      // writes, so there's nothing here to pin or reheat.
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
        if (simNode) {
          simNode.fx = null;
          simNode.fy = null;
        }
        draggedNodeRef.current = null;
      }
      container.style.cursor = hoverStateRef.current.hovered ? "pointer" : "default";
      // Natural decay is the inertia tail; reduced motion skips it outright.
      if (prefersReducedMotion()) sim.stop();
      else sim.alphaTarget(0);
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
      // Sigma removes its own canvases; the underlay is ours to remove.
      underlay.remove();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [model]);

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
      color: attrs.bridge ? palette.bridge : palette.edge,
    }));
    renderer.setSetting("labelColor", { color: palette.label });
    // refresh() re-fires afterRender, which repaints the cartography underlay
    // with the palette paletteRef now carries.
    renderer.refresh();
  }, [palette]);

  useEffect(() => {
    const underlay = underlayRef.current;
    if (!underlay) return;
    underlay.style.display = showRegions ? "" : "none";
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
  if (entitiesError || detailsError) {
    return (
      <div data-testid="atlas-view" style={statusStyle}>
        <p className="entity-empty" style={{ color: "var(--mem-status-danger-text)" }}>
          {t("constellationMap.loadError")}
        </p>
        <button
          type="button"
          className="memory-detail-text-button"
          onClick={() => {
            refetchEntities();
            refetchDetails();
          }}
        >
          {t("constellationMap.retry")}
        </button>
      </div>
    );
  }

  if (entitiesLoading || detailsLoading) {
    return (
      <div data-testid="atlas-view" style={statusStyle}>
        <span className="entity-empty">{t("constellationMap.loading")}</span>
      </div>
    );
  }

  if (entities.length === 0) {
    return (
      <div data-testid="atlas-view" style={statusStyle}>
        <span className="entity-empty">{t("constellationMap.empty")}</span>
      </div>
    );
  }

  const entityCount = model.nodes.length;
  const countLine =
    regionInfo.count > 0
      ? `${t("atlas.countEntities", { count: entityCount })} · ${t("atlas.countRegions", { count: regionInfo.count })}`
      : t("atlas.countEntities", { count: entityCount });
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
                      backgroundColor: colorForEntityType(node.entityType, palette),
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

      <div
        style={{
          display: "grid",
          gridTemplateColumns: insights.length > 0 ? "minmax(0, 1fr) 292px" : "minmax(0, 1fr)",
          flex: 1,
          minHeight: 0,
        }}
      >
      <div style={{ position: "relative", minWidth: 0 }}>
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
                backgroundColor: colorForEntityType(key, palette),
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

      {/* Insight rail — artifact screen 01's 292px column. Renders only
          cards with live data behind them; every action is a real fly. */}
      {insights.length > 0 && (
        <aside
          style={{
            borderLeft: "1px solid var(--mem-border)",
            background: "var(--mem-surface)",
            padding: "16px 16px 20px",
            display: "flex",
            flexDirection: "column",
            gap: 14,
            overflowY: "auto",
            fontFamily: "var(--mem-font-body)",
          }}
        >
          {insights.map((card) => (
            <div
              key={card.key}
              style={{
                background: "var(--mem-detail-surface-raised)",
                border: "1px solid var(--mem-border)",
                borderRadius: "var(--mem-radius-md)",
                padding: "12px 14px",
              }}
            >
              <div
                style={{
                  font: "500 10px var(--mem-font-mono)",
                  letterSpacing: ".14em",
                  textTransform: "uppercase",
                  color: "var(--mem-accent-warm)",
                }}
              >
                {card.title}
              </div>
              <p
                style={{
                  margin: "6px 0 0",
                  fontSize: 12.5,
                  lineHeight: 1.5,
                  color: "var(--mem-text-secondary)",
                }}
              >
                {card.body}
              </p>
              <button
                type="button"
                onClick={() => focusEntity(card.focusId)}
                style={{
                  display: "inline-block",
                  background: "none",
                  border: "none",
                  padding: 0,
                  marginTop: 8,
                  fontSize: 11.5,
                  color: "var(--mem-accent-indigo)",
                  cursor: "pointer",
                  fontFamily: "inherit",
                }}
              >
                {t("atlas.rail.showInAtlas")}
              </button>
            </div>
          ))}
        </aside>
      )}
      </div>
    </div>
  );
}
