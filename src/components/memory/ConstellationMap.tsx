// SPDX-License-Identifier: AGPL-3.0-only
import { useRef, useMemo, useCallback, useEffect, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import ForceGraph2D from "react-force-graph-2d";
import { listEntities, getEntityDetail, listMemoriesRich } from "../../lib/tauri";
import type { Entity } from "../../lib/tauri";

/* ------------------------------------------------------------------ */
/*  Types                                                              */
/* ------------------------------------------------------------------ */

interface ConstellationMapProps {
  onClick?: () => void;
  fullScreen?: boolean;
  onNodeClick?: (entityId: string) => void;
}

interface GraphNode {
  id: string;
  name: string;
  entityType: string;
  stability: string;
  connectionCount: number;
  isMemory?: boolean;
  isDistilled?: boolean;
}

interface GraphLink {
  source: string;
  target: string;
}

/* ------------------------------------------------------------------ */
/*  Color mapping                                                      */
/* ------------------------------------------------------------------ */

// Dark mode: warm, luminous tones that glow on dark backgrounds
const TYPE_COLORS: Record<string, string> = {
  person:       "#E0926A",  // terracotta — human warmth
  project:      "#6DB87E",  // verdant — growth, building
  technology:   "#5BA4C9",  // cool steel — precision, tools
  organization: "#D4A45E",  // golden amber — institutional
  place:        "#C4956E",  // sandstone — grounded
  concept:      "#A48BBF",  // amethyst — abstract thought
  event:        "#D17089",  // dusty rose — memorable moments
};

// Light mode: deeper, richer versions for readability on light backgrounds
const TYPE_COLORS_LIGHT: Record<string, string> = {
  person:       "#C07550",
  project:      "#4E9A62",
  technology:   "#3E87AD",
  organization: "#B88A42",
  place:        "#A87852",
  concept:      "#8670A5",
  event:        "#B5566E",
};

function getThemeColors() {
  const isDark = document.documentElement.getAttribute("data-theme") !== "light";
  return isDark ? TYPE_COLORS : TYPE_COLORS_LIGHT;
}

function colorForType(entityType: string, colors: Record<string, string>): string {
  return colors[entityType] ?? colors.concept;
}

function nodeRadius(stability: string, connectionCount: number): number {
  const base = stability === "confirmed" ? 4 : stability === "learned" ? 3.5 : 3;
  return Math.min(8, Math.max(3, base + connectionCount * 0.5));
}

/* ------------------------------------------------------------------ */
/*  Component                                                          */
/* ------------------------------------------------------------------ */

export default function ConstellationMap({ onClick, fullScreen, onNodeClick }: ConstellationMapProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const fgRef = useRef<any>(null);
  const defaultHeight = fullScreen ? 600 : 280;
  const [dimensions, setDimensions] = useState({ width: 400, height: defaultHeight });
  const [showMemories, setShowMemories] = useState(() => localStorage.getItem("constellation-show-memories") === "true");
  const [showLabels, setShowLabels] = useState(() => localStorage.getItem("constellation-show-labels") !== "false");

  // Responsive sizing
  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const ro = new ResizeObserver((entries) => {
      for (const entry of entries) {
        const w = entry.contentRect.width;
        const h = fullScreen ? entry.contentRect.height : 280;
        setDimensions({ width: w, height: h });
        // Re-zoom after dimension change settles (ForceGraph2D resets zoom on resize).
        // Matches the fit logic in `onEngineStop` for each mode.
        setTimeout(() => {
          if (fullScreen) {
            fgRef.current?.zoomToFit?.(0, -40);
          } else if (focusNodeIds.size > 0) {
            fgRef.current?.zoomToFit?.(
              0,
              24,
              (n: any) => focusNodeIds.has(n.id),
            );
          } else {
            fgRef.current?.zoomToFit?.(0, 8);
          }
        }, 50);
      }
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, [fullScreen]);

  // Fetch entities
  const { data: entities = [] } = useQuery({
    queryKey: ["constellation-entities"],
    queryFn: () => listEntities(),
    refetchInterval: 120_000,
  });

  // Fetch relations for top 20 entities
  const top20Ids = useMemo(
    () => entities.slice(0, 20).map((e: Entity) => e.id),
    [entities],
  );

  const { data: details = [] } = useQuery({
    queryKey: ["constellation-relations", top20Ids],
    queryFn: () => Promise.all(top20Ids.map((id) => getEntityDetail(id))),
    enabled: top20Ids.length > 0,
    refetchInterval: 300_000,
    staleTime: 120_000,
  });

  // Fetch memories when toggle is on (fullScreen only)
  const { data: memories = [] } = useQuery({
    queryKey: ["constellation-memories"],
    queryFn: () => listMemoriesRich(undefined, undefined, undefined, 200),
    enabled: showMemories,
    refetchInterval: 120_000,
  });

  // Build graph data
  const graphData = useMemo(() => {
    if (entities.length === 0) return { nodes: [] as GraphNode[], links: [] as GraphLink[] };

    const edgeSet = new Map<string, GraphLink>();
    const connectionCounts = new Map<string, number>();

    for (const detail of details) {
      for (const rel of detail.relations) {
        const sourceId = detail.entity.id;
        const targetId = rel.entity_id;
        const key = [sourceId, targetId].sort().join("-");
        if (!edgeSet.has(key) && entities.some((e: Entity) => e.id === targetId)) {
          edgeSet.set(key, { source: sourceId, target: targetId });
        }
        connectionCounts.set(sourceId, (connectionCounts.get(sourceId) ?? 0) + 1);
        connectionCounts.set(targetId, (connectionCounts.get(targetId) ?? 0) + 1);
      }
    }

    const nodes: GraphNode[] = entities.map((e: Entity) => ({
      id: e.id,
      name: e.name,
      entityType: e.entity_type,
      stability: e.confirmed ? "confirmed" : "new",
      connectionCount: connectionCounts.get(e.id) ?? 0,
      isMemory: false,
    }));

    const links = [...edgeSet.values()];

    // Add memory nodes when toggle is on
    if (showMemories && memories.length > 0) {
      const entityIds = new Set(entities.map((e: Entity) => e.id));
      // Build name→id map for client-side matching (entity_id is rarely set)
      const entityByName = new Map<string, string>();
      for (const e of entities) {
        entityByName.set(e.name.toLowerCase(), e.id);
      }

      const addedMemIds = new Set<string>();
      for (const mem of memories) {
        // Try entity_id first (most precise)
        let linkedEntityId = mem.entity_id && entityIds.has(mem.entity_id) ? mem.entity_id : null;

        // Fallback: match entity name in memory title or domain
        if (!linkedEntityId) {
          const titleLower = (mem.title || "").toLowerCase();
          const domainLower = (mem.domain || "").toLowerCase();
          for (const [eName, eId] of entityByName) {
            if (eName.length >= 3 && (titleLower.includes(eName) || domainLower === eName)) {
              linkedEntityId = eId;
              break;
            }
          }
        }

        if (linkedEntityId && !addedMemIds.has(mem.source_id)) {
          addedMemIds.add(mem.source_id);
          const memId = `mem:${mem.source_id}`;
          nodes.push({
            id: memId,
            name: mem.title || "",
            entityType: mem.memory_type ?? "fact",
            stability: mem.stability ?? (mem.confirmed ? "confirmed" : "new"),
            connectionCount: 0,
            isMemory: true,
            isDistilled: mem.is_recap === true,
          });
          links.push({ source: memId, target: linkedEntityId });
        }
      }
    }

    return { nodes, links };
  }, [entities, details, showMemories, memories]);

  // Minimap view: drop orphan nodes (entities with zero edges in the current
  // fetch window). The backend returns entities ordered by `updated_at DESC`
  // and we only fetch relations for the top-20 — so `entities` always contains
  // recent-but-unconnected solo nodes. Leaving them in forced `zoomToFit` to
  // accommodate their spread, pulling back so far the dense core became a dot.
  // Filter them out here so the minimap shows only the "meaningful connections"
  // (nodes that actually have an edge in this view). Fullscreen keeps everything.
  const displayGraph = useMemo(() => {
    if (fullScreen) return graphData;
    const connectedIds = new Set<string>();
    for (const l of graphData.links) {
      const src = typeof l.source === "object" ? (l.source as any).id : l.source;
      const tgt = typeof l.target === "object" ? (l.target as any).id : l.target;
      if (src) connectedIds.add(src as string);
      if (tgt) connectedIds.add(tgt as string);
    }
    if (connectedIds.size === 0) return graphData; // nothing to filter
    return {
      nodes: graphData.nodes.filter((n) => connectedIds.has(n.id)),
      links: graphData.links,
    };
  }, [graphData, fullScreen]);

  // Stabilize graphData reference — only update when node/link IDs actually change
  const prevGraphRef = useRef(displayGraph);
  const stableGraphData = useMemo(() => {
    const prev = prevGraphRef.current;
    const sameNodes = prev.nodes.length === displayGraph.nodes.length
      && prev.nodes.every((n, i) => n.id === displayGraph.nodes[i]?.id);
    const sameLinks = prev.links.length === displayGraph.links.length
      && prev.links.every((l, i) => {
        const gl = displayGraph.links[i];
        const lSrc = typeof l.source === "object" ? (l.source as any).id : l.source;
        const glSrc = typeof gl?.source === "object" ? (gl.source as any).id : gl?.source;
        const lTgt = typeof l.target === "object" ? (l.target as any).id : l.target;
        const glTgt = typeof gl?.target === "object" ? (gl.target as any).id : gl?.target;
        return lSrc === glSrc && lTgt === glTgt;
      });
    if (sameNodes && sameLinks) return prev;
    prevGraphRef.current = displayGraph;
    return displayGraph;
  }, [displayGraph]);

  // Stats
  const entityCount = entities.length;
  const connectionCount = graphData.links.length;
  const cornerLabel = useMemo(() => {
    if (entityCount === 0) return "";
    const parts = [`${entityCount} ${entityCount === 1 ? "entity" : "entities"}`];
    if (connectionCount > 0) {
      parts.push(`${connectionCount} ${connectionCount === 1 ? "connection" : "connections"}`);
    }
    return parts.join(", ");
  }, [entityCount, connectionCount]);

  // Zoom to fit is triggered by onEngineStop below — runs when simulation settles

  // Helper: get node radius
  const getNodeRadius = useCallback((node: any) => {
    if (node.isMemory) return node.stability === "confirmed" || node.isDistilled ? 4.5 : 3;
    return nodeRadius(node.stability ?? "new", node.connectionCount);
  }, []);

  // Identify the user's entity — the person node with the most connections.
  // Matches IdentityCard's "first person" convention but picks the most-central
  // person instead of `[0]`, which avoids the "first recently-updated contact
  // isn't you" failure mode when the graph has many person entities.
  const userNodeId = useMemo(() => {
    const persons = displayGraph.nodes
      .filter((n) => !n.isMemory && n.entityType === "person")
      .sort((a, b) => (b.connectionCount ?? 0) - (a.connectionCount ?? 0));
    return persons[0]?.id ?? null;
  }, [displayGraph.nodes]);

  // Top-N hubs by connection count (excluding memories — those are embers,
  // not named entities). In the minimap N is fixed at 3 (the sweet spot you
  // landed on), except when the graph is smaller than 3 — then we show
  // whatever's there so the view isn't padded with nothing.
  const topHubIds = useMemo(() => {
    const sorted = [...displayGraph.nodes]
      .filter((n) => !n.isMemory)
      .sort((a, b) => (b.connectionCount ?? 0) - (a.connectionCount ?? 0));
    const N = fullScreen ? 20 : Math.min(3, sorted.length);
    return sorted.slice(0, N).map((n) => n.id);
  }, [displayGraph.nodes, fullScreen]);

  // Which nodes get persistent text labels. In fullscreen that's the top-20
  // hubs (earlier fix for overlapping labels). In the minimap it's the
  // auto-tuned top hubs + the user node — "me + my biggest connections".
  const labeledNodeIds = useMemo(() => {
    const ids = new Set<string>(topHubIds);
    if (userNodeId) ids.add(userNodeId);
    return ids;
  }, [topHubIds, userNodeId]);

  // The set of nodes we want the camera to frame when in minimap mode:
  // the user + top hubs. `zoomToFit`'s third arg is a node filter that only
  // those nodes contribute to the computed bbox. Other nodes still render
  // but the camera aims at the "user and your top hubs" cluster.
  const focusNodeIds = labeledNodeIds;

  // Auto-tuned fit padding for the minimap. The hub count is normally pinned
  // to 3, so the common case is 3 hubs + user = 4 focus nodes. That lands in
  // the 44-padding sweet spot. Edge cases (fewer hubs available, user is
  // already a hub so focus set dedupes to 3) stay in the same band. Larger
  // focus sets only happen if we ever revisit the N-auto-tune.
  const minimapFitPadding = useMemo(() => {
    const n = focusNodeIds.size;
    if (n <= 2) return 88;   // lonely cluster — lots of margin so it doesn't feel isolated
    if (n <= 5) return 72;   // cozy sweet spot — more breathing room, zoomed out
    if (n <= 8) return 50;   // denser — tighter so content doesn't shrink
    return 30;               // very dense — minimal margin
  }, [focusNodeIds]);

  // Re-apply the minimap fit whenever the padding value or focus set changes.
  // `onEngineStop` only fires once (the simulation is frozen with
  // `cooldownTicks={0}`), so tweaking `minimapFitPadding` at dev-time or when
  // the focus nodes change without the engine restarting would never reach
  // the camera otherwise. This effect catches those cases — it's the
  // canonical path for re-centering the minimap after config or data changes.
  useEffect(() => {
    if (fullScreen) return;
    if (stableGraphData.nodes.length === 0) return;
    // Tiny delay: on first mount the ForceGraph ref may not be populated yet,
    // and we want the layout to have applied before we fit.
    const t = setTimeout(() => {
      if (!fgRef.current) return;
      if (focusNodeIds.size > 0) {
        fgRef.current.zoomToFit(
          300,
          minimapFitPadding,
          (n: any) => focusNodeIds.has(n.id),
        );
      } else {
        fgRef.current.zoomToFit(300, 20);
      }
    }, 120);
    return () => clearTimeout(t);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [fullScreen, minimapFitPadding, focusNodeIds, stableGraphData]);

  // Configure the d3 charge force via ref (not available as JSX props in types)
  useEffect(() => {
    if (!fgRef.current) return;
    fgRef.current.d3Force("charge")?.strength(-40);
  }, [stableGraphData]);

  // Custom node rendering. `globalScale` is the current zoom factor from
  // react-force-graph — used to keep label size screen-constant so text
  // doesn't balloon when zoomed in or shrink to invisible when zoomed out.
  const paintNode = useCallback((node: any, ctx: CanvasRenderingContext2D, globalScale: number) => {
    const colors = getThemeColors();

    // Memory nodes: warm embers
    if (node.isMemory) {
      const isStrong = node.stability === "confirmed" || node.isDistilled;
      const isLearned = node.stability === "learned";
      const r = isStrong ? 4.5 : 3;
      const warmColor = isStrong ? "#E8A87C" : "#C49878";

      if (isStrong) {
        ctx.beginPath();
        ctx.arc(node.x, node.y, r + 3, 0, 2 * Math.PI);
        ctx.fillStyle = "#E8A87C";
        ctx.globalAlpha = 0.12;
        ctx.fill();
        ctx.globalAlpha = 1;
      }

      ctx.beginPath();
      ctx.arc(node.x, node.y, r, 0, 2 * Math.PI);
      ctx.fillStyle = warmColor;
      ctx.globalAlpha = isStrong ? 0.8 : isLearned ? 0.55 : 0.3;
      ctx.fill();
      ctx.globalAlpha = 1;
      return;
    }

    const r = nodeRadius(node.stability ?? "new", node.connectionCount);
    const color = colorForType(node.entityType, colors);

    // Subtle glow for confirmed
    if (node.stability === "confirmed") {
      ctx.beginPath();
      ctx.arc(node.x, node.y, r + 3, 0, 2 * Math.PI);
      ctx.fillStyle = color;
      ctx.globalAlpha = 0.1;
      ctx.fill();
      ctx.globalAlpha = 1;
    }

    // Main node
    ctx.beginPath();
    ctx.arc(node.x, node.y, r, 0, 2 * Math.PI);
    ctx.fillStyle = color;
    ctx.globalAlpha = node.stability === "confirmed" ? 0.9 : node.stability === "learned" ? 0.7 : 0.5;
    ctx.fill();
    ctx.globalAlpha = 1;

    // Label — only for top-N hub nodes (minimap: top 3 + user;
    // fullscreen: top 20), when the label toggle is on.
    if (showLabels && labeledNodeIds.has(node.id)) {
      // Use the tertiary text color + muted alpha — matches the quieter
      // "previous" aesthetic. Screen-constant sizing stays (we still want
      // labels legible at any zoom level).
      const labelColor = getComputedStyle(document.documentElement)
        .getPropertyValue("--mem-text-tertiary").trim() || "#6a6a8a";

      const screenFontPx = 12;
      const screenPadPx = 8;
      const fontSize = screenFontPx / globalScale;
      const pad = r + screenPadPx / globalScale;

      // Position in one of 4 directions based on node's angle from center.
      const angle = Math.atan2(node.y, node.x);
      const sector = Math.round((angle + Math.PI) / (Math.PI / 2)) % 4;

      ctx.font = `${fontSize}px -apple-system, sans-serif`;
      ctx.fillStyle = labelColor;
      ctx.globalAlpha = 0.85;

      if (sector === 0 || sector === 2) {
        // Left or right
        const isRight = sector === 0;
        ctx.textAlign = isRight ? "left" : "right";
        ctx.textBaseline = "middle";
        ctx.fillText(node.name, node.x + (isRight ? pad : -pad), node.y);
      } else {
        // Above or below
        const isBelow = sector === 1;
        ctx.textAlign = "center";
        ctx.textBaseline = isBelow ? "top" : "bottom";
        ctx.fillText(node.name, node.x, node.y + (isBelow ? pad : -pad));
      }
      ctx.globalAlpha = 1;
    }
  }, [showLabels, fullScreen, labeledNodeIds]);

  // Custom link rendering — lines stop at node borders
  const paintLink = useCallback((link: any, ctx: CanvasRenderingContext2D) => {
    const src = link.source;
    const tgt = link.target;
    if (!src || !tgt || src.x == null || tgt.x == null) return;

    const dx = tgt.x - src.x;
    const dy = tgt.y - src.y;
    const dist = Math.sqrt(dx * dx + dy * dy);
    if (dist === 0) return;

    const srcR = getNodeRadius(src);
    const tgtR = getNodeRadius(tgt);

    // Shorten line by each node's radius
    const ux = dx / dist;
    const uy = dy / dist;
    const x1 = src.x + ux * srcR;
    const y1 = src.y + uy * srcR;
    const x2 = tgt.x - ux * tgtR;
    const y2 = tgt.y - uy * tgtR;

    const isDark = document.documentElement.getAttribute("data-theme") !== "light";
    const isMemLink = src.isMemory || tgt.isMemory;

    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.lineTo(x2, y2);
    ctx.strokeStyle = isMemLink
      ? (isDark ? "rgba(232, 168, 124, 0.15)" : "rgba(192, 117, 80, 0.12)")
      : (isDark ? "rgba(180, 165, 200, 0.4)" : "rgba(100, 85, 130, 0.3)");
    ctx.lineWidth = isMemLink ? 0.5 : 1;
    if (isMemLink) ctx.setLineDash([2, 3]);
    else ctx.setLineDash([]);
    ctx.stroke();
    ctx.setLineDash([]);
  }, [getNodeRadius]);

  // Empty state
  if (entities.length === 0) {
    return (
      <div
        data-testid="constellation-map"
        ref={containerRef}
        onClick={onClick}
        style={{
          height: fullScreen ? "100%" : 280,
          width: "100%",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          background: "var(--mem-surface)",
          border: fullScreen ? "none" : "1px solid var(--mem-border)",
          borderRadius: fullScreen ? 0 : "var(--radius-md, 10px)",
          cursor: onClick ? "pointer" : undefined,
          fontFamily: "var(--mem-font-body)",
        }}
      >
        <span style={{ color: "var(--mem-text-tertiary)", fontSize: 13 }}>
          Your constellation will appear as knowledge grows
        </span>
      </div>
    );
  }

  return (
    <div
      data-testid="constellation-map"
      ref={containerRef}
      style={{
        position: "relative",
        height: fullScreen ? "100%" : 280,
        width: "100%",
        background: "var(--mem-surface)",
        border: fullScreen ? "none" : "1px solid var(--mem-border)",
        borderRadius: fullScreen ? 0 : "var(--radius-md, 10px)",
        overflow: "hidden",
        cursor: onClick ? "pointer" : undefined,
      }}
    >
      <ForceGraph2D
        ref={fgRef}
        graphData={stableGraphData}
        width={dimensions.width}
        height={dimensions.height}
        nodeCanvasObject={paintNode}
        nodePointerAreaPaint={(node: any, color: string, ctx: CanvasRenderingContext2D) => {
          const r = node.isMemory ? 3 : nodeRadius(node.stability ?? "new", node.connectionCount);
          ctx.beginPath();
          ctx.arc(node.x, node.y, r + 2, 0, 2 * Math.PI);
          ctx.fillStyle = color;
          ctx.fill();
        }}
        linkCanvasObject={paintLink}
        linkCanvasObjectMode={() => "replace"}
        nodeLabel={(node: any) => node.name}
        nodeRelSize={1}
        nodeVal={(node: any) => {
          if (node.isMemory) return node.stability === "confirmed" || node.isDistilled ? 4.5 : 3;
          return nodeRadius(node.stability ?? "new", node.connectionCount);
        }}
        backgroundColor="rgba(0,0,0,0)"
        enableNodeDrag={fullScreen}
        enableZoomInteraction={fullScreen}
        enablePanInteraction={fullScreen}
        warmupTicks={fullScreen ? 0 : 100}
        cooldownTicks={fullScreen ? 100 : 0}
        d3AlphaDecay={0.03}
        d3VelocityDecay={0.25}
        onNodeClick={(node: any) => {
          if (onNodeClick) {
            // Memory nodes pass source_id (without mem: prefix), entity nodes pass entity id
            onNodeClick(node.isMemory ? `memory:${node.id.replace("mem:", "")}` : node.id);
          } else {
            onClick?.();
          }
        }}
        onBackgroundClick={() => {
          if (!fullScreen) {
            onClick?.();
          }
        }}
        onEngineStop={() => {
          if (fullScreen) {
            // Negative padding here bleeds the bbox past the viewport edges,
            // producing a noticeable "zoom in to see meaningful connections"
            // effect. Without it, the graph sat comfortably in the middle with
            // a band of empty space top and bottom.
            fgRef.current?.zoomToFit?.(400, -40);
          } else {
            // Minimap: fit ONLY user + top hubs. Other nodes still render but
            // the camera frames the meaningful core. This is the "user + hub
            // together" centering — the viewport is aimed at the smallest bbox
            // that contains both your entity and your most-connected hubs.
            // Fallback: if we have no focus nodes yet, fit everything.
            if (focusNodeIds.size > 0) {
              fgRef.current?.zoomToFit?.(
                0,
                minimapFitPadding,
                (n: any) => focusNodeIds.has(n.id),
              );
            } else {
              fgRef.current?.zoomToFit?.(0, 8);
            }
          }
        }}
      />

      {/* Corner label */}
      <div
        style={{
          position: "absolute",
          bottom: 6,
          right: 8,
          fontSize: 10,
          color: "var(--mem-text-tertiary)",
          fontFamily: "var(--mem-font-body)",
          pointerEvents: "none",
        }}
      >
        {cornerLabel}
      </div>

      {/* Legend — fullScreen only, top-right */}
      {fullScreen && (
        <div
          style={{
            position: "absolute",
            top: 10,
            right: 10,
            display: "flex",
            flexDirection: "column",
            gap: 3,
            pointerEvents: "none",
          }}
        >
          {/* Legend box */}
          <div
            style={{
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
            }}
          >
            {[
              { label: "Person", key: "person" },
              { label: "Project", key: "project" },
              { label: "Technology", key: "technology" },
              { label: "Organization", key: "organization" },
              { label: "Place", key: "place" },
              { label: "Concept", key: "concept" },
              { label: "Event", key: "event" },
            ].map(({ label, key }) => (
              <div key={key} style={{ display: "flex", alignItems: "center", gap: 5 }}>
                <span
                  style={{
                    display: "inline-block",
                    width: 8,
                    height: 8,
                    borderRadius: "50%",
                    backgroundColor: colorForType(key, getThemeColors()),
                    opacity: 0.7,
                    flexShrink: 0,
                  }}
                />
                <span>{label}</span>
              </div>
            ))}
            {showMemories && (
              <div style={{ display: "flex", alignItems: "center", gap: 5 }}>
                <span
                  style={{
                    display: "inline-block",
                    width: 6,
                    height: 6,
                    borderRadius: "50%",
                    backgroundColor: "#E8A87C",
                    opacity: 0.7,
                    flexShrink: 0,
                    marginLeft: 1,
                  }}
                />
                <span style={{ opacity: 0.7 }}>Memory</span>
              </div>
            )}
            <div style={{ display: "flex", alignItems: "center", gap: 5 }}>
              <span style={{
                display: "inline-block",
                width: 12,
                height: 0,
                borderTop: "1px solid var(--mem-text-tertiary)",
                opacity: 0.5,
                flexShrink: 0,
              }} />
              <span>Connection</span>
            </div>
          </div>

          {/* Toggles — grouped below legend */}
          <div style={{
            display: "flex",
            flexDirection: "column",
            gap: 2,
            padding: "4px 6px",
            background: "var(--mem-surface)",
            border: "1px solid var(--mem-border)",
            borderRadius: 6,
            opacity: 0.85,
            pointerEvents: "auto",
          }}>
            {[
              { label: "Memories", on: showMemories, toggle: () => setShowMemories((v) => { const next = !v; localStorage.setItem("constellation-show-memories", String(next)); return next; }), testId: "memory-toggle" },
              { label: "Labels", on: showLabels, toggle: () => setShowLabels((v) => { const next = !v; localStorage.setItem("constellation-show-labels", String(next)); return next; }) },
            ].map(({ label, on, toggle, testId }) => (
              <button
                key={label}
                onClick={(e) => { e.stopPropagation(); toggle(); }}
                data-testid={testId}
                style={{
                  display: "flex",
                  alignItems: "center",
                  gap: 4,
                  padding: "2px 4px",
                  fontSize: 10,
                  fontFamily: "var(--mem-font-body)",
                  color: on ? "var(--mem-text-primary)" : "var(--mem-text-tertiary)",
                  background: "none",
                  border: "none",
                  cursor: "pointer",
                  opacity: on ? 1 : 0.5,
                }}
              >
                <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  {on
                    ? <><rect x="3" y="3" width="18" height="18" rx="3" /><path d="M9 12l2 2 4-4" /></>
                    : <rect x="3" y="3" width="18" height="18" rx="3" />
                  }
                </svg>
                {label}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Explore button — minimap only */}
      {!fullScreen && onClick && (
        <button
          onClick={(e) => { e.stopPropagation(); onClick(); }}
          className="transition-opacity duration-200"
          style={{
            position: "absolute",
            top: 10,
            right: 10,
            display: "flex",
            alignItems: "center",
            gap: 4,
            padding: "4px 10px",
            fontSize: 11,
            fontFamily: "var(--mem-font-body)",
            fontWeight: 500,
            color: "var(--mem-text-secondary)",
            background: "var(--mem-surface)",
            border: "1px solid var(--mem-border)",
            borderRadius: 6,
            cursor: "pointer",
            opacity: 0.7,
            backdropFilter: "blur(8px)",
          }}
          onMouseEnter={(e) => { e.currentTarget.style.opacity = "1"; }}
          onMouseLeave={(e) => { e.currentTarget.style.opacity = "0.7"; }}
        >
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <polyline points="15 3 21 3 21 9" />
            <polyline points="9 21 3 21 3 15" />
            <line x1="21" y1="3" x2="14" y2="10" />
            <line x1="3" y1="21" x2="10" y2="14" />
          </svg>
          Explore
        </button>
      )}
    </div>
  );
}
