// SPDX-License-Identifier: AGPL-3.0-only
import { useEffect, useId, useMemo, useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import type { EntityDetail } from "../../lib/tauri";
import { buildEgoModel, type GraphModel } from "../../lib/graph/model";
import { slotForEntityType, useGraphPalette } from "../../lib/graph/palette";

const GRAPH_NODE_CAP = 8;

interface FocusGraphProps {
  detail: EntityDetail;
  onEntityClick: (entityId: string) => void;
  // When set, height comes from the container (100%) instead of the
  // content-driven cap — for a full-screen host, not more graph data.
  fill?: boolean;
  // Verb labels on the edges — on by default; the overlay's "Show verbs"
  // chip is the only thing that turns them off.
  showVerbs?: boolean;
  // Linked-memory count for this entity. In fill mode it renders as the
  // artifact's small dot cluster hanging off the center node; the accessible
  // count lives in the overlay toolbar, so the cluster is decoration.
  memoriesCount?: number;
}

interface FocusNeighbor {
  entityId: string;
  name: string;
  entityType: string;
  verbs: string[];
  direction: "incoming" | "outgoing";
  confidence: number | null;
}

// A neighbor node aggregates every edge to it in one direction; confidence is
// the weakest link (null anywhere → unknown, so no fake opacity downstream).
function combineConfidence(a: number | null, b: number | null): number | null {
  if (a == null || b == null) return null;
  return Math.min(a, b);
}

// Collapse the ego model's edges into one display node per (neighbor,
// direction), verbs gathered onto the edge — the current EntityGraph shape,
// now sourced from GraphModel instead of raw relations.
function deriveNeighbors(model: GraphModel, centerId: string): FocusNeighbor[] {
  const nodeById = new Map(model.nodes.map((n) => [n.id, n]));
  const byKey = new Map<string, FocusNeighbor>();
  for (const edge of model.edges) {
    const outgoing = edge.source === centerId;
    const neighborId = outgoing ? edge.target : edge.source;
    if (neighborId === centerId) continue; // self-loop: center is not its own neighbor
    const node = nodeById.get(neighborId);
    if (!node) continue;
    const direction: "incoming" | "outgoing" = outgoing ? "outgoing" : "incoming";
    const key = `${neighborId}:${direction}`;
    const existing = byKey.get(key);
    if (existing) {
      if (!existing.verbs.includes(edge.type)) existing.verbs.push(edge.type);
      existing.confidence = combineConfidence(existing.confidence, edge.confidence);
    } else {
      byKey.set(key, {
        entityId: neighborId,
        name: node.name,
        entityType: node.entityType,
        verbs: [edge.type],
        direction,
        confidence: edge.confidence,
      });
    }
  }
  return Array.from(byKey.values());
}

interface PlacedNeighbor extends FocusNeighbor {
  x: number; // percent of container width
  y: number; // percent of container height
}

/**
 * One-hop ego view: this entity centered, incoming relations arced left,
 * outgoing arced right. SVG edge layer carries arrowheads along the semantic
 * direction; DOM `<button>`s keep every neighbor keyboard-reachable (the
 * canvas renderer can't). Consumes buildEgoModel + useGraphPalette so
 * no daemon shape is read for drawing and colors track the theme.
 */
export default function FocusGraph({
  detail,
  onEntityClick,
  fill,
  showVerbs = true,
  memoriesCount = 0,
}: FocusGraphProps) {
  const { t } = useTranslation();
  const palette = useGraphPalette();
  const rawId = useId();
  const markerId = `focus-arrow-${rawId.replace(/:/g, "")}`;

  const model = useMemo(() => buildEgoModel(detail), [detail]);
  const centerId = detail.entity.id;
  const allNeighbors = useMemo(() => deriveNeighbors(model, centerId), [model, centerId]);

  // GRAPH_NODE_CAP is one shared budget across BOTH directions, not a
  // per-direction cap — balanced below so one lopsided side can't crowd the
  // other out of the map (ported verbatim from the old EntityGraph placement
  // in EntityDetail).
  const incomingAll = allNeighbors.filter((n) => n.direction === "incoming");
  const outgoingAll = allNeighbors.filter((n) => n.direction === "outgoing");
  const outgoingShown = Math.min(
    outgoingAll.length,
    GRAPH_NODE_CAP - Math.min(incomingAll.length, Math.ceil(GRAPH_NODE_CAP / 2)),
  );
  const shown = [
    ...incomingAll.slice(0, GRAPH_NODE_CAP - outgoingShown),
    ...outgoingAll.slice(0, outgoingShown),
  ];
  const hiddenCount = allNeighbors.length - shown.length;

  const shownIncoming = shown.filter((n) => n.direction === "incoming");
  const shownOutgoing = shown.filter((n) => n.direction === "outgoing");
  const maxSide = Math.max(shownIncoming.length, shownOutgoing.length);
  const contentHeight = Math.min(280, 128 + maxSide * 30);

  // viewBox width tracks the rendered pixel width so arrowhead markers render
  // undistorted (percent-positioned DOM nodes stay aligned regardless). In
  // fill mode, height is container-driven (CSS 100%) rather than the content
  // cap, so it's tracked the same way as width to keep the viewBox accurate.
  const containerRef = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState(400);
  const [measuredHeight, setMeasuredHeight] = useState(400);
  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const ro = new ResizeObserver((entries) => {
      for (const entry of entries) {
        setWidth(entry.contentRect.width || 400);
        setMeasuredHeight(entry.contentRect.height || 400);
      }
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);
  const height = fill ? measuredHeight : contentHeight;

  const place = (list: FocusNeighbor[], side: "left" | "right"): PlacedNeighbor[] =>
    list.map((n, i) => {
      const y = ((i + 1) / (list.length + 1)) * 100;
      const spread = (i % 2) * 6;
      const x = side === "right" ? 76 - spread : 24 + spread;
      return { ...n, x, y };
    });
  const placed = [...place(shownIncoming, "left"), ...place(shownOutgoing, "right")];

  const centerColor = palette[slotForEntityType(detail.entity.entity_type)];

  // Artifact screen 02: memories hang off the center as a small dot cluster on
  // a thin plain edge — attached to their entity, not dust across the map.
  // Fill mode only; the inline card is too shallow for the offset.
  const memoryCluster = fill && memoriesCount > 0;
  const clusterDX = 72;
  const clusterDY = 112;

  return (
    <div
      ref={containerRef}
      className="entity-graph"
      style={{ height: fill ? "100%" : contentHeight }}
      role="group"
      aria-label={t("entityDetail.graphLabel", { name: detail.entity.name })}
    >
      <svg
        className="entity-graph-edges"
        viewBox={`0 0 ${width} ${height}`}
        preserveAspectRatio="none"
        aria-hidden="true"
      >
        <defs>
          <marker
            id={markerId}
            viewBox="0 0 8 8"
            refX="7"
            refY="4"
            markerWidth="7"
            markerHeight="7"
            orient="auto"
          >
            <path d="M0 0 L8 4 L0 8 Z" fill={palette.edgeStrong} />
          </marker>
        </defs>
        {placed.map((n) => {
          const nx = (n.x / 100) * width;
          const ny = (n.y / 100) * height;
          const cx = width / 2;
          const cy = height / 2;
          const outgoing = n.direction === "outgoing";
          // Draw source→target so the arrowhead lands on the semantic object:
          // outgoing points at the neighbor, incoming points back at center.
          const x1 = outgoing ? cx : nx;
          const y1 = outgoing ? cy : ny;
          const x2 = outgoing ? nx : cx;
          const y2 = outgoing ? ny : cy;
          return (
            <line
              key={`edge-${n.entityId}-${n.direction}`}
              x1={x1}
              y1={y1}
              x2={x2}
              y2={y2}
              style={{
                stroke: palette.edgeStrong,
                strokeWidth: 1.3,
                strokeOpacity: Math.max(n.confidence ?? 1, 0.15),
              }}
              markerEnd={`url(#${markerId})`}
            />
          );
        })}
        {memoryCluster && (
          <line
            x1={width / 2}
            y1={height / 2}
            x2={width / 2 + clusterDX - 10}
            y2={height / 2 + clusterDY - 14}
            style={{ stroke: palette.edge, strokeWidth: 1 }}
          />
        )}
      </svg>
      {shownIncoming.length > 0 && (
        <span className="entity-graph-caption is-left" aria-hidden="true">
          ← {t("entityDetail.incoming")}
        </span>
      )}
      {shownOutgoing.length > 0 && (
        <span className="entity-graph-caption is-right" aria-hidden="true">
          {t("entityDetail.outgoing")} →
        </span>
      )}
      {showVerbs &&
        placed.map((n) => (
          <span
            key={`verb-${n.entityId}-${n.direction}`}
            className="entity-graph-verb"
            style={{ left: `${50 + (n.x - 50) * 0.55}%`, top: `${50 + (n.y - 50) * 0.55}%` }}
            aria-hidden="true"
          >
            {n.verbs[0]}
            {n.verbs.length > 1 ? ` +${n.verbs.length - 1}` : ""}
          </span>
        ))}
      <div className="entity-graph-center" aria-hidden="true">
        <span
          className="entity-graph-center-dot focus-graph-center-dot"
          style={{ backgroundColor: centerColor }}
        />
        <span className="entity-graph-center-name">{detail.entity.name}</span>
      </div>
      {placed.map((n) => {
        const isIncoming = n.direction === "incoming";
        const dirLabel = t(isIncoming ? "entityDetail.incoming" : "entityDetail.outgoing");
        const color = palette[slotForEntityType(n.entityType)];
        return (
          <button
            key={`node-${n.entityId}-${n.direction}`}
            type="button"
            className={`entity-graph-node ${isIncoming ? "is-left" : "is-right"}`}
            style={{ left: `${n.x}%`, top: `${n.y}%` }}
            onClick={() => onEntityClick(n.entityId)}
            title={`${n.verbs.join(", ")} — ${n.name}`}
            aria-label={`${n.name} (${n.entityType}) · ${dirLabel} · ${n.verbs.join(", ")}`}
          >
            <span
              className="entity-graph-node-dot"
              style={{ backgroundColor: color, borderColor: color }}
              aria-hidden="true"
            />
            <span className="entity-graph-node-text">
              <span className="entity-graph-node-name">{n.name}</span>
              <span className="entity-graph-node-type">{n.entityType}</span>
            </span>
          </button>
        );
      })}
      {memoryCluster && (
        <span
          aria-hidden="true"
          style={{
            position: "absolute",
            left: `calc(50% + ${clusterDX}px)`,
            top: `calc(50% + ${clusterDY}px)`,
            transform: "translateY(-50%)",
            display: "flex",
            alignItems: "center",
            gap: 6,
          }}
        >
          <span style={{ display: "flex", gap: 3 }}>
            {Array.from({ length: Math.min(memoriesCount, 3) }, (_, i) => (
              <span
                key={i}
                style={{
                  width: 6,
                  height: 6,
                  borderRadius: "50%",
                  background: "var(--mem-accent-indigo)",
                  opacity: 0.6,
                  // Loose cluster, not a row — echoes the artifact's scatter.
                  transform: `translateY(${(i % 2) * 5 - 2}px)`,
                }}
              />
            ))}
          </span>
          <span style={{ font: "400 10px var(--mem-font-mono)", color: "var(--mem-text-tertiary)" }}>
            {t("focus.memoriesLabel", { count: memoriesCount })}
          </span>
        </span>
      )}
      {fill && (
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
          {t("focus.hint")}
        </span>
      )}
      {hiddenCount > 0 && (
        <span className="entity-graph-more">
          {t("entityDetail.moreConnections", { count: hiddenCount })}
        </span>
      )}
    </div>
  );
}

// Exposed for direct unit exercise of the neighbor-collapse logic without a DOM.
export { deriveNeighbors };
