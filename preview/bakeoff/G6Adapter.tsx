// SPDX-License-Identifier: AGPL-3.0-only
// @antv/g6 v5 candidate (canvas/WebGL runtime, built-in "force" layout) for
// the bake-off. Preview-only, untested per spec.
import { useEffect, useRef } from "react";
import { Graph, NodeEvent, type NodeData } from "@antv/g6";
import { useGraphPalette, colorForEntityType } from "../../src/lib/graph/palette";
import type { BakeoffModel } from "./synthetic";
import { hubIds, sizeForDegree, withLayoutTimeout, writeBakeoffResult } from "./bakeoffResult";

export default function G6Adapter({ model }: { model: BakeoffModel }) {
  const containerRef = useRef<HTMLDivElement>(null);
  const graphRef = useRef<Graph | null>(null);
  const palette = useGraphPalette();

  // Full remount on theme flip and StrictMode double-invoke, same as the
  // other two adapters — see CytoscapeAdapter's comment.
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;
    let cancelled = false;

    const t0 = performance.now();
    const hubs = hubIds(model.nodes);
    const graph = new Graph({
      container,
      animation: false,
      data: {
        nodes: model.nodes.map((node) => ({
          id: node.id,
          style: { x: node.x, y: node.y },
          data: { entityType: node.entityType, degree: node.degree },
        })),
        edges: model.edges.map((edge) => ({ id: edge.id, source: edge.source, target: edge.target })),
      },
      node: {
        style: {
          fill: (d: NodeData) => colorForEntityType(String(d.data?.entityType), palette),
          size: (d: NodeData) => sizeForDegree(Number(d.data?.degree)),
          labelText: (d: NodeData) => (hubs.has(String(d.id)) ? String(d.id) : ""),
        },
      },
      edge: { style: { stroke: palette.edge, lineWidth: 1 } },
      layout: { type: "force", animation: false },
      behaviors: ["zoom-canvas", "drag-canvas"],
    });
    graphRef.current = graph;
    const t1 = performance.now();

    graph.on(NodeEvent.CLICK, (evt: { target?: { id?: string } }) => {
      console.log("[bakeoff] g6 node click:", evt.target?.id);
    });

    withLayoutTimeout(graph.render()).then(({ timedOut }) => {
      if (cancelled) return;
      const t2 = performance.now();
      if (timedOut) {
        // Finding: force layout did not settle synchronously at this n — the
        // nodes already carry the generator's precomputed x/y as their
        // initial style, so there's nothing further to reset here.
        graph.stopLayout();
      }
      writeBakeoffResult({
        renderer: "g6",
        n: model.nodes.length,
        buildMs: t1 - t0,
        layoutMs: t2 - t1,
        mountMs: t2 - t0,
        nodesDrawn: graph.getNodeData().length,
        ...(timedOut ? { error: "force layout did not settle synchronously; used precomputed positions" } : {}),
      });
    });

    return () => {
      cancelled = true;
      graphRef.current = null;
      graph.destroy();
    };
  }, [model, palette]);

  // Container-driven resize, same pattern as ConstellationMap's ResizeObserver:
  // no-arg resize() fits the canvas back to the container's current size.
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;
    const ro = new ResizeObserver(() => graphRef.current?.resize());
    ro.observe(container);
    return () => ro.disconnect();
  }, []);

  return <div ref={containerRef} style={{ width: "100%", height: "100%" }} />;
}
