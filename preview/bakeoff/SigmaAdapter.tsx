// SPDX-License-Identifier: AGPL-3.0-only
// sigma v3 + graphology candidate (WebGL renderer, no canvas fallback — note
// for the decision record) for the bake-off. Preview-only, untested per spec.
import { useEffect, useRef } from "react";
import Graph from "graphology";
import Sigma from "sigma";
import forceAtlas2 from "graphology-layout-forceatlas2";
import { useGraphPalette, colorForEntityType } from "../../src/lib/graph/palette";
import type { BakeoffModel } from "./synthetic";
import { hubIds, sizeForDegree, writeBakeoffResult } from "./bakeoffResult";

export default function SigmaAdapter({ model }: { model: BakeoffModel }) {
  const containerRef = useRef<HTMLDivElement>(null);
  const rendererRef = useRef<Sigma | null>(null);
  const palette = useGraphPalette();

  // Full remount on theme flip and StrictMode double-invoke, same as the
  // other two adapters — see CytoscapeAdapter's comment.
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const t0 = performance.now();
    const hubs = hubIds(model.nodes);
    const graph = new Graph();
    // Seed positions come straight from the generator's deterministic x/y —
    // forceAtlas2 needs a starting layout to refine, and reusing the
    // generator's fallback geometry means we don't need a second seed scheme.
    for (const node of model.nodes) {
      graph.addNode(node.id, {
        x: node.x,
        y: node.y,
        size: sizeForDegree(node.degree),
        color: colorForEntityType(node.entityType, palette),
        label: hubs.has(node.id) ? node.id : null,
      });
    }
    for (const edge of model.edges) {
      // mergeEdge is idempotent — the generator can produce a rare duplicate
      // pair (a bridge edge landing on an already-connected pair), and a plain
      // addEdge would throw on the second call.
      graph.mergeEdge(edge.source, edge.target);
    }
    const t1 = performance.now();

    // forceAtlas2.assign is a synchronous, positions-mutate-in-place call —
    // no rAF, no event to await, so there's no "did it settle" ambiguity here
    // the way there is for cytoscape/G6.
    forceAtlas2.assign(graph, { iterations: 100 });
    const t2 = performance.now();

    const renderer = new Sigma(graph, container);
    rendererRef.current = renderer;
    renderer.on("clickNode", ({ node }) => {
      console.log("[bakeoff] sigma node click:", node);
    });
    const t3 = performance.now();

    writeBakeoffResult({
      renderer: "sigma",
      n: model.nodes.length,
      buildMs: t1 - t0,
      layoutMs: t2 - t1,
      mountMs: t3 - t0,
      nodesDrawn: graph.order,
    });

    return () => {
      rendererRef.current = null;
      renderer.kill();
    };
  }, [model, palette]);

  // Container-driven resize, same pattern as ConstellationMap's ResizeObserver.
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;
    const ro = new ResizeObserver(() => rendererRef.current?.resize());
    ro.observe(container);
    return () => ro.disconnect();
  }, []);

  return <div ref={containerRef} style={{ width: "100%", height: "100%" }} />;
}
