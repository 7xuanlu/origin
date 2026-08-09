// SPDX-License-Identifier: AGPL-3.0-only
// Cytoscape.js candidate (canvas renderer) for the bake-off. Preview-only,
// untested per the spec — same convention as the rest of preview/.
import { useEffect, useRef } from "react";
import cytoscape from "cytoscape";
import { useGraphPalette, colorForEntityType } from "../../src/lib/graph/palette";
import type { BakeoffModel } from "./synthetic";
import { hubIds, sizeForDegree, withLayoutTimeout, writeBakeoffResult } from "./bakeoffResult";

export default function CytoscapeAdapter({ model }: { model: BakeoffModel }) {
  const containerRef = useRef<HTMLDivElement>(null);
  const cyRef = useRef<cytoscape.Core | null>(null);
  const palette = useGraphPalette();

  // Re-runs on every theme flip (palette changes), tearing down and rebuilding
  // the whole cytoscape instance — a full remount, not an in-place recolor.
  // Cheapest correct option for a spike; the decision record should note it.
  // The teardown in the effect cleanup also makes StrictMode's double-invoke
  // (mount → cleanup → mount) safe: each cy instance is destroyed before the
  // next is created.
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;
    let cancelled = false;

    const t0 = performance.now();
    const hubs = hubIds(model.nodes);
    const byId = new Map(model.nodes.map((node) => [node.id, node]));
    const cy = cytoscape({
      container,
      elements: [
        ...model.nodes.map((node) => ({
          data: { id: node.id, entityType: node.entityType, degree: node.degree },
        })),
        ...model.edges.map((edge) => ({
          data: { id: edge.id, source: edge.source, target: edge.target },
        })),
      ],
      style: [
        {
          selector: "node",
          style: {
            "background-color": (ele: cytoscape.NodeSingular) =>
              colorForEntityType(ele.data("entityType"), palette),
            width: (ele: cytoscape.NodeSingular) => sizeForDegree(ele.data("degree")),
            height: (ele: cytoscape.NodeSingular) => sizeForDegree(ele.data("degree")),
            label: (ele: cytoscape.NodeSingular) => (hubs.has(ele.id()) ? ele.data("id") : ""),
            "font-size": 8,
            color: palette.neutral,
          },
        },
        {
          selector: "edge",
          style: { width: 1, "line-color": palette.edge, "curve-style": "haystack" },
        },
      ],
      // preset (no positions yet) — the cose layout below assigns them.
      layout: { name: "preset" },
    });
    cyRef.current = cy;
    const t1 = performance.now();

    cy.on("tap", "node", (evt) => {
      console.log("[bakeoff] cytoscape node click:", evt.target.id());
    });

    const layout = cy.layout({ name: "cose", animate: false });
    const settled = new Promise<void>((resolve) => layout.one("layoutstop", () => resolve()));
    layout.run();

    withLayoutTimeout(settled).then(({ timedOut }) => {
      if (cancelled) return;
      const t2 = performance.now();
      if (timedOut) {
        // Finding: cose did not settle synchronously at this n — fall back to
        // the generator's precomputed positions instead of an indefinite wait.
        cy.nodes().positions((ele) => {
          const node = byId.get(ele.id());
          return node ? { x: node.x, y: node.y } : { x: 0, y: 0 };
        });
      }
      writeBakeoffResult({
        renderer: "cytoscape",
        n: model.nodes.length,
        buildMs: t1 - t0,
        layoutMs: t2 - t1,
        mountMs: t2 - t0,
        nodesDrawn: cy.nodes().length,
        ...(timedOut ? { error: "cose layout did not settle synchronously; used precomputed positions" } : {}),
      });
    });

    return () => {
      cancelled = true;
      cyRef.current = null;
      cy.destroy();
    };
  }, [model, palette]);

  // Container-driven resize, same pattern as ConstellationMap's ResizeObserver:
  // cytoscape doesn't auto-observe its container, so without this the canvas
  // keeps its mount-time dimensions after the wrapper resizes.
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;
    const ro = new ResizeObserver(() => cyRef.current?.resize());
    ro.observe(container);
    return () => ro.disconnect();
  }, []);

  return <div ref={containerRef} style={{ width: "100%", height: "100%" }} />;
}
