// SPDX-License-Identifier: AGPL-3.0-only
// Top-level bake-off harness (task #17): builds the synthetic model once per
// n and mounts the selected renderer's adapter. Each adapter is imported via
// React.lazy so `pnpm build` emits one chunk per candidate — that per-chunk
// size is itself one of the bake-off's measures.
import { Suspense, lazy, useMemo } from "react";
import { generateBakeoffGraph } from "./synthetic";
import type { BakeoffRenderer } from "./bakeoffResult";

const CytoscapeAdapter = lazy(() => import("./CytoscapeAdapter"));
const SigmaAdapter = lazy(() => import("./SigmaAdapter"));
const G6Adapter = lazy(() => import("./G6Adapter"));

export default function BakeoffHarness({ renderer, n }: { renderer: BakeoffRenderer; n: number }) {
  const model = useMemo(() => generateBakeoffGraph(n), [n]);

  return (
    <Suspense fallback={<div style={{ padding: 16 }}>Loading {renderer}…</div>}>
      {renderer === "cytoscape" ? (
        <CytoscapeAdapter model={model} />
      ) : renderer === "sigma" ? (
        <SigmaAdapter model={model} />
      ) : (
        <G6Adapter model={model} />
      )}
    </Suspense>
  );
}
