// SPDX-License-Identifier: AGPL-3.0-only
// Shared instrumentation for the renderer bake-off (task #17, spec at
// ~/.wenlan/sessions/2026-07-16-knowledge-graph-redesign/atlas-bakeoff-spec.md).
// Every adapter writes exactly one BakeoffResult here so task #19 can read
// timings/counts from an occluded tab without depending on rAF-driven signals
// (see the occluded-tab-raf-suspended lesson: rAF is fully suspended there).

export type BakeoffRenderer = "cytoscape" | "sigma" | "g6";

export interface BakeoffResult {
  renderer: BakeoffRenderer;
  n: number;
  buildMs: number;
  layoutMs: number;
  mountMs: number;
  nodesDrawn: number;
  /** Set when a candidate couldn't lay out synchronously and the adapter fell
   * back to the generator's precomputed x/y — that fallback is itself a
   * bake-off finding, not a bug to hide. */
  error?: string;
}

declare global {
  interface Window {
    __BAKEOFF?: BakeoffResult;
  }
}

export function writeBakeoffResult(result: BakeoffResult): void {
  window.__BAKEOFF = result;
}

/** Top-degree node ids, for hub-only labeling — every adapter labels the same set. */
export function hubIds(nodes: { id: string; degree: number }[], limit = 20): Set<string> {
  return new Set(
    [...nodes]
      .sort((a, b) => b.degree - a.degree)
      .slice(0, limit)
      .map((node) => node.id),
  );
}

/** Node visual size from degree — shared scale so the three renderers are comparable. */
export function sizeForDegree(degree: number): number {
  return 4 + Math.min(16, Math.sqrt(degree) * 3);
}

/**
 * Race a layout's completion against a plain setTimeout (never rAF, so this
 * still fires in an occluded tab). If the layout hasn't settled within `ms`,
 * that's the "cannot lay out synchronously" finding the spec asks to record —
 * the caller falls back to the generator's precomputed positions instead of
 * waiting forever.
 */
export function withLayoutTimeout(work: Promise<void>, ms = 1000): Promise<{ timedOut: boolean }> {
  return new Promise((resolve) => {
    const timer = setTimeout(() => resolve({ timedOut: true }), ms);
    work.then(() => {
      clearTimeout(timer);
      resolve({ timedOut: false });
    });
  });
}
