// SPDX-License-Identifier: AGPL-3.0-only
import { describe, it, expect } from "vitest";
import { existsSync, readFileSync } from "node:fs";
import type { KnowledgeGraph } from "../tauri";
import { buildKnowledgeGraphModel, drawableModel } from "./model";
import type { GraphModel } from "./model";
import { buildAtlasGraph, runAtlasLayout, createAtlasSimulation } from "./atlas";
import { communitiesFor } from "./cartography";
import type { GraphPalette } from "./palette";

// A stopwatch over the whole main-thread layer flip, run against a real
// `GET /api/memory/graph` capture. It is a measurement lane, not a gate: there
// is no wall-clock assertion (a laptop under load would flake one), only the
// printed per-stage milliseconds that the perf work in round 3 is argued from.
//
// The capture is a developer artifact, never checked in, so the whole suite
// skips itself when the file is absent — which is every CI run.
const CAPTURE = "/tmp/claude-501/graph.json";

const PALETTE: GraphPalette = {
  project: "#111111",
  tool: "#222222",
  org: "#333333",
  person: "#444444",
  concept: "#555555",
  neutral: "#666666",
  edge: "#777777",
  edgeStrong: "#888888",
  label: "#999999",
  labelMuted: "#aaaaaa",
  surface: "#000000",
  graticule: "rgba(4,5,6,0.13)",
  bridge: "#bbbbbb",
  memory: "#cccccc",
  page: "#dddddd",
};

function ms<T>(label: string, run: () => T, into: string[]): T {
  const started = performance.now();
  const value = run();
  into.push(`${label}: ${(performance.now() - started).toFixed(1)} ms`);
  return value;
}

describe.skipIf(!existsSync(CAPTURE))("graph pipeline timing on a real capture", () => {
  it(
    "reports per-stage milliseconds with every layer on",
    () => {
      // Read inside the test, not in the describe body: vitest still EXECUTES
      // a skipped describe's body to collect it, so a top-level read would
      // throw ENOENT on every machine without the capture — which is all of
      // CI, and the whole point of the skip.
      const graph = JSON.parse(readFileSync(CAPTURE, "utf8")) as KnowledgeGraph;
      const lines: string[] = [
        `capture: ${graph.entities.length} entities, ${graph.relations.length} relations, ` +
          `${graph.memories.length} memories, ${graph.memory_links.length} memory links, ` +
          `${graph.pages.length} pages, ${graph.page_links.length} page links`,
      ];

      const model = ms(
        "buildKnowledgeGraphModel",
        () => buildKnowledgeGraphModel(graph, { layers: { entity: true, page: true, memory: true } }),
        lines,
      );
      // Present only after round 3; running this file against an older tree
      // still reports the other stages.
      const drawable: GraphModel =
        typeof drawableModel === "function"
          ? ms("drawableModel", () => drawableModel(model), lines)
          : model;
      ms("communityDetection", () => communitiesFor(drawable, new Map()), lines);
      const atlas = ms("toSigmaGraph", () => buildAtlasGraph(drawable, PALETTE), lines);
      ms("runAtlasLayout", () => runAtlasLayout(atlas), lines);
      const sim = ms("createAtlasSimulation", () => createAtlasSimulation(atlas), lines);
      sim.stop();

      lines.push(
        `model: ${model.nodes.length} nodes / ${model.edges.length} edges; ` +
          `drawn: ${drawable.nodes.length} nodes / ${drawable.edges.length} edges; ` +
          `simulated: ${sim.nodes().length} nodes`,
      );
      // eslint-disable-next-line no-console
      console.log(`\n${lines.join("\n")}\n`);

      expect(atlas.order).toBe(drawable.nodes.length);
    },
    600_000,
  );
});
