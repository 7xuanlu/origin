// @vitest-environment jsdom
import { describe, it, expect, vi } from "vitest";
import Graph from "graphology";
import { DUST_BADGE_MIN, drawDustCounts, dustBadgeAnchors } from "./dust";
import { lodFor } from "./atlas";
import type { GraphPalette } from "./palette";

const PALETTE = { surface: "#111", labelMuted: "#999" } as GraphPalette;

function mockCtx() {
  const texts: { text: string; x: number; y: number }[] = [];
  const ctx = {
    save: vi.fn(),
    restore: vi.fn(),
    strokeText: vi.fn(),
    fillText: vi.fn((text: string, x: number, y: number) => texts.push({ text, x, y })),
  };
  return { ctx: ctx as unknown as CanvasRenderingContext2D, texts };
}

function graphWith(anchors: Record<string, Partial<{ dustCount: number; island: boolean; hidden: boolean; x: number }>>): Graph {
  const graph = new Graph();
  for (const [id, attrs] of Object.entries(anchors)) {
    graph.addNode(id, { x: 0, y: 0, size: 6, ...attrs });
  }
  return graph;
}

const identity = (pos: { x: number; y: number }) => pos;
const viewport = { width: 800, height: 600 };

describe("dust counts", () => {
  it("badges only anchors with at least DUST_BADGE_MIN memories", () => {
    const graph = graphWith({ busy: { dustCount: DUST_BADGE_MIN }, quiet: { dustCount: DUST_BADGE_MIN - 1 }, none: {} });
    expect(dustBadgeAnchors(graph)).toEqual(["busy"]);
  });

  it("draws the count up and to the right of the disc while the halo is thinned", () => {
    const graph = graphWith({ busy: { dustCount: 40, x: 100 } });
    const { ctx, texts } = mockCtx();
    drawDustCounts(ctx, graph, ["busy"], identity, PALETTE, lodFor(1), null, viewport);
    expect(texts).toEqual([{ text: "40", x: 100 + 6 * 0.8 + 2, y: -(6 * 0.8) - 3 }]);
  });

  it("draws nothing once the zoom shows every memory, or while a hover is live", () => {
    const graph = graphWith({ busy: { dustCount: 40 } });
    const a = mockCtx();
    drawDustCounts(a.ctx, graph, ["busy"], identity, PALETTE, lodFor(4), null, viewport);
    expect(a.texts).toEqual([]);
    const b = mockCtx();
    drawDustCounts(b.ctx, graph, ["busy"], identity, PALETTE, lodFor(1), "busy", viewport);
    expect(b.texts).toEqual([]);
  });

  it("skips a count the zoom already covers, a dim island anchor, a hidden anchor, and one off screen", () => {
    const graph = graphWith({
      covered: { dustCount: 5 },
      island: { dustCount: 40, island: true },
      hidden: { dustCount: 40, hidden: true },
      away: { dustCount: 40, x: 5000 },
      shown: { dustCount: 40 },
    });
    const { ctx, texts } = mockCtx();
    drawDustCounts(ctx, graph, ["covered", "island", "hidden", "away", "shown"], identity, PALETTE, lodFor(1), null, viewport);
    expect(texts.map((t) => t.text)).toEqual(["40"]);
    // The island's count comes back with its colour.
    const solid = mockCtx();
    drawDustCounts(solid.ctx, graph, ["island"], identity, PALETTE, lodFor(2), null, viewport);
    expect(solid.texts.map((t) => t.text)).toEqual(["40"]);
  });
});
