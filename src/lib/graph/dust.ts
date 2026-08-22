import type Graph from "graphology";
import type { LodState } from "./atlas";
import type { GraphPalette } from "./palette";

/**
 * The memory count beside a busy anchor. At the opening view an anchor shows
 * only a pinch of its memories (see atlas.ts's dustVisibleCount); when it
 * has at least DUST_BADGE_MIN the number says how much is really there.
 * Below that the dots themselves say "a few" and a number would be clutter.
 * Drawn on AtlasView's overlay canvas, above sigma, like the place names.
 */
export const DUST_BADGE_MIN = 25;

/** The anchors that can carry a count — fixed for a mounted graph. */
export function dustBadgeAnchors(graph: Graph): string[] {
  return graph.filterNodes(
    (_id, attrs) => ((attrs.dustCount as number | undefined) ?? 0) >= DUST_BADGE_MIN,
  );
}

const BADGE_FONT = "500 9px 'JetBrains Mono', ui-monospace, monospace";
const BADGE_HALO_WIDTH = 3;
/** Extra viewport margin inside which a badge is still drawn. */
const BADGE_CULL_PAD = 24;

/**
 * Paint the counts. Nothing is drawn while a hover is live (the hovered
 * anchor shows its whole halo; everything else is muted), when the zoom
 * already shows every memory, for a dim island node, or for an anchor off
 * screen. `project` maps graph coords to viewport CSS px.
 */
export function drawDustCounts(
  ctx: CanvasRenderingContext2D,
  graph: Graph,
  anchors: string[],
  project: (pos: { x: number; y: number }) => { x: number; y: number },
  palette: GraphPalette,
  lod: LodState,
  hovered: string | null,
  viewport: { width: number; height: number },
): void {
  if (lod.dustVisible === Infinity || hovered !== null) return;
  ctx.save();
  ctx.font = BADGE_FONT;
  ctx.textAlign = "left";
  ctx.textBaseline = "middle";
  ctx.lineJoin = "round";
  ctx.lineWidth = BADGE_HALO_WIDTH;
  ctx.strokeStyle = palette.surface;
  ctx.fillStyle = palette.labelMuted;
  for (const id of anchors) {
    const attrs = graph.getNodeAttributes(id);
    if (attrs.hidden) continue;
    if (attrs.island && !lod.islandsSolid) continue;
    const count = attrs.dustCount as number;
    if (count <= lod.dustVisible) continue;
    const p = project({ x: attrs.x as number, y: attrs.y as number });
    if (
      p.x < -BADGE_CULL_PAD ||
      p.y < -BADGE_CULL_PAD ||
      p.x > viewport.width + BADGE_CULL_PAD ||
      p.y > viewport.height + BADGE_CULL_PAD
    )
      continue;
    const r = (attrs.size as number) ?? 4;
    // Up and to the right of the disc, clear of the radial node label,
    // which sits on the side facing away from the cluster centre.
    const x = p.x + r * 0.8 + 2;
    const y = p.y - r * 0.8 - 3;
    const text = String(count);
    ctx.strokeText(text, x, y);
    ctx.fillText(text, x, y);
  }
  ctx.restore();
}
