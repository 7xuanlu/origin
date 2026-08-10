// Radial layout spike for the Page Map (docs/superpowers/plans/
// 2026-07-18-page-map-mind-map.md). The council gated the tooling decision on
// proving that variable-width boxes survive a radial mapping: a naive
// cartesian-then-angle transform re-collides wide boxes near the center.
// `radialNaive` exists to demonstrate that failure; `radialPolar` is the fix —
// it runs flextree directly in polar coordinates (angular width = px / radius)
// and grows the rings when a circle overflows. Separation is solved on circles;
// the rings are then squashed onto ellipses to match the frame (RING_SQUASH),
// which only ever reduces vertical clearance, never radial.

import flextree from './flextree/flextree.js';

export interface MapNodeInput {
  id: string;
  /** box size in px, as the DOM would render it */
  width: number;
  height: number;
  children?: MapNodeInput[];
}

export interface PlacedNode {
  id: string;
  /** box center, px, root at origin */
  x: number;
  y: number;
  width: number;
  height: number;
  depth: number;
}

const TAU = Math.PI * 2;

interface RadialOptions {
  /** clear space between ring edges, px */
  ringGap?: number;
  /** minimum tangential clearance between sibling boxes, px */
  spacing?: number;
}

// Ring center-line radii per depth, safe at any angle: separate consecutive
// rings by the boxes' half-diagonals so rotation can never make them touch
// radially. ponytail: conservative (wide short boxes get roomy rings); tune
// per-ring by actual angular occupancy if maps look too sparse.
function ringRadii(root: MapNodeInput, ringGap: number): number[] {
  const halfDiag: number[] = [];
  const walk = (n: MapNodeInput, depth: number) => {
    const hd = Math.hypot(n.width, n.height) / 2;
    halfDiag[depth] = Math.max(halfDiag[depth] ?? 0, hd);
    for (const c of n.children ?? []) walk(c, depth + 1);
  };
  walk(root, 0);
  const radii = [0];
  for (let d = 1; d < halfDiag.length; d++) {
    radii[d] = radii[d - 1] + halfDiag[d - 1] + ringGap + halfDiag[d];
  }
  return radii;
}

/**
 * How much wider than tall each ring is.
 *
 * The rings are ellipses rather than circles because the frame they are read in
 * is not square: the canvas is full width and `min(68vh, 620px)` tall, about
 * 1.8:1, while circular rings produce a roughly square map. Fitting a square
 * map into a wide frame wastes the width and spends the height, so a page with
 * thirteen sections opened either unreadably small or with its top and bottom
 * boxes cut off by the frame edge.
 *
 * 1.55 rather than the frame's own 1.8: squashing the y axis compresses the
 * tangential clearance between siblings stacked either side of the vertical
 * axis, and that clearance is what `radialPolar` works to establish. Measured on
 * a 24-box map, the tightest gap between any two boxes goes from 4px at 1.0
 * (circular) to 2px at 1.55 to boxes touching at 1.7. 1.55 is the most squash
 * the layout's own margin pays for.
 */
export const RING_SQUASH = 1.55;

function place(
  nodes: { data: MapNodeInput; depth: number; angle: number }[],
  radii: number[],
): PlacedNode[] {
  return nodes.map(({ data, depth, angle }) => ({
    id: data.id,
    x: depth === 0 ? 0 : radii[depth] * Math.cos(angle),
    y: depth === 0 ? 0 : (radii[depth] / RING_SQUASH) * Math.sin(angle),
    width: data.width,
    height: data.height,
    depth,
  }));
}

/**
 * The transform the council warned about: lay out in cartesian px, then map
 * the breadth axis linearly onto the full circle. Angular box widths are
 * never considered, so wide boxes on small rings overlap. Kept exported so
 * the test suite can keep demonstrating WHY it is not used.
 */
export function radialNaive(
  root: MapNodeInput,
  { ringGap = 48, spacing = 16 }: RadialOptions = {},
): PlacedNode[] {
  const radii = ringRadii(root, ringGap);
  const layout = flextree<MapNodeInput>({
    nodeSize: (n) => [n.data.width, 1],
    spacing,
  });
  const tree = layout.hierarchy(root);
  layout(tree);
  const all = tree.descendants();
  const minLeft = Math.min(...all.map((n) => n.x - n.xSize / 2));
  const maxRight = Math.max(...all.map((n) => n.x + n.xSize / 2));
  const span = Math.max(maxRight - minLeft, 1e-9);
  return place(
    all.map((n) => ({
      data: n.data,
      depth: n.depth,
      angle: ((n.x - minLeft) / span) * TAU,
    })),
    radii,
  );
}

/**
 * Polar-aware layout: flextree runs with breadth measured in radians
 * (box width / ring radius), so separation accounts for box size AT its
 * radius. If the tree needs more than a full circle, all rings grow by the
 * overflow factor and the layout re-runs (converges geometrically). If it
 * needs less, angles spread to fill the circle — spreading only ever adds
 * separation, so it cannot create collisions. The seam (first vs last node
 * meeting across 0/TAU after spreading) is held open by one spacing at the
 * innermost occupied ring.
 */
export function radialPolar(
  root: MapNodeInput,
  { ringGap = 48, spacing = 16 }: RadialOptions = {},
): PlacedNode[] {
  const baseRadii = ringRadii(root, ringGap);
  if (baseRadii.length === 1) {
    return place([{ data: root, depth: 0, angle: 0 }], baseRadii);
  }
  let scale = 1;
  for (let attempt = 0; attempt < 12; attempt++) {
    const radii = baseRadii.map((r) => r * scale);
    const layout = flextree<MapNodeInput>({
      nodeSize: (n) =>
        n.depth === 0 ? [0, 1] : [n.data.width / radii[n.depth], 1],
      spacing: (a, b) =>
        spacing / radii[Math.min(Math.max(a.depth, 1), Math.max(b.depth, 1))],
    });
    const tree = layout.hierarchy(root);
    layout(tree);
    const all = tree.descendants();
    const ringed = all.filter((n) => n.depth > 0);
    const minLeft = Math.min(...ringed.map((n) => n.x - n.xSize / 2));
    const maxRight = Math.max(...ringed.map((n) => n.x + n.xSize / 2));
    const span = Math.max(maxRight - minLeft, 1e-9);
    const seam = spacing / radii[1];
    if (span + seam > TAU) {
      // Overflow: shrink angular widths by growing every ring, then re-run.
      scale *= (span + seam) / TAU;
      continue;
    }
    const spread = (TAU - seam) / span;
    const angleOf = (n: { x: number }) => (n.x - minLeft) * spread + seam / 2;
    // The first branch sits at three o'clock and the rest fan clockwise from
    // there. Without the rotation the fan starts wherever the seam falls, and
    // two branches land at 90 and 270 degrees — a vertical chain down a canvas
    // twice as wide as it is tall, which reads as a list rather than a map.
    //
    // Rotating the whole fan is free: rings are separated by their boxes'
    // half-diagonals and siblings by full box width at their own radius, so
    // neither clearance depends on which way the map happens to point.
    const branchAngles = ringed.filter((n) => n.depth === 1).map(angleOf);
    const offset = branchAngles.length ? Math.min(...branchAngles) : 0;
    return place(
      all.map((n) => ({
        data: n.data,
        depth: n.depth,
        angle: n.depth === 0 ? 0 : angleOf(n) - offset,
      })),
      radii,
    );
  }
  throw new Error('radialPolar: ring growth did not converge in 12 rounds');
}

/** Axis-aligned pairwise overlap check (React Flow nodes don't rotate). */
export function findCollisions(
  nodes: PlacedNode[],
  tolerance = 0.5,
): [string, string][] {
  const hits: [string, string][] = [];
  for (let i = 0; i < nodes.length; i++) {
    for (let j = i + 1; j < nodes.length; j++) {
      const a = nodes[i];
      const b = nodes[j];
      if (
        Math.abs(a.x - b.x) < (a.width + b.width) / 2 - tolerance &&
        Math.abs(a.y - b.y) < (a.height + b.height) / 2 - tolerance
      ) {
        hits.push([a.id, b.id]);
      }
    }
  }
  return hits;
}
