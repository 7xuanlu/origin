import { describe, expect, it } from 'vitest';
import {
  findCollisions,
  radialNaive,
  radialPolar,
  RING_SQUASH,
  type MapNodeInput,
  type PlacedNode,
} from './radial';

// Box sizes measured from the approved mockup (screen 01): root card,
// branch pills, leaf cards, ghost suggestion cards.
const ROOT = { width: 210, height: 60 };
const BRANCH = { width: 170, height: 36 };
const LEAF = { width: 230, height: 52 };
const GHOST = { width: 250, height: 76 };

let seq = 0;
const node = (
  size: { width: number; height: number },
  children?: MapNodeInput[],
): MapNodeInput => ({ id: `n${seq++}`, ...size, children });

/** 12 wide branches, one narrow child each — the council's failure case:
 * inner-ring arc per branch is smaller than the branch box itself. */
const crowdedTree = (): MapNodeInput =>
  node(
    ROOT,
    Array.from({ length: 12 }, () =>
      node(BRANCH, [node({ width: 40, height: 24 })]),
    ),
  );

/** Page-map-shaped tree: 3 branches, mixed leaf/ghost fan-outs, depth 3. */
const pageTree = (): MapNodeInput =>
  node(ROOT, [
    node(BRANCH, [node(LEAF), node(LEAF, [node(GHOST)])]),
    node(BRANCH, [node(LEAF), node(LEAF), node(GHOST)]),
    node(BRANCH, [node(LEAF, [node(LEAF), node(GHOST)])]),
  ]);

const byDepth = (placed: PlacedNode[], depth: number) =>
  placed.filter((p) => p.depth === depth);

describe('radialNaive (kept only as the documented failure)', () => {
  it('re-collides wide boxes near the center — why it is not used', () => {
    expect(findCollisions(radialNaive(crowdedTree())).length).toBeGreaterThan(
      0,
    );
  });
});

describe('radialPolar', () => {
  it('lays out the crowded tree with zero collisions', () => {
    expect(findCollisions(radialPolar(crowdedTree()))).toEqual([]);
  });

  it('grows ring 1 until the full circumference fits every branch box', () => {
    const placed = radialPolar(crowdedTree());
    const ring1 = byDepth(placed, 1);
    const r1 = Math.hypot(ring1[0].x, ring1[0].y);
    const needed = ring1.reduce((sum, p) => sum + p.width, 0);
    expect(2 * Math.PI * r1).toBeGreaterThanOrEqual(needed);
  });

  it('lays out a page-map-shaped tree with zero collisions', () => {
    expect(findCollisions(radialPolar(pageTree()))).toEqual([]);
  });

  it('survives the 0/TAU seam when spreading to a full circle', () => {
    // Few, wide children: large spread factor, so the first and last boxes
    // meet across the seam — the case a plain fill-the-circle scale breaks.
    const tree = node(
      ROOT,
      Array.from({ length: 4 }, () => node(GHOST)),
    );
    expect(findCollisions(radialPolar(tree))).toEqual([]);
  });

  it('keeps every node exactly on its depth ring', () => {
    const placed = radialPolar(pageTree());
    // The rings are ellipses, not circles — squashed on y to match the frame the
    // map is read in — so the invariant is the ellipse equation rather than a
    // constant distance from the centre. Every node on a ring shares one
    // (semi-major, semi-minor) pair, which is what "on its ring" now means.
    // Un-squashing y turns the ellipse back into the circle the separation was
    // solved on, so every node on a ring must report the same radius there. A
    // node's own coordinates give it exactly, at any angle.
    const semiMajor = (p: PlacedNode) => Math.hypot(p.x, p.y * RING_SQUASH);
    for (let depth = 1; depth <= 3; depth++) {
      const ring = byDepth(placed, depth);
      const radius = semiMajor(ring[0]);
      for (const p of ring) {
        expect(semiMajor(p)).toBeCloseTo(radius, 6);
      }
    }
  });

  it('squashes the rings toward the shape of the frame they are read in', () => {
    // The whole point of the squash: a map that is wider than it is tall fits a
    // frame that is wider than it is tall. Circular rings made this ~1.0.
    const placed = radialPolar(pageTree()).filter((p) => p.depth > 0);
    const spanX =
      Math.max(...placed.map((p) => p.x + p.width / 2)) -
      Math.min(...placed.map((p) => p.x - p.width / 2));
    const spanY =
      Math.max(...placed.map((p) => p.y + p.height / 2)) -
      Math.min(...placed.map((p) => p.y - p.height / 2));
    expect(spanX / spanY).toBeGreaterThan(1.3);
  });

  it('preserves sibling input order as monotonic angles', () => {
    const tree = node(ROOT, [node(BRANCH), node(BRANCH), node(BRANCH)]);
    const ring1 = byDepth(radialPolar(tree), 1);
    const angles = ring1.map((p) => Math.atan2(p.y, p.x));
    const unwrapped = angles.map((a) => (a < angles[0] ? a + 2 * Math.PI : a));
    expect(unwrapped).toEqual([...unwrapped].sort((a, b) => a - b));
  });

  it('puts the first branch at three o clock', () => {
    const ring1 = byDepth(radialPolar(node(ROOT, [node(BRANCH), node(BRANCH)])), 1);
    expect(Math.atan2(ring1[0].y, ring1[0].x)).toBeCloseTo(0, 6);
  });

  it('sends two branches sideways, never into a vertical chain', () => {
    // The regression this rotation exists for: filling the circle from wherever
    // the seam fell put two branches at 90 and 270 degrees, stacking them down
    // a canvas twice as wide as it is tall. Both must clear the vertical.
    const ring1 = byDepth(radialPolar(node(ROOT, [node(BRANCH), node(BRANCH)])), 1);
    expect(ring1).toHaveLength(2);
    for (const p of ring1) {
      expect(Math.abs(p.x)).toBeGreaterThan(Math.abs(p.y));
    }
    // ...and on opposite sides of it, not both to the right.
    expect(ring1[0].x * ring1[1].x).toBeLessThan(0);
  });

  it('stays collision-free through the rotation, at every branch count', () => {
    for (let n = 1; n <= 8; n++) {
      const tree = node(
        ROOT,
        Array.from({ length: n }, () => node(BRANCH, [node(LEAF)])),
      );
      expect(findCollisions(radialPolar(tree))).toEqual([]);
    }
  });

  it('handles a root-only map', () => {
    const placed = radialPolar(node(ROOT));
    expect(placed).toHaveLength(1);
    expect(placed[0]).toMatchObject({ x: 0, y: 0, depth: 0 });
  });
});

describe('findCollisions', () => {
  it('reports touching-but-not-overlapping boxes as clear', () => {
    const a = { id: 'a', x: 0, y: 0, width: 100, height: 40, depth: 1 };
    const b = { id: 'b', x: 100, y: 0, width: 100, height: 40, depth: 1 };
    expect(findCollisions([a, b])).toEqual([]);
    expect(findCollisions([a, { ...b, x: 90 }])).toEqual([['a', 'b']]);
  });
});
