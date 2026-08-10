// SPDX-License-Identifier: AGPL-3.0-only
import { BaseEdge, type Edge, type EdgeProps } from "@xyflow/react";

/**
 * The stroke between a box and its parent, drawn as a mind map draws it.
 *
 * React Flow's own bezier is wrong here, and not by a little: it derives its
 * control points from the handle's `Position` enum, so with our handles pinned
 * at the box center as Top and Bottom every curve leaves the parent going down
 * and enters the child going down. On a radial map, where a child can sit at
 * any angle, that reads as an S laid over the spoke. `straight` avoided the S
 * but drew the map as a wire diagram.
 *
 * A quadratic with its control point pushed off the chord's own midpoint has no
 * opinion about which way is down: the arc follows wherever the child actually
 * is, which is the only direction that means anything on a ring.
 */

/**
 * How far the arc bows, as a fraction of the distance between the two boxes.
 *
 * Proportional rather than fixed so a long spoke to the outer ring and a short
 * one between siblings bow by the same visual amount. Kept small on purpose:
 * boxes on one ring are separated by their own width, not by room for a curve,
 * so a deep bow on a crowded ring will lean a stroke into the neighbouring box.
 */
const BOW = 0.11;

/**
 * The most the drawn arc may sag away from the straight line, px.
 *
 * A proportional bow is right for the spans the layout produces — a ring is one
 * box-depth out from the last, so spokes run 150-250px and sag 8-14px. It stops
 * being right once a box has been dragged: pull a child 600px out and a purely
 * proportional bow sags 33px, which stops reading as a curve and starts reading
 * as a detour around something.
 */
const MAX_SAG = 24;

function SpineEdge({
  sourceX,
  sourceY,
  targetX,
  targetY,
  style,
  markerEnd,
}: EdgeProps) {
  const dx = targetX - sourceX;
  const dy = targetY - sourceY;
  const span = Math.hypot(dx, dy);
  // Perpendicular to the chord, one consistent rotational sense for every edge
  // on the map. Mirroring the sign so a fan bows symmetrically is what a static
  // bilateral mind map does, and it is wrong for a canvas you can edit: the sign
  // flips the moment a child is dragged across its fan's bisector, so the arc
  // visibly snaps to the other side mid-drag. One sense is stable under every
  // edit, and the slight pinwheel reads as deliberate.
  //
  // The drawn arc sags half this offset — the midpoint of a quadratic sits
  // halfway to its control point — so the cap is doubled here.
  const offset = Math.min(span * BOW, MAX_SAG * 2);
  const cx = (sourceX + targetX) / 2 + (-dy / (span || 1)) * offset;
  const cy = (sourceY + targetY) / 2 + (dx / (span || 1)) * offset;

  return (
    <BaseEdge
      path={`M${sourceX},${sourceY} Q${cx},${cy} ${targetX},${targetY}`}
      style={style}
      markerEnd={markerEnd}
    />
  );
}

export default SpineEdge;
export type SpineEdgeType = Edge<Record<string, unknown>, "spine">;
