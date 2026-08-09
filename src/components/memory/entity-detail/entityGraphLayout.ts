// SPDX-License-Identifier: AGPL-3.0-only
export type GraphDirection = "incoming" | "outgoing";

export type GraphNeighbor = {
  readonly entityId: string;
  readonly name: string;
  readonly entityType: string;
  readonly verbs: readonly string[];
  readonly direction: GraphDirection;
};

export type PlacedGraphNeighbor = GraphNeighbor & {
  readonly x: number;
  readonly y: number;
  readonly labelX: number;
  readonly labelY: number;
};

function bandPositions(count: number, start: number, end: number, singleton: number): number[] {
  if (count === 0) return [];
  if (count === 1) return [singleton];
  return Array.from({ length: count }, (_, index) => start + (index * (end - start)) / (count - 1));
}

function placeCompact(
  neighbors: readonly GraphNeighbor[],
  direction: GraphDirection,
  hasOppositeSide: boolean,
): readonly PlacedGraphNeighbor[] {
  const positions = hasOppositeSide
    ? direction === "incoming"
      ? bandPositions(neighbors.length, 10, 38, 20)
      : bandPositions(neighbors.length, 62, 90, 80)
    : [
        ...bandPositions(Math.ceil(neighbors.length / 2), 10, 38, 20),
        ...bandPositions(Math.floor(neighbors.length / 2), 62, 90, 80),
      ];
  return neighbors.map((neighbor, index) => {
    const y = positions[index];
    return {
      ...neighbor,
      x: 50,
      y,
      labelX: 50,
      labelY: 50 + (y - 50) * 0.55,
    };
  });
}

function placeWide(
  neighbors: readonly GraphNeighbor[],
  side: "left" | "right",
  pairedSingletons: boolean,
): readonly PlacedGraphNeighbor[] {
  const labelPositions =
    side === "left"
      ? bandPositions(neighbors.length, 16, 36, 28)
      : bandPositions(neighbors.length, 64, 84, 72);
  return neighbors.map((neighbor, index) => {
    const y = pairedSingletons
      ? side === "left"
        ? 35
        : 65
      : ((index + 1) / (neighbors.length + 1)) * 100;
    const spread = (index % 2) * 6;
    return {
      ...neighbor,
      x: side === "right" ? 72 - spread : 28 + spread,
      y,
      labelX: 50,
      labelY: labelPositions[index],
    };
  });
}

export function layoutGraphNeighbors(
  neighbors: readonly GraphNeighbor[],
  compact: boolean,
): readonly PlacedGraphNeighbor[] {
  const incoming = neighbors.filter((neighbor) => neighbor.direction === "incoming");
  const outgoing = neighbors.filter((neighbor) => neighbor.direction === "outgoing");
  if (compact) {
    return [
      ...placeCompact(incoming, "incoming", outgoing.length > 0),
      ...placeCompact(outgoing, "outgoing", incoming.length > 0),
    ];
  }
  const pairedSingletons = incoming.length === 1 && outgoing.length === 1;
  return [
    ...placeWide(incoming, "left", pairedSingletons),
    ...placeWide(outgoing, "right", pairedSingletons),
  ];
}
