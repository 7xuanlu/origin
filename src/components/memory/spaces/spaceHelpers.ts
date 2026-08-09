import type { Space } from "../../../lib/tauri";
import type { MoveDirection } from "./spacesTypes";

function normalized(value: string): string {
  return value.trim().toLocaleLowerCase();
}

function compareNames(left: Space, right: Space): number {
  return left.name.localeCompare(right.name, undefined, { sensitivity: "base" });
}

export function sortConfirmedSpaces(spaces: readonly Space[]): readonly Space[] {
  return spaces
    .filter((space) => !space.suggested)
    .sort((left, right) => {
      if (left.starred !== right.starred) return left.starred ? -1 : 1;
      return left.sort_order - right.sort_order || compareNames(left, right);
    });
}

export function sortSuggestedSpaces(spaces: readonly Space[]): readonly Space[] {
  return spaces
    .filter((space) => space.suggested)
    .sort((left, right) => left.sort_order - right.sort_order || compareNames(left, right));
}

export function filterSpaces(spaces: readonly Space[], filter: string): readonly Space[] {
  const query = normalized(filter);
  if (query.length === 0) return spaces;
  return spaces.filter((space) => {
    const description = space.description ?? "";
    return normalized(space.name).includes(query) || normalized(description).includes(query);
  });
}

export function isDuplicateSpaceName(
  spaces: readonly Space[],
  candidate: string,
  currentName?: string,
): boolean {
  const target = normalized(candidate);
  const current = currentName === undefined ? null : normalized(currentName);
  return spaces.some((space) => {
    const existing = normalized(space.name);
    return existing === target && existing !== current;
  });
}

export function findReorderTarget(
  orderedSpaces: readonly Space[],
  sourceId: string,
  direction: MoveDirection,
): Space | null {
  const source = orderedSpaces.find((space) => space.id === sourceId);
  if (source === undefined) return null;
  const group = orderedSpaces.filter((space) => space.starred === source.starred);
  const index = group.findIndex((space) => space.id === sourceId);
  const targetIndex = direction === "up" ? index - 1 : index + 1;
  return group[targetIndex] ?? null;
}

export function canReorderTogether(source: Space, target: Space): boolean {
  return source.id !== target.id && source.starred === target.starred;
}
