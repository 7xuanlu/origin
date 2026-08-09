import { describe, expect, it } from "vitest";
import {
  filterSpaces,
  findReorderTarget,
  isDuplicateSpaceName,
  sortConfirmedSpaces,
} from "./spaceHelpers";
import { makeSpace } from "./SpacesOverview.testUtils";

describe("space overview helpers", () => {
  it("sorts starred spaces first, then by sort order and name", () => {
    // Given mixed starred state, tied sort orders, and unsorted input
    const spaces = [
      makeSpace({ id: "b", name: "Beta", starred: false, sort_order: 2 }),
      makeSpace({ id: "z", name: "Zulu", starred: true, sort_order: 1 }),
      makeSpace({ id: "a", name: "Alpha", starred: false, sort_order: 2 }),
      makeSpace({ id: "y", name: "Yankee", starred: true, sort_order: 0 }),
    ];

    // When the confirmed inventory is sorted
    const result = sortConfirmedSpaces(spaces);

    // Then starred grouping and deterministic ties are preserved
    expect(result.map((space) => space.id)).toEqual(["y", "z", "a", "b"]);
  });

  it("filters trimmed case-insensitive text across name and description", () => {
    // Given spaces whose matches live in different fields
    const spaces = [
      makeSpace({ id: "work", name: "Work", description: "Projects" }),
      makeSpace({ id: "health", name: "Health", description: "Daily FITNESS notes" }),
    ];

    // When filters contain whitespace and mixed case
    const byName = filterSpaces(spaces, "  woRK ");
    const byDescription = filterSpaces(spaces, " fitness ");

    // Then matching is normalized without changing the source
    expect(byName.map((space) => space.id)).toEqual(["work"]);
    expect(byDescription.map((space) => space.id)).toEqual(["health"]);
    expect(spaces).toHaveLength(2);
  });

  it("detects duplicate names after trimming and case folding", () => {
    // Given one existing space
    const spaces = [makeSpace({ id: "work", name: "Work" })];

    // When names vary only by case or whitespace
    const duplicate = isDuplicateSpaceName(spaces, "  wOrK  ");
    const selfRename = isDuplicateSpaceName(spaces, " work ", "work");

    // Then new duplicates are rejected while a no-op self rename is allowed
    expect(duplicate).toBe(true);
    expect(selfRename).toBe(false);
  });

  it("finds reorder targets only inside the source starred group", () => {
    // Given adjacent starred and unstarred groups
    const spaces = sortConfirmedSpaces([
      makeSpace({ id: "star-a", name: "Star A", starred: true, sort_order: 0 }),
      makeSpace({ id: "star-b", name: "Star B", starred: true, sort_order: 1 }),
      makeSpace({ id: "plain-a", name: "Plain A", starred: false, sort_order: 0 }),
      makeSpace({ id: "plain-b", name: "Plain B", starred: false, sort_order: 1 }),
    ]);

    // When moving within and across the visual group boundary
    const withinGroup = findReorderTarget(spaces, "plain-b", "up");
    const blockedBoundary = findReorderTarget(spaces, "plain-a", "up");

    // Then only the same-group neighbor is returned
    expect(withinGroup?.id).toBe("plain-a");
    expect(blockedBoundary).toBeNull();
  });
});
