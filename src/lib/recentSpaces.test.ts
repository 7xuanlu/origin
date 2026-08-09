// SPDX-License-Identifier: AGPL-3.0-only
import { describe, expect, it } from "vitest";
import type { Space } from "./tauri";
import {
  RECENT_SPACES_STORAGE_KEY,
  deleteRecentSpace,
  rankRecentSpaces,
  readRecentSpaceHistory,
  recordRecentSpaceVisit,
  renameRecentSpace,
  type RecentSpaceHistory,
  type RecentSpacesStorage,
} from "./recentSpaces";

const NOW = 2_000_000_000_000;
const DAY_MS = 86_400_000;

function space(
  id: string,
  name: string,
  overrides: Partial<Space> = {},
): Space {
  return {
    id,
    name,
    description: null,
    suggested: false,
    starred: false,
    sort_order: 0,
    memory_count: 0,
    entity_count: 0,
    created_at: 0,
    updated_at: 0,
    ...overrides,
  };
}

function history(
  entries: RecentSpaceHistory["entries"],
): RecentSpaceHistory {
  return { version: 1, entries };
}

function memoryStorage(initial: string | null = null): {
  readonly storage: RecentSpacesStorage;
  readonly value: () => string | null;
} {
  let value = initial;
  return {
    storage: {
      getItem: (key) => key === RECENT_SPACES_STORAGE_KEY ? value : null,
      setItem: (key, next) => {
        if (key === RECENT_SPACES_STORAGE_KEY) value = next;
      },
    },
    value: () => value,
  };
}

describe("recent space ranking", () => {
  it("returns only actual confirmed visits in MRU order and caps the result at four", () => {
    // Given
    const spaces = [
      space("fallback-z", "Zulu", { updated_at: 50 }),
      space("star-b", "Beta", { starred: true, sort_order: 2 }),
      space("suggested", "Suggested", { suggested: true, starred: true }),
      space("mru-old", "Old"),
      space("star-a", "Alpha", { starred: true, sort_order: 2 }),
      space("mru-new", "New"),
      space("mru-newest", "Newest"),
      space("mru-middle", "Middle"),
    ];
    const visits = history([
      { id: "mru-old", name: "Old", visitedAt: NOW - 20 },
      { id: "mru-new", name: "New", visitedAt: NOW - 10 },
      { id: "mru-newest", name: "Newest", visitedAt: NOW - 1 },
      { id: "mru-middle", name: "Middle", visitedAt: NOW - 15 },
      { id: "star-a", name: "Alpha", visitedAt: NOW - 30 },
      { id: "suggested", name: "Suggested", visitedAt: NOW - 1 },
    ]);

    // When
    const ranked = rankRecentSpaces(spaces, visits, NOW);

    // Then
    expect(ranked.map(({ id }) => id)).toEqual([
      "mru-newest",
      "mru-new",
      "mru-middle",
      "mru-old",
    ]);
    expect(ranked.map(({ id }) => id)).not.toContain("star-b");
    expect(ranked.map(({ id }) => id)).not.toContain("fallback-z");
  });

  it("uses stable ids across rename and deduplicates ids without merging duplicate names", () => {
    // Given
    const spaces = [
      space("same-id", "Renamed"),
      space("same-id", "Duplicate server row"),
      space("other-id", "Renamed"),
    ];
    const visits = history([
      { id: "same-id", name: "Old name", visitedAt: NOW - 50 },
      { id: "same-id", name: "Older duplicate", visitedAt: NOW - 100 },
      { id: "other-id", name: "Renamed", visitedAt: NOW - 25 },
    ]);

    // When
    const ranked = rankRecentSpaces(spaces, visits, NOW);

    // Then
    expect(ranked.map(({ id, name }) => [id, name])).toEqual([
      ["other-id", "Renamed"],
      ["same-id", "Renamed"],
    ]);
  });

  it("ignores expired, negative, NaN, and future visit timestamps instead of falling back", () => {
    // Given
    const spaces = [
      space("fallback", "Fallback", { updated_at: 10 }),
      space("expired", "Expired", { updated_at: 40 }),
      space("negative", "Negative", { updated_at: 30 }),
      space("nan", "NaN", { updated_at: 20 }),
      space("future", "Future", { updated_at: 50 }),
    ];
    const visits = history([
      { id: "expired", name: "Expired", visitedAt: NOW - 30 * DAY_MS },
      { id: "negative", name: "Negative", visitedAt: -1 },
      { id: "nan", name: "NaN", visitedAt: Number.NaN },
      { id: "future", name: "Future", visitedAt: NOW + 1 },
    ]);

    // When
    const ranked = rankRecentSpaces(spaces, visits, NOW);

    // Then
    expect(ranked).toEqual([]);
  });
});

describe("recent space persistence", () => {
  it("records version one history, prunes stale entries, and retains fifty visits", () => {
    // Given
    const spaces = Array.from({ length: 53 }, (_, index) =>
      space(`space-${index}`, `Space ${index}`),
    );
    const seededEntries = spaces.map(({ id, name }, index) => ({
      id,
      name,
      visitedAt: NOW - index - 1,
    }));
    const cache = memoryStorage(JSON.stringify(history([
      ...seededEntries,
      { id: "missing", name: "Missing", visitedAt: NOW - 1 },
      { id: "expired", name: "Expired", visitedAt: NOW - 31 * DAY_MS },
    ])));

    // When
    recordRecentSpaceVisit(spaces[52], {
      storage: cache.storage,
      spaces,
      now: NOW,
    });

    // Then
    const stored = readRecentSpaceHistory({ storage: cache.storage, now: NOW });
    expect(stored.version).toBe(1);
    expect(stored.entries).toHaveLength(50);
    expect(stored.entries[0]).toEqual({
      id: "space-52",
      name: "Space 52",
      visitedAt: NOW,
    });
    expect(stored.entries.some(({ id }) => id === "missing" || id === "expired")).toBe(false);
    expect(cache.value()).toContain('"version":1');
  });

  it("renames and deletes history entries by stable id", () => {
    // Given
    const cache = memoryStorage(JSON.stringify(history([
      { id: "keep", name: "Duplicate", visitedAt: NOW - 1 },
      { id: "change", name: "Duplicate", visitedAt: NOW - 2 },
    ])));
    const runtime = {
      storage: cache.storage,
      spaces: [space("keep", "Duplicate"), space("change", "Renamed")],
      now: NOW,
    };

    // When
    renameRecentSpace({ id: "change", name: "Renamed" }, runtime);
    deleteRecentSpace("keep", runtime);

    // Then
    expect(readRecentSpaceHistory(runtime)).toEqual(history([
      { id: "change", name: "Renamed", visitedAt: NOW - 2 },
    ]));
  });

  it("treats absent, corrupt, invalid-schema, and throwing storage as recoverable", () => {
    // Given
    const corrupt = memoryStorage("not-json");
    const invalid = memoryStorage(JSON.stringify({
      version: 1,
      entries: [{ id: "x", name: "X", visitedAt: null }],
    }));
    const throwing: RecentSpacesStorage = {
      getItem: () => { throw new Error("storage disabled"); },
      setItem: () => { throw new Error("quota denied"); },
    };
    const current = space("current", "Current");

    // When / Then
    expect(readRecentSpaceHistory({ storage: null, now: NOW })).toEqual(history([]));
    expect(readRecentSpaceHistory({ storage: corrupt.storage, now: NOW })).toEqual(history([]));
    expect(readRecentSpaceHistory({ storage: invalid.storage, now: NOW })).toEqual(history([]));
    expect(readRecentSpaceHistory({ storage: throwing, now: NOW })).toEqual(history([]));
    expect(() => recordRecentSpaceVisit(current, { storage: throwing, now: NOW })).not.toThrow();
    expect(() => renameRecentSpace(current, { storage: throwing, now: NOW })).not.toThrow();
    expect(() => deleteRecentSpace(current.id, { storage: throwing, now: NOW })).not.toThrow();
  });
});
