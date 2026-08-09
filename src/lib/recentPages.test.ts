// SPDX-License-Identifier: AGPL-3.0-only
import { describe, expect, it } from "vitest";
import type { Page } from "./tauri";
import {
  RECENT_PAGES_STORAGE_KEY,
  rankRecentPages,
  readRecentPageHistory,
  recordRecentPageVisit,
  type RecentPageHistory,
  type RecentPagesStorage,
} from "./recentPages";

const NOW = 2_000_000_000_000;
const DAY_MS = 86_400_000;

function page(id: string, title: string, status = "active"): Page {
  return {
    id,
    title,
    summary: null,
    content: "",
    entity_id: null,
    domain: null,
    source_memory_ids: [],
    version: 1,
    status,
    created_at: "2026-07-16T00:00:00Z",
    last_compiled: "2026-07-16T00:00:00Z",
    last_modified: "2026-07-16T00:00:00Z",
  };
}

function history(entries: RecentPageHistory["entries"]): RecentPageHistory {
  return { version: 1, entries };
}

function memoryStorage(initial: string | null = null): {
  readonly storage: RecentPagesStorage;
  readonly value: () => string | null;
} {
  let value = initial;
  return {
    storage: {
      getItem: (key) => key === RECENT_PAGES_STORAGE_KEY ? value : null,
      setItem: (key, next) => {
        if (key === RECENT_PAGES_STORAGE_KEY) value = next;
      },
    },
    value: () => value,
  };
}

describe("recent page ranking", () => {
  it("returns only actual active Page visits in MRU order, refreshes titles, and caps at four", () => {
    const pages = [
      page("unvisited", "Unvisited"),
      page("old", "Old"),
      page("middle", "Middle"),
      page("new", "Renamed New"),
      page("newest", "Newest"),
      page("fifth", "Fifth"),
      page("archived", "Archived", "archived"),
    ];
    const visits = history([
      { id: "old", title: "Old", visitedAt: NOW - 50 },
      { id: "middle", title: "Middle", visitedAt: NOW - 30 },
      { id: "new", title: "Old title", visitedAt: NOW - 20 },
      { id: "newest", title: "Newest", visitedAt: NOW - 10 },
      { id: "fifth", title: "Fifth", visitedAt: NOW - 60 },
      { id: "archived", title: "Archived", visitedAt: NOW - 1 },
    ]);

    expect(rankRecentPages(pages, visits, NOW).map(({ id, title }) => [id, title])).toEqual([
      ["newest", "Newest"],
      ["new", "Renamed New"],
      ["middle", "Middle"],
      ["old", "Old"],
    ]);
  });

  it("ignores invalid and expired visits without adding unvisited Pages", () => {
    const pages = [page("fallback", "Fallback"), page("expired", "Expired"), page("future", "Future")];
    const visits = history([
      { id: "expired", title: "Expired", visitedAt: NOW - 30 * DAY_MS },
      { id: "future", title: "Future", visitedAt: NOW + 1 },
    ]);

    expect(rankRecentPages(pages, visits, NOW)).toEqual([]);
  });
});

describe("recent page persistence", () => {
  it("records a real visit by stable id and refreshes its title", () => {
    const cache = memoryStorage(JSON.stringify(history([
      { id: "page", title: "Old title", visitedAt: NOW - 10 },
    ])));
    const current = page("page", "Current title");

    recordRecentPageVisit(current, { storage: cache.storage, pages: [current], now: NOW });

    expect(readRecentPageHistory({ storage: cache.storage, pages: [current], now: NOW })).toEqual(history([
      { id: "page", title: "Current title", visitedAt: NOW },
    ]));
    expect(cache.value()).toContain('"version":1');
  });

  it("treats absent, corrupt, and throwing storage as recoverable", () => {
    const corrupt = memoryStorage("not-json");
    const throwing: RecentPagesStorage = {
      getItem: () => { throw new Error("storage disabled"); },
      setItem: () => { throw new Error("quota denied"); },
    };
    const current = page("page", "Page");

    expect(readRecentPageHistory({ storage: null, now: NOW })).toEqual(history([]));
    expect(readRecentPageHistory({ storage: corrupt.storage, now: NOW })).toEqual(history([]));
    expect(readRecentPageHistory({ storage: throwing, now: NOW })).toEqual(history([]));
    expect(() => recordRecentPageVisit(current, { storage: throwing, now: NOW })).not.toThrow();
  });
});
