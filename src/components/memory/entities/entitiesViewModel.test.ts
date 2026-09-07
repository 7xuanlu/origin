// SPDX-License-Identifier: AGPL-3.0-only
import { describe, expect, it } from "vitest";
import type { TFunction } from "i18next";
import type { Entity } from "../../../lib/tauri";
import {
  DEFAULT_FILTERS,
  ENTITIES_PAGE_SIZE,
  entityListRequest,
  establishedByLabel,
  filterReadback,
  filtersActive,
  matchLabel,
  selectAllState,
  selectionSummary,
  subtitle,
  toggleSelectAll,
  toggleSelected,
  type EntityFilters,
} from "./entitiesViewModel";

// Same convention as pageReviewNotice.test.ts: a `t` that echoes the key (and,
// when interpolation is used, the options) rather than real copy, so these
// tests lock the logic that picks a key, not the translated string.
const t = ((key: string, options?: Record<string, unknown>) =>
  options ? `${key}:${JSON.stringify(options)}` : key) as unknown as TFunction;

function makeEntity(overrides: Partial<Entity> = {}): Entity {
  return {
    id: "entity-1",
    name: "Ada Lovelace",
    entity_type: "person",
    domain: null,
    source_agent: null,
    confidence: null,
    confirmed: true,
    created_at: 0,
    updated_at: 0,
    memory_count: 3,
    status: "established",
    established_by: "manual",
    ...overrides,
  };
}

describe("entityListRequest", () => {
  it("maps each tab to its status and pages with the fixed page size", () => {
    expect(entityListRequest("established", DEFAULT_FILTERS, 0)).toEqual({
      status: "established",
      limit: ENTITIES_PAGE_SIZE,
      offset: 0,
    });
    expect(entityListRequest("archived", DEFAULT_FILTERS, 200)).toEqual({
      status: "archived",
      limit: ENTITIES_PAGE_SIZE,
      offset: 200,
    });
  });

  it("ignores filters on tabs other than Detected", () => {
    const filters: EntityFilters = { query: "ada", type: "person", memories: "none" };
    expect(entityListRequest("established", filters, 0)).toEqual({
      status: "established",
      limit: ENTITIES_PAGE_SIZE,
      offset: 0,
    });
    expect(entityListRequest("archived", filters, 0)).toEqual({
      status: "archived",
      limit: ENTITIES_PAGE_SIZE,
      offset: 0,
    });
  });

  it("applies the type chip, trimmed query, and 'no memories' as an exact zero range", () => {
    const filters: EntityFilters = { query: "  ada  ", type: "concept", memories: "none" };
    expect(entityListRequest("detected", filters, 0)).toEqual({
      status: "detected",
      limit: ENTITIES_PAGE_SIZE,
      offset: 0,
      entity_type: "concept",
      min_memories: 0,
      max_memories: 0,
      query: "ada",
    });
  });

  it("maps 'some memories' to a lower bound of one, with no upper bound", () => {
    const request = entityListRequest("detected", { ...DEFAULT_FILTERS, memories: "some" }, 0);
    expect(request.min_memories).toBe(1);
    expect(request.max_memories).toBeUndefined();
  });

  it("omits entity_type, memory bounds, and query when filters are at their defaults", () => {
    const request = entityListRequest("detected", DEFAULT_FILTERS, 0);
    expect(request).toEqual({ status: "detected", limit: ENTITIES_PAGE_SIZE, offset: 0 });
  });

  it("drops a whitespace-only query rather than sending an empty filter", () => {
    const request = entityListRequest("detected", { ...DEFAULT_FILTERS, query: "   " }, 0);
    expect(request.query).toBeUndefined();
  });
});

describe("filtersActive", () => {
  it("is false only when every filter is at its default", () => {
    expect(filtersActive(DEFAULT_FILTERS)).toBe(false);
    expect(filtersActive({ ...DEFAULT_FILTERS, type: "place" })).toBe(true);
    expect(filtersActive({ ...DEFAULT_FILTERS, memories: "some" })).toBe(true);
    expect(filtersActive({ ...DEFAULT_FILTERS, query: "ada" })).toBe(true);
  });

  it("treats a whitespace-only query as inactive", () => {
    expect(filtersActive({ ...DEFAULT_FILTERS, query: "   " })).toBe(false);
  });
});

describe("matchLabel", () => {
  it("picks the filtered key only for a filtered Detected tab", () => {
    expect(matchLabel("detected", 5, false, t)).toBe('entities.match_detected:{"count":5}');
    expect(matchLabel("detected", 5, true, t)).toBe('entities.matchFiltered_detected:{"count":5}');
  });

  it("always reads Archived as unfiltered", () => {
    expect(matchLabel("archived", 2, true, t)).toBe('entities.match_archived:{"count":2}');
  });
});

describe("filterReadback", () => {
  it("reads back the type chip, the memories chip, and a present query", () => {
    const filters: EntityFilters = { query: "ada", type: "concept", memories: "none" };
    const readback = filterReadback(filters, t);
    expect(readback).toContain("entities.filters.type_concept");
    expect(readback).toContain("entities.filters.memoriesNoneDescription");
    expect(readback).toContain('entities.filters.queryDescription:{"query":"ada"}');
  });

  it("omits the query clause when the search box is empty", () => {
    const readback = filterReadback(DEFAULT_FILTERS, t);
    expect(readback).toContain("entities.filters.typeAny");
    expect(readback).toContain("entities.filters.memoriesAnyDescription");
    expect(readback).not.toContain("queryDescription");
  });
});

describe("subtitle and selectionSummary", () => {
  it("passes the three tab counts straight through", () => {
    expect(subtitle({ established: 1, detected: 2, archived: 3 }, t)).toBe(
      'entities.subtitle:{"established":1,"detected":2,"archived":3}',
    );
  });

  it("passes the selected count straight through", () => {
    expect(selectionSummary(4, t)).toBe('entities.selection:{"count":4}');
  });
});

describe("establishedByLabel", () => {
  it("names a manual establish", () => {
    expect(establishedByLabel(makeEntity({ established_by: "manual" }), t)).toBe(
      "entities.establishedBy.manual",
    );
  });

  it("names an auto-citation establish", () => {
    expect(establishedByLabel(makeEntity({ established_by: "auto:citation" }), t)).toBe(
      "entities.establishedBy.citation",
    );
  });

  it("falls back to the memory count for any other reason, such as the auto-memories threshold", () => {
    expect(
      establishedByLabel(makeEntity({ established_by: "auto:memories", memory_count: 7 }), t),
    ).toBe('entities.establishedBy.memories:{"count":7}');
  });
});

describe("selection model", () => {
  it("toggles a single id in and out without disturbing the rest", () => {
    const selected = new Set(["a", "b"]);
    expect([...toggleSelected(selected, "c")].sort()).toEqual(["a", "b", "c"]);
    expect([...toggleSelected(selected, "a")].sort()).toEqual(["b"]);
  });

  it("selects every id on screen and leaves ids from other pages untouched", () => {
    const selected = new Set(["off-screen"]);
    const next = toggleSelectAll(selected, ["a", "b"], true);
    expect([...next].sort()).toEqual(["a", "b", "off-screen"]);
  });

  it("clears only the ids on screen", () => {
    const selected = new Set(["a", "b", "off-screen"]);
    const next = toggleSelectAll(selected, ["a", "b"], false);
    expect([...next]).toEqual(["off-screen"]);
  });

  it("reports checked, indeterminate, and empty header-checkbox states", () => {
    expect(selectAllState(new Set(), ["a", "b"])).toEqual({ checked: false, indeterminate: false });
    expect(selectAllState(new Set(["a"]), ["a", "b"])).toEqual({ checked: false, indeterminate: true });
    expect(selectAllState(new Set(["a", "b"]), ["a", "b"])).toEqual({ checked: true, indeterminate: false });
    expect(selectAllState(new Set(), [])).toEqual({ checked: false, indeterminate: false });
  });
});
