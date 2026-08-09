// SPDX-License-Identifier: AGPL-3.0-only
import { describe, expect, it } from "vitest";
import type { SearchResult } from "./tauri";
import { searchResultTarget } from "./searchResultTarget";

function result(overrides: Partial<SearchResult>): SearchResult {
  return {
    id: "hit-1",
    content: "hit content",
    source: "memory",
    source_id: "mem-1",
    title: "Hit",
    url: null,
    chunk_index: 0,
    last_modified: 0,
    score: 0.5,
    ...overrides,
  };
}

describe("searchResultTarget", () => {
  it("routes page-channel search results to page detail", () => {
    expect(searchResultTarget(result({ source: "page", source_id: "page-1" }))).toEqual({
      kind: "page",
      pageId: "page-1",
    });
  });

  it("routes URL-backed search results to files", () => {
    expect(searchResultTarget(result({ url: "/tmp/note.md" }))).toEqual({
      kind: "file",
      url: "/tmp/note.md",
    });
  });

  it("falls back to copy for non-page rows without URLs", () => {
    expect(searchResultTarget(result({ source: "memory", source_id: "mem-1" }))).toEqual({
      kind: "copy",
    });
  });
});
