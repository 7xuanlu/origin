// SPDX-License-Identifier: AGPL-3.0-only
import type { SearchResult } from "./tauri";

export type SearchResultTarget =
  | { kind: "page"; pageId: string }
  | { kind: "file"; url: string }
  | { kind: "copy" };

export function searchResultTarget(result: SearchResult): SearchResultTarget {
  if (result.source === "page" && result.source_id) {
    return { kind: "page", pageId: result.source_id };
  }
  if (result.url) {
    return { kind: "file", url: result.url };
  }
  return { kind: "copy" };
}
