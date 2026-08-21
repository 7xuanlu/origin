import {
  listPages,
  listPagesExplicitBrowse,
  type Page,
} from "../../../lib/tauri";

export { EXPLICIT_BROWSE_QUERY_POLICY } from "../../../hooks/useTruthStatus";

const PAGE_BATCH_SIZE = 500;

type PageBatchFetcher = (
  status: "active" | "draft",
  domain: undefined,
  limit: number,
  offset: number,
) => Promise<Page[]>;

async function listAllPagesWithStatus(
  status: "active" | "draft",
  fetchBatch: PageBatchFetcher,
): Promise<Page[]> {
  const pages: Page[] = [];
  const seenIds = new Set<string>();
  let offset = 0;

  while (true) {
    const batch = await fetchBatch(status, undefined, PAGE_BATCH_SIZE, offset);
    let added = 0;
    for (const page of batch) {
      if (seenIds.has(page.id)) continue;
      seenIds.add(page.id);
      pages.push(page);
      added += 1;
    }
    if (batch.length < PAGE_BATCH_SIZE || added === 0) return pages;
    offset += batch.length;
  }
}

/**
 * True for a page a person wrote, distilled, imported, or researched; false
 * for an entity "shadow" page (the auto-made page behind every entity).
 *
 * `pages.kind` never serializes, but every shadow page carries
 * `creation_kind = "entity"` — the same invariant the daemon's knowledge-graph
 * query relies on (`crates/wenlan-core/src/db/scoped_entities.rs`,
 * `get_knowledge_graph_scoped`). A missing `creation_kind` (older daemons)
 * counts as a knowledge page. Browse lists still include shadow pages by
 * contract; this is for counts that should not double-count entities.
 */
export function isKnowledgePage(page: Page): boolean {
  return page.creation_kind !== "entity";
}

export function listAllActivePages(): Promise<Page[]> {
  return listAllPagesWithStatus("active", listPages);
}

export function listAllDraftPages(): Promise<Page[]> {
  return listAllPagesWithStatus("draft", listPages);
}

/**
 * Explicit-browse siblings for the Wiki overview screen, the one place a
 * person deliberately browses the full page list. Every other caller of
 * {@link listAllActivePages}/{@link listAllDraftPages} (sidebar recents, the
 * home feed, space page counts) is a background or ambient read and must
 * keep using the automatic functions above.
 */
export function listAllActivePagesExplicitBrowse(): Promise<Page[]> {
  return listAllPagesWithStatus("active", listPagesExplicitBrowse);
}

export function listAllDraftPagesExplicitBrowse(): Promise<Page[]> {
  return listAllPagesWithStatus("draft", listPagesExplicitBrowse);
}
