import { listPages, type Page } from "../../../lib/tauri";

const PAGE_BATCH_SIZE = 500;

async function listAllPagesWithStatus(status: "active" | "draft"): Promise<Page[]> {
  const pages: Page[] = [];
  const seenIds = new Set<string>();
  let offset = 0;

  while (true) {
    const batch = await listPages(status, undefined, PAGE_BATCH_SIZE, offset);
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

export function listAllActivePages(): Promise<Page[]> {
  return listAllPagesWithStatus("active");
}

export function listAllDraftPages(): Promise<Page[]> {
  return listAllPagesWithStatus("draft");
}
