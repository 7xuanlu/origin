import type { Page } from "../../../lib/tauri";

export type PagePresentationType = "entity" | "page";

export function classifyPage(page: Page): PagePresentationType {
  return page.entity_id ? "entity" : "page";
}

export function pageSpaceContext(page: Page): string | undefined {
  if (page.space !== undefined) return page.space?.trim() || undefined;
  return page.domain?.trim() || undefined;
}
