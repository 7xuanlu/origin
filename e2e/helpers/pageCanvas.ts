// SPDX-License-Identifier: AGPL-3.0-only
import { expect, type Locator, type Page } from "@playwright/test";

/** Open the "Fixture architecture" page and switch it to Canvas view. */
export async function openCanvas(page: Page): Promise<void> {
  await page.goto("/");
  await page
    .getByRole("navigation", { name: "Primary navigation" })
    .getByRole("button", { name: "Wiki", exact: true })
    .click();
  await page.getByRole("button", { name: "Open Fixture architecture" }).click();
  await expect(page.getByRole("heading", { level: 1, name: "Fixture architecture" })).toBeVisible();
  await page.getByRole("button", { name: "Canvas" }).click();
  await expect(page.getByRole("region", { name: "Canvas for Fixture architecture" })).toBeVisible();
}

/** A box on the canvas, addressed the way a reader sees it: by its name. */
export function box(page: Page, label: string): Locator {
  return page.locator(".react-flow__node").filter({ hasText: label }).first();
}

/**
 * A map the size a real page grows to: thirteen boxes over three rings. Seed
 * the mock straight through its own command surface before the canvas ever
 * asks for it.
 */
export async function seedLargeMap(page: Page): Promise<void> {
  await page.evaluate(async () => {
    const add = async (parent: string, label: string): Promise<string> => {
      const result = (await window.__wenlanTauriInvoke("create_page_map_node", {
        pageId: "page-architecture",
        body: { parent_id: parent, label, ref_kind: "section", ref_id: label },
      })) as { node: { id: string } };
      return result.node.id;
    };
    const ingest = await add("n_root", "Ingest pipeline");
    await add(ingest, "Chunking");
    await add(ingest, "Embedding queue");
    const api = await add("n_root", "HTTP surface");
    await add(api, "Auth");
    await add(api, "Rate limits");
    await add("n_root", "Observability");
    await add("n_query", "Reranking");
    await add("n_query", "Filters");
  });
}
