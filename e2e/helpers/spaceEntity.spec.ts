// SPDX-License-Identifier: AGPL-3.0-only
import { expect, test } from "@playwright/test";
import { getSpaceEntityButton } from "./spaceEntity";

test("returns the entity when it and View all are both visible", async ({ page }) => {
  await page.setContent(`
    <section aria-label="Key entities" role="region">
      <button>Ada Lovelace</button>
      <button>View all 2</button>
    </section>
  `);

  const entity = await getSpaceEntityButton(page, "Ada Lovelace");

  await expect(entity).toBeVisible();
});
