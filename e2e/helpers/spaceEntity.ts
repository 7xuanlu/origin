// SPDX-License-Identifier: AGPL-3.0-only
import { expect, type Locator, type Page } from "@playwright/test";

export type SpaceEntityLocale = "en" | "zh-Hans" | "zh-Hant";

const labels: Record<
  SpaceEntityLocale,
  { readonly keyEntities: string; readonly viewAllEntities: RegExp }
> = {
  en: {
    keyEntities: "Key entities",
    viewAllEntities: /^View all \d+$/,
  },
  "zh-Hans": {
    keyEntities: "关键实体",
    viewAllEntities: /^查看全部 \d+ 个$/,
  },
  "zh-Hant": {
    keyEntities: "關鍵實體",
    viewAllEntities: /^檢視全部 \d+ 個$/,
  },
};

export async function getSpaceEntityButton(
  page: Page,
  entityName: string,
  locale: SpaceEntityLocale = "en",
): Promise<Locator> {
  const copy = labels[locale];
  const region = page.getByRole("region", {
    name: copy.keyEntities,
    exact: true,
  });
  await expect(region).toBeVisible();

  const entity = region.getByRole("button", {
    name: entityName,
    exact: true,
  });
  const viewAll = region.getByRole("button", {
    name: copy.viewAllEntities,
  });
  if (await entity.isVisible()) {
    return entity;
  }

  await expect(viewAll).toBeVisible();
  await viewAll.click();
  await expect(entity).toBeVisible();
  return entity;
}

export async function openSpaceEntity(
  page: Page,
  entityName: string,
  locale: SpaceEntityLocale = "en",
): Promise<void> {
  await (await getSpaceEntityButton(page, entityName, locale)).click();
}
