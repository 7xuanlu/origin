// SPDX-License-Identifier: AGPL-3.0-only
import { expect, test } from "@playwright/test";
import { installTauriMock } from "./tauriMock";
import type { AppLocale } from "../src/i18n/locales";

const rawActions = [
  "page_skip_user_edited",
  "observation_add",
  "entity_create",
  "relation_create",
] as const;

const localeCases: Record<Exclude<AppLocale, "en">, {
  home: string;
  navigation: string;
  activity: string;
  action: string;
  spaces: string;
  actionLabels: string[];
}> = {
  "zh-Hans": {
    home: "首页",
    navigation: "主导航",
    activity: "活动",
    action: "动作",
    spaces: "空间",
    actionLabels: [
      "跳过页面更新",
      "新增观察",
      "新增实体",
      "新增关系",
    ],
  },
  "zh-Hant": {
    home: "首頁",
    navigation: "主要導覽",
    activity: "活動",
    action: "動作",
    spaces: "空間",
    actionLabels: [
      "略過頁面更新",
      "新增觀察",
      "新增實體",
      "新增關聯",
    ],
  },
};

test.describe("Chinese interface localization", () => {
  for (const [locale, labels] of Object.entries(localeCases) as [Exclude<AppLocale, "en">, typeof localeCases[Exclude<AppLocale, "en">]][]) {
    test(`${locale} localizes the shell and activity filter`, async ({ page }) => {
      const pageErrors: string[] = [];
      const consoleErrors: string[] = [];
      page.on("pageerror", (error) => pageErrors.push(error.message));
      page.on("console", (message) => {
        if (message.type() === "error") {
          consoleErrors.push(message.text());
        }
      });

      await installTauriMock(page, { locale, rawActions: [...rawActions] });
      await page.goto("/");

      const header = page.getByRole("banner");
      const primaryNavigation = page.getByRole("navigation", { name: labels.navigation });
      await expect(primaryNavigation.getByRole("button", { name: labels.home, exact: true })).toBeVisible();
      await expect(primaryNavigation.getByRole("button", { name: "Wiki", exact: true })).toBeVisible();
      await expect(header.getByRole("button", { name: labels.home })).toHaveCount(0);
      await expect(header.getByRole("button", { name: labels.activity })).toBeVisible();
      await expect(page.getByText(labels.spaces, { exact: true })).toBeVisible();
      await expect(page.getByText("Wenlan 文瀾", { exact: true })).toHaveCount(0);
      await expect(page.getByText("SPACES", { exact: true })).toHaveCount(0);
      await expect(page.getByText("Home", { exact: true })).toHaveCount(0);
      await expect(page.getByText("Activity", { exact: true })).toHaveCount(0);

      await header.getByRole("button", { name: labels.activity }).click();

      const actionFilter = page.getByLabel(labels.action);
      await expect(actionFilter).toBeVisible();
      await expect(actionFilter.locator("option")).toHaveText([
        labels.action,
        ...labels.actionLabels,
      ]);

      const renderedText = await page.locator("body").innerText();
      for (const action of rawActions) {
        expect(renderedText).not.toContain(action);
      }
      expect(pageErrors).toEqual([]);
      expect(consoleErrors).toEqual([]);
    });
  }
});
