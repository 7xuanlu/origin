import { describe, expect, it } from "vitest";

import { resources, supportedAppLocales } from "./resources";

function flattenKeys(value: unknown, prefix = ""): string[] {
  if (!value || typeof value !== "object") {
    return [prefix];
  }

  return Object.entries(value as Record<string, unknown>).flatMap(
    ([key, child]) => flattenKeys(child, prefix ? `${prefix}.${key}` : key),
  );
}

describe("translation resources", () => {
  it("defines the supported app locales in a stable order", () => {
    expect(supportedAppLocales).toEqual(["en", "zh-Hans", "zh-Hant"]);
  });

  it("contains representative primary interface keys", () => {
    expect(resources.en.translation.setup.welcomeTitle).toBe(
      "Welcome to Wenlan",
    );
    expect(resources.en.translation.main.searchPlaceholder).toBe(
      "Search pages, memories, sources...",
    );
    expect(resources.en.translation.settings.language.label).toBe("Language");
  });

  it("keeps the Review fixture boundary explicit in every locale", () => {
    expect([
      resources.en.translation.reviewEnvironment.testData,
      resources["zh-Hans"].translation.reviewEnvironment.testData,
      resources["zh-Hant"].translation.reviewEnvironment.testData,
    ]).toEqual(["TEST DATA", "测试数据", "測試資料"]);
    expect([
      resources.en.translation.reviewEnvironment.fixtureNotice,
      resources["zh-Hans"].translation.reviewEnvironment.fixtureNotice,
      resources["zh-Hant"].translation.reviewEnvironment.fixtureNotice,
    ]).toEqual([
      "Fixture data · resets on relaunch",
      "固定测试数据 · 重新启动后重置",
      "固定測試資料 · 重新啟動後重設",
    ]);
    expect([
      resources.en.translation.reviewEnvironment.reset,
      resources["zh-Hans"].translation.reviewEnvironment.reset,
      resources["zh-Hant"].translation.reviewEnvironment.reset,
    ]).toEqual(["Reset test data", "重置测试数据", "重設測試資料"]);
  });

  it("presents the Pages route as Wiki in every locale", () => {
    expect([
      resources.en.translation.sidebar.pages,
      resources["zh-Hans"].translation.sidebar.pages,
      resources["zh-Hant"].translation.sidebar.pages,
    ]).toEqual(["Wiki", "Wiki", "Wiki"]);
    expect([
      resources.en.translation.pages.overview.title,
      resources["zh-Hans"].translation.pages.overview.title,
      resources["zh-Hant"].translation.pages.overview.title,
    ]).toEqual(["Wiki", "Wiki", "Wiki"]);
  });

  it("names the honest Wiki Page kinds without inferring schemas from prose", () => {
    const overviews = [
      resources.en.translation.pages.overview,
      resources["zh-Hans"].translation.pages.overview,
      resources["zh-Hant"].translation.pages.overview,
    ];

    expect(overviews.map((overview) => overview.typeLabel)).toEqual([
      "Kind",
      "类别",
      "類別",
    ]);
    expect(overviews.map((overview) => overview.columns.type)).toEqual([
      "Kind",
      "类别",
      "類別",
    ]);
    expect(overviews.map((overview) => overview.types)).toEqual([
      { page: "Page", entity: "Entity" },
      { page: "页面", entity: "实体" },
      { page: "頁面", entity: "實體" },
    ]);
  });

  it("defines the direct Page draft lifecycle copy in every locale", () => {
    const editors = [
      resources.en.translation.pages.editor,
      resources["zh-Hans"].translation.pages.editor,
      resources["zh-Hant"].translation.pages.editor,
    ];

    expect(editors.map((editor) => editor.publish)).toEqual([
      "Publish",
      "发布",
      "發佈",
    ]);
    expect(editors.map((editor) => editor.reloadLatest)).toEqual([
      "Reload latest",
      "载入最新版本",
      "載入最新版本",
    ]);
    expect(editors.map((editor) => editor.openExisting)).toEqual([
      "Open existing",
      "打开现有页面",
      "開啟現有頁面",
    ]);
  });

  it("names Wiki context-link destinations in every locale", () => {
    expect([
      resources.en.translation.pages.overview.openEntity,
      resources["zh-Hans"].translation.pages.overview.openEntity,
      resources["zh-Hant"].translation.pages.overview.openEntity,
    ]).toEqual([
      "Open Entity dossier: {{title}}",
      "打开实体档案：{{title}}",
      "開啟實體檔案：{{title}}",
    ]);
    expect([
      resources.en.translation.pages.overview.openSpace,
      resources["zh-Hans"].translation.pages.overview.openSpace,
      resources["zh-Hant"].translation.pages.overview.openSpace,
    ]).toEqual([
      "Open Space: {{space}}",
      "打开空间：{{space}}",
      "開啟空間：{{space}}",
    ]);
  });

  it("keeps Simplified and Traditional Chinese key sets in parity with English", () => {
    const englishKeys = flattenKeys(resources.en.translation).sort();

    expect(flattenKeys(resources["zh-Hans"].translation).sort()).toEqual(
      englishKeys,
    );
    expect(flattenKeys(resources["zh-Hant"].translation).sort()).toEqual(
      englishKeys,
    );
  });

  it("defines the complete Page dateline vocabulary in every locale", () => {
    const datelines = [
      resources.en.translation.pageDetail.dateline,
      resources["zh-Hans"].translation.pageDetail.dateline,
      resources["zh-Hant"].translation.pageDetail.dateline,
    ];

    expect(datelines).toEqual([
      {
        lastDistilled: "Last distilled {{time}}",
        sourceMemories_one: "from {{count}} memory",
        sourceMemories_other: "from {{count}} memories",
        relativeJustNow: "just now",
        relativeMinutesAgo: "{{count}}m ago",
        relativeHoursAgo: "{{count}}h ago",
        relativeDaysAgo: "{{count}}d ago",
        needsReview: "needs review",
        updating: "updating...",
      },
      {
        lastDistilled: "上次精炼：{{time}}",
        sourceMemories_one: "来自 {{count}} 条记忆",
        sourceMemories_other: "来自 {{count}} 条记忆",
        relativeJustNow: "刚刚",
        relativeMinutesAgo: "{{count}} 分钟前",
        relativeHoursAgo: "{{count}} 小时前",
        relativeDaysAgo: "{{count}} 天前",
        needsReview: "需要审核",
        updating: "正在更新…",
      },
      {
        lastDistilled: "上次精煉：{{time}}",
        sourceMemories_one: "來自 {{count}} 則記憶",
        sourceMemories_other: "來自 {{count}} 則記憶",
        relativeJustNow: "剛剛",
        relativeMinutesAgo: "{{count}} 分鐘前",
        relativeHoursAgo: "{{count}} 小時前",
        relativeDaysAgo: "{{count}} 天前",
        needsReview: "需要審核",
        updating: "正在更新…",
      },
    ]);
  });

  it("defines Spaces navigation, overview, and dossier copy in every locale", () => {
    expect([
      resources.en.translation.sidebar.recentSpaces,
      resources["zh-Hans"].translation.sidebar.recentSpaces,
      resources["zh-Hant"].translation.sidebar.recentSpaces,
    ]).toEqual(["Recent spaces", "最近空间", "最近空間"]);
    expect([
      resources.en.translation.spaces.overview.newSpace,
      resources["zh-Hans"].translation.spaces.overview.newSpace,
      resources["zh-Hant"].translation.spaces.overview.newSpace,
    ]).toEqual(["New space", "新增空间", "新增空間"]);
    expect([
      resources.en.translation.spaceDetail.backToSpaces,
      resources["zh-Hans"].translation.spaceDetail.backToSpaces,
      resources["zh-Hant"].translation.spaceDetail.backToSpaces,
    ]).toEqual(["Spaces", "返回空间", "返回空間"]);
  });

  it("uses the approved inventory heading in every locale", () => {
    // Given the existing confirmed inventory heading key
    const headings = [
      resources.en.translation.spaces.overview.confirmedHeading,
      resources["zh-Hans"].translation.spaces.overview.confirmedHeading,
      resources["zh-Hant"].translation.spaces.overview.confirmedHeading,
    ];

    // When the three supported locale values are read
    // Then they match the approved All spaces copy
    expect(headings).toEqual(["All spaces", "所有空间", "所有空間"]);
  });

  it("defines Home overview copy without the user-facing Index label", () => {
    // Given the three supported Home resource groups
    const homeResources = [
      resources.en.translation.home,
      resources["zh-Hans"].translation.home,
      resources["zh-Hant"].translation.home,
    ];

    // When their overview labels are read
    // Then every locale has meaningful Home copy and no legacy Index key
    expect(homeResources.map((home) => home.overview)).toEqual([
      "Home overview",
      "首页概览",
      "首頁概覽",
    ]);
    expect(homeResources.every((home) => !("index" in home))).toBe(true);
  });

  it("localizes all Spaces overview metadata labels", () => {
    // Given the desktop and mobile overview labels share one resource contract
    const overviewResources = [
      resources.en.translation.spaces.overview,
      resources["zh-Hans"].translation.spaces.overview,
      resources["zh-Hant"].translation.spaces.overview,
    ];

    // When Pages, Memories, and Updated are read
    // Then all supported locales provide the approved copy
    expect(overviewResources.map((overview) => overview.pages)).toEqual([
      "Pages",
      "页面",
      "頁面",
    ]);
    expect(overviewResources.map((overview) => overview.memories)).toEqual([
      "Memories",
      "记忆",
      "記憶",
    ]);
    expect(overviewResources.map((overview) => overview.updated)).toEqual([
      "Updated",
      "更新",
      "更新",
    ]);
  });
});

// Standing rule: never announce that Wenlan itself is a plugin. The only
// exceptions are real CLI commands/slash-commands and claude.ai's own menu
// names, referenced verbatim while walking the user through its UI.
function flattenStringEntries(
  value: unknown,
  prefix = "",
): Array<[string, string]> {
  if (typeof value === "string") return [[prefix, value]];
  if (!value || typeof value !== "object") return [];
  return Object.entries(value as Record<string, unknown>).flatMap(
    ([key, child]) =>
      flattenStringEntries(child, prefix ? `${prefix}.${key}` : key),
  );
}

const PLUGIN_WORD_ALLOWLIST = new Set([
  "connectMatrix.claudeCodeCommand1",
  "connectMatrix.claudeCodeCommand2",
  "connectMatrix.claudeCodeReload",
  "connectMatrix.claudeCodePrompt", // real `/plugin` menu + `claude plugin` CLI commands
  "connectMatrix.claudePluginStep1",
  "connectMatrix.claudePluginStep2",
  "connectMatrix.claudePluginStep3",
  "connectMatrix.chatgptStep1", // ChatGPT's own "Plugins" (插件/外掛程式) menu name, referenced verbatim
]);
const BANNED_PLUGIN_WORDS = ["plugin", "插件", "外掛"];

describe("banned self-referential 'plugin' copy", () => {
  it.each(supportedAppLocales)(
    "%s never describes Wenlan itself as a plugin",
    (locale) => {
      const entries = flattenStringEntries(resources[locale].translation);
      const offenders = entries.filter(
        ([key, value]) =>
          !PLUGIN_WORD_ALLOWLIST.has(key) &&
          BANNED_PLUGIN_WORDS.some((word) => value.includes(word)),
      );
      expect(offenders).toEqual([]);
    },
  );
});
