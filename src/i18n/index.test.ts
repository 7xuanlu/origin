import { createInstance } from "i18next";
import { describe, expect, it } from "vitest";

import {
  initializeI18n,
  readStoredLocalePreference,
  setLocalePreference,
} from ".";
import { APP_LOCALE_STORAGE_KEY } from "./locales";

function createStorage(initialValue?: string): Storage {
  const items = new Map<string, string>();
  if (initialValue !== undefined) {
    items.set(APP_LOCALE_STORAGE_KEY, initialValue);
  }

  return {
    get length() {
      return items.size;
    },
    clear: () => items.clear(),
    getItem: (key: string) => items.get(key) ?? null,
    key: (index: number) => Array.from(items.keys())[index] ?? null,
    removeItem: (key: string) => {
      items.delete(key);
    },
    setItem: (key: string, value: string) => {
      items.set(key, value);
    },
  };
}

describe("i18n runtime", () => {
  it("initializes with real English resources by default", async () => {
    const instance = createInstance();

    await initializeI18n(instance, {
      storage: createStorage(),
      systemLanguages: ["fr-FR"],
    });

    expect(instance.language).toBe("en");
    expect(instance.t("setup.welcomeTitle")).toBe("Welcome to Wenlan");
  });

  it("uses explicit stored language before system language", async () => {
    const instance = createInstance();

    await initializeI18n(instance, {
      storage: createStorage("zh-Hans"),
      systemLanguages: ["zh-TW", "en-US"],
    });

    expect(instance.language).toBe("zh-Hans");
    expect(instance.t("setup.getStarted")).toBe("开始使用");
  });

  it("reads invalid stored language as system", () => {
    expect(readStoredLocalePreference(createStorage("zh-CN"))).toBe("system");
  });

  it("persists explicit language changes and updates the i18n instance", async () => {
    const instance = createInstance();
    const storage = createStorage();

    await initializeI18n(instance, {
      storage,
      systemLanguages: ["en-US"],
    });
    await setLocalePreference("zh-Hant", {
      instance,
      storage,
      systemLanguages: ["en-US"],
    });

    expect(storage.getItem(APP_LOCALE_STORAGE_KEY)).toBe("zh-Hant");
    expect(instance.language).toBe("zh-Hant");
    expect(instance.t("setup.getStarted")).toBe("開始使用");
  });

  it("persists system preference and resolves against current system languages", async () => {
    const instance = createInstance();
    const storage = createStorage("zh-Hant");

    await initializeI18n(instance, {
      storage,
      systemLanguages: ["en-US"],
    });
    await setLocalePreference("system", {
      instance,
      storage,
      systemLanguages: ["zh-CN"],
    });

    expect(storage.getItem(APP_LOCALE_STORAGE_KEY)).toBe("system");
    expect(instance.language).toBe("zh-Hans");
  });
});
