import { describe, expect, it } from "vitest";

import {
  APP_LOCALE_STORAGE_KEY,
  isStoredLocale,
  resolveAppLocale,
  resolveInitialLocale,
} from "./locales";

describe("resolveAppLocale", () => {
  it.each([
    ["zh", "zh-Hans"],
    ["zh-CN", "zh-Hans"],
    ["zh-SG", "zh-Hans"],
    ["zh-Hans", "zh-Hans"],
    ["zh-Hans-CN", "zh-Hans"],
    ["ZH_hans_sg", "zh-Hans"],
  ])("maps %s to Simplified Chinese", (input, expected) => {
    expect(resolveAppLocale(input)).toBe(expected);
  });

  it.each([
    ["zh-TW", "zh-Hant"],
    ["zh-HK", "zh-Hant"],
    ["zh-MO", "zh-Hant"],
    ["zh-Hant", "zh-Hant"],
    ["zh-Hant-TW", "zh-Hant"],
    ["ZH_hant_hk", "zh-Hant"],
  ])("maps %s to Traditional Chinese", (input, expected) => {
    expect(resolveAppLocale(input)).toBe(expected);
  });

  it.each(["en", "en-US", "fr", "", undefined])(
    "falls back to English for %s",
    (input) => {
      expect(resolveAppLocale(input)).toBe("en");
    },
  );
});

describe("stored locale choices", () => {
  it("uses the documented localStorage key", () => {
    expect(APP_LOCALE_STORAGE_KEY).toBe("wenlan-locale");
  });

  it.each(["system", "en", "zh-Hans", "zh-Hant"])(
    "accepts %s as a stored preference",
    (input) => {
      expect(isStoredLocale(input)).toBe(true);
    },
  );

  it.each(["zh", "zh-CN", "fr", "", null, undefined])(
    "rejects %s as a stored preference",
    (input) => {
      expect(isStoredLocale(input)).toBe(false);
    },
  );
});

describe("resolveInitialLocale", () => {
  it("uses explicit supported choices before system languages", () => {
    expect(resolveInitialLocale("zh-Hant", ["zh-CN", "en-US"])).toBe(
      "zh-Hant",
    );
  });

  it("uses the first supported system language when preference is system", () => {
    expect(resolveInitialLocale("system", ["fr-FR", "zh-TW", "en-US"])).toBe(
      "zh-Hant",
    );
  });

  it("falls back to English for invalid stored preferences and unsupported system languages", () => {
    expect(resolveInitialLocale("zh-CN", ["fr-FR"])).toBe("en");
  });
});
